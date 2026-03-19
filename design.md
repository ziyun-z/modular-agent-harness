# Modular Agent Harness: Design Doc & Implementation Plan

## 1. Project Overview

### Goal
Build a modular agent harness framework where memory, context compression, and communication architecture components can be swapped independently, then run a rigorous ablation study on SWE-bench tasks to compare performance across configurations.

### Why SWE-bench
- Ground-truth patches exist → binary pass/fail metric
- Tasks span difficulty levels (easy lint fixes → complex cross-file refactors)
- Repos don't fit in one context window → forces real memory/compression solutions
- Well-known in the community → results are interpretable and comparable
- SWE-bench Lite (300 tasks) is a tractable subset for individual work

### Success Criteria
- A working agent that can attempt SWE-bench Lite tasks end-to-end
- At least 3 memory variants, 3 compression variants, 2 communication variants implemented
- A results table comparing solve rates across configurations
- A written analysis of tradeoffs and findings

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Runner / CLI                       │
│  (loads config, picks task, runs agent, scores result)  │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                    AgentOrchestrator                     │
│  (main loop: observe → think → act → repeat)            │
│                                                         │
│  Pluggable modules:                                     │
│  ┌─────────────┐ ┌──────────────┐ ┌──────────────────┐  │
│  │   Memory     │ │ Compression  │ │  Communication   │  │
│  │   Module     │ │   Module     │ │    Module        │  │
│  └─────────────┘ └──────────────┘ └──────────────────┘  │
│                                                         │
│  Shared:                                                │
│  ┌─────────────┐ ┌──────────────┐ ┌──────────────────┐  │
│  │  Tool        │ │   LLM        │ │   Logger /       │  │
│  │  Executor    │ │   Client     │ │   Tracer         │  │
│  └─────────────┘ └──────────────┘ └──────────────────┘  │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                   Sandboxed Environment                  │
│  (Docker container with cloned repo, test runner)       │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Directory Structure

```
agent-harness/
├── README.md
├── pyproject.toml
├── configs/                    # YAML configs for each experiment
│   ├── baseline.yaml
│   ├── rag_memory.yaml
│   ├── hierarchical_compression.yaml
│   ├── multi_agent.yaml
│   └── ...
├── src/
│   ├── __init__.py
│   ├── runner.py               # CLI entry point
│   ├── orchestrator.py         # Main agent loop
│   ├── llm_client.py           # Wrapper around Anthropic/OpenAI API
│   ├── tool_executor.py        # Executes bash, file read/write, etc.
│   ├── sandbox.py              # Docker sandbox management
│   ├── logger.py               # Structured logging + trajectory recording
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── base.py             # Abstract base class
│   │   ├── naive.py            # Full context, hard truncation
│   │   ├── scratchpad.py       # Agent-maintained notes
│   │   ├── rag.py              # Vector DB retrieval
│   │   └── hybrid.py           # Episodic + semantic
│   ├── compression/
│   │   ├── __init__.py
│   │   ├── base.py             # Abstract base class
│   │   ├── none.py             # No compression (baseline)
│   │   ├── rolling_summary.py  # Summarize oldest turns periodically
│   │   └── hierarchical.py     # Multi-level summaries + landmarks
│   ├── communication/
│   │   ├── __init__.py
│   │   ├── base.py             # Abstract base class
│   │   ├── single_agent.py     # One agent does everything
│   │   ├── orchestrated.py     # Planner + specialist workers
│   │   └── blackboard.py       # Shared workspace, autonomous agents
│   └── evaluation/
│       ├── __init__.py
│       ├── swebench_loader.py  # Load tasks from SWE-bench dataset
│       ├── scorer.py           # Run tests, compute pass/fail
│       └── analysis.py         # Aggregate results, generate tables
├── tests/
│   ├── test_memory.py
│   ├── test_compression.py
│   ├── test_communication.py
│   └── test_orchestrator.py
├── experiments/
│   ├── run_ablation.py         # Script to run full ablation matrix
│   └── results/                # JSON results per run
└── docs/
    └── writeup.md              # Final analysis and findings
```

---

## 4. Core Interfaces (Abstract Base Classes)

### 4.1 Memory Module

```python
# src/memory/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

@dataclass
class MemoryEntry:
    """A single unit of memory."""
    step: int                   # Which agent step produced this
    entry_type: str             # "observation", "action", "thought", "error"
    content: str                # The actual content
    metadata: dict[str, Any]    # Arbitrary metadata (tool name, file path, etc.)
    timestamp: float            # Unix timestamp

class MemoryModule(ABC):
    """Interface for all memory implementations."""

    @abstractmethod
    def store(self, entry: MemoryEntry) -> None:
        """Store a new memory entry."""
        ...

    @abstractmethod
    def retrieve(self, query: str, max_tokens: int) -> list[MemoryEntry]:
        """
        Retrieve relevant memories given a query and token budget.

        Args:
            query: The current task context or question to retrieve memories for.
            max_tokens: Maximum number of tokens the returned memories should
                        consume (approximate). Implementations should respect
                        this budget.

        Returns:
            List of MemoryEntry objects, ordered by relevance or recency
            depending on implementation.
        """
        ...

    @abstractmethod
    def get_context_block(self, max_tokens: int) -> str:
        """
        Produce a formatted string to inject into the LLM prompt.
        This is the main interface the orchestrator calls each turn.

        Args:
            max_tokens: Token budget for the memory block.

        Returns:
            A formatted string ready to insert into the system/user prompt.
        """
        ...

    @abstractmethod
    def clear(self) -> None:
        """Reset all stored memories. Called between tasks."""
        ...

    def get_stats(self) -> dict:
        """Return metrics about memory usage (entries stored, tokens used, etc.)."""
        return {}
```

### 4.2 Compression Module

```python
# src/compression/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class ConversationTurn:
    """A single turn in the agent's trajectory."""
    role: str               # "assistant" or "tool_result"
    content: str            # The message content
    step: int               # Step number
    is_landmark: bool       # Whether this turn should be preserved during compression
    token_count: int        # Pre-computed token count

class CompressionModule(ABC):
    """Interface for all compression implementations."""

    @abstractmethod
    def compress(
        self,
        turns: list[ConversationTurn],
        target_tokens: int,
        llm_client: "LLMClient"
    ) -> list[ConversationTurn]:
        """
        Compress a conversation history to fit within target_tokens.

        Args:
            turns: Full conversation history.
            target_tokens: Target total token count after compression.
            llm_client: LLM client for summarization calls (if needed).

        Returns:
            Compressed list of ConversationTurn objects.
        """
        ...

    @abstractmethod
    def should_compress(self, turns: list[ConversationTurn], max_tokens: int) -> bool:
        """Check if compression is needed given current history and token limit."""
        ...

    def get_stats(self) -> dict:
        """Return compression metrics (compressions performed, tokens saved, etc.)."""
        return {}
```

### 4.3 Communication Module

```python
# src/communication/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

@dataclass
class AgentConfig:
    """Configuration for a single agent in the system."""
    name: str                           # e.g., "planner", "code_reader", "writer"
    system_prompt: str                  # Role-specific system prompt
    tools: list[str]                    # Which tools this agent can use
    memory_module: "MemoryModule"       # Each agent can have its own memory
    compression_module: "CompressionModule"

@dataclass
class AgentMessage:
    """A message passed between agents."""
    sender: str
    recipient: str          # Or "broadcast" for blackboard-style
    content: str
    message_type: str       # "task", "result", "status", "query"
    metadata: dict[str, Any]

class CommunicationModule(ABC):
    """Interface for all communication/coordination implementations."""

    @abstractmethod
    def setup(self, task_description: str, llm_client: "LLMClient") -> None:
        """
        Initialize the communication structure for a new task.
        For single agent: no-op.
        For multi-agent: spawn agent configs, set up channels.
        """
        ...

    @abstractmethod
    def run_step(
        self,
        llm_client: "LLMClient",
        tool_executor: "ToolExecutor",
    ) -> dict[str, Any]:
        """
        Execute one step of the agent system.

        For single agent: one LLM call + tool execution.
        For multi-agent: may involve multiple LLM calls coordinated
                         according to the architecture.

        Returns:
            Dict with keys:
                - "actions_taken": list of actions executed
                - "done": bool, whether the agent thinks the task is complete
                - "result": optional final result if done=True
                - "llm_calls": int, number of LLM calls made this step
        """
        ...

    @abstractmethod
    def get_trajectory(self) -> list[dict]:
        """Return the full trajectory for logging/analysis."""
        ...

    def get_stats(self) -> dict:
        """Return communication metrics (messages sent, agents used, etc.)."""
        return {}
```

---

## 5. Orchestrator (Main Agent Loop)

```python
# src/orchestrator.py (simplified sketch)
"""
The orchestrator wires together the three modules and runs the agent loop.
It does NOT contain agent logic itself — that lives in the communication module.
"""

@dataclass
class OrchestratorConfig:
    max_steps: int = 50
    max_tokens_context: int = 100_000      # Context window budget
    memory_token_budget: int = 20_000      # How much context to give memory
    compression_target: float = 0.6        # Compress to 60% of max when triggered
    model: str = "claude-sonnet-4-20250514"

class Orchestrator:
    def __init__(
        self,
        config: OrchestratorConfig,
        memory: MemoryModule,
        compression: CompressionModule,
        communication: CommunicationModule,
        llm_client: LLMClient,
        tool_executor: ToolExecutor,
        logger: TrajectoryLogger,
    ):
        self.config = config
        self.memory = memory
        self.compression = compression
        self.communication = communication
        self.llm = llm_client
        self.tools = tool_executor
        self.logger = logger

    def run_task(self, task: SWEBenchTask) -> TaskResult:
        """
        Main entry point. Runs the agent on a single SWE-bench task.
        Returns structured result with pass/fail, trajectory, and metrics.
        """
        # 1. Setup sandbox (clone repo, checkout base commit)
        sandbox = self.tools.setup_sandbox(task)

        # 2. Initialize communication module with task description
        self.communication.setup(
            task_description=self._format_task_prompt(task),
            llm_client=self.llm,
        )

        # 3. Main loop
        for step in range(self.config.max_steps):
            # 3a. Check if compression is needed
            trajectory = self.communication.get_trajectory()
            turns = self._trajectory_to_turns(trajectory)
            if self.compression.should_compress(turns, self.config.max_tokens_context):
                compressed = self.compression.compress(
                    turns,
                    target_tokens=int(
                        self.config.max_tokens_context * self.config.compression_target
                    ),
                    llm_client=self.llm,
                )
                self.communication.update_trajectory(compressed)

            # 3b. Run one step of the agent
            step_result = self.communication.run_step(
                llm_client=self.llm,
                tool_executor=self.tools,
            )

            # 3c. Store observations in memory
            for action in step_result["actions_taken"]:
                self.memory.store(MemoryEntry(
                    step=step,
                    entry_type=action["type"],
                    content=action["content"],
                    metadata=action.get("metadata", {}),
                    timestamp=time.time(),
                ))

            # 3d. Log step
            self.logger.log_step(step, step_result)

            # 3e. Check if done
            if step_result["done"]:
                break

        # 4. Extract patch and score
        patch = sandbox.get_diff()
        passed = self._run_tests(sandbox, task)

        return TaskResult(
            task_id=task.instance_id,
            passed=passed,
            patch=patch,
            steps=step + 1,
            trajectory=self.logger.get_full_trajectory(),
            metrics=self._collect_metrics(),
        )

    def _collect_metrics(self) -> dict:
        """Aggregate metrics from all modules."""
        return {
            "memory": self.memory.get_stats(),
            "compression": self.compression.get_stats(),
            "communication": self.communication.get_stats(),
            "llm": self.llm.get_stats(),  # total calls, tokens used, cost
        }
```

---

## 6. Module Implementations (Detailed Specs)

### 6.1 Memory Implementations

#### 6.1.1 Naive (baseline)
- **How it works**: No separate memory. The agent's only "memory" is whatever fits
  in the conversation history. When history exceeds the context window, oldest turns
  are dropped.
- **Implementation**: `get_context_block()` returns empty string. Memory is entirely
  implicit in the conversation history managed by the communication module.
- **Purpose**: Baseline to show that memory modules add value.

#### 6.1.2 Scratchpad
- **How it works**: The agent has a special tool `update_scratchpad(content: str)` that
  writes to a persistent string. Each turn, the current scratchpad is injected into
  the prompt.
- **Implementation**:
  - Maintain a `self.scratchpad: str` that the agent can overwrite.
  - `get_context_block()` returns the scratchpad content (truncated to `max_tokens`).
  - Agent is prompted: "You have a scratchpad for notes. Use it to track your plan,
    findings, and key decisions. You can overwrite it each turn."
- **Key design choice**: The agent controls what to remember. This tests whether the
  LLM can self-manage memory effectively.

#### 6.1.3 RAG (Retrieval-Augmented)
- **How it works**: Every observation/action is embedded and stored in a vector DB.
  Each turn, the current task context is used as a query to retrieve the top-k most
  relevant past entries.
- **Implementation**:
  - Use ChromaDB (local, no server needed) for vector storage.
  - Embed with `sentence-transformers/all-MiniLM-L6-v2`.
  - `store()` embeds and inserts into ChromaDB.
  - `retrieve(query, max_tokens)` does similarity search, returns top entries that fit
    in budget.
  - `get_context_block()` calls `retrieve()` with the current turn's content as query.
- **Key design choice**: Chunk by individual tool results (not by turn). Attach metadata
  (step number, tool name, file path) to each chunk for filtering.

#### 6.1.4 Hybrid (Episodic + Semantic)
- **How it works**: Maintains two stores:
  1. **Episodic**: Raw log of all events, stored in ChromaDB (same as RAG). Retrieved
     by similarity.
  2. **Semantic**: A structured "knowledge base" (dict/JSON) that the agent updates via
     an `update_knowledge(key, value)` tool. Contains distilled facts.
- **Implementation**:
  - Episodic store: identical to RAG implementation.
  - Semantic store: `dict[str, str]` serialized to JSON, injected at top of context each
    turn.
  - Agent is prompted: "You have two memory systems. A knowledge base (always visible)
    for key facts, and a retrieval system that surfaces past observations. Use
    `update_knowledge` to store important discoveries."
  - `get_context_block()` returns: semantic JSON block + top-k episodic retrievals.
- **Key design choice**: Semantic store gives the agent "always on" memory for critical
  facts, while episodic handles the long tail.

### 6.2 Compression Implementations

#### 6.2.1 None (baseline)
- **How it works**: No compression. When context exceeds limit, oldest turns are hard-truncated.
- **Implementation**: `should_compress()` always returns False. The communication module
  handles overflow by dropping oldest turns.
- **Purpose**: Baseline.

#### 6.2.2 Rolling Summary
- **How it works**: When context exceeds threshold, take the oldest N turns (excluding
  landmarks), summarize them into a single "summary turn", and replace them.
- **Implementation**:
  - Trigger when total tokens > 80% of `max_tokens_context`.
  - Select oldest 30% of non-landmark turns.
  - Call LLM with prompt: "Summarize the following agent trajectory into a concise
    summary. Preserve: key decisions, errors encountered, files modified, current
    hypothesis. Be factual and specific."
  - Replace selected turns with one
    `ConversationTurn(role="system", content=summary, is_landmark=True)`.
- **Landmark heuristic**: Mark as landmark: the initial task description, any turn where
  the agent found a bug or error, any turn where the agent modified a file. Landmarks
  are never compressed away.

#### 6.2.3 Hierarchical with Landmarks
- **How it works**: Maintain three levels of context:
  1. **Mission summary** (~500 tokens): High-level task + overall approach. Always present.
  2. **Phase summaries** (~2000 tokens): One paragraph per logical phase of work.
     Always present.
  3. **Recent turns** (last N turns): Full detail for recent work.
  4. **Landmark turns**: Key decision points preserved verbatim.
- **Implementation**:
  - Every 10 steps, ask LLM: "What phase of work did the agent just complete?
    Summarize it in one paragraph."
  - Every 20 steps, ask LLM: "Update the mission summary given the latest phase
    summaries."
  - `compress()` returns:
    `[mission_summary_turn, phase_summary_turns..., landmark_turns..., recent_turns...]`
- **Key design choice**: Phase boundaries are detected by the LLM, not hardcoded.
  More expensive (extra LLM calls) but adapts to different task structures.

### 6.3 Communication Implementations

#### 6.3.1 Single Agent
- **How it works**: One agent with access to all tools. Standard ReAct-style loop.
- **Implementation**:
  - System prompt defines the agent's role, available tools, and task.
  - Each `run_step()`: build prompt (system + memory context + conversation history),
    call LLM, parse tool calls, execute tools, append results.
  - Agent decides it's done by calling a `submit_patch()` tool.
- **Tools available**: `bash`, `read_file`, `write_file`, `edit_file`, `search_code`,
  `list_files`, `submit_patch`, plus memory-specific tools (scratchpad, update_knowledge).

#### 6.3.2 Orchestrated (Planner + Specialists)
- **How it works**:
  - **Planner agent**: Reads the task, makes a plan, delegates subtasks to specialists,
    reviews their work, decides when done.
  - **Explorer agent**: Navigates the repo, reads files, searches for relevant code.
    Returns findings to planner.
  - **Editor agent**: Given specific instructions ("modify function X in file Y to do Z"),
    writes the code changes.
  - **Tester agent**: Runs tests, interprets failures, reports back.
- **Implementation**:
  - Planner has tools: `delegate_to(agent_name, instruction)`, `submit_patch()`.
  - Each specialist has a focused system prompt and limited tools:
    - Explorer: `read_file`, `search_code`, `list_files`, `bash` (read-only commands)
    - Editor: `read_file`, `write_file`, `edit_file`
    - Tester: `bash`, `read_file`
  - Planner sees a summary of each delegation result (not the full specialist trajectory).
  - Each specialist has its own memory module instance.
  - `run_step()` in orchestrated mode = one planner turn, which may trigger 0-N
    specialist runs.
- **Key design choice**: Specialists have independent context windows. This effectively
  multiplies your total context budget, but introduces communication overhead.

#### 6.3.3 Blackboard
- **How it works**: Multiple agents share a "blackboard" (a structured document). Each
  step, one agent is selected to act. It reads the blackboard, does work, and posts
  findings back.
- **Implementation**:
  - Blackboard is a structured document with sections: `task`, `findings`, `hypotheses`,
    `changes_made`, `test_results`, `open_questions`.
  - Agent pool: same roles as orchestrated (explorer, editor, tester) but NO planner.
  - Agent selection: round-robin, or LLM-based ("given the current blackboard state,
    which agent should act next?").
  - Each agent's prompt includes the full blackboard + its role-specific instructions.
  - After acting, agent writes its findings to the appropriate blackboard section.
  - Termination: a `reviewer` agent periodically checks if the task seems solved and
    calls `submit_patch()`.
- **Key design choice**: Decentralized coordination. Agents don't talk to each other —
  they communicate through shared state.

---

## 7. Tools Available to Agents

```python
TOOL_DEFINITIONS = {
    "bash": {
        "description": "Execute a bash command in the sandboxed repo environment.",
        "parameters": {
            "command": {"type": "string", "description": "The command to run."},
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds.",
                "default": 30,
            },
        },
    },
    "read_file": {
        "description": "Read contents of a file. Returns the file content with line numbers.",
        "parameters": {
            "path": {"type": "string"},
            "start_line": {"type": "integer", "optional": True},
            "end_line": {"type": "integer", "optional": True},
        },
    },
    "write_file": {
        "description": "Write content to a file. Overwrites existing content.",
        "parameters": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
    },
    "edit_file": {
        "description": "Replace a specific string in a file (like str_replace).",
        "parameters": {
            "path": {"type": "string"},
            "old_str": {"type": "string"},
            "new_str": {"type": "string"},
        },
    },
    "search_code": {
        "description": "Search for a pattern across the repo using ripgrep.",
        "parameters": {
            "pattern": {"type": "string"},
            "file_glob": {"type": "string", "optional": True},
        },
    },
    "list_files": {
        "description": "List files in a directory, optionally recursive.",
        "parameters": {
            "path": {"type": "string"},
            "recursive": {"type": "boolean", "default": False},
            "max_depth": {"type": "integer", "default": 2},
        },
    },
    "submit_patch": {
        "description": (
            "Signal that the agent is done and the current repo state "
            "should be evaluated."
        ),
        "parameters": {},
    },
    # Memory-specific tools (injected based on memory module config)
    "update_scratchpad": {
        "description": "Overwrite your scratchpad with new notes.",
        "parameters": {"content": {"type": "string"}},
    },
    "update_knowledge": {
        "description": "Update a key-value pair in your knowledge base.",
        "parameters": {
            "key": {"type": "string"},
            "value": {"type": "string"},
        },
    },
}
```

---

## 8. Configuration System

Each experiment is defined by a YAML config:

```yaml
# configs/rag_hierarchical_single.yaml
experiment_name: "rag_memory_hierarchical_compression_single_agent"

model: "claude-sonnet-4-20250514"

orchestrator:
  max_steps: 50
  max_tokens_context: 100000
  memory_token_budget: 20000
  compression_target: 0.6

memory:
  type: "rag"                     # naive | scratchpad | rag | hybrid
  params:
    embedding_model: "all-MiniLM-L6-v2"
    top_k: 20
    chunk_strategy: "per_tool_result"

compression:
  type: "hierarchical"            # none | rolling_summary | hierarchical
  params:
    trigger_threshold: 0.8        # Compress at 80% of context budget
    phase_interval: 10            # Detect phases every 10 steps
    mission_update_interval: 20
    max_landmarks: 15

communication:
  type: "single_agent"            # single_agent | orchestrated | blackboard
  params: {}

evaluation:
  dataset: "swebench_lite"
  task_ids: null                  # null = all tasks, or list of specific IDs
  num_tasks: 50                   # Random sample if task_ids is null
  seed: 42

sandbox:
  docker_image: "swebench-sandbox:latest"
  timeout_per_task: 600           # 10 minutes per task
```

---

## 9. Evaluation & Analysis

### Metrics to Collect Per Task

```python
@dataclass
class TaskResult:
    task_id: str
    passed: bool                      # Did the patch pass the test suite?
    steps: int                        # Number of agent steps taken
    wall_time_seconds: float          # Total wall clock time
    llm_calls: int                    # Total LLM API calls
    total_input_tokens: int           # Total tokens sent to LLM
    total_output_tokens: int          # Total tokens received from LLM
    estimated_cost_usd: float         # Estimated API cost
    compression_events: int           # How many times compression was triggered
    memory_entries_stored: int        # Total entries in memory at end
    memory_retrievals: int            # Total retrieval queries made
    agent_messages: int               # Inter-agent messages (0 for single agent)
    trajectory: list[dict]            # Full trajectory for qualitative analysis
    error: str | None                 # If the run crashed, why
```

### Ablation Matrix

Phase 1 ablation (single agent, vary memory x compression):

```
                  | none_compress | rolling_summary | hierarchical
naive_memory      |    run_01     |     run_02      |    run_03
scratchpad_memory |    run_04     |     run_05      |    run_06
rag_memory        |    run_07     |     run_08      |    run_09
hybrid_memory     |    run_10     |     run_11      |    run_12
```

Phase 2 ablation (best memory x compression from phase 1, vary communication):

```
best_config + single_agent   → run_13
best_config + orchestrated   → run_14
best_config + blackboard     → run_15
```

### Analysis Script Output
- Solve rate per config (with 95% confidence intervals)
- Median steps to solve (for solved tasks only)
- Cost per task (median, p90)
- Solve rate by task difficulty tier (easy/medium/hard)
- Pairwise statistical significance tests (McNemar's test for solve rate)
- Failure mode categorization (wrong file, wrong fix, timeout, crash, etc.)

---

## 10. Implementation Plan (Phased)

### Phase 1: Foundation (Week 1)

**Goal**: A single-agent baseline that can attempt SWE-bench tasks end-to-end.

#### Day 1: Project setup
- Initialize Python project with pyproject.toml
- Dependencies: anthropic, chromadb, sentence-transformers, docker, pyyaml, tiktoken,
  pytest, datasets, pandas, scipy, matplotlib, rich
- Set up directory structure as specified in Section 3
- Set up basic CLI: `python -m src.runner --config configs/baseline.yaml --task <task_id>`

#### Day 1-2: SWE-bench integration
- `swebench_loader.py`: download SWE-bench Lite from HuggingFace `datasets`, parse into
  `SWEBenchTask` dataclass (instance_id, repo, base_commit, problem_statement,
  test_patch, hints)
- `sandbox.py`: Docker-based sandbox that clones repo at correct commit, applies test
  patch, runs test suite. Use the `docker` Python SDK.
- `scorer.py`: apply agent's patch to fresh sandbox, run tests, report pass/fail
- **Validation**: manually verify sandbox works for 3 different SWE-bench tasks by
  applying the gold patch and confirming tests pass

#### Day 2: LLM Client
- `llm_client.py`: wrapper around Anthropic API
  - Support for tool use (function calling)
  - Token counting via tiktoken
  - Cost tracking (input/output tokens x model price)
  - Exponential backoff retry (3 attempts)
  - Rate limit handling
  - Metrics: track total calls, tokens, cost across a run
- **Validation**: verify a tool-use round trip (send message with tools, get tool_use
  response, send tool_result, get final response)

#### Day 2-3: Tool Executor
- `tool_executor.py`: routes tool call names to sandbox operations
  - `bash` → `sandbox.exec(command, timeout)`
  - `read_file` → `sandbox.read_file(path, start_line, end_line)`
  - `write_file` → `sandbox.write_file(path, content)`
  - `edit_file` → `sandbox.edit_file(path, old_str, new_str)`
  - `search_code` → `sandbox.exec(f"rg {pattern} {glob}")`
  - `list_files` → `sandbox.exec(f"find {path} -maxdepth {depth}")`
  - `submit_patch` → signal done
- Output truncation: cap tool output at 10,000 tokens. If exceeded, show first 4000 +
  "[... truncated {N} tokens ...]" + last 4000.
- Error handling: catch timeouts, format errors nicely for the agent.
- **Validation**: verify each tool works against a real sandbox

#### Day 3-4: Single Agent Communication Module
- `single_agent.py`:
  - Manages conversation history (list of messages)
  - `setup()`: initializes system prompt + first user message with task description
  - `run_step()`:
    1. Inject memory context block into system prompt or as a prefixed user message
    2. Call LLM with conversation history + tools
    3. Parse response: if tool_use, execute tool, append tool_result
    4. If text-only response (no tool), append and continue
    5. If `submit_patch` called, set done=True
    6. Return step result dict
  - `get_trajectory()`: return conversation history
  - `update_trajectory(compressed)`: replace history with compressed version
- System prompt template (store in `src/prompts/single_agent.txt`):
  ```
  You are a software engineer tasked with fixing a bug in a repository.
  You have access to tools to navigate and modify the codebase.

  Strategy:
  1. Read the issue carefully and understand what's expected
  2. Explore the repository structure to find relevant files
  3. Search for related code and understand the current behavior
  4. Develop a fix and implement it
  5. Test your fix to make sure it works
  6. Call submit_patch when you're confident in your solution

  Be methodical. Think step by step. If something doesn't work, re-read
  the error and try a different approach.
  ```
- **Validation**: run on 3 easy SWE-bench tasks, verify it produces patches

#### Day 4: Naive Memory + No Compression
- `naive.py`: all methods are no-ops or return empty
- `none.py`: `should_compress()` returns False, `compress()` just truncates from front
- Wire into orchestrator

#### Day 4-5: Orchestrator + Logger
- `orchestrator.py`: implement `run_task()` as specified in Section 5
- `logger.py`:
  - `log_step(step_num, step_result)`: append to in-memory list
  - `get_full_trajectory()`: return complete log
  - `save(path)`: write JSON to disk
  - Include timestamps, token counts, and cost per step
- **Milestone**: `python -m src.runner --config configs/baseline.yaml` runs a task
  end-to-end and reports pass/fail with full trajectory saved to JSON

#### Day 5: Config system + end-to-end validation
- Implement YAML config loading in `runner.py`
- Module registry: map config strings to classes
  ```python
  MEMORY_REGISTRY = {
      "naive": NaiveMemory,
      "scratchpad": ScratchpadMemory,
      "rag": RAGMemory,
      "hybrid": HybridMemory,
  }
  # Same pattern for compression and communication
  ```
- Write `configs/baseline.yaml` (naive memory, no compression, single agent)
- Run on 5 tasks end-to-end. Fix any crashes. Verify scoring works.

---

### Phase 2: Memory Variants (Week 2, Days 1-3)

**Goal**: Implement all 4 memory modules and verify they work.

#### Day 1: Scratchpad memory
- Implement `scratchpad.py`:
  - `self.scratchpad: str = ""`
  - `store()`: no-op (agent manages its own scratchpad via tool)
  - `retrieve()`: returns scratchpad content
  - `get_context_block()`: returns formatted scratchpad in XML tags
  - `clear()`: reset scratchpad to empty
  - `get_stats()`: return scratchpad length, number of updates
- Register `update_scratchpad` tool in tool executor when scratchpad memory is active
- Add to system prompt: "You have a scratchpad for persistent notes. Use
  `update_scratchpad` to save your plan, key findings, and important details. The
  scratchpad content is shown to you each turn. Manage it actively — overwrite with
  updated notes rather than appending endlessly."
- **Validation**: run 3 tasks, inspect trajectories to confirm scratchpad is being used

#### Day 1-2: RAG memory
- Implement `rag.py`:
  - `__init__()`: initialize ChromaDB client (ephemeral, in-memory), create collection,
    load sentence-transformer model
  - `store(entry)`: format, embed, insert into ChromaDB with metadata
  - `retrieve(query, max_tokens)`: embed query, search top-k, accumulate within budget
  - `get_context_block()`: call retrieve with current turn content as query, format results
  - `clear()`: delete and recreate ChromaDB collection
  - `get_stats()`: entries stored, total retrievals, avg retrieval time
- **Validation**: unit test retrieval quality (store 50 entries about different files,
  query "error in parser", verify parser-related entries rank highest). Then 3 tasks e2e.

#### Day 2-3: Hybrid memory
- Implement `hybrid.py`:
  - Combines RAG (episodic) + dict (semantic)
  - `self.knowledge_base: dict[str, str] = {}`
  - `self.episodic_store: RAGMemory` (reuse RAG implementation)
  - Allocate 30% token budget to semantic, 70% to episodic
  - Register `update_knowledge` tool
  - System prompt additions explaining both memory systems
- **Validation**: run 3 tasks, inspect that both stores are populated

#### Day 3: Memory module tests
- `test_memory.py`: store/retrieve cycle, token budget compliance, clear, get_stats
- Integration test: same task with each memory module, all produce valid patches

---

### Phase 3: Compression Variants (Week 2, Days 3-5)

**Goal**: Implement all 3 compression modules.

#### Day 3-4: Rolling summary
- Implement `rolling_summary.py`:
  - Trigger at 80% context budget
  - Select oldest 30% non-landmark turns
  - Summarize via LLM call
  - Replace with summary turn
  - Landmark detection heuristic (errors, file edits, test results)
- **Validation**: run a 30+ step task, verify compression triggers and summary quality

#### Day 4-5: Hierarchical compression
- Implement `hierarchical.py`:
  - Phase detection every 10 steps via LLM
  - Mission summary update every 20 steps via LLM
  - Context assembly: mission → phases → landmarks → recent
- **Validation**: run a 40+ step task, inspect hierarchy at various points

#### Day 5: Compression tests
- Token budget compliance, landmark preservation, correct trigger thresholds

---

### Phase 4: Multi-Agent Communication (Week 3, Days 1-3)

**Goal**: Implement orchestrated and blackboard communication modules.

#### Day 1-2: Orchestrated multi-agent
- Implement `orchestrated.py`:
  - Planner with `delegate_to(agent_name, instruction)` tool
  - Specialist agents: explorer, editor, tester (each with focused tools + own memory)
  - Specialist runs capped at 10 steps, results summarized to 1000 tokens for planner
  - Edge cases: specialist timeout, unknown agent, specialist crash
- **Validation**: 3 tasks, verify delegation and specialist execution

#### Day 2-3: Blackboard multi-agent
- Implement `blackboard.py`:
  - Structured blackboard document with typed sections
  - Agent pool: explorer, editor, tester, reviewer (no planner)
  - Round-robin agent selection (optional: LLM-based)
  - Reviewer checks for completion periodically
- **Validation**: 3 tasks, inspect blackboard evolution

#### Day 3: Communication module tests
- All modules can initialize, run, and terminate
- Correct metrics (llm_calls, agent_messages)

---

### Phase 5: Ablation Study (Week 3, Days 3-5 + Week 4, Days 1-2)

**Goal**: Run systematic experiments and collect results.

#### Day 3: Experiment runner
- `experiments/run_ablation.py`: config matrix, resume support, optional parallelism
- Generate all 12 Phase 1 configs programmatically
- `--dry-run` flag

#### Day 3-4: Validation run
- 12 configs x 5 tasks = 60 runs
- Fix crashes, estimate cost for full run

#### Day 4-5: Phase 1 full ablation
- 12 configs x 50 tasks = 600 runs
- Monitor with rich progress bar

#### Week 4, Day 1: Phase 2 ablation
- Best memory x compression combo + 3 communication variants x 50 tasks = 150 runs

#### Week 4, Day 2: Results consolidation
- Verify completeness, re-run failures, consolidate to analysis dataset

---

### Phase 6: Analysis & Writeup (Week 4, Days 2-5)

**Goal**: Analyze results and produce compelling writeup.

#### Day 2-3: Analysis script
- `src/evaluation/analysis.py`: solve rates, confidence intervals, McNemar's test,
  cost analysis, difficulty breakdown
- Generate plots: bar charts, scatter (cost vs solve rate), heatmap (ablation matrix)

#### Day 3-4: Failure mode analysis
- Auto-categorize: crash, timeout, no_attempt, wrong_file, wrong_fix, other
- Manual inspection: 5 trajectories per failure mode for best/worst configs

#### Day 4-5: Writeup
- `docs/writeup.md`: motivation, framework design, experiments, results, analysis,
  lessons learned
- Target: 3-5 pages, dense and technical
- This writeup + GitHub repo = application project

---

## 11. Key Implementation Notes for Claude Code

### Dependencies (pyproject.toml)

```toml
[project]
name = "agent-harness"
requires-python = ">=3.11"
dependencies = [
    "anthropic>=0.40.0",
    "chromadb>=0.5.0",
    "sentence-transformers>=3.0.0",
    "tiktoken>=0.7.0",
    "pyyaml>=6.0",
    "docker>=7.0.0",
    "datasets>=2.0.0",
    "pandas>=2.0.0",
    "scipy>=1.10.0",
    "matplotlib>=3.7.0",
    "rich>=13.0.0",
    "pytest>=8.0.0",
]
```

### Testing Strategy
- Unit tests for each module in isolation (mock the LLM client with deterministic
  responses)
- Integration tests that run real LLM calls on 1-2 simple SWE-bench tasks
- Mark integration tests with `@pytest.mark.integration` so they can be skipped
- Run unit tests in CI, integration tests manually before ablation

### Error Handling Priorities
- **LLM API failures**: exponential backoff with 3 retries, jitter
- **Sandbox crashes**: catch, log, mark task as "error", continue ablation
- **Token limit exceeded**: compress aggressively; if still over, truncate and log warning
- **Agent infinite loops**: detect via (a) step limit AND (b) repeated identical actions
  (same tool + same args 3x in a row → force termination)
- **Malformed tool calls**: return error message to agent, do not crash

### Prompt Engineering Notes
- Keep system prompts under 2000 tokens. Put task-specific info in user message.
- Use extended thinking if available (helps with planning steps)
- Tool output formatting: truncate at 10,000 tokens with
  "[truncated, showing first/last N lines]"
- For multi-agent: specialist summaries max 1000 tokens returned to planner
- Include in system prompt: "If you're stuck, re-read the original issue and try a
  different approach. Don't repeat the same action more than twice."

### Cost Optimization
- Use `claude-sonnet-4-20250514` for all agent steps
- For compression summarization calls, consider using `claude-haiku-4-5-20251001`
- Cache deterministic tool outputs (e.g., `list_files` same path = same result)
- Set aggressive timeouts: 30s per tool call, 600s per task, 7200s per ablation row
- Run a small pilot (5 tasks x 3 configs) first to estimate total cost

### Git Strategy
- Commit after each phase milestone
- Tag releases: v0.1 (baseline), v0.2 (memory), v0.3 (compression),
  v0.4 (multi-agent), v0.5 (ablation complete), v1.0 (writeup done)
- Keep experiment configs and results in the repo for reproducibility