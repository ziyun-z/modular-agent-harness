"""
Orchestrated multi-agent communication module.

Architecture
------------
                    ┌────────────────────────────────────┐
                    │  Planner Agent                     │
                    │  - Plans and decomposes the task   │
                    │  - Delegates sub-tasks via tool    │
                    │  - Reads specialist summaries      │
                    │  - Calls submit_patch when done    │
                    └──────────────┬─────────────────────┘
                                   │  delegate_task(task=...)
                                   ▼
                    ┌────────────────────────────────────┐
                    │  Specialist Agent (per delegation) │
                    │  - Runs for up to max_steps        │
                    │  - Has all code tools              │
                    │  - Returns a detailed summary      │
                    └────────────────────────────────────┘

Planner tools   : delegate_task, submit_patch
Specialist tools: bash, read_file, write_file, edit_file, search_code, list_files

Only the planner's conversation is tracked as ConversationTurns (for compression).
Specialist conversations are ephemeral — discarded after each delegation.

Config params (passed as `params:` in YAML):
    specialist_max_steps          int   10    max LLM calls per specialist run
    specialist_summary_max_tokens int  1000   max tokens for specialist summary
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

from src.communication.base import CommunicationModule
from src.compression.base import ConversationTurn
from src.tool_executor import PatchSubmitted, TOOL_DEFINITIONS

if TYPE_CHECKING:
    from src.llm_client import LLMClient
    from src.tool_executor import ToolExecutor
    from src.memory.base import MemoryModule
    from src.sandbox import DockerSandbox

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

_DELEGATE_TOOL: dict[str, Any] = {
    "name": "delegate_task",
    "description": (
        "Delegate a sub-task to a specialist worker agent who has access to all "
        "code tools (bash, read_file, write_file, edit_file, search_code, list_files). "
        "The specialist will complete the task and return a detailed summary of "
        "findings, changes made, and any errors encountered. "
        "Use this for: exploring the codebase, reading or editing files, "
        "running tests, and verifying fixes. "
        "You cannot use code tools directly — all code work must be delegated."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": (
                    "Detailed description of the sub-task. Be specific: include "
                    "file paths, function names, and expected outcomes when known. "
                    "State clearly whether the specialist should only read/explore "
                    "or also make changes."
                ),
            },
        },
        "required": ["task"],
    },
}

# Specialist tool names — standard tools minus submit_patch and delegate_task
_SPECIALIST_TOOL_NAMES: frozenset[str] = frozenset(
    {"bash", "read_file", "write_file", "edit_file", "search_code", "list_files"}
)

# submit_patch definition (pulled from the shared TOOL_DEFINITIONS list)
_SUBMIT_PATCH_DEF: list[dict[str, Any]] = [
    t for t in TOOL_DEFINITIONS if t["name"] == "submit_patch"
]

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_PLANNER_SYSTEM = """\
You are a lead software engineer coordinating a team to fix a bug in a repository.

You have two tools:
  - delegate_task: send a sub-task to a specialist worker who has full code access
  - submit_patch: signal that the fix is complete

You CANNOT use code tools (bash, read_file, etc.) directly — all code work must
be delegated. Each delegate_task call returns a detailed summary of what the
specialist found or changed.

Recommended workflow:
  1. Delegate exploration to understand the bug and relevant code.
  2. Delegate code editing to implement the fix.
  3. Delegate test execution to verify the fix works.
  4. Call submit_patch once tests pass.

Be specific in task descriptions: include file paths, function names, and line
numbers when known. Do not re-delegate tasks that have already been completed.\
"""

_SPECIALIST_SYSTEM = """\
You are a specialist software engineering assistant working on a specific sub-task.
Complete the task thoroughly using the available tools.

When you have finished, end your response with a comprehensive summary that includes:
  - What you explored and found (file paths, function names, line numbers)
  - What changes you made (exact files and edits, if any)
  - Test results (if you ran tests)
  - Any issues or uncertainties encountered

Be specific and concise in your summary.\
"""

_SYNTHESIS_PROMPT = (
    "Summarize everything you found and any changes you made in 2–3 paragraphs. "
    "Include specific file paths, function names, error messages, and test results."
)


# ---------------------------------------------------------------------------
# OrchestratedCommunication
# ---------------------------------------------------------------------------


class OrchestratedCommunication(CommunicationModule):
    """
    Planner + specialist multi-agent architecture.

    The planner decides what to do; specialists execute all code operations.
    The planner's conversation is tracked for compression; specialist
    conversations are discarded after each delegation.
    """

    def __init__(
        self,
        specialist_max_steps: int = 10,
        specialist_summary_max_tokens: int = 1000,
    ) -> None:
        self._specialist_max_steps = specialist_max_steps
        self._specialist_summary_max_tokens = specialist_summary_max_tokens

        # Planner state (reset on each setup())
        self._planner_messages: list[dict[str, Any]] = []
        self._planner_turns: list[ConversationTurn] = []
        self._task_message: str = ""
        self._sandbox: Optional["DockerSandbox"] = None
        self._memory: Optional["MemoryModule"] = None
        self._step: int = 0

        # Stats
        self._delegations: int = 0
        self._specialist_steps_total: int = 0
        self._specialist_llm_calls_total: int = 0

    # ------------------------------------------------------------------
    # CommunicationModule interface
    # ------------------------------------------------------------------

    def setup(
        self,
        task_description: str,
        llm_client: "LLMClient",
        sandbox: Optional["DockerSandbox"] = None,
    ) -> None:
        """Initialise planner conversation for a new task."""
        self._task_message = task_description
        self._planner_messages = [{"role": "user", "content": task_description}]
        self._planner_turns = []
        self._step = 0
        self._delegations = 0
        self._specialist_steps_total = 0
        self._specialist_llm_calls_total = 0
        if sandbox is not None:
            self._sandbox = sandbox
        logger.debug("OrchestratedCommunication set up for new task.")

    def set_sandbox(self, sandbox: "DockerSandbox") -> None:
        """Inject the sandbox after setup (called by orchestrator)."""
        self._sandbox = sandbox

    def set_memory(self, memory: "MemoryModule") -> None:
        """Inject the memory module after setup (called by orchestrator)."""
        self._memory = memory

    def run_step(
        self,
        llm_client: "LLMClient",
        tool_executor: "ToolExecutor",
    ) -> dict[str, Any]:
        """
        Execute one planner step (one LLM call + zero or more specialist runs).

        Returns:
            {
                "actions_taken": flat list of all tool calls (planner + specialists),
                "done":          True if planner called submit_patch,
                "result":        submit_patch message (or None),
                "llm_calls":     total LLM calls made this step,
            }
        """
        if self._sandbox is None:
            raise RuntimeError(
                "Sandbox not set. Call setup(..., sandbox=...) or set_sandbox() first."
            )

        self._step += 1
        llm_call_count = 0

        # Build planner system prompt (base + optional memory)
        system = self._build_planner_system()

        # Planner's available tools
        planner_tools = [_DELEGATE_TOOL] + _SUBMIT_PATCH_DEF

        # Planner LLM call
        response = llm_client.complete(
            messages=self._planner_messages,
            tools=planner_tools,
            system=system,
        )
        llm_call_count += 1
        logger.debug(
            "Planner step %d — stop_reason=%s  blocks=%d",
            self._step, response.stop_reason, len(response.content),
        )

        # Append raw assistant response
        self._planner_messages.append({"role": "assistant", "content": response.content})

        # Process planner response
        text_parts: list[str] = []
        tool_results: list[dict[str, Any]] = []
        actions_taken: list[dict[str, Any]] = []
        done = False
        result_message: Optional[str] = None

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)

            elif block.type == "tool_use":
                if block.name == "delegate_task":
                    task_desc = block.input.get("task", "")
                    summary, spec_actions, spec_steps, spec_llm = self._run_specialist(
                        task_desc, llm_client, tool_executor
                    )
                    self._delegations += 1
                    self._specialist_steps_total += spec_steps
                    self._specialist_llm_calls_total += spec_llm
                    llm_call_count += spec_llm

                    delegation_result = f"Specialist completed task.\n\n{summary}"
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": delegation_result,
                    })
                    actions_taken.append({
                        "tool": "delegate_task",
                        "input": block.input,
                        "output": summary,
                    })
                    actions_taken.extend(spec_actions)
                    logger.info(
                        "Planner step %d — delegation #%d done (%d spec steps, %d spec calls)",
                        self._step, self._delegations, spec_steps, spec_llm,
                    )

                elif block.name == "submit_patch":
                    try:
                        tool_executor.execute("submit_patch", block.input, self._sandbox)
                    except PatchSubmitted as exc:
                        done = True
                        result_message = str(exc)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": f"Patch submitted. {exc}".strip(),
                        })
                        actions_taken.append({
                            "tool": "submit_patch",
                            "input": block.input,
                            "output": result_message,
                        })
                        logger.info(
                            "Planner step %d — submit_patch called: %s",
                            self._step, exc,
                        )
                        break  # no more processing after submit

        # Append tool results as a user message
        if tool_results:
            self._planner_messages.append({"role": "user", "content": tool_results})

        # Record ConversationTurns for the compression module
        assistant_text = "\n".join(text_parts)
        if assistant_text or tool_results:
            tc = llm_client.count_tokens([{"role": "assistant", "content": assistant_text}])
            self._planner_turns.append(ConversationTurn(
                role="assistant",
                content=assistant_text,
                step=self._step,
                is_landmark=done,
                token_count=tc,
            ))

        if tool_results:
            results_text = "\n---\n".join(r["content"] for r in tool_results)
            tc = llm_client.count_tokens([{"role": "user", "content": results_text}])
            self._planner_turns.append(ConversationTurn(
                role="tool_result",
                content=results_text,
                step=self._step,
                is_landmark=done,
                token_count=tc,
            ))

        return {
            "actions_taken": actions_taken,
            "done": done,
            "result": result_message,
            "llm_calls": llm_call_count,
        }

    def get_trajectory(self) -> list[dict]:
        """Return planner ConversationTurn list for the compression module."""
        return self._planner_turns  # type: ignore[return-value]

    def update_trajectory(self, compressed_turns: list) -> None:
        """Replace planner turn list with compressed version; rebuild messages."""
        self._planner_turns = list(compressed_turns)
        self._rebuild_planner_messages()

    def get_stats(self) -> dict[str, Any]:
        return {
            "type": "orchestrated",
            "planner_steps": self._step,
            "delegations": self._delegations,
            "specialist_steps_total": self._specialist_steps_total,
            "specialist_llm_calls_total": self._specialist_llm_calls_total,
        }

    # ------------------------------------------------------------------
    # Specialist runner
    # ------------------------------------------------------------------

    def _run_specialist(
        self,
        task: str,
        llm_client: "LLMClient",
        tool_executor: "ToolExecutor",
    ) -> tuple[str, list[dict[str, Any]], int, int]:
        """
        Run a specialist agent for up to specialist_max_steps steps.

        Returns:
            (summary_text, actions_list, steps_taken, llm_call_count)
        """
        # Build specialist tool list (all standard tools except submit_patch)
        specialist_tools = [
            t for t in tool_executor.tool_definitions
            if t["name"] in _SPECIALIST_TOOL_NAMES
        ]

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": f"Sub-task: {task}"}
        ]
        actions: list[dict[str, Any]] = []
        steps = 0
        llm_calls = 0

        for _ in range(self._specialist_max_steps):
            steps += 1
            response = llm_client.complete(
                messages=messages,
                tools=specialist_tools,
                system=_SPECIALIST_SYSTEM,
            )
            llm_calls += 1
            messages.append({"role": "assistant", "content": response.content})

            tool_results: list[dict[str, Any]] = []
            for block in response.content:
                if block.type == "tool_use":
                    try:
                        output = tool_executor.execute(
                            block.name, block.input, self._sandbox
                        )
                    except PatchSubmitted:
                        # Should not happen (no submit_patch in specialist tools)
                        output = "[submit_patch not available to specialist]"
                    except Exception as exc:
                        output = f"Error: {exc}"

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": output,
                    })
                    actions.append({
                        "tool": block.name,
                        "input": block.input,
                        "output": output,
                    })

            if tool_results:
                messages.append({"role": "user", "content": tool_results})

            if response.stop_reason == "end_turn":
                # Specialist finished naturally
                break

        # Extract summary from the last assistant text or synthesize one
        summary, synthesized = self._extract_summary(messages, llm_client)
        if synthesized:
            llm_calls += 1

        logger.debug(
            "Specialist done: %d steps, %d tool calls, synthesized=%s",
            steps, len(actions), synthesized,
        )
        return summary, actions, steps, llm_calls

    def _extract_summary(
        self,
        specialist_messages: list[dict[str, Any]],
        llm_client: "LLMClient",
    ) -> tuple[str, bool]:
        """
        Try to extract a summary from the specialist's last text response.
        If no text is found, make a synthesis LLM call.

        Returns: (summary_text, was_synthesized)
        """
        # Walk backwards through messages looking for the last assistant text block
        for msg in reversed(specialist_messages):
            if msg.get("role") != "assistant":
                continue
            content = msg["content"]
            # content is a list of Anthropic content blocks
            if isinstance(content, list):
                text_blocks = [
                    b for b in content
                    if hasattr(b, "type") and b.type == "text" and b.text.strip()
                ]
                if text_blocks:
                    return text_blocks[-1].text, False
            elif isinstance(content, str) and content.strip():
                return content, False
            break  # found an assistant message but no usable text

        # No text found — ask for a synthesis
        synthesis_messages = specialist_messages + [
            {"role": "user", "content": _SYNTHESIS_PROMPT}
        ]
        response = llm_client.complete(
            messages=synthesis_messages,
            system=_SPECIALIST_SYSTEM,
            max_tokens=self._specialist_summary_max_tokens,
        )
        text_blocks = [b for b in response.content if b.type == "text"]
        if text_blocks:
            return text_blocks[0].text, True
        return "[specialist summary unavailable]", True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_planner_system(self) -> str:
        """Combine planner system prompt with optional memory context block."""
        if self._memory is None:
            return _PLANNER_SYSTEM
        memory_block = self._memory.get_context_block(max_tokens=2_000)
        if not memory_block:
            return _PLANNER_SYSTEM
        return f"{_PLANNER_SYSTEM}\n\n## Memory Context\n{memory_block}"

    def _rebuild_planner_messages(self) -> None:
        """
        Reconstruct _planner_messages from _planner_turns after compression.

        Mirrors the same logic as SingleAgentCommunication._rebuild_messages().
        Consecutive same-role turns are merged to keep valid user/assistant alternation.
        """
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": self._task_message}
        ]
        for turn in self._planner_turns:
            role = "assistant" if turn.role == "assistant" else "user"
            content = turn.content
            if messages and messages[-1]["role"] == role:
                messages[-1]["content"] = str(messages[-1]["content"]) + "\n\n" + content
            else:
                messages.append({"role": role, "content": content})

        self._planner_messages = messages
        logger.debug(
            "Rebuilt planner messages from %d compressed turns → %d messages",
            len(self._planner_turns), len(self._planner_messages),
        )
