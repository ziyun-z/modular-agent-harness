"""
Single-agent communication module.

One agent with access to all tools. Standard ReAct-style agentic loop:
  - System prompt defines the agent role and task.
  - Each run_step(): call LLM → parse tool uses → execute tools → loop.
  - Agent signals completion by calling the submit_patch tool.

Conversation history is stored in two parallel forms:
  - _messages: Anthropic API format (list[dict]), used for LLM calls.
  - _turns: list[ConversationTurn], used by the compression module.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

from src.communication.base import CommunicationModule
from src.compression.base import ConversationTurn
from src.tool_executor import PatchSubmitted

if TYPE_CHECKING:
    from src.llm_client import LLMClient
    from src.tool_executor import ToolExecutor
    from src.memory.base import MemoryModule
    from src.sandbox import DockerSandbox

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "single_agent.txt"


class SingleAgentCommunication(CommunicationModule):
    """
    Single-agent ReAct loop.

    Usage (with orchestrator)::

        comm = SingleAgentCommunication(memory=memory_module)
        comm.setup(task_description, llm_client, sandbox=sandbox)

        while not done:
            if compression.should_compress(comm.get_trajectory(), limit):
                comm.update_trajectory(compression.compress(...))
            result = comm.run_step(llm_client, tool_executor)
            done = result["done"]

        diff = sandbox.get_diff()
    """

    def __init__(
        self,
        memory: Optional["MemoryModule"] = None,
        compression=None,           # accepted but not used (orchestrator drives it)
    ) -> None:
        self._memory = memory

        # State reset by setup()
        self._system_base: str = _load_prompt_template()
        self._messages: list[dict[str, Any]] = []
        self._turns: list[ConversationTurn] = []
        self._task_message: str = ""
        self._sandbox: Optional["DockerSandbox"] = None
        self._step: int = 0

    # ------------------------------------------------------------------
    # CommunicationModule interface
    # ------------------------------------------------------------------

    def setup(
        self,
        task_description: str,
        llm_client: "LLMClient",
        sandbox: Optional["DockerSandbox"] = None,
    ) -> None:
        """
        Initialise for a new task.

        Args:
            task_description: Formatted task prompt to send as the first user message.
            llm_client:       Used only to pre-compute token counts for turns.
            sandbox:          Running DockerSandbox; required before run_step() calls.
        """
        self._task_message = task_description
        self._messages = [{"role": "user", "content": task_description}]
        self._turns = []
        self._step = 0
        self._sandbox = sandbox
        logger.debug("SingleAgentCommunication set up for new task.")

    def set_sandbox(self, sandbox: "DockerSandbox") -> None:
        """Inject the sandbox after setup (used by orchestrator)."""
        self._sandbox = sandbox

    def set_memory(self, memory: "MemoryModule") -> None:
        """Inject the memory module after setup (used by orchestrator)."""
        self._memory = memory

    def run_step(
        self,
        llm_client: "LLMClient",
        tool_executor: "ToolExecutor",
    ) -> dict[str, Any]:
        """
        Execute one agent step (one LLM call + all resulting tool calls).

        Returns:
            {
                "actions_taken": list of {tool, input, output} dicts,
                "done":          True if the agent called submit_patch,
                "result":        submit_patch message (or None),
                "llm_calls":     number of LLM calls made this step (always 1),
            }
        """
        if self._sandbox is None:
            raise RuntimeError(
                "Sandbox not set. Call setup(..., sandbox=sandbox) or set_sandbox() first."
            )

        self._step += 1

        # 1. Build system prompt: base template + optional memory context block
        system = self._build_system_prompt()

        # 2. Call LLM
        response = llm_client.complete(
            messages=self._messages,
            tools=tool_executor.tool_definitions,
            system=system,
        )
        logger.debug(
            "Step %d — stop_reason=%s  blocks=%d",
            self._step,
            response.stop_reason,
            len(response.content),
        )

        # 3. Append the raw assistant message (TextBlock/ToolUseBlock objects)
        self._messages.append({"role": "assistant", "content": response.content})

        # 4. Process content blocks
        text_parts: list[str] = []
        tool_results: list[dict[str, Any]] = []
        actions_taken: list[dict[str, Any]] = []
        done = False
        result_message: Optional[str] = None

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)

            elif block.type == "tool_use":
                try:
                    tool_output = tool_executor.execute(
                        block.name, block.input, self._sandbox
                    )
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": tool_output,
                    })
                    actions_taken.append({
                        "tool": block.name,
                        "input": block.input,
                        "output": tool_output,
                    })
                    logger.debug("Tool %s → %d chars", block.name, len(tool_output))

                except PatchSubmitted as e:
                    done = True
                    result_message = str(e)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"Patch submitted. {e}".strip(),
                    })
                    actions_taken.append({
                        "tool": "submit_patch",
                        "input": block.input,
                        "output": result_message,
                    })
                    logger.info("Step %d — agent called submit_patch: %s", self._step, e)
                    break   # no more tool calls after submit

        # 5. Append tool results as a single user message (Anthropic format)
        if tool_results:
            self._messages.append({"role": "user", "content": tool_results})

        # 6. Record ConversationTurns for the compression module
        assistant_text = "\n".join(text_parts)
        if assistant_text or tool_results:
            token_count = llm_client.count_tokens(
                [{"role": "assistant", "content": assistant_text}]
            )
            self._turns.append(ConversationTurn(
                role="assistant",
                content=assistant_text,
                step=self._step,
                is_landmark=done,           # landmarks are preserved during compression
                token_count=token_count,
            ))

        if tool_results:
            results_text = "\n---\n".join(r["content"] for r in tool_results)
            token_count = llm_client.count_tokens(
                [{"role": "user", "content": results_text}]
            )
            self._turns.append(ConversationTurn(
                role="tool_result",
                content=results_text,
                step=self._step,
                is_landmark=done,
                token_count=token_count,
            ))

        return {
            "actions_taken": actions_taken,
            "done": done,
            "result": result_message,
            "llm_calls": 1,
        }

    def get_trajectory(self) -> list[dict]:
        """Return ConversationTurn list for the compression module."""
        return self._turns  # type: ignore[return-value]  # compression uses ConversationTurn

    def update_trajectory(self, compressed_turns: list) -> None:
        """
        Replace the internal turn list with compressed turns and rebuild
        the Anthropic-format message list.

        Called by the orchestrator after compression.compress() returns.
        """
        self._turns = list(compressed_turns)
        self._rebuild_messages()

    def get_stats(self) -> dict[str, Any]:
        return {
            "steps": self._step,
            "turns": len(self._turns),
            "messages": len(self._messages),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        """Combine the base template with an optional memory context block."""
        if self._memory is None:
            return self._system_base

        memory_block = self._memory.get_context_block(max_tokens=2_000)
        if not memory_block:
            return self._system_base

        return f"{self._system_base}\n\n## Memory Context\n{memory_block}"

    def _rebuild_messages(self) -> None:
        """
        Reconstruct _messages from _turns after compression.

        Starts with the original task message (always preserved), then
        converts each ConversationTurn back to an Anthropic message dict.
        The conversation must alternate user/assistant, so consecutive
        turns of the same role are merged.
        """
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": self._task_message}
        ]

        for turn in self._turns:
            if turn.role == "assistant":
                role = "assistant"
            else:
                # "tool_result", "summary", or anything else → user message
                role = "user"

            content = turn.content

            # Merge consecutive same-role messages to keep valid alternation
            if messages and messages[-1]["role"] == role:
                messages[-1]["content"] = (
                    str(messages[-1]["content"]) + "\n\n" + content
                )
            else:
                messages.append({"role": role, "content": content})

        self._messages = messages
        logger.debug(
            "Rebuilt messages from %d compressed turns → %d messages",
            len(self._turns),
            len(self._messages),
        )


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def _load_prompt_template() -> str:
    """Load the system prompt template from disk."""
    try:
        return _PROMPT_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        logger.warning("Prompt template not found at %s; using fallback.", _PROMPT_PATH)
        return (
            "You are a software engineer tasked with fixing a bug in a repository. "
            "Use the available tools to explore, edit, and test the code. "
            "Call submit_patch when you are confident your fix is correct."
        )
