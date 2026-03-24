"""
Scratchpad memory module.

The agent manages its own persistent notes via the update_scratchpad tool.
The scratchpad content is injected into the system prompt each turn so the
agent always sees its latest notes.

Unlike RAG or hybrid memory, the agent has full control — it decides what to
write, overwrite, or delete. This mirrors how a human engineer keeps a running
notes file while debugging.

Architecture
------------
- store():            no-op (agent writes the scratchpad explicitly via tool)
- retrieve():         returns current scratchpad as a MemoryEntry
- get_context_block(): instructions + current scratchpad injected each turn
- clear():            resets scratchpad between tasks
- handle_tool_call(): called by ToolExecutor when the agent uses update_scratchpad

The TOOL_DEFINITION constant holds the Anthropic-format tool schema.
Runner registers it with the ToolExecutor when this module is active.
"""

from __future__ import annotations

import time
from typing import Any

import tiktoken

from src.memory.base import MemoryModule, MemoryEntry

_encoding = tiktoken.get_encoding("cl100k_base")

# ---------------------------------------------------------------------------
# Tool definition (registered with ToolExecutor when scratchpad is active)
# ---------------------------------------------------------------------------

TOOL_DEFINITION: dict[str, Any] = {
    "name": "update_scratchpad",
    "description": (
        "Overwrite your persistent scratchpad with new content. "
        "The scratchpad is shown to you at the top of every turn. "
        "Use it to track your plan, key findings, hypotheses, and important file locations. "
        "Manage it actively — overwrite with condensed, current notes rather than appending endlessly."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": (
                    "New scratchpad content. Replaces the existing scratchpad entirely. "
                    "Keep it concise (under 500 words). Focus on what you still need to do "
                    "and what you have already learned."
                ),
            },
        },
        "required": ["content"],
    },
}

# System prompt instructions added whenever scratchpad memory is active
_INSTRUCTIONS = (
    "You have a scratchpad for persistent notes. "
    "Use `update_scratchpad` to save your plan, key findings, and important details. "
    "The scratchpad content is shown to you each turn. "
    "Manage it actively — overwrite with updated notes rather than appending endlessly."
)


# ---------------------------------------------------------------------------
# ScratchpadMemory
# ---------------------------------------------------------------------------


class ScratchpadMemory(MemoryModule):
    """
    Agent-managed persistent notes.

    The agent calls update_scratchpad(content) to overwrite the scratchpad.
    The content is injected into the system prompt each turn via
    get_context_block() so the agent always sees its latest notes.

    store() is a no-op — the orchestrator pipes every tool result through it,
    but the scratchpad only changes when the agent explicitly calls
    update_scratchpad.
    """

    def __init__(self) -> None:
        self._scratchpad: str = ""
        self._updates: int = 0
        self._total_chars_written: int = 0

    # ------------------------------------------------------------------
    # MemoryModule interface
    # ------------------------------------------------------------------

    def store(self, entry: MemoryEntry) -> None:
        """No-op. The agent controls the scratchpad via the tool."""
        pass

    def retrieve(self, query: str, max_tokens: int) -> list[MemoryEntry]:
        """
        Return the current scratchpad as a single MemoryEntry.
        Returns an empty list if the scratchpad is blank.
        """
        if not self._scratchpad.strip():
            return []
        return [MemoryEntry(
            step=-1,
            entry_type="scratchpad",
            content=self._scratchpad,
            metadata={"updates": self._updates},
            timestamp=time.time(),
        )]

    def get_context_block(self, max_tokens: int) -> str:
        """
        Return the block injected into the system prompt every turn.

        Always includes scratchpad usage instructions so the agent knows the
        tool exists. When the scratchpad has content, it is appended inside
        <scratchpad> XML tags, truncated to fit max_tokens.
        """
        if not self._scratchpad.strip():
            return _INSTRUCTIONS

        content = _truncate_to_tokens(self._scratchpad, max(0, max_tokens - 60))
        return f"{_INSTRUCTIONS}\n\n<scratchpad>\n{content}\n</scratchpad>"

    def clear(self) -> None:
        """Reset scratchpad to empty. Called at the start of each task."""
        self._scratchpad = ""
        self._updates = 0
        self._total_chars_written = 0

    def get_stats(self) -> dict[str, Any]:
        return {
            "type": "scratchpad",
            "scratchpad_chars": len(self._scratchpad),
            "updates": self._updates,
            "total_chars_written": self._total_chars_written,
        }

    # ------------------------------------------------------------------
    # Tool handler
    # ------------------------------------------------------------------

    def handle_tool_call(self, inp: dict, sandbox: Any) -> str:
        """
        Handle an update_scratchpad tool call from the agent.

        Called by ToolExecutor when the LLM invokes the update_scratchpad tool.
        Replaces the scratchpad content and returns a confirmation message.
        """
        content = inp.get("content", "")
        self._scratchpad = content
        self._updates += 1
        self._total_chars_written += len(content)
        return (
            f"Scratchpad updated ({len(content)} chars). "
            f"Total updates this task: {self._updates}."
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to at most max_tokens tokens, decoded back to a string."""
    if max_tokens <= 0:
        return ""
    tokens = _encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _encoding.decode(tokens[:max_tokens])
