from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.llm_client import LLMClient


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
        llm_client: LLMClient,
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
