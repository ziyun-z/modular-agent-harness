from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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
