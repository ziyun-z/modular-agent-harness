"""
Naive memory module — no-op implementation.

All methods are no-ops or return empty results. Used as the baseline
in ablation experiments where memory is disabled.
"""

from __future__ import annotations

import time
from typing import Any

from src.memory.base import MemoryModule, MemoryEntry


class NaiveMemory(MemoryModule):
    """
    No-op memory. Stores nothing, retrieves nothing, returns an empty
    context block. Used as the baseline (memory=none) in ablation runs.
    """

    def store(self, entry: MemoryEntry) -> None:
        """Accept but discard the entry."""
        pass

    def retrieve(self, query: str, max_tokens: int) -> list[MemoryEntry]:
        """Always returns an empty list."""
        return []

    def get_context_block(self, max_tokens: int) -> str:
        """Always returns an empty string (no memory to inject)."""
        return ""

    def clear(self) -> None:
        """No-op."""
        pass

    def get_stats(self) -> dict[str, Any]:
        return {"type": "naive", "entries": 0}
