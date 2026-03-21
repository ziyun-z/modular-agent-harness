"""
No-compression module — baseline implementation.

should_compress() always returns False.
compress() truncates from the front (drops oldest non-landmark turns)
as a last-resort safety net if the orchestrator ever calls it despite
should_compress() returning False.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from src.compression.base import CompressionModule, ConversationTurn

if TYPE_CHECKING:
    from src.llm_client import LLMClient


class NoCompression(CompressionModule):
    """
    No-op compression. Used as the baseline (compression=none) in ablation runs.

    Compression is never triggered. If compress() is called anyway (e.g.
    by a future orchestrator path), it truncates the oldest non-landmark
    turns from the front until the history fits within target_tokens.
    """

    def should_compress(
        self,
        turns: list[ConversationTurn],
        max_tokens: int,
    ) -> bool:
        """Always returns False — no compression is ever triggered."""
        return False

    def compress(
        self,
        turns: list[ConversationTurn],
        target_tokens: int,
        llm_client: "LLMClient",
    ) -> list[ConversationTurn]:
        """
        Fallback-only truncation: drop oldest non-landmark turns from the
        front until the total token count fits within target_tokens.

        No LLM call is made.
        """
        result = list(turns)

        while result and _total_tokens(result) > target_tokens:
            # Find the first non-landmark turn and remove it
            for i, turn in enumerate(result):
                if not turn.is_landmark:
                    result.pop(i)
                    break
            else:
                # All remaining turns are landmarks — nothing left to drop
                break

        return result

    def get_stats(self) -> dict[str, Any]:
        return {"type": "none", "compressions": 0}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _total_tokens(turns: list[ConversationTurn]) -> int:
    return sum(t.token_count for t in turns)
