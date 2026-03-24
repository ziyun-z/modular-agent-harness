"""
Rolling-summary compression module.

Strategy:
  When context usage exceeds trigger_ratio * max_tokens, take all turns
  older than the most recent `keep_recent` turns, summarize the non-landmark
  ones into a single "summary" ConversationTurn via an LLM call, and discard
  the originals. Landmark turns (e.g. submit_patch) are always preserved.

Resulting history after compression:
  [summary_turn] + [preserved old landmarks] + [recent_turns]

The summary role is "summary"; _rebuild_messages() in SingleAgentCommunication
maps any non-"assistant" role to a user message, so the summary is injected
as a user-side context block.

Config params (all optional):
    trigger_ratio      float  0.8   trigger when tokens > max_tokens * ratio
    keep_recent        int    10    number of recent turns to keep verbatim
    summary_model      str    "claude-haiku-4-5-20251001"   model for summaries
    max_summary_tokens int    1000  max tokens for the generated summary
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from src.compression.base import CompressionModule, ConversationTurn

if TYPE_CHECKING:
    from src.llm_client import LLMClient

logger = logging.getLogger(__name__)

_DEFAULT_SUMMARY_MODEL = "claude-haiku-4-5-20251001"
_SUMMARIZER_SYSTEM = (
    "You are a concise technical summarizer. "
    "Extract and preserve key facts, findings, file paths, function names, "
    "error messages, and code snippets from agent conversation history. "
    "Be specific and brief."
)


class RollingSummaryCompression(CompressionModule):
    """
    Periodic LLM-based summarization of the oldest conversation turns.

    should_compress() triggers when total turn tokens exceed
    trigger_ratio * max_tokens. compress() collapses non-landmark old turns
    into a single summary block while keeping recent turns verbatim.
    """

    def __init__(
        self,
        trigger_ratio: float = 0.8,
        keep_recent: int = 10,
        summary_model: str = _DEFAULT_SUMMARY_MODEL,
        max_summary_tokens: int = 1000,
    ) -> None:
        self._trigger_ratio = trigger_ratio
        self._keep_recent = keep_recent
        self._summary_model = summary_model
        self._max_summary_tokens = max_summary_tokens

        # Accumulated stats
        self._compressions: int = 0
        self._tokens_before_total: int = 0
        self._tokens_after_total: int = 0

    # ------------------------------------------------------------------
    # CompressionModule interface
    # ------------------------------------------------------------------

    def should_compress(
        self,
        turns: list[ConversationTurn],
        max_tokens: int,
    ) -> bool:
        """Trigger when accumulated turn tokens exceed trigger_ratio * max_tokens."""
        total = _total_tokens(turns)
        threshold = int(max_tokens * self._trigger_ratio)
        return total > threshold

    def compress(
        self,
        turns: list[ConversationTurn],
        target_tokens: int,
        llm_client: "LLMClient",
    ) -> list[ConversationTurn]:
        """
        Summarize the oldest non-landmark turns, keeping the most recent
        `keep_recent` turns verbatim and all landmark turns intact.

        Returns a new list: [summary_turn] + [old landmarks] + [recent turns].
        If there is nothing to summarize, returns the original list unchanged.
        """
        if not turns:
            return turns

        tokens_before = _total_tokens(turns)
        self._tokens_before_total += tokens_before

        # Split into old / recent
        if len(turns) <= self._keep_recent:
            logger.debug(
                "RollingSummary: only %d turns, nothing old enough to compress",
                len(turns),
            )
            return turns

        old_turns = turns[: -self._keep_recent]
        recent_turns = turns[-self._keep_recent :]

        # Among old turns, separate landmarks (preserve) from non-landmarks (summarize)
        old_landmarks = [t for t in old_turns if t.is_landmark]
        to_summarize = [t for t in old_turns if not t.is_landmark]

        if not to_summarize:
            logger.debug("RollingSummary: all old turns are landmarks — nothing to summarize")
            return turns

        # Call LLM to summarize
        summary_text = self._call_summarizer(to_summarize, llm_client)
        summary_tokens = llm_client.count_tokens(
            [{"role": "user", "content": summary_text}]
        )
        summary_turn = ConversationTurn(
            role="summary",
            content=summary_text,
            step=to_summarize[0].step,
            is_landmark=False,
            token_count=summary_tokens,
        )

        result = [summary_turn] + old_landmarks + recent_turns

        tokens_after = _total_tokens(result)
        self._tokens_after_total += tokens_after
        self._compressions += 1

        logger.info(
            "RollingSummary: compressed %d old turns → 1 summary  "
            "(%d → %d tokens, saved %d)",
            len(to_summarize),
            tokens_before,
            tokens_after,
            tokens_before - tokens_after,
        )

        return result

    def get_stats(self) -> dict[str, Any]:
        saved = self._tokens_before_total - self._tokens_after_total
        return {
            "type": "rolling_summary",
            "compressions": self._compressions,
            "tokens_before_total": self._tokens_before_total,
            "tokens_after_total": self._tokens_after_total,
            "tokens_saved_total": saved,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_summarizer(
        self,
        turns: list[ConversationTurn],
        llm_client: "LLMClient",
    ) -> str:
        """
        Build a prompt from the given turns and ask the LLM to summarize them.
        Uses a cheaper model (summary_model) to keep costs low.
        """
        turns_text = "\n\n".join(
            f"[{t.role.upper()} | step {t.step}]\n{t.content}"
            for t in turns
        )
        prompt = (
            "Summarize the following agent conversation history. "
            "Preserve key technical details: file paths, function names, "
            "error messages, test results, and code changes attempted.\n\n"
            f"{turns_text}"
        )
        response = llm_client.complete(
            messages=[{"role": "user", "content": prompt}],
            system=_SUMMARIZER_SYSTEM,
            max_tokens=self._max_summary_tokens,
            model=self._summary_model,
        )
        text_blocks = [b for b in response.content if b.type == "text"]
        if not text_blocks:
            logger.warning("RollingSummary: summarizer returned no text blocks")
            return "[summary unavailable]"
        return text_blocks[0].text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _total_tokens(turns: list[ConversationTurn]) -> int:
    return sum(t.token_count for t in turns)
