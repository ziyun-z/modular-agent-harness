"""
Tests for compression modules: NoCompression and RollingSummaryCompression.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.compression.base import ConversationTurn
from src.compression.none import NoCompression
from src.compression.rolling_summary import RollingSummaryCompression


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_turn(
    role: str = "assistant",
    content: str = "some text",
    step: int = 1,
    is_landmark: bool = False,
    token_count: int = 100,
) -> ConversationTurn:
    return ConversationTurn(
        role=role,
        content=content,
        step=step,
        is_landmark=is_landmark,
        token_count=token_count,
    )


def make_turns(n: int, tokens_each: int = 100, landmark_at: int = -1) -> list[ConversationTurn]:
    """Create n turns, optionally marking one as a landmark."""
    turns = []
    for i in range(n):
        turns.append(make_turn(
            role="assistant" if i % 2 == 0 else "tool_result",
            content=f"turn content {i}",
            step=i + 1,
            is_landmark=(i == landmark_at),
            token_count=tokens_each,
        ))
    return turns


def make_mock_llm(summary_text: str = "Summary of old turns.") -> MagicMock:
    """Return a mock LLMClient whose complete() returns a single text block."""
    mock_llm = MagicMock()

    # count_tokens: approximate by counting words * 1.3
    mock_llm.count_tokens.side_effect = lambda msgs, system=None: (
        len(" ".join(
            m.get("content", "") if isinstance(m.get("content"), str) else ""
            for m in msgs
        ).split())
    )

    # complete(): return a mock response with one TextBlock
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = summary_text

    mock_response = MagicMock()
    mock_response.content = [text_block]

    mock_llm.complete.return_value = mock_response
    return mock_llm


# ===========================================================================
# NoCompression tests
# ===========================================================================

class TestNoCompression:
    def setup_method(self):
        self.comp = NoCompression()

    def test_should_compress_always_false(self):
        turns = make_turns(20, tokens_each=5000)  # 100k total
        assert self.comp.should_compress(turns, max_tokens=10_000) is False

    def test_should_compress_empty(self):
        assert self.comp.should_compress([], max_tokens=100_000) is False

    def test_compress_no_op_when_within_budget(self):
        turns = make_turns(5, tokens_each=100)
        mock_llm = MagicMock()
        result = self.comp.compress(turns, target_tokens=10_000, llm_client=mock_llm)
        assert result == turns
        mock_llm.complete.assert_not_called()

    def test_compress_drops_oldest_non_landmark(self):
        turns = make_turns(5, tokens_each=300)  # 1500 total
        mock_llm = MagicMock()
        result = self.comp.compress(turns, target_tokens=900, llm_client=mock_llm)
        total = sum(t.token_count for t in result)
        assert total <= 900

    def test_compress_preserves_landmarks(self):
        # All turns are landmarks
        turns = [make_turn(is_landmark=True, token_count=500) for _ in range(5)]
        mock_llm = MagicMock()
        result = self.comp.compress(turns, target_tokens=100, llm_client=mock_llm)
        # Cannot drop landmarks; returns all
        assert len(result) == 5

    def test_compress_drops_non_landmark_before_landmark(self):
        turns = [
            make_turn(is_landmark=False, token_count=400, step=1),
            make_turn(is_landmark=True, token_count=400, step=2),
        ]
        mock_llm = MagicMock()
        result = self.comp.compress(turns, target_tokens=500, llm_client=mock_llm)
        assert len(result) == 1
        assert result[0].is_landmark is True

    def test_get_stats(self):
        stats = self.comp.get_stats()
        assert stats["type"] == "none"
        assert stats["compressions"] == 0


# ===========================================================================
# RollingSummaryCompression tests
# ===========================================================================

class TestRollingSummaryCompression:
    def setup_method(self):
        self.comp = RollingSummaryCompression(
            trigger_ratio=0.8,
            keep_recent=3,
            summary_model="claude-haiku-4-5-20251001",
            max_summary_tokens=500,
        )

    # ------------------------------------------------------------------
    # should_compress
    # ------------------------------------------------------------------

    def test_should_compress_below_threshold(self):
        turns = make_turns(5, tokens_each=100)  # 500 total
        # threshold = 0.8 * 1000 = 800
        assert self.comp.should_compress(turns, max_tokens=1000) is False

    def test_should_compress_at_threshold(self):
        turns = make_turns(8, tokens_each=100)  # 800 total
        # threshold = 0.8 * 1000 = 800; 800 is NOT > 800
        assert self.comp.should_compress(turns, max_tokens=1000) is False

    def test_should_compress_above_threshold(self):
        turns = make_turns(9, tokens_each=100)  # 900 total
        # threshold = 0.8 * 1000 = 800; 900 > 800
        assert self.comp.should_compress(turns, max_tokens=1000) is True

    def test_should_compress_empty(self):
        assert self.comp.should_compress([], max_tokens=1000) is False

    # ------------------------------------------------------------------
    # compress — basic behaviour
    # ------------------------------------------------------------------

    def test_compress_empty_returns_empty(self):
        mock_llm = make_mock_llm()
        result = self.comp.compress([], target_tokens=1000, llm_client=mock_llm)
        assert result == []
        mock_llm.complete.assert_not_called()

    def test_compress_fewer_than_keep_recent_unchanged(self):
        turns = make_turns(3, tokens_each=100)  # == keep_recent
        mock_llm = make_mock_llm()
        result = self.comp.compress(turns, target_tokens=100, llm_client=mock_llm)
        assert result == turns
        mock_llm.complete.assert_not_called()

    def test_compress_creates_summary_turn(self):
        turns = make_turns(10, tokens_each=100)  # 7 old + 3 recent
        mock_llm = make_mock_llm("This is a summary.")
        result = self.comp.compress(turns, target_tokens=500, llm_client=mock_llm)

        # First turn should be the summary
        assert result[0].role == "summary"
        assert result[0].content == "This is a summary."
        mock_llm.complete.assert_called_once()

    def test_compress_keeps_recent_turns_verbatim(self):
        turns = make_turns(10, tokens_each=100)
        recent_expected = turns[-3:]  # keep_recent=3
        mock_llm = make_mock_llm()
        result = self.comp.compress(turns, target_tokens=500, llm_client=mock_llm)

        # Last 3 turns of result should match original recent turns
        assert result[-3:] == recent_expected

    def test_compress_reduces_total_tokens(self):
        turns = make_turns(10, tokens_each=200)  # 2000 total
        mock_llm = make_mock_llm("Short summary.")
        result = self.comp.compress(turns, target_tokens=1000, llm_client=mock_llm)

        total_before = sum(t.token_count for t in turns)
        total_after = sum(t.token_count for t in result)
        assert total_after < total_before

    # ------------------------------------------------------------------
    # compress — landmark handling
    # ------------------------------------------------------------------

    def test_compress_preserves_old_landmark_turns(self):
        # landmark at position 2 (old turn, not in recent)
        turns = make_turns(10, tokens_each=100, landmark_at=2)
        mock_llm = make_mock_llm()
        result = self.comp.compress(turns, target_tokens=500, llm_client=mock_llm)

        landmark_turns = [t for t in result if t.is_landmark]
        assert len(landmark_turns) == 1
        assert landmark_turns[0].step == turns[2].step

    def test_compress_all_old_are_landmarks_no_summarize(self):
        # Make all old turns landmarks — nothing should be summarized
        old = [make_turn(is_landmark=True, token_count=100, step=i) for i in range(7)]
        recent = [make_turn(is_landmark=False, token_count=100, step=i + 7) for i in range(3)]
        turns = old + recent

        mock_llm = make_mock_llm()
        result = self.comp.compress(turns, target_tokens=500, llm_client=mock_llm)

        # No LLM call should be made
        mock_llm.complete.assert_not_called()
        assert result == turns

    def test_compress_landmark_in_recent_preserved(self):
        # landmark is one of the last keep_recent turns
        turns = make_turns(10, tokens_each=100, landmark_at=9)  # last turn is landmark
        mock_llm = make_mock_llm()
        result = self.comp.compress(turns, target_tokens=500, llm_client=mock_llm)

        # The landmark (last turn) must still be in the result
        landmark_turns = [t for t in result if t.is_landmark]
        assert len(landmark_turns) == 1
        assert landmark_turns[0].step == turns[9].step

    # ------------------------------------------------------------------
    # compress — summary step metadata
    # ------------------------------------------------------------------

    def test_summary_turn_step_is_first_summarized_step(self):
        turns = make_turns(8, tokens_each=100)
        # old_turns = turns[0:5], first step = 1
        mock_llm = make_mock_llm()
        result = self.comp.compress(turns, target_tokens=500, llm_client=mock_llm)

        assert result[0].step == turns[0].step

    def test_summary_turn_is_not_landmark(self):
        turns = make_turns(8, tokens_each=100)
        mock_llm = make_mock_llm()
        result = self.comp.compress(turns, target_tokens=500, llm_client=mock_llm)
        assert result[0].is_landmark is False

    # ------------------------------------------------------------------
    # compress — fallback when summarizer returns no text
    # ------------------------------------------------------------------

    def test_compress_fallback_on_empty_summarizer_response(self):
        mock_llm = make_mock_llm()
        empty_response = MagicMock()
        empty_response.content = []  # no text blocks
        mock_llm.complete.return_value = empty_response

        turns = make_turns(8, tokens_each=100)
        result = self.comp.compress(turns, target_tokens=500, llm_client=mock_llm)

        assert result[0].role == "summary"
        assert result[0].content == "[summary unavailable]"

    # ------------------------------------------------------------------
    # get_stats
    # ------------------------------------------------------------------

    def test_get_stats_initial(self):
        stats = self.comp.get_stats()
        assert stats["type"] == "rolling_summary"
        assert stats["compressions"] == 0
        assert stats["tokens_saved_total"] == 0

    def test_get_stats_after_compression(self):
        turns = make_turns(10, tokens_each=200)  # 2000 tokens
        mock_llm = make_mock_llm("Summary text here.")
        self.comp.compress(turns, target_tokens=1000, llm_client=mock_llm)

        stats = self.comp.get_stats()
        assert stats["compressions"] == 1
        assert stats["tokens_before_total"] == 2000
        assert stats["tokens_saved_total"] > 0

    def test_get_stats_accumulates_across_calls(self):
        turns = make_turns(10, tokens_each=100)
        mock_llm = make_mock_llm()
        self.comp.compress(turns, target_tokens=500, llm_client=mock_llm)
        self.comp.compress(turns, target_tokens=500, llm_client=mock_llm)

        stats = self.comp.get_stats()
        assert stats["compressions"] == 2
        assert stats["tokens_before_total"] == 2000

    # ------------------------------------------------------------------
    # configuration params
    # ------------------------------------------------------------------

    def test_custom_keep_recent(self):
        comp = RollingSummaryCompression(keep_recent=5)
        turns = make_turns(10, tokens_each=100)
        mock_llm = make_mock_llm()
        result = comp.compress(turns, target_tokens=500, llm_client=mock_llm)

        # summary + 5 recent = 6 items (no old landmarks here)
        assert result[-5:] == turns[-5:]

    def test_custom_trigger_ratio(self):
        comp = RollingSummaryCompression(trigger_ratio=0.5)
        turns = make_turns(6, tokens_each=100)  # 600 tokens
        # threshold = 0.5 * 1000 = 500; 600 > 500
        assert comp.should_compress(turns, max_tokens=1000) is True

    def test_summarizer_uses_configured_model(self):
        comp = RollingSummaryCompression(summary_model="claude-opus-4-6", keep_recent=3)
        turns = make_turns(8, tokens_each=100)
        mock_llm = make_mock_llm()
        comp.compress(turns, target_tokens=500, llm_client=mock_llm)

        call_kwargs = mock_llm.complete.call_args
        assert call_kwargs.kwargs.get("model") == "claude-opus-4-6"

    def test_summarizer_respects_max_summary_tokens(self):
        comp = RollingSummaryCompression(max_summary_tokens=256, keep_recent=3)
        turns = make_turns(8, tokens_each=100)
        mock_llm = make_mock_llm()
        comp.compress(turns, target_tokens=500, llm_client=mock_llm)

        call_kwargs = mock_llm.complete.call_args
        assert call_kwargs.kwargs.get("max_tokens") == 256


# ===========================================================================
# Runner registry integration smoke test
# ===========================================================================

class TestRunnerRegistry:
    def test_rolling_summary_in_registry(self):
        from src.runner import COMPRESSION_REGISTRY
        assert "rolling_summary" in COMPRESSION_REGISTRY
        assert COMPRESSION_REGISTRY["rolling_summary"] is RollingSummaryCompression

    def test_none_still_in_registry(self):
        from src.runner import COMPRESSION_REGISTRY
        assert "none" in COMPRESSION_REGISTRY

    def test_build_compression_module_rolling_summary(self):
        from src.runner import build_compression_module
        cfg = {
            "compression": {
                "type": "rolling_summary",
                "params": {"trigger_ratio": 0.75, "keep_recent": 8},
            }
        }
        module = build_compression_module(cfg)
        assert isinstance(module, RollingSummaryCompression)
        assert module._trigger_ratio == 0.75
        assert module._keep_recent == 8
