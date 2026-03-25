"""
Tests for compression modules: NoCompression, RollingSummaryCompression,
and HierarchicalCompression.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.compression.base import ConversationTurn
from src.compression.none import NoCompression
from src.compression.rolling_summary import RollingSummaryCompression
from src.compression.hierarchical import HierarchicalCompression, _partition, _ROLE_MISSION, _ROLE_PHASE


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
# HierarchicalCompression tests
# ===========================================================================

def make_phase_turn(content: str = "phase text", step: int = 1, token_count: int = 50) -> ConversationTurn:
    return ConversationTurn(role=_ROLE_PHASE, content=content, step=step, is_landmark=False, token_count=token_count)

def make_mission_turn(content: str = "mission text", step: int = 0, token_count: int = 80) -> ConversationTurn:
    return ConversationTurn(role=_ROLE_MISSION, content=content, step=step, is_landmark=False, token_count=token_count)


class TestPartition:
    """Unit tests for the _partition helper."""

    def test_empty(self):
        mission, phases, other = _partition([])
        assert mission == [] and phases == [] and other == []

    def test_only_regular_turns(self):
        turns = make_turns(4)
        mission, phases, other = _partition(turns)
        assert mission == [] and phases == [] and other == turns

    def test_identifies_mission_turn(self):
        t = make_mission_turn()
        mission, phases, other = _partition([t])
        assert mission == [t] and phases == [] and other == []

    def test_identifies_phase_turns(self):
        p1, p2 = make_phase_turn(step=1), make_phase_turn(step=2)
        mission, phases, other = _partition([p1, p2])
        assert phases == [p1, p2] and mission == [] and other == []

    def test_mixed_partition(self):
        m = make_mission_turn()
        p = make_phase_turn()
        r = make_turn(step=10)
        mission, phases, other = _partition([m, p, r])
        assert mission == [m]
        assert phases == [p]
        assert other == [r]

    def test_duplicate_mission_keeps_last(self):
        m1 = make_mission_turn(content="old", step=1)
        m2 = make_mission_turn(content="new", step=2)
        mission, _, _ = _partition([m1, m2])
        assert len(mission) == 1
        assert mission[0].content == "new"

    def test_landmark_goes_to_other(self):
        t = make_turn(is_landmark=True)
        _, _, other = _partition([t])
        assert other == [t]


class TestHierarchicalCompression:
    def setup_method(self):
        self.comp = HierarchicalCompression(
            trigger_ratio=0.8,
            keep_recent=3,
            max_phases=3,
            summary_model="claude-haiku-4-5-20251001",
            max_phase_tokens=100,
            max_mission_tokens=150,
        )

    # ------------------------------------------------------------------
    # should_compress
    # ------------------------------------------------------------------

    def test_should_compress_below_threshold(self):
        turns = make_turns(5, tokens_each=100)  # 500 tokens
        assert self.comp.should_compress(turns, max_tokens=1000) is False  # 500 <= 800

    def test_should_compress_above_threshold(self):
        turns = make_turns(9, tokens_each=100)  # 900 tokens
        assert self.comp.should_compress(turns, max_tokens=1000) is True   # 900 > 800

    def test_should_compress_empty(self):
        assert self.comp.should_compress([], max_tokens=1000) is False

    # ------------------------------------------------------------------
    # compress — basic shape
    # ------------------------------------------------------------------

    def test_compress_empty_returns_empty(self):
        result = self.comp.compress([], target_tokens=1000, llm_client=make_mock_llm())
        assert result == []

    def test_compress_fewer_than_keep_recent_unchanged(self):
        turns = make_turns(3, tokens_each=100)  # == keep_recent
        mock_llm = make_mock_llm()
        result = self.comp.compress(turns, target_tokens=500, llm_client=mock_llm)
        assert result == turns
        mock_llm.complete.assert_not_called()

    def test_compress_creates_phase_summary_turn(self):
        turns = make_turns(8, tokens_each=100)   # 5 old + 3 recent
        mock_llm = make_mock_llm("Phase one summary.")
        result = self.comp.compress(turns, target_tokens=500, llm_client=mock_llm)

        phase_turns = [t for t in result if t.role == _ROLE_PHASE]
        assert len(phase_turns) == 1
        assert phase_turns[0].content == "Phase one summary."

    def test_compress_recent_turns_preserved_verbatim(self):
        turns = make_turns(8, tokens_each=100)
        recent_expected = turns[-3:]
        result = self.comp.compress(turns, target_tokens=500, llm_client=make_mock_llm())
        assert result[-3:] == recent_expected

    def test_compress_reduces_token_count(self):
        turns = make_turns(10, tokens_each=200)  # 2000 tokens
        result = self.comp.compress(turns, target_tokens=1000, llm_client=make_mock_llm("Short."))
        assert sum(t.token_count for t in result) < sum(t.token_count for t in turns)

    # ------------------------------------------------------------------
    # compress — output structure/ordering
    # ------------------------------------------------------------------

    def test_output_order_no_mission(self):
        # Before mission is created: [phases] + [recent]
        turns = make_turns(8, tokens_each=100)
        result = self.comp.compress(turns, target_tokens=500, llm_client=make_mock_llm())

        roles = [t.role for t in result]
        # Phase summary must come before regular turns
        last_phase = max((i for i, r in enumerate(roles) if r == _ROLE_PHASE), default=-1)
        first_regular = next((i for i, r in enumerate(roles) if r not in (_ROLE_MISSION, _ROLE_PHASE)), len(roles))
        assert last_phase < first_regular

    def test_mission_comes_first_when_present(self):
        # Pre-inject a mission turn; it should appear first in output
        mission = make_mission_turn()
        phase = make_phase_turn()
        regular = make_turns(8, tokens_each=100)
        turns = [mission, phase] + regular

        result = self.comp.compress(turns, target_tokens=500, llm_client=make_mock_llm())
        assert result[0].role == _ROLE_MISSION

    # ------------------------------------------------------------------
    # compress — landmark preservation
    # ------------------------------------------------------------------

    def test_old_landmark_preserved_verbatim(self):
        turns = make_turns(8, tokens_each=100, landmark_at=1)  # landmark in old window
        result = self.comp.compress(turns, target_tokens=500, llm_client=make_mock_llm())

        landmark_turns = [t for t in result if t.is_landmark]
        assert len(landmark_turns) == 1
        assert landmark_turns[0].step == turns[1].step

    def test_landmark_in_recent_preserved(self):
        turns = make_turns(8, tokens_each=100, landmark_at=7)  # last turn
        result = self.comp.compress(turns, target_tokens=500, llm_client=make_mock_llm())
        landmark_turns = [t for t in result if t.is_landmark]
        assert len(landmark_turns) == 1

    def test_all_old_are_landmarks_no_phase_created(self):
        old = [make_turn(is_landmark=True, token_count=100, step=i) for i in range(5)]
        recent = make_turns(3, tokens_each=100)
        turns = old + recent

        mock_llm = make_mock_llm()
        result = self.comp.compress(turns, target_tokens=500, llm_client=mock_llm)
        mock_llm.complete.assert_not_called()
        phase_turns = [t for t in result if t.role == _ROLE_PHASE]
        assert len(phase_turns) == 0

    # ------------------------------------------------------------------
    # compress — phase roll-up into mission
    # ------------------------------------------------------------------

    def test_phase_rollup_triggers_at_max_phases(self):
        # max_phases=3; inject 2 existing phases + create 1 new → triggers roll-up
        existing_phases = [make_phase_turn(step=i, content=f"phase {i}") for i in range(2)]
        regular = make_turns(8, tokens_each=100)
        turns = existing_phases + regular

        mock_llm = make_mock_llm("Mission summary after rollup.")
        result = self.comp.compress(turns, target_tokens=500, llm_client=mock_llm)

        mission_turns = [t for t in result if t.role == _ROLE_MISSION]
        assert len(mission_turns) == 1
        # Two LLM calls: one for new phase summary, one for mission update
        assert mock_llm.complete.call_count == 2

    def test_phase_rollup_reduces_phase_count(self):
        # max_phases=3; after rollup only (max_phases - 1) = 2 phases kept
        existing_phases = [make_phase_turn(step=i) for i in range(2)]
        regular = make_turns(8, tokens_each=100)
        turns = existing_phases + regular

        result = self.comp.compress(turns, target_tokens=500, llm_client=make_mock_llm())

        phase_turns = [t for t in result if t.role == _ROLE_PHASE]
        assert len(phase_turns) <= self.comp._max_phases - 1

    def test_no_rollup_below_max_phases(self):
        # max_phases=3; inject 1 existing phase + create 1 new = 2 < 3 → no mission
        existing_phases = [make_phase_turn(step=0)]
        regular = make_turns(8, tokens_each=100)
        turns = existing_phases + regular

        result = self.comp.compress(turns, target_tokens=500, llm_client=make_mock_llm())

        mission_turns = [t for t in result if t.role == _ROLE_MISSION]
        assert len(mission_turns) == 0

    def test_existing_mission_updated_on_rollup(self):
        # Existing mission + 2 phases; adding 1 more triggers rollup → mission updated
        existing_mission = make_mission_turn(content="old mission")
        existing_phases = [make_phase_turn(step=i) for i in range(2)]
        regular = make_turns(8, tokens_each=100)
        turns = [existing_mission] + existing_phases + regular

        mock_llm = make_mock_llm("Updated mission text.")
        result = self.comp.compress(turns, target_tokens=500, llm_client=mock_llm)

        mission_turns = [t for t in result if t.role == _ROLE_MISSION]
        assert len(mission_turns) == 1
        assert mission_turns[0].content == "Updated mission text."

    def test_existing_phases_carried_forward_no_rollup(self):
        # Inject 1 phase; add 1 new → total 2 < max_phases=3, no rollup
        existing_phase = make_phase_turn(step=0, content="earlier phase")
        regular = make_turns(8, tokens_each=100)
        turns = [existing_phase] + regular

        result = self.comp.compress(turns, target_tokens=500, llm_client=make_mock_llm("new phase"))

        phase_turns = [t for t in result if t.role == _ROLE_PHASE]
        assert len(phase_turns) == 2
        assert phase_turns[0].content == "earlier phase"
        assert phase_turns[1].content == "new phase"

    # ------------------------------------------------------------------
    # compress — phase turn metadata
    # ------------------------------------------------------------------

    def test_phase_turn_step_is_first_old_turn_step(self):
        turns = make_turns(8, tokens_each=100)
        result = self.comp.compress(turns, target_tokens=500, llm_client=make_mock_llm())
        phase_turns = [t for t in result if t.role == _ROLE_PHASE]
        assert phase_turns[0].step == turns[0].step

    def test_phase_turn_is_not_landmark(self):
        turns = make_turns(8, tokens_each=100)
        result = self.comp.compress(turns, target_tokens=500, llm_client=make_mock_llm())
        phase_turns = [t for t in result if t.role == _ROLE_PHASE]
        assert all(not t.is_landmark for t in phase_turns)

    # ------------------------------------------------------------------
    # compress — fallback on empty LLM response
    # ------------------------------------------------------------------

    def test_phase_fallback_on_empty_response(self):
        mock_llm = make_mock_llm()
        mock_llm.complete.return_value = MagicMock(content=[])

        turns = make_turns(8, tokens_each=100)
        result = self.comp.compress(turns, target_tokens=500, llm_client=mock_llm)
        phase_turns = [t for t in result if t.role == _ROLE_PHASE]
        assert phase_turns[0].content == "[summary unavailable]"

    # ------------------------------------------------------------------
    # get_stats
    # ------------------------------------------------------------------

    def test_stats_initial(self):
        stats = self.comp.get_stats()
        assert stats["type"] == "hierarchical"
        assert stats["compressions"] == 0
        assert stats["phase_summaries_created"] == 0
        assert stats["mission_updates"] == 0

    def test_stats_after_compression(self):
        turns = make_turns(8, tokens_each=100)
        self.comp.compress(turns, target_tokens=500, llm_client=make_mock_llm())
        stats = self.comp.get_stats()
        assert stats["compressions"] == 1
        assert stats["phase_summaries_created"] == 1
        assert stats["tokens_before_total"] == 800

    def test_stats_mission_update_counted(self):
        existing_phases = [make_phase_turn(step=i) for i in range(2)]
        regular = make_turns(8, tokens_each=100)
        turns = existing_phases + regular

        self.comp.compress(turns, target_tokens=500, llm_client=make_mock_llm())
        stats = self.comp.get_stats()
        assert stats["mission_updates"] == 1

    def test_stats_accumulate_across_calls(self):
        turns = make_turns(8, tokens_each=100)
        self.comp.compress(turns, target_tokens=500, llm_client=make_mock_llm())
        self.comp.compress(turns, target_tokens=500, llm_client=make_mock_llm())
        stats = self.comp.get_stats()
        assert stats["compressions"] == 2

    # ------------------------------------------------------------------
    # configuration
    # ------------------------------------------------------------------

    def test_custom_keep_recent(self):
        comp = HierarchicalCompression(keep_recent=5, max_phases=3)
        turns = make_turns(10, tokens_each=100)
        result = comp.compress(turns, target_tokens=500, llm_client=make_mock_llm())
        assert result[-5:] == turns[-5:]

    def test_summarizer_uses_configured_model(self):
        comp = HierarchicalCompression(summary_model="claude-opus-4-6", keep_recent=3, max_phases=3)
        turns = make_turns(8, tokens_each=100)
        mock_llm = make_mock_llm()
        comp.compress(turns, target_tokens=500, llm_client=mock_llm)
        for call in mock_llm.complete.call_args_list:
            assert call.kwargs.get("model") == "claude-opus-4-6"

    def test_max_phase_tokens_passed_to_llm(self):
        comp = HierarchicalCompression(max_phase_tokens=42, keep_recent=3, max_phases=3)
        turns = make_turns(8, tokens_each=100)
        mock_llm = make_mock_llm()
        comp.compress(turns, target_tokens=500, llm_client=mock_llm)
        # First call is the phase summary; check max_tokens
        first_call = mock_llm.complete.call_args_list[0]
        assert first_call.kwargs.get("max_tokens") == 42


# ===========================================================================
# Runner registry integration smoke test
# ===========================================================================

class TestRunnerRegistry:
    def test_rolling_summary_in_registry(self):
        from src.runner import COMPRESSION_REGISTRY
        assert "rolling_summary" in COMPRESSION_REGISTRY
        assert COMPRESSION_REGISTRY["rolling_summary"] is RollingSummaryCompression

    def test_hierarchical_in_registry(self):
        from src.runner import COMPRESSION_REGISTRY
        assert "hierarchical" in COMPRESSION_REGISTRY
        assert COMPRESSION_REGISTRY["hierarchical"] is HierarchicalCompression

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

    def test_build_compression_module_hierarchical(self):
        from src.runner import build_compression_module
        cfg = {
            "compression": {
                "type": "hierarchical",
                "params": {"trigger_ratio": 0.7, "keep_recent": 6, "max_phases": 4},
            }
        }
        module = build_compression_module(cfg)
        assert isinstance(module, HierarchicalCompression)
        assert module._trigger_ratio == 0.7
        assert module._keep_recent == 6
        assert module._max_phases == 4
