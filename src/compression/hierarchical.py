"""
Hierarchical compression module.

Maintains three tiers of context, from coarsest to finest:

  1. Mission summary  (≤ max_mission_tokens)
       High-level recap of the overall task and approach.
       Created once the first phase summary exists; updated whenever old
       phase summaries are rolled up (i.e. when phase count ≥ max_phases).

  2. Phase summaries  (≤ max_phase_tokens each, up to max_phases kept verbatim)
       One terse paragraph per completed "work phase".  A new phase summary
       is generated from the old non-landmark turns on every compress() call.
       When the number of phase summaries reaches max_phases, the oldest
       phases are condensed into the mission summary and removed from the
       phase list so the list never grows unboundedly.

  3. Recent turns  (last keep_recent regular turns, verbatim)
       Full detail for the most recent work.

  (+) Landmark turns are always preserved verbatim, regardless of age.

Output shape after each compress() call:
    [mission_turn?] + [phase_turns…] + [old_landmark_turns…] + [recent_turns…]

Special ConversationTurn roles used internally:
    "mission_summary"  – the single mission-level turn (at most one)
    "phase_summary"    – one per completed phase
    All other roles ("assistant", "tool_result", "summary", …) are treated
    as regular turns and subject to the keep_recent / phase-summary logic.

Config params (all optional):
    trigger_ratio      float  0.8     compress when tokens > max_tokens * ratio
    keep_recent        int    10      regular turns to keep verbatim
    max_phases         int    5       max phase summaries before rolling up
    summary_model      str    "claude-haiku-4-5-20251001"
    max_phase_tokens   int    300     max tokens per phase summary
    max_mission_tokens int    500     max tokens for the mission summary
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from src.compression.base import CompressionModule, ConversationTurn

if TYPE_CHECKING:
    from src.llm_client import LLMClient

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "claude-haiku-4-5-20251001"

_ROLE_MISSION = "mission_summary"
_ROLE_PHASE = "phase_summary"

_SYSTEM_PHASE = (
    "You are a concise technical summarizer for an AI coding agent. "
    "Produce a single dense paragraph capturing: what was explored, "
    "what was found (file paths, function names, errors), what was "
    "attempted, and what outcome resulted. Be specific, omit filler."
)
_SYSTEM_MISSION = (
    "You are a concise technical summarizer for an AI coding agent. "
    "Produce a single dense paragraph that serves as a high-level mission "
    "summary: the overall task goal, the approach taken so far, key "
    "discoveries, and current status. Incorporate the provided phase "
    "summaries and any existing mission summary. Be specific and brief."
)


class HierarchicalCompression(CompressionModule):
    """
    Three-tier hierarchical summarization: mission → phases → recent turns.

    See module docstring for full description.
    """

    def __init__(
        self,
        trigger_ratio: float = 0.8,
        keep_recent: int = 10,
        max_phases: int = 5,
        summary_model: str = _DEFAULT_MODEL,
        max_phase_tokens: int = 300,
        max_mission_tokens: int = 500,
    ) -> None:
        self._trigger_ratio = trigger_ratio
        self._keep_recent = keep_recent
        self._max_phases = max_phases
        self._summary_model = summary_model
        self._max_phase_tokens = max_phase_tokens
        self._max_mission_tokens = max_mission_tokens

        # Stats
        self._compressions: int = 0
        self._phase_summaries_created: int = 0
        self._mission_updates: int = 0
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
        return _total_tokens(turns) > int(max_tokens * self._trigger_ratio)

    def compress(
        self,
        turns: list[ConversationTurn],
        target_tokens: int,
        llm_client: "LLMClient",
    ) -> list[ConversationTurn]:
        """
        Compress turns into the three-tier hierarchical representation.

        Steps:
          1. Partition incoming turns into existing summary turns and regular turns.
          2. Split regular turns into old (subject to compression) and recent (kept).
          3. If there are old non-landmark regular turns, summarize them as a new phase.
          4. If phase count reaches max_phases, roll the oldest phases into the
             mission summary, then drop them from the phase list.
          5. Reconstruct: [mission?] + [phases] + [old landmarks] + [recent].
        """
        if not turns:
            return turns

        tokens_before = _total_tokens(turns)
        self._tokens_before_total += tokens_before

        # ------------------------------------------------------------------
        # 1. Partition
        # ------------------------------------------------------------------
        existing_mission, existing_phases, other_turns = _partition(turns)

        # ------------------------------------------------------------------
        # 2. Split other_turns into old / recent
        # ------------------------------------------------------------------
        if len(other_turns) <= self._keep_recent:
            # Nothing old enough to process; reassemble unchanged
            result = existing_mission + existing_phases + other_turns
            self._tokens_after_total += _total_tokens(result)
            return result

        old_other = other_turns[: -self._keep_recent]
        recent_other = other_turns[-self._keep_recent :]

        old_landmarks = [t for t in old_other if t.is_landmark]
        old_normal = [t for t in old_other if not t.is_landmark]

        # ------------------------------------------------------------------
        # 3. Summarize old non-landmark turns as a new phase
        # ------------------------------------------------------------------
        if old_normal:
            phase_turn = self._make_phase_summary(old_normal, llm_client)
            existing_phases.append(phase_turn)
            self._phase_summaries_created += 1
            logger.info(
                "Hierarchical: new phase summary from %d turns (step %d–%d)",
                len(old_normal),
                old_normal[0].step,
                old_normal[-1].step,
            )

        # ------------------------------------------------------------------
        # 4. Roll up oldest phases into mission when limit reached
        # ------------------------------------------------------------------
        if len(existing_phases) >= self._max_phases:
            # Phases to absorb: all except the most recent (max_phases - 1)
            n_keep = self._max_phases - 1
            phases_to_absorb = existing_phases[: -n_keep] if n_keep else existing_phases
            phases_to_keep = existing_phases[-n_keep:] if n_keep else []

            existing_mission = [
                self._update_mission_summary(
                    existing_mission, phases_to_absorb, llm_client
                )
            ]
            existing_phases = phases_to_keep
            self._mission_updates += 1
            logger.info(
                "Hierarchical: mission summary updated from %d phases; "
                "%d phases retained",
                len(phases_to_absorb),
                len(existing_phases),
            )

        # ------------------------------------------------------------------
        # 5. Reconstruct
        # ------------------------------------------------------------------
        result = existing_mission + existing_phases + old_landmarks + recent_other

        tokens_after = _total_tokens(result)
        self._tokens_after_total += tokens_after
        self._compressions += 1

        logger.info(
            "Hierarchical: compression #%d  %d→%d tokens  "
            "(mission=%d, phases=%d, landmarks=%d, recent=%d)",
            self._compressions,
            tokens_before,
            tokens_after,
            len(existing_mission),
            len(existing_phases),
            len(old_landmarks),
            len(recent_other),
        )
        return result

    def get_stats(self) -> dict[str, Any]:
        saved = self._tokens_before_total - self._tokens_after_total
        return {
            "type": "hierarchical",
            "compressions": self._compressions,
            "phase_summaries_created": self._phase_summaries_created,
            "mission_updates": self._mission_updates,
            "tokens_before_total": self._tokens_before_total,
            "tokens_after_total": self._tokens_after_total,
            "tokens_saved_total": saved,
        }

    # ------------------------------------------------------------------
    # Internal helpers — LLM calls
    # ------------------------------------------------------------------

    def _make_phase_summary(
        self,
        turns: list[ConversationTurn],
        llm_client: "LLMClient",
    ) -> ConversationTurn:
        """Summarize a list of regular turns into a single phase-summary turn."""
        turns_text = "\n\n".join(
            f"[{t.role.upper()} | step {t.step}]\n{t.content}" for t in turns
        )
        prompt = (
            "Summarize the following agent work phase into one concise paragraph. "
            "Preserve specific details: file paths, function names, error messages, "
            "test results, and code changes made.\n\n"
            f"{turns_text}"
        )
        summary_text = self._llm_summarize(prompt, _SYSTEM_PHASE, self._max_phase_tokens, llm_client)
        token_count = llm_client.count_tokens([{"role": "user", "content": summary_text}])
        return ConversationTurn(
            role=_ROLE_PHASE,
            content=summary_text,
            step=turns[0].step,
            is_landmark=False,
            token_count=token_count,
        )

    def _update_mission_summary(
        self,
        existing_mission: list[ConversationTurn],
        phases_to_absorb: list[ConversationTurn],
        llm_client: "LLMClient",
    ) -> ConversationTurn:
        """
        Create or update the mission summary by absorbing the given phase summaries.
        The existing mission text (if any) is included so information is not lost.
        """
        parts: list[str] = []
        if existing_mission:
            parts.append(f"## Existing mission summary\n{existing_mission[0].content}")
        if phases_to_absorb:
            phases_text = "\n\n".join(
                f"Phase (step {t.step}): {t.content}" for t in phases_to_absorb
            )
            parts.append(f"## Phase summaries to absorb\n{phases_text}")

        prompt = (
            "Update the mission summary for an AI coding agent by incorporating "
            "the phase summaries below. Produce a single dense paragraph covering: "
            "the overall task goal, approach taken, key discoveries, and current "
            "status. Include specific details (file paths, key findings).\n\n"
            + "\n\n".join(parts)
        )
        summary_text = self._llm_summarize(prompt, _SYSTEM_MISSION, self._max_mission_tokens, llm_client)
        token_count = llm_client.count_tokens([{"role": "user", "content": summary_text}])

        first_step = phases_to_absorb[0].step if phases_to_absorb else (
            existing_mission[0].step if existing_mission else 0
        )
        return ConversationTurn(
            role=_ROLE_MISSION,
            content=summary_text,
            step=first_step,
            is_landmark=False,
            token_count=token_count,
        )

    def _llm_summarize(
        self,
        prompt: str,
        system: str,
        max_tokens: int,
        llm_client: "LLMClient",
    ) -> str:
        """Single LLM call; returns the text of the first TextBlock."""
        response = llm_client.complete(
            messages=[{"role": "user", "content": prompt}],
            system=system,
            max_tokens=max_tokens,
            model=self._summary_model,
        )
        text_blocks = [b for b in response.content if b.type == "text"]
        if not text_blocks:
            logger.warning("Hierarchical: summarizer returned no text blocks")
            return "[summary unavailable]"
        return text_blocks[0].text


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _partition(
    turns: list[ConversationTurn],
) -> tuple[list[ConversationTurn], list[ConversationTurn], list[ConversationTurn]]:
    """
    Split turns into (mission_list, phase_list, other_list).

    mission_list: at most one turn with role == "mission_summary"
    phase_list:   turns with role == "phase_summary", in original order
    other_list:   everything else (regular assistant/tool_result/landmark turns)
    """
    mission: list[ConversationTurn] = []
    phases: list[ConversationTurn] = []
    other: list[ConversationTurn] = []

    for turn in turns:
        if turn.role == _ROLE_MISSION:
            mission.append(turn)
        elif turn.role == _ROLE_PHASE:
            phases.append(turn)
        else:
            other.append(turn)

    # Sanity: keep only the most recent mission turn if somehow duplicated
    if len(mission) > 1:
        mission = [mission[-1]]

    return mission, phases, other


def _total_tokens(turns: list[ConversationTurn]) -> int:
    return sum(t.token_count for t in turns)
