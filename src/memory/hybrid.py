"""
Hybrid memory module — episodic (RAG) + semantic (knowledge base).

Two complementary stores
------------------------
1. **Episodic store** (RAGMemory, automatic):
   Every tool result is embedded and stored automatically by the framework.
   The most relevant past observations are retrieved each turn via semantic
   search.  The agent does not need to think about this — it just works.

2. **Semantic / knowledge base** (dict, agent-driven):
   The agent explicitly writes permanent key-value facts via the
   `update_knowledge` tool.  Facts are always shown in full every turn —
   there is no retrieval step because the agent chose to record them
   precisely because they are important enough to always remember.
   Examples: bug location, root cause, key file paths, patterns found.
   Setting value="" deletes a fact.

Token budget split
------------------
  30% of max_tokens → knowledge base (semantic)
  70% of max_tokens → episodic RAG context

The knowledge-base fraction is configurable; it defaults to 0.3 because
explicit facts tend to be short and dense.

Context block format
--------------------
    You have two memory systems: ...   ← always-present instructions

    ## Knowledge Base
    <knowledge>
    bug_location: parser.py line 42
    root_cause: off-by-one in token loop
    </knowledge>

    ## Relevant Past Observations
    Relevant observations from earlier in this task:

    [memory 1]
    [step=3 type=observation]
    ...
"""

from __future__ import annotations

import logging
from typing import Any

import tiktoken

from src.memory.base import MemoryModule, MemoryEntry
from src.memory.rag import RAGMemory

logger = logging.getLogger(__name__)

_encoding = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_encoding.encode(text))


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    tokens = _encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _encoding.decode(tokens[:max_tokens])


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------

KNOWLEDGE_TOOL_DEFINITION: dict[str, Any] = {
    "name": "update_knowledge",
    "description": (
        "Add, update, or delete a fact in your persistent knowledge base. "
        "Facts are always shown to you at the start of every turn — use this "
        "for information you never want to lose: bug location, root cause, "
        "key file paths, API signatures, or important patterns. "
        "Set value to an empty string to delete a fact."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": (
                    "Short fact name (e.g. 'bug_location', 'root_cause', 'affected_files'). "
                    "Use snake_case. Reusing a key overwrites the previous value."
                ),
            },
            "value": {
                "type": "string",
                "description": (
                    "Fact value. Keep concise — one or two sentences max. "
                    "Empty string removes the fact from the knowledge base."
                ),
            },
        },
        "required": ["key", "value"],
    },
}

_INSTRUCTIONS = """\
You have two memory systems:

1. Knowledge base (use `update_knowledge`): store permanent facts as key-value \
pairs — bug location, root cause, key file paths, important patterns. \
Facts are always visible every turn. Set value="" to delete a fact.

2. Episodic memory (automatic): past observations are stored and retrieved \
automatically based on relevance. No action needed on your part.\
"""


# ---------------------------------------------------------------------------
# HybridMemory
# ---------------------------------------------------------------------------


class HybridMemory(MemoryModule):
    """
    Hybrid memory: automatic episodic recall (RAG) + explicit semantic facts (dict).

    Parameters
    ----------
    embedding_model : str
        Embedding model for the episodic RAG store.
    top_k : int
        Number of episodic candidates to fetch from ChromaDB per query.
    semantic_budget_fraction : float
        Fraction of max_tokens reserved for the knowledge base (default 0.3).
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        top_k: int = 20,
        semantic_budget_fraction: float = 0.3,
    ) -> None:
        self.knowledge_base: dict[str, str] = {}
        self.episodic_store: RAGMemory = RAGMemory(
            embedding_model=embedding_model,
            top_k=top_k,
        )
        self._semantic_fraction = semantic_budget_fraction
        self._knowledge_updates: int = 0

    # ------------------------------------------------------------------
    # MemoryModule interface
    # ------------------------------------------------------------------

    def store(self, entry: MemoryEntry) -> None:
        """Feed every tool observation into the episodic (RAG) store."""
        self.episodic_store.store(entry)

    def retrieve(self, query: str, max_tokens: int) -> list[MemoryEntry]:
        """
        Retrieve from the episodic store only.

        The knowledge base is always shown complete in get_context_block(),
        so it does not participate in semantic retrieval.
        """
        episodic_budget = int(max_tokens * (1 - self._semantic_fraction))
        return self.episodic_store.retrieve(query, episodic_budget)

    def get_context_block(self, max_tokens: int) -> str:
        """
        Build the full memory context block for the system prompt.

        Layout (each section only included if non-empty):
            instructions
            ## Knowledge Base  +  <knowledge> block
            ## Relevant Past Observations  +  RAG block
        """
        instructions_tokens = _count_tokens(_INSTRUCTIONS)
        remaining = max_tokens - instructions_tokens - 10

        semantic_budget = int(remaining * self._semantic_fraction)
        episodic_budget = remaining - semantic_budget

        parts: list[str] = [_INSTRUCTIONS]

        # Knowledge base (semantic)
        kb_block = self._format_knowledge_base(semantic_budget)
        if kb_block:
            parts.append(f"## Knowledge Base\n{kb_block}")

        # Episodic RAG context
        episodic_block = self.episodic_store.get_context_block(episodic_budget)
        if episodic_block:
            parts.append(f"## Relevant Past Observations\n{episodic_block}")

        # If nothing beyond instructions, at least return the instructions so
        # the agent knows the tools exist.
        return "\n\n".join(parts)

    def clear(self) -> None:
        """Reset both stores. Called between tasks."""
        self.knowledge_base = {}
        self.episodic_store.clear()
        self._knowledge_updates = 0
        logger.debug("HybridMemory cleared")

    def get_stats(self) -> dict[str, Any]:
        return {
            "type": "hybrid",
            "knowledge_entries": len(self.knowledge_base),
            "knowledge_updates": self._knowledge_updates,
            "episodic": self.episodic_store.get_stats(),
        }

    # ------------------------------------------------------------------
    # Tool handler
    # ------------------------------------------------------------------

    def handle_knowledge_tool_call(self, inp: dict, sandbox: Any) -> str:
        """
        Handle an update_knowledge tool call from the agent.

        - Non-empty value  → add or overwrite the fact
        - Empty value      → delete the fact (no-op if key not present)
        """
        key = inp.get("key", "").strip()
        value = inp.get("value", "")

        if not key:
            return "Error: 'key' cannot be empty."

        if value == "":
            removed = self.knowledge_base.pop(key, None)
            if removed is not None:
                self._knowledge_updates += 1
                return f"Removed '{key}' from knowledge base ({len(self.knowledge_base)} facts remaining)."
            return f"Key '{key}' not found in knowledge base (no change)."

        action = "Updated" if key in self.knowledge_base else "Added"
        self.knowledge_base[key] = value
        self._knowledge_updates += 1
        preview = value[:80] + ("…" if len(value) > 80 else "")
        return f"{action} knowledge — '{key}': {preview!r} ({len(self.knowledge_base)} facts total)."

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _format_knowledge_base(self, max_tokens: int) -> str:
        """Format knowledge_base as a <knowledge> block, token-limited."""
        if not self.knowledge_base:
            return ""
        lines = [f"{k}: {v}" for k, v in self.knowledge_base.items()]
        content = _truncate_to_tokens("\n".join(lines), max(0, max_tokens - 20))
        return f"<knowledge>\n{content}\n</knowledge>"
