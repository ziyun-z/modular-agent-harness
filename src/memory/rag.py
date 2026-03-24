"""
RAG memory module — retrieval-augmented generation via ChromaDB.

Architecture
------------
Every tool result the orchestrator observes is automatically stored as an
embedding in an in-memory ChromaDB collection.  Each turn, get_context_block()
queries the collection with the most recent observation as the search query and
injects the top-k most semantically relevant past entries into the system prompt.

This is system-driven (automatic) memory: the agent never calls a tool to
write to it — the framework handles all persistence.  The agent benefits
transparently: relevant past observations surface in its context even after
they would otherwise have scrolled off the context window.

Key design decisions
--------------------
- Embedding model: ChromaDB's DefaultEmbeddingFunction (ONNX-based
  all-MiniLM-L6-v2, 384-d cosine similarity). Works without PyTorch.
- Storage: one ChromaDB document per MemoryEntry; metadata preserved.
- Query for get_context_block(): the content of the most recently stored
  entry.  This approximates "what am I doing right now?" without requiring
  external coordination.
- Token budget: entries are accumulated in relevance order until max_tokens
  is exhausted; the remainder are dropped.
- Chunk strategy: "per_tool_result" (default) — each MemoryEntry is one
  document.  Large entries are stored whole; retrieval truncates on output.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

import tiktoken

from src.memory.base import MemoryModule, MemoryEntry

logger = logging.getLogger(__name__)

_encoding = tiktoken.get_encoding("cl100k_base")

# Prefix added to every context block so the agent understands the source
_CONTEXT_HEADER = "Relevant observations from earlier in this task:"


def _count_tokens(text: str) -> int:
    return len(_encoding.encode(text))


def _format_entry(entry: MemoryEntry) -> str:
    """Format a MemoryEntry as the text stored in ChromaDB."""
    parts = [f"[step={entry.step} type={entry.entry_type}]"]
    if entry.metadata.get("input"):
        # Trim long inputs to keep documents compact
        inp = str(entry.metadata["input"])[:200]
        parts.append(f"input: {inp}")
    parts.append(entry.content)
    return "\n".join(parts)


def _format_result_for_prompt(doc: str, rank: int) -> str:
    """Format a retrieved document for injection into the system prompt."""
    return f"[memory {rank}]\n{doc}"


class RAGMemory(MemoryModule):
    """
    Retrieval-augmented memory backed by an in-memory ChromaDB collection.

    Usage flow:
        1. Orchestrator calls store(entry) after every tool result.
        2. Communication module calls get_context_block(max_tokens) each turn.
        3. get_context_block() queries ChromaDB with the latest observation
           and returns the top-k most relevant past entries, formatted for
           injection into the system prompt.

    Parameters
    ----------
    embedding_model : str
        Which embedding model to use.  Currently only "all-MiniLM-L6-v2" is
        supported (via ChromaDB's built-in ONNX runtime).
    top_k : int
        Maximum number of candidates to fetch from ChromaDB before applying
        the token budget filter.
    chunk_strategy : str
        How to split large entries before storage.  "per_tool_result" (default)
        stores each MemoryEntry as a single document.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        top_k: int = 20,
        chunk_strategy: str = "per_tool_result",
    ) -> None:
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.chunk_strategy = chunk_strategy

        self._collection = self._make_collection()
        self._entry_count: int = 0
        self._total_retrievals: int = 0
        self._total_retrieval_time: float = 0.0
        self._last_stored_content: str = ""   # used as query in get_context_block()

    # ------------------------------------------------------------------
    # MemoryModule interface
    # ------------------------------------------------------------------

    def store(self, entry: MemoryEntry) -> None:
        """Embed and insert a MemoryEntry into ChromaDB."""
        doc = _format_entry(entry)
        doc_id = f"entry_{self._entry_count}_{uuid.uuid4().hex[:8]}"

        self._collection.add(
            documents=[doc],
            metadatas=[{
                "step": entry.step,
                "entry_type": entry.entry_type,
                "timestamp": entry.timestamp,
            }],
            ids=[doc_id],
        )
        self._entry_count += 1
        self._last_stored_content = entry.content
        logger.debug("RAG stored entry %d (type=%s, %d chars)", self._entry_count, entry.entry_type, len(doc))

    def retrieve(self, query: str, max_tokens: int) -> list[MemoryEntry]:
        """
        Query ChromaDB for the most semantically relevant past entries.

        Fetches up to top_k candidates, then accumulates in relevance order
        until the token budget is exhausted.

        Returns MemoryEntry objects reconstructed from stored documents.
        """
        if self._entry_count == 0 or not query.strip():
            return []

        t0 = time.time()
        n = min(self.top_k, self._entry_count)

        results = self._collection.query(
            query_texts=[query],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )

        self._total_retrievals += 1
        self._total_retrieval_time += time.time() - t0

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        budget = max_tokens
        entries: list[MemoryEntry] = []

        for doc, meta in zip(docs, metas):
            tokens = _count_tokens(doc)
            if tokens > budget:
                break
            entries.append(MemoryEntry(
                step=meta.get("step", -1),
                entry_type=meta.get("entry_type", "unknown"),
                content=doc,
                metadata=meta,
                timestamp=meta.get("timestamp", 0.0),
            ))
            budget -= tokens

        logger.debug(
            "RAG retrieve: query=%r → %d/%d entries (budget=%d→%d tokens)",
            query[:60], len(entries), n, max_tokens, budget,
        )
        return entries

    def get_context_block(self, max_tokens: int) -> str:
        """
        Return formatted relevant memories for injection into the system prompt.

        Uses the content of the most recently stored entry as the search query,
        which approximates "what am I currently working on?"
        Returns an empty string if no entries have been stored yet.
        """
        if self._entry_count == 0:
            return ""

        entries = self.retrieve(self._last_stored_content, max_tokens - _count_tokens(_CONTEXT_HEADER) - 10)
        if not entries:
            return ""

        parts = [_CONTEXT_HEADER]
        for i, entry in enumerate(entries, start=1):
            parts.append(_format_result_for_prompt(entry.content, i))

        return "\n\n".join(parts)

    def clear(self) -> None:
        """Delete and recreate the ChromaDB collection. Called between tasks."""
        try:
            self._collection.delete(ids=self._collection.get()["ids"])
        except Exception:
            # Recreate from scratch if delete fails
            self._collection = self._make_collection()

        self._entry_count = 0
        self._total_retrievals = 0
        self._total_retrieval_time = 0.0
        self._last_stored_content = ""
        logger.debug("RAG memory cleared")

    def get_stats(self) -> dict[str, Any]:
        avg_ms = (
            (self._total_retrieval_time / self._total_retrievals) * 1000
            if self._total_retrievals > 0
            else 0.0
        )
        return {
            "type": "rag",
            "entries_stored": self._entry_count,
            "total_retrievals": self._total_retrievals,
            "avg_retrieval_ms": round(avg_ms, 2),
            "embedding_model": self.embedding_model,
            "top_k": self.top_k,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_collection(self):
        """Create a fresh ephemeral (in-memory) ChromaDB collection."""
        import chromadb
        from chromadb.utils import embedding_functions

        client = chromadb.Client()   # ephemeral — data lives only in process memory

        ef = embedding_functions.DefaultEmbeddingFunction()

        return client.create_collection(
            name=f"rag_memory_{uuid.uuid4().hex[:8]}",
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
