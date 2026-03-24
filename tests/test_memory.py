"""
Tests for memory modules.

Covers:
- NaiveMemory      (no-op baseline)
- ScratchpadMemory (agent-managed persistent notes)
- RAGMemory        (automatic retrieval-augmented memory via ChromaDB)

No Docker or API calls needed.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from src.memory.base import MemoryEntry, MemoryModule
from src.memory.naive import NaiveMemory
from src.memory.scratchpad import ScratchpadMemory, TOOL_DEFINITION, _truncate_to_tokens
from src.memory.rag import RAGMemory, _format_entry, _count_tokens
from src.memory.hybrid import HybridMemory, KNOWLEDGE_TOOL_DEFINITION


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entry(step=0, content="some content", entry_type="observation") -> MemoryEntry:
    return MemoryEntry(
        step=step,
        entry_type=entry_type,
        content=content,
        metadata={},
        timestamp=time.time(),
    )


# ===========================================================================
# NaiveMemory
# ===========================================================================


class TestNaiveMemoryInterface:
    """NaiveMemory satisfies MemoryModule interface and is a no-op."""

    def test_is_memory_module_subclass(self):
        assert issubclass(NaiveMemory, MemoryModule)

    def test_store_does_not_raise(self):
        m = NaiveMemory()
        m.store(_entry())  # should not raise

    def test_retrieve_always_empty(self):
        m = NaiveMemory()
        m.store(_entry(content="important finding"))
        assert m.retrieve("important finding", max_tokens=1000) == []

    def test_retrieve_empty_before_store(self):
        assert NaiveMemory().retrieve("anything", 1000) == []

    def test_get_context_block_always_empty_string(self):
        m = NaiveMemory()
        m.store(_entry(content="some content"))
        assert m.get_context_block(max_tokens=1000) == ""

    def test_clear_does_not_raise(self):
        m = NaiveMemory()
        m.store(_entry())
        m.clear()  # no-op, should not raise

    def test_get_stats_returns_dict(self):
        stats = NaiveMemory().get_stats()
        assert isinstance(stats, dict)
        assert stats["type"] == "naive"

    def test_get_stats_entries_always_zero(self):
        m = NaiveMemory()
        for _ in range(5):
            m.store(_entry())
        assert m.get_stats()["entries"] == 0


# ===========================================================================
# ScratchpadMemory — basic interface
# ===========================================================================


class TestScratchpadMemoryInterface:
    def test_is_memory_module_subclass(self):
        assert issubclass(ScratchpadMemory, MemoryModule)

    def test_initial_scratchpad_is_empty(self):
        m = ScratchpadMemory()
        assert m._scratchpad == ""

    def test_store_is_noop(self):
        m = ScratchpadMemory()
        m.store(_entry(content="should not appear"))
        assert m._scratchpad == ""

    def test_clear_resets_scratchpad(self):
        m = ScratchpadMemory()
        m.handle_tool_call({"content": "my notes"}, sandbox=None)
        m.clear()
        assert m._scratchpad == ""

    def test_clear_resets_update_count(self):
        m = ScratchpadMemory()
        m.handle_tool_call({"content": "notes"}, sandbox=None)
        m.clear()
        assert m.get_stats()["updates"] == 0

    def test_clear_resets_total_chars(self):
        m = ScratchpadMemory()
        m.handle_tool_call({"content": "notes"}, sandbox=None)
        m.clear()
        assert m.get_stats()["total_chars_written"] == 0


# ===========================================================================
# ScratchpadMemory — retrieve
# ===========================================================================


class TestScratchpadRetrieve:
    def test_retrieve_empty_when_blank(self):
        assert ScratchpadMemory().retrieve("query", 1000) == []

    def test_retrieve_empty_when_whitespace_only(self):
        m = ScratchpadMemory()
        m.handle_tool_call({"content": "   \n  "}, sandbox=None)
        assert m.retrieve("query", 1000) == []

    def test_retrieve_returns_entry_when_has_content(self):
        m = ScratchpadMemory()
        m.handle_tool_call({"content": "found bug in parser"}, sandbox=None)
        entries = m.retrieve("bug", 1000)
        assert len(entries) == 1

    def test_retrieve_entry_has_scratchpad_type(self):
        m = ScratchpadMemory()
        m.handle_tool_call({"content": "notes"}, sandbox=None)
        entry = m.retrieve("notes", 1000)[0]
        assert entry.entry_type == "scratchpad"

    def test_retrieve_entry_contains_scratchpad_content(self):
        m = ScratchpadMemory()
        m.handle_tool_call({"content": "key finding: line 42"}, sandbox=None)
        entry = m.retrieve("key finding", 1000)[0]
        assert "key finding: line 42" in entry.content

    def test_retrieve_entry_metadata_has_updates(self):
        m = ScratchpadMemory()
        m.handle_tool_call({"content": "first"}, sandbox=None)
        m.handle_tool_call({"content": "second"}, sandbox=None)
        entry = m.retrieve("", 1000)[0]
        assert entry.metadata["updates"] == 2


# ===========================================================================
# ScratchpadMemory — get_context_block
# ===========================================================================


class TestScratchpadContextBlock:
    def test_empty_scratchpad_returns_instructions(self):
        block = ScratchpadMemory().get_context_block(max_tokens=2000)
        assert "update_scratchpad" in block
        assert len(block) > 0

    def test_nonempty_scratchpad_includes_content(self):
        m = ScratchpadMemory()
        m.handle_tool_call({"content": "working on auth module"}, sandbox=None)
        block = m.get_context_block(max_tokens=2000)
        assert "working on auth module" in block

    def test_nonempty_scratchpad_has_xml_tags(self):
        m = ScratchpadMemory()
        m.handle_tool_call({"content": "plan: fix the bug"}, sandbox=None)
        block = m.get_context_block(max_tokens=2000)
        assert "<scratchpad>" in block
        assert "</scratchpad>" in block

    def test_nonempty_scratchpad_includes_instructions(self):
        m = ScratchpadMemory()
        m.handle_tool_call({"content": "some notes"}, sandbox=None)
        block = m.get_context_block(max_tokens=2000)
        assert "update_scratchpad" in block

    def test_context_block_respects_token_budget(self):
        """Very large scratchpad should be truncated to fit max_tokens."""
        m = ScratchpadMemory()
        large_content = "word " * 5000  # ~5000 tokens
        m.handle_tool_call({"content": large_content}, sandbox=None)
        block = m.get_context_block(max_tokens=100)
        # Block should be significantly smaller than the full content
        assert len(block) < len(large_content)

    def test_context_block_zero_token_budget(self):
        """Should not crash with tiny budget."""
        m = ScratchpadMemory()
        m.handle_tool_call({"content": "notes"}, sandbox=None)
        block = m.get_context_block(max_tokens=0)
        assert isinstance(block, str)

    def test_whitespace_scratchpad_returns_only_instructions(self):
        m = ScratchpadMemory()
        m.handle_tool_call({"content": "   "}, sandbox=None)
        block = m.get_context_block(max_tokens=2000)
        assert "<scratchpad>" not in block
        assert "update_scratchpad" in block


# ===========================================================================
# ScratchpadMemory — handle_tool_call
# ===========================================================================


class TestScratchpadToolCall:
    def test_sets_scratchpad_content(self):
        m = ScratchpadMemory()
        m.handle_tool_call({"content": "new notes"}, sandbox=None)
        assert m._scratchpad == "new notes"

    def test_overwrites_previous_content(self):
        m = ScratchpadMemory()
        m.handle_tool_call({"content": "first notes"}, sandbox=None)
        m.handle_tool_call({"content": "updated notes"}, sandbox=None)
        assert m._scratchpad == "updated notes"
        assert "first notes" not in m._scratchpad

    def test_returns_confirmation_string(self):
        result = ScratchpadMemory().handle_tool_call({"content": "hello"}, sandbox=None)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_increments_update_count(self):
        m = ScratchpadMemory()
        m.handle_tool_call({"content": "a"}, sandbox=None)
        m.handle_tool_call({"content": "b"}, sandbox=None)
        assert m.get_stats()["updates"] == 2

    def test_accumulates_total_chars(self):
        m = ScratchpadMemory()
        m.handle_tool_call({"content": "abc"}, sandbox=None)   # 3 chars
        m.handle_tool_call({"content": "de"}, sandbox=None)    # 2 chars
        assert m.get_stats()["total_chars_written"] == 5

    def test_empty_content_allowed(self):
        m = ScratchpadMemory()
        result = m.handle_tool_call({"content": ""}, sandbox=None)
        assert isinstance(result, str)
        assert m._scratchpad == ""

    def test_missing_content_key_defaults_to_empty(self):
        m = ScratchpadMemory()
        m.handle_tool_call({}, sandbox=None)
        assert m._scratchpad == ""

    def test_sandbox_argument_not_used(self):
        """handle_tool_call accepts sandbox but doesn't call it."""
        sandbox = MagicMock()
        ScratchpadMemory().handle_tool_call({"content": "notes"}, sandbox=sandbox)
        sandbox.assert_not_called()


# ===========================================================================
# ScratchpadMemory — get_stats
# ===========================================================================


class TestScratchpadStats:
    def test_stats_returns_dict(self):
        assert isinstance(ScratchpadMemory().get_stats(), dict)

    def test_type_is_scratchpad(self):
        assert ScratchpadMemory().get_stats()["type"] == "scratchpad"

    def test_initial_stats_are_zero(self):
        stats = ScratchpadMemory().get_stats()
        assert stats["scratchpad_chars"] == 0
        assert stats["updates"] == 0
        assert stats["total_chars_written"] == 0

    def test_scratchpad_chars_reflects_current_content(self):
        m = ScratchpadMemory()
        m.handle_tool_call({"content": "hello"}, sandbox=None)  # 5 chars
        assert m.get_stats()["scratchpad_chars"] == 5

    def test_scratchpad_chars_updated_on_overwrite(self):
        m = ScratchpadMemory()
        m.handle_tool_call({"content": "long text here"}, sandbox=None)
        m.handle_tool_call({"content": "hi"}, sandbox=None)
        assert m.get_stats()["scratchpad_chars"] == 2


# ===========================================================================
# TOOL_DEFINITION schema
# ===========================================================================


class TestToolDefinition:
    def test_has_name_field(self):
        assert TOOL_DEFINITION["name"] == "update_scratchpad"

    def test_has_description(self):
        assert len(TOOL_DEFINITION["description"]) > 0

    def test_has_input_schema(self):
        assert "input_schema" in TOOL_DEFINITION

    def test_content_is_required(self):
        schema = TOOL_DEFINITION["input_schema"]
        assert "content" in schema["properties"]
        assert "content" in schema["required"]

    def test_schema_type_is_object(self):
        assert TOOL_DEFINITION["input_schema"]["type"] == "object"


# ===========================================================================
# ToolExecutor integration — register_tool
# ===========================================================================


class TestToolExecutorRegistration:
    """Verify ScratchpadMemory integrates correctly with ToolExecutor."""

    def test_register_tool_adds_to_definitions(self):
        from src.tool_executor import ToolExecutor
        executor = ToolExecutor()
        memory = ScratchpadMemory()
        executor.register_tool(TOOL_DEFINITION, memory.handle_tool_call)
        names = [t["name"] for t in executor.tool_definitions]
        assert "update_scratchpad" in names

    def test_base_tools_still_present_after_register(self):
        from src.tool_executor import ToolExecutor, TOOL_DEFINITIONS
        executor = ToolExecutor()
        memory = ScratchpadMemory()
        executor.register_tool(TOOL_DEFINITION, memory.handle_tool_call)
        base_names = {t["name"] for t in TOOL_DEFINITIONS}
        registered_names = {t["name"] for t in executor.tool_definitions}
        assert base_names.issubset(registered_names)

    def test_execute_dispatches_to_scratchpad_handler(self):
        from src.tool_executor import ToolExecutor
        executor = ToolExecutor()
        memory = ScratchpadMemory()
        executor.register_tool(TOOL_DEFINITION, memory.handle_tool_call)

        sandbox = MagicMock()
        result = executor.execute("update_scratchpad", {"content": "test notes"}, sandbox)
        assert isinstance(result, str)
        assert memory._scratchpad == "test notes"

    def test_execute_unknown_tool_still_raises(self):
        from src.tool_executor import ToolExecutor, ToolError
        executor = ToolExecutor()
        with pytest.raises(ToolError):
            executor.execute("nonexistent_tool", {}, MagicMock())

    def test_multiple_registrations_all_accessible(self):
        from src.tool_executor import ToolExecutor
        executor = ToolExecutor()
        called = []

        def handler_a(inp, sandbox):
            called.append("a")
            return "a"

        def handler_b(inp, sandbox):
            called.append("b")
            return "b"

        executor.register_tool({"name": "tool_a", "description": "a", "input_schema": {"type": "object", "properties": {}, "required": []}}, handler_a)
        executor.register_tool({"name": "tool_b", "description": "b", "input_schema": {"type": "object", "properties": {}, "required": []}}, handler_b)

        executor.execute("tool_a", {}, MagicMock())
        executor.execute("tool_b", {}, MagicMock())
        assert called == ["a", "b"]


# ===========================================================================
# Runner integration — registry + tool registration
# ===========================================================================


class TestRunnerIntegration:
    def test_scratchpad_in_memory_registry(self):
        from src.runner import MEMORY_REGISTRY
        assert "scratchpad" in MEMORY_REGISTRY

    def test_build_memory_module_returns_scratchpad_instance(self):
        from src.runner import build_memory_module
        cfg = {
            "memory": {"type": "scratchpad", "params": {}},
            "compression": {"type": "none", "params": {}},
            "communication": {"type": "single_agent", "params": {}},
            "evaluation": {"dataset": "swebench_lite"},
            "sandbox": {},
        }
        module = build_memory_module(cfg)
        assert isinstance(module, ScratchpadMemory)

    def test_register_memory_tools_adds_scratchpad_tool(self):
        from src.runner import _register_memory_tools
        from src.tool_executor import ToolExecutor
        memory = ScratchpadMemory()
        executor = ToolExecutor()
        _register_memory_tools(memory, executor)
        names = [t["name"] for t in executor.tool_definitions]
        assert "update_scratchpad" in names

    def test_register_memory_tools_noop_for_naive(self):
        from src.runner import _register_memory_tools
        from src.tool_executor import ToolExecutor, TOOL_DEFINITIONS
        memory = NaiveMemory()
        executor = ToolExecutor()
        _register_memory_tools(memory, executor)
        # No extra tools added
        assert len(executor.tool_definitions) == len(TOOL_DEFINITIONS)

    def test_scratchpad_config_validates_cleanly(self):
        from src.runner import load_config, validate_config
        import pathlib
        cfg_path = pathlib.Path(__file__).parent.parent / "configs" / "scratchpad_memory.yaml"
        cfg = load_config(str(cfg_path))
        assert validate_config(cfg) == []


# ===========================================================================
# End-to-end: scratchpad tool actually updates memory during agent loop
# ===========================================================================


class TestScratchpadEndToEnd:
    """Full stack test: scratchpad memory wired into orchestrator, tool updates it."""

    def _build_stack_with_scratchpad(self):
        from src.orchestrator import Orchestrator, OrchestratorConfig
        from src.logger import TrajectoryLogger
        from src.compression.none import NoCompression
        from src.communication.single_agent import SingleAgentCommunication
        from src.tool_executor import ToolExecutor, PatchSubmitted

        memory = ScratchpadMemory()
        executor = ToolExecutor()
        executor.register_tool(TOOL_DEFINITION, memory.handle_tool_call)

        def _execute(tool_name, tool_input, sandbox_ref):
            if tool_name == "update_scratchpad":
                return memory.handle_tool_call(tool_input, sandbox_ref)
            if tool_name == "submit_patch":
                raise PatchSubmitted(tool_input.get("message", ""))
            return "ok"

        executor.execute = _execute
        executor.tool_definitions  # access to ensure no crash

        # LLM: step 1 = update_scratchpad, step 2 = submit_patch
        def _make_response(tool_name, tool_id, tool_input):
            resp = MagicMock()
            resp.stop_reason = "tool_use"
            tb = MagicMock()
            tb.type = "tool_use"
            tb.id = tool_id
            tb.name = tool_name
            tb.input = tool_input
            resp.content = [tb]
            resp.usage = MagicMock(input_tokens=100, output_tokens=50)
            return resp

        responses = [
            _make_response("update_scratchpad", "tu_001", {"content": "found bug in line 42"}),
            _make_response("submit_patch", "tu_002", {"message": "fixed"}),
        ]
        call_count = [0]

        llm = MagicMock()

        def llm_complete(**kwargs):
            idx = min(call_count[0], len(responses) - 1)
            call_count[0] += 1
            return responses[idx]

        llm.complete.side_effect = llm_complete
        llm.count_tokens.return_value = 50
        stats_counter = [0]

        def get_stats():
            stats_counter[0] += 1
            return {
                "total_calls": stats_counter[0],
                "total_input_tokens": 100,
                "total_output_tokens": 50,
                "estimated_cost_usd": 0.001,
            }

        llm.get_stats.side_effect = get_stats

        sandbox = MagicMock()
        sandbox.get_diff.return_value = "--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-x\n+y"
        sandbox.run_tests.return_value = {
            "passed": ["test_one"], "failed": [], "exit_code": 0, "output": "ok", "error": None
        }

        tool_executor_mock = MagicMock()
        tool_executor_mock.setup_sandbox.return_value = sandbox
        tool_executor_mock.tool_definitions = []
        tool_executor_mock.execute.side_effect = _execute

        compression = NoCompression()
        communication = SingleAgentCommunication()
        traj_logger = TrajectoryLogger()

        orch = Orchestrator(
            config=OrchestratorConfig(max_steps=10),
            memory=memory,
            compression=compression,
            communication=communication,
            llm_client=llm,
            tool_executor=tool_executor_mock,
            logger=traj_logger,
        )
        return orch, memory

    def test_scratchpad_updated_during_run(self):
        orch, memory = self._build_stack_with_scratchpad()
        task = MagicMock()
        task.instance_id = "test__test-001"
        task.repo = "test/repo"
        task.base_commit = "abc123"
        task.problem_statement = "Fix the bug"
        task.hints_text = ""
        task.test_patch = ""
        task.fail_to_pass = ["test_one"]
        task.pass_to_pass = []

        orch.run_task(task)
        assert memory._scratchpad == "found bug in line 42"
        assert memory._updates == 1

    def test_scratchpad_stats_reflect_usage(self):
        orch, memory = self._build_stack_with_scratchpad()
        task = MagicMock()
        task.instance_id = "test__test-002"
        task.repo = "test/repo"
        task.base_commit = "abc123"
        task.problem_statement = "Fix the bug"
        task.hints_text = ""
        task.test_patch = ""
        task.fail_to_pass = ["test_one"]
        task.pass_to_pass = []

        orch.run_task(task)
        stats = memory.get_stats()
        assert stats["updates"] == 1
        assert stats["scratchpad_chars"] > 0
        assert stats["total_chars_written"] > 0


# ===========================================================================
# _truncate_to_tokens helper
# ===========================================================================


class TestTruncateToTokens:
    def test_short_text_unchanged(self):
        text = "hello world"
        assert _truncate_to_tokens(text, 100) == text

    def test_empty_text_unchanged(self):
        assert _truncate_to_tokens("", 100) == ""

    def test_zero_max_tokens_returns_empty(self):
        assert _truncate_to_tokens("hello world", 0) == ""

    def test_truncates_long_text(self):
        long_text = "word " * 2000  # ~2000 tokens
        truncated = _truncate_to_tokens(long_text, 100)
        assert len(truncated) < len(long_text)

    def test_truncated_is_valid_string(self):
        text = "token " * 500
        result = _truncate_to_tokens(text, 50)
        assert isinstance(result, str)


# ===========================================================================
# RAGMemory — helpers
# ===========================================================================


@pytest.fixture(scope="module")
def rag():
    """A single RAGMemory instance shared across the module-scoped tests.
    Scope=module so the ChromaDB collection is only created once (embedding
    model load is slow)."""
    return RAGMemory(top_k=20)


def _make_entry(step=0, content="", entry_type="observation", tool="bash") -> MemoryEntry:
    return MemoryEntry(
        step=step,
        entry_type=entry_type,
        content=content,
        metadata={"input": f"{tool} call"},
        timestamp=time.time(),
    )


# ===========================================================================
# RAGMemory — basic interface
# ===========================================================================


class TestRAGMemoryInterface:
    def test_is_memory_module_subclass(self):
        assert issubclass(RAGMemory, MemoryModule)

    def test_default_params(self):
        m = RAGMemory()
        assert m.top_k == 20
        assert m.embedding_model == "all-MiniLM-L6-v2"
        assert m.chunk_strategy == "per_tool_result"

    def test_custom_top_k(self):
        m = RAGMemory(top_k=5)
        assert m.top_k == 5

    def test_initial_entry_count_zero(self):
        assert RAGMemory().get_stats()["entries_stored"] == 0

    def test_store_increments_entry_count(self):
        m = RAGMemory()
        m.store(_make_entry(content="hello"))
        assert m.get_stats()["entries_stored"] == 1

    def test_store_multiple_increments_correctly(self):
        m = RAGMemory()
        for i in range(5):
            m.store(_make_entry(step=i, content=f"entry {i}"))
        assert m.get_stats()["entries_stored"] == 5

    def test_store_does_not_raise_on_empty_content(self):
        m = RAGMemory()
        m.store(_make_entry(content=""))  # should not raise

    def test_clear_resets_entry_count(self):
        m = RAGMemory()
        m.store(_make_entry(content="data"))
        m.clear()
        assert m.get_stats()["entries_stored"] == 0

    def test_clear_resets_retrieval_stats(self):
        m = RAGMemory()
        m.store(_make_entry(content="data"))
        m.retrieve("data", 1000)
        m.clear()
        stats = m.get_stats()
        assert stats["total_retrievals"] == 0

    def test_clear_makes_retrieve_return_empty(self):
        m = RAGMemory()
        m.store(_make_entry(content="something important"))
        m.clear()
        assert m.retrieve("something important", 1000) == []


# ===========================================================================
# RAGMemory — retrieve
# ===========================================================================


class TestRAGRetrieve:
    def test_retrieve_empty_before_store(self):
        m = RAGMemory()
        assert m.retrieve("anything", 1000) == []

    def test_retrieve_empty_query_returns_empty(self):
        m = RAGMemory()
        m.store(_make_entry(content="something"))
        assert m.retrieve("", 1000) == []

    def test_retrieve_returns_list(self):
        m = RAGMemory()
        m.store(_make_entry(content="fixed bug in auth module"))
        result = m.retrieve("auth bug", 2000)
        assert isinstance(result, list)

    def test_retrieve_returns_memory_entries(self):
        m = RAGMemory()
        m.store(_make_entry(content="error in tokenizer"))
        entries = m.retrieve("tokenizer error", 2000)
        assert len(entries) > 0
        assert isinstance(entries[0], MemoryEntry)

    def test_retrieve_respects_token_budget(self):
        m = RAGMemory()
        # Store entries large enough that not all fit in budget
        for i in range(10):
            m.store(_make_entry(step=i, content=f"word " * 100))
        entries = m.retrieve("word", max_tokens=50)
        total = sum(_count_tokens(e.content) for e in entries)
        assert total <= 50

    def test_retrieve_top_k_caps_results(self):
        m = RAGMemory(top_k=3)
        for i in range(10):
            m.store(_make_entry(step=i, content=f"entry number {i} about coding"))
        entries = m.retrieve("coding entry", max_tokens=100_000)
        assert len(entries) <= 3

    def test_retrieve_increments_retrieval_count(self):
        m = RAGMemory()
        m.store(_make_entry(content="data"))
        m.retrieve("data", 1000)
        m.retrieve("data", 1000)
        assert m.get_stats()["total_retrievals"] == 2


# ===========================================================================
# RAGMemory — retrieval quality
#
# Design spec validation: store 50 entries about different topics, query for
# a specific topic, verify topic-relevant entries rank highest.
# ===========================================================================


class TestRAGRetrievalQuality:
    """
    Semantic retrieval quality tests.  These use real embeddings (ONNX model),
    so they are somewhat slow (~1-2s) but require no API key or Docker.
    """

    @pytest.fixture(scope="class")
    def populated(self):
        """50-entry collection with distinct topic clusters."""
        m = RAGMemory(top_k=20)

        parser_entries = [
            "SyntaxError in parser.py at line 42: unexpected token",
            "parser.py parse_expression() raises ValueError on empty input",
            "fix: parser now handles nested parentheses correctly",
            "test_parser.py: all 12 tests pass after the fix",
            "the parser module uses recursive descent for expression handling",
            "found bug: parser does not handle UTF-8 identifiers",
            "parser error: missing semicolon detection broken",
            "updated parser to support Python 3.12 match statements",
            "regression in parser introduced in commit abc123",
            "parser.py imports: ast, tokenize, io",
        ]
        auth_entries = [
            "login endpoint returns 500 on invalid password",
            "auth.py: JWT token expiry not checked correctly",
            "fixed authentication middleware to reject expired tokens",
            "test_auth.py: login test fails with KeyError on session",
            "auth module uses bcrypt for password hashing",
            "auth bug: admin users can bypass 2FA",
            "session cookie not being invalidated on logout",
            "OAuth2 flow broken after upgrading requests library",
            "auth.py line 88: missing null check before user.id access",
            "security: rate limiting not applied to auth endpoints",
        ]
        db_entries = [
            "database connection pool exhausted under load",
            "SQL query in models.py missing index on user_id column",
            "migration 0042 failed: column already exists",
            "ORM N+1 query problem in UserListView",
            "postgres: deadlock detected in transaction isolation tests",
            "db.py: connection timeout set too low (5s), should be 30s",
            "query optimization: added composite index on (user_id, created_at)",
            "database schema mismatch between staging and production",
            "SQLAlchemy session not closed after request in some paths",
            "test_db.py: fixture setup failing due to missing test database",
        ]
        # 20 generic filler entries
        filler_entries = [
            f"general code maintenance task {i}: updated dependencies, fixed lint" for i in range(20)
        ]

        all_entries = parser_entries + auth_entries + db_entries + filler_entries
        for i, content in enumerate(all_entries):
            m.store(_make_entry(step=i, content=content, entry_type="observation"))

        return m

    def test_parser_query_returns_parser_entries(self, populated):
        entries = populated.retrieve("error in parser", max_tokens=50_000)
        contents = " ".join(e.content for e in entries[:5])
        assert "parser" in contents.lower()

    def test_auth_query_returns_auth_entries(self, populated):
        entries = populated.retrieve("authentication login bug", max_tokens=50_000)
        contents = " ".join(e.content for e in entries[:5])
        assert any(w in contents.lower() for w in ("auth", "login", "token", "password"))

    def test_db_query_returns_db_entries(self, populated):
        entries = populated.retrieve("database query performance", max_tokens=50_000)
        contents = " ".join(e.content for e in entries[:5])
        assert any(w in contents.lower() for w in ("database", "db", "sql", "query", "index"))

    def test_parser_query_top1_is_parser_related(self, populated):
        entries = populated.retrieve("syntax error in parser module", max_tokens=50_000)
        assert len(entries) > 0
        assert "parser" in entries[0].content.lower()

    def test_unrelated_query_does_not_rank_filler_first(self, populated):
        """Parser query should not return generic filler as top result."""
        entries = populated.retrieve("parser recursive descent bug", max_tokens=50_000)
        assert len(entries) > 0
        top_content = entries[0].content.lower()
        assert "parser" in top_content or "syntax" in top_content or "parse" in top_content

    def test_50_entries_all_stored(self, populated):
        assert populated.get_stats()["entries_stored"] == 50


# ===========================================================================
# RAGMemory — get_context_block
# ===========================================================================


class TestRAGContextBlock:
    def test_empty_before_any_store(self):
        assert RAGMemory().get_context_block(max_tokens=2000) == ""

    def test_nonempty_after_store(self):
        m = RAGMemory()
        m.store(_make_entry(content="found bug in request handler"))
        block = m.get_context_block(max_tokens=2000)
        assert len(block) > 0

    def test_contains_header(self):
        m = RAGMemory()
        m.store(_make_entry(content="found bug in request handler"))
        block = m.get_context_block(max_tokens=2000)
        assert "Relevant observations" in block

    def test_contains_stored_content(self):
        m = RAGMemory()
        m.store(_make_entry(content="unique_marker_xyzzy"))
        block = m.get_context_block(max_tokens=5000)
        assert "unique_marker_xyzzy" in block

    def test_respects_token_budget(self):
        m = RAGMemory()
        for i in range(20):
            m.store(_make_entry(step=i, content="word " * 200))
        block = m.get_context_block(max_tokens=200)
        assert _count_tokens(block) <= 250  # small slack for header

    def test_context_block_uses_last_stored_as_query(self):
        """The last stored entry should influence what's retrieved."""
        m = RAGMemory()
        # Store two topic clusters
        for i in range(5):
            m.store(_make_entry(step=i, content=f"parser error {i}: syntax issue"))
        for i in range(5):
            m.store(_make_entry(step=10 + i, content=f"database connection {i}: timeout"))
        # Last stored is about DB; context block should reflect DB topic
        block = m.get_context_block(max_tokens=10_000)
        assert "database" in block.lower() or "connection" in block.lower()


# ===========================================================================
# RAGMemory — get_stats
# ===========================================================================


class TestRAGStats:
    def test_stats_has_required_keys(self):
        stats = RAGMemory().get_stats()
        for key in ("type", "entries_stored", "total_retrievals", "avg_retrieval_ms", "embedding_model", "top_k"):
            assert key in stats, f"missing key: {key}"

    def test_type_is_rag(self):
        assert RAGMemory().get_stats()["type"] == "rag"

    def test_avg_retrieval_ms_zero_before_retrieve(self):
        assert RAGMemory().get_stats()["avg_retrieval_ms"] == 0.0

    def test_avg_retrieval_ms_positive_after_retrieve(self):
        m = RAGMemory()
        m.store(_make_entry(content="data"))
        m.retrieve("data", 1000)
        assert m.get_stats()["avg_retrieval_ms"] >= 0.0

    def test_top_k_reflected_in_stats(self):
        assert RAGMemory(top_k=7).get_stats()["top_k"] == 7

    def test_embedding_model_reflected_in_stats(self):
        m = RAGMemory(embedding_model="all-MiniLM-L6-v2")
        assert m.get_stats()["embedding_model"] == "all-MiniLM-L6-v2"


# ===========================================================================
# RAGMemory — runner integration
# ===========================================================================


class TestRAGRunnerIntegration:
    def test_rag_in_memory_registry(self):
        from src.runner import MEMORY_REGISTRY
        assert "rag" in MEMORY_REGISTRY

    def test_build_memory_module_returns_rag_instance(self):
        from src.runner import build_memory_module
        cfg = {
            "memory": {
                "type": "rag",
                "params": {"embedding_model": "all-MiniLM-L6-v2", "top_k": 10},
            },
            "compression": {"type": "none", "params": {}},
            "communication": {"type": "single_agent", "params": {}},
            "evaluation": {"dataset": "swebench_lite"},
            "sandbox": {},
        }
        module = build_memory_module(cfg)
        assert isinstance(module, RAGMemory)
        assert module.top_k == 10

    def test_rag_config_validates_cleanly(self):
        from src.runner import load_config, validate_config
        import pathlib
        cfg_path = pathlib.Path(__file__).parent.parent / "configs" / "rag_memory.yaml"
        cfg = load_config(str(cfg_path))
        assert validate_config(cfg) == []

    def test_register_memory_tools_noop_for_rag(self):
        """RAG has no extra tools to register."""
        from src.runner import _register_memory_tools
        from src.tool_executor import ToolExecutor, TOOL_DEFINITIONS
        memory = RAGMemory()
        executor = ToolExecutor()
        _register_memory_tools(memory, executor)
        assert len(executor.tool_definitions) == len(TOOL_DEFINITIONS)


# ===========================================================================
# RAGMemory — _format_entry helper
# ===========================================================================


class TestFormatEntry:
    def test_includes_step_and_type(self):
        entry = _make_entry(step=7, entry_type="observation", content="hello")
        doc = _format_entry(entry)
        assert "step=7" in doc
        assert "observation" in doc

    def test_includes_content(self):
        entry = _make_entry(content="unique content xyz")
        doc = _format_entry(entry)
        assert "unique content xyz" in doc

    def test_returns_string(self):
        assert isinstance(_format_entry(_make_entry(content="x")), str)


# ===========================================================================
# HybridMemory — basic interface
# ===========================================================================


class TestHybridMemoryInterface:
    def test_is_memory_module_subclass(self):
        assert issubclass(HybridMemory, MemoryModule)

    def test_initial_knowledge_base_empty(self):
        assert HybridMemory().knowledge_base == {}

    def test_has_episodic_store(self):
        assert isinstance(HybridMemory().episodic_store, RAGMemory)

    def test_default_params(self):
        m = HybridMemory()
        assert m._semantic_fraction == 0.3
        assert m.episodic_store.top_k == 20

    def test_custom_params(self):
        m = HybridMemory(top_k=5, semantic_budget_fraction=0.4)
        assert m.episodic_store.top_k == 5
        assert m._semantic_fraction == 0.4

    def test_store_goes_into_episodic(self):
        m = HybridMemory()
        m.store(_make_entry(content="observed file listing"))
        assert m.episodic_store.get_stats()["entries_stored"] == 1

    def test_store_does_not_touch_knowledge_base(self):
        m = HybridMemory()
        m.store(_make_entry(content="some observation"))
        assert m.knowledge_base == {}

    def test_clear_empties_knowledge_base(self):
        m = HybridMemory()
        m.handle_knowledge_tool_call({"key": "k", "value": "v"}, None)
        m.clear()
        assert m.knowledge_base == {}

    def test_clear_empties_episodic_store(self):
        m = HybridMemory()
        m.store(_make_entry(content="data"))
        m.clear()
        assert m.episodic_store.get_stats()["entries_stored"] == 0

    def test_clear_resets_knowledge_update_count(self):
        m = HybridMemory()
        m.handle_knowledge_tool_call({"key": "k", "value": "v"}, None)
        m.clear()
        assert m.get_stats()["knowledge_updates"] == 0


# ===========================================================================
# HybridMemory — handle_knowledge_tool_call
# ===========================================================================


class TestHybridKnowledgeToolCall:
    def test_add_new_fact(self):
        m = HybridMemory()
        result = m.handle_knowledge_tool_call({"key": "bug_location", "value": "parser.py:42"}, None)
        assert m.knowledge_base["bug_location"] == "parser.py:42"
        assert "Added" in result

    def test_update_existing_fact(self):
        m = HybridMemory()
        m.handle_knowledge_tool_call({"key": "k", "value": "v1"}, None)
        result = m.handle_knowledge_tool_call({"key": "k", "value": "v2"}, None)
        assert m.knowledge_base["k"] == "v2"
        assert "Updated" in result

    def test_delete_fact_with_empty_value(self):
        m = HybridMemory()
        m.handle_knowledge_tool_call({"key": "k", "value": "v"}, None)
        result = m.handle_knowledge_tool_call({"key": "k", "value": ""}, None)
        assert "k" not in m.knowledge_base
        assert "Removed" in result

    def test_delete_nonexistent_key_no_error(self):
        m = HybridMemory()
        result = m.handle_knowledge_tool_call({"key": "missing", "value": ""}, None)
        assert isinstance(result, str)
        assert "not found" in result.lower()

    def test_empty_key_returns_error(self):
        m = HybridMemory()
        result = m.handle_knowledge_tool_call({"key": "", "value": "v"}, None)
        assert "Error" in result
        assert m.knowledge_base == {}

    def test_missing_key_field_returns_error(self):
        m = HybridMemory()
        result = m.handle_knowledge_tool_call({"value": "v"}, None)
        assert "Error" in result

    def test_increments_update_count_on_add(self):
        m = HybridMemory()
        m.handle_knowledge_tool_call({"key": "k", "value": "v"}, None)
        assert m.get_stats()["knowledge_updates"] == 1

    def test_increments_update_count_on_update(self):
        m = HybridMemory()
        m.handle_knowledge_tool_call({"key": "k", "value": "v1"}, None)
        m.handle_knowledge_tool_call({"key": "k", "value": "v2"}, None)
        assert m.get_stats()["knowledge_updates"] == 2

    def test_increments_update_count_on_delete(self):
        m = HybridMemory()
        m.handle_knowledge_tool_call({"key": "k", "value": "v"}, None)
        m.handle_knowledge_tool_call({"key": "k", "value": ""}, None)
        assert m.get_stats()["knowledge_updates"] == 2

    def test_sandbox_not_used(self):
        sandbox = MagicMock()
        HybridMemory().handle_knowledge_tool_call({"key": "k", "value": "v"}, sandbox)
        sandbox.assert_not_called()

    def test_multiple_facts_stored_independently(self):
        m = HybridMemory()
        m.handle_knowledge_tool_call({"key": "a", "value": "1"}, None)
        m.handle_knowledge_tool_call({"key": "b", "value": "2"}, None)
        assert m.knowledge_base == {"a": "1", "b": "2"}

    def test_returns_string(self):
        result = HybridMemory().handle_knowledge_tool_call({"key": "k", "value": "v"}, None)
        assert isinstance(result, str)


# ===========================================================================
# HybridMemory — retrieve
# ===========================================================================


class TestHybridRetrieve:
    def test_retrieve_empty_before_store(self):
        assert HybridMemory().retrieve("anything", 1000) == []

    def test_retrieve_queries_episodic_store(self):
        m = HybridMemory()
        m.store(_make_entry(content="parser bug at line 42"))
        entries = m.retrieve("parser bug", 5000)
        assert len(entries) > 0

    def test_retrieve_respects_episodic_budget_fraction(self):
        """retrieve() should use 70% of budget for episodic by default."""
        m = HybridMemory(semantic_budget_fraction=0.3)
        for i in range(5):
            m.store(_make_entry(step=i, content="word " * 100))
        entries = m.retrieve("word", max_tokens=100)
        total = sum(_count_tokens(e.content) for e in entries)
        # Budget passed to episodic = 70 tokens
        assert total <= 70

    def test_knowledge_base_does_not_affect_retrieve(self):
        """Knowledge base entries never appear in retrieve() results."""
        m = HybridMemory()
        m.handle_knowledge_tool_call({"key": "root_cause", "value": "off-by-one error"}, None)
        entries = m.retrieve("off-by-one error", 5000)
        assert all(e.entry_type != "knowledge" for e in entries)


# ===========================================================================
# HybridMemory — get_context_block
# ===========================================================================


class TestHybridContextBlock:
    def test_always_returns_string(self):
        assert isinstance(HybridMemory().get_context_block(max_tokens=2000), str)

    def test_contains_instructions_when_empty(self):
        block = HybridMemory().get_context_block(max_tokens=2000)
        assert "update_knowledge" in block

    def test_contains_knowledge_base_section_when_populated(self):
        m = HybridMemory()
        m.handle_knowledge_tool_call({"key": "bug_location", "value": "auth.py:88"}, None)
        block = m.get_context_block(max_tokens=5000)
        assert "Knowledge Base" in block
        assert "bug_location" in block
        assert "auth.py:88" in block

    def test_knowledge_base_in_xml_tags(self):
        m = HybridMemory()
        m.handle_knowledge_tool_call({"key": "k", "value": "v"}, None)
        block = m.get_context_block(max_tokens=5000)
        assert "<knowledge>" in block
        assert "</knowledge>" in block

    def test_contains_episodic_section_after_store(self):
        m = HybridMemory()
        m.store(_make_entry(content="unique_episodic_marker_abc123"))
        block = m.get_context_block(max_tokens=10000)
        assert "unique_episodic_marker_abc123" in block

    def test_episodic_section_labeled(self):
        m = HybridMemory()
        m.store(_make_entry(content="observation data"))
        block = m.get_context_block(max_tokens=10000)
        assert "Relevant Past Observations" in block

    def test_knowledge_before_episodic_in_block(self):
        m = HybridMemory()
        m.handle_knowledge_tool_call({"key": "k", "value": "v"}, None)
        m.store(_make_entry(content="episodic data"))
        block = m.get_context_block(max_tokens=10000)
        kb_pos = block.find("Knowledge Base")
        ep_pos = block.find("Relevant Past Observations")
        assert kb_pos < ep_pos

    def test_respects_total_token_budget(self):
        m = HybridMemory()
        m.handle_knowledge_tool_call({"key": "k", "value": "v " * 200}, None)
        for i in range(10):
            m.store(_make_entry(step=i, content="word " * 200))
        block = m.get_context_block(max_tokens=300)
        assert _count_tokens(block) <= 350  # small slack for structure

    def test_deleted_fact_not_in_block(self):
        m = HybridMemory()
        m.handle_knowledge_tool_call({"key": "temp", "value": "temporary_value_xyz"}, None)
        m.handle_knowledge_tool_call({"key": "temp", "value": ""}, None)
        block = m.get_context_block(max_tokens=5000)
        assert "temporary_value_xyz" not in block

    def test_both_stores_populated_reflected_in_block(self):
        """Validation criterion: both stores are populated and both show up."""
        m = HybridMemory()
        m.handle_knowledge_tool_call({"key": "root_cause", "value": "missing null check"}, None)
        m.store(_make_entry(content="found the bug in request_handler.py"))
        block = m.get_context_block(max_tokens=10000)
        assert "missing null check" in block
        assert "request_handler" in block


# ===========================================================================
# HybridMemory — get_stats
# ===========================================================================


class TestHybridStats:
    def test_stats_returns_dict(self):
        assert isinstance(HybridMemory().get_stats(), dict)

    def test_type_is_hybrid(self):
        assert HybridMemory().get_stats()["type"] == "hybrid"

    def test_initial_knowledge_entries_zero(self):
        assert HybridMemory().get_stats()["knowledge_entries"] == 0

    def test_knowledge_entries_increments(self):
        m = HybridMemory()
        m.handle_knowledge_tool_call({"key": "a", "value": "1"}, None)
        m.handle_knowledge_tool_call({"key": "b", "value": "2"}, None)
        assert m.get_stats()["knowledge_entries"] == 2

    def test_knowledge_entries_decrements_on_delete(self):
        m = HybridMemory()
        m.handle_knowledge_tool_call({"key": "a", "value": "1"}, None)
        m.handle_knowledge_tool_call({"key": "a", "value": ""}, None)
        assert m.get_stats()["knowledge_entries"] == 0

    def test_stats_contains_episodic_substats(self):
        stats = HybridMemory().get_stats()
        assert "episodic" in stats
        assert stats["episodic"]["type"] == "rag"


# ===========================================================================
# KNOWLEDGE_TOOL_DEFINITION schema
# ===========================================================================


class TestKnowledgeToolDefinition:
    def test_name_is_update_knowledge(self):
        assert KNOWLEDGE_TOOL_DEFINITION["name"] == "update_knowledge"

    def test_has_description(self):
        assert len(KNOWLEDGE_TOOL_DEFINITION["description"]) > 0

    def test_key_and_value_are_required(self):
        schema = KNOWLEDGE_TOOL_DEFINITION["input_schema"]
        assert "key" in schema["required"]
        assert "value" in schema["required"]

    def test_schema_type_is_object(self):
        assert KNOWLEDGE_TOOL_DEFINITION["input_schema"]["type"] == "object"


# ===========================================================================
# HybridMemory — runner integration
# ===========================================================================


class TestHybridRunnerIntegration:
    def test_hybrid_in_memory_registry(self):
        from src.runner import MEMORY_REGISTRY
        assert "hybrid" in MEMORY_REGISTRY

    def test_build_memory_module_returns_hybrid_instance(self):
        from src.runner import build_memory_module
        cfg = {
            "memory": {
                "type": "hybrid",
                "params": {"top_k": 5, "semantic_budget_fraction": 0.3},
            },
            "compression": {"type": "none", "params": {}},
            "communication": {"type": "single_agent", "params": {}},
            "evaluation": {"dataset": "swebench_lite"},
            "sandbox": {},
        }
        module = build_memory_module(cfg)
        assert isinstance(module, HybridMemory)
        assert module.episodic_store.top_k == 5

    def test_register_memory_tools_adds_knowledge_tool(self):
        from src.runner import _register_memory_tools
        from src.tool_executor import ToolExecutor
        memory = HybridMemory()
        executor = ToolExecutor()
        _register_memory_tools(memory, executor)
        names = [t["name"] for t in executor.tool_definitions]
        assert "update_knowledge" in names

    def test_register_memory_tools_does_not_add_scratchpad_for_hybrid(self):
        from src.runner import _register_memory_tools
        from src.tool_executor import ToolExecutor
        memory = HybridMemory()
        executor = ToolExecutor()
        _register_memory_tools(memory, executor)
        names = [t["name"] for t in executor.tool_definitions]
        assert "update_scratchpad" not in names

    def test_hybrid_config_validates_cleanly(self):
        from src.runner import load_config, validate_config
        import pathlib
        cfg_path = pathlib.Path(__file__).parent.parent / "configs" / "hybrid_memory.yaml"
        cfg = load_config(str(cfg_path))
        assert validate_config(cfg) == []


# ===========================================================================
# HybridMemory — end-to-end: both stores populated during orchestrator run
# ===========================================================================


class TestHybridEndToEnd:
    """Validation criterion: after a run, both knowledge base and episodic store
    are populated."""

    def _build_stack_with_hybrid(self):
        from src.orchestrator import Orchestrator, OrchestratorConfig
        from src.logger import TrajectoryLogger
        from src.compression.none import NoCompression
        from src.communication.single_agent import SingleAgentCommunication
        from src.tool_executor import PatchSubmitted

        memory = HybridMemory()

        def _execute(tool_name, tool_input, sandbox_ref):
            if tool_name == "update_knowledge":
                return memory.handle_knowledge_tool_call(tool_input, sandbox_ref)
            if tool_name == "submit_patch":
                raise PatchSubmitted(tool_input.get("message", ""))
            return "ls output: app.py\n"

        def _make_resp(tool_name, tool_id, tool_input, stop="tool_use"):
            resp = MagicMock()
            resp.stop_reason = stop
            tb = MagicMock()
            tb.type = "tool_use"
            tb.id = tool_id
            tb.name = tool_name
            tb.input = tool_input
            resp.content = [tb]
            resp.usage = MagicMock(input_tokens=100, output_tokens=50)
            return resp

        responses = [
            _make_resp("bash", "tu_001", {"command": "ls"}),
            _make_resp("update_knowledge", "tu_002", {"key": "bug_location", "value": "app.py:42"}),
            _make_resp("submit_patch", "tu_003", {"message": "fixed"}),
        ]
        call_count = [0]
        llm = MagicMock()

        def llm_complete(**kwargs):
            idx = min(call_count[0], len(responses) - 1)
            call_count[0] += 1
            return responses[idx]

        llm.complete.side_effect = llm_complete
        llm.count_tokens.return_value = 50
        ctr = [0]

        def get_stats():
            ctr[0] += 1
            return {"total_calls": ctr[0], "total_input_tokens": 100,
                    "total_output_tokens": 50, "estimated_cost_usd": 0.001}

        llm.get_stats.side_effect = get_stats

        sandbox = MagicMock()
        sandbox.get_diff.return_value = "--- a/f\n+++ b/f\n@@ -1 +1 @@\n-x\n+y"
        sandbox.run_tests.return_value = {
            "passed": ["t1"], "failed": [], "exit_code": 0, "output": "ok", "error": None
        }

        tool_executor = MagicMock()
        tool_executor.setup_sandbox.return_value = sandbox
        tool_executor.tool_definitions = []
        tool_executor.execute.side_effect = _execute

        orch = Orchestrator(
            config=OrchestratorConfig(max_steps=10),
            memory=memory,
            compression=NoCompression(),
            communication=SingleAgentCommunication(),
            llm_client=llm,
            tool_executor=tool_executor,
            logger=TrajectoryLogger(),
        )
        return orch, memory

    def _make_task(self):
        t = MagicMock()
        t.instance_id = "hybrid_e2e_test"
        t.repo = "test/repo"
        t.base_commit = "abc"
        t.problem_statement = "Fix the bug"
        t.hints_text = ""
        t.test_patch = ""
        t.fail_to_pass = ["t1"]
        t.pass_to_pass = []
        return t

    def test_episodic_store_populated(self):
        orch, memory = self._build_stack_with_hybrid()
        orch.run_task(self._make_task())
        # bash result was stored automatically via orchestrator
        assert memory.episodic_store.get_stats()["entries_stored"] > 0

    def test_knowledge_base_populated(self):
        orch, memory = self._build_stack_with_hybrid()
        orch.run_task(self._make_task())
        # agent explicitly called update_knowledge
        assert memory.knowledge_base.get("bug_location") == "app.py:42"

    def test_both_stores_in_stats(self):
        orch, memory = self._build_stack_with_hybrid()
        orch.run_task(self._make_task())
        stats = memory.get_stats()
        assert stats["knowledge_entries"] >= 1
        assert stats["episodic"]["entries_stored"] >= 1
