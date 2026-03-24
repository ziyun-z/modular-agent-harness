"""
Tests for memory modules.

Covers:
- NaiveMemory  (no-op baseline)
- ScratchpadMemory (agent-managed persistent notes)

No Docker or API calls needed.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from src.memory.base import MemoryEntry, MemoryModule
from src.memory.naive import NaiveMemory
from src.memory.scratchpad import ScratchpadMemory, TOOL_DEFINITION, _truncate_to_tokens


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
