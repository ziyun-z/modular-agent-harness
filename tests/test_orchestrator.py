"""
Tests for Orchestrator and TrajectoryLogger.

All tests use mocked dependencies — no Docker, no API calls needed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from src.orchestrator import Orchestrator, OrchestratorConfig, TaskResult
from src.logger import TrajectoryLogger


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_task(
    instance_id="django__django-001",
    repo="django/django",
    base_commit="abc123",
    problem_statement="Fix off-by-one in queryset",
    fail_to_pass=None,
    pass_to_pass=None,
    test_patch="",
    hints_text="",
):
    task = MagicMock()
    task.instance_id = instance_id
    task.repo = repo
    task.base_commit = base_commit
    task.problem_statement = problem_statement
    task.hints_text = hints_text
    task.test_patch = test_patch
    task.fail_to_pass = fail_to_pass or ["tests/test_queryset.py::TestOff"]
    task.pass_to_pass = pass_to_pass or []
    return task


def _make_step_result(done=False, actions=None, result=None):
    return {
        "actions_taken": actions or [],
        "done": done,
        "result": result,
        "llm_calls": 1,
    }


def _make_orchestrator(
    max_steps=10,
    step_results=None,
    tests_pass=True,
    compression_triggers=False,
):
    """
    Build an Orchestrator with all dependencies mocked.

    step_results: list of dicts to return from communication.run_step(),
                  cycled if the loop runs longer.
    """
    cfg = OrchestratorConfig(max_steps=max_steps)

    # Sandbox
    sandbox = MagicMock()
    sandbox.get_diff.return_value = "--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-old\n+new"
    sandbox.run_tests.return_value = {
        "passed": ["test_one"] if tests_pass else [],
        "failed": [] if tests_pass else ["test_one"],
        "exit_code": 0,
        "output": "ok",
        "error": None,
    }

    # Tool executor
    tool_executor = MagicMock()
    tool_executor.setup_sandbox.return_value = sandbox

    # LLM client
    llm = MagicMock()
    _stats = {"total_input_tokens": 0, "total_output_tokens": 0, "estimated_cost_usd": 0.0}
    call_count = [0]

    def get_stats_side_effect():
        call_count[0] += 10
        return {
            "total_input_tokens": call_count[0] * 100,
            "total_output_tokens": call_count[0] * 50,
            "estimated_cost_usd": call_count[0] * 0.001,
        }

    llm.get_stats.side_effect = get_stats_side_effect

    # Communication module
    if step_results is None:
        step_results = [_make_step_result(done=True, result="patch submitted")]

    comm = MagicMock()
    comm.get_trajectory.return_value = []
    step_iter = iter(step_results + [_make_step_result(done=True)] * 100)
    comm.run_step.side_effect = lambda **kw: next(step_iter)

    # Compression
    compression = MagicMock()
    compression.should_compress.return_value = compression_triggers
    compression.compress.return_value = []
    compression.get_stats.return_value = {}

    # Memory
    memory = MagicMock()
    memory.get_stats.return_value = {}

    # Logger
    traj_logger = TrajectoryLogger()

    orch = Orchestrator(
        config=cfg,
        memory=memory,
        compression=compression,
        communication=comm,
        llm_client=llm,
        tool_executor=tool_executor,
        logger=traj_logger,
    )
    return orch, sandbox, tool_executor, llm, comm, compression, memory, traj_logger


# ===========================================================================
# Orchestrator — lifecycle
# ===========================================================================


class TestOrchestratorLifecycle:
    def test_setup_sandbox_called(self):
        orch, _, tool_executor, _, _, _, _, _ = _make_orchestrator()
        task = _make_task()
        orch.run_task(task)
        tool_executor.setup_sandbox.assert_called_once_with(task)

    def test_sandbox_teardown_always_called(self):
        orch, _, tool_executor, _, _, _, _, _ = _make_orchestrator()
        task = _make_task()
        orch.run_task(task)
        tool_executor.teardown_sandbox.assert_called_once()

    def test_sandbox_teardown_called_even_on_crash(self):
        orch, _, tool_executor, _, comm, _, _, _ = _make_orchestrator()
        comm.run_step.side_effect = RuntimeError("boom")
        task = _make_task()
        orch.run_task(task)   # should not raise
        tool_executor.teardown_sandbox.assert_called_once()

    def test_communication_setup_called(self):
        orch, _, _, llm, comm, _, _, _ = _make_orchestrator()
        task = _make_task()
        orch.run_task(task)
        comm.setup.assert_called_once()
        call_kwargs = comm.setup.call_args[1]
        assert "task_description" in call_kwargs

    def test_sandbox_injected_into_communication(self):
        orch, sandbox, _, _, comm, _, _, _ = _make_orchestrator()
        orch.run_task(_make_task())
        comm.set_sandbox.assert_called_once_with(sandbox)

    def test_memory_injected_into_communication(self):
        orch, _, _, _, comm, _, memory, _ = _make_orchestrator()
        orch.run_task(_make_task())
        comm.set_memory.assert_called_once_with(memory)

    def test_memory_cleared_at_task_start(self):
        orch, _, _, _, _, _, memory, _ = _make_orchestrator()
        orch.run_task(_make_task())
        memory.clear.assert_called_once()


# ===========================================================================
# Orchestrator — main loop
# ===========================================================================


class TestOrchestratorLoop:
    def test_stops_when_done(self):
        steps = [
            _make_step_result(done=False),
            _make_step_result(done=False),
            _make_step_result(done=True),
        ]
        orch, _, _, _, comm, _, _, _ = _make_orchestrator(step_results=steps)
        result = orch.run_task(_make_task())
        assert result.steps == 3

    def test_stops_at_max_steps(self):
        orch, _, _, _, comm, _, _, _ = _make_orchestrator(
            max_steps=3,
            step_results=[_make_step_result(done=False)] * 10,
        )
        result = orch.run_task(_make_task())
        assert result.steps == 3

    def test_actions_stored_in_memory(self):
        action = {"tool": "bash", "input": {"command": "ls"}, "output": "foo.py"}
        steps = [_make_step_result(done=True, actions=[action])]
        orch, _, _, _, _, _, memory, _ = _make_orchestrator(step_results=steps)
        orch.run_task(_make_task())
        memory.store.assert_called_once()
        entry = memory.store.call_args[0][0]
        assert entry.entry_type == "bash"
        assert "foo.py" in entry.content

    def test_compression_triggered_when_needed(self):
        orch, _, _, _, comm, compression, _, _ = _make_orchestrator(
            compression_triggers=True,
            step_results=[_make_step_result(done=True)],
        )
        orch.run_task(_make_task())
        compression.compress.assert_called_once()

    def test_compression_not_triggered_when_not_needed(self):
        orch, _, _, _, _, compression, _, _ = _make_orchestrator(compression_triggers=False)
        orch.run_task(_make_task())
        compression.compress.assert_not_called()

    def test_compression_events_counted(self):
        orch, _, _, _, _, compression, _, _ = _make_orchestrator(
            compression_triggers=True,
            step_results=[
                _make_step_result(done=False),
                _make_step_result(done=True),
            ],
        )
        result = orch.run_task(_make_task())
        assert result.compression_events == 2   # one per step


# ===========================================================================
# Orchestrator — TaskResult
# ===========================================================================


class TestTaskResult:
    def test_passed_true_when_tests_pass(self):
        orch, _, _, _, _, _, _, _ = _make_orchestrator(tests_pass=True)
        result = orch.run_task(_make_task())
        assert result.passed is True

    def test_passed_false_when_tests_fail(self):
        orch, _, _, _, _, _, _, _ = _make_orchestrator(tests_pass=False)
        result = orch.run_task(_make_task())
        assert result.passed is False

    def test_patch_is_diff_string(self):
        orch, _, _, _, _, _, _, _ = _make_orchestrator()
        result = orch.run_task(_make_task())
        assert "--- a/foo.py" in result.patch

    def test_error_is_none_on_success(self):
        orch, _, _, _, _, _, _, _ = _make_orchestrator()
        result = orch.run_task(_make_task())
        assert result.error is None

    def test_error_captured_on_crash(self):
        orch, _, _, _, comm, _, _, _ = _make_orchestrator()
        comm.run_step.side_effect = RuntimeError("exploded")
        result = orch.run_task(_make_task())
        assert result.error is not None
        assert "RuntimeError" in result.error

    def test_trajectory_matches_logger(self):
        steps = [
            _make_step_result(done=False),
            _make_step_result(done=True),
        ]
        orch, _, _, _, _, _, _, traj_logger = _make_orchestrator(step_results=steps)
        result = orch.run_task(_make_task())
        assert result.trajectory == traj_logger.get_full_trajectory()
        assert len(result.trajectory) == 2

    def test_metrics_dict_has_expected_keys(self):
        orch, _, _, _, _, _, _, _ = _make_orchestrator()
        result = orch.run_task(_make_task())
        assert "memory" in result.metrics
        assert "compression" in result.metrics
        assert "communication" in result.metrics
        assert "llm" in result.metrics

    def test_task_id_set_correctly(self):
        orch, _, _, _, _, _, _, _ = _make_orchestrator()
        result = orch.run_task(_make_task(instance_id="flask__flask-42"))
        assert result.task_id == "flask__flask-42"

    def test_wall_time_is_non_negative(self):
        orch, _, _, _, _, _, _, _ = _make_orchestrator()
        result = orch.run_task(_make_task())
        assert result.wall_time_seconds >= 0


# ===========================================================================
# TrajectoryLogger
# ===========================================================================


class TestTrajectoryLogger:
    @pytest.fixture()
    def tlog(self):
        log = TrajectoryLogger()
        log.start_task("test-task-001")
        return log

    def test_start_task_resets_steps(self, tlog):
        tlog.log_step(0, {"done": False, "actions_taken": [], "llm_calls": 1})
        tlog.start_task("new-task")
        assert tlog.get_full_trajectory() == []

    def test_log_step_appends(self, tlog):
        tlog.log_step(0, {"done": False, "actions_taken": [], "llm_calls": 1})
        tlog.log_step(1, {"done": True, "actions_taken": [], "llm_calls": 1})
        assert len(tlog.get_full_trajectory()) == 2

    def test_step_record_has_required_fields(self, tlog):
        tlog.log_step(3, {"done": False, "actions_taken": [], "llm_calls": 2,
                          "input_tokens": 100, "output_tokens": 50, "cost_usd": 0.001})
        record = tlog.get_full_trajectory()[0]
        assert record["step"] == 3
        assert "timestamp" in record
        assert "elapsed_s" in record
        assert record["input_tokens"] == 100
        assert record["output_tokens"] == 50
        assert record["cost_usd"] == 0.001

    def test_missing_token_fields_default_to_zero(self, tlog):
        tlog.log_step(0, {"done": False, "actions_taken": [], "llm_calls": 1})
        record = tlog.get_full_trajectory()[0]
        assert record["input_tokens"] == 0
        assert record["output_tokens"] == 0
        assert record["cost_usd"] == 0.0

    def test_get_full_trajectory_returns_copy(self, tlog):
        tlog.log_step(0, {"done": False, "actions_taken": [], "llm_calls": 1})
        t1 = tlog.get_full_trajectory()
        t2 = tlog.get_full_trajectory()
        assert t1 == t2
        assert t1 is not t2   # separate list objects

    def test_save_writes_valid_json(self, tlog, tmp_path):
        tlog.log_step(0, {"done": True, "actions_taken": [
            {"tool": "bash", "input": {"command": "ls"}, "output": "ok"}
        ], "llm_calls": 1})
        out = tmp_path / "traj.json"
        tlog.save(str(out))
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["task_id"] == "test-task-001"
        assert len(data["trajectory"]) == 1

    def test_save_creates_parent_dirs(self, tlog, tmp_path):
        out = tmp_path / "nested" / "dir" / "traj.json"
        tlog.save(str(out))
        assert out.exists()

    def test_save_handles_non_serialisable_values(self, tlog, tmp_path):
        """Non-JSON-serialisable values (e.g. objects) should be stringified."""
        class Obj:
            def __repr__(self): return "<Obj>"

        tlog.log_step(0, {
            "done": False,
            "actions_taken": [{"tool": "bash", "input": Obj(), "output": "ok"}],
            "llm_calls": 1,
        })
        out = tmp_path / "traj.json"
        tlog.save(str(out))   # should not raise
        data = json.loads(out.read_text())
        assert data is not None

    def test_format_task_prompt_includes_problem(self):
        from src.orchestrator import Orchestrator, OrchestratorConfig
        orch = Orchestrator(
            config=OrchestratorConfig(),
            memory=MagicMock(),
            compression=MagicMock(),
            communication=MagicMock(),
            llm_client=MagicMock(),
            tool_executor=MagicMock(),
            logger=MagicMock(),
        )
        task = _make_task(problem_statement="Widget is broken")
        prompt = orch._format_task_prompt(task)
        assert "Widget is broken" in prompt
        assert "django/django" in prompt
