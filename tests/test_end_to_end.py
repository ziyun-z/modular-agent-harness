"""
End-to-end integration tests for the full agent pipeline.

These tests exercise the complete stack — Runner → Orchestrator → Communication
→ Memory → Compression — using mocked LLM and sandbox so no Docker or API key
is needed.

One test is marked @pytest.mark.integration and requires real Docker + API key.
The rest are fast unit-level end-to-end tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from src.orchestrator import Orchestrator, OrchestratorConfig, TaskResult
from src.logger import TrajectoryLogger
from src.memory.naive import NaiveMemory
from src.compression.none import NoCompression
from src.communication.single_agent import SingleAgentCommunication
from src.tool_executor import PatchSubmitted


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_task(
    instance_id="flask__flask-1234",
    repo="pallets/flask",
    base_commit="deadbeef",
    problem_statement="NullPointerException in request handler",
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
    task.fail_to_pass = fail_to_pass or ["tests/test_app.py::TestApp::test_handler"]
    task.pass_to_pass = pass_to_pass or []
    return task


def _make_llm_response(text="I'll fix this.", tool_calls=None, stop_reason="end_turn"):
    """Build a mock Anthropic Message object."""
    response = MagicMock()
    response.stop_reason = stop_reason
    blocks = []

    if text:
        tb = MagicMock()
        tb.type = "text"
        tb.text = text
        blocks.append(tb)

    for tc in (tool_calls or []):
        ub = MagicMock()
        ub.type = "tool_use"
        ub.id = tc["id"]
        ub.name = tc["name"]
        ub.input = tc["input"]
        blocks.append(ub)

    response.content = blocks
    response.usage = MagicMock()
    response.usage.input_tokens = 500
    response.usage.output_tokens = 150
    return response


def _build_full_stack(
    step_responses=None,
    tests_pass=True,
    max_steps=10,
):
    """
    Build a fully wired Orchestrator with all real module instances
    but mocked LLM and sandbox.

    step_responses: list of mock LLM response objects, one per step.
                    Defaults to [text-only, submit_patch].
    """
    if step_responses is None:
        # Default: one exploration step, then submit_patch
        step_responses = [
            _make_llm_response(
                text="Let me look at the code.",
                tool_calls=[{
                    "id": "tu_001",
                    "name": "bash",
                    "input": {"command": "ls src/"},
                }],
                stop_reason="tool_use",
            ),
            _make_llm_response(
                text="I found the bug, fixing now.",
                tool_calls=[{
                    "id": "tu_002",
                    "name": "submit_patch",
                    "input": {"message": "Fixed NullPointerException"},
                }],
                stop_reason="tool_use",
            ),
        ]

    # LLM mock
    llm = MagicMock()
    call_count = [0]

    def llm_complete(**kwargs):
        idx = min(call_count[0], len(step_responses) - 1)
        call_count[0] += 1
        return step_responses[idx]

    llm.complete.side_effect = llm_complete
    llm.count_tokens.return_value = 100

    stats_counter = [0]
    def get_stats():
        stats_counter[0] += 5
        return {
            "total_calls": stats_counter[0],
            "total_input_tokens": stats_counter[0] * 500,
            "total_output_tokens": stats_counter[0] * 150,
            "estimated_cost_usd": stats_counter[0] * 0.002,
        }
    llm.get_stats.side_effect = get_stats

    # Sandbox mock
    sandbox = MagicMock()
    sandbox.get_diff.return_value = "--- a/src/app.py\n+++ b/src/app.py\n@@ -1 +1 @@\n-bug\n+fix"
    sandbox.run_tests.return_value = {
        "passed": ["tests/test_app.py::TestApp::test_handler"] if tests_pass else [],
        "failed": [] if tests_pass else ["tests/test_app.py::TestApp::test_handler"],
        "exit_code": 0 if tests_pass else 1,
        "output": "ok",
        "error": None,
    }
    sandbox.exec.return_value = MagicMock(stdout="app.py\n", exit_code=0, ok=True)

    # Tool executor mock — raises PatchSubmitted for submit_patch calls
    tool_executor = MagicMock()
    tool_executor.setup_sandbox.return_value = sandbox
    tool_executor.tool_definitions = []

    def _execute(tool_name, tool_input, sandbox_ref):
        if tool_name == "submit_patch":
            raise PatchSubmitted(tool_input.get("message", ""))
        return "app.py\n"

    tool_executor.execute.side_effect = _execute

    # Real modules
    memory = NaiveMemory()
    compression = NoCompression()
    communication = SingleAgentCommunication()
    traj_logger = TrajectoryLogger()

    cfg = OrchestratorConfig(max_steps=max_steps)
    orchestrator = Orchestrator(
        config=cfg,
        memory=memory,
        compression=compression,
        communication=communication,
        llm_client=llm,
        tool_executor=tool_executor,
        logger=traj_logger,
    )
    return orchestrator, traj_logger, sandbox, tool_executor, llm


# ===========================================================================
# Full pipeline — happy path
# ===========================================================================


class TestEndToEndHappyPath:
    def test_task_completes_and_passes(self):
        orch, _, _, _, _ = _build_full_stack(tests_pass=True)
        result = orch.run_task(_make_task())
        assert result.passed is True

    def test_result_has_patch(self):
        orch, _, _, _, _ = _build_full_stack()
        result = orch.run_task(_make_task())
        assert "fix" in result.patch

    def test_result_has_task_id(self):
        orch, _, _, _, _ = _build_full_stack()
        result = orch.run_task(_make_task(instance_id="flask__flask-9999"))
        assert result.task_id == "flask__flask-9999"

    def test_result_has_no_error_on_success(self):
        orch, _, _, _, _ = _build_full_stack()
        result = orch.run_task(_make_task())
        assert result.error is None

    def test_steps_counted(self):
        orch, _, _, _, _ = _build_full_stack()
        result = orch.run_task(_make_task())
        # 2 LLM steps: bash + submit_patch
        assert result.steps == 2

    def test_trajectory_recorded(self):
        orch, traj_logger, _, _, _ = _build_full_stack()
        result = orch.run_task(_make_task())
        assert len(result.trajectory) == 2
        assert result.trajectory == traj_logger.get_full_trajectory()

    def test_metrics_populated(self):
        orch, _, _, _, _ = _build_full_stack()
        result = orch.run_task(_make_task())
        assert "memory" in result.metrics
        assert "compression" in result.metrics
        assert "llm" in result.metrics

    def test_sandbox_setup_called(self):
        orch, _, _, tool_executor, _ = _build_full_stack()
        task = _make_task()
        orch.run_task(task)
        tool_executor.setup_sandbox.assert_called_once_with(task)

    def test_sandbox_teardown_always_called(self):
        orch, _, _, tool_executor, _ = _build_full_stack()
        orch.run_task(_make_task())
        tool_executor.teardown_sandbox.assert_called_once()


# ===========================================================================
# Full pipeline — failure cases
# ===========================================================================


class TestEndToEndFailureCases:
    def test_tests_fail_marks_result_failed(self):
        orch, _, _, _, _ = _build_full_stack(tests_pass=False)
        result = orch.run_task(_make_task())
        assert result.passed is False

    def test_error_captured_when_llm_crashes(self):
        orch, _, _, _, llm = _build_full_stack()
        llm.complete.side_effect = RuntimeError("API down")
        result = orch.run_task(_make_task())
        assert result.error is not None
        assert "RuntimeError" in result.error

    def test_sandbox_teardown_even_on_llm_crash(self):
        orch, _, _, tool_executor, llm = _build_full_stack()
        llm.complete.side_effect = RuntimeError("boom")
        orch.run_task(_make_task())
        tool_executor.teardown_sandbox.assert_called_once()

    def test_max_steps_respected(self):
        # Provide responses that never submit_patch
        never_done = [
            _make_llm_response(text="Still thinking...", stop_reason="end_turn")
        ] * 20
        orch, _, _, _, _ = _build_full_stack(
            step_responses=never_done, max_steps=5
        )
        result = orch.run_task(_make_task())
        assert result.steps == 5


# ===========================================================================
# Module wiring — real NaiveMemory + NoCompression
# ===========================================================================


class TestModuleWiring:
    def test_naive_memory_stats_in_metrics(self):
        orch, _, _, _, _ = _build_full_stack()
        result = orch.run_task(_make_task())
        assert result.metrics["memory"]["type"] == "naive"

    def test_no_compression_stats_in_metrics(self):
        orch, _, _, _, _ = _build_full_stack()
        result = orch.run_task(_make_task())
        assert result.metrics["compression"]["type"] == "none"

    def test_no_compression_events_triggered(self):
        orch, _, _, _, _ = _build_full_stack()
        result = orch.run_task(_make_task())
        assert result.compression_events == 0


# ===========================================================================
# Trajectory saving
# ===========================================================================


class TestTrajectorySaving:
    def test_trajectory_saves_to_json(self, tmp_path):
        orch, traj_logger, _, _, _ = _build_full_stack()
        result = orch.run_task(_make_task(instance_id="save-test-001"))

        out = tmp_path / "save-test-001.json"
        traj_logger.save(str(out))

        assert out.exists()
        data = json.loads(out.read_text())
        assert data["task_id"] == "save-test-001"
        assert data["total_steps"] == result.steps
        assert len(data["trajectory"]) == result.steps

    def test_trajectory_json_has_cost_fields(self, tmp_path):
        orch, traj_logger, _, _, _ = _build_full_stack()
        orch.run_task(_make_task())

        out = tmp_path / "traj.json"
        traj_logger.save(str(out))
        data = json.loads(out.read_text())

        for step in data["trajectory"]:
            assert "input_tokens" in step
            assert "output_tokens" in step
            assert "cost_usd" in step

    def test_trajectory_json_has_actions(self, tmp_path):
        orch, traj_logger, _, _, _ = _build_full_stack()
        orch.run_task(_make_task())

        out = tmp_path / "traj.json"
        traj_logger.save(str(out))
        data = json.loads(out.read_text())

        # First step has a bash action
        step_0 = data["trajectory"][0]
        assert len(step_0["actions_taken"]) >= 1
        assert step_0["actions_taken"][0]["tool"] == "bash"


# ===========================================================================
# Config → full run integration (no Docker/API)
# ===========================================================================


class TestConfigToRun:
    def test_baseline_config_builds_correct_modules(self):
        """Verify the baseline.yaml config instantiates the right module types."""
        from src.runner import load_config, build_memory_module, build_compression_module, build_communication_module
        from src.memory.naive import NaiveMemory
        from src.compression.none import NoCompression
        from src.communication.single_agent import SingleAgentCommunication

        baseline = Path(__file__).parent.parent / "configs" / "baseline.yaml"
        cfg = load_config(str(baseline))

        assert isinstance(build_memory_module(cfg), NaiveMemory)
        assert isinstance(build_compression_module(cfg), NoCompression)
        assert isinstance(build_communication_module(cfg), SingleAgentCommunication)

    def test_baseline_config_validates_cleanly(self):
        from src.runner import load_config, validate_config
        baseline = Path(__file__).parent.parent / "configs" / "baseline.yaml"
        cfg = load_config(str(baseline))
        assert validate_config(cfg) == []
