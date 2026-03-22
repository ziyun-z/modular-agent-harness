"""
Trajectory logger for the agent harness.

Records every agent step with timestamps, actions, and LLM cost deltas.
Saves the full trajectory to JSON at the end of a task run.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class TrajectoryLogger:
    """
    In-memory trajectory recorder.

    Usage::

        logger = TrajectoryLogger()
        logger.start_task("django__django-1234")

        # Inside the orchestrator loop:
        logger.log_step(step_num, step_result)   # step_result may include token fields

        # After the task:
        trajectory = logger.get_full_trajectory()
        logger.save("experiments/results/django__django-1234.json")
    """

    def __init__(self) -> None:
        self._task_id: str | None = None
        self._start_time: float = time.time()
        self._steps: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_task(self, task_id: str) -> None:
        """Reset state and record the task ID and wall-clock start time."""
        self._task_id = task_id
        self._start_time = time.time()
        self._steps = []

    def log_step(self, step_num: int, step_result: dict[str, Any]) -> None:
        """
        Append a step record.

        step_result is the dict returned by communication.run_step(), optionally
        enriched by the orchestrator with:
            - input_tokens   (int)
            - output_tokens  (int)
            - cost_usd       (float)

        The logger adds:
            - step           step number
            - timestamp      unix time of logging
            - elapsed_s      seconds since task start
        """
        record: dict[str, Any] = {
            "step": step_num,
            "timestamp": time.time(),
            "elapsed_s": round(time.time() - self._start_time, 3),
            # Core step data (from communication.run_step)
            "actions_taken": _serialise(step_result.get("actions_taken", [])),
            "done": step_result.get("done", False),
            "result": step_result.get("result"),
            "llm_calls": step_result.get("llm_calls", 0),
            # LLM cost data (enriched by orchestrator)
            "input_tokens": step_result.get("input_tokens", 0),
            "output_tokens": step_result.get("output_tokens", 0),
            "cost_usd": step_result.get("cost_usd", 0.0),
        }
        self._steps.append(record)

    def get_full_trajectory(self) -> list[dict[str, Any]]:
        """Return the complete list of step records."""
        return list(self._steps)

    def save(self, path: str) -> None:
        """
        Write the full trajectory to a JSON file.

        The file contains task_id, wall_time_seconds, and the step list.
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "task_id": self._task_id,
            "wall_time_seconds": round(time.time() - self._start_time, 3),
            "total_steps": len(self._steps),
            "trajectory": self._steps,
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _serialise(obj: Any) -> Any:
    """
    Recursively make an object JSON-safe.
    Converts non-serialisable values (e.g. Anthropic block objects) to strings.
    """
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialise(v) for v in obj]
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)
