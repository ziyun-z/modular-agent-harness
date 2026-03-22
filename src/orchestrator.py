"""
Orchestrator: task-level controller for the agent harness.

Wires together memory, compression, communication, LLM, tools, and logger.
Does NOT contain agent reasoning — that lives in the communication module.

Flow for a single task:
  1. Start sandbox
  2. Inject sandbox + memory into communication module
  3. Main loop (up to max_steps):
       a. Compress if context budget exceeded
       b. Run one agent step
       c. Store observations in memory
       d. Log step
       e. Break if done
  4. Extract diff, run tests, return TaskResult
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from src.memory.base import MemoryEntry
from src.compression.base import ConversationTurn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class OrchestratorConfig:
    max_steps: int = 50
    max_tokens_context: int = 100_000      # context window budget
    memory_token_budget: int = 20_000      # tokens reserved for memory block
    compression_target: float = 0.6        # compress to this fraction of max
    model: str = "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class TaskResult:
    task_id: str
    passed: bool
    patch: str
    steps: int
    wall_time_seconds: float
    llm_calls: int
    total_input_tokens: int
    total_output_tokens: int
    estimated_cost_usd: float
    compression_events: int
    trajectory: list[dict]
    metrics: dict
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class Orchestrator:
    """
    Runs a single SWE-bench task end-to-end.

    All agent coordination lives in self.communication (CommunicationModule).
    This class only manages the outer loop and resource lifecycle.
    """

    def __init__(
        self,
        config: OrchestratorConfig,
        memory,
        compression,
        communication,
        llm_client,
        tool_executor,
        logger,
    ):
        self.config = config
        self.memory = memory
        self.compression = compression
        self.communication = communication
        self.llm = llm_client
        self.tool_executor = tool_executor
        self.logger = logger

    def run_task(self, task) -> TaskResult:
        """
        Run the agent on a single SWE-bench task.

        Returns a TaskResult with pass/fail, patch diff, full trajectory,
        and cost/token metrics.
        """
        start_time = time.time()
        error: Optional[str] = None
        compression_events = 0
        total_llm_calls = 0
        step = 0

        self.logger.start_task(task.instance_id)

        sandbox = None
        try:
            # 1. Start sandbox
            sandbox = self.tool_executor.setup_sandbox(task)
            logger.info("Sandbox ready for %s", task.instance_id)

            # 2. Initialise communication module
            self.communication.setup(
                task_description=self._format_task_prompt(task),
                llm_client=self.llm,
            )
            self.communication.set_sandbox(sandbox)
            self.communication.set_memory(self.memory)
            self.memory.clear()

            # 3. Main agent loop
            for step in range(self.config.max_steps):

                # 3a. Compress if context is getting large
                turns = self._get_turns()
                if self.compression.should_compress(turns, self.config.max_tokens_context):
                    target = int(self.config.max_tokens_context * self.config.compression_target)
                    compressed = self.compression.compress(
                        turns,
                        target_tokens=target,
                        llm_client=self.llm,
                    )
                    self.communication.update_trajectory(compressed)
                    compression_events += 1
                    logger.info("Step %d — compression triggered (%d→%d turns)",
                                step, len(turns), len(compressed))

                # 3b. Run one agent step, track LLM cost delta
                llm_before = self.llm.get_stats()
                step_result = self.communication.run_step(
                    llm_client=self.llm,
                    tool_executor=self.tool_executor,
                )
                llm_after = self.llm.get_stats()

                # Enrich step_result with per-step token/cost data
                step_result["input_tokens"] = (
                    llm_after["total_input_tokens"] - llm_before["total_input_tokens"]
                )
                step_result["output_tokens"] = (
                    llm_after["total_output_tokens"] - llm_before["total_output_tokens"]
                )
                step_result["cost_usd"] = round(
                    llm_after["estimated_cost_usd"] - llm_before["estimated_cost_usd"], 6
                )
                total_llm_calls += step_result.get("llm_calls", 0)

                # 3c. Store each tool action in memory
                for action in step_result.get("actions_taken", []):
                    self.memory.store(MemoryEntry(
                        step=step,
                        entry_type=action.get("tool", "unknown"),
                        content=str(action.get("output", "")),
                        metadata={"input": str(action.get("input", {}))},
                        timestamp=time.time(),
                    ))

                # 3d. Log step
                self.logger.log_step(step, step_result)

                # 3e. Check completion
                if step_result.get("done"):
                    logger.info("Task %s done at step %d", task.instance_id, step)
                    break

        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            logger.error("Task %s crashed at step %d: %s", task.instance_id, step, error)
        finally:
            self.tool_executor.teardown_sandbox()

        # 4. Extract patch and score
        patch = ""
        passed = False
        if sandbox is not None and error is None:
            try:
                patch = sandbox.get_diff()
                passed = self._run_tests(sandbox, task)
            except Exception as exc:
                error = error or f"Scoring failed: {exc}"
                logger.error("Scoring error for %s: %s", task.instance_id, exc)

        llm_stats = self.llm.get_stats()
        return TaskResult(
            task_id=task.instance_id,
            passed=passed,
            patch=patch,
            steps=step + 1,
            wall_time_seconds=round(time.time() - start_time, 2),
            llm_calls=total_llm_calls,
            total_input_tokens=llm_stats["total_input_tokens"],
            total_output_tokens=llm_stats["total_output_tokens"],
            estimated_cost_usd=llm_stats["estimated_cost_usd"],
            compression_events=compression_events,
            trajectory=self.logger.get_full_trajectory(),
            metrics=self._collect_metrics(),
            error=error,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _format_task_prompt(self, task) -> str:
        """Format a SWEBenchTask into the agent's opening user message."""
        lines = [
            f"Repository: {task.repo}",
            f"Base commit: {task.base_commit}",
            "",
            "## Issue",
            task.problem_statement,
        ]
        if task.hints_text:
            lines += ["", "## Hints", task.hints_text]
        lines += [
            "",
            "## Your task",
            "Fix the issue described above. The repository is already cloned at /workspace/repo.",
            "When you are confident your fix is correct, call submit_patch.",
        ]
        return "\n".join(lines)

    def _run_tests(self, sandbox, task) -> bool:
        """
        Run the fail_to_pass tests. Returns True only if all tests pass.
        """
        if not task.fail_to_pass:
            logger.warning("No fail_to_pass tests defined for %s", task.instance_id)
            return False

        result = sandbox.run_tests(task.fail_to_pass, timeout=120)
        n_failed = len(result.get("failed", []))
        n_passed = len(result.get("passed", []))
        logger.info(
            "Tests for %s: %d passed, %d failed",
            task.instance_id, n_passed, n_failed,
        )
        return n_failed == 0 and n_passed > 0

    def _get_turns(self) -> list[ConversationTurn]:
        """Get current trajectory as ConversationTurn list for compression."""
        raw = self.communication.get_trajectory()
        # get_trajectory() returns ConversationTurn objects; cast for type checkers
        return list(raw)  # type: ignore[arg-type]

    def _collect_metrics(self) -> dict[str, Any]:
        """Aggregate metrics from all modules."""
        return {
            "memory": self.memory.get_stats(),
            "compression": self.compression.get_stats(),
            "communication": self.communication.get_stats(),
            "llm": self.llm.get_stats(),
        }
