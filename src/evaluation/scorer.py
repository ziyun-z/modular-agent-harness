"""
Scorer: apply the agent's patch to a fresh sandbox and run the evaluation tests.

The scorer is called by the orchestrator after the agent calls submit_patch().
It creates a clean sandbox (no test patch, no agent changes), applies the agent's
diff, then runs FAIL_TO_PASS and PASS_TO_PASS tests to determine pass/fail.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.evaluation.swebench_loader import SWEBenchTask
from src.sandbox import DockerSandbox

logger = logging.getLogger(__name__)


@dataclass
class ScoringResult:
    passed: bool                        # True iff all FAIL_TO_PASS pass and PASS_TO_PASS don't regress
    fail_to_pass_results: dict          # test results for the required-to-fix tests
    pass_to_pass_results: dict          # test results for the must-not-regress tests
    patch_applied: bool                 # whether the patch applied cleanly
    patch_error: str = ""              # error message if patch failed to apply
    error: str = ""                    # unexpected error during scoring


class Scorer:
    """Scores an agent patch against the SWE-bench test suite."""

    def __init__(self, sandbox_config: dict | None = None):
        self._sandbox_config = sandbox_config or {}

    def score(self, task: SWEBenchTask, patch: str) -> ScoringResult:
        """
        Apply patch to a fresh sandbox and run the evaluation tests.

        Args:
            task: The SWE-bench task being evaluated.
            patch: The agent's unified diff patch.

        Returns:
            ScoringResult with pass/fail and per-test details.
        """
        docker_image = self._sandbox_config.get("docker_image", "swebench-sandbox:latest")
        timeout = self._sandbox_config.get("timeout_per_task", 600)

        sandbox = DockerSandbox(docker_image=docker_image, timeout_per_task=timeout)
        try:
            return self._score_in_sandbox(sandbox, task, patch)
        except Exception as e:
            logger.exception(f"Unexpected error scoring {task.instance_id}")
            return ScoringResult(
                passed=False,
                fail_to_pass_results={},
                pass_to_pass_results={},
                patch_applied=False,
                error=str(e),
            )
        finally:
            sandbox.teardown()

    def score_gold(self, task: SWEBenchTask) -> ScoringResult:
        """
        Score the gold patch. Used for validation: confirms the sandbox and
        test suite work correctly before running experiments.
        """
        return self.score(task, task.patch)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _score_in_sandbox(
        self, sandbox: DockerSandbox, task: SWEBenchTask, patch: str
    ) -> ScoringResult:
        # 1. Set up a fresh sandbox (no test patch yet — we apply it below)
        #    We need a task-like object without the test_patch so setup()
        #    doesn't apply it before we apply the agent's patch.
        class _TaskNoTestPatch:
            instance_id = task.instance_id
            repo = task.repo
            base_commit = task.base_commit
            test_patch = ""  # don't apply yet

        sandbox.setup(_TaskNoTestPatch())

        # 2. Apply the agent's patch
        applied, apply_msg = sandbox.apply_patch(patch)
        if not applied:
            logger.warning(f"Patch apply failed for {task.instance_id}: {apply_msg}")
            return ScoringResult(
                passed=False,
                fail_to_pass_results={},
                pass_to_pass_results={},
                patch_applied=False,
                patch_error=apply_msg,
            )

        # 3. Apply the test patch (evaluation tests)
        if task.test_patch:
            test_applied, test_msg = sandbox.apply_patch(task.test_patch)
            if not test_applied:
                logger.warning(f"Test patch apply failed: {test_msg}")
                return ScoringResult(
                    passed=False,
                    fail_to_pass_results={},
                    pass_to_pass_results={},
                    patch_applied=True,
                    error=f"Test patch failed to apply: {test_msg}",
                )

        # 4. Run FAIL_TO_PASS tests (these must pass for the task to be solved)
        test_timeout = min(self._sandbox_config.get("timeout_per_task", 600) // 2, 300)
        f2p_results = sandbox.run_tests(task.fail_to_pass, timeout=test_timeout)
        logger.debug(
            f"{task.instance_id} FAIL_TO_PASS: "
            f"{len(f2p_results['passed'])} passed, {len(f2p_results['failed'])} failed"
        )

        # 5. Run PASS_TO_PASS tests (these must not regress)
        p2p_results = {}
        if task.pass_to_pass:
            p2p_results = sandbox.run_tests(task.pass_to_pass, timeout=test_timeout)
            logger.debug(
                f"{task.instance_id} PASS_TO_PASS: "
                f"{len(p2p_results['passed'])} passed, {len(p2p_results['failed'])} failed"
            )

        # 6. Determine overall pass/fail
        f2p_all_passed = (
            f2p_results["exit_code"] == 0
            and len(f2p_results["failed"]) == 0
            and len(f2p_results["passed"]) > 0
        )
        p2p_no_regression = not p2p_results or len(p2p_results.get("failed", [])) == 0

        passed = f2p_all_passed and p2p_no_regression

        return ScoringResult(
            passed=passed,
            fail_to_pass_results=f2p_results,
            pass_to_pass_results=p2p_results,
            patch_applied=True,
        )
