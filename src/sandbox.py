"""
Docker-based sandbox for running SWE-bench tasks.

Each task gets a fresh container with the repo cloned at the correct commit.
The sandbox provides exec/read/write/diff operations used by the tool executor.
"""

from __future__ import annotations

import base64
import io
import logging
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import docker
from docker.models.containers import Container

logger = logging.getLogger(__name__)


@dataclass
class ExecResult:
    stdout: str
    exit_code: int

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


class SandboxError(Exception):
    pass


class DockerSandbox:
    """
    Docker-based sandbox. Lifecycle:
        sandbox = DockerSandbox()
        sandbox.setup(task)   # starts container, clones repo, applies test patch
        sandbox.exec(...)     # agent uses these via ToolExecutor
        sandbox.get_diff()    # called by orchestrator at the end
        sandbox.teardown()    # always call in a finally block
    """

    REPO_DIR = "/workspace/repo"

    def __init__(
        self,
        docker_image: str = "swebench-sandbox:latest",
        timeout_per_task: int = 600,
    ):
        self.docker_image = docker_image
        self.timeout_per_task = timeout_per_task
        self._client = docker.from_env()
        self._container: Optional[Container] = None
        self._task_instance_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self, task) -> None:
        """Start a container and prepare the repo for the given task."""
        self._task_instance_id = task.instance_id
        logger.info(f"Setting up sandbox for {task.instance_id}")

        self._start_container()
        self._bootstrap_tools()
        self._clone_repo(task.repo, task.base_commit)
        self._install_package()
        if task.test_patch:
            self._apply_patch_str(task.test_patch, label="test patch")

        logger.info(f"Sandbox ready: {task.instance_id}")

    def teardown(self) -> None:
        """Stop and remove the container."""
        if self._container:
            try:
                self._container.stop(timeout=5)
                self._container.remove(force=True)
                logger.info(f"Sandbox torn down: {self._task_instance_id}")
            except Exception as e:
                logger.warning(f"Error tearing down sandbox: {e}")
            finally:
                self._container = None

    # ------------------------------------------------------------------
    # Agent-facing operations (called by ToolExecutor)
    # ------------------------------------------------------------------

    def exec(self, command: str, timeout: int = 30) -> ExecResult:
        """Run a shell command inside the repo directory."""
        return self._run(f"cd {self.REPO_DIR} && {command}", timeout=timeout)

    def read_file(
        self,
        path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
    ) -> str:
        """Read file contents with optional line range. Returns numbered lines."""
        abs_path = self._abs(path)

        if start_line is not None and end_line is not None:
            result = self._run(f"sed -n '{start_line},{end_line}p' {abs_path}")
        elif start_line is not None:
            result = self._run(f"tail -n +{start_line} {abs_path}")
        else:
            result = self._run(f"cat {abs_path}")

        if result.exit_code != 0:
            raise FileNotFoundError(f"Cannot read '{path}': {result.stdout.strip()}")

        offset = (start_line or 1) - 1
        lines = result.stdout.splitlines()
        return "\n".join(f"{offset + i + 1}: {line}" for i, line in enumerate(lines))

    def write_file(self, path: str, content: str) -> None:
        """Write (overwrite) a file in the container."""
        abs_path = self._abs(path)
        dir_path = str(Path(abs_path).parent)
        self._run(f"mkdir -p {dir_path}")

        # Use base64 to safely transfer arbitrary content
        b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
        result = self._run(f"echo '{b64}' | base64 -d > {abs_path}")
        if result.exit_code != 0:
            raise SandboxError(f"write_file failed for '{path}': {result.stdout}")

    def edit_file(self, path: str, old_str: str, new_str: str) -> str:
        """Replace the first occurrence of old_str in a file."""
        abs_path = self._abs(path)
        result = self._run(f"cat {abs_path}")
        if result.exit_code != 0:
            raise FileNotFoundError(f"Cannot read '{path}' for editing")

        content = result.stdout
        if old_str not in content:
            raise ValueError(
                f"String not found in '{path}'.\n"
                f"Looking for:\n{old_str[:200]}"
            )

        new_content = content.replace(old_str, new_str, 1)
        self.write_file(path, new_content)
        return new_content

    def list_files(self, path: str, recursive: bool = False, max_depth: int = 2) -> str:
        """List files in a directory."""
        abs_path = self._abs(path)
        if recursive:
            result = self._run(f"find {abs_path} -maxdepth {max_depth} -type f | sort")
        else:
            result = self._run(f"ls -la {abs_path}")
        return result.stdout

    def search_code(self, pattern: str, file_glob: Optional[str] = None) -> str:
        """Search for a regex pattern using ripgrep."""
        glob_flag = f"--glob '{file_glob}'" if file_glob else ""
        result = self._run(
            f"cd {self.REPO_DIR} && rg --line-number {glob_flag} {pattern!r} 2>&1 | head -200"
        )
        return result.stdout

    # ------------------------------------------------------------------
    # Patch / diff operations
    # ------------------------------------------------------------------

    def get_diff(self) -> str:
        """Return the current git diff (agent's changes vs base commit)."""
        result = self._run(f"cd {self.REPO_DIR} && git diff")
        return result.stdout

    def apply_patch(self, patch: str) -> tuple[bool, str]:
        """Apply a unified diff patch. Returns (success, message)."""
        try:
            self._apply_patch_str(patch, label="agent patch")
            return True, "Patch applied successfully"
        except SandboxError as e:
            return False, str(e)

    def reset_to_base(self) -> None:
        """Discard all agent changes, restoring the base commit state."""
        self._run(f"cd {self.REPO_DIR} && git checkout -- .")
        self._run(f"cd {self.REPO_DIR} && git clean -fd")

    # ------------------------------------------------------------------
    # Test running
    # ------------------------------------------------------------------

    def run_tests(self, test_ids: list[str], timeout: int = 120) -> dict:
        """
        Run specific test IDs and return structured results.

        Detects whether the repo uses Django's custom test runner (runtests.py)
        or standard pytest, and invokes the correct runner automatically.

        Returns:
            {
                "passed":    list of passing test names,
                "failed":    list of failing test names,
                "error":     top-level error string or None,
                "exit_code": int,
                "output":    full stdout/stderr as a string,
            }
        """
        if not test_ids:
            return {"passed": [], "failed": [], "error": None, "exit_code": 0, "output": ""}

        if self._is_django_repo():
            result = self._run_django_tests(test_ids, timeout)
        else:
            result = self._run_pytest_tests(test_ids, timeout)

        passed, failed = self._parse_test_output(result.stdout)

        return {
            "passed": passed,
            "failed": failed,
            "error": None,
            "exit_code": result.exit_code,
            "output": result.stdout,
        }

    def _is_django_repo(self) -> bool:
        """Return True if this repo uses Django's runtests.py test runner."""
        check = self._run(
            f"test -f {self.REPO_DIR}/tests/runtests.py && echo yes || echo no"
        )
        return check.stdout.strip() == "yes"

    def _run_django_tests(self, test_ids: list[str], timeout: int) -> "ExecResult":
        """
        Run tests via Django's runtests.py.

        SWE-bench test IDs look like:
            tests/auth_tests/test_tokens.py::TokenGeneratorTest::test_make_token

        Django's runtests.py expects dotted module names like:
            auth_tests.test_tokens

        So we strip the leading "tests/", drop the filename extension,
        replace "/" with ".", and discard the "::ClassName::method" suffix.
        """
        modules: set[str] = set()
        for tid in test_ids:
            # Normalise: strip leading "tests/" if present
            tid = tid.replace("\\", "/")
            if tid.startswith("tests/"):
                tid = tid[len("tests/"):]
            # Drop class / method suffix (everything from "::" onward)
            tid = tid.split("::")[0]
            # Drop .py extension and convert path separators to dots
            module = tid.replace(".py", "").replace("/", ".")
            modules.add(module)

        spec = " ".join(sorted(modules))
        return self._run(
            f"cd {self.REPO_DIR} && python tests/runtests.py --verbosity=2 {spec} 2>&1",
            timeout=timeout,
        )

    def _run_pytest_tests(self, test_ids: list[str], timeout: int) -> "ExecResult":
        """Run tests via standard pytest."""
        test_spec = " ".join(f'"{t}"' for t in test_ids)
        return self._run(
            f"cd {self.REPO_DIR} && python -m pytest {test_spec} -v --tb=short --no-header 2>&1",
            timeout=timeout,
        )

    def _parse_test_output(self, output: str) -> tuple[list[str], list[str]]:
        """
        Parse test runner output and return (passed, failed) name lists.

        Handles both pytest and Django runtests.py output formats:
          pytest:  "tests/foo.py::Bar::test_baz PASSED"
          django:  "test_baz (auth_tests.test_tokens.TokenGeneratorTest) ... ok"
                   "test_baz (auth_tests.test_tokens.TokenGeneratorTest) ... FAIL"
        """
        passed, failed = [], []
        for line in output.splitlines():
            line_s = line.strip()
            # pytest format
            if " PASSED" in line_s:
                passed.append(line_s)
            elif " FAILED" in line_s or " ERROR" in line_s:
                failed.append(line_s)
            # Django unittest format
            elif line_s.endswith(" ... ok") or line_s.endswith(" ... OK"):
                passed.append(line_s)
            elif line_s.endswith(" ... FAIL") or line_s.endswith(" ... ERROR"):
                failed.append(line_s)
        return passed, failed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _start_container(self) -> None:
        try:
            self._client.ping()
        except Exception as e:
            raise SandboxError(
                "Cannot connect to Docker daemon. "
                "Make sure Docker Desktop (or dockerd) is running.\n"
                f"Original error: {e}"
            ) from e

        self._container = self._client.containers.run(
            self.docker_image,
            command="sleep infinity",
            detach=True,
            remove=False,
            mem_limit="4g",
            environment={"PYTHONDONTWRITEBYTECODE": "1", "PIP_NO_CACHE_DIR": "1"},
            working_dir="/workspace",
        )

    def _bootstrap_tools(self) -> None:
        """Ensure git, patch, rg are available in the container."""
        self._run(
            "apt-get update -qq && "
            "apt-get install -y --no-install-recommends git patch ripgrep 2>/dev/null || true",
            timeout=120,
        )
        # Ensure pip is available
        self._run("python -m ensurepip --upgrade 2>/dev/null || true", timeout=60)
        self._run("pip install --upgrade pip -q 2>/dev/null || true", timeout=60)

    def _clone_repo(self, repo: str, base_commit: str) -> None:
        repo_url = f"https://github.com/{repo}.git"
        result = self._run(
            f"git clone --quiet --filter=blob:none {repo_url} {self.REPO_DIR} 2>&1",
            timeout=300,
        )
        if not result.ok:
            raise SandboxError(f"git clone failed for {repo}:\n{result.stdout}")

        result = self._run(
            f"cd {self.REPO_DIR} && git checkout {base_commit} 2>&1"
        )
        if not result.ok:
            raise SandboxError(
                f"git checkout {base_commit} failed:\n{result.stdout}"
            )

    def _install_package(self) -> None:
        """Best-effort package installation — errors are logged but not fatal."""
        result = self._run(
            f"cd {self.REPO_DIR} && "
            "pip install -e . -q 2>&1 || "
            "pip install -r requirements.txt -q 2>&1 || "
            "pip install -r requirements-dev.txt -q 2>&1 || true",
            timeout=300,
        )
        if not result.ok:
            logger.warning(f"Package install may have failed:\n{result.stdout[:500]}")

    def _apply_patch_str(self, patch: str, label: str = "patch") -> None:
        """Write patch content to container and apply it."""
        self.write_file("/tmp/apply.patch", patch)
        result = self._run(
            f"cd {self.REPO_DIR} && git apply /tmp/apply.patch 2>&1"
        )
        if not result.ok:
            # Fallback to GNU patch
            result2 = self._run(
                f"cd {self.REPO_DIR} && patch -p1 < /tmp/apply.patch 2>&1"
            )
            if not result2.ok:
                raise SandboxError(
                    f"Failed to apply {label}:\n"
                    f"git apply: {result.stdout}\n"
                    f"patch -p1: {result2.stdout}"
                )

    def _run(self, command: str, timeout: int = 60) -> ExecResult:
        """Execute a raw bash command in the container."""
        if not self._container:
            raise SandboxError("Sandbox not started. Call setup() first.")

        docker_result = self._container.exec_run(
            ["timeout", str(timeout), "bash", "-c", command],
            stdout=True,
            stderr=True,
        )
        return ExecResult(
            stdout=docker_result.output.decode("utf-8", errors="replace"),
            exit_code=docker_result.exit_code,
        )

    def _abs(self, path: str) -> str:
        """Resolve a relative path to absolute within the repo."""
        if path.startswith("/"):
            return path
        return f"{self.REPO_DIR}/{path}"

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.teardown()
