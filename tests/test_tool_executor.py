"""
Tests for ToolExecutor.

Fast unit tests mock the DockerSandbox — no Docker needed.
Integration tests (marked @pytest.mark.integration) require a running Docker
daemon and the swebench-sandbox:latest image. Run with:

    pytest -m integration tests/test_tool_executor.py
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.tool_executor import (
    TOOL_DEFINITIONS,
    HEAD_TOKENS,
    MAX_TOKENS,
    TAIL_TOKENS,
    PatchSubmitted,
    ToolError,
    ToolExecutor,
)
from src.sandbox import DockerSandbox, ExecResult, SandboxError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def executor() -> ToolExecutor:
    return ToolExecutor()


@pytest.fixture()
def sandbox() -> MagicMock:
    """Mocked DockerSandbox — no Docker required."""
    sb = MagicMock(spec=DockerSandbox)
    sb.exec.return_value = ExecResult(stdout="ok", exit_code=0)
    sb.read_file.return_value = "1: hello\n2: world"
    sb.write_file.return_value = None
    sb.edit_file.return_value = "new content"
    sb.search_code.return_value = "src/foo.py:1: match"
    sb.list_files.return_value = "file1.py\nfile2.py"
    return sb


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


class TestToolDefinitions:
    EXPECTED_TOOLS = {
        "bash",
        "read_file",
        "write_file",
        "edit_file",
        "search_code",
        "list_files",
        "submit_patch",
    }

    def test_all_tools_present(self, executor: ToolExecutor):
        names = {t["name"] for t in executor.tool_definitions}
        assert names == self.EXPECTED_TOOLS

    def test_each_tool_has_input_schema(self, executor: ToolExecutor):
        for tool in executor.tool_definitions:
            assert "input_schema" in tool, f"{tool['name']} missing input_schema"

    def test_each_tool_has_description(self, executor: ToolExecutor):
        for tool in executor.tool_definitions:
            assert tool.get("description"), f"{tool['name']} missing description"

    def test_tool_definitions_property_returns_list(self, executor: ToolExecutor):
        assert isinstance(executor.tool_definitions, list)
        assert len(executor.tool_definitions) == len(TOOL_DEFINITIONS)


# ---------------------------------------------------------------------------
# bash
# ---------------------------------------------------------------------------


class TestBash:
    def test_returns_stdout_on_success(self, executor, sandbox):
        sandbox.exec.return_value = ExecResult(stdout="hello\n", exit_code=0)
        result = executor.execute("bash", {"command": "echo hello"}, sandbox)
        assert "hello" in result

    def test_calls_sandbox_exec(self, executor, sandbox):
        executor.execute("bash", {"command": "ls"}, sandbox)
        sandbox.exec.assert_called_once_with("ls", timeout=30)

    def test_respects_timeout_param(self, executor, sandbox):
        executor.execute("bash", {"command": "sleep 1", "timeout": 10}, sandbox)
        sandbox.exec.assert_called_once_with("sleep 1", timeout=10)

    def test_caps_timeout_at_120(self, executor, sandbox):
        executor.execute("bash", {"command": "x", "timeout": 9999}, sandbox)
        _, kwargs = sandbox.exec.call_args
        assert kwargs["timeout"] == 120

    def test_nonzero_exit_includes_exit_code(self, executor, sandbox):
        sandbox.exec.return_value = ExecResult(stdout="not found", exit_code=1)
        result = executor.execute("bash", {"command": "false"}, sandbox)
        assert "Exit code 1" in result
        assert "not found" in result

    def test_empty_stdout_returns_placeholder(self, executor, sandbox):
        sandbox.exec.return_value = ExecResult(stdout="", exit_code=0)
        result = executor.execute("bash", {"command": "true"}, sandbox)
        assert result == "(no output)"


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------


class TestReadFile:
    def test_returns_file_content(self, executor, sandbox):
        result = executor.execute("read_file", {"path": "foo.py"}, sandbox)
        assert "hello" in result

    def test_passes_path_to_sandbox(self, executor, sandbox):
        executor.execute("read_file", {"path": "src/bar.py"}, sandbox)
        sandbox.read_file.assert_called_once_with(
            "src/bar.py", start_line=None, end_line=None
        )

    def test_passes_line_range(self, executor, sandbox):
        executor.execute(
            "read_file", {"path": "foo.py", "start_line": 5, "end_line": 10}, sandbox
        )
        sandbox.read_file.assert_called_once_with("foo.py", start_line=5, end_line=10)

    def test_file_not_found_returns_error_string(self, executor, sandbox):
        sandbox.read_file.side_effect = FileNotFoundError("no such file")
        result = executor.execute("read_file", {"path": "missing.py"}, sandbox)
        assert "Error" in result


# ---------------------------------------------------------------------------
# write_file
# ---------------------------------------------------------------------------


class TestWriteFile:
    def test_returns_written_path(self, executor, sandbox):
        result = executor.execute(
            "write_file", {"path": "out.py", "content": "x = 1"}, sandbox
        )
        assert "out.py" in result

    def test_calls_sandbox_write_file(self, executor, sandbox):
        executor.execute(
            "write_file", {"path": "out.py", "content": "x = 1"}, sandbox
        )
        sandbox.write_file.assert_called_once_with("out.py", "x = 1")

    def test_sandbox_error_returned_as_string(self, executor, sandbox):
        sandbox.write_file.side_effect = SandboxError("disk full")
        result = executor.execute(
            "write_file", {"path": "out.py", "content": ""}, sandbox
        )
        assert "Error" in result


# ---------------------------------------------------------------------------
# edit_file
# ---------------------------------------------------------------------------


class TestEditFile:
    def test_returns_edited_path(self, executor, sandbox):
        result = executor.execute(
            "edit_file",
            {"path": "foo.py", "old_str": "old", "new_str": "new"},
            sandbox,
        )
        assert "foo.py" in result

    def test_calls_sandbox_edit_file(self, executor, sandbox):
        executor.execute(
            "edit_file",
            {"path": "foo.py", "old_str": "old", "new_str": "new"},
            sandbox,
        )
        sandbox.edit_file.assert_called_once_with("foo.py", "old", "new")

    def test_string_not_found_returns_error(self, executor, sandbox):
        sandbox.edit_file.side_effect = ValueError("String not found")
        result = executor.execute(
            "edit_file",
            {"path": "foo.py", "old_str": "missing", "new_str": "x"},
            sandbox,
        )
        assert "Error" in result


# ---------------------------------------------------------------------------
# search_code
# ---------------------------------------------------------------------------


class TestSearchCode:
    def test_returns_search_results(self, executor, sandbox):
        result = executor.execute(
            "search_code", {"pattern": "def foo"}, sandbox
        )
        assert "match" in result

    def test_passes_pattern_to_sandbox(self, executor, sandbox):
        executor.execute("search_code", {"pattern": "class Bar"}, sandbox)
        sandbox.search_code.assert_called_once_with("class Bar", file_glob=None)

    def test_passes_file_glob(self, executor, sandbox):
        executor.execute(
            "search_code", {"pattern": "TODO", "file_glob": "*.py"}, sandbox
        )
        sandbox.search_code.assert_called_once_with("TODO", file_glob="*.py")


# ---------------------------------------------------------------------------
# list_files
# ---------------------------------------------------------------------------


class TestListFiles:
    def test_returns_file_list(self, executor, sandbox):
        result = executor.execute("list_files", {}, sandbox)
        assert "file1.py" in result

    def test_default_path_is_dot(self, executor, sandbox):
        executor.execute("list_files", {}, sandbox)
        sandbox.list_files.assert_called_once_with(".", recursive=False, max_depth=2)

    def test_passes_path_and_flags(self, executor, sandbox):
        executor.execute(
            "list_files",
            {"path": "src", "recursive": True, "max_depth": 3},
            sandbox,
        )
        sandbox.list_files.assert_called_once_with("src", recursive=True, max_depth=3)


# ---------------------------------------------------------------------------
# submit_patch
# ---------------------------------------------------------------------------


class TestSubmitPatch:
    def test_raises_patch_submitted(self, executor, sandbox):
        with pytest.raises(PatchSubmitted):
            executor.execute("submit_patch", {}, sandbox)

    def test_patch_submitted_carries_message(self, executor, sandbox):
        with pytest.raises(PatchSubmitted, match="fixed the bug"):
            executor.execute(
                "submit_patch", {"message": "fixed the bug"}, sandbox
            )

    def test_patch_submitted_not_caught_as_tool_error(self, executor, sandbox):
        """PatchSubmitted must propagate, not be swallowed by the error handler."""
        raised = False
        try:
            executor.execute("submit_patch", {}, sandbox)
        except PatchSubmitted:
            raised = True
        except Exception:
            pass
        assert raised


# ---------------------------------------------------------------------------
# Unknown tool
# ---------------------------------------------------------------------------


class TestUnknownTool:
    def test_raises_tool_error(self, executor, sandbox):
        with pytest.raises(ToolError, match="unknown_tool"):
            executor.execute("unknown_tool", {}, sandbox)


# ---------------------------------------------------------------------------
# Output truncation
# ---------------------------------------------------------------------------


class TestTruncation:
    def test_short_output_not_truncated(self, executor):
        text = "hello world"
        assert executor._truncate(text) == text

    def test_long_output_truncated(self, executor):
        # Build a string that exceeds MAX_TOKENS
        word = "token " * (MAX_TOKENS + 500)
        result = executor._truncate(word)
        assert "[... truncated" in result

    def test_truncated_output_has_head_and_tail(self, executor):
        # Create a string where head and tail are identifiable
        head_marker = "STARTOFFILE " * HEAD_TOKENS
        tail_marker = "ENDOFFILE " * TAIL_TOKENS
        big_middle = "MIDDLE " * 3_000
        text = head_marker + big_middle + tail_marker

        result = executor._truncate(text)
        assert "STARTOFFILE" in result
        assert "ENDOFFILE" in result
        assert "[... truncated" in result

    def test_truncation_message_includes_count(self, executor):
        word = "x " * (MAX_TOKENS + 1_000)
        result = executor._truncate(word)
        # Should contain a number in the truncation notice
        import re
        assert re.search(r"\[\.\.\. truncated [\d,]+ tokens \.\.\.\]", result)

    def test_truncated_token_count_at_most_head_plus_tail(self, executor):
        word = "word " * (MAX_TOKENS * 2)
        result = executor._truncate(word)
        encoding = executor._encoding
        token_count = len(encoding.encode(result))
        # Allow some slack for the truncation notice itself
        assert token_count <= HEAD_TOKENS + TAIL_TOKENS + 50

    def test_execute_truncates_large_output(self, executor, sandbox):
        big_output = "line\n" * 50_000
        sandbox.exec.return_value = ExecResult(stdout=big_output, exit_code=0)
        result = executor.execute("bash", {"command": "cat big.txt"}, sandbox)
        encoding = executor._encoding
        assert len(encoding.encode(result)) <= HEAD_TOKENS + TAIL_TOKENS + 50


# ---------------------------------------------------------------------------
# Integration tests — require Docker + swebench-sandbox:latest image
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestIntegration:
    """
    Real sandbox tests. Requires:
      - Docker daemon running
      - swebench-sandbox:latest image built  (`bash docker/build.sh`)
    """

    @pytest.fixture(scope="class")
    def live_sandbox(self):
        """Start a bare container (no repo clone) for tool smoke-tests."""
        from src.sandbox import DockerSandbox
        import docker

        try:
            client = docker.from_env()
            client.ping()
        except Exception:
            pytest.skip("Docker daemon not available")
        container = client.containers.run(
            "swebench-sandbox:latest",
            command="sleep infinity",
            detach=True,
            remove=False,
            working_dir="/workspace",
        )
        # Minimal stub that wraps the container for exec() only
        sb = MagicMock(spec=DockerSandbox)
        sb.REPO_DIR = "/workspace"

        def _exec(command, timeout=30):
            result = container.exec_run(
                ["bash", "-c", command], stdout=True, stderr=True
            )
            return ExecResult(
                stdout=result.output.decode("utf-8", errors="replace"),
                exit_code=result.exit_code,
            )

        sb.exec.side_effect = _exec
        sb.read_file.side_effect = lambda p, **kw: _exec(f"cat /workspace/{p}").stdout
        sb.write_file.side_effect = lambda p, c: _exec(
            f"echo '{c}' > /workspace/{p}"
        )
        sb.list_files.side_effect = lambda p, **kw: _exec(f"ls /workspace/{p}").stdout
        sb.search_code.side_effect = lambda pattern, **kw: _exec(
            f"echo 'rg {pattern}'"
        ).stdout

        yield sb

        container.stop(timeout=5)
        container.remove(force=True)

    def test_bash_echo(self, live_sandbox):
        executor = ToolExecutor()
        result = executor.execute(
            "bash", {"command": "echo hello_integration"}, live_sandbox
        )
        assert "hello_integration" in result

    def test_bash_nonzero_exit(self, live_sandbox):
        executor = ToolExecutor()
        result = executor.execute(
            "bash", {"command": "exit 42"}, live_sandbox
        )
        assert "42" in result

    def test_write_then_read(self, live_sandbox):
        executor = ToolExecutor()
        executor.execute(
            "write_file",
            {"path": "test_write.txt", "content": "integration_test_content"},
            live_sandbox,
        )
        result = executor.execute(
            "read_file", {"path": "test_write.txt"}, live_sandbox
        )
        assert "integration_test_content" in result

    def test_submit_patch_raises(self, live_sandbox):
        executor = ToolExecutor()
        with pytest.raises(PatchSubmitted):
            executor.execute("submit_patch", {"message": "done"}, live_sandbox)
