"""
Tool executor: routes LLM tool-use calls to DockerSandbox operations.

Supported tools
---------------
bash          – run an arbitrary shell command
read_file     – read a file, optionally between two line numbers
write_file    – overwrite a file with given content
edit_file     – replace the first occurrence of a string in a file
search_code   – search source with ripgrep
list_files    – list directory contents
submit_patch  – signal the agent is done (raises PatchSubmitted)

Output truncation
-----------------
Each tool result is capped at MAX_TOKENS tokens.  If it exceeds the cap the
output is shown as:

    <first HEAD_TOKENS tokens>
    [... truncated N tokens ...]
    <last TAIL_TOKENS tokens>
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import tiktoken

from src.sandbox import DockerSandbox, SandboxError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_TOKENS = 10_000
HEAD_TOKENS = 4_000
TAIL_TOKENS = 4_000


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class PatchSubmitted(Exception):
    """Raised by execute() when the agent calls submit_patch."""


class ToolError(Exception):
    """Raised for unknown tool names or missing required inputs."""


# ---------------------------------------------------------------------------
# Tool definitions  (Anthropic tool_use / JSON-schema format)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "bash",
        "description": (
            "Run a shell command inside the repository directory. "
            "Stdout and stderr are merged. Exit code is shown if non-zero."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default 30, max 120).",
                    "default": 30,
                },
            },
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": (
            "Read a file from the repository. Returns numbered lines. "
            "Optionally restrict to a line range."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to the repo root (or absolute).",
                },
                "start_line": {
                    "type": "integer",
                    "description": "First line to return (1-indexed, inclusive).",
                },
                "end_line": {
                    "type": "integer",
                    "description": "Last line to return (1-indexed, inclusive).",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Overwrite (or create) a file with the given content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to the repo root (or absolute).",
                },
                "content": {
                    "type": "string",
                    "description": "Full file content to write.",
                },
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": (
            "Replace the first occurrence of old_str with new_str in a file. "
            "Fails if old_str is not found."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to the repo root (or absolute).",
                },
                "old_str": {
                    "type": "string",
                    "description": "Exact string to replace (must be present in the file).",
                },
                "new_str": {
                    "type": "string",
                    "description": "Replacement string.",
                },
            },
            "required": ["path", "old_str", "new_str"],
        },
    },
    {
        "name": "search_code",
        "description": (
            "Search the repository for a regex pattern using ripgrep. "
            "Returns matching lines with file paths and line numbers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for.",
                },
                "file_glob": {
                    "type": "string",
                    "description": "Optional file glob to restrict the search (e.g. '*.py').",
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "list_files",
        "description": "List files in a directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path relative to the repo root (or absolute).",
                    "default": ".",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "If true, list files recursively up to max_depth.",
                    "default": False,
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Maximum directory depth when recursive=true (default 2).",
                    "default": 2,
                },
            },
            "required": [],
        },
    },
    {
        "name": "submit_patch",
        "description": (
            "Signal that you have finished editing the code. "
            "Call this once you believe your changes fix the issue. "
            "No further tool calls will be accepted after this."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Optional summary of what you changed and why.",
                },
            },
            "required": [],
        },
    },
]


# ---------------------------------------------------------------------------
# ToolExecutor
# ---------------------------------------------------------------------------


class ToolExecutor:
    """
    Routes Anthropic tool_use blocks to DockerSandbox operations.

    Usage::

        executor = ToolExecutor()

        # Pass executor.tool_definitions to LLMClient.complete()
        response = llm_client.complete(messages, tools=executor.tool_definitions)

        # Dispatch each tool_use block
        for block in response.content:
            if block.type == "tool_use":
                try:
                    result = executor.execute(block.name, block.input, sandbox)
                except PatchSubmitted:
                    diff = sandbox.get_diff()   # agent is done
                    break
                except ToolError as e:
                    result = f"Error: {e}"
    """

    def __init__(
        self,
        docker_image: str = "swebench-sandbox:latest",
        timeout_per_task: int = 600,
    ) -> None:
        self._encoding = tiktoken.get_encoding("cl100k_base")
        self._docker_image = docker_image
        self._timeout_per_task = timeout_per_task
        self._sandbox: Optional[DockerSandbox] = None

    # ------------------------------------------------------------------
    # Sandbox lifecycle (called by orchestrator)
    # ------------------------------------------------------------------

    def setup_sandbox(self, task) -> DockerSandbox:
        """
        Create, start, and return a DockerSandbox for the given task.
        Stores a reference so teardown_sandbox() can clean it up.
        """
        from src.sandbox import DockerSandbox as _DockerSandbox
        sandbox = _DockerSandbox(
            docker_image=self._docker_image,
            timeout_per_task=self._timeout_per_task,
        )
        sandbox.setup(task)
        self._sandbox = sandbox
        return sandbox

    def teardown_sandbox(self) -> None:
        """Stop and remove the current sandbox container."""
        if self._sandbox is not None:
            self._sandbox.teardown()
            self._sandbox = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def tool_definitions(self) -> list[dict[str, Any]]:
        """Return the list of tool definitions to pass to LLMClient.complete()."""
        return TOOL_DEFINITIONS

    def execute(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        sandbox: DockerSandbox,
    ) -> str:
        """
        Dispatch a tool call to the sandbox and return the result string.

        Args:
            tool_name:  Name from the LLM's tool_use block.
            tool_input: Input dict from the LLM's tool_use block.
            sandbox:    A running DockerSandbox instance.

        Returns:
            String to send back as the tool_result content.

        Raises:
            PatchSubmitted: when tool_name == "submit_patch".
            ToolError:      for unknown tool names.
        """
        handler = {
            "bash":         self._bash,
            "read_file":    self._read_file,
            "write_file":   self._write_file,
            "edit_file":    self._edit_file,
            "search_code":  self._search_code,
            "list_files":   self._list_files,
            "submit_patch": self._submit_patch,
        }.get(tool_name)

        if handler is None:
            raise ToolError(f"Unknown tool: '{tool_name}'")

        logger.debug(
            "Tool call: %s %s",
            tool_name,
            {k: str(v)[:80] for k, v in tool_input.items()},
        )

        try:
            raw = handler(tool_input, sandbox)
        except PatchSubmitted:
            raise
        except (FileNotFoundError, ValueError, SandboxError) as exc:
            raw = f"Error: {exc}"
        except Exception as exc:
            raw = f"Unexpected error in {tool_name}: {exc}"

        result = self._truncate(raw)
        logger.debug(
            "Tool result (%s): %d chars",
            tool_name,
            len(result),
        )
        return result

    # ------------------------------------------------------------------
    # Tool handlers
    # ------------------------------------------------------------------

    def _bash(self, inp: dict, sandbox: DockerSandbox) -> str:
        command = inp["command"]
        timeout = min(int(inp.get("timeout", 30)), 120)
        result = sandbox.exec(command, timeout=timeout)
        if result.ok:
            return result.stdout or "(no output)"
        return f"Exit code {result.exit_code}:\n{result.stdout}"

    def _read_file(self, inp: dict, sandbox: DockerSandbox) -> str:
        return sandbox.read_file(
            inp["path"],
            start_line=inp.get("start_line"),
            end_line=inp.get("end_line"),
        )

    def _write_file(self, inp: dict, sandbox: DockerSandbox) -> str:
        sandbox.write_file(inp["path"], inp["content"])
        return f"Written: {inp['path']}"

    def _edit_file(self, inp: dict, sandbox: DockerSandbox) -> str:
        sandbox.edit_file(inp["path"], inp["old_str"], inp["new_str"])
        return f"Edited: {inp['path']}"

    def _search_code(self, inp: dict, sandbox: DockerSandbox) -> str:
        return sandbox.search_code(
            inp["pattern"],
            file_glob=inp.get("file_glob"),
        )

    def _list_files(self, inp: dict, sandbox: DockerSandbox) -> str:
        return sandbox.list_files(
            inp.get("path", "."),
            recursive=inp.get("recursive", False),
            max_depth=inp.get("max_depth", 2),
        )

    def _submit_patch(self, inp: dict, sandbox: DockerSandbox) -> str:
        message = inp.get("message", "")
        raise PatchSubmitted(message)

    # ------------------------------------------------------------------
    # Output truncation
    # ------------------------------------------------------------------

    def _truncate(self, text: str) -> str:
        """
        Cap output at MAX_TOKENS tokens.

        If the text exceeds the cap, keep the first HEAD_TOKENS tokens and
        the last TAIL_TOKENS tokens, separated by a truncation notice.
        """
        tokens = self._encoding.encode(text)
        if len(tokens) <= MAX_TOKENS:
            return text

        head = self._encoding.decode(tokens[:HEAD_TOKENS])
        tail = self._encoding.decode(tokens[-TAIL_TOKENS:])
        dropped = len(tokens) - HEAD_TOKENS - TAIL_TOKENS
        return f"{head}\n[... truncated {dropped:,} tokens ...]\n{tail}"
