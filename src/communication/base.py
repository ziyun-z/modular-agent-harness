from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.llm_client import LLMClient
    from src.tool_executor import ToolExecutor
    from src.memory.base import MemoryModule
    from src.compression.base import CompressionModule


@dataclass
class AgentConfig:
    """Configuration for a single agent in the system."""
    name: str                               # e.g., "planner", "code_reader", "writer"
    system_prompt: str                      # Role-specific system prompt
    tools: list[str]                        # Which tools this agent can use
    memory_module: MemoryModule             # Each agent can have its own memory
    compression_module: CompressionModule


@dataclass
class AgentMessage:
    """A message passed between agents."""
    sender: str
    recipient: str          # Or "broadcast" for blackboard-style
    content: str
    message_type: str       # "task", "result", "status", "query"
    metadata: dict[str, Any] = field(default_factory=dict)


class CommunicationModule(ABC):
    """Interface for all communication/coordination implementations."""

    @abstractmethod
    def setup(self, task_description: str, llm_client: LLMClient) -> None:
        """
        Initialize the communication structure for a new task.
        For single agent: no-op.
        For multi-agent: spawn agent configs, set up channels.
        """
        ...

    @abstractmethod
    def run_step(
        self,
        llm_client: LLMClient,
        tool_executor: ToolExecutor,
    ) -> dict[str, Any]:
        """
        Execute one step of the agent system.

        For single agent: one LLM call + tool execution.
        For multi-agent: may involve multiple LLM calls coordinated
                         according to the architecture.

        Returns:
            Dict with keys:
                - "actions_taken": list of actions executed
                - "done": bool, whether the agent thinks the task is complete
                - "result": optional final result if done=True
                - "llm_calls": int, number of LLM calls made this step
        """
        ...

    @abstractmethod
    def get_trajectory(self) -> list[dict]:
        """Return the full trajectory for logging/analysis."""
        ...

    def update_trajectory(self, compressed_turns: list) -> None:
        """Replace internal trajectory with compressed version."""
        raise NotImplementedError

    def get_stats(self) -> dict:
        """Return communication metrics (messages sent, agents used, etc.)."""
        return {}
