"""
LLM client wrapping the Anthropic API.

Features:
- Tool use (function calling)
- Token counting via tiktoken (cl100k_base approximation)
- Cost tracking (input/output tokens × model price)
- Exponential backoff retry (3 attempts, handles rate limits + 5xx errors)
- Metrics: total calls, tokens, cost across a run
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any

import anthropic
import tiktoken

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

# USD per million tokens, as of 2025
_MODEL_PRICING: dict[str, dict[str, float]] = {
    "claude-opus-4-6":           {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-6":         {"input":  3.00, "output": 15.00},
    "claude-haiku-4-5":          {"input":  0.25, "output":  1.25},
    "claude-haiku-4-5-20251001": {"input":  0.25, "output":  1.25},
}
_DEFAULT_PRICING: dict[str, float] = {"input": 3.00, "output": 15.00}  # sonnet fallback


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class LLMError(Exception):
    """Raised when all retry attempts are exhausted or an unrecoverable error occurs."""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class LLMClient:
    """
    Anthropic API wrapper with tool use, token counting, cost tracking,
    and exponential backoff retries.

    Lifecycle:
        client = LLMClient(model="claude-sonnet-4-6")
        response = client.complete(
            messages=[{"role": "user", "content": "Hello"}],
            system="You are a helpful assistant.",
        )
        block = response.content[0]          # TextBlock or ToolUseBlock
        print(client.get_stats())
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        max_retries: int = 3,
        base_delay: float = 1.0,
    ):
        self.model = model
        self.max_retries = max_retries
        self.base_delay = base_delay

        self._client = anthropic.Anthropic()
        self._encoding = tiktoken.get_encoding("cl100k_base")

        # Accumulated metrics
        self._total_calls: int = 0
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._total_cost_usd: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system: str | None = None,
        max_tokens: int = 4096,
        model: str | None = None,
    ) -> anthropic.types.Message:
        """
        Call the Anthropic Messages API with automatic retry.

        Args:
            messages:   Conversation history in Anthropic format.
            tools:      Tool definitions in Anthropic tool_use format.
            system:     System prompt string.
            max_tokens: Maximum tokens in the response.
            model:      Override the instance model for this single call.

        Returns:
            anthropic.types.Message — inspect .content, .stop_reason, .usage.

        Raises:
            LLMError: if all retries are exhausted.
        """
        effective_model = model or self.model
        kwargs: dict[str, Any] = {
            "model": effective_model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self._client.messages.create(**kwargs)
                self._record(response, effective_model)
                return response

            except anthropic.RateLimitError as exc:
                last_exc = exc
                delay = self._backoff(attempt)
                logger.warning(
                    "Rate limit hit (attempt %d/%d), retrying in %.1fs",
                    attempt + 1, self.max_retries, delay,
                )
                time.sleep(delay)

            except anthropic.APIStatusError as exc:
                if exc.status_code >= 500:
                    last_exc = exc
                    delay = self._backoff(attempt)
                    logger.warning(
                        "API error %d (attempt %d/%d), retrying in %.1fs",
                        exc.status_code, attempt + 1, self.max_retries, delay,
                    )
                    time.sleep(delay)
                else:
                    raise LLMError(
                        f"Anthropic API error {exc.status_code}: {exc.message}"
                    ) from exc

            except anthropic.APIConnectionError as exc:
                last_exc = exc
                delay = self._backoff(attempt)
                logger.warning(
                    "Connection error (attempt %d/%d), retrying in %.1fs",
                    attempt + 1, self.max_retries, delay,
                )
                time.sleep(delay)

        raise LLMError(
            f"LLM call failed after {self.max_retries} attempts"
        ) from last_exc

    def count_tokens(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
    ) -> int:
        """
        Estimate token count for a message list using tiktoken (cl100k_base).

        This is an approximation — Claude uses its own tokeniser, but cl100k_base
        is close enough for budget checks (~5% error in practice).
        """
        total = 0
        if system:
            total += len(self._encoding.encode(system))
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += len(self._encoding.encode(content))
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        text = block.get("text") or str(block.get("input", ""))
                        total += len(self._encoding.encode(text))
        return total

    def get_stats(self) -> dict[str, Any]:
        """Return accumulated metrics for all calls made by this client instance."""
        return {
            "total_calls": self._total_calls,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "estimated_cost_usd": round(self._total_cost_usd, 6),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record(self, response: anthropic.types.Message, model: str) -> None:
        """Update accumulated metrics from a successful API response."""
        pricing = _MODEL_PRICING.get(model, _DEFAULT_PRICING)
        input_tok = response.usage.input_tokens
        output_tok = response.usage.output_tokens
        cost = (
            (input_tok  / 1_000_000) * pricing["input"]
            + (output_tok / 1_000_000) * pricing["output"]
        )
        self._total_calls += 1
        self._total_input_tokens += input_tok
        self._total_output_tokens += output_tok
        self._total_cost_usd += cost
        logger.debug(
            "LLM call: %d in / %d out / $%.6f  |  session total: %d calls $%.4f",
            input_tok, output_tok, cost, self._total_calls, self._total_cost_usd,
        )

    def _backoff(self, attempt: int) -> float:
        """Exponential backoff with full jitter: uniform(0, base * 2^attempt)."""
        cap = self.base_delay * (2 ** attempt)
        return random.uniform(0, cap)
