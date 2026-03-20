"""
Tests for LLMClient.

Fast unit tests mock the Anthropic SDK — no network or API key needed.
Integration tests (marked @pytest.mark.integration) make real API calls
and require ANTHROPIC_API_KEY in the environment. Run with:

    pytest -m integration tests/test_llm_client.py
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, call
import time

import pytest
import anthropic

from src.llm_client import LLMClient, LLMError, _MODEL_PRICING, _DEFAULT_PRICING


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(
    input_tokens: int = 10,
    output_tokens: int = 20,
    content: list | None = None,
    stop_reason: str = "end_turn",
) -> MagicMock:
    """Build a fake anthropic.types.Message."""
    resp = MagicMock(spec=anthropic.types.Message)
    resp.usage = MagicMock()
    resp.usage.input_tokens = input_tokens
    resp.usage.output_tokens = output_tokens
    resp.stop_reason = stop_reason
    resp.content = content or [MagicMock(type="text", text="Hello!")]
    return resp


def _make_tool_use_response(tool_name: str = "bash", tool_input: dict | None = None) -> MagicMock:
    """Build a fake response with a tool_use block."""
    block = MagicMock()
    block.type = "tool_use"
    block.id = "toolu_01"
    block.name = tool_name
    block.input = tool_input or {"command": "echo hello"}

    resp = _make_response(stop_reason="tool_use", content=[block])
    return resp


@pytest.fixture()
def client() -> LLMClient:
    """LLMClient with mocked Anthropic internals — no API calls."""
    with patch("src.llm_client.anthropic.Anthropic"):
        c = LLMClient(model="claude-sonnet-4-6")
    return c


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_model(self):
        with patch("src.llm_client.anthropic.Anthropic"):
            c = LLMClient()
        assert c.model == "claude-sonnet-4-6"

    def test_custom_model(self):
        with patch("src.llm_client.anthropic.Anthropic"):
            c = LLMClient(model="claude-haiku-4-5")
        assert c.model == "claude-haiku-4-5"

    def test_initial_stats_are_zero(self, client: LLMClient):
        stats = client.get_stats()
        assert stats["total_calls"] == 0
        assert stats["total_input_tokens"] == 0
        assert stats["total_output_tokens"] == 0
        assert stats["estimated_cost_usd"] == 0.0


# ---------------------------------------------------------------------------
# complete() — happy path
# ---------------------------------------------------------------------------

class TestComplete:
    def test_returns_response(self, client: LLMClient):
        resp = _make_response()
        client._client.messages.create.return_value = resp
        result = client.complete([{"role": "user", "content": "Hi"}])
        assert result is resp

    def test_passes_messages(self, client: LLMClient):
        client._client.messages.create.return_value = _make_response()
        messages = [{"role": "user", "content": "Hi"}]
        client.complete(messages)
        kwargs = client._client.messages.create.call_args.kwargs
        assert kwargs["messages"] == messages

    def test_passes_system(self, client: LLMClient):
        client._client.messages.create.return_value = _make_response()
        client.complete([{"role": "user", "content": "Hi"}], system="Be helpful.")
        kwargs = client._client.messages.create.call_args.kwargs
        assert kwargs["system"] == "Be helpful."

    def test_omits_system_when_none(self, client: LLMClient):
        client._client.messages.create.return_value = _make_response()
        client.complete([{"role": "user", "content": "Hi"}])
        kwargs = client._client.messages.create.call_args.kwargs
        assert "system" not in kwargs

    def test_passes_tools(self, client: LLMClient):
        client._client.messages.create.return_value = _make_response()
        tools = [{"name": "bash", "description": "Run a command", "input_schema": {}}]
        client.complete([{"role": "user", "content": "Hi"}], tools=tools)
        kwargs = client._client.messages.create.call_args.kwargs
        assert kwargs["tools"] == tools

    def test_omits_tools_when_none(self, client: LLMClient):
        client._client.messages.create.return_value = _make_response()
        client.complete([{"role": "user", "content": "Hi"}])
        kwargs = client._client.messages.create.call_args.kwargs
        assert "tools" not in kwargs

    def test_model_override(self, client: LLMClient):
        client._client.messages.create.return_value = _make_response()
        client.complete([{"role": "user", "content": "Hi"}], model="claude-haiku-4-5")
        kwargs = client._client.messages.create.call_args.kwargs
        assert kwargs["model"] == "claude-haiku-4-5"

    def test_default_model_used_when_no_override(self, client: LLMClient):
        client._client.messages.create.return_value = _make_response()
        client.complete([{"role": "user", "content": "Hi"}])
        kwargs = client._client.messages.create.call_args.kwargs
        assert kwargs["model"] == "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Metrics tracking
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_increments_call_count(self, client: LLMClient):
        client._client.messages.create.return_value = _make_response(10, 20)
        client.complete([{"role": "user", "content": "Hi"}])
        assert client.get_stats()["total_calls"] == 1

    def test_accumulates_tokens(self, client: LLMClient):
        client._client.messages.create.return_value = _make_response(100, 200)
        client.complete([{"role": "user", "content": "Hi"}])
        client.complete([{"role": "user", "content": "Hi"}])
        stats = client.get_stats()
        assert stats["total_input_tokens"] == 200
        assert stats["total_output_tokens"] == 400

    def test_cost_calculation_sonnet(self, client: LLMClient):
        # 1M input @ $3, 1M output @ $15 → $18 per million of each
        client._client.messages.create.return_value = _make_response(1_000_000, 1_000_000)
        client.complete([{"role": "user", "content": "Hi"}])
        stats = client.get_stats()
        assert abs(stats["estimated_cost_usd"] - 18.0) < 0.001

    def test_cost_uses_model_override_pricing(self, client: LLMClient):
        # Haiku: $0.25 input + $1.25 output per million
        client._client.messages.create.return_value = _make_response(1_000_000, 1_000_000)
        client.complete([{"role": "user", "content": "Hi"}], model="claude-haiku-4-5")
        stats = client.get_stats()
        assert abs(stats["estimated_cost_usd"] - 1.50) < 0.001

    def test_stats_accumulate_across_calls(self, client: LLMClient):
        client._client.messages.create.side_effect = [
            _make_response(10, 5),
            _make_response(20, 10),
        ]
        client.complete([{"role": "user", "content": "A"}])
        client.complete([{"role": "user", "content": "B"}])
        stats = client.get_stats()
        assert stats["total_calls"] == 2
        assert stats["total_input_tokens"] == 30
        assert stats["total_output_tokens"] == 15


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------

class TestRetry:
    def test_retries_on_rate_limit(self, client: LLMClient):
        client._client.messages.create.side_effect = [
            anthropic.RateLimitError("rate limit", response=MagicMock(), body={}),
            _make_response(),
        ]
        with patch("src.llm_client.time.sleep"):
            result = client.complete([{"role": "user", "content": "Hi"}])
        assert result is not None
        assert client._client.messages.create.call_count == 2

    def test_retries_on_500_error(self, client: LLMClient):
        err = anthropic.APIStatusError("server error", response=MagicMock(status_code=500), body={})
        client._client.messages.create.side_effect = [err, _make_response()]
        with patch("src.llm_client.time.sleep"):
            result = client.complete([{"role": "user", "content": "Hi"}])
        assert result is not None

    def test_raises_llm_error_after_max_retries(self, client: LLMClient):
        client.max_retries = 3
        client._client.messages.create.side_effect = anthropic.RateLimitError(
            "rate limit", response=MagicMock(), body={}
        )
        with patch("src.llm_client.time.sleep"):
            with pytest.raises(LLMError):
                client.complete([{"role": "user", "content": "Hi"}])
        assert client._client.messages.create.call_count == 3

    def test_does_not_retry_on_4xx(self, client: LLMClient):
        err = anthropic.APIStatusError(
            "bad request", response=MagicMock(status_code=400), body={}
        )
        client._client.messages.create.side_effect = err
        with pytest.raises(LLMError):
            client.complete([{"role": "user", "content": "Hi"}])
        assert client._client.messages.create.call_count == 1

    def test_sleeps_between_retries(self, client: LLMClient):
        client.max_retries = 2
        client._client.messages.create.side_effect = anthropic.RateLimitError(
            "rate limit", response=MagicMock(), body={}
        )
        with patch("src.llm_client.time.sleep") as mock_sleep:
            with pytest.raises(LLMError):
                client.complete([{"role": "user", "content": "Hi"}])
        assert mock_sleep.call_count == 2


# ---------------------------------------------------------------------------
# count_tokens
# ---------------------------------------------------------------------------

class TestCountTokens:
    def test_empty_messages_returns_zero(self, client: LLMClient):
        assert client.count_tokens([]) == 0

    def test_counts_system_tokens(self, client: LLMClient):
        assert client.count_tokens([], system="Hello world") > 0

    def test_counts_message_tokens(self, client: LLMClient):
        count = client.count_tokens([{"role": "user", "content": "Hello world"}])
        assert count > 0

    def test_longer_content_has_more_tokens(self, client: LLMClient):
        short = client.count_tokens([{"role": "user", "content": "Hi"}])
        long = client.count_tokens([{"role": "user", "content": "Hi " * 100}])
        assert long > short

    def test_counts_list_content_blocks(self, client: LLMClient):
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": "Hello world"}],
        }]
        assert client.count_tokens(messages) > 0


# ---------------------------------------------------------------------------
# Pricing table
# ---------------------------------------------------------------------------

class TestPricing:
    def test_all_known_models_have_pricing(self):
        for model in ["claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5"]:
            assert model in _MODEL_PRICING

    def test_opus_more_expensive_than_sonnet(self):
        assert _MODEL_PRICING["claude-opus-4-6"]["input"] > _MODEL_PRICING["claude-sonnet-4-6"]["input"]

    def test_sonnet_more_expensive_than_haiku(self):
        assert _MODEL_PRICING["claude-sonnet-4-6"]["input"] > _MODEL_PRICING["claude-haiku-4-5"]["input"]


# ---------------------------------------------------------------------------
# Integration tests — require ANTHROPIC_API_KEY, skipped by default
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "The city name."}
            },
            "required": ["city"],
        },
    }
]


@pytest.mark.integration
class TestIntegration:
    """Real API calls. Requires ANTHROPIC_API_KEY environment variable."""

    @pytest.fixture(scope="class")
    def real_client(self) -> LLMClient:
        return LLMClient(model="claude-haiku-4-5")  # cheapest model for tests

    def test_simple_text_response(self, real_client: LLMClient):
        response = real_client.complete(
            messages=[{"role": "user", "content": "Say exactly: hello"}],
            max_tokens=20,
        )
        assert response.stop_reason in ("end_turn", "max_tokens")
        text_blocks = [b for b in response.content if b.type == "text"]
        assert len(text_blocks) > 0

    def test_tool_use_round_trip(self, real_client: LLMClient):
        """
        Full tool-use round trip:
          1. Send message + tool definitions → get tool_use response
          2. Send tool_result back → get final text response
        """
        messages = [
            {"role": "user", "content": "What's the weather in Paris? Use the tool."}
        ]

        # Step 1: get tool_use
        response1 = real_client.complete(
            messages=messages,
            tools=TOOL_DEFINITIONS,
            max_tokens=256,
        )
        assert response1.stop_reason == "tool_use"
        tool_block = next(b for b in response1.content if b.type == "tool_use")
        assert tool_block.name == "get_weather"
        assert "city" in tool_block.input

        # Step 2: send tool result and get final response
        messages = messages + [
            {"role": "assistant", "content": response1.content},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_block.id,
                        "content": "Sunny, 22°C",
                    }
                ],
            },
        ]
        response2 = real_client.complete(
            messages=messages,
            tools=TOOL_DEFINITIONS,
            max_tokens=256,
        )
        assert response2.stop_reason == "end_turn"
        text = " ".join(b.text for b in response2.content if b.type == "text")
        assert len(text) > 0

    def test_stats_populated_after_calls(self, real_client: LLMClient):
        real_client.complete(
            messages=[{"role": "user", "content": "Say hi"}],
            max_tokens=10,
        )
        stats = real_client.get_stats()
        assert stats["total_calls"] >= 1
        assert stats["total_input_tokens"] > 0
        assert stats["total_output_tokens"] > 0
        assert stats["estimated_cost_usd"] > 0

    def test_count_tokens_reasonable_estimate(self, real_client: LLMClient):
        messages = [{"role": "user", "content": "Hello, how are you today?"}]
        count = real_client.count_tokens(messages)
        # "Hello, how are you today?" is ~7 tokens; tiktoken may be slightly off
        assert 4 <= count <= 15
