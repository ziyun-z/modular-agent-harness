"""
Tests for SingleAgentCommunication, NaiveMemory, and NoCompression.

All tests are fast unit tests with mocked LLM/sandbox — no Docker or API needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Helpers to build fake Anthropic response objects
# ---------------------------------------------------------------------------


def _text_block(text: str):
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _tool_use_block(name: str, tool_input: dict, block_id: str = "toolu_001"):
    block = MagicMock()
    block.type = "tool_use"
    block.name = name
    block.input = tool_input
    block.id = block_id
    return block


def _llm_response(blocks, stop_reason: str = "tool_use"):
    resp = MagicMock()
    resp.content = blocks
    resp.stop_reason = stop_reason
    resp.usage = MagicMock(input_tokens=100, output_tokens=50)
    return resp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def llm():
    client = MagicMock()
    client.count_tokens.return_value = 10
    return client


@pytest.fixture()
def sandbox():
    return MagicMock()


@pytest.fixture()
def tool_executor():
    te = MagicMock()
    te.tool_definitions = [{"name": "bash"}]
    te.execute.return_value = "command output"
    return te


@pytest.fixture()
def comm():
    from src.communication.single_agent import SingleAgentCommunication
    return SingleAgentCommunication()


@pytest.fixture()
def comm_ready(comm, llm, sandbox):
    """A communication module that has been set up and is ready to run."""
    comm.setup("Fix the bug in foo.py", llm, sandbox=sandbox)
    return comm


# ===========================================================================
# SingleAgentCommunication — setup
# ===========================================================================


class TestSetup:
    def test_initial_message_is_task_description(self, comm, llm, sandbox):
        from src.communication.single_agent import SingleAgentCommunication
        comm.setup("Fix the bug", llm, sandbox=sandbox)
        assert comm._messages[0] == {"role": "user", "content": "Fix the bug"}

    def test_setup_resets_step_counter(self, comm, llm, sandbox):
        comm.setup("task", llm, sandbox=sandbox)
        comm._step = 5
        comm.setup("new task", llm, sandbox=sandbox)
        assert comm._step == 0

    def test_setup_resets_turns(self, comm, llm, sandbox):
        from src.compression.base import ConversationTurn
        comm.setup("task", llm, sandbox=sandbox)
        comm._turns.append(
            ConversationTurn(role="assistant", content="hi", step=1,
                             is_landmark=False, token_count=5)
        )
        comm.setup("new task", llm, sandbox=sandbox)
        assert comm._turns == []

    def test_set_sandbox(self, comm, llm):
        comm.setup("task", llm)
        sb = MagicMock()
        comm.set_sandbox(sb)
        assert comm._sandbox is sb

    def test_run_step_raises_without_sandbox(self, comm, llm, tool_executor):
        comm.setup("task", llm)          # no sandbox
        with pytest.raises(RuntimeError, match="Sandbox not set"):
            comm.run_step(llm, tool_executor)


# ===========================================================================
# SingleAgentCommunication — run_step with text-only response
# ===========================================================================


class TestRunStepTextOnly:
    def test_returns_done_false_for_text_response(self, comm_ready, llm, tool_executor):
        llm.complete.return_value = _llm_response(
            [_text_block("I see the issue.")], stop_reason="end_turn"
        )
        result = comm_ready.run_step(llm, tool_executor)
        assert result["done"] is False

    def test_no_tool_calls_made(self, comm_ready, llm, tool_executor):
        llm.complete.return_value = _llm_response(
            [_text_block("thinking...")], stop_reason="end_turn"
        )
        comm_ready.run_step(llm, tool_executor)
        tool_executor.execute.assert_not_called()

    def test_llm_calls_count_is_one(self, comm_ready, llm, tool_executor):
        llm.complete.return_value = _llm_response(
            [_text_block("ok")], stop_reason="end_turn"
        )
        result = comm_ready.run_step(llm, tool_executor)
        assert result["llm_calls"] == 1

    def test_actions_taken_is_empty(self, comm_ready, llm, tool_executor):
        llm.complete.return_value = _llm_response(
            [_text_block("thinking")], stop_reason="end_turn"
        )
        result = comm_ready.run_step(llm, tool_executor)
        assert result["actions_taken"] == []

    def test_assistant_message_appended(self, comm_ready, llm, tool_executor):
        blocks = [_text_block("hello")]
        llm.complete.return_value = _llm_response(blocks, stop_reason="end_turn")
        before = len(comm_ready._messages)
        comm_ready.run_step(llm, tool_executor)
        assert len(comm_ready._messages) == before + 1
        assert comm_ready._messages[-1]["role"] == "assistant"


# ===========================================================================
# SingleAgentCommunication — run_step with tool use
# ===========================================================================


class TestRunStepToolUse:
    def test_tool_executed(self, comm_ready, llm, tool_executor, sandbox):
        llm.complete.return_value = _llm_response([
            _tool_use_block("bash", {"command": "ls"})
        ])
        comm_ready.run_step(llm, tool_executor)
        tool_executor.execute.assert_called_once_with(
            "bash", {"command": "ls"}, comm_ready._sandbox
        )

    def test_tool_result_appended_as_user_message(self, comm_ready, llm, tool_executor):
        tool_executor.execute.return_value = "file1.py\nfile2.py"
        llm.complete.return_value = _llm_response([
            _tool_use_block("bash", {"command": "ls"})
        ])
        comm_ready.run_step(llm, tool_executor)
        last = comm_ready._messages[-1]
        assert last["role"] == "user"
        assert last["content"][0]["type"] == "tool_result"
        assert "file1.py" in last["content"][0]["content"]

    def test_action_recorded(self, comm_ready, llm, tool_executor):
        tool_executor.execute.return_value = "result"
        llm.complete.return_value = _llm_response([
            _tool_use_block("read_file", {"path": "foo.py"})
        ])
        result = comm_ready.run_step(llm, tool_executor)
        assert len(result["actions_taken"]) == 1
        assert result["actions_taken"][0]["tool"] == "read_file"

    def test_multiple_tools_in_one_step(self, comm_ready, llm, tool_executor):
        tool_executor.execute.return_value = "output"
        llm.complete.return_value = _llm_response([
            _tool_use_block("bash", {"command": "ls"}, "id_1"),
            _tool_use_block("read_file", {"path": "f.py"}, "id_2"),
        ])
        result = comm_ready.run_step(llm, tool_executor)
        assert len(result["actions_taken"]) == 2
        assert tool_executor.execute.call_count == 2

    def test_tool_results_use_matching_ids(self, comm_ready, llm, tool_executor):
        tool_executor.execute.return_value = "output"
        llm.complete.return_value = _llm_response([
            _tool_use_block("bash", {"command": "ls"}, "toolu_abc")
        ])
        comm_ready.run_step(llm, tool_executor)
        user_msg = comm_ready._messages[-1]
        assert user_msg["content"][0]["tool_use_id"] == "toolu_abc"


# ===========================================================================
# SingleAgentCommunication — submit_patch / done signal
# ===========================================================================


class TestSubmitPatch:
    def test_done_true_when_patch_submitted(self, comm_ready, llm, tool_executor):
        from src.tool_executor import PatchSubmitted
        tool_executor.execute.side_effect = PatchSubmitted("fixed it")
        llm.complete.return_value = _llm_response([
            _tool_use_block("submit_patch", {"message": "fixed it"})
        ])
        result = comm_ready.run_step(llm, tool_executor)
        assert result["done"] is True

    def test_result_carries_patch_message(self, comm_ready, llm, tool_executor):
        from src.tool_executor import PatchSubmitted
        tool_executor.execute.side_effect = PatchSubmitted("fixed the off-by-one")
        llm.complete.return_value = _llm_response([
            _tool_use_block("submit_patch", {"message": "fixed"})
        ])
        result = comm_ready.run_step(llm, tool_executor)
        assert "fixed the off-by-one" in result["result"]

    def test_no_tools_called_after_submit(self, comm_ready, llm, tool_executor):
        """Tools after submit_patch in the same response should be skipped."""
        from src.tool_executor import PatchSubmitted

        call_count = 0

        def side_effect(name, inp, sandbox):
            nonlocal call_count
            call_count += 1
            if name == "submit_patch":
                raise PatchSubmitted("done")
            return "output"

        tool_executor.execute.side_effect = side_effect
        llm.complete.return_value = _llm_response([
            _tool_use_block("submit_patch", {}, "id_1"),
            _tool_use_block("bash", {"command": "ls"}, "id_2"),
        ])
        comm_ready.run_step(llm, tool_executor)
        assert call_count == 1   # only submit_patch, bash skipped


# ===========================================================================
# SingleAgentCommunication — trajectory / compression interface
# ===========================================================================


class TestTrajectory:
    def test_get_trajectory_empty_before_any_step(self, comm_ready):
        assert comm_ready.get_trajectory() == []

    def test_trajectory_grows_after_step(self, comm_ready, llm, tool_executor):
        llm.complete.return_value = _llm_response(
            [_text_block("hi")], stop_reason="end_turn"
        )
        comm_ready.run_step(llm, tool_executor)
        assert len(comm_ready.get_trajectory()) > 0

    def test_update_trajectory_replaces_turns(self, comm_ready, llm, tool_executor):
        from src.compression.base import ConversationTurn
        llm.complete.return_value = _llm_response(
            [_text_block("step1")], stop_reason="end_turn"
        )
        comm_ready.run_step(llm, tool_executor)

        compressed = [
            ConversationTurn(
                role="assistant",
                content="[summary of step 1]",
                step=1,
                is_landmark=False,
                token_count=20,
            )
        ]
        comm_ready.update_trajectory(compressed)
        assert comm_ready._turns == compressed

    def test_update_trajectory_rebuilds_messages(self, comm_ready, llm, tool_executor):
        from src.compression.base import ConversationTurn
        compressed = [
            ConversationTurn(
                role="assistant",
                content="[summary]",
                step=1,
                is_landmark=False,
                token_count=10,
            )
        ]
        comm_ready.update_trajectory(compressed)
        # Must still have the original task message + rebuilt messages
        assert comm_ready._messages[0]["role"] == "user"
        assert comm_ready._messages[0]["content"] == "Fix the bug in foo.py"


# ===========================================================================
# SingleAgentCommunication — memory injection
# ===========================================================================


class TestMemoryInjection:
    def test_memory_context_appended_to_system(self, llm, sandbox, tool_executor):
        from src.communication.single_agent import SingleAgentCommunication

        memory = MagicMock()
        memory.get_context_block.return_value = "## Prior observations\n- foo is broken"

        comm = SingleAgentCommunication(memory=memory)
        comm.setup("Fix foo", llm, sandbox=sandbox)

        llm.complete.return_value = _llm_response(
            [_text_block("ok")], stop_reason="end_turn"
        )
        comm.run_step(llm, tool_executor)

        call_kwargs = llm.complete.call_args[1]
        system = call_kwargs.get("system") or llm.complete.call_args[0][2]
        assert "Prior observations" in system

    def test_no_memory_uses_base_prompt_only(self, comm_ready, llm, tool_executor):
        llm.complete.return_value = _llm_response(
            [_text_block("ok")], stop_reason="end_turn"
        )
        comm_ready.run_step(llm, tool_executor)

        call_kwargs = llm.complete.call_args[1]
        system = call_kwargs.get("system", "")
        assert "software engineer" in system

    def test_empty_memory_block_not_injected(self, llm, sandbox, tool_executor):
        from src.communication.single_agent import SingleAgentCommunication

        memory = MagicMock()
        memory.get_context_block.return_value = ""

        comm = SingleAgentCommunication(memory=memory)
        comm.setup("Fix foo", llm, sandbox=sandbox)

        llm.complete.return_value = _llm_response(
            [_text_block("ok")], stop_reason="end_turn"
        )
        comm.run_step(llm, tool_executor)

        call_kwargs = llm.complete.call_args[1]
        system = call_kwargs.get("system", "")
        assert "Memory Context" not in system


# ===========================================================================
# NaiveMemory
# ===========================================================================


class TestNaiveMemory:
    @pytest.fixture()
    def memory(self):
        from src.memory.naive import NaiveMemory
        return NaiveMemory()

    def test_store_does_not_raise(self, memory):
        from src.memory.base import MemoryEntry
        import time
        entry = MemoryEntry(
            step=1, entry_type="observation",
            content="foo", metadata={}, timestamp=time.time()
        )
        memory.store(entry)  # should not raise

    def test_retrieve_returns_empty(self, memory):
        assert memory.retrieve("query", max_tokens=1000) == []

    def test_get_context_block_returns_empty_string(self, memory):
        assert memory.get_context_block(max_tokens=1000) == ""

    def test_clear_does_not_raise(self, memory):
        memory.clear()

    def test_get_stats_returns_dict(self, memory):
        stats = memory.get_stats()
        assert isinstance(stats, dict)


# ===========================================================================
# NoCompression
# ===========================================================================


class TestNoCompression:
    @pytest.fixture()
    def compression(self):
        from src.compression.none import NoCompression
        return NoCompression()

    def _make_turn(self, step=1, tokens=100, landmark=False) -> "ConversationTurn":
        from src.compression.base import ConversationTurn
        return ConversationTurn(
            role="assistant",
            content=f"content at step {step}",
            step=step,
            is_landmark=landmark,
            token_count=tokens,
        )

    def test_should_compress_always_false(self, compression):
        turns = [self._make_turn(tokens=50_000)]
        assert compression.should_compress(turns, max_tokens=1000) is False

    def test_compress_returns_list(self, compression):
        turns = [self._make_turn()]
        result = compression.compress(turns, target_tokens=500, llm_client=MagicMock())
        assert isinstance(result, list)

    def test_compress_drops_oldest_non_landmark(self, compression):
        turns = [
            self._make_turn(step=1, tokens=200, landmark=False),
            self._make_turn(step=2, tokens=200, landmark=False),
            self._make_turn(step=3, tokens=200, landmark=False),
        ]
        # target=400 → need to drop 1 turn (600 total → 400)
        result = compression.compress(turns, target_tokens=400, llm_client=MagicMock())
        assert len(result) == 2
        # Oldest (step 1) should be dropped first
        assert result[0].step == 2

    def test_compress_preserves_landmarks(self, compression):
        turns = [
            self._make_turn(step=1, tokens=500, landmark=True),   # landmark, keep
            self._make_turn(step=2, tokens=500, landmark=False),   # non-landmark
        ]
        # target=600 but total=1000; only step=2 can be dropped
        result = compression.compress(turns, target_tokens=600, llm_client=MagicMock())
        assert any(t.step == 1 for t in result)  # landmark preserved

    def test_compress_does_not_drop_all_landmarks(self, compression):
        turns = [
            self._make_turn(step=1, tokens=500, landmark=True),
            self._make_turn(step=2, tokens=500, landmark=True),
        ]
        # Can't drop landmarks — should return as-is even if over budget
        result = compression.compress(turns, target_tokens=100, llm_client=MagicMock())
        assert len(result) == 2

    def test_compress_makes_no_llm_call(self, compression):
        llm = MagicMock()
        turns = [self._make_turn(tokens=9999)]
        compression.compress(turns, target_tokens=100, llm_client=llm)
        llm.complete.assert_not_called()

    def test_get_stats_returns_dict(self, compression):
        assert isinstance(compression.get_stats(), dict)
