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


# ===========================================================================
# OrchestratedCommunication
# ===========================================================================


def _text_block_plain(text: str):
    """A simple text block mock (plain attribute access)."""
    b = MagicMock()
    b.type = "text"
    b.text = text
    return b


def _tool_use_block_plain(name: str, inp: dict, block_id: str = "toolu_001"):
    b = MagicMock()
    b.type = "tool_use"
    b.name = name
    b.input = inp
    b.id = block_id
    return b


def _llm_response_plain(blocks, stop_reason: str = "end_turn"):
    resp = MagicMock()
    resp.content = blocks
    resp.stop_reason = stop_reason
    resp.usage = MagicMock(input_tokens=50, output_tokens=20)
    return resp


@pytest.fixture()
def orch_comm():
    from src.communication.orchestrated import OrchestratedCommunication
    return OrchestratedCommunication(specialist_max_steps=3, specialist_summary_max_tokens=200)


@pytest.fixture()
def orch_sandbox():
    return MagicMock()


@pytest.fixture()
def orch_tool_executor():
    from src.tool_executor import TOOL_DEFINITIONS
    te = MagicMock()
    te.tool_definitions = TOOL_DEFINITIONS  # real definitions so filtering works
    te.execute.return_value = "tool output"
    return te


@pytest.fixture()
def orch_llm():
    llm = MagicMock()
    llm.count_tokens.return_value = 10
    return llm


@pytest.fixture()
def orch_ready(orch_comm, orch_llm, orch_sandbox):
    orch_comm.setup("Fix the null-pointer bug in parser.py", orch_llm)
    orch_comm.set_sandbox(orch_sandbox)
    return orch_comm


class TestOrchestratedSetup:
    def test_setup_puts_task_in_messages(self, orch_comm, orch_llm):
        orch_comm.setup("My task", orch_llm)
        assert orch_comm._planner_messages[0] == {
            "role": "user", "content": "My task"
        }

    def test_setup_resets_step_counter(self, orch_comm, orch_llm):
        orch_comm.setup("task", orch_llm)
        orch_comm._step = 99
        orch_comm.setup("new task", orch_llm)
        assert orch_comm._step == 0

    def test_setup_clears_turns(self, orch_comm, orch_llm):
        from src.compression.base import ConversationTurn
        orch_comm.setup("task", orch_llm)
        orch_comm._planner_turns.append(
            ConversationTurn("assistant", "hi", 1, False, 5)
        )
        orch_comm.setup("new task", orch_llm)
        assert orch_comm._planner_turns == []

    def test_setup_resets_delegation_stats(self, orch_comm, orch_llm):
        orch_comm.setup("task", orch_llm)
        orch_comm._delegations = 5
        orch_comm.setup("new task", orch_llm)
        assert orch_comm._delegations == 0

    def test_set_sandbox(self, orch_comm, orch_llm, orch_sandbox):
        orch_comm.setup("task", orch_llm)
        orch_comm.set_sandbox(orch_sandbox)
        assert orch_comm._sandbox is orch_sandbox

    def test_run_step_raises_without_sandbox(self, orch_comm, orch_llm, orch_tool_executor):
        orch_comm.setup("task", orch_llm)  # no sandbox
        with pytest.raises(RuntimeError, match="Sandbox not set"):
            orch_comm.run_step(orch_llm, orch_tool_executor)


class TestOrchestratedPlannerTextOnly:
    """Planner responds with pure text (no tool use)."""

    def test_done_is_false(self, orch_ready, orch_llm, orch_tool_executor):
        orch_llm.complete.return_value = _llm_response_plain(
            [_text_block_plain("I am thinking...")], stop_reason="end_turn"
        )
        result = orch_ready.run_step(orch_llm, orch_tool_executor)
        assert result["done"] is False

    def test_actions_empty(self, orch_ready, orch_llm, orch_tool_executor):
        orch_llm.complete.return_value = _llm_response_plain(
            [_text_block_plain("Hmm.")], stop_reason="end_turn"
        )
        result = orch_ready.run_step(orch_llm, orch_tool_executor)
        assert result["actions_taken"] == []

    def test_llm_calls_is_one(self, orch_ready, orch_llm, orch_tool_executor):
        orch_llm.complete.return_value = _llm_response_plain(
            [_text_block_plain("ok")], stop_reason="end_turn"
        )
        result = orch_ready.run_step(orch_llm, orch_tool_executor)
        assert result["llm_calls"] == 1

    def test_assistant_message_appended(self, orch_ready, orch_llm, orch_tool_executor):
        orch_llm.complete.return_value = _llm_response_plain(
            [_text_block_plain("thinking")], stop_reason="end_turn"
        )
        before = len(orch_ready._planner_messages)
        orch_ready.run_step(orch_llm, orch_tool_executor)
        assert len(orch_ready._planner_messages) == before + 1
        assert orch_ready._planner_messages[-1]["role"] == "assistant"

    def test_turn_recorded(self, orch_ready, orch_llm, orch_tool_executor):
        orch_llm.complete.return_value = _llm_response_plain(
            [_text_block_plain("reasoning")], stop_reason="end_turn"
        )
        orch_ready.run_step(orch_llm, orch_tool_executor)
        assert len(orch_ready._planner_turns) > 0


class TestOrchestratedDelegation:
    """Planner calls delegate_task; specialist runs and returns a summary."""

    def _make_specialist_response(self, text: str = "Found the bug in foo.py."):
        """A specialist response: one text block, stop_reason=end_turn."""
        return _llm_response_plain(
            [_text_block_plain(text)], stop_reason="end_turn"
        )

    def test_delegation_increments_count(self, orch_ready, orch_llm, orch_tool_executor):
        # First call = planner with delegate_task
        planner_resp = _llm_response_plain(
            [_tool_use_block_plain("delegate_task", {"task": "explore the repo"}, "id1")],
            stop_reason="tool_use",
        )
        # Second call = specialist (text-only, done)
        specialist_resp = self._make_specialist_response("Explored. Found bug at line 42.")

        orch_llm.complete.side_effect = [planner_resp, specialist_resp]
        orch_ready.run_step(orch_llm, orch_tool_executor)

        assert orch_ready._delegations == 1

    def test_delegation_result_injected_as_tool_result(
        self, orch_ready, orch_llm, orch_tool_executor
    ):
        planner_resp = _llm_response_plain(
            [_tool_use_block_plain("delegate_task", {"task": "explore"}, "id1")],
            stop_reason="tool_use",
        )
        specialist_resp = self._make_specialist_response("Found foo.py line 42.")
        orch_llm.complete.side_effect = [planner_resp, specialist_resp]

        orch_ready.run_step(orch_llm, orch_tool_executor)

        # Planner's messages should now have a user message with tool_result content
        user_msgs = [m for m in orch_ready._planner_messages if m["role"] == "user"]
        last_user = user_msgs[-1]
        assert isinstance(last_user["content"], list)
        assert last_user["content"][0]["type"] == "tool_result"
        assert "foo.py" in last_user["content"][0]["content"]

    def test_specialist_actions_in_actions_taken(
        self, orch_ready, orch_llm, orch_tool_executor
    ):
        planner_resp = _llm_response_plain(
            [_tool_use_block_plain("delegate_task", {"task": "read foo.py"}, "id1")],
            stop_reason="tool_use",
        )
        # Specialist makes one tool call then ends
        tool_block = _tool_use_block_plain("read_file", {"path": "foo.py"}, "spec_id1")
        spec_resp_1 = _llm_response_plain([tool_block], stop_reason="tool_use")
        spec_resp_2 = self._make_specialist_response("Read foo.py, saw the bug.")
        orch_llm.complete.side_effect = [planner_resp, spec_resp_1, spec_resp_2]

        result = orch_ready.run_step(orch_llm, orch_tool_executor)

        # Should have delegate_task action + read_file action
        tool_names = [a["tool"] for a in result["actions_taken"]]
        assert "delegate_task" in tool_names
        assert "read_file" in tool_names

    def test_llm_calls_includes_specialist_calls(
        self, orch_ready, orch_llm, orch_tool_executor
    ):
        planner_resp = _llm_response_plain(
            [_tool_use_block_plain("delegate_task", {"task": "run tests"}, "id1")],
            stop_reason="tool_use",
        )
        # Specialist makes 2 LLM calls
        spec_resp_1 = _llm_response_plain(
            [_tool_use_block_plain("bash", {"command": "pytest"}, "s1")],
            stop_reason="tool_use",
        )
        spec_resp_2 = self._make_specialist_response("Tests pass.")
        orch_llm.complete.side_effect = [planner_resp, spec_resp_1, spec_resp_2]

        result = orch_ready.run_step(orch_llm, orch_tool_executor)
        # 1 planner call + 2 specialist calls
        assert result["llm_calls"] == 3

    def test_specialist_tools_exclude_submit_and_delegate(
        self, orch_ready, orch_llm, orch_tool_executor
    ):
        """Specialist should not receive submit_patch or delegate_task."""
        captured_tools: list = []

        def capture_complete(messages, tools=None, system=None, **kw):
            if tools:
                captured_tools.extend(tools)
            # First call = planner (has delegate_task)
            if any(t.get("name") == "delegate_task" for t in (tools or [])):
                return _llm_response_plain(
                    [_tool_use_block_plain("delegate_task", {"task": "explore"}, "id1")],
                    stop_reason="tool_use",
                )
            # Subsequent calls = specialist
            return self._make_specialist_response("Done.")

        orch_llm.complete.side_effect = capture_complete

        orch_ready.run_step(orch_llm, orch_tool_executor)

        # Specialist tools are those passed to calls WITHOUT delegate_task
        specialist_tool_sets = [
            set(t.get("name") for t in call_tools)
            for call_tools in [
                # rebuild per-call sets from captured list
                # simpler: just check the specialist call directly via call_args_list
            ]
        ]
        # Use call_args_list to inspect the second call (specialist)
        calls = orch_llm.complete.call_args_list
        assert len(calls) >= 2
        specialist_call_tools = calls[1][1].get("tools") or calls[1][0][1]
        specialist_tool_names = {t["name"] for t in specialist_call_tools}
        assert "submit_patch" not in specialist_tool_names
        assert "delegate_task" not in specialist_tool_names

    def test_multiple_delegations_in_one_step(
        self, orch_ready, orch_llm, orch_tool_executor
    ):
        """Planner can issue multiple delegate_task calls in one response."""
        planner_resp = _llm_response_plain(
            [
                _tool_use_block_plain("delegate_task", {"task": "explore"}, "id1"),
                _tool_use_block_plain("delegate_task", {"task": "edit"}, "id2"),
            ],
            stop_reason="tool_use",
        )
        spec_resp = self._make_specialist_response("Done.")
        orch_llm.complete.side_effect = [planner_resp, spec_resp, spec_resp]

        orch_ready.run_step(orch_llm, orch_tool_executor)
        assert orch_ready._delegations == 2


class TestOrchestratedSubmitPatch:
    def test_done_true_on_submit_patch(self, orch_ready, orch_llm, orch_tool_executor):
        from src.tool_executor import PatchSubmitted
        orch_tool_executor.execute.side_effect = PatchSubmitted("fixed it")

        planner_resp = _llm_response_plain(
            [_tool_use_block_plain("submit_patch", {"message": "fixed"}, "id1")],
            stop_reason="tool_use",
        )
        orch_llm.complete.return_value = planner_resp
        result = orch_ready.run_step(orch_llm, orch_tool_executor)

        assert result["done"] is True

    def test_result_contains_patch_message(self, orch_ready, orch_llm, orch_tool_executor):
        from src.tool_executor import PatchSubmitted
        orch_tool_executor.execute.side_effect = PatchSubmitted("null check added")

        planner_resp = _llm_response_plain(
            [_tool_use_block_plain("submit_patch", {}, "id1")],
            stop_reason="tool_use",
        )
        orch_llm.complete.return_value = planner_resp
        result = orch_ready.run_step(orch_llm, orch_tool_executor)

        assert "null check added" in result["result"]

    def test_submit_patch_landmark_in_turn(
        self, orch_ready, orch_llm, orch_tool_executor
    ):
        from src.tool_executor import PatchSubmitted
        orch_tool_executor.execute.side_effect = PatchSubmitted("done")

        planner_resp = _llm_response_plain(
            [_tool_use_block_plain("submit_patch", {}, "id1")],
            stop_reason="tool_use",
        )
        orch_llm.complete.return_value = planner_resp
        orch_ready.run_step(orch_llm, orch_tool_executor)

        landmark_turns = [t for t in orch_ready._planner_turns if t.is_landmark]
        assert len(landmark_turns) >= 1


class TestOrchestratedSpecialistMaxSteps:
    def test_specialist_stops_at_max_steps(self, orch_llm, orch_sandbox, orch_tool_executor):
        from src.communication.orchestrated import OrchestratedCommunication
        comm = OrchestratedCommunication(specialist_max_steps=2, specialist_summary_max_tokens=200)
        comm.setup("task", orch_llm)
        comm.set_sandbox(orch_sandbox)

        # Specialist always calls a tool — forces max_steps
        def always_tool_call(messages, tools=None, system=None, **kw):
            if tools and any(t.get("name") == "delegate_task" for t in tools):
                # Planner delegates
                return _llm_response_plain(
                    [_tool_use_block_plain("delegate_task", {"task": "explore"}, "p1")],
                    stop_reason="tool_use",
                )
            # Specialist always issues tool_use (never end_turn)
            return _llm_response_plain(
                [_tool_use_block_plain("bash", {"command": "ls"}, "s1")],
                stop_reason="tool_use",
            )

        orch_llm.complete.side_effect = always_tool_call
        comm.run_step(orch_llm, orch_tool_executor)

        # specialist_max_steps=2; plus possibly 1 synthesis call
        # planner call + 2 specialist calls + 1 synthesis = 4 total
        total_calls = orch_llm.complete.call_count
        assert total_calls <= 4  # bounded by max_steps + synthesis

    def test_specialist_steps_tracked_in_stats(
        self, orch_ready, orch_llm, orch_tool_executor
    ):
        planner_resp = _llm_response_plain(
            [_tool_use_block_plain("delegate_task", {"task": "explore"}, "id1")],
            stop_reason="tool_use",
        )
        spec_tool = _llm_response_plain(
            [_tool_use_block_plain("bash", {"command": "ls"}, "s1")],
            stop_reason="tool_use",
        )
        spec_end = _llm_response_plain(
            [_text_block_plain("Done. Found bug at line 10.")], stop_reason="end_turn"
        )
        orch_llm.complete.side_effect = [planner_resp, spec_tool, spec_end]

        orch_ready.run_step(orch_llm, orch_tool_executor)
        stats = orch_ready.get_stats()
        assert stats["specialist_steps_total"] == 2


class TestOrchestratedSpecialistSynthesis:
    """Specialist produces no text → synthesis call is made."""

    def test_synthesis_called_when_no_text(
        self, orch_ready, orch_llm, orch_tool_executor
    ):
        planner_resp = _llm_response_plain(
            [_tool_use_block_plain("delegate_task", {"task": "run tests"}, "id1")],
            stop_reason="tool_use",
        )
        # Specialist ends with a pure tool-call block (no text)
        tool_only = MagicMock()
        tool_only.type = "tool_use"
        tool_only.name = "bash"
        tool_only.input = {"command": "pytest"}
        tool_only.id = "s1"
        spec_resp = _llm_response_plain([tool_only], stop_reason="end_turn")

        # Synthesis response
        synthesis_resp = _llm_response_plain(
            [_text_block_plain("Tests passed after running pytest.")],
            stop_reason="end_turn",
        )
        orch_llm.complete.side_effect = [planner_resp, spec_resp, synthesis_resp]

        result = orch_ready.run_step(orch_llm, orch_tool_executor)

        # The delegation result should contain the synthesis text
        assert "Tests passed" in result["actions_taken"][0]["output"]

    def test_synthesis_fallback_on_empty_response(
        self, orch_ready, orch_llm, orch_tool_executor
    ):
        planner_resp = _llm_response_plain(
            [_tool_use_block_plain("delegate_task", {"task": "explore"}, "id1")],
            stop_reason="tool_use",
        )
        # Specialist ends immediately with empty content
        spec_resp = MagicMock()
        spec_resp.content = []
        spec_resp.stop_reason = "end_turn"

        # Synthesis also returns empty
        synthesis_resp = MagicMock()
        synthesis_resp.content = []
        synthesis_resp.stop_reason = "end_turn"

        orch_llm.complete.side_effect = [planner_resp, spec_resp, synthesis_resp]
        result = orch_ready.run_step(orch_llm, orch_tool_executor)

        assert result["actions_taken"][0]["output"] == "[specialist summary unavailable]"


class TestOrchestratedTrajectory:
    def test_get_trajectory_empty_before_step(self, orch_ready):
        assert orch_ready.get_trajectory() == []

    def test_trajectory_grows_after_step(self, orch_ready, orch_llm, orch_tool_executor):
        orch_llm.complete.return_value = _llm_response_plain(
            [_text_block_plain("thinking")], stop_reason="end_turn"
        )
        orch_ready.run_step(orch_llm, orch_tool_executor)
        assert len(orch_ready.get_trajectory()) > 0

    def test_update_trajectory_replaces_turns(
        self, orch_ready, orch_llm, orch_tool_executor
    ):
        from src.compression.base import ConversationTurn
        orch_llm.complete.return_value = _llm_response_plain(
            [_text_block_plain("step1")], stop_reason="end_turn"
        )
        orch_ready.run_step(orch_llm, orch_tool_executor)

        compressed = [
            ConversationTurn("assistant", "[summary]", 1, False, 10)
        ]
        orch_ready.update_trajectory(compressed)
        assert orch_ready._planner_turns == compressed

    def test_update_trajectory_rebuilds_messages(
        self, orch_ready, orch_llm, orch_tool_executor
    ):
        from src.compression.base import ConversationTurn
        compressed = [
            ConversationTurn("assistant", "Compressed history.", 1, False, 10)
        ]
        orch_ready.update_trajectory(compressed)

        # Task message is always first
        assert orch_ready._planner_messages[0]["role"] == "user"
        assert orch_ready._planner_messages[0]["content"] == (
            "Fix the null-pointer bug in parser.py"
        )


class TestOrchestratedMemoryInjection:
    def test_memory_block_in_planner_system(
        self, orch_ready, orch_llm, orch_tool_executor
    ):
        memory = MagicMock()
        memory.get_context_block.return_value = "## Prior: found bug in query.py"
        orch_ready.set_memory(memory)

        orch_llm.complete.return_value = _llm_response_plain(
            [_text_block_plain("ok")], stop_reason="end_turn"
        )
        orch_ready.run_step(orch_llm, orch_tool_executor)

        call_kwargs = orch_llm.complete.call_args[1]
        system = call_kwargs.get("system", "")
        assert "Prior: found bug in query.py" in system

    def test_empty_memory_not_injected(self, orch_ready, orch_llm, orch_tool_executor):
        memory = MagicMock()
        memory.get_context_block.return_value = ""
        orch_ready.set_memory(memory)

        orch_llm.complete.return_value = _llm_response_plain(
            [_text_block_plain("ok")], stop_reason="end_turn"
        )
        orch_ready.run_step(orch_llm, orch_tool_executor)

        call_kwargs = orch_llm.complete.call_args[1]
        system = call_kwargs.get("system", "")
        assert "Memory Context" not in system


class TestOrchestratedStats:
    def test_initial_stats(self, orch_ready):
        stats = orch_ready.get_stats()
        assert stats["type"] == "orchestrated"
        assert stats["planner_steps"] == 0
        assert stats["delegations"] == 0
        assert stats["specialist_steps_total"] == 0

    def test_stats_after_step(self, orch_ready, orch_llm, orch_tool_executor):
        orch_llm.complete.return_value = _llm_response_plain(
            [_text_block_plain("ok")], stop_reason="end_turn"
        )
        orch_ready.run_step(orch_llm, orch_tool_executor)
        assert orch_ready.get_stats()["planner_steps"] == 1


class TestOrchestratedRunnerRegistry:
    def test_orchestrated_in_registry(self):
        from src.runner import COMMUNICATION_REGISTRY
        from src.communication.orchestrated import OrchestratedCommunication
        assert "orchestrated" in COMMUNICATION_REGISTRY
        assert COMMUNICATION_REGISTRY["orchestrated"] is OrchestratedCommunication

    def test_single_agent_still_in_registry(self):
        from src.runner import COMMUNICATION_REGISTRY
        assert "single_agent" in COMMUNICATION_REGISTRY

    def test_build_orchestrated_module(self):
        from src.runner import build_communication_module
        from src.communication.orchestrated import OrchestratedCommunication
        cfg = {
            "communication": {
                "type": "orchestrated",
                "params": {"specialist_max_steps": 5, "specialist_summary_max_tokens": 500},
            }
        }
        module = build_communication_module(cfg)
        assert isinstance(module, OrchestratedCommunication)
        assert module._specialist_max_steps == 5
        assert module._specialist_summary_max_tokens == 500
