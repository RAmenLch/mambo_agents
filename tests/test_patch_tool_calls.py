"""Tests for PatchToolCallsMiddleware."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Overwrite

from mambo_agents.middleware.patch_tool_calls import PatchToolCallsMiddleware


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EMPTY_RUNTIME = Runtime(
    context=None,
    store=None,
    stream_writer=lambda v: None,
    previous=None,
    execution_info=None,
    server_info=None,
)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestPatchToolCallsMiddleware:
    """Unit tests for PatchToolCallsMiddleware.before_agent."""

    def test_no_messages_returns_none(self):
        """Empty message list → no patching needed."""
        mw = PatchToolCallsMiddleware()
        state = {"messages": []}
        result = mw.before_agent(state, _EMPTY_RUNTIME)
        assert result is None

    def test_no_tool_calls_returns_none(self):
        """Only human messages → no patching needed."""
        mw = PatchToolCallsMiddleware()
        state = {"messages": [HumanMessage(content="hello")]}
        result = mw.before_agent(state, _EMPTY_RUNTIME)
        assert result is None

    def test_all_tool_calls_answered_returns_none(self):
        """Every AIMessage tool_call has a matching ToolMessage → no patching."""
        mw = PatchToolCallsMiddleware()
        state = {
            "messages": [
                HumanMessage(content="read /file.txt"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "read", "args": {"path": "/file.txt"}, "id": "call_1"}
                    ],
                ),
                ToolMessage(content="hello", name="read", tool_call_id="call_1"),
                AIMessage(content="Done!"),
            ]
        }
        result = mw.before_agent(state, _EMPTY_RUNTIME)
        assert result is None

    def test_dangling_tool_call_is_patched(self):
        """AIMessage with unanswered tool_call → synthetic ToolMessage injected."""
        mw = PatchToolCallsMiddleware()
        state = {
            "messages": [
                HumanMessage(content="do stuff"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "write", "args": {"path": "/f.txt", "content": "x"}, "id": "call_2"}
                    ],
                ),
                # No ToolMessage for call_2 → dangling!
            ]
        }
        result = mw.before_agent(state, _EMPTY_RUNTIME)
        assert result is not None
        assert "messages" in result
        assert isinstance(result["messages"], Overwrite)

        patched = result["messages"].value
        # Should have original HumanMessage + AIMessage + injected ToolMessage
        assert len(patched) == 3
        injected = patched[-1]
        assert isinstance(injected, ToolMessage)
        assert injected.tool_call_id == "call_2"
        assert "cancelled" in injected.content
        assert injected.name == "write"

    def test_mixed_answered_and_dangling(self):
        """Only unanswered tool_calls get patched; answered ones are left alone."""
        mw = PatchToolCallsMiddleware()
        state = {
            "messages": [
                HumanMessage(content="task"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "read", "args": {"path": "/a.txt"}, "id": "call_a"},
                        {"name": "write", "args": {"path": "/b.txt", "content": "b"}, "id": "call_b"},
                    ],
                ),
                ToolMessage(content="content of a", name="read", tool_call_id="call_a"),
                # call_b has NO ToolMessage → dangling
            ]
        }
        result = mw.before_agent(state, _EMPTY_RUNTIME)
        assert result is not None

        patched = result["messages"].value
        # Order: Human, AI, *injectedTool(call_b)*, Tool(call_a)
        assert len(patched) == 4
        # The injected ToolMessage for call_b (inserted right after AIMessage)
        injected = patched[2]
        assert isinstance(injected, ToolMessage)
        assert injected.tool_call_id == "call_b"
        assert "cancelled" in injected.content
        # The original ToolMessage for call_a still preserved
        existing_tool = patched[3]
        assert isinstance(existing_tool, ToolMessage)
        assert existing_tool.tool_call_id == "call_a"
        assert "cancelled" not in existing_tool.content

    def test_multiple_aimessages_mixed(self):
        """Two AIMessages, each with one dangling call → both patched."""
        mw = PatchToolCallsMiddleware()
        state = {
            "messages": [
                HumanMessage(content="step 1"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "read", "args": {"path": "/1.txt"}, "id": "call_1"}
                    ],
                ),
                # No ToolMessage for call_1
                HumanMessage(content="step 2 (interrupted)"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "write", "args": {"path": "/2.txt", "content": "2"}, "id": "call_2"}
                    ],
                ),
                # No ToolMessage for call_2
            ]
        }
        result = mw.before_agent(state, _EMPTY_RUNTIME)
        assert result is not None

        patched = result["messages"].value
        # Original: Human, AI(call_1), Human, AI(call_2)
        # Patched:  Human, AI(call_1), Tool(call_1 cancelled), Human, AI(call_2), Tool(call_2 cancelled)
        assert len(patched) == 6

        tool_messages = [m for m in patched if isinstance(m, ToolMessage)]
        assert len(tool_messages) == 2
        assert {t.tool_call_id for t in tool_messages} == {"call_1", "call_2"}
        expected_names = {"call_1": "read", "call_2": "write"}
        for tm in tool_messages:
            assert "cancelled" in tm.content
            assert tm.name == expected_names[tm.tool_call_id]

    def test_preserves_non_ai_tool_messages(self):
        """Existing ToolMessages and HumanMessages are preserved unchanged."""
        mw = PatchToolCallsMiddleware()
        state = {
            "messages": [
                HumanMessage(content="query"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "read", "args": {"path": "/x.txt"}, "id": "call_x"}
                    ],
                ),
                ToolMessage(content="file content", name="read", tool_call_id="call_x"),
                HumanMessage(content="thanks"),
            ]
        }
        result = mw.before_agent(state, _EMPTY_RUNTIME)
        assert result is None  # No dangling → no change

        # But let's verify the messages are exactly the same
        # (overwrite would have replaced them, but since no dangling we got None)
        assert state["messages"][0] == HumanMessage(content="query")
        assert state["messages"][3] == HumanMessage(content="thanks")
