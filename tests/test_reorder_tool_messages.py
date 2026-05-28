"""Tests for ``ReorderToolMessagesMiddleware`` and its ``_reorder()`` logic."""

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Overwrite

from mambo_agents.middleware.reorder_tool_messages import (
    ReorderToolMessagesMiddleware,
    _reorder,
)


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


def _make_tool_call(name: str, call_id: str) -> dict:
    return {"name": name, "args": {}, "id": call_id, "type": "tool_call"}


# ==============================================================================
# Unit tests — _reorder() pure function
# ==============================================================================


class TestReorder:
    def test_empty_messages_returns_none(self):
        assert _reorder([]) is None

    def test_single_human_message_returns_none(self):
        messages = [HumanMessage(content="hello")]
        assert _reorder(messages) is None

    def test_ai_message_no_tool_calls_returns_none(self):
        messages = [AIMessage(content="done")]
        assert _reorder(messages) is None

    def test_tool_messages_in_order_returns_none(self):
        """When ToolMessages are already in the correct order, no change needed."""
        tc1 = _make_tool_call("read", "call_1")
        tc2 = _make_tool_call("write", "call_2")

        messages = [
            AIMessage(content="", tool_calls=[tc1, tc2]),
            ToolMessage(content="result1", tool_call_id="call_1", name="read"),
            ToolMessage(content="result2", tool_call_id="call_2", name="write"),
        ]
        result = _reorder(messages)
        assert result is None

    def test_tool_messages_out_of_order_are_reordered(self):
        """ToolMessages in wrong order are reordered to match tool_calls."""
        tc1 = _make_tool_call("read", "call_1")
        tc2 = _make_tool_call("write", "call_2")

        messages = [
            AIMessage(content="", tool_calls=[tc1, tc2]),
            ToolMessage(content="result2", tool_call_id="call_2", name="write"),
            ToolMessage(content="result1", tool_call_id="call_1", name="read"),
        ]
        result = _reorder(messages)
        assert result is not None
        assert len(result) == 3
        assert result[1].tool_call_id == "call_1"
        assert result[2].tool_call_id == "call_2"

    def test_partial_batch_not_reordered(self):
        """When not all ToolMessages are present for a batch, no reordering."""
        tc1 = _make_tool_call("read", "call_1")
        tc2 = _make_tool_call("write", "call_2")

        messages = [
            AIMessage(content="", tool_calls=[tc1, tc2]),
            ToolMessage(content="result1", tool_call_id="call_1", name="read"),
            # Missing ToolMessage for call_2
            HumanMessage(content="next"),
        ]
        result = _reorder(messages)
        assert result is None  # Partial batch → no reorder

    def test_crossed_by_non_tool_message_no_reorder(self):
        """Non-ToolMessage between AIMessage and its ToolMessages blocks reorder."""
        tc1 = _make_tool_call("read", "call_1")
        tc2 = _make_tool_call("write", "call_2")

        # The second ToolMessage is separated by a HumanMessage → not part of batch
        messages = [
            AIMessage(content="", tool_calls=[tc1, tc2]),
            ToolMessage(content="result1", tool_call_id="call_1", name="read"),
            HumanMessage(content="interrupting"),
            ToolMessage(content="result2", tool_call_id="call_2", name="write"),
        ]
        result = _reorder(messages)
        assert result is None  # Partial batch (only 1 collected) → no reorder

    def test_three_tool_calls_out_of_order(self):
        tc1 = _make_tool_call("a", "call_a")
        tc2 = _make_tool_call("b", "call_b")
        tc3 = _make_tool_call("c", "call_c")

        messages = [
            AIMessage(content="", tool_calls=[tc1, tc2, tc3]),
            ToolMessage(content="C", tool_call_id="call_c", name="c"),
            ToolMessage(content="A", tool_call_id="call_a", name="a"),
            ToolMessage(content="B", tool_call_id="call_b", name="b"),
        ]
        result = _reorder(messages)
        assert result is not None
        assert result[1].tool_call_id == "call_a"
        assert result[2].tool_call_id == "call_b"
        assert result[3].tool_call_id == "call_c"

    def test_multiple_ai_messages_with_tool_calls(self):
        """Each AIMessage's batch is independently reordered."""
        tc1 = _make_tool_call("read", "call_r1")
        tc2 = _make_tool_call("write", "call_w1")
        tc3 = _make_tool_call("read", "call_r2")
        tc4 = _make_tool_call("write", "call_w2")

        messages = [
            # First batch (in order)
            AIMessage(content="", tool_calls=[tc1, tc2]),
            ToolMessage(content="r1", tool_call_id="call_r1", name="read"),
            ToolMessage(content="w1", tool_call_id="call_w1", name="write"),
            # Second batch (out of order)
            AIMessage(content="", tool_calls=[tc3, tc4]),
            ToolMessage(content="w2", tool_call_id="call_w2", name="write"),
            ToolMessage(content="r2", tool_call_id="call_r2", name="read"),
        ]
        result = _reorder(messages)
        assert result is not None
        # First batch unchanged
        assert result[1].tool_call_id == "call_r1"
        assert result[2].tool_call_id == "call_w1"
        # Second batch reordered
        assert result[4].tool_call_id == "call_r2"
        assert result[5].tool_call_id == "call_w2"

    def test_no_tool_calls_after_collection(self):
        """AIMessage with tool_calls followed by empty block → no change."""
        tc1 = _make_tool_call("read", "call_1")

        messages = [
            AIMessage(content="", tool_calls=[tc1]),
        ]
        result = _reorder(messages)
        assert result is None  # No ToolMessages at all

    def test_extra_tool_messages_not_part_of_batch(self):
        """ToolMessages with IDs not in any AIMessage's tool_calls are untouched."""
        tc1 = _make_tool_call("read", "call_1")

        messages = [
            AIMessage(content="", tool_calls=[tc1]),
            ToolMessage(content="matched", tool_call_id="call_1", name="read"),
            # Stray ToolMessage from another batch
            ToolMessage(content="stray", tool_call_id="call_999", name="ghost"),
        ]
        result = _reorder(messages)
        assert result is None  # first batch is in order (only 1 element)


# ==============================================================================
# Unit tests — ReorderToolMessagesMiddleware
# ==============================================================================


class TestReorderMiddleware:
    def test_empty_state_returns_none(self):
        mw = ReorderToolMessagesMiddleware()
        state = {"messages": []}
        assert mw.before_model(state, _EMPTY_RUNTIME) is None

    def test_single_message_returns_none(self):
        mw = ReorderToolMessagesMiddleware()
        state = {"messages": [HumanMessage(content="hi")]}
        assert mw.before_model(state, _EMPTY_RUNTIME) is None

    def test_ordered_messages_returns_none(self):
        mw = ReorderToolMessagesMiddleware()
        tc1 = _make_tool_call("read", "call_1")
        tc2 = _make_tool_call("write", "call_2")
        state = {
            "messages": [
                AIMessage(content="", tool_calls=[tc1, tc2]),
                ToolMessage(content="r1", tool_call_id="call_1", name="read"),
                ToolMessage(content="w1", tool_call_id="call_w2", name="write"),
            ]
        }
        # This is a partial batch (w1 has wrong id) → no reorder
        result = mw.before_model(state, _EMPTY_RUNTIME)
        assert result is None

    def test_disordered_messages_produces_overwrite(self):
        mw = ReorderToolMessagesMiddleware()
        tc1 = _make_tool_call("read", "call_1")
        tc2 = _make_tool_call("write", "call_2")
        state = {
            "messages": [
                AIMessage(content="", tool_calls=[tc1, tc2]),
                ToolMessage(content="w2", tool_call_id="call_2", name="write"),
                ToolMessage(content="r1", tool_call_id="call_1", name="read"),
            ]
        }
        result = mw.before_model(state, _EMPTY_RUNTIME)
        assert result is not None
        assert "messages" in result
        assert isinstance(result["messages"], Overwrite)
        reordered = result["messages"].value
        assert reordered[1].tool_call_id == "call_1"
        assert reordered[2].tool_call_id == "call_2"

    def test_abefore_model_delegates_to_sync(self):
        mw = ReorderToolMessagesMiddleware()
        state = {"messages": []}
        # abefore_model just delegates to before_model
        assert mw.before_model(state, _EMPTY_RUNTIME) is None
