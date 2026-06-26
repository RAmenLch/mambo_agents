"""Middleware to patch dangling tool calls in the messages history.

In some edge cases (e.g. human interruption, user sends a new message
mid-execution), an ``AIMessage`` may contain ``tool_calls`` that never
received a corresponding ``ToolMessage`` response.  This leaves the
message history in an invalid state that can confuse the LLM.

This middleware runs :meth:`before_agent` to detect and patch such
dangling tool calls by injecting synthetic "cancelled" ``ToolMessage``
objects, ensuring the agent always sees a coherent message history.
"""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, AgentState, ResponseT
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Overwrite
from langgraph.typing import ContextT

from mambo_agents.diagnostics import get_tracker, is_enabled

# LangGraph internal key used in the dict-form of Overwrite:
#   {'__overwrite__': <value>}
_OVERWRITE_KEY = "__overwrite__"


def _unwrap_overwrite(value: Any) -> Any:
    """Return the actual value if *value* is wrapped in an Overwrite marker."""
    if isinstance(value, Overwrite):
        return value.value
    if isinstance(value, dict) and set(value.keys()) == {_OVERWRITE_KEY}:
        return value[_OVERWRITE_KEY]
    return value


class PatchToolCallsMiddleware(AgentMiddleware[AgentState, ContextT, ResponseT]):
    """Patch dangling tool calls before each agent turn.

    Collects all ``ToolMessage.tool_call_id`` values already present in
    the message history.  For every ``AIMessage`` tool call whose
    ``id`` is **not** in that set, a synthetic ``ToolMessage`` is
    injected right after the ``AIMessage``, marking the call as
    cancelled.

    The middleware is a pure safety net: when no dangling calls exist
    it returns ``None`` (zero overhead).
    """

    state_schema = AgentState

    def before_agent(
        self, state: AgentState, runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Detect and patch dangling tool calls before each agent turn."""
        raw_messages = state.get("messages")
        messages = _unwrap_overwrite(raw_messages)

        # ---- 诊断：检查 state["messages"] 是否已经是 Overwrite（异常信号） ----
        if is_enabled() and isinstance(raw_messages, Overwrite):
            get_tracker().log_state_messages_type_mismatch(
                source="patch_tool_calls.before_agent",
                runtime=runtime,
                value=raw_messages,
            )

        if not messages:
            return None

        # Collect all tool_call_ids that already have a response
        answered_ids: set[str] = {
            msg.tool_call_id  # type: ignore[union-attr]
            for msg in messages
            if isinstance(msg, ToolMessage) and msg.tool_call_id
        }

        # Check if any AIMessage has unanswered tool calls
        has_dangling = any(
            tool_call["id"] not in answered_ids
            for msg in messages
            if isinstance(msg, AIMessage) and msg.tool_calls
            for tool_call in msg.tool_calls
        )
        if not has_dangling:
            return None

        # Patch: inject synthetic ToolMessage for each dangling tool call
        patched: list = []
        for msg in messages:
            patched.append(msg)
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    if tool_call["id"] not in answered_ids:
                        patched.append(
                            ToolMessage(
                                content=(
                                    f"Tool call {tool_call['name']} with id "
                                    f"{tool_call['id']} was cancelled — "
                                    "another message came in before it could "
                                    "be completed."
                                ),
                                name=tool_call["name"],
                                tool_call_id=tool_call["id"],
                            )
                        )

        # ---- 诊断：记录 Overwrite 被产生 ----
        if is_enabled():
            get_tracker().log_overwrite_produced(
                source="patch_tool_calls.before_agent",
                runtime=runtime,
                original_count=len(messages),
                patched_count=len(patched),
                dangling_count=len(patched) - len(messages),
            )

        return {"messages": Overwrite(patched)}
