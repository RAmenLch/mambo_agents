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

from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import AgentMiddleware, AgentState, ResponseT
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Overwrite
from langgraph.typing import ContextT

if TYPE_CHECKING:
    from langgraph.store.base import BaseStore

    from mambo_agents.middleware.subagent_snapshot import SubAgentSnapshotNode

# LangGraph internal key used in the dict-form of Overwrite:
#   {'__overwrite__': <value>}
_OVERWRITE_KEY = "__overwrite__"

#: Name of the subagent spawner tool — only its dangling calls carry a
#: ``continue_id``.
_TASK_TOOL_NAME = "task"


def _unwrap_overwrite(value: Any) -> Any:
    """Return the actual value if *value* is wrapped in an Overwrite marker."""
    if isinstance(value, Overwrite):
        return value.value
    if isinstance(value, dict) and set(value.keys()) == {_OVERWRITE_KEY}:
        return value[_OVERWRITE_KEY]
    return value


def _thread_id(runtime: Runtime) -> str | None:
    config = getattr(runtime, "config", None) or {}
    configurable = config.get("configurable") if isinstance(config, dict) else None
    if not isinstance(configurable, dict):
        return None
    thread_id = configurable.get("thread_id")
    return str(thread_id) if thread_id else None


def _continuation_hint(
    node: SubAgentSnapshotNode | None,
) -> tuple[str, dict[str, Any]] | None:
    """Text + metadata telling the model it may resume this ``task`` call."""
    if node is None or not node.available:
        return None
    text = (
        "\n\nProgress up to that point was saved. To continue without redoing "
        f'completed steps, call task(subagent_type="{node.subagent_type}", '
        f'continue_id="{node.node_id}").'
    )
    return text, {
        "subagent_continuation": {
            "continue_id": node.node_id,
            "subagent_type": node.subagent_type,
            "status": node.status,
            "depth": node.depth,
            "retryable": node.retryable,
            "steps": node.steps,
            "available": True,
        }
    }


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
                        hint = None
                        if tool_call.get("name") == _TASK_TOOL_NAME:
                            hint = self._lookup_hint(runtime, tool_call["id"])
                        patched.append(_synthetic_message(tool_call, hint))

        return {"messages": Overwrite(patched)}

    async def abefore_agent(
        self, state: AgentState, runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Async twin of :meth:`before_agent` (needs ``store.aget``)."""
        raw_messages = state.get("messages")
        messages = _unwrap_overwrite(raw_messages)

        if not messages:
            return None

        answered_ids: set[str] = {
            msg.tool_call_id  # type: ignore[union-attr]
            for msg in messages
            if isinstance(msg, ToolMessage) and msg.tool_call_id
        }

        dangling = [
            tool_call
            for msg in messages
            if isinstance(msg, AIMessage) and msg.tool_calls
            for tool_call in msg.tool_calls
            if tool_call["id"] not in answered_ids
        ]
        if not dangling:
            return None

        hints: dict[str, tuple[str, dict[str, Any]]] = {}
        for tool_call in dangling:
            if tool_call.get("name") != _TASK_TOOL_NAME:
                continue
            hint = await self._alookup_hint(runtime, tool_call["id"])
            if hint is not None:
                hints[tool_call["id"]] = hint

        patched: list = []
        for msg in messages:
            patched.append(msg)
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    if tool_call["id"] not in answered_ids:
                        patched.append(
                            _synthetic_message(tool_call, hints.get(tool_call["id"]))
                        )

        return {"messages": Overwrite(patched)}

    # ------------------------------------------------------------------
    # continuation lookup (best-effort — never fails the patch)
    # ------------------------------------------------------------------

    @staticmethod
    def _store_and_thread(
        runtime: Runtime,
    ) -> tuple[BaseStore | None, str | None]:
        store = getattr(runtime, "store", None)
        thread_id = _thread_id(runtime)
        if store is None or thread_id is None:
            return None, None
        return store, thread_id

    def _lookup_hint(
        self, runtime: Runtime, tool_call_id: str
    ) -> tuple[str, dict[str, Any]] | None:
        store, thread_id = self._store_and_thread(runtime)
        if store is None or thread_id is None:
            return None
        try:
            from mambo_agents.middleware.subagent_snapshot import get_node

            return _continuation_hint(get_node(store, thread_id, tool_call_id))
        except Exception:  # pragma: no cover - store unavailable / bad backend
            return None

    async def _alookup_hint(
        self, runtime: Runtime, tool_call_id: str
    ) -> tuple[str, dict[str, Any]] | None:
        store, thread_id = self._store_and_thread(runtime)
        if store is None or thread_id is None:
            return None
        try:
            from mambo_agents.middleware.subagent_snapshot import aget_node

            return _continuation_hint(await aget_node(store, thread_id, tool_call_id))
        except Exception:  # pragma: no cover - store unavailable / bad backend
            return None


def _synthetic_message(
    tool_call: dict[str, Any],
    hint: tuple[str, dict[str, Any]] | None,
) -> ToolMessage:
    """Build the "was cancelled" ``ToolMessage`` for a dangling call."""
    content = (
        f"Tool call {tool_call['name']} with id "
        f"{tool_call['id']} was cancelled — "
        "another message came in before it could "
        "be completed."
    )
    kwargs: dict[str, Any] = {}
    if hint is not None:
        suffix, extra = hint
        content += suffix
        kwargs.update(extra)
    return ToolMessage(
        content=content,
        name=tool_call["name"],
        tool_call_id=tool_call["id"],
        additional_kwargs=kwargs,
    )
