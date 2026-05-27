"""Middleware that reorders ``ToolMessage`` entries to match ``AIMessage.tool_calls`` order.

When multiple tools are invoked in parallel (e.g. reading several images at once),
the ``ToolMessage`` results may arrive in a different order than the ``tool_calls``
list on the originating ``AIMessage``.  While the LLM matches ToolMessages by
``tool_call_id`` (not position), some multi-modal models are sensitive to the
ordering of content blocks and may misinterpret results when ToolMessages appear
out of order relative to their corresponding tool calls.

This middleware scans the message history before each model call and, for every
``AIMessage`` with ``tool_calls``, reorders the immediately following block of
``ToolMessage`` entries to match the ``tool_call`` order.

.. important::

   This middleware requires **complete** tool-call batches (i.e. every
   ``tool_call_id`` must have a corresponding ``ToolMessage``).  Partial
   batches (e.g. resulting from dangling tool calls) are passed through
   unchanged.  ``PatchToolCallsMiddleware`` **must** run before this
   middleware to fill in any missing entries so that reordering can
   operate on a complete batch.

Only ToolMessages within a **contiguous block** following their parent AIMessage
are reordered; messages that cross non-``ToolMessage`` boundaries are left in
place, preserving the integrity of multi-turn conversations.
"""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, AgentState, ResponseT
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Overwrite
from langgraph.typing import ContextT


class ReorderToolMessagesMiddleware(AgentMiddleware[AgentState, ContextT, ResponseT]):
    """Reorder ``ToolMessage`` entries to match ``AIMessage.tool_calls`` order.

    Before each model call, scans the message history for ``AIMessage`` objects
    that carry ``tool_calls``.  If the immediately following ``ToolMessage``
    block is out of order with respect to the tool-call sequence, the block is
    rewritten in the correct order.

    The middleware is zero-overhead when no reordering is needed: it returns
    ``None`` from ``before_model``, leaving the state unchanged.
    """

    state_schema = AgentState

    # ------------------------------------------------------------------
    # before_model / abefore_model
    # ------------------------------------------------------------------

    def before_model(
        self, state: AgentState, runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Reorder ToolMessages before each model call (sync)."""
        messages = state.get("messages")
        if not messages or len(messages) < 2:
            return None

        reordered = _reorder(messages)
        if reordered is None:
            return None

        return {"messages": Overwrite(value=reordered)}

    async def abefore_model(
        self, state: AgentState, runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Reorder ToolMessages before each model call (async)."""
        # Pure computation, no I/O – delegate to sync.
        return self.before_model(state, runtime)


# ---------------------------------------------------------------------------
# Reordering logic (module-level for testability)
# ---------------------------------------------------------------------------


def _reorder(messages: list) -> list | None:
    """Reorder ``ToolMessage`` entries within each AIMessage's tool_calls batch.

    For each ``AIMessage`` that carries ``tool_calls``, collects the
    immediately following ``ToolMessage`` entries whose ``tool_call_id``
    matches one of the call IDs.  If the collected block is not in the
    expected order (as defined by ``tool_calls``), the block is rewritten.

    Args:
        messages: The full message history.

    Returns:
        A new ``list`` if reordering was necessary, otherwise ``None``.
    """
    reordered: list = []
    changed = False
    i = 0
    n = len(messages)

    while i < n:
        msg = messages[i]
        reordered.append(msg)
        i += 1

        if not isinstance(msg, AIMessage) or not msg.tool_calls:
            continue

        ordered_ids = [tc["id"] for tc in msg.tool_calls]
        id_set = set(ordered_ids)

        # Collect consecutive ToolMessages belonging to this batch
        buffered: dict[str, Any] = {}
        while i < n:
            nm = messages[i]
            if isinstance(nm, ToolMessage) and nm.tool_call_id in id_set:
                buffered[nm.tool_call_id] = nm
                i += 1
            else:
                break

        if not buffered:
            continue

        # Only reorder when we have the **full** complement of ToolMessages
        # for this batch.  Partial blocks (e.g. dangling tool calls) are
        # appended in situ and left for PatchToolCallsMiddleware to resolve.
        if len(buffered) != len(ordered_ids):
            reordered.extend(buffered.values())
            continue

        # Compare current order to expected order
        current_order = list(buffered.keys())
        if current_order == ordered_ids:
            reordered.extend(buffered.values())
            continue

        changed = True
        for tc_id in ordered_ids:
            # Each ID is guaranteed present because we verified
            # len(buffered) == len(ordered_ids)
            reordered.append(buffered[tc_id])

    if not changed:
        return None
    return reordered
