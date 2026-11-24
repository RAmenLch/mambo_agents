"""Internalize tool-execution exceptions into ``ToolMessage`` results.

``ToolNode``'s default error handling only converts *invocation* errors
(argument validation, unknown tool) into ``ToolMessage(status="error")``;
anything raised while the tool body runs is re-raised and aborts the whole
superstep.  That is destructive when tool calls run in parallel: the sibling
results are dropped before they are merged into the channel, and a dangling
``tool_call`` is left behind (patched later with "was cancelled", which makes
the model redo work that already succeeded).

This middleware wraps the tool call itself and converts such exceptions into a
``ToolMessage`` so the superstep commits normally and every sibling result
survives.  It is a tool-call hook only — it does not add graph nodes, and it
does not touch model calls.

What is *not* internalized:

* ``GraphBubbleUp`` (``GraphInterrupt``, ``ParentCommand``, ...) — control flow,
  not an error; swallowing it would break interrupts and human review.
* Programming errors (``TypeError``, ``AttributeError``, ``InvalidUpdateError``,
  ...) — they signal a defect in the code, and internalizing them would hide the
  bug and invite the model to retry on top of it.
* Cancellation / shutdown signals (including ones wrapped into a plain
  exception by the tool or its HTTP client) — see
  :func:`mambo_agents.utils.cancellation.is_cancellation_related`.

Everything else (timeouts, connection errors, rate limits, 4xx/5xx, missing
files, permission errors, external service failures, ...) is internalized: the
model can see the failure and choose another strategy.

Escalation: a per-thread failure streak is tracked so that a tool subsystem
that is consistently broken becomes visible.  Reaching
``max_consecutive_failures`` does **not** re-raise — breaking the superstep is
exactly what this middleware exists to prevent — it raises the log level and
flags the message (``additional_kwargs["tool_error"]["escalated"]``) so the
application layer can alert.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import AgentMiddleware, AgentState, ResponseT
from langchain_core.messages import ToolMessage
from langgraph.errors import EmptyInputError, GraphBubbleUp, InvalidUpdateError
from langgraph.typing import ContextT

from mambo_agents.utils.cancellation import (
    is_cancellation_related,
    iter_exception_chain,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain.agents.middleware.types import ToolCallRequest
    from langgraph.types import Command

logger = logging.getLogger(__name__)

__all__ = ["ToolCallErrorMiddleware", "is_retryable"]

# Errors that point at a defect rather than at a runtime condition.  These keep
# propagating: internalizing them would mask the bug (and the model would keep
# retrying on top of broken code).
_PROGRAMMING_ERRORS: tuple[type[BaseException], ...] = (
    TypeError,
    AttributeError,
    NameError,
    AssertionError,
    NotImplementedError,
    ImportError,
    # Raised when a node (or a nested subagent) writes a key the state schema
    # does not declare — a schema bug, never a tool failure.
    InvalidUpdateError,
    # Raised when a graph is invoked with empty input — a wiring bug.
    EmptyInputError,
)

# HTTP status codes worth retrying (rate limiting / transient server side).
_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_NON_RETRYABLE_STATUS = frozenset({400, 401, 403, 404, 405, 406, 410, 413, 415, 422})

_RETRYABLE_NAME_HINTS = (
    "Timeout",
    "TimeoutError",
    "Connection",
    "RateLimit",
    "TooManyRequests",
    "ServiceUnavailable",
    "InternalServerError",
    "Unavailable",
    "Deadline",
    "Temporary",
    "Reset",
    "Broken",
    "Disconnected",
)
_NON_RETRYABLE_NAME_HINTS = (
    "BadRequest",
    "Authentication",
    "Unauthorized",
    "Forbidden",
    "PermissionDenied",
    "NotFound",
    "Validation",
    "InvalidRequest",
    "ContextOverflow",
)

# Guard rails for what ends up in the transcript.
_CONTENT_LIMIT = 2000
_DETAIL_LIMIT = 1000


def _status_code(exc: BaseException) -> int | None:
    """Best-effort HTTP status code of *exc* (directly or through its chain)."""
    for node in iter_exception_chain(exc):
        code = getattr(node, "status_code", None)
        if isinstance(code, int):
            return code
        response = getattr(node, "response", None)
        code = getattr(response, "status_code", None)
        if isinstance(code, int):
            return code
    return None


def _is_retryable(exc: BaseException) -> bool:
    """Whether retrying the same call could plausibly succeed."""
    code = _status_code(exc)
    if code is not None:
        if code in _NON_RETRYABLE_STATUS:
            return False
        return code in _RETRYABLE_STATUS or 500 <= code < 600

    names = " ".join(type(node).__name__ for node in iter_exception_chain(exc))
    if any(hint in names for hint in _NON_RETRYABLE_NAME_HINTS):
        return False
    return any(hint in names for hint in _RETRYABLE_NAME_HINTS)


#: Public alias — shared with the subagent continuation layer, which needs the
#: exact same classification.  Keep a single implementation.
is_retryable = _is_retryable


class ToolCallErrorMiddleware(AgentMiddleware[AgentState, ContextT, ResponseT]):
    """Turn tool-execution exceptions into ``ToolMessage(status="error")``.

    Args:
        enabled: Master switch.  When ``False`` the hook is a pure pass-through
            and the framework default (exception propagates) applies.
        max_consecutive_failures: Number of consecutive internalized failures for
            the same ``thread_id`` after which failures are logged at ``ERROR``
            level and flagged as ``escalated``.  The exception is still
            internalized.
        max_tracked_threads: Upper bound for the in-process streak table; the
            table is dropped wholesale when exceeded.

    The streak counter is process-local and only tracked when a ``thread_id`` is
    available in the runnable config — otherwise counting would lump unrelated
    runs together.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        max_consecutive_failures: int = 3,
        max_tracked_threads: int = 1024,
    ) -> None:
        super().__init__()
        self.enabled = enabled
        self.max_consecutive_failures = max(1, int(max_consecutive_failures))
        self.max_tracked_threads = max(1, int(max_tracked_threads))
        self._lock = threading.Lock()
        self._fail_streak: dict[str, int] = {}

    # ------------------------------------------------------------------
    # wrap_tool_call / awrap_tool_call
    # ------------------------------------------------------------------

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Execute the sync tool call, internalizing runtime failures."""
        if not self.enabled:
            return handler(request)

        try:
            result = handler(request)
        except GraphBubbleUp:
            raise
        except _PROGRAMMING_ERRORS:
            raise
        except Exception as exc:  # noqa: BLE001 - deliberately broad
            if is_cancellation_related(exc):
                raise
            return self._internalize(request, exc)

        self._clear_streak(request)
        return result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Execute the async tool call, internalizing runtime failures."""
        if not self.enabled:
            return await handler(request)

        try:
            result = await handler(request)
        except GraphBubbleUp:
            raise
        except _PROGRAMMING_ERRORS:
            raise
        except Exception as exc:  # noqa: BLE001 - deliberately broad
            if is_cancellation_related(exc):
                raise
            return self._internalize(request, exc)

        self._clear_streak(request)
        return result

    # ------------------------------------------------------------------
    # Internalization
    # ------------------------------------------------------------------

    def _internalize(self, request: ToolCallRequest, exc: Exception) -> ToolMessage:
        """Build the error ``ToolMessage``; must never raise."""
        tool_call = getattr(request, "tool_call", None) or {}
        name = str(tool_call.get("name") or "unknown_tool")
        call_id = tool_call.get("id")
        # ``ToolMessage.tool_call_id`` must be a string.  A missing id is a
        # upstream bug (``PatchToolCallsMiddleware`` normally prevents it) — fall
        # back to an empty id rather than raising from inside an error handler,
        # which would abort the superstep.
        call_id = "" if call_id is None else str(call_id)

        detail = f"{type(exc).__name__}: {exc}"
        thread_id = self._thread_id(request)
        streak = self._bump_streak(thread_id)
        escalated = streak > self.max_consecutive_failures
        retryable = _is_retryable(exc)

        log = logger.error if escalated else logger.warning
        log(
            "Tool call failed and was internalized: tool=%s id=%s thread=%s "
            "streak=%d retryable=%s error=%s",
            name,
            call_id or "<none>",
            thread_id or "<none>",
            streak,
            retryable,
            detail,
            exc_info=exc if escalated else False,
        )
        if escalated:
            logger.error(
                "Tool '%s' failed %d times in a row (thread=%s); escalate to the "
                "application layer — the tool or its backend is likely down.",
                name,
                streak,
                thread_id or "<none>",
            )

        return ToolMessage(
            content=f"Tool '{name}' failed: {detail}"[:_CONTENT_LIMIT],
            name=name,
            tool_call_id=call_id,
            status="error",
            additional_kwargs={
                "tool_error": {
                    "tool_name": name,
                    "error_type": type(exc).__name__,
                    "message": detail[:_DETAIL_LIMIT],
                    "retryable": retryable,
                    "streak": streak,
                    "escalated": escalated,
                }
            },
        )

    # ------------------------------------------------------------------
    # Failure streak bookkeeping
    # ------------------------------------------------------------------

    @staticmethod
    def _thread_id(request: ToolCallRequest) -> str | None:
        """The ``thread_id`` of the current run, if any."""
        runtime = getattr(request, "runtime", None)
        config = getattr(runtime, "config", None)
        if not isinstance(config, dict):
            return None
        configurable = config.get("configurable")
        if not isinstance(configurable, dict):
            return None
        thread_id = configurable.get("thread_id")
        return str(thread_id) if thread_id else None

    def _bump_streak(self, thread_id: str | None) -> int:
        """Increment and return the failure streak for *thread_id*."""
        if thread_id is None:
            return 0
        with self._lock:
            if len(self._fail_streak) > self.max_tracked_threads:
                self._fail_streak.clear()
            streak = self._fail_streak.get(thread_id, 0) + 1
            self._fail_streak[thread_id] = streak
            return streak

    def _clear_streak(self, request: ToolCallRequest) -> None:
        """Reset the failure streak after a successful tool call."""
        thread_id = self._thread_id(request)
        if thread_id is None:
            return
        with self._lock:
            self._fail_streak.pop(thread_id, None)
