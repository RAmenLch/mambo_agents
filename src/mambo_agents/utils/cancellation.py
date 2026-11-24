"""Helpers for telling cancellation / shutdown signals apart from real errors.

Cancellation is not an error.  When the caller stops a run (``asyncio`` task
cancellation) or the interpreter is shutting down, the exception raised at the
tool boundary must propagate so that LangGraph aborts the superstep instead of
committing it and calling the model again after the user pressed "stop".

Detecting it reliably is not trivial, because the signal travels through the
tool body, the HTTP client and possibly the serialization layer before it
reaches a middleware — and any of those layers may wrap it into a plain
``Exception``.  Two independent signals are therefore combined:

1. **Task state (primary).**  ``asyncio.Task.cancelling()`` stays non-zero while
   a cancellation request is outstanding, so it survives a layer that swallows
   ``CancelledError`` and raises something else afterwards.  Libraries that
   implement timeouts *through* task cancellation (``anyio.fail_after``, used by
   ``httpcore`` and therefore by ``httpx`` / the vendor SDKs) call ``uncancel()``
   when their cancel scope exits, so a plain timeout does **not** look like a
   cancellation here.
2. **Exception chain (secondary).**  ``__cause__`` / ``__context__`` still carry
   the ``CancelledError`` when it was wrapped with ``raise ... from ...`` or
   raised from inside an ``except`` block.  A ``TimeoutError`` anywhere in the
   chain means the ``CancelledError`` is only the transport of a timeout scope,
   so the error is classified as a timeout instead.

The module is deliberately dependency-free (standard library only) so it can be
shared by the tool middleware and the subagent task helpers.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

__all__ = [
    "is_cancellation_related",
    "iter_exception_chain",
]

# Hard bound so that a pathological (self-referencing) chain cannot hang us.
_MAX_CHAIN_NODES = 24

_CANCELLATION_TYPES: tuple[type[BaseException], ...] = (
    asyncio.CancelledError,
    GeneratorExit,
)

# ``asyncio.TimeoutError`` and ``concurrent.futures.TimeoutError`` are aliases of
# the builtin since Python 3.11, so a single type check covers all of them.
_TIMEOUT_TYPES: tuple[type[BaseException], ...] = (TimeoutError,)


def iter_exception_chain(exc: BaseException) -> Iterator[BaseException]:
    """Yield *exc* plus everything reachable through causes/contexts/groups.

    Both ``__cause__`` and ``__context__`` are followed (an exception may carry
    both), and ``ExceptionGroup`` members are flattened, so callers see every
    exception involved in the failure.  The walk is bounded and cycle-safe.
    """
    pending: list[BaseException] = [exc]
    seen: set[int] = set()

    while pending and len(seen) < _MAX_CHAIN_NODES:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current

        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)

        members = getattr(current, "exceptions", None)
        if isinstance(members, tuple):
            pending.extend(m for m in members if isinstance(m, BaseException))


def _task_is_cancelling() -> bool:
    """Whether the *current* asyncio task has an outstanding cancel request."""
    try:
        task = asyncio.current_task()
    except RuntimeError:
        # No running event loop (sync execution path) — degrade gracefully.
        return False
    if task is None:
        return False
    cancelling = getattr(task, "cancelling", None)
    if cancelling is None:
        return False
    try:
        return bool(cancelling())
    except Exception:  # pragma: no cover - defensive
        return False


def is_cancellation_related(exc: BaseException) -> bool:
    """Return ``True`` when *exc* is a cancellation/shutdown signal, not an error.

    ``True`` means "do not internalize": the exception must keep propagating so
    the run is actually aborted.
    """
    if _task_is_cancelling():
        return True

    saw_cancellation = False
    saw_timeout = False
    for node in iter_exception_chain(exc):
        if isinstance(node, _TIMEOUT_TYPES):
            saw_timeout = True
        elif isinstance(node, _CANCELLATION_TYPES):
            saw_cancellation = True

    return saw_cancellation and not saw_timeout
