"""Tests for ``ToolCallErrorMiddleware`` and the cancellation helpers."""

from __future__ import annotations

import asyncio

import httpx
import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from mambo_agents.middleware.tool_call_error import (
    ToolCallErrorMiddleware,
    _is_retryable,
    resolve_tool_error_middleware,
)
from mambo_agents.utils.cancellation import is_cancellation_related


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
class _FakeRuntime:
    def __init__(self, thread_id: str | None = None) -> None:
        self.config = {"configurable": {"thread_id": thread_id}} if thread_id else {}


class _FakeRequest:
    def __init__(
        self,
        name: str = "demo",
        call_id: str = "c1",
        thread_id: str | None = None,
    ) -> None:
        self.tool_call = {"name": name, "args": {}, "id": call_id, "type": "tool_call"}
        self.runtime = _FakeRuntime(thread_id)


def _raise(exc: BaseException):
    def handler(_request):
        raise exc

    return handler


@tool
def _ok_tool(x: int) -> str:
    """Always succeeds."""
    return f"ok:{x}"


@tool
def _boom_tool(x: int) -> str:
    """Always raises a business error."""
    raise RuntimeError("backend exploded")


@tool
def _typo_tool(x: int) -> str:
    """Raises a programming error."""
    raise TypeError("bad usage")


def _tool_graph(tools, wrapper=None, awrapper=None):
    node = ToolNode(tools, wrap_tool_call=wrapper, awrap_tool_call=awrapper)
    graph = StateGraph(MessagesState)
    graph.add_node("tools", node)
    graph.add_edge(START, "tools")
    graph.add_edge("tools", END)
    return graph.compile()


def _call(*pairs: tuple[str, str]) -> dict:
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": name, "args": {"x": 1}, "id": cid, "type": "tool_call"}
                    for name, cid in pairs
                ],
            )
        ]
    }


def _tool_messages(result: dict) -> list[ToolMessage]:
    return [m for m in result["messages"] if isinstance(m, ToolMessage)]


# ---------------------------------------------------------------------------
# is_cancellation_related
# ---------------------------------------------------------------------------
class TestCancellationDetection:
    def test_plain_error_is_not_cancellation(self):
        assert is_cancellation_related(ValueError("boom")) is False

    def test_cancelled_error_is_cancellation(self):
        assert is_cancellation_related(asyncio.CancelledError()) is True

    def test_wrapped_cancellation_is_detected(self):
        try:
            try:
                raise asyncio.CancelledError()
            except asyncio.CancelledError:
                raise RuntimeError("tool wrapped the cancel") from None
        except RuntimeError as exc:
            assert is_cancellation_related(exc) is True

    def test_timeout_is_not_cancellation(self):
        assert is_cancellation_related(TimeoutError()) is False

    def test_timeout_via_cancel_scope_is_not_cancellation(self):
        """``anyio``/``httpcore`` build timeouts out of a cancel scope.

        The resulting ``TimeoutError`` carries a ``CancelledError`` in its
        context chain (``reason: deadline exc``) but is a timeout, not a
        cancellation.  Misclassifying it would stop every HTTP timeout from
        being internalized.
        """
        try:
            try:
                raise asyncio.CancelledError(
                    "Cancelled via cancel scope 0x1; reason: deadline exc"
                )
            except asyncio.CancelledError as inner:
                raise TimeoutError() from inner
        except TimeoutError as exc:
            assert is_cancellation_related(exc) is False

    def test_httpx_read_timeout_shape_is_not_cancellation(self):
        """Same chain as a real ``httpx.ReadTimeout`` raised by httpcore."""
        context: BaseException | None = None
        try:
            try:
                raise asyncio.CancelledError("reason: deadline exc")
            except asyncio.CancelledError as inner:
                context = inner
                raise TimeoutError() from inner
        except TimeoutError as inner_timeout:
            exc = httpx.ReadTimeout("")
            exc.__cause__ = inner_timeout
        assert context is not None
        assert is_cancellation_related(exc) is False


@pytest.mark.asyncio
async def test_task_cancelling_flag_is_detected():
    """A tool that swallows ``CancelledError`` is still recognised as cancelled."""

    async def tool_body() -> None:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            pass
        raise RuntimeError("raised after swallowing the cancel")

    task = asyncio.create_task(tool_body())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(RuntimeError):
        await task

    # Re-run the body inside a cancelling task so the flag is observable from
    # within the same task (that is what the middleware sees).
    seen: list[bool] = []

    async def probe() -> None:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            try:
                raise RuntimeError("post-cancel failure")
            except RuntimeError as exc:
                seen.append(is_cancellation_related(exc))
        raise asyncio.CancelledError()

    t = asyncio.create_task(probe())
    await asyncio.sleep(0.05)
    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t
    assert seen == [True]


# ---------------------------------------------------------------------------
# passthrough behaviour
# ---------------------------------------------------------------------------
class TestPassthrough:
    def test_graph_bubble_up_is_reraised(self):
        middleware = ToolCallErrorMiddleware()
        with pytest.raises(GraphInterrupt):
            middleware.wrap_tool_call(_FakeRequest(), _raise(GraphInterrupt(())))

    def test_programming_errors_are_reraised(self):
        middleware = ToolCallErrorMiddleware()
        for exc in (TypeError("x"), AttributeError("x"), NameError("x"),
                    AssertionError("x"), NotImplementedError("x"), ImportError("x")):
            with pytest.raises(type(exc)):
                middleware.wrap_tool_call(_FakeRequest(), _raise(exc))

    def test_cancellation_is_reraised(self):
        middleware = ToolCallErrorMiddleware()
        with pytest.raises(asyncio.CancelledError):
            middleware.wrap_tool_call(_FakeRequest(), _raise(asyncio.CancelledError()))

    def test_enabled_false_is_passthrough(self):
        middleware = ToolCallErrorMiddleware(enabled=False)
        with pytest.raises(RuntimeError):
            middleware.wrap_tool_call(_FakeRequest(), _raise(RuntimeError("boom")))


# ---------------------------------------------------------------------------
# internalization
# ---------------------------------------------------------------------------
class TestInternalization:
    def test_business_error_becomes_tool_message(self):
        middleware = ToolCallErrorMiddleware()
        message = middleware.wrap_tool_call(
            _FakeRequest(name="write", call_id="c7"), _raise(RuntimeError("disk full"))
        )
        assert isinstance(message, ToolMessage)
        assert message.status == "error"
        assert message.tool_call_id == "c7"
        assert message.name == "write"
        assert "disk full" in message.content
        info = message.additional_kwargs["tool_error"]
        assert info["error_type"] == "RuntimeError"
        assert info["retryable"] is False
        assert info["escalated"] is False

    def test_successful_result_is_returned_unchanged(self):
        middleware = ToolCallErrorMiddleware()
        sentinel = ToolMessage(content="fine", tool_call_id="c1", name="demo")
        assert middleware.wrap_tool_call(_FakeRequest(), lambda _r: sentinel) is sentinel

    def test_streak_escalates_without_raising(self):
        middleware = ToolCallErrorMiddleware(max_consecutive_failures=2)
        request = _FakeRequest(thread_id="t-1")
        flags = []
        for _ in range(4):
            message = middleware.wrap_tool_call(request, _raise(RuntimeError("down")))
            flags.append(message.additional_kwargs["tool_error"]["escalated"])
        assert flags == [False, False, True, True]

    def test_success_resets_the_streak(self):
        middleware = ToolCallErrorMiddleware(max_consecutive_failures=2)
        request = _FakeRequest(thread_id="t-2")
        middleware.wrap_tool_call(request, _raise(RuntimeError("1")))
        middleware.wrap_tool_call(request, _raise(RuntimeError("2")))
        middleware.wrap_tool_call(request, lambda _r: ToolMessage(content="ok", tool_call_id="c1"))
        message = middleware.wrap_tool_call(request, _raise(RuntimeError("3")))
        assert message.additional_kwargs["tool_error"]["escalated"] is False

    def test_missing_thread_id_does_not_share_a_bucket(self):
        middleware = ToolCallErrorMiddleware(max_consecutive_failures=1)
        request = _FakeRequest(thread_id=None)
        flags = [
            middleware.wrap_tool_call(request, _raise(RuntimeError("x")))
            .additional_kwargs["tool_error"]["escalated"]
            for _ in range(3)
        ]
        assert flags == [False, False, False]


class TestEscalationHook:
    def test_hook_fires_only_when_escalated(self):
        seen: list[tuple] = []
        middleware = ToolCallErrorMiddleware(
            max_consecutive_failures=2,
            on_escalate=lambda *args: seen.append(args),
        )
        request = _FakeRequest(name="write", thread_id="t-hook")
        for _ in range(3):
            middleware.wrap_tool_call(request, _raise(RuntimeError("down")))
        assert [(t, n, s) for t, n, s, _e in seen] == [("t-hook", "write", 3)]

    def test_hook_exception_is_swallowed(self):
        def boom(*_args):
            raise ValueError("bad host callback")

        middleware = ToolCallErrorMiddleware(
            max_consecutive_failures=1, on_escalate=boom
        )
        request = _FakeRequest(thread_id="t-boom")
        middleware.wrap_tool_call(request, _raise(RuntimeError("down")))
        message = middleware.wrap_tool_call(request, _raise(RuntimeError("down")))
        assert isinstance(message, ToolMessage)
        assert message.additional_kwargs["tool_error"]["escalated"] is True


class TestResolveToolErrorMiddleware:
    def test_true_and_none_yield_a_default_instance(self):
        assert isinstance(resolve_tool_error_middleware(True), ToolCallErrorMiddleware)
        assert isinstance(
            resolve_tool_error_middleware(None), ToolCallErrorMiddleware
        )

    def test_false_disables(self):
        assert resolve_tool_error_middleware(False) is None

    def test_instance_is_used_as_is(self):
        instance = ToolCallErrorMiddleware(max_consecutive_failures=9)
        assert resolve_tool_error_middleware(instance) is instance

    def test_bad_type_raises(self):
        with pytest.raises(TypeError):
            resolve_tool_error_middleware("yes")  # type: ignore[arg-type]


class TestRetryableClassification:
    def test_status_codes(self):
        for code, expected in ((429, True), (503, True), (500, True), (400, False), (403, False)):
            exc = RuntimeError("x")
            exc.status_code = code  # type: ignore[attr-defined]
            assert _is_retryable(exc) is expected, code

    def test_timeout_and_connection_names(self):
        assert _is_retryable(httpx.ReadTimeout("")) is True
        assert _is_retryable(ConnectionResetError()) is True
        assert _is_retryable(FileNotFoundError("nope")) is False


# ---------------------------------------------------------------------------
# end to end through ToolNode
# ---------------------------------------------------------------------------
class TestToolNodeIntegration:
    def test_parallel_sibling_result_survives(self):
        graph = _tool_graph([_ok_tool, _boom_tool], wrapper=ToolCallErrorMiddleware().wrap_tool_call)
        messages = {m.tool_call_id: m for m in _tool_messages(graph.invoke(_call(("_ok_tool", "c1"), ("_boom_tool", "c2"))))}
        assert set(messages) == {"c1", "c2"}
        assert messages["c1"].status == "success"
        assert messages["c2"].status == "error"

    def test_without_middleware_the_exception_propagates(self):
        graph = _tool_graph([_ok_tool, _boom_tool])
        with pytest.raises(RuntimeError):
            graph.invoke(_call(("_ok_tool", "c1"), ("_boom_tool", "c2")))

    def test_programming_error_still_breaks_the_run(self):
        graph = _tool_graph([_typo_tool], wrapper=ToolCallErrorMiddleware().wrap_tool_call)
        with pytest.raises(TypeError):
            graph.invoke(_call(("_typo_tool", "c1")))

    @pytest.mark.asyncio
    async def test_async_parallel_sibling_result_survives(self):
        graph = _tool_graph(
            [_ok_tool, _boom_tool],
            awrapper=ToolCallErrorMiddleware().awrap_tool_call,
        )
        result = await graph.ainvoke(_call(("_ok_tool", "c1"), ("_boom_tool", "c2")))
        messages = {m.tool_call_id: m for m in _tool_messages(result)}
        assert set(messages) == {"c1", "c2"}
        assert messages["c2"].status == "error"
