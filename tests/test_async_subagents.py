"""Tests for AsyncSubAgentMiddleware (background subagents)."""

from __future__ import annotations

import time
import threading
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import Runnable, RunnableLambda

from langgraph.store.memory import InMemoryStore

from mambo_agents.backends.schemas import VirtualPath
from mambo_agents.middleware.async_subagents import (
    ASYNC_TASK_SYSTEM_PROMPT,
    AsyncSubAgentMiddleware,
    AsyncTaskData,
    AsyncTaskInput,
    AsyncStatusInput,
    AsyncListInput,
    AsyncCancelInput,
    ReportProgressInput,
    _AsyncTaskTracker,
    _build_report_progress_tool,
    _build_async_task_tool,
    _build_async_status_tool,
    _build_async_list_tool,
    _build_async_cancel_tool,
    _format_status_message,
    _generate_task_id,
    _run_subagent_in_thread,
    _set_current_async_task,
    _get_current_async_task,
    _append_to_system_message,
)
from mambo_agents.middleware.subagents import CompiledSubAgent
from mambo_agents.backends.store import StoreBackend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool_runtime(
    state: dict | None = None,
    tool_call_id: str = "call-123",
    config: dict | None = None,
):
    """Create a minimal ToolRuntime with all required args."""
    from langgraph.prebuilt.tool_node import ToolRuntime as TR

    return TR(
        state=state or {"messages": []},
        config={"configurable": config or {}},
        tool_call_id=tool_call_id,
        context={},
        stream_writer=lambda data: None,
        store=None,
    )


def _make_stub_runnable(
    name: str = "stub",
    result_content: str = "stub result",
    delay: float = 0,
) -> Runnable:
    """Create a runnable that returns a fixed state after an optional delay.

    The underlying functions accept ``**kwargs`` so that the stub can be
    used with ``runnable.stream(state, config, stream_mode=[...])`` — a
    real ``CompiledStateGraph`` would consume ``stream_mode``, but our
    stub simply ignores it.
    """

    def _invoke(state, config=None, **kwargs):
        if delay:
            time.sleep(delay)
        return {"messages": [AIMessage(content=result_content)]}

    async def _ainvoke(state, config=None, **kwargs):
        if delay:
            time.sleep(delay)
        return {"messages": [AIMessage(content=result_content)]}

    runnable = RunnableLambda(_invoke, afunc=_ainvoke)
    return runnable


def _stub_subagent_spec(name: str = "test-agent", description: str = "A test agent"):
    """Build a minimal CompiledSubAgent with a stub runnable."""
    return CompiledSubAgent(
        name=name,
        description=description,
        runnable=_make_stub_runnable(name),
    )


# ---------------------------------------------------------------------------
# Tests: AsyncTaskTracker
# ---------------------------------------------------------------------------


class TestAsyncTaskTracker:
    """Unit tests for _AsyncTaskTracker (thread-safe task state manager)."""

    def test_register_and_get_snapshot(self):
        tracker = _AsyncTaskTracker()
        data = tracker.register("task-1", "agent-a", threading.Thread(), threading.Event())
        assert data.task_id == "task-1"
        assert data.agent_name == "agent-a"
        assert data.status == "running"
        assert data.percentage == 0.0

        snap = tracker.get_snapshot("task-1")
        assert snap is not None
        assert snap.task_id == "task-1"
        assert snap is not data  # deep copy

    def test_nonexistent_task_returns_none(self):
        tracker = _AsyncTaskTracker()
        assert tracker.get_snapshot("nonexistent") is None

    def test_update_progress(self):
        tracker = _AsyncTaskTracker()
        tracker.register("t1", "a", threading.Thread(), threading.Event())
        tracker.update_progress("t1", "pulling image...", 0.3)
        snap = tracker.get_snapshot("t1")
        assert snap.progress_message == "pulling image..."
        assert snap.percentage == 0.3

    def test_update_progress_clamps_percentage(self):
        tracker = _AsyncTaskTracker()
        tracker.register("t1", "a", threading.Thread(), threading.Event())
        tracker.update_progress("t1", "test", 5.0)  # > 1.0
        assert tracker.get_snapshot("t1").percentage == 1.0
        tracker.update_progress("t1", "test", -1.0)  # < 0.0
        assert tracker.get_snapshot("t1").percentage == 0.0

    def test_mark_success(self):
        tracker = _AsyncTaskTracker()
        tracker.register("t1", "a", threading.Thread(), threading.Event())
        tracker.mark_success("t1", "deploy done")
        snap = tracker.get_snapshot("t1")
        assert snap.status == "success"
        assert snap.result == "deploy done"
        assert snap.finished_at is not None

    def test_mark_error(self):
        tracker = _AsyncTaskTracker()
        tracker.register("t1", "a", threading.Thread(), threading.Event())
        tracker.mark_error("t1", "connection refused")
        snap = tracker.get_snapshot("t1")
        assert snap.status == "error"
        assert snap.error_message == "connection refused"

    def test_mark_cancelled(self):
        tracker = _AsyncTaskTracker()
        cancel_ev = threading.Event()
        tracker.register("t1", "a", threading.Thread(), cancel_ev)
        tracker.request_cancel("t1")
        tracker.mark_cancelled("t1")
        snap = tracker.get_snapshot("t1")
        assert snap.status == "cancelled"

    def test_request_cancel_sets_event(self):
        tracker = _AsyncTaskTracker()
        cancel_ev = threading.Event()
        tracker.register("t1", "a", threading.Thread(), cancel_ev)
        assert not cancel_ev.is_set()
        ok = tracker.request_cancel("t1")
        assert ok
        assert cancel_ev.is_set()

    def test_request_cancel_nonexistent(self):
        tracker = _AsyncTaskTracker()
        assert not tracker.request_cancel("ghost")

    def test_list_snapshots_with_filter(self):
        tracker = _AsyncTaskTracker()
        tracker.register("t1", "a", threading.Thread(), threading.Event())
        tracker.register("t2", "b", threading.Thread(), threading.Event())
        tracker.mark_success("t1", "ok")

        all_tasks = tracker.list_snapshots()
        assert len(all_tasks) == 2

        running = tracker.list_snapshots("running")
        assert len(running) == 1
        assert running[0].task_id == "t2"

        success = tracker.list_snapshots("success")
        assert len(success) == 1
        assert success[0].task_id == "t1"

    def test_is_tracked(self):
        tracker = _AsyncTaskTracker()
        tracker.register("t1", "a", threading.Thread(), threading.Event())
        assert tracker.is_tracked("t1")
        assert not tracker.is_tracked("t2")

    def test_cleanup_finished_removes_dead_threads(self):
        tracker = _AsyncTaskTracker()
        tracker.register("t1", "a", threading.Thread(), threading.Event())
        tracker.mark_success("t1", "done")
        # Thread was never started, so is_alive() → False
        removed = tracker.cleanup_finished()
        assert removed == 1
        assert not tracker.is_tracked("t1")


# ---------------------------------------------------------------------------
# Tests: AsyncTaskData (Pydantic model)
# ---------------------------------------------------------------------------


class TestAsyncTaskData:
    def test_defaults(self):
        data = AsyncTaskData(
            task_id="abc",
            agent_name="deployer",
            status="running",
            created_at="2024-01-01T00:00:00Z",
        )
        assert data.progress_message is None
        assert data.percentage == 0.0
        assert data.result is None
        assert data.error_message is None
        assert data.finished_at is None
        assert data.cancelled_at is None

    def test_percentage_range(self):
        data = AsyncTaskData(
            task_id="x",
            agent_name="y",
            status="running",
            created_at="2024-01-01T00:00:00Z",
            percentage=0.75,
        )
        assert data.percentage == 0.75

    def test_status_literal(self):
        """Only allowed status values pass validation."""
        for good in ("running", "success", "error", "cancelled", "crashed"):
            data = AsyncTaskData(
                task_id="x",
                agent_name="y",
                status=good,
                created_at="2024-01-01T00:00:00Z",
            )
            assert data.status == good

    def test_invalid_status_raises(self):
        with pytest.raises(Exception):
            AsyncTaskData(
                task_id="x",
                agent_name="y",
                status="unknown",
                created_at="2024-01-01T00:00:00Z",
            )


# ---------------------------------------------------------------------------
# Tests: _format_status_message
# ---------------------------------------------------------------------------


class TestFormatStatusMessage:
    def test_running(self):
        data = AsyncTaskData(
            task_id="a1",
            agent_name="deployer",
            status="running",
            created_at="2024-01-01T00:00:00Z",
            progress_message="pulling image",
            percentage=0.5,
        )
        msg = _format_status_message(data)
        assert "a1" in msg
        assert "deployer" in msg
        assert "running" in msg
        assert "pulling image" in msg
        assert "50%" in msg

    def test_success(self):
        data = AsyncTaskData(
            task_id="a2",
            agent_name="deployer",
            status="success",
            created_at="2024-01-01T00:00:00Z",
            finished_at="2024-01-01T00:10:00Z",
            result="deployment successful",
        )
        msg = _format_status_message(data)
        assert "success" in msg.lower()
        assert "deployment successful" in msg

    def test_error(self):
        data = AsyncTaskData(
            task_id="a3",
            agent_name="deployer",
            status="error",
            created_at="2024-01-01T00:00:00Z",
            error_message="connection timeout",
        )
        msg = _format_status_message(data)
        assert "error" in msg.lower()
        assert "connection timeout" in msg

    def test_cancelled(self):
        data = AsyncTaskData(
            task_id="a4",
            agent_name="deployer",
            status="cancelled",
            created_at="2024-01-01T00:00:00Z",
            cancelled_at="2024-01-01T00:05:00Z",
        )
        msg = _format_status_message(data)
        assert "cancelled" in msg.lower()

    def test_crashed(self):
        data = AsyncTaskData(
            task_id="a5",
            agent_name="deployer",
            status="crashed",
            created_at="2024-01-01T00:00:00Z",
            error_message="系统重启，任务已丢失",
        )
        msg = _format_status_message(data)
        assert "crashed" in msg.lower()


# ---------------------------------------------------------------------------
# Tests: _generate_task_id
# ---------------------------------------------------------------------------


class TestGenerateTaskId:
    def test_format(self):
        tid = _generate_task_id()
        assert len(tid) == 8
        assert all(c in "0123456789abcdef" for c in tid)

    def test_uniqueness(self):
        ids = {_generate_task_id() for _ in range(100)}
        assert len(ids) == 100


# ---------------------------------------------------------------------------
# Tests: Thread-local context
# ---------------------------------------------------------------------------


class TestThreadLocalContext:
    def test_set_and_get(self):
        _set_current_async_task("my-task-123")
        assert _get_current_async_task() == "my-task-123"

    def test_isolation(self):
        """Thread-local is isolated across threads."""
        _set_current_async_task("main-task")
        result_container = []

        def _thread():
            assert _get_current_async_task() is None  # New thread → empty
            _set_current_async_task("bg-task")
            result_container.append(_get_current_async_task())

        t = threading.Thread(target=_thread)
        t.start()
        t.join()

        assert result_container == ["bg-task"]
        assert _get_current_async_task() == "main-task"  # main thread unchanged


# ---------------------------------------------------------------------------
# Tests: report_progress tool
# ---------------------------------------------------------------------------


class TestReportProgressTool:
    def test_updates_tracker(self):
        tracker = _AsyncTaskTracker()
        tracker.register("t1", "a", threading.Thread(), threading.Event())
        _set_current_async_task("t1")

        tool = _build_report_progress_tool(tracker)
        result = tool.invoke({"message": "step 1 done", "percentage": 0.3})
        assert "进度已汇报" in result

        snap = tracker.get_snapshot("t1")
        assert snap.progress_message == "step 1 done"
        assert snap.percentage == 0.3

    def test_outside_async_context(self):
        """Calling report_progress without thread-local context returns error."""
        tracker = _AsyncTaskTracker()
        _set_current_async_task(None)  # ensure clean

        tool = _build_report_progress_tool(tracker)
        result = tool.invoke({"message": "hello"})
        assert "错误" in result  # error message about missing context


# ---------------------------------------------------------------------------
# Tests: async_task tool
# ---------------------------------------------------------------------------


class TestAsyncTaskTool:
    def test_launch_returns_task_id(self):
        tracker = _AsyncTaskTracker()
        graphs = {"worker": _make_stub_runnable("worker", "job done", delay=0.1)}
        tool = _build_async_task_tool(graphs, "- worker: does work", tracker, default_timeout=10)

        runtime = _make_tool_runtime(state={"messages": [HumanMessage(content="do work")]})
        result = tool.invoke({"description": "do the thing", "subagent_type": "worker", "runtime": runtime})

        # Should be a Command
        from langgraph.types import Command
        assert isinstance(result, Command)

        # Extract task_id from the ToolMessage
        tool_msgs = [m for m in result.update["messages"] if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 1
        content = tool_msgs[0].content
        assert "后台任务已启动" in content
        # Verify task_id is present (8 hex chars)
        import re
        match = re.search(r"task_id: ([0-9a-f]{8})", content)
        assert match is not None
        task_id = match.group(1)

        # Verify async_tasks in Command update
        assert "async_tasks" in result.update
        assert task_id in result.update["async_tasks"]
        assert result.update["async_tasks"][task_id]["status"] == "running"

    def test_unknown_subagent_type(self):
        tracker = _AsyncTaskTracker()
        graphs = {"worker": _make_stub_runnable("worker")}
        tool = _build_async_task_tool(graphs, "- worker: does work", tracker, default_timeout=10)

        runtime = _make_tool_runtime()
        result = tool.invoke({
            "description": "do the thing",
            "subagent_type": "nonexistent",
            "runtime": runtime,
        })
        assert "未知" in str(result)

    def test_background_thread_completes(self):
        tracker = _AsyncTaskTracker()
        graphs = {"worker": _make_stub_runnable("worker", "all done", delay=0.05)}
        tool = _build_async_task_tool(graphs, "- worker: does work", tracker, default_timeout=10)

        runtime = _make_tool_runtime()
        from langgraph.types import Command
        result = tool.invoke({"description": "work", "subagent_type": "worker", "runtime": runtime})
        assert isinstance(result, Command)

        # Wait for background thread to finish
        time.sleep(0.3)

        # Task should be marked success
        import re
        content = result.update["messages"][0].content
        match = re.search(r"task_id: ([0-9a-f]{8})", content)
        task_id = match.group(1)
        snap = tracker.get_snapshot(task_id)
        assert snap is not None
        assert snap.status == "success"
        assert snap.result == "all done"


# ---------------------------------------------------------------------------
# Tests: async_status tool
# ---------------------------------------------------------------------------


class TestAsyncStatusTool:
    def test_running_task(self):
        tracker = _AsyncTaskTracker()
        t = threading.Thread()
        tracker.register("abc", "deployer", t, threading.Event())
        tracker.update_progress("abc", "pulling image", 0.4)

        tool = _build_async_status_tool(tracker)
        runtime = _make_tool_runtime()
        result = tool.invoke({"task_id": "abc", "runtime": runtime})
        assert "running" in result
        assert "pulling image" in result

    def test_success_task(self):
        tracker = _AsyncTaskTracker()
        t = threading.Thread()
        tracker.register("abc", "deployer", t, threading.Event())
        tracker.mark_success("abc", "deployment complete")

        tool = _build_async_status_tool(tracker)
        runtime = _make_tool_runtime()
        result = tool.invoke({"task_id": "abc", "runtime": runtime})
        assert "success" in result.lower()
        assert "deployment complete" in result

    def test_error_task(self):
        tracker = _AsyncTaskTracker()
        t = threading.Thread()
        tracker.register("abc", "deployer", t, threading.Event())
        tracker.mark_error("abc", "kubectl timeout")

        tool = _build_async_status_tool(tracker)
        runtime = _make_tool_runtime()
        result = tool.invoke({"task_id": "abc", "runtime": runtime})
        assert "error" in result.lower()
        assert "kubectl timeout" in result

    def test_not_found_no_state(self):
        tracker = _AsyncTaskTracker()
        tool = _build_async_status_tool(tracker)
        runtime = _make_tool_runtime()
        result = tool.invoke({"task_id": "ghost", "runtime": runtime})
        assert "未找到" in result

    def test_crash_recovery_orphaned_task(self):
        """Orphaned 'running' task in state but not tracker → marked crashed."""
        tracker = _AsyncTaskTracker()
        tool = _build_async_status_tool(tracker)

        state = {
            "messages": [],
            "async_tasks": {
                "orphan-1": {
                    "task_id": "orphan-1",
                    "agent_name": "deployer",
                    "status": "running",
                    "created_at": "2024-01-01T00:00:00Z",
                }
            },
        }
        runtime = _make_tool_runtime(state=state)
        result = tool.invoke({"task_id": "orphan-1", "runtime": runtime})

        # Should be a Command (state update) marking task as crashed
        from langgraph.types import Command
        assert isinstance(result, Command)
        assert "async_tasks" in result.update
        assert result.update["async_tasks"]["orphan-1"]["status"] == "crashed"
        assert "系统重启" in result.update["async_tasks"]["orphan-1"]["error_message"]


# ---------------------------------------------------------------------------
# Tests: async_list tool
# ---------------------------------------------------------------------------


class TestAsyncListTool:
    def test_empty(self):
        tracker = _AsyncTaskTracker()
        tool = _build_async_list_tool(tracker)
        runtime = _make_tool_runtime()
        result = tool.invoke({"runtime": runtime})
        assert "没有" in str(result)

    def test_lists_tasks(self):
        tracker = _AsyncTaskTracker()
        tracker.register("t1", "a", threading.Thread(), threading.Event())
        tracker.register("t2", "b", threading.Thread(), threading.Event())
        tracker.mark_success("t1", "ok")

        tool = _build_async_list_tool(tracker)
        runtime = _make_tool_runtime()
        result = tool.invoke({"runtime": runtime})
        assert "t1" in str(result)
        assert "t2" in str(result)

    def test_detects_orphaned_tasks(self):
        """async_list detects orphaned 'running' tasks from checkpoint state."""
        tracker = _AsyncTaskTracker()
        tool = _build_async_list_tool(tracker)

        state = {
            "messages": [],
            "async_tasks": {
                "orphan-1": {
                    "task_id": "orphan-1",
                    "agent_name": "deployer",
                    "status": "running",
                    "created_at": "2024-01-01T00:00:00Z",
                },
                "orphan-2": {
                    "task_id": "orphan-2",
                    "agent_name": "builder",
                    "status": "success",
                    "created_at": "2024-01-01T00:00:00Z",
                    "result": "build done",
                },
            },
        }
        runtime = _make_tool_runtime(state=state)
        result = tool.invoke({"runtime": runtime})

        from langgraph.types import Command
        assert isinstance(result, Command)
        assert "async_tasks" in result.update
        # orphan-1 should be marked crashed
        assert result.update["async_tasks"]["orphan-1"]["status"] == "crashed"
        # List output should mention both
        content = result.update["messages"][0].content
        assert "orphan-1" in content
        assert "orphan-2" in content


# ---------------------------------------------------------------------------
# Tests: async_cancel tool
# ---------------------------------------------------------------------------


class TestAsyncCancelTool:
    def test_cancel_running_task(self):
        tracker = _AsyncTaskTracker()
        cancel_ev = threading.Event()
        tracker.register("t1", "a", threading.Thread(), cancel_ev)

        tool = _build_async_cancel_tool(tracker)
        runtime = _make_tool_runtime()
        result = tool.invoke({"task_id": "t1", "runtime": runtime})

        from langgraph.types import Command
        assert isinstance(result, Command)
        assert "取消" in result.update["messages"][0].content
        assert cancel_ev.is_set()

    def test_cancel_nonexistent(self):
        tracker = _AsyncTaskTracker()
        tool = _build_async_cancel_tool(tracker)
        runtime = _make_tool_runtime()
        result = tool.invoke({"task_id": "ghost", "runtime": runtime})
        assert "未找到" in result

    def test_cancel_already_completed(self):
        tracker = _AsyncTaskTracker()
        tracker.register("t1", "a", threading.Thread(), threading.Event())
        tracker.mark_success("t1", "done")

        tool = _build_async_cancel_tool(tracker)
        runtime = _make_tool_runtime()
        result = tool.invoke({"task_id": "t1", "runtime": runtime})
        assert "success" in result.lower()


# ---------------------------------------------------------------------------
# Tests: _run_subagent_in_thread
# ---------------------------------------------------------------------------


class TestRunSubagentInThread:
    def test_normal_completion(self):
        tracker = _AsyncTaskTracker()
        cancel_ev = threading.Event()
        task_id = "my-task"
        tracker.register(task_id, "test", threading.Thread(), cancel_ev)

        runnable = _make_stub_runnable("test", "hello world", delay=0.01)
        _run_subagent_in_thread(
            runnable=runnable,
            state={"messages": [HumanMessage(content="go")]},
            config={"configurable": {}},
            task_id=task_id,
            tracker=tracker,
            cancel_event=cancel_ev,
            timeout=10,
        )

        snap = tracker.get_snapshot(task_id)
        assert snap.status == "success"
        assert "hello world" in (snap.result or "")

    def test_timeout(self):
        tracker = _AsyncTaskTracker()
        cancel_ev = threading.Event()
        task_id = "timeout-task"
        tracker.register(task_id, "test", threading.Thread(), cancel_ev)

        # Runnable that takes 1 second but timeout is 0.01
        runnable = _make_stub_runnable("test", "too slow", delay=1.0)
        _run_subagent_in_thread(
            runnable=runnable,
            state={"messages": [HumanMessage(content="go")]},
            config={"configurable": {}},
            task_id=task_id,
            tracker=tracker,
            cancel_event=cancel_ev,
            timeout=0.01,
        )

        snap = tracker.get_snapshot(task_id)
        assert snap.status == "error"
        assert "超时" in (snap.error_message or "")

    def test_cancellation(self):
        tracker = _AsyncTaskTracker()
        cancel_ev = threading.Event()
        task_id = "cancel-task"
        tracker.register(task_id, "test", threading.Thread(), cancel_ev)

        # Runnable that takes a while — we'll cancel it externally
        runnable = _make_stub_runnable("test", "result", delay=2.0)

        # Run in background thread
        def _run():
            _run_subagent_in_thread(
                runnable=runnable,
                state={"messages": [HumanMessage(content="go")]},
                config={"configurable": {}},
                task_id=task_id,
                tracker=tracker,
                cancel_event=cancel_ev,
                timeout=10,
            )

        t = threading.Thread(target=_run)
        t.start()
        time.sleep(0.05)  # Let it start running
        cancel_ev.set()  # Signal cancellation
        t.join(timeout=3)

        snap = tracker.get_snapshot(task_id)
        assert snap.status in ("cancelled", "error")


# ---------------------------------------------------------------------------
# Tests: _append_to_system_message
# ---------------------------------------------------------------------------


class TestAppendToSystemMessage:
    def test_append_to_str_content(self):
        from langchain.agents.middleware.types import ModelRequest
        from langchain_core.messages import SystemMessage

        req = ModelRequest(
            system_message=SystemMessage(content="original"),
            messages=[],
            tools=[],
            model="test-model",
        )
        result = _append_to_system_message(req, "appended")
        content = result.system_message.content
        assert "original" in str(content)
        assert "appended" in str(content)

    def test_none_system_message(self):
        from langchain.agents.middleware.types import ModelRequest

        req = ModelRequest(
            system_message=None,
            messages=[],
            tools=[],
            model="test-model",
        )
        result = _append_to_system_message(req, "only")
        assert "only" in str(result.system_message.content)


# ---------------------------------------------------------------------------
# Tests: AsyncSubAgentMiddleware initialization
# ---------------------------------------------------------------------------


class TestAsyncSubAgentMiddlewareInit:
    def test_requires_at_least_one_subagent(self):
        backend = StoreBackend(store=InMemoryStore())
        with pytest.raises(ValueError, match="At least one async_subagent"):
            AsyncSubAgentMiddleware(backend=backend, async_subagents=[])

    def test_basic_initialization(self):
        backend = StoreBackend(store=InMemoryStore())
        mw = AsyncSubAgentMiddleware(
            backend=backend,
            async_subagents=[_stub_subagent_spec("deployer")],
        )
        assert mw is not None
        assert len(mw.tools) == 4  # async_task, async_status, async_list, async_cancel
        tool_names = {t.name for t in mw.tools}
        assert tool_names == {"async_task", "async_status", "async_list", "async_cancel"}

    def test_duplicate_names_raises(self):
        backend = StoreBackend(store=InMemoryStore())
        with pytest.raises(ValueError, match="Duplicate"):
            AsyncSubAgentMiddleware(
                backend=backend,
                async_subagents=[
                    _stub_subagent_spec("deployer"),
                    _stub_subagent_spec("deployer"),
                ],
            )

    def test_multiple_subagent_types(self):
        backend = StoreBackend(store=InMemoryStore())
        mw = AsyncSubAgentMiddleware(
            backend=backend,
            async_subagents=[
                _stub_subagent_spec("deployer", "deploy things"),
                _stub_subagent_spec("builder", "build things"),
            ],
        )
        assert mw is not None

    def test_system_prompt_injected(self):
        backend = StoreBackend(store=InMemoryStore())
        mw = AsyncSubAgentMiddleware(
            backend=backend,
            async_subagents=[_stub_subagent_spec("deployer")],
        )
        assert mw._system_prompt is not None
        assert "deployer" in mw._system_prompt

    def test_wrap_model_call_injects_prompt(self):
        from langchain.agents.middleware.types import ModelRequest
        from langchain_core.messages import SystemMessage

        backend = StoreBackend(store=InMemoryStore())
        mw = AsyncSubAgentMiddleware(
            backend=backend,
            async_subagents=[_stub_subagent_spec("deployer")],
        )

        req = ModelRequest(
            system_message=SystemMessage(content="base prompt"),
            messages=[],
            tools=[],
            model="test-model",
        )

        def _handler(r):
            return r  # echo back

        result = mw.wrap_model_call(req, _handler)
        content = str(result.system_message.content)
        assert "异步子代理" in content
        assert "deployer" in content


# ---------------------------------------------------------------------------
# Tests: Thread safety (concurrent tasks)
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_multiple_tasks(self):
        """Launch multiple async tasks concurrently and verify all complete."""
        tracker = _AsyncTaskTracker()
        graphs = {
            "worker": _make_stub_runnable("worker", "done", delay=0.05),
        }
        tool = _build_async_task_tool(graphs, "- worker: works", tracker, default_timeout=10)

        task_ids = []
        for i in range(5):
            runtime = _make_tool_runtime(tool_call_id=f"call-{i}")
            from langgraph.types import Command
            result = tool.invoke({
                "description": f"task {i}",
                "subagent_type": "worker",
                "runtime": runtime,
            })
            import re
            content = result.update["messages"][0].content
            tid = re.search(r"task_id: ([0-9a-f]{8})", content).group(1)
            task_ids.append(tid)

        # Wait for all to complete
        deadline = time.time() + 5
        while time.time() < deadline:
            all_done = all(
                tracker.get_snapshot(tid).status != "running"
                for tid in task_ids
            )
            if all_done:
                break
            time.sleep(0.1)

        for tid in task_ids:
            snap = tracker.get_snapshot(tid)
            assert snap.status == "success"

    def test_thread_isolation_of_report_progress(self):
        """Each background thread sees only its own task_id via thread-local."""
        task_ids_seen = {}

        def _worker(task_id):
            _set_current_async_task(task_id)
            seen = _get_current_async_task()
            task_ids_seen[task_id] = seen
            time.sleep(0.05)
            still_seen = _get_current_async_task()
            task_ids_seen[f"{task_id}-again"] = still_seen

        threads = []
        for i in range(3):
            tid = f"thread-{i}"
            t = threading.Thread(target=_worker, args=(tid,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert task_ids_seen["thread-0"] == "thread-0"
        assert task_ids_seen["thread-1"] == "thread-1"
        assert task_ids_seen["thread-2"] == "thread-2"
        assert task_ids_seen["thread-0-again"] == "thread-0"
