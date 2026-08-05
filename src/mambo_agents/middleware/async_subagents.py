"""Middleware for async (background) subagents that run locally.

Unlike ``SubAgentMiddleware`` which blocks until the subagent completes,
async subagents run in a background thread.  The main agent launches a task
via ``async_task()``, receives a ``task_id`` immediately, and can later
query progress/results via ``async_status(task_id)``.

Key differences from the sync ``task`` tool:
- ``async_task`` returns a ``task_id`` immediately; the subagent runs in the
  background.
- The subagent can call ``report_progress(message, percentage)`` to report
  intermediate progress.
- ``async_status(task_id)`` returns current status: ``running`` (with
  progress), ``success`` (with result), ``error``, ``cancelled``, or
  ``crashed``.
- ``async_list(status_filter)`` rediscover tasks when the LLM forgets
  ``task_id`` (e.g. after context compaction).

Crash recovery: on system restart, tasks that were ``running`` before the
crash are detected via checkpointer state and marked as ``crashed``.

.. warning::

    **Experimental (实验性):** 本模块使用较少、测试覆盖不足，属于实验性质。
    API 可能在没有弃用期的情况下直接变更或移除，请勿在关键路径依赖它。
    实例化 :class:`AsyncSubAgentMiddleware` 时会发出 ``FutureWarning``。
"""

from __future__ import annotations

import threading
import time
import uuid
import warnings
from collections.abc import Awaitable, Callable, Sequence
from typing import Annotated, Any, Literal, NotRequired

from langchain.agents.factory import create_agent as _langchain_create_agent
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelRequest,
    ModelResponse,
    ResponseT,
)
from langchain.tools import BaseTool, ToolRuntime
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import StructuredTool
from langgraph.types import Command, Overwrite
from pydantic import BaseModel, ConfigDict, Field

from mambo_agents.backends.protocol import BackendProtocol
from mambo_agents.middleware.subagents import CompiledSubAgent, SubAgent, _EXCLUDED_STATE_KEYS, _SubagentSpec

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class AsyncTaskData(BaseModel):
    """Serialisable snapshot of a single async subagent task."""

    task_id: str
    agent_name: str
    status: Literal["running", "success", "error", "cancelled", "crashed"]
    progress_message: str | None = Field(default=None)
    percentage: float = Field(default=0.0, ge=0.0, le=1.0)
    result: str | None = Field(default=None)
    error_message: str | None = Field(default=None)
    created_at: str
    finished_at: str | None = Field(default=None)
    cancelled_at: str | None = Field(default=None)


# ---- Tool input schemas --------------------------------------------------


class AsyncTaskInput(BaseModel):
    """Input for ``async_task``."""

    model_config = ConfigDict(extra="allow")

    description: str = Field(
        description=(
            "任务的详细描述。子代理会一次性收到这个描述并自主执行。"
            "包含所有上下文和预期的输出格式。"
        )
    )
    subagent_type: str = Field(
        description="要使用的子代理类型。必须是可用类型之一。"
    )


class AsyncStatusInput(BaseModel):
    """Input for ``async_status``."""

    model_config = ConfigDict(extra="allow")

    task_id: str = Field(
        description=(
            "async_task 返回的 task_id。必须是完整精确的值，"
            "如 'a3f4b2c1'。不要截断或推测。"
        )
    )


class AsyncListInput(BaseModel):
    """Input for ``async_list``."""

    model_config = ConfigDict(extra="allow")

    status_filter: Literal["running", "success", "error", "cancelled", "crashed", "all"] | None = Field(
        default=None,
        description="可选的状态过滤。默认返回全部。",
    )


class AsyncCancelInput(BaseModel):
    """Input for ``async_cancel``."""

    model_config = ConfigDict(extra="allow")

    task_id: str = Field(
        description="要取消的任务的 task_id。必须是完整精确的值。"
    )


class ReportProgressInput(BaseModel):
    """Input for ``report_progress`` (injected into subagents)."""

    message: str = Field(description="当前进度描述，如 '正在拉取镜像...'")
    percentage: float | None = Field(
        default=None,
        description="可选，0-1之间的完成度。如 0.3 表示 30%。",
    )


# ---------------------------------------------------------------------------
# AsyncTaskTracker — thread-safe state manager
# ---------------------------------------------------------------------------


class _AsyncTaskTracker:
    """Thread-safe in-memory tracker for async subagent tasks.

    Tracks live tasks (with their threads and cancel events).  Serialisable
    snapshots are written to agent state via ``Command.update`` so that
    checkpointer-based crash recovery can detect orphaned tasks after a
    restart.

    After a system restart this tracker is empty.  Tasks that appear
    ``running`` in agent state but are absent from the tracker are re-marked
    as ``crashed`` by the tools that encounter them.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # Live tasks: {task_id: {"data": AsyncTaskData, "thread": Thread, "cancel": Event}}
        self._live: dict[str, dict[str, Any]] = {}

    # -- mutators ----------------------------------------------------------

    def register(
        self,
        task_id: str,
        agent_name: str,
        thread: threading.Thread,
        cancel_event: threading.Event,
    ) -> AsyncTaskData:
        """Register a newly launched task."""
        now = _utc_now()
        data = AsyncTaskData(
            task_id=task_id,
            agent_name=agent_name,
            status="running",
            progress_message=None,
            percentage=0.0,
            result=None,
            error_message=None,
            created_at=now,
        )
        with self._lock:
            self._live[task_id] = {
                "data": data,
                "thread": thread,
                "cancel": cancel_event,
            }
        return data

    def update_progress(
        self, task_id: str, message: str, percentage: float | None
    ) -> None:
        """Called by the subagent's ``report_progress`` tool."""
        with self._lock:
            entry = self._live.get(task_id)
            if entry is None:
                return
            data: AsyncTaskData = entry["data"]
            data.progress_message = message
            if percentage is not None:
                data.percentage = max(0.0, min(1.0, percentage))

    def mark_success(self, task_id: str, result: str) -> None:
        with self._lock:
            entry = self._live.get(task_id)
            if entry is None:
                return
            entry["data"].status = "success"
            entry["data"].result = result
            entry["data"].finished_at = _utc_now()

    def mark_error(self, task_id: str, error: str) -> None:
        with self._lock:
            entry = self._live.get(task_id)
            if entry is None:
                return
            entry["data"].status = "error"
            entry["data"].error_message = error
            entry["data"].finished_at = _utc_now()

    def mark_cancelled(self, task_id: str) -> None:
        with self._lock:
            entry = self._live.get(task_id)
            if entry is None:
                return
            entry["data"].status = "cancelled"
            entry["data"].cancelled_at = _utc_now()

    def request_cancel(self, task_id: str) -> bool:
        """Set the cancel event for a running task.  Returns True if the
        task exists and was running."""
        with self._lock:
            entry = self._live.get(task_id)
            if entry is None or entry["data"].status != "running":
                return False
            entry["cancel"].set()
            return True

    # -- accessors ---------------------------------------------------------

    def get_snapshot(self, task_id: str) -> AsyncTaskData | None:
        """Return a copy of the tracked data or None."""
        with self._lock:
            entry = self._live.get(task_id)
            if entry is None:
                return None
            return entry["data"].model_copy(deep=True)

    def list_snapshots(
        self, status_filter: str | None = None
    ) -> list[AsyncTaskData]:
        """Return copies of all tracked tasks, optionally filtered."""
        with self._lock:
            items = list(self._live.values())
        result = [e["data"].model_copy(deep=True) for e in items]
        if status_filter and status_filter != "all":
            result = [d for d in result if d.status == status_filter]
        return result

    def is_tracked(self, task_id: str) -> bool:
        with self._lock:
            return task_id in self._live

    def cleanup_finished(self) -> int:
        """Remove completed/errored/cancelled tasks that are no longer
        needed by live threads.  Returns count of removed entries."""
        with self._lock:
            to_remove = [
                tid
                for tid, entry in self._live.items()
                if entry["data"].status in ("success", "error", "cancelled", "crashed")
                and not entry["thread"].is_alive()
            ]
            for tid in to_remove:
                del self._live[tid]
            return len(to_remove)

    def shutdown(self) -> None:
        """Cancel all running tasks and wait for their threads to join."""
        with self._lock:
            entries = list(self._live.items())
        for tid, entry in entries:
            if entry["data"].status == "running":
                entry["cancel"].set()
        for tid, entry in entries:
            entry["thread"].join(timeout=5.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    """ISO-8601 UTC timestamp with second precision."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _generate_task_id() -> str:
    """Short human-readable task id (8 hex chars, UUID4-derived)."""
    return uuid.uuid4().hex[:8]


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------


def _async_tasks_reducer(
    existing: dict[str, dict[str, Any]] | None,
    update: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Merge async_tasks updates into the existing dict."""
    merged = dict(existing or {})
    merged.update(update)
    return merged


class AsyncSubAgentState(AgentState):
    """Agent state extended with ``async_tasks`` for async subagent tracking."""

    async_tasks: Annotated[
        NotRequired[dict[str, dict[str, Any]]], _async_tasks_reducer
    ]
    """Dict of ``task_id`` → serialised ``AsyncTaskData``."""


# ---------------------------------------------------------------------------
# Thread-local context for report_progress
# ---------------------------------------------------------------------------

_current_async_task = threading.local()


def _set_current_async_task(task_id: str) -> None:
    _current_async_task.task_id = task_id


def _get_current_async_task() -> str | None:
    return getattr(_current_async_task, "task_id", None)


# ---------------------------------------------------------------------------
# report_progress tool (injected into subagents)
# ---------------------------------------------------------------------------


def _build_report_progress_tool(tracker: _AsyncTaskTracker) -> StructuredTool:
    """Build the ``report_progress`` tool that subagents use internally."""

    def report_progress(
        message: str,
        percentage: float | None = None,
    ) -> str:
        task_id = _get_current_async_task()
        if task_id is None:
            return "错误：report_progress 只能在异步子代理内部调用。"
        tracker.update_progress(task_id, message, percentage)
        return f"进度已汇报"

    return StructuredTool.from_function(
        name="report_progress",
        func=report_progress,
        description=(
            "汇报当前任务的进度。当你完成一个阶段时调用此工具。"
            "message: 进度描述（如 '正在拉取镜像...'），"
            "percentage: 可选，0-1之间的完成度。"
        ),
        args_schema=ReportProgressInput,
    )


# ---------------------------------------------------------------------------
# async_task tool
# ---------------------------------------------------------------------------

ASYNC_TASK_TOOL_DESCRIPTION = """启动后台子代理任务。子代理在后台独立运行，该工具立即返回 task_id。

可用子代理类型：
{available_agents}

使用说明：
1. 调用后立即返回 task_id（如 'a3f4b2c1'）
2. **必须将 task_id 完整告知用户**，例如"已启动后台任务 a3f4b2c1"
3. 启动后不要立即查询状态——返回控制权给用户
4. 子代理可在后台调用 report_progress 汇报进度
5. 用户想查询时使用 async_status(task_id) 获取状态"""


def _build_async_task_tool(
    subagent_graphs: dict[str, Runnable],
    subagent_descriptions: str,
    tracker: _AsyncTaskTracker,
    default_timeout: float,
) -> StructuredTool:
    """Build the ``async_task`` tool."""

    description = ASYNC_TASK_TOOL_DESCRIPTION.format(
        available_agents=subagent_descriptions
    )

    def async_task(
        description: str,
        subagent_type: str,
        runtime: ToolRuntime,
        **_extra: object,
    ) -> str | Command:
        if subagent_type not in subagent_graphs:
            allowed = ", ".join(f"`{k}`" for k in subagent_graphs)
            return (
                f"未知的子代理类型 {subagent_type}。可用类型: {allowed}"
            )
        if not runtime.tool_call_id:
            raise ValueError("Tool call ID is required for async_task")

        task_id = _generate_task_id()
        cancel_event = threading.Event()

        # Prepare subagent state (same logic as sync subagents)
        subagent_state = {
            k: v
            for k, v in runtime.state.items()
            if k not in _EXCLUDED_STATE_KEYS
        }
        subagent_state["messages"] = [HumanMessage(content=description)]

        subagent_config: RunnableConfig = {
            "configurable": {
                **runtime.config.get("configurable", {}),
                "ls_agent_type": "async_subagent",
            }
        }

        runnable = subagent_graphs[subagent_type]

        thread = threading.Thread(
            target=_run_subagent_in_thread,
            args=(
                runnable,
                subagent_state,
                subagent_config,
                task_id,
                tracker,
                cancel_event,
                default_timeout,
            ),
            daemon=True,
            name=f"async_subagent_{task_id}",
        )

        data = tracker.register(task_id, subagent_type, thread, cancel_event)
        thread.start()

        result_msg = (
            f"✅ 后台任务已启动\n\n"
            f"task_id: {task_id}\n"
            f"agent: {subagent_type}\n"
            f"status: running\n\n"
            f"⚠️ 请向用户展示此 task_id {task_id}，方便用户后续追踪。"
        )
        return Command(
            update={
                "messages": [
                    ToolMessage(result_msg, tool_call_id=runtime.tool_call_id)
                ],
                "async_tasks": {task_id: data.model_dump()},
            }
        )

    async def aasync_task(
        description: str,
        subagent_type: str,
        runtime: ToolRuntime,
        **_extra: object,
    ) -> str | Command:
        # Async variant delegates to sync (thread creation is sync by nature)
        return async_task(description, subagent_type, runtime)

    return StructuredTool.from_function(
        name="async_task",
        func=async_task,
        coroutine=aasync_task,
        description=description,
        infer_schema=False,
        args_schema=AsyncTaskInput,
    )


# ---------------------------------------------------------------------------
# async_status tool
# ---------------------------------------------------------------------------


def _build_async_status_tool(tracker: _AsyncTaskTracker) -> StructuredTool:
    """Build the ``async_status`` tool."""

    def async_status(
        task_id: str,
        runtime: ToolRuntime,
        **_extra: object,
    ) -> str | Command:
        data = tracker.get_snapshot(task_id)

        if data is None:
            # Check agent state for orphaned task (crash recovery)
            tasks_state: dict[str, dict[str, Any]] = (
                runtime.state.get("async_tasks") or {}
            )
            raw = tasks_state.get(task_id)
            if raw is None:
                return f"未找到任务 {task_id}。使用 async_list 查看所有任务。"
            if raw.get("status") == "running":
                # Orphaned task → mark as crashed in state
                raw["status"] = "crashed"
                raw["error_message"] = "系统重启，任务已丢失"
                msg = (
                    f"💥 任务已丢失（系统重启）\n\n"
                    f"task_id: {task_id}\n"
                    f"agent: {raw.get('agent_name', 'unknown')}\n"
                    f"status: crashed\n"
                    f"原因: 系统在任务执行期间发生了重启，该任务已不可恢复。\n"
                    f"建议: 重新启动该任务。"
                )
                return Command(
                    update={
                        "messages": [ToolMessage(msg, tool_call_id=runtime.tool_call_id)],
                        "async_tasks": {task_id: raw},
                    }
                )
            # Task exists but with non-running status – just return the
            # state data as-is (e.g. was marked success/error before)
            data = AsyncTaskData(**raw)

        msg = _format_status_message(data)
        return msg

    async def aasync_status(
        task_id: str,
        runtime: ToolRuntime,
        **_extra: object,
    ) -> str | Command:
        return async_status(task_id, runtime)

    return StructuredTool.from_function(
        name="async_status",
        func=async_status,
        coroutine=aasync_status,
        description=(
            "查询指定异步子代理任务的当前状态。"
            "task_id 必须是 async_task 返回的精确值。"
            "返回状态含 running（含进度）、success（含结果）、error、cancelled、crashed。"
        ),
        infer_schema=False,
        args_schema=AsyncStatusInput,
    )


def _format_status_message(data: AsyncTaskData) -> str:
    """Format an ``AsyncTaskData`` into a human+LLM-readable string."""
    status = data.status

    if status == "running":
        progress = data.progress_message or "无进度信息"
        pct = f"{data.percentage:.0%}" if data.percentage else "N/A"
        return (
            f"🔵 任务仍在执行中\n\n"
            f"task_id: {data.task_id}\n"
            f"agent: {data.agent_name}\n"
            f"status: running\n"
            f"progress: {progress}\n"
            f"完成度: {pct}\n"
            f"启动时间: {data.created_at}\n\n"
            f"提示：任务还在后台运行，你可以稍后再次查询。"
        )

    if status == "success":
        result = data.result or "(无输出)"
        return (
            f"✅ 任务已完成\n\n"
            f"task_id: {data.task_id}\n"
            f"agent: {data.agent_name}\n"
            f"status: success\n"
            f"完成时间: {data.finished_at}\n\n"
            f"--- 子代理输出 ---\n"
            f"{result}\n\n"
            f"提示：请将以上结果消化后用自然语言告知用户。"
        )

    if status == "error":
        error = data.error_message or "未知错误"
        return (
            f"❌ 任务执行失败\n\n"
            f"task_id: {data.task_id}\n"
            f"agent: {data.agent_name}\n"
            f"status: error\n"
            f"错误信息: {error}\n\n"
            f"提示：请分析错误原因并告知用户，必要时建议重新启动任务。"
        )

    if status == "cancelled":
        return (
            f"⚪ 任务已取消\n\n"
            f"task_id: {data.task_id}\n"
            f"agent: {data.agent_name}\n"
            f"status: cancelled\n"
            f"取消时间: {data.cancelled_at or 'N/A'}"
        )

    if status == "crashed":
        return (
            f"💥 任务已丢失（系统重启）\n\n"
            f"task_id: {data.task_id}\n"
            f"agent: {data.agent_name}\n"
            f"status: crashed\n"
            f"建议: 重新启动该任务。"
        )

    return f"未知状态: {status}"


# ---------------------------------------------------------------------------
# async_list tool
# ---------------------------------------------------------------------------


def _build_async_list_tool(tracker: _AsyncTaskTracker) -> StructuredTool:
    """Build the ``async_list`` tool."""

    def async_list(
        runtime: ToolRuntime,
        status_filter: str | None = None,
        **_extra: object,
    ) -> str | Command:
        # 1) Live tasks from tracker
        tasks = tracker.list_snapshots(status_filter if status_filter != "all" else None)

        # 2) Also scan agent state for orphaned / completed tasks not in tracker
        tasks_state: dict[str, dict[str, Any]] = (
            runtime.state.get("async_tasks") or {}
        )
        tracked_ids = {t.task_id for t in tasks}
        crashed_updates: dict[str, dict[str, Any]] = {}

        for tid, raw in tasks_state.items():
            if tid in tracked_ids:
                continue
            if raw.get("status") == "running":
                raw["status"] = "crashed"
                raw["error_message"] = "系统重启，任务已丢失"
                crashed_updates[tid] = raw
                try:
                    tasks.append(AsyncTaskData(**raw))
                except Exception:
                    # Skip malformed entries
                    pass
            else:
                try:
                    tasks.append(AsyncTaskData(**raw))
                except Exception:
                    pass

        if not tasks:
            return "当前没有异步子代理任务。"

        lines = [f"共 {len(tasks)} 个异步任务：\n"]
        for t in tasks:
            icon = {
                "running": "🔵",
                "success": "✅",
                "error": "❌",
                "cancelled": "⚪",
                "crashed": "💥",
            }.get(t.status, "❓")
            progress = t.progress_message or "N/A"
            lines.append(
                f"{icon} task_id: {t.task_id}  "
                f"agent: {t.agent_name}  "
                f"status: {t.status}  "
                f"progress: {progress}"
            )

        msg = "\n".join(lines)

        if crashed_updates:
            return Command(
                update={
                    "messages": [ToolMessage(msg, tool_call_id=runtime.tool_call_id)],
                    "async_tasks": crashed_updates,
                }
            )
        return msg

    async def aasync_list(
        runtime: ToolRuntime,
        status_filter: str | None = None,
        **_extra: object,
    ) -> str | Command:
        return async_list(runtime, status_filter)

    return StructuredTool.from_function(
        name="async_list",
        func=async_list,
        coroutine=aasync_list,
        description=(
            "列出所有异步子代理任务及其当前状态。"
            "当你忘记了 task_id 时使用此工具找回。"
            "可用 status_filter 过滤：running/success/error/cancelled/crashed"
        ),
        infer_schema=False,
        args_schema=AsyncListInput,
    )


# ---------------------------------------------------------------------------
# async_cancel tool
# ---------------------------------------------------------------------------


def _build_async_cancel_tool(tracker: _AsyncTaskTracker) -> StructuredTool:
    """Build the ``async_cancel`` tool."""

    def async_cancel(
        task_id: str,
        runtime: ToolRuntime,
        **_extra: object,
    ) -> str | Command:
        data = tracker.get_snapshot(task_id)
        if data is None:
            return f"未找到任务 {task_id}。"

        if data.status != "running":
            return f"任务 {task_id} 当前状态为 {data.status}，无法取消。"

        cancelled = tracker.request_cancel(task_id)
        if not cancelled:
            return f"无法取消任务 {task_id}。"

        # Build state update preserving all existing fields
        state_data = data.model_dump()
        state_data["status"] = "cancelled"
        state_data["cancelled_at"] = _utc_now()

        msg = (
            f"⚪ 任务取消请求已发送\n\n"
            f"task_id: {task_id}\n"
            f"agent: {data.agent_name}\n"
            f"注意：任务会在下一个 report_progress 调用时终止。"
        )
        return Command(
            update={
                "messages": [ToolMessage(msg, tool_call_id=runtime.tool_call_id)],
                "async_tasks": {task_id: state_data},
            }
        )

    async def aasync_cancel(
        task_id: str,
        runtime: ToolRuntime,
        **_extra: object,
    ) -> str | Command:
        return async_cancel(task_id, runtime)

    return StructuredTool.from_function(
        name="async_cancel",
        func=async_cancel,
        coroutine=aasync_cancel,
        description="取消运行中的异步子代理任务。task_id 必须是 async_task 返回的精確值。",
        infer_schema=False,
        args_schema=AsyncCancelInput,
    )


# ---------------------------------------------------------------------------
# Background thread runner
# ---------------------------------------------------------------------------

def _run_subagent_in_thread(
    runnable: Runnable,
    state: dict[str, Any],
    config: RunnableConfig,
    task_id: str,
    tracker: _AsyncTaskTracker,
    cancel_event: threading.Event,
    timeout: float,
) -> None:
    """Run a subagent in a background thread.

    Sets thread-local context so that ``report_progress`` knows which task
    it belongs to.  Periodically checks the cancel_event during execution.
    """
    _set_current_async_task(task_id)
    start_time = time.monotonic()
    try:
        # Run the subagent with streaming to allow cancellation checks
        final_state: dict[str, Any] = {}
        for chunk in runnable.stream(state, config, stream_mode=["values"]):
            # Check for timeout
            if time.monotonic() - start_time > timeout:
                tracker.mark_error(task_id, f"任务超时（{timeout:.0f}秒）")
                return
            # Check for cancellation
            if cancel_event.is_set():
                tracker.mark_cancelled(task_id)
                return
            # Update final state with latest chunk
            if isinstance(chunk, tuple) and len(chunk) == 2:
                _, data = chunk
                final_state = data if isinstance(data, dict) else {}
            else:
                final_state = chunk if isinstance(chunk, dict) else {}

        # Subagent completed successfully
        messages = final_state.get("messages", [])
        if messages:
            last_msg = messages[-1]
            result = (
                last_msg.text.rstrip()
                if hasattr(last_msg, "text") and last_msg.text
                else str(last_msg)
            )
        else:
            result = "(子代理完成但无输出消息)"

        tracker.mark_success(task_id, result)

    except Exception as exc:
        tracker.mark_error(task_id, str(exc))
    finally:
        # Cleanup thread-local
        try:
            del _current_async_task.task_id
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------


ASYNC_TASK_SYSTEM_PROMPT = """## 异步子代理 (async_task / async_status / async_list / async_cancel)

你有异步子代理工具来启动后台任务，它们会在后台独立运行，而你可以继续当前对话。

### 可用工具：
- `async_task(description, subagent_type)` — 启动后台子代理，立即返回 task_id
- `async_status(task_id)` — 查询指定任务的状态和结果
- `async_list(status_filter)` — 列出所有已知任务及其当前状态
- `async_cancel(task_id)` — 取消运行中的任务

### 启动任务时：
1. 调用 `async_task()` 后会立即得到 task_id（如 `a3f4b2c1`）
2. **必须将该 task_id 完整告诉用户**，例如："已启动后台任务 `a3f4b2c1`，正在执行中"
3. 启动后不要立即查询状态——任务在后台运行，你可以返回控制权给用户

### 查询任务时：
1. 用户要求"查进度"时，使用 `async_status(task_id="...")` 查询
2. task_id 必须是你之前从 `async_task` 返回值中看到的**精确值**
3. 如果对话中有多个任务，用户可能说"查下部署任务"——根据上下文推断 task_id
4. **如果你忘记了 task_id**（对话被压缩后），使用 `async_list()` 查看所有任务找回

### 何时用异步 vs 同步：
- **异步 (`async_task`)**：长时间运行的任务（部署、训练、批量处理），任务在后台跑，用户可以稍后查进度
- **同步 (`task`)**：需要立即得到结果的任务（研究、分析、代码审查），任务完成后你才能继续回答

### 禁止的行为：
- ❌ 不要在调用 async_task 后立即调用 async_status 轮询等待
- ❌ 不要猜测或编造 task_id——它必须来自 async_task 的返回值
- ❌ 不要把 task_id 截断或修改——始终使用完整的值
- ❌ 不要假设任务已经完成——始终用 async_status 获取最新状态"""


# ---------------------------------------------------------------------------
# Subagent construction (shared logic with SubAgentMiddleware)
# ---------------------------------------------------------------------------


def _build_subagent_specs(
    subagents: Sequence[SubAgent | CompiledSubAgent],
    tracker: _AsyncTaskTracker,
) -> list[_SubagentSpec]:
    """Build runnable subagents from specs, injecting ``report_progress``.

    ``SubAgent`` specs get ``report_progress`` added to their tools list.
    ``CompiledSubAgent`` specs are left as-is (report_progress not
    available).
    """
    from langchain.agents.middleware import HumanInTheLoopMiddleware

    report_progress_tool = _build_report_progress_tool(tracker)
    specs: list[_SubagentSpec] = []

    for spec in subagents:
        if isinstance(spec, CompiledSubAgent):
            runnable = spec.runnable.with_config(
                {
                    "metadata": {"lc_agent_name": spec.name},
                    "run_name": spec.name,
                }
            )
            specs.append(
                _SubagentSpec(
                    name=spec.name,
                    description=spec.description,
                    runnable=runnable,
                )
            )
            continue

        # SubAgent
        if spec.model is None:
            raise ValueError(f"SubAgent '{spec.name}' must specify 'model'")

        middleware: list[Any] = list(spec.middleware)
        if spec.interrupt_on:
            middleware.append(HumanInTheLoopMiddleware(interrupt_on=spec.interrupt_on))

        # Inject report_progress
        tools = list(spec.tools) + [report_progress_tool]

        specs.append(
            _SubagentSpec(
                name=spec.name,
                description=spec.description,
                runnable=_langchain_create_agent(
                    spec.model,
                    system_prompt=spec.system_prompt,
                    tools=tools,  # type: ignore[arg-type]
                    middleware=middleware,
                    name=spec.name,
                ),
            )
        )

    return specs


# ---------------------------------------------------------------------------
# AsyncSubAgentMiddleware
# ---------------------------------------------------------------------------


class AsyncSubAgentMiddleware(AgentMiddleware[AgentState, ContextT, ResponseT]):
    """Middleware that provides async (background) subagents.

    .. warning::

        **Experimental (实验性):** 本类使用较少、测试覆盖不足。
        API 可能在没有弃用期的情况下直接变更或移除（PEP 411 provisional 风格）。
        实例化时会发出 ``FutureWarning``。

    Unlike :class:`SubAgentMiddleware` which blocks until completion,
    async subagents run in background threads and are queried via
    ``async_status`` / ``async_list``.

    The subagents are **local** — they run in the same process.  No remote
    server deployment is required.  Users can give subagents tools that
    access remote resources if needed, but the async mechanism itself is
    purely local.

    Args:
        backend: Backend for file operations (passed to subagent
            construction if subagents need it in their middleware).
        async_subagents: List of subagent specs.  Each can be a
            ``SubAgent`` or ``CompiledSubAgent``.
        system_prompt: Instructions appended to the main agent's system
            prompt about async subagent usage.
        default_timeout: Maximum seconds a subagent may run before being
            force-cancelled.  Default: 3600 (1 hour).

    Example::

        from mambo_agents.middleware import AsyncSubAgentMiddleware

        middleware = AsyncSubAgentMiddleware(
            backend=my_backend,
            async_subagents=[
                {
                    "name": "deployer",
                    "description": "将服务部署到 K8s 集群",
                    "system_prompt": "你是部署专家...",
                    "model": "gpt-4o",
                    "tools": [kubectl_tool, helm_tool],
                },
            ],
            default_timeout=1800,  # 30 min
        )
    """

    state_schema = AsyncSubAgentState

    def __init__(
        self,
        *,
        backend: BackendProtocol,
        async_subagents: Sequence[SubAgent | CompiledSubAgent],
        system_prompt: str | None = ASYNC_TASK_SYSTEM_PROMPT,
        default_timeout: float = 3600.0,
    ) -> None:
        super().__init__()
        warnings.warn(
            "AsyncSubAgentMiddleware is experimental and not fully tested; "
            "its API may change or be removed without a deprecation period.",
            FutureWarning,
            stacklevel=2,
        )
        if not async_subagents:
            raise ValueError("At least one async_subagent must be specified")

        names = [a.name for a in async_subagents]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"Duplicate async subagent names: {dupes}")

        self._backend = backend
        self._tracker = _AsyncTaskTracker()
        self._default_timeout = default_timeout

        # Build subagent specs + inject report_progress
        subagent_specs = _build_subagent_specs(async_subagents, self._tracker)
        subagent_graphs = {s.name: s.runnable for s in subagent_specs}
        subagent_descriptions = "\n".join(
            f"- {s.name}: {s.description}" for s in subagent_specs
        )

        # Build tools
        self._tools: list[StructuredTool] = [
            _build_async_task_tool(
                subagent_graphs,
                subagent_descriptions,
                self._tracker,
                default_timeout,
            ),
            _build_async_status_tool(self._tracker),
            _build_async_list_tool(self._tracker),
            _build_async_cancel_tool(self._tracker),
        ]

        # Build system prompt
        if system_prompt and subagent_specs:
            agents_desc = "\n".join(
                f"- {s.name}: {s.description}" for s in subagent_specs
            )
            self._system_prompt = (
                system_prompt
                + "\n\n可用异步子代理类型：\n"
                + agents_desc
            )
        else:
            self._system_prompt = system_prompt

    @property
    def tools(self) -> list[BaseTool]:
        return self._tools  # type: ignore[return-value]

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[
            [ModelRequest[ContextT]], ModelResponse[ResponseT]
        ],
    ) -> ModelResponse[ResponseT]:
        """Inject async subagent instructions into system prompt."""
        if self._system_prompt is not None:
            request = _append_to_system_message(request, self._system_prompt)
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[
            [ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]
        ],
    ) -> ModelResponse[ResponseT]:
        """(async) Inject async subagent instructions into system prompt."""
        if self._system_prompt is not None:
            request = _append_to_system_message(request, self._system_prompt)
        return await handler(request)


def _append_to_system_message(
    request: ModelRequest,
    extra: str,
) -> ModelRequest:
    """Append *extra* text to the system message in a ModelRequest."""
    from langchain_core.messages import SystemMessage

    existing = request.system_message
    if existing is None:
        return request.override(system_message=SystemMessage(content=extra))

    content = existing.content
    if isinstance(content, str):
        new_content = content + "\n\n" + extra
    elif isinstance(content, list):
        new_content = [*content, {"type": "text", "text": "\n\n" + extra}]
    else:
        new_content = f"{content}\n\n{extra}"
    return request.override(system_message=SystemMessage(content=new_content))
