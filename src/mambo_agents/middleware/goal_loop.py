"""Goal-driven loop control middleware (mambo version).

Provides :class:`GoalLoopMiddleware` — a general loop-control core that
forces the agent loop to continue from ``after_agent`` until a goal is
satisfied, a completion condition is met, or the round budget is
exhausted.

Two usage modes (instances of the same core):

1. **Preset mode** (``mode="preset"``): only ``get_goal`` is registered.
   The goal is **preset** via config (``objective``) and completion is
   decided by ``conditions`` callbacks — e.g. "the LLM must call ``show``
   at least once".  This constrains what the LLM *must* do before
   finishing.

2. **LLM mode** (``mode="llm"``): ``create_goal`` / ``update_goal`` /
   ``get_goal`` are registered.  The LLM autonomously creates a long
   running goal and drives it to completion (``update_goal(complete)``),
   blockage (``update_goal(blocked)``) or round exhaustion.

Mechanism (shared by both modes):

- ``after_agent`` inspects the goal state every time the agent finishes a
  turn without pending tool calls.
- While the goal is ``active`` and the exit conditions are not met, an
  ``AIMessage`` with a synthetic ``get_goal()`` tool call is injected and
  the graph is routed back into the loop via ``jump_to="tools"``.
- ``get_goal`` executes and returns a ``ToolMessage`` carrying the goal,
  the current round and instructions — the model sees it and keeps
  working.
- The loop ends when the goal is satisfied / completed / blocked / timed
  out; the middleware then returns ``None`` and the graph flows to END.

Round counting:

- **preset goals**: rounds are counted as injected ``get_goal`` calls
  inside the current user-turn window (``messages`` scanned backwards
  until the first ``HumanMessage``).  A new user message naturally resets
  the budget for the next turn.
- **LLM-created goals**: rounds accumulate on ``goal["rounds"]`` across
  turns, until ``complete`` / ``blocked`` / ``max_rounds``.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Annotated, Any, Literal, TypedDict, cast

from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain.agents.middleware.types import OmitFromInput
from langchain.tools import InjectedState
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, StructuredTool
from langgraph.runtime import Runtime
from langgraph.types import Command
from langgraph.typing import ContextT
from pydantic import BaseModel, Field
from typing_extensions import NotRequired

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Prefix for tool_call ids of injected get_goal calls — used to distinguish
#: middleware-injected calls from model-initiated ones when counting rounds
#: and when evaluating conditions.
_INJECT_PREFIX = "goal-loop-"

_PRESET_GOAL_ID = "preset-goal"

#: Tool description for get_goal (both modes).
GET_GOAL_TOOL_DESCRIPTION = """读取当前循环目标与进度。

每轮结束时系统可能自动调用本工具来驱动你继续工作;你也可以主动调用它查看当前目标、轮数与完成条件。"""

#: Tool description for create_goal (LLM-controlled mode only).
CREATE_GOAL_TOOL_DESCRIPTION = """创建长程任务目标,进入自动续跑模式。

当任务复杂、需要多轮持续工作时调用本工具。创建后,每轮结束系统会自动让你继续,
直到你调用 update_goal(action='complete') 声明完成、声明阻塞或轮数用尽。

简单任务(< 3 步)不要创建目标,直接完成即可。"""

#: Tool description for update_goal (LLM-controlled mode only).
UPDATE_GOAL_TOOL_DESCRIPTION = """更新长程任务目标状态。

必须先调用 get_goal 获取精确的 goal_id 与 revision,再传入本工具(先读后写,防止过期操作)。

action:
- complete: 声明目标已完成,循环将结束。必须基于实际验证,不能空口完成。
- blocked: 声明阻塞。blocked_reason 必填;需先工作满 blocked_threshold
  轮(默认 3),此后声明一次即终止循环。
- edit: 修改 objective 或 max_goal_rounds。"""

#: System prompt appended to the model call in user-controlled mode.
_PRESET_SYSTEM_PROMPT = """## 预设目标

本次会话存在一个预设目标,通过 `get_goal` 查看详情。

每轮结束后,如果目标尚未达成,系统会强制你继续工作(自动注入 `get_goal` 调用)。
请持续工作直到完成条件满足——条件满足后循环会自动结束,无需你额外操作。"""

#: System prompt appended to the model call in LLM-controlled mode.
_LLM_SYSTEM_PROMPT = """## 长程任务模式

你可以在任务复杂、需要多轮持续工作时调用 `create_goal` 创建目标,进入自动续跑模式:

- 创建后,每轮结束系统会自动让你继续(自动注入 `get_goal` 调用,返回目标与轮数);
- 全部完成后,先调用 `get_goal` 获取精确的 goal_id / revision,再调用
  `update_goal(action='complete')` 声明完成并结束循环;
- 工作满 3 轮(默认,可配置)后仍无法推进时,调用
  `update_goal(action='blocked', blocked_reason='...')` 声明阻塞并结束循环;
- 简单任务(< 3 步)不要创建目标,直接完成即可。"""


# ---------------------------------------------------------------------------
# Goal state
# ---------------------------------------------------------------------------

GoalStatus = Literal["active", "complete", "blocked", "timeout"]


class GoalState(TypedDict):
    """Persistent goal state stored in the ``goal`` channel."""

    id: str
    objective: str
    status: GoalStatus
    rounds: int
    max_rounds: int
    revision: int
    blocked_reason: str | None
    created_by: Literal["preset", "llm"]


def _goal_last_wins(a: GoalState | None, b: GoalState | None) -> GoalState | None:
    """Last-write-wins reducer for the ``goal`` channel (whole-dict replace)."""
    return b


class GoalLoopState(AgentState):
    """State schema for the goal-loop middleware.

    ``goal`` is excluded from user-facing input — callers never pass it via
    ``.invoke({"goal": ...})``; it is managed entirely by the middleware
    and the goal tools.
    """

    goal: Annotated[NotRequired[GoalState], OmitFromInput, _goal_last_wins]
    """Current goal state (``None`` when no goal is active)."""


# ---------------------------------------------------------------------------
# Conditions (user-controlled mode)
# ---------------------------------------------------------------------------

ConditionCheck = Callable[[dict[str, Any], Runtime], bool]


@dataclass(frozen=True)
class Condition:
    """A completion condition for user-controlled mode.

    ``check(state, runtime) -> bool`` returns ``True`` when the condition
    is satisfied (the loop may end).  ``description`` is a human-readable
    summary shown to the LLM inside ``get_goal`` results.  ``progress``
    (optional) returns a **dynamic** status line — e.g. how many times a
    tool has been called so far — which ``get_goal`` uses to report the
    current condition state instead of a fixed objective.
    """

    check: ConditionCheck
    description: str
    progress: Callable[[dict[str, Any], Runtime], str] | None = None

    def __call__(self, state: dict[str, Any], runtime: Runtime) -> bool:
        """Evaluate the condition against the current agent state."""
        return self.check(state, runtime)


def _scan_window_start(messages: list) -> int:
    """Return the start index of the current user-turn window.

    Scans backwards from the end of *messages* until the first
    ``HumanMessage`` (a real user message) and returns the index right
    after it.  Returns ``0`` when no user message is present.
    """
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return i + 1
    return 0


def _tool_call_count_in_window(
    messages: list,
    name: str,
    args_subset: dict[str, Any] | None,
) -> int:
    """Count model-initiated calls of *name* in the current turn window.

    Calls injected by the middleware (``_INJECT_PREFIX`` ids) are excluded
    so they never satisfy a condition on their own.
    """
    start = _scan_window_start(messages)
    count = 0
    for msg in messages[start:]:
        if not isinstance(msg, AIMessage):
            continue
        for tool_call in msg.tool_calls:
            if tool_call.get("name") != name:
                continue
            if str(tool_call.get("id", "")).startswith(_INJECT_PREFIX):
                continue
            if args_subset and not all(
                tool_call.get("args", {}).get(k) == v for k, v in args_subset.items()
            ):
                continue
            count += 1
    return count


def _count_injected(messages: list, start: int | None = None) -> int:
    """Count middleware-injected get_goal calls in the window."""
    start = _scan_window_start(messages) if start is None else start
    count = 0
    for msg in messages[start:]:
        if isinstance(msg, AIMessage):
            for tool_call in msg.tool_calls:
                if str(tool_call.get("id", "")).startswith(_INJECT_PREFIX):
                    count += 1
    return count


def tool_called_at_least(
    name: str,
    times: int = 1,
    args_subset: dict[str, Any] | None = None,
) -> Condition:
    """Condition: the LLM must call *name* at least *times* times in the
    current turn window (optionally with matching args).

    Example::

        conditions=[tool_called_at_least("show", 1)]
        conditions=[tool_called_at_least("read", 2, {"path": "/config.json"})]
    """

    def _check(state: dict[str, Any], runtime: Runtime) -> bool:
        return (
            _tool_call_count_in_window(state.get("messages") or [], name, args_subset)
            >= times
        )

    def _progress(state: dict[str, Any], runtime: Runtime) -> str:
        count = _tool_call_count_in_window(
            state.get("messages") or [], name, args_subset
        )
        status = "已满足" if count >= times else "未满足"
        if args_subset:
            return (
                f"工具 `{name}`(参数需匹配 {args_subset}) "
                f"已调用 {count}/{times} 次({status})"
            )
        return f"工具 `{name}` 已调用 {count}/{times} 次({status})"

    description = f"工具 `{name}` 至少调用 {times} 次"
    if args_subset:
        description += f"(参数需匹配 {args_subset})"
    return Condition(check=_check, description=description, progress=_progress)


# ---------------------------------------------------------------------------
# Tool input schemas
# ---------------------------------------------------------------------------


class GetGoalInput(BaseModel):
    """Input schema for ``get_goal`` (no LLM-visible arguments)."""

    tool_call_id: Annotated[str, InjectedToolCallId] = Field(
        default="", description="Injected tool call ID (hidden from LLM)"
    )
    state: Annotated[dict[str, Any], InjectedState] = Field(
        default_factory=dict, description="Injected graph state (hidden from LLM)"
    )


class CreateGoalInput(BaseModel):
    """Input schema for ``create_goal``."""

    objective: str = Field(description="从用户请求推断出的具体完成目标")
    max_goal_rounds: int | None = Field(
        default=None,
        description="自动续跑轮数上限(正整数);不传则用中间件配置的默认值",
    )
    tool_call_id: Annotated[str, InjectedToolCallId] = Field(
        default="", description="Injected tool call ID (hidden from LLM)"
    )
    state: Annotated[dict[str, Any], InjectedState] = Field(
        default_factory=dict, description="Injected graph state (hidden from LLM)"
    )


class UpdateGoalInput(BaseModel):
    """Input schema for ``update_goal``."""

    goal_id: str = Field(description="get_goal 返回的精确目标 ID")
    revision: int = Field(description="get_goal 返回的精确版本号")
    action: Literal["complete", "blocked", "edit"] = Field(
        description="complete=完成;blocked=阻塞;edit=修改目标"
    )
    objective: str | None = Field(default=None, description="仅 edit:替换目标文本")
    max_goal_rounds: int | None = Field(
        default=None, description="仅 edit:调整轮数上限"
    )
    blocked_reason: str | None = Field(
        default=None, description="仅 blocked:必填,具体阻塞原因"
    )
    tool_call_id: Annotated[str, InjectedToolCallId] = Field(
        default="", description="Injected tool call ID (hidden from LLM)"
    )
    state: Annotated[dict[str, Any], InjectedState] = Field(
        default_factory=dict, description="Injected graph state (hidden from LLM)"
    )


def _error_tool_message(
    content: str, tool_call_id: str, tool_name: str = "update_goal",
) -> ToolMessage:
    """Build an error ToolMessage (status='error')."""
    return ToolMessage(
        content=f"Error: {content}",
        name=tool_name,
        tool_call_id=tool_call_id,
        status="error",
    )


# ---------------------------------------------------------------------------
# GoalLoopMiddleware
# ---------------------------------------------------------------------------


class GoalLoopMiddleware(AgentMiddleware[GoalLoopState, ContextT, Any]):
    """Goal-driven loop control middleware.

    See the module docstring for the full design.

    Args:
        max_rounds: Maximum number of ``after_agent`` visits for one goal
            loop (the round budget).  ``max_rounds=3`` means: the agent may
            reach ``after_agent`` 3 times; if the goal is still unmet on
            the last visit the loop ends (timeout).  Default ``256``.
        mode: ``"preset"`` (user-controlled) registers only ``get_goal``
            and requires ``objective``; ``"llm"`` (LLM-controlled) also
            registers ``create_goal`` / ``update_goal`` and requires
            ``objective`` / ``conditions`` to be left ``None``.  Default
            ``"llm"``.
        objective: Preset objective for **preset** mode — the middleware
            injects this goal on the first turn.  Must be non-empty when
            ``mode="preset"``; must be ``None`` when ``mode="llm"``.
        conditions: Completion conditions for preset mode (``Condition``
            instances, AND semantics — all must be satisfied to end the
            loop).  Ignored in LLM mode.  Empty = the loop ends only when
            the round budget is exhausted.
        blocked_threshold: Number of rounds the model must have worked
            through (``after_agent`` visits) before a single
            ``update_goal(action='blocked')`` declaration is accepted and
            the goal is marked ``blocked`` and the loop ends.  Only used
            in LLM mode (``update_goal`` is not registered in preset
            mode).  Default ``3``.
        tool_prefix: Optional prefix for all registered tool names, to
            avoid collisions with user-provided tools (e.g. ``"mambo_"``).
        system_prompt: Custom system prompt appended to the model call.
            ``None`` (default) uses the built-in prompt for the selected
            mode; pass ``""`` to disable injection entirely.
    """

    state_schema = cast(type[GoalLoopState], GoalLoopState)

    def __init__(
        self,
        *,
        max_rounds: int = 256,
        mode: Literal["preset", "llm"] = "llm",
        objective: str | None = None,
        conditions: list[Condition] | None = None,
        blocked_threshold: int = 3,
        tool_prefix: str = "",
        system_prompt: str | None = None,
    ) -> None:
        super().__init__()
        if max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")
        if blocked_threshold < 1:
            raise ValueError("blocked_threshold must be >= 1")
        if mode not in ("preset", "llm"):
            raise ValueError("mode must be 'preset' or 'llm'")
        if mode == "preset":
            if not objective or not objective.strip():
                raise ValueError("mode='preset' 需要非空的 objective")
            if blocked_threshold != 3:
                raise ValueError("blocked_threshold 仅在 mode='llm' 时可用")
        else:
            if objective is not None:
                raise ValueError("objective 仅在 mode='preset' 时可用")
            if conditions:
                raise ValueError("conditions 仅在 mode='preset' 时可用")
        self._max_rounds = max_rounds
        self._blocked_threshold = blocked_threshold
        self._objective = objective
        self._conditions = list(conditions or [])
        self._mode = mode
        self._tool_prefix = tool_prefix
        self._system_prompt = system_prompt
        self._goal_tool_name = f"{tool_prefix}get_goal"
        self.tools = self._build_tools()

    # ------------------------------------------------------------------
    # Tool construction
    # ------------------------------------------------------------------

    def _build_tools(self) -> list[StructuredTool]:
        """Register the goal tools according to the selected mode."""
        tools = [
            StructuredTool.from_function(
                name=self._goal_tool_name,
                description=GET_GOAL_TOOL_DESCRIPTION,
                func=self._make_get_goal(),
                coroutine=self._make_aget_goal(),
                args_schema=GetGoalInput,
                infer_schema=False,
            )
        ]
        if self._mode == "llm":
            tools.append(
                StructuredTool.from_function(
                    name=f"{self._tool_prefix}create_goal",
                    description=CREATE_GOAL_TOOL_DESCRIPTION,
                    func=self._make_create_goal(),
                    coroutine=self._make_acreate_goal(),
                    args_schema=CreateGoalInput,
                    infer_schema=False,
                )
            )
            tools.append(
                StructuredTool.from_function(
                    name=f"{self._tool_prefix}update_goal",
                    description=UPDATE_GOAL_TOOL_DESCRIPTION,
                    func=self._make_update_goal(),
                    coroutine=self._make_aupdate_goal(),
                    args_schema=UpdateGoalInput,
                    infer_schema=False,
                )
            )
        return tools

    # ------------------------------------------------------------------
    # get_goal
    # ------------------------------------------------------------------

    def _make_get_goal(self) -> Callable[[str, dict[str, Any]], Command]:
        """Build the ``get_goal`` tool function (bound to this instance)."""

        def _get_goal(tool_call_id: str, state: dict[str, Any]) -> Command:
            goal = state.get("goal")
            messages = state.get("messages") or []
            if goal is None:
                payload = {
                    "goal": None,
                    "message": "当前没有进行中的目标。",
                }
            elif goal.get("created_by") == "preset":
                # get_goal 执行时本次注入的调用已在 state 中,
                # injected = 已完成的轮数,展示的轮次 = injected + 1(进行中的下一轮)
                injected = _count_injected(messages)
                rounds = injected + 1

                # objective 与条件进度根据当前状态动态生成(而非固定文本)
                # 注意:工具执行环境没有 langgraph Runtime,条件求值传 None
                # (内置条件只读 state;after_agent 路径仍使用真实 runtime)
                progress_lines = [
                    c.progress(state, None)
                    if c.progress is not None
                    else c.description
                    for c in self._conditions
                ]
                satisfied = (
                    all(c.check(state, None) for c in self._conditions)
                    if self._conditions
                    else False
                )
                goal_out = dict(goal)
                if self._conditions:
                    goal_out["objective"] = (
                        f"{self._objective}"
                        f"({'已完成' if satisfied else '进行中'})"
                    )
                    goal_out["conditions"] = progress_lines

                if satisfied:
                    tail = "全部条件已满足,本轮结束后循环将自动结束。"
                elif rounds >= self._max_rounds:
                    tail = (
                        "当前条件未满足,且本轮已是最后一轮:本轮结束后循环将因"
                        "轮数用尽终止。请尽量完成条件,并在回复中说明当前进展"
                        "与未完成部分。"
                    )
                else:
                    tail = "当前条件未满足,请继续执行。条件满足后循环将自动结束。"
                progress_body = "\n".join(
                    f"  - {line}" for line in progress_lines
                ) or "  (无附加条件,达到轮数上限后自动结束)"
                payload = {
                    "goal": goal_out,
                    "message": (
                        f"【预设目标·第 {rounds}/{self._max_rounds} 轮】\n"
                        f"目标: {goal_out['objective']}\n"
                        f"条件进度:\n{progress_body}\n"
                        "提示: 以当前工作区与工具结果为准,早前叙述可能已过时,"
                        "行动前先检查现状。\n"
                        f"{tail}"
                    ),
                }
            else:
                # goal.rounds = 已完成的轮数,展示的轮次 = rounds + 1(进行中的下一轮)
                rounds = goal.get("rounds", 0) + 1
                max_rounds = goal.get("max_rounds", self._max_rounds)
                tail = (
                    "\n提示: 本轮为最后一轮,若无法完成,循环将因轮数用尽终止;"
                    "请在回复中说明当前进展与未完成部分。"
                    if rounds >= max_rounds
                    else ""
                )
                payload = {
                    "goal": goal,
                    "message": (
                        f"【目标循环·第 {rounds}/{max_rounds} 轮】\n"
                        f"目标: {goal['objective']}\n"
                        "指令: 请继续执行。以当前工作区与工具结果为准,行动前先检查"
                        "现状,不要假设早前叙述仍然有效。全部完成后先调用 get_goal "
                        "获取最新 goal_id/revision,再调用 update_goal(action='complete') "
                        "结束循环;工作满 3 轮(默认,可配置)仍无进展时,调用 "
                        "update_goal(action='blocked', blocked_reason='具体原因')"
                        " 声明阻塞。"
                        f"{tail}"
                    ),
                }
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            content=json.dumps(payload, ensure_ascii=False),
                            tool_call_id=tool_call_id,
                        )
                    ]
                }
            )

        return _get_goal

    def _make_aget_goal(self) -> Callable[[str, dict[str, Any]], Awaitable[Command]]:
        """Async variant of :meth:`_make_get_goal`."""

        async def _aget_goal(tool_call_id: str, state: dict[str, Any]) -> Command:
            return self._make_get_goal()(tool_call_id, state)

        return _aget_goal

    # ------------------------------------------------------------------
    # create_goal
    # ------------------------------------------------------------------

    def _make_create_goal(self) -> Callable[[str, str, int | None, dict[str, Any]], Command]:
        """Build the ``create_goal`` tool function (LLM-controlled mode)."""
        tool_name = f"{self._tool_prefix}create_goal"

        def _create_goal(
            objective: str,
            max_goal_rounds: int | None,
            tool_call_id: str,
            state: dict[str, Any],
        ) -> Command:
            goal = state.get("goal")
            if goal is not None and goal.get("status") == "active":
                return Command(
                    update={
                        "messages": [
                            _error_tool_message(
                                f"已有进行中的目标: {goal['objective']}。"
                                "请先 update_goal(complete/blocked) 结束它,"
                                "或 update_goal(edit) 修改现有目标。",
                                tool_call_id,
                                tool_name,
                            )
                        ]
                    }
                )
            if not objective or not objective.strip():
                return Command(
                    update={
                        "messages": [
                            _error_tool_message(
                                "objective 不能为空。", tool_call_id, tool_name
                            )
                        ]
                    }
                )
            max_rounds = self._max_rounds
            if max_goal_rounds is not None:
                if max_goal_rounds < 1:
                    return Command(
                        update={
                            "messages": [
                                _error_tool_message(
                                    "max_goal_rounds 必须为正整数。",
                                    tool_call_id,
                                    tool_name,
                                )
                            ]
                        }
                    )
                max_rounds = min(max_goal_rounds, self._max_rounds)
            new_goal: GoalState = {
                "id": f"goal-{uuid.uuid4().hex[:8]}",
                "objective": objective.strip(),
                "status": "active",
                "rounds": 0,
                "max_rounds": max_rounds,
                "revision": 1,
                "blocked_reason": None,
                "created_by": "llm",
            }
            message = (
                f"目标已创建,进入长程模式(上限 {max_rounds} 轮)。"
                "请开始执行;完成后调用 update_goal(action='complete') 结束循环。"
            )
            return Command(
                update={
                    "goal": new_goal,
                    "messages": [ToolMessage(content=message, tool_call_id=tool_call_id)],
                }
            )

        return _create_goal

    def _make_acreate_goal(self) -> Callable[..., Awaitable[Command]]:
        """Async variant of :meth:`_make_create_goal`."""

        async def _acreate_goal(
            objective: str,
            max_goal_rounds: int | None,
            tool_call_id: str,
            state: dict[str, Any],
        ) -> Command:
            return self._make_create_goal()(objective, max_goal_rounds, tool_call_id, state)

        return _acreate_goal

    # ------------------------------------------------------------------
    # update_goal
    # ------------------------------------------------------------------

    def _make_update_goal(self) -> Callable[..., Command]:
        """Build the ``update_goal`` tool function (LLM-controlled mode)."""
        tool_name = f"{self._tool_prefix}update_goal"

        def _update_goal(
            goal_id: str,
            revision: int,
            action: str,
            objective: str | None,
            max_goal_rounds: int | None,
            blocked_reason: str | None,
            tool_call_id: str,
            state: dict[str, Any],
        ) -> Command:
            goal = state.get("goal")
            if goal is None:
                return Command(
                    update={
                        "messages": [
                            _error_tool_message(
                                "没有进行中的目标,请先调用 create_goal。",
                                tool_call_id,
                                tool_name,
                            )
                        ]
                    }
                )
            if goal["id"] != goal_id or goal["revision"] != revision:
                return Command(
                    update={
                        "messages": [
                            _error_tool_message(
                                f"goal_id/revision 不匹配(当前 id={goal['id']}, "
                                f"revision={goal['revision']})。请先调用 get_goal "
                                "获取最新值再重试。",
                                tool_call_id,
                                tool_name,
                            )
                        ]
                    }
                )

            new_goal = dict(goal)
            new_revision = revision + 1

            if action == "complete":
                new_goal["status"] = "complete"
                new_goal["revision"] = new_revision
                message = (
                    "目标已标记完成,循环将在本轮结束后终止。"
                    "请向用户给出收尾总结:\n"
                    "1. 结果:完成了什么;\n"
                    "2. 验证:如何验证、证据是什么(只基于本会话实际执行过的"
                    "工具结果,未验证的不要声称完成);\n"
                    "3. 产出物:文件、代码或其他可查看的成果位置。\n"
                    "只陈述本会话中实际发生的事实;未确认的细节要明说,不要编造。"
                    "本轮不要再调用任何工具。"
                )
            elif action == "edit":
                if objective is not None:
                    if not objective.strip():
                        return Command(
                            update={
                                "messages": [
                                    _error_tool_message(
                                        "edit 的 objective 不能为空。",
                                        tool_call_id,
                                        tool_name,
                                    )
                                ]
                            }
                        )
                    new_goal["objective"] = objective.strip()
                if max_goal_rounds is not None:
                    if max_goal_rounds < 1:
                        return Command(
                            update={
                                "messages": [
                                    _error_tool_message(
                                        "max_goal_rounds 必须为正整数。",
                                        tool_call_id,
                                        tool_name,
                                    )
                                ]
                            }
                        )
                    new_goal["max_rounds"] = min(max_goal_rounds, self._max_rounds)
                new_goal["revision"] = new_revision
                message = (
                    f"目标已更新(revision={new_revision}): "
                    f"objective={new_goal['objective']}, "
                    f"max_rounds={new_goal['max_rounds']}。请继续执行。"
                )
            elif action == "blocked":
                if not blocked_reason or not blocked_reason.strip():
                    return Command(
                        update={
                            "messages": [
                                _error_tool_message(
                                    "blocked 必须提供 blocked_reason。",
                                    tool_call_id,
                                    tool_name,
                                )
                            ]
                        }
                    )
                # 资格 = 实际工作轮数(当前进行中的轮次 = 已完成轮数 + 1);
                # 未达 blocked_threshold 轮直接拒绝,不记录任何状态。
                current_round = new_goal.get("rounds", 0) + 1
                if current_round < self._blocked_threshold:
                    return Command(
                        update={
                            "messages": [
                                _error_tool_message(
                                    f"当前为第 {current_round} 轮,至少需工作满 "
                                    f"{self._blocked_threshold} 轮才能声明阻塞。"
                                    "请继续尝试解决;若确无进展,后续轮次再次声明。",
                                    tool_call_id,
                                    tool_name,
                                )
                            ]
                        }
                    )
                new_goal["status"] = "blocked"
                new_goal["blocked_reason"] = blocked_reason
                new_goal["revision"] = new_revision
                message = (
                    f"阻塞已确认(已工作 {current_round} 轮,达到 "
                    f"{self._blocked_threshold} 轮门槛),循环终止。"
                    "请向用户给出收尾消息:\n"
                    "1. 已完成的部分;\n"
                    "2. 具体阻塞条件与已尝试过的方案;\n"
                    "3. 需要用户提供什么才能继续。\n"
                    "只陈述本会话中实际发生的事实,不要编造。"
                    "本轮不要再调用任何工具。"
                )
            else:  # pragma: no cover — Literal restricts action
                return Command(
                    update={
                        "messages": [
                            _error_tool_message(
                                f"未知 action: {action}", tool_call_id, tool_name
                            )
                        ]
                    }
                )

            return Command(
                update={
                    "goal": new_goal,
                    "messages": [ToolMessage(content=message, tool_call_id=tool_call_id)],
                }
            )

        return _update_goal

    def _make_aupdate_goal(self) -> Callable[..., Awaitable[Command]]:
        """Async variant of :meth:`_make_update_goal`."""

        async def _aupdate_goal(
            goal_id: str,
            revision: int,
            action: str,
            objective: str | None,
            max_goal_rounds: int | None,
            blocked_reason: str | None,
            tool_call_id: str,
            state: dict[str, Any],
        ) -> Command:
            return self._make_update_goal()(
                goal_id,
                revision,
                action,
                objective,
                max_goal_rounds,
                blocked_reason,
                tool_call_id,
                state,
            )

        return _aupdate_goal

    # ------------------------------------------------------------------
    # before_agent — preset goal bootstrap
    # ------------------------------------------------------------------

    def before_agent(
        self, state: GoalLoopState, runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Inject the preset goal on the first turn (user-controlled mode)."""
        if self._mode == "preset" and state.get("goal") is None:
            return {
                "goal": {
                    "id": _PRESET_GOAL_ID,
                    "objective": self._objective,
                    "status": "active",
                    "rounds": 0,
                    "max_rounds": self._max_rounds,
                    "revision": 1,
                    "blocked_reason": None,
                    "created_by": "preset",
                }
            }
        return None

    async def abefore_agent(
        self, state: GoalLoopState, runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Async variant of :meth:`before_agent`."""
        return self.before_agent(state, runtime)

    # ------------------------------------------------------------------
    # after_model — declare the loop-exit node (required for jump_to)
    # ------------------------------------------------------------------

    def after_model(
        self, state: GoalLoopState, runtime: Runtime,
    ) -> dict[str, Any] | None:
        """No-op ``after_model`` hook.

        Declaring this hook makes the agent factory treat this
        middleware's ``after_model`` node as the loop exit node, which
        adds the loop-entry node to the model-to-tools conditional edge
        destinations.  This is required for the injected (artificial)
        ``get_goal`` calls: once the tools node executes them, the
        model-to-tools edge can route back into the model loop via its
        "all tool calls answered" branch.
        """
        return None

    async def aafter_model(
        self, state: GoalLoopState, runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Async variant of :meth:`after_model` (no-op)."""
        return None

    # ------------------------------------------------------------------
    # after_agent — loop control
    # ------------------------------------------------------------------

    def _inject_get_goal(self, state: GoalLoopState) -> dict[str, Any] | None:
        """Inject a synthetic ``get_goal`` tool call into the LAST AI message
        and jump to the tools node.

        Instead of appending a brand-new ``AIMessage`` (which would add a
        synthetic assistant turn to the history — problematic for
        providers that require every assistant message to carry e.g.
        ``reasoning_content`` back to the API), the last AI message is
        **replaced in place** (same message id, so ``add_messages``
        semantics replace rather than append) with a copy that
        additionally carries the ``get_goal`` tool call.  The history
        length is unchanged and the message keeps its original
        ``additional_kwargs``.

        Returns ``None`` when there is no AI message to attach to (the
        loop then simply ends).
        """
        messages = state.get("messages") or []
        if not messages:
            return None
        last = messages[-1]
        if not isinstance(last, AIMessage):
            return None
        call_id = f"{_INJECT_PREFIX}{uuid.uuid4().hex}"
        new_last = last.model_copy(
            update={
                "id": last.id,
                "tool_calls": [
                    *list(last.tool_calls),
                    {
                        "name": self._goal_tool_name,
                        "args": {},
                        "id": call_id,
                    },
                ],
            }
        )
        return {"jump_to": "tools", "messages": [new_last]}

    @hook_config(can_jump_to=["tools"])
    def after_agent(
        self, state: GoalLoopState, runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Enforce the goal loop.

        Round semantics: each ``after_agent`` visit means one round has
        **completed**; the injected ``get_goal`` call is the start of the
        **next** round (so the first injection is round 2).

        - No goal → return ``None`` (normal mode, no loop).
        - Preset goal: end when all conditions are satisfied or when the
          round budget is exhausted; otherwise inject ``get_goal``.
        - LLM-created goal: end when ``complete`` / ``blocked`` or when
          ``rounds >= max_rounds`` (timeout); otherwise inject.
        """
        goal = state.get("goal")
        if goal is None:
            return None

        if goal.get("created_by") == "preset":
            injected = _count_injected(state.get("messages") or [])
            if self._conditions and all(
                c.check(state, runtime) for c in self._conditions
            ):
                return None
            if injected >= self._max_rounds - 1:
                # Last allowed after_agent visit — end without injecting.
                return None
            return self._inject_get_goal(state)

        # LLM-created goal
        rounds = goal.get("rounds", 0) + 1
        if goal.get("status") in ("complete", "blocked", "timeout"):
            return None
        max_rounds = goal.get("max_rounds") or self._max_rounds
        if rounds >= max_rounds:
            # 轮数用尽:只更新 goal 状态,不注入 get_goal、不返回消息,
            # 图将直接流向 END 结束本次执行。
            return {"goal": {**goal, "rounds": rounds, "status": "timeout"}}
        injection = self._inject_get_goal(state)
        if injection is None:
            return None
        return {**injection, "goal": {**goal, "rounds": rounds}}

    async def aafter_agent(
        self, state: GoalLoopState, runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Async variant of :meth:`after_agent`."""
        return self.after_agent(state, runtime)

    # ------------------------------------------------------------------
    # System prompt injection
    # ------------------------------------------------------------------

    def _resolve_system_prompt(self) -> str | None:
        """Return the system prompt to append (``None`` disables)."""
        if self._system_prompt is not None:
            return self._system_prompt or None
        if self._mode == "preset":
            return _PRESET_SYSTEM_PROMPT
        return _LLM_SYSTEM_PROMPT

    def wrap_model_call(
        self,
        request: Any,
        handler: Callable[[Any], Any],
    ) -> Any:
        """Append the goal-loop system prompt to the model call."""
        prompt = self._resolve_system_prompt()
        if prompt is None:
            return handler(request)
        if request.system_message is not None:
            new_system_content = [
                *request.system_message.content_blocks,
                {"type": "text", "text": f"\n\n{prompt}"},
            ]
        else:
            new_system_content = [{"type": "text", "text": prompt}]
        new_system_message = SystemMessage(
            content=cast("list[str | dict[str, str]]", new_system_content)
        )
        return handler(request.override(system_message=new_system_message))

    async def awrap_model_call(
        self,
        request: Any,
        handler: Callable[[Any], Any],
    ) -> Any:
        """Async variant of :meth:`wrap_model_call`."""
        prompt = self._resolve_system_prompt()
        if prompt is None:
            return await handler(request)
        if request.system_message is not None:
            new_system_content = [
                *request.system_message.content_blocks,
                {"type": "text", "text": f"\n\n{prompt}"},
            ]
        else:
            new_system_content = [{"type": "text", "text": prompt}]
        new_system_message = SystemMessage(
            content=cast("list[str | dict[str, str]]", new_system_content)
        )
        return await handler(request.override(system_message=new_system_message))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class GoalLoopConfig:
    """Configuration for :class:`GoalLoopMiddleware`.

    Args:
        mode: ``"preset"`` (user-controlled, requires ``objective``) or
            ``"llm"`` (LLM-controlled).  Default ``"llm"``.
        max_rounds: Round budget (number of ``after_agent`` visits) before
            the loop is force-stopped.  Default ``256``.
        objective: Preset objective (preset mode only, required).
        conditions: Completion conditions (preset mode only, AND).
            Empty = the loop ends only when the round budget is
            exhausted.
        blocked_threshold: Number of worked rounds required before a
            single ``update_goal(action='blocked')`` declaration is
            accepted and the goal is marked ``blocked``.  LLM mode only.
            Default ``3``.
        tool_prefix: Optional prefix for goal tool names.
        system_prompt: Custom system prompt (``None`` = built-in per mode,
            ``""`` disables).
    """

    mode: Literal["preset", "llm"] = "llm"
    max_rounds: int = 256
    objective: str | None = None
    conditions: list[Condition] | None = None
    blocked_threshold: int = 3
    tool_prefix: str = ""
    system_prompt: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("preset", "llm"):
            raise ValueError("mode must be 'preset' or 'llm'")
        if self.mode == "preset":
            if not self.objective or not self.objective.strip():
                raise ValueError("mode='preset' 需要非空的 objective")
            if self.blocked_threshold != 3:
                raise ValueError("blocked_threshold 仅在 mode='llm' 时可用")
        else:
            if self.objective is not None:
                raise ValueError("objective 仅在 mode='preset' 时可用")
            if self.conditions:
                raise ValueError("conditions 仅在 mode='preset' 时可用")

    def build_middleware(self) -> GoalLoopMiddleware:
        """Instantiate the middleware from this config."""
        return GoalLoopMiddleware(
            max_rounds=self.max_rounds,
            mode=self.mode,
            objective=self.objective,
            conditions=self.conditions,
            blocked_threshold=self.blocked_threshold,
            tool_prefix=self.tool_prefix,
            system_prompt=self.system_prompt,
        )
