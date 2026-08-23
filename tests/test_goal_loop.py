"""Tests for GoalLoopMiddleware (goal-driven loop control).

Covers:
- Condition / tool_called_at_least window scanning
- before_agent preset-goal bootstrap
- after_agent loop-control decisions (preset + LLM-created goals)
- get_goal / create_goal / update_goal tool logic
- Full-graph integration with a scripted fake model
"""

from __future__ import annotations

import json
import uuid

import pytest
from langchain.agents.middleware.types import AgentState
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import StructuredTool
from langgraph.runtime import Runtime
from langgraph.types import Command
from pydantic import Field

from mambo_agents import create_mambo_agent
from mambo_agents.backends.store import StoreBackend
from mambo_agents.middleware.goal_loop import (
    GoalLoopConfig,
    GoalLoopMiddleware,
    _INJECT_PREFIX,
    tool_called_at_least,
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


def _preset_goal(**overrides):
    goal = {
        "id": "preset-goal",
        "objective": "必须调用 show 工具",
        "status": "active",
        "rounds": 0,
        "max_rounds": 3,
        "revision": 1,
        "blocked_reason": None,
        "created_by": "preset",
    }
    goal.update(overrides)
    return goal


def _llm_goal(**overrides):
    goal = {
        "id": "goal-abc123",
        "objective": "写一个贪吃蛇游戏",
        "status": "active",
        "rounds": 2,
        "max_rounds": 5,
        "revision": 3,
        "blocked_reason": None,
        "created_by": "llm",
    }
    goal.update(overrides)
    return goal


def _injected(msg: AIMessage) -> bool:
    """True if *msg* is a middleware-injected get_goal call."""
    return any(
        str(tc.get("id", "")).startswith(_INJECT_PREFIX) for tc in msg.tool_calls
    )


def _injected_count(messages) -> int:
    return sum(1 for m in messages if isinstance(m, AIMessage) and _injected(m))


class _ScriptedModel(BaseChatModel):
    """Fake chat model that replays a script of AIMessages."""

    script: list = Field(default_factory=list, exclude=True)
    default_response: AIMessage = Field(
        default_factory=lambda: AIMessage(content="done"), exclude=True
    )
    invocations: int = Field(default=0, exclude=True)

    def __init__(self, script, default: str = "done"):
        super().__init__()
        self.script = list(script)
        self.default_response = AIMessage(content=default)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):  # noqa: ARG002
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ARG002
        self.invocations += 1
        msg = self.script.pop(0) if self.script else self.default_response
        # Fresh message id per invocation — otherwise add_messages dedupes
        # same-id messages (replace instead of append).
        msg = msg.model_copy(update={"id": str(uuid.uuid4())})
        return ChatResult(generations=[ChatGeneration(message=msg)])


def _show_tool() -> StructuredTool:
    def _show(text: str) -> str:
        """Show some text (fake tool)."""
        return f"showing: {text}"

    return StructuredTool.from_function(func=_show, name="show")


def _read_tool() -> StructuredTool:
    def _read(path: str) -> str:
        """Read a file (fake tool)."""
        return f"reading: {path}"

    return StructuredTool.from_function(func=_read, name="read")


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------


class TestToolCalledAtLeast:
    def test_count_in_window_until_user_message(self):
        """Only calls after the last user message are counted."""
        cond = tool_called_at_least("read", 2)
        messages = [
            HumanMessage(content="第一回合"),
            AIMessage(content="", tool_calls=[{"name": "read", "args": {}, "id": "a1"}]),
            ToolMessage(content="ok", tool_call_id="a1"),
            AIMessage(content="", tool_calls=[{"name": "read", "args": {}, "id": "a2"}]),
            ToolMessage(content="ok", tool_call_id="a2"),
            HumanMessage(content="第二回合"),
            AIMessage(content="", tool_calls=[{"name": "read", "args": {}, "id": "a3"}]),
            ToolMessage(content="ok", tool_call_id="a3"),
        ]
        # Second turn window has only one read call → not satisfied
        assert cond.check({"messages": messages}, _EMPTY_RUNTIME) is False

    def test_injected_calls_are_excluded(self):
        """Middleware-injected get_goal calls never satisfy a condition."""
        cond = tool_called_at_least("get_goal", 1)
        messages = [
            HumanMessage(content="hi"),
            AIMessage(
                content="",
                tool_calls=[{"name": "get_goal", "args": {}, "id": f"{_INJECT_PREFIX}abc"}],
            ),
        ]
        assert cond.check({"messages": messages}, _EMPTY_RUNTIME) is False

        # Same name, model-initiated id → counted
        messages = [
            HumanMessage(content="hi"),
            AIMessage(
                content="",
                tool_calls=[{"name": "get_goal", "args": {}, "id": "model-1"}],
            ),
        ]
        assert cond.check({"messages": messages}, _EMPTY_RUNTIME) is True

    def test_args_subset_matching(self):
        cond = tool_called_at_least("read", 1, {"path": "/config.json"})
        messages = [
            HumanMessage(content="hi"),
            AIMessage(
                content="",
                tool_calls=[{"name": "read", "args": {"path": "/other.txt"}, "id": "a1"}],
            ),
        ]
        assert cond.check({"messages": messages}, _EMPTY_RUNTIME) is False

        messages = [
            HumanMessage(content="hi"),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "read", "args": {"path": "/config.json", "offset": 0}, "id": "a1"}
                ],
            ),
        ]
        assert cond.check({"messages": messages}, _EMPTY_RUNTIME) is True

    def test_times_threshold(self):
        cond = tool_called_at_least("show", 3)
        messages = [
            HumanMessage(content="hi"),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "show", "args": {}, "id": "a1"},
                    {"name": "show", "args": {}, "id": "a2"},
                ],
            ),
        ]
        assert cond.check({"messages": messages}, _EMPTY_RUNTIME) is False

    def test_description(self):
        cond = tool_called_at_least("show", 1)
        assert "show" in cond.description and "1" in cond.description
        cond2 = tool_called_at_least("read", 2, {"path": "/x"})
        assert "2" in cond2.description and "{'path': '/x'}" in cond2.description

    def test_progress_dynamic(self):
        """progress 返回动态计数,而非固定描述。"""
        cond = tool_called_at_least("show", 3)
        messages = [
            HumanMessage(content="hi"),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "show", "args": {}, "id": "a1"},
                    {"name": "show", "args": {}, "id": "a2"},
                ],
            ),
        ]
        assert cond.progress({"messages": messages}, _EMPTY_RUNTIME) == (
            "工具 `show` 已调用 2/3 次(未满足)"
        )

        # 注入的调用不计入进度
        messages2 = [
            HumanMessage(content="hi"),
            AIMessage(
                content="",
                tool_calls=[{"name": "get_goal", "args": {}, "id": f"{_INJECT_PREFIX}x"}],
            ),
        ]
        cond2 = tool_called_at_least("get_goal", 1)
        assert cond2.progress({"messages": messages2}, _EMPTY_RUNTIME) == (
            "工具 `get_goal` 已调用 0/1 次(未满足)"
        )

    def test_progress_with_args_subset(self):
        """带参数子集时,progress 显示参数要求。"""
        cond = tool_called_at_least("show", 1, {"mode": "abc"})
        messages = [
            HumanMessage(content="hi"),
            AIMessage(
                content="",
                tool_calls=[{"name": "show", "args": {"mode": "xyz"}, "id": "a1"}],
            ),
        ]
        # mode=xyz 不匹配 → 0/1
        assert cond.progress({"messages": messages}, _EMPTY_RUNTIME) == (
            "工具 `show`(参数需匹配 {'mode': 'abc'}) 已调用 0/1 次(未满足)"
        )
        assert cond.check({"messages": messages}, _EMPTY_RUNTIME) is False

        messages2 = [
            HumanMessage(content="hi"),
            AIMessage(
                content="",
                tool_calls=[{"name": "show", "args": {"mode": "abc"}, "id": "a1"}],
            ),
        ]
        assert cond.progress({"messages": messages2}, _EMPTY_RUNTIME) == (
            "工具 `show`(参数需匹配 {'mode': 'abc'}) 已调用 1/1 次(已满足)"
        )
        assert cond.check({"messages": messages2}, _EMPTY_RUNTIME) is True


# ---------------------------------------------------------------------------
# before_agent
# ---------------------------------------------------------------------------


class TestBeforeAgent:
    def test_preset_goal_injected_once(self):
        mw = GoalLoopMiddleware(
            mode="preset",
            objective="必须调用 show 工具",
            conditions=[tool_called_at_least("show", 1)],
            max_rounds=3,
        )
        state: AgentState = {"messages": [HumanMessage(content="hi")]}
        update = mw.before_agent(state, _EMPTY_RUNTIME)
        assert update is not None
        goal = update["goal"]
        assert goal["created_by"] == "preset"
        assert goal["status"] == "active"
        assert goal["max_rounds"] == 3

        # Second call with goal already present → no update
        state2 = {"messages": [HumanMessage(content="hi")], "goal": goal}
        assert mw.before_agent(state2, _EMPTY_RUNTIME) is None

    def test_no_preset_goal_no_bootstrap(self):
        mw = GoalLoopMiddleware(mode="llm", max_rounds=10)
        state: AgentState = {"messages": [HumanMessage(content="hi")]}
        assert mw.before_agent(state, _EMPTY_RUNTIME) is None

    def test_async_variant(self):
        mw = GoalLoopMiddleware(mode="preset", objective="目标", max_rounds=3)
        state: AgentState = {"messages": [HumanMessage(content="hi")]}
        update = mw.aafter_agent  # ensure async method exists
        assert callable(update)


# ---------------------------------------------------------------------------
# after_agent
# ---------------------------------------------------------------------------


class TestAfterAgentPreset:
    def test_no_goal_returns_none(self):
        mw = GoalLoopMiddleware(mode="preset", objective="x", max_rounds=3)
        state: AgentState = {"messages": [HumanMessage(content="hi")]}
        assert mw.after_agent(state, _EMPTY_RUNTIME) is None

    def test_condition_met_ends_loop(self):
        mw = GoalLoopMiddleware(
            mode="preset",
            objective="必须调用 show 工具",
            conditions=[tool_called_at_least("show", 1)],
            max_rounds=3,
        )
        state: AgentState = {
            "goal": _preset_goal(),
            "messages": [
                HumanMessage(content="hi"),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "show", "args": {"text": "x"}, "id": "a1"}],
                ),
                ToolMessage(content="ok", tool_call_id="a1"),
                AIMessage(content="done"),
            ],
        }
        assert mw.after_agent(state, _EMPTY_RUNTIME) is None

    def test_partial_conditions_continue_loop(self):
        """多个条件只满足一个 → 不结束,继续注入 get_goal。"""
        mw = GoalLoopMiddleware(
            mode="preset",
            objective="必须调用 show 和 read",
            conditions=[
                tool_called_at_least("show", 1),
                tool_called_at_least("read", 1),
            ],
            max_rounds=3,
        )
        state: AgentState = {
            "goal": _preset_goal(),
            "messages": [
                HumanMessage(content="hi"),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "show", "args": {"text": "x"}, "id": "a1"}],
                ),
                ToolMessage(content="ok", tool_call_id="a1"),
                AIMessage(content="done"),
            ],
        }
        update = mw.after_agent(state, _EMPTY_RUNTIME)
        assert update is not None
        assert update["jump_to"] == "tools"
        assert len(update["messages"]) == 1
        assert update["messages"][0].tool_calls[0]["name"] == "get_goal"

    def test_all_conditions_met_ends_loop(self):
        """所有条件都满足 → 循环结束。"""
        mw = GoalLoopMiddleware(
            mode="preset",
            objective="必须调用 show 和 read",
            conditions=[
                tool_called_at_least("show", 1),
                tool_called_at_least("read", 1),
            ],
            max_rounds=3,
        )
        state: AgentState = {
            "goal": _preset_goal(),
            "messages": [
                HumanMessage(content="hi"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "show", "args": {"text": "x"}, "id": "a1"},
                        {"name": "read", "args": {"path": "/x"}, "id": "a2"},
                    ],
                ),
                ToolMessage(content="ok", tool_call_id="a1"),
                ToolMessage(content="ok", tool_call_id="a2"),
                AIMessage(content="done"),
            ],
        }
        assert mw.after_agent(state, _EMPTY_RUNTIME) is None

    def test_no_conditions_does_not_end_early(self):
        """无 conditions 时不因空列表提前结束(回归保护)。"""
        mw = GoalLoopMiddleware(
            mode="preset",
            objective="无条件目标",
            max_rounds=3,
        )
        state: AgentState = {
            "goal": _preset_goal(),
            "messages": [HumanMessage(content="hi"), AIMessage(content="done")],
        }
        update = mw.after_agent(state, _EMPTY_RUNTIME)
        assert update is not None
        assert update["jump_to"] == "tools"

    def test_round_budget_exhausted_ends_loop(self):
        mw = GoalLoopMiddleware(
            mode="preset",
            objective="必须调用 show 工具",
            conditions=[tool_called_at_least("show", 1)],
            max_rounds=3,
        )
        # 2 injections already happened → this is the 3rd (last) visit
        injected_msg = AIMessage(
            content="",
            tool_calls=[{"name": "get_goal", "args": {}, "id": f"{_INJECT_PREFIX}x1"}],
        )
        injected_msg2 = AIMessage(
            content="",
            tool_calls=[{"name": "get_goal", "args": {}, "id": f"{_INJECT_PREFIX}x2"}],
        )
        state: AgentState = {
            "goal": _preset_goal(),
            "messages": [
                HumanMessage(content="hi"),
                injected_msg,
                ToolMessage(content="{}", tool_call_id=f"{_INJECT_PREFIX}x1"),
                injected_msg2,
                ToolMessage(content="{}", tool_call_id=f"{_INJECT_PREFIX}x2"),
                AIMessage(content="still not done"),
            ],
        }
        assert mw.after_agent(state, _EMPTY_RUNTIME) is None

    def test_injects_get_goal_and_jumps_to_tools(self):
        mw = GoalLoopMiddleware(
            mode="preset",
            objective="必须调用 show 工具",
            conditions=[tool_called_at_least("show", 1)],
            max_rounds=3,
        )
        last_ai = AIMessage(content="done", id="ai-last-1")
        state: AgentState = {
            "goal": _preset_goal(),
            "messages": [HumanMessage(content="hi"), last_ai],
        }
        update = mw.after_agent(state, _EMPTY_RUNTIME)
        assert update is not None
        assert update["jump_to"] == "tools"
        assert len(update["messages"]) == 1
        ai = update["messages"][0]
        # 原地替换:同 id、保留原 content,仅追加 get_goal tool_call
        assert isinstance(ai, AIMessage)
        assert ai.id == "ai-last-1"
        assert ai.content == "done"
        assert len(ai.tool_calls) == 1
        assert ai.tool_calls[0]["name"] == "get_goal"
        assert str(ai.tool_calls[0]["id"]).startswith(_INJECT_PREFIX)

    def test_inject_keeps_additional_kwargs(self):
        """替换注入必须保留原消息的 additional_kwargs(如 reasoning_content)。"""
        mw = GoalLoopMiddleware(mode="preset", objective="x", max_rounds=3)
        last_ai = AIMessage(
            content="done",
            id="ai-last-2",
            additional_kwargs={"reasoning_content": "思考过程"},
        )
        state: AgentState = {
            "goal": _preset_goal(),
            "messages": [HumanMessage(content="hi"), last_ai],
        }
        update = mw.after_agent(state, _EMPTY_RUNTIME)
        ai = update["messages"][0]
        assert ai.additional_kwargs.get("reasoning_content") == "思考过程"

    def test_inject_no_ai_message_returns_none(self):
        """最后一条消息不是 AIMessage 时防御性放行。"""
        mw = GoalLoopMiddleware(mode="preset", objective="x", max_rounds=3)
        state: AgentState = {
            "goal": _preset_goal(),
            "messages": [HumanMessage(content="hi")],
        }
        assert mw.after_agent(state, _EMPTY_RUNTIME) is None

    def test_can_jump_to_declared(self):
        mw = GoalLoopMiddleware(mode="preset", objective="x", max_rounds=3)
        assert getattr(mw.after_agent, "__can_jump_to__", []) == ["tools"]


class TestAfterAgentLLM:
    def test_active_injects_and_increments_rounds(self):
        mw = GoalLoopMiddleware(mode="llm", max_rounds=10)
        last_ai = AIMessage(content="done", id="ai-llm-1")
        state: AgentState = {
            "goal": _llm_goal(rounds=2),
            "messages": [HumanMessage(content="hi"), last_ai],
        }
        update = mw.after_agent(state, _EMPTY_RUNTIME)
        assert update is not None
        assert update["jump_to"] == "tools"
        assert update["goal"]["rounds"] == 3
        assert update["goal"]["status"] == "active"
        ai = update["messages"][0]
        assert ai.id == "ai-llm-1"
        assert ai.tool_calls[0]["name"] == "get_goal"

    def test_complete_ends_loop(self):
        mw = GoalLoopMiddleware(mode="llm", max_rounds=10)
        state: AgentState = {
            "goal": _llm_goal(status="complete", rounds=4),
            "messages": [HumanMessage(content="hi"), AIMessage(content="done")],
        }
        assert mw.after_agent(state, _EMPTY_RUNTIME) is None

    def test_blocked_ends_loop(self):
        mw = GoalLoopMiddleware(mode="llm", max_rounds=10)
        state: AgentState = {
            "goal": _llm_goal(status="blocked", rounds=4),
            "messages": [HumanMessage(content="hi"), AIMessage(content="done")],
        }
        assert mw.after_agent(state, _EMPTY_RUNTIME) is None

    def test_round_budget_timeout(self):
        mw = GoalLoopMiddleware(mode="llm", max_rounds=5)
        state: AgentState = {
            "goal": _llm_goal(rounds=4),
            "messages": [HumanMessage(content="hi"), AIMessage(content="done")],
        }
        update = mw.after_agent(state, _EMPTY_RUNTIME)
        assert update is not None
        assert update["goal"]["rounds"] == 5
        assert update["goal"]["status"] == "timeout"
        assert "jump_to" not in update

    def test_async_variant_matches_sync(self):
        import asyncio

        mw = GoalLoopMiddleware(mode="llm", max_rounds=10)
        last_ai = AIMessage(content="done", id="ai-async-1")
        state: AgentState = {
            "goal": _llm_goal(rounds=2),
            "messages": [HumanMessage(content="hi"), last_ai],
        }
        sync_update = mw.after_agent(state, _EMPTY_RUNTIME)
        async_update = asyncio.run(mw.aafter_agent(state, _EMPTY_RUNTIME))
        # tool_call ids are random uuids — compare structure instead
        assert async_update["jump_to"] == sync_update["jump_to"] == "tools"
        assert async_update["goal"] == sync_update["goal"] == {
            **_llm_goal(rounds=2),
            "rounds": 3,
        }
        for update in (sync_update, async_update):
            ai = update["messages"][0]
            assert isinstance(ai, AIMessage)
            assert ai.id == "ai-async-1"  # 原地替换,id 不变
            assert ai.tool_calls[0]["name"] == "get_goal"


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


class TestGetGoalTool:
    def test_no_goal(self):
        mw = GoalLoopMiddleware(mode="preset", objective="预设目标", max_rounds=3)
        get_goal = mw.tools[0]
        cmd = get_goal.func(tool_call_id="t1", state={"messages": []})
        assert isinstance(cmd, Command)
        msg = cmd.update["messages"][0]
        payload = json.loads(msg.content)
        assert payload["goal"] is None

    def test_preset_goal_content(self):
        mw = GoalLoopMiddleware(
            mode="preset",
            objective="必须调用 show 工具",
            conditions=[tool_called_at_least("show", 1)],
            max_rounds=3,
        )
        get_goal = mw.tools[0]
        state = {
            "goal": _preset_goal(),
            "messages": [
                HumanMessage(content="hi"),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "get_goal", "args": {}, "id": f"{_INJECT_PREFIX}a"}],
                ),
            ],
        }
        cmd = get_goal.func(tool_call_id="t1", state=state)
        payload = json.loads(cmd.update["messages"][0].content)
        assert "第 2/3 轮" in payload["message"]
        assert payload["goal"]["created_by"] == "preset"
        # objective 动态生成:携带进行中状态
        assert payload["goal"]["objective"] == "必须调用 show 工具(进行中)"
        # 条件进度动态反映当前计数
        assert payload["goal"]["conditions"] == ["工具 `show` 已调用 0/1 次(未满足)"]
        assert "0/1 次" in payload["message"]
        assert "当前条件未满足" in payload["message"]

    def test_preset_goal_content_when_satisfied(self):
        """条件已满足时,objective 标记完成、message 提示将结束。"""
        mw = GoalLoopMiddleware(
            mode="preset",
            objective="必须调用 show 工具",
            conditions=[tool_called_at_least("show", 1)],
            max_rounds=3,
        )
        get_goal = mw.tools[0]
        state = {
            "goal": _preset_goal(),
            "messages": [
                HumanMessage(content="hi"),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "show", "args": {"text": "x"}, "id": "s1"}],
                ),
                ToolMessage(content="ok", tool_call_id="s1"),
            ],
        }
        cmd = get_goal.func(tool_call_id="t1", state=state)
        payload = json.loads(cmd.update["messages"][0].content)
        assert payload["goal"]["objective"] == "必须调用 show 工具(已完成)"
        assert payload["goal"]["conditions"] == ["工具 `show` 已调用 1/1 次(已满足)"]
        assert "全部条件已满足" in payload["message"]

    def test_llm_goal_content(self):
        mw = GoalLoopMiddleware(mode="llm", max_rounds=10)
        get_goal = mw.tools[0]
        state = {"goal": _llm_goal(), "messages": []}
        cmd = get_goal.func(tool_call_id="t1", state=state)
        payload = json.loads(cmd.update["messages"][0].content)
        # rounds=2 已完成 → 展示进行中的第 3 轮
        assert "第 3/5 轮" in payload["message"]
        assert "update_goal" in payload["message"]

    def test_marks_injected_vs_active_call_source(self):
        """返回中明确标注调用来源:系统注入 vs 模型主动。"""
        mw = GoalLoopMiddleware(
            mode="preset",
            objective="必须调用 show 工具",
            conditions=[tool_called_at_least("show", 1)],
            max_rounds=3,
        )
        get_goal = mw.tools[0]
        state = {
            "goal": _preset_goal(),
            "messages": [
                HumanMessage(content="hi"),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "get_goal", "args": {}, "id": f"{_INJECT_PREFIX}a"}],
                ),
            ],
        }
        # 注入调用(id 带前缀)标注「系统注入·自动续跑」,并点明"已结束本轮生成"
        cmd = get_goal.func(tool_call_id=f"{_INJECT_PREFIX}abc", state=state)
        msg = json.loads(cmd.update["messages"][0].content)["message"]
        assert "【系统注入·自动续跑】" in msg
        assert "结束本轮生成" in msg
        # 主动调用(id 无前缀)标注「主动查询」,澄清不会推进轮次
        cmd = get_goal.func(tool_call_id="model-1", state=state)
        msg = json.loads(cmd.update["messages"][0].content)["message"]
        assert "【主动查询】" in msg
        assert "不会推进轮次" in msg


class TestCreateGoalTool:
    def test_creates_goal(self):
        mw = GoalLoopMiddleware(mode="llm", max_rounds=10)
        create_goal = mw.tools[1]
        cmd = create_goal.func(
            objective="写一个贪吃蛇游戏",
            max_goal_rounds=None,
            tool_call_id="t1",
            state={"messages": []},
        )
        goal = cmd.update["goal"]
        assert goal["objective"] == "写一个贪吃蛇游戏"
        assert goal["status"] == "active"
        assert goal["max_rounds"] == 10
        assert goal["revision"] == 1
        assert goal["created_by"] == "llm"

    def test_max_goal_rounds_clamped(self):
        mw = GoalLoopMiddleware(mode="llm", max_rounds=5)
        create_goal = mw.tools[1]
        cmd = create_goal.func(
            objective="大任务", max_goal_rounds=100, tool_call_id="t1", state={}
        )
        assert cmd.update["goal"]["max_rounds"] == 5

        cmd2 = create_goal.func(
            objective="小任务", max_goal_rounds=2, tool_call_id="t1", state={}
        )
        assert cmd2.update["goal"]["max_rounds"] == 2

    def test_rejects_active_goal(self):
        mw = GoalLoopMiddleware(mode="llm", max_rounds=5)
        create_goal = mw.tools[1]
        state = {"goal": _llm_goal(), "messages": []}
        cmd = create_goal.func(
            objective="另一个任务", max_goal_rounds=None, tool_call_id="t1", state=state
        )
        msg = cmd.update["messages"][0]
        assert msg.status == "error"
        assert "已有进行中的目标" in msg.content

    def test_rejects_empty_objective(self):
        mw = GoalLoopMiddleware(mode="llm", max_rounds=5)
        create_goal = mw.tools[1]
        cmd = create_goal.func(
            objective="   ", max_goal_rounds=None, tool_call_id="t1", state={}
        )
        assert cmd.update["messages"][0].status == "error"

    def test_allows_replace_after_complete(self):
        mw = GoalLoopMiddleware(mode="llm", max_rounds=5)
        create_goal = mw.tools[1]
        state = {"goal": _llm_goal(status="complete"), "messages": []}
        cmd = create_goal.func(
            objective="新任务", max_goal_rounds=None, tool_call_id="t1", state=state
        )
        assert cmd.update["goal"]["objective"] == "新任务"


class TestUpdateGoalTool:
    def _mw(self):
        return GoalLoopMiddleware(mode="llm", max_rounds=10)

    def _call(self, mw, goal, **kwargs):
        update_goal = mw.tools[2]
        defaults = {
            "goal_id": goal["id"],
            "revision": goal["revision"],
            "action": "complete",
            "objective": None,
            "max_goal_rounds": None,
            "blocked_reason": None,
            "tool_call_id": "t1",
            "state": {"goal": goal, "messages": []},
        }
        defaults.update(kwargs)
        return update_goal.func(**defaults)

    def test_wrong_id_rejected(self):
        mw = self._mw()
        goal = _llm_goal()
        cmd = self._call(mw, goal, goal_id="goal-wrong")
        msg = cmd.update["messages"][0]
        assert msg.status == "error"
        assert "不匹配" in msg.content
        assert "goal" not in cmd.update

    def test_stale_revision_rejected(self):
        mw = self._mw()
        goal = _llm_goal(revision=3)
        cmd = self._call(mw, goal, revision=2)
        assert cmd.update["messages"][0].status == "error"

    def test_complete(self):
        mw = self._mw()
        goal = _llm_goal()
        cmd = self._call(mw, goal, action="complete")
        new_goal = cmd.update["goal"]
        assert new_goal["status"] == "complete"
        assert new_goal["revision"] == goal["revision"] + 1
        assert "标记完成" in cmd.update["messages"][0].content

    def test_blocked_requires_three_worked_rounds(self):
        """需工作满 3 轮才能声明 blocked;未满直接拒绝且状态不变。"""
        mw = self._mw()
        goal = _llm_goal()
        # 第 1 轮(rounds=0)声明 → 拒绝,零状态变更
        cmd = self._call(mw, {**goal, "rounds": 0}, action="blocked", blocked_reason="API 已废弃")
        assert cmd.update["messages"][0].status == "error"
        assert "至少需工作满 3 轮" in cmd.update["messages"][0].content
        assert "goal" not in cmd.update

        # 第 2 轮(rounds=1)声明 → 拒绝
        cmd = self._call(mw, {**goal, "rounds": 1}, action="blocked", blocked_reason="API 已废弃")
        assert cmd.update["messages"][0].status == "error"
        assert "goal" not in cmd.update

        # 第 3 轮(rounds=2)声明 → 接受,blocked
        cmd = self._call(mw, {**goal, "rounds": 2}, action="blocked", blocked_reason="API 已废弃")
        g3 = cmd.update["goal"]
        assert g3["status"] == "blocked"
        assert g3["blocked_reason"] == "API 已废弃"
        assert g3["revision"] == goal["revision"] + 1
        assert "阻塞已确认" in cmd.update["messages"][0].content

    def test_blocked_threshold_configurable(self):
        """blocked_threshold 可配置。"""
        mw = GoalLoopMiddleware(mode="llm", max_rounds=10, blocked_threshold=2)
        goal = _llm_goal()
        # 第 1 轮(rounds=0)声明 → 拒绝
        cmd = self._call(mw, {**goal, "rounds": 0}, action="blocked", blocked_reason="依赖缺失")
        assert cmd.update["messages"][0].status == "error"
        assert "至少需工作满 2 轮" in cmd.update["messages"][0].content
        assert "goal" not in cmd.update
        # 第 2 轮(rounds=1)声明 → 接受,blocked
        cmd = self._call(mw, {**goal, "rounds": 1}, action="blocked", blocked_reason="依赖缺失")
        g2 = cmd.update["goal"]
        assert g2["status"] == "blocked"
        assert "阻塞已确认" in cmd.update["messages"][0].content

    def test_blocked_reason_does_not_affect_eligibility(self):
        """资格取决于工作轮数,与 reason 是否更换无关。"""
        mw = self._mw()
        goal = _llm_goal(rounds=4, blocked_reason="旧原因")
        # 第 5 轮(rounds=4)声明新原因 → 轮数足够,直接 blocked
        cmd = self._call(mw, goal, action="blocked", blocked_reason="新原因")
        new_goal = cmd.update["goal"]
        assert new_goal["status"] == "blocked"
        assert new_goal["blocked_reason"] == "新原因"

    def test_blocked_without_reason_rejected(self):
        mw = self._mw()
        goal = _llm_goal()
        cmd = self._call(mw, goal, action="blocked", blocked_reason=None)
        assert cmd.update["messages"][0].status == "error"
        assert "blocked_reason" in cmd.update["messages"][0].content

    def test_edit(self):
        mw = self._mw()
        goal = _llm_goal()
        cmd = self._call(
            mw, goal, action="edit", objective="新目标", max_goal_rounds=100
        )
        new_goal = cmd.update["goal"]
        assert new_goal["objective"] == "新目标"
        assert new_goal["max_rounds"] == 10  # clamped to middleware max
        assert new_goal["revision"] == goal["revision"] + 1


# ---------------------------------------------------------------------------
# Integration (full graph with scripted model)
# ---------------------------------------------------------------------------


def _make_agent(model, goal_loop_config, extra_tools=None):
    return create_mambo_agent(
        model,
        backend=StoreBackend(),
        goal_loop=goal_loop_config,
        tools=list(extra_tools or []),
    )


def _invoke(agent, content: str = "测试"):
    return agent.invoke(
        {"messages": [HumanMessage(content=content)]},
        config={"configurable": {"thread_id": "goal-loop-test"}},
    )


class TestIntegrationUserControlled:
    def test_uncooperative_model_stops_at_budget(self):
        """LLM never calls show → forced loop runs max_rounds-1 times then stops."""
        model = _ScriptedModel(script=[])
        agent = _make_agent(
            model,
            GoalLoopConfig(
                mode="preset",
                objective="必须调用 show 工具",
                conditions=[tool_called_at_least("show", 1)],
                max_rounds=3,
            ),
            extra_tools=[_show_tool()],
        )
        result = _invoke(agent)

        # Model called 3 times: done, injected-get_goal, done, injected, done
        assert model.invocations == 3
        assert _injected_count(result["messages"]) == 2
        # No show call anywhere
        assert not any(
            isinstance(m, AIMessage)
            and any(tc.get("name") == "show" for tc in m.tool_calls)
            for m in result["messages"]
        )
        assert result["goal"]["status"] == "active"

    def test_model_complies_after_injection(self):
        """Model calls show on the second turn → loop ends right after."""
        model = _ScriptedModel(
            script=[
                AIMessage(content="first reply"),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "show", "args": {"text": "结果"}, "id": "s1"}],
                ),
                AIMessage(content="done"),
            ]
        )
        agent = _make_agent(
            model,
            GoalLoopConfig(
                mode="preset",
                objective="必须调用 show 工具",
                conditions=[tool_called_at_least("show", 1)],
                max_rounds=3,
            ),
            extra_tools=[_show_tool()],
        )
        result = _invoke(agent)

        assert _injected_count(result["messages"]) == 1
        # show was actually executed
        assert any(
            isinstance(m, ToolMessage) and m.name == "show" for m in result["messages"]
        )

    def test_model_meets_condition_first_turn(self):
        """Condition met on the first pass → no injection at all."""
        model = _ScriptedModel(
            script=[
                AIMessage(
                    content="",
                    tool_calls=[{"name": "show", "args": {"text": "结果"}, "id": "s1"}],
                ),
                AIMessage(content="done"),
            ]
        )
        agent = _make_agent(
            model,
            GoalLoopConfig(
                mode="preset",
                objective="必须调用 show 工具",
                conditions=[tool_called_at_least("show", 1)],
                max_rounds=3,
            ),
            extra_tools=[_show_tool()],
        )
        result = _invoke(agent)

        assert model.invocations == 2
        assert _injected_count(result["messages"]) == 0

    def test_partial_conditions_keep_looping(self):
        """两个条件只满足一个 → 注入继续;全部满足后才结束。"""
        model = _ScriptedModel(
            script=[
                AIMessage(
                    content="",
                    tool_calls=[{"name": "show", "args": {"text": "结果"}, "id": "s1"}],
                ),
                AIMessage(content="只完成了 show,还没调用 read"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "read", "args": {"path": "/config.json"}, "id": "r1"}
                    ],
                ),
                AIMessage(content="done"),
            ]
        )
        agent = _make_agent(
            model,
            GoalLoopConfig(
                mode="preset",
                objective="必须调用 show 和 read",
                conditions=[
                    tool_called_at_least("show", 1),
                    tool_called_at_least("read", 1),
                ],
                max_rounds=3,
            ),
            extra_tools=[_show_tool(), _read_tool()],
        )
        result = _invoke(agent)

        # show 后只满足一个条件 → after_agent 注入 get_goal 继续循环;
        # read 后全部满足才结束
        assert _injected_count(result["messages"]) == 1
        assert model.invocations == 4
        # 两个工具都被实际执行
        tool_names = [
            m.name for m in result["messages"] if isinstance(m, ToolMessage)
        ]
        assert tool_names.count("show") == 1
        assert tool_names.count("read") == 1


class TestIntegrationLLMControlled:
    def test_goal_times_out_at_budget(self):
        """LLM creates a goal but never completes → forced loop until timeout."""
        model = _ScriptedModel(
            script=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "create_goal",
                            "args": {"objective": "写一个贪吃蛇"},
                            "id": "c1",
                        }
                    ],
                ),
                AIMessage(content="开始"),
                AIMessage(content="继续1"),
                AIMessage(content="继续2"),
            ]
        )
        agent = _make_agent(model, GoalLoopConfig(mode="llm", max_rounds=3))
        result = _invoke(agent)

        assert model.invocations == 4  # create + 3 turns
        assert _injected_count(result["messages"]) == 2
        goal = result["goal"]
        assert goal["status"] == "timeout"
        assert goal["rounds"] == 3
        assert goal["created_by"] == "llm"

    def test_no_goal_normal_mode(self):
        """No create_goal → no loop, single turn."""
        model = _ScriptedModel(script=[])
        agent = _make_agent(model, GoalLoopConfig(mode="llm", max_rounds=5))
        result = _invoke(agent)

        assert model.invocations == 1
        assert _injected_count(result["messages"]) == 0
        assert result.get("goal") is None
