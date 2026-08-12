"""Tests for SubAgentMiddleware and subagent support in create_mambo_agent."""

from typing import ClassVar

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import StructuredTool
from langgraph.store.memory import InMemoryStore

from mambo_agents import (
    CompiledSubAgent,
    EventGranularity,
    MamboPlanMiddleware,
    SubAgent,
    SubAgentMiddleware,
    create_mambo_agent,
)
from mambo_agents.backends.store import StoreBackend
from mambo_agents.middleware.subagents import (
    DEFAULT_GENERAL_PURPOSE_DESCRIPTION,
    DEFAULT_SUBAGENT_PROMPT,
    GENERAL_PURPOSE_NAME,
    TASK_SYSTEM_PROMPT,
    _SubagentSpec,
    _build_task_tool,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stub_tool(name: str) -> StructuredTool:
    """Create a minimal stub tool for testing."""
    return StructuredTool.from_function(
        name=name,
        func=lambda **kw: f"{name} result",
        description=f"Stub {name} tool",
    )


def _make_stub_runnable(name: str) -> Runnable:
    """Create a minimal runnable that returns a single message state."""
    from langchain_core.runnables import RunnableLambda

    def _invoke(state, config=None):
        return {"messages": [HumanMessage(content=f"Hello from {name}")]}

    async def _ainvoke(state, config=None):
        return {"messages": [HumanMessage(content=f"Hello from {name}")]}

    return RunnableLambda(_invoke, afunc=_ainvoke)


def _make_tool_runtime(
    state: dict | None = None,
    tool_call_id: str = "call-123",
):
    """Create a minimal ToolRuntime with all required args."""
    from langgraph.prebuilt.tool_node import ToolRuntime as TR

    return TR(
        state=state or {"messages": []},
        config={"configurable": {}},
        tool_call_id=tool_call_id,
        context={},
        stream_writer=lambda data: None,
        store=None,
    )


# ---------------------------------------------------------------------------
# Unit tests – SubAgentMiddleware initialization
# ---------------------------------------------------------------------------


class TestSubAgentMiddlewareInit:
    def test_requires_at_least_one_subagent(self):
        """SubAgentMiddleware must not be created without subagents."""
        backend = StoreBackend(store=InMemoryStore())
        with pytest.raises(ValueError, match="At least one subagent"):
            SubAgentMiddleware(backend=backend, subagents=[])

    def test_requires_model_for_subagent(self):
        """SubAgent must specify 'model' — validated before model init."""
        backend = StoreBackend(store=InMemoryStore())
        subagent = SubAgent(
            name="bad-agent",
            description="Missing model",
            system_prompt="...",
        )
        with pytest.raises(ValueError, match="must specify 'model'"):
            SubAgentMiddleware(backend=backend, subagents=[subagent])

    def test_compiled_subagent(self):
        """CompiledSubAgent works without model/tools fields."""
        backend = StoreBackend(store=InMemoryStore())
        compiled = CompiledSubAgent(
            name="pre-built",
            description="Pre-compiled agent",
            runnable=_make_stub_runnable("pre-built"),
        )
        mw = SubAgentMiddleware(backend=backend, subagents=[compiled])
        assert mw is not None
        assert len(mw.tools) == 1


# ---------------------------------------------------------------------------
# Tests – streaming structure (no model needed)
# ---------------------------------------------------------------------------


class TestStreamingStructure:
    def test_task_tool_accepts_granularity(self):
        """_build_task_tool accepts event_granularity kwarg."""
        specs: list[_SubagentSpec] = [
            _SubagentSpec(
                name="test",
                description="A test",
                runnable=_make_stub_runnable("test"),
            )
        ]
        tool = _build_task_tool(specs, event_granularity="messages")
        assert tool.name == "task"
        assert tool.description is not None
        assert "test" in tool.description

    def test_task_tool_accepts_values_granularity(self):
        """_build_task_tool with 'values' granularity doesn't dup mode."""
        specs: list[_SubagentSpec] = [
            _SubagentSpec(
                name="test",
                description="A test",
                runnable=_make_stub_runnable("test"),
            )
        ]
        tool = _build_task_tool(specs, event_granularity="values")
        assert tool is not None

    def test_task_tool_description_formats_agents(self):
        """Task tool description includes all subagent names and descriptions."""
        specs: list[_SubagentSpec] = [
            _SubagentSpec(
                name="agent-a",
                description="Does A",
                runnable=_make_stub_runnable("a"),
            ),
            _SubagentSpec(
                name="agent-b",
                description="Does B",
                runnable=_make_stub_runnable("b"),
            ),
        ]
        tool = _build_task_tool(specs, event_granularity="updates")
        desc = tool.description
        assert "agent-a" in desc
        assert "Does A" in desc
        assert "agent-b" in desc
        assert "Does B" in desc

    def test_task_tool_custom_description(self):
        """Custom task_description with placeholder is formatted."""
        specs: list[_SubagentSpec] = [
            _SubagentSpec(
                name="x",
                description="X agent",
                runnable=_make_stub_runnable("x"),
            )
        ]
        tool = _build_task_tool(
            specs,
            event_granularity="updates",
            task_description="Available: {available_agents}",
        )
        assert "Available:" in tool.description
        assert "x: X agent" in tool.description

    def test_task_tool_custom_description_no_placeholder(self):
        """Custom task_description without placeholder is used as-is."""
        specs: list[_SubagentSpec] = [
            _SubagentSpec(
                name="x",
                description="X",
                runnable=_make_stub_runnable("x"),
            )
        ]
        tool = _build_task_tool(
            specs,
            event_granularity="updates",
            task_description="Just use the tool.",
        )
        assert tool.description == "Just use the tool."


# ---------------------------------------------------------------------------
# Unit tests – task tool invocation (sync)
# ---------------------------------------------------------------------------


class TestTaskToolInvocation:
    def test_task_rejects_unknown_subagent_type(self):
        """task tool returns error string for unknown subagent_type."""
        specs: list[_SubagentSpec] = [
            _SubagentSpec(
                name="known",
                description="Known agent",
                runnable=_make_stub_runnable("known"),
            )
        ]
        tool = _build_task_tool(specs, event_granularity="updates")

        runtime = _make_tool_runtime(tool_call_id="call-123")
        result = tool.func(
            description="do something",
            subagent_type="unknown",
            runtime=runtime,
        )
        assert isinstance(result, str)
        assert "does not exist" in result
        assert "`known`" in result

    def test_task_requires_tool_call_id(self):
        """task tool raises ValueError without tool_call_id."""
        specs: list[_SubagentSpec] = [
            _SubagentSpec(
                name="agent",
                description="Agent",
                runnable=_make_stub_runnable("agent"),
            )
        ]
        tool = _build_task_tool(specs, event_granularity="updates")

        runtime = _make_tool_runtime(tool_call_id="")  # empty / falsy
        with pytest.raises(ValueError, match="Tool call ID"):
            tool.func(
                description="do",
                subagent_type="agent",
                runtime=runtime,
            )


# ---------------------------------------------------------------------------
# Unit tests – constants
# ---------------------------------------------------------------------------


class TestPromptConstants:
    def test_default_subagent_prompt_not_empty(self):
        assert len(DEFAULT_SUBAGENT_PROMPT) > 0

    def test_task_system_prompt_not_empty(self):
        assert len(TASK_SYSTEM_PROMPT) > 0

    def test_general_purpose_description_not_empty(self):
        assert len(DEFAULT_GENERAL_PURPOSE_DESCRIPTION) > 0

    def test_general_purpose_name_correct(self):
        assert GENERAL_PURPOSE_NAME == "general-purpose"


# ---------------------------------------------------------------------------
# Unit tests – EventGranularity type
# ---------------------------------------------------------------------------


class TestEventGranularityType:
    def test_valid_values(self):
        """Only three valid granularity levels exist."""
        valid: set[EventGranularity] = {"messages", "updates", "values"}
        assert "messages" in valid
        assert "updates" in valid
        assert "values" in valid
        assert len(valid) == 3


# ---------------------------------------------------------------------------
# Integration tests – plans state isolation (parent / subagent)
# ---------------------------------------------------------------------------


class _PlansParentModel(BaseChatModel):
    """Round 1: parallel ``write_plans`` + ``task``; round 2: wrap up."""

    calls: ClassVar[int] = 0

    @property
    def _llm_type(self) -> str:
        return "fake"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        type(self).calls += 1
        if self.calls == 1:
            msg = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_plans",
                        "args": {
                            "plans": [
                                {"content": "parent task", "status": "pending"}
                            ]
                        },
                        "id": "wp1",
                        "type": "tool_call",
                    },
                    {
                        "name": "task",
                        "args": {
                            "description": "sub work",
                            "subagent_type": "sub",
                        },
                        "id": "task1",
                        "type": "tool_call",
                    },
                ],
            )
        else:
            msg = AIMessage(content="all done")
        return ChatResult(generations=[ChatGeneration(message=msg)])


class _PlansSubModel(BaseChatModel):
    """Sub agent: writes its own plans, then finishes."""

    calls: ClassVar[int] = 0

    @property
    def _llm_type(self) -> str:
        return "fake"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        type(self).calls += 1
        if self.calls == 1:
            msg = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_plans",
                        "args": {
                            "plans": [
                                {"content": "sub task", "status": "pending"}
                            ]
                        },
                        "id": "swp1",
                        "type": "tool_call",
                    }
                ],
            )
        else:
            msg = AIMessage(content="subagent finished")
        return ChatResult(generations=[ChatGeneration(message=msg)])


class TestPlansStateIsolation:
    """Regression: parallel ``write_plans`` + ``task`` double-writes parent ``plans``.

    The subagent's ``plans`` must never be forwarded into the parent graph —
    otherwise the parent's ``plans`` channel receives two writes in one
    super-step and ``apply_writes`` raises ``InvalidUpdateError``.
    """

    @pytest.mark.asyncio
    async def test_parallel_write_plans_and_task_keeps_parent_plans(self):
        _PlansParentModel.calls = 0
        _PlansSubModel.calls = 0

        sub_agent = create_mambo_agent(
            _PlansSubModel(), middleware=[MamboPlanMiddleware()]
        )
        parent = create_mambo_agent(
            _PlansParentModel(),
            middleware=[MamboPlanMiddleware()],
            subagents=[
                CompiledSubAgent(name="sub", description="sub", runnable=sub_agent)
            ],
        )
        final = await parent.ainvoke(
            {"messages": [HumanMessage("delegate and plan")]},
            {"configurable": {"thread_id": "t-sub"}},
        )

        plans = final.get("plans")
        assert plans is not None
        assert [p.content for p in plans] == ["parent task"]
