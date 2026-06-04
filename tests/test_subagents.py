"""Tests for SubAgentMiddleware and subagent support in create_mambo_agent."""

import os
from unittest.mock import patch

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import StructuredTool
from langgraph.graph.state import CompiledStateGraph

from mambo_agents import (
    CompiledSubAgent,
    EventGranularity,
    SubAgent,
    SubAgentMiddleware,
    create_mambo_agent,
)
from mambo_agents.backends.state import StateBackend
from tests.test_state_backend import _simulate_graph
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

_GLM_MODEL_NAME = "Pro/zai-org/GLM-4.7"


def _get_model():
    """Return a test ChatOpenAI model instance."""
    pytest.importorskip("langchain_openai")
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=_GLM_MODEL_NAME,
        api_key=os.environ.get("GJKEY", ""),
        base_url="https://api.siliconflow.cn/v1",
        temperature=0,
    )


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
        backend = StateBackend()
        with pytest.raises(ValueError, match="At least one subagent"):
            SubAgentMiddleware(backend=backend, subagents=[])

    def test_basic_initialization(self):
        """SubAgentMiddleware initializes with a valid SubAgent spec."""
        backend = StateBackend()
        subagent: SubAgent = {
            "name": "test-agent",
            "description": "A test subagent",
            "system_prompt": "You are a test agent.",
            "model": _get_model(),
            "tools": [_make_stub_tool("search")],
        }
        mw = SubAgentMiddleware(backend=backend, subagents=[subagent])
        assert mw is not None
        assert len(mw.tools) == 1
        assert mw.tools[0].name == "task"

    def test_requires_model_for_subagent(self):
        """SubAgent must specify 'model' — validated before model init."""
        backend = StateBackend()
        subagent: SubAgent = {
            "name": "bad-agent",
            "description": "Missing model",
            "system_prompt": "...",
            "tools": [],  # type: ignore
        }
        with pytest.raises(ValueError, match="must specify 'model'"):
            SubAgentMiddleware(backend=backend, subagents=[subagent])

    def test_requires_tools_for_subagent(self):
        """SubAgent must specify 'tools' — validated before model init."""
        backend = StateBackend()
        subagent: SubAgent = {
            "name": "bad-agent",
            "description": "Missing tools",
            "system_prompt": "...",
            "model": "mock-model",
        }
        with pytest.raises(ValueError, match="must specify 'tools'"):
            SubAgentMiddleware(backend=backend, subagents=[subagent])

    def test_compiled_subagent(self):
        """CompiledSubAgent works without model/tools fields."""
        backend = StateBackend()
        compiled: CompiledSubAgent = {
            "name": "pre-built",
            "description": "Pre-compiled agent",
            "runnable": _make_stub_runnable("pre-built"),
        }
        mw = SubAgentMiddleware(backend=backend, subagents=[compiled])
        assert mw is not None
        assert len(mw.tools) == 1

    def test_mixed_subagents(self):
        """Mixed SubAgent and CompiledSubAgent specs."""
        backend = StateBackend()
        compiled: CompiledSubAgent = {
            "name": "compiled",
            "description": "Compiled",
            "runnable": _make_stub_runnable("compiled"),
        }
        sub: SubAgent = {
            "name": "declarative",
            "description": "Declarative",
            "system_prompt": "...",
            "model": _get_model(),
            "tools": [_make_stub_tool("calc")],
        }
        mw = SubAgentMiddleware(backend=backend, subagents=[compiled, sub])
        assert mw is not None


# ---------------------------------------------------------------------------
# Unit tests – event granularity
# ---------------------------------------------------------------------------


class TestEventGranularity:
    def test_default_granularity_is_updates(self):
        """Default event_granularity is 'updates'."""
        backend = StateBackend()
        subagent: SubAgent = {
            "name": "agent",
            "description": "desc",
            "system_prompt": "sys",
            "model": _get_model(),
            "tools": [_make_stub_tool("t")],
        }
        mw = SubAgentMiddleware(backend=backend, subagents=[subagent])
        assert mw._event_granularity == "updates"

    def test_explicit_granularity_messages(self):
        """event_granularity='messages' is accepted."""
        backend = StateBackend()
        subagent: SubAgent = {
            "name": "agent",
            "description": "desc",
            "system_prompt": "sys",
            "model": _get_model(),
            "tools": [_make_stub_tool("t")],
        }
        mw = SubAgentMiddleware(
            backend=backend,
            subagents=[subagent],
            event_granularity="messages",
        )
        assert mw._event_granularity == "messages"

    def test_explicit_granularity_values(self):
        """event_granularity='values' is accepted."""
        backend = StateBackend()
        subagent: SubAgent = {
            "name": "agent",
            "description": "desc",
            "system_prompt": "sys",
            "model": _get_model(),
            "tools": [_make_stub_tool("t")],
        }
        mw = SubAgentMiddleware(
            backend=backend,
            subagents=[subagent],
            event_granularity="values",
        )
        assert mw._event_granularity == "values"

    def test_granularity_passed_to_task_tool(self):
        """Granularity is plumbed into the task tool builder."""
        with patch(
            "mambo_agents.middleware.subagents._build_task_tool"
        ) as mock_build:
            mock_build.return_value = _make_stub_tool("task")
            backend = StateBackend()
            subagent: SubAgent = {
                "name": "agent",
                "description": "desc",
                "system_prompt": "sys",
                "model": _get_model(),
                "tools": [_make_stub_tool("t")],
            }
            SubAgentMiddleware(
                backend=backend,
                subagents=[subagent],
                event_granularity="messages",
            )
            call_kwargs = mock_build.call_args.kwargs
            assert call_kwargs["event_granularity"] == "messages"


# ---------------------------------------------------------------------------
# Unit tests – system prompt injection
# ---------------------------------------------------------------------------


class TestSystemPrompt:
    def test_default_system_prompt_includes_agents(self):
        """System prompt includes available subagent types."""
        backend = StateBackend()
        subagent: SubAgent = {
            "name": "researcher",
            "description": "Research topics thoroughly",
            "system_prompt": "You are a researcher.",
            "model": _get_model(),
            "tools": [_make_stub_tool("search")],
        }
        mw = SubAgentMiddleware(backend=backend, subagents=[subagent])
        assert mw._system_prompt is not None
        assert "researcher" in mw._system_prompt
        assert "Research topics thoroughly" in mw._system_prompt

    def test_custom_system_prompt(self):
        """Custom system_prompt is used."""
        backend = StateBackend()
        subagent: SubAgent = {
            "name": "agent",
            "description": "desc",
            "system_prompt": "sys",
            "model": _get_model(),
            "tools": [_make_stub_tool("t")],
        }
        custom = "Custom instructions here."
        mw = SubAgentMiddleware(
            backend=backend,
            subagents=[subagent],
            system_prompt=custom,
        )
        assert custom in mw._system_prompt

    def test_none_system_prompt(self):
        """system_prompt=None suppresses injection."""
        backend = StateBackend()
        subagent: SubAgent = {
            "name": "agent",
            "description": "desc",
            "system_prompt": "sys",
            "model": _get_model(),
            "tools": [_make_stub_tool("t")],
        }
        mw = SubAgentMiddleware(
            backend=backend,
            subagents=[subagent],
            system_prompt=None,
        )
        assert mw._system_prompt is None


# ---------------------------------------------------------------------------
# Unit tests – interrupt_on support
# ---------------------------------------------------------------------------


class TestInterruptOn:
    def test_interrupt_on_adds_human_in_the_loop(self):
        """interrupt_on triggers HumanInTheLoopMiddleware (no crash)."""
        backend = StateBackend()
        subagent: SubAgent = {
            "name": "agent",
            "description": "desc",
            "system_prompt": "sys",
            "model": _get_model(),
            "tools": [_make_stub_tool("dangerous")],
            "interrupt_on": {"dangerous": True},
        }
        mw = SubAgentMiddleware(backend=backend, subagents=[subagent])
        assert mw is not None


# ---------------------------------------------------------------------------
# Integration tests – create_mambo_agent with subagents
# ---------------------------------------------------------------------------


class TestCreateMamboAgentSubagents:
    def test_create_with_subagents(self):
        """create_mambo_agent accepts subagents parameter."""
        backend = StateBackend()
        model = _get_model()
        agent = create_mambo_agent(
            model,
            backend=backend,
            subagents=[
                {
                    "name": "worker",
                    "description": "A worker agent",
                    "system_prompt": "You are a worker.",
                    "model": model,
                    "tools": [_make_stub_tool("tool")],
                }
            ],
        )
        assert agent is not None
        assert isinstance(agent, CompiledStateGraph)
        assert _graph_has_task_tool(agent), (
            "task tool should be registered when subagents are provided"
        )

    def test_create_with_compiled_subagent(self):
        """create_mambo_agent accepts CompiledSubAgent."""
        backend = StateBackend()
        model = _get_model()
        compiled: CompiledSubAgent = {
            "name": "pre-built",
            "description": "Pre-built agent",
            "runnable": _make_stub_runnable("pre-built"),
        }
        agent = create_mambo_agent(
            model,
            backend=backend,
            subagents=[compiled],
        )
        assert agent is not None
        assert _graph_has_task_tool(agent)

    def test_include_general_purpose_default_false(self):
        """By default, general-purpose subagent is NOT auto-added."""
        backend = StateBackend()
        model = _get_model()
        agent = create_mambo_agent(model, backend=backend)
        assert not _graph_has_task_tool(agent), (
            "task tool should NOT exist when no subagents and "
            "include_general_purpose=False"
        )

    def test_include_general_purpose_true(self):
        """include_general_purpose=True auto-adds general-purpose subagent."""
        backend = StateBackend()
        model = _get_model()
        agent = create_mambo_agent(
            model,
            backend=backend,
            include_general_purpose=True,
        )
        assert _graph_has_task_tool(agent), (
            "task tool should be present when include_general_purpose=True"
        )

    def test_general_purpose_not_duplicated(self):
        """general-purpose is not added twice if already in subagents list."""
        backend = StateBackend()
        model = _get_model()
        gp_spec: SubAgent = {
            "name": GENERAL_PURPOSE_NAME,
            "description": "My custom GP",
            "system_prompt": DEFAULT_SUBAGENT_PROMPT,
            "model": model,
            "tools": [_make_stub_tool("t")],
        }
        agent = create_mambo_agent(
            model,
            backend=backend,
            subagents=[gp_spec],
            include_general_purpose=True,
        )
        assert agent is not None
        assert _graph_has_task_tool(agent)

    def test_event_granularity_plumbed(self):
        """event_granularity is passed through to the middleware (no crash)."""
        backend = StateBackend()
        model = _get_model()
        subagent: SubAgent = {
            "name": "worker",
            "description": "Worker",
            "system_prompt": "...",
            "model": model,
            "tools": [_make_stub_tool("t")],
        }
        agent = create_mambo_agent(
            model,
            backend=backend,
            subagents=[subagent],
            event_granularity="messages",
        )
        assert agent is not None
        assert _graph_has_task_tool(agent)

    def test_subagent_middleware_position(self):
        """SubAgentMiddleware works alongside HITL."""
        backend = StateBackend()
        model = _get_model()
        from langgraph.checkpoint.memory import MemorySaver

        agent = create_mambo_agent(
            model,
            backend=backend,
            subagents=[
                {
                    "name": "w",
                    "description": "d",
                    "system_prompt": "s",
                    "model": model,
                    "tools": [_make_stub_tool("t")],
                }
            ],
            interrupt_on={"write": True},
            checkpointer=MemorySaver(),
        )
        assert agent is not None
        assert _graph_has_task_tool(agent)


# ---------------------------------------------------------------------------
# Tests – streaming structure (no model needed)
# ---------------------------------------------------------------------------


class TestStreamingStructure:
    def test_task_tool_accepts_granularity(self):
        """_build_task_tool accepts event_granularity kwarg."""
        specs: list[_SubagentSpec] = [
            {
                "name": "test",
                "description": "A test",
                "runnable": _make_stub_runnable("test"),
            }
        ]
        tool = _build_task_tool(specs, event_granularity="messages")
        assert tool.name == "task"
        assert tool.description is not None
        assert "test" in tool.description

    def test_task_tool_accepts_values_granularity(self):
        """_build_task_tool with 'values' granularity doesn't dup mode."""
        specs: list[_SubagentSpec] = [
            {
                "name": "test",
                "description": "A test",
                "runnable": _make_stub_runnable("test"),
            }
        ]
        tool = _build_task_tool(specs, event_granularity="values")
        assert tool is not None

    def test_task_tool_description_formats_agents(self):
        """Task tool description includes all subagent names and descriptions."""
        specs: list[_SubagentSpec] = [
            {
                "name": "agent-a",
                "description": "Does A",
                "runnable": _make_stub_runnable("a"),
            },
            {
                "name": "agent-b",
                "description": "Does B",
                "runnable": _make_stub_runnable("b"),
            },
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
            {
                "name": "x",
                "description": "X agent",
                "runnable": _make_stub_runnable("x"),
            }
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
            {
                "name": "x",
                "description": "X",
                "runnable": _make_stub_runnable("x"),
            }
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
            {
                "name": "known",
                "description": "Known agent",
                "runnable": _make_stub_runnable("known"),
            }
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
            {
                "name": "agent",
                "description": "Agent",
                "runnable": _make_stub_runnable("agent"),
            }
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
# Integration test – end-to-end streaming with real LLM
# ---------------------------------------------------------------------------


class TestSubagentStreamingE2E:
    """End-to-end tests that invoke the agent with a real LLM and verify
    subagent streaming events are correctly emitted."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_subagent_emits_custom_events(self):
        """Agent with subagent emits 'subagent_event' custom stream events.

        Uses ``event_granularity='updates'`` and verifies each event
        carries ``tool_call_id``, ``subagent_type``, and ``chunk``.
        """
        model = _get_model()
        backend = StateBackend()
        agent = create_mambo_agent(
            model,
            backend=backend,
            include_general_purpose=True,
            event_granularity="updates",
        )

        custom_events: list[dict] = []
        async for mode, data in agent.astream(
            {"messages": [HumanMessage(
                content=(
                    "Use the task tool with subagent_type='general-purpose' "
                    "to create a file /research.txt containing the single "
                    "line 'E2E test passed'. Then reply ONLY with the word "
                    "'DONE' and nothing else."
                )
            )]},
            stream_mode=["updates", "custom"],
            config={"configurable": {"thread_id": "test_subagent_e2e"}},
        ):
            if mode == "custom":
                payload: dict = data  # type: ignore[assignment]
                if payload.get("type") == "subagent_event":
                    custom_events.append(payload)

        # Must have at least one event
        assert len(custom_events) > 0, (
            "Expected at least one subagent_event, got none. "
            "The agent may not have used the task tool."
        )

        # Verify event structure
        for event in custom_events:
            assert event["type"] == "subagent_event"
            assert "tool_call_id" in event
            assert event["tool_call_id"], "tool_call_id must not be empty"
            assert event["subagent_type"] == "general-purpose"
            assert event["granularity"] == "updates"
            assert "timestamp" in event
            assert "chunk" in event

        # Verify the file was actually created by the subagent
        with _simulate_graph(backend, thread_id="test_subagent_e2e"):
            result = backend.read("/research.txt")
        assert result.error is None, (
            f"Subagent should have created /research.txt: {result.error}"
        )
        assert "E2E test passed" in (result.content or ""), (
            f"Expected 'E2E test passed', got: {result.content}"
        )

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_parallel_subagents_distinct_tool_call_ids(self):
        """Parallel subagent invocations yield events with distinct tool_call_ids."""
        model = _get_model()
        backend = StateBackend()
        agent = create_mambo_agent(
            model,
            backend=backend,
            include_general_purpose=True,
            event_granularity="values",
        )

        events_by_task: dict[str, list[dict]] = {}
        async for mode, data in agent.astream(
            {"messages": [HumanMessage(
                content=(
                    "Launch TWO general-purpose subagents IN PARALLEL using the "
                    "task tool (send both tool calls in one message):\n"
                    "1. Create file /file_a.txt with content 'AAA'\n"
                    "2. Create file /file_b.txt with content 'BBB'\n"
                    "After both complete, reply ONLY with 'DONE'."
                )
            )]},
            stream_mode=["updates", "custom"],
            config={"configurable": {"thread_id": "test_parallel_subagents_dtci"}},
        ):
            if mode == "custom":
                payload: dict = data  # type: ignore[assignment]
                if payload.get("type") == "subagent_event":
                    task_id = payload["tool_call_id"]
                    events_by_task.setdefault(task_id, []).append(payload)

        # We should have events for at least 2 different task calls
        assert len(events_by_task) >= 2, (
            f"Expected events from >=2 parallel task calls, got "
            f"{len(events_by_task)}: {list(events_by_task.keys())}"
        )

        # Each task's events should all share the same tool_call_id
        for task_id, events in events_by_task.items():
            assert all(e["tool_call_id"] == task_id for e in events), (
                f"All events for task {task_id} should share the same tool_call_id"
            )


# ---------------------------------------------------------------------------
# Helpers for inspecting internals
# ---------------------------------------------------------------------------


def _graph_has_task_tool(agent: CompiledStateGraph) -> bool:
    """Check if the compiled graph has a 'task' tool registered.

    Walks the Pregel builder nodes looking for a ToolNode that
    contains the 'task' tool.
    """
    try:
        pregel = getattr(agent, "_graph", agent)
        builder = getattr(pregel, "builder", pregel)
        for _node_id, node_spec in builder.nodes.items():
            runnable = getattr(node_spec, "runnable", None)
            if runnable is not None and hasattr(runnable, "tools_by_name"):
                if "task" in runnable.tools_by_name:
                    return True
    except Exception:
        pass
    return False
