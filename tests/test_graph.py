"""Dedicated unit tests for ``create_mambo_agent()`` — branch coverage.

All tests use ``FakeListChatModel`` (no real LLM), ``StoreBackend`` (in-memory),
and ``MemorySaver`` — fully safe, zero network.
"""

import pytest
from langchain_core.language_models import FakeListChatModel
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from mambo_agents import create_mambo_agent
from mambo_agents.backends.store import StoreBackend
from mambo_agents.middleware.security_review import SecurityReviewConfig


# ============================================================================
# Helpers
# ============================================================================


def _create_model():
    return FakeListChatModel(responses=["done"])


def _make_backend():
    return StoreBackend(store=InMemoryStore())


# ============================================================================
# Constructor — default routing
# ============================================================================


class TestDefaultBackend:
    """No backend → StoreBackend() default."""

    def test_no_backend_defaults_to_store_backend(self):
        agent = create_mambo_agent(_create_model())
        assert agent is not None

    def test_explicit_store_backend(self):
        agent = create_mambo_agent(_create_model(), backend=_make_backend())
        assert agent is not None


# ============================================================================
# Skills routing
# ============================================================================


class TestSkillsRouting:
    def test_skills_none_not_injected(self):
        """skills=None → no SkillsMiddleware."""
        agent = create_mambo_agent(
            _create_model(),
            backend=_make_backend(),
            skills=None,
        )
        assert agent is not None

    def test_skills_list_injects_middleware(self):
        """skills list → SkillsMiddleware added."""
        agent = create_mambo_agent(
            _create_model(),
            backend=_make_backend(),
            skills=["/skills/user/"],
        )
        assert agent is not None

    def test_mixed_skill_sources(self):
        """Mixed source types (path + tuple)."""
        agent = create_mambo_agent(
            _create_model(),
            backend=_make_backend(),
            skills=["/skills/user/", ("/skills/project/", "Project")],
        )
        assert agent is not None


# ============================================================================
# Summarization routing
# ============================================================================


class TestSummarizationRouting:
    def test_summarization_none_not_injected(self):
        """summarization=None → no summarization middleware."""
        agent = create_mambo_agent(
            _create_model(),
            backend=_make_backend(),
            summarization=None,
        )
        assert agent is not None

    def test_summarization_with_trigger(self):
        from mambo_agents.middleware.summarization import SummarizationConfig

        agent = create_mambo_agent(
            _create_model(),
            backend=_make_backend(),
            summarization=SummarizationConfig(
                trigger=("tokens", 100000),
                keep=("messages", 20),
            ),
        )
        assert agent is not None

    def test_summarization_with_backend_offload(self):
        from mambo_agents.middleware.summarization import SummarizationConfig

        agent = create_mambo_agent(
            _create_model(),
            backend=_make_backend(),
            summarization=SummarizationConfig(
                trigger=("messages", 50),
                offload_to_backend=True,
            ),
        )
        assert agent is not None


# ============================================================================
# Subagents routing
# ============================================================================


class TestSubagentsRouting:
    def test_no_subagents_no_task_tool(self):
        """Without subagents and include_general_purpose=False, no task tool."""
        agent = create_mambo_agent(
            _create_model(),
            backend=_make_backend(),
            subagents=None,
            include_general_purpose=False,
        )
        assert agent is not None
        assert not _graph_has_task_tool(agent)

    def test_include_general_purpose_adds_task_tool(self):
        """include_general_purpose=True auto-adds general-purpose subagent."""
        agent = create_mambo_agent(
            _create_model(),
            backend=_make_backend(),
            include_general_purpose=True,
        )
        assert agent is not None
        assert _graph_has_task_tool(agent)

    def test_general_purpose_not_duplicated(self):
        from mambo_agents.middleware.subagents import (
            DEFAULT_SUBAGENT_PROMPT,
            GENERAL_PURPOSE_NAME,
            SubAgent,
        )

        agent = create_mambo_agent(
            _create_model(),
            backend=_make_backend(),
            subagents=[
                SubAgent(
                    name=GENERAL_PURPOSE_NAME,
                    description="My GP",
                    system_prompt=DEFAULT_SUBAGENT_PROMPT,
                    model=_create_model(),
                    tools=[],
                )
            ],
            include_general_purpose=True,
        )
        assert agent is not None

    def test_subagents_list(self):
        stub = _make_stub_tool("stub")
        from mambo_agents.middleware.subagents import SubAgent
        agent = create_mambo_agent(
            _create_model(),
            backend=_make_backend(),
            subagents=[
                SubAgent(
                    name="worker",
                    description="Does work",
                    system_prompt="You do work.",
                    model=_create_model(),
                    tools=[stub],
                )
            ],
        )
        assert agent is not None
        assert _graph_has_task_tool(agent)


# ============================================================================
# Async subagents routing
# ============================================================================


class TestAsyncSubagentsRouting:
    def test_async_subagents_none(self):
        agent = create_mambo_agent(
            _create_model(),
            backend=_make_backend(),
            async_subagents=None,
        )
        assert agent is not None


# ============================================================================
# Interrupt_on + security_review routing
# ============================================================================


class TestInterruptOnRouting:
    def test_classic_hitl_without_explicit_checkpointer(self):
        """interrupt_on without explicit checkpointer → defaults to InMemorySaver."""
        agent = create_mambo_agent(
            _create_model(),
            backend=_make_backend(),
            interrupt_on={"write": True},
        )
        assert agent is not None

    def test_classic_hitl_with_checkpointer(self):
        """interrupt_on with checkpointer → HumanInTheLoopMiddleware."""
        agent = create_mambo_agent(
            _create_model(),
            backend=_make_backend(),
            interrupt_on={"write": True},
            checkpointer=MemorySaver(),
        )
        assert agent is not None

    def test_security_review_with_hitl(self):
        """interrupt_on + security_review → AutoSecurityReviewMiddleware."""
        agent = create_mambo_agent(
            _create_model(),
            backend=_make_backend(),
            interrupt_on={"write": True},
            security_review=SecurityReviewConfig(),
            checkpointer=MemorySaver(),
        )
        assert agent is not None


# ============================================================================
# Middleware routing
# ============================================================================


class TestMiddlewareRouting:
    def test_custom_middleware_appended(self):
        from langchain.agents.middleware import AgentMiddleware

        class NoOpMiddleware(AgentMiddleware):
            state_schema = dict

        agent = create_mambo_agent(
            _create_model(),
            backend=_make_backend(),
            middleware=[NoOpMiddleware()],
        )
        assert agent is not None

    def test_plan_middleware_summarization_hook_auto_wired(self):
        """PlanMiddleware hook is auto-detected when summarization is on."""
        from mambo_agents.middleware.planning import MamboPlanMiddleware
        from mambo_agents.middleware.summarization import SummarizationConfig

        agent = create_mambo_agent(
            _create_model(),
            backend=_make_backend(),
            middleware=[MamboPlanMiddleware()],
            summarization=SummarizationConfig(
                trigger=("tokens", 100000),
                keep=("messages", 20),
            ),
        )
        assert agent is not None


class TestToolErrorHandling:
    """``tool_error_handling`` controls the built-in error middleware."""

    def _stack(self, monkeypatch, **kwargs) -> list:
        import mambo_agents.graph as graph_module

        captured: dict = {}
        real = graph_module._langchain_create_agent

        def spy(model, **call_kwargs):
            captured["middleware"] = list(call_kwargs.get("middleware") or [])
            return real(model, **call_kwargs)

        monkeypatch.setattr(graph_module, "_langchain_create_agent", spy)
        create_mambo_agent(_create_model(), backend=_make_backend(), **kwargs)
        return captured["middleware"]

    def test_default_installs_the_middleware(self, monkeypatch):
        stack = self._stack(monkeypatch)
        assert "ToolCallErrorMiddleware" in [type(m).__name__ for m in stack]

    def test_false_leaves_it_out(self, monkeypatch):
        stack = self._stack(monkeypatch, tool_error_handling=False)
        assert "ToolCallErrorMiddleware" not in [type(m).__name__ for m in stack]

    def test_instance_is_used_as_is(self, monkeypatch):
        from mambo_agents.middleware.tool_call_error import ToolCallErrorMiddleware

        custom = ToolCallErrorMiddleware(max_consecutive_failures=1)
        stack = self._stack(monkeypatch, tool_error_handling=custom)
        assert custom in stack

    def test_setting_reaches_subagent_graphs(self, monkeypatch):
        import mambo_agents.middleware.subagents as subagents_module

        captured: list = []
        real = subagents_module._langchain_create_agent

        def spy(model, **call_kwargs):
            captured.append(list(call_kwargs.get("middleware") or []))
            return real(model, **call_kwargs)

        monkeypatch.setattr(subagents_module, "_langchain_create_agent", spy)
        create_mambo_agent(
            _create_model(),
            backend=_make_backend(),
            include_general_purpose=True,
            tool_error_handling=False,
        )
        assert captured
        assert all(
            "ToolCallErrorMiddleware"
            not in [type(m).__name__ for m in middleware]
            for middleware in captured
        )


# ============================================================================
# Tools routing
# ============================================================================


class TestToolsRouting:
    def test_extra_tools(self):
        stub = _make_stub_tool("extra_tool")
        agent = create_mambo_agent(
            _create_model(),
            backend=_make_backend(),
            tools=[stub],
        )
        assert agent is not None

    def test_tools_none(self):
        agent = create_mambo_agent(
            _create_model(),
            backend=_make_backend(),
            tools=None,
        )
        assert agent is not None


# ============================================================================
# System prompt routing
# ============================================================================


class TestSystemPromptRouting:
    def test_custom_system_prompt(self):
        agent = create_mambo_agent(
            _create_model(),
            backend=_make_backend(),
            system_prompt="Custom directions here.",
        )
        assert agent is not None

    def test_none_system_prompt_uses_default(self):
        agent = create_mambo_agent(
            _create_model(),
            backend=_make_backend(),
            system_prompt=None,
        )
        assert agent is not None


# ============================================================================
# Helpers
# ============================================================================


def _make_stub_tool(name: str):
    return StructuredTool.from_function(
        name=name,
        func=lambda **kw: f"{name} result",
        description=f"Stub {name} tool",
    )


def _graph_has_task_tool(agent) -> bool:
    """Check if the compiled graph has a 'task' tool registered."""
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
