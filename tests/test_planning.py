"""Tests for MamboPlanMiddleware and Summarization hook integration."""

import os
from typing import Annotated, get_args, get_origin, get_type_hints
from unittest.mock import MagicMock

import pytest
from langchain.agents.middleware.types import AgentState
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)
from langgraph.types import Overwrite

from mambo_agents import (
    MamboPlanMiddleware,
    MamboSummarizationMiddleware,
    Plan,
    SummarizationConfig,
    SummaryHook,
    SummaryHookContext,
    WritePlansInput,
    create_mambo_agent,
)
from langgraph.store.memory import InMemoryStore

from mambo_agents.backends.store import StoreBackend
from tests.test_store_backend import _simulate_graph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GLM_MODEL_NAME = "Pro/zai-org/GLM-4.7"


def _get_model():
    pytest.importorskip("langchain_openai")
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=_GLM_MODEL_NAME,
        api_key=os.environ.get("GJKEY", ""),
        base_url="https://api.siliconflow.cn/v1",
        temperature=0,
    )


def _make_mock_summary_model(summary_text: str = "Mock summary: plan stuff happened."):
    from unittest.mock import AsyncMock

    mock = MagicMock()
    mock_response = MagicMock()
    mock_response.content = summary_text
    mock.invoke.return_value = mock_response
    mock_async_response = MagicMock()
    mock_async_response.content = summary_text
    mock.ainvoke = AsyncMock(return_value=mock_async_response)
    mock._llm_type = "mock-chat"
    return mock


def _make_request(messages: list, state: dict | None = None):
    def _override(**kwargs):
        new_req = MagicMock()
        new_req.messages = kwargs.get("messages", messages)
        new_req.system_message = kwargs.get("system_message", None)
        new_req.tools = kwargs.get("tools", None)
        new_req.state = kwargs.get("state", state or {})
        new_req.runtime = MagicMock()
        new_req.override = _override
        return new_req

    req = MagicMock()
    req.messages = list(messages)
    req.system_message = None
    req.tools = None
    req.state = state or {}
    req.runtime = MagicMock()
    req.override = _override
    return req


# ---------------------------------------------------------------------------
# Unit tests – Plan model
# ---------------------------------------------------------------------------


class TestPlanModel:
    """Pydantic Plan model – correctness and frozen immutability."""

    def test_create_plan(self):
        p = Plan(content="Refactor payment module", status="pending")
        assert p.content == "Refactor payment module"
        assert p.status == "pending"

    def test_default_status_is_pending(self):
        p = Plan(content="Write tests")
        assert p.status == "pending"

    def test_frozen_prevents_mutation(self):
        p = Plan(content="Task A", status="pending")
        with pytest.raises(Exception):
            p.status = "completed"  # type: ignore[misc]

    def test_invalid_status_rejected(self):
        with pytest.raises(Exception):
            Plan(content="Bad status", status="unknown")  # type: ignore[arg-type]

    def test_serialize_to_dict(self):
        p = Plan(content="Read file", status="in_progress")
        d = p.model_dump()
        assert d == {"content": "Read file", "status": "in_progress"}

    def test_deserialize_from_dict(self):
        d = {"content": "Write unit tests", "status": "completed"}
        p = Plan(**d)
        assert p.content == "Write unit tests"
        assert p.status == "completed"

    def test_list_of_plans_validation(self):
        wi = WritePlansInput(
            plans=[
                Plan(content="A", status="pending"),
                Plan(content="B", status="in_progress"),
            ]
        )
        assert len(wi.plans) == 2


# ---------------------------------------------------------------------------
# Unit tests – MamboPlanMiddleware
# ---------------------------------------------------------------------------


class TestPlanMiddleware:
    def test_initialization(self):
        mw = MamboPlanMiddleware()
        assert mw is not None
        assert len(mw.tools) == 1
        assert mw.tools[0].name == "write_plans"

    def test_custom_prompts(self):
        mw = MamboPlanMiddleware(
            system_prompt="Custom system prompt.",
            tool_description="Custom tool desc.",
        )
        assert mw._system_prompt == "Custom system prompt."
        assert mw._tool_description == "Custom tool desc."

    def test_tool_has_correct_schema(self):
        mw = MamboPlanMiddleware()
        tool = mw.tools[0]
        assert tool.args_schema is WritePlansInput

    def test_wrap_model_call_injects_prompt(self):
        mw = MamboPlanMiddleware(system_prompt="USE PLANS")
        messages = [HumanMessage(content="Hello")]
        request = _make_request(messages)

        received = []

        def handler(req):
            received.append(req)
            return "response"

        mw.wrap_model_call(request, handler)
        assert len(received) == 1
        sys_msg = received[0].system_message
        assert sys_msg is not None
        # Content blocks should contain our prompt
        content_str = sys_msg.text if hasattr(sys_msg, 'text') else str(sys_msg.content)
        assert "USE PLANS" in content_str

    def test_wrap_model_call_no_system_message(self):
        """When no system_message exists, a new one is created."""
        mw = MamboPlanMiddleware(system_prompt="PLAN PROMPT")
        messages = [HumanMessage(content="Hi")]
        request = _make_request(messages)
        request.system_message = None

        received = []

        def handler(req):
            received.append(req)
            return "ok"

        mw.wrap_model_call(request, handler)
        assert received[0].system_message is not None


# ---------------------------------------------------------------------------
# Unit tests – Parallel write_plans prevention
# ---------------------------------------------------------------------------


class TestParallelWritePlansPrevention:
    def test_single_write_plans_allowed(self):
        """One write_plans call per turn → no error."""
        mw = MamboPlanMiddleware()
        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "write_plans", "id": "t1", "args": {"plans": []}}],
                )
            ]
        }
        result = mw.after_model(state, MagicMock())
        assert result is None

    def test_multiple_write_plans_rejected(self):
        """Multiple parallel write_plans calls → error ToolMessages."""
        mw = MamboPlanMiddleware()
        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "write_plans", "id": "t1", "args": {"plans": []}},
                        {"name": "write_plans", "id": "t2", "args": {"plans": []}},
                    ],
                )
            ]
        }
        result = mw.after_model(state, MagicMock())
        assert result is not None
        assert "messages" in result
        error_msgs = result["messages"]
        assert len(error_msgs) == 2
        for em in error_msgs:
            assert em.status == "error"
            assert "multiple times" in em.content

    def test_mixed_tool_calls_ignore_non_write_plans(self):
        """Other tool calls alongside single write_plans → fine."""
        mw = MamboPlanMiddleware()
        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "read", "id": "r1", "args": {}},
                        {"name": "write_plans", "id": "t1", "args": {"plans": []}},
                        {"name": "grep", "id": "g1", "args": {}},
                    ],
                )
            ]
        }
        result = mw.after_model(state, MagicMock())
        assert result is None

    def test_no_tool_calls(self):
        mw = MamboPlanMiddleware()
        state = {"messages": [AIMessage(content="Just chatting")]}
        result = mw.after_model(state, MagicMock())
        assert result is None

    def test_no_messages(self):
        mw = MamboPlanMiddleware()
        state = {"messages": []}
        result = mw.after_model(state, MagicMock())
        assert result is None


# ---------------------------------------------------------------------------
# Unit tests – build_summary_hook
# ---------------------------------------------------------------------------


class TestBuildSummaryHook:
    def test_returns_callable(self):
        hook = MamboPlanMiddleware.build_summary_hook()
        assert callable(hook)

    def test_no_plans_returns_none(self):
        hook = MamboPlanMiddleware.build_summary_hook()
        ctx = SummaryHookContext(
            state={"plans": []},
            messages_to_summarize=[],
            preserved_messages=[],
        )
        result = hook(ctx)
        assert result is None

    def test_plans_key_missing_returns_none(self):
        hook = MamboPlanMiddleware.build_summary_hook()
        ctx = SummaryHookContext(
            state={},
            messages_to_summarize=[],
            preserved_messages=[],
        )
        result = hook(ctx)
        assert result is None

    def test_all_completed_returns_none(self):
        """When every plan is completed, suppress to avoid hallucination."""
        hook = MamboPlanMiddleware.build_summary_hook()
        ctx = SummaryHookContext(
            state={
                "plans": [
                    Plan(content="Task A", status="completed"),
                    Plan(content="Task B", status="completed"),
                ]
            },
            messages_to_summarize=[],
            preserved_messages=[],
        )
        result = hook(ctx)
        assert result is None

    def test_active_plans_return_boundary_wrapped_content(self):
        """Has active plans → return formatted text with boundary markers."""
        hook = MamboPlanMiddleware.build_summary_hook()
        ctx = SummaryHookContext(
            state={
                "plans": [
                    Plan(content="Analyze routes", status="completed"),
                    Plan(content="Decouple cache", status="in_progress"),
                    Plan(content="Update tests", status="pending"),
                ]
            },
            messages_to_summarize=[],
            preserved_messages=[],
        )
        result = hook(ctx)
        assert result is not None

        # Boundary markers present
        assert "====" in result
        assert "[ATTACHED STATE:" in result
        assert "Current Task List" in result
        assert "write_plans" in result
        assert "authoritative" in result

        # Each plan status icon
        assert "✅" in result
        assert "🔄" in result
        assert "⬜" in result

        # Content included
        assert "Decouple cache" in result
        assert "Update tests" in result

    def test_coerces_plain_dicts_to_plan(self):
        """Hook handles plain dicts (e.g. from checkpoint restore)."""
        hook = MamboPlanMiddleware.build_summary_hook()
        ctx = SummaryHookContext(
            state={
                "plans": [
                    {"content": "Dict task", "status": "pending"},
                ]
            },
            messages_to_summarize=[],
            preserved_messages=[],
        )
        result = hook(ctx)
        assert result is not None
        assert "Dict task" in result
        assert "⬜" in result

    def test_empty_after_coercion_returns_none(self):
        """Empty dict list yields empty after coercion → None."""
        hook = MamboPlanMiddleware.build_summary_hook()
        ctx = SummaryHookContext(
            state={
                "plans": [],
            },
            messages_to_summarize=[],
            preserved_messages=[],
        )
        result = hook(ctx)
        assert result is None


# ---------------------------------------------------------------------------
# Unit tests – SummaryHookContext
# ---------------------------------------------------------------------------


class TestSummaryHookContext:
    def test_frozen_immutable(self):
        ctx = SummaryHookContext(
            state={"plans": []},
            messages_to_summarize=[],
            preserved_messages=[],
        )
        with pytest.raises(Exception):
            ctx.state = {}  # type: ignore[misc]

    def test_holds_all_fields(self):
        ctx = SummaryHookContext(
            state={"a": 1},
            messages_to_summarize=[HumanMessage(content="H1")],
            preserved_messages=[AIMessage(content="A1")],
        )
        assert ctx.state == {"a": 1}
        assert ctx.messages_to_summarize is not None
        assert ctx.preserved_messages is not None


# ---------------------------------------------------------------------------
# Unit tests – MamboSummarizationMiddleware with hooks
# ---------------------------------------------------------------------------


class TestSummarizationWithHooks:
    def test_no_hooks_does_not_inject(self):
        """Without hooks, summary message has no extra content beyond default."""
        mock = _make_mock_summary_model("Summary text.")
        mw = MamboSummarizationMiddleware(
            model=mock,
            trigger=("messages", 3),
            keep=("messages", 2),
        )

        messages = [
            HumanMessage(content="U1"),
            AIMessage(content="A1"),
            ToolMessage(content="T1", tool_call_id="c1"),
            HumanMessage(content="U2"),
            AIMessage(content="A2"),
        ]
        request = _make_request(messages)
        received = []

        def handler(req):
            received.append(list(req.messages))
            return "ok"

        mw.wrap_model_call(request, handler)
        modified = received[0]
        assert isinstance(modified[0], HumanMessage)
        content = modified[0].content
        assert "Summary text" in content
        # No boundary markers from hooks
        assert "====" not in content
        assert "[ATTACHED STATE:" not in content

    def test_with_plan_hook_injects_content(self):
        """Hook registered → summary message includes plan state."""
        mock = _make_mock_summary_model("Summary text.")

        def custom_hook(ctx: SummaryHookContext) -> str | None:
            plans = ctx.state.get("plans", [])
            if not plans:
                return None
            return "CUSTOM_PLAN_MARKER: " + ", ".join(
                p.content for p in plans if isinstance(p, Plan)
            )

        mw = MamboSummarizationMiddleware(
            model=mock,
            trigger=("messages", 3),
            keep=("messages", 2),
            summary_hooks=[custom_hook],
        )

        messages = [
            HumanMessage(content="U1"),
            AIMessage(content="A1"),
            ToolMessage(content="T1", tool_call_id="c1"),
            HumanMessage(content="U2"),
            AIMessage(content="A2"),
        ]
        state = {"plans": [Plan(content="Refactor", status="in_progress")]}
        request = _make_request(messages, state=state)
        received = []

        def handler(req):
            received.append(list(req.messages))
            return "ok"

        mw.wrap_model_call(request, handler)
        modified = received[0]
        assert isinstance(modified[0], HumanMessage)
        content = modified[0].content

        assert "Summary text" in content
        assert "---" in content, "Hook separator '---' should appear"
        assert "CUSTOM_PLAN_MARKER" in content
        assert "Refactor" in content

    def test_hook_returns_none__no_injection(self):
        """Hook returning None → no extra content appended."""
        mock = _make_mock_summary_model("Summary.")

        def noop_hook(ctx: SummaryHookContext) -> str | None:
            return None

        mw = MamboSummarizationMiddleware(
            model=mock,
            trigger=("messages", 3),
            keep=("messages", 2),
            summary_hooks=[noop_hook],
        )

        messages = [
            HumanMessage(content="U1"),
            AIMessage(content="A1"),
            ToolMessage(content="T1", tool_call_id="c1"),
            HumanMessage(content="U2"),
            AIMessage(content="A2"),
        ]
        request = _make_request(messages)
        received = []

        def handler(req):
            received.append(list(req.messages))
            return "ok"

        mw.wrap_model_call(request, handler)
        modified = received[0]
        content = modified[0].content
        assert "Summary." in content
        # No separator because nothing was injected
        assert "---" not in content

    def test_register_hook_runtime(self):
        """register_hook() can add hooks after construction."""
        mock = _make_mock_summary_model("Summary.")
        mw = MamboSummarizationMiddleware(model=mock, trigger=("messages", 3), keep=("messages", 2))

        def late_hook(ctx: SummaryHookContext) -> str | None:
            return "LATE_HOOK_CONTENT"

        mw.register_hook(late_hook)

        messages = [
            HumanMessage(content="U1"),
            AIMessage(content="A1"),
            ToolMessage(content="T1", tool_call_id="c1"),
            HumanMessage(content="U2"),
            AIMessage(content="A2"),
        ]
        request = _make_request(messages)
        received = []

        def handler(req):
            received.append(list(req.messages))
            return "ok"

        mw.wrap_model_call(request, handler)
        content = received[0][0].content
        assert "LATE_HOOK_CONTENT" in content


# ---------------------------------------------------------------------------
# Integration tests – create_mambo_agent auto-wiring
# ---------------------------------------------------------------------------


class TestCreateAgentAutoWiring:
    def test_plan_hook_auto_wired_with_summarization(self):
        """When both PlanMiddleware and summarization are active,
        the plan hook is automatically wired."""
        backend = StoreBackend(store=InMemoryStore())

        plan_mw = MamboPlanMiddleware()
        hook = plan_mw.build_summary_hook()

        ctx = SummaryHookContext(
            state={"plans": [Plan(content="Task 1", status="in_progress")]},
            messages_to_summarize=[],
            preserved_messages=[],
        )
        result = hook(ctx)
        assert result is not None
        assert "Task 1" in result
        assert "🔄" in result

    @pytest.mark.integration
    def test_create_agent_with_plan_and_summarization(self):
        """Agent creation with both PlanMiddleware and summarization succeeds."""
        model = _get_model()
        backend = StoreBackend(store=InMemoryStore())
        agent = create_mambo_agent(
            model,
            backend=backend,
            middleware=[MamboPlanMiddleware()],
            summarization=SummarizationConfig(
                trigger=("tokens", 100000),
                keep=("messages", 20),
            ),
        )
        assert agent is not None

    @pytest.mark.integration
    def test_summarization_alone__no_plan(self):
        """Summarization without PlanMiddleware works fine."""
        model = _get_model()
        backend = StoreBackend(store=InMemoryStore())
        agent = create_mambo_agent(
            model,
            backend=backend,
            summarization=SummarizationConfig(trigger=("messages", 50)),
        )
        assert agent is not None

    @pytest.mark.integration
    def test_plan_alone__no_summarization(self):
        """PlanMiddleware without summarization works fine."""
        model = _get_model()
        backend = StoreBackend(store=InMemoryStore())
        agent = create_mambo_agent(
            model,
            backend=backend,
            middleware=[MamboPlanMiddleware()],
        )
        assert agent is not None


# ---------------------------------------------------------------------------
# Integration tests – E2E with real LLM
# ---------------------------------------------------------------------------


class TestPlanSummarizationE2E:
    """End-to-end: agent with PlanMiddleware + Summarization preserves plan state."""

    @pytest.mark.integration
    def test_e2e_plan_agent_writes_and_updates_plans(self):
        """Agent uses write_plans and state is tracked."""
        model = _get_model()
        backend = StoreBackend(store=InMemoryStore())
        agent = create_mambo_agent(
            model,
            backend=backend,
            middleware=[MamboPlanMiddleware()],
        )

        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "You have a write_plans tool. Please use it to create "
                            "a plan list with exactly these 3 items:\n"
                            "1. 'Create file /t1.txt' (mark as in_progress)\n"
                            "2. 'Create file /t2.txt'\n"
                            "3. 'Create file /t3.txt'\n\n"
                            "After writing the plan list, actually create all "
                            "three files and update the status accordingly. "
                            "All three files should contain the text 'E2E_OK'.\n"
                            "After all files are created, reply with 'ALL_COMPLETE'."
                        )
                    )
                ]
            },
            config={"configurable": {"thread_id": "test_plan_e2e_1"}},
        )

        # Verify plans were tracked
        plans = result.get("plans")
        assert plans is not None, "Agent should have used write_plans"
        assert len(plans) >= 1, "At least one plan item should exist"

        # All should be completed
        for p in plans:
            if isinstance(p, Plan):
                assert p.status == "completed", (
                    f"Plan '{p.content}' should be completed, got {p.status}"
                )
            elif isinstance(p, dict):
                assert p.get("status") == "completed", (
                    f"Plan dict should be completed: {p}"
                )

        # Verify files were actually created
        for fname in ["/t1.txt", "/t2.txt", "/t3.txt"]:
            with _simulate_graph(backend, thread_id="test_plan_e2e_1"):
                r = backend.read(fname)
            assert r.error is None, f"File {fname} should exist: {r.error}"
            assert "E2E_OK" in (r.content or ""), (
                f"File {fname} content mismatch: {r.content}"
            )

        # Verify agent completed
        messages = result.get("messages", [])
        last_ai = None
        for m in reversed(messages):
            if isinstance(m, AIMessage) and not m.tool_calls:
                last_ai = m
                break
        assert last_ai is not None
        assert "ALL_COMPLETE" in (last_ai.content or "").upper()

    @pytest.mark.integration
    def test_e2e_plan_with_aggressive_summarization(self):
        """Plan state survives summarization compaction (key scenario)."""
        model = _get_model()
        backend = StoreBackend(store=InMemoryStore())

        plan_mw = MamboPlanMiddleware()

        agent = create_mambo_agent(
            model,
            backend=backend,
            middleware=[plan_mw],
            summarization=SummarizationConfig(
                trigger=("messages", 8),
                keep=("messages", 3),
            ),
        )

        thread_cfg = {"configurable": {"thread_id": "test_plan_e2e_2"}}

        # Phase 1: Create plan list and do some work
        result1 = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "Use write_plans to create a plan list with exactly 2 items:\n"
                            "1. 'Write file A' (mark it in_progress immediately)\n"
                            "2. 'Write file B'\n\n"
                            "Then do item 1: create /a.txt with content 'PHASE1_DONE'. "
                            "After that, reply 'PHASE1_COMPLETE'."
                        )
                    )
                ]
            },
            config=thread_cfg,
        )

        plans1 = result1.get("plans")
        assert plans1 is not None, "Agent should have used write_plans in phase 1"

        # Verify a.txt was created in phase 1 (needed for phase 2 assertions)
        with _simulate_graph(backend, thread_id="test_plan_e2e_2"):
            r_a_phase1 = backend.read("/a.txt")
        if r_a_phase1.error is not None:
            last_ai_msg = None
            for m in reversed(result1.get("messages", [])):
                if isinstance(m, AIMessage) and not m.tool_calls:
                    last_ai_msg = m
                    break
            pytest.fail(
                f"Phase 1 agent did NOT create /a.txt (required for subsequent steps). "
                f"Last AI response: {last_ai_msg.content if last_ai_msg else 'N/A'}. "
                f"Read error: {r_a_phase1.error}"
            )

        # Phase 2: Continue working — this will generate enough messages
        # to trigger summarization, compacting the phase 1 plan interaction.
        result2 = agent.invoke(
            {
                "messages": [
                    *result1["messages"],
                    HumanMessage(
                        content=(
                            "Now do item 2 from your plan list: create /b.txt "
                            "with content 'PHASE2_DONE'. "
                            "Also read /a.txt to confirm it still exists. "
                            "Then create /c.txt with the combined content of "
                            "both a.txt and b.txt (format: 'A:<content> B:<content>').\n"
                            "Use write_plans to update the list. "
                            "Reply with 'ALL_DONE'."
                        )
                    ),
                ]
            },
            config=thread_cfg,
        )

        # Verify b.txt was created
        with _simulate_graph(backend, thread_id="test_plan_e2e_2"):
            r_b = backend.read("/b.txt")
        assert r_b.error is None
        assert "PHASE2_DONE" in (r_b.content or "")

        # Verify a.txt still exists (phase 1 work not lost)
        with _simulate_graph(backend, thread_id="test_plan_e2e_2"):
            r_a = backend.read("/a.txt")
        assert r_a.error is None
        assert "PHASE1_DONE" in (r_a.content or "")

        # Verify c.txt was created with combined content
        with _simulate_graph(backend, thread_id="test_plan_e2e_2"):
            r_c = backend.read("/c.txt")
        assert r_c.error is None
        assert "PHASE1_DONE" in (r_c.content or "")
        assert "PHASE2_DONE" in (r_c.content or "")

        # Verify plans are complete
        plans2 = result2.get("plans")
        assert plans2 is not None, "Plan list should persist after summarization"
        for p in plans2:
            if isinstance(p, Plan):
                assert p.status == "completed", f"Plan {p.content} should be completed"
            elif isinstance(p, dict):
                assert p.get("status") == "completed", f"Plan dict should be completed: {p}"

        messages = result2.get("messages", [])
        last_ai = None
        for m in reversed(messages):
            if isinstance(m, AIMessage) and not m.tool_calls:
                last_ai = m
                break
        assert last_ai is not None
        assert "ALL_DONE" in (last_ai.content or "").upper()


# =============================================================================
# PlanningState schema regression — Overwrite leak bug
# =============================================================================


class TestPlanningStateSchema:
    """Verify PlanningState correctly inherits AgentState with ``add_messages``.

    Regression test for: ``PlanningState`` was an independent ``TypedDict`` that
    re-declared ``messages`` with ``...`` (Ellipsis) instead of the ``add_messages``
    reducer.  When the factory merged multiple middleware state schemas with
    "last wins" semantics, the reducer-less annotation could overwrite the correct
    one, degrading the ``messages`` channel from ``BinaryOperatorAggregate`` to
    ``LastValue``.  ``LastValue`` does not recognise ``Overwrite``, so the raw
    ``Overwrite`` object leaked into ``state["messages"]`` and caused
    ``TypeError: 'Overwrite' object is not iterable`` downstream.
    """

    # ------------------------------------------------------------------
    # Schema structure
    # ------------------------------------------------------------------

    def test_planning_state_inherits_from_agent_state(self):
        """PlanningState must be a subclass of AgentState (not an isolated TypedDict).

        Uses ``__orig_bases__`` because TypedDict's metaclass rewrites
        ``__bases__`` to ``(Generic, dict)`` at runtime.
        """
        from mambo_agents.middleware.planning import PlanningState

        orig_bases = getattr(PlanningState, "__orig_bases__", ())
        assert AgentState in orig_bases, (
            "PlanningState must inherit AgentState so that 'messages' gets the "
            "add_messages reducer. Found orig_bases: {orig_bases}"
        )

    def test_messages_field_has_callable_reducer(self):
        """messages annotation on PlanningState must carry a callable reducer."""
        from mambo_agents.middleware.planning import PlanningState

        hints = get_type_hints(PlanningState, include_extras=True)
        assert "messages" in hints, (
            "PlanningState must expose 'messages' (inherited from AgentState)"
        )

        msg_type = hints["messages"]
        metadata = _get_annotated_metadata(msg_type)

        assert any(callable(m) for m in metadata), (
            "messages must have a callable reducer (add_messages), "
            f"got metadata: {metadata}"
        )

    def test_planning_state_does_not_redeclare_messages_without_reducer(self):
        """PlanningState must not redeclare ``messages`` with a non-callable reducer.

        Before the fix it redundantly declared
        ``messages: Annotated[list[AnyMessage], ...]`` (Ellipsis instead of
        ``add_messages``).  After the fix it simply inherits from AgentState.
        """
        from mambo_agents.middleware.planning import PlanningState

        own_anns = getattr(PlanningState, "__annotations__", {})
        assert "messages" in own_anns, "PlanningState must inherit 'messages' from AgentState"

        # Check the raw annotation string — it must reference add_messages,
        # not Ellipsis (which would appear as '...' or 'Ellipsis').
        raw = own_anns["messages"]
        raw_str = str(raw)
        assert "add_messages" in raw_str, (
            f"messages annotation must reference add_messages, got: {raw_str[:120]}"
        )

    # ------------------------------------------------------------------
    # Factory schema-merge simulation
    # ------------------------------------------------------------------

    def test_schema_merge_preserves_reducer_in_all_orders(self):
        """Simulate factory ``_resolve_schema`` — messages always has a reducer.

        The factory collects middleware ``state_schema`` values into a ``set``,
        then iterates with ``all_annotations[name] = type`` (last wins).
        This test verifies that **regardless of iteration order**, the
        ``messages`` annotation always carries a callable reducer.
        """
        from mambo_agents.middleware.memory import MemoryState
        from mambo_agents.middleware.planning import PlanningState
        from mambo_agents.middleware.skills import SkillsState
        from mambo_agents.middleware.summarization import SummarizationState

        schemas = [
            AgentState,
            MemoryState,
            SummarizationState,
            SkillsState,
            PlanningState,
        ]

        # Try three different iteration orders
        orders = [
            schemas,                        # natural order
            list(reversed(schemas)),        # reversed
            [s for s in schemas if s is not PlanningState] + [PlanningState],  # planning last
        ]

        for order in orders:
            all_annotations: dict = {}
            for schema in order:
                hints = get_type_hints(schema, include_extras=True)
                for fname, ftype in hints.items():
                    all_annotations[fname] = ftype  # last-wins

            assert "messages" in all_annotations
            msg_type = all_annotations["messages"]
            metadata = _get_annotated_metadata(msg_type)

            assert any(callable(m) for m in metadata), (
                f"messages must have callable reducer in all merge orders. "
                f"Order: {[s.__name__ for s in order]}, "
                f"metadata: {metadata}"
            )

    # ------------------------------------------------------------------
    # LangGraph channel behaviour
    # ------------------------------------------------------------------

    def test_binary_operator_aggregate_unwraps_overwrite(self):
        """``BinaryOperatorAggregate`` (created by ``add_messages``) unwraps ``Overwrite``.

        This is the healthy behaviour — when the channel sees an ``Overwrite``
        it stores the inner value, not the wrapper.
        """
        from langgraph.graph.message import add_messages
        from langgraph.channels.binop import BinaryOperatorAggregate

        channel = BinaryOperatorAggregate(typ=list, operator=add_messages)

        # Normal update
        channel.update([[HumanMessage(content="first")]])
        assert isinstance(channel.get(), list)
        assert len(channel.get()) == 1

        # Overwrite — must be unwrapped
        channel.update([Overwrite(value=[HumanMessage(content="replaced")])])
        result = channel.get()
        assert isinstance(result, list), (
            f"BinaryOperatorAggregate must unwrap Overwrite, got {type(result)}"
        )
        assert len(result) == 1
        assert result[0].content == "replaced"

    def test_last_value_channel_leaks_overwrite(self):
        """Document the failure mode: ``LastValue`` channel leaks ``Overwrite``.

        When ``PlanningState`` did **not** carry ``add_messages``, LangGraph
        created a ``LastValue`` channel for ``messages``.  ``LastValue`` stores
        ``Overwrite`` as-is, causing ``TypeError`` downstream.
        """
        from langgraph.channels.last_value import LastValue

        channel = LastValue(typ=object)

        # Normal update
        channel.update([[HumanMessage(content="first")]])

        # Overwrite — LastValue stores the wrapper (the bug)
        channel.update([Overwrite(value=[HumanMessage(content="replaced")])])
        result = channel.get()

        assert isinstance(result, Overwrite), (
            "LastValue should NOT unwrap Overwrite — this is the documented failure"
        )
        with pytest.raises(TypeError, match="not iterable"):
            list(result)

    # ------------------------------------------------------------------
    # Defensive unwrap in summarization middleware
    # ------------------------------------------------------------------

    def test__apply_event__unwraps_overwrite(self):
        """``_apply_event_to_messages`` defensively unwraps ``Overwrite`` input."""
        from mambo_agents.middleware.summarization import MamboSummarizationMiddleware

        # Call the static method directly
        messages = [HumanMessage(content="hi"), AIMessage(content="hey")]
        result = MamboSummarizationMiddleware._apply_event_to_messages(messages, None)
        assert result == messages

        # When messages accidentally arrive as Overwrite (defense)
        wrapped = Overwrite(value=messages)
        result = MamboSummarizationMiddleware._apply_event_to_messages(wrapped, None)
        assert result == messages

    def test__apply_event__with_summarization_event__unwraps_overwrite(self):
        """``_apply_event_to_messages`` handles Overwrite + summarization event."""
        from mambo_agents.middleware.summarization import MamboSummarizationMiddleware

        summary_msg = AIMessage(content="Summary so far...")
        event = {
            "summary_message": summary_msg,
            "cutoff_index": 1,
        }
        messages = [
            HumanMessage(content="U1"),
            HumanMessage(content="U2"),
            HumanMessage(content="U3"),
        ]

        # Normal path
        result = MamboSummarizationMiddleware._apply_event_to_messages(messages, event)
        assert isinstance(result, list)
        assert result[0] is summary_msg
        assert result[1:] == messages[1:]

        # Defensive Overwrite unwrap with event
        wrapped = Overwrite(value=messages)
        result = MamboSummarizationMiddleware._apply_event_to_messages(wrapped, event)
        assert isinstance(result, list)
        assert result[0] is summary_msg
        assert result[1:] == messages[1:]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _get_annotated_metadata(hint):
    """Return metadata list from an ``Annotated[T, m1, m2, ...]`` type.

    Unwraps ``Required`` / ``NotRequired`` wrappers that TypedDict fields
    commonly carry (e.g. ``Required[Annotated[list[AnyMessage], add_messages]]``).

    Returns an empty list if *hint* is not ``Annotated`` (even after unwrapping).
    """
    # Unwrap Required / NotRequired
    origin = get_origin(hint)
    if origin is not None:
        origin_name = getattr(origin, "__name__", str(origin))
        if origin_name in ("Required", "NotRequired"):
            hint = get_args(hint)[0]

    if get_origin(hint) is Annotated:
        return list(get_args(hint))[1:]
    return []
