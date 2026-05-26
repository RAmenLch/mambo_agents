"""Tests for AutoSecurityReviewMiddleware — AI-based gate before HITL.

**Safety**: all tests use mocked LLM calls (no real API), mocked
``interrupt()``, and ``StateBackend`` (in-memory — no real filesystem).
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.language_models import BaseChatModel, FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.runtime import Runtime
from pydantic import ValidationError

from mambo_agents.middleware.security_review import (
    DEFAULT_SECURITY_REVIEW_SYSTEM_PROMPT,
    AutoSecurityReviewMiddleware,
    SecurityReviewConfig,
    SecurityReviewResult,
    _build_review_messages,
)


# =============================================================================
# Shared fixtures / helpers
# =============================================================================


def _make_tool_call(
    name: str = "write",
    args: dict[str, Any] | None = None,
    call_id: str = "call_1",
) -> dict[str, Any]:
    return {"name": name, "args": args or {"file_path": "/test.txt", "content": "x"}, "id": call_id, "type": "tool_call"}


_EMPTY_RUNTIME = Runtime(
    context=None,
    store=None,
    stream_writer=lambda v: None,
    previous=None,
    execution_info=None,
    server_info=None,
)

_SAFE_RESULT = SecurityReviewResult(
    is_safe=True, reason="Looks fine", risk_level="low",
)
_UNSAFE_RESULT = SecurityReviewResult(
    is_safe=False, reason="Writes outside project workspace", risk_level="high",
)


@pytest.fixture
def mock_model() -> Generator[MagicMock, None, None]:
    """A fully mocked BaseChatModel — safe, zero real calls."""
    model = MagicMock(spec=BaseChatModel)
    yield model


# =============================================================================
# 1. Pydantic models
# =============================================================================


class TestSecurityReviewConfig:
    def test_defaults(self) -> None:
        cfg = SecurityReviewConfig()
        assert cfg.model is None
        assert cfg.system_prompt is None
        assert cfg.review_tools == "all"

    def test_custom_model(self) -> None:
        cfg = SecurityReviewConfig(model="gpt-4o-mini")
        assert cfg.model == "gpt-4o-mini"

    def test_custom_review_tools_frozenset(self) -> None:
        cfg = SecurityReviewConfig(review_tools=frozenset(["write", "edit"]))
        assert cfg.review_tools == frozenset(["write", "edit"])

    def test_frozen_prevents_mutation(self) -> None:
        cfg = SecurityReviewConfig()
        with pytest.raises(ValidationError):
            cfg.model = "something"  # type: ignore[misc]


class TestSecurityReviewResult:
    def test_default_risk_level(self) -> None:
        r = SecurityReviewResult(is_safe=True, reason="ok")
        assert r.risk_level == "low"

    def test_frozen(self) -> None:
        r = SecurityReviewResult(is_safe=True, reason="ok")
        with pytest.raises(ValidationError):
            r.is_safe = False  # type: ignore[misc]

    def test_invalid_risk_level_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SecurityReviewResult(is_safe=True, reason="ok", risk_level="invalid")  # type: ignore[arg-type]


# =============================================================================
# 2. _build_review_messages helper
# =============================================================================


class TestBuildReviewMessages:
    def test_without_description(self) -> None:
        tc = _make_tool_call("write", {"file_path": "/a.txt", "content": "hi"})
        msgs = _build_review_messages("system prompt", tc)
        assert len(msgs) == 2
        assert msgs[0].content == "system prompt"
        assert "Tool description" not in str(msgs[1].content)
        assert "/a.txt" in str(msgs[1].content)

    def test_with_description(self) -> None:
        tc = _make_tool_call("write", {"file_path": "/a.txt", "content": "hi"})
        msgs = _build_review_messages(
            "system prompt", tc,
            tool_description="Writes content to a file (create or overwrite).",
        )
        assert "Tool description" in str(msgs[1].content)
        assert "Writes content to a file" in str(msgs[1].content)


# =============================================================================
# 3. Constructor & config resolution
# =============================================================================


class TestConstructor:
    def test_interrupt_on_bool_true_resolves(self) -> None:
        mw = AutoSecurityReviewMiddleware(
            interrupt_on={"write": True},
            model=FakeListChatModel(responses=[]),
        )
        assert "write" in mw._interrupt_on
        cfg = mw._interrupt_on["write"]
        assert cfg["allowed_decisions"] == ["approve", "edit", "reject", "respond"]

    def test_interrupt_on_bool_false_ignored(self) -> None:
        mw = AutoSecurityReviewMiddleware(
            interrupt_on={"write": False},
            model=FakeListChatModel(responses=[]),
        )
        assert "write" not in mw._interrupt_on

    def test_interrupt_on_custom_config(self) -> None:
        mw = AutoSecurityReviewMiddleware(
            interrupt_on={"write": {"allowed_decisions": ["approve", "reject"]}},
            model=FakeListChatModel(responses=[]),
        )
        assert mw._interrupt_on["write"]["allowed_decisions"] == ["approve", "reject"]

    def test_default_review_tools_is_all(self) -> None:
        mw = AutoSecurityReviewMiddleware(
            interrupt_on={"write": True},
            model=FakeListChatModel(responses=[]),
        )
        assert mw._review_tools == "all"

    def test_custom_review_tools_frozenset(self) -> None:
        mw = AutoSecurityReviewMiddleware(
            interrupt_on={"write": True, "edit": True},
            model=FakeListChatModel(responses=[]),
            review_tools=frozenset(["write"]),
        )
        assert mw._review_tools == frozenset(["write"])

    def test_tool_descriptions_stored(self) -> None:
        descs = {"write": "Write a file", "read": "Read a file"}
        mw = AutoSecurityReviewMiddleware(
            interrupt_on={"write": True},
            model=FakeListChatModel(responses=[]),
            tool_descriptions=descs,
        )
        assert mw._tool_descriptions == descs

    def test_tool_descriptions_default_empty(self) -> None:
        mw = AutoSecurityReviewMiddleware(
            interrupt_on={"write": True},
            model=FakeListChatModel(responses=[]),
        )
        assert mw._tool_descriptions == {}


# =============================================================================
# 4. _should_ai_review routing
# =============================================================================


class TestShouldAiReview:
    def test_all_reviews_every_interrupt_on_tool(self) -> None:
        mw = AutoSecurityReviewMiddleware(
            interrupt_on={"write": True, "edit": True},
            model=FakeListChatModel(responses=[]),
        )
        assert mw._should_ai_review("write") is True
        assert mw._should_ai_review("edit") is True

    def test_all_excludes_non_interrupt_on(self) -> None:
        mw = AutoSecurityReviewMiddleware(
            interrupt_on={"write": True},
            model=FakeListChatModel(responses=[]),
        )
        assert mw._should_ai_review("read") is False

    def test_frozenset_only_reviews_specified(self) -> None:
        mw = AutoSecurityReviewMiddleware(
            interrupt_on={"write": True, "edit": True, "delete": True},
            model=FakeListChatModel(responses=[]),
            review_tools=frozenset(["write", "delete"]),
        )
        assert mw._should_ai_review("write") is True
        assert mw._should_ai_review("delete") is True
        assert mw._should_ai_review("edit") is False  # direct HITL


# =============================================================================
# 5. _ai_review — with mocked model
# =============================================================================


class TestAiReview:
    def test_ai_review_safe_path(self, mock_model: MagicMock) -> None:
        """AI returns structured safe result."""
        mw = AutoSecurityReviewMiddleware(
            interrupt_on={"write": True},
            model=mock_model,
        )
        # Patch `with_structured_output` to return a callable that returns SAFE
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = _SAFE_RESULT
        mock_model.with_structured_output.return_value = mock_structured

        result = mw._ai_review(_make_tool_call("write"))
        assert result.is_safe is True
        assert result.risk_level == "low"
        mock_model.with_structured_output.assert_called()

    def test_ai_review_unsafe_path(self, mock_model: MagicMock) -> None:
        """AI returns structured unsafe result."""
        mw = AutoSecurityReviewMiddleware(
            interrupt_on={"write": True},
            model=mock_model,
        )
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = _UNSAFE_RESULT
        mock_model.with_structured_output.return_value = mock_structured

        result = mw._ai_review(_make_tool_call("write"))
        assert result.is_safe is False
        assert "outside" in result.reason

    def test_ai_review_fallback_parses_unsafe_text(self, mock_model: MagicMock) -> None:
        """When structured output fails, raw text containing 'unsafe' is detected."""
        mw = AutoSecurityReviewMiddleware(
            interrupt_on={"write": True},
            model=mock_model,
        )
        # Make structured output fail
        mock_model.with_structured_output.side_effect = ValueError("unsupported")

        # Raw invoke returns text with "unsafe"
        mock_msg = MagicMock()
        mock_msg.content = "This tool call is UNSAFE because it writes to /etc"
        mock_model.invoke.return_value = mock_msg

        result = mw._ai_review(_make_tool_call("write"))
        assert result.is_safe is False
        assert "UNSAFE" in result.reason

    def test_ai_review_fallback_parses_safe_text(self, mock_model: MagicMock) -> None:
        """When structured output fails, raw text without 'unsafe' is deemed safe."""
        mw = AutoSecurityReviewMiddleware(
            interrupt_on={"write": True},
            model=mock_model,
        )
        mock_model.with_structured_output.side_effect = ValueError("unsupported")
        mock_msg = MagicMock()
        mock_msg.content = "This looks completely fine."
        mock_model.invoke.return_value = mock_msg

        result = mw._ai_review(_make_tool_call("write"))
        assert result.is_safe is True

    def test_ai_review_includes_tool_description(self, mock_model: MagicMock) -> None:
        """Verify that tool description is sent to the model."""
        descs = {"write": "Creates or overwrites files on disk."}
        mw = AutoSecurityReviewMiddleware(
            interrupt_on={"write": True},
            model=mock_model,
            tool_descriptions=descs,
        )
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = _SAFE_RESULT
        mock_model.with_structured_output.return_value = mock_structured

        mw._ai_review(_make_tool_call("write"))
        # Confirm the call was made & the arg included description
        call_args = mock_structured.invoke.call_args[0][0]  # messages list
        combined = str([m.content for m in call_args])
        assert "Creates or overwrites files" in combined


# =============================================================================
# 6. _build_action_and_config — human-facing description
# =============================================================================


class TestBuildActionAndConfig:
    def test_no_ai_review(self) -> None:
        mw = AutoSecurityReviewMiddleware(
            interrupt_on={"write": True},
            model=FakeListChatModel(responses=[]),
        )
        tc = _make_tool_call("write")
        cfg = mw._interrupt_on["write"]
        action, review = mw._build_action_and_config(tc, cfg)
        assert action["name"] == "write"
        assert action["args"] == tc["args"]
        assert "AI" not in str(action.get("description", ""))
        assert "/test.txt" in str(action.get("description", ""))

    def test_with_ai_review_result(self) -> None:
        mw = AutoSecurityReviewMiddleware(
            interrupt_on={"write": True},
            model=FakeListChatModel(responses=[]),
        )
        tc = _make_tool_call("write")
        cfg = mw._interrupt_on["write"]
        review = SecurityReviewResult(
            is_safe=False,
            reason="Modifying /etc/hosts is dangerous",
            risk_level="critical",
        )
        action, _review_cfg = mw._build_action_and_config(tc, cfg, ai_review=review)
        desc = str(action.get("description", ""))
        assert "UNSAFE" in desc or "安全审查" in desc  # AI header
        assert "CRITICAL" in desc or "critical" in desc.lower()
        assert "Modifying" in desc or "/etc/hosts" in desc


# =============================================================================
# 7. _process_decision
# =============================================================================


class TestProcessDecision:
    @pytest.fixture
    def mw(self) -> AutoSecurityReviewMiddleware:
        return AutoSecurityReviewMiddleware(
            interrupt_on={"write": True},
            model=FakeListChatModel(responses=[]),
        )

    def test_approve(self, mw: AutoSecurityReviewMiddleware) -> None:
        tc = _make_tool_call("write")
        cfg: Any = mw._interrupt_on["write"]
        revised, msg = mw._process_decision({"type": "approve"}, tc, cfg)
        assert revised == tc
        assert msg is None

    def test_edit(self, mw: AutoSecurityReviewMiddleware) -> None:
        tc = _make_tool_call("write")
        cfg = mw._interrupt_on["write"]
        revised, msg = mw._process_decision(
            {"type": "edit", "edited_action": {"name": "read", "args": {"path": "/x"}}},
            tc, cfg,
        )
        assert revised is not None
        assert revised["name"] == "read"
        assert revised["args"] == {"path": "/x"}
        assert msg is None

    def test_reject_with_message(self, mw: AutoSecurityReviewMiddleware) -> None:
        tc = _make_tool_call("write")
        cfg = mw._interrupt_on["write"]
        revised, msg = mw._process_decision(
            {"type": "reject", "message": "Don't touch that file!"}, tc, cfg,
        )
        assert revised == tc
        assert msg is not None
        assert isinstance(msg, ToolMessage)
        assert "Don't touch that file!" in str(msg.content)
        assert msg.tool_call_id == "call_1"
        assert msg.status == "error"

    def test_reject_default_message(self, mw: AutoSecurityReviewMiddleware) -> None:
        tc = _make_tool_call("write")
        cfg = mw._interrupt_on["write"]
        _revised, msg = mw._process_decision({"type": "reject"}, tc, cfg)
        assert msg is not None
        assert "User rejected" in str(msg.content)

    def test_respond(self, mw: AutoSecurityReviewMiddleware) -> None:
        tc = _make_tool_call("write")
        cfg = mw._interrupt_on["write"]
        _revised, msg = mw._process_decision(
            {"type": "respond", "message": "The file already looks correct."}, tc, cfg,
        )
        assert msg is not None
        assert msg.status == "success"
        assert "already looks correct" in str(msg.content)


# =============================================================================
# 8. after_model — main interception logic (mocked)
# =============================================================================


class TestAfterModelNoOp:
    """Cases where after_model returns None (no interception)."""

    def test_empty_messages(self) -> None:
        mw = AutoSecurityReviewMiddleware(
            interrupt_on={"write": True},
            model=FakeListChatModel(responses=[]),
        )
        assert mw.after_model({"messages": []}, _EMPTY_RUNTIME) is None

    def test_no_aimessage(self) -> None:
        mw = AutoSecurityReviewMiddleware(
            interrupt_on={"write": True},
            model=FakeListChatModel(responses=[]),
        )
        state = {"messages": [HumanMessage(content="hello")]}
        assert mw.after_model(state, _EMPTY_RUNTIME) is None

    def test_aimessage_without_tool_calls(self) -> None:
        mw = AutoSecurityReviewMiddleware(
            interrupt_on={"write": True},
            model=FakeListChatModel(responses=[]),
        )
        state = {"messages": [AIMessage(content="all done")]}
        assert mw.after_model(state, _EMPTY_RUNTIME) is None

    def test_tool_call_not_in_interrupt_on(self) -> None:
        mw = AutoSecurityReviewMiddleware(
            interrupt_on={"write": True},
            model=FakeListChatModel(responses=[]),
        )
        state = {
            "messages": [
                AIMessage(content="", tool_calls=[_make_tool_call("read")]),
            ]
        }
        assert mw.after_model(state, _EMPTY_RUNTIME) is None


class TestAfterModelAiAutoApproved:
    """AI reviews and auto-approves → no human interrupt."""

    def test_single_safe_tool_call(self) -> None:
        mw = AutoSecurityReviewMiddleware(
            interrupt_on={"write": True},
            model=FakeListChatModel(responses=[]),
        )
        # Mock _ai_review to return safe
        mw._ai_review = MagicMock(return_value=_SAFE_RESULT)  # type: ignore[method-assign]

        state = {
            "messages": [
                AIMessage(content="", tool_calls=[_make_tool_call("write")]),
            ]
        }
        result = mw.after_model(state, _EMPTY_RUNTIME)
        # All safe → no interrupt
        assert result is None

    def test_multiple_safe_tool_calls(self) -> None:
        mw = AutoSecurityReviewMiddleware(
            interrupt_on={"write": True, "edit": True},
            model=FakeListChatModel(responses=[]),
        )
        mw._ai_review = MagicMock(return_value=_SAFE_RESULT)  # type: ignore[method-assign]

        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        _make_tool_call("write", call_id="call_1"),
                        _make_tool_call("edit", {"file_path": "/f", "old_str": "a", "new_str": "b"}, call_id="call_2"),
                    ],
                ),
            ]
        }
        result = mw.after_model(state, _EMPTY_RUNTIME)
        assert result is None


class TestAfterModelAiUnsafeEscalatesToHuman:
    """AI flags unsafe → interrupt() called, decisions processed."""

    def test_single_unsafe_approved_by_human(self) -> None:
        mw = AutoSecurityReviewMiddleware(
            interrupt_on={"write": True},
            model=FakeListChatModel(responses=[]),
        )
        mw._ai_review = MagicMock(return_value=_UNSAFE_RESULT)  # type: ignore[method-assign]

        tc = _make_tool_call("write", call_id="call_1")
        state = {
            "messages": [
                AIMessage(content="creating file", tool_calls=[tc]),
            ]
        }

        with patch(
            "mambo_agents.middleware.security_review.interrupt",
            return_value={"decisions": [{"type": "approve"}]},
        ) as mock_interrupt:
            result = mw.after_model(state, _EMPTY_RUNTIME)

        # interrupt was called with HITLRequest containing AI assessment
        mock_interrupt.assert_called_once()
        hitl: Any = mock_interrupt.call_args[0][0]
        assert hitl["action_requests"][0]["name"] == "write"
        # AI review reason should be in the human-facing description
        desc = str(hitl["action_requests"][0].get("description", ""))
        assert "unsafe" in desc.lower() or "UNSAFE" in desc

        # Approved → tool_calls preserved as-is
        assert result is not None
        assert len(result["messages"]) == 1  # only the AIMessage
        assert result["messages"][0].tool_calls == [tc]

    def test_single_unsafe_rejected_by_human(self) -> None:
        mw = AutoSecurityReviewMiddleware(
            interrupt_on={"write": True},
            model=FakeListChatModel(responses=[]),
        )
        mw._ai_review = MagicMock(return_value=_UNSAFE_RESULT)  # type: ignore[method-assign]

        tc = _make_tool_call("write", call_id="call_1")
        state = {
            "messages": [
                AIMessage(content="creating file", tool_calls=[tc]),
            ]
        }

        with patch(
            "mambo_agents.middleware.security_review.interrupt",
            return_value={"decisions": [{"type": "reject", "message": "No, don't write there"}]},
        ):
            result = mw.after_model(state, _EMPTY_RUNTIME)

        assert result is not None
        msgs = result["messages"]
        # AIMessage + synthetic ToolMessage (reject)
        assert len(msgs) == 2
        # AIMessage tool_calls cleared (rejected)
        assert msgs[0].tool_calls == [tc]  # tool call still present
        # ToolMessage with error
        assert isinstance(msgs[1], ToolMessage)
        assert msgs[1].tool_call_id == "call_1"
        assert msgs[1].status == "error"
        assert "don't write there" in str(msgs[1].content).lower()

    def test_mixed_safe_and_unsafe(self) -> None:
        """Two tool calls: one safe (auto-approve), one unsafe → human."""
        mw = AutoSecurityReviewMiddleware(
            interrupt_on={"write": True, "edit": True},
            model=FakeListChatModel(responses=[]),
        )
        # write = safe, edit = unsafe
        def _mock_review(tc: Any) -> SecurityReviewResult:
            if tc["name"] == "write":
                return _SAFE_RESULT
            return _UNSAFE_RESULT
        mw._ai_review = _mock_review  # type: ignore[method-assign]

        tc_write = _make_tool_call("write", {"file_path": "/safe.txt", "content": "ok"}, "call_safe")
        tc_edit = _make_tool_call("edit", {"file_path": "/unsafe.txt", "old_str": "a", "new_str": "b"}, "call_unsafe")
        state = {
            "messages": [
                AIMessage(content="", tool_calls=[tc_write, tc_edit]),
            ]
        }

        with patch(
            "mambo_agents.middleware.security_review.interrupt",
            return_value={"decisions": [{"type": "approve"}]},
        ) as mock_interrupt:
            result = mw.after_model(state, _EMPTY_RUNTIME)

        # Only the unsafe tool should trigger interrupt
        mock_interrupt.assert_called_once()
        hitl: Any = mock_interrupt.call_args[0][0]
        assert len(hitl["action_requests"]) == 1
        assert hitl["action_requests"][0]["name"] == "edit"

        # Both tool calls preserved (safe auto-approved, unsafe human-approved)
        assert result is not None
        revised = result["messages"][0].tool_calls
        assert len(revised) == 2
        names = [t["name"] for t in revised]
        assert "write" in names
        assert "edit" in names


class TestAfterModelDirectHitl:
    """Tools not in review_tools → direct HITL (no AI)."""

    def test_direct_hitl_no_ai_review(self) -> None:
        mw = AutoSecurityReviewMiddleware(
            interrupt_on={"write": True, "edit": True},
            model=FakeListChatModel(responses=[]),
            review_tools=frozenset(["edit"]),  # only edit gets AI review
        )
        # Spy on _ai_review
        mw._ai_review = MagicMock(side_effect=RuntimeError("should not be called"))  # type: ignore[method-assign]

        tc = _make_tool_call("write", call_id="call_direct")
        state = {
            "messages": [
                AIMessage(content="", tool_calls=[tc]),
            ]
        }

        with patch(
            "mambo_agents.middleware.security_review.interrupt",
            return_value={"decisions": [{"type": "approve"}]},
        ):
            result = mw.after_model(state, _EMPTY_RUNTIME)

        # _ai_review was NOT called (write not in review_tools)
        mw._ai_review.assert_not_called()
        # But interrupt WAS called (direct HITL)
        assert result is not None

    def test_direct_hitl_description_has_no_ai_info(self) -> None:
        """Direct HITL description should NOT include AI review header."""
        mw = AutoSecurityReviewMiddleware(
            interrupt_on={"write": True},
            model=FakeListChatModel(responses=[]),
            review_tools=frozenset(),  # empty → nothing gets AI-reviewed
        )
        tc = _make_tool_call("write")
        state = {
            "messages": [
                AIMessage(content="", tool_calls=[tc]),
            ]
        }

        with patch(
            "mambo_agents.middleware.security_review.interrupt",
            return_value={"decisions": [{"type": "approve"}]},
        ) as mock_interrupt:
            mw.after_model(state, _EMPTY_RUNTIME)

        hitl: Any = mock_interrupt.call_args[0][0]
        desc = str(hitl["action_requests"][0].get("description", ""))
        # No AI review header
        assert "AI 安全审查" not in desc
        assert "UNSAFE" not in desc


class TestAfterModelDecisionMismatch:
    def test_decision_count_mismatch_raises(self) -> None:
        mw = AutoSecurityReviewMiddleware(
            interrupt_on={"write": True, "edit": True},
            model=FakeListChatModel(responses=[]),
        )
        mw._ai_review = MagicMock(return_value=_UNSAFE_RESULT)  # type: ignore[method-assign]

        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        _make_tool_call("write", call_id="call_1"),
                        _make_tool_call("edit", {"file_path": "/f", "old_str": "a", "new_str": "b"}, call_id="call_2"),
                    ],
                ),
            ]
        }

        with patch(
            "mambo_agents.middleware.security_review.interrupt",
            return_value={"decisions": [{"type": "approve"}]},  # only 1, expected 2
        ):
            with pytest.raises(ValueError, match="Mismatch"):
                mw.after_model(state, _EMPTY_RUNTIME)


class TestAfterModelPreservesNonInterruptTools:
    """Tool calls not in interrupt_on should pass through unchanged."""

    def test_mixed_intercept_and_pass_through(self) -> None:
        mw = AutoSecurityReviewMiddleware(
            interrupt_on={"write": True},
            model=FakeListChatModel(responses=[]),
        )
        mw._ai_review = MagicMock(return_value=_SAFE_RESULT)  # type: ignore[method-assign]

        tc_write = _make_tool_call("write", call_id="call_w")
        tc_read = _make_tool_call("read", {"file_path": "/f.txt"}, "call_r")
        state = {
            "messages": [
                AIMessage(content="", tool_calls=[tc_write, tc_read]),
            ]
        }

        result = mw.after_model(state, _EMPTY_RUNTIME)
        # write was auto-approved (safe), read was never intercepted → None
        assert result is None  # no human interrupt needed


# =============================================================================
# 9. Integration test — create_mambo_agent + security_review config
# =============================================================================


class TestIntegrationWithCreateAgent:
    """End-to-end: create_mambo_agent with security_review using mocked model.

    Uses FakeListChatModel + StateBackend + MemorySaver so no real LLM
    calls, no real filesystem, no real persistence — fully safe.
    """

    def test_agent_created_with_security_review_config(self) -> None:
        """Smoke: agent creation succeeds with security_review."""
        from langgraph.checkpoint.memory import MemorySaver

        from mambo_agents import create_mambo_agent
        from mambo_agents.backends.state import StateBackend
        from mambo_agents.middleware import SecurityReviewConfig

        agent = create_mambo_agent(
            FakeListChatModel(responses=["done"]),
            backend=StateBackend(),
            interrupt_on={"write": True},
            security_review=SecurityReviewConfig(),
            checkpointer=MemorySaver(),
        )
        assert agent is not None

    def test_agent_with_security_review_selective_tools(self) -> None:
        """Only edit gets AI-reviewed; write is direct HITL."""
        from langgraph.checkpoint.memory import MemorySaver

        from mambo_agents import create_mambo_agent
        from mambo_agents.backends.state import StateBackend
        from mambo_agents.middleware import SecurityReviewConfig

        agent = create_mambo_agent(
            FakeListChatModel(responses=["done"]),
            backend=StateBackend(),
            interrupt_on={"write": True, "edit": True},
            security_review=SecurityReviewConfig(
                model=FakeListChatModel(responses=[]),
                review_tools=frozenset(["edit"]),
            ),
            checkpointer=MemorySaver(),
        )
        assert agent is not None

    def test_agent_with_custom_security_prompt(self) -> None:
        """Custom security review prompt is accepted."""
        from langgraph.checkpoint.memory import MemorySaver

        from mambo_agents import create_mambo_agent
        from mambo_agents.backends.state import StateBackend
        from mambo_agents.middleware import SecurityReviewConfig

        agent = create_mambo_agent(
            FakeListChatModel(responses=["done"]),
            backend=StateBackend(),
            interrupt_on={"write": True},
            security_review=SecurityReviewConfig(
                system_prompt="You are a paranoid security auditor. Flag EVERYTHING.",
            ),
            checkpointer=MemorySaver(),
        )
        assert agent is not None
