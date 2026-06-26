"""Tests for MamboSummarizationMiddleware and summarization support."""

import os
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)

from mambo_agents import (
    MamboSummarizationMiddleware,
    SummarizationConfig,
    create_mambo_agent,
)
from mambo_agents.backends.state import StateBackend
from mambo_agents.backends.schemas import VirtualPath
from tests.test_state_backend import _simulate_graph
from mambo_agents.middleware.summarization import (
    DEFAULT_MAMBO_CHAINED_SUMMARY_PROMPT,
    DEFAULT_MAMBO_SUMMARY_PROMPT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_summary_model(summary_text: str = "Mock summary: plan stuff happened."):
    from unittest.mock import AsyncMock

    mock = MagicMock()
    # Sync invoke: response with .content attribute
    mock_response = MagicMock()
    mock_response.content = summary_text
    mock.invoke.return_value = mock_response
    # Async invoke: must be AsyncMock
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


def _make_mock_backend():
    """Create a MagicMock backend for offload testing."""
    from unittest.mock import AsyncMock

    mock = MagicMock()
    mock.download_files.return_value = []
    mock.write.return_value = MagicMock(error=None)
    mock.edit.return_value = MagicMock(error=None)
    mock.adownload_files = AsyncMock(return_value=[])
    mock.awrite = AsyncMock(return_value=MagicMock(error=None))
    mock.aedit = AsyncMock(return_value=MagicMock(error=None))
    return mock


# ---------------------------------------------------------------------------
# Unit tests – is_summary_message / filter_summary_messages
# ---------------------------------------------------------------------------


class TestIsSummaryMessage:
    """Tests for ``MamboSummarizationMiddleware._is_summary_message``."""

    def test_summary_human_message(self):
        msg = HumanMessage(
            content="Summary...",
            additional_kwargs={"lc_source": "summarization"},
        )
        assert MamboSummarizationMiddleware._is_summary_message(msg) is True

    def test_regular_human_message(self):
        msg = HumanMessage(content="Hello")
        assert MamboSummarizationMiddleware._is_summary_message(msg) is False

    def test_human_no_additional_kwargs(self):
        msg = HumanMessage(content="Hello")
        assert MamboSummarizationMiddleware._is_summary_message(msg) is False

    def test_ai_message(self):
        msg = AIMessage(content="AI response")
        assert MamboSummarizationMiddleware._is_summary_message(msg) is False

    def test_tool_message(self):
        msg = ToolMessage(content="tool result", tool_call_id="c1")
        assert MamboSummarizationMiddleware._is_summary_message(msg) is False


class TestFilterSummaryMessages:
    """Tests for ``MamboSummarizationMiddleware._filter_summary_messages``."""

    def test_filters_summary_messages(self):
        messages = [
            HumanMessage(
                content="Previous summary",
                additional_kwargs={"lc_source": "summarization"},
            ),
            HumanMessage(content="Real user message"),
            AIMessage(content="AI response"),
        ]
        model = _make_mock_summary_model()
        mw = MamboSummarizationMiddleware(model=model)
        result = mw._filter_summary_messages(messages)
        assert len(result) == 2
        assert result[0].content == "Real user message"
        assert result[1].content == "AI response"

    def test_no_summary_messages(self):
        messages = [
            HumanMessage(content="User 1"),
            AIMessage(content="AI 1"),
        ]
        model = _make_mock_summary_model()
        mw = MamboSummarizationMiddleware(model=model)
        result = mw._filter_summary_messages(messages)
        assert result == messages

    def test_empty_list(self):
        model = _make_mock_summary_model()
        mw = MamboSummarizationMiddleware(model=model)
        result = mw._filter_summary_messages([])
        assert result == []


# ---------------------------------------------------------------------------
# Unit tests – _build_new_messages_with_path
# ---------------------------------------------------------------------------


class TestBuildNewMessagesWithPath:
    """Tests for ``MamboSummarizationMiddleware._build_new_messages_with_path``."""

    def test_with_file_path(self):
        model = _make_mock_summary_model()
        mw = MamboSummarizationMiddleware(model=model)
        result = mw._build_new_messages_with_path("summary text", "/conv_history/t1.md")
        assert len(result) == 1
        msg = result[0]
        assert isinstance(msg, HumanMessage)
        assert msg.additional_kwargs["lc_source"] == "summarization"
        assert "/conv_history/t1.md" in msg.content
        assert "<summary>" in msg.content
        assert "summary text" in msg.content
        assert "conversation that has been summarized" in msg.content

    def test_without_file_path(self):
        model = _make_mock_summary_model()
        mw = MamboSummarizationMiddleware(model=model)
        result = mw._build_new_messages_with_path("summary only", None)
        assert len(result) == 1
        msg = result[0]
        assert isinstance(msg, HumanMessage)
        assert msg.additional_kwargs["lc_source"] == "summarization"
        assert "Here is a summary of the conversation to date" in msg.content
        assert "summary only" in msg.content
        assert "saved to" not in msg.content


# ---------------------------------------------------------------------------
# Unit tests – _get_thread_id
# ---------------------------------------------------------------------------


class TestGetThreadId:
    """Tests for ``MamboSummarizationMiddleware._get_thread_id``."""

    def test_from_config_thread_id(self):
        model = _make_mock_summary_model()
        mw = MamboSummarizationMiddleware(model=model)
        runtime = MagicMock()
        runtime.config = {"configurable": {"thread_id": "my-thread-123"}}
        tid = mw._get_thread_id(runtime)
        assert tid == "my-thread-123"

    def test_no_config_uses_session_fallback(self):
        model = _make_mock_summary_model()
        mw = MamboSummarizationMiddleware(model=model)
        runtime = MagicMock()
        runtime.config = {}
        tid = mw._get_thread_id(runtime)
        assert tid.startswith("session_")

    def test_no_configurable_key(self):
        model = _make_mock_summary_model()
        mw = MamboSummarizationMiddleware(model=model)
        runtime = MagicMock()
        runtime.config = {"configurable": {}}
        tid = mw._get_thread_id(runtime)
        assert tid.startswith("session_")

    def test_runtime_without_config_attr(self):
        model = _make_mock_summary_model()
        mw = MamboSummarizationMiddleware(model=model)
        runtime = MagicMock(spec=[])
        tid = mw._get_thread_id(runtime)
        assert tid.startswith("session_")


# ---------------------------------------------------------------------------
# Unit tests – _get_history_path
# ---------------------------------------------------------------------------


class TestGetHistoryPath:
    """Tests for ``MamboSummarizationMiddleware._get_history_path``."""

    def test_path_contains_thread_id(self):
        model = _make_mock_summary_model()
        mw = MamboSummarizationMiddleware(model=model)
        runtime = MagicMock()
        runtime.config = {"configurable": {"thread_id": "t42"}}
        path = mw._get_history_path(runtime)
        assert path == "/.mambo/conversation_history/t42.md"

    def test_path_with_session_fallback(self):
        model = _make_mock_summary_model()
        mw = MamboSummarizationMiddleware(model=model)
        runtime = MagicMock(spec=[])
        path = mw._get_history_path(runtime)
        assert str(path).startswith("/.mambo/conversation_history/session_")
        assert path.endswith(".md")


# ---------------------------------------------------------------------------
# Unit tests – _offload_to_backend
# ---------------------------------------------------------------------------


class TestOffloadToBackend:
    """Tests for ``MamboSummarizationMiddleware._offload_to_backend``."""

    def test_returns_none_when_no_backend(self):
        model = _make_mock_summary_model()
        mw = MamboSummarizationMiddleware(model=model)
        runtime = MagicMock(spec=[])
        result = mw._offload_to_backend(
            [HumanMessage(content="H")], runtime
        )
        assert result is None

    def test_first_offload_writes_new_file(self):
        model = _make_mock_summary_model()
        backend = _make_mock_backend()
        backend.download_files.return_value = []
        mw = MamboSummarizationMiddleware(model=model, backend=backend, offload_to_backend=True)

        runtime = MagicMock()
        runtime.config = {"configurable": {"thread_id": "t1"}}

        messages = [
            HumanMessage(content="User question"),
            AIMessage(content="AI answer"),
        ]
        path = mw._offload_to_backend(messages, runtime)

        assert path == "/.mambo/conversation_history/t1.md"
        backend.download_files.assert_called_once_with(["/.mambo/conversation_history/t1.md"])
        backend.write.assert_called_once()
        call_args = backend.write.call_args[0]
        assert call_args[0] == "/.mambo/conversation_history/t1.md"
        written = call_args[1]
        assert "## Summarized at" in written
        assert "Human: User question" in written
        assert "AI: AI answer" in written
        backend.edit.assert_not_called()

    def test_subsequent_offload_appends_to_existing(self):
        model = _make_mock_summary_model()
        backend = _make_mock_backend()
        existing = """## Summarized at 2026-01-01T00:00:00

User: old stuff
AI: old response

"""
        download_result = MagicMock()
        download_result.content = existing.encode("utf-8")
        download_result.error = None
        backend.download_files.return_value = [download_result]

        mw = MamboSummarizationMiddleware(model=model, backend=backend, offload_to_backend=True)

        runtime = MagicMock()
        runtime.config = {"configurable": {"thread_id": "t1"}}

        messages = [HumanMessage(content="new message")]
        path = mw._offload_to_backend(messages, runtime)

        assert path == "/.mambo/conversation_history/t1.md"
        backend.edit.assert_called_once()
        edit_args = backend.edit.call_args[0]
        assert edit_args[0] == "/.mambo/conversation_history/t1.md"
        assert edit_args[1] == existing
        new_content = edit_args[2]
        assert new_content.startswith(existing)
        assert "new message" in new_content
        backend.write.assert_not_called()

    def test_filters_previous_summary_messages(self):
        """Previous summary messages are NOT offloaded again."""
        model = _make_mock_summary_model()
        backend = _make_mock_backend()
        backend.download_files.return_value = []
        mw = MamboSummarizationMiddleware(model=model, backend=backend, offload_to_backend=True)

        runtime = MagicMock()
        runtime.config = {"configurable": {"thread_id": "t1"}}

        messages = [
            HumanMessage(
                content="Old summary...",
                additional_kwargs={"lc_source": "summarization"},
            ),
            HumanMessage(content="Real user message"),
            AIMessage(content="Real AI response"),
        ]
        path = mw._offload_to_backend(messages, runtime)

        assert path == "/.mambo/conversation_history/t1.md"
        written = backend.write.call_args[0][1]
        assert "Old summary" not in written
        assert "Real user message" in written
        assert "Real AI response" in written

    def test_offload_write_error_returns_none(self):
        model = _make_mock_summary_model()
        backend = _make_mock_backend()
        backend.download_files.return_value = []
        backend.write.return_value = MagicMock(error="disk full")

        mw = MamboSummarizationMiddleware(model=model, backend=backend, offload_to_backend=True)
        runtime = MagicMock(spec=[])
        result = mw._offload_to_backend(
            [HumanMessage(content="H")], runtime
        )
        assert result is None

    def test_offload_exception_returns_none(self):
        model = _make_mock_summary_model()
        backend = _make_mock_backend()
        backend.download_files.return_value = []
        backend.write.side_effect = RuntimeError("connection lost")

        mw = MamboSummarizationMiddleware(model=model, backend=backend, offload_to_backend=True)
        runtime = MagicMock(spec=[])
        result = mw._offload_to_backend(
            [HumanMessage(content="H")], runtime
        )
        assert result is None


# ---------------------------------------------------------------------------
# Unit tests – wrap_model_call with backend offload
# ---------------------------------------------------------------------------


class TestWrapModelCallWithBackend:
    """Test summarization + backend offload in the wrap_model_call flow."""

    def test_offload_called_during_summarization(self):
        """Backend offload is invoked before the summary replaces messages."""
        mock = _make_mock_summary_model("Summary text.")
        backend = _make_mock_backend()
        backend.download_files.return_value = []
        mw = MamboSummarizationMiddleware(
            model=mock,
            backend=backend,
            offload_to_backend=True,
            trigger=("messages", 3),
            keep=("messages", 2),
        )

        messages = [
            HumanMessage(content="User 1"),
            AIMessage(content="AI 1"),
            ToolMessage(content="Tool 1", tool_call_id="c1"),
            HumanMessage(content="User 2"),
            AIMessage(content="AI 2"),
            ToolMessage(content="Tool 2", tool_call_id="c2"),
        ]
        request = _make_request(messages)

        def handler(req):
            return "response"

        result = mw.wrap_model_call(request, handler)
        assert result.model_response == "response"
        assert backend.write.called or backend.edit.called, "offload should be called"

    def test_summary_includes_file_path_when_backend(self):
        """When backend offload succeeds, summary message includes the file path."""
        mock = _make_mock_summary_model("Summary: user asked about files.")
        backend = _make_mock_backend()
        backend.download_files.return_value = []
        mw = MamboSummarizationMiddleware(
            model=mock,
            backend=backend,
            offload_to_backend=True,
            trigger=("messages", 3),
            keep=("messages", 2),
        )

        messages = [
            HumanMessage(content="User 1"),
            AIMessage(content="AI 1"),
            ToolMessage(content="Tool 1", tool_call_id="c1"),
            HumanMessage(content="User 2"),
            AIMessage(content="AI 2"),
            ToolMessage(content="Tool 2", tool_call_id="c2"),
        ]
        request = _make_request(messages)
        received = []

        def handler(req):
            received.append(list(req.messages))
            return "ok"

        mw.wrap_model_call(request, handler)

        modified = received[0]
        summary_msg = modified[0]
        assert isinstance(summary_msg, HumanMessage)
        assert "conversation that has been summarized" in summary_msg.content
        assert "/conversation_history/" in summary_msg.content
        assert "<summary>" in summary_msg.content
        assert "Summary:" in summary_msg.content

    def test_offload_failure_does_not_block_summarization(self):
        """Even when offload fails, summarization still proceeds."""
        mock = _make_mock_summary_model("Fallback summary.")
        backend = _make_mock_backend()
        backend.download_files.return_value = []
        backend.write.side_effect = Exception("BOOM")

        mw = MamboSummarizationMiddleware(
            model=mock,
            backend=backend,
            offload_to_backend=True,
            trigger=("messages", 3),
            keep=("messages", 2),
        )

        messages = [
            HumanMessage(content="User 1"),
            AIMessage(content="AI 1"),
            ToolMessage(content="Tool 1", tool_call_id="c1"),
            HumanMessage(content="User 2"),
            AIMessage(content="AI 2"),
            ToolMessage(content="Tool 2", tool_call_id="c2"),
        ]
        request = _make_request(messages)
        received = []

        with pytest.warns(UserWarning, match="Offloading.*failed"):
            def handler(req):
                received.append(list(req.messages))
                return "ok"

            result = mw.wrap_model_call(request, handler)

        assert result.model_response == "ok"
        modified = received[0]
        assert isinstance(modified[0], HumanMessage)
        assert modified[0].additional_kwargs["lc_source"] == "summarization"
        assert "Here is a summary" in modified[0].content

    def test_no_offload_when_backend_is_none(self):
        """Without backend, no offload and summary uses old format."""
        mock = _make_mock_summary_model("Plain summary.")
        mw = MamboSummarizationMiddleware(
            model=mock,
            trigger=("messages", 3),
            keep=("messages", 2),
        )

        messages = [
            HumanMessage(content="User 1"),
            AIMessage(content="AI 1"),
            ToolMessage(content="Tool 1", tool_call_id="c1"),
            HumanMessage(content="User 2"),
            AIMessage(content="AI 2"),
        ]
        request = _make_request(messages)
        received = []

        def handler(req):
            received.append(list(req.messages))
            return "ok"

        mw.wrap_model_call(request, handler)

        modified = received[0]
        summary_msg = modified[0]
        assert "Here is a summary of the conversation to date" in summary_msg.content
        assert "/conversation_history" not in summary_msg.content


# ---------------------------------------------------------------------------
# Unit tests – async offload
# ---------------------------------------------------------------------------


class TestAsyncOffload:
    """Async variant tests for backend offload."""

    @pytest.mark.asyncio
    async def test_aoffload_returns_none_when_no_backend(self):
        model = _make_mock_summary_model()
        mw = MamboSummarizationMiddleware(model=model)
        runtime = MagicMock(spec=[])
        result = await mw._aoffload_to_backend(
            [HumanMessage(content="H")], runtime
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_aoffload_writes_new_file(self):
        model = _make_mock_summary_model()
        backend = _make_mock_backend()
        backend.adownload_files.return_value = []
        mw = MamboSummarizationMiddleware(model=model, backend=backend, offload_to_backend=True)

        runtime = MagicMock()
        runtime.config = {"configurable": {"thread_id": "t-async"}}

        messages = [HumanMessage(content="async test")]
        path = await mw._aoffload_to_backend(messages, runtime)

        assert path == "/.mambo/conversation_history/t-async.md"
        backend.awrite.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_summarize_with_backend(self):
        """Async summarization triggers offload and includes file path."""
        mock = _make_mock_summary_model("Async summary text.")
        backend = _make_mock_backend()
        backend.adownload_files.return_value = []
        mw = MamboSummarizationMiddleware(
            model=mock,
            backend=backend,
            offload_to_backend=True,
            trigger=("messages", 3),
            keep=("messages", 2),
        )

        messages = [
            HumanMessage(content="User 1"),
            AIMessage(content="AI 1"),
            ToolMessage(content="Tool 1", tool_call_id="c1"),
            HumanMessage(content="User 2"),
            AIMessage(content="AI 2"),
        ]
        request = _make_request(messages)
        received = []

        async def handler(req):
            received.append(list(req.messages))
            return "async_ok"

        result = await mw.awrap_model_call(request, handler)
        assert result.model_response == "async_ok"
        assert len(received) == 1

        modified = received[0]
        summary_msg = modified[0]
        assert isinstance(summary_msg, HumanMessage)
        assert summary_msg.additional_kwargs["lc_source"] == "summarization"
        assert backend.awrite.called, "async offload should be called"


# ---------------------------------------------------------------------------
# Unit tests – DEFAULT_MAMBO_SUMMARY_PROMPT
# ---------------------------------------------------------------------------


class TestDefaultPrompt:
    def test_no_next_steps_section(self):
        """DEFAULT_MAMBO_SUMMARY_PROMPT must NOT contain NEXT STEPS."""
        assert "NEXT STEPS" not in DEFAULT_MAMBO_SUMMARY_PROMPT, (
            "Mambo summary prompt should not include speculative NEXT STEPS"
        )

    def test_has_required_sections(self):
        """Prompt must contain SESSION INTENT, SUMMARY, and ARTIFACTS."""
        assert "SESSION INTENT" in DEFAULT_MAMBO_SUMMARY_PROMPT
        assert "## SUMMARY" in DEFAULT_MAMBO_SUMMARY_PROMPT
        assert "ARTIFACTS" in DEFAULT_MAMBO_SUMMARY_PROMPT

    def test_has_messages_placeholder(self):
        """Prompt must contain {messages} for string formatting."""
        assert "{messages}" in DEFAULT_MAMBO_SUMMARY_PROMPT

    def test_not_empty(self):
        """Prompt should be substantial."""
        assert len(DEFAULT_MAMBO_SUMMARY_PROMPT) > 100

    def test_chained_has_required_sections(self):
        """Chained prompt must contain SESSION INTENT, SUMMARY, ARTIFACTS, and {previous_summaries}."""
        assert "SESSION INTENT" in DEFAULT_MAMBO_CHAINED_SUMMARY_PROMPT
        assert "ARTIFACTS" in DEFAULT_MAMBO_CHAINED_SUMMARY_PROMPT
        assert "{previous_summaries}" in DEFAULT_MAMBO_CHAINED_SUMMARY_PROMPT
        assert "{messages}" in DEFAULT_MAMBO_CHAINED_SUMMARY_PROMPT


# ---------------------------------------------------------------------------
# Unit tests – _adjust_cutoff_for_user_message
# ---------------------------------------------------------------------------


class TestAdjustCutoffForUserMessage:
    """Tests for ``MamboSummarizationMiddleware._adjust_cutoff_for_user_message``."""

    def test_user_already_in_preserve_zone(self):
        """Last user message is in the preserved suffix — cutoff unchanged."""
        messages = [
            HumanMessage(content="User 1"),
            AIMessage(content="AI 1"),
            ToolMessage(content="Tool 1", tool_call_id="c1"),
            HumanMessage(content="User 2"),
            AIMessage(content="AI 2"),
            ToolMessage(content="Tool 2", tool_call_id="c2"),
        ]
        cutoff = 3
        result = MamboSummarizationMiddleware._adjust_cutoff_for_user_message(
            messages, cutoff
        )
        assert result == 3, "cutoff should stay at 3 — last user message is preserved"

    def test_user_in_summarize_zone__cutoff_not_moved(self):
        """Last user message is in the to-summarize zone — cutoff NOT moved.

        The simplified boundary-alignment strategy intentionally leaves
        user messages in the summarize zone.  Their intent is captured
        via ``_LATEST_USER_INTENT_SECTION`` in the summary prompt instead.
        """
        messages = [
            HumanMessage(content="User 1"),
            AIMessage(content="AI 1"),
            ToolMessage(content="Tool 1", tool_call_id="c1"),
            HumanMessage(content="User 2"),
            AIMessage(content="AI 2"),
            ToolMessage(content="Tool 2", tool_call_id="c2"),
        ]
        cutoff = 5
        result = MamboSummarizationMiddleware._adjust_cutoff_for_user_message(
            messages, cutoff
        )
        assert result == 5, (
            f"Expected cutoff=5 (boundary unchanged — intent handled via prompt), got {result}"
        )

    def test_summary_human_message_skipped(self):
        """Previous summary HumanMessage at boundary — skipped, cutoff unchanged.

        The summary message (lc_source="summarization") at the boundary
        is not a genuine user message, so the cutoff stays put.  The
        genuine user question in the summarize zone is intentionally
        left there — its intent is captured via the prompt.
        """
        messages = [
            HumanMessage(
                content="Summary of earlier...",
                additional_kwargs={"lc_source": "summarization"},
            ),
            HumanMessage(content="User question"),
            AIMessage(content="AI answer"),
            ToolMessage(content="Tool result", tool_call_id="c1"),
        ]
        cutoff = 3
        result = MamboSummarizationMiddleware._adjust_cutoff_for_user_message(
            messages, cutoff
        )
        assert result == 3, (
            f"Expected cutoff=3 (boundary unchanged — summary skipped, user intent via prompt), got {result}"
        )

    def test_no_user_message_at_all(self):
        """Only AI and Tool messages — no adjustment needed."""
        messages = [
            AIMessage(content="AI 1"),
            ToolMessage(content="Tool 1", tool_call_id="c1"),
            AIMessage(content="AI 2"),
            ToolMessage(content="Tool 2", tool_call_id="c2"),
        ]
        cutoff = 2
        result = MamboSummarizationMiddleware._adjust_cutoff_for_user_message(
            messages, cutoff
        )
        assert result == 2, "no user messages — cutoff unchanged"

    def test_single_user_as_last_message(self):
        """User is the very last message — already safe, no change."""
        messages = [
            AIMessage(content="AI 1"),
            ToolMessage(content="Tool 1", tool_call_id="c1"),
            HumanMessage(content="User query"),
        ]
        cutoff = 2
        result = MamboSummarizationMiddleware._adjust_cutoff_for_user_message(
            messages, cutoff
        )
        assert result == 2, "last message IS the user — cutoff unchanged"

    def test_multiple_user_messages__boundary_only(self):
        """Multiple user messages — only boundary alignment, not preservation.

        The last user message (U2) is in the summarize zone.  The simplified
        strategy does NOT move the cutoff to preserve it — that responsibility
        is handled by ``_LATEST_USER_INTENT_SECTION`` in the summary prompt.
        """
        messages = [
            HumanMessage(content="User 1"),
            AIMessage(content="AI 1"),
            HumanMessage(content="User 2"),
            AIMessage(content="AI 2"),
            ToolMessage(content="Tool 2", tool_call_id="c2"),
        ]
        cutoff = 4
        result = MamboSummarizationMiddleware._adjust_cutoff_for_user_message(
            messages, cutoff
        )
        assert result == 4, (
            f"Expected cutoff=4 (boundary unchanged — intent via prompt), got {result}"
        )

    def test_user_at_cutoff_boundary__with_user_in_preserved_zone(self):
        """User message at cutoff-1 with another user in preserved zone.

        Regression test: Phase 1 used to stop after finding the user in
        the preserved zone, leaving the user message at the cutoff
        boundary unprotected.
        """
        messages = [
            HumanMessage(content="User 1"),
            AIMessage(content="AI 1"),
            HumanMessage(content="User 2"),   # index 2 – at cutoff boundary!
            AIMessage(content="AI 2"),
            ToolMessage(content="Tool 2", tool_call_id="c2"),
            AIMessage(content="AI 3"),
            HumanMessage(content="User 3"),   # index 6 – in preserved zone
            AIMessage(content="AI 4"),
        ]
        cutoff = 3
        result = MamboSummarizationMiddleware._adjust_cutoff_for_user_message(
            messages, cutoff
        )
        assert result == 2, (
            f"Expected cutoff=2 to preserve User 2 at the boundary, got {result}"
        )


# ---------------------------------------------------------------------------
# Unit tests – MamboSummarizationMiddleware initialization
# ---------------------------------------------------------------------------


class TestMiddlewareInit:
    def test_basic_initialization(self):
        """Middleware initializes with minimum required args."""
        model = _make_mock_summary_model()
        mw = MamboSummarizationMiddleware(model=model)
        assert mw is not None
        assert mw._lc_helper is not None

    def test_initialization_with_trigger(self):
        """Middleware accepts all configurable parameters."""
        model = _make_mock_summary_model()
        mw = MamboSummarizationMiddleware(
            model=model,
            trigger=("messages", 10),
            keep=("messages", 5),
            summary_prompt="Custom: {messages}",
            trim_tokens_to_summarize=2000,
        )
        assert mw is not None

    def test_initialization_with_fraction_trigger(self):
        """Fraction-based trigger is accepted."""
        model = _make_mock_summary_model()
        model.profile = {"max_input_tokens": 128000}
        mw = MamboSummarizationMiddleware(
            model=model,
            trigger=("fraction", 0.85),
            keep=("fraction", 0.10),
        )
        assert mw is not None

    def test_string_model_is_resolved(self):
        """String model name is resolved via init_chat_model."""
        from unittest.mock import patch

        with patch(
            "mambo_agents.middleware.summarization.init_chat_model"
        ) as mock_init:
            mock_model = _make_mock_summary_model()
            mock_init.return_value = mock_model
            mw = MamboSummarizationMiddleware(model="gpt-4o-mini")
            mock_init.assert_called_once_with("gpt-4o-mini")
            assert mw is not None

    def test_defaults_are_applied(self):
        """When optional params omitted, defaults kick in."""
        model = _make_mock_summary_model()
        mw = MamboSummarizationMiddleware(model=model)
        keep_type, keep_value = mw._lc_helper.keep
        assert keep_type == "messages"
        assert keep_value == 20

    def test_custom_token_counter(self):
        """Custom token_counter is respected."""
        model = _make_mock_summary_model()
        custom_counter = lambda msgs: 42  # noqa: E731
        mw = MamboSummarizationMiddleware(
            model=model,
            token_counter=custom_counter,
        )
        assert mw._token_counter is custom_counter


# ---------------------------------------------------------------------------
# Unit tests – SummarizationConfig
# ---------------------------------------------------------------------------


class TestSummarizationConfig:
    def test_all_fields_optional(self):
        """SummarizationConfig has total=False — every field is optional."""
        config: SummarizationConfig = {}
        assert len(config) == 0

    def test_fields_accessible(self):
        """Config fields can be set and read."""
        config: SummarizationConfig = {
            "trigger": ("tokens", 50000),
            "keep": ("messages", 10),
            "summary_prompt": "My prompt: {messages}",
            "trim_tokens_to_summarize": 2000,
        }
        assert config["trigger"] == ("tokens", 50000)
        assert config["keep"] == ("messages", 10)
        assert config["summary_prompt"] == "My prompt: {messages}"
        assert config["trim_tokens_to_summarize"] == 2000
