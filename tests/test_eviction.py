"""Tests for large tool result eviction in BackendToolsMiddleware."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

from mambo_agents.backends.state import StateBackend
from mambo_agents.middleware.backend_tools import (
    BackendToolsMiddleware,
    _build_preview,
    _estimate_tokens,
    _sanitize_tool_call_id,
    _EVICTION_PREFIX,
    _TOOL_RESULT_EVICTED_MSG,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_THRESHOLD = 100  # Small threshold for test purposes


def _make_request(tool_name: str, tool_call_id: str = "call_test_001") -> ToolCallRequest:
    """Create a minimal ToolCallRequest for testing."""
    return ToolCallRequest(
        tool_call={"name": tool_name, "args": {}, "id": tool_call_id},
        tool=None,
        state={"messages": []},
        runtime=MagicMock(),
    )


def _make_middleware(threshold: int | None = _THRESHOLD) -> BackendToolsMiddleware:
    """Create a middleware with a StateBackend and the given threshold."""
    return BackendToolsMiddleware(
        backend=StateBackend(),
        tool_token_limit_before_evict=threshold,
    )


# ---------------------------------------------------------------------------
# Pure function tests
# ---------------------------------------------------------------------------


class TestSanitizeToolCallId:
    def test_alphanumeric_unchanged(self):
        assert _sanitize_tool_call_id("call_abc123") == "call_abc123"

    def test_slashes_replaced(self):
        assert _sanitize_tool_call_id("call/with/slash") == "call_with_slash"

    def test_colons_replaced(self):
        assert _sanitize_tool_call_id("chatcmpl-ABC:123") == "chatcmpl-ABC_123"

    def test_spaces_replaced(self):
        assert _sanitize_tool_call_id("call with space") == "call_with_space"

    def test_dots_and_hyphens_preserved(self):
        assert _sanitize_tool_call_id("call-v1.2.3") == "call-v1.2.3"


class TestEstimateTokens:
    def test_empty_string(self):
        assert _estimate_tokens("") == 0

    def test_short_text(self):
        assert _estimate_tokens("hello") == 1  # 5 // 4

    def test_exact_boundary(self):
        assert _estimate_tokens("1234") == 1  # 4 // 4

    def test_long_text(self):
        assert _estimate_tokens("x" * 400) == 100  # 400 // 4


class TestBuildPreview:
    def test_small_content_returns_numbered(self):
        """Content within 10 lines returns the whole thing with line numbers."""
        text = "line1\nline2\nline3"
        result = _build_preview(text)
        assert "     1|line1" in result
        assert "     2|line2" in result
        assert "     3|line3" in result
        assert "truncated" not in result

    def test_large_content_shows_head_and_tail(self):
        """Content > 10 lines shows head 5 + truncation notice + tail 5."""
        lines = [f"line {i}" for i in range(100)]
        text = "\n".join(lines)
        result = _build_preview(text)

        # Head
        assert "     1|line 0" in result
        assert "     5|line 4" in result
        # Truncation notice
        assert "[90 lines truncated]" in result
        # Tail
        assert "    96|line 95" in result
        assert "   100|line 99" in result

    def test_exactly_ten_lines_returns_all(self):
        """Exactly 10 lines = window size, return all without truncation."""
        text = "\n".join(f"L{i}" for i in range(10))
        result = _build_preview(text)
        assert "truncated" not in result
        assert len(result.splitlines()) == 10


# ---------------------------------------------------------------------------
# Eviction – sync
# ---------------------------------------------------------------------------


class TestEviction:
    def test_small_result_not_evicted(self):
        """A result below the token threshold is returned unchanged."""
        mw = _make_middleware(threshold=200)
        result = ToolMessage(content="short", tool_call_id="call_001", name="tree")
        request = _make_request("tree")

        final = mw._maybe_evict(result, request)
        assert final is result  # Same object, no eviction

    def test_large_result_evicted(self):
        """A result above the threshold is evicted to the filesystem."""
        mw = _make_middleware(threshold=100)
        huge = "x" * 500  # ~125 tokens, above 100
        result = ToolMessage(content=huge, tool_call_id="call_002", name="tree")
        request = _make_request("tree", tool_call_id="call_002")

        final = mw._maybe_evict(result, request)
        assert final is not result
        assert "Tool result too large" in final.content
        assert _EVICTION_PREFIX in final.content

    def test_eviction_writes_to_backend(self):
        """Evicted content is persisted to the backend."""
        mw = _make_middleware(threshold=100)
        huge = "HELLO_EVICTED_CONTENT" * 20
        result = ToolMessage(content=huge, tool_call_id="call_003", name="tree")
        request = _make_request("tree", tool_call_id="call_003")

        mw._maybe_evict(result, request)

        # Read back from backend
        sane_id = _sanitize_tool_call_id("call_003")
        read_result = mw.backend.read(f"{_EVICTION_PREFIX}/{sane_id}")
        assert read_result.error is None
        assert "HELLO_EVICTED_CONTENT" in (read_result.content or "")

    def test_preview_present_in_evicted_message(self):
        """The evicted message includes a head+tail preview."""
        mw = _make_middleware(threshold=50)
        lines = [f"line_{i:03d}" for i in range(60)]
        huge = "\n".join(lines)
        result = ToolMessage(content=huge, tool_call_id="call_004", name="tree")
        request = _make_request("tree", tool_call_id="call_004")

        final = mw._maybe_evict(result, request)

        assert "     1|line_000" in final.content
        assert "TOOL_RESULT_EVICTED" not in final.content  # should not be the raw template
        assert "50 lines truncated" in final.content

    def test_eviction_preserves_tool_call_id(self):
        """The returned ToolMessage retains the original tool_call_id."""
        mw = _make_middleware(threshold=10)
        huge = "x" * 100
        result = ToolMessage(content=huge, tool_call_id="call_007", name="tree")
        request = _make_request("tree", tool_call_id="call_007")

        final = mw._maybe_evict(result, request)
        assert final.tool_call_id == "call_007"

    def test_eviction_preserves_tool_name(self):
        """The returned ToolMessage retains the original tool name."""
        mw = _make_middleware(threshold=10)
        huge = "x" * 100
        result = ToolMessage(content=huge, tool_call_id="call_008", name="execute")
        request = _make_request("execute", tool_call_id="call_008")

        final = mw._maybe_evict(result, request)
        assert final.name == "execute"

    def test_excluded_tool_ls_not_evicted(self):
        """ls tool results are never evicted."""
        mw = _make_middleware(threshold=10)
        huge = "x" * 100
        result = ToolMessage(content=huge, tool_call_id="call_ls", name="ls")
        request = _make_request("ls", tool_call_id="call_ls")

        final = mw._maybe_evict(result, request)
        assert final is result

    def test_excluded_tool_read_not_evicted(self):
        """read tool results are never evicted."""
        mw = _make_middleware(threshold=10)
        huge = "x" * 100
        result = ToolMessage(content=huge, tool_call_id="call_read", name="read")
        request = _make_request("read", tool_call_id="call_read")

        final = mw._maybe_evict(result, request)
        assert final is result

    def test_excluded_tool_write_not_evicted(self):
        """write tool results are never evicted."""
        mw = _make_middleware(threshold=10)
        huge = "x" * 100
        result = ToolMessage(content=huge, tool_call_id="call_write", name="write")
        request = _make_request("write", tool_call_id="call_write")

        final = mw._maybe_evict(result, request)
        assert final is result

    def test_excluded_tool_edit_not_evicted(self):
        """edit tool results are never evicted."""
        mw = _make_middleware(threshold=10)
        huge = "x" * 100
        result = ToolMessage(content=huge, tool_call_id="call_edit", name="edit")
        request = _make_request("edit", tool_call_id="call_edit")

        final = mw._maybe_evict(result, request)
        assert final is result

    def test_excluded_tool_grep_not_evicted(self):
        """grep tool results are never evicted."""
        mw = _make_middleware(threshold=10)
        huge = "x" * 100
        result = ToolMessage(content=huge, tool_call_id="call_grep", name="grep")
        request = _make_request("grep", tool_call_id="call_grep")

        final = mw._maybe_evict(result, request)
        assert final is result

    def test_excluded_tool_glob_not_evicted(self):
        """glob tool results are never evicted."""
        mw = _make_middleware(threshold=10)
        huge = "x" * 100
        result = ToolMessage(content=huge, tool_call_id="call_glob", name="glob")
        request = _make_request("glob", tool_call_id="call_glob")

        final = mw._maybe_evict(result, request)
        assert final is result

    def test_eviction_disabled_by_none(self):
        """When tool_token_limit_before_evict is None, nothing is evicted."""
        mw = _make_middleware(threshold=None)
        huge = "x" * 10_000
        result = ToolMessage(content=huge, tool_call_id="call_009", name="execute")
        request = _make_request("execute", tool_call_id="call_009")

        final = mw._maybe_evict(result, request)
        assert final is result  # Unchanged

    def test_same_tool_call_id_no_conflict(self):
        """Two evictions with the same tool_call_id: the second overwrites.

        Since StateBackend.write fails on existing files, we patch write
        to allow overwrites for this test only.
        """
        mw = _make_middleware(threshold=30)

        # Monkey-patch write to allow overwrites
        _orig_write = mw.backend.write
        _files = mw.backend._files

        def _allow_overwrite_write(file_path: str, content: str):
            # Always overwrite: remove existing entry first
            _files.pop(file_path, None)
            return _orig_write(file_path, content)

        mw.backend.write = _allow_overwrite_write  # type: ignore[method-assign]

        # First eviction
        huge1 = "FIRST_CONTENT" * 40  # ~130 tokens
        result1 = ToolMessage(content=huge1, tool_call_id="call_dup", name="tree")
        req1 = _make_request("tree", tool_call_id="call_dup")
        mw._maybe_evict(result1, req1)

        # Second eviction
        huge2 = "SECOND_CONTENT" * 40  # ~130 tokens
        result2 = ToolMessage(content=huge2, tool_call_id="call_dup", name="tree")
        req2 = _make_request("tree", tool_call_id="call_dup")
        mw._maybe_evict(result2, req2)

        sane_id = _sanitize_tool_call_id("call_dup")
        read = mw.backend.read(f"{_EVICTION_PREFIX}/{sane_id}")
        assert "SECOND_CONTENT" in (read.content or "")
        assert "FIRST_CONTENT" not in (read.content or "")

    def test_write_failure_returns_original(self):
        """When backend.write fails, the original message is returned unchanged."""
        mw = _make_middleware(threshold=10)

        # Use a backend that will fail on any write
        class FailWriteBackend(StateBackend):
            def write(self, file_path: str, content: str):
                from mambo_agents.backends.protocol import WriteResult
                return WriteResult(error="Disk full", path=file_path)

        mw.backend = FailWriteBackend()

        huge = "x" * 100
        result = ToolMessage(content=huge, tool_call_id="call_fail", name="tree")
        request = _make_request("tree", tool_call_id="call_fail")

        final = mw._maybe_evict(result, request)
        assert final is result  # Fallback to original


# ---------------------------------------------------------------------------
# Eviction – async
# ---------------------------------------------------------------------------


class TestEvictionAsync:
    @pytest.mark.asyncio
    async def test_small_result_not_evicted_async(self):
        """Async path: result below threshold returned unchanged."""
        mw = _make_middleware(threshold=200)
        result = ToolMessage(content="short", tool_call_id="call_async_01", name="tree")
        request = _make_request("tree", tool_call_id="call_async_01")

        final = await mw._amaybe_evict(result, request)
        assert final is result

    @pytest.mark.asyncio
    async def test_large_result_evicted_async(self):
        """Async path: result above threshold is evicted."""
        mw = _make_middleware(threshold=100)
        huge = "x" * 500
        result = ToolMessage(content=huge, tool_call_id="call_async_02", name="tree")
        request = _make_request("tree", tool_call_id="call_async_02")

        final = await mw._amaybe_evict(result, request)
        assert final is not result
        assert "Tool result too large" in final.content

    @pytest.mark.asyncio
    async def test_eviction_writes_to_backend_async(self):
        """Async path: evicted content persisted to backend."""
        mw = _make_middleware(threshold=30)
        huge = "ASYNC_CONTENT" * 40  # ~130 tokens
        result = ToolMessage(content=huge, tool_call_id="call_async_03", name="tree")
        request = _make_request("tree", tool_call_id="call_async_03")

        await mw._amaybe_evict(result, request)

        sane_id = _sanitize_tool_call_id("call_async_03")
        read_result = await mw.backend.aread(f"{_EVICTION_PREFIX}/{sane_id}")
        assert read_result.error is None
        assert "ASYNC_CONTENT" in (read_result.content or "")

    @pytest.mark.asyncio
    async def test_disabled_async(self):
        """Async path: eviction disabled returns original."""
        mw = _make_middleware(threshold=None)
        huge = "x" * 10_000
        result = ToolMessage(content=huge, tool_call_id="call_async_04", name="execute")
        request = _make_request("execute", tool_call_id="call_async_04")

        final = await mw._amaybe_evict(result, request)
        assert final is result
