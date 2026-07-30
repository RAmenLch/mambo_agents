"""Tests for multimodal support in protocol, backends, and middleware."""

from __future__ import annotations

import base64

import pytest
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolNode

from langgraph.store.memory import InMemoryStore

from mambo_agents.backends.schemas import BackendError, ErrorCode, VirtualPath, ReadResult
from mambo_agents.backends.utils.multimodal import (
    EXTENSION_TO_FILE_TYPE,
    get_file_type,
    get_mime_type,
)
from mambo_agents.backends.store import StoreBackend
from tests.test_store_backend import _simulate_graph
from mambo_agents.middleware.backend_tools import (
    BackendToolsMiddleware,
    _build_evicted_content,
    build_core_tools,
)


# ============================================================================
# Phase 1: Protocol layer — file type classification
# ============================================================================


class TestGetFileType:
    """Tests for ``get_file_type`` in protocol.py."""

    def test_image_png(self):
        assert get_file_type("photo.png") == "image"

    def test_image_jpeg(self):
        assert get_file_type("photo.jpeg") == "image"

    def test_image_jpg(self):
        assert get_file_type("photo.jpg") == "image"

    def test_image_webp(self):
        assert get_file_type("photo.webp") == "image"

    def test_video_mp4(self):
        assert get_file_type("clip.mp4") == "video"

    def test_video_mov(self):
        assert get_file_type("clip.mov") == "video"

    def test_audio_mp3(self):
        assert get_file_type("song.mp3") == "audio"

    def test_audio_wav(self):
        assert get_file_type("sound.wav") == "audio"

    def test_file_pdf(self):
        assert get_file_type("doc.pdf") == "file"

    def test_file_pptx(self):
        assert get_file_type("slides.pptx") == "file"

    def test_text_python(self):
        assert get_file_type("script.py") == "text"

    def test_text_txt(self):
        assert get_file_type("notes.txt") == "text"

    def test_text_unknown_extension(self):
        assert get_file_type("data.xyz123") == "text"

    def test_case_insensitive(self):
        assert get_file_type("photo.PNG") == "image"
        assert get_file_type("clip.MP4") == "video"

    def test_path_with_dirs(self):
        assert get_file_type("/home/user/photo.png") == "image"


class TestGetMimeType:
    """Tests for ``get_mime_type`` in protocol.py."""

    def test_png(self):
        assert get_mime_type("photo.png") == "image/png"

    def test_jpeg(self):
        assert get_mime_type("photo.jpg") == "image/jpeg"

    def test_mp3(self):
        assert get_mime_type("song.mp3") == "audio/mpeg"

    def test_mp4(self):
        mime = get_mime_type("clip.mp4")
        assert mime in ("video/mp4", "application/octet-stream")

    def test_pdf(self):
        assert get_mime_type("doc.pdf") == "application/pdf"

    def test_unknown_fallback(self):
        assert get_mime_type("file.xyz123") == "application/octet-stream"


class TestReadResultMultimodal:
    """Tests for ``ReadResult.is_multimodal`` property."""

    def test_text_file_not_multimodal(self):
        r = ReadResult(content="hello", encoding="utf-8", file_type="text")
        assert r.is_multimodal is False

    def test_image_base64_is_multimodal(self):
        r = ReadResult(
            content="AAAA", encoding="base64", file_type="image", mime_type="image/png"
        )
        assert r.is_multimodal is True

    def test_audio_base64_is_multimodal(self):
        r = ReadResult(
            content="AAAA", encoding="base64", file_type="audio", mime_type="audio/mpeg"
        )
        assert r.is_multimodal is True

    def test_video_base64_is_multimodal(self):
        r = ReadResult(
            content="AAAA", encoding="base64", file_type="video", mime_type="video/mp4"
        )
        assert r.is_multimodal is True

    def test_file_base64_is_multimodal(self):
        r = ReadResult(
            content="AAAA", encoding="base64", file_type="file", mime_type="application/pdf"
        )
        assert r.is_multimodal is True

    def test_text_encoding_base64_not_multimodal(self):
        """A text file that happens to be base64-encoded is not multimodal."""
        r = ReadResult(content="AAAA", encoding="base64", file_type="text")
        assert r.is_multimodal is False

    def test_default_values(self):
        """Default ReadResult has text-compatible defaults."""
        r = ReadResult(content="hello")
        assert r.file_type == "text"
        assert r.mime_type == ""
        assert r.is_multimodal is False

    def test_str_with_error(self):
        r = ReadResult(error=BackendError(code=ErrorCode.NOT_FOUND, message="File not found"))
        assert "File not found" in str(r)

    def test_str_with_content(self):
        r = ReadResult(content="hello", encoding="utf-8")
        assert str(r) == "hello"


# ============================================================================
# Phase 2: Backend layer — StoreBackend multimodal read
# ============================================================================


class TestStoreBackendMultimodal:
    """Tests for StoreBackend.read() with multimodal files."""

    def test_read_image_returns_file_type(self):
        b64 = base64.b64encode(b"\x89PNG\r\n\x1a\n\x00").decode("ascii")
        backend = StoreBackend(store=InMemoryStore(), initial_files={"/photo.png": b64})
        with _simulate_graph(backend):
            result = backend.read(VirtualPath("/photo.png"))
        assert result.error is None
        assert result.encoding == "base64"
        assert result.file_type == "image"
        assert result.mime_type == "image/png"

    def test_read_audio_returns_file_type(self):
        b64 = base64.b64encode(b"\xff\xfb\x00").decode("ascii")
        backend = StoreBackend(store=InMemoryStore(), initial_files={"/song.mp3": b64})
        with _simulate_graph(backend):
            result = backend.read(VirtualPath("/song.mp3"))
        assert result.file_type == "audio"
        assert result.mime_type == "audio/mpeg"

    def test_read_video_returns_file_type(self):
        b64 = base64.b64encode(b"\x00\x00\x00\x08ftyp").decode("ascii")
        backend = StoreBackend(store=InMemoryStore(), initial_files={"/clip.mp4": b64})
        with _simulate_graph(backend):
            result = backend.read(VirtualPath("/clip.mp4"))
        assert result.file_type == "video"

    def test_read_pdf_returns_file_type(self):
        b64 = base64.b64encode(b"%PDF").decode("ascii")
        backend = StoreBackend(store=InMemoryStore(), initial_files={"/doc.pdf": b64})
        with _simulate_graph(backend):
            result = backend.read(VirtualPath("/doc.pdf"))
        assert result.file_type == "file"
        assert result.mime_type == "application/pdf"

    def test_read_text_file_type_unchanged(self):
        backend = StoreBackend(store=InMemoryStore(), initial_files={"/hello.py": "print('hi')"})
        with _simulate_graph(backend):
            result = backend.read(VirtualPath("/hello.py"))
        assert result.file_type == "text"
        assert result.mime_type == ""
        assert result.encoding == "utf-8"
        assert "print" in (result.content or "")

    def test_write_image_auto_encoding(self):
        backend = StoreBackend(store=InMemoryStore())
        b64 = base64.b64encode(b"\x89PNG\r\n\x1a\n\x00").decode("ascii")
        with _simulate_graph(backend):
            r = backend.write(VirtualPath("/new_img.png"), b64)
        assert r.error is not None
        assert "非文本" in str(r.error)

    def test_read_not_found(self):
        backend = StoreBackend(store=InMemoryStore())
        with _simulate_graph(backend):
            result = backend.read(VirtualPath("/missing.png"))
        assert result.error is not None
        assert result.file_type == "text"  # default

    def test_image_no_line_numbers(self):
        """Binary file content should NOT have line number formatting."""
        b64 = base64.b64encode(b"\x89PNG\r\n\x1a\n\x00").decode("ascii")
        backend = StoreBackend(store=InMemoryStore(), initial_files={"/photo.png": b64})
        with _simulate_graph(backend):
            result = backend.read(VirtualPath("/photo.png"))
        assert "\t" not in (result.content or "")


# ============================================================================
# Phase 3: Middleware layer — build_core_tools with multimodal read
# ============================================================================


class TestBuildCoreToolsMultimodal:
    """Tests for build_core_tools producing multimodal ToolMessage for read."""

    def test_read_image_returns_tool_message(self):
        b64 = base64.b64encode(b"\x89PNG\r\n\x1a\n\x00").decode("ascii")
        backend = StoreBackend(store=InMemoryStore(), initial_files={"/photo.png": b64})
        tools = build_core_tools(backend)
        read_tool = next(t for t in tools if t.name == "read")

        with _simulate_graph(backend):
            result = read_tool.invoke({"file_path": "/photo.png"})

        assert isinstance(result, ToolMessage)
        assert result.content_blocks is not None
        blocks = result.content_blocks
        assert len(blocks) == 1
        assert blocks[0]["type"] == "image"
        assert blocks[0]["base64"] == b64
        assert blocks[0]["mime_type"] == "image/png"

    def test_read_text_returns_string(self):
        backend = StoreBackend(store=InMemoryStore(), initial_files={"/hello.py": "print('hi')"})
        tools = build_core_tools(backend)
        read_tool = next(t for t in tools if t.name == "read")

        with _simulate_graph(backend):
            result = read_tool.invoke({"file_path": "/hello.py"})

        assert isinstance(result, str)
        assert "print" in result

    def test_read_audio_returns_tool_message(self):
        b64 = base64.b64encode(b"\xff\xfb\x00").decode("ascii")
        backend = StoreBackend(store=InMemoryStore(), initial_files={"/song.mp3": b64})
        tools = build_core_tools(backend)
        read_tool = next(t for t in tools if t.name == "read")

        with _simulate_graph(backend):
            result = read_tool.invoke({"file_path": "/song.mp3"})

        assert isinstance(result, ToolMessage)
        blocks = result.content_blocks
        assert blocks[0]["type"] == "audio"

    def test_read_video_returns_tool_message(self):
        b64 = base64.b64encode(b"\x00\x00\x00\x08ftyp").decode("ascii")
        backend = StoreBackend(store=InMemoryStore(), initial_files={"/clip.mp4": b64})
        tools = build_core_tools(backend)
        read_tool = next(t for t in tools if t.name == "read")

        with _simulate_graph(backend):
            result = read_tool.invoke({"file_path": "/clip.mp4"})

        assert isinstance(result, ToolMessage)
        blocks = result.content_blocks
        assert blocks[0]["type"] == "video"

    def test_read_pdf_returns_tool_message(self):
        b64 = base64.b64encode(b"%PDF").decode("ascii")
        backend = StoreBackend(store=InMemoryStore(), initial_files={"/doc.pdf": b64})
        tools = build_core_tools(backend)
        read_tool = next(t for t in tools if t.name == "read")

        with _simulate_graph(backend):
            result = read_tool.invoke({"file_path": "/doc.pdf"})

        assert isinstance(result, ToolMessage)
        blocks = result.content_blocks
        assert blocks[0]["type"] == "file"

    def test_read_error_returns_string(self):
        backend = StoreBackend(store=InMemoryStore())
        tools = build_core_tools(backend)
        read_tool = next(t for t in tools if t.name == "read")

        with _simulate_graph(backend):
            result = read_tool.invoke({"file_path": "/missing.png"})

        assert isinstance(result, str)
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_async_read_image_returns_tool_message(self):
        b64 = base64.b64encode(b"\x89PNG\r\n\x1a\n\x00").decode("ascii")
        backend = StoreBackend(store=InMemoryStore(), initial_files={"/photo.png": b64})
        tools = build_core_tools(backend)
        read_tool = next(t for t in tools if t.name == "read")

        with _simulate_graph(backend):
            result = await read_tool.ainvoke({"file_path": "/photo.png"})

        assert isinstance(result, ToolMessage)
        blocks = result.content_blocks
        assert blocks[0]["type"] == "image"

    @pytest.mark.asyncio
    async def test_async_read_text_returns_string(self):
        backend = StoreBackend(store=InMemoryStore(), initial_files={"/hello.py": "print('hi')"})
        tools = build_core_tools(backend)
        read_tool = next(t for t in tools if t.name == "read")

        with _simulate_graph(backend):
            result = await read_tool.ainvoke({"file_path": "/hello.py"})

        assert isinstance(result, str)
        assert "print" in result


# ============================================================================
# Phase 4: Eviction — multimodal block preservation
# ============================================================================


class TestBuildEvictedContent:
    """Tests for ``_build_evicted_content`` helper."""

    def test_str_content_returns_replacement(self):
        msg = ToolMessage(content="original text", tool_call_id="c1", name="tree")
        result = _build_evicted_content(msg, "replaced")
        assert result == "replaced"

    def test_list_text_only_returns_replacement(self):
        msg = ToolMessage(
            content=[{"type": "text", "text": "original"}],
            tool_call_id="c1",
            name="tree",
        )
        result = _build_evicted_content(msg, "replaced")
        assert result == "replaced"

    def test_list_mixed_preserves_media_blocks(self):
        msg = ToolMessage(
            content=[
                {"type": "text", "text": "original"},
                {"type": "image", "base64": "AAAA", "mime_type": "image/png"},
            ],
            tool_call_id="c1",
            name="tree",
        )
        result = _build_evicted_content(msg, "replaced")
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == {"type": "text", "text": "replaced"}
        assert result[1] == {"type": "image", "base64": "AAAA", "mime_type": "image/png"}

    def test_list_media_only_keeps_media(self):
        msg = ToolMessage(
            content=[
                {"type": "image", "base64": "AAAA", "mime_type": "image/png"},
            ],
            tool_call_id="c1",
            name="tree",
        )
        result = _build_evicted_content(msg, "replaced")
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == {"type": "text", "text": "replaced"}
        assert result[1] == {"type": "image", "base64": "AAAA", "mime_type": "image/png"}

    def test_list_multiple_media_blocks(self):
        msg = ToolMessage(
            content=[
                {"type": "text", "text": "original"},
                {"type": "image", "base64": "img1", "mime_type": "image/png"},
                {"type": "audio", "base64": "aud1", "mime_type": "audio/mpeg"},
            ],
            tool_call_id="c1",
            name="tree",
        )
        result = _build_evicted_content(msg, "replaced")
        assert isinstance(result, list)
        assert len(result) == 3
        assert result[0]["type"] == "text"
        assert result[1]["type"] == "image"
        assert result[2]["type"] == "audio"


class TestEvictionMultimodalPreservation:
    """Integration tests: eviction preserves multimodal blocks."""

    def test_eviction_preserves_image_block(self):
        """Evicting a ToolMessage with text + image keeps the image."""
        mw = BackendToolsMiddleware(
            backend=StoreBackend(store=InMemoryStore()),
            tool_token_limit_before_evict=10,
        )
        huge_text = "x" * 200  # ~50 tokens, above 10
        msg = ToolMessage(
            content=[
                {"type": "text", "text": huge_text},
                {"type": "image", "base64": "AAAA", "mime_type": "image/png"},
            ],
            tool_call_id="call_mm_01",
            name="tree",
        )
        from mambo_agents.middleware.backend_tools import _sanitize_tool_call_id
        sane_id = _sanitize_tool_call_id("call_mm_01")

        with _simulate_graph(mw.backend):
            result = mw._evict(msg, sane_id)

        # Content should be a list with text replacement + image block
        assert isinstance(result.content, list)
        text_block = result.content[0]
        assert text_block["type"] == "text"
        assert "Tool result too large" in text_block["text"]
        image_block = result.content[1]
        assert image_block["type"] == "image"
        assert image_block["base64"] == "AAAA"

    @pytest.mark.asyncio
    async def test_async_eviction_preserves_image_block(self):
        """Async path: eviction preserves multimodal blocks."""
        mw = BackendToolsMiddleware(
            backend=StoreBackend(store=InMemoryStore()),
            tool_token_limit_before_evict=10,
        )
        huge_text = "x" * 200
        msg = ToolMessage(
            content=[
                {"type": "text", "text": huge_text},
                {"type": "audio", "base64": "BBB=", "mime_type": "audio/mpeg"},
            ],
            tool_call_id="call_mm_async",
            name="tree",
        )
        from mambo_agents.middleware.backend_tools import _sanitize_tool_call_id
        sane_id = _sanitize_tool_call_id("call_mm_async")

        with _simulate_graph(mw.backend):
            result = await mw._aevict(msg, sane_id)

        assert isinstance(result.content, list)
        audio_block = next(b for b in result.content if b["type"] == "audio")
        assert audio_block["base64"] == "BBB="


# ============================================================================
# Phase 5: ToolRuntime injection for multimodal read (regression coverage)
# ============================================================================


class TestReadToolRuntimeInjection:
    """Verify ``runtime`` is correctly recognized as an injected arg.

    Without the fix (removing ``from __future__ import annotations``),
    ``StructuredTool._injected_args_keys`` would fail to detect ``runtime``
    because PEP 563 turns the type annotation into a string that
    ``_is_injected_arg_type`` cannot interpret.  This caused ``tool_call_id``
    to be empty in multimodal ToolMessage results.
    """

    def test_injected_args_keys_includes_runtime(self):
        """``_injected_args_keys`` must contain ``"runtime"`` for the read tool."""
        backend = StoreBackend(store=InMemoryStore())
        tools = build_core_tools(backend)
        read_tool = next(t for t in tools if t.name == "read")

        injected_keys = read_tool._injected_args_keys
        assert "runtime" in injected_keys, (
            "read tool's _injected_args_keys must contain 'runtime'; "
            "otherwise ToolNode-injected ToolRuntime will be dropped by "
            "_parse_input() and sync_read receives runtime=None, "
            "causing tool_call_id to be empty."
        )

    def test_toolnode_call_multimodal_gives_valid_tool_call_id(self):
        """Multimodal read via ToolNode produces ToolMessage with correct tool_call_id."""
        b64 = base64.b64encode(b"\x89PNG\r\n\x1a\n\x00").decode("ascii")
        backend = StoreBackend(store=InMemoryStore(), initial_files={"/photo.png": b64})
        tools = build_core_tools(backend)
        read_tool = next(t for t in tools if t.name == "read")

        # Simulate a ToolNode flow: the model calls the read tool with a tool_call
        tool_call = {
            "name": "read",
            "args": {"file_path": "/photo.png"},
            "id": "call_multimodal_001",
            "type": "tool_call",
        }
        ai_msg = AIMessage(content="read the photo", tool_calls=[tool_call])

        tool_node = ToolNode(tools=tools)
        with _simulate_graph(backend):
            result = tool_node._func(
                {"messages": [HumanMessage("read photo"), ai_msg]},
                config={},
                runtime=_FakeRuntime(),
            )

        # Extract the single ToolMessage from the result
        assert isinstance(result, dict)
        msgs = result.get("messages", [])
        assert len(msgs) == 1
        tm = msgs[0]
        assert isinstance(tm, ToolMessage)
        assert tm.tool_call_id == "call_multimodal_001", (
            "ToolMessage.tool_call_id should match the original tool call id, "
            "not be empty. If empty, the runtime injection was lost."
        )
        # Multimodal content blocks should be present
        assert tm.content_blocks is not None

    @pytest.mark.asyncio
    async def test_toolnode_async_call_multimodal_tool_call_id(self):
        """Async multimodal read via ToolNode produces correct tool_call_id."""
        b64 = base64.b64encode(b"\x89PNG\r\n\x1a\n\x00").decode("ascii")
        backend = StoreBackend(store=InMemoryStore(), initial_files={"/photo.png": b64})
        tools = build_core_tools(backend)
        read_tool = next(t for t in tools if t.name == "read")

        tool_call = {
            "name": "read",
            "args": {"file_path": "/photo.png"},
            "id": "call_async_multimodal_002",
            "type": "tool_call",
        }
        ai_msg = AIMessage(content="read the photo", tool_calls=[tool_call])

        tool_node = ToolNode(tools=tools)
        with _simulate_graph(backend):
            result = await tool_node._afunc(
                {"messages": [HumanMessage("read photo"), ai_msg]},
                config={},
                runtime=_FakeRuntime(),
            )

        assert isinstance(result, dict)
        msgs = result.get("messages", [])
        assert len(msgs) == 1
        tm = msgs[0]
        assert isinstance(tm, ToolMessage)
        assert tm.tool_call_id == "call_async_multimodal_002"

    def test_direct_invoke_with_runtime_injection(self):
        """Direct invoke of read tool with runtime arg produces correct tool_call_id.

        This simulates what ToolNode._inject_tool_args does: it merges the
        ToolRuntime instance into the tool call args before calling tool.invoke().
        """
        b64 = base64.b64encode(b"\x89PNG\r\n\x1a\n\x00").decode("ascii")
        backend = StoreBackend(store=InMemoryStore(), initial_files={"/photo.png": b64})
        tools = build_core_tools(backend)
        read_tool = next(t for t in tools if t.name == "read")

        # Simulate ToolNode's injected call format: args + runtime injected
        fake_runtime = ToolRuntime(
            state={},
            context=None,
            config={},
            stream_writer=None,
            tool_call_id="call_injected_003",
            store=None,
            tools=list(tools),
        )
        call_args = {
            "name": "read",
            "args": {"file_path": "/photo.png", "runtime": fake_runtime},
            "id": "call_injected_003",
            "type": "tool_call",
        }

        with _simulate_graph(backend):
            result = read_tool.invoke(call_args)

        assert isinstance(result, ToolMessage)
        assert result.tool_call_id == "call_injected_003", (
            f"Expected tool_call_id='call_injected_003', got {result.tool_call_id!r}. "
            "The runtime arg may have been dropped by _parse_input() due to "
            "_injected_args_keys not recognizing it."
        )


class _FakeRuntime:
    """Minimal fake of langgraph Runtime for ToolNode execution without a real graph."""
    context = None
    store = None
    stream_writer = None
    execution_info = None
    server_info = None
