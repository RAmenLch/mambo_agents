"""Tests for multimodal support in protocol, backends, and middleware."""

from __future__ import annotations

import base64

import pytest
from langchain_core.messages import ToolMessage

from mambo_agents.backends.protocol import (
    ReadResult,
    _EXTENSION_TO_FILE_TYPE,
    _get_file_type,
    _get_mime_type,
)
from mambo_agents.backends.state import StateBackend
from mambo_agents.middleware.backend_tools import (
    BackendToolsMiddleware,
    _build_evicted_content,
    build_core_tools,
)


# ============================================================================
# Phase 1: Protocol layer — file type classification
# ============================================================================


class TestGetFileType:
    """Tests for ``_get_file_type`` in protocol.py."""

    def test_image_png(self):
        assert _get_file_type("photo.png") == "image"

    def test_image_jpeg(self):
        assert _get_file_type("photo.jpeg") == "image"

    def test_image_jpg(self):
        assert _get_file_type("photo.jpg") == "image"

    def test_image_webp(self):
        assert _get_file_type("photo.webp") == "image"

    def test_video_mp4(self):
        assert _get_file_type("clip.mp4") == "video"

    def test_video_mov(self):
        assert _get_file_type("clip.mov") == "video"

    def test_audio_mp3(self):
        assert _get_file_type("song.mp3") == "audio"

    def test_audio_wav(self):
        assert _get_file_type("sound.wav") == "audio"

    def test_file_pdf(self):
        assert _get_file_type("doc.pdf") == "file"

    def test_file_pptx(self):
        assert _get_file_type("slides.pptx") == "file"

    def test_text_python(self):
        assert _get_file_type("script.py") == "text"

    def test_text_txt(self):
        assert _get_file_type("notes.txt") == "text"

    def test_text_unknown_extension(self):
        assert _get_file_type("data.xyz123") == "text"

    def test_case_insensitive(self):
        assert _get_file_type("photo.PNG") == "image"
        assert _get_file_type("clip.MP4") == "video"

    def test_path_with_dirs(self):
        assert _get_file_type("/home/user/photo.png") == "image"


class TestGetMimeType:
    """Tests for ``_get_mime_type`` in protocol.py."""

    def test_png(self):
        assert _get_mime_type("photo.png") == "image/png"

    def test_jpeg(self):
        assert _get_mime_type("photo.jpg") == "image/jpeg"

    def test_mp3(self):
        assert _get_mime_type("song.mp3") == "audio/mpeg"

    def test_mp4(self):
        mime = _get_mime_type("clip.mp4")
        assert mime in ("video/mp4", "application/octet-stream")

    def test_pdf(self):
        assert _get_mime_type("doc.pdf") == "application/pdf"

    def test_unknown_fallback(self):
        assert _get_mime_type("file.xyz123") == "application/octet-stream"


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
        r = ReadResult(error="File not found")
        assert str(r) == "Error: File not found"

    def test_str_with_content(self):
        r = ReadResult(content="hello", encoding="utf-8")
        assert str(r) == "hello"


# ============================================================================
# Phase 2: Backend layer — StateBackend multimodal read
# ============================================================================


class TestStateBackendMultimodal:
    """Tests for StateBackend.read() with multimodal files."""

    def test_read_image_returns_file_type(self):
        b64 = base64.b64encode(b"\x89PNG\r\n").decode("ascii")
        backend = StateBackend(initial_files={"/photo.png": b64})
        result = backend.read("/photo.png")
        assert result.error is None
        assert result.encoding == "base64"
        assert result.file_type == "image"
        assert result.mime_type == "image/png"

    def test_read_audio_returns_file_type(self):
        b64 = base64.b64encode(b"fake_audio").decode("ascii")
        backend = StateBackend(initial_files={"/song.mp3": b64})
        result = backend.read("/song.mp3")
        assert result.file_type == "audio"
        assert result.mime_type == "audio/mpeg"

    def test_read_video_returns_file_type(self):
        b64 = base64.b64encode(b"fake_video").decode("ascii")
        backend = StateBackend(initial_files={"/clip.mp4": b64})
        result = backend.read("/clip.mp4")
        assert result.file_type == "video"

    def test_read_pdf_returns_file_type(self):
        b64 = base64.b64encode(b"fake_pdf").decode("ascii")
        backend = StateBackend(initial_files={"/doc.pdf": b64})
        result = backend.read("/doc.pdf")
        assert result.file_type == "file"
        assert result.mime_type == "application/pdf"

    def test_read_text_file_type_unchanged(self):
        backend = StateBackend(initial_files={"/hello.py": "print('hi')"})
        result = backend.read("/hello.py")
        assert result.file_type == "text"
        assert result.mime_type == ""
        assert result.encoding == "utf-8"
        assert "print" in (result.content or "")

    def test_write_image_auto_encoding(self):
        backend = StateBackend()
        b64 = base64.b64encode(b"img_data").decode("ascii")
        backend.write("/new_img.png", b64)
        result = backend.read("/new_img.png")
        assert result.encoding == "base64"
        assert result.file_type == "image"

    def test_read_not_found(self):
        backend = StateBackend()
        result = backend.read("/missing.png")
        assert result.error is not None
        assert result.file_type == "text"  # default

    def test_image_no_line_numbers(self):
        """Binary file content should NOT have line number formatting."""
        b64 = base64.b64encode(b"\x89PNG\r\n").decode("ascii")
        backend = StateBackend(initial_files={"/photo.png": b64})
        result = backend.read("/photo.png")
        assert "\t" not in (result.content or "")


# ============================================================================
# Phase 3: Middleware layer — build_core_tools with multimodal read
# ============================================================================


class TestBuildCoreToolsMultimodal:
    """Tests for build_core_tools producing multimodal ToolMessage for read."""

    def test_read_image_returns_tool_message(self):
        b64 = base64.b64encode(b"\x89PNG\r\n").decode("ascii")
        backend = StateBackend(initial_files={"/photo.png": b64})
        tools = build_core_tools(backend)
        read_tool = next(t for t in tools if t.name == "read")

        result = read_tool.invoke({"file_path": "/photo.png"})

        assert isinstance(result, ToolMessage)
        assert result.content_blocks is not None
        blocks = result.content_blocks
        assert len(blocks) == 1
        assert blocks[0]["type"] == "image"
        assert blocks[0]["base64"] == b64
        assert blocks[0]["mime_type"] == "image/png"

    def test_read_text_returns_string(self):
        backend = StateBackend(initial_files={"/hello.py": "print('hi')"})
        tools = build_core_tools(backend)
        read_tool = next(t for t in tools if t.name == "read")

        result = read_tool.invoke({"file_path": "/hello.py"})

        assert isinstance(result, str)
        assert "print" in result

    def test_read_audio_returns_tool_message(self):
        b64 = base64.b64encode(b"fake_audio").decode("ascii")
        backend = StateBackend(initial_files={"/song.mp3": b64})
        tools = build_core_tools(backend)
        read_tool = next(t for t in tools if t.name == "read")

        result = read_tool.invoke({"file_path": "/song.mp3"})

        assert isinstance(result, ToolMessage)
        blocks = result.content_blocks
        assert blocks[0]["type"] == "audio"

    def test_read_video_returns_tool_message(self):
        b64 = base64.b64encode(b"fake_video").decode("ascii")
        backend = StateBackend(initial_files={"/clip.mp4": b64})
        tools = build_core_tools(backend)
        read_tool = next(t for t in tools if t.name == "read")

        result = read_tool.invoke({"file_path": "/clip.mp4"})

        assert isinstance(result, ToolMessage)
        blocks = result.content_blocks
        assert blocks[0]["type"] == "video"

    def test_read_pdf_returns_tool_message(self):
        b64 = base64.b64encode(b"fake_pdf").decode("ascii")
        backend = StateBackend(initial_files={"/doc.pdf": b64})
        tools = build_core_tools(backend)
        read_tool = next(t for t in tools if t.name == "read")

        result = read_tool.invoke({"file_path": "/doc.pdf"})

        assert isinstance(result, ToolMessage)
        blocks = result.content_blocks
        assert blocks[0]["type"] == "file"

    def test_read_error_returns_string(self):
        backend = StateBackend()
        tools = build_core_tools(backend)
        read_tool = next(t for t in tools if t.name == "read")

        result = read_tool.invoke({"file_path": "/missing.png"})

        assert isinstance(result, str)
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_async_read_image_returns_tool_message(self):
        b64 = base64.b64encode(b"\x89PNG\r\n").decode("ascii")
        backend = StateBackend(initial_files={"/photo.png": b64})
        tools = build_core_tools(backend)
        read_tool = next(t for t in tools if t.name == "read")

        result = await read_tool.ainvoke({"file_path": "/photo.png"})

        assert isinstance(result, ToolMessage)
        blocks = result.content_blocks
        assert blocks[0]["type"] == "image"

    @pytest.mark.asyncio
    async def test_async_read_text_returns_string(self):
        backend = StateBackend(initial_files={"/hello.py": "print('hi')"})
        tools = build_core_tools(backend)
        read_tool = next(t for t in tools if t.name == "read")

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
            backend=StateBackend(),
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
            backend=StateBackend(),
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

        result = await mw._aevict(msg, sane_id)

        assert isinstance(result.content, list)
        audio_block = next(b for b in result.content if b["type"] == "audio")
        assert audio_block["base64"] == "BBB="
