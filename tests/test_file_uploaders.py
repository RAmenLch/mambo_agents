"""Tests for the read-tool file uploader hook and the DeepSeek uploader.

Covers:

- ``build_core_tools`` read tool: hook invocation (sync / async), fallback to
  inline base64 on ``None`` / exceptions / timeouts, text reads untouched.
- Backend wiring: ``file_uploader`` constructor argument and
  ``ReadOnlyBackend`` inheritance.
- ``deepseek_file_uploader``: upload once + store/memo caching, expiry-based
  re-upload, provider scoping, non-image skip, config validation.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import threading
import time

import pytest
from langgraph.store.memory import InMemoryStore

from mambo_agents.backends.local import LocalBackend
from mambo_agents.backends.protocol import ToolTimeouts
from mambo_agents.backends.readonly import ReadOnlyBackend
from mambo_agents.backends.schemas import VirtualPath
from mambo_agents.backends.store import StoreBackend
from mambo_agents.file_uploaders import deepseek_file_uploader
from mambo_agents.file_uploaders._deepseek import _DEFAULT_NAMESPACE
from mambo_agents.middleware.backend_tools import build_core_tools
from tests.test_store_backend import _simulate_graph

# ============================================================================
# Helpers
# ============================================================================

_PNG_BYTES = b"\x89PNG\r\n\x1a\n\x00"
_PNG_B64 = base64.b64encode(_PNG_BYTES).decode("ascii")

_UPLOADED_BLOCK = {"type": "file", "file_id": "file-api-test"}


def _make_backend(file_uploader=None, **kwargs) -> StoreBackend:
    """StoreBackend holding ``/photo.png`` (and optional extra files)."""
    initial_files = {"/photo.png": _PNG_B64, **kwargs.pop("initial_files", {})}
    return StoreBackend(
        store=kwargs.pop("store", InMemoryStore()),
        initial_files=initial_files,
        file_uploader=file_uploader,
        **kwargs,
    )


def _read_tool(backend):
    return next(t for t in build_core_tools(backend) if t.name == "read")


class _FakeUploadedFile:
    def __init__(self, file_id: str, expires_at: float | None = None) -> None:
        self.id = file_id
        self.expires_at = expires_at


class _FakeFilesAPI:
    """Records ``files.create`` calls and returns canned responses."""

    def __init__(self, responses: list[tuple[str, float | None]]) -> None:
        self.calls: list[dict] = []
        self._responses = list(responses)

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("unexpected extra upload call")
        file_id, expires_at = self._responses.pop(0)
        return _FakeUploadedFile(file_id, expires_at)


class _FakeAsyncOpenAI:
    """Minimal ``AsyncOpenAI`` stand-in exposing ``files.create``."""

    def __init__(self, responses: list[tuple[str, float | None]]) -> None:
        self.files = _FakeFilesAPI(responses)
        self.api_key = "test-key"
        self.base_url = "https://api.deepseek.com"


# ============================================================================
# read tool — FileUploader hook
# ============================================================================


class TestReadToolUploaderHook:
    """The read tool replaces the inline base64 block with hook output."""

    def test_sync_uploader_replaces_base64_block(self):
        calls: list[tuple] = []

        def uploader(file_path, base64_content, mime_type):
            calls.append((file_path.value, mime_type))
            assert base64_content == _PNG_B64
            return [dict(_UPLOADED_BLOCK)]

        backend = _make_backend(uploader)
        with _simulate_graph(backend):
            result = _read_tool(backend).invoke({"file_path": "/photo.png"})

        assert result.content_blocks == [_UPLOADED_BLOCK]
        assert calls == [("/photo.png", "image/png")]

    def test_uploader_returns_none_falls_back_to_inline_base64(self):
        backend = _make_backend(lambda file_path, content, mime: None)
        with _simulate_graph(backend):
            result = _read_tool(backend).invoke({"file_path": "/photo.png"})

        blocks = result.content_blocks
        assert blocks[0]["type"] == "image"
        assert blocks[0]["base64"] == _PNG_B64

    def test_uploader_failure_falls_back_to_inline_base64(self, caplog):
        def boom(file_path, content, mime):
            raise RuntimeError("upload failed")

        backend = _make_backend(boom)
        with caplog.at_level(logging.WARNING, logger="mambo_agents.middleware.backend_tools"):
            with _simulate_graph(backend):
                result = _read_tool(backend).invoke({"file_path": "/photo.png"})

        assert result.content_blocks[0]["base64"] == _PNG_B64
        assert any("file_uploader 执行失败" in r.getMessage() for r in caplog.records)

    def test_uploader_timeout_falls_back_to_inline_base64(self):
        def slow(file_path, content, mime):
            time.sleep(0.3)
            return [dict(_UPLOADED_BLOCK)]

        backend = _make_backend(slow, tool_timeouts=ToolTimeouts(file_upload=0.05))
        with _simulate_graph(backend):
            result = _read_tool(backend).invoke({"file_path": "/photo.png"})

        assert result.content_blocks[0]["base64"] == _PNG_B64

    def test_uploader_not_called_for_text_files(self):
        calls: list[str] = []

        def uploader(file_path, content, mime):
            calls.append(file_path.value)
            return [dict(_UPLOADED_BLOCK)]

        backend = _make_backend(
            uploader, initial_files={"/hello.py": "print('hi')"},
        )
        with _simulate_graph(backend):
            result = _read_tool(backend).invoke({"file_path": "/hello.py"})

        assert isinstance(result, str)
        assert calls == []

    def test_single_dict_result_is_normalized(self):
        backend = _make_backend(lambda file_path, content, mime: dict(_UPLOADED_BLOCK))
        with _simulate_graph(backend):
            result = _read_tool(backend).invoke({"file_path": "/photo.png"})

        assert result.content_blocks == [_UPLOADED_BLOCK]

    def test_unsupported_result_type_falls_back(self, caplog):
        backend = _make_backend(lambda file_path, content, mime: "not-a-block")
        with caplog.at_level(logging.WARNING, logger="mambo_agents.middleware.backend_tools"):
            with _simulate_graph(backend):
                result = _read_tool(backend).invoke({"file_path": "/photo.png"})

        assert result.content_blocks[0]["base64"] == _PNG_B64
        assert any("不受支持" in r.getMessage() for r in caplog.records)

    def test_readonly_backend_inherits_uploader(self):
        inner = _make_backend(lambda file_path, content, mime: [dict(_UPLOADED_BLOCK)])
        readonly = ReadOnlyBackend(inner)

        with _simulate_graph(inner):
            result = _read_tool(readonly).invoke({"file_path": "/photo.png"})

        assert result.content_blocks == [_UPLOADED_BLOCK]

    def test_read_tool_description_mentions_pre_upload(self):
        with_uploader = _read_tool(_make_backend(lambda p, c, m: None))
        without_uploader = _read_tool(_make_backend())

        assert "pre-uploaded" in with_uploader.description
        assert "pre-uploaded" not in without_uploader.description

    @pytest.mark.asyncio
    async def test_async_uploader_awaited_on_same_loop(self):
        loop_ids: list[int] = []

        async def uploader(file_path, base64_content, mime_type):
            loop_ids.append(id(asyncio.get_running_loop()))
            return [dict(_UPLOADED_BLOCK)]

        backend = _make_backend(uploader)
        with _simulate_graph(backend):
            result = await _read_tool(backend).ainvoke({"file_path": "/photo.png"})

        assert result.content_blocks == [_UPLOADED_BLOCK]
        assert loop_ids == [id(asyncio.get_running_loop())]

    @pytest.mark.asyncio
    async def test_async_read_runs_sync_uploader_in_thread(self):
        seen: dict[str, threading.Thread] = {}

        def uploader(file_path, base64_content, mime_type):
            seen["thread"] = threading.current_thread()
            return [dict(_UPLOADED_BLOCK)]

        backend = _make_backend(uploader)
        with _simulate_graph(backend):
            result = await _read_tool(backend).ainvoke({"file_path": "/photo.png"})

        assert result.content_blocks == [_UPLOADED_BLOCK]
        assert seen["thread"] is not threading.main_thread()

    def test_sync_read_runs_async_uploader_via_asyncio_run(self):
        async def uploader(file_path, base64_content, mime_type):
            return [dict(_UPLOADED_BLOCK)]

        backend = _make_backend(uploader)
        with _simulate_graph(backend):
            result = _read_tool(backend).invoke({"file_path": "/photo.png"})

        assert result.content_blocks == [_UPLOADED_BLOCK]


# ============================================================================
# Backend wiring
# ============================================================================


class TestBackendWiring:
    def test_default_file_upload_timeout(self):
        assert ToolTimeouts().get("file_upload") == 120.0

    def test_backends_accept_file_uploader(self, tmp_path):
        hook = lambda file_path, content, mime: None  # noqa: E731

        store_backend = StoreBackend(store=InMemoryStore(), file_uploader=hook)
        local_backend = LocalBackend(root_dir=tmp_path, file_uploader=hook)

        assert store_backend._file_uploader is hook
        assert local_backend._file_uploader is hook


# ============================================================================
# deepseek_file_uploader
# ============================================================================


class TestDeepSeekFileUploader:
    @pytest.mark.asyncio
    async def test_upload_once_then_cache_hit(self):
        fake = _FakeAsyncOpenAI([("file-api-1", time.time() + 3600)])
        hook = deepseek_file_uploader(client=fake, store=InMemoryStore())

        first = await hook(VirtualPath("/workspace/photo.png"), _PNG_B64, "image/png")
        second = await hook(VirtualPath("/workspace/photo.png"), _PNG_B64, "image/png")

        expected = [{
            "type": "file",
            "file_id": "file-api-1",
            "mime_type": "image/png",
            "filename": "photo.png",
        }]
        assert first == expected
        assert second == expected
        assert len(fake.files.calls) == 1

        kwargs = fake.files.calls[0]
        assert kwargs["purpose"] == "user_data"
        assert kwargs["file"] == ("photo.png", _PNG_BYTES, "image/png")
        assert kwargs["expires_after"] == {
            "anchor": "created_at",
            "seconds": 30 * 24 * 3600,
        }

    @pytest.mark.asyncio
    async def test_upload_record_persisted_in_store(self):
        fake = _FakeAsyncOpenAI([("file-api-1", time.time() + 3600)])
        store = InMemoryStore()
        hook = deepseek_file_uploader(client=fake, store=store)

        await hook(VirtualPath("/workspace/photo.png"), _PNG_B64, "image/png")

        items = await store.asearch(_DEFAULT_NAMESPACE)
        assert len(items) == 1
        record = items[0].value
        assert record["file_id"] == "file-api-1"
        assert record["mime_type"] == "image/png"
        assert record["filename"] == "photo.png"
        assert record["size"] == len(_PNG_BYTES)
        assert record["expires_at"] > time.time()

    @pytest.mark.asyncio
    async def test_new_hook_instance_reuses_store_cache(self):
        store = InMemoryStore()

        first = _FakeAsyncOpenAI([("file-api-1", time.time() + 3600)])
        await deepseek_file_uploader(client=first, store=store)(
            VirtualPath("/workspace/photo.png"), _PNG_B64, "image/png",
        )

        # Second instance has an empty memo and a fake that would fail on upload.
        second = _FakeAsyncOpenAI([])
        blocks = await deepseek_file_uploader(client=second, store=store)(
            VirtualPath("/workspace/photo.png"), _PNG_B64, "image/png",
        )

        assert blocks[0]["file_id"] == "file-api-1"
        assert second.files.calls == []

    @pytest.mark.asyncio
    async def test_expired_record_triggers_reupload(self):
        fake = _FakeAsyncOpenAI([
            ("file-api-1", time.time() - 1),           # already expired
            ("file-api-2", time.time() + 3600),
        ])
        hook = deepseek_file_uploader(client=fake, store=InMemoryStore())

        first = await hook(VirtualPath("/workspace/photo.png"), _PNG_B64, "image/png")
        second = await hook(VirtualPath("/workspace/photo.png"), _PNG_B64, "image/png")

        assert first[0]["file_id"] == "file-api-1"
        assert second[0]["file_id"] == "file-api-2"
        assert len(fake.files.calls) == 2

    @pytest.mark.asyncio
    async def test_non_image_file_returns_none(self):
        fake = _FakeAsyncOpenAI([])
        hook = deepseek_file_uploader(client=fake, store=InMemoryStore())

        result = await hook(VirtualPath("/workspace/doc.pdf"), _PNG_B64, "application/pdf")

        assert result is None
        assert fake.files.calls == []

    @pytest.mark.asyncio
    async def test_agent_read_via_deepseek_hook_has_no_base64(self):
        fake = _FakeAsyncOpenAI([("file-api-img", time.time() + 3600)])
        backend = _make_backend(
            deepseek_file_uploader(client=fake, store=InMemoryStore()),
        )

        with _simulate_graph(backend):
            result = await _read_tool(backend).ainvoke({"file_path": "/photo.png"})

        blocks = result.content_blocks
        assert blocks[0]["type"] == "file"
        assert blocks[0]["file_id"] == "file-api-img"
        assert "base64" not in blocks[0]

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "env-key")

        hook = deepseek_file_uploader(store=InMemoryStore())

        assert hook._api_key == "env-key"

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("DEEPSEEK_API_BASE", raising=False)

        with pytest.raises(ValueError, match="DeepSeek API key"):
            deepseek_file_uploader(store=InMemoryStore())

    def test_invalid_ttl_raises(self):
        with pytest.raises(ValueError, match="ttl_seconds"):
            deepseek_file_uploader(api_key="k", ttl_seconds=10)

    @pytest.mark.asyncio
    async def test_ttl_none_omits_expires_after_and_never_expires(self):
        fake = _FakeAsyncOpenAI([("file-api-perm", None)])
        hook = deepseek_file_uploader(
            client=fake, store=InMemoryStore(), ttl_seconds=None,
        )

        first = await hook(VirtualPath("/workspace/photo.png"), _PNG_B64, "image/png")
        second = await hook(VirtualPath("/workspace/photo.png"), _PNG_B64, "image/png")

        assert first[0]["file_id"] == "file-api-perm"
        assert second[0]["file_id"] == "file-api-perm"
        assert "expires_after" not in fake.files.calls[0]
        assert len(fake.files.calls) == 1

    @pytest.mark.asyncio
    async def test_store_failure_is_non_fatal(self):
        class _BrokenStore(InMemoryStore):
            async def aget(self, namespace, key, *, refresh_ttl=None):
                raise RuntimeError("store down")

            async def aput(self, namespace, key, value, index=None, *, ttl=None):
                raise RuntimeError("store down")

        fake = _FakeAsyncOpenAI([("file-api-1", time.time() + 3600)])
        hook = deepseek_file_uploader(client=fake, store=_BrokenStore())

        blocks = await hook(VirtualPath("/workspace/photo.png"), _PNG_B64, "image/png")

        assert blocks[0]["file_id"] == "file-api-1"
        assert len(fake.files.calls) == 1
