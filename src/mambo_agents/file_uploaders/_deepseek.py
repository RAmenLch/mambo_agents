"""DeepSeek Files API uploader — built-in ``FileUploader`` implementation.

The ``read`` tool returns image / document content as inline base64 blocks.
That base64 ends up both in the LangGraph checkpoint and in every model
request — large files blow up checkpoints and can hit the 32 MiB
inline-image limit of the DeepSeek API.  The DeepSeek Files API solves the
request-size problem by uploading the file once and referencing it via
``file_id`` (see https://api-docs.deepseek.com/zh-cn/guides/files_api).

:func:`deepseek_file_uploader` returns an **async** :data:`FileUploader`
hook that:

1. hashes the file content and reuses a previously uploaded ``file_id``
   (process-memory memo + LangGraph ``store`` upload records);
2. otherwise uploads the file through the DeepSeek Files API
   (``purpose="user_data"``, TTL between 1 hour and 30 days);
3. returns ``[{"type": "file", "file_id": ...}]`` — so the ``read``
   ``ToolMessage``, and therefore the checkpoint, only carries a small
   reference block.

The hook is never enabled implicitly: pass it to a backend via the
``file_uploader`` constructor argument.

Requires the ``openai`` package (``pip install openai``) unless a ready-made
``AsyncOpenAI`` client is passed via ``client=``.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import os
import threading
import time
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

from mambo_agents.backends.protocol import FileUploader
from mambo_agents.backends.schemas import VirtualPath

if TYPE_CHECKING:
    from langgraph.store.base import BaseStore
    from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

_FILE_TTL_SECONDS = 30 * 24 * 3600
"""Maximum DeepSeek file TTL (30 days)."""

_DEFAULT_API_BASE = "https://api.deepseek.com"

_DEFAULT_NAMESPACE: tuple[str, ...] = ("mambo_agents", "deepseek_files")
"""Default store namespace for upload records."""


class _DeepSeekFilesUploader:
    """Async ``FileUploader`` hook backed by the DeepSeek Files API.

    Upload records (``file_id`` / expiry / mime / filename / size) are cached
    by content hash in a LangGraph store, so repeated reads of the same file
    reuse the same ``file_id`` without uploading again.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        client: "AsyncOpenAI | None" = None,
        store: "BaseStore | None" = None,
        namespace: tuple[str, ...] = _DEFAULT_NAMESPACE,
        ttl_seconds: int | None = _FILE_TTL_SECONDS,
        request_timeout: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        if ttl_seconds is not None and not (3600 <= ttl_seconds <= _FILE_TTL_SECONDS):
            raise ValueError(
                "ttl_seconds 必须在 3600 到 2592000（30 天）之间，或为 None（永久有效），"
                f"当前为 {ttl_seconds}"
            )
        if client is not None:
            api_key = getattr(client, "api_key", None) or api_key or ""
            api_base = api_base or str(getattr(client, "base_url", "") or "")
        else:
            api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
            api_base = api_base or os.getenv("DEEPSEEK_API_BASE", "") or _DEFAULT_API_BASE
            if not api_key:
                raise ValueError(
                    "deepseek_file_uploader 需要 DeepSeek API key：请传入 api_key=、"
                    "设置环境变量 DEEPSEEK_API_KEY，或通过 client= 传入已创建的 AsyncOpenAI 实例。"
                )
        self._api_key = api_key
        self._api_base = api_base
        self._client = client
        self._ttl_seconds = ttl_seconds
        self._request_timeout = request_timeout
        self._max_retries = max_retries

        if store is None:
            from langgraph.store.memory import InMemoryStore

            store = InMemoryStore()
        self._store = store
        self._namespace = tuple(namespace)

        # Provider-scoped cache key: DeepSeek files belong to the API key that
        # uploaded them, so different key / base URL combinations must not
        # share cache entries.
        self._provider_key = hashlib.sha256(
            f"{api_base}|{api_key}".encode("utf-8")
        ).hexdigest()[:32]

        self._memo: dict[str, tuple[str, float | None]] = {}
        self._memo_lock = threading.Lock()
        self._lock: asyncio.Lock | None = None
        self._lock_loop: Any = None

    # ------------------------------------------------------------------
    # FileUploader contract (async)
    # ------------------------------------------------------------------

    async def __call__(
        self,
        file_path: VirtualPath,
        base64_content: str,
        mime_type: str,
    ) -> list[dict] | None:
        if not (mime_type or "").startswith("image/"):
            logger.debug(
                "deepseek_file_uploader 仅支持图片（DeepSeek Files API 限制），"
                "跳过上传并回退内联 base64：%s（%s）",
                file_path.value, mime_type,
            )
            return None

        raw = base64.b64decode(base64_content)
        digest = hashlib.sha256(raw).hexdigest()
        cache_key = f"{self._provider_key}:{digest}"

        now = time.time()
        file_id = self._memo_get(cache_key, now)
        if file_id is None:
            async with self._lock_for_current_loop():
                file_id = self._memo_get(cache_key, now)
                if file_id is None:
                    file_id = await self._store_lookup(cache_key, now)
                if file_id is None:
                    file_id, expires_at = await self._upload(raw, mime_type, file_path)
                    self._memo_put(cache_key, file_id, expires_at)
                    await self._store_save(
                        cache_key, file_id, expires_at, raw, mime_type, file_path,
                    )
        return self._build_blocks(file_id, mime_type, file_path)

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _memo_get(self, cache_key: str, now: float) -> str | None:
        """Process-memory lookup — avoids the store round-trip on repeat reads."""
        with self._memo_lock:
            entry = self._memo.get(cache_key)
            if entry is None:
                return None
            file_id, expires_at = entry
            if expires_at is not None and now >= expires_at:
                self._memo.pop(cache_key, None)
                return None
            return file_id

    def _memo_put(self, cache_key: str, file_id: str, expires_at: float | None) -> None:
        with self._memo_lock:
            self._memo[cache_key] = (file_id, expires_at)

    async def _store_lookup(self, cache_key: str, now: float) -> str | None:
        """Look up a persisted upload record; store failures are non-fatal."""
        try:
            item = await self._store.aget(self._namespace, cache_key)
        except Exception as exc:  # noqa: BLE001 — cache is best-effort
            logger.warning(
                "DeepSeek 上传缓存读取失败（忽略缓存继续上传）：%s: %s",
                type(exc).__name__, exc,
            )
            return None
        record = item.value if item is not None else None
        if not isinstance(record, dict):
            return None
        file_id = record.get("file_id")
        expires_at = record.get("expires_at")
        if not isinstance(file_id, str) or not file_id:
            return None
        if isinstance(expires_at, (int, float)) and now >= float(expires_at):
            return None
        self._memo_put(
            cache_key, file_id,
            float(expires_at) if isinstance(expires_at, (int, float)) else None,
        )
        return file_id

    async def _store_save(
        self,
        cache_key: str,
        file_id: str,
        expires_at: float | None,
        raw: bytes,
        mime_type: str,
        file_path: VirtualPath,
    ) -> None:
        """Persist an upload record; store failures are non-fatal."""
        record = {
            "file_id": file_id,
            "expires_at": expires_at,
            "mime_type": mime_type or "",
            "filename": PurePosixPath(file_path.value).name,
            "size": len(raw),
            "created_at": time.time(),
            "cache_key": cache_key,
        }
        try:
            await self._store.aput(self._namespace, cache_key, record)
        except Exception as exc:  # noqa: BLE001 — cache is best-effort
            logger.warning(
                "DeepSeek 上传缓存写入失败（不影响本次结果）：%s: %s",
                type(exc).__name__, exc,
            )

    # ------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------

    async def _upload(
        self, raw: bytes, mime_type: str, file_path: VirtualPath
    ) -> tuple[str, float | None]:
        """Upload *raw* via the Files API; returns ``(file_id, expires_at)``."""
        filename = PurePosixPath(file_path.value).name or "file"
        kwargs: dict[str, Any] = {
            "file": (filename, raw, mime_type or "application/octet-stream"),
            "purpose": "user_data",
        }
        if self._ttl_seconds is not None:
            kwargs["expires_after"] = {
                "anchor": "created_at",
                "seconds": self._ttl_seconds,
            }
        if self._client is not None:
            uploaded = await self._client.files.create(**kwargs)
        else:
            async with self._new_client() as client:
                uploaded = await client.files.create(**kwargs)

        file_id = getattr(uploaded, "id", None)
        if not isinstance(file_id, str) or not file_id:
            raise RuntimeError(f"DeepSeek Files API 未返回合法 file_id：{uploaded!r}")
        expires_at = getattr(uploaded, "expires_at", None)
        if not isinstance(expires_at, (int, float)) or expires_at <= 0:
            expires_at = (
                time.time() + self._ttl_seconds if self._ttl_seconds else None
            )
        return file_id, float(expires_at) if expires_at is not None else None

    def _new_client(self) -> "AsyncOpenAI":
        """Create a short-lived AsyncOpenAI client for a single upload."""
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "deepseek_file_uploader 需要 openai 包：请先 `pip install openai`，"
                "或通过 client= 传入已创建的 openai.AsyncOpenAI 实例。"
            ) from exc
        return AsyncOpenAI(
            api_key=self._api_key,
            base_url=self._api_base,
            timeout=self._request_timeout,
            max_retries=self._max_retries,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _lock_for_current_loop(self) -> asyncio.Lock:
        """Per-loop dedup lock (recreated when the hook runs on a new loop)."""
        loop = asyncio.get_running_loop()
        if self._lock is None or self._lock_loop is not loop:
            self._lock = asyncio.Lock()
            self._lock_loop = loop
        return self._lock

    @staticmethod
    def _build_blocks(
        file_id: str, mime_type: str, file_path: VirtualPath
    ) -> list[dict]:
        """Build the replacement content block for the ``read`` ``ToolMessage``."""
        block: dict[str, Any] = {"type": "file", "file_id": file_id}
        if mime_type:
            block["mime_type"] = mime_type
        filename = PurePosixPath(file_path.value).name
        if filename:
            block["filename"] = filename
        return [block]


def deepseek_file_uploader(
    *,
    api_key: str | None = None,
    api_base: str | None = None,
    client: "AsyncOpenAI | None" = None,
    store: "BaseStore | None" = None,
    namespace: tuple[str, ...] = _DEFAULT_NAMESPACE,
    ttl_seconds: int | None = _FILE_TTL_SECONDS,
    request_timeout: float = 60.0,
    max_retries: int = 2,
) -> FileUploader:
    """Return an async :data:`FileUploader` backed by the DeepSeek Files API.

    The hook uploads each image once (``POST /files``, ``purpose=user_data``)
    and returns ``[{"type": "file", "file_id": ...}]`` as the replacement
    content block, so the ``read`` ``ToolMessage`` — and therefore the
    LangGraph checkpoint and every model request — only carries a small
    reference instead of the raw base64 body.

    Upload records are cached by content hash in *store* (a LangGraph
    ``BaseStore``): repeated reads of the same file reuse the same
    ``file_id`` without uploading again.  Pass the same store to several
    backends / processes to share the cache; ``store=None`` uses an
    in-process ``InMemoryStore``.

    The hook is never enabled implicitly — configure it via the backend's
    ``file_uploader`` constructor argument.  DeepSeek's Files API only
    supports images (JPEG / PNG / GIF / WebP); other multimodal files are
    left inline (the hook returns ``None``).

    Args:
        api_key: DeepSeek API key.  Falls back to ``DEEPSEEK_API_KEY``.
        api_base: API base URL.  Falls back to ``DEEPSEEK_API_BASE`` or
            ``https://api.deepseek.com``.
        client: A ready-made ``openai.AsyncOpenAI`` instance to reuse
            (recommended for long-lived async applications).  When omitted,
            an ``AsyncOpenAI`` client is created per upload and closed
            afterwards — requires the ``openai`` package.
        store: LangGraph store used to persist upload records across
            processes.  Defaults to an in-process ``InMemoryStore``.
        namespace: Store namespace for the upload records.
        ttl_seconds: Uploaded-file TTL (3600 – 2592000 = 30 days), or
            ``None`` for files that never expire.  Defaults to 30 days.
        request_timeout: Per-request timeout for the Files API call.
        max_retries: Retry count for the Files API call.

    Example::

        from langgraph.store.memory import InMemoryStore
        from mambo_agents.backends.local import LocalBackend
        from mambo_agents.file_uploaders import deepseek_file_uploader

        backend = LocalBackend(
            root_dir="/data",
            file_uploader=deepseek_file_uploader(store=InMemoryStore()),
        )
    """
    return _DeepSeekFilesUploader(
        api_key=api_key,
        api_base=api_base,
        client=client,
        store=store,
        namespace=namespace,
        ttl_seconds=ttl_seconds,
        request_timeout=request_timeout,
        max_retries=max_retries,
    )
