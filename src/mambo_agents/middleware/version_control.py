"""Version-control middleware — checkpoint-scoped file snapshots with manual rollback.

Provides a user-facing (not Agent-facing) mechanism to:

*   Automatically back up files before ``write``/``edit``/``delete`` tool calls,
    associating every backed-up file with the **parent checkpoint** (the checkpoint
    this ``invoke()`` call started from).
*   Emit ``BackupEvent`` as custom stream events so consumers can track what
    was backed up in real time.
*   Restore files to a previous checkpoint via :meth:`VersionControlMiddleware.restore_files`
    (called manually by the user — no automatic rollback).

.. note::

    This middleware does **not** expose any tools to the LLM — all version
    data is purely for human consumption via :class:`VersionStore`.

Design principles
-----------------
* **Storage via LangGraph BaseStore** — blobs and indices are persisted through
  ``BaseStore``, using namespaces ``(thread_id, "mambo_vc_blobs")`` and
  ``(thread_id, "mambo_vc_index")``.  Works with ``InMemoryStore``, Postgres, or
  any ``BaseStore`` implementation.  No local filesystem dependency.
* **Write-time persistence** — each ``wrap_tool_call`` backup writes its
  blob AND updates the index atomically via ``store.put()``.  Survives
  ``astream`` interruption.
* **Incremental** — only files actually mutated by the LLM are backed up.
* **Content-addressed** — SHA256 blobs; identical content is only stored once.
* **Manual rollback only** — users call ``restore_files()`` explicitly.
  No automatic rollback via config.

Usage::

    from langgraph.store.memory import InMemoryStore

    from mambo_agents.middleware.version_control import (
        BackupEvent,
        RestoreFileResult,
        VersionStore,
        VersionControlMiddleware,
    )

    store = VersionStore(store=InMemoryStore())
    middleware = VersionControlMiddleware(store=store, backend=backend)

    # Graph-external inspection
    snapshots = store.list_snapshots("thread-1")
    old = store.get_file("thread-1", "cp_abc123", "/workspace/src/main.py")

    agent = create_mambo_agent(
        "gpt-4o",
        backend=LocalBackend(),
        middleware=[middleware],
    )

    # Receive real-time backup events via custom stream events
    async for mode, chunk in agent.astream(
        {"messages": [...]}, config, stream_mode=["updates", "custom"],
    ):
        if mode == "custom":
            event = BackupEvent(**chunk)
            print(f"[backup] ckpt={event.checkpoint_id} file={event.file_path}")

    # Restore files manually (outside the graph)
    results = middleware.restore_files("t1", "cp_abc123", files=["/workspace/file1.py"])
    for r in results:
        if r.success:
            print(f"[restored] {r.path}")
        else:
            print(f"[failed] {r.path}: {r.message}")
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Literal

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import ToolCall, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_config, get_store, get_stream_writer
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.store.base import BaseStore
from langgraph.typing import ContextT
from pydantic import BaseModel, ConfigDict, Field

from langchain.agents.middleware.types import AgentState as _AgentState

from mambo_agents.backends.protocol import BackendProtocol
from mambo_agents.backends.schemas import BackendError, VirtualPath

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MUTATING_TOOLS = frozenset({"write", "edit", "delete"})
"""Tool names whose invocation triggers a backup."""


# ============================================================================
# Data models
# ============================================================================


class Snapshot(BaseModel):
    """File snapshot scoped to one ``parent_checkpoint``.

    Only files that were **mutated** during the scope are recorded — not the
    entire workspace.  Each path maps to a SHA256 blob identifier; the actual
    file content lives under ``blobs/<sha256>``.
    """

    model_config = ConfigDict(frozen=False)

    checkpoint_id: str
    """The parent checkpoint this snapshot is attached to."""

    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    """ISO-8601 creation time."""

    file_blobs: dict[str, str] = Field(default_factory=dict)
    """``{path_normalized: sha256_hex}`` — files changed under this checkpoint."""


class VersionIndex(BaseModel):
    """Per-thread index of all snapshots.

    Persisted via :class:`VersionStore` using LangGraph ``BaseStore``
    under namespace ``(thread_id, "mambo_vc_index")`` with key ``"index"``.
    """

    model_config = ConfigDict(frozen=False)

    thread_id: str
    """Thread / session identifier."""

    snapshots: list[Snapshot] = Field(default_factory=list)
    """Ordered list (oldest first).  Branching is implied by checkpoint hierarchy."""


class BackupEvent(BaseModel):
    """Emitted as a **custom stream event** every time a file is backed up.

    Consume via ``astream(..., stream_mode=["updates", "custom"])``::

        async for mode, chunk in agent.astream(..., stream_mode=["updates", "custom"]):
            if mode == "custom":
                event = BackupEvent(**chunk)
                print(event.checkpoint_id, event.file_path)
    """

    model_config = ConfigDict(frozen=True)

    type: Literal["backup"] = "backup"
    """Discriminator for custom event routing."""

    source: Literal["version_control"] = Field(
        default="version_control",
        description="Identifies the middleware source of this event.",
    )

    thread_id: str
    """Thread / session identifier."""

    checkpoint_id: str
    """Parent checkpoint this backup is attached to."""

    file_path: str
    """Normalised absolute virtual path that was backed up."""

    sha256: str
    """SHA-256 hex digest of the backed-up content."""

    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    """ISO-8601 time the backup was created."""


class RestoreFileResult(BaseModel):
    """Per-file result from :meth:`VersionControlMiddleware.restore_files`.

    Each file in the restore operation gets one of these, regardless of
    whether the restore succeeded or failed.  Failures carry a
    human-readable ``message`` so the caller can display per-file errors.
    """

    model_config = ConfigDict(frozen=True)

    path: str
    """Normalised absolute virtual path that was targeted for restore."""

    success: bool
    """``True`` if the file content was written to disk successfully."""

    message: str = ""
    """Human-readable status or error description.  Empty on success."""


class VersionControlConfig(BaseModel):
    """Top-level configuration for :class:`VersionControlMiddleware`."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    store: BaseStore | None = None
    """LangGraph ``BaseStore`` to persist version data.  When ``None`` (default),
    auto-resolved from the graph execution context via ``get_store()``,
    falling back to a lazy ``InMemoryStore``.  Ignored when a
    :class:`VersionStore` is passed directly to :func:`create_mambo_agent`."""

    auto_snapshot: bool = True
    """When ``True`` (default), mutate-tool calls automatically trigger backups."""

    whitelist_folders: list[VirtualPath] = Field(default_factory=list)
    """Absolute virtual paths of folders to monitor in whitelist mode.

    When non-empty, only files located under one of these folders are
    backed up and available for rollback.  An empty list means **no**
    files will be processed (strict whitelist mode).
    """

    mutating_tool_names: list[str] = Field(default_factory=lambda: ["write", "edit", "delete"])
    """Tool names that trigger a pre-mutation backup.

    Defaults to ``["write", "edit", "delete"]``.  Extend this when your
    backend exposes custom mutating tools (e.g. ``"patch"``, ``"rename"``).
    """


# ============================================================================
# VersionStore — BaseStore-backed persistent storage
# ============================================================================


class VersionStore:
    """Per-thread version journal backed by LangGraph ``BaseStore``.

    Uses two namespaces per thread:

    * ``(thread_id, "mambo_vc_blobs")`` — content-addressed blob storage.
      Key = SHA256 hex digest, value = ``{"content": "..."}``.
    * ``(thread_id, "mambo_vc_index")`` — serialised ``VersionIndex``
      under key ``"index"``.

    When *store* is ``None`` at construction, the store is resolved lazily
    via ``get_store()`` (graph context) with an ``InMemoryStore`` fallback.
    """

    def __init__(self, store: BaseStore | None = None) -> None:
        self._store = store

    # ------------------------------------------------------------------
    # Store resolution
    # ------------------------------------------------------------------

    def _get_store(self) -> BaseStore:
        """Return the store instance.

        Priority: (1) explicit *store* from constructor,
        (2) ``get_store()`` from graph execution context,
        (3) lazy-created ``InMemoryStore`` fallback.
        """
        if self._store is not None:
            return self._store
        try:
            return get_store()
        except RuntimeError:
            if self._store is None:
                from langgraph.store.memory import InMemoryStore
                self._store = InMemoryStore()
            return self._store

    # ------------------------------------------------------------------
    # Namespace helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _blob_ns(thread_id: str) -> tuple[str, str]:
        return (thread_id, "mambo_vc_blobs")

    @staticmethod
    def _index_ns(thread_id: str) -> tuple[str, str]:
        return (thread_id, "mambo_vc_index")

    # ------------------------------------------------------------------
    # Public graph-external query API (sync)
    # ------------------------------------------------------------------

    def list_snapshots(self, thread_id: str) -> list[Snapshot]:
        """Return every snapshot for *thread_id* in chronological order."""
        index = self._load_index(thread_id)
        return list(index.snapshots)

    def get_file(
        self, thread_id: str, checkpoint_id: str, path: str,
    ) -> str | None:
        """Read the content of *path* as of *checkpoint_id*, or ``None``."""
        index = self._load_index(thread_id)
        snapshot = self._find_snapshot(index, checkpoint_id)
        if snapshot is None:
            return None
        sha = snapshot.file_blobs.get(path)
        if sha is None:
            return None
        return self._read_blob(thread_id, sha)

    def get_changed_files(
        self, thread_id: str, checkpoint_id: str,
    ) -> list[str]:
        """Paths recorded for *checkpoint_id*."""
        index = self._load_index(thread_id)
        snapshot = self._find_snapshot(index, checkpoint_id)
        if snapshot is None:
            return []
        return list(snapshot.file_blobs.keys())

    # ── High-level session queries ──

    def get_all_changed_files(self, thread_id: str) -> frozenset[str]:
        """Return the deduplicated set of all files changed across every
        checkpoint in *thread_id*.

        Answers "what has been modified in this entire conversation?".
        """
        index = self._load_index(thread_id)
        all_files: set[str] = set()
        for snap in index.snapshots:
            all_files.update(snap.file_blobs.keys())
        return frozenset(all_files)

    def get_latest_snapshot(self, thread_id: str) -> Snapshot | None:
        """Return the most recent :class:`Snapshot` for *thread_id*, or ``None``.

        Answers "what happened in the latest turn?".
        """
        index = self._load_index(thread_id)
        return index.snapshots[-1] if index.snapshots else None

    def get_latest_changed_files(self, thread_id: str) -> list[str]:
        """Return files changed in the most recent checkpoint for *thread_id*.

        Convenience shortcut for ``get_latest_snapshot(…).file_blobs.keys()``.
        """
        latest = self.get_latest_snapshot(thread_id)
        if latest is None:
            return []
        return list(latest.file_blobs.keys())

    # ------------------------------------------------------------------
    # Public graph-external query API (async)
    # ------------------------------------------------------------------

    async def alist_snapshots(self, thread_id: str) -> list[Snapshot]:
        """Async: return every snapshot for *thread_id* in chronological order."""
        index = await self._aload_index(thread_id)
        return list(index.snapshots)

    async def aget_file(
        self, thread_id: str, checkpoint_id: str, path: str,
    ) -> str | None:
        """Async: read the content of *path* as of *checkpoint_id*, or ``None``."""
        index = await self._aload_index(thread_id)
        snapshot = self._find_snapshot(index, checkpoint_id)
        if snapshot is None:
            return None
        sha = snapshot.file_blobs.get(path)
        if sha is None:
            return None
        return await self._aread_blob(thread_id, sha)

    async def aget_changed_files(
        self, thread_id: str, checkpoint_id: str,
    ) -> list[str]:
        """Async: paths recorded for *checkpoint_id*."""
        index = await self._aload_index(thread_id)
        snapshot = self._find_snapshot(index, checkpoint_id)
        if snapshot is None:
            return []
        return list(snapshot.file_blobs.keys())

    # ── High-level session queries (async) ──

    async def aget_all_changed_files(self, thread_id: str) -> frozenset[str]:
        """Async: deduplicated set of all files changed across every checkpoint."""
        index = await self._aload_index(thread_id)
        all_files: set[str] = set()
        for snap in index.snapshots:
            all_files.update(snap.file_blobs.keys())
        return frozenset(all_files)

    async def aget_latest_snapshot(self, thread_id: str) -> Snapshot | None:
        """Async: most recent snapshot for *thread_id*, or ``None``."""
        index = await self._aload_index(thread_id)
        return index.snapshots[-1] if index.snapshots else None

    async def aget_latest_changed_files(self, thread_id: str) -> list[str]:
        """Async: files changed in the most recent checkpoint for *thread_id*."""
        latest = await self.aget_latest_snapshot(thread_id)
        if latest is None:
            return []
        return list(latest.file_blobs.keys())

    # ------------------------------------------------------------------
    # Public graph-internal write API (sync)
    # ------------------------------------------------------------------

    def add_file_to_snapshot(
        self,
        thread_id: str,
        checkpoint_id: str,
        path: str,
        sha: str,
    ) -> None:
        """Record *path* → *sha* in the snapshot for *checkpoint_id*.

        Creates the snapshot automatically if it does not yet exist.
        The index is re-serialised on every call, so the data
        survives graph interruptions.
        """
        index = self._load_index(thread_id)
        snapshot = self._find_snapshot(index, checkpoint_id)
        if snapshot is None:
            snapshot = Snapshot(checkpoint_id=checkpoint_id)
            index.snapshots.append(snapshot)
        snapshot.file_blobs[path] = sha
        self._save_index(thread_id, index)

    def save_blob(self, thread_id: str, sha256: str, content: str) -> None:
        """Write *content* to the blob store.  No-op if the blob exists."""
        store = self._get_store()
        ns = self._blob_ns(thread_id)
        existing = store.get(ns, sha256)
        if existing is not None:
            return
        store.put(ns, sha256, {"content": content})

    # ------------------------------------------------------------------
    # Public graph-internal write API (async)
    # ------------------------------------------------------------------

    async def aadd_file_to_snapshot(
        self,
        thread_id: str,
        checkpoint_id: str,
        path: str,
        sha: str,
    ) -> None:
        """Async: record *path* → *sha* in the snapshot for *checkpoint_id*."""
        index = await self._aload_index(thread_id)
        snapshot = self._find_snapshot(index, checkpoint_id)
        if snapshot is None:
            snapshot = Snapshot(checkpoint_id=checkpoint_id)
            index.snapshots.append(snapshot)
        snapshot.file_blobs[path] = sha
        await self._asave_index(thread_id, index)

    async def asave_blob(self, thread_id: str, sha256: str, content: str) -> None:
        """Async: write *content* to the blob store.  No-op if the blob exists."""
        store = self._get_store()
        ns = self._blob_ns(thread_id)
        existing = await store.aget(ns, sha256)
        if existing is not None:
            return
        await store.aput(ns, sha256, {"content": content})

    # ------------------------------------------------------------------
    # Internal helpers (sync)
    # ------------------------------------------------------------------

    def _read_blob(self, thread_id: str, sha256: str) -> str | None:
        store = self._get_store()
        ns = self._blob_ns(thread_id)
        item = store.get(ns, sha256)
        if item is None:
            return None
        return item.value.get("content")

    def _load_index(self, thread_id: str) -> VersionIndex:
        store = self._get_store()
        ns = self._index_ns(thread_id)
        item = store.get(ns, "index")
        if item is None:
            return VersionIndex(thread_id=thread_id)
        return VersionIndex(**item.value)

    def _save_index(self, thread_id: str, index: VersionIndex) -> None:
        store = self._get_store()
        ns = self._index_ns(thread_id)
        store.put(ns, "index", index.model_dump(exclude_defaults=False))

    # ------------------------------------------------------------------
    # Internal helpers (async)
    # ------------------------------------------------------------------

    async def _aread_blob(self, thread_id: str, sha256: str) -> str | None:
        store = self._get_store()
        ns = self._blob_ns(thread_id)
        item = await store.aget(ns, sha256)
        if item is None:
            return None
        return item.value.get("content")

    async def _aload_index(self, thread_id: str) -> VersionIndex:
        store = self._get_store()
        ns = self._index_ns(thread_id)
        item = await store.aget(ns, "index")
        if item is None:
            return VersionIndex(thread_id=thread_id)
        return VersionIndex(**item.value)

    async def _asave_index(self, thread_id: str, index: VersionIndex) -> None:
        store = self._get_store()
        ns = self._index_ns(thread_id)
        await store.aput(ns, "index", index.model_dump(exclude_defaults=False))

    @staticmethod
    def _find_snapshot(index: VersionIndex, checkpoint_id: str) -> Snapshot | None:
        return next(
            (s for s in index.snapshots if s.checkpoint_id == checkpoint_id),
            None,
        )


# ============================================================================
# VersionControlMiddleware
# ============================================================================


class VersionControlMiddleware(AgentMiddleware[_AgentState, ContextT, Any]):

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        *,
        store: VersionStore,
        backend: BackendProtocol,
        whitelist_folders: list[VirtualPath] | None = None,
        mutating_tool_names: list[str] | None = None,
    ) -> None:
        super().__init__()
        self._store = store
        self._backend = backend

        # ── whitelist ──
        self._whitelist: list[VirtualPath] = whitelist_folders or []

        # ── mutating tools ──
        self._mutating_tool_names: frozenset[str] = frozenset(
            mutating_tool_names if mutating_tool_names is not None
            else _MUTATING_TOOLS
        )

        # ── per-thread state (cleared each invoke) ──
        self._current_parent_cp: dict[str, str] = {}
        """{thread_id: parent_cp} — the checkpoint this invoke started from."""
        self._backed_up: dict[str, set[str]] = {}
        """{thread_id: {path_normalized...}} — already backed-up in this invoke."""

    # ------------------------------------------------------------------
    # before_agent — record parent checkpoint
    # ------------------------------------------------------------------

    def before_agent(
        self,
        state: _AgentState,
        runtime: Any,
    ) -> dict[str, Any] | None:
        config = get_config()
        tid = self._resolve_thread_id(config)

        self._current_parent_cp[tid] = self._resolve_parent_checkpoint_id(config)
        self._backed_up[tid] = set()

        return None

    async def abefore_agent(
        self,
        state: _AgentState,
        runtime: Any,
    ) -> dict[str, Any] | None:
        return self.before_agent(state, runtime)

    # ------------------------------------------------------------------
    # wrap_tool_call — backup-on-write
    # ------------------------------------------------------------------

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Any,
    ) -> ToolMessage | Any:
        tool_call = request.tool_call
        if tool_call["name"] in self._mutating_tool_names:
            path = _extract_file_path(tool_call)
            if path:
                self._backup_file(path)
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Any,
    ) -> ToolMessage | Any:
        tool_call = request.tool_call
        if tool_call["name"] in self._mutating_tool_names:
            path = _extract_file_path(tool_call)
            if path:
                await self._abackup_file(path)
        return await handler(request)

    # ------------------------------------------------------------------
    # Whitelist helpers
    # ------------------------------------------------------------------

    def _is_in_whitelist(self, path: VirtualPath) -> bool:
        """Return ``True`` if *path* is under any whitelisted folder.

        When the whitelist is empty (strict mode), no file passes.
        """
        if not self._whitelist:
            return False
        return any(path.is_under(prefix.normalized) for prefix in self._whitelist)

    # ------------------------------------------------------------------
    # Backup — read → blob → index (immediate persistence)
    # ------------------------------------------------------------------

    def _backup_file(self, path: VirtualPath) -> None:
        """Read *path* from backend, persist blob + update index atomically."""
        # ── whitelist guard ──
        if not self._is_in_whitelist(path):
            return

        normalized = path.normalized
        tid = self._resolve_thread_id(get_config())

        if normalized in self._backed_up.get(tid, set()):
            return  # already backed up in this invoke

        result = self._backend.read_raw(path, limit=None)
        if result.error or result.content is None:
            return

        sha = hashlib.sha256(result.content.encode("utf-8")).hexdigest()

        # blob + index — both immediately durable
        self._store.save_blob(tid, sha, result.content)
        cp = self._current_parent_cp.get(tid, "") or self._resolve_parent_checkpoint_id(get_config())
        if cp:
            self._current_parent_cp.setdefault(tid, cp)
            self._store.add_file_to_snapshot(tid, cp, normalized, sha)

        self._backed_up.setdefault(tid, set()).add(normalized)

        # ── emit custom stream event (like subagents.py) ──
        writer = get_stream_writer()
        if writer is not None:
            writer(
                BackupEvent(
                    thread_id=tid,
                    checkpoint_id=cp,
                    file_path=normalized,
                    sha256=sha,
                ).model_dump()
            )

    async def _abackup_file(self, path: VirtualPath) -> None:
        """Async backup — read file + persist blob + index via async store pipeline.

        Uses ``read_raw`` (sync, but I/O-bound and GIL-safe) and then
        calls async ``asave_blob`` / ``aadd_file_to_snapshot`` to avoid
        ``asyncio.to_thread`` threading issues with async-only stores
        (e.g. aiosqlite-based stores).
        """
        # ── whitelist guard ──
        if not self._is_in_whitelist(path):
            return

        normalized = path.normalized
        tid = self._resolve_thread_id(get_config())

        if normalized in self._backed_up.get(tid, set()):
            return  # already backed up in this invoke

        result = await self._backend.aread_raw(path, limit=None)
        if result.error or result.content is None:
            return

        sha = hashlib.sha256(result.content.encode("utf-8")).hexdigest()

        # blob + index — async durable persistence
        await self._store.asave_blob(tid, sha, result.content)
        cp = self._current_parent_cp.get(tid, "") or self._resolve_parent_checkpoint_id(get_config())
        if cp:
            self._current_parent_cp.setdefault(tid, cp)
            await self._store.aadd_file_to_snapshot(tid, cp, normalized, sha)

        self._backed_up.setdefault(tid, set()).add(normalized)

        # ── emit custom stream event ──
        writer = get_stream_writer()
        if writer is not None:
            writer(
                BackupEvent(
                    thread_id=tid,
                    checkpoint_id=cp,
                    file_path=normalized,
                    sha256=sha,
                ).model_dump()
            )

    # ------------------------------------------------------------------
    # Rollback execution
    # ------------------------------------------------------------------

    def restore_files(
        self,
        thread_id: str,
        checkpoint_id: str,
        *,
        files: list[str] | None = None,
        all: bool = False,
    ) -> list[RestoreFileResult]:
        """Restore files to their state as of *checkpoint_id*.

        Can be called at any time — inside or outside the graph —
        and is the **only** supported rollback mechanism.

        Args:
            thread_id: Thread / session identifier.
            checkpoint_id: Which checkpoint's snapshot to restore from.
            files: Explicit list of file paths to restore.  Mutually
                   exclusive with ``all=True``.
            all: If ``True``, restore every file recorded in the snapshot
                 for *checkpoint_id*.

        Returns:
            Per-file :class:`RestoreFileResult` entries — one for every
            file that passed the whitelist.  Check ``result.success``
            and ``result.message`` for per-file error details.
        """
        if all and files:
            raise ValueError("'all' and 'files' are mutually exclusive")
        if not all and not files:
            raise ValueError("either 'all=True' or 'files' must be specified")

        paths: list[str]
        if all:
            paths = self._store.get_changed_files(thread_id, checkpoint_id)
        else:
            paths = files  # type: ignore[assignment]

        results: list[RestoreFileResult] = []
        for path_str in paths:
            vp = VirtualPath(path_str)
            if not self._is_in_whitelist(vp):
                results.append(RestoreFileResult(
                    path=vp.normalized,
                    success=False,
                    message="Path is outside the whitelist",
                ))
                continue

            content = self._store.get_file(thread_id, checkpoint_id, path_str)
            if content is None:
                results.append(RestoreFileResult(
                    path=vp.normalized,
                    success=False,
                    message=f"Backup content not found for checkpoint '{checkpoint_id}'",
                ))
                continue

            write_result = self._backend.write(vp, content, overwrite=True)
            if write_result.error is not None:
                results.append(RestoreFileResult(
                    path=vp.normalized,
                    success=False,
                    message=str(write_result.error),
                ))
            else:
                results.append(RestoreFileResult(
                    path=vp.normalized,
                    success=True,
                ))

        return results

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_thread_id(config: RunnableConfig) -> str:
        return config.get("configurable", {}).get("thread_id", "__default__")

    @staticmethod
    def _resolve_parent_checkpoint_id(config: RunnableConfig) -> str:
        configurable = config.get("configurable", {})
        ## 主动指向的保存点
        vc_parent = configurable.get("version_control_ckpt_id")
        if vc_parent:
            return vc_parent

        ## 时间旅行的ckpt_id -> 预期为父节点的父节点或爷节点
        cp_id = config.get("metadata",{}).get("checkpoint_id")
        if cp_id:
            return cp_id

        ## 父节点 -> 预期为 INPUT
        cp_map = configurable.get("checkpoint_map", {})
        checkpoint_ns = configurable.get("checkpoint_ns", "")
        parent_cp = cp_map.get(checkpoint_ns) or cp_map.get("")

        if parent_cp:
            return parent_cp

        return "__initial__"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_file_path(tool_call: ToolCall) -> VirtualPath | None:
    """Pull a ``VirtualPath`` from the tool-call arguments."""
    args = tool_call["args"]
    # write / edit / delete all use "file_path" or "path"
    for key in ("file_path", "path"):
        raw = args.get(key)
        if raw is not None:
            try:
                return VirtualPath(raw)
            except (ValueError, TypeError, BackendError):
                return None
    return None
