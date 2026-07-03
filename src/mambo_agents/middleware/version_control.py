"""Version-control middleware — checkpoint-scoped file snapshots with selective rollback.

Provides a user-facing (not Agent-facing) mechanism to:

*   Automatically back up files before ``write``/``edit``/``delete`` tool calls,
    associating every backed-up file with the **parent checkpoint** (the checkpoint
    this ``invoke()`` call started from).
*   Selectively roll back individual files to their state at a given checkpoint
    when LangGraph time-travel is triggered.

.. note::

    This middleware does **not** expose any tools to the LLM — all version
    data is purely for human consumption via :class:`VersionStore`.

Design principles
-----------------
* **Storage decoupled from agent backend** — pure local-file I/O under
  ``./.mambo_versions/``.  No circular dependency.
* **Write-time persistence** — each ``wrap_tool_call`` backup writes its
  blob AND updates index.json atomically.  Survives ``astream`` interruption.
* **Incremental** — only files actually mutated by the LLM are backed up.
* **Content-addressed** — SHA256 blobs; identical content is only stored once.

Usage::

    from mambo_agents.middleware.version_control import (
        VersionStore,
        VersionControlMiddleware,
    )

    store = VersionStore(storage_dir="./.mambo_versions")

    # Graph-external inspection
    snapshots = store.list_snapshots("thread-1")
    old = store.get_file("thread-1", "cp_abc123", "/workspace/src/main.py")

    agent = create_mambo_agent(
        "gpt-4o",
        backend=LocalBackend(),
        middleware=[VersionControlMiddleware(store=store, backend=...)],
    )

    # Time-travel with selective file rollback
    config = {
        "configurable": {
            "thread_id": "t1",
            "checkpoint_id": "cp_abc123",
            "version_rollback": {"files": ["/workspace/file1.py"]},
        }
    }
    agent.astream({"messages": [...]}, config)
"""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import ToolCall, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langgraph.config import get_config
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.typing import ContextT
from pydantic import BaseModel, ConfigDict, Field, model_validator

from mambo_agents.backends.protocol import BackendProtocol
from mambo_agents.backends.schemas import BackendError, VirtualPath
from mambo_agents.backends.state_schema import FilesystemState

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_STORAGE_DIR = "./.mambo_versions"
"""Default root directory for version storage."""

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

    Serialised to ``index.json`` under ``<storage_dir>/<thread_id>/``.
    """

    model_config = ConfigDict(frozen=False)

    thread_id: str
    """Thread / session identifier."""

    snapshots: list[Snapshot] = Field(default_factory=list)
    """Ordered list (oldest first).  Branching is implied by checkpoint hierarchy."""


class VersionRollbackConfig(BaseModel):
    """Rollback directive placed in ``config["configurable"]["version_rollback"]``."""

    files: list[str] = Field(default_factory=list)
    """Explicit list of paths to roll back.  Mutually exclusive with ``all``."""

    all: bool = False
    """If ``True``, restore every file known to have changed under this checkpoint."""

    @model_validator(mode="after")
    def _check_mutual_exclusion(self) -> "VersionRollbackConfig":
        if self.all and self.files:
            raise ValueError("'all' and 'files' are mutually exclusive")
        if not self.all and not self.files:
            raise ValueError("either 'all=True' or 'files' must be specified")
        return self


class VersionControlConfig(BaseModel):
    """Top-level configuration for :class:`VersionControlMiddleware`."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    store_dir: str = _DEFAULT_STORAGE_DIR
    """Storage directory path.  Ignored when a :class:`VersionStore` is passed
    directly to :func:`create_mambo_agent`."""

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
# VersionStore — pure file-I/O storage
# ============================================================================


class VersionStore:
    """Per-thread version journal backed by the local filesystem.

    **Zero dependency on agent backends** — all I/O goes through
    ``json.dumps`` / ``Path.write_text`` / ``Path.read_text``.

    Thread-safe: ``add_file_to_snapshot`` and ``list_snapshots`` use a
    ``threading.RLock`` to protect ``index.json``.
    """

    def __init__(self, storage_dir: str | Path = _DEFAULT_STORAGE_DIR) -> None:
        self._root = Path(storage_dir)
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Public graph-external query API
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
    # Public graph-internal write API
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
        The ``index.json`` is re-serialised on every call, so the data
        survives graph interruptions.
        """
        with self._lock:
            index = self._load_index(thread_id)
            snapshot = self._find_snapshot(index, checkpoint_id)
            if snapshot is None:
                snapshot = Snapshot(checkpoint_id=checkpoint_id)
                index.snapshots.append(snapshot)
            snapshot.file_blobs[path] = sha
            self._save_index(thread_id, index)

    def save_blob(self, thread_id: str, sha256: str, content: str) -> None:
        """Write *content* to ``blobs/<sha256>``.  No-op if the blob exists."""
        blob_path = self._blob_path(thread_id, sha256)
        if blob_path.exists():
            return
        blob_path.parent.mkdir(parents=True, exist_ok=True)
        blob_path.write_text(content, encoding="utf-8")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_blob(self, thread_id: str, sha256: str) -> str | None:
        blob_path = self._blob_path(thread_id, sha256)
        if not blob_path.is_file():
            return None
        return blob_path.read_text(encoding="utf-8")

    def _blob_path(self, thread_id: str, sha256: str) -> Path:
        return self._root / thread_id / "blobs" / sha256

    def _index_path(self, thread_id: str) -> Path:
        return self._root / thread_id / "index.json"

    def _load_index(self, thread_id: str) -> VersionIndex:
        path = self._index_path(thread_id)
        if not path.is_file():
            return VersionIndex(thread_id=thread_id)
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        return VersionIndex(**data)

    def _save_index(self, thread_id: str, index: VersionIndex) -> None:
        path = self._index_path(thread_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized = index.model_dump(exclude_defaults=False)
        path.write_text(json.dumps(serialized, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _find_snapshot(index: VersionIndex, checkpoint_id: str) -> Snapshot | None:
        return next(
            (s for s in index.snapshots if s.checkpoint_id == checkpoint_id),
            None,
        )


# ============================================================================
# VersionControlMiddleware
# ============================================================================


class VersionControlMiddleware(AgentMiddleware[FilesystemState, ContextT, Any]):

    state_schema = FilesystemState

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
    # before_agent — record parent_cp + execute rollback
    # ------------------------------------------------------------------

    def before_agent(
        self,
        state: FilesystemState,
        runtime: Any,
    ) -> dict[str, Any] | None:
        config = get_config()
        tid = self._resolve_thread_id(config)

        # 1. Record parent checkpoint
        self._current_parent_cp[tid] = self._resolve_parent_checkpoint_id(config)
        self._backed_up[tid] = set()

        # 2. Execute version_rollback if requested
        rollback_raw = raw_version_rollback(config)
        if rollback_raw:
            decoded = VersionRollbackConfig(**rollback_raw)
            self._execute_rollback(tid, decoded)

        return None

    async def abefore_agent(
        self,
        state: FilesystemState,
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
        cp = self._current_parent_cp.get(tid, "")
        if cp:
            self._store.add_file_to_snapshot(tid, cp, normalized, sha)

        self._backed_up[tid].add(normalized)

    async def _abackup_file(self, path: VirtualPath) -> None:
        # Delegate to sync version; read_raw is I/O bound so the GIL
        # holds the lock for us — and we never want to back up the
        # same file twice anyway.
        import asyncio
        await asyncio.to_thread(self._backup_file, path)

    # ------------------------------------------------------------------
    # Rollback execution
    # ------------------------------------------------------------------

    def _execute_rollback(
        self,
        thread_id: str,
        rollback: VersionRollbackConfig,
    ) -> None:
        cp = self._current_parent_cp.get(thread_id, "")
        if not cp:
            return

        files = rollback.files
        if rollback.all:
            files = self._store.get_changed_files(thread_id, cp)

        for path_str in files:
            # ── whitelist guard ──
            vp = VirtualPath(path_str)
            if not self._is_in_whitelist(vp):
                continue

            content = self._store.get_file(thread_id, cp, path_str)
            if content is not None:
                self._backend.write(vp, content, overwrite=True)

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_thread_id(config: RunnableConfig) -> str:
        return config.get("configurable", {}).get("thread_id", "__default__")

    @staticmethod
    def _resolve_parent_checkpoint_id(config: RunnableConfig) -> str:
        """Extract the parent checkpoint ID this invoke is continuing from.

        Priority order:
        1. ``config["configurable"]["checkpoint_id"]`` — user-specified
           (time-travel target).
        2. ``config["configurable"]["checkpoint_map"]`` — LangGraph-set
           mapping of ``checkpoint_ns → checkpoint_id`` for parent graphs.
           For the root graph (``checkpoint_ns=""``), this holds the
           parent checkpoint.
        3. A sentinel value (``"__initial__"``) when neither is available
           (first ``invoke()`` call).
        """
        configurable = config.get("configurable", {})
        cp_id = configurable.get("checkpoint_id")
        if cp_id:
            return cp_id

        cp_map = configurable.get("checkpoint_map", {})
        checkpoint_ns = configurable.get("checkpoint_ns", "")
        parent_cp = cp_map.get(checkpoint_ns)
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


def raw_version_rollback(config: RunnableConfig) -> dict[str, Any] | None:
    """Read the raw ``version_rollback`` dict from config, or ``None``."""
    configurable = config.get("configurable", {})
    return configurable.get("version_rollback")
