"""StateBackend — file storage via LangGraph Pregel state channels.

Files are stored in the ``files`` channel of ``FilesystemState``, which
participates in LangGraph checkpointing automatically.  Per-thread
snapshots (``_snapshots``) mirror the channel faithfully, using full
replacement (not incremental merge) to correctly handle checkpoint
rollbacks, branch switches, and file deletions.
"""

from __future__ import annotations

import asyncio
import base64
import fnmatch
import threading

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langgraph._internal._constants import CONFIG_KEY_READ, CONFIG_KEY_SEND
from langgraph.config import get_config
from pydantic import Field, create_model

from mambo_agents.backends.protocol import (
    DownloadFileResult,
    EditResult,
    FileInfo,
    GlobResult,
    GrepMatch,
    GrepResult,
    LsResult,
    ReadResult,
    ReadSummarizer,
    ThreadAwareWorkspace,
    ToolTimeouts,
    UploadFileResult,
    WriteResult,
    _get_file_type,
    _get_mime_type,
)
from mambo_agents.backends.state_schema import FileData
from mambo_agents.backends.utils import (
    detect_trailing_newline_mismatch,
    format_with_line_numbers,
)
from mambo_agents.backends.schemas import VirtualPath,human_size

# ---------------------------------------------------------------------------
# StateBackend
# ---------------------------------------------------------------------------


class StateBackend(ThreadAwareWorkspace):
    """File storage backed by a LangGraph Pregel state channel.

    Files live in the ``files`` field of ``FilesystemState``, which is
    automatically checkpointed after each agent step.

    **Per-thread snapshots** mirror the Pregel channel faithfully: each
    ``_read_files()`` inside a graph **fully replaces** the snapshot for
    that thread, so checkpoint rollbacks and file deletions are honoured.
    Graph-outside uploads are queued per-thread and flushed on first access.

    Extra tool provided: ``tree`` — displays directory structure.

    Parameters:
        initial_files: Optional ``{path: content_str}`` mapping to
            pre-populate new threads on first access.  Files already
            present in the checkpoint are *not* overwritten — only
            genuinely new paths are injected.
    """

    def __init__(
        self,
        initial_files: dict[str, str] | None = None,
        *,
        max_read_chars: int = 100_000,
        summarizer: "ReadSummarizer | None" = None,
        tool_timeouts: ToolTimeouts | None = None,
    ) -> None:
        super().__init__(max_read_chars=max_read_chars, summarizer=summarizer, tool_timeouts=tool_timeouts)
        self._lock = threading.RLock()

        # --- Per-thread channel mirrors (full-replaced on every _read_files) ---
        self._snapshots: dict[str, dict[str, FileData]] = {}

        # --- Per-thread pending uploads (queued outside graph, flushed on
        #     first _read_files for that thread) ---
        self._pending_uploads: dict[str, dict[str, FileData]] = {}

        # --- Immutable template – injected on each new thread's first access
        #     (only for paths NOT already in the checkpoint) ---
        self._initial_files: dict[str, FileData] = {}
        if initial_files:
            for raw_path, content_ in initial_files.items():
                vp = VirtualPath(raw_path)  # validate absolute, no "..", no "//"
                if vp.value.endswith("/"):
                    raise ValueError(
                        f"initial_files path must not end with '/': {raw_path!r} "
                        f"(directories are not supported)"
                    )
                encoding = "base64" if _get_file_type(vp.normalized) != "text" else "utf-8"
                self._initial_files[vp.normalized] = FileData(
                    content=content_, encoding=encoding,
                )

    # ------------------------------------------------------------------
    # Per-thread read / write — graph context vs. outside-graph
    # ------------------------------------------------------------------

    def _get_config(self) -> RunnableConfig:
        """Return the current LangGraph config, or raise a clear error.

        Raises *RuntimeError* when called outside a graph context.
        """
        try:
            config = get_config()
        except RuntimeError:
            raise RuntimeError(
                "StateBackend must be used inside a LangGraph graph execution. "
            ) from None
        configurable = config.get("configurable", {})
        if CONFIG_KEY_READ not in configurable:
            raise RuntimeError(
                "StateBackend requires CONFIG_KEY_READ / CONFIG_KEY_SEND "
                "in the LangGraph config."
            )
        return config

    def _in_graph_context(self) -> bool:
        """Return ``True`` if we are inside a LangGraph execution with
        the ``files`` channel available.

        Checks both that ``get_config()`` succeeds AND that
        ``CONFIG_KEY_READ`` is present in the configurable dict.
        This correctly returns ``False`` for:
        - Direct ``read_tool.invoke()`` calls (no Pregel context)
        - Worker threads spawned by ``asyncio.to_thread`` where the
          context variable may yield a default config without the keys
        """
        try:
            config = get_config()
            configurable = config.get("configurable", {})
            return CONFIG_KEY_READ in configurable
        except RuntimeError:
            return False

    def _resolve_thread_id(self, thread_id: str | None) -> str:
        """Resolve the thread_id in three-tier priority order.

        1. **Explicit** — caller passed a ``thread_id`` value (graph-in or
           graph-out).  Always honoured first.
        2. **Implicit (graph context)** — ``thread_id`` is ``None`` but we
           are inside a Pregel execution.  Extract from
           ``config["configurable"]["thread_id"]``, defaulting to
           ``"__default__"`` if absent.
        3. **Error (outside graph, no explicit value)** — raise
           ``ValueError`` because there is no way to infer which session
           the caller intends.
        """
        if thread_id is not None:
            return thread_id
        if self._in_graph_context():
            config = self._get_config()
            return config["configurable"].get("thread_id", "__default__")
        raise ValueError(
            "thread_id is required when calling upload_files/download_files "
            "outside a graph context."
        )

    def _read_files(self) -> dict[str, FileData]:
        """Read files from Pregel channel with full-replace snapshot sync.

        **Execution order** (each step is critical for correctness):

        1. **Read channel FIRST** — the current channel content determines
           which ``_initial_files`` entries are genuinely new (paths
           absent from the channel).  This ordering prevents accidentally
           overwriting existing checkpoint data with stale initial values.
        2. **Build flush set** — two sources:
           - ``_initial_files`` (only paths missing from the channel)
           - ``_pending_uploads[tid]`` (always flushed — user-explicit)
        3. **Flush to channel** via ``CONFIG_KEY_SEND``, then **re-read**
           to capture the merged state.
        4. **Full-replace snapshot** — ``_snapshots[tid]`` is set to
           ``dict(channel_files)``, *not* incrementally ``.update()``.
           This guarantees that file deletions and checkpoint rollbacks
           are reflected immediately.

        **Subagent fallback**: when the ``files`` channel is absent
        (this graph is a subagent without ``FilesystemState``),
        ``read("files")`` raises ``KeyError``.  We catch it and proceed
        with an empty dict — ``_initial_files`` and pending uploads will
        be flushed to the parent graph's channel on next re-entry.

        Requires graph context.  For outside-graph access, use
        ``download_files(thread_id=...)``.

        Returns a **shallow copy** of the files dict so callers cannot
        mutate the internal snapshot by accident.
        """
        if not self._in_graph_context():
            raise RuntimeError(
                "StateBackend._read_files requires graph context. "
                "Use download_files(thread_id=...) for outside-graph access."
            )

        config = self._get_config()
        thread_id = config["configurable"].get("thread_id", "__default__")
        read = config["configurable"][CONFIG_KEY_READ]

        # (1) Read channel FIRST to determine which initial_files are new.
        #     ``fresh=True`` ensures Pregel reads the latest committed
        #     value for this key, bypassing any in-flight writes.
        try:
            channel_files: dict[str, FileData] = read("files", fresh=True) or {}
        except KeyError:
            # No ``files`` channel in this graph (e.g. subagent) —
            # initial_files and pending uploads will be flushed to the
            # parent graph on next re-entry.
            channel_files = {}

        # (2) Build flush set under lock — guard against concurrent
        #     graph-outside upload_files() calls.
        to_flush: dict[str, FileData] = {}
        with self._lock:
            if thread_id not in self._snapshots:
                # First access for this thread → inject initial_files,
                # but ONLY for paths not already in the checkpoint.
                # Existing checkpoint data is *never* overwritten by
                # initial_files — this preserves edits made by the LLM
                # across backend instance restarts.
                for path, fd in self._initial_files.items():
                    if path not in channel_files:
                        to_flush[path] = fd

            # Pending uploads are always flushed regardless of whether
            # the path exists in the channel — they represent an explicit
            # user action and should overwrite.
            to_flush.update(self._pending_uploads.pop(thread_id, {}))

        # (3) Flush to channel *outside* the lock (Pregel send() may
        #     interact with the event loop).  Re-read immediately after
        #     to capture the reducer-merged state.
        if to_flush:
            send = config["configurable"][CONFIG_KEY_SEND]
            send([("files", to_flush)])
            channel_files = read("files", fresh=True) or {}

        # (4) Full-replace snapshot — guarantees that any files deleted
        #     since the last _read_files (e.g. via channel ``{path: None}``
        #     updates) are gone from our mirror too.
        with self._lock:
            self._snapshots[thread_id] = dict(channel_files)

        return dict(channel_files)

    def _send_files_update(self, update: dict[str, FileData]) -> None:
        """Write a partial files update — to Pregel channel + per-thread snapshot.

        1. Validates we are in graph context (raises ``RuntimeError``
           otherwise — graph-outside writes must go through
           ``upload_files(thread_id=...)``).
        2. Queues the update via ``CONFIG_KEY_SEND`` so LangGraph's
           channel reducer processes it and includes it in the checkpoint.
        3. Lazily mirrors the update to the per-thread snapshot with
           ``.update()``.  This mirror is *best-effort* — the next
           ``_read_files()`` call will full-replace the snapshot from
           the channel, correcting any discrepancies.  So the lazy
           ``.update()`` cannot cause permanent data corruption; it
           merely provides a temporary read cache between channel reads.

        **Note on subagents**: if the current graph has no ``files``
        channel, ``send()`` will raise ``KeyError``.  This is propagated
        to the caller — subagent middleware is expected to handle it.
        """
        if not self._in_graph_context():
            raise RuntimeError(
                "StateBackend._send_files_update requires graph context. "
                "Use upload_files(thread_id=...) for outside-graph access."
            )

        config = self._get_config()
        thread_id = config["configurable"].get("thread_id", "__default__")

        # Queue write to Pregel channel (participates in checkpointing)
        send = config["configurable"][CONFIG_KEY_SEND]
        send([("files", update)])

        # Best-effort mirror to snapshot — next _read_files full-replace
        # will correct any drift.  Using .update() is acceptable here
        # because it's additive only (new or updated files), never stale
        # deletions — those are corrected by _read_files.
        with self._lock:
            if thread_id not in self._snapshots:
                self._snapshots[thread_id] = {}
            self._snapshots[thread_id].update(update)

    # ------------------------------------------------------------------
    # tools – extra tools only (core tools are built by middleware)
    # ------------------------------------------------------------------

    @property
    def tools(self) -> list[StructuredTool]:
        return [
            StructuredTool(
                name="tree",
                description=(
                    "View the directory tree structure. "
                    "Shows directories and files with their sizes in a tree format."
                ),
                args_schema=create_model(
                    "TreeSchema",
                    path=(str, Field(..., description="Root directory to display")),
                    depth=(int, Field(default=3, description="Maximum recursion depth")),
                ),
                func=lambda **kwargs: self.tree(**kwargs),
                coroutine=lambda **kwargs: self.atree(**kwargs),
            ),
        ]

    # ------------------------------------------------------------------
    # Core file operations
    # ------------------------------------------------------------------

    def ls(self, path: VirtualPath) -> LsResult:
        normalized = path.normalized + "/"

        files = self._read_files()
        # Check if path itself is a file, not a directory
        if path.normalized in files:
            return LsResult(error=f"'{path}' is a file, not a directory")

        infos: list[FileInfo] = []
        subdirs: set[str] = set()
        for fpath, fd in files.items():
            if not fpath.startswith(normalized):
                continue
            relative = fpath[len(normalized):]
            if "/" in relative:
                subdirs.add(normalized + relative.split("/")[0] + "/")
            else:
                content = fd.get("content", "")
                infos.append(
                    FileInfo(path=fpath, is_dir=False, size=len(content))
                )

        if not infos and not subdirs:
            # workspace_root always exists even when empty
            if path.normalized == self.workspace_root.normalized:
                return LsResult(entries=[])
            return LsResult(error=f"Path '{path}' not found")

        for sd in sorted(subdirs):
            infos.append(FileInfo(path=sd, is_dir=True, size=0))

        infos.sort(key=lambda fi: fi.path)
        return LsResult(entries=infos)

    def read_raw(
        self,
        file_path: VirtualPath,
        offset: int = 0,
        limit: int | None = 2000,
        include_line_numbers: bool = False,
    ) -> ReadResult:
        if file_path.value.endswith("/"):
            return ReadResult(error=f"Cannot read '{file_path}': looks like a directory")
        files = self._read_files()
        fd = files.get(file_path.normalized)
        if fd is None:
            return ReadResult(error=f"File '{file_path}' not found")

        content = fd.get("content", "")
        encoding = fd.get("encoding", "utf-8")

        if encoding == "base64":
            file_type = _get_file_type(file_path.normalized)
            return ReadResult(
                content=content,
                total_lines=1,
                encoding="base64",
                file_type=file_type,
                mime_type=_get_mime_type(file_path.normalized),
            )

        lines = content.split("\n")
        if lines and lines[-1] == "":
            lines = lines[:-1]
        total = len(lines)

        end = offset + limit if limit is not None else None
        sliced = lines[offset:end]
        raw_slice = "\n".join(sliced)
        content = (
            format_with_line_numbers(raw_slice, start_line=offset + 1)
            if include_line_numbers
            else raw_slice
        )
        return ReadResult(
            content=content,
            total_lines=total,
            encoding="utf-8",
        )

    def write(
        self, file_path: VirtualPath, content: str, overwrite: bool = False,
    ) -> WriteResult:
        if file_path.value.endswith("/"):
            return WriteResult(error=f"'{file_path}' looks like a directory — use ls() to list it")
        files = self._read_files()
        # Check if file_path is a "directory" (contains child files)
        prefix = file_path.normalized + "/"
        for fpath in files:
            if fpath.startswith(prefix):
                return WriteResult(
                    error=f"'{file_path}' is a directory, cannot write to it"
                )
        if file_path.normalized in files and not overwrite:
            return WriteResult(
                error=(
                    f"Cannot write '{file_path}': file already exists. "
                    "Read the file and use edit() to modify it, "
                    "or use overwrite=True to replace the file."
                ),
            )
        encoding = "base64" if _get_file_type(file_path.normalized) != "text" else "utf-8"
        fd: FileData = {"content": content, "encoding": encoding}
        self._send_files_update({file_path.normalized: fd})
        return WriteResult(path=file_path.normalized)

    def edit(
        self,
        file_path: VirtualPath,
        old_str: str,
        new_str: str,
        *,
        replace_all: bool = False,
    ) -> EditResult:
        if file_path.value.endswith("/"):
            return EditResult(error=f"'{file_path}' looks like a directory — use ls() to list it")
        if not old_str:
            return EditResult(error="old_str must not be empty")
        files = self._read_files()
        # Check if file_path is a "directory" (contains child files)
        prefix = file_path.normalized + "/"
        for fpath in files:
            if fpath.startswith(prefix):
                return EditResult(
                    error=f"'{file_path}' is a directory, cannot edit it"
                )
        existing_fd = files.get(file_path.normalized)
        if existing_fd is None:
            return EditResult(
                error=(
                    f"Cannot edit '{file_path}': file not found. "
                    "To create a new file, use write()."
                ),
            )
        if existing_fd.get("encoding") == "base64":
            return EditResult(
                error=(
                    f"Cannot edit '{file_path}': file is binary "
                    f"(base64 encoded). Use write() to replace the file."
                ),
            )

        existing_content = existing_fd.get("content", "")

        occurrences = existing_content.count(old_str)

        if occurrences == 0:
            # Detect trailing-newline mismatch: model adds \n to old_str
            # but the file does not end with a newline.
            mismatch = detect_trailing_newline_mismatch(
                file_path.normalized, old_str, existing_content,
            )
            if mismatch is not None:
                return mismatch
            return EditResult(
                error=(
                    f"Cannot edit '{file_path}': old_str not found in file. "
                    "Read the file first to see its exact content."
                ),
            )

        if occurrences > 1 and not replace_all:
            return EditResult(
                error=(
                    f"Cannot edit '{file_path}': old_str appears {occurrences} times "
                    f"in the file. Use replace_all=True to replace all occurrences, "
                    f"or provide a more specific old_str with surrounding context."
                ),
            )

        fd: FileData = {
            "content": existing_content.replace(old_str, new_str),
            "encoding": existing_fd.get("encoding", "utf-8"),
        }
        self._send_files_update({file_path.normalized: fd})
        return EditResult(path=file_path.normalized, occurrences=occurrences)

    def grep(
        self,
        pattern: str,
        path: VirtualPath,
        glob: str | None = None,
        regex: bool = False,
        offset: int = 0,
        limit: int | None = None,
    ) -> GrepResult:
        if not pattern:
            return GrepResult(error="pattern must not be empty")
        if regex:
            import re as _re
            try:
                _re.compile(pattern)
            except _re.error as e:
                return GrepResult(error=f"Invalid regex pattern: {e}")
        files = self._read_files()
        raw_matches = _grep_in_memory(files, pattern, path.normalized, glob, regex, self._max_grep_matches)
        return self._apply_grep_limit(raw_matches, offset, limit)

    def glob(self, pattern: str, path: VirtualPath) -> GlobResult:
        files = self._read_files()
        return _glob_in_memory(files, pattern, path.normalized)

    # ------------------------------------------------------------------
    # Extra operations
    # ------------------------------------------------------------------

    def tree(self, path: VirtualPath, depth: int = 3) -> str:
        """Render a directory tree by recursively calling ``ls()``."""
        if isinstance(path, str):
            path = VirtualPath(path)
        entries = _collect_tree_entries(self, path, depth)
        return _format_tree(entries)

    async def atree(self, path: VirtualPath, depth: int = 3) -> str:
        """Async: Render a directory tree."""
        if isinstance(path, str):
            path = VirtualPath(path)
        return await asyncio.to_thread(self.tree, path, depth)

    # ------------------------------------------------------------------
    # Developer API — upload / download
    # ------------------------------------------------------------------

    def upload_files(
        self,
        files: list[tuple[VirtualPath, bytes]],
        *,
        thread_id: str | None = None,
    ) -> list[UploadFileResult]:
        """Upload multiple files.

        **Dual-path branching** (based on ``_in_graph_context()``):

        **Graph-in path** (``thread_id`` auto-resolved from config):
        writes directly to the Pregel channel via ``_send_files_update()``,
        which also lazily mirrors to the per-thread snapshot.  The files
        immediately participate in checkpointing.

        **Graph-out path** (``thread_id`` **required**): queues files in
        ``_pending_uploads[tid]``.  They are flushed to the channel on
        the next ``_read_files()`` call for that thread — which happens
        on the first file operation (read/write/ls/etc.) during the next
        graph execution.

        Parameters:
            files: List of ``(path, raw_bytes)`` tuples.
            thread_id: Explicit thread identifier.  Required outside a
                graph; optional (auto-resolved) inside a graph.
        """
        tid = self._resolve_thread_id(thread_id)
        results: list[UploadFileResult] = []
        update: dict[str, FileData] = {}
        for path, raw_content in files:
            try:
                text = raw_content.decode("utf-8")
                encoding = "utf-8"
            except UnicodeDecodeError:
                text = base64.b64encode(raw_content).decode("ascii")
                encoding = "base64"
            fd: FileData = {"content": text, "encoding": encoding}
            update[path.normalized] = fd
            results.append(UploadFileResult(path=path.normalized, error=None))

        # Branch: graph-in → write channel + snapshot; graph-out → queue
        if self._in_graph_context():
            self._send_files_update(update)
        else:
            with self._lock:
                pending = self._pending_uploads.setdefault(tid, {})
                pending.update(update)

        return results

    def download_files(
        self,
        paths: list[VirtualPath],
        *,
        thread_id: str | None = None,
    ) -> list[DownloadFileResult]:
        """Download multiple files.

        **Dual-path branching** (based on ``_in_graph_context()``):

        **Graph-in path** (``thread_id`` auto-resolved from config):
        delegates to ``_read_files()``, which reads the Pregel channel and
        full-replaces the per-thread snapshot.  This guarantees the files
        are consistent with the current checkpoint.

        **Graph-out path** (``thread_id`` **required**): reads from the
        last-synced per-thread snapshot, merged with any pending uploads
        that haven't been flushed yet.  This is the best-effort view of
        files *between* graph executions — no channel access is possible
        outside the graph.

        Parameters:
            paths: List of absolute file paths to download.
            thread_id: Explicit thread identifier.  Required outside a
                graph; optional (auto-resolved) inside a graph.
        """
        tid = self._resolve_thread_id(thread_id)
        results: list[DownloadFileResult] = []

        # Branch: graph-in → read channel directly; graph-out → read cache
        if self._in_graph_context():
            files = self._read_files()
        else:
            # Merge snapshot (last synced channel state) with pending
            # uploads (graph-outside writes not yet flushed).  Pending
            # takes precedence via .update() so callers see their own
            # recently-uploaded files.
            with self._lock:
                snapshot = self._snapshots.get(tid, {})
                pending = self._pending_uploads.get(tid, {})
                files = dict(snapshot)
                files.update(pending)

        for path in paths:
            fd = files.get(path.normalized)
            if fd is None:
                results.append(
                    DownloadFileResult(path=path.normalized, content=None, error="file_not_found")
                )
                continue
            content = fd.get("content", "")
            encoding = fd.get("encoding", "utf-8")
            if encoding == "utf-8":
                content_bytes = content.encode("utf-8")
            else:
                content_bytes = base64.standard_b64decode(content)
            results.append(
                DownloadFileResult(path=path.normalized, content=content_bytes, error=None)
            )
        return results


# ============================================================================
# Internal helpers
# ============================================================================


def _grep_in_memory(
    files: dict[str, FileData],
    pattern: str,
    path: str = "/",
    file_glob: str | None = None,
    regex: bool = False,
    max_matches: int = 1000,
) -> list[GrepMatch]:
    """Collect grep matches from in-memory files.

    Returns:
        Raw list of matches collected up to *max_matches* (no offset/limit applied).
    """
    import re as _re
    if regex:
        compiled = _re.compile(pattern)
    else:
        compiled = _re.compile(_re.escape(pattern))

    path_prefix = path.rstrip("/") if path != "/" else "/"

    matches: list[GrepMatch] = []
    for fpath, fd in sorted(files.items()):
        if len(matches) >= max_matches:
            break
        if path_prefix != "/" and not fpath.startswith(path_prefix):
            continue
        if file_glob and not fnmatch.fnmatch(fpath, file_glob):
            continue
        if fd.get("encoding") == "base64":
            continue
        content = fd.get("content", "")
        for li, line in enumerate(content.split("\n"), start=1):
            if len(matches) >= max_matches:
                break
            if compiled.search(line):
                matches.append(GrepMatch(path=fpath, line=li, text=line))

    return matches


def _glob_in_memory(
    files: dict[str, FileData],
    pattern: str,
    path: str = "/",
) -> GlobResult:
    path_prefix = path.rstrip("/") if path != "/" else ""

    results: list[FileInfo] = []
    for fpath in sorted(files):
        if path_prefix and fpath != path_prefix and not fpath.startswith(path_prefix + "/"):
            continue
        if not fnmatch.fnmatch(fpath, pattern):
            continue
        content = files[fpath].get("content", "")
        results.append(FileInfo(path=fpath, is_dir=False, size=len(content)))

    return GlobResult(matches=results)


def _collect_tree_entries(
    backend: StateBackend,
    path: VirtualPath,
    depth: int,
) -> list[tuple[str, bool, int]]:
    """Recurse via ``ls()`` and collect ``(path, is_dir, size)`` tuples."""
    result = backend.ls(path)
    if result.error or not result.entries:
        return []

    entries: list[tuple[str, bool, int]] = []
    for fi in result.entries:
        if fi.is_dir and depth > 1:
            sub_path = fi.path.value if fi.path.value.endswith("/") else fi.path.value + "/"
            entries.extend(
                _collect_tree_entries(
                    backend,
                    VirtualPath(sub_path),
                    depth - 1,
                )
            )
    # Direct children after subdirectories (visual order)
    for fi in result.entries:
        entries.append((str(fi.path), fi.is_dir, fi.size))
    return entries


def _format_tree(
    entries: list[tuple[str, bool, int]],
    prefix: str = "",
) -> str:
    """Render ``(path, is_dir, size)`` entries as a visual tree."""
    import os as _os

    lines: list[str] = []
    for i, (p, is_dir, size) in enumerate(entries):
        is_last = i == len(entries) - 1
        connector = "└── " if is_last else "├── "
        name = _os.path.basename(p.rstrip("/")) or p.rstrip("/")
        if is_dir:
            name += "/"
            size_str = ""
        else:
            size_str = f" ({human_size(size)})"
        lines.append(f"{prefix}{connector}{name}{size_str}")
    return "\n".join(lines)
