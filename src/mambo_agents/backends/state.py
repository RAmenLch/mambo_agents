"""StateBackend — file storage via LangGraph Pregel state channels.

Files are stored in the ``files`` channel of ``FilesystemState``, which
participates in LangGraph checkpointing automatically.  A memory cache
(``_pending_files``) bridges the gap between construction-time file
population and the first graph execution.
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
    BackendProtocol,
    DownloadFileResult,
    EditResult,
    FileInfo,
    GlobResult,
    GrepMatch,
    GrepResult,
    LsResult,
    ReadResult,
    UploadFileResult,
    WriteResult,
    _get_file_type,
    _get_mime_type,
)
from mambo_agents.backends.state_schema import FileData
from mambo_agents.backends.utils import (
    LINE_NUMBER_WIDTH,
    MAX_LINE_LENGTH,
    detect_trailing_newline_mismatch,
    format_with_line_numbers,
    human_size,
)

# ---------------------------------------------------------------------------
# StateBackend
# ---------------------------------------------------------------------------


class StateBackend(BackendProtocol):
    """File storage backed by a LangGraph Pregel state channel.

    Files live in the ``files`` field of ``FilesystemState``, which is
    automatically checkpointed after each agent step.  A memory cache
    (``_pending_files``) allows files to be populated *before* the graph
    starts — via ``initial_files`` or ``upload_files()`` — and is flushed
    to the Pregel channel on the first file access during execution.

    Extra tool provided: ``tree`` — displays directory structure.

    Parameters:
        initial_files: Optional ``{path: content_str}`` mapping to
            pre-populate the store before graph execution.
    """

    def __init__(self, initial_files: dict[str, str] | None = None) -> None:
        # Memory cache for files populated outside graph context.
        # Flushed to the Pregel channel on the first ``_read_files()``
        # call inside a graph execution.
        self._lock = threading.Lock()
        self._pending_files: dict[str, FileData] = {}
        if initial_files:
            for path, content_ in initial_files.items():
                encoding = "base64" if _get_file_type(path) != "text" else "utf-8"
                self._pending_files[path] = FileData(
                    content=content_, encoding=encoding,
                )

    # ------------------------------------------------------------------
    # Dual-mode read / write (graph context or memory cache)
    # ------------------------------------------------------------------
    #
    # **Mirror pattern**: ``_pending_files`` is kept in sync with the
    # Pregel ``files`` channel during graph execution.  Writes inside a
    # graph go to *both* the channel and the cache; reads inside a graph
    # read from the channel and then mirror the result back to the cache.
    #
    # This ensures that after ``invoke()`` returns — when we are outside
    # the graph context — ``_pending_files`` still has the latest state
    # and callers can read files directly.

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

    def _read_files(self) -> dict[str, FileData]:
        """Read the files dict — from Pregel channel or memory cache.

        **Mirror pattern**: ``_pending_files`` is the authoritative
        mutable cache.  It is *never* cleared and *never* overwritten
        with ``=`` — only ``.update()`` is used so that data written
        via ``_send_files_update`` cannot be lost by a stale channel
        read.

        **Inside a graph**: flushes pending (outside-graph) files to
        the channel on first call, reads from the channel, then merges
        the channel data into the cache.

        **Subagents** (no ``files`` channel): returns the cache directly.

        **Outside a graph**: returns the cache directly.

        Returns a **shallow copy** of the files dict so callers cannot
        mutate the internal cache by accident.
        """
        if not self._in_graph_context():
            with self._lock:
                return dict(self._pending_files)

        # Snapshot pending files under lock, then flush to channel
        # outside the lock (Pregel operations may be slow).
        with self._lock:
            pending = dict(self._pending_files) if self._pending_files else None
        if pending:
            self._send_files_update(pending)

        config = self._get_config()
        read = config["configurable"][CONFIG_KEY_READ]
        try:
            channel_files = read("files", fresh=True) or {}
        except KeyError:
            # No ``files`` channel in this graph (e.g. subagent)
            with self._lock:
                return dict(self._pending_files)

        # Merge channel data into cache under lock (never overwrite,
        # never clear).  ``_send_files_update`` always mirrors first,
        # so the cache is a superset of the channel — merging ensures
        # we pick up files written by other nodes / restored from
        # checkpoint.
        with self._lock:
            self._pending_files.update(channel_files)
            return dict(self._pending_files)

    def _send_files_update(self, update: dict[str, FileData]) -> None:
        """Write a partial files update — to Pregel channel + memory cache.

        **Always mirrors to ``_pending_files``** so the cache stays in
        sync with the channel.  Inside a graph, also queues the update
        via ``CONFIG_KEY_SEND`` for checkpoint participation.

        Silently degrades to cache-only when the ``files`` channel is
        absent (e.g. inside subagents that lack ``FilesystemState``).

        Cache mutation is protected by ``self._lock`` to guard against
        concurrent writes from the ``asyncio.to_thread`` pool.  Pregel
        ``send()`` is called **outside** the lock to avoid holding it
        during I/O.
        """
        # Mirror into cache under lock
        with self._lock:
            self._pending_files.update(update)

        if not self._in_graph_context():
            return  # Outside graph: cache is the only storage

        try:
            config = self._get_config()
            send = config["configurable"][CONFIG_KEY_SEND]
            send([("files", update)])
        except (KeyError, RuntimeError):
            # Channel not available (subagent without files channel) —
            # mirror already done above, silently OK.
            pass

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
                    path=(str, Field(default="/", description="Root directory to display")),
                    depth=(int, Field(default=3, description="Maximum recursion depth")),
                ),
                func=lambda **kwargs: self.tree(**kwargs),
                coroutine=lambda **kwargs: self.atree(**kwargs),
            ),
        ]

    # ------------------------------------------------------------------
    # Core file operations
    # ------------------------------------------------------------------

    def ls(self, path: str) -> LsResult:
        normalized = path.rstrip("/") + "/" if path != "/" else "/"
        infos: list[FileInfo] = []
        subdirs: set[str] = set()

        files = self._read_files()
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

        for sd in sorted(subdirs):
            infos.append(FileInfo(path=sd, is_dir=True, size=0))

        infos.sort(key=lambda fi: fi.path)
        return LsResult(entries=infos)

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> ReadResult:
        files = self._read_files()
        fd = files.get(file_path)
        if fd is None:
            return ReadResult(error=f"File '{file_path}' not found")

        content = fd.get("content", "")
        encoding = fd.get("encoding", "utf-8")

        if encoding == "base64":
            file_type = _get_file_type(file_path)
            return ReadResult(
                content=content,
                total_lines=1,
                encoding="base64",
                file_type=file_type,
                mime_type=_get_mime_type(file_path),
            )

        lines = content.split("\n")
        if lines and lines[-1] == "":
            lines = lines[:-1]
        total = len(lines)

        sliced = lines[offset: offset + limit]
        return ReadResult(
            content=format_with_line_numbers(
                "\n".join(sliced), start_line=offset + 1,
            ),
            total_lines=total,
            encoding="utf-8",
        )

    def write(
        self, file_path: str, content: str, overwrite: bool = False,
    ) -> WriteResult:
        files = self._read_files()
        if file_path in files and not overwrite:
            return WriteResult(
                error=(
                    f"Cannot write '{file_path}': file already exists. "
                    "Read the file and use edit() to modify it, "
                    "or use overwrite=True to replace the file."
                ),
            )
        encoding = "base64" if _get_file_type(file_path) != "text" else "utf-8"
        fd: FileData = {"content": content, "encoding": encoding}
        self._send_files_update({file_path: fd})
        return WriteResult(path=file_path)

    def edit(
        self,
        file_path: str,
        old_str: str,
        new_str: str,
        *,
        replace_all: bool = False,
    ) -> EditResult:
        files = self._read_files()
        existing_fd = files.get(file_path)
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
                file_path, old_str, existing_content,
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
        self._send_files_update({file_path: fd})
        return EditResult(path=file_path, occurrences=occurrences)

    def grep(
        self,
        pattern: str,
        path: str = "/",
        glob: str | None = None,
    ) -> GrepResult:
        files = self._read_files()
        return _grep_in_memory(files, pattern, path, glob)

    def glob(self, pattern: str, path: str = "/") -> GlobResult:
        files = self._read_files()
        return _glob_in_memory(files, pattern, path)

    # ------------------------------------------------------------------
    # Extra operations
    # ------------------------------------------------------------------

    def tree(self, path: str = "/", depth: int = 3) -> str:
        """Render a directory tree by recursively calling ``ls()``."""
        entries = _collect_tree_entries(self, path, depth)
        return _format_tree(entries)

    async def atree(self, path: str = "/", depth: int = 3) -> str:
        """Async: Render a directory tree."""
        return await asyncio.to_thread(self.tree, path, depth)

    # ------------------------------------------------------------------
    # Developer API — upload / download
    # ------------------------------------------------------------------

    def upload_files(
        self, files: list[tuple[str, bytes]]
    ) -> list[UploadFileResult]:
        """Upload multiple files (works inside or outside graph context).

        Inside a graph the files go directly to the Pregel channel;
        outside a graph they are cached in ``_pending_files`` and
        injected on the first ``_read_files()`` call.
        """
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
            update[path] = fd
            results.append(UploadFileResult(path=path, error=None))

        self._send_files_update(update)
        return results

    def download_files(
        self, paths: list[str]
    ) -> list[DownloadFileResult]:
        results: list[DownloadFileResult] = []
        files = self._read_files()
        for path in paths:
            fd = files.get(path)
            if fd is None:
                results.append(
                    DownloadFileResult(path=path, content=None, error="file_not_found")
                )
                continue
            content = fd.get("content", "")
            encoding = fd.get("encoding", "utf-8")
            if encoding == "utf-8":
                content_bytes = content.encode("utf-8")
            else:
                content_bytes = base64.standard_b64decode(content)
            results.append(
                DownloadFileResult(path=path, content=content_bytes, error=None)
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
) -> GrepResult:
    path_prefix = path.rstrip("/") if path != "/" else "/"

    matches: list[GrepMatch] = []
    for fpath, fd in sorted(files.items()):
        if path_prefix != "/" and not fpath.startswith(path_prefix):
            continue
        if file_glob and not fnmatch.fnmatch(fpath, file_glob):
            continue
        if fd.get("encoding") == "base64":
            continue
        content = fd.get("content", "")
        for li, line in enumerate(content.split("\n"), start=1):
            if pattern in line:
                matches.append(GrepMatch(path=fpath, line=li, text=line))

    return GrepResult(matches=matches)


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
    path: str,
    depth: int,
) -> list[tuple[str, bool, int]]:
    """Recurse via ``ls()`` and collect ``(path, is_dir, size)`` tuples."""
    result = backend.ls(path)
    if result.error or not result.entries:
        return []

    entries: list[tuple[str, bool, int]] = []
    for fi in result.entries:
        if fi.is_dir and depth > 1:
            entries.extend(
                _collect_tree_entries(
                    backend,
                    fi.path if fi.path.endswith("/") else fi.path + "/",
                    depth - 1,
                )
            )
    # Direct children after subdirectories (visual order)
    for fi in result.entries:
        entries.append((fi.path, fi.is_dir, fi.size))
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
