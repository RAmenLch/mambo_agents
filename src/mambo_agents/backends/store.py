"""StoreBackend — persistent file storage via LangGraph BaseStore.

Files are stored as LangGraph store Items with ``(thread_id, "mambo_fs")``
namespace for automatic session isolation.  The ``thread_id`` is locked in
at construction — no per-call parameter needed.
"""

from __future__ import annotations

import asyncio
import base64
import fnmatch
import threading
from typing import Any

from langchain_core.tools import StructuredTool
from langgraph.config import get_config, get_store
from langgraph.store.base import BaseStore, SearchItem
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
    ReadSummarizer,
    ToolTimeouts,
    UploadFileResult,
    WriteResult,
    _get_file_type,
    _get_mime_type,
)
from mambo_agents.backends.schemas import BackendError, ErrorCode, VirtualPath, human_size
from mambo_agents.backends.utils import (
    detect_trailing_newline_mismatch,
    format_with_line_numbers,
)

# ---------------------------------------------------------------------------
# Namespace constants
# ---------------------------------------------------------------------------

_MAMBO_FS_SUFFIX = "mambo_fs"
"""Store namespace suffix for file-system data under per-thread prefix."""


# ---------------------------------------------------------------------------
# StoreBackend
# ---------------------------------------------------------------------------


class StoreBackend(BackendProtocol):
    """File storage backed by LangGraph ``BaseStore`` with session isolation.

    Each thread (session) gets its own namespace ``(thread_id, "mambo_fs")``,
    so files from different conversations never leak.

    Parameters:
        thread_id: Session identifier.  When ``None`` (default), resolved
            automatically from the graph execution config.  Set explicitly for
            graph-outside / test usage.
        store: Optional ``BaseStore`` to use directly.  When ``None``,
            obtained at call-time via ``get_store()``, or a lazy
            ``InMemoryStore`` is created as fallback.
        initial_files: ``{path: content_str}`` mapping pre-populated on the
            very first access for each thread.
    """

    def __init__(
        self,
        *,
        thread_id: str | None = None,
        store: BaseStore | None = None,
        initial_files: dict[str, str] | None = None,
        max_read_chars: int = 100_000,
        max_grep_matches: int = 1000,
        summarizer: "ReadSummarizer | None" = None,
        tool_timeouts: ToolTimeouts | None = None,
    ) -> None:
        super().__init__(
            max_read_chars=max_read_chars,
            max_grep_matches=max_grep_matches,
            summarizer=summarizer,
            tool_timeouts=tool_timeouts,
        )
        self._thread_id = thread_id
        self._store = store
        self._lock = threading.RLock()

        # --- Immutable template — injected on first access per thread ---
        self._initial_files: dict[str, dict[str, str]] = {}
        if initial_files:
            for raw_path, content_ in initial_files.items():
                vp = VirtualPath(raw_path)
                if vp.value.endswith("/"):
                    raise BackendError(
                        code=ErrorCode.INVALID,
                        message=f"initial_files 路径不能以 '/' 结尾: {raw_path!r}",
                    )
                encoding = "base64" if _get_file_type(vp.normalized) != "text" else "utf-8"
                self._initial_files[vp.normalized] = {"content": content_, "encoding": encoding}

        # Track which threads have received initial_files (thread-safe)
        self._initialized_threads: set[str] = set()

    # ------------------------------------------------------------------
    # Store / namespace / thread_id resolution
    # ------------------------------------------------------------------

    def _get_store(self) -> BaseStore:
        """Return the store instance.

        Priority: (1) explicit ``store`` injected at construction,
        (2) ``get_store()`` from graph execution context.

        A ``RuntimeError`` from ``get_store()`` (not inside a runnable context)
        is handled by falling back to ``InMemoryStore``.

        If ``get_store()`` returns ``None`` (which means the graph was
        compiled without a ``store=``), a clear error is raised — this is a
        configuration mistake that must be fixed by the caller.
        """
        if self._store is not None:
            return self._store
        try:
            store_result = get_store()
        except RuntimeError:
            # Not inside a graph runnable context — use InMemoryStore as fallback
            from langgraph.store.memory import InMemoryStore
            self._store = InMemoryStore()
            return self._store

        if store_result is None:
            raise RuntimeError(
                "StoreBackend requires a LangGraph store, but get_store() returned None. "
                "The graph was likely compiled without a store= parameter. "
                "Fix: pass store= when constructing StoreBackend, "
                "or pass store= to graph.compile()."
            )
        return store_result

    @staticmethod
    def _get_namespace(thread_id: str) -> tuple[str, str]:
        """Build the per-thread namespace tuple."""
        return (thread_id, _MAMBO_FS_SUFFIX)

    def _resolve_thread_id(self) -> str:
        """Resolve thread_id in priority order:

        1. Explicit value from constructor.
        2. Implicit from ``config["configurable"]["thread_id"]`` (graph context).
        3. Fallback ``"__default__"``.
        """
        if self._thread_id is not None:
            return self._thread_id
        try:
            config = get_config()
            return config["configurable"].get("thread_id", "__default__")
        except RuntimeError:
            return "__default__"

    # ------------------------------------------------------------------
    # File read / write helpers
    # ------------------------------------------------------------------

    def _inject_initial_files(self, store: BaseStore, namespace: tuple[str, str]) -> None:
        """On first access for a thread, inject ``_initial_files`` if the
        namespace is empty."""
        if not self._initial_files:
            return
        existing = store.search(namespace, limit=1)
        if existing:
            return
        for path, value in self._initial_files.items():
            store.put(namespace, path, value)

    def _search_store_paginated(
        self,
        store: BaseStore,
        namespace: tuple[str, ...],
    ) -> list[SearchItem]:
        """Search store with automatic pagination to retrieve all items."""
        all_items: list[SearchItem] = []
        offset = 0
        page_size = 100
        while True:
            page = store.search(namespace, limit=page_size, offset=offset)
            if not page:
                break
            all_items.extend(page)
            if len(page) < page_size:
                break
            offset += page_size
        return all_items

    def _get_all_files(self, thread_id: str) -> dict[str, dict[str, str]]:
        """Return ``{path: {content, encoding}}`` for all files in *thread_id*."""
        store = self._get_store()
        namespace = self._get_namespace(thread_id)

        with self._lock:
            if thread_id not in self._initialized_threads:
                self._inject_initial_files(store, namespace)
                self._initialized_threads.add(thread_id)

            items = self._search_store_paginated(store, namespace)
            files: dict[str, dict[str, str]] = {}
            for item in items:
                files[item.key] = {"content": item.value.get("content", ""),
                                   "encoding": item.value.get("encoding", "utf-8")}
        return files

    def _get_file(self, thread_id: str, file_path: str) -> dict[str, str] | None:
        """Return ``{content, encoding}`` for a single file, or ``None``."""
        store = self._get_store()
        namespace = self._get_namespace(thread_id)

        with self._lock:
            if thread_id not in self._initialized_threads:
                self._inject_initial_files(store, namespace)
                self._initialized_threads.add(thread_id)

            item = store.get(namespace, file_path)
        if item is None:
            return None
        return {"content": item.value.get("content", ""),
                "encoding": item.value.get("encoding", "utf-8")}

    def _put_file(self, thread_id: str, file_path: str, content: str, encoding: str) -> None:
        """Store a single file."""
        store = self._get_store()
        namespace = self._get_namespace(thread_id)
        with self._lock:
            store.put(namespace, file_path, {"content": content, "encoding": encoding})

    # ------------------------------------------------------------------
    # Async file read / write helpers
    # ------------------------------------------------------------------

    async def _aget_all_files(self, thread_id: str) -> dict[str, dict[str, str]]:
        """Async: return ``{path: {content, encoding}}`` for all files in *thread_id*."""
        store = self._get_store()
        namespace = self._get_namespace(thread_id)

        with self._lock:
            if thread_id not in self._initialized_threads:
                self._inject_initial_files(store, namespace)
                self._initialized_threads.add(thread_id)

            items = await self._asearch_store_paginated(store, namespace)
            files: dict[str, dict[str, str]] = {}
            for item in items:
                files[item.key] = {"content": item.value.get("content", ""),
                                   "encoding": item.value.get("encoding", "utf-8")}
        return files

    async def _aget_file(self, thread_id: str, file_path: str) -> dict[str, str] | None:
        """Async: return ``{content, encoding}`` for a single file, or ``None``."""
        store = self._get_store()
        namespace = self._get_namespace(thread_id)

        with self._lock:
            if thread_id not in self._initialized_threads:
                self._inject_initial_files(store, namespace)
                self._initialized_threads.add(thread_id)

            item = await store.aget(namespace, file_path)
        if item is None:
            return None
        return {"content": item.value.get("content", ""),
                "encoding": item.value.get("encoding", "utf-8")}

    async def _aput_file(self, thread_id: str, file_path: str, content: str, encoding: str) -> None:
        """Async: store a single file."""
        store = self._get_store()
        namespace = self._get_namespace(thread_id)
        with self._lock:
            await store.aput(namespace, file_path, {"content": content, "encoding": encoding})

    async def _asearch_store_paginated(
        self,
        store: BaseStore,
        namespace: tuple[str, ...],
    ) -> list[SearchItem]:
        """Async: search store with automatic pagination to retrieve all items."""
        all_items: list[SearchItem] = []
        offset = 0
        page_size = 100
        while True:
            page = await store.asearch(namespace, limit=page_size, offset=offset)
            if not page:
                break
            all_items.extend(page)
            if len(page) < page_size:
                break
            offset += page_size
        return all_items

    # ------------------------------------------------------------------
    # tools — extra (core tools are built by middleware)
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
                    path=(VirtualPath, Field(description="Root directory to display")),
                    depth=(int, Field(default=3, description="Maximum recursion depth")),
                ),
                func=lambda **kwargs: self.tree(**kwargs),
                coroutine=lambda **kwargs: self.atree(**kwargs),
            ),
        ]

    # ------------------------------------------------------------------
    # Core file operations — use _resolve_thread_id() everywhere
    # ------------------------------------------------------------------

    def ls(self, path: VirtualPath) -> LsResult:
        normalized = path.normalized + "/"
        tid = self._resolve_thread_id()
        files = self._get_all_files(tid)

        if path.normalized in files:
            return LsResult(
                error=BackendError(code=ErrorCode.NOT_DIR, path=path, message="目标是文件，不是目录")
            )

        infos: list[FileInfo] = []
        subdirs: set[str] = set()
        for fpath, fd in files.items():
            if not fpath.startswith(normalized):
                continue
            relative = fpath[len(normalized):]
            if "/" in relative:
                subdirs.add(normalized + relative.split("/")[0])
            else:
                infos.append(FileInfo(path=VirtualPath(fpath), is_dir=False, size=len(fd["content"])))

        if not infos and not subdirs:
            if path.normalized == self.workspace_root.normalized:
                return LsResult(entries=[])
            return LsResult(error=BackendError(code=ErrorCode.NOT_FOUND, path=path, message="路径不存在"))

        for sd in sorted(subdirs):
            infos.append(FileInfo(path=VirtualPath(sd), is_dir=True, size=0))

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
            return ReadResult(error=BackendError(code=ErrorCode.IS_DIR, path=file_path, message="目标是目录"))
        tid = self._resolve_thread_id()
        fd = self._get_file(tid, file_path.normalized)
        if fd is None:
            return ReadResult(error=BackendError(code=ErrorCode.NOT_FOUND, path=file_path, message="文件不存在"))

        content = fd["content"]
        encoding = fd["encoding"]

        if encoding == "base64":
            file_type = _get_file_type(file_path.normalized)
            return ReadResult(
                content=content, total_lines=1, encoding="base64",
                file_type=file_type, mime_type=_get_mime_type(file_path.normalized),
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
            if include_line_numbers else raw_slice
        )
        return ReadResult(content=content, total_lines=total, encoding="utf-8")

    def write(
        self, file_path: VirtualPath, content: str, overwrite: bool = False,
    ) -> WriteResult:
        if file_path.value.endswith("/"):
            return WriteResult(error=BackendError(code=ErrorCode.IS_DIR, path=file_path, message="目标是目录"))
        tid = self._resolve_thread_id()
        files = self._get_all_files(tid)

        prefix = file_path.normalized + "/"
        for fpath in files:
            if fpath.startswith(prefix):
                return WriteResult(
                    error=BackendError(code=ErrorCode.IS_DIR, path=file_path, message="目标是目录，无法写入"),
                )
        if file_path.normalized in files and not overwrite:
            return WriteResult(
                error=BackendError(code=ErrorCode.ALREADY_EXISTS, path=file_path, message="文件已存在，请用 edit() 修改或用 overwrite=True 覆盖"),
            )
        encoding = "base64" if _get_file_type(file_path.normalized) != "text" else "utf-8"
        self._put_file(tid, file_path.normalized, content, encoding)
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
            return EditResult(error=BackendError(code=ErrorCode.IS_DIR, path=file_path, message="目标是目录"))
        if not old_str:
            return EditResult(error=BackendError(code=ErrorCode.INVALID, message="old_str 不能为空"))
        tid = self._resolve_thread_id()
        files = self._get_all_files(tid)

        prefix = file_path.normalized + "/"
        for fpath in files:
            if fpath.startswith(prefix):
                return EditResult(
                    error=BackendError(code=ErrorCode.IS_DIR, path=file_path, message="目标是目录，无法编辑"),
                )
        existing_fd = files.get(file_path.normalized)
        if existing_fd is None:
            return EditResult(
                error=BackendError(code=ErrorCode.NOT_FOUND, path=file_path, message="文件不存在，请用 write() 创建新文件"),
            )
        if existing_fd["encoding"] == "base64":
            return EditResult(
                error=BackendError(code=ErrorCode.INVALID, path=file_path, message="文件是二进制格式，请用 write() 覆盖"),
            )

        existing_content = existing_fd["content"]
        occurrences = existing_content.count(old_str)

        if occurrences == 0:
            mismatch = detect_trailing_newline_mismatch(old_str, existing_content)
            if mismatch is not None:
                return mismatch
            return EditResult(
                error=BackendError(code=ErrorCode.OLD_STR_NOT_FOUND, path=file_path, message="未找到要替换的文本"),
            )

        if occurrences > 1 and not replace_all:
            return EditResult(
                error=BackendError(code=ErrorCode.MULTI_OCCURRENCES, path=file_path, message=f"匹配到 {occurrences} 处，请用 replace_all=True 或提供更精确的上下文"),
            )

        new_content = existing_content.replace(old_str, new_str)
        self._put_file(tid, file_path.normalized, new_content, existing_fd["encoding"])
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
            return GrepResult(error=BackendError(code=ErrorCode.INVALID, message="搜索模式不能为空"))
        if regex:
            import re as _re
            try:
                _re.compile(pattern)
            except _re.error as e:
                return GrepResult(error=BackendError(code=ErrorCode.INVALID, message=f"无效正则: {e}"))
        tid = self._resolve_thread_id()
        files = self._get_all_files(tid)
        raw_matches = _grep_in_memory(files, pattern, path.normalized, glob, regex, self._max_grep_matches)
        return self._apply_grep_limit(raw_matches, offset, limit)

    def glob(self, pattern: str, path: VirtualPath) -> GlobResult:
        tid = self._resolve_thread_id()
        files = self._get_all_files(tid)
        return _glob_in_memory(files, pattern, path.normalized)

    # ------------------------------------------------------------------
    # Extra operations
    # ------------------------------------------------------------------

    def tree(self, path: VirtualPath, depth: int = 3) -> str:
        if depth < 1:
            return f"Invalid depth value: {depth}. Depth must be a positive integer (>= 1)."
        if isinstance(path, str):
            path = VirtualPath(path)
        entries = _collect_tree_entries(self, path, depth)
        return _format_tree(entries)

    async def atree(self, path: VirtualPath, depth: int = 3) -> str:
        if isinstance(path, str):
            path = VirtualPath(path)
        return await asyncio.to_thread(self.tree, path, depth)

    # ------------------------------------------------------------------
    # Developer API — upload / download (no thread_id param — locked at init)
    # ------------------------------------------------------------------

    def upload_files(
        self,
        files: list[tuple[VirtualPath, bytes]],
    ) -> list[UploadFileResult]:
        tid = self._resolve_thread_id()
        results: list[UploadFileResult] = []
        for path, raw_content in files:
            try:
                text = raw_content.decode("utf-8")
                encoding = "utf-8"
            except UnicodeDecodeError:
                text = base64.b64encode(raw_content).decode("ascii")
                encoding = "base64"
            self._put_file(tid, path.normalized, text, encoding)
            results.append(UploadFileResult(path=path.normalized, error=None))
        return results

    def download_files(
        self,
        paths: list[VirtualPath],
    ) -> list[DownloadFileResult]:
        tid = self._resolve_thread_id()
        results: list[DownloadFileResult] = []
        for path in paths:
            fd = self._get_file(tid, path.normalized)
            if fd is None:
                results.append(
                    DownloadFileResult(
                        path=path.normalized, content=None,
                        error=BackendError(code=ErrorCode.NOT_FOUND, path=path, message="文件不存在"),
                    )
                )
                continue
            content = fd["content"]
            encoding = fd["encoding"]
            if encoding == "utf-8":
                content_bytes = content.encode("utf-8")
            else:
                content_bytes = base64.standard_b64decode(content)
            results.append(
                DownloadFileResult(path=path.normalized, content=content_bytes, error=None)
            )
        return results


# ============================================================================
# Internal helpers — in-memory grep / glob / tree logic
# ============================================================================


def _grep_in_memory(
    files: dict[str, dict[str, str]],
    pattern: str,
    path: str = "/",
    file_glob: str | None = None,
    regex: bool = False,
    max_matches: int = 1000,
) -> list[GrepMatch]:
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
                matches.append(GrepMatch(path=VirtualPath(fpath), line=li, text=line))

    return matches


def _glob_in_memory(
    files: dict[str, dict[str, str]],
    pattern: str,
    path: str = "/",
) -> GlobResult:
    path_prefix = path.rstrip("/") if path != "/" else ""

    results: list[FileInfo] = []
    dirs_seen: set[str] = set()

    for fpath in sorted(files):
        if path_prefix and fpath != path_prefix and not fpath.startswith(path_prefix + "/"):
            continue

        parent = fpath.rpartition("/")[0]
        while parent and parent != path_prefix and parent not in dirs_seen:
            dirs_seen.add(parent)
            parent = parent.rpartition("/")[0]

        if not fnmatch.fnmatch(fpath, pattern):
            continue
        results.append(FileInfo(path=VirtualPath(fpath), is_dir=False, size=len(files[fpath].get("content", ""))))

    for dpath in sorted(dirs_seen):
        if not fnmatch.fnmatch(dpath, pattern):
            continue
        results.append(FileInfo(path=VirtualPath(dpath), is_dir=True, size=0))

    return GlobResult(matches=results)


def _collect_tree_entries(
    backend: StoreBackend,
    path: VirtualPath,
    depth: int,
) -> list[tuple[str, bool, int]]:
    result = backend.ls(path)
    if result.error or not result.entries:
        return []

    entries: list[tuple[str, bool, int]] = []
    for fi in result.entries:
        if fi.is_dir and depth > 1:
            sub_path = fi.path.value if fi.path.value.endswith("/") else fi.path.value + "/"
            entries.extend(
                _collect_tree_entries(backend, VirtualPath(sub_path), depth - 1)
            )
    for fi in result.entries:
        entries.append((str(fi.path), fi.is_dir, fi.size))
    return entries


def _format_tree(
    entries: list[tuple[str, bool, int]],
    prefix: str = "",
) -> str:
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
