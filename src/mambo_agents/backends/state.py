"""StateBackend – ephemeral in-memory file storage."""

from __future__ import annotations

import asyncio
import base64
import fnmatch

from langchain_core.tools import StructuredTool
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LINE_NUMBER_WIDTH = 6
MAX_LINE_LENGTH = 5000


# ---------------------------------------------------------------------------
# StateBackend
# ---------------------------------------------------------------------------


class StateBackend(BackendProtocol):
    """In-memory file storage backed by a ``dict``.

    Files are keyed by absolute path.  Each value is::

        {"content": str, "encoding": "utf-8" | "base64"}

    Extra tool provided: ``tree`` — displays directory structure.

    Parameters:
        initial_files: Optional ``{path: content_str}`` mapping to
            pre-populate the store.
    """

    def __init__(self, initial_files: dict[str, str] | None = None) -> None:
        self._files: dict[str, dict[str, str]] = {}
        if initial_files:
            for path, content in initial_files.items():
                encoding = "base64" if _get_file_type(path) != "text" else "utf-8"
                self._files[path] = {"content": content, "encoding": encoding}

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

        for fpath, fd in self._files.items():
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
        fd = self._files.get(file_path)
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
            content=_format_with_line_numbers(
                "\n".join(sliced), start_line=offset + 1,
            ),
            total_lines=total,
            encoding="utf-8",
        )

    def write(
        self, file_path: str, content: str, overwrite: bool = False,
    ) -> WriteResult:
        if file_path in self._files and not overwrite:
            return WriteResult(
                error=(
                    f"Cannot write '{file_path}': file already exists. "
                    "Read the file and use edit() to modify it, "
                    "or use overwrite=True to replace the file."
                ),
            )
        encoding = "base64" if _get_file_type(file_path) != "text" else "utf-8"
        self._files[file_path] = {"content": content, "encoding": encoding}
        return WriteResult(path=file_path)

    def edit(
        self,
        file_path: str,
        old_str: str,
        new_str: str,
        *,
        replace_all: bool = False,
    ) -> EditResult:
        existing_fd = self._files.get(file_path)
        if existing_fd is None:
            return EditResult(
                error=(
                    f"Cannot edit '{file_path}': file not found. "
                    "To create a new file, use write()."
                ),
            )
        existing_content = existing_fd.get("content", "")

        occurrences = existing_content.count(old_str)

        if occurrences == 0:
            # Detect trailing-newline mismatch: model adds \n to old_str
            # but the file does not end with a newline.
            if (
                old_str.endswith("\n")
                and len(old_str) > 1
                and existing_content.endswith(old_str.removesuffix("\n"))
            ):
                stripped = old_str.removesuffix("\n")
                stripped_count = existing_content.count(stripped)
                if stripped_count == 1:
                    return EditResult(
                        error=(
                            "old_str ends with a newline, but the file does not "
                            "end with a newline. Retry with the trailing newline "
                            "removed from old_str (and from new_str if it also "
                            "ends with a newline)."
                        ),
                    )
                return EditResult(
                    error=(
                        f"old_str ends with a newline, but the file does not "
                        f"end with a newline. With the trailing newline removed, "
                        f"old_str would appear {stripped_count} times. "
                        f"Retry with the trailing newline removed and add "
                        f"surrounding context so the match is unique."
                    ),
                )
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

        self._files[file_path] = {
            "content": existing_content.replace(old_str, new_str),
            "encoding": existing_fd.get("encoding", "utf-8"),
        }
        return EditResult(path=file_path, occurrences=occurrences)

    def grep(
        self,
        pattern: str,
        path: str = "/",
        glob: str | None = None,
    ) -> GrepResult:
        return _grep_in_memory(self._files, pattern, path, glob)

    def glob(self, pattern: str, path: str = "/") -> GlobResult:
        return _glob_in_memory(self._files, pattern, path)

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
    # Developer API
    # ------------------------------------------------------------------

    def upload_files(
        self, files: list[tuple[str, bytes]]
    ) -> list[UploadFileResult]:
        results: list[UploadFileResult] = []
        for path, raw_content in files:
            try:
                text = raw_content.decode("utf-8")
                encoding = "utf-8"
            except UnicodeDecodeError:
                text = base64.b64encode(raw_content).decode("ascii")
                encoding = "base64"
            self._files[path] = {"content": text, "encoding": encoding}
            results.append(UploadFileResult(path=path, error=None))
        return results

    def download_files(
        self, paths: list[str]
    ) -> list[DownloadFileResult]:
        results: list[DownloadFileResult] = []
        for path in paths:
            fd = self._files.get(path)
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


def _format_with_line_numbers(content: str, start_line: int = 1) -> str:
    """Format content with line numbers (``cat -n`` style)."""
    lines = content.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]

    result: list[str] = []
    for i, line in enumerate(lines):
        num = i + start_line
        if len(line) <= MAX_LINE_LENGTH:
            result.append(f"{num:{LINE_NUMBER_WIDTH}d}\t{line}")
        else:
            chunk_count = (len(line) + MAX_LINE_LENGTH - 1) // MAX_LINE_LENGTH
            for ci in range(chunk_count):
                chunk = line[ci * MAX_LINE_LENGTH:(ci + 1) * MAX_LINE_LENGTH]
                marker = f"{num}.{ci}" if ci > 0 else str(num)
                result.append(f"{marker:>{LINE_NUMBER_WIDTH}}\t{chunk}")
    return "\n".join(result)


def _grep_in_memory(
    files: dict[str, dict[str, str]],
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
    files: dict[str, dict[str, str]],
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
            size_str = f" ({_human_size(size)})"
        lines.append(f"{prefix}{connector}{name}{size_str}")
    return "\n".join(lines)


def _human_size(size: int) -> str:
    """Format *size* as a human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size} {unit}"
        size //= 1024
    return f"{size} TB"
