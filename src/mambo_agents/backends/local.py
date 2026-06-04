"""LocalBackend – real filesystem backend with shell execution support.

Provides direct file-system access plus extra tools: ``tree`` (directory
structure), ``delete`` (remove files/directories), and ``execute`` (run
shell commands).  Supports both Windows and Linux.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

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
    ReadSummarizer,
    UploadFileResult,
    WriteResult,
    _get_file_type,
    _get_mime_type,
)
from mambo_agents.backends.utils import (
    TreeEntry,
    check_path_allowed,
    detect_trailing_newline_mismatch,
    format_tree_entries,
    format_with_line_numbers,
    human_size,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_EXECUTE_TIMEOUT = 120
_MAX_OUTPUT_BYTES = 100_000
_DEFAULT_MAX_FILE_SIZE_MB = 10


# ---------------------------------------------------------------------------
# LocalBackend
# ---------------------------------------------------------------------------


class LocalBackend(BackendProtocol):
    """Real filesystem backend with shell command execution.

    Operates directly on the local filesystem.  Provides the six core
    file operations plus three extra tools:

    - ``tree``  — visual directory tree
    - ``delete`` — remove files and directories
    - ``execute`` — run arbitrary shell commands (Win: ``cmd /c``, Linux: ``sh -c``)

    !!! warning
        This backend grants the agent direct filesystem access AND
        arbitrary shell execution.  Use ``BackendToolsMiddleware``
        with Human-in-the-Loop review.

    Parameters:
        root_dir: Working directory for both file ops and shell commands.
            Defaults to ``os.getcwd()``.
        timeout: Default timeout in seconds for shell execution (default 120).
        max_output_bytes: Max bytes to capture from command output.
        env: Environment variables for shell commands.
            If ``None``, starts with an empty environment (unless ``inherit_env=True``).
        inherit_env: Inherit ``os.environ`` into the command environment.
        enable_execute: Whether to provide the ``execute`` tool to the
            LLM, allowing it to run arbitrary shell commands.
            Default ``False``. Set to ``True`` to enable shell execution.
        max_file_size_mb: Maximum file size in MB for ``grep`` search.
            Files exceeding this limit are skipped.  Default 10 MB.
        edit_whitelist: Virtual path prefixes allowed for edit/write/delete.
            Mutually exclusive with *edit_blacklist*.  When set, any
            path not matching a prefix is rejected.
        edit_blacklist: Virtual path prefixes forbidden for edit/write/delete.
            Mutually exclusive with *edit_whitelist*.  When set, any
            path matching a prefix is rejected.
        ignore_dirs: Virtual directory paths whose children are hidden
            in ``tree()`` output.  The directory itself is shown with
            an ``/(ignore)`` marker but its content is not expanded.
    """

    def __init__(
        self,
        root_dir: str | Path | None = None,
        *,
        timeout: int = _DEFAULT_EXECUTE_TIMEOUT,
        max_output_bytes: int = _MAX_OUTPUT_BYTES,
        env: dict[str, str] | None = None,
        inherit_env: bool = False,
        enable_execute: bool = False,
        max_file_size_mb: int = _DEFAULT_MAX_FILE_SIZE_MB,
        edit_whitelist: frozenset[str] | None = None,
        edit_blacklist: frozenset[str] | None = None,
        ignore_dirs: frozenset[str] | None = None,
        max_read_chars: int = 100_000,
        summarizer: "ReadSummarizer | None" = None,
    ) -> None:
        super().__init__(max_read_chars=max_read_chars, summarizer=summarizer)
        if timeout <= 0:
            msg = f"timeout must be positive, got {timeout}"
            raise ValueError(msg)

        if edit_whitelist is not None and edit_blacklist is not None:
            raise ValueError(
                "edit_whitelist and edit_blacklist are mutually exclusive. "
                "Provide at most one of them."
            )

        self._cwd = Path(root_dir).resolve() if root_dir else Path.cwd()
        self._default_timeout = timeout
        self._max_output_bytes = max_output_bytes
        self._enable_execute = enable_execute
        self._max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self._edit_whitelist = edit_whitelist or frozenset()
        self._edit_blacklist = edit_blacklist or frozenset()
        self._ignore_dirs = ignore_dirs or frozenset()

        # Build environment
        if inherit_env:
            self._env = os.environ.copy()
            if env is not None:
                self._env.update(env)
        else:
            self._env = env if env is not None else {}

    # ------------------------------------------------------------------
    # tools — extra tools only (core tools are built by middleware)
    # ------------------------------------------------------------------

    @property
    def tools(self) -> list[StructuredTool]:
        tools: list[StructuredTool] = [
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
            StructuredTool(
                name="delete",
                description=(
                    "Delete a file or directory. "
                    "For directories, removes the directory and all its contents recursively."
                ),
                args_schema=create_model(
                    "DeleteSchema",
                    path=(str, Field(description="Absolute path to delete")),
                ),
                func=lambda **kwargs: self.delete(**kwargs),
                coroutine=lambda **kwargs: self.adelete(**kwargs),
            ),
        ]

        if self._enable_execute:
            tools.append(
                StructuredTool(
                    name="execute",
                    description=(
                        "Execute a shell command on the local system. "
                        "On Windows, commands run via cmd /c. "
                        "On Linux/macOS, commands run via sh -c. "
                        "Returns combined stdout and stderr output."
                    ),
                    args_schema=create_model(
                        "ExecuteSchema",
                        command=(str, Field(description="Shell command to execute")),
                        timeout=(int | None, Field(default=None, description="Optional timeout in seconds")),
                    ),
                    func=lambda **kwargs: self.execute(**kwargs),
                    coroutine=lambda **kwargs: self.aexecute(**kwargs),
                )
            )

        return tools

    @property
    def description(self) -> str:
        desc = (
            "Local file system backend with shell command execution support "
            f"(working directory: {self._cwd})"
        )
        if not self._enable_execute:
            desc += " [shell execution disabled]"
        return desc

    # ------------------------------------------------------------------
    # Core file operations
    # ------------------------------------------------------------------

    def _resolve(self, path: str) -> Path:
        """Resolve a virtual absolute path to a real filesystem path under ``_cwd``.

        Uses ``PurePosixPath`` to parse the virtual path consistently
        across platforms (Windows, Linux, macOS).  The virtual path
        scheme uses ``"/"`` separators and treats ``"/"`` as the root
        of *backing* directory (``_cwd``).
        """
        from pathlib import PurePosixPath

        pp = PurePosixPath(path)
        if pp.is_absolute():
            # "/foo/bar" → parts = ("/", "foo", "bar") → skip root → ("foo", "bar")
            relative_parts = pp.parts[1:]
        else:
            relative_parts = pp.parts

        if not relative_parts:
            return self._cwd

        return (self._cwd.joinpath(*relative_parts)).resolve()

    def _check_edit_allowed(self, path: str) -> bool:
        """Check whether *path* is allowed for edit/write/delete.

        Delegates to :func:`~mambo_agents.backends.utils.check_path_allowed`
        with this backend's whitelist / blacklist.
        """
        whitelist = self._edit_whitelist or None
        blacklist = self._edit_blacklist or None
        return check_path_allowed(path, whitelist=whitelist, blacklist=blacklist)

    def ls(self, path: str) -> LsResult:
        resolved = self._resolve(path)
        try:
            if not resolved.exists() or not resolved.is_dir():
                return LsResult(error=f"Directory '{path}' not found")
        except OSError as e:
            return LsResult(error=f"Cannot access '{path}': {e}")

        infos: list[FileInfo] = []
        errors: list[str] = []
        try:
            for child in sorted(resolved.iterdir(), key=lambda c: (not c.is_dir(), c.name)):
                try:
                    st = child.stat()
                    modified_at = ""
                    ts = getattr(st, "st_mtime", 0)
                    if ts:
                        modified_at = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
                    if child.is_dir():
                        infos.append(FileInfo(
                            path="/" + str(child.relative_to(self._cwd)).replace("\\", "/") + "/",
                            is_dir=True,
                            size=0,
                            modified_at=modified_at,
                        ))
                    else:
                        infos.append(FileInfo(
                            path="/" + str(child.relative_to(self._cwd)).replace("\\", "/"),
                            is_dir=False,
                            size=st.st_size,
                            modified_at=modified_at,
                        ))
                except OSError as e:
                    errors.append(f"Cannot stat '{child.name}': {e}")
        except OSError as e:
            errors.append(f"Listing aborted: {e}")

        error_msg = "\n".join(errors) if errors else None
        return LsResult(error=error_msg, entries=infos)

    def read_raw(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
        include_line_numbers: bool = False,
    ) -> ReadResult:
        resolved = self._resolve(file_path)
        try:
            if not resolved.exists():
                return ReadResult(error=f"File '{file_path}' not found")
            if resolved.is_dir():
                return ReadResult(error=f"'{file_path}' is a directory")
        except OSError as e:
            return ReadResult(error=f"Error accessing '{file_path}': {e}")

        _O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)

        if _get_file_type(file_path) != "text":
            try:
                fd = os.open(resolved, os.O_RDONLY | _O_NOFOLLOW)
                with os.fdopen(fd, "rb") as f:
                    raw = f.read()
            except OSError as e:
                return ReadResult(error=f"Error reading '{file_path}': {e}")
            encoded = base64.b64encode(raw).decode("ascii")
            return ReadResult(
                content=encoded,
                total_lines=1,
                encoding="base64",
                file_type=_get_file_type(file_path),
                mime_type=_get_mime_type(file_path),
            )

        # Text file: attempt UTF-8 read, fallback to base64
        try:
            fd = os.open(resolved, os.O_RDONLY | _O_NOFOLLOW)
            with os.fdopen(fd, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            try:
                fd = os.open(resolved, os.O_RDONLY | _O_NOFOLLOW)
                with os.fdopen(fd, "rb") as f:
                    raw = f.read()
            except OSError as e:
                return ReadResult(error=f"Error reading '{file_path}': {e}")
            encoded = base64.b64encode(raw).decode("ascii")
            return ReadResult(
                content=encoded,
                total_lines=1,
                encoding="base64",
                file_type=_get_file_type(file_path),
                mime_type=_get_mime_type(file_path),
            )
        except OSError as e:
            return ReadResult(error=f"Error reading '{file_path}': {e}")

        lines = content.splitlines(keepends=True)
        total = len(lines)

        if offset >= total:
            return ReadResult(
                error=f"Line offset {offset} exceeds file length ({total} lines)",
            )

        sliced = lines[offset : offset + limit]
        raw_slice = "".join(sliced)
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
        self, file_path: str, content: str, overwrite: bool = False,
    ) -> WriteResult:
        if not self._check_edit_allowed(file_path):
            return WriteResult(
                error=(
                    f"Path '{file_path}' is not allowed for write. "
                    "Check edit_whitelist / edit_blacklist."
                ),
            )
        resolved = self._resolve(file_path)
        if resolved.exists() and not overwrite:
            return WriteResult(
                error=(
                    f"Cannot write '{file_path}': file already exists. "
                    "Read the file and use edit() to modify it, "
                    "or use overwrite=True to replace the file."
                ),
            )

        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            _O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | _O_NOFOLLOW
            fd = os.open(resolved, flags, 0o644)
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
                f.write(content)
        except OSError as e:
            return WriteResult(error=f"Error writing '{file_path}': {e}")

        return WriteResult(path=file_path)

    def edit(
        self,
        file_path: str,
        old_str: str,
        new_str: str,
        *,
        replace_all: bool = False,
    ) -> EditResult:
        if not self._check_edit_allowed(file_path):
            return EditResult(
                error=(
                    f"Path '{file_path}' is not allowed for edit. "
                    "Check edit_whitelist / edit_blacklist."
                ),
            )
        resolved = self._resolve(file_path)
        try:
            if not resolved.exists():
                return EditResult(
                    error=(
                        f"Cannot edit '{file_path}': file not found. "
                        "To create a new file, use write()."
                    ),
                )
        except OSError as e:
            return EditResult(error=f"Error accessing '{file_path}': {e}")

        _O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(resolved, os.O_RDONLY | _O_NOFOLLOW)
            with os.fdopen(fd, "r", encoding="utf-8") as f:
                content = f.read()
        except OSError as e:
            return EditResult(error=f"Error reading '{file_path}': {e}")

        # Normalize line endings in old_str / new_str so that LLM-provided
        # CRLF strings match files that were read as LF by Python text mode.
        old_str = old_str.replace("\r\n", "\n").replace("\r", "\n")
        new_str = new_str.replace("\r\n", "\n").replace("\r", "\n")

        occurrences = content.count(old_str)

        if occurrences == 0:
            # Detect trailing-newline mismatch
            mismatch = detect_trailing_newline_mismatch(
                file_path, old_str, content,
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

        try:
            flags = os.O_WRONLY | os.O_TRUNC | _O_NOFOLLOW
            fd = os.open(resolved, flags)
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
                f.write(content.replace(old_str, new_str))
        except OSError as e:
            return EditResult(error=f"Error writing '{file_path}': {e}")

        return EditResult(path=file_path, occurrences=occurrences)

    # ------------------------------------------------------------------
    # grep — ripgrep-first with Python fallback and file-size guard
    # ------------------------------------------------------------------

    def grep(
        self,
        pattern: str,
        path: str = "/",
        glob: str | None = None,
    ) -> GrepResult:
        resolved = self._resolve(path)
        try:
            if not resolved.exists():
                return GrepResult(error=f"Path '{path}' not found")
        except OSError as e:
            return GrepResult(error=f"Error accessing '{path}': {e}")

        search_dir = resolved if resolved.is_dir() else resolved.parent

        # 1) Try ripgrep (orders of magnitude faster on large trees)
        results = self._ripgrep_grep(pattern, search_dir, glob)
        if results is not None:
            # ripgrep paths are physical → convert to virtual
            matches: list[GrepMatch] = []
            for fpath, items in results.items():
                try:
                    virt = "/" + str(Path(fpath).relative_to(self._cwd)).replace("\\", "/")
                except ValueError:
                    continue
                for li, text in items:
                    matches.append(GrepMatch(path=virt, line=li, text=text))
            return GrepResult(matches=matches)

        # 2) Python fallback with file-size guard
        import fnmatch as _fnmatch

        matches: list[GrepMatch] = []
        skipped: int = 0
        regex = re.compile(re.escape(pattern))

        try:
            for fp in search_dir.rglob("*"):
                try:
                    if not fp.is_file():
                        continue
                except OSError:
                    continue

                if glob and not _fnmatch.fnmatch(fp.name, glob):
                    continue
                if _get_file_type(fp.suffix) != "text":
                    continue

                # Skip files exceeding the size limit
                try:
                    if fp.stat().st_size > self._max_file_size_bytes:
                        skipped += 1
                        continue
                except OSError:
                    continue

                try:
                    lines = fp.read_text(encoding="utf-8").split("\n")
                except (UnicodeDecodeError, OSError):
                    continue

                for li, line in enumerate(lines, start=1):
                    if regex.search(line):
                        virt_path = "/" + str(fp.relative_to(self._cwd)).replace("\\", "/")
                        matches.append(GrepMatch(path=virt_path, line=li, text=line))
        except OSError as e:
            return GrepResult(
                error=f"Error during grep: {e}",
                matches=matches,
            )

        error_msg: str | None = None
        if skipped:
            error_msg = (
                f"Skipped {skipped} file(s) exceeding "
                f"{self._max_file_size_bytes // (1024 * 1024)} MB size limit"
            )

        return GrepResult(error=error_msg, matches=matches)

    # ------------------------------------------------------------------
    # ripgrep helper
    # ------------------------------------------------------------------

    def _ripgrep_grep(
        self,
        pattern: str,
        base_dir: Path,
        include_glob: str | None,
    ) -> dict[str, list[tuple[int, str]]] | None:
        """Search with ripgrep (literal mode, JSON output).

        Returns:
            Dict mapping physical file paths → list of (line, text), or
            ``None`` if ripgrep is unavailable or times out.
        """
        if not shutil.which("rg"):
            return None

        cmd = ["rg", "--json", "-F"]
        if include_glob:
            cmd.extend(["--glob", include_glob])
        cmd.extend(["--", pattern, str(base_dir)])

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError):
            return None

        results: dict[str, list[tuple[int, str]]] = {}
        for line in proc.stdout.splitlines():
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if data.get("type") != "match":
                continue
            pdata = data.get("data", {})
            ftext = pdata.get("path", {}).get("text")
            if not ftext:
                continue
            ln = pdata.get("line_number")
            lt = pdata.get("lines", {}).get("text", "").rstrip("\n")
            if ln is None:
                continue
            results.setdefault(ftext, []).append((int(ln), lt))

        return results

    def glob(self, pattern: str, path: str = "/") -> GlobResult:
        resolved = self._resolve(path)

        try:
            if not resolved.exists() or not resolved.is_dir():
                return GlobResult(error=f"Path '{path}' not found")
        except OSError as e:
            return GlobResult(error=f"Error accessing '{path}': {e}")

        matches: list[FileInfo] = []
        errors: list[str] = []
        try:
            for fp in resolved.rglob(pattern):
                try:
                    if not fp.is_file():
                        continue
                except OSError:
                    continue
                try:
                    virt_path = "/" + str(fp.relative_to(self._cwd)).replace("\\", "/")
                    st = fp.stat()
                except OSError as e:
                    errors.append(f"Cannot stat '{fp}': {e}")
                    continue
                matches.append(FileInfo(
                    path=virt_path,
                    is_dir=False,
                    size=st.st_size,
                ))
        except OSError as e:
            errors.append(f"Glob aborted: {e}")

        error_msg = "\n".join(errors) if errors else None
        return GlobResult(error=error_msg, matches=matches)

    # ------------------------------------------------------------------
    # Extra operations: tree, delete, execute
    # ------------------------------------------------------------------

    def tree(self, path: str = "/", depth: int = 3) -> str:
        """Render a directory tree.

        Args:
            path: Root directory to display (virtual path, default ``"/"``).
            depth: Maximum recursion depth (default 3).

        Returns:
            Formatted tree string.
        """
        resolved = self._resolve(path)
        if not resolved.exists():
            return f"Path '{path}' not found."
        if not resolved.is_dir():
            return f"Path '{path}' is not a directory."

        entries = _walk_tree(
            resolved,
            depth,
            cwd=self._cwd,
            ignore_dirs=self._ignore_dirs,
        )
        return format_tree_entries(entries)

    def delete(self, path: str) -> str:
        """Delete a file or directory.

        For directories, removes recursively (like ``rm -rf``).

        Args:
            path: Virtual path to delete.

        Returns:
            Success or error message.
        """
        if not self._check_edit_allowed(path):
            return (
                f"Error: Path '{path}' is not allowed for delete. "
                "Check edit_whitelist / edit_blacklist."
            )
        resolved = self._resolve(path)

        # Safety: refuse to delete the root_dir itself
        if resolved == self._cwd:
            return "Error: cannot delete root working directory."

        if not resolved.exists():
            return f"Error: path '{path}' does not exist."

        try:
            if resolved.is_dir():
                shutil.rmtree(resolved)
            else:
                resolved.unlink()
        except OSError as e:
            return f"Error deleting '{path}': {e}"

        return f"Deleted: {path}"

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> str:
        r"""Execute a shell command on the local system.

        On Windows, uses ``cmd /c``.  On Linux/macOS, uses ``sh -c``.

        Args:
            command: Shell command string to execute.
            timeout: Override the default timeout (seconds).

        Returns:
            Formatted output with stdout, stderr, and exit code.
        """
        if not command or not isinstance(command, str):
            return "Error: Command must be a non-empty string."

        effective_timeout = timeout if timeout is not None else self._default_timeout
        if effective_timeout <= 0:
            msg = f"timeout must be positive, got {effective_timeout}"
            raise ValueError(msg)

        # Build platform-specific shell invocation
        if sys.platform == "win32":
            cmd_list: list[str] = ["cmd", "/c", command]
        else:
            cmd_list = ["sh", "-c", command]

        try:
            result = subprocess.run(
                cmd_list,
                check=False,
                capture_output=True,
                stdin=subprocess.DEVNULL,
                text=True,
                timeout=effective_timeout,
                env=self._env if self._env else None,
                cwd=str(self._cwd),
            )

            output_parts: list[str] = []
            if result.stdout:
                output_parts.append(result.stdout.rstrip())
            if result.stderr:
                for line in result.stderr.strip().split("\n"):
                    output_parts.append(f"[stderr] {line}")

            output = "\n".join(output_parts) if output_parts else "<no output>"

            # Truncation
            if len(output) > self._max_output_bytes:
                output = output[: self._max_output_bytes]
                output += f"\n\n... Output truncated at {self._max_output_bytes} bytes."

            # Add exit code if non-zero
            if result.returncode != 0:
                output = f"{output.rstrip()}\n\nExit code: {result.returncode}"

            return output

        except subprocess.TimeoutExpired:
            return (
                f"Error: Command timed out after {effective_timeout} seconds. "
                "For long-running commands, re-run with a longer timeout."
            )

        except Exception as e:
            return f"Error executing command ({type(e).__name__}): {e}"

    # ------------------------------------------------------------------
    # Async extra operations
    # ------------------------------------------------------------------

    async def atree(self, path: str = "/", depth: int = 3) -> str:
        """Async: Render a directory tree."""
        return await asyncio.to_thread(self.tree, path, depth)

    async def adelete(self, path: str) -> str:
        """Async: Delete a file or directory."""
        return await asyncio.to_thread(self.delete, path)

    async def aexecute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> str:
        """Async: Execute a shell command on the local system."""
        return await asyncio.to_thread(self.execute, command, timeout=timeout)

    # ------------------------------------------------------------------
    # Developer API — bulk upload / download (raw bytes for filesystem)
    # ------------------------------------------------------------------

    def upload_files(
        self,
        files: list[tuple[str, bytes]],
    ) -> list[UploadFileResult]:
        """Upload multiple files as raw bytes directly to the filesystem.

        Overrides the base ``BackendProtocol.upload_files`` to write raw
        bytes to disk (instead of going through ``write()`` which would
        base64-encode binary files).

        Returns:
            Per-file results with ``path`` and optional ``error``.
        """
        results: list[UploadFileResult] = []
        for path, raw_content in files:
            try:
                resolved = self._resolve(path)
                resolved.parent.mkdir(parents=True, exist_ok=True)
                resolved.write_bytes(raw_content)
                results.append(UploadFileResult(path=path))
            except OSError as e:
                results.append(UploadFileResult(path=path, error=str(e)))
            except Exception as e:
                results.append(
                    UploadFileResult(
                        path=path, error=f"{type(e).__name__}: {e}"
                    )
                )
        return results

    def download_files(
        self,
        paths: list[str],
    ) -> list[DownloadFileResult]:
        """Download multiple files as raw bytes from the filesystem.

        Overrides the base ``BackendProtocol.download_files`` to read
        raw bytes directly (instead of going through ``read()`` and
        reverse-engineering the line-numbered format).

        Returns:
            Per-file results with ``path``, ``content`` (bytes or ``None``),
            and optional ``error``.
        """
        results: list[DownloadFileResult] = []
        for path in paths:
            try:
                resolved = self._resolve(path)
                if not resolved.exists():
                    results.append(
                        DownloadFileResult(
                            path=path, content=None, error="file_not_found"
                        )
                    )
                    continue
                if resolved.is_dir():
                    results.append(
                        DownloadFileResult(
                            path=path, content=None, error="is_directory"
                        )
                    )
                    continue
                raw = resolved.read_bytes()
                results.append(
                    DownloadFileResult(path=path, content=raw)
                )
            except OSError as e:
                results.append(
                    DownloadFileResult(path=path, content=None, error=str(e))
                )
            except Exception as e:
                results.append(
                    DownloadFileResult(
                        path=path,
                        content=None,
                        error=f"{type(e).__name__}: {e}",
                    )
                )
        return results


# ---------------------------------------------------------------------------
# Tree helpers
# ---------------------------------------------------------------------------

def _walk_tree(
    root: Path,
    max_depth: int,
    *,
    cwd: Path | None = None,
    current_depth: int = 0,
    ignore_dirs: frozenset[str] = frozenset(),
) -> list[TreeEntry]:
    """Recursively walk a directory tree.

    Returns:
        List of ``TreeEntry`` in DFS order (directories first, then files).
        Directories may carry a ``marker``: ``"empty"``, ``"ignore"``, or
        ``"depth_exceeded"``.
    """
    if current_depth >= max_depth:
        return []

    entries: list[TreeEntry] = []
    try:
        children = sorted(root.iterdir(), key=lambda c: (not c.is_dir(), c.name))
    except OSError:
        return []

    # Directories first
    for child in children:
        if not child.is_dir():
            continue

        virt = _virtual_path(child, cwd) if cwd else ""

        # Ignored directory: show it but skip children
        if virt and virt in ignore_dirs:
            entries.append(TreeEntry(
                name=child.name + "/",
                depth=current_depth,
                marker="ignore",
            ))
            continue

        # At depth limit: check if the directory has children
        if current_depth + 1 >= max_depth:
            try:
                has_children = any(True for _ in child.iterdir())
            except OSError:
                has_children = False
            marker: Literal["", "empty", "ignore", "depth_exceeded"] = (
                "depth_exceeded" if has_children else "empty"
            )
            entries.append(TreeEntry(
                name=child.name + "/",
                depth=current_depth,
                marker=marker,
            ))
            continue

        # Non-limit, non-ignored: check emptiness first
        try:
            sub_children = list(child.iterdir())
        except OSError:
            sub_children = []

        if not sub_children:
            entries.append(TreeEntry(
                name=child.name + "/",
                depth=current_depth,
                marker="empty",
            ))
        else:
            entries.append(TreeEntry(
                name=child.name + "/",
                depth=current_depth,
            ))
            sub = _walk_tree(
                child,
                max_depth,
                cwd=cwd,
                current_depth=current_depth + 1,
                ignore_dirs=ignore_dirs,
            )
            entries.extend(sub)

    # Then files
    for child in children:
        if child.is_file():
            try:
                size = child.stat().st_size
            except OSError:
                size = 0
            entries.append(TreeEntry(
                name=f"{child.name} ({human_size(size)})",
                depth=current_depth,
            ))

    return entries


def _virtual_path(child: Path, cwd: Path) -> str:
    """Convert a child Path to a virtual absolute path (POSIX separators)."""
    try:
        return "/" + str(child.relative_to(cwd)).replace("\\", "/")
    except ValueError:
        return ""
