"""LocalBackend – real filesystem backend with shell execution support.

Provides direct file-system access plus extra tools: ``tree`` (directory
structure), ``delete`` (remove files/directories), and ``execute`` (run
shell commands).  Supports both Windows and Linux.
"""

from __future__ import annotations

import asyncio
import base64
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

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

_DEFAULT_EXECUTE_TIMEOUT = 120
_MAX_OUTPUT_BYTES = 100_000
_LINE_NUMBER_WIDTH = 6
_MAX_LINE_LENGTH = 5000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _human_size(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size} {unit}"
        size //= 1024
    return f"{size} TB"


def _format_with_line_numbers(content: str, start_line: int = 1) -> str:
    """Format content with line numbers (``cat -n`` style)."""
    lines = content.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]

    result: list[str] = []
    for i, line in enumerate(lines):
        num = i + start_line
        if len(line) <= _MAX_LINE_LENGTH:
            result.append(f"{num:{_LINE_NUMBER_WIDTH}d}\t{line}")
        else:
            chunk_count = (len(line) + _MAX_LINE_LENGTH - 1) // _MAX_LINE_LENGTH
            for ci in range(chunk_count):
                chunk = line[ci * _MAX_LINE_LENGTH:(ci + 1) * _MAX_LINE_LENGTH]
                marker = f"{num}.{ci}" if ci > 0 else str(num)
                result.append(f"{marker:>{_LINE_NUMBER_WIDTH}}\t{chunk}")
    return "\n".join(result)


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
    ) -> None:
        if timeout <= 0:
            msg = f"timeout must be positive, got {timeout}"
            raise ValueError(msg)

        self._cwd = Path(root_dir).resolve() if root_dir else Path.cwd()
        self._default_timeout = timeout
        self._max_output_bytes = max_output_bytes
        self._enable_execute = enable_execute

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

    def ls(self, path: str) -> LsResult:
        resolved = self._resolve(path)
        if not resolved.exists() or not resolved.is_dir():
            return LsResult(error=f"Directory '{path}' not found")

        infos: list[FileInfo] = []
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
                except OSError:
                    continue
        except OSError as e:
            return LsResult(error=f"Error listing '{path}': {e}")

        return LsResult(entries=infos)

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> ReadResult:
        resolved = self._resolve(file_path)
        if not resolved.exists():
            return ReadResult(error=f"File '{file_path}' not found")
        if resolved.is_dir():
            return ReadResult(error=f"'{file_path}' is a directory")

        if _get_file_type(file_path) != "text":
            raw = resolved.read_bytes()
            encoded = base64.b64encode(raw).decode("ascii")
            file_type = _get_file_type(file_path)
            return ReadResult(
                content=encoded,
                total_lines=1,
                encoding="base64",
                file_type=file_type,
                mime_type=_get_mime_type(file_path),
            )

        try:
            content = resolved.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raw = resolved.read_bytes()
            encoded = base64.b64encode(raw).decode("ascii")
            return ReadResult(
                content=encoded,
                total_lines=1,
                encoding="base64",
                file_type=_get_file_type(file_path),
                mime_type=_get_mime_type(file_path),
            )

        lines = content.split("\n")
        if lines and lines[-1] == "":
            lines = lines[:-1]
        total = len(lines)

        sliced = lines[offset: offset + limit]
        return ReadResult(
            content=_format_with_line_numbers("\n".join(sliced), start_line=offset + 1),
            total_lines=total,
            encoding="utf-8",
        )

    def write(
        self, file_path: str, content: str, overwrite: bool = False,
    ) -> WriteResult:
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
            resolved.write_text(content, encoding="utf-8")
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
        resolved = self._resolve(file_path)
        if not resolved.exists():
            return EditResult(
                error=(
                    f"Cannot edit '{file_path}': file not found. "
                    "To create a new file, use write()."
                ),
            )

        try:
            content = resolved.read_text(encoding="utf-8")
        except OSError as e:
            return EditResult(error=f"Error reading '{file_path}': {e}")

        occurrences = content.count(old_str)

        if occurrences == 0:
            # Detect trailing-newline mismatch
            if (
                old_str.endswith("\n")
                and len(old_str) > 1
                and content.endswith(old_str.removesuffix("\n"))
            ):
                stripped = old_str.removesuffix("\n")
                stripped_count = content.count(stripped)
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

        try:
            resolved.write_text(content.replace(old_str, new_str), encoding="utf-8")
        except OSError as e:
            return EditResult(error=f"Error writing '{file_path}': {e}")

        return EditResult(path=file_path, occurrences=occurrences)

    def grep(
        self,
        pattern: str,
        path: str = "/",
        glob: str | None = None,
    ) -> GrepResult:
        resolved = self._resolve(path)
        if not resolved.exists():
            return GrepResult(error=f"Path '{path}' not found")

        import fnmatch as _fnmatch

        matches: list[GrepMatch] = []
        search_dir = resolved if resolved.is_dir() else resolved.parent

        try:
            for fp in search_dir.rglob("*"):
                if not fp.is_file():
                    continue
                if glob and not _fnmatch.fnmatch(fp.name, glob):
                    continue
                if _get_file_type(fp.suffix) != "text":
                    continue
                try:
                    lines = fp.read_text(encoding="utf-8").split("\n")
                except (UnicodeDecodeError, OSError):
                    continue
                for li, line in enumerate(lines, start=1):
                    if pattern in line:
                        virt_path = "/" + str(fp.relative_to(self._cwd)).replace("\\", "/")
                        matches.append(GrepMatch(path=virt_path, line=li, text=line))
        except OSError as e:
            return GrepResult(error=f"Error during grep: {e}", matches=matches)

        return GrepResult(matches=matches)

    def glob(self, pattern: str, path: str = "/") -> GlobResult:
        resolved = self._resolve(path)

        if not resolved.exists() or not resolved.is_dir():
            return GlobResult(error=f"Path '{path}' not found")

        matches: list[FileInfo] = []
        try:
            for fp in resolved.rglob(pattern):
                if not fp.is_file():
                    continue
                virt_path = "/" + str(fp.relative_to(self._cwd)).replace("\\", "/")
                st = fp.stat()
                matches.append(FileInfo(
                    path=virt_path,
                    is_dir=False,
                    size=st.st_size,
                ))
        except OSError as e:
            return GlobResult(error=f"Error during glob: {e}")

        return GlobResult(matches=matches)

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

        entries = _walk_tree(resolved, depth)
        return _format_tree_entries(entries)

    def delete(self, path: str) -> str:
        """Delete a file or directory.

        For directories, removes recursively (like ``rm -rf``).

        Args:
            path: Virtual path to delete.

        Returns:
            Success or error message.
        """
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
    current_depth: int = 0,
) -> list[tuple[str, int]]:
    """Recursively walk a directory tree.

    Returns:
        List of ``(display_name, depth)`` tuples in DFS order
        (directories first with their name-only and depth, then children,
         then files with size embedded in the name).
    """
    if current_depth >= max_depth:
        return []

    entries: list[tuple[str, int]] = []
    try:
        children = sorted(root.iterdir(), key=lambda c: (not c.is_dir(), c.name))
    except OSError:
        return []

    # Directories first
    for child in children:
        if child.is_dir():
            entries.append((child.name + "/", current_depth))
            sub = _walk_tree(child, max_depth, current_depth=current_depth + 1)
            entries.extend(sub)

    # Then files
    for child in children:
        if child.is_file():
            try:
                size = child.stat().st_size
            except OSError:
                size = 0
            entries.append((f"{child.name} ({_human_size(size)})", current_depth))

    return entries


def _format_tree_entries(
    entries: list[tuple[str, int]],
) -> str:
    """Render tree entries with indent-based tree display."""
    if not entries:
        return "(empty)"

    lines: list[str] = []
    for i, (display, depth) in enumerate(entries):
        # Determine connector: look ahead to see if there are siblings at the same depth
        has_more_siblings = False
        for j in range(i + 1, len(entries)):
            next_depth = entries[j][1]
            if next_depth < depth:
                break
            if next_depth == depth:
                has_more_siblings = True
                break

        connector = "├── " if has_more_siblings else "└── "
        if depth == 0:
            # Root-level entry — no indent prefix, just show it
            lines.append(display)
        else:
            # Build prefix: for each parent level, decide "│   " or "    "
            prefix_parts: list[str] = []
            # Walk backward to find active parent lines
            for level in range(1, depth + 1):
                # Check if there's any future entry at this level or deeper after us
                active = False
                for j in range(i + 1, len(entries)):
                    if entries[j][1] < level:
                        break
                    if entries[j][1] == level:
                        active = True
                        break
                prefix_parts.append("│   " if active else "    ")

            prefix = "".join(prefix_parts)
            lines.append(f"{prefix}{connector}{display}")

    return "\n".join(lines)
