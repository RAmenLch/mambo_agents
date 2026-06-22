"""Backend protocol – the minimal abstract contract all backends must implement.

Defines ``BackendProtocol`` (ABC) and strictly-typed Pydantic result
models.  Each concrete backend builds its own ``tools`` list by hand
and may freely add backend-specific operations beyond the core six.
"""

import abc
import asyncio
import base64
import mimetypes
import threading
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Awaitable, Callable, Literal, TypeVar

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict

from mambo_agents.backends.utils import human_size


# ============================================================================
# File type classification
# ============================================================================


FileType = Literal["text", "image", "audio", "video", "file"]
"""Classification of a file by its extension for multimodal dispatch."""

_EXTENSION_TO_FILE_TYPE: dict[str, FileType] = {
    # Images (https://ai.google.dev/gemini-api/docs/image-understanding)
    ".png": "image",
    ".jpeg": "image",
    ".jpg": "image",
    ".webp": "image",
    ".gif": "image",
    ".heic": "image",
    ".heif": "image",
    # Video (https://ai.google.dev/gemini-api/docs/video-understanding)
    ".mp4": "video",
    ".mpeg": "video",
    ".mov": "video",
    ".avi": "video",
    ".flv": "video",
    ".mpg": "video",
    ".webm": "video",
    ".wmv": "video",
    ".3gpp": "video",
    # Audio (https://ai.google.dev/gemini-api/docs/audio)
    ".wav": "audio",
    ".mp3": "audio",
    ".aiff": "audio",
    ".aac": "audio",
    ".ogg": "audio",
    ".flac": "audio",
    # Documents
    ".pdf": "file",
    ".ppt": "file",
    ".pptx": "file",
}


def _get_file_type(path: str) -> FileType:
    """Classify a file by its extension.

    Returns:
        One of ``"text"``, ``"image"``, ``"audio"``, ``"video"``, or ``"file"``.
        Defaults to ``"text"`` for unrecognized extensions.
    """
    return _EXTENSION_TO_FILE_TYPE.get(
        PurePosixPath(path).suffix.lower(), "text"
    )


def _get_mime_type(path: str) -> str:
    """Guess the MIME type for a file path."""
    return (
        mimetypes.guess_type("file" + PurePosixPath(path).suffix)[0]
        or "application/octet-stream"
    )


# ============================================================================
# Callback types
# ============================================================================


ReadSummarizer = Callable[[str, str, int], str]
"""Callback that summarizes oversized text content.

Args:
    file_path: The virtual path of the file being read.
    content: The full text content that exceeded the character limit.
    max_chars: The configured character limit (``max_read_chars``).

Returns:
    A short summary string to replace the oversized content.
"""


# ============================================================================
# Tool timeout configuration
# ============================================================================


@dataclass(frozen=True)
class ToolTimeouts:
    """Per-tool timeout configuration (seconds).

    Each field maps to a tool name.  Pass an instance to any backend's
    constructor via the ``tool_timeouts`` parameter to customise limits.

    Defaults are chosen conservatively — actual values should be tuned
    per deployment based on repository size and network latency.
    """

    ls: float = 20.0
    """``ls`` — directory listing (synchronous SFTP / filesystem call)."""

    read: float = 60.0
    """``read`` — single-file read (SFTP transfer with optional line-number formatting)."""

    write: float = 60.0
    """``write`` — single-file create / overwrite."""

    edit: float = 60.0
    """``edit`` — text replacement in an existing file (remote python3 or local)."""

    grep: float = 120.0
    """``grep`` — recursive text search (may fall back to remote ``os.walk``)."""

    glob: float = 60.0
    """``glob`` — recursive pattern match (remote ``glob.glob`` or ``find``)."""

    tree: float = 60.0
    """``tree`` — directory tree render (remote ``os.walk`` or ``find``)."""

    delete: float = 30.0
    """``delete`` — single-file removal (directories are rejected)."""

    execute: float = 180.0
    """``execute`` — arbitrary shell command on the remote server."""

    upload: float = 60.0
    """``upload_files`` — bulk file upload."""

    download: float = 60.0
    """``download_files`` — bulk file download."""


class ToolTimeoutError(Exception):
    """Raised when a backend tool exceeds its configured timeout.

    Caught by the agent runtime and presented as a tool error, giving
    the LLM a chance to retry with narrower scope.
    """

    def __init__(self, tool_name: str, timeout_seconds: float) -> None:
        self.tool_name = tool_name
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"Tool '{tool_name}' timed out after {timeout_seconds:.0f}s. "
            f"Try narrowing the scope — use a more specific path, glob "
            f"pattern, or read limit."
        )


# ============================================================================
# Value objects
# ============================================================================


class FileInfo(BaseModel):
    """Structured file / directory listing entry."""

    model_config = ConfigDict(frozen=True)

    path: str
    """Absolute virtual path."""
    is_dir: bool = False
    size: int = 0
    """Size in bytes (0 for directories)."""
    modified_at: str = ""
    """ISO-8601 timestamp or empty string."""


class GrepMatch(BaseModel):
    """A single match from a grep search."""

    model_config = ConfigDict(frozen=True)

    path: str
    """File path."""
    line: int
    """1-indexed line number."""
    text: str
    """Content of the matching line."""


# ============================================================================
# Result types
# ============================================================================


class LsResult(BaseModel):
    """Result from ``ls()``."""

    error: str | None = None
    entries: list[FileInfo] | None = None

    def __str__(self) -> str:
        lines: list[str] = []
        if self.error is not None:
            lines.append(f"Warning: {self.error}")
        if self.entries is not None:
            for fi in self.entries:
                if fi.is_dir:
                    lines.append(f"{fi.path}/")
                else:
                    lines.append(f"{fi.path}  ({human_size(fi.size)})")
        if not lines:
            return "(empty directory)"
        return "\n".join(lines)


class ReadResult(BaseModel):
    """Result from ``read()``.

    For text files, ``content`` carries plain text (no line numbers by
    default) and ``encoding`` is ``"utf-8"``.  Pass
    ``include_line_numbers=True`` to the ``read()`` call to get
    ``cat -n``-style line-numbered output.  For binary files,
    ``content`` is base64-encoded and ``encoding`` is ``"base64"``.

    For non-text (multimodal) files, ``file_type`` is set to the
    appropriate classification (``"image"``, ``"audio"``, ``"video"``,
    or ``"file"``) and ``mime_type`` provides the IANA media type.
    """

    error: str | None = None
    content: str | None = None
    total_lines: int = 0
    encoding: str | None = None
    file_type: FileType = "text"
    """File type classification for multimodal dispatch."""
    mime_type: str = ""
    """IANA media type (e.g. ``"image/png"``) when ``file_type`` is not ``"text"``."""

    @property
    def is_multimodal(self) -> bool:
        """Return ``True`` if the file should be delivered as a multimodal content block."""
        return self.encoding == "base64" and self.file_type != "text"

    def __str__(self) -> str:
        if self.error is not None:
            return f"Error: {self.error}"
        return self.content or ""


class WriteResult(BaseModel):
    """Result from ``write()`` — create or overwrite a file.

    By default ``overwrite=False`` and creating a file that already
    exists is an error.  Set ``overwrite=True`` to replace the file
    contents entirely.
    """

    error: str | None = None
    path: str | None = None

    def __str__(self) -> str:
        if self.error is not None:
            return f"Error: {self.error}"
        return f"File written: {self.path}"


class EditResult(BaseModel):
    """Result from ``edit()`` — replace text in an existing file."""

    error: str | None = None
    path: str | None = None
    occurrences: int = 0

    def __str__(self) -> str:
        if self.error is not None:
            return f"Error: {self.error}"
        return f"File edited: {self.path} ({self.occurrences} replacement(s))"


class GrepResult(BaseModel):
    """Result from ``grep()``."""

    error: str | None = None
    matches: list[GrepMatch] | None = None
    truncated: bool = False
    """``True`` when the result was truncated by ``max_grep_matches`` or offset/limit."""
    total_matches: int = 0
    """Total number of matches found before offset/limit slicing."""

    def __str__(self) -> str:
        lines: list[str] = []
        if self.error is not None:
            lines.append(f"Warning: {self.error}")
        if self.matches:
            lines.extend(f"{m.path}:{m.line}: {m.text}" for m in self.matches)
        if not lines:
            return "No matches found."
        if self.truncated:
            shown = len(self.matches) if self.matches else 0
            total = self.total_matches
            lines.append(
                f"\n[Truncated: showing {shown} of {total} matches. "
                f"Use a narrower pattern/path/glob or adjust offset/limit to paginate.]"
            )
        return "\n".join(lines)


class GlobResult(BaseModel):
    """Result from ``glob()``."""

    error: str | None = None
    matches: list[FileInfo] | None = None

    def __str__(self) -> str:
        lines: list[str] = []
        if self.error is not None:
            lines.append(f"Warning: {self.error}")
        if self.matches:
            lines.extend(fi.path for fi in self.matches)
        if not lines:
            return "No files found."
        return "\n".join(lines)


class UploadFileResult(BaseModel):
    """Result for a single file in a bulk upload."""

    path: str
    error: str | None = None


class DownloadFileResult(BaseModel):
    """Result for a single file in a bulk download."""

    path: str
    content: bytes | None = None
    error: str | None = None


# ============================================================================
# Workspace path error
# ============================================================================


class WorkspacePathError(ValueError):
    """Raised when a virtual path falls outside the workspace root.

    Each backend defines a ``workspace_root`` (default ``"/workspace"``)
    that serves as the only valid anchor for file operations.  Any path
    outside this prefix is rejected so the AI never perceives the virtual
    filesystem as a real system root.
    """


# ============================================================================
# Protocol
# ============================================================================


class BackendProtocol(abc.ABC):
    """Minimal abstract contract all file-system backends must implement.

    Each concrete backend:

    - Implements the six core operations (``ls``, ``read_raw``, ``write``,
      ``edit``, ``grep``, ``glob``).  ``read()`` is a concrete template
      method that delegates to :meth:`read_raw` and then applies
      ``max_read_chars`` safety limits.
    - Provides a ``tools`` property returning the ``StructuredTool``
      list exposed to the agent.  Extra tools (``delete``, ``execute``,
      ``sandbox_run``, …) are freely added per backend.
    - Overrides ``description`` to inject backend-specific context into
      the system prompt.

    The protocol includes default implementations for ``upload_files()``
    and ``download_files()`` that delegate to the core operations —
    subclasses may override these for better performance.

    All paths are absolute and start with the :attr:`workspace_root`
    (default ``"/workspace"``).  Paths outside this prefix are rejected.
    """
    # ------------------------------------------------------------------
    # Workspace root
    # ------------------------------------------------------------------

    workspace_root: str = "/workspace"
    """The virtual path prefix that serves as the root of the workspace.

    All file operations must target paths under this prefix (e.g.
    ``"/workspace/src/main.py"``).  Paths outside are rejected with
    :class:`WorkspacePathError` so the AI never treats the virtual
    filesystem as a real system root.
    """

    # ------------------------------------------------------------------
    # Required properties
    # ------------------------------------------------------------------

    @property
    @abc.abstractmethod
    def tools(self) -> list[StructuredTool]:
        """Extra (non-core) tools specific to this backend.

        The six core tools (``ls``, ``read``, ``write``, ``edit``,
        ``grep``, ``glob``) are built by the middleware — they are
        **not** included here.  Return only backend-specific extras
        (e.g. ``tree``, ``delete``, ``execute``).  Return an empty
        list if the backend has no extras.

        Note: ``read`` is a concrete template method; subclasses
        implement :meth:`read_raw` instead.
        """
        ...

    @property
    def description(self) -> str:
        """One-line description injected into the system prompt.

        Override this to tell the agent about backend-specific
        capabilities (e.g. "Local file system with delete and shell
        execute support").

        .. note::

            The ``read`` tool defaults to **no line numbers**.  When
            you need to reference specific lines (e.g. for targeted
            edits), set ``include_line_numbers=True`` so that each
            output line is prefixed with its 1-indexed line number.
        """
        return (
            f"File system backend: {type(self).__name__.replace('Backend', '').lower()}. "
            "The read tool defaults to no line numbers. "
            "Set include_line_numbers=True when you need to reference specific lines "
            "(e.g. for edits or patching)."
        )

    # ------------------------------------------------------------------
    # Construction & read-limit configuration
    # ------------------------------------------------------------------

    def __init__(
        self,
        *,
        max_read_chars: int = 100_000,
        max_grep_matches: int = 1000,
        summarizer: ReadSummarizer | None = None,
        tool_timeouts: ToolTimeouts | None = None,
    ) -> None:
        if max_read_chars < 1:
            raise ValueError(f"max_read_chars must be >= 1, got {max_read_chars}")
        if max_grep_matches < 1:
            raise ValueError(f"max_grep_matches must be >= 1, got {max_grep_matches}")
        self._max_read_chars = max_read_chars
        self._max_grep_matches = max_grep_matches
        self._summarizer: ReadSummarizer = summarizer or self._default_summarizer
        self._tool_timeouts = tool_timeouts or ToolTimeouts()

    def _timeout_for(self, tool_name: str) -> float:
        """Return the timeout (seconds) configured for *tool_name*.

        Falls back to 60 s for unknown tool names.
        """
        return getattr(self._tool_timeouts, tool_name, 60.0)

    _T = TypeVar("_T")

    async def _await_with_timeout(
        self, tool_name: str, coro: Awaitable[_T],
    ) -> _T:
        """Await *coro* with the timeout configured for *tool_name*.

        Raises :class:`ToolTimeoutError` when the configured time limit
        is exceeded.
        """
        timeout = self._timeout_for(tool_name)
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            raise ToolTimeoutError(tool_name, timeout)

    def _wrap_tool_coroutine(
        self, tool_name: str, async_method: Callable[..., Awaitable[Any]],
    ) -> Callable[..., Awaitable[Any]]:
        """Return an async wrapper that applies timeout to *async_method*.

        Intended for use when building ``StructuredTool`` instances so
        that timeout enforcement happens at the tool boundary rather
        than inside each async method.  Backend implementors write
        plain async methods; the framework wraps them here.

        Usage::

            StructuredTool(
                name="grep",
                coroutine=backend._wrap_tool_coroutine("grep", backend.agrep),
                ...
            )
        """
        async def _wrapped(*args: Any, **kwargs: Any) -> Any:
            return await self._await_with_timeout(
                tool_name, async_method(*args, **kwargs),
            )
        _wrapped.__name__ = f"_timeout_wrapped_{tool_name}"
        return _wrapped

    def _wrap_sync_with_timeout(
        self, tool_name: str, sync_method: Callable[..., Any],
    ) -> Callable[..., Any]:
        """Return a sync wrapper that applies timeout to *sync_method* via threading.

        Runs *sync_method* in a daemon thread and joins with the configured
        timeout.  Raises :class:`ToolTimeoutError` if the thread does not
        finish within the limit.

        Intended for use when building ``StructuredTool`` sync functions so
        that timeout enforcement works for both ``invoke()`` and ``ainvoke()``.
        """
        timeout = self._timeout_for(tool_name)

        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            result_holder: list[Any] = []
            error_holder: list[BaseException] = []

            def _target() -> None:
                try:
                    result_holder.append(sync_method(*args, **kwargs))
                except Exception as exc:
                    error_holder.append(exc)

            t = threading.Thread(target=_target, daemon=True)
            t.start()
            t.join(timeout=timeout)

            if t.is_alive():
                raise ToolTimeoutError(tool_name, timeout)
            if error_holder:
                raise error_holder[0]
            return result_holder[0]

        _wrapped.__name__ = f"_timeout_wrapped_sync_{tool_name}"
        return _wrapped

    def _safe_tool_coroutine(
        self, tool_name: str, async_method: Callable[..., Awaitable[Any]],
    ) -> Callable[..., Awaitable[str]]:
        """Return an async coroutine that catches :class:`ToolTimeoutError`.

        Wraps the timeout-protected coroutine from :meth:`_wrap_tool_coroutine`
        and converts :class:`ToolTimeoutError` to an error string so the LLM
        sees a graceful timeout message instead of a crashed run.
        """
        wrapped = self._wrap_tool_coroutine(tool_name, async_method)

        async def _safe(*args: Any, **kwargs: Any) -> str:
            try:
                return await wrapped(*args, **kwargs)
            except ToolTimeoutError as e:
                return str(e)

        _safe.__name__ = f"_safe_wrapped_{tool_name}"
        return _safe

    def _safe_tool_func(
        self, tool_name: str, sync_method: Callable[..., Any],
    ) -> Callable[..., str]:
        """Return a sync func that catches :class:`ToolTimeoutError`.

        Wraps the timeout-protected func from :meth:`_wrap_sync_with_timeout`
        and converts :class:`ToolTimeoutError` to an error string so the LLM
        sees a graceful timeout message instead of a crashed run.
        """
        wrapped = self._wrap_sync_with_timeout(tool_name, sync_method)

        def _safe(*args: Any, **kwargs: Any) -> str:
            try:
                return wrapped(*args, **kwargs)
            except ToolTimeoutError as e:
                return str(e)

        _safe.__name__ = f"_safe_wrapped_sync_{tool_name}"
        return _safe

    @staticmethod
    def _default_summarizer(file_path: str, content: str, max_chars: int) -> str:
        """Default summarizer: prompt the caller to specify offset + limit."""
        total_lines = content.count("\n") + 1
        return (
            f"[返回结果过大（{len(content):,} 字符，{total_lines:,} 行），"
            f"超过读取上限 {max_chars:,} 字符。"
            f"请重新指定 offset + limit 参数后读取。"
            f"示例: read(file_path='{file_path}', offset=0, limit=500)]"
        )

    def _apply_read_limit(self, result: ReadResult, file_path: str) -> ReadResult:
        """Apply character-limit to a text ``ReadResult``.

        Binary / multimodal files are never truncated.  When the text
        content exceeds ``_max_read_chars``, the ``summarizer`` callback
        replaces the content with a short prompt.
        """
        if result.error or result.encoding == "base64" or result.file_type != "text":
            return result
        if result.content and len(result.content) > self._max_read_chars:
            result = result.model_copy(update={
                "content": self._summarizer(file_path, result.content, self._max_read_chars),
            })
        return result

    def _apply_grep_limit(
        self,
        matches: list[GrepMatch],
        offset: int,
        limit: int | None,
    ) -> GrepResult:
        """Apply offset / limit slicing and ``max_grep_matches`` cap to grep results.

        Args:
            matches: All collected matches (up to ``max_grep_matches + 1``
                to detect truncation).
            offset: 0-based starting index into *matches*.
            limit: Max number of matches to return.  ``None`` means
                "use ``max_grep_matches``".

        Returns:
            A ``GrepResult`` with ``truncated`` and ``total_matches`` set.
        """
        total = len(matches)
        effective_limit = limit if limit is not None else self._max_grep_matches
        effective_limit = min(effective_limit, self._max_grep_matches)
        if offset < 0:
            offset = 0
        sliced = matches[offset : offset + effective_limit]
        truncated = (offset + effective_limit) < total
        return GrepResult(
            matches=sliced if sliced else None,
            truncated=truncated,
            total_matches=total,
        )

    # ------------------------------------------------------------------
    # Core file operations (abstract — every backend MUST implement
    # read_raw instead of read; read is a concrete template method)
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def ls(self, path: str) -> LsResult:
        """List files and directories in *path* (non-recursive)."""
        ...

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
        include_line_numbers: bool = False,
        *,
        _apply_max_chars: bool = True,
    ) -> ReadResult:
        """Read the contents of *file_path*.

        For text files, the returned ``content`` is plain text by
        default (no line numbers).  Set *include_line_numbers* to
        ``True`` to get ``cat -n``-style output where each line is
        prefixed with its 1-indexed line number.

        Text files exceeding ``max_read_chars`` are summarized using
        the configured ``summarizer`` callback.  Binary / multimodal
        files are never truncated.  Pass ``_apply_max_chars=False`` to
        bypass the limit (used internally by ``download_files``).
        """
        if offset < 0:
            return ReadResult(error=f"offset must be non-negative, got {offset}")
        if limit < 1:
            return ReadResult(error=f"limit must be positive, got {limit}")
        result = self.read_raw(file_path, offset, limit, include_line_numbers)
        if _apply_max_chars:
            result = self._apply_read_limit(result, file_path)
        return result

    @abc.abstractmethod
    def read_raw(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
        include_line_numbers: bool = False,
    ) -> ReadResult:
        """Read file contents **without** character-limit safety checks.

        Subclasses must implement this method to perform the actual
        file I/O.  The returned ``ReadResult`` is the raw, untruncated
        result — the caller (typically :meth:`read`) is responsible for
        applying ``max_read_chars`` limits.
        """
        ...

    @abc.abstractmethod
    def write(
        self, file_path: str, content: str, overwrite: bool = False,
    ) -> WriteResult:
        """Create a new file or overwrite an existing one.

        When *overwrite* is ``False`` (default), the operation must
        fail if the file already exists.  Set *overwrite* to ``True``
        to replace the entire file content.
        """
        ...

    @abc.abstractmethod
    def edit(
        self,
        file_path: str,
        old_str: str,
        new_str: str,
        *,
        replace_all: bool = False,
    ) -> EditResult:
        """Replace *old_str* with *new_str* in an existing file.

        Args:
            replace_all: If ``False`` (default), *old_str* must appear
                exactly once.  If it appears multiple times, an error is
                returned.  Set to ``True`` to replace all occurrences.
        """
        ...

    @abc.abstractmethod
    def grep(
        self,
        pattern: str,
        path: str = "/workspace",
        glob: str | None = None,
        regex: bool = False,
        offset: int = 0,
        limit: int | None = None,
    ) -> GrepResult:
        """Search for a text pattern in files under *path*.

        When *regex* is ``False`` (default), performs exact substring
        matching (literal mode, fast and safe).  Set *regex* to ``True``
        to interpret *pattern* as a Python regex with alternation,
        character classes, quantifiers, etc.

        Args:
            offset: 0-based index into matches to start from (for pagination).
            limit: Max matches to return.  ``None`` means up to
                ``max_grep_matches`` (default 1000).
        """
        ...

    @abc.abstractmethod
    def glob(self, pattern: str, path: str = "/workspace") -> GlobResult:
        """Find files matching a glob pattern under *path*."""
        ...

    # ------------------------------------------------------------------
    # Async core file operations
    # ------------------------------------------------------------------

    async def als(self, path: str) -> LsResult:
        """Async: List files and directories in *path* (non-recursive)."""
        return await asyncio.to_thread(self.ls, path)

    async def aread(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
        include_line_numbers: bool = False,
        *,
        _apply_max_chars: bool = True,
    ) -> ReadResult:
        """Async: Read the contents of *file_path*.

        See :meth:`read` for full documentation.
        """
        return await asyncio.to_thread(
            self.read, file_path, offset, limit, include_line_numbers,
            _apply_max_chars=_apply_max_chars,
        )

    async def awrite(
        self, file_path: str, content: str, overwrite: bool = False,
    ) -> WriteResult:
        """Async: Create a new file or overwrite an existing one."""
        return await asyncio.to_thread(self.write, file_path, content, overwrite)

    async def aedit(
        self,
        file_path: str,
        old_str: str,
        new_str: str,
        *,
        replace_all: bool = False,
    ) -> EditResult:
        """Async: Replace *old_str* with *new_str* in an existing file."""
        return await asyncio.to_thread(
            self.edit, file_path, old_str, new_str, replace_all=replace_all,
        )

    async def agrep(
        self,
        pattern: str,
        path: str = "/workspace",
        glob: str | None = None,
        regex: bool = False,
        offset: int = 0,
        limit: int | None = None,
    ) -> GrepResult:
        """Async: Search for a text pattern in files under *path*."""
        return await asyncio.to_thread(self.grep, pattern, path, glob, regex, offset, limit)

    async def aglob(self, pattern: str, path: str = "/workspace") -> GlobResult:
        """Async: Find files matching a glob pattern under *path*."""
        return await asyncio.to_thread(self.glob, pattern, path)

    # ------------------------------------------------------------------
    # Developer API — bulk upload / download
    # ------------------------------------------------------------------

    def upload_files(
        self, files: list[tuple[str, bytes]]
    ) -> list[UploadFileResult]:
        """Upload multiple files as raw bytes.

        Each file's bytes are inspected: valid UTF-8 is stored as text,
        otherwise base64-encoded.

        Returns:
            Per-file results with ``path`` and optional ``error``.
        """
        results: list[UploadFileResult] = []
        for path, raw_content in files:
            try:
                text = raw_content.decode("utf-8")
                w = self.write(path, text)
            except UnicodeDecodeError:
                encoded = base64.b64encode(raw_content).decode("ascii")
                w = self.write(path, encoded)
            results.append(
                UploadFileResult(path=path, error=w.error)
            )
        return results

    def download_files(
        self, paths: list[str]
    ) -> list[DownloadFileResult]:
        """Download multiple files as original bytes.

        Returns:
            Per-file results with ``path``, ``content`` (bytes or
            ``None``), and optional ``error``.
        """
        results: list[DownloadFileResult] = []
        for path in paths:
            r = self.read(path, _apply_max_chars=False)
            if r.error:
                results.append(
                    DownloadFileResult(
                        path=path, content=None, error=r.error,
                    )
                )
                continue
            if r.encoding == "base64" and r.content is not None:
                content_bytes = base64.standard_b64decode(r.content)
            elif r.content is not None:
                content_bytes = r.content.encode("utf-8")
            else:
                content_bytes = None
            results.append(
                DownloadFileResult(
                    path=path, content=content_bytes, error=None,
                )
            )
        return results

    async def adownload_files(
        self, paths: list[str]
    ) -> list[DownloadFileResult]:
        """Async: Download multiple files as original bytes."""
        return await asyncio.to_thread(self.download_files, paths)


# ============================================================================
# ThreadAwareWorkspace — thread/session-aware backends
# ============================================================================


class ThreadAwareWorkspace(BackendProtocol, abc.ABC):
    """Protocol for backends aware of LangGraph thread/session context.

    Backends implementing this protocol need access to a *thread_id* for
    ``upload_files()`` and ``download_files()`` so they can correctly
    isolate state across concurrent sessions (e.g. multiple graph
    execution threads).

    Concrete implementors:

    - :class:`~mambo_agents.backends.state.StateBackend` — files live in
      per-thread Pregel state channels.
    - :class:`~mambo_agents.backends.hybrid_workspace.HybridWorkspaceBackend`
      — routes bulk operations to backends that may be thread-aware.

    Router backends (like ``HybridWorkspaceBackend``) can check
    ``isinstance(target, ThreadAwareWorkspace)`` to decide whether
    ``thread_id`` should be forwarded, without coupling to a concrete
    implementation.
    """

    @abc.abstractmethod
    def upload_files(
        self,
        files: list[tuple[str, bytes]],
        *,
        thread_id: str | None = None,
    ) -> list[UploadFileResult]:
        """Upload multiple files with optional thread/session context.

        Parameters:
            files: List of ``(path, raw_bytes)`` tuples.
            thread_id: Thread/session identifier.  Required by backends
                that isolate state per thread (e.g. StateBackend outside
                graph context); optional for simple backends.
        """
        ...

    @abc.abstractmethod
    def download_files(
        self,
        paths: list[str],
        *,
        thread_id: str | None = None,
    ) -> list[DownloadFileResult]:
        """Download multiple files with optional thread/session context.

        Parameters:
            paths: List of absolute file paths to download.
            thread_id: Thread/session identifier.  Required by backends
                that isolate state per thread (e.g. StateBackend outside
                graph context); optional for simple backends.
        """
        ...

