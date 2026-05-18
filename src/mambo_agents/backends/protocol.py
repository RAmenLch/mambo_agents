"""Backend protocol – the minimal abstract contract all backends must implement.

Defines ``BackendProtocol`` (ABC) and strictly-typed Pydantic result
models.  Each concrete backend builds its own ``tools`` list by hand
and may freely add backend-specific operations beyond the core six.
"""

import abc
import asyncio
import base64
import mimetypes
from pathlib import PurePosixPath
from typing import Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict


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
        if self.error is not None:
            return f"Error: {self.error}"
        if self.entries is None or not self.entries:
            return "(empty directory)"
        lines: list[str] = []
        for fi in self.entries:
            if fi.is_dir:
                lines.append(f"{fi.path}/")
            else:
                lines.append(f"{fi.path}  ({_human_size(fi.size)})")
        return "\n".join(lines)


class ReadResult(BaseModel):
    """Result from ``read()``.

    For text files, ``content`` carries line-numbered text and
    ``encoding`` is ``"utf-8"``.  For binary files, ``content`` is
    base64-encoded (no line numbers) and ``encoding`` is ``"base64"``.

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

    def __str__(self) -> str:
        if self.error is not None:
            return f"Error: {self.error}"
        if not self.matches:
            return "No matches found."
        return "\n".join(
            f"{m.path}:{m.line}: {m.text}" for m in self.matches
        )


class GlobResult(BaseModel):
    """Result from ``glob()``."""

    error: str | None = None
    matches: list[FileInfo] | None = None

    def __str__(self) -> str:
        if self.error is not None:
            return f"Error: {self.error}"
        if not self.matches:
            return "No files found."
        return "\n".join(fi.path for fi in self.matches)


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
# Protocol
# ============================================================================


class BackendProtocol(abc.ABC):
    """Minimal abstract contract all file-system backends must implement.

    Each concrete backend:

    - Implements the six core operations (``ls``, ``read``, ``write``,
      ``edit``, ``grep``, ``glob``).
    - Provides a ``tools`` property returning the ``StructuredTool``
      list exposed to the agent.  Extra tools (``delete``, ``execute``,
      ``sandbox_run``, …) are freely added per backend.
    - Overrides ``description`` to inject backend-specific context into
      the system prompt.

    The protocol includes default implementations for ``upload_files()``
    and ``download_files()`` that delegate to the core operations —
    subclasses may override these for better performance.

    All paths are absolute and start with ``"/"``.
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
        """
        ...

    @property
    def description(self) -> str:
        """One-line description injected into the system prompt.

        Override this to tell the agent about backend-specific
        capabilities (e.g. "Local file system with delete and shell
        execute support").
        """
        return f"File system backend: {type(self).__name__.replace('Backend', '').lower()}"

    # ------------------------------------------------------------------
    # Core file operations (abstract — every backend MUST implement)
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def ls(self, path: str) -> LsResult:
        """List files and directories in *path* (non-recursive)."""
        ...

    @abc.abstractmethod
    def read(
        self, file_path: str, offset: int = 0, limit: int = 2000
    ) -> ReadResult:
        """Read the contents of *file_path*."""
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
        path: str = "/",
        glob: str | None = None,
    ) -> GrepResult:
        """Search for a literal substring in files under *path*."""
        ...

    @abc.abstractmethod
    def glob(self, pattern: str, path: str = "/") -> GlobResult:
        """Find files matching a glob pattern under *path*."""
        ...

    # ------------------------------------------------------------------
    # Async core file operations
    # ------------------------------------------------------------------

    async def als(self, path: str) -> LsResult:
        """Async: List files and directories in *path* (non-recursive)."""
        return await asyncio.to_thread(self.ls, path)

    async def aread(
        self, file_path: str, offset: int = 0, limit: int = 2000
    ) -> ReadResult:
        """Async: Read the contents of *file_path*."""
        return await asyncio.to_thread(self.read, file_path, offset, limit)

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
        path: str = "/",
        glob: str | None = None,
    ) -> GrepResult:
        """Async: Search for a literal substring in files under *path*."""
        return await asyncio.to_thread(self.grep, pattern, path, glob)

    async def aglob(self, pattern: str, path: str = "/") -> GlobResult:
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
            r = self.read(path)
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
# Internal helpers
# ============================================================================


def _human_size(size: int) -> str:
    """Format *size* as a human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size} {unit}"
        size //= 1024
    return f"{size} TB"
