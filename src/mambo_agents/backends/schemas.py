"""Domain data types for the virtual file-system backend.

Plain Pydantic models and value objects — zero business logic.
Safe to import from anywhere (utils, protocol, backends, middleware)
without circular-dependency risk.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator


# ============================================================================
# Formatter — converts byte counts to human-readable strings for Result display
# ============================================================================


def human_size(size: int) -> str:
    """Format *size* as a human-readable string (e.g. ``"1.2 MB"``)."""
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size} {unit}"
        size //= 1024
    return f"{size} TB"


# ============================================================================
# Path validation
# ============================================================================


def check_no_path_traversal(path: str, *, name: str = "path") -> None:
    """Raise :class:`ValueError` if *path* contains ``".."`` or ``"//"``.

    This check MUST run on the **raw, un-normalized** path string —
    **before** passing it through :class:`~pathlib.PurePosixPath` or any
    other normalizer.  Normalizers like ``PurePosixPath`` silently
    collapse ``".."`` segments, which would defeat a subsequent prefix
    check.

    Args:
        path: The raw path string to check.
        name: Human-readable name for the parameter (used in error messages).

    Raises:
        ValueError: If *path* contains ``".."`` path traversal or ``"//"``
            double slashes.
    """
    # Split on "/" separates the raw segments without collapsing ".."
    if ".." in path.split("/"):
        raise ValueError(
            f"{name} must not contain '..' path traversal, got {path!r}"
        )
    if "//" in path:
        raise ValueError(f"{name} must not contain '//', got {path!r}")


def _normalize_path_str(value: str) -> str:
    """Validate + normalize a path string for :class:`VirtualPath` construction.

    Raises ValueError if the path is empty, non-absolute, contains ``..``
    traversal, ``//``, or is the root directory ``"/"`` (trailing slash
    is preserved — ``"/workspace/"`` is valid, ``"/"`` is not).
    Returns the validated path string as-is.
    """
    check_no_path_traversal(value, name="path")
    s = value.strip()
    if not s:
        raise ValueError("path must not be empty")
    if not s.startswith("/"):
        raise ValueError(
            f"path must be an absolute path starting with '/', got {s!r}"
        )
    # Reject root — but "/workspace/" with trailing slash is fine
    if s.rstrip("/") == "":
        raise ValueError(
            "path must not be the root directory '/'; use a subdirectory like '/workspace'"
        )
    return s


class VirtualPath(BaseModel):
    """Canonical virtual-filesystem path.

    Construction validates the path **immediately** (fails fast for
    ``".."`` traversal, ``"//"``, empty / non-absolute / root ``"/"``).
    The model is **frozen** so a validated ``VirtualPath`` is immutable
    and safe to pass across backend / middleware boundaries.

    Trailing slashes are preserved — ``"/workspace/"`` is distinct from
    ``"/workspace"``.  Use :attr:`normalized` for the no-trailing-slash
    form in internal file-system lookups.  Methods like
    :meth:`StateBackend.write` reject paths ending in ``"/"``.

    Supports three construction styles::

        VirtualPath("/workspace/src")   # positional str (most common)
        VirtualPath(value="/workspace/src")  # keyword
        VirtualPath(existing_vp)        # pass-through

    When used as a Pydantic field type (e.g. in a ``StructuredTool``
    args_schema), the :meth:`model_validator` with ``mode='before'``
    automatically coerces a plain ``str`` from the LLM into a
    ``VirtualPath`` — no manual conversion needed.

    Attributes:
        value: Absolute path as provided (may end with ``"/"``).
        normalized: Same path guaranteed without trailing slash.
    """

    model_config = {"frozen": True}

    value: str = Field(
        default="",
        description="Absolute path, exactly as provided (may end with '/')",
    )

    @property
    def normalized(self) -> str:
        """Path without trailing slash, for internal file-system operations."""
        return self.value.rstrip("/") or self.value

    def __init__(self, _value: str | VirtualPath | None = None, /, **data):
        """Accept ``VirtualPath("/path")`` (positional) or keyword styles."""
        if isinstance(_value, VirtualPath):
            data["value"] = _value.value
        elif isinstance(_value, str):
            data["value"] = _value
        elif _value is not None:
            raise TypeError(
                f"Expected str or VirtualPath, got {type(_value).__name__}: {_value!r}"
            )
        super().__init__(**data)

    @model_validator(mode="before")
    @classmethod
    def _validate_and_normalize(cls, data: object) -> dict:
        """Coerce ``str`` → ``VirtualPath`` or validate an existing one."""
        if isinstance(data, VirtualPath):
            return {"value": data.value}
        if isinstance(data, str):
            return {"value": _normalize_path_str(data)}
        if isinstance(data, dict):
            # Keyword construction: VirtualPath(value="/path")
            v = data.get("value", None)
            if v is None:
                raise ValueError("VirtualPath requires a 'value' field")
            if isinstance(v, VirtualPath):
                return {"value": v.value}
            return {"value": _normalize_path_str(v)}
        raise TypeError(
            f"Expected str or VirtualPath, got {type(data).__name__}: {data!r}"
        )

    @model_serializer(mode="plain")
    def _serialize(self) -> str:
        """Serialize VirtualPath as a plain string (not a nested dict)."""
        return self.value

    # -- hash / display --------------------------------------------------

    def __hash__(self) -> int:
        return hash(self.value)

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"VirtualPath({self.value!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, VirtualPath):
            return self.value == other.value
        if isinstance(other, str):
            return self.value == other
        return NotImplemented

    def __lt__(self, other: object) -> bool:
        if isinstance(other, VirtualPath):
            return self.value < other.value
        if isinstance(other, str):
            return self.value < other
        return NotImplemented

    # -- business methods -------------------------------------------------

    def is_under(self, prefix: str) -> bool:
        """Return ``True`` when this path lives under *prefix*.

        The *prefix* is normalized (trailing slash stripped) so both
        ``"/.mambo"`` and ``"/.mambo/"`` work identically.
        """
        prefix = prefix.rstrip("/")
        if prefix == "":
            return True  # root — everything is under "/"
        return self.value == prefix or self.value.startswith(prefix + "/")

    def relative_to(self, prefix: str) -> str:
        """Return the relative path fragment without leading ``'/'``.

        Raises :class:`ValueError` when this path is not under *prefix*.

        The *prefix* is normalized (trailing slash stripped) so both
        ``"/.mambo"`` and ``"/.mambo/"`` work identically.
        """
        prefix = prefix.rstrip("/")
        if self.value == prefix:
            return ""
        if self.value.startswith(prefix + "/"):
            return self.value[len(prefix) + 1:]
        raise ValueError(f"{self.value!r} is not under {prefix!r}")

    def endswith(self, suffix: str) -> bool:
        """Return ``True`` if this path's value ends with *suffix*."""
        return self.value.endswith(suffix)

    def join(self, *parts: str) -> VirtualPath:
        """Safely join sub-path segments (no ``..`` collapsing)."""
        new = self.value.rstrip("/") + "/" + "/".join(
            p.strip("/") for p in parts
        )
        return VirtualPath(new)


# ============================================================================
# File type classification
# ============================================================================


FileType = Literal["text", "image", "audio", "video", "file"]
"""Classification of a file by its extension for multimodal dispatch."""


# ============================================================================
# Value objects
# ============================================================================


class FileInfo(BaseModel):
    """Structured file / directory listing entry."""

    model_config = ConfigDict(frozen=True)

    path: VirtualPath
    """Absolute virtual path."""
    is_dir: bool = False
    size: int = 0
    """Size in bytes (0 for directories)."""
    modified_at: str = ""
    """ISO-8601 timestamp or empty string."""
    desc: str = ""
    """Optional human-readable description / summary for the file or directory."""


class GrepMatch(BaseModel):
    """A single match from a grep search."""

    model_config = ConfigDict(frozen=True)

    path: VirtualPath
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
                desc_part = f"  -- {fi.desc.replace(chr(10), ' ')}" if fi.desc else ""
                if fi.is_dir:
                    lines.append(f"{fi.path}/{desc_part}")
                else:
                    lines.append(f"{fi.path}({human_size(fi.size)}){desc_part}")
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
    path: VirtualPath | None = None

    def __str__(self) -> str:
        if self.error is not None:
            return f"Error: {self.error}"
        return f"File written: {self.path}"


class EditResult(BaseModel):
    """Result from ``edit()`` — replace text in an existing file."""

    error: str | None = None
    path: VirtualPath | None = None
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
            for fi in self.matches:
                desc_part = f"  -- {fi.desc.replace(chr(10), ' ')}" if fi.desc else ""
                lines.append(f"{fi.path}({human_size(fi.size)}){desc_part}")
        if not lines:
            return "No files found."
        return "\n".join(lines)


class UploadFileResult(BaseModel):
    """Result for a single file in a bulk upload."""

    path: VirtualPath
    error: str | None = None


class DownloadFileResult(BaseModel):
    """Result for a single file in a bulk download."""

    path: VirtualPath
    content: bytes | None = None
    error: str | None = None
