"""Domain data types for the virtual file-system backend.

Plain Pydantic models and value objects — zero business logic.
Safe to import from anywhere (utils, protocol, backends, middleware)
without circular-dependency risk.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import Literal, Self

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
    """Raise :class:`BackendError` if *path* contains ``".."`` or ``"//"``.

    This check MUST run on the **raw, un-normalized** path string —
    **before** passing it through :class:`~pathlib.PurePosixPath` or any
    other normalizer.  Normalizers like ``PurePosixPath`` silently
    collapse ``".."`` segments, which would defeat a subsequent prefix
    check.

    Args:
        path: The raw path string to check.
        name: Human-readable name for the parameter (used in error messages).

    Raises:
        BackendError: If *path* contains ``".."`` path traversal or ``"//"``
            double slashes.  The raw *path* is **not** embedded in the
            message — only the AI-facing ``path`` field when the input is
            a valid absolute path.
    """
    # Split on "/" separates the raw segments without collapsing ".."
    if ".." in path.split("/"):
        raise BackendError(
            code=ErrorCode.PATH_TRAVERSAL,
            message=f"{name} 不能包含 '..' 路径穿越",
        )
    if "//" in path:
        raise BackendError(
            code=ErrorCode.PATH_DOUBLE_SLASH,
            message=f"{name} 不能包含 '//'",
        )


def _normalize_path_str(value: str) -> str:
    """Validate + normalize a path string for :class:`VirtualPath` construction.

    Raises :class:`BackendError` if the path is empty, non-absolute,
    contains ``..`` traversal, ``//``, or is the root directory ``"/"``
    (trailing slash is preserved — ``"/workspace/"`` is valid,
    ``"/"`` is not).

    Returns the validated path string as-is.
    """
    check_no_path_traversal(value, name="path")
    s = value.strip()
    if not s:
        raise BackendError(code=ErrorCode.PATH_EMPTY, message="路径不能为空")
    if not s.startswith("/"):
        raise BackendError(
            code=ErrorCode.PATH_NOT_ABSOLUTE,
            message="路径必须以 '/' 开头",
        )
    # Reject root — but "/workspace/" with trailing slash is fine
    if s.rstrip("/") == "":
        raise BackendError(
            code=ErrorCode.PATH_IS_ROOT,
            message="路径不能是根目录 '/'；请使用子目录如 '/workspace'",
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
    ``write()`` reject paths ending in ``"/"``.

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

    # ``extra="forbid"`` ensures stray keyword arguments (e.g. ``limit`` /
    # ``offset`` mistakenly passed alongside ``value``) fail loudly instead of
    # being silently ignored — which previously caused read/write to run with
    # default arguments after the extra fields vanished.
    model_config = ConfigDict(frozen=True, extra="forbid")

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
            # Reject any stray keys — only ``value`` is allowed.  Silently
            # dropping extras (the default pydantic behaviour combined with
            # this validator) let caller errors like ``limit`` / ``offset``
            # sneak through and degrade tool calls to their defaults.
            extras = [k for k in data if k != "value"]
            if extras:
                raise ValueError(
                    f"VirtualPath does not accept extra fields: {', '.join(map(str, extras))}. "
                    f"Only 'value' is allowed (got keys: {', '.join(map(str, data))})."
                )
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

        Raises :class:`BackendError` when this path is not under *prefix*.

        The *prefix* is normalized (trailing slash stripped) so both
        ``"/.mambo"`` and ``"/.mambo/"`` work identically.
        """
        prefix = prefix.rstrip("/")
        if self.value == prefix:
            return ""
        if self.value.startswith(prefix + "/"):
            return self.value[len(prefix) + 1:]
        raise BackendError(
            code=ErrorCode.PATH_NOT_UNDER,
            path=self,
            message=f"路径不在 '{prefix}/' 下",
        )

    def endswith(self, suffix: str) -> bool:
        """Return ``True`` if this path's value ends with *suffix*."""
        return self.value.endswith(suffix)

    def join(self, *parts: str) -> VirtualPath:
        """Safely join sub-path segments (no ``..`` collapsing)."""
        new = self.value.rstrip("/") + "/" + "/".join(
            p.strip("/") for p in parts
        )
        return VirtualPath(new)

    def as_dir(self) -> VirtualPath:
        """Return a copy with trailing ``"/"`` to mark as a directory."""
        if self.value.endswith("/"):
            return self
        return VirtualPath(self.value + "/")

    @property
    def name(self) -> str:
        """Final path component (like :attr:`pathlib.PurePosixPath.name`)."""
        return self.normalized.rsplit("/", 1)[-1]

    @property
    def parent(self) -> VirtualPath:
        """Parent directory (like :attr:`pathlib.PurePosixPath.parent`).

        Returns *self* when already at workspace root (``VirtualPath("/")``
        is not valid in this system)."""
        normalized = self.normalized
        parent_str = normalized.rsplit("/", 1)[0]
        if not parent_str:
            # "/workspace" → parent would be "/" → return self
            return self
        return VirtualPath(parent_str)


# ============================================================================
# File type classification
# ============================================================================


FileType = Literal["text", "image", "audio", "video", "file"]
"""Classification of a file by its extension for multimodal dispatch."""


# ============================================================================
# ErrorCode — machine-readable error classification
# ============================================================================


class ErrorCode(str, Enum):
    """Structured error codes for :class:`BackendError`.

    Replaces ad-hoc string matching (``"file_not_found"``, ``"not found"``
    substring checks, etc.) with typed, machine-readable codes.
    """

    # -- 路径验证 ----------------------------------------------------------
    PATH_TRAVERSAL = "PATH_TRAVERSAL"
    """Path contains ``".."`` traversal segments."""
    PATH_DOUBLE_SLASH = "PATH_DOUBLE_SLASH"
    """Path contains ``"//"``."""
    PATH_NOT_ABSOLUTE = "PATH_NOT_ABSOLUTE"
    """Path does not start with ``"/"``."""
    PATH_IS_ROOT = "PATH_IS_ROOT"
    """Path is ``"/"`` (the root, which is forbidden)."""
    PATH_EMPTY = "PATH_EMPTY"
    """Path is empty or whitespace-only."""
    PATH_NOT_UNDER = "PATH_NOT_UNDER"
    """Path is not under the required prefix."""

    # -- 工作区边界 --------------------------------------------------------
    OUTSIDE_WORKSPACE = "OUTSIDE_WORKSPACE"
    """Path is outside the configured workspace root."""
    SYMLINK_ESCAPE = "SYMLINK_ESCAPE"
    """Symlink resolves outside the working directory."""

    # -- 文件操作 ----------------------------------------------------------
    NOT_FOUND = "NOT_FOUND"
    """File or directory does not exist."""
    IS_DIR = "IS_DIR"
    """Target is a directory when a file was expected."""
    NOT_DIR = "NOT_DIR"
    """Target is a file when a directory was expected."""
    ALREADY_EXISTS = "ALREADY_EXISTS"
    """File already exists and overwrite was not requested."""
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    """File exceeds the configured size limit."""

    # -- 编辑 --------------------------------------------------------------
    EDIT_NOT_ALLOWED = "EDIT_NOT_ALLOWED"
    """Path is blocked by edit whitelist / blacklist."""
    OLD_STR_NOT_FOUND = "OLD_STR_NOT_FOUND"
    """The ``old_str`` was not found in the file."""
    MULTI_OCCURRENCES = "MULTI_OCCURRENCES"
    """``old_str`` appears more than once and ``replace_all`` was not set."""

    # -- IO / 系统 ---------------------------------------------------------
    IO_ERROR = "IO_ERROR"
    """Generic filesystem I/O error."""
    OS_ERROR = "OS_ERROR"
    """Operating-system-level error (permissions, disk full, etc.)."""
    INVALID = "INVALID"
    """Catch-all for errors that don't fit a specific code."""


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
# BackendError — unified structured error (Model + Exception)
# ============================================================================


class BackendError(Exception):
    """Unified structured error — raiseable AND storable as ``Result.error``.

    Inherits :class:`Exception` so it can be raised / caught, and stores
    typed attributes (``code``, ``path``, ``message``) so the hybrid
    router can reverse-translate the path via :meth:`apply_reverse_translation`.

    Replaces three string-based error mechanisms:

    - ``ValueError`` from :func:`check_no_path_traversal` / etc.
    - ``WorkspacePathError`` from backend ``_resolve()``
    - ``str | None`` in ``Result.error``

    Attributes:
        code: Machine-readable :class:`ErrorCode`.
        path: The offending :class:`VirtualPath`, or ``None``.
        message: Human-readable description — does **not** embed raw paths.
    """

    def __init__(
        self,
        code: ErrorCode,
        *,
        path: VirtualPath | None = None,
        message: str,
    ) -> None:
        self.code: ErrorCode = code
        self.path: VirtualPath | None = path
        self.message: str = message
        super().__init__(message)

    def __str__(self) -> str:
        return f"[{self.code.value}] {self.message}"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, BackendError):
            return self.code == other.code and self.path == other.path and self.message == other.message
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.code, self.path, self.message))

    def apply_reverse_translation(
        self,
        reverse_fn: "ReversePathFn",
        target_ws_root: VirtualPath,
        virtual_prefix: VirtualPath,
    ) -> "BackendError":
        """Reverse-translate the *path* field through the hybrid boundary."""
        if self.path is None:
            return self
        return BackendError(
            code=self.code,
            path=reverse_fn(self.path, target_ws_root, virtual_prefix),
            message=self.message,
        )


# ============================================================================
# Result — base class for all tool result types
# ============================================================================

ReversePathFn = Callable[[VirtualPath, VirtualPath, VirtualPath], VirtualPath]
"""Signature of ``_reverse_path(internal_path, target_ws_root, virtual_prefix)``."""


class Result(BaseModel):
    """Base class for all tool results.

    Subclasses that contain translatable ``VirtualPath`` fields override
    :meth:`apply_reverse_translation` to return a new instance with
    all path fields reverse-translated.

    The default implementation handles the ``error`` field (a
    :class:`BackendError` whose ``path`` is also reverse-translated)
    and returns *self* unchanged otherwise.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    error: BackendError | None = None
    """Structured error, or ``None`` on success.

    :class:`BackendError` carries a machine-readable :class:`ErrorCode`,
    the offending :class:`VirtualPath` (if applicable), and a human-readable
    ``message`` that never embeds raw filesystem paths.
    """

    def apply_reverse_translation(
        self, reverse_fn: ReversePathFn, target_ws_root: VirtualPath, virtual_prefix: VirtualPath,
    ) -> Self:
        """Return a new instance with ``VirtualPath`` fields reverse-translated.

        The default implementation reverse-translates the ``error`` field
        (if set) and returns *self* unchanged when there is no error.
        Subclasses with additional path fields override, call ``super()``
        first, then apply their own translations.
        """
        if self.error is None:
            return self
        return self.model_copy(update={
            "error": self.error.apply_reverse_translation(
                reverse_fn, target_ws_root, virtual_prefix,
            ),
        })


# ============================================================================
# Concrete result types
# ============================================================================


class LsResult(Result):
    """Result from ``ls()``."""

    entries: list[FileInfo] | None = None

    def apply_reverse_translation(
        self, reverse_fn: ReversePathFn, target_ws_root: VirtualPath, virtual_prefix: VirtualPath,
    ) -> "LsResult":
        result = super().apply_reverse_translation(reverse_fn, target_ws_root, virtual_prefix)
        if self.entries is None:
            return result
        entries = [
            e.model_copy(update={"path": reverse_fn(e.path, target_ws_root, virtual_prefix)})
            for e in self.entries
        ]
        return result.model_copy(update={"entries": entries})

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


class ReadResult(Result):
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


class WriteResult(Result):
    """Result from ``write()`` — create or overwrite a file.

    By default ``overwrite=False`` and creating a file that already
    exists is an error.  Set ``overwrite=True`` to replace the file
    contents entirely.
    """

    path: VirtualPath | None = None

    def apply_reverse_translation(
        self, reverse_fn: ReversePathFn, target_ws_root: VirtualPath, virtual_prefix: VirtualPath,
    ) -> "WriteResult":
        result = super().apply_reverse_translation(reverse_fn, target_ws_root, virtual_prefix)
        if self.path is None:
            return result
        return result.model_copy(update={"path": reverse_fn(self.path, target_ws_root, virtual_prefix)})

    def __str__(self) -> str:
        if self.error is not None:
            return f"Error: {self.error}"
        return f"File written: {self.path}"


class EditResult(Result):
    """Result from ``edit()`` — replace text in an existing file."""

    path: VirtualPath | None = None
    occurrences: int = 0

    def apply_reverse_translation(
        self, reverse_fn: ReversePathFn, target_ws_root: VirtualPath, virtual_prefix: VirtualPath,
    ) -> "EditResult":
        result = super().apply_reverse_translation(reverse_fn, target_ws_root, virtual_prefix)
        if self.path is None:
            return result
        return result.model_copy(update={"path": reverse_fn(self.path, target_ws_root, virtual_prefix)})

    def __str__(self) -> str:
        if self.error is not None:
            return f"Error: {self.error}"
        return f"File edited: {self.path} ({self.occurrences} replacement(s))"


class GrepResult(Result):
    """Result from ``grep()``."""

    matches: list[GrepMatch] | None = None
    truncated: bool = False
    """``True`` when the result was truncated by ``max_grep_matches`` or offset/limit."""
    total_matches: int = 0
    """Total number of matches found before offset/limit slicing."""

    def apply_reverse_translation(
        self, reverse_fn: ReversePathFn, target_ws_root: VirtualPath, virtual_prefix: VirtualPath,
    ) -> "GrepResult":
        result = super().apply_reverse_translation(reverse_fn, target_ws_root, virtual_prefix)
        if self.matches is None:
            return result
        matches = [
            m.model_copy(update={"path": reverse_fn(m.path, target_ws_root, virtual_prefix)})
            for m in self.matches
        ]
        return result.model_copy(update={"matches": matches})

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


class GlobResult(Result):
    """Result from ``glob()``."""

    matches: list[FileInfo] | None = None

    def apply_reverse_translation(
        self, reverse_fn: ReversePathFn, target_ws_root: VirtualPath, virtual_prefix: VirtualPath,
    ) -> "GlobResult":
        result = super().apply_reverse_translation(reverse_fn, target_ws_root, virtual_prefix)
        if self.matches is None:
            return result
        matches = [
            e.model_copy(update={"path": reverse_fn(e.path, target_ws_root, virtual_prefix)})
            for e in self.matches
        ]
        return result.model_copy(update={"matches": matches})

    def __str__(self) -> str:
        lines: list[str] = []
        if self.error is not None:
            lines.append(f"Warning: {self.error}")
        if self.matches:
            for fi in self.matches:
                desc_part = f"  -- {fi.desc.replace(chr(10), ' ')}" if fi.desc else ""
                if fi.is_dir:
                    lines.append(f"{fi.path}/{desc_part}")
                else:
                    lines.append(f"{fi.path}({human_size(fi.size)}){desc_part}")
        if not lines:
            return "No matches found."
        return "\n".join(lines)


class UploadFileResult(Result):
    """Result for a single file in a bulk upload."""

    path: VirtualPath

    def apply_reverse_translation(
        self, reverse_fn: ReversePathFn, target_ws_root: VirtualPath, virtual_prefix: VirtualPath,
    ) -> "UploadFileResult":
        result = super().apply_reverse_translation(reverse_fn, target_ws_root, virtual_prefix)
        return result.model_copy(update={"path": reverse_fn(self.path, target_ws_root, virtual_prefix)})


class DownloadFileResult(Result):
    """Result for a single file in a bulk download."""

    path: VirtualPath
    content: bytes | None = None

    def apply_reverse_translation(
        self, reverse_fn: ReversePathFn, target_ws_root: VirtualPath, virtual_prefix: VirtualPath,
    ) -> "DownloadFileResult":
        result = super().apply_reverse_translation(reverse_fn, target_ws_root, virtual_prefix)
        return result.model_copy(update={"path": reverse_fn(self.path, target_ws_root, virtual_prefix)})


class DeleteResult(Result):
    """Result from ``delete()`` — delete a single file."""

    path: VirtualPath | None = None

    def apply_reverse_translation(
        self, reverse_fn: ReversePathFn, target_ws_root: VirtualPath, virtual_prefix: VirtualPath,
    ) -> "DeleteResult":
        result = super().apply_reverse_translation(reverse_fn, target_ws_root, virtual_prefix)
        if self.path is None:
            return result
        return result.model_copy(update={"path": reverse_fn(self.path, target_ws_root, virtual_prefix)})

    def __str__(self) -> str:
        if self.error is not None:
            return f"Error: {self.error}"
        return f"Deleted: {self.path}"
