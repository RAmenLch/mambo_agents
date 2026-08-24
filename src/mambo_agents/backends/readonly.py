"""Read-only backend — wraps any ``BackendProtocol`` and strips write/destructive ops.

Exposes only safe, read-only operations (``ls``, ``read``, ``grep``,
``glob``) while rejecting ``write``, ``edit``, and any backend-specific
destructive tools (``delete``, ``execute``, …).

Extra backend tools can be selectively enabled via *allowed_extra_tools*
(e.g. ``"tree"`` is safe, ``"execute"`` is not).
"""

from __future__ import annotations

from typing import Iterable

from langchain_core.tools import StructuredTool

from mambo_agents.backends.protocol import (
    BackendProtocol,
    EditResult,
    GlobResult,
    GrepResult,
    LsResult,
    ReadResult,
    WriteResult,
)
from mambo_agents.backends.schemas import BackendError, ErrorCode, VirtualPath


class ReadOnlyBackend(BackendProtocol):
    """Thin wrapper that makes any backend read-only.

    Delegates ``ls``, ``read_raw``, ``grep``, ``glob`` to the wrapped
    *backend*.  ``write`` and ``edit`` always return an error.
    Destructive extra tools (``delete``, ``execute``, …) are excluded
    from ``tools`` by default — pass *allowed_extra_tools* to
    selectively include safe extras like ``tree``.

    Parameters
    ----------
    backend:
        The underlying backend to wrap.
    allowed_extra_tools:
        Names of extra tools from ``backend.tools`` that are safe to
        expose (e.g. ``"tree"``, ``"copy"``).  Tools not in this list
        are excluded.  Defaults to an empty frozenset (no extras).
    """

    def __init__(
        self,
        backend: BackendProtocol,
        *,
        allowed_extra_tools: Iterable[str] = frozenset(),
    ) -> None:
        super().__init__(
            max_read_chars=backend._max_read_chars,
            summarizer=backend._summarizer,
            multimodal_describer=getattr(backend, "_multimodal_describer", None),
            tool_timeouts=backend._tool_timeouts,
        )
        self._backend = backend
        self._allowed = frozenset(allowed_extra_tools)
        self.workspace_root = backend.workspace_root

    # ------------------------------------------------------------------
    # Pass-through read-only ops — catch exceptions from underlying
    # backend so the review agent can recover gracefully.
    # ------------------------------------------------------------------

    def ls(self, path: VirtualPath) -> LsResult:
        try:
            return self._backend.ls(path)
        except BackendError as exc:
            return LsResult(error=exc)
        except Exception as exc:
            return LsResult(error=BackendError(code=ErrorCode.INVALID, message=f"{type(exc).__name__}: {exc}"))

    def read_raw(
        self,
        file_path: VirtualPath,
        offset: int = 0,
        limit: int | None = 2000,
        include_line_numbers: bool = False,
    ) -> ReadResult:
        try:
            return self._backend.read_raw(file_path, offset, limit, include_line_numbers)
        except BackendError as exc:
            return ReadResult(error=exc)
        except Exception as exc:
            return ReadResult(error=BackendError(code=ErrorCode.INVALID, message=f"{type(exc).__name__}: {exc}"))

    async def aread_raw(
        self,
        file_path: VirtualPath,
        offset: int = 0,
        limit: int | None = 2000,
        include_line_numbers: bool = False,
    ) -> ReadResult:
        try:
            return await self._backend.aread_raw(file_path, offset, limit, include_line_numbers)
        except BackendError as exc:
            return ReadResult(error=exc)
        except Exception as exc:
            return ReadResult(error=BackendError(code=ErrorCode.INVALID, message=f"{type(exc).__name__}: {exc}"))

    def grep(
        self,
        pattern: str,
        path: VirtualPath,
        glob: str | None = None,
        regex: bool = True,
        offset: int = 0,
        limit: int | None = None,
    ) -> GrepResult:
        try:
            return self._backend.grep(pattern, path, glob, regex, offset, limit)
        except BackendError as exc:
            return GrepResult(error=exc)
        except Exception as exc:
            return GrepResult(error=BackendError(code=ErrorCode.INVALID, message=f"{type(exc).__name__}: {exc}"))

    def glob(self, pattern: str, path: VirtualPath) -> GlobResult:
        try:
            return self._backend.glob(pattern, path)
        except BackendError as exc:
            return GlobResult(error=exc)
        except Exception as exc:
            return GlobResult(error=BackendError(code=ErrorCode.INVALID, message=f"{type(exc).__name__}: {exc}"))

    # ------------------------------------------------------------------
    # Rejected write ops
    # ------------------------------------------------------------------

    def write(
        self, file_path: VirtualPath, content: str, overwrite: bool = False,
    ) -> WriteResult:
        return WriteResult(
            error=BackendError(code=ErrorCode.EDIT_NOT_ALLOWED, path=file_path, message="写入被拒绝：后端是只读的"),
            path=file_path.value,
        )

    def edit(
        self,
        file_path: VirtualPath,
        old_str: str,
        new_str: str,
        *,
        replace_all: bool = False,
    ) -> EditResult:
        return EditResult(
            error=BackendError(code=ErrorCode.EDIT_NOT_ALLOWED, path=file_path, message="编辑被拒绝：后端是只读的"),
            path=file_path.value,
            occurrences=0,
        )

    # ------------------------------------------------------------------
    # Tools — only safe extras
    # ------------------------------------------------------------------

    @property
    def tools(self) -> list[StructuredTool]:
        return [
            t for t in self._backend.tools
            if t.name in self._allowed
        ]

    @property
    def path_mapping_info(self) -> dict[str, str]:
        return self._backend.path_mapping_info

    @property
    def description(self) -> str:
        base = self._backend.description
        return (
            f"{base} [read-only mode — write/edit/execute blocked, "
            f"allowed extras: {sorted(self._allowed) or 'none'}]"
        )
