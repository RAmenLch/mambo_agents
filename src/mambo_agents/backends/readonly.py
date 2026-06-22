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
            tool_timeouts=backend._tool_timeouts,
        )
        self._backend = backend
        self._allowed = frozenset(allowed_extra_tools)
        self.workspace_root = backend.workspace_root

    # ------------------------------------------------------------------
    # Pass-through read-only ops — catch exceptions from underlying
    # backend so the review agent can recover gracefully.
    # ------------------------------------------------------------------

    def ls(self, path: str) -> LsResult:
        try:
            return self._backend.ls(path)
        except Exception as exc:
            return LsResult(error=f"{type(exc).__name__}: {exc}")

    def read_raw(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
        include_line_numbers: bool = False,
    ) -> ReadResult:
        try:
            return self._backend.read_raw(file_path, offset, limit, include_line_numbers)
        except Exception as exc:
            return ReadResult(error=f"{type(exc).__name__}: {exc}")

    def grep(
        self,
        pattern: str,
        path: str = "/workspace",
        glob: str | None = None,
        regex: bool = False,
        offset: int = 0,
        limit: int | None = None,
    ) -> GrepResult:
        try:
            return self._backend.grep(pattern, path, glob, regex, offset, limit)
        except Exception as exc:
            return GrepResult(error=f"{type(exc).__name__}: {exc}")

    def glob(self, pattern: str, path: str = "/workspace") -> GlobResult:
        try:
            return self._backend.glob(pattern, path)
        except Exception as exc:
            return GlobResult(error=f"{type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------
    # Rejected write ops
    # ------------------------------------------------------------------

    def write(
        self, file_path: str, content: str, overwrite: bool = False,
    ) -> WriteResult:
        return WriteResult(
            error="Write denied: audit backend is read-only.",
            path=file_path,
        )

    def edit(
        self,
        file_path: str,
        old_str: str,
        new_str: str,
        *,
        replace_all: bool = False,
    ) -> EditResult:
        return EditResult(
            error="Edit denied: audit backend is read-only.",
            path=file_path,
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
    def description(self) -> str:
        base = self._backend.description
        return (
            f"{base} [read-only mode — write/edit/execute blocked, "
            f"allowed extras: {sorted(self._allowed) or 'none'}]"
        )
