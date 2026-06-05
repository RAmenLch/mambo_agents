"""TempWorkspaceBackend — dual-backend routing: StateBackend for /.mambo/ + delegate for rest.

Routes file operations based on path prefix:

- ``/.mambo/`` and all sub-paths → internal :class:`StateBackend` (virtual, isolated, checkpointable)
- Everything else → delegate ``_backend`` (e.g. :class:`LocalBackend` for real files)

The AI is told via the system prompt that ``/.mambo/`` is a scratchpad for:

- Middleware file storage (large result eviction, chat history dumps)
- Agent-internal scratch files
- Subagent communication files

The system prompt explicitly lists the core tools (``ls``, ``read``, ``write``,
``edit``, ``grep``, ``glob``) as the **only** tools that can operate on
``/.mambo/`` paths.  When ``execute`` is enabled on the delegate backend,
the prompt also explains the virtual-to-real path mapping — ``/`` maps
to the delegate's working directory, and ``execute`` commands must use
real filesystem paths, never ``/.mambo/`` paths.
"""

from __future__ import annotations

from langchain_core.tools import StructuredTool

from mambo_agents.backends.protocol import (
    BackendProtocol,
    DownloadFileResult,
    EditResult,
    GlobResult,
    GrepResult,
    LsResult,
    ReadResult,
    ReadSummarizer,
    UploadFileResult,
    WriteResult,
)
from mambo_agents.backends.state import StateBackend

# ---------------------------------------------------------------------------
# Default workspace prefix
# ---------------------------------------------------------------------------

DEFAULT_WORKSPACE_PREFIX = "/.mambo/"
"""Default path prefix for the virtual scratchpad managed by StateBackend."""


# ============================================================================
# TempWorkspaceBackend
# ============================================================================


class TempWorkspaceBackend(BackendProtocol):
    """Dual-backend that routes ``/.mambo/`` paths to StateBackend, rest to delegate.

    Parameters:
        backend:
            Delegate for all files *outside* the workspace prefix
            (e.g. :class:`LocalBackend`).
        workspace_prefix:
            Path prefix managed by the internal :class:`StateBackend`.
            Default ``"/.mambo/"``.
        custom_description:
            Extra text appended to the system-prompt description.
    """

    def __init__(
        self,
        backend: BackendProtocol,
        workspace_prefix: str = DEFAULT_WORKSPACE_PREFIX,
        custom_description: str | None = None,
        *,
        max_read_chars: int = 100_000,
        summarizer: "ReadSummarizer | None" = None,
    ) -> None:
        super().__init__(max_read_chars=max_read_chars, summarizer=summarizer)
        self._backend = backend
        self._state = StateBackend(max_read_chars=max_read_chars, summarizer=summarizer)
        self._prefix = workspace_prefix.rstrip("/") + "/"
        self._custom_description = custom_description

    # ------------------------------------------------------------------
    # tools — delegate to _backend
    # ------------------------------------------------------------------

    @property
    def tools(self) -> list[StructuredTool]:
        return self._backend.tools

    # ------------------------------------------------------------------
    # description — system-prompt injection
    # ------------------------------------------------------------------

    @property
    def description(self) -> str:
        core_tools = (
            "`ls`, `read`, `write`, `edit`, `grep`, `glob`"
        )
        base = (
            f"A virtual temporary workspace is available at **{self._prefix}**.  "
            f"Use it for intermediate files, chat history dumps, large "
            f"tool-result evictions, and subagent communication.  "
            f"This workspace is isolated from the real filesystem — files here "
            f"do not persist across sessions.  "
            f"Only the following core file tools work on **{self._prefix}** paths: "
            f"{core_tools}.  "
            f"Other tools (tree, delete, execute, etc.) must NOT target "
            f"**{self._prefix}** paths."
        )

        # Delegate backend's own description (path mapping, etc.)
        delegate_desc = self._backend.description
        if delegate_desc:
            base += f"\n\n{delegate_desc}"

        if self._custom_description:
            base = self._custom_description + "\n\n" + base
        return base

    # ------------------------------------------------------------------
    # Path routing
    # ------------------------------------------------------------------

    def _is_workspace(self, path: str) -> bool:
        """Return ``True`` if *path* falls under the workspace prefix."""
        p = path.rstrip("/")
        prefix_clean = self._prefix.rstrip("/")
        return p == prefix_clean or p.startswith(prefix_clean + "/")

    def _route(self, path: str) -> tuple[BackendProtocol, str]:
        """Resolve ``(target_backend, path)`` for routing.

        Workspace paths → :attr:`_state`; everything else → :attr:`_backend`.
        """
        if self._is_workspace(path):
            return self._state, path
        return self._backend, path

    # ------------------------------------------------------------------
    # Core file operations — prefix-routed
    # ------------------------------------------------------------------

    def ls(self, path: str) -> LsResult:
        target, p = self._route(path)
        return target.ls(p)

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
        include_line_numbers: bool = False,
        *,
        _apply_max_chars: bool = True,
    ) -> ReadResult:
        target, p = self._route(file_path)
        return target.read(p, offset, limit, include_line_numbers, _apply_max_chars=_apply_max_chars)

    def read_raw(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
        include_line_numbers: bool = False,
    ) -> ReadResult:
        target, p = self._route(file_path)
        return target.read_raw(p, offset, limit, include_line_numbers)

    def write(
        self, file_path: str, content: str, overwrite: bool = False,
    ) -> WriteResult:
        target, p = self._route(file_path)
        return target.write(p, content, overwrite)

    def edit(
        self,
        file_path: str,
        old_str: str,
        new_str: str,
        *,
        replace_all: bool = False,
    ) -> EditResult:
        target, p = self._route(file_path)
        return target.edit(p, old_str, new_str, replace_all=replace_all)

    def grep(
        self,
        pattern: str,
        path: str = "/",
        glob: str | None = None,
    ) -> GrepResult:
        target, p = self._route(path)
        return target.grep(pattern, p, glob)

    def glob(self, pattern: str, path: str = "/") -> GlobResult:
        target, p = self._route(path)
        return target.glob(pattern, p)

    # ------------------------------------------------------------------
    # Developer API — upload / download (split by prefix)
    # ------------------------------------------------------------------

    def upload_files(
        self,
        files: list[tuple[str, bytes]],
        *,
        thread_id: str | None = None,
    ) -> list[UploadFileResult]:
        """Upload files, routing each to the correct backend by prefix."""
        ws_files: list[tuple[str, bytes]] = []
        be_files: list[tuple[str, bytes]] = []
        index_map: list[tuple[bool, int]] = []  # (is_workspace, original_index)

        for i, (p, data) in enumerate(files):
            if self._is_workspace(p):
                ws_files.append((p, data))
                index_map.append((True, len(ws_files) - 1))
            else:
                be_files.append((p, data))
                index_map.append((False, len(be_files) - 1))

        # Collect results
        ws_results: list[UploadFileResult] = []
        be_results: list[UploadFileResult] = []

        if ws_files:
            ws_results = list(self._state.upload_files(ws_files, thread_id=thread_id))
        if be_files:
            be_results = list(self._backend.upload_files(be_files))

        # Reassemble in original order
        results: list[UploadFileResult] = []
        ws_idx = 0
        be_idx = 0
        for is_ws, _ in index_map:
            if is_ws:
                results.append(ws_results[ws_idx])
                ws_idx += 1
            else:
                results.append(be_results[be_idx])
                be_idx += 1
        return results

    def download_files(
        self,
        paths: list[str],
        *,
        thread_id: str | None = None,
    ) -> list[DownloadFileResult]:
        """Download files, routing each to the correct backend by prefix."""
        ws_paths: list[str] = []
        be_paths: list[str] = []
        index_map: list[tuple[bool, int]] = []

        for i, p in enumerate(paths):
            if self._is_workspace(p):
                ws_paths.append(p)
                index_map.append((True, len(ws_paths) - 1))
            else:
                be_paths.append(p)
                index_map.append((False, len(be_paths) - 1))

        ws_results: list[DownloadFileResult] = []
        be_results: list[DownloadFileResult] = []

        if ws_paths:
            ws_results = list(self._state.download_files(ws_paths, thread_id=thread_id))
        if be_paths:
            be_results = list(self._backend.download_files(be_paths))

        results: list[DownloadFileResult] = []
        ws_idx = 0
        be_idx = 0
        for is_ws, _ in index_map:
            if is_ws:
                results.append(ws_results[ws_idx])
                ws_idx += 1
            else:
                results.append(be_results[be_idx])
                be_idx += 1
        return results
