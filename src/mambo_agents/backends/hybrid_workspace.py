"""HybridWorkspaceBackend — multi-backend routing inside /.mambo/ + one real backend.

Routes file operations based on path prefix:

- ``/.mambo/<name>/`` → named virtual workspace (StateBackend)
- ``/.mambo/``         → default virtual workspace (StateBackend, auto-created)
- Everything else      → real backend (e.g. LocalBackend for real files)

The default StateBackend at ``/.mambo/`` is always present.  Pass ``"."``
in *virtual_workspaces* to override it with a custom StateBackend instance.

The AI is told via the system prompt that ``/.mambo/`` is a virtual scratchpad:

- Middleware file storage (large result eviction, chat history dumps)
- Agent-internal scratch files
- Subagent communication files

The system prompt explicitly lists the core tools (``ls``, ``read``, ``write``,
``edit``, ``grep``, ``glob``) as the **only** tools that can operate on
``/.mambo/`` paths.  When ``execute`` is enabled on the real backend,
the prompt also explains the virtual-to-real path mapping.
"""

from __future__ import annotations

import asyncio

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

from mambo_agents.backends.protocol import (
    BackendProtocol,
    DownloadFileResult,
    EditResult,
    FileInfo,
    GlobResult,
    GrepResult,
    LsResult,
    ReadResult,
    ReadSummarizer,
    ThreadAwareWorkspace,
    UploadFileResult,
    WriteResult,
)
from mambo_agents.backends.state import StateBackend
from mambo_agents.backends.utils import validate_canonical_path

# ---------------------------------------------------------------------------
# Default workspace prefix
# ---------------------------------------------------------------------------

DEFAULT_MAMBO_PREFIX = "/.mambo/"
"""Default path prefix for the virtual scratchpad managed by StateBackend."""


# ============================================================================
# CopyResult
# ============================================================================


class CopyResult(BaseModel):
    """Result from ``copy()`` — single-file copy, potentially cross-backend."""

    error: str | None = None
    source: str | None = None
    destination: str | None = None

    def __str__(self) -> str:
        if self.error is not None:
            return f"Error: {self.error}"
        return f"Copied: {self.source} -> {self.destination}"


# ============================================================================
# HybridWorkspaceBackend
# ============================================================================


class HybridWorkspaceBackend(ThreadAwareWorkspace):
    """Multi-backend router: 1 real + N virtual workspaces under ``/.mambo/``.

    A default StateBackend is always mounted at ``/.mambo/``.  Additional
    named virtual workspaces live at ``/.mambo/<name>/``.  Everything outside
    ``/.mambo/`` routes to *real_backend*.

    Virtual workspaces only support the 6 core protocol tools
    (``ls``, ``read``, ``write``, ``edit``, ``grep``, ``glob``).

    Parameters:
        real_backend:
            The one real backend for paths outside ``/.mambo/``
            (e.g. :class:`LocalBackend`).
        virtual_workspaces:
            Named virtual workspaces.  Key ``"."`` overrides the default
            ``/.mambo/`` StateBackend; other keys create
            ``/.mambo/<name>/`` namespaces.  Each value is a
            :class:`BackendProtocol` (typically :class:`StateBackend`).
            Virtual backends see their workspace as rooted at ``"/"`` —
            the ``/.mambo/<name>/`` prefix is stripped before delegation.
        mambo_prefix:
            Shared prefix for all virtual workspaces.  Default ``"/.mambo/"``.
        workspace_root:
            AI-facing namespace root for real-filesystem paths.
            Default ``"/workspace"``.  All file operations targeting the
            real backend must use paths under this prefix.  Hybrid strips
            this prefix and prepends the real backend's ``workspace_root``
            before delegating — the two can differ freely.
        custom_description:
            Extra text appended to the system-prompt description.
    """

    def __init__(
        self,
        real_backend: BackendProtocol,
        virtual_workspaces: dict[str, BackendProtocol] | None = None,
        mambo_prefix: str = DEFAULT_MAMBO_PREFIX,
        custom_description: str | None = None,
        *,
        workspace_root: str = "/workspace",
        max_read_chars: int = 100_000,
        summarizer: "ReadSummarizer | None" = None,
    ) -> None:
        super().__init__(max_read_chars=max_read_chars, summarizer=summarizer)

        self.workspace_root = validate_canonical_path(workspace_root, "workspace_root")
        self._real = real_backend
        self._prefix = validate_canonical_path(mambo_prefix, "mambo_prefix")  # "/.mambo"
        self._custom_description = custom_description

        # --- Build virtual workspaces ---------------------------------
        vw = dict(virtual_workspaces or {})

        # "." key → override default /.mambo/ StateBackend
        if "." in vw:
            self._default_mambo = vw.pop(".")
        else:
            self._default_mambo = StateBackend(
                max_read_chars=max_read_chars, summarizer=summarizer,
            )

        # Remaining entries → /.mambo/<name>/ namespaces
        self._virtual: dict[str, BackendProtocol] = {}
        for name, be in vw.items():
            name = name.strip("/")
            if not name:
                raise ValueError(f"Virtual workspace name must be non-empty, got {name!r}")
            if name == ".":
                raise ValueError(
                    "Use '.' only to override the default workspace; "
                    "it cannot appear alongside other names."
                )
            if not isinstance(be, BackendProtocol):
                raise TypeError(
                    f"Virtual workspace '{name}' must be a BackendProtocol instance, "
                    f"got {type(be).__name__}"
                )
            self._virtual[name] = be

    # ------------------------------------------------------------------
    # tools — real_backend extras + copy
    # ------------------------------------------------------------------

    @property
    def tools(self) -> list[StructuredTool]:
        return self._real.tools + [
            StructuredTool(
                name="copy",
                description=(
                    "Copy a single file from source to destination. "
                    "Supports cross-backend copies (e.g. from a virtual "
                    f"**{self._prefix}/** workspace to the real filesystem, "
                    "or between virtual workspaces). "
                    "Overwrites the destination if it already exists."
                ),
                args_schema=create_model(
                    "CopySchema",
                    source=(str, Field(description="Absolute source file path")),
                    destination=(str, Field(description="Absolute destination file path")),
                ),
                func=lambda source, destination: self.copy(source, destination),
                coroutine=lambda source, destination: self.acopy(source, destination),
            ),
        ]

    # ------------------------------------------------------------------
    # description — system-prompt injection
    # ------------------------------------------------------------------

    @property
    def description(self) -> str:
        wr = self.workspace_root
        core_tools = "`ls`, `read`, `write`, `edit`, `grep`, `glob`, `copy`"
        base = (
            f"A virtual temporary workspace is available at **{self._prefix}/**.  "
            f"Use it for intermediate files, chat history dumps, large "
            f"tool-result evictions, and subagent communication.  "
            f"This workspace is isolated from the real filesystem — files here "
            f"do not persist across sessions.  "
            f"All real-filesystem paths must start with **{wr}/**.  "
            f"The following file tools work on **{self._prefix}/** paths: "
            f"{core_tools}.  "
            f"`copy` can move files between the virtual workspace and the real "
            f"filesystem.  "
            f"Other tools (tree, delete, execute, etc.) must NOT target "
            f"**{self._prefix}/** paths."
        )

        delegate_desc = self._real.description
        if delegate_desc:
            base += f"\n\n{delegate_desc}"

        if self._custom_description:
            base = self._custom_description + "\n\n" + base
        return base

    # ------------------------------------------------------------------
    # Path routing
    # ------------------------------------------------------------------

    def _is_mambo(self, path: str) -> bool:
        """Return ``True`` if *path* falls under the mambo prefix."""
        p = path.rstrip("/")
        mambo = self._prefix  # "/.mambo"
        return p == mambo or p.startswith(mambo + "/")

    @staticmethod
    def _rewrite(normalized_path: str, strip_prefix: str, target_ws_root: str) -> str:
        """Strip *strip_prefix* from *normalized_path*, then prepend *target_ws_root*.

        Always rewrites: strip the prefix, prepend the target backend's
        ``workspace_root``.  No special cases — predictable regardless of
        what directory names the user chooses.

        Raises
        ------
        ValueError
            If *normalized_path* is not under *strip_prefix*.
        """
        if normalized_path == strip_prefix:
            rel = ""
        elif normalized_path.startswith(strip_prefix + "/"):
            rel = normalized_path[len(strip_prefix):].lstrip("/")
        else:
            raise ValueError(
                f"Cannot rewrite path {normalized_path!r}: "
                f"not under prefix {strip_prefix!r}"
            )
        ws = target_ws_root.rstrip("/")
        if not rel:
            return ws
        return ws + "/" + rel

    def _route(self, path: str) -> tuple[BackendProtocol, str]:
        """Resolve ``(target_backend, rewritten_path)`` for routing.

        Every path is rewritten: the relevant prefix (Hybrid workspace root
        or ``/.mambo/<name>/``) is stripped and the target backend's
        ``workspace_root`` is prepended.  This guarantees that **any**
        backend can be placed in any routing slot without implicit naming
        conventions.

        Routing priority:
        1. Named virtual workspace (``/.mambo/<name>/...``)
        2. Default virtual workspace (``/.mambo/...``)
        3. Real backend (everything else)
        """
        if not isinstance(path, str):
            raise TypeError(
                f"path must be a string, got {type(path).__name__}: {path!r}"
            )
        p = path.rstrip("/")
        mambo = self._prefix  # "/.mambo"

        # (1) Check named virtual workspaces
        for name, be in self._virtual.items():
            ns_prefix = f"{mambo}/{name}"
            if p == ns_prefix or p.startswith(ns_prefix + "/"):
                return be, self._rewrite(p, ns_prefix, be.workspace_root)

        # (2) Fallback: default /.mambo/
        if p == mambo or p.startswith(mambo + "/"):
            return self._default_mambo, self._rewrite(
                p, mambo, self._default_mambo.workspace_root,
            )

        # (3) Real backend — strip Hybrid.ws_root, prepend real.ws_root
        return self._real, self._rewrite(p, self.workspace_root, self._real.workspace_root)

    # ------------------------------------------------------------------
    # Reverse path translation for results (ls / grep / glob)
    # ------------------------------------------------------------------

    def _get_virtual_prefix(self, path: str) -> str:
        """Return the external prefix that *path* would be routed from.

        This is the mirror of :meth:`_route` — given the same *path*, it
        returns the prefix that ``_route`` strips before delegation.
        Used to reverse-translate internal paths in result objects.
        """
        p = path.rstrip("/")
        mambo = self._prefix  # "/.mambo"

        # (1) Named virtual workspace
        for name in self._virtual:
            ns_prefix = f"{mambo}/{name}"
            if p == ns_prefix or p.startswith(ns_prefix + "/"):
                return ns_prefix

        # (2) Default mambo
        if p == mambo or p.startswith(mambo + "/"):
            return mambo

        # (3) Real backend
        return self.workspace_root

    @staticmethod
    def _reverse_path(
        internal_path: str, target_ws_root: str, virtual_prefix: str,
    ) -> str:
        """Reverse a ``_rewrite``: strip *target_ws_root*, prepend *virtual_prefix*.

        Example::
            _reverse_path("/skill-a", "/", "/.mambo/skills")
            # → "/.mambo/skills/skill-a"
        """
        twsr = target_ws_root.rstrip("/")
        if internal_path == twsr:
            return virtual_prefix
        if internal_path.startswith(twsr + "/"):
            rel = internal_path[len(twsr) + 1:]
        elif internal_path.startswith(twsr):
            rel = internal_path[len(twsr):].lstrip("/")
        else:
            rel = internal_path.lstrip("/")
        if not rel:
            return virtual_prefix
        return virtual_prefix + "/" + rel

    def _valid_paths_description(self) -> str:
        """Build a human-readable description of all valid path prefixes,
        including the mapping from AI-facing workspace to real backend root.
        """
        parts = [f"'{self.workspace_root}/'（映射至真实路径 '{self._real.workspace_root}/'）"]
        parts.append(f"'{self._prefix}/'（虚拟临时工作区）")
        for name in sorted(self._virtual):
            parts.append(f"'{self._prefix}/{name}/'（虚拟工作区）")
        return "、".join(parts)

    # ------------------------------------------------------------------
    # Core file operations — prefix-routed
    # ------------------------------------------------------------------

    def ls(self, path: str) -> LsResult:
        try:
            target, p = self._route(path)
        except ValueError:
            return LsResult(error=f"路径 '{path}' 无效，仅可访问：{self._valid_paths_description()}")
        result = target.ls(p)

        entries: list[FileInfo] = []
        if result.entries:
            vprefix = self._get_virtual_prefix(path)
            twsr = target.workspace_root
            for e in result.entries:
                entries.append(e.model_copy(update={
                    "path": self._reverse_path(e.path, twsr, vprefix),
                }))

        # Inject named virtual workspace directories when listing /.mambo root
        if path.rstrip("/") == self._prefix:
            for name in self._virtual:
                entries.append(FileInfo(
                    path=f"{self._prefix}/{name}",
                    is_dir=True,
                ))

        return LsResult(
            error=result.error,
            entries=entries if entries else None,
        )

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
        include_line_numbers: bool = False,
        *,
        _apply_max_chars: bool = True,
    ) -> ReadResult:
        try:
            target, p = self._route(file_path)
        except ValueError:
            return ReadResult(error=f"路径 '{file_path}' 无效，仅可访问：{self._valid_paths_description()}")
        return target.read(p, offset, limit, include_line_numbers, _apply_max_chars=_apply_max_chars)

    def read_raw(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
        include_line_numbers: bool = False,
    ) -> ReadResult:
        try:
            target, p = self._route(file_path)
        except ValueError:
            return ReadResult(error=f"路径 '{file_path}' 无效，仅可访问：{self._valid_paths_description()}")
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
        # Fan-out to all virtual backends when searching /.mambo root
        if path.rstrip("/") == self._prefix:
            return self._grep_all_virtual(pattern, glob)

        try:
            target, p = self._route(path)
        except ValueError:
            return GrepResult(error=f"路径 '{path}' 无效，仅可访问：{self._valid_paths_description()}")
        result = target.grep(pattern, p, glob)
        if result.matches:
            vprefix = self._get_virtual_prefix(path)
            twsr = target.workspace_root
            result = GrepResult(
                error=result.error,
                matches=[
                    m.model_copy(update={
                        "path": self._reverse_path(m.path, twsr, vprefix),
                    })
                    for m in result.matches
                ],
            )
        return result

    def glob(self, pattern: str, path: str = "/") -> GlobResult:
        # Fan-out to all virtual backends when searching /.mambo root
        if path.rstrip("/") == self._prefix:
            return self._glob_all_virtual(pattern)

        try:
            target, p = self._route(path)
        except ValueError:
            return GlobResult(error=f"路径 '{path}' 无效，仅可访问：{self._valid_paths_description()}")
        result = target.glob(pattern, p)
        if result.matches:
            vprefix = self._get_virtual_prefix(path)
            twsr = target.workspace_root
            result = GlobResult(
                error=result.error,
                matches=[
                    e.model_copy(update={
                        "path": self._reverse_path(e.path, twsr, vprefix),
                    })
                    for e in result.matches
                ],
            )
        return result

    # ------------------------------------------------------------------
    # Async core file operations — route to target's async method
    # so that per-backend locks (e.g. SshBackend._async_lock) are
    # properly acquired.  Without these overrides the inherited
    # BackendProtocol.async methods call self.write() etc. on a
    # thread-pool thread, bypassing the target backend's async guards.
    # ------------------------------------------------------------------

    async def als(self, path: str) -> LsResult:
        target, p = self._route(path)
        result = await target.als(p)

        entries: list[FileInfo] = []
        if result.entries:
            vprefix = self._get_virtual_prefix(path)
            twsr = target.workspace_root
            for e in result.entries:
                entries.append(e.model_copy(update={
                    "path": self._reverse_path(e.path, twsr, vprefix),
                }))

        # Inject named virtual workspace directories when listing /.mambo root
        if path.rstrip("/") == self._prefix:
            for name in self._virtual:
                entries.append(FileInfo(
                    path=f"{self._prefix}/{name}",
                    is_dir=True,
                ))

        return LsResult(
            error=result.error,
            entries=entries if entries else None,
        )

    async def aread(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
        include_line_numbers: bool = False,
        *,
        _apply_max_chars: bool = True,
    ) -> ReadResult:
        target, p = self._route(file_path)
        return await target.aread(
            p, offset, limit, include_line_numbers,
            _apply_max_chars=_apply_max_chars,
        )

    async def awrite(
        self, file_path: str, content: str, overwrite: bool = False,
    ) -> WriteResult:
        target, p = self._route(file_path)
        return await target.awrite(p, content, overwrite)

    async def aedit(
        self,
        file_path: str,
        old_str: str,
        new_str: str,
        *,
        replace_all: bool = False,
    ) -> EditResult:
        target, p = self._route(file_path)
        return await target.aedit(p, old_str, new_str, replace_all=replace_all)

    async def agrep(
        self,
        pattern: str,
        path: str = "/",
        glob: str | None = None,
    ) -> GrepResult:
        # Fan-out to all virtual backends when searching /.mambo root
        if path.rstrip("/") == self._prefix:
            return await self._agrep_all_virtual(pattern, glob)

        target, p = self._route(path)
        result = await target.agrep(pattern, p, glob)
        if result.matches:
            vprefix = self._get_virtual_prefix(path)
            twsr = target.workspace_root
            result = GrepResult(
                error=result.error,
                matches=[
                    m.model_copy(update={
                        "path": self._reverse_path(m.path, twsr, vprefix),
                    })
                    for m in result.matches
                ],
            )
        return result

    async def aglob(self, pattern: str, path: str = "/") -> GlobResult:
        # Fan-out to all virtual backends when searching /.mambo root
        if path.rstrip("/") == self._prefix:
            return await self._aglob_all_virtual(pattern)

        target, p = self._route(path)
        result = await target.aglob(pattern, p)
        if result.matches:
            vprefix = self._get_virtual_prefix(path)
            twsr = target.workspace_root
            result = GlobResult(
                error=result.error,
                matches=[
                    e.model_copy(update={
                        "path": self._reverse_path(e.path, twsr, vprefix),
                    })
                    for e in result.matches
                ],
            )
        return result

    # ------------------------------------------------------------------
    # Fan-out helpers — search across all virtual backends at /.mambo root
    # ------------------------------------------------------------------

    def _grep_all_virtual(self, pattern: str, glob: str | None) -> GrepResult:
        all_matches: list = []
        errors: list[str] = []

        # Default mambo
        p = self._rewrite(self._prefix, self._prefix, self._default_mambo.workspace_root)
        result = self._default_mambo.grep(pattern, p, glob)
        if result.error:
            errors.append(f"[{self._prefix}]: {result.error}")
        if result.matches:
            twsr = self._default_mambo.workspace_root
            all_matches.extend(
                m.model_copy(update={"path": self._reverse_path(m.path, twsr, self._prefix)})
                for m in result.matches
            )

        # Named virtual workspaces
        for name, be in self._virtual.items():
            vprefix = f"{self._prefix}/{name}"
            result = be.grep(pattern, be.workspace_root, glob)
            if result.error:
                errors.append(f"[{vprefix}]: {result.error}")
            if result.matches:
                twsr = be.workspace_root
                all_matches.extend(
                    m.model_copy(update={"path": self._reverse_path(m.path, twsr, vprefix)})
                    for m in result.matches
                )

        return GrepResult(
            error=" | ".join(errors) if errors else None,
            matches=all_matches if all_matches else None,
        )

    def _glob_all_virtual(self, pattern: str) -> GlobResult:
        all_matches: list[FileInfo] = []
        errors: list[str] = []

        # Default mambo
        p = self._rewrite(self._prefix, self._prefix, self._default_mambo.workspace_root)
        result = self._default_mambo.glob(pattern, p)
        if result.error:
            errors.append(f"[{self._prefix}]: {result.error}")
        if result.matches:
            twsr = self._default_mambo.workspace_root
            all_matches.extend(
                e.model_copy(update={"path": self._reverse_path(e.path, twsr, self._prefix)})
                for e in result.matches
            )

        # Named virtual workspaces
        for name, be in self._virtual.items():
            vprefix = f"{self._prefix}/{name}"
            result = be.glob(pattern, be.workspace_root)
            if result.error:
                errors.append(f"[{vprefix}]: {result.error}")
            if result.matches:
                twsr = be.workspace_root
                all_matches.extend(
                    e.model_copy(update={"path": self._reverse_path(e.path, twsr, vprefix)})
                    for e in result.matches
                )

        return GlobResult(
            error=" | ".join(errors) if errors else None,
            matches=all_matches if all_matches else None,
        )

    async def _agrep_all_virtual(self, pattern: str, glob: str | None) -> GrepResult:
        all_matches: list = []
        errors: list[str] = []

        # Default mambo
        p = self._rewrite(self._prefix, self._prefix, self._default_mambo.workspace_root)
        result = await self._default_mambo.agrep(pattern, p, glob)
        if result.error:
            errors.append(f"[{self._prefix}]: {result.error}")
        if result.matches:
            twsr = self._default_mambo.workspace_root
            all_matches.extend(
                m.model_copy(update={"path": self._reverse_path(m.path, twsr, self._prefix)})
                for m in result.matches
            )

        # Named virtual workspaces
        for name, be in self._virtual.items():
            vprefix = f"{self._prefix}/{name}"
            result = await be.agrep(pattern, be.workspace_root, glob)
            if result.error:
                errors.append(f"[{vprefix}]: {result.error}")
            if result.matches:
                twsr = be.workspace_root
                all_matches.extend(
                    m.model_copy(update={"path": self._reverse_path(m.path, twsr, vprefix)})
                    for m in result.matches
                )

        return GrepResult(
            error=" | ".join(errors) if errors else None,
            matches=all_matches if all_matches else None,
        )

    async def _aglob_all_virtual(self, pattern: str) -> GlobResult:
        all_matches: list[FileInfo] = []
        errors: list[str] = []

        # Default mambo
        p = self._rewrite(self._prefix, self._prefix, self._default_mambo.workspace_root)
        result = await self._default_mambo.aglob(pattern, p)
        if result.error:
            errors.append(f"[{self._prefix}]: {result.error}")
        if result.matches:
            twsr = self._default_mambo.workspace_root
            all_matches.extend(
                e.model_copy(update={"path": self._reverse_path(e.path, twsr, self._prefix)})
                for e in result.matches
            )

        # Named virtual workspaces
        for name, be in self._virtual.items():
            vprefix = f"{self._prefix}/{name}"
            result = await be.aglob(pattern, be.workspace_root)
            if result.error:
                errors.append(f"[{vprefix}]: {result.error}")
            if result.matches:
                twsr = be.workspace_root
                all_matches.extend(
                    e.model_copy(update={"path": self._reverse_path(e.path, twsr, vprefix)})
                    for e in result.matches
                )

        return GlobResult(
            error=" | ".join(errors) if errors else None,
            matches=all_matches if all_matches else None,
        )

    # ------------------------------------------------------------------
    # copy — cross-backend single-file copy
    # ------------------------------------------------------------------

    def copy(self, source: str, destination: str) -> CopyResult:
        """Copy a single file, potentially across different backends.

        Reads the source file as raw bytes via :meth:`download_files` on the
        source backend, then writes to the destination via :meth:`upload_files`
        on the destination backend.  This correctly handles both text and
        binary files without going through the ``write(str)`` path.
        """
        src_be, src_path = self._route(source)
        dst_be, dst_path = self._route(destination)

        # Download source as raw bytes
        dl_results = list(src_be.download_files([src_path]))
        if not dl_results:
            return CopyResult(error=f"No result returned when reading '{source}'")
        dl = dl_results[0]
        if dl.error:
            return CopyResult(error=f"Failed to read '{source}': {dl.error}")
        if dl.content is None:
            return CopyResult(error=f"'{source}' is empty or unreadable")

        # Upload raw bytes to destination
        ul_results = list(dst_be.upload_files([(dst_path, dl.content)]))
        if not ul_results:
            return CopyResult(error=f"No result returned when writing '{destination}'")
        ul = ul_results[0]
        if ul.error:
            return CopyResult(error=f"Failed to write '{destination}': {ul.error}")

        return CopyResult(source=source, destination=destination)

    async def acopy(self, source: str, destination: str) -> CopyResult:
        """Async: Copy a single file across backends."""
        return await asyncio.to_thread(self.copy, source, destination)

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
        # Group files by target backend
        groups: dict[int, list[tuple[str, bytes]]] = {}  # backend_id → files
        targets: list[BackendProtocol] = []
        index_map: list[tuple[int, int]] = []  # (target_idx, file_idx_in_group)

        for orig_path, data in files:
            target, stripped = self._route(orig_path)
            # Find or register target
            idx = None
            for i, t in enumerate(targets):
                if t is target:
                    idx = i
                    break
            if idx is None:
                idx = len(targets)
                targets.append(target)
                groups[idx] = []
            groups[idx].append((stripped, data))
            index_map.append((idx, len(groups[idx]) - 1))

        # Execute uploads per target
        results_per_target: dict[int, list[UploadFileResult]] = {}
        for idx, group_files in groups.items():
            be = targets[idx]
            if isinstance(be, ThreadAwareWorkspace):
                results_per_target[idx] = list(
                    be.upload_files(group_files, thread_id=thread_id)
                )
            else:
                results_per_target[idx] = list(be.upload_files(group_files))

        # Reassemble in original order
        results: list[UploadFileResult] = []
        for target_idx, file_idx in index_map:
            results.append(results_per_target[target_idx][file_idx])
        return results

    def download_files(
        self,
        paths: list[str],
        *,
        thread_id: str | None = None,
    ) -> list[DownloadFileResult]:
        """Download files, routing each to the correct backend by prefix."""
        groups: dict[int, list[str]] = {}
        targets: list[BackendProtocol] = []
        index_map: list[tuple[int, int]] = []

        for orig_path in paths:
            target, stripped = self._route(orig_path)
            idx = None
            for i, t in enumerate(targets):
                if t is target:
                    idx = i
                    break
            if idx is None:
                idx = len(targets)
                targets.append(target)
                groups[idx] = []
            groups[idx].append(stripped)
            index_map.append((idx, len(groups[idx]) - 1))

        results_per_target: dict[int, list[DownloadFileResult]] = {}
        for idx, group_paths in groups.items():
            be = targets[idx]
            if isinstance(be, ThreadAwareWorkspace):
                results_per_target[idx] = list(
                    be.download_files(group_paths, thread_id=thread_id)
                )
            else:
                results_per_target[idx] = list(be.download_files(group_paths))

        results: list[DownloadFileResult] = []
        for target_idx, file_idx in index_map:
            results.append(results_per_target[target_idx][file_idx])
        return results
