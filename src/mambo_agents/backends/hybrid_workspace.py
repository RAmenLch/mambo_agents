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
import re

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

from mambo_agents.backends.protocol import (
    BackendProtocol,
    DeleteResult,
    DownloadFileResult,
    EditResult,
    FileInfo,
    GlobResult,
    GrepResult,
    LsResult,
    ReadResult,
    ReadSummarizer,
    Result,
    ThreadAwareWorkspace,
    ToolTimeouts,
    UploadFileResult,
    WriteResult,
)
from mambo_agents.backends.schemas import check_no_path_traversal, GrepMatch,VirtualPath
from mambo_agents.backends.state import StateBackend

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
# Virtual workspace name validation
# ============================================================================

_WORKSPACE_NAME_RE = re.compile(r"^[a-zA-Z0-9_][a-zA-Z0-9_.\-]*$")
"""Allowed characters for virtual workspace names (single segment, no ``/``)."""


def _validate_workspace_name(name: str) -> str:
    """Validate a virtual workspace name (single path segment under ``/.mambo/``).

    Raises :class:`ValueError` if *name*:
    - contains ``..`` path traversal
    - contains ``//`` double slashes
    - contains ``/`` (nested names are forbidden — ``a/b`` is not a valid name)
    - contains illegal characters outside ``[a-zA-Z0-9_.-]``
    - starts with a character other than ``[a-zA-Z0-9_]``

    Returns *name* unchanged on success.
    """
    check_no_path_traversal(name, name="virtual workspace name")
    if "/" in name:
        raise ValueError(
            f"Virtual workspace name must be a single segment (no '/'), got {name!r}"
        )
    if not _WORKSPACE_NAME_RE.match(name):
        raise ValueError(
            f"Virtual workspace name must match [a-zA-Z0-9_]"
            f"[a-zA-Z0-9_.\\-]* but got {name!r}"
        )
    return name


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

    # Default per-tool timeout values specific to this backend (overridable via __init__).
    _BACKEND_DEFAULT_TIMEOUTS = ToolTimeouts(copy=120.0)

    def __init__(
        self,
        real_backend: BackendProtocol,
        virtual_workspaces: dict[str, BackendProtocol] | None = None,
        mambo_prefix: VirtualPath = VirtualPath(DEFAULT_MAMBO_PREFIX),
        custom_description: str | None = None,
        *,
        workspace_root: VirtualPath = VirtualPath("/workspace"),
        max_read_chars: int = 100_000,
        summarizer: "ReadSummarizer | None" = None,
        tool_timeouts: ToolTimeouts | None = None,
    ) -> None:
        # Merge backend-specific defaults with user overrides (user wins).
        _user = tool_timeouts.model_dump() if tool_timeouts else {}
        _merged = ToolTimeouts(**{**self._BACKEND_DEFAULT_TIMEOUTS.model_dump(), **_user})
        super().__init__(max_read_chars=max_read_chars, summarizer=summarizer, tool_timeouts=_merged)

        self.workspace_root = VirtualPath(workspace_root)
        self._real = real_backend
        self._prefix = VirtualPath(mambo_prefix)  # "/.mambo"
        self._custom_description = custom_description

        # --- Build virtual workspaces ---------------------------------
        vw = dict(virtual_workspaces or {})

        # "." key → override default /.mambo/ StateBackend
        if "." in vw:
            self._default_mambo = vw.pop(".")
        else:
            self._default_mambo = StateBackend(
                max_read_chars=max_read_chars, summarizer=summarizer, tool_timeouts=_merged,
            )

        # Remaining entries → /.mambo/<name>/ namespaces
        self._virtual: dict[str, BackendProtocol] = {}
        for name, be in vw.items():
            name = _validate_workspace_name(name.strip("/"))
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
    # tools — real_backend extras (path-translated) + copy
    # ------------------------------------------------------------------

    @property
    def tools(self) -> list[StructuredTool]:
        wrapped = [self._wrap_extra_tool(t) for t in self._real.tools]
        wrapped.append(
            StructuredTool(
                name="copy",
                description=(
                    "Copy a single file from source to destination. "
                    "Supports cross-backend copies (e.g. from a virtual "
                    f"**{self._prefix.normalized}/** workspace to the real filesystem, "
                    "or between virtual workspaces). "
                    "Overwrites the destination if it already exists."
                ),
                args_schema=create_model(
                    "CopySchema",
                    source=(VirtualPath, Field(description="Absolute source file path")),
                    destination=(VirtualPath, Field(description="Absolute destination file path")),
                ),
                func=self._safe_tool_func("copy", self.copy),
                coroutine=self._safe_tool_coroutine("copy", self.acopy),
            ),
        )
        return wrapped

    @staticmethod
    def _tool_has_path_param(tool: StructuredTool) -> bool:
        """Return ``True`` if *tool*'s args_schema declares a ``path`` field."""
        schema = tool.args_schema
        if schema is None:
            return False
        return "path" in schema.model_fields

    def _translate_path_kwarg(self, kwargs: dict) -> tuple[BackendProtocol, VirtualPath] | None:
        """Rewrite ``kwargs["path"]`` via :meth:`_route`
        when the path routes to the real backend.

        Returns ``(target_backend, virtual_prefix)`` for reverse
        translation of results, or ``None`` when no translation occurred.

        Only modifies the dict when all of these hold:

        - ``"path"`` is in *kwargs*
        - The value is a non-empty string or VirtualPath
        - ``_route()`` succeeds and returns ``self._real`` as the target

        Paths that route to a virtual backend or fail validation are left
        unchanged — the real backend's own ``_resolve()`` will raise a
        :class:`WorkspacePathError` with a descriptive message.
        """
        if "path" not in kwargs:
            return None
        raw_path = kwargs["path"]
        if not isinstance(raw_path, (str, VirtualPath)) or not raw_path:
            return None
        try:
            if isinstance(raw_path, str):
                raw_path = VirtualPath(raw_path)
            target, rewritten = self._route(raw_path)
        except (ValueError, TypeError):
            return None
        if target is not self._real:
            return None
        vprefix = self._get_virtual_prefix(raw_path)
        kwargs["path"] = rewritten
        return (target, vprefix)

    def _wrap_extra_tool(self, tool: StructuredTool) -> StructuredTool:
        """Wrap a real-backend tool so that its ``path`` argument is
        translated through :meth:`_route` before delegation, and
        ``VirtualPath`` fields in the result are reverse-translated.

        Only rewrites when ``_route`` maps to ``self._real`` — paths
        that route to a virtual backend are left unchanged so the real
        backend's own ``_resolve()`` returns a clear error.
        """
        if not self._tool_has_path_param(tool):
            return tool

        original_func = tool.func
        original_coroutine = tool.coroutine

        def wrapped_func(*args, **kwargs):
            route_info = self._translate_path_kwarg(kwargs)
            if original_func is not None:
                result = original_func(*args, **kwargs)
            else:
                result = None
            if result is not None and route_info is not None and isinstance(result, Result):
                target, vprefix = route_info
                result = result.apply_reverse_translation(self._reverse_path, target.workspace_root, vprefix)
            return result

        async def wrapped_coroutine(*args, **kwargs):
            route_info = self._translate_path_kwarg(kwargs)
            if original_coroutine is not None:
                result = await original_coroutine(*args, **kwargs)
            elif original_func is not None:
                result = await asyncio.to_thread(original_func, *args, **kwargs)
            else:
                result = None
            if result is not None and route_info is not None and isinstance(result, Result):
                target, vprefix = route_info
                result = result.apply_reverse_translation(self._reverse_path, target.workspace_root, vprefix)
            return result

        return StructuredTool(
            name=tool.name,
            description=tool.description,
            args_schema=tool.args_schema,
            func=wrapped_func,
            coroutine=wrapped_coroutine
        )

    # ------------------------------------------------------------------
    # description — system-prompt injection
    # ------------------------------------------------------------------

    @property
    def description(self) -> str:
        wr = self.workspace_root.value
        core_tools = "`ls`, `read`, `write`, `edit`, `grep`, `glob`, `copy`"
        base = (
            f"A virtual temporary workspace is available at **{self._prefix.normalized}/**.  "
            f"Use it for intermediate files, chat history dumps, large "
            f"tool-result evictions, and subagent communication.  "
            f"This workspace is isolated from the real filesystem — files here "
            f"do not persist across sessions.  "
            f"All real-filesystem paths must start with **{wr}/**.  "
            f"The following file tools work on **{self._prefix.normalized}/** paths: "
            f"{core_tools}.  "
            f"`copy` can move files between the virtual workspace and the real "
            f"filesystem.  "
        )

        # Dynamically detect which extra tools the real backend provides
        real_tool_names = {t.name for t in self._real.tools}
        auto_translate_tools = [n for n in ("tree", "delete") if n in real_tool_names]
        if auto_translate_tools:
            quoted = ", ".join(f"`{n}`" for n in auto_translate_tools)
            base += (
                f"{quoted} automatically translate paths through the "
                f"workspace namespace.  "
            )
        if "execute" in real_tool_names:
            base += (
                f"`execute` must use real filesystem paths — do NOT target "
                f"**{self._prefix.normalized}/** paths with shell commands."
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


    @staticmethod
    def _rewrite(vp: VirtualPath, strip_prefix: str, target_ws_root: VirtualPath) -> VirtualPath:
        """Strip *strip_prefix* from *vp*, then prepend *target_ws_root*.

        Always rewrites: strip the prefix, prepend the target backend's
        ``workspace_root``.  No special cases — predictable regardless of
        what directory names the user chooses.

        Raises
        ------
        ValueError
            If *vp* is not under *strip_prefix*.
        """
        rel = vp.relative_to(strip_prefix)
        ws = target_ws_root.value.rstrip("/")
        if not rel:
            return VirtualPath(ws)
        return VirtualPath(ws + "/" + rel)

    def _route(self, path: VirtualPath) -> tuple[BackendProtocol, VirtualPath]:
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
        # VirtualPath already validates on construction (no .., no //, absolute)
        if not path.normalized:
            raise ValueError(
                f"不能使用空字符串或 '/' 作为路径，"
                f"请提供 {self.workspace_root.value}/ 或 {self._prefix.normalized}/ 下的完整路径。"
            )
        mambo_base = self._prefix.normalized  # "/.mambo"

        # (1) Check named virtual workspaces
        for name, be in self._virtual.items():
            ns_vp = self._prefix.join(name)
            if path == ns_vp or path.is_under(ns_vp.normalized):
                return be, self._rewrite(path, ns_vp.value, be.workspace_root)

        # (2) Fallback: default /.mambo/
        if path.is_under(mambo_base):
            return self._default_mambo, self._rewrite(
                path, mambo_base, self._default_mambo.workspace_root,
            )

        # (3) Real backend — strip Hybrid.ws_root, prepend real.ws_root
        return self._real, self._rewrite(path, self.workspace_root.value, self._real.workspace_root)

    # ------------------------------------------------------------------
    # Reverse path translation for results (ls / grep / glob)
    # ------------------------------------------------------------------

    def _get_virtual_prefix(self, path: str | VirtualPath) -> VirtualPath:
        """Return the external prefix that *path* would be routed from.

        This is the mirror of :meth:`_route` — given the same *path*, it
        returns the prefix that ``_route`` strips before delegation.
        Used to reverse-translate internal paths in result objects.
        """
        vp = VirtualPath(path) if not isinstance(path, VirtualPath) else path

        # (1) Named virtual workspace
        for name in self._virtual:
            ns_vp = self._prefix.join(name)
            if vp == ns_vp or vp.is_under(ns_vp.normalized):
                return ns_vp

        # (2) Default mambo
        if vp.is_under(self._prefix.normalized):
            return VirtualPath(self._prefix.normalized)

        # (3) Real backend
        return VirtualPath(self.workspace_root.normalized)

    @staticmethod
    def _reverse_path(
        internal_path: VirtualPath,
        target_ws_root: VirtualPath,
        virtual_prefix: VirtualPath,
    ) -> VirtualPath:
        """Reverse a ``_rewrite``: strip *target_ws_root*, prepend *virtual_prefix*.

        Example::
            _reverse_path(VirtualPath("/skill-a"), VirtualPath("/"), VirtualPath("/.mambo/skills"))
            # → VirtualPath("/.mambo/skills/skill-a")
        """
        rel = internal_path.relative_to(target_ws_root.normalized)
        if not rel:
            return VirtualPath(virtual_prefix.normalized)
        return virtual_prefix.join(rel)

    def _valid_paths_description(self) -> str:
        """Build a human-readable description of all valid path prefixes,
        including the mapping from AI-facing workspace to real backend root.
        """
        parts = [f"'{self.workspace_root.value}/'（映射至真实路径 '{self._real.workspace_root.value}/'）",
                 f"'{self._prefix.normalized}/'（虚拟临时工作区）"]
        for name in sorted(self._virtual):
            parts.append(f"'{self._prefix.join(name).value}/'（虚拟工作区）")
        return "、".join(parts)

    # ------------------------------------------------------------------
    # Core file operations — prefix-routed
    # ------------------------------------------------------------------

    def ls(self, path: VirtualPath) -> LsResult:
        try:
            target, p = self._route(path)
        except (ValueError, TypeError) as e:
            return LsResult(error=f"路径 '{path}' 无效：{e}。仅可访问：{self._valid_paths_description()}")
        result = target.ls(p)
        result = result.apply_reverse_translation(
            self._reverse_path, target.workspace_root, self._get_virtual_prefix(path),
        )

        # Inject named virtual workspace directories when listing /.mambo root
        if str(path).rstrip("/") == self._prefix.normalized:
            entries = list(result.entries) if result.entries else []
            for name in self._virtual:
                entries.append(FileInfo(
                    path=self._prefix.join(name),
                    is_dir=True,
                ))
            result = result.model_copy(update={"entries": entries})

        return result

    def read(
        self,
        file_path: VirtualPath,
        offset: int = 0,
        limit: int = 2000,
        include_line_numbers: bool = False,
    ) -> ReadResult:
        try:
            target, p = self._route(file_path)
        except (ValueError, TypeError) as e:
            return ReadResult(error=f"路径 '{file_path}' 无效：{e}。仅可访问：{self._valid_paths_description()}")
        return target.read(p, offset, limit, include_line_numbers)

    def read_raw(
        self,
        file_path: VirtualPath,
        offset: int = 0,
        limit: int | None = 2000,
        include_line_numbers: bool = False,
    ) -> ReadResult:
        try:
            target, p = self._route(file_path)
        except (ValueError, TypeError) as e:
            return ReadResult(error=f"路径 '{file_path}' 无效：{e}。仅可访问：{self._valid_paths_description()}")
        return target.read_raw(p, offset, limit, include_line_numbers)

    def write(
        self, file_path: VirtualPath, content: str, overwrite: bool = False,
    ) -> WriteResult:
        try:
            target, p = self._route(file_path)
        except (ValueError, TypeError) as e:
            return WriteResult(error=f"路径 '{file_path}' 无效：{e}。仅可访问：{self._valid_paths_description()}")
        result = target.write(p, content, overwrite)
        return result.apply_reverse_translation(
            self._reverse_path, target.workspace_root, self._get_virtual_prefix(file_path),
        )

    def edit(
        self,
        file_path: VirtualPath,
        old_str: str,
        new_str: str,
        *,
        replace_all: bool = False,
    ) -> EditResult:
        try:
            target, p = self._route(file_path)
        except (ValueError, TypeError) as e:
            return EditResult(error=f"路径 '{file_path}' 无效：{e}。仅可访问：{self._valid_paths_description()}")
        result = target.edit(p, old_str, new_str, replace_all=replace_all)
        return result.apply_reverse_translation(
            self._reverse_path, target.workspace_root, self._get_virtual_prefix(file_path),
        )

    def grep(
        self,
        pattern: str,
        path: VirtualPath,
        glob: str | None = None,
        regex: bool = False,
        offset: int = 0,
        limit: int | None = None,
    ) -> GrepResult:
        # Fan-out to all virtual backends when searching /.mambo root
        if str(path).rstrip("/") == self._prefix.normalized:
            return self._apply_grep_limit(
                self._grep_all_virtual(pattern, glob, regex),
                offset, limit,
            )

        try:
            target, p = self._route(path)
        except (ValueError, TypeError) as e:
            return GrepResult(error=f"路径 '{path}' 无效：{e}。仅可访问：{self._valid_paths_description()}")
        result = target.grep(pattern, p, glob, regex, offset, limit)
        return result.apply_reverse_translation(
            self._reverse_path, target.workspace_root, self._get_virtual_prefix(path),
        )

    def glob(self, pattern: str, path: VirtualPath) -> GlobResult:
        # Fan-out to all virtual backends when searching /.mambo root
        if str(path).rstrip("/") == self._prefix.normalized:
            return self._glob_all_virtual(pattern)

        try:
            target, p = self._route(path)
        except (ValueError, TypeError) as e:
            return GlobResult(error=f"路径 '{path}' 无效：{e}。仅可访问：{self._valid_paths_description()}")
        result = target.glob(pattern, p)
        return result.apply_reverse_translation(
            self._reverse_path, target.workspace_root, self._get_virtual_prefix(path),
        )

    # ------------------------------------------------------------------
    # Async core file operations — route to target's async method
    # so that per-backend locks (e.g. SshBackend._async_lock) are
    # properly acquired.  Without these overrides the inherited
    # BackendProtocol.async methods call self.write() etc. on a
    # thread-pool thread, bypassing the target backend's async guards.
    # ------------------------------------------------------------------

    async def als(self, path: VirtualPath) -> LsResult:
        try:
            target, p = self._route(path)
        except (ValueError, TypeError) as e:
            return LsResult(error=f"路径 '{path}' 无效：{e}。仅可访问：{self._valid_paths_description()}")
        result = await target.als(p)
        result = result.apply_reverse_translation(
            self._reverse_path, target.workspace_root, self._get_virtual_prefix(path),
        )

        # Inject named virtual workspace directories when listing /.mambo root
        if str(path).rstrip("/") == self._prefix.normalized:
            entries = list(result.entries) if result.entries else []
            for name in self._virtual:
                entries.append(FileInfo(
                    path=self._prefix.join(name),
                    is_dir=True,
                ))
            result = result.model_copy(update={"entries": entries})

        return result

    async def aread(
        self,
        file_path: VirtualPath,
        offset: int = 0,
        limit: int = 2000,
        include_line_numbers: bool = False,
    ) -> ReadResult:
        try:
            target, p = self._route(file_path)
        except (ValueError, TypeError) as e:
            return ReadResult(error=f"路径 '{file_path}' 无效：{e}。仅可访问：{self._valid_paths_description()}")
        return await target.aread(
            p, offset, limit, include_line_numbers,
        )

    async def awrite(
        self, file_path: VirtualPath, content: str, overwrite: bool = False,
    ) -> WriteResult:
        try:
            target, p = self._route(file_path)
        except (ValueError, TypeError) as e:
            return WriteResult(error=f"路径 '{file_path}' 无效：{e}。仅可访问：{self._valid_paths_description()}")
        result = await target.awrite(p, content, overwrite)
        return result.apply_reverse_translation(
            self._reverse_path, target.workspace_root, self._get_virtual_prefix(file_path),
        )

    async def aedit(
        self,
        file_path: VirtualPath,
        old_str: str,
        new_str: str,
        *,
        replace_all: bool = False,
    ) -> EditResult:
        try:
            target, p = self._route(file_path)
        except (ValueError, TypeError) as e:
            return EditResult(error=f"路径 '{file_path}' 无效：{e}。仅可访问：{self._valid_paths_description()}")
        result = await target.aedit(p, old_str, new_str, replace_all=replace_all)
        return result.apply_reverse_translation(
            self._reverse_path, target.workspace_root, self._get_virtual_prefix(file_path),
        )

    async def agrep(
        self,
        pattern: str,
        path: VirtualPath,
        glob: str | None = None,
        regex: bool = False,
        offset: int = 0,
        limit: int | None = None,
    ) -> GrepResult:
        # Fan-out to all virtual backends when searching /.mambo root
        if str(path).rstrip("/") == self._prefix.normalized:
            raw_matches = await self._agrep_all_virtual(pattern, glob, regex)
            return self._apply_grep_limit(raw_matches, offset, limit)

        try:
            target, p = self._route(path)
        except (ValueError, TypeError) as e:
            return GrepResult(error=f"路径 '{path}' 无效：{e}。仅可访问：{self._valid_paths_description()}")
        result = await target.agrep(pattern, p, glob, regex, offset, limit)
        return result.apply_reverse_translation(
            self._reverse_path, target.workspace_root, self._get_virtual_prefix(path),
        )

    async def aglob(self, pattern: str, path: VirtualPath) -> GlobResult:
        # Fan-out to all virtual backends when searching /.mambo root
        if str(path).rstrip("/") == self._prefix.normalized:
            return await self._aglob_all_virtual(pattern)

        try:
            target, p = self._route(path)
        except (ValueError, TypeError) as e:
            return GlobResult(error=f"路径 '{path}' 无效：{e}。仅可访问：{self._valid_paths_description()}")
        result = await target.aglob(pattern, p)
        return result.apply_reverse_translation(
            self._reverse_path, target.workspace_root, self._get_virtual_prefix(path),
        )

    # ------------------------------------------------------------------
    # Fan-out helpers — search across all virtual backends at /.mambo root
    # ------------------------------------------------------------------

    def _grep_all_virtual(self, pattern: str, glob: str | None, regex: bool = False) -> list[GrepMatch]:
        """Collect raw grep matches from all virtual backends (no offset/limit)."""
        all_matches: list = []

        # Default mambo
        mambo_prefix = self._prefix
        p = self._rewrite(mambo_prefix, mambo_prefix.normalized, self._default_mambo.workspace_root)
        result = self._default_mambo.grep(pattern, p, glob, regex, offset=0, limit=None)
        if result.matches:
            twsr = self._default_mambo.workspace_root
            all_matches.extend(
                m.model_copy(update={"path": self._reverse_path(m.path, twsr, mambo_prefix)})
                for m in result.matches
            )

        # Named virtual workspaces
        for name, be in self._virtual.items():
            vprefix = self._prefix.join(name)
            result = be.grep(pattern, VirtualPath(be.workspace_root), glob, regex, offset=0, limit=None)
            if result.matches:
                twsr = be.workspace_root
                all_matches.extend(
                    m.model_copy(update={"path": self._reverse_path(m.path, twsr, vprefix)})
                    for m in result.matches
                )

        return all_matches

    def _glob_all_virtual(self, pattern: str) -> GlobResult:
        all_matches: list[FileInfo] = []
        errors: list[str] = []

        # Default mambo
        mambo_prefix = self._prefix
        p = self._rewrite(mambo_prefix, mambo_prefix.normalized, self._default_mambo.workspace_root)
        result = self._default_mambo.glob(pattern, p)
        if result.error:
            errors.append(f"[{mambo_prefix.normalized}]: {result.error}")
        if result.matches:
            twsr = self._default_mambo.workspace_root
            all_matches.extend(
                e.model_copy(update={"path": self._reverse_path(e.path, twsr, mambo_prefix)})
                for e in result.matches
            )

        # Named virtual workspaces
        for name, be in self._virtual.items():
            vprefix = self._prefix.join(name)
            result = be.glob(pattern, VirtualPath(be.workspace_root))
            if result.error:
                errors.append(f"[{vprefix.value}]: {result.error}")
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

    async def _agrep_all_virtual(self, pattern: str, glob: str | None, regex: bool = False) -> list[GrepMatch]:
        """Collect raw grep matches from all virtual backends (async, no offset/limit)."""
        all_matches: list = []

        # Default mambo
        mambo_prefix = self._prefix
        p = self._rewrite(mambo_prefix, mambo_prefix.normalized, self._default_mambo.workspace_root)
        result = await self._default_mambo.agrep(pattern, p, glob, regex, offset=0, limit=None)
        if result.matches:
            twsr = self._default_mambo.workspace_root
            all_matches.extend(
                m.model_copy(update={"path": self._reverse_path(m.path, twsr, mambo_prefix)})
                for m in result.matches
            )

        # Named virtual workspaces
        for name, be in self._virtual.items():
            vprefix = self._prefix.join(name)
            result = await be.agrep(pattern, VirtualPath(be.workspace_root), glob, regex, offset=0, limit=None)
            if result.matches:
                twsr = be.workspace_root
                all_matches.extend(
                    m.model_copy(update={"path": self._reverse_path(m.path, twsr, vprefix)})
                    for m in result.matches
                )

        return all_matches

    async def _aglob_all_virtual(self, pattern: str) -> GlobResult:
        all_matches: list[FileInfo] = []
        errors: list[str] = []

        # Default mambo
        mambo_prefix = self._prefix
        p = self._rewrite(mambo_prefix, mambo_prefix.normalized, self._default_mambo.workspace_root)
        result = await self._default_mambo.aglob(pattern, p)
        if result.error:
            errors.append(f"[{mambo_prefix.normalized}]: {result.error}")
        if result.matches:
            twsr = self._default_mambo.workspace_root
            all_matches.extend(
                e.model_copy(update={"path": self._reverse_path(e.path, twsr, mambo_prefix)})
                for e in result.matches
            )

        # Named virtual workspaces
        for name, be in self._virtual.items():
            vprefix = self._prefix.join(name)
            result = await be.aglob(pattern, VirtualPath(be.workspace_root))
            if result.error:
                errors.append(f"[{vprefix.value}]: {result.error}")
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

    def copy(self, source: VirtualPath, destination: VirtualPath) -> CopyResult:
        """Copy a single file, potentially across different backends.

        Reads the source file as raw bytes via :meth:`download_files` on the
        source backend, then writes to the destination via :meth:`upload_files`
        on the destination backend.  This correctly handles both text and
        binary files without going through the ``write(str)`` path.
        """
        try:
            src_be, src_path = self._route(source)
        except (ValueError, TypeError) as e:
            return CopyResult(error=f"源路径 '{source}' 无效：{e}。仅可访问：{self._valid_paths_description()}")
        try:
            dst_be, dst_path = self._route(destination)
        except (ValueError, TypeError) as e:
            return CopyResult(error=f"目标路径 '{destination}' 无效：{e}。仅可访问：{self._valid_paths_description()}")

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

        return CopyResult(source=str(source), destination=str(destination))

    async def acopy(self, source: VirtualPath, destination: VirtualPath) -> CopyResult:
        """Async: Copy a single file across backends."""
        return await asyncio.to_thread(self.copy, source, destination)

    # ------------------------------------------------------------------
    # Developer API — upload / download (split by prefix)
    # ------------------------------------------------------------------

    def upload_files(
        self,
        files: list[tuple[VirtualPath, bytes]],
        *,
        thread_id: str | None = None,
    ) -> list[UploadFileResult]:
        """Upload files, routing each to the correct backend by prefix."""
        # Group files by target backend
        groups: dict[int, list[tuple[VirtualPath, bytes]]] = {}  # backend_id → files
        targets: list[BackendProtocol] = []
        index_map: list[tuple[int, int]] = []  # (target_idx, file_idx_in_group)
        routing_errors: dict[int, str] = {}  # file_orig_index → error message

        for orig_idx, (orig_path, data) in enumerate(files):
            try:
                target, stripped = self._route(orig_path)
            except (ValueError, TypeError) as e:
                routing_errors[orig_idx] = (
                    f"路径 '{orig_path}' 无效：{e}。仅可访问：{self._valid_paths_description()}"
                )
                continue
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
            index_map.append((orig_idx, idx, len(groups[idx]) - 1))

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
        for orig_idx in range(len(files)):
            if orig_idx in routing_errors:
                path = files[orig_idx][0]
                path_str = path.value if isinstance(path, VirtualPath) else str(path)
                results.append(UploadFileResult(path=path_str, error=routing_errors[orig_idx]))
            else:
                # find the matching entry in index_map
                for oi, target_idx, file_idx in index_map:
                    if oi == orig_idx:
                        results.append(results_per_target[target_idx][file_idx])
                        break
        return results

    def download_files(
        self,
        paths: list[VirtualPath],
        *,
        thread_id: str | None = None,
    ) -> list[DownloadFileResult]:
        """Download files, routing each to the correct backend by prefix."""
        groups: dict[int, list[VirtualPath]] = {}
        targets: list[BackendProtocol] = []
        index_map: list[tuple[int, int, int]] = []
        routing_errors: dict[int, str] = {}

        for orig_idx, orig_path in enumerate(paths):
            try:
                target, stripped = self._route(orig_path)
            except (ValueError, TypeError) as e:
                routing_errors[orig_idx] = (
                    f"路径 '{orig_path}' 无效：{e}。仅可访问：{self._valid_paths_description()}"
                )
                continue
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
            index_map.append((orig_idx, idx, len(groups[idx]) - 1))

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
        for orig_idx in range(len(paths)):
            if orig_idx in routing_errors:
                path = paths[orig_idx]
                path_str = path.value if isinstance(path, VirtualPath) else str(path)
                results.append(DownloadFileResult(path=path_str, error=routing_errors[orig_idx]))
            else:
                for oi, target_idx, file_idx in index_map:
                    if oi == orig_idx:
                        results.append(results_per_target[target_idx][file_idx])
                        break
        return results

    # ------------------------------------------------------------------
    # Async bulk — route per-group to each target backend's async method
    # so that backends with _async_lock (e.g. SshBackend) serialize
    # SFTP access and avoid paramiko deadlocks.
    # ------------------------------------------------------------------

    async def aupload_files(
        self,
        files: list[tuple[VirtualPath, bytes]],
        *,
        thread_id: str | None = None,
    ) -> list[UploadFileResult]:
        """Async: upload files, routing each to the correct backend."""
        # Group files by target backend (same logic as sync upload_files)
        groups: dict[int, list[tuple[VirtualPath, bytes]]] = {}
        targets: list[BackendProtocol] = []
        index_map: list[tuple[int, int, int]] = []
        routing_errors: dict[int, str] = {}

        for orig_idx, (orig_path, data) in enumerate(files):
            try:
                target, stripped = self._route(orig_path)
            except (ValueError, TypeError) as e:
                routing_errors[orig_idx] = (
                    f"路径 '{orig_path}' 无效：{e}。仅可访问：{self._valid_paths_description()}"
                )
                continue
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
            index_map.append((orig_idx, idx, len(groups[idx]) - 1))

        # Execute uploads per target via each backend's async method
        results_per_target: dict[int, list[UploadFileResult]] = {}
        for idx, group_files in groups.items():
            be = targets[idx]
            results_per_target[idx] = list(
                await be.aupload_files(group_files)
            )

        # Reassemble in original order
        results: list[UploadFileResult] = []
        for orig_idx in range(len(files)):
            if orig_idx in routing_errors:
                path = files[orig_idx][0]
                path_str = path.value if isinstance(path, VirtualPath) else str(path)
                results.append(UploadFileResult(path=path_str, error=routing_errors[orig_idx]))
            else:
                for oi, target_idx, file_idx in index_map:
                    if oi == orig_idx:
                        results.append(results_per_target[target_idx][file_idx])
                        break
        return results

    async def adownload_files(
        self,
        paths: list[VirtualPath],
        *,
        thread_id: str | None = None,
    ) -> list[DownloadFileResult]:
        """Async: download files, routing each to the correct backend."""
        # Group paths by target backend (same logic as sync download_files)
        groups: dict[int, list[VirtualPath]] = {}
        targets: list[BackendProtocol] = []
        index_map: list[tuple[int, int, int]] = []
        routing_errors: dict[int, str] = {}

        for orig_idx, orig_path in enumerate(paths):
            try:
                target, stripped = self._route(orig_path)
            except (ValueError, TypeError) as e:
                routing_errors[orig_idx] = (
                    f"路径 '{orig_path}' 无效：{e}。仅可访问：{self._valid_paths_description()}"
                )
                continue
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
            index_map.append((orig_idx, idx, len(groups[idx]) - 1))

        # Execute downloads per target via each backend's async method
        results_per_target: dict[int, list[DownloadFileResult]] = {}
        for idx, group_paths in groups.items():
            be = targets[idx]
            results_per_target[idx] = list(
                await be.adownload_files(group_paths)
            )

        # Reassemble in original order
        results: list[DownloadFileResult] = []
        for orig_idx in range(len(paths)):
            if orig_idx in routing_errors:
                path = paths[orig_idx]
                path_str = path.value if isinstance(path, VirtualPath) else str(path)
                results.append(DownloadFileResult(path=path_str, error=routing_errors[orig_idx]))
            else:
                for oi, target_idx, file_idx in index_map:
                    if oi == orig_idx:
                        results.append(results_per_target[target_idx][file_idx])
                        break
        return results
