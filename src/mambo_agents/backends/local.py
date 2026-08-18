"""LocalBackend – real filesystem backend with shell execution support.

Provides direct file-system access plus extra tools: ``tree`` (directory
structure), ``delete`` (remove files/directories), and ``execute`` (run
shell commands).  Supports both Windows and Linux.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

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
    ReadSummarizer,
    ToolTimeouts,
    UploadFileResult,
    WriteResult,
)
from mambo_agents.backends.utils.multimodal import (
    get_file_type,
    get_mime_type,
)
from mambo_agents.backends.utils import (
    TreeEntry,
    check_path_allowed,
    decode_output,
    detect_trailing_newline_mismatch,
    normalize_line_endings,
    fnmatch_path,
    format_tree_entries,
    format_validation_error,
    format_with_line_numbers,
)
from mambo_agents.backends.schemas import BackendError, DeleteResult, ErrorCode, VirtualPath, human_size

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_EXECUTE_TIMEOUT = 120
_MAX_OUTPUT_BYTES = 100_000
_DEFAULT_MAX_FILE_SIZE_MB = 10


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
        workspace_root: Virtual path prefix acting as the workspace root
            (default ``"/workspace"``).  All file paths must live under
            this prefix — paths outside are rejected so the AI never
            perceives the virtual filesystem as a real system root.
        timeout: Default timeout in seconds for shell execution (default 120).
        max_output_bytes: Max bytes to capture from command output.
        env: Environment variables for shell commands.
            If ``None``, starts with an empty environment (unless ``inherit_env=True``).
        inherit_env: Inherit ``os.environ`` into the command environment.
        enable_execute: Whether to provide the ``execute`` tool to the
            LLM, allowing it to run arbitrary shell commands.
            Default ``False``. Set to ``True`` to enable shell execution.
        max_file_size_mb: Maximum file size in MB for ``grep`` search.
            Files exceeding this limit are skipped.  Default 10 MB.
        edit_whitelist: Virtual path prefixes allowed for edit/write/delete.
            Mutually exclusive with *edit_blacklist*.  When set, any
            path not matching a prefix is rejected.
        edit_blacklist: Virtual path prefixes forbidden for edit/write/delete.
            Mutually exclusive with *edit_whitelist*.  When set, any
            path matching a prefix is rejected.
        ignore_dirs: Directory names (bare basenames, e.g. ``"node_modules"``)
            whose children are hidden in ``tree()`` output and whose files are
            excluded from ``grep()`` results.  Any directory matching one of
            these names (at any depth) is shown with an ``/(ignore)`` marker
            but its content is not expanded.
    """

    # Default per-tool timeout values specific to this backend (overridable via __init__).
    _BACKEND_DEFAULT_TIMEOUTS = ToolTimeouts(tree=60.0, delete=30.0, execute=500.0)

    def __init__(
        self,
        root_dir: str | Path | None = None,
        *,
        workspace_root: VirtualPath = VirtualPath("/workspace"),
        timeout: int = _DEFAULT_EXECUTE_TIMEOUT,
        max_output_bytes: int = _MAX_OUTPUT_BYTES,
        env: dict[str, str] | None = None,
        inherit_env: bool = False,
        enable_execute: bool = False,
        max_file_size_mb: int = _DEFAULT_MAX_FILE_SIZE_MB,
        edit_whitelist: frozenset[VirtualPath] | None = None,
        edit_blacklist: frozenset[VirtualPath] | None = None,
        ignore_dirs: frozenset[str] | None = None,
        max_read_chars: int = 100_000,
        max_grep_matches: int = 1000,
        max_grep_match_chars: int = 500,
        summarizer: "ReadSummarizer | None" = None,
        tool_timeouts: ToolTimeouts | None = None,
    ) -> None:
        # Merge backend-specific timeout defaults with user overrides.
        _user = tool_timeouts.model_dump() if tool_timeouts else {}
        _merged = ToolTimeouts(**{**self._BACKEND_DEFAULT_TIMEOUTS.model_dump(), **_user})
        super().__init__(
            max_read_chars=max_read_chars,
            max_grep_matches=max_grep_matches,
            max_grep_match_chars=max_grep_match_chars,
            summarizer=summarizer,
            tool_timeouts=_merged,
        )
        if timeout <= 0:
            msg = f"timeout must be positive, got {timeout}"
            raise ValueError(msg)

        if edit_whitelist is not None and edit_blacklist is not None:
            raise ValueError(
                "edit_whitelist and edit_blacklist are mutually exclusive. "
                "Provide at most one of them."
            )

        self.workspace_root = VirtualPath(workspace_root)
        self._cwd = Path(root_dir).resolve() if root_dir else Path.cwd()
        if os.name == "nt":
            # Normalize once so downstream lexical comparisons (relative_to)
            # never see the extended-length (\\\\?\\) form.
            self._cwd = self._strip_win_extended_prefix(self._cwd)
        self._default_timeout = timeout
        self._max_output_bytes = max_output_bytes
        self._enable_execute = enable_execute
        self._max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self._edit_whitelist = edit_whitelist or frozenset()
        self._edit_blacklist = edit_blacklist or frozenset()
        self._ignore_dirs = ignore_dirs or frozenset()

        # Per-file locks for serializing concurrent edit/write/delete.
        # Keyed by resolved real path to prevent read-modify-write races.
        self._file_locks: dict[str, threading.Lock] = {}
        self._file_locks_guard = threading.Lock()

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
        wr = self.workspace_root.value
        tools: list[StructuredTool] = [
            StructuredTool(
                name="tree",
                description=(
                    "View the directory tree structure. "
                    "Shows directories and files with their sizes in a tree format."
                ),
                args_schema=create_model(
                    "TreeSchema",
                    path=(VirtualPath, Field(default=VirtualPath(wr), description="Root directory to display")),
                    depth=(int, Field(default=3, description="Maximum recursion depth")),
                ),
                func=self._safe_tool_func("tree", self.tree),
                coroutine=self._safe_tool_coroutine("tree", self.atree),
                handle_validation_error=format_validation_error,
            ),
            StructuredTool(
                name="delete",
                description=(
                    "Delete a single file. "
                    "Directories are NOT supported — remove files inside the "
                    "directory first, then the empty directory disappears naturally."
                ),
                args_schema=create_model(
                    "DeleteSchema",
                    path=(VirtualPath, Field(description="Absolute file path to delete")),
                ),
                func=self._safe_tool_func("delete", self.delete),
                coroutine=self._safe_tool_coroutine("delete", self.adelete),
                handle_validation_error=format_validation_error,
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
                        "Returns combined stdout and stderr output.\n\n"
                        "**CRITICAL — Real vs. virtual path mapping:** "
                        f"The workspace root `{wr}` is a virtual path "
                        f"that maps to the real directory `{self._cwd}`. "
                        f"File tools (ls/read/write/edit/grep/glob) accept "
                        f"`{wr}/...` virtual paths, but shell commands "
                        f"in **execute** run directly on the real filesystem. "
                        f"You MUST use real filesystem paths (e.g. "
                        f"`{self._cwd}/src/main.py`) in commands — "
                        f"virtual paths like `{wr}/src/main.py` do NOT "
                        f"exist on the real filesystem and will fail."
                    ),
                    args_schema=create_model(
                        "ExecuteSchema",
                        command=(str, Field(description="Shell command to execute")),
                    ),
                    func=self._safe_tool_func("execute", self.execute),
                    coroutine=self._safe_tool_coroutine("execute", self.aexecute),
                )
            )

        return tools

    @property
    def path_mapping_info(self) -> dict[str, str]:
        wr = self.workspace_root.value
        return {
            "workspace_root": wr,
            "real_root": str(self._cwd),
            "virtual_prefixes": "",
            "path_mapping": f"\n- 虚拟路径 `{wr}/` → 真实路径 `{self._cwd}/`",
        }

    @property
    def description(self) -> str:
        wr = self.workspace_root.value
        os_label = {"win32": "Windows", "darwin": "macOS"}.get(sys.platform, "Unix")
        shell = "cmd /c" if sys.platform == "win32" else "sh -c"
        desc = (
            f"**Environment:** Local {os_label} filesystem "
            f"(working directory: {self._cwd}, shell: {shell}).\n"
            f"**Path mapping:** the workspace root `{wr}` maps to the real "
            f"directory `{self._cwd}` — all file tools must use paths under "
            f"`{wr}`. Paths outside `{wr}` (including `/`) are rejected."
        )
        if self._enable_execute:
            desc += (
                f"\n**execute tool:** shell commands run in `{self._cwd}`.  "
                f"Use real filesystem paths in commands, NOT `{wr}` paths "
                f"— the virtual workspace path does not exist on the real filesystem."
            )
            if sys.platform == "win32":
                desc += (
                    "\n**Windows quoting rules:** cmd.exe has no single-quote "
                    "support and no backslash escaping (unlike bash). "
                    "``python -c \"print('x')\"`` works — double quotes "
                    "outside, single quotes inside the code. But "
                    "``python -c 'print(\"x\")'`` (single-quoted delimiter) "
                    "fails silently (exit 0, no output). For anything "
                    "non-trivial, prefer writing a temporary script file "
                    "with the write tool, then executing it."
                )
        else:
            desc += " [shell execution disabled]"
        return desc

    # ------------------------------------------------------------------
    # Core file operations
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_win_extended_prefix(p: Path) -> Path:
        """Return *p* in ordinary Windows path form.

        ``os.path.realpath`` on Windows obtains the canonical path via
        ``GetFinalPathNameByHandleW``, which always returns an extended-length
        form (``\\\\?\\C:\\...`` or ``\\\\?\\UNC\\server\\share\\...``) and
        normally strips the prefix itself.  Under concurrent directory
        creation it can fail to strip the prefix, leaving a path that is
        physically identical but lexically different from the ordinary form.
        Stripping the prefix is a pure form conversion — it never changes
        what the path points to — and lets ``relative_to`` comparisons work.
        """
        s = str(p)
        if s.startswith("\\\\?\\UNC\\"):
            s = "\\\\" + s[len("\\\\?\\UNC\\"):]
        elif s.startswith("\\\\?\\"):
            s = s[len("\\\\?\\"):]
        return Path(s)

    def _resolve(self, path: VirtualPath) -> Path:
        """Resolve a virtual absolute path to a real filesystem path under ``_cwd``.

        Validates that *path* is under :attr:`workspace_root` and strips
        the prefix before resolving against ``_cwd``.  Raises
        :class:`~mambo_agents.backends.schemas.BackendError` for paths outside the workspace.

        VirtualPath already validates on construction (no ``..``, no
        ``//``, absolute, non-empty), so no additional traversal checks
        are needed here.
        """
        v = path.value
        wr = self.workspace_root.value

        # Must start with workspace root
        if v != wr and not v.startswith(wr + "/"):
            raise BackendError(
                code=ErrorCode.OUTSIDE_WORKSPACE,
                path=path,
                message=f"路径超出工作区，所有文件操作必须在 '{wr}/' 下进行",
            )

        rel = v[len(wr):].lstrip("/")
        if not rel:
            return self._cwd

        resolved = (self._cwd / rel).resolve()
        if os.name == "nt":
            resolved = self._strip_win_extended_prefix(resolved)

        # Prevent symlink escape from _cwd
        try:
            resolved.relative_to(self._cwd)
        except ValueError:
            raise BackendError(
                code=ErrorCode.SYMLINK_ESCAPE,
                path=path,
                message="路径通过符号链接解析到工作区外部",
            ) from None

        return resolved

    def _check_edit_allowed(self, path: VirtualPath) -> bool:
        """Check whether *path* is allowed for edit/write/delete.

        Delegates to :func:`~mambo_agents.backends.utils.check_path_allowed`
        with this backend's whitelist / blacklist.
        """
        return check_path_allowed(
            path.value,
            whitelist=self._edit_whitelist or None,
            blacklist=self._edit_blacklist or None,
        )

    @contextmanager
    def _file_lock(self, resolved: Path):
        """Acquire a per-file lock to serialize concurrent mutations.

        Keyed by the resolved real filesystem path so that edit, write,
        and delete on the same file are strictly serialized, preventing
        read-modify-write race conditions (e.g. concurrent edit+write
        silently overwriting each other's changes).
        """
        key = str(resolved)
        with self._file_locks_guard:
            lock = self._file_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._file_locks[key] = lock
        lock.acquire()
        try:
            yield
        finally:
            lock.release()

    def ls(self, path: VirtualPath) -> LsResult:
        try:
            resolved = self._resolve(path)
        except BackendError as e:
            return LsResult(error=e)
        try:
            if not resolved.exists():
                return LsResult(error=BackendError(code=ErrorCode.NOT_FOUND, path=path, message="路径不存在"))
            if not resolved.is_dir():
                return LsResult(error=BackendError(code=ErrorCode.NOT_DIR, path=path, message="目标是文件，不是目录"))
        except OSError as e:
            return LsResult(error=BackendError(code=ErrorCode.OS_ERROR, path=path, message=str(e)))

        wr = self.workspace_root.value
        infos: list[FileInfo] = []
        errors: list[str] = []
        try:
            for child in sorted(resolved.iterdir(), key=lambda c: (not c.is_dir(), c.name)):
                try:
                    st = child.stat()
                    modified_at = ""
                    ts = getattr(st, "st_mtime", 0)
                    if ts:
                        modified_at = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
                    rel = str(child.relative_to(self._cwd)).replace("\\", "/")
                    if child.is_dir():
                        infos.append(FileInfo(
                            path=f"{wr}/{rel}",
                            is_dir=True,
                            size=0,
                            modified_at=modified_at,
                        ))
                    else:
                        infos.append(FileInfo(
                            path=f"{wr}/{rel}",
                            is_dir=False,
                            size=st.st_size,
                            modified_at=modified_at,
                        ))
                except OSError as e:
                    errors.append(f"Cannot stat '{child.name}': {e}")
        except OSError as e:
            errors.append(f"Listing aborted: {e}")

        error_msg = BackendError(code=ErrorCode.IO_ERROR, message="\n".join(errors)) if errors else None
        return LsResult(error=error_msg, entries=infos)

    def read_raw(
        self,
        file_path: VirtualPath,
        offset: int = 0,
        limit: int | None = 2000,
        include_line_numbers: bool = False,
    ) -> ReadResult:
        if offset < 0:
            return ReadResult(error=BackendError(
                code=ErrorCode.INVALID, path=file_path,
                message=f"offset must be non-negative, got {offset}",
            ))
        if limit is not None and limit < 1:
            return ReadResult(error=BackendError(
                code=ErrorCode.INVALID, path=file_path,
                message=f"limit must be >= 1 (or None for unlimited), got {limit}",
            ))
        try:
            resolved = self._resolve(file_path)
        except BackendError as e:
            return ReadResult(error=e)
        try:
            if not resolved.exists():
                return ReadResult(error=BackendError(code=ErrorCode.NOT_FOUND, path=file_path, message="文件不存在"))
            if resolved.is_dir():
                return ReadResult(error=BackendError(code=ErrorCode.IS_DIR, path=file_path, message="目标是目录"))
        except OSError as e:
            return ReadResult(error=BackendError(code=ErrorCode.OS_ERROR, path=file_path, message=str(e)))

        _O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)

        if get_file_type(file_path.value) != "text":
            try:
                fd = os.open(resolved, os.O_RDONLY | _O_NOFOLLOW)
                with os.fdopen(fd, "rb") as f:
                    raw = f.read()
            except OSError as e:
                return ReadResult(error=BackendError(code=ErrorCode.IO_ERROR, path=file_path, message=str(e)))
            encoded = base64.b64encode(raw).decode("ascii")
            return ReadResult(
                content=encoded,
                total_lines=1,
                encoding="base64",
                file_type=get_file_type(file_path.value),
                mime_type=get_mime_type(file_path.value),
            )

        # Text file: attempt UTF-8 read.
        try:
            fd = os.open(resolved, os.O_RDONLY | _O_NOFOLLOW)
            with os.fdopen(fd, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            return ReadResult(
                error=BackendError(code=ErrorCode.INVALID, path=file_path, message="无法读取，不是可识别的文本或多媒体格式"),
            )
        except OSError as e:
            return ReadResult(error=BackendError(code=ErrorCode.IO_ERROR, path=file_path, message=str(e)))

        lines = content.splitlines(keepends=True)
        total = len(lines)

        if total > 0 and offset >= total:
            return ReadResult(
                error=BackendError(code=ErrorCode.INVALID, path=file_path, message=f"偏移量 {offset} 超过文件长度 ({total} 行)"),
            )

        sliced = lines[offset : offset + limit] if limit is not None else lines[offset:]
        raw_slice = "".join(sliced)
        content = (
            format_with_line_numbers(raw_slice, start_line=offset + 1)
            if include_line_numbers
            else raw_slice
        )
        return ReadResult(
            content=content,
            total_lines=total,
            encoding="utf-8",
        )

    def write(
        self, file_path: VirtualPath, content: str, overwrite: bool = False,
    ) -> WriteResult:
        if not self._check_edit_allowed(file_path):
            return WriteResult(
                error=BackendError(code=ErrorCode.EDIT_NOT_ALLOWED, path=file_path, message="路径不允许写入"),
            )
        if get_file_type(file_path.value) != "text":
            return WriteResult(error=BackendError(
                code=ErrorCode.INVALID, path=file_path,
                message="无法写入该文件，非文本格式不支持写入",
            ))
        try:
            resolved = self._resolve(file_path)
        except BackendError as e:
            return WriteResult(error=e)
        with self._file_lock(resolved):
            try:
                if resolved.is_dir():
                    return WriteResult(
                        error=BackendError(code=ErrorCode.IS_DIR, path=file_path, message="目标是目录，无法写入"),
                    )
            except OSError as e:
                return WriteResult(error=BackendError(code=ErrorCode.OS_ERROR, path=file_path, message=str(e)))
            if resolved.exists() and not overwrite:
                return WriteResult(
                    error=BackendError(code=ErrorCode.ALREADY_EXISTS, path=file_path, message="文件已存在，请用 edit() 修改或用 overwrite=True 覆盖"),
                )
            if resolved.exists():
                _O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
                try:
                    fd = os.open(resolved, os.O_RDONLY | _O_NOFOLLOW)
                    with os.fdopen(fd, "r", encoding="utf-8") as f:
                        f.read(1)
                except UnicodeDecodeError:
                    return WriteResult(error=BackendError(
                        code=ErrorCode.INVALID, path=file_path,
                        message="无法写入该文件，非文本文件仅支持读取",
                    ))
                except OSError as e:
                    return WriteResult(error=BackendError(
                        code=ErrorCode.IO_ERROR, path=file_path, message=str(e),
                    ))

            try:
                resolved.parent.mkdir(parents=True, exist_ok=True)
                _O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
                flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | _O_NOFOLLOW
                fd = os.open(resolved, flags, 0o644)
                with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
                    f.write(content)
            except OSError as e:
                return WriteResult(error=BackendError(code=ErrorCode.IO_ERROR, path=file_path, message=str(e)))

            return WriteResult(path=file_path)

    def edit(
        self,
        file_path: VirtualPath,
        old_str: str,
        new_str: str,
        *,
        replace_all: bool = False,
    ) -> EditResult:
        if not old_str:
            return EditResult(error=BackendError(code=ErrorCode.INVALID, message="old_str 不能为空"))
        if not self._check_edit_allowed(file_path):
            return EditResult(
                error=BackendError(code=ErrorCode.EDIT_NOT_ALLOWED, path=file_path, message="路径不允许编辑"),
            )
        if get_file_type(file_path.value) != "text":
            return EditResult(error=BackendError(
                code=ErrorCode.INVALID, path=file_path,
                message="无法编辑该文件，非文本格式不支持编辑",
            ))
        try:
            resolved = self._resolve(file_path)
        except BackendError as e:
            return EditResult(error=e)
        with self._file_lock(resolved):
            try:
                if not resolved.exists():
                    return EditResult(
                        error=BackendError(code=ErrorCode.NOT_FOUND, path=file_path, message="文件不存在，请用 write() 创建新文件"),
                    )
                if resolved.is_dir():
                    return EditResult(
                        error=BackendError(code=ErrorCode.IS_DIR, path=file_path, message="目标是目录，无法编辑"),
                    )
            except OSError as e:
                return EditResult(error=BackendError(code=ErrorCode.OS_ERROR, path=file_path, message=str(e)))

            _O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
            try:
                fd = os.open(resolved, os.O_RDONLY | _O_NOFOLLOW)
                with os.fdopen(fd, "r", encoding="utf-8") as f:
                    content = f.read()
            except UnicodeDecodeError:
                return EditResult(error=BackendError(
                    code=ErrorCode.INVALID, path=file_path,
                    message="无法编辑该文件，非文本文件仅支持读取",
                ))
            except OSError as e:
                return EditResult(error=BackendError(code=ErrorCode.IO_ERROR, path=file_path, message=str(e)))

            old_str = normalize_line_endings(old_str)
            new_str = normalize_line_endings(new_str)

            occurrences = content.count(old_str)

            if occurrences == 0:
                # Detect trailing-newline mismatch
                mismatch = detect_trailing_newline_mismatch(
                    old_str, content,
                )
                if mismatch is not None:
                    return mismatch
                return EditResult(
                    error=BackendError(code=ErrorCode.OLD_STR_NOT_FOUND, path=file_path, message="未找到要替换的文本，请先读文件确认内容"),
                )

            if occurrences > 1 and not replace_all:
                return EditResult(
                    error=BackendError(code=ErrorCode.MULTI_OCCURRENCES, path=file_path, message=f"匹配到 {occurrences} 处，请用 replace_all=True 替换全部或提供更精确的上下文"),
                )

            try:
                flags = os.O_WRONLY | os.O_TRUNC | _O_NOFOLLOW
                fd = os.open(resolved, flags)
                with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
                    f.write(content.replace(old_str, new_str))
            except OSError as e:
                return EditResult(error=BackendError(code=ErrorCode.IO_ERROR, path=file_path, message=str(e)))

            return EditResult(path=file_path, occurrences=occurrences)

    # ------------------------------------------------------------------
    # grep — ripgrep-first with Python fallback and file-size guard
    # ------------------------------------------------------------------

    def grep(
        self,
        pattern: str,
        path: VirtualPath = VirtualPath("/workspace"),
        glob: str | None = None,
        regex: bool = True,
        offset: int = 0,
        limit: int | None = None,
    ) -> GrepResult:
        if not pattern:
            return GrepResult(error=BackendError(code=ErrorCode.INVALID, message="搜索模式不能为空"))
        try:
            resolved = self._resolve(path)
        except BackendError as e:
            return GrepResult(error=e)
        try:
            if not resolved.exists():
                return GrepResult(error=BackendError(code=ErrorCode.NOT_FOUND, path=path, message="路径不存在"))
        except OSError as e:
            return GrepResult(error=BackendError(code=ErrorCode.OS_ERROR, path=path, message=str(e)))

        is_dir = resolved.is_dir()
        wr = self.workspace_root.value

        # 1) Try ripgrep (handles files and dirs equally; --glob prunes files
        # early for performance, but uses gitignore semantics so we still
        # post-filter with POSIX fnmatch_path below).
        rg_glob = glob if is_dir else None
        results = self._ripgrep_grep(pattern, resolved, rg_glob, regex)
        if results is not None:
            # ripgrep paths are physical → convert to virtual
            matches: list[GrepMatch] = []
            for fpath, items in results.items():
                try:
                    rel = str(Path(fpath).relative_to(self._cwd)).replace("\\", "/")
                    virt = f"{wr}/{rel}"
                except ValueError:
                    continue
                # Post-filter with POSIX glob (ripgrep --glob uses gitignore semantics,
                # not POSIX, so we filter on our side to keep consistent behaviour).
                if glob and is_dir:
                    try:
                        rel_path = str(Path(fpath).relative_to(resolved)).replace("\\", "/")
                    except ValueError:
                        continue
                    if not fnmatch_path(rel_path, glob):
                        continue
                for li, text in items:
                    if _in_ignored_dir(virt, self._ignore_dirs, wr):
                        continue
                    matches.append(GrepMatch(path=virt, line=li, text=text))
            matches.sort(key=lambda m: (str(m.path), m.line))
            return self._apply_grep_limit(matches, offset, limit, pattern=pattern, regex=regex)

        # 2) Python fallback with file-size guard

        matches: list[GrepMatch] = []
        skipped: int = 0
        try:
            if regex:
                compiled = re.compile(pattern)
            else:
                compiled = re.compile(re.escape(pattern))
        except re.error as e:
            return GrepResult(error=BackendError(code=ErrorCode.INVALID, message=f"无效正则: {e}"))

        if not is_dir:
            # Single-file search
            try:
                lines = resolved.read_text(encoding="utf-8").split("\n")
            except (UnicodeDecodeError, OSError):
                return self._apply_grep_limit([], offset, limit, pattern=pattern, regex=regex)
            for li, line in enumerate(lines, start=1):
                if len(matches) >= self._max_grep_matches:
                    break
                if compiled.search(line):
                    rel = str(resolved.relative_to(self._cwd)).replace("\\", "/")
                    virt_path = f"{wr}/{rel}"
                    if _in_ignored_dir(virt_path, self._ignore_dirs, wr):
                        continue
                    matches.append(GrepMatch(path=virt_path, line=li, text=line))
            return self._apply_grep_limit(matches, offset, limit, pattern=pattern, regex=regex)

        search_dir = resolved
        error_msg: BackendError | None = None
        try:
            for fp in sorted(search_dir.rglob("*")):
                if len(matches) >= self._max_grep_matches:
                    break
                try:
                    if not fp.is_file():
                        continue
                except OSError:
                    continue

                if glob:
                    try:
                        rel_path = str(fp.relative_to(search_dir)).replace("\\", "/")
                    except ValueError:
                        continue
                    if not fnmatch_path(rel_path, glob):
                        continue
                if get_file_type(fp.suffix) != "text":
                    continue

                # Skip files exceeding the size limit
                try:
                    if fp.stat().st_size > self._max_file_size_bytes:
                        skipped += 1
                        continue
                except OSError:
                    continue

                try:
                    lines = fp.read_text(encoding="utf-8").split("\n")
                except (UnicodeDecodeError, OSError):
                    continue

                for li, line in enumerate(lines, start=1):
                    if len(matches) >= self._max_grep_matches:
                        break
                    if compiled.search(line):
                        rel = str(fp.relative_to(self._cwd)).replace("\\", "/")
                        virt_path = f"{wr}/{rel}"
                        if _in_ignored_dir(virt_path, self._ignore_dirs, wr):
                            continue
                        matches.append(GrepMatch(path=virt_path, line=li, text=line))
        except OSError as e:
            matches.sort(key=lambda m: (str(m.path), m.line))
            result = self._apply_grep_limit(matches, offset, limit, pattern=pattern, regex=regex)
            return GrepResult(
                error=BackendError(code=ErrorCode.OS_ERROR, path=path, message=str(e)),
                matches=result.matches,
                truncated=result.truncated,
                total_matches=result.total_matches,
            )

        if skipped:
            error_msg = BackendError(
                code=ErrorCode.FILE_TOO_LARGE,
                path=path,
                message=f"跳过 {skipped} 个大文件 (>{self._max_file_size_bytes // (1024 * 1024)} MB)",
            )

        matches.sort(key=lambda m: (str(m.path), m.line))
        result = self._apply_grep_limit(matches, offset, limit, pattern=pattern, regex=regex)
        if error_msg:
            result = GrepResult(
                error=error_msg,
                matches=result.matches,
                truncated=result.truncated,
                total_matches=result.total_matches,
            )
        return result

    # ------------------------------------------------------------------
    # ripgrep helper
    # ------------------------------------------------------------------

    def _ripgrep_grep(
        self,
        pattern: str,
        base_dir: Path,
        include_glob: str | None,
        regex: bool = True,
    ) -> dict[str, list[tuple[int, str]]] | None:
        """Search with ripgrep (JSON output).

        Returns:
            Dict mapping physical file paths → list of (line, text), or
            ``None`` if ripgrep is unavailable or times out.
        """
        if not shutil.which("rg"):
            return None

        # Search everything: hidden files/dirs and gitignored paths are included;
        # hiding is controlled exclusively by ignore_dirs (see _in_ignored_dir).
        cmd = ["rg", "--json", "--hidden", "--no-ignore"]
        if not regex:
            cmd.append("-F")
        if include_glob:
            cmd.extend(["--glob", include_glob])
        cmd.extend(["--", pattern, str(base_dir)])

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=30,
                check=False,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError):
            return None

        if proc.stdout is None:
            return None

        results: dict[str, list[tuple[int, str]]] = {}
        for line in proc.stdout.splitlines():
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if data.get("type") != "match":
                continue
            pdata = data.get("data", {})
            ftext = pdata.get("path", {}).get("text")
            if not ftext:
                continue
            ln = pdata.get("line_number")
            lt = pdata.get("lines", {}).get("text", "").rstrip("\n")
            if ln is None:
                continue
            results.setdefault(ftext, []).append((int(ln), lt))

        return results

    def glob(self, pattern: str, path: VirtualPath = VirtualPath("/workspace")) -> GlobResult:
        if not pattern:
            return GlobResult(error=BackendError(code=ErrorCode.INVALID, message="搜索模式不能为空"))
        try:
            resolved = self._resolve(path)
        except BackendError as e:
            return GlobResult(error=e)

        try:
            if not resolved.exists():
                return GlobResult(error=BackendError(code=ErrorCode.NOT_FOUND, path=path, message="路径不存在"))
            if not resolved.is_dir():
                return GlobResult(error=BackendError(code=ErrorCode.NOT_DIR, path=path, message="目标是文件，不是目录"))
        except OSError as e:
            return GlobResult(error=BackendError(code=ErrorCode.OS_ERROR, path=path, message=str(e)))

        wr = self.workspace_root.value
        matches: list[FileInfo] = []
        errors: list[str] = []
        try:
            for fp in resolved.glob(pattern):
                try:
                    is_dir = fp.is_dir()
                except OSError:
                    continue
                try:
                    rel = str(fp.relative_to(self._cwd)).replace("\\", "/")
                    virt_path = f"{wr}/{rel}"
                    st = fp.stat()
                except OSError as e:
                    errors.append(f"无法获取文件信息 '{fp}': {e}")
                    continue
                matches.append(FileInfo(
                    path=virt_path,
                    is_dir=is_dir,
                    size=st.st_size if not is_dir else 0,
                ))
        except OSError as e:
            errors.append(f"Glob 搜索中断: {e}")

        error_msg = BackendError(code=ErrorCode.IO_ERROR, message="\n".join(errors)) if errors else None
        return GlobResult(error=error_msg, matches=matches)

    # ------------------------------------------------------------------
    # Extra operations: tree, delete, execute
    # ------------------------------------------------------------------

    def tree(self, path: VirtualPath = VirtualPath("/workspace"), depth: int = 3) -> str:
        """Render a directory tree.

        Args:
            path: Root directory to display (virtual path, default to workspace root).
            depth: Maximum recursion depth (default 3, must be >= 1).

        Returns:
            Formatted tree string.
        """
        if depth < 1:
            return f"Invalid depth value: {depth}. Depth must be a positive integer (>= 1)."

        try:
            resolved = self._resolve(path)
        except BackendError as e:
            return str(e)
        if not resolved.exists():
            return f"Path '{path}' not found."
        if not resolved.is_dir():
            return f"Path '{path}' is not a directory."

        entries = _walk_tree(
            resolved,
            depth,
            cwd=self._cwd,
            ignore_dirs=self._ignore_dirs,
        )
        return format_tree_entries(entries)

    def delete(self, path: VirtualPath) -> DeleteResult:
        """Delete a single **file**.

        Directories are rejected — the agent must remove files inside
        the directory individually before the directory can be deleted.

        Args:
            path: Virtual path to delete.

        Returns:
            Success or error result.
        """
        if not self._check_edit_allowed(path):
            return DeleteResult(
                error=BackendError(code=ErrorCode.EDIT_NOT_ALLOWED, path=path, message="路径不允许删除"),
                path=path,
            )
        try:
            resolved = self._resolve(path)
        except BackendError as e:
            return DeleteResult(error=e, path=path)

        # Safety: refuse to delete the root_dir itself
        if resolved == self._cwd:
            return DeleteResult(error=BackendError(code=ErrorCode.INVALID, path=path, message="不能删除根工作目录"), path=path)

        with self._file_lock(resolved):
            if not resolved.exists():
                return DeleteResult(error=BackendError(code=ErrorCode.NOT_FOUND, path=path, message="路径不存在"), path=path)

            if resolved.is_dir():
                return DeleteResult(
                    error=BackendError(code=ErrorCode.IS_DIR, path=path, message="目标是目录，delete 工具只能删除单个文件"),
                    path=path,
                )

            try:
                resolved.unlink()
            except OSError as e:
                return DeleteResult(error=BackendError(code=ErrorCode.IO_ERROR, path=path, message=str(e)), path=path)

            return DeleteResult(path=path)

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

        # Build platform-specific shell invocation.
        # Windows: pass the raw command string with shell=True so cmd /c
        # receives it verbatim, exactly as typed at the prompt.  The list
        # form ["cmd", "/c", command] went through subprocess.list2cmdline,
        # which escapes " as \" — cmd.exe has no backslash escaping, so any
        # command containing double quotes (e.g. python -c "print('x')") was
        # silently mangled.  POSIX keeps the list form: execve passes argv
        # directly, no escaping involved.
        if sys.platform == "win32":
            shell_cmd: str | list[str] = command
            use_shell = True
        else:
            shell_cmd = ["sh", "-c", command]
            use_shell = False
        display_cmd = command if isinstance(shell_cmd, str) else " ".join(shell_cmd)

        try:
            # Capture raw bytes and decode manually.  text=True decodes in a
            # reader thread: on a cp936 system, UTF-8 child output raises
            # UnicodeDecodeError inside that thread (stdout silently becomes
            # None) and the fallback below never runs.  Decoding here in the
            # main thread keeps the fallback working.
            result = subprocess.run(
                shell_cmd,
                check=False,
                capture_output=True,
                stdin=subprocess.DEVNULL,
                timeout=effective_timeout,
                env=self._env if self._env else None,
                cwd=str(self._cwd),
                shell=use_shell,
            )

            # Decode raw bytes with UTF-8 first (modern CLIs emit UTF-8;
            # see decode_output for the full fallback strategy).
            output_parts: list[str] = []
            if result.stdout:
                output_parts.append(decode_output(result.stdout).rstrip())
            if result.stderr:
                for line in decode_output(result.stderr).strip().split("\n"):
                    output_parts.append(f"[stderr] {line}")

            if output_parts:
                output = "\n".join(output_parts)
            else:
                if result.returncode != 0:
                    output = (
                        f"<no output> — command failed with exit code {result.returncode}\n"
                        f"Ran: {display_cmd}"
                    )
                else:
                    output = "<no output>"

            # Truncation
            if len(output) > self._max_output_bytes:
                output = output[: self._max_output_bytes]
                output += f"\n\n... Output truncated at {self._max_output_bytes} bytes."

            # Add exit code if non-zero (only when there was output to display)
            if result.returncode != 0 and output_parts:
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

    async def atree(self, path: VirtualPath = VirtualPath("/workspace"), depth: int = 3) -> str:
        """Async: Render a directory tree."""
        return await asyncio.to_thread(self.tree, path, depth)

    async def adelete(self, path: VirtualPath) -> DeleteResult:
        """Async: Delete a file."""
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
        files: list[tuple[VirtualPath, bytes]],
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
            if not self._check_edit_allowed(path):
                results.append(UploadFileResult(
                    path=path,
                    error=BackendError(code=ErrorCode.EDIT_NOT_ALLOWED, path=path, message="路径不允许写入"),
                ))
                continue
            try:
                resolved = self._resolve(path)
                with self._file_lock(resolved):
                    resolved.parent.mkdir(parents=True, exist_ok=True)
                    resolved.write_bytes(raw_content)
                results.append(UploadFileResult(path=path))
            except BackendError as e:
                results.append(UploadFileResult(path=path, error=e))
            except OSError as e:
                results.append(UploadFileResult(path=path, error=BackendError(code=ErrorCode.IO_ERROR, path=path, message=str(e))))
            except Exception as e:
                results.append(
                    UploadFileResult(
                        path=path, error=BackendError(code=ErrorCode.INVALID, path=path, message=f"{type(e).__name__}: {e}")
                    )
                )
        return results

    def download_files(
        self,
        paths: list[VirtualPath],
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
                with self._file_lock(resolved):
                    if not resolved.exists():
                        results.append(
                        DownloadFileResult(
                            path=path, content=None, error=BackendError(code=ErrorCode.NOT_FOUND, path=path, message="文件不存在")
                        )
                        )
                        continue
                    if resolved.is_dir():
                        results.append(
                            DownloadFileResult(
                                path=path, content=None, error=BackendError(code=ErrorCode.IS_DIR, path=path, message="目标是目录")
                            )
                        )
                        continue
                    raw = resolved.read_bytes()
                results.append(
                    DownloadFileResult(path=path, content=raw)
                )
            except OSError as e:
                results.append(
                    DownloadFileResult(path=path, content=None, error=BackendError(code=ErrorCode.IO_ERROR, path=path, message=str(e)))
                )
            except Exception as e:
                results.append(
                    DownloadFileResult(
                        path=path,
                        content=None,
                        error=BackendError(code=ErrorCode.INVALID, path=path, message=f"{type(e).__name__}: {e}"),
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
    cwd: Path | None = None,
    current_depth: int = 0,
    ignore_dirs: frozenset[str] = frozenset(),
) -> list[TreeEntry]:
    """Recursively walk a directory tree.

    Returns:
        List of ``TreeEntry`` in DFS order (directories first, then files).
        Directories may carry a ``marker``: ``"empty"``, ``"ignore"``, or
        ``"depth_exceeded"``.
    """
    if current_depth >= max_depth:
        return []

    entries: list[TreeEntry] = []
    try:
        children = sorted(root.iterdir(), key=lambda c: (not c.is_dir(), c.name))
    except OSError:
        return []

    # Directories first
    for child in children:
        if not child.is_dir():
            continue

        # Ignored directory: show it but skip children
        if child.name in ignore_dirs:
            entries.append(TreeEntry(
                name=child.name + "/",
                depth=current_depth,
                marker="ignore",
            ))
            continue

        # At depth limit: check if the directory has children
        if current_depth + 1 >= max_depth:
            try:
                has_children = any(True for _ in child.iterdir())
            except OSError:
                has_children = False
            marker: Literal["", "empty", "ignore", "depth_exceeded"] = (
                "depth_exceeded" if has_children else "empty"
            )
            entries.append(TreeEntry(
                name=child.name + "/",
                depth=current_depth,
                marker=marker,
            ))
            continue

        # Non-limit, non-ignored: check emptiness first
        try:
            sub_children = list(child.iterdir())
        except OSError:
            sub_children = []

        if not sub_children:
            entries.append(TreeEntry(
                name=child.name + "/",
                depth=current_depth,
                marker="empty",
            ))
        else:
            entries.append(TreeEntry(
                name=child.name + "/",
                depth=current_depth,
            ))
            sub = _walk_tree(
                child,
                max_depth,
                cwd=cwd,
                current_depth=current_depth + 1,
                ignore_dirs=ignore_dirs,
            )
            entries.extend(sub)

    # Then files
    for child in children:
        if child.is_file():
            try:
                size = child.stat().st_size
            except OSError:
                size = 0
            entries.append(TreeEntry(
                name=f"{child.name} ({human_size(size)})",
                depth=current_depth,
            ))

    return entries


def _in_ignored_dir(
    virt_path: str,
    ignore_dirs: frozenset[str],
    workspace_root: str,
) -> bool:
    """Segment-level check: True if any parent-directory segment of *virt_path*
    matches one of the bare directory names in *ignore_dirs*.

    Matches path segments only, not substrings — e.g. with ``ignore_dirs={"a"}``,
    ``/workspace/ac/x.txt`` passes (segment ``"ac"`` != ``"a"``) while
    ``/workspace/a/c/x.txt`` is excluded (segment ``"a"`` matches).
    """
    virt_path = str(virt_path)
    rel = virt_path[len(workspace_root):].lstrip("/")
    return any(seg in ignore_dirs for seg in rel.split("/")[:-1])
