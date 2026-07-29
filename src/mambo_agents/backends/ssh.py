"""SshBackend – remote filesystem backend via SSH / SFTP.

Connects to a remote server through SSH using paramiko, providing the
six core file operations plus ``tree``, ``delete``, and ``execute``.
Heavy operations (grep, glob, edit, tree) are pushed to the remote
shell to minimise network round-trips.
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
import shlex
import socket
import threading
from datetime import datetime, timezone
from pathlib import PurePosixPath
from types import TracebackType
from typing import Literal

import paramiko
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
    _get_file_type,
    _get_mime_type,
)
from mambo_agents.backends.utils import (
    TreeEntry,
    check_path_allowed,
    normalize_line_endings,
    format_tree_entries,
    format_validation_error,
    format_with_line_numbers,
)
from mambo_agents.backends.schemas import BackendError, DeleteResult, ErrorCode, VirtualPath, human_size

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_SSH_CONNECT_TIMEOUT = 30
_DEFAULT_EXECUTE_TIMEOUT = 120
_MAX_OUTPUT_BYTES = 100_000


# ---------------------------------------------------------------------------
# SshBackend
# ---------------------------------------------------------------------------


class SshBackend(BackendProtocol):
    """Remote filesystem backend via SSH with shell command execution.

    Operates on a remote server's filesystem through an SSH connection.
    Provides the six core file operations plus three extra tools:

    - ``tree``    — visual directory tree via remote ``find``
    - ``delete``  — remove files and directories via ``rm -rf``
    - ``execute`` — run arbitrary shell commands on the remote server

    **Performance strategy**: bulk / traversal operations (``grep``,
    ``glob``, ``edit``, ``tree``, ``ls``) are executed *on the remote
    machine* via ``exec_command``, avoiding per-file SFTP round-trips.
    Only ``read`` and ``write`` transfer file contents through SFTP.

    Parameters:
        host: SSH server hostname or IP address.
        username: SSH login username.
        port: SSH port (default 22).
        password: SSH password for authentication.
            Either *password* or *key_filename* must be provided.
        key_filename: Path to a private key file for key-based auth.
        remote_root: Remote working directory.  ``"~"`` (default) is
            expanded to the user's home directory on connect.
        workspace_root: Virtual path prefix acting as the workspace root
            (default ``"/workspace"``).  All file paths must live under
            this prefix — paths outside are rejected so the AI never
            perceives the virtual filesystem as a real system root.
        connect_timeout: SSH connection timeout in seconds (default 30).
        execute_timeout: Default timeout for shell commands (default 120).
        max_output_bytes: Max bytes captured from command output.
        edit_whitelist: Virtual path prefixes allowed for edit/write/delete.
            Mutually exclusive with *edit_blacklist*.
        edit_blacklist: Virtual path prefixes forbidden for edit/write/delete.
            Mutually exclusive with *edit_whitelist*.
        ignore_dirs: Virtual directory paths whose children are hidden
            in ``tree()`` output.  The directory itself is shown with
            an ``/(ignore)`` marker but its content is not expanded.
    """

    # Default per-tool timeout values specific to this backend (overridable via __init__).
    _BACKEND_DEFAULT_TIMEOUTS = ToolTimeouts(tree=60.0, delete=30.0, execute=500.0)

    def __init__(
        self,
        host: str,
        username: str,
        *,
        port: int = 22,
        password: str | None = None,
        key_filename: str | None = None,
        remote_root: str = "~",
        workspace_root: VirtualPath = VirtualPath("/workspace"),
        connect_timeout: int = _DEFAULT_SSH_CONNECT_TIMEOUT,
        execute_timeout: int = _DEFAULT_EXECUTE_TIMEOUT,
        max_output_bytes: int = _MAX_OUTPUT_BYTES,
        enable_execute: bool = False,
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
        if password is None and key_filename is None:
            raise ValueError(
                "Either 'password' or 'key_filename' must be provided for SSH authentication."
            )

        if edit_whitelist is not None and edit_blacklist is not None:
            raise ValueError(
                "edit_whitelist and edit_blacklist are mutually exclusive. "
                "Provide at most one of them."
            )

        self.workspace_root = VirtualPath(workspace_root)
        self._host = host
        self._username = username
        self._port = port
        self._password = password
        self._key_filename = key_filename
        self._raw_remote_root = remote_root
        self._connect_timeout = connect_timeout
        self._execute_timeout = execute_timeout
        self._max_output_bytes = max_output_bytes
        self._enable_execute = enable_execute
        self._edit_whitelist = edit_whitelist or frozenset()
        self._edit_blacklist = edit_blacklist or frozenset()
        self._ignore_dirs = ignore_dirs or frozenset()

        self._client: paramiko.SSHClient | None = None
        self._sftp: paramiko.SFTPClient | None = None
        self._remote_root: str = ""  # resolved absolute path, set by _connect()
        self._has_python3: bool = False  # detected in _connect()
        self._async_lock: asyncio.Lock = asyncio.Lock()
        self._sync_lock = threading.RLock()

        self._connect()

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def _connect(self) -> None:
        """Establish SSH + SFTP connection and resolve *remote_root* to
        an absolute path."""
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        connect_kwargs: dict = {
            "hostname": self._host,
            "port": self._port,
            "username": self._username,
            "timeout": self._connect_timeout,
        }
        # Optional auth – at least one has been validated in __init__
        if self._password is not None:
            connect_kwargs["password"] = self._password
        if self._key_filename is not None:
            connect_kwargs["key_filename"] = self._key_filename

        client.connect(**connect_kwargs)
        self._client = client
        self._sftp = client.open_sftp()

        # Resolve remote_root: expand "~" → $HOME
        remote_root = self._raw_remote_root
        if remote_root.startswith("~"):
            _, stdout, _ = client.exec_command("echo $HOME")
            home = stdout.read().decode().strip()
            # Replace the leading ~ with $HOME
            remote_root = home + remote_root[1:]

        # Normalise: strip trailing slashes (both / and \ for Windows remotes)
        remote_root = remote_root.rstrip("/").rstrip("\\")

        # Validate: remote_root must be non-empty and MUST exist on the remote
        if not remote_root:
            raise ValueError(
                f"remote_root '{self._raw_remote_root}' resolved to an empty path. "
                f"Please provide a valid absolute path on the remote host."
            )
        try:
            _ = self._sftp.stat(remote_root)
        except FileNotFoundError:
            raise ValueError(
                f"remote_root '{remote_root}' does not exist on the remote host "
                f"({self._host}). Please create it first or use an existing directory."
            )
        except PermissionError as e:
            raise PermissionError(
                f"remote_root '{remote_root}' is not accessible on the remote host "
                f"({self._host}): {e}"
            )

        self._remote_root = remote_root

        # Detect whether python3 is available (needed by edit())
        self._has_python3 = False
        try:
            _stdin, stdout, _stderr = client.exec_command("python3 --version", timeout=10)
            if stdout.channel.recv_exit_status() == 0:
                self._has_python3 = True
        except Exception:
            pass

    def close(self) -> None:
        """Close the SSH and SFTP connections."""
        if self._sftp is not None:
            self._sftp.close()
            self._sftp = None
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> SshBackend:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    @property
    def is_connected(self) -> bool:
        """Return ``True`` if the SSH transport is active."""
        transport = self._client.get_transport() if self._client else None
        return transport is not None and transport.is_active()

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------

    def _resolve(self, path: VirtualPath) -> str:
        """Map a virtual absolute path to a remote filesystem path.

        Validates that *path* is under :attr:`workspace_root` and strips
        the prefix before resolving against *remote_root*.  Raises
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
            return self._remote_root

        return f"{self._remote_root}/{rel}"

    # ------------------------------------------------------------------
    # Remote command execution
    # ------------------------------------------------------------------

    def _exec(
        self, command: str, *, timeout: int | None = None
    ) -> tuple[str, str, int]:
        """Execute *command* on the remote server via SSH.

        Enforces a hard timeout via a background watchdog timer that
        forcibly closes the channel after *timeout* seconds, guarding
        against cases where ``channel.settimeout()`` is defeated by
        SSH keep-alive traffic (e.g. ``sleep N`` produces no I/O
        stalls).  Reads stdout and stderr concurrently in background
        threads to avoid pipe-buffer deadlocks.

        Returns:
            ``(stdout, stderr, exit_code)`` tuple.
        """
        t = timeout if timeout is not None else self._execute_timeout
        _stdin, stdout, stderr = self._client.exec_command(command, timeout=t)

        channel = stdout.channel
        channel.settimeout(t)

        # Hard-timeout watchdog: force-close the channel after *t* seconds
        # even if SSH keep-alive traffic prevents socket.timeout from firing.
        # Uses threading.Event so the main thread can detect when the hard
        # timer has fired even if recv_exit_status() returns without raising.
        _timed_out = threading.Event()

        def _close_channel() -> None:
            _timed_out.set()
            try:
                channel.close()
            except Exception:
                pass

        _hard_timer = threading.Timer(t, _close_channel)
        _hard_timer.start()

        # Read stdout / stderr in background threads to prevent pipe-
        # buffer deadlock: if the remote process writes a lot to stderr
        # we must drain it even while we are still reading stdout.
        out_chunks: list[bytes] = []
        err_chunks: list[bytes] = []

        def _drain(stream: object, chunks: list[bytes]) -> None:
            try:
                while True:
                    data = stream.read(65536)  # type: ignore[attr-defined]
                    if not data:
                        break
                    chunks.append(data)
            except Exception:
                pass

        out_thread = threading.Thread(target=_drain, args=(stdout, out_chunks), daemon=True)
        err_thread = threading.Thread(target=_drain, args=(stderr, err_chunks), daemon=True)
        out_thread.start()
        err_thread.start()

        try:
            try:
                exit_code = channel.recv_exit_status()
            except socket.timeout:
                _timed_out.set()
                channel.close()
                # Give drain threads a brief window to collect partial data
                out_thread.join(timeout=3)
                err_thread.join(timeout=3)
                partial_out = b"".join(out_chunks).decode(errors="replace")
                partial_err = b"".join(err_chunks).decode(errors="replace")
                hint = f"\n(stderr) {partial_err}" if partial_err.strip() else ""
                return (partial_out, f"Timeout: command exceeded {t} seconds.{hint}", -1)
            except Exception as exc:
                channel.close()
                return ("", f"Command execution error: {exc}", -1)

            # Check if hard timer fired (channel was force-closed → recv_exit_status()
            # returns -1 without raising).  Must be checked BEFORE the finally block
            # cancels the timer.
            if _timed_out.is_set():
                out_thread.join(timeout=3)
                err_thread.join(timeout=3)
                partial_out = b"".join(out_chunks).decode(errors="replace")
                partial_err = b"".join(err_chunks).decode(errors="replace")
                hint = f"\n(stderr) {partial_err}" if partial_err.strip() else ""
                return (partial_out, f"Timeout: command exceeded {t} seconds.{hint}", -1)
        finally:
            _hard_timer.cancel()

        out_thread.join(timeout=5)
        err_thread.join(timeout=5)

        out = b"".join(out_chunks).decode(errors="replace")
        err = b"".join(err_chunks).decode(errors="replace")
        return out, err, exit_code

    # ------------------------------------------------------------------
    # tools property
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
                        "Execute a shell command on the remote server via SSH. "
                        "Returns combined stdout and stderr output.\n\n"
                        "**CRITICAL — Real vs. virtual path mapping:** "
                        f"The workspace root `{wr}` is a virtual path "
                        f"that maps to the remote directory `{self._remote_root}`. "
                        f"File tools (ls/read/write/edit/grep/glob) accept "
                        f"`{wr}/...` virtual paths, but shell commands "
                        f"in **execute** run directly on the remote filesystem. "
                        f"You MUST use real remote filesystem paths (e.g. "
                        f"`{self._remote_root}/src/main.py`) in commands — "
                        f"virtual paths like `{wr}/src/main.py` do NOT "
                        f"exist on the remote filesystem and will fail."
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
            "real_root": str(self._remote_root),
            "virtual_prefixes": "",
            "path_mapping": f"\n- 虚拟路径 `{wr}/` → 远程真实路径 `{self._remote_root}/`",
        }

    @property
    def description(self) -> str:
        wr = self.workspace_root.value
        py3_note = (
            "" if self._has_python3
            else (
                "\n⚠️  **python3 not detected** — grep/glob/tree may use GNU-only "
                "fallback commands (``find -printf``, ``grep --include=``) "
                "which can produce inaccurate results on non-GNU systems."
            )
        )
        exec_note = ""
        if self._enable_execute:
            exec_note = (
                f"\n**execute tool:** shell commands run in `{self._remote_root}`.  "
                f"Use real filesystem paths in commands, NOT `{wr}` paths "
                f"— the virtual workspace path does not exist on the remote filesystem."
            )
        else:
            exec_note = " [shell execution disabled]"

        return (
            f"**Environment:** Remote Linux server via SSH "
            f"(host: {self._host}, working directory: {self._remote_root}).{exec_note}\n"
            f"**Path mapping:** the workspace root `{wr}` maps to the remote "
            f"directory `{self._remote_root}` — all file tools must use paths under "
            f"`{wr}`. Paths outside `{wr}` (including `/`) are rejected."
            f"{py3_note}"
        )

    def ls(self, path: VirtualPath) -> LsResult:
        """List files and directories under *path* (non-recursive).

        Uses SFTP ``listdir_attr`` for structured, locale-independent output.
        """
        with self._sync_lock:
            try:
                remote = self._resolve(path)
            except BackendError as e:
                return LsResult(error=e)
            try:
                st_mode = self._sftp.stat(remote).st_mode
                if not self._attr_is_dir_maybe(st_mode):
                    return LsResult(error=BackendError(code=ErrorCode.NOT_DIR, path=path, message="目标是文件，不是目录"))
                attrs = self._sftp.listdir_attr(remote)
            except FileNotFoundError:
                return LsResult(error=BackendError(code=ErrorCode.NOT_FOUND, path=path, message="路径不存在"))
            except OSError as e:
                return LsResult(error=BackendError(code=ErrorCode.OS_ERROR, path=path, message=str(e)))

            infos: list[FileInfo] = []
            # Sort: directories first, then by name
            sorted_attrs = sorted(attrs, key=lambda a: (not self._attr_is_dir(a), a.filename))

            for attr in sorted_attrs:
                is_dir = self._attr_is_dir(attr)
                size = attr.st_size or 0
                modified_at = ""
                if attr.st_mtime is not None:
                    modified_at = datetime.fromtimestamp(attr.st_mtime, tz=timezone.utc).isoformat()

                vp = path.join(attr.filename)

                infos.append(FileInfo(
                    path=vp,
                    is_dir=is_dir,
                    size=size,
                    modified_at=modified_at,
                ))

            return LsResult(entries=infos)

    @staticmethod
    def _attr_is_dir(attr: paramiko.SFTPAttributes) -> bool:
        """Check whether an SFTP attribute represents a directory."""
        return SshBackend._attr_is_dir_maybe(attr.st_mode)

    # ------------------------------------------------------------------
    # Core: read
    # ------------------------------------------------------------------

    def read_raw(
        self,
        file_path: VirtualPath,
        offset: int = 0,
        limit: int | None = 2000,
        include_line_numbers: bool = False,
    ) -> ReadResult:
        with self._sync_lock:
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
                remote = self._resolve(file_path)
            except BackendError as e:
                return ReadResult(error=e)
            try:
                is_dir = self._sftp.stat(remote).st_mode
                if self._attr_is_dir_maybe(is_dir):
                    return ReadResult(error=BackendError(code=ErrorCode.IS_DIR, path=file_path, message="目标是目录"))
            except FileNotFoundError:
                return ReadResult(error=BackendError(code=ErrorCode.NOT_FOUND, path=file_path, message="文件不存在"))
            except OSError as e:
                return ReadResult(error=BackendError(code=ErrorCode.OS_ERROR, path=file_path, message=str(e)))

            file_type = _get_file_type(file_path.value)
            mime_type = _get_mime_type(file_path.value)

            if file_type != "text":
                try:
                    with self._sftp.open(remote, "rb") as f:
                        raw = f.read()
                except OSError as e:
                    return ReadResult(error=BackendError(code=ErrorCode.IO_ERROR, path=file_path, message=str(e)))
                encoded = base64.b64encode(raw).decode("ascii")
                return ReadResult(
                    content=encoded,
                    total_lines=1,
                    encoding="base64",
                    file_type=file_type,
                    mime_type=mime_type,
                )

            # Text file: attempt UTF-8 read.
            # NOTE: Paramiko's SFTPClient.open(…, "r") may return bytes on
            # some server implementations.  Always open in binary mode and
            # decode explicitly to avoid TypeError downstream.
            try:
                with self._sftp.open(remote, "rb") as f:
                    content = f.read().decode("utf-8")
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
                    error=BackendError(code=ErrorCode.INVALID, path=file_path, message=f"偏移量 {offset} 超过文件长度({total} 行)"),
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

    @staticmethod
    def _attr_is_dir_maybe(st_mode: int | None) -> bool:
        """Check if *st_mode* indicates a directory."""
        return st_mode is not None and (st_mode & 0o40000) != 0

    # ------------------------------------------------------------------
    # Core: write
    # ------------------------------------------------------------------

    def write(
        self, file_path: VirtualPath, content: str, overwrite: bool = False,
    ) -> WriteResult:
        with self._sync_lock:
            if not self._check_edit_allowed(file_path):
                return WriteResult(
                    error=BackendError(code=ErrorCode.EDIT_NOT_ALLOWED, path=file_path, message="路径不允许写入"),
                )
            try:
                remote = self._resolve(file_path)
            except BackendError as e:
                return WriteResult(error=e)

            # Check if it's a directory
            try:
                st_mode = self._sftp.stat(remote).st_mode
                if self._attr_is_dir_maybe(st_mode):
                    return WriteResult(
                        error=BackendError(code=ErrorCode.IS_DIR, path=file_path, message="目标是目录，无法写入"),
                    )
            except FileNotFoundError:
                pass  # Doesn't exist yet – fine for write

            # Check existence
            try:
                self._sftp.stat(remote)
                exists = True
            except FileNotFoundError:
                exists = False

            if exists and not overwrite:
                return WriteResult(
                    error=BackendError(code=ErrorCode.ALREADY_EXISTS, path=file_path, message="文件已存在，请用 edit() 修改或用 overwrite=True 覆盖"),
                )

            # Ensure parent directories exist
            self._ensure_remote_dir(str(PurePosixPath(remote).parent))

            try:
                with self._sftp.open(remote, "w") as f:
                    f.write(content)
            except OSError as e:
                return WriteResult(error=BackendError(code=ErrorCode.IO_ERROR, path=file_path, message=str(e)))

            return WriteResult(path=file_path)

    def _ensure_remote_dir(self, remote_path: str) -> None:
        """Create remote directory tree, like ``mkdir -p``."""
        if remote_path in ("/", "", "."):
            return
        try:
            self._sftp.stat(remote_path)
        except FileNotFoundError:
            parent = str(PurePosixPath(remote_path).parent)
            self._ensure_remote_dir(parent)
            self._sftp.mkdir(remote_path)

    # ------------------------------------------------------------------
    # Core: edit
    # ------------------------------------------------------------------

    def edit(
        self,
        file_path: VirtualPath,
        old_str: str,
        new_str: str,
        *,
        replace_all: bool = False,
    ) -> EditResult:
        """Replace *old_str* with *new_str* in a remote file.

        If ``python3`` is available on the remote server the edit is
        performed remotely via a ``python3 -c`` one-liner (no SFTP
        round-trips).  Otherwise falls back to SFTP: download → local
        replace → upload.
        """
        with self._sync_lock:
            if not old_str:
                return EditResult(error=BackendError(code=ErrorCode.INVALID, message="old_str 不能为空"))
            if not self._check_edit_allowed(file_path):
                return EditResult(
                    error=BackendError(code=ErrorCode.EDIT_NOT_ALLOWED, path=file_path, message="路径不允许编辑"),
                )
            try:
                remote = self._resolve(file_path)
            except BackendError as e:
                return EditResult(error=e)

            # Check if path exists and is not a directory
            try:
                st_mode = self._sftp.stat(remote).st_mode
                if self._attr_is_dir_maybe(st_mode):
                    return EditResult(
                        error=BackendError(code=ErrorCode.IS_DIR, path=file_path, message="目标是目录，无法编辑"),
                    )
            except FileNotFoundError:
                return EditResult(
                    error=BackendError(
                        code=ErrorCode.NOT_FOUND,
                        path=file_path,
                        message=f"Cannot edit '{file_path}': file not found. To create a new file, use write().",
                    ),
                )

            old_str = normalize_line_endings(old_str)
            new_str = normalize_line_endings(new_str)

            if self._has_python3:
                return self._edit_remote(file_path, remote, old_str, new_str, replace_all)
            else:
                return self._edit_via_sftp(file_path, remote, old_str, new_str, replace_all)

    def _edit_remote(
        self,
        file_path: VirtualPath,
        remote: str,
        old_str: str,
        new_str: str,
        replace_all: bool,
    ) -> EditResult:
        """Execute the edit on the remote server via ``python3`` heredoc.

        Uses a POSIX heredoc with a quoted delimiter (``<< 'PYEOF'``)
        so the remote shell treats the entire script body verbatim — no
        variable expansion, no backslash interpretation, no cross‑platform
        quoting issues with ``shlex.quote``.
        """
        old_b64 = base64.b64encode(old_str.encode()).decode()
        new_b64 = base64.b64encode(new_str.encode()).decode()
        replace_count = "-1" if replace_all else "1"

        remote_repr = repr(remote)

        script = (
            f"import base64\n"
            f"old = base64.b64decode({old_b64!r}).decode()\n"
            f"new = base64.b64decode({new_b64!r}).decode()\n"
            f"c = open({remote_repr}, 'r', encoding='utf-8').read()\n"
            f"n = c.count(old)\n"
            f"print(n)\n"
            f"if not (n > 1 and not {replace_all}):\n"
            f"    open({remote_repr}, 'w', encoding='utf-8').write(c.replace(old, new, {replace_count}))\n"
        )

        # Quoted delimiter ('PYEOF') → no shell expansion inside the body
        cmd = f"python3 << 'PYEOF'\n{script}PYEOF"

        out, err, exit_code = self._exec(cmd)

        if exit_code != 0:
            return EditResult(error=BackendError(code=ErrorCode.IO_ERROR, path=file_path, message=f"编辑失败(exit {exit_code}): {err or out}"))

        try:
            occurrences = int(out.strip())
        except ValueError:
            return EditResult(error=BackendError(code=ErrorCode.INVALID, path=file_path, message=f"编辑返回异常: {out or err}"))

        if occurrences == 0:
            return EditResult(error=BackendError(code=ErrorCode.OLD_STR_NOT_FOUND, path=file_path, message="未找到要替换的文本"))

        if occurrences > 1 and not replace_all:
            return EditResult(error=BackendError(code=ErrorCode.MULTI_OCCURRENCES, path=file_path, message=f"匹配到 {occurrences} 处"))

        return EditResult(path=file_path, occurrences=occurrences)

    def _edit_via_sftp(
        self,
        file_path: VirtualPath,
        remote: str,
        old_str: str,
        new_str: str,
        replace_all: bool,
    ) -> EditResult:
        """Fallback edit: download file via SFTP, replace locally, upload."""
        try:
            with self._sftp.open(remote, "rb") as f:
                content = f.read().decode("utf-8")
        except UnicodeDecodeError:
            return EditResult(
                error=BackendError(code=ErrorCode.INVALID, path=file_path, message="文件不是有效 UTF-8"),
            )
        except OSError as e:
            return EditResult(error=BackendError(code=ErrorCode.IO_ERROR, path=file_path, message=str(e)))

        content = normalize_line_endings(content)
        occurrences = content.count(old_str)

        if occurrences == 0:
            return EditResult(
                error=BackendError(code=ErrorCode.OLD_STR_NOT_FOUND, path=file_path, message="未找到要替换的文本"),
            )

        if occurrences > 1 and not replace_all:
            return EditResult(
                error=BackendError(code=ErrorCode.MULTI_OCCURRENCES, path=file_path, message=f"匹配到 {occurrences} 处"),
            )

        new_content = content.replace(old_str, new_str, -1 if replace_all else 1)

        try:
            with self._sftp.open(remote, "w") as f:
                f.write(new_content)
        except OSError as e:
            return EditResult(error=BackendError(code=ErrorCode.IO_ERROR, path=file_path, message=str(e)))

        return EditResult(path=file_path, occurrences=occurrences)

    # ------------------------------------------------------------------
    # Core: grep
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
        """Search for a text pattern in files under *path*.

        Execution order:
        1. ``rg --json`` (ripgrep, fastest)
        2. ``grep -rnIsH`` (fast C fallback)
        3. ``python3`` os.walk (portable last resort, used when
           GNU grep fails e.g. BSD/macOS with ``--include`` glob)
        """
        with self._sync_lock:
            # Delegate to the internal collector which returns raw full matches,
            # then apply limit at this level.
            raw = self._grep_raw(pattern, path, glob, regex)
            if raw.error and not raw.matches:
                return raw  # fatal error, no matches
            return self._apply_grep_limit(raw.matches or [], offset, limit, pattern=pattern, regex=regex)

    def _grep_raw(
        self,
        pattern: str,
        path: VirtualPath = VirtualPath("/workspace"),
        glob: str | None = None,
        regex: bool = True,
    ) -> GrepResult:
        """Collect raw grep matches without offset/limit truncation."""
        if not pattern:
            return GrepResult(error=BackendError(code=ErrorCode.INVALID, message="搜索模式不能为空"))
        try:
            remote = self._resolve(path)
        except BackendError as e:
            return GrepResult(error=e)
        try:
            st_mode = self._sftp.stat(remote).st_mode
        except FileNotFoundError:
            return GrepResult(error=BackendError(code=ErrorCode.NOT_FOUND, path=path, message="路径不存在"))
        except OSError as e:
            return GrepResult(error=BackendError(code=ErrorCode.OS_ERROR, path=path, message=str(e)))
        is_dir = self._attr_is_dir_maybe(st_mode)
        remote_escaped = shlex.quote(remote)
        pattern_escaped = shlex.quote(pattern)

        # 1) Try ripgrep on the remote (fastest; --glob prunes files early
        # for performance, but uses gitignore semantics so we still
        # post-filter with POSIX fnmatch_path below).
        matches = self._rg_remote(pattern_escaped, remote, glob, regex)
        if matches is not None:
            if glob and is_dir and matches:
                matches = self._apply_glob_filter(matches, path.value, glob)
            return GrepResult(matches=matches)

        # 2) GNU grep (fast C program, handles most cases) — post-filter with POSIX glob
        result = self._grep_remote(pattern_escaped, remote_escaped, regex)
        if result.error is None:
            if glob and is_dir and result.matches:
                result = GrepResult(matches=self._apply_glob_filter(result.matches, path.value, glob))
            return result

        # 3) python3 last resort — post-filter with POSIX glob
        if self._has_python3:
            result = self._grep_python(pattern, remote, regex)
            if glob and is_dir and result.matches:
                result = GrepResult(matches=self._apply_glob_filter(result.matches, path.value, glob))
            return result

        return result

    def _rg_remote(
        self,
        pattern_escaped: str,
        remote: str,
        glob: str | None,
        regex: bool = True,
    ) -> list[GrepMatch] | None:
        """Run ripgrep on the remote server.  Returns ``None`` if unavailable."""
        cmd = "rg --json"
        if not regex:
            cmd += " -F"
        if glob:
            cmd += f" --glob {shlex.quote(glob)}"
        cmd += f" -- {pattern_escaped} {shlex.quote(remote)}"

        out, _err, exit_code = self._exec(cmd, timeout=60)

        # rg exit codes: 0 = matches found, 1 = no matches, anything else = error/unavailable
        if exit_code == 1:
            return []  # rg found no matches
        if exit_code != 0:
            return None  # rg not available or error → fall through to grep
        if not out.strip():
            return []

        matches: list[GrepMatch] = []
        for line in out.splitlines():
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

            # Convert physical remote path → virtual path
            virt = self._physical_to_virtual(ftext)
            matches.append(GrepMatch(path=virt, line=int(ln), text=lt))

        return matches

    def _grep_python(
        self,
        pattern: str,
        remote: str,
        regex: bool = True,
    ) -> GrepResult:
        """Portable grep via remote ``python3``.

        For a single file, searches directly.  For a directory,
        uses :func:`os.walk`.  Respects :attr:`_ignore_dirs` —
        directories listed there are skipped entirely.  Files > 1 MB
        and known binary extensions are also skipped to keep traversal
        fast.
        """
        pattern_b64 = base64.b64encode(pattern.encode()).decode()
        regex_b64 = base64.b64encode(str(regex).encode()).decode()
        remote_repr = repr(remote)

        # Convert virtual ignore_dirs → relative names for remote os.walk
        wr = self.workspace_root.value
        ignore_names_b64 = base64.b64encode(
            json.dumps([
                p[len(wr):].lstrip("/").rsplit("/", 1)[-1]
                for p in self._ignore_dirs
            ]).encode()
        ).decode()

        script = (
            f"import base64, os, json, re\n"
            f"SKIP_DIRS = set(json.loads("
            f"base64.b64decode({ignore_names_b64!r}).decode()))\n"
            f"BINARY_EXTS = {{'.pyc', '.pyo', '.so', '.o', '.a', '.bin',\n"
            f"                '.exe', '.dll', '.pyd', '.zip', '.tar', '.gz',\n"
            f"                '.bz2', '.xz', '.7z', '.png', '.jpg', '.jpeg',\n"
            f"                '.gif', '.ico', '.pdf', '.mp3', '.mp4', '.avi'}}\n"
            f"MAX_SIZE = 1_048_576\n"  # 1 MB
            f"pat = base64.b64decode({pattern_b64!r}).decode()\n"
            f"use_regex = base64.b64decode({regex_b64!r}).decode() == 'True'\n"
            f"if use_regex:\n"
            f"    _regex = re.compile(pat)\n"
            f"else:\n"
            f"    _regex = re.compile(re.escape(pat))\n"
            f"def _search_file(fp):\n"
            f"    r = []\n"
            f"    try:\n"
            f"        if os.path.getsize(fp) > MAX_SIZE:\n"
            f"            return r\n"
            f"        with open(fp, 'r', encoding='utf-8', errors='ignore') as f:\n"
            f"            for li, line in enumerate(f, 1):\n"
            f"                if _regex.search(line):\n"
            f"                    r.append({{'p': fp, 'l': li,"
            f" 't': line.rstrip(chr(10))}})\n"
            f"    except Exception:\n"
            f"        pass\n"
            f"    return r\n"
            f"d = {remote_repr}\n"
            f"if os.path.isfile(d):\n"
            f"    res = _search_file(d)\n"
            f"else:\n"
            f"    res = []\n"
            f"    for root, dirs, files in os.walk(d):\n"
            f"        dirs[:] = [x for x in dirs\n"
            f"                   if not x.startswith('.') and x not in SKIP_DIRS]\n"
            f"        for fname in files:\n"
            f"            if fname.startswith('.'):\n"
            f"                continue\n"
            f"            if os.path.splitext(fname)[1].lower() in BINARY_EXTS:\n"
            f"                continue\n"
            f"            fp = os.path.join(root, fname)\n"
            f"            res.extend(_search_file(fp))\n"
            f"print(json.dumps(res))\n"
        )

        cmd = f"python3 << 'PYEOF'\n{script}PYEOF"
        out, err, exit_code = self._exec(cmd, timeout=60)

        if exit_code != 0:
            return GrepResult(error=BackendError(code=ErrorCode.IO_ERROR, message=f"grep 错误(exit {exit_code}): {err or out}"))

        try:
            data = json.loads(out.strip())
        except json.JSONDecodeError:
            return GrepResult(error=BackendError(code=ErrorCode.INVALID, message=f"grep 返回异常: {out[:200]}"))

        matches: list[GrepMatch] = []
        for item in data:
            virt = self._physical_to_virtual(item["p"])
            matches.append(GrepMatch(
                path=virt,
                line=item["l"],
                text=item["t"],
            ))

        return GrepResult(matches=matches)

    def _grep_remote(
        self,
        pattern_escaped: str,
        remote_escaped: str,
        regex: bool = True,
    ) -> GrepResult:
        """Fallback grep using ``grep -rnIsH`` on the remote.

        ``-r`` recursive, ``-n`` line numbers, ``-I`` skip binary files,
        ``-s`` suppress error messages, ``-H`` always print filename
        (needed for single-file searches where grep would otherwise omit it).
        """
        if regex:
            base_flags = "grep -rnIsHE"
        else:
            base_flags = "grep -rnIsHF"
        cmd = f"{base_flags} -- {pattern_escaped} {remote_escaped}"

        out, err, exit_code = self._exec(cmd, timeout=60)

        # grep exit codes: 0 = matches, 1 = no matches, >1 = errors in some files
        if not out.strip():
            if exit_code > 1 and err.strip():
                return GrepResult(error=BackendError(code=ErrorCode.IO_ERROR, message=f"grep 错误: {err.strip()}"))
            return GrepResult(matches=[])

        matches: list[GrepMatch] = []
        for line in out.splitlines():
            # Output format: file:line_number:text
            if ":" not in line:
                continue
            first_colon = line.index(":")
            ftext = line[:first_colon]
            rest = line[first_colon + 1:]
            if ":" not in rest:
                continue
            second_colon = rest.index(":")
            try:
                ln = int(rest[:second_colon])
            except ValueError:
                continue
            text = rest[second_colon + 1:]
            virt = self._physical_to_virtual(ftext)
            matches.append(GrepMatch(path=virt, line=ln, text=text))

        return GrepResult(matches=matches)

    def _physical_to_virtual(self, physical_path: str) -> VirtualPath:
        """Convert a remote absolute path to the virtual path scheme.

        Example: ``/home/user/project/src/main.py`` → ``/workspace/src/main.py``
        (when *remote_root* is ``/home/user/project`` and workspace_root is ``/workspace``).
        """
        # Normalize separators
        physical_path = physical_path.replace("\\", "/")
        wr = self.workspace_root.value
        if physical_path.startswith(self._remote_root):
            suffix = physical_path[len(self._remote_root):]
            rel = suffix if suffix.startswith("/") else ("/" + suffix if suffix else "")
            return VirtualPath(f"{wr}{rel}")
        # If the path doesn't start with remote_root, return as-is
        return VirtualPath(physical_path)

    def _check_edit_allowed(self, path: VirtualPath) -> bool:
        """Check whether *path* is allowed for edit/write/delete."""
        return check_path_allowed(
            path.value,
            whitelist=self._edit_whitelist or None,
            blacklist=self._edit_blacklist or None,
        )

    @staticmethod
    def _apply_glob_filter(
        matches: list[GrepMatch],
        search_path: str,
        glob_pattern: str,
    ) -> list[GrepMatch]:
        """Filter grep matches by POSIX glob pattern on the relative path.

        Each match's path is a **virtual** path (already converted from
        physical remote path).  We strip *search_path* to get a relative
        path and then apply POSIX glob matching (``*`` does not cross
        ``/``, ``**`` matches any depth).
        """
        from mambo_agents.backends.utils import fnmatch_path

        prefix = search_path.rstrip("/") + "/"
        filtered: list[GrepMatch] = []
        for m in matches:
            path_str = str(m.path)
            if path_str.startswith(prefix):
                rel = path_str[len(prefix):]
            elif path_str == search_path.rstrip("/"):
                rel = ""
            else:
                continue
            if fnmatch_path(rel, glob_pattern):
                filtered.append(m)
        return filtered

    # ------------------------------------------------------------------
    # Core: glob
    # ------------------------------------------------------------------

    def glob(self, pattern: str, path: VirtualPath = VirtualPath("/workspace")) -> GlobResult:
        """Find files and directories matching *pattern* under *path*.

        Uses remote ``python3`` with ``glob.glob(recursive=True)``
        for full wildcard support (``*``, ``**``, ``?``, ``[...]``).
        Falls back to ``find`` when python3 is unavailable.
        """
        with self._sync_lock:
            if not pattern:
                return GlobResult(error=BackendError(code=ErrorCode.INVALID, message="搜索模式不能为空"))
            try:
                remote = self._resolve(path)
            except BackendError as e:
                return GlobResult(error=e)

            # Check that path is a directory
            try:
                st_mode = self._sftp.stat(remote).st_mode
                if not self._attr_is_dir_maybe(st_mode):
                    return GlobResult(error=BackendError(code=ErrorCode.NOT_DIR, path=path, message="目标是文件，不是目录"))
            except FileNotFoundError:
                return GlobResult(error=BackendError(code=ErrorCode.NOT_FOUND, path=path, message="路径不存在"))

            if self._has_python3:
                return self._glob_python(pattern, remote)
            else:
                return self._glob_find(pattern, remote)

    # ------------------------------------------------------------------
    # Glob via remote Python (full ** / path-prefix support)
    # ------------------------------------------------------------------

    def _glob_python(self, pattern: str, remote: str) -> GlobResult:
        """Glob via remote ``python3`` using :func:`glob.glob`.

        Transmits *pattern* and *remote* safely via base64 encoding
        inside a POSIX heredoc with quoted delimiter (``<< 'PYEOF'``),
        avoiding any shell-level interpretation of special characters.

        Validates that the resolved search path stays within the base
        directory, preventing path-traversal attacks via ``..`` in the
        pattern string.
        """
        pattern_b64 = base64.b64encode(pattern.encode()).decode()
        remote_repr = repr(remote)

        script = (
            f"import base64, glob, os, json\n"
            f"p = base64.b64decode({pattern_b64!r}).decode()\n"
            f"d = {remote_repr}\n"
            f"prefix = d.rstrip('/') + '/'\n"
            f"full = os.path.normpath(os.path.join(d, p))\n"
            f"if not (full == d.rstrip('/') or full.startswith(prefix)):\n"
            f"    print(json.dumps({{'error': "
            f"'Pattern resolves outside base directory'}}))\n"
            f"else:\n"
            f"    res = []\n"
            f"    for f in glob.glob(full, recursive=True):\n"
            f"        is_dir = os.path.isdir(f)\n"
            f"        res.append({{'p': f, 'd': is_dir, 's': os.path.getsize(f) if not is_dir else 0}})\n"
            f"    print(json.dumps(res))\n"
        )

        cmd = f"python3 << 'PYEOF'\n{script}PYEOF"
        out, err, exit_code = self._exec(cmd, timeout=60)

        if exit_code != 0:
            return GlobResult(error=BackendError(code=ErrorCode.IO_ERROR, message=f"glob 错误(exit {exit_code}): {err or out}"))

        try:
            data = json.loads(out.strip())
        except json.JSONDecodeError:
            return GlobResult(error=BackendError(code=ErrorCode.INVALID, message=f"glob 返回异常: {out[:200]}"))

        if isinstance(data, dict) and "error" in data:
            return GlobResult(error=BackendError(code=ErrorCode.IO_ERROR, message=str(data["error"])))

        matches: list[FileInfo] = []
        for item in data:
            virt = self._physical_to_virtual(item["p"])
            matches.append(FileInfo(
                path=virt,
                is_dir=item.get("d", False),
                size=item.get("s", 0),
            ))

        return GlobResult(matches=matches)

    def _glob_find(self, pattern: str, remote: str) -> GlobResult:
        """Fallback glob: split pattern into directory prefix + filename glob.

        Parses the *pattern* to extract a static directory prefix
        (everything before the first glob metacharacter's parent ``/``)
        and uses it as the ``find`` search root, with the remainder
        passed to ``-name``.  ``**/`` segments are treated as a
        recursive marker (default ``find`` behaviour).

        Limitations (vs. the python3 path):
        - ``**`` in the **middle** of a path (e.g. ``a/**/b/*.txt``)
          cannot constrain intermediate directories — ``find`` has no
          equivalent of ``**``.
        - Patterns with no ``**`` use ``-maxdepth 1`` for single-level
          matching, but ``find`` has no way to enforce exact depth
          patterns like ``a/*/b/*.txt``.
        """
        # 1) Strip all **/ segments — find is already recursive,
        #    and **/ carries no information that find can use.
        clean = pattern
        has_recursive = "**/" in clean
        if has_recursive:
            clean = clean.replace("**/", "")

        # 2) Find the first glob metacharacter
        m = re.search(r"[*?\[\]]", clean)
        if m is None:
            # No glob chars — treat as a literal filename
            search_dir = remote
            name_pat = clean
            maxdepth = "1"
        else:
            idx = m.start()
            prefix = clean[:idx]

            # Find the last '/' before the first glob char
            last_slash = prefix.rfind("/")
            if last_slash >= 0:
                dir_part = prefix[:last_slash]
                file_part = prefix[last_slash + 1:] + clean[idx:]
                search_dir = f"{remote}/{dir_part}" if dir_part else remote
                name_pat = file_part
            else:
                search_dir = remote
                name_pat = clean

            # Non-recursive unless pattern contains **/
            maxdepth = "" if (has_recursive or "**/" in pattern) else "1"

        # 3) Build and run the find command
        search_esc = shlex.quote(search_dir)
        name_esc = shlex.quote(name_pat)
        maxdepth_flag = f"-maxdepth {maxdepth} " if maxdepth else ""

        cmd = (
            f"find {search_esc} {maxdepth_flag}"
            f"-name {name_esc} "
            f"-printf '%Y\\t%s\\t%p\\n' 2>/dev/null"
        )

        out, err, exit_code = self._exec(cmd, timeout=60)
        if exit_code > 1:
            return GlobResult(error=BackendError(code=ErrorCode.IO_ERROR, message=f"glob 错误: {err}"))

        matches: list[FileInfo] = []
        for line in out.splitlines():
            if not line.strip():
                continue
            parts = line.split("\t", 2)
            if len(parts) < 3:
                continue
            type_char, size_str, physical_path = parts
            try:
                size = int(size_str)
            except ValueError:
                size = 0
            is_dir = (type_char == "d")
            virt = self._physical_to_virtual(physical_path.strip())
            matches.append(FileInfo(
                path=virt,
                is_dir=is_dir,
                size=0 if is_dir else size,
            ))

        return GlobResult(matches=matches)

    # ------------------------------------------------------------------
    # Extra: tree
    # ------------------------------------------------------------------

    def tree(self, path: VirtualPath = VirtualPath("/workspace"), depth: int = 3) -> str:
        """Render a directory tree.

        Uses remote ``python3`` with :func:`os.walk` for portable
        directory traversal (works on any Unix).  Falls back to
        ``find -printf`` (GNU only) when python3 is unavailable.

        Args:
            path: Root directory to display (default workspace root).
            depth: Maximum recursion depth (default 3, must be >= 1).

        Returns:
            Formatted tree string.
        """
        with self._sync_lock:
            if depth < 1:
                return f"Invalid depth value: {depth}. Depth must be a positive integer (>= 1)."

            try:
                remote = self._resolve(path)
            except BackendError as e:
                return str(e)

            # Check if path exists and is a directory (T1 & T2 fix)
            try:
                st_mode = self._sftp.stat(remote).st_mode
                if not self._attr_is_dir_maybe(st_mode):
                    return f"'{path}' is a file, not a directory"
            except FileNotFoundError:
                return f"Path '{path}' not found"
            except OSError as e:
                return f"Cannot access '{path}': {e}"

            if self._has_python3:
                out, err, exit_code = self._tree_python(remote, depth)
            else:
                out, err, exit_code = self._tree_find(remote, depth)

            # Map virtual ignore_dirs → relative paths expected in output
            wr = self.workspace_root.value
            ignore_rel_paths: frozenset[str] = frozenset(
                p[len(wr):].lstrip("/") for p in self._ignore_dirs
            )

            if exit_code > 1 or not out.strip():
                return f"(empty or inaccessible)"

            # Parse find output into tree entries
            dirs: set[str] = set()
            file_entries: list[tuple[str, str, int]] = []  # (parent, name, size)

            for line in out.splitlines():
                if not line.strip():
                    continue
                parts = line.split(" ", 2)
                if len(parts) < 2:
                    continue
                entry_type = parts[0]
                if entry_type == "d":
                    dirs.add(parts[1])
                elif entry_type == "f" and len(parts) >= 3:
                    rel_path = parts[1]
                    try:
                        size = int(parts[2])
                    except ValueError:
                        size = 0
                    parent = str(PurePosixPath(rel_path).parent) if "/" in rel_path else "."
                    name = PurePosixPath(rel_path).name
                    file_entries.append((parent, name, size))

            # Build depth from relative paths
            def _path_depth(rel: str) -> int:
                if rel == "." or not rel:
                    return 1
                return rel.count("/") + 1

            # Filter out children of ignored dirs AND the ignored dirs themselves
            # (we'll add ignored dirs back later with the ignore marker)
            filtered_dirs: set[str] = set()
            for d in dirs:
                if any(
                    d == ign or d.startswith(ign + "/")
                    for ign in ignore_rel_paths
                ):
                    continue
                filtered_dirs.add(d)

            filtered_files: list[tuple[str, str, int]] = []
            for parent, name, size in file_entries:
                # Build full relative path for checking
                full_rel = f"{parent}/{name}" if parent != "." else name
                if any(
                    full_rel == ign or full_rel.startswith(ign + "/")
                    for ign in ignore_rel_paths
                ):
                    continue
                filtered_files.append((parent, name, size))

            # Build (sort_path, TreeEntry) tuples — sort by full path so that
            # files appear immediately after their parent directory instead of
            # all being grouped at the bottom.
            dir_file_entries: list[tuple[str, TreeEntry]] = []

            # Directories
            for d in filtered_dirs:
                d_depth = _path_depth(d)
                if d_depth > depth:
                    continue

                # Determine marker
                marker: Literal["", "empty", "ignore", "depth_exceeded"] = ""

                if d_depth == depth:
                    # At depth limit — check via SFTP if this dir has children
                    dir_remote_path = f"{remote}/{d}"
                    try:
                        dir_attrs = self._sftp.listdir_attr(dir_remote_path)
                        has_children = len(dir_attrs) > 0
                    except (FileNotFoundError, OSError):
                        has_children = False
                    marker = "depth_exceeded" if has_children else "empty"
                else:
                    # Check emptiness from parsed data
                    has_children = any(
                        fp == d for fp, _n, _s in filtered_files
                    ) or any(
                        sub != d and sub.startswith(d + "/")
                        for sub in filtered_dirs
                    )
                    if not has_children:
                        marker = "empty"

                # Sort path: directory paths get trailing "/" so they sort
                # before files in the same directory (e.g. "0849fe09/" < "0849fe09/file.png")
                sort_path = d + "/"
                dir_file_entries.append((sort_path, TreeEntry(
                    name=PurePosixPath(d).name + "/",
                    depth=d_depth,
                    marker=marker,
                )))

            # Add ignored dirs (show but with ignore marker, no children)
            for ign in sorted(ignore_rel_paths):
                if ign in dirs:  # only if it actually exists
                    sort_path = ign + "/"
                    dir_file_entries.append((sort_path, TreeEntry(
                        name=ign.split("/")[-1] + "/",
                        depth=ign.count("/") + 1,
                        marker="ignore",
                    )))

            # Files
            for parent, name, size in filtered_files:
                d = (parent + "/" + name).count("/") + 1 if parent != "." else 1
                if d > depth:
                    continue
                sort_path = (parent + "/" + name) if parent != "." else name
                dir_file_entries.append((sort_path, TreeEntry(
                    name=f"{name} ({human_size(size)})",
                    depth=d,
                )))

            # Sort all entries by their full relative path so children
            # appear right after their parent directory
            dir_file_entries.sort(key=lambda x: x[0])

            # Build final entries list
            entries: list[TreeEntry] = []
            root_name = path.name
            entries.append(TreeEntry(name=root_name + "/", depth=0))
            entries.extend(e[1] for e in dir_file_entries)

            return format_tree_entries(entries)

    # ------------------------------------------------------------------
    # Tree via remote Python (portable, no GNU find dependencies)
    # ------------------------------------------------------------------

    def _tree_python(
        self, remote: str, depth: int
    ) -> tuple[str, str, int]:
        """List directory tree via remote ``python3`` with :func:`os.walk`.

        Output format is identical to the ``find`` fallback:
        ``d <relative_path>`` for directories,
        ``f <relative_path> <size>`` for files.
        """
        remote_repr = repr(remote)

        script = (
            f"import os\n"
            f"d = {remote_repr}\n"
            f"md = {depth}\n"
            f"res = []\n"
            f"for root, dirs, files in os.walk(d):\n"
            f"    rd = root[len(d):].lstrip('/')\n"
            f"    cur_d = (rd.count('/') + 1) if rd else 1\n"
            f"    if cur_d > md:\n"
            f"        dirs[:] = []\n"
            f"        continue\n"
            f"    dirs[:] = [x for x in dirs if not x.startswith('.')]\n"
            f"    for name in dirs:\n"
            f"        p = rd + '/' + name if rd else name\n"
            f"        res.append(('d', p, 0))\n"
            f"    for name in files:\n"
            f"        if name.startswith('.'):\n"
            f"            continue\n"
            f"        p = rd + '/' + name if rd else name\n"
            f"        fp = os.path.join(root, name)\n"
            f"        try:\n"
            f"            sz = os.path.getsize(fp)\n"
            f"        except OSError:\n"
            f"            sz = 0\n"
            f"        res.append(('f', p, sz))\n"
            f"res.sort()\n"
            f"for t, p, sz in res:\n"
            f"    print(f'{{t}} {{p}} {{sz}}' if t == 'f' else f'{{t}} {{p}}')\n"
        )

        cmd = f"python3 << 'PYEOF'\n{script}PYEOF"
        return self._exec(cmd, timeout=30)

    def _tree_find(
        self, remote: str, depth: int
    ) -> tuple[str, str, int]:
        """Fallback tree via ``find -printf`` (GNU find only)."""
        remote_escaped = shlex.quote(remote)
        cmd = (
            f"find {remote_escaped} -maxdepth {depth} -not -path '*/.*' "
            f"\\( -type d -printf 'd %P\\n' , -type f -printf 'f %P %s\\n' \\) "
            f"2>/dev/null | sort"
        )
        return self._exec(cmd, timeout=30)

    # ------------------------------------------------------------------
    # Extra: delete
    # ------------------------------------------------------------------

    def delete(self, path: VirtualPath) -> DeleteResult:
        """Delete a single **file** on the remote server.

        Directories are rejected — the agent must remove files inside
        the directory individually before the directory can be deleted.
        """
        with self._sync_lock:
            if not self._check_edit_allowed(path):
                return DeleteResult(
                    error=BackendError(code=ErrorCode.EDIT_NOT_ALLOWED, path=path, message="路径不允许删除"),
                    path=path,
                )
            try:
                remote = self._resolve(path)
            except BackendError as e:
                return DeleteResult(error=e, path=path)

            # Safety: refuse to delete the remote root
            if remote.rstrip("/") == self._remote_root.rstrip("/"):
                return DeleteResult(error=BackendError(code=ErrorCode.INVALID, path=path, message="不能删除根工作目录"), path=path)

            # Reject directories
            try:
                st_mode = self._sftp.stat(remote).st_mode
                if self._attr_is_dir_maybe(st_mode):
                    return DeleteResult(
                        error=BackendError(code=ErrorCode.IS_DIR, path=path, message="目标是目录，delete 工具只能删除单个文件"),
                        path=path,
                    )
            except FileNotFoundError:
                return DeleteResult(error=BackendError(code=ErrorCode.NOT_FOUND, path=path, message="路径不存在"), path=path)

            remote_escaped = shlex.quote(remote)
            out, err, exit_code = self._exec(f"rm -f {remote_escaped}")

            if exit_code != 0:
                return DeleteResult(error=BackendError(code=ErrorCode.IO_ERROR, path=path, message=err or out), path=path)

            return DeleteResult(path=path)

    # ------------------------------------------------------------------
    # Extra: execute
    # ------------------------------------------------------------------

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> str:
        """Execute a shell command on the remote server.

        Args:
            command: Shell command string to execute.
            timeout: Override the default timeout (seconds).

        Returns:
            Formatted output with stdout, stderr, and exit code.
        """
        with self._sync_lock:
            if not command or not isinstance(command, str):
                return "Error: Command must be a non-empty string."

            effective_timeout = timeout if timeout is not None else self._execute_timeout
            if effective_timeout <= 0:
                msg = f"timeout must be positive, got {effective_timeout}"
                raise ValueError(msg)

            # Prepend a cd to the remote root so the command runs in the
            # expected working directory
            full_cmd = f"cd {shlex.quote(self._remote_root)} && {command}"

            try:
                out, err, exit_code = self._exec(full_cmd, timeout=effective_timeout)
            except Exception as e:
                return f"Error executing command ({type(e).__name__}): {e}"

            output_parts: list[str] = []
            if out:
                output_parts.append(out.rstrip())

            # Timeout message: display prominently, not wrapped in [stderr]
            is_timeout = exit_code == -1 and err.startswith("Timeout:")
            if err:
                for line in err.strip().split("\n"):
                    if is_timeout:
                        output_parts.append(line)
                    else:
                        output_parts.append(f"[stderr] {line}")

            output = "\n".join(output_parts) if output_parts else "<no output>"

            # Truncation
            if len(output) > self._max_output_bytes:
                output = output[: self._max_output_bytes]
                output += f"\n\n... Output truncated at {self._max_output_bytes} bytes."

            if exit_code != 0:
                output = f"{output.rstrip()}\n\nExit code: {exit_code}"

            return output

    # ------------------------------------------------------------------
    # Async variants — all serialized via _async_lock to prevent
    # paramiko Transport deadlocks when ToolNode runs tools in parallel.
    # ------------------------------------------------------------------

    async def als(self, path: VirtualPath) -> LsResult:
        async with self._async_lock:
            return await asyncio.to_thread(self.ls, path)

    async def aread(
        self,
        file_path: VirtualPath,
        offset: int = 0,
        limit: int = 2000,
        include_line_numbers: bool = False,
    ) -> ReadResult:
        async with self._async_lock:
            return await asyncio.to_thread(
                self.read, file_path, offset, limit, include_line_numbers,
            )

    async def awrite(
        self, file_path: VirtualPath, content: str, overwrite: bool = False,
    ) -> WriteResult:
        async with self._async_lock:
            return await asyncio.to_thread(self.write, file_path, content, overwrite)

    async def aedit(
        self,
        file_path: VirtualPath,
        old_str: str,
        new_str: str,
        *,
        replace_all: bool = False,
    ) -> EditResult:
        async with self._async_lock:
            return await asyncio.to_thread(
                self.edit, file_path, old_str, new_str, replace_all=replace_all,
            )

    async def agrep(
        self,
        pattern: str,
        path: VirtualPath = VirtualPath("/workspace"),
        glob: str | None = None,
        regex: bool = True,
        offset: int = 0,
        limit: int | None = None,
    ) -> GrepResult:
        async with self._async_lock:
            return await asyncio.to_thread(self.grep, pattern, path, glob, regex, offset, limit)

    async def aglob(self, pattern: str, path: VirtualPath = VirtualPath("/workspace")) -> GlobResult:
        async with self._async_lock:
            return await asyncio.to_thread(self.glob, pattern, path)

    async def atree(self, path: VirtualPath = VirtualPath("/workspace"), depth: int = 3) -> str:
        async with self._async_lock:
            return await asyncio.to_thread(self.tree, path, depth)

    async def adelete(self, path: VirtualPath) -> DeleteResult:
        async with self._async_lock:
            return await asyncio.to_thread(self.delete, path)

    async def aexecute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> str:
        async with self._async_lock:
            return await asyncio.to_thread(self.execute, command, timeout=timeout)

    async def aupload_files(
        self,
        files: list[tuple[VirtualPath, bytes]],
    ) -> list[UploadFileResult]:
        async with self._async_lock:
            return await asyncio.to_thread(self.upload_files, files)

    async def adownload_files(
        self,
        paths: list[VirtualPath],
    ) -> list[DownloadFileResult]:
        async with self._async_lock:
            return await asyncio.to_thread(self.download_files, paths)

    # ------------------------------------------------------------------
    # Developer API — bulk upload / download
    # ------------------------------------------------------------------

    def upload_files(
        self,
        files: list[tuple[VirtualPath, bytes]],
    ) -> list[UploadFileResult]:
        """Upload multiple files as raw bytes to the remote server.

        Binary files are written directly via SFTP (not base64-encoded
        as text), mirroring ``LocalBackend.upload_files``.
        """
        with self._sync_lock:
            results: list[UploadFileResult] = []
            for path, raw_content in files:
                if not self._check_edit_allowed(path):
                    results.append(UploadFileResult(
                        path=path,
                        error=BackendError(code=ErrorCode.EDIT_NOT_ALLOWED, path=path, message="路径不允许写入"),
                    ))
                    continue
                try:
                    remote = self._resolve(path)
                    self._ensure_remote_dir(str(PurePosixPath(remote).parent))
                    with self._sftp.open(remote, "wb") as f:
                        f.write(raw_content)
                    results.append(UploadFileResult(path=path))
                except OSError as e:
                    results.append(UploadFileResult(path=path, error=BackendError(code=ErrorCode.IO_ERROR, path=path, message=str(e))))
                except Exception as e:
                    results.append(
                        UploadFileResult(
                            path=path, error=BackendError(code=ErrorCode.INVALID, path=path, message=f"{type(e).__name__}: {e}")
                        )
                    )
            return results

