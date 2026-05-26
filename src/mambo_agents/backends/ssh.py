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
import shlex
from datetime import datetime, timezone
from pathlib import PurePosixPath
from types import TracebackType

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
    UploadFileResult,
    WriteResult,
    _get_file_type,
    _get_mime_type,
)
from mambo_agents.backends.utils import (
    format_tree_entries,
    format_with_line_numbers,
    human_size,
)

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
        connect_timeout: SSH connection timeout in seconds (default 30).
        execute_timeout: Default timeout for shell commands (default 120).
        max_output_bytes: Max bytes captured from command output.
    """

    def __init__(
        self,
        host: str,
        username: str,
        *,
        port: int = 22,
        password: str | None = None,
        key_filename: str | None = None,
        remote_root: str = "~",
        connect_timeout: int = _DEFAULT_SSH_CONNECT_TIMEOUT,
        execute_timeout: int = _DEFAULT_EXECUTE_TIMEOUT,
        max_output_bytes: int = _MAX_OUTPUT_BYTES,
    ) -> None:
        if password is None and key_filename is None:
            raise ValueError(
                "Either 'password' or 'key_filename' must be provided for SSH authentication."
            )

        self._host = host
        self._username = username
        self._port = port
        self._password = password
        self._key_filename = key_filename
        self._raw_remote_root = remote_root
        self._connect_timeout = connect_timeout
        self._execute_timeout = execute_timeout
        self._max_output_bytes = max_output_bytes

        self._client: paramiko.SSHClient | None = None
        self._sftp: paramiko.SFTPClient | None = None
        self._remote_root: str = ""  # resolved absolute path, set by _connect()

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
        # Normalise: strip trailing slash (except "/")
        self._remote_root = remote_root.rstrip("/") or "/"

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

    def _resolve(self, path: str) -> str:
        """Map a virtual absolute path to a remote filesystem path.

        Virtual paths use POSIX separators and ``"/"`` as the root of
        the backing directory (*remote_root*).  Traversal beyond the
        root via ``..`` is not explicitly prevented (same as
        ``LocalBackend``).
        """
        pp = PurePosixPath(path)
        if pp.is_absolute():
            relative_parts = pp.parts[1:]  # skip the root "/"
        else:
            relative_parts = pp.parts

        if not relative_parts:
            return self._remote_root

        joined = "/".join(relative_parts)
        return f"{self._remote_root}/{joined}"

    # ------------------------------------------------------------------
    # Remote command execution
    # ------------------------------------------------------------------

    def _exec(
        self, command: str, *, timeout: int | None = None
    ) -> tuple[str, str, int]:
        """Execute *command* on the remote server via SSH.

        Returns:
            ``(stdout, stderr, exit_code)`` tuple.
        """
        t = timeout if timeout is not None else self._execute_timeout
        _stdin, stdout, stderr = self._client.exec_command(command, timeout=t)
        exit_code = stdout.channel.recv_exit_status()
        out = stdout.read().decode(errors="replace")
        err = stderr.read().decode(errors="replace")
        return out, err, exit_code

    # ------------------------------------------------------------------
    # tools property
    # ------------------------------------------------------------------

    @property
    def tools(self) -> list[StructuredTool]:
        return [
            StructuredTool(
                name="tree",
                description=(
                    "View the directory tree structure. "
                    "Shows directories and files with their sizes in a tree format."
                ),
                args_schema=create_model(
                    "TreeSchema",
                    path=(str, Field(default="/", description="Root directory to display")),
                    depth=(int, Field(default=3, description="Maximum recursion depth")),
                ),
                func=lambda **kwargs: self.tree(**kwargs),
                coroutine=lambda **kwargs: self.atree(**kwargs),
            ),
            StructuredTool(
                name="delete",
                description=(
                    "Delete a file or directory. "
                    "For directories, removes the directory and all its contents recursively."
                ),
                args_schema=create_model(
                    "DeleteSchema",
                    path=(str, Field(description="Absolute path to delete")),
                ),
                func=lambda **kwargs: self.delete(**kwargs),
                coroutine=lambda **kwargs: self.adelete(**kwargs),
            ),
            StructuredTool(
                name="execute",
                description=(
                    "Execute a shell command on the remote server via SSH. "
                    "Returns combined stdout and stderr output."
                ),
                args_schema=create_model(
                    "ExecuteSchema",
                    command=(str, Field(description="Shell command to execute")),
                    timeout=(int | None, Field(default=None, description="Optional timeout in seconds")),
                ),
                func=lambda **kwargs: self.execute(**kwargs),
                coroutine=lambda **kwargs: self.aexecute(**kwargs),
            ),
        ]

    @property
    def description(self) -> str:
        return (
            f"Remote file system backend via SSH "
            f"(host: {self._host}, working directory: {self._remote_root})"
        )

    # ------------------------------------------------------------------
    # Core: ls
    # ------------------------------------------------------------------

    def ls(self, path: str) -> LsResult:
        """List files and directories under *path* (non-recursive).

        Uses SFTP ``listdir_attr`` for structured, locale-independent output.
        """
        remote = self._resolve(path)
        try:
            attrs = self._sftp.listdir_attr(remote)
        except FileNotFoundError:
            return LsResult(error=f"Directory '{path}' not found")
        except OSError as e:
            return LsResult(error=f"Cannot access '{path}': {e}")

        infos: list[FileInfo] = []
        # Sort: directories first, then by name
        sorted_attrs = sorted(attrs, key=lambda a: (not self._attr_is_dir(a), a.filename))

        for attr in sorted_attrs:
            is_dir = self._attr_is_dir(attr)
            size = attr.st_size or 0
            modified_at = ""
            if attr.st_mtime is not None:
                modified_at = datetime.fromtimestamp(attr.st_mtime, tz=timezone.utc).isoformat()

            pp = PurePosixPath(path) / attr.filename
            virtual_path = str(pp) + ("/" if is_dir else "")

            infos.append(FileInfo(
                path=virtual_path,
                is_dir=is_dir,
                size=size,
                modified_at=modified_at,
            ))

        return LsResult(entries=infos)

    @staticmethod
    def _attr_is_dir(attr: paramiko.SFTPAttributes) -> bool:
        """Check whether an SFTP attribute represents a directory."""
        mode = attr.st_mode
        return mode is not None and (mode & 0o40000) != 0

    # ------------------------------------------------------------------
    # Core: read
    # ------------------------------------------------------------------

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
        include_line_numbers: bool = False,
    ) -> ReadResult:
        remote = self._resolve(file_path)
        try:
            is_dir = self._sftp.stat(remote).st_mode
            if self._attr_is_dir_maybe(is_dir):
                return ReadResult(error=f"'{file_path}' is a directory")
        except FileNotFoundError:
            return ReadResult(error=f"File '{file_path}' not found")
        except OSError as e:
            return ReadResult(error=f"Error accessing '{file_path}': {e}")

        file_type = _get_file_type(file_path)
        mime_type = _get_mime_type(file_path)

        if file_type != "text":
            try:
                with self._sftp.open(remote, "rb") as f:
                    raw = f.read()
            except OSError as e:
                return ReadResult(error=f"Error reading '{file_path}': {e}")
            encoded = base64.b64encode(raw).decode("ascii")
            return ReadResult(
                content=encoded,
                total_lines=1,
                encoding="base64",
                file_type=file_type,
                mime_type=mime_type,
            )

        # Text file: attempt UTF-8, fallback to base64
        try:
            with self._sftp.open(remote, "r") as f:
                content = f.read()
        except UnicodeDecodeError:
            # Re-read as binary and encode
            try:
                with self._sftp.open(remote, "rb") as f:
                    raw = f.read()
            except OSError as e:
                return ReadResult(error=f"Error reading '{file_path}': {e}")
            encoded = base64.b64encode(raw).decode("ascii")
            return ReadResult(
                content=encoded,
                total_lines=1,
                encoding="base64",
                file_type=file_type,
                mime_type=mime_type,
            )
        except OSError as e:
            return ReadResult(error=f"Error reading '{file_path}': {e}")

        lines = content.splitlines(keepends=True)
        total = len(lines)

        if offset >= total:
            return ReadResult(
                error=f"Line offset {offset} exceeds file length ({total} lines)",
            )

        sliced = lines[offset : offset + limit]
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
        self, file_path: str, content: str, overwrite: bool = False,
    ) -> WriteResult:
        remote = self._resolve(file_path)

        # Check existence
        try:
            self._sftp.stat(remote)
            exists = True
        except FileNotFoundError:
            exists = False

        if exists and not overwrite:
            return WriteResult(
                error=(
                    f"Cannot write '{file_path}': file already exists. "
                    "Read the file and use edit() to modify it, "
                    "or use overwrite=True to replace the file."
                ),
            )

        # Ensure parent directories exist
        self._ensure_remote_dir(str(PurePosixPath(remote).parent))

        try:
            with self._sftp.open(remote, "w") as f:
                f.write(content)
        except OSError as e:
            return WriteResult(error=f"Error writing '{file_path}': {e}")

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
        file_path: str,
        old_str: str,
        new_str: str,
        *,
        replace_all: bool = False,
    ) -> EditResult:
        """Replace *old_str* with *new_str* in a remote file.

        The edit is performed **on the remote machine** via a single
        ``python3 -c`` call with base64-encoded strings to avoid shell
        escaping issues.  This requires ``python3`` to be installed on
        the remote server.
        """
        remote = self._resolve(file_path)

        # Check file exists
        try:
            self._sftp.stat(remote)
        except FileNotFoundError:
            return EditResult(
                error=(
                    f"Cannot edit '{file_path}': file not found. "
                    "To create a new file, use write()."
                ),
            )

        # Normalize line endings
        old_str = old_str.replace("\r\n", "\n").replace("\r", "\n")
        new_str = new_str.replace("\r\n", "\n").replace("\r", "\n")

        old_b64 = base64.b64encode(old_str.encode()).decode()
        new_b64 = base64.b64encode(new_str.encode()).decode()
        replace_count = "-1" if replace_all else "1"

        remote_escaped = shlex.quote(remote)

        script = (
            "import base64; "
            f"old = base64.b64decode('{old_b64}').decode(); "
            f"new = base64.b64decode('{new_b64}').decode(); "
            f"c = open({remote_escaped}, 'r', encoding='utf-8').read(); "
            "n = c.count(old); "
            "print(n); "
            f"if not (n > 1 and {replace_count} == '1'): "
            f"    open({remote_escaped}, 'w', encoding='utf-8').write(c.replace(old, new, {replace_count}))"
        )

        out, err, exit_code = self._exec(f"python3 -c {shlex.quote(script)}")

        if exit_code != 0:
            return EditResult(error=f"Edit failed (exit {exit_code}): {err or out}")

        try:
            occurrences = int(out.strip())
        except ValueError:
            return EditResult(error=f"Edit failed: unexpected output: {out or err}")

        if occurrences == 0:
            return EditResult(
                error=(
                    f"Cannot edit '{file_path}': old_str not found in file. "
                    "Read the file first to see its exact content."
                ),
            )

        if occurrences > 1 and not replace_all:
            return EditResult(
                error=(
                    f"Cannot edit '{file_path}': old_str appears {occurrences} times "
                    f"in the file. Use replace_all=True to replace all occurrences, "
                    f"or provide a more specific old_str with surrounding context."
                ),
            )

        return EditResult(path=file_path, occurrences=occurrences)

    # ------------------------------------------------------------------
    # Core: grep
    # ------------------------------------------------------------------

    def grep(
        self,
        pattern: str,
        path: str = "/",
        glob: str | None = None,
    ) -> GrepResult:
        """Search for a literal pattern in files under *path*.

        Execution order: (1) try remote ``rg --json -F``, (2) fallback
        to ``grep -rn --``.  Both run on the remote server in a single
        command — no per-file SFTP round-trips.
        """
        remote = self._resolve(path)
        remote_escaped = shlex.quote(remote)
        pattern_escaped = shlex.quote(pattern)

        # 1) Try ripgrep on the remote
        matches = self._rg_remote(pattern_escaped, remote, glob)
        if matches is not None:
            return GrepResult(matches=matches)

        # 2) Fallback to GNU grep
        return self._grep_remote(pattern_escaped, remote_escaped, glob)

    def _rg_remote(
        self,
        pattern_escaped: str,
        remote: str,
        glob: str | None,
    ) -> list[GrepMatch] | None:
        """Run ripgrep on the remote server.  Returns ``None`` if unavailable."""
        cmd = f"rg --json -F"
        if glob:
            cmd += f" --glob {shlex.quote(glob)}"
        cmd += f" -- {pattern_escaped} {shlex.quote(remote)}"

        out, _err, exit_code = self._exec(cmd, timeout=60)

        # rg returns exit 1 when no matches found; exit 2 on error
        if exit_code == 2:
            return None  # rg not available
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

    def _grep_remote(
        self,
        pattern_escaped: str,
        remote_escaped: str,
        glob: str | None,
    ) -> GrepResult:
        """Fallback grep using ``grep -rn`` on the remote."""
        cmd = f"grep -rn -- {pattern_escaped} {remote_escaped}"
        # --include for glob filtering
        if glob:
            cmd = f"grep -rn --include={shlex.quote(glob)} -- {pattern_escaped} {remote_escaped}"

        out, err, exit_code = self._exec(cmd, timeout=60)

        # grep returns exit 1 if no matches
        if exit_code > 1:
            return GrepResult(error=f"grep error: {err}")

        if not out.strip():
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

    def _physical_to_virtual(self, physical_path: str) -> str:
        """Convert a remote absolute path to the virtual path scheme.

        Example: ``/home/user/project/src/main.py`` → ``/project/src/main.py``
        (when *remote_root* is ``/home/user/project``).
        """
        # Normalize separators
        physical_path = physical_path.replace("\\", "/")
        if physical_path.startswith(self._remote_root):
            suffix = physical_path[len(self._remote_root):]
            return suffix if suffix.startswith("/") else "/" + suffix
        # If the path doesn't start with remote_root, return as-is
        return physical_path

    # ------------------------------------------------------------------
    # Core: glob
    # ------------------------------------------------------------------

    def glob(self, pattern: str, path: str = "/") -> GlobResult:
        """Find files matching *pattern* under *path*.

        Uses remote ``find`` — one command covers the entire subtree.
        """
        remote = self._resolve(path)
        remote_escaped = shlex.quote(remote)
        pattern_escaped = shlex.quote(pattern)

        cmd = (
            f"find {remote_escaped} -type f -name {pattern_escaped} "
            f"-printf '%s\\t%p\\n' 2>/dev/null"
        )

        out, err, exit_code = self._exec(cmd, timeout=60)
        if exit_code > 1:
            return GlobResult(error=f"glob error: {err}")

        matches: list[FileInfo] = []
        for line in out.splitlines():
            if not line.strip():
                continue
            if "\t" not in line:
                continue
            size_str, physical_path = line.split("\t", 1)
            try:
                size = int(size_str)
            except ValueError:
                size = 0
            virt = self._physical_to_virtual(physical_path.strip())
            matches.append(FileInfo(
                path=virt,
                is_dir=False,
                size=size,
            ))

        return GlobResult(matches=matches)

    # ------------------------------------------------------------------
    # Extra: tree
    # ------------------------------------------------------------------

    def tree(self, path: str = "/", depth: int = 3) -> str:
        """Render a directory tree using remote ``find``.

        Args:
            path: Root directory to display (default ``"/"``).
            depth: Maximum recursion depth (default 3).

        Returns:
            Formatted tree string.
        """
        remote = self._resolve(path)
        remote_escaped = shlex.quote(remote)

        # Use find to list dirs + files with sizes, respecting max depth
        cmd = (
            f"find {remote_escaped} -maxdepth {depth} -not -path '*/.*' "
            f"\\( -type d -printf 'd %P\\n' , -type f -printf 'f %P %s\\n' \\) "
            f"2>/dev/null | sort"
        )

        out, _err, exit_code = self._exec(cmd, timeout=30)
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

        # Build tree entries list: (display_name, depth)
        # We build it from the dirs and files parsed from find output
        entries: list[tuple[str, int]] = []

        # Add path root itself
        pp = PurePosixPath(path)
        root_name = pp.name or remote.split("/")[-1] or "/"
        entries.append((root_name + "/", 0))

        # Build depth from relative paths
        def _path_depth(rel: str) -> int:
            if rel == "." or not rel:
                return 1
            return rel.count("/") + 1

        # Sort dirs by depth then name
        sorted_dirs = sorted(dirs, key=lambda d: (_path_depth(d), d))
        for d in sorted_dirs:
            if _path_depth(d) > depth:
                continue
            entries.append((PurePosixPath(d).name + "/", _path_depth(d)))

        # Sort files by depth then name
        for parent, name, size in sorted(file_entries, key=lambda f: (f[0].count("/"), f[0], f[1])):
            d = parent.count("/") + 1 if parent != "." else 1
            if d > depth:
                continue
            entries.append((f"{name} ({human_size(size)})", d))

        return format_tree_entries(entries)

    # ------------------------------------------------------------------
    # Extra: delete
    # ------------------------------------------------------------------

    def delete(self, path: str) -> str:
        """Delete a file or directory on the remote server.

        Uses ``rm -rf`` on the remote side.  Refuses to delete the
        remote root directory.
        """
        remote = self._resolve(path)

        # Safety: refuse to delete the root dir
        if remote.rstrip("/") == self._remote_root.rstrip("/"):
            return "Error: cannot delete root working directory."

        remote_escaped = shlex.quote(remote)
        out, err, exit_code = self._exec(f"rm -rf {remote_escaped}")

        if exit_code != 0:
            return f"Error deleting '{path}': {err or out}"

        return f"Deleted: {path}"

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
        if err:
            for line in err.strip().split("\n"):
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
    # Async variants
    # ------------------------------------------------------------------

    async def atree(self, path: str = "/", depth: int = 3) -> str:
        return await asyncio.to_thread(self.tree, path, depth)

    async def adelete(self, path: str) -> str:
        return await asyncio.to_thread(self.delete, path)

    async def aexecute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> str:
        return await asyncio.to_thread(self.execute, command, timeout=timeout)

    # ------------------------------------------------------------------
    # Developer API — bulk upload / download
    # ------------------------------------------------------------------

    def upload_files(
        self,
        files: list[tuple[str, bytes]],
    ) -> list[UploadFileResult]:
        """Upload multiple files as raw bytes to the remote server.

        Binary files are written directly via SFTP (not base64-encoded
        as text), mirroring ``LocalBackend.upload_files``.
        """
        results: list[UploadFileResult] = []
        for path, raw_content in files:
            try:
                remote = self._resolve(path)
                self._ensure_remote_dir(str(PurePosixPath(remote).parent))
                with self._sftp.open(remote, "wb") as f:
                    f.write(raw_content)
                results.append(UploadFileResult(path=path))
            except OSError as e:
                results.append(UploadFileResult(path=path, error=str(e)))
            except Exception as e:
                results.append(
                    UploadFileResult(
                        path=path, error=f"{type(e).__name__}: {e}"
                    )
                )
        return results

    def download_files(
        self,
        paths: list[str],
    ) -> list[DownloadFileResult]:
        """Download multiple files as raw bytes from the remote server."""
        results: list[DownloadFileResult] = []
        for path in paths:
            try:
                remote = self._resolve(path)
                try:
                    attr = self._sftp.stat(remote)
                except FileNotFoundError:
                    results.append(
                        DownloadFileResult(
                            path=path, content=None, error="file_not_found"
                        )
                    )
                    continue

                if self._attr_is_dir_maybe(attr.st_mode):
                    results.append(
                        DownloadFileResult(
                            path=path, content=None, error="is_directory"
                        )
                    )
                    continue

                with self._sftp.open(remote, "rb") as f:
                    raw = f.read()
                results.append(
                    DownloadFileResult(path=path, content=raw)
                )
            except OSError as e:
                results.append(
                    DownloadFileResult(path=path, content=None, error=str(e))
                )
            except Exception as e:
                results.append(
                    DownloadFileResult(
                        path=path,
                        content=None,
                        error=f"{type(e).__name__}: {e}",
                    )
                )
        return results
