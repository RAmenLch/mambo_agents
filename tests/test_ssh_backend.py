"""Tests for SshBackend – unit tests with mocked paramiko SFTP client.

These tests verify the core read/write/edit logic without requiring a
real SSH server, focusing on edge cases the local-backend tests can't
cover (e.g. bytes-returning SFTP, non-UTF-8 decode errors).
"""

from __future__ import annotations

import base64
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from mambo_agents.backends.protocol import ReadResult, WriteResult
from mambo_agents.backends.schemas import BackendError, VirtualPath
from mambo_agents.backends.ssh import SshBackend


_W = "/workspace"


# ---------------------------------------------------------------------------
# Helpers – mock an SFTP file handle
# ---------------------------------------------------------------------------


def _sftp_bytes_file(content: bytes) -> MagicMock:
    """Return a mock file handle whose ``read()`` returns *content*."""
    mock_file = MagicMock()
    mock_file.__enter__.return_value = mock_file
    mock_file.__exit__.return_value = None
    mock_file.read.return_value = content
    return mock_file


def _sftp_dir_stat() -> MagicMock:
    """Mock SFTPAttributes for a directory (st_mode with S_IFDIR)."""
    stat = MagicMock()
    stat.st_mode = 0o040755  # directory
    return stat


def _sftp_file_stat() -> MagicMock:
    """Mock SFTPAttributes for a regular file."""
    stat = MagicMock()
    stat.st_mode = 0o100644  # regular file
    return stat


# ---------------------------------------------------------------------------
# Fixture – SshBackend with mocked SSH + SFTP
# ---------------------------------------------------------------------------


def _build_mocked_ssh_client(*, remote_root: str, home_dir: str = "/home/test"):
    """Build a trio of (mock_client, mock_sftp) with *remote_root* support.

    *home_dir* controls what ``echo $HOME`` returns when ``~`` is expanded.
    """
    mock_client = MagicMock()
    mock_sftp = MagicMock()

    # Simulate ``exec_command("echo $HOME")`` to resolve ~ path
    mock_home_stdout = MagicMock()
    mock_home_stdout.read.return_value = (home_dir + "\n").encode("utf-8")
    mock_home_stderr = MagicMock()
    mock_home_stderr.read.return_value = b""
    mock_client.exec_command.return_value = (
        MagicMock(),  # stdin
        mock_home_stdout,
        mock_home_stderr,
    )
    mock_client.open_sftp.return_value = mock_sftp

    return mock_client, mock_sftp


def _make_ssh_backend(remote_root: str, **kwargs):
    """Create an SshBackend with mocked paramiko, validating *remote_root*."""
    with patch("mambo_agents.backends.ssh.paramiko.SSHClient") as mock_client_cls, \
         patch("mambo_agents.backends.ssh.paramiko.AutoAddPolicy"):
        mock_client, mock_sftp = _build_mocked_ssh_client(remote_root=remote_root)
        mock_client_cls.return_value = mock_client

        # _connect() validates remote_root via sftp.stat — we MUST provide
        # a valid directory stat to avoid a spurious FileNotFoundError.
        mock_sftp.stat.return_value = _sftp_dir_stat()

        return SshBackend(
            host="fake-host",
            username="testuser",
            password="testpass",
            remote_root=remote_root,
            **kwargs,
        )


@pytest.fixture
def ssh_backend():
    """Create an SshBackend with a fully mocked paramiko layer.

    The backend connects to a fake remote_root of ``/home/test``.
    Callers can replace ``backend._sftp.open``, ``backend._sftp.stat``,
    etc. on a per-test basis.
    """
    return _make_ssh_backend(remote_root="~")


# ============================================================================
# read_raw
# ============================================================================


class TestReadRaw:
    """Unit tests for SshBackend.read_raw()."""

    def test_read_text_file_utf8(self, ssh_backend):
        """Happy path: read a UTF-8 text file."""
        ssh_backend._sftp.stat.return_value = _sftp_file_stat()
        text = "line1\nline2\nline3\n"
        ssh_backend._sftp.open.return_value = _sftp_bytes_file(text.encode("utf-8"))

        result = ssh_backend.read_raw(VirtualPath(f"{_W}/test.txt"))

        assert isinstance(result, ReadResult)
        assert result.content == "line1\nline2\nline3\n"
        assert result.total_lines == 3
        assert result.error is None
        assert result.encoding == "utf-8"

    def test_read_with_offset_limit(self, ssh_backend):
        """Offset and limit should slice lines correctly."""
        ssh_backend._sftp.stat.return_value = _sftp_file_stat()
        text = "A\nB\nC\nD\nE\n"
        ssh_backend._sftp.open.return_value = _sftp_bytes_file(text.encode("utf-8"))

        result = ssh_backend.read_raw(VirtualPath(f"{_W}/test.txt"), offset=1, limit=2)

        assert result.content == "B\nC\n"

    def test_read_with_line_numbers(self, ssh_backend):
        """include_line_numbers=True should prefix lines with cat -n style numbers."""
        ssh_backend._sftp.stat.return_value = _sftp_file_stat()
        text = "hello\nworld\n"
        ssh_backend._sftp.open.return_value = _sftp_bytes_file(text.encode("utf-8"))

        result = ssh_backend.read_raw(VirtualPath(f"{_W}/test.txt"), include_line_numbers=True)

        # format_with_line_numbers uses {num:6d}\t{line} style
        assert "     1\thello" in result.content
        assert "     2\tworld" in result.content

    def test_read_binary_file_returns_base64(self, ssh_backend):
        """Non-text files should be base64-encoded."""
        ssh_backend._sftp.stat.return_value = _sftp_file_stat()
        raw_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        ssh_backend._sftp.open.return_value = _sftp_bytes_file(raw_bytes)

        result = ssh_backend.read_raw(VirtualPath(f"{_W}/image.png"))

        assert result.encoding == "base64"
        assert result.file_type == "image"
        assert result.content == base64.b64encode(raw_bytes).decode("ascii")

    def test_read_directory_returns_error(self, ssh_backend):
        """Reading a directory should return an error."""
        ssh_backend._sftp.stat.return_value = _sftp_dir_stat()

        result = ssh_backend.read_raw(VirtualPath(f"{_W}/some_dir"))

        assert result.content is None
        assert result.error is not None
        assert "目录" in str(result.error)

    def test_read_file_not_found(self, ssh_backend):
        """Non-existent file should return an error."""
        from paramiko import SFTPError
        ssh_backend._sftp.stat.side_effect = FileNotFoundError()

        result = ssh_backend.read_raw(VirtualPath(f"{_W}/nonexistent.txt"))

        assert result.content is None
        assert result.error is not None
        assert "不存在" in str(result.error)

    def test_read_unicode_decode_error(self, ssh_backend):
        """When UTF-8 decode fails on a non-multimedia file, return an error."""
        ssh_backend._sftp.stat.return_value = _sftp_file_stat()
        # Non-UTF-8 bytes that can't be decoded as UTF-8
        raw_bytes = b"\x80\x81\x82\x83"
        ssh_backend._sftp.open.return_value = _sftp_bytes_file(raw_bytes)

        result = ssh_backend.read_raw(VirtualPath(f"{_W}/broken.txt"))

        assert result.error is not None
        assert "无法读取" in str(result.error)
        assert result.content is None

    def test_read_empty_file(self, ssh_backend):
        """Empty text file should return empty content, not an error."""
        ssh_backend._sftp.stat.return_value = _sftp_file_stat()
        ssh_backend._sftp.open.return_value = _sftp_bytes_file(b"")

        result = ssh_backend.read_raw(VirtualPath(f"{_W}/empty.txt"))

        assert result.content == ""
        assert result.total_lines == 0
        assert result.error is None

    def test_path_outside_workspace_rejected(self, ssh_backend):
        """Paths outside /workspace are rejected even before SFTP access."""
        result = ssh_backend.read_raw(VirtualPath("/etc/passwd"))
        assert result.error is not None
        assert "超出工作区" in str(result.error)


# ============================================================================
# Regression: SFTP "r" mode returning bytes (now fixed with "rb" mode)
# ============================================================================


class TestReadRawBytesRegression:
    """Verify the fix for TypeError when SFTP.read() returned bytes.

    The old implementation used ``self._sftp.open(remote, "r")`` which
    could return ``bytes`` on some Paramiko/SFTP server combinations,
    leading to ``TypeError: sequence item 0: expected str instance,
    bytes found`` when calling ``"".join(sliced)``.

    The fix opens files in binary mode (``"rb"``) and explicitly
    decodes UTF-8, avoiding any ambiguity about the return type.
    """

    def test_open_mode_is_rb_not_r(self, ssh_backend):
        """Verify read_raw opens files in binary mode (rb), not text mode (r)."""
        ssh_backend._sftp.stat.return_value = _sftp_file_stat()
        text = "content\n"
        ssh_backend._sftp.open.return_value = _sftp_bytes_file(text.encode("utf-8"))

        ssh_backend.read_raw(VirtualPath(f"{_W}/test.txt"))

        call_kwargs = ssh_backend._sftp.open.call_args
        assert call_kwargs is not None
        # First positional arg is the path, second should be "rb"
        assert call_kwargs[0][1] == "rb", (
            f"Expected open mode 'rb', got {call_kwargs[0][1]!r}. "
            "Binary mode ensures .read() always returns bytes (not str)."
        )

    def test_bytes_from_sftp_handled_correctly(self, ssh_backend):
        """Even if some SFTP impl returns bytes-like, decode handles it."""
        ssh_backend._sftp.stat.return_value = _sftp_file_stat()
        # Simulate content that would be bytes — our "rb" mode guarantees this
        ssh_backend._sftp.open.return_value = _sftp_bytes_file(b"hello\nworld\n")

        result = ssh_backend.read_raw(VirtualPath(f"{_W}/test.txt"), include_line_numbers=True)

        assert "hello" in result.content
        assert "world" in result.content
        assert result.error is None
        # No TypeError — the fix works


# ============================================================================
# write
# ============================================================================


class TestWrite:
    """Unit tests for SshBackend.write()."""

    def test_write_text_file(self, ssh_backend):
        """Write a text file via SFTP."""
        ssh_backend._sftp.stat.side_effect = FileNotFoundError()  # file doesn't exist yet

        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.__exit__.return_value = None
        ssh_backend._sftp.open.return_value = mock_file

        result = ssh_backend.write(VirtualPath(f"{_W}/new_file.txt"), "hello world")

        assert isinstance(result, WriteResult)
        assert result.error is None
        mock_file.write.assert_called_once_with("hello world")

    def test_write_blocked_by_whitelist(self, ssh_backend):
        """Write outside edit_whitelist should be rejected."""
        ssh_backend._edit_whitelist = frozenset([VirtualPath(f"{_W}/allowed")])
        ssh_backend._sftp.stat.side_effect = FileNotFoundError()

        result = ssh_backend.write(VirtualPath(f"{_W}/forbidden/file.txt"), "data")

        assert result.error is not None
        assert "不允许" in str(result.error)

    def test_write_outside_workspace_rejected(self, ssh_backend):
        """Write to path outside /workspace is rejected."""
        result = ssh_backend.write(VirtualPath("/etc/hosts"), "evil")
        assert result.error is not None
        assert "超出工作区" in str(result.error)


# ============================================================================
# _remote_root validation during _connect()
# ============================================================================


class TestRemoteRootValidation:
    """Tests for _remote_root SSH validation performed during _connect()."""

    # ------------------------------------------------------------------
    # Happy path
    # ------------------------------------------------------------------

    def test_valid_absolute_path(self):
        """Non-tilde absolute path with existing directory → backend created."""
        backend = _make_ssh_backend(remote_root="/opt/project")
        assert backend._remote_root == "/opt/project"

    def test_tilde_expands_to_home(self):
        """~ is expanded to $HOME, then validated via stat."""
        backend = _make_ssh_backend(remote_root="~")
        assert backend._remote_root == "/home/test"

    def test_trailing_slash_normalized(self):
        """Trailing / is stripped after validation."""
        backend = _make_ssh_backend(remote_root="/opt/project/")
        assert backend._remote_root == "/opt/project"

    def test_trailing_backslash_normalized(self):
        """Trailing \\ (Windows remote) is stripped."""
        with patch("mambo_agents.backends.ssh.paramiko.SSHClient") as mock_cls, \
             patch("mambo_agents.backends.ssh.paramiko.AutoAddPolicy"):
            mock_client, mock_sftp = _build_mocked_ssh_client(remote_root="C:\\Users\\admin")
            mock_cls.return_value = mock_client
            mock_sftp.stat.return_value = _sftp_dir_stat()

            backend = SshBackend(
                host="fake-host",
                username="testuser",
                password="testpass",
                remote_root="C:\\Users\\admin\\",
            )
            assert backend._remote_root == "C:\\Users\\admin"

    # ------------------------------------------------------------------
    # Directory does not exist
    # ------------------------------------------------------------------

    def test_directory_not_found_raises_valueerror(self):
        """sftp.stat raises FileNotFoundError → ValueError with clear message."""
        with patch("mambo_agents.backends.ssh.paramiko.SSHClient") as mock_cls, \
             patch("mambo_agents.backends.ssh.paramiko.AutoAddPolicy"):
            mock_client, mock_sftp = _build_mocked_ssh_client(
                remote_root="/nonexistent"
            )
            mock_cls.return_value = mock_client
            mock_sftp.stat.side_effect = FileNotFoundError()

            with pytest.raises(ValueError, match="does not exist"):
                SshBackend(
                    host="fake-host",
                    username="testuser",
                    password="testpass",
                    remote_root="/nonexistent",
                )

    def test_directory_not_found_message_includes_host(self):
        """Error message includes the host and the path for debugging."""
        with patch("mambo_agents.backends.ssh.paramiko.SSHClient") as mock_cls, \
             patch("mambo_agents.backends.ssh.paramiko.AutoAddPolicy"):
            mock_client, mock_sftp = _build_mocked_ssh_client(
                remote_root="/bad/path"
            )
            mock_cls.return_value = mock_client
            mock_sftp.stat.side_effect = FileNotFoundError()

            with pytest.raises(ValueError) as exc_info:
                SshBackend(
                    host="my-server",
                    username="testuser",
                    password="testpass",
                    remote_root="/bad/path",
                )
            msg = str(exc_info.value)
            assert "/bad/path" in msg
            assert "my-server" in msg

    # ------------------------------------------------------------------
    # Permission denied
    # ------------------------------------------------------------------

    def test_permission_denied_raises_permissionerror(self):
        """sftp.stat raises PermissionError → PermissionError propagated."""
        with patch("mambo_agents.backends.ssh.paramiko.SSHClient") as mock_cls, \
             patch("mambo_agents.backends.ssh.paramiko.AutoAddPolicy"):
            mock_client, mock_sftp = _build_mocked_ssh_client(
                remote_root="/root"
            )
            mock_cls.return_value = mock_client
            mock_sftp.stat.side_effect = PermissionError("permission denied")

            with pytest.raises(PermissionError, match="not accessible"):
                SshBackend(
                    host="fake-host",
                    username="testuser",
                    password="testpass",
                    remote_root="/root",
                )

    # ------------------------------------------------------------------
    # Empty / all-slash values – no fallback to "/"
    # ------------------------------------------------------------------

    def test_empty_string_raises_valueerror(self):
        """remote_root="" → empty after strip → ValueError, NOT falling back to /."""
        with patch("mambo_agents.backends.ssh.paramiko.SSHClient") as mock_cls, \
             patch("mambo_agents.backends.ssh.paramiko.AutoAddPolicy"):
            mock_client, mock_sftp = _build_mocked_ssh_client(remote_root="")
            mock_cls.return_value = mock_client
            mock_sftp.stat.return_value = _sftp_dir_stat()

            with pytest.raises(ValueError, match="empty path"):
                SshBackend(
                    host="fake-host",
                    username="testuser",
                    password="testpass",
                    remote_root="",
                )

    def test_all_slashes_raises_valueerror(self):
        """remote_root="///" → all stripped → empty → ValueError, NOT /."""
        with patch("mambo_agents.backends.ssh.paramiko.SSHClient") as mock_cls, \
             patch("mambo_agents.backends.ssh.paramiko.AutoAddPolicy"):
            mock_client, mock_sftp = _build_mocked_ssh_client(remote_root="///")
            mock_cls.return_value = mock_client
            mock_sftp.stat.return_value = _sftp_dir_stat()

            with pytest.raises(ValueError, match="empty path"):
                SshBackend(
                    host="fake-host",
                    username="testuser",
                    password="testpass",
                    remote_root="///",
                )

    def test_tilde_only_with_no_trailing_path_works(self):
        """remote_root="~" → expands to /home/test → validated → ok."""
        backend = _make_ssh_backend(remote_root="~")
        assert backend._remote_root == "/home/test"


# ============================================================================
# tree – depth calculation regression
# ============================================================================


def _connector_pos(line: str) -> int:
    """Return the character position of the tree connector (├── or └──)."""
    for c in ("├──", "└──"):
        if c in line:
            return line.index(c)
    return -1


class TestTreeDepthCalculation:
    """Regression: files inside subdirectories must be nested at the correct depth.

    The bug: file depth was calculated from ``parent.count("/") + 1``,
    which gives depth 1 for files directly inside a subdirectory (e.g.
    ``parent="0849fe09"`` → depth=0+1=1), placing files at the same
    level as their parent directory instead of one level deeper.

    The fix uses ``(parent + "/" + name).count("/") + 1`` to calculate
    depth from the file's full relative path.

    We verify nesting by checking that the tree connector (├── / └──)
    on child lines appears further to the right than the connector on
    the parent directory line.
    """

    def test_files_nested_under_subdirectories(self, ssh_backend):
        """Files in a subdirectory appear indented, not at the same level as the dir."""
        ssh_backend._has_python3 = False

        # Simulate find output: one dir with two image files inside it
        fake_out = (
            "d 0849fe09\n"
            "f 0849fe09/Anima_00013_.png 1048576\n"
            "f 0849fe09/Anima_00014_.png 2048\n"
        )
        from unittest.mock import MagicMock
        ssh_backend._exec = MagicMock(return_value=(fake_out, "", 0))

        result = ssh_backend.tree(VirtualPath("/workspace"), depth=3)
        lines = result.split("\n")

        # Find line indices
        dir_line = next(i for i, l in enumerate(lines) if "0849fe09/" in l)
        file_a_line = next(i for i, l in enumerate(lines) if "Anima_00013_" in l)
        file_b_line = next(i for i, l in enumerate(lines) if "Anima_00014_" in l)

        # Files must appear after their parent directory
        assert file_a_line > dir_line, f"file_a after its parent dir:\n{result}"
        assert file_b_line > dir_line, f"file_b after its parent dir:\n{result}"

        # Files must be deeper: their connector positions should be > dir's
        dir_cp = _connector_pos(lines[dir_line])
        file_a_cp = _connector_pos(lines[file_a_line])
        file_b_cp = _connector_pos(lines[file_b_line])

        assert file_a_cp > dir_cp, (
            f"file_a should be nested deeper (connector pos {file_a_cp} <= dir {dir_cp}).\n{result}"
        )
        assert file_b_cp > dir_cp, (
            f"file_b should be nested deeper (connector pos {file_b_cp} <= dir {dir_cp}).\n{result}"
        )

        # Sibling files share same depth → same connector position
        assert file_a_cp == file_b_cp, (
            f"Sibling files should share depth; "
            f"connector pos {file_a_cp} != {file_b_cp}.\n{result}"
        )

    def test_files_at_root_level_appear_flat(self, ssh_backend):
        """Files directly under workspace/ should appear at depth 1 (same as subdirs)."""
        ssh_backend._has_python3 = False

        fake_out = (
            "f root_file.txt 100\n"
            "d subdir\n"
            "f subdir/nested.txt 200\n"
        )
        from unittest.mock import MagicMock
        ssh_backend._exec = MagicMock(return_value=(fake_out, "", 0))

        result = ssh_backend.tree(VirtualPath("/workspace"), depth=3)
        lines = result.split("\n")

        # Find each entry
        root_file_line = next(i for i, l in enumerate(lines) if "root_file.txt" in l)
        subdir_line = next(i for i, l in enumerate(lines) if "subdir/" in l)
        nested_line = next(i for i, l in enumerate(lines) if "nested.txt" in l)

        root_cp = _connector_pos(lines[root_file_line])
        subdir_cp = _connector_pos(lines[subdir_line])
        nested_cp = _connector_pos(lines[nested_line])

        # root_file and subdir at same level (both depth 1) → same connector position
        assert root_cp == subdir_cp, (
            f"root_file.txt and subdir/ should share depth 1; "
            f"connector pos {root_cp} vs {subdir_cp}.\n{result}"
        )
        # nested.txt one level deeper (depth 2) → connector further right
        assert nested_cp > subdir_cp, (
            f"nested.txt should be deeper than subdir/; "
            f"connector pos {nested_cp} <= {subdir_cp}.\n{result}"
        )
