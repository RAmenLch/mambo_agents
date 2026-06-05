"""Tests for SshBackend – unit tests with mocked paramiko SFTP client.

These tests verify the core read/write/edit logic without requiring a
real SSH server, focusing on edge cases the local-backend tests can't
cover (e.g. bytes-returning SFTP, base64 fallback for binary files).
"""

from __future__ import annotations

import base64
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from mambo_agents.backends.protocol import ReadResult, WriteResult, WorkspacePathError
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


@pytest.fixture
def ssh_backend():
    """Create an SshBackend with a fully mocked paramiko layer.

    The backend connects to a fake remote_root of ``/home/test``.
    Callers can replace ``backend._sftp.open``, ``backend._sftp.stat``,
    etc. on a per-test basis.
    """
    with patch("mambo_agents.backends.ssh.paramiko.SSHClient") as mock_client_cls, \
         patch("mambo_agents.backends.ssh.paramiko.AutoAddPolicy"):
        mock_client = mock_client_cls.return_value
        mock_sftp = MagicMock()

        # Simulate ``exec_command("echo $HOME")`` to resolve remote_root
        mock_home_stdout = MagicMock()
        mock_home_stdout.read.return_value = b"/home/test\n"
        mock_home_stderr = MagicMock()
        mock_home_stderr.read.return_value = b""
        mock_client.exec_command.return_value = (
            MagicMock(),  # stdin
            mock_home_stdout,
            mock_home_stderr,
        )
        mock_client.open_sftp.return_value = mock_sftp

        backend = SshBackend(
            host="fake-host",
            username="testuser",
            password="testpass",
            remote_root="~",
        )
        return backend


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

        result = ssh_backend.read_raw(f"{_W}/test.txt")

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

        result = ssh_backend.read_raw(f"{_W}/test.txt", offset=1, limit=2)

        assert result.content == "B\nC\n"

    def test_read_with_line_numbers(self, ssh_backend):
        """include_line_numbers=True should prefix lines with cat -n style numbers."""
        ssh_backend._sftp.stat.return_value = _sftp_file_stat()
        text = "hello\nworld\n"
        ssh_backend._sftp.open.return_value = _sftp_bytes_file(text.encode("utf-8"))

        result = ssh_backend.read_raw(f"{_W}/test.txt", include_line_numbers=True)

        # format_with_line_numbers uses {num:6d}\t{line} style
        assert "     1\thello" in result.content
        assert "     2\tworld" in result.content

    def test_read_binary_file_returns_base64(self, ssh_backend):
        """Non-text files should be base64-encoded."""
        ssh_backend._sftp.stat.return_value = _sftp_file_stat()
        raw_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
        ssh_backend._sftp.open.return_value = _sftp_bytes_file(raw_bytes)

        result = ssh_backend.read_raw(f"{_W}/image.png")

        assert result.encoding == "base64"
        assert result.file_type == "image"
        assert result.content == base64.b64encode(raw_bytes).decode("ascii")

    def test_read_directory_returns_error(self, ssh_backend):
        """Reading a directory should return an error."""
        ssh_backend._sftp.stat.return_value = _sftp_dir_stat()

        result = ssh_backend.read_raw(f"{_W}/some_dir")

        assert result.content is None
        assert result.error is not None
        assert "directory" in result.error

    def test_read_file_not_found(self, ssh_backend):
        """Non-existent file should return an error."""
        from paramiko import SFTPError
        ssh_backend._sftp.stat.side_effect = FileNotFoundError()

        result = ssh_backend.read_raw(f"{_W}/nonexistent.txt")

        assert result.content is None
        assert result.error is not None
        assert "not found" in result.error.lower()

    def test_read_unicode_decode_error_fallback(self, ssh_backend):
        """When UTF-8 decode fails, fall back to base64 encoding."""
        ssh_backend._sftp.stat.return_value = _sftp_file_stat()
        # Non-UTF-8 bytes that can't be decoded as UTF-8
        raw_bytes = b"\x80\x81\x82\x83"
        ssh_backend._sftp.open.return_value = _sftp_bytes_file(raw_bytes)

        result = ssh_backend.read_raw(f"{_W}/broken.txt")

        assert result.encoding == "base64"
        assert result.content == base64.b64encode(raw_bytes).decode("ascii")

    def test_read_empty_file(self, ssh_backend):
        """Empty text file should return empty content, not an error."""
        ssh_backend._sftp.stat.return_value = _sftp_file_stat()
        ssh_backend._sftp.open.return_value = _sftp_bytes_file(b"")

        result = ssh_backend.read_raw(f"{_W}/empty.txt")

        assert result.content == ""
        assert result.total_lines == 0
        assert result.error is None

    def test_path_outside_workspace_rejected(self, ssh_backend):
        """Paths outside /workspace are rejected even before SFTP access."""
        result = ssh_backend.read_raw("/etc/passwd")
        assert result.error is not None
        assert "outside the workspace" in (result.error or "")


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

        ssh_backend.read_raw(f"{_W}/test.txt")

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

        result = ssh_backend.read_raw(f"{_W}/test.txt", include_line_numbers=True)

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

        result = ssh_backend.write(f"{_W}/new_file.txt", "hello world")

        assert isinstance(result, WriteResult)
        assert result.error is None
        mock_file.write.assert_called_once_with("hello world")

    def test_write_blocked_by_whitelist(self, ssh_backend):
        """Write outside edit_whitelist should be rejected."""
        ssh_backend._edit_whitelist = frozenset([f"{_W}/allowed"])
        ssh_backend._sftp.stat.side_effect = FileNotFoundError()

        result = ssh_backend.write(f"{_W}/forbidden/file.txt", "data")

        assert result.error is not None
        assert "not allowed" in result.error.lower()

    def test_write_outside_workspace_rejected(self, ssh_backend):
        """Write to path outside /workspace is rejected."""
        result = ssh_backend.write("/etc/hosts", "evil")
        assert result.error is not None
        assert "outside the workspace" in (result.error or "")
