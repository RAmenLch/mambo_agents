"""Integration test for SshBackend._edit_remote().

Requires a real SSH server.  Set ``MAMBO_TEST_SSH_PASSWORD`` in your
environment before running:

    pytest tests/test_edit_remote_integration.py -v -s
"""

from __future__ import annotations

import os
import sys
import textwrap
from datetime import datetime

import pytest

from mambo_agents.backends.ssh import SshBackend

# ---------------------------------------------------------------------------
# Config – host/port are hardcoded; password comes from env
# ---------------------------------------------------------------------------

_HOST = "jd.ramenl.top"
_USER = "ramenl"
_PORT = 22


def _get_password() -> str:
    pw = os.environ.get("MAMBO_TEST_SSH_PASSWORD", "")
    if not pw:
        pytest.fail("MAMBO_TEST_SSH_PASSWORD not set in environment")
    return pw


# ---------------------------------------------------------------------------
# Fixture – shared backend
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def backend():
    """Create one SshBackend that lives for the whole test module."""
    pw = _get_password()
    be = SshBackend(
        host=_HOST,
        username=_USER,
        port=_PORT,
        password=pw,
        remote_root="~",
        workspace_root="/workspace",
        connect_timeout=15,
        execute_timeout=30,
    )
    yield be
    be.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_WORKER_ID = os.environ.get("PYTEST_XDIST_WORKER", "gw0")
_TEST_FILE = f"/workspace/_test_edit_remote_{_WORKER_ID}.txt"


def _write_test_data(be: SshBackend, content: str) -> None:
    """Write *content* to the test file, overwriting if it exists."""
    # Delete first in case it already exists
    be.delete(_TEST_FILE)
    result = be.write(_TEST_FILE, content)
    if result.error:
        pytest.fail(f"write() failed: {result.error}")


def _read_back(be: SshBackend) -> str:
    """Read back the test file as plain text."""
    result = be.read(_TEST_FILE)
    if result.error:
        pytest.fail(f"read() failed: {result.error}")
    assert result.content is not None
    return result.content


def _cleanup(be: SshBackend) -> None:
    be.delete(_TEST_FILE)


# ---------------------------------------------------------------------------
# Pre-flight check
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_connection_ok(backend: SshBackend):
    """Verify we can connect and python3 is available (required for _edit_remote)."""
    assert backend.is_connected, "SSH connection is not active"
    assert backend._has_python3, (
        "python3 is **required** on the remote for _edit_remote(). "
        "If not available the edit falls back to SFTP (not this test's target)."
    )
    print(f"\n  ✅ Connected to {_HOST}:{_PORT} as {_USER}")
    print(f"  ✅ remote_root = {backend._remote_root}")
    print(f"  ✅ _has_python3 = {backend._has_python3}")


# ---------------------------------------------------------------------------
# Test: single occurrence edit
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_edit_single_occurrence(backend: SshBackend):
    """Replace a string that appears exactly once in the file."""
    _write_test_data(backend, "line-A\nline-B\nline-C\n")

    result = backend.edit(
        file_path=_TEST_FILE,
        old_str="line-B",
        new_str="line-B-EDITED",
    )

    assert result.error is None, f"Edit should succeed: {result.error}"
    assert result.occurrences == 1, f"Expected 1 occurrence, got {result.occurrences}"
    assert result.path == _TEST_FILE

    content = _read_back(backend)
    assert "line-B-EDITED" in content
    assert "line-B\n" not in content  # old line gone
    print(f"\n  ✅ single occurrence: {result}")


# ---------------------------------------------------------------------------
# Test: multiple occurrences WITHOUT replace_all (should fail)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_edit_multiple_without_replace_all(backend: SshBackend):
    """Edit a string that appears twice – should fail without replace_all=True."""
    _write_test_data(backend, "apple\nbanana\napple\n")

    result = backend.edit(
        file_path=_TEST_FILE,
        old_str="apple",
        new_str="orange",
        replace_all=False,  # explicit default
    )

    # Should be an error: 2 occurrences, no replace_all
    assert result.error is not None, (
        f"Edit with 2 occurrences and replace_all=False should FAIL. "
        f"Got: {result}"
    )
    assert "2 times" in result.error or "appears" in result.error, (
        f"Error should mention multiple occurrences. Got: {result.error}"
    )

    # File should be UNCHANGED
    content = _read_back(backend)
    assert content.count("apple") == 2, "File should NOT have been modified"
    assert content.count("orange") == 0
    print(f"\n  ✅ multi occurrence blocked correctly: {result.error}")


# ---------------------------------------------------------------------------
# Test: multiple occurrences WITH replace_all (should succeed)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_edit_replace_all(backend: SshBackend):
    """Replace all occurrences of a string with replace_all=True."""
    _write_test_data(backend, "apple\nbanana\napple\n")

    result = backend.edit(
        file_path=_TEST_FILE,
        old_str="apple",
        new_str="orange",
        replace_all=True,
    )

    assert result.error is None, f"Edit with replace_all=True should succeed: {result.error}"
    assert result.occurrences == 2

    content = _read_back(backend)
    assert content.count("orange") == 2
    assert "apple" not in content
    print(f"\n  ✅ replace_all: {result}")


# ---------------------------------------------------------------------------
# Test: non-existent old_str (should fail)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_edit_not_found(backend: SshBackend):
    """Search for a string that isn't in the file."""
    _write_test_data(backend, "hello world\n")

    result = backend.edit(
        file_path=_TEST_FILE,
        old_str="xyz-not-there",
        new_str="replaced",
    )

    assert result.error is not None
    assert "not found" in result.error.lower()
    print(f"\n  ✅ not found: {result.error}")


# ---------------------------------------------------------------------------
# Test: unicode / emoji in old_str & new_str (regression for SyntaxError)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_edit_with_unicode_and_emoji(backend: SshBackend):
    """Verify edit works with Chinese characters and emoji (the bug report case)."""
    _write_test_data(backend, "- 测试 edit ✅\n")

    result = backend.edit(
        file_path=_TEST_FILE,
        old_str="- 测试 edit ✅",
        new_str="- 测试 edit ✅ ✅ (已编辑过!)",
    )

    assert result.error is None, f"Unicode edit should succeed: {result.error}"
    assert result.occurrences == 1

    content = _read_back(backend)
    assert "已编辑过" in content
    assert "✅ ✅" in content
    print(f"\n  ✅ unicode/emoji edit works: {result}")


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_cleanup(backend: SshBackend):
    """Remove the test file."""
    _cleanup(backend)
    # Verify deletion
    result = backend.read(_TEST_FILE)
    assert result.error is not None
    assert "not found" in (result.error or "").lower()
    print("\n  ✅ test file cleaned up")
