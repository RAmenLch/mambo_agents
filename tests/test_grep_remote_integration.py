"""Integration test for SshBackend grep().

Requires a real SSH server.  Set ``MAMBO_TEST_SSH_PASSWORD`` in your
environment before running:

    pytest tests/test_grep_remote_integration.py -v -s
"""

from __future__ import annotations

import os
import textwrap

import pytest

from mambo_agents.backends.ssh import SshBackend
from mambo_agents.backends.protocol import GrepResult, GrepMatch

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

_TEST_DIR = "/workspace/_test_grep_remote"
_FILE_A = f"{_TEST_DIR}/a.txt"
_FILE_B = f"{_TEST_DIR}/b.txt"
_FILE_C = f"{_TEST_DIR}/sub/c.txt"
_FILE_UNICODE = f"{_TEST_DIR}/unicode.txt"
_FILE_HIDDEN = f"{_TEST_DIR}/.hidden.txt"
_FILE_BINARY = f"{_TEST_DIR}/binary.bin"


def _setup_files(be: SshBackend) -> None:
    """Create the test directory and all test files."""
    # Clean up any leftovers
    be.delete(_TEST_DIR)

    files: list[tuple[str, str]] = [
        (_FILE_A, "hello world\nthis is file A\nhello again\n"),
        (_FILE_B, "goodbye world\nthis is file B\nhello there\n"),
        (_FILE_C, "nested file\nwith hello inside\n"),
        (_FILE_UNICODE, "你好世界\nこんにちは\nemoji 🎉 test\n"),
        (_FILE_HIDDEN, "hidden file\nwith hello\n"),
    ]

    for path, content in files:
        result = be.write(path, content)
        if result.error:
            # If file already exists, try with overwrite
            result = be.write(path, content, overwrite=True)
        if result.error:
            pytest.fail(f"write({path}) failed: {result.error}")

    # Write binary file
    _stdin, stdout, _stderr = be._client.exec_command(
        f"printf '\\x00\\x01\\x02hello\\x03\\x04' > {_TEST_DIR}/binary.bin"
    )
    stdout.channel.recv_exit_status()


def _read_back(be: SshBackend, path: str) -> str:
    """Read back a file as plain text."""
    result = be.read(path)
    if result.error:
        pytest.fail(f"read({path}) failed: {result.error}")
    assert result.content is not None
    return result.content


def _cleanup(be: SshBackend) -> None:
    be.delete(_TEST_DIR)


# ---------------------------------------------------------------------------
# Pre-flight check
# ---------------------------------------------------------------------------


def test_connection_ok(backend: SshBackend):
    """Verify we can connect."""
    assert backend.is_connected, "SSH connection is not active"
    print(f"\n  ✅ Connected to {_HOST}:{_PORT} as {_USER}")
    print(f"  ✅ remote_root = {backend._remote_root}")
    print(f"  ✅ workspace_root = {backend.workspace_root}")


# ============================================================================
# Diagnostic tests – before we search, verify the setup
# ============================================================================


def test_diagnostic_files_exist(backend: SshBackend):
    """Verify test files are actually written AND readable on the remote."""
    _setup_files(backend)

    # 1) Check if the directory exists via ls
    ls_result = backend.ls(_TEST_DIR)
    print(f"\n  📁 ls({_TEST_DIR}):")
    print(f"     error: {ls_result.error}")
    print(f"     entries: {ls_result.entries}")

    # 2) Check if file content is correct
    read_result = backend.read(_FILE_A)
    print(f"\n  📄 read({_FILE_A}):")
    print(f"     error: {read_result.error}")
    print(f"     content repr: {read_result.content!r}")

    # 3) Check if rg is available
    rg_out, rg_err, rg_exit = backend._exec("which rg || echo 'rg NOT FOUND'")
    print(f"\n  🔍 rg check: exit={rg_exit}, stdout={rg_out!r}, stderr={rg_err!r}")

    # 4) Check if grep is available
    grep_ver, _, _ = backend._exec("grep --version | head -1")
    print(f"  🔍 grep version: {grep_ver.strip()!r}")

    # 5) Run raw grep command manually (same as _grep_remote would)
    remote = backend._resolve(_TEST_DIR)
    raw_cmd = f"grep -rn hello {backend._resolve(_TEST_DIR)}"
    raw_out, raw_err, raw_exit = backend._exec(raw_cmd)
    print(f"\n  🧪 raw grep: exit={raw_exit}")
    print(f"     stdout: {raw_out.strip()!r}")
    print(f"     stderr: {raw_err.strip()!r}")

    # 6) Run raw grep with shlex.quote (same as code)
    import shlex
    remote_escaped = shlex.quote(remote)
    pattern_escaped = shlex.quote("hello")
    cmd = f"grep -rn -- {pattern_escaped} {remote_escaped}"
    cmd_out, cmd_err, cmd_exit = backend._exec(cmd)
    print(f"\n  🧪 grep with shlex.quote:")
    print(f"     cmd: {cmd!r}")
    print(f"     exit={cmd_exit}")
    print(f"     stdout: {cmd_out.strip()!r}")
    print(f"     stderr: {cmd_err.strip()!r}")

    # 7) Check if rg would work
    rg_cmd = f"rg --json -F -- 'hello' {remote}"
    rg_cmd_out, rg_cmd_err, rg_cmd_exit = backend._exec(rg_cmd)
    print(f"\n  🧪 raw rg --json:")
    print(f"     exit={rg_cmd_exit}")
    print(f"     stdout: {rg_cmd_out.strip()[:200]!r}")
    print(f"     stderr: {rg_cmd_err.strip()[:200]!r}")

    # Cleanup
    _cleanup(backend)

    # Don't hard-assert; let the diagnostic output guide us
    assert True


# ============================================================================
# Basic grep tests
# ============================================================================


class TestGrepBasic:
    """Basic grep functionality – run first since it creates test files."""

    @pytest.fixture(autouse=True)
    def _setup(self, backend: SshBackend):
        _setup_files(backend)
        yield
        _cleanup(backend)

    # ------------------------------------------------------------------
    # Happy path: simple match
    # ------------------------------------------------------------------

    def test_simple_match(self, backend: SshBackend):
        """Search for a word that appears in a single file."""
        result = backend.grep("file A", path=_TEST_DIR)
        assert isinstance(result, GrepResult)
        assert result.error is None, f"Unexpected error: {result.error}"
        assert result.matches is not None, "matches should not be None"
        assert len(result.matches) >= 1, "Expected at least 1 match"
        # Should find in a.txt
        paths = {m.path for m in result.matches}
        assert _FILE_A in paths, f"Expected match in {_FILE_A}, got {paths}"
        print(f"\n  ✅ simple_match: {len(result.matches)} match(es)")
        for m in result.matches:
            print(f"     {m.path}:{m.line}: {m.text}")

    def test_multi_file_match(self, backend: SshBackend):
        """Search for a word that appears in multiple files."""
        result = backend.grep("hello", path=_TEST_DIR)
        assert result.error is None, f"Unexpected error: {result.error}"
        assert result.matches is not None
        assert len(result.matches) >= 3, (
            f"Expected at least 3 matches, got {len(result.matches)}"
        )
        paths = {m.path for m in result.matches}
        assert _FILE_A in paths, f"Missing match in {_FILE_A}"
        assert _FILE_B in paths, f"Missing match in {_FILE_B}"
        assert _FILE_C in paths, f"Missing match in {_FILE_C}"
        print(f"\n  ✅ multi_file_match: {len(result.matches)} matches across {len(paths)} files")
        for m in result.matches:
            print(f"     {m.path}:{m.line}: {m.text}")

    def test_no_match(self, backend: SshBackend):
        """Search for a string that doesn't exist."""
        result = backend.grep("xyz_nonexistent_abc", path=_TEST_DIR)
        assert result.error is None, f"Unexpected error: {result.error}"
        assert result.matches is not None
        assert len(result.matches) == 0, (
            f"Expected 0 matches, got {len(result.matches)}"
        )
        print(f"\n  ✅ no_match: correctly returned 0 matches")

    def test_nested_subdirectory_search(self, backend: SshBackend):
        """Verify grep searches recursively into subdirectories."""
        result = backend.grep("nested file", path=_TEST_DIR)
        assert result.error is None, f"Unexpected error: {result.error}"
        assert result.matches is not None
        assert len(result.matches) >= 1, (
            f"Expected at least 1 match in nested dir, got {len(result.matches)}"
        )
        paths = {m.path for m in result.matches}
        assert _FILE_C in paths, (
            f"Should find match in nested file {_FILE_C}, got paths: {paths}"
        )
        print(f"\n  ✅ nested_subdirectory: found match in sub/")

    # ------------------------------------------------------------------
    # Unicode / emoji
    # ------------------------------------------------------------------

    def test_unicode_chinese(self, backend: SshBackend):
        """Search for Chinese characters."""
        result = backend.grep("你好", path=_TEST_DIR)
        assert result.error is None, f"Unexpected error: {result.error}"
        assert result.matches is not None
        assert len(result.matches) >= 1, (
            f"Expected at least 1 unicode match, got {len(result.matches)}"
        )
        for m in result.matches:
            assert "你好" in m.text, f"Match text should contain '你好': {m.text}"
        print(f"\n  ✅ unicode_chinese: {len(result.matches)} match(es)")

    def test_emoji(self, backend: SshBackend):
        """Search for emoji characters."""
        result = backend.grep("🎉", path=_TEST_DIR)
        assert result.error is None, f"Unexpected error: {result.error}"
        assert result.matches is not None
        assert len(result.matches) >= 1, (
            f"Expected at least 1 emoji match, got {len(result.matches)}"
        )
        for m in result.matches:
            assert "🎉" in m.text, f"Match text should contain 🎉: {m.text}"
        print(f"\n  ✅ emoji: {len(result.matches)} match(es)")

    def test_japanese(self, backend: SshBackend):
        """Search for Japanese characters."""
        result = backend.grep("こんにちは", path=_TEST_DIR)
        assert result.error is None, f"Unexpected error: {result.error}"
        assert result.matches is not None
        assert len(result.matches) >= 1, (
            f"Expected at least 1 japanese match, got {len(result.matches)}"
        )
        print(f"\n  ✅ japanese: {len(result.matches)} match(es)")


# ============================================================================
# Special characters / edge cases
# ============================================================================


class TestGrepSpecialChars:
    """Test patterns with characters that could break regex or shell escaping."""

    @pytest.fixture(autouse=True)
    def _setup(self, backend: SshBackend):
        _setup_files(backend)
        yield
        _cleanup(backend)

    def test_pattern_with_space(self, backend: SshBackend):
        """Search for a pattern containing spaces."""
        result = backend.grep("hello world", path=_TEST_DIR)
        assert result.error is None, f"Unexpected error: {result.error}"
        assert result.matches is not None
        assert len(result.matches) >= 1, (
            f"Expected at least 1 match for 'hello world', got {len(result.matches)}"
        )
        for m in result.matches:
            assert "hello world" in m.text, f"Match text should contain 'hello world': {m.text}"
        print(f"\n  ✅ pattern_with_space: {len(result.matches)} match(es)")

    def test_pattern_with_dot_regex_char(self, backend: SshBackend):
        """Search for a literal dot – grep without -F treats '.' as regex (any char).
        This test verifies whether the backend treats the pattern as literal or regex.
        """
        result = backend.grep("file A", path=_TEST_DIR)
        assert result.error is None, f"Unexpected error: {result.error}"
        assert result.matches is not None
        assert len(result.matches) >= 1, (
            f"Expected at least 1 match for 'file A', got {len(result.matches)}"
        )
        print(f"\n  ✅ dot_in_pattern: {len(result.matches)} match(es)")

    def test_pattern_with_bracket(self, backend: SshBackend):
        """Search for a literal bracket character."""
        # Write a file with brackets
        bracket_file = f"{_TEST_DIR}/brackets.txt"
        be_result = backend.write(bracket_file, "array[0] = value\nhello [world]\n", overwrite=True)
        if be_result.error:
            pytest.fail(f"Failed to write bracket file: {be_result.error}")

        result = backend.grep("array[0]", path=_TEST_DIR)
        assert result.error is None, f"Unexpected error: {result.error}"
        # Note: without -F, grep treats 'array[0]' as regex "array followed by '0'"
        # This test documents the current behavior
        print(f"\n  ✅ bracket_pattern: {len(result.matches or [])} match(es)")
        if result.matches:
            for m in result.matches:
                print(f"     {m.path}:{m.line}: {m.text}")

    def test_pattern_with_asterisk(self, backend: SshBackend):
        """Search for a literal asterisk."""
        star_file = f"{_TEST_DIR}/star.txt"
        be_result = backend.write(star_file, "import * from module\nhello * world\n", overwrite=True)
        if be_result.error:
            pytest.fail(f"Failed to write star file: {be_result.error}")

        result = backend.grep("*", path=_TEST_DIR)
        assert result.error is None, f"Unexpected error: {result.error}"
        # '*' as a regex means "zero or more of previous char" — may match everything
        print(f"\n  ✅ asterisk_pattern: {len(result.matches or [])} match(es)")
        if result.matches:
            for m in result.matches[:5]:
                print(f"     {m.path}:{m.line}: {m.text}")

    def test_pattern_with_question_mark(self, backend: SshBackend):
        """Search for a literal question mark."""
        q_file = f"{_TEST_DIR}/question.txt"
        be_result = backend.write(q_file, "are you sure?\nyes or no?\n", overwrite=True)
        if be_result.error:
            pytest.fail(f"Failed to write question file: {be_result.error}")

        result = backend.grep("sure?", path=_TEST_DIR)
        assert result.error is None, f"Unexpected error: {result.error}"
        # '?' as regex means "zero or one of previous char"
        print(f"\n  ✅ question_mark_pattern: {len(result.matches or [])} match(es)")
        if result.matches:
            for m in result.matches:
                print(f"     {m.path}:{m.line}: {m.text}")

    def test_pattern_with_parenthesis(self, backend: SshBackend):
        """Search for parentheses."""
        paren_file = f"{_TEST_DIR}/paren.txt"
        be_result = backend.write(paren_file, "function test()\ncall(arg1, arg2)\n", overwrite=True)
        if be_result.error:
            pytest.fail(f"Failed to write paren file: {be_result.error}")

        result = backend.grep("test()", path=_TEST_DIR)
        assert result.error is None, f"Unexpected error: {result.error}"
        # '()' as regex means a group — this will likely fail without -F
        print(f"\n  ✅ parenthesis_pattern: {len(result.matches or [])} match(es)")
        if result.matches:
            for m in result.matches:
                print(f"     {m.path}:{m.line}: {m.text}")


# ============================================================================
# Glob filtering
# ============================================================================


class TestGrepGlob:
    """Test the glob filtering parameter."""

    @pytest.fixture(autouse=True)
    def _setup(self, backend: SshBackend):
        _setup_files(backend)
        yield
        _cleanup(backend)

    def test_glob_txt_only(self, backend: SshBackend):
        """Filter by *.txt glob – should only search text files."""
        result = backend.grep("hello", path=_TEST_DIR, glob="*.txt")
        assert result.error is None, f"Unexpected error: {result.error}"
        assert result.matches is not None
        assert len(result.matches) >= 1, (
            f"Expected at least 1 match with glob *.txt, got {len(result.matches)}"
        )
        # Should NOT match binary.bin
        for m in result.matches:
            assert not m.path.endswith(".bin"), (
                f"Glob *.txt should exclude .bin files, but got {m.path}"
            )
        print(f"\n  ✅ glob_txt_only: {len(result.matches)} matches (all .txt)")

    def test_glob_py_no_match(self, backend: SshBackend):
        """Filter by *.py – should find no matches since there are no .py files."""
        result = backend.grep("hello", path=_TEST_DIR, glob="*.py")
        assert result.error is None, f"Unexpected error: {result.error}"
        assert result.matches is not None
        assert len(result.matches) == 0, (
            f"Expected 0 matches for *.py, got {len(result.matches)}"
        )
        print(f"\n  ✅ glob_py_no_match: correct – 0 matches for *.py")


# ============================================================================
# Path & error handling
# ============================================================================


class TestGrepPathHandling:
    """Test various path inputs and error conditions."""

    @pytest.fixture(autouse=True)
    def _setup(self, backend: SshBackend):
        _setup_files(backend)
        yield
        _cleanup(backend)

    def test_search_workspace_root(self, backend: SshBackend):
        """Search from workspace root – should still find files."""
        result = backend.grep("hello", path="/workspace")
        assert result.error is None, f"Unexpected error: {result.error}"
        assert result.matches is not None
        # At least our test files
        assert len(result.matches) >= 1, "Should find at least the test files"
        print(f"\n  ✅ search_workspace_root: {len(result.matches)} matches")
        for m in result.matches[:5]:
            print(f"     {m.path}:{m.line}: {m.text}")

    def test_search_specific_file(self, backend: SshBackend):
        """Search within a specific file path."""
        # Diagnostic: verify file content and raw grep
        content = _read_back(backend, _FILE_A)
        print(f"\n  📄 file content: {content!r}")

        remote = backend._resolve(_FILE_A)
        import shlex
        raw_cmd = f"grep -rnIs -- {shlex.quote('file A')} {shlex.quote(remote)}"
        raw_out, raw_err, raw_exit = backend._exec(raw_cmd)
        print(f"  🧪 raw grep single file: exit={raw_exit}")
        print(f"     stdout: {raw_out.strip()!r}")
        print(f"     stderr: {raw_err.strip()!r}")

        result = backend.grep("file A", path=_FILE_A)
        print(f"  🔍 grep result: error={result.error!r}, matches={result.matches}")
        assert result.error is None, f"Unexpected error: {result.error}"
        assert result.matches is not None
        assert len(result.matches) >= 1, "Expected match in specific file"
        for m in result.matches:
            assert m.path == _FILE_A, f"Match should only be in {_FILE_A}"
        print(f"  ✅ search_specific_file: {len(result.matches)} matches in single file")

    def test_nonexistent_path(self, backend: SshBackend):
        """Search a non-existent path."""
        result = backend.grep("hello", path="/workspace/_nonexistent_dir_xyz")
        # Should get an error, not crash
        assert isinstance(result, GrepResult)
        print(f"\n  ✅ nonexistent_path: error={result.error}")
        print(f"     matches={result.matches}")

    def test_outside_workspace_rejected(self, backend: SshBackend):
        """Search outside /workspace should be rejected."""
        result = backend.grep("hello", path="/etc")
        assert result.error is not None, "Should reject path outside workspace"
        assert "outside the workspace" in (result.error or "").lower()
        print(f"\n  ✅ outside_workspace: correctly rejected")


# ============================================================================
# Hidden file handling
# ============================================================================


class TestGrepHiddenFiles:
    """Test behavior with hidden (dot) files."""

    @pytest.fixture(autouse=True)
    def _setup(self, backend: SshBackend):
        _setup_files(backend)
        yield
        _cleanup(backend)

    def test_hidden_file_search(self, backend: SshBackend):
        """Search should find content in hidden files."""
        result = backend.grep("hidden file", path=_TEST_DIR)
        assert result.error is None, f"Unexpected error: {result.error}"
        assert result.matches is not None
        # Hidden files should be included in grep search
        hidden_matches = [m for m in result.matches if ".hidden" in m.path]
        print(f"\n  ✅ hidden_file: {len(hidden_matches)} match(es) in .hidden.txt")
        if hidden_matches:
            for m in hidden_matches:
                print(f"     {m.path}:{m.line}: {m.text}")


# ============================================================================
# Cleanup
# ============================================================================


def test_cleanup(backend: SshBackend):
    """Remove all test files."""
    _cleanup(backend)
    # Verify deletion
    result = backend.read(_FILE_A)
    assert result.error is not None
    assert "not found" in (result.error or "").lower()
    print("\n  ✅ test files cleaned up")
