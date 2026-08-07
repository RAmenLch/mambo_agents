"""Tests for shared backend utilities in ``mambo_agents.backends.utils``."""

import pytest

from mambo_agents.backends.protocol import EditResult
from mambo_agents.backends.schemas import human_size, VirtualPath
from mambo_agents.backends.utils import (
    TreeEntry,
    check_path_allowed,
    decode_output,
    detect_trailing_newline_mismatch,
    format_tree_entries,
    format_with_line_numbers,
)


# ============================================================================
# decode_output
# ============================================================================


class TestDecodeOutput:
    def test_utf8_bytes(self):
        assert decode_output("你好世界".encode("utf-8")) == "你好世界"

    def test_utf8_preferred_over_gbk_system(self, monkeypatch):
        """UTF-8 字节流在 cp936 系统上必须按 UTF-8 解码(旧实现会解成乱码)。"""
        monkeypatch.setattr("locale.getpreferredencoding", lambda *a, **k: "cp936")
        assert decode_output("你好世界".encode("utf-8")) == "你好世界"

    def test_gbk_bytes_falls_back_to_system_encoding(self, monkeypatch):
        monkeypatch.setattr("locale.getpreferredencoding", lambda *a, **k: "cp936")
        assert decode_output("你好世界".encode("gbk")) == "你好世界"

    def test_crlf_normalized(self):
        assert decode_output(b"a\r\nb\r\n") == "a\nb\n"

    def test_empty_bytes(self):
        assert decode_output(b"") == ""

    def test_invalid_bytes_use_replacement(self, monkeypatch):
        monkeypatch.setattr("locale.getpreferredencoding", lambda *a, **k: "ascii")
        assert decode_output(b"\xff\xfe\xfd") == "\ufffd\ufffd\ufffd"


# ============================================================================
# human_size
# ============================================================================


class TestHumanSize:
    def test_zero_bytes(self):
        assert human_size(0) == "0 B"

    def test_bytes(self):
        assert human_size(512) == "512 B"

    def test_kb(self):
        assert human_size(1024) == "1 KB"
        assert human_size(2048) == "2 KB"

    def test_mb(self):
        assert human_size(1048576) == "1 MB"
        assert human_size(2_097_152) == "2 MB"

    def test_gb(self):
        assert human_size(1073741824) == "1 GB"

    def test_tb(self):
        assert human_size(1099511627776) == "1 TB"
        assert human_size(2_199_023_255_552) == "2 TB"


# ============================================================================
# format_with_line_numbers
# ============================================================================


class TestFormatWithLineNumbers:
    def test_single_line(self):
        result = format_with_line_numbers("hello")
        assert "     1\thello" in result

    def test_multiple_lines(self):
        result = format_with_line_numbers("a\nb\nc")
        lines = result.split("\n")
        assert len(lines) == 3
        assert "     1\ta" in lines[0]
        assert "     2\tb" in lines[1]
        assert "     3\tc" in lines[2]

    def test_trailing_newline_stripped(self):
        """Trailing empty line (final newline) is stripped."""
        result = format_with_line_numbers("a\n")
        lines = result.split("\n")
        assert len(lines) == 1
        assert "     1\ta" in lines[0]

    def test_start_line_offset(self):
        result = format_with_line_numbers("a\nb", start_line=100)
        assert "   100\ta" in result
        assert "   101\tb" in result

    def test_long_line_is_chunked(self):
        """Lines longer than MAX_LINE_LENGTH are split into sub-chunks."""
        long_line = "x" * 10001  # > 5000
        result = format_with_line_numbers(long_line)
        # Should have line number 1, then 1.1 as the chunk marker
        assert "     1\t" in result
        assert "   1.1\t" in result

    def test_empty_string(self):
        result = format_with_line_numbers("")
        assert result == ""


# ============================================================================
# format_tree_entries
# ============================================================================


class TestFormatTreeEntries:
    def test_empty(self):
        assert format_tree_entries([]) == "(empty)"

    def test_single_root(self):
        entries = [TreeEntry(name="root", depth=0)]
        result = format_tree_entries(entries)
        assert result == "root"

    def test_nested(self):
        entries = [
            TreeEntry(name="root", depth=0),
            TreeEntry(name="child1", depth=1),
            TreeEntry(name="child2", depth=1),
            TreeEntry(name="grandchild", depth=2),
        ]
        result = format_tree_entries(entries)
        lines = result.split("\n")
        assert lines[0] == "root"
        assert "child1" in lines[1]
        assert "child2" in lines[2]
        assert "grandchild" in lines[3]

    def test_connectors_are_correct(self):
        """Verify tree connectors are correct (├── for siblings, └── for last)."""
        entries = [
            TreeEntry(name="a.py", depth=1),
            TreeEntry(name="b.py", depth=1),
            TreeEntry(name="c.py", depth=1),
        ]
        result = format_tree_entries(entries)
        lines = result.split("\n")
        # First two items have more siblings → ├──
        assert "├── a.py" in lines[0]
        assert "├── b.py" in lines[1]
        # Last item → └──
        assert "└── c.py" in lines[2]

    # ------------------------------------------------------------------
    # Marker tests
    # ------------------------------------------------------------------

    def test_empty_marker(self):
        entries = [TreeEntry(name="emptydir/", depth=1, marker="empty")]
        result = format_tree_entries(entries)
        assert "emptydir/(empty)" in result

    def test_ignore_marker(self):
        entries = [TreeEntry(name="node_modules/", depth=1, marker="ignore")]
        result = format_tree_entries(entries)
        assert "node_modules/(ignore)" in result

    def test_depth_exceeded_marker(self):
        entries = [TreeEntry(name="deepdir/", depth=3, marker="depth_exceeded")]
        result = format_tree_entries(entries)
        assert "deepdir/(...)" in result

    def test_mixed_markers(self):
        entries = [
            TreeEntry(name="root/", depth=0),
            TreeEntry(name="emptydir/", depth=1, marker="empty"),
            TreeEntry(name="normal/", depth=1),
            TreeEntry(name="bigfile.txt (1 MB)", depth=2),
            TreeEntry(name="ignored/", depth=1, marker="ignore"),
            TreeEntry(name="deep/", depth=1, marker="depth_exceeded"),
        ]
        result = format_tree_entries(entries)
        assert "emptydir/(empty)" in result
        assert "ignored/(ignore)" in result
        assert "deep/(...)" in result
        assert "bigfile.txt" in result
        assert "normal/" in result

    def test_files_nested_under_directories(self):
        """Files at depth 2 must render under their parent dir at depth 1.

        Regression test for a bug where file depth was miscalculated as
        ``parent.count("/") + 1`` instead of using the full file path,
        causing files to render at the same level as their parent directory.
        """
        entries = [
            TreeEntry(name="workspace/", depth=0),
            TreeEntry(name="0849fe09/", depth=1),
            TreeEntry(name="Anima_00013_.png (1 MB)", depth=2),
            TreeEntry(name="Anima_00014_.png (1 MB)", depth=2),
        ]
        result = format_tree_entries(entries)
        lines = result.split("\n")

        dir_line = next(i for i, l in enumerate(lines) if "0849fe09/" in l)
        file_a_line = next(i for i, l in enumerate(lines) if "Anima_00013_" in l)
        file_b_line = next(i for i, l in enumerate(lines) if "Anima_00014_" in l)

        # Files must appear after their parent directory
        assert file_a_line > dir_line
        assert file_b_line > dir_line

        # Root line has no tree connector (depth 0)
        assert lines[0] == "workspace/"

        # Depth-2 file connectors must be further right than depth-1 dir connector
        def _cp(line: str) -> int:
            """Position of the tree connector (├── or └──) in *line*."""
            for c in ("├──", "└──"):
                if c in line:
                    return line.index(c)
            return -1

        dir_cp = _cp(lines[dir_line])
        file_a_cp = _cp(lines[file_a_line])
        file_b_cp = _cp(lines[file_b_line])

        assert file_a_cp > dir_cp, (
            f"Depth-2 file should indent more than depth-1 dir; "
            f"connector pos {file_a_cp} <= {dir_cp}.\n{result}"
        )
        assert file_b_cp > dir_cp, (
            f"Depth-2 file should indent more than depth-1 dir; "
            f"connector pos {file_b_cp} <= {dir_cp}.\n{result}"
        )
        # Siblings share same depth → same connector position
        assert file_a_cp == file_b_cp, (
            f"Sibling files should share depth; "
            f"connector pos {file_a_cp} != {file_b_cp}.\n{result}"
        )


# ============================================================================
# check_path_allowed
# ============================================================================


class TestCheckPathAllowed:
    def test_no_restrictions(self):
        assert check_path_allowed("/any/path.py") is True

    def test_whitelist_exact_match(self):
        assert check_path_allowed("/src/main.py", whitelist=frozenset({VirtualPath("/src/main.py")})) is True

    def test_whitelist_prefix_match(self):
        assert check_path_allowed("/src/sub/file.py", whitelist=frozenset({VirtualPath("/src")})) is True

    def test_whitelist_block(self):
        assert check_path_allowed("/build/output.o", whitelist=frozenset({VirtualPath("/src")})) is False

    def test_whitelist_root_match(self):
        assert check_path_allowed("/src", whitelist=frozenset({VirtualPath("/src")})) is True

    def test_blacklist_exact_match(self):
        assert check_path_allowed("/build/output.o", blacklist=frozenset({VirtualPath("/build/output.o")})) is False

    def test_blacklist_prefix_match(self):
        assert check_path_allowed("/build/output.o", blacklist=frozenset({VirtualPath("/build")})) is False

    def test_blacklist_allow(self):
        assert check_path_allowed("/src/main.py", blacklist=frozenset({VirtualPath("/build")})) is True

    def test_blacklist_partial_name_no_match(self):
        """Partial name match should NOT trigger (prefix-based)."""
        assert check_path_allowed("/build_scripts/run.sh", blacklist=frozenset({VirtualPath("/build")})) is True


# ============================================================================
# detect_trailing_newline_mismatch
# ============================================================================


class TestDetectTrailingNewlineMismatch:
    def test_no_mismatch__old_str_has_no_newline(self):
        """old_str without trailing newline → no mismatch to detect."""
        result = detect_trailing_newline_mismatch(
            "def foo():", "def foo():\n  pass"
        )
        assert result is None

    def test_no_mismatch__content_has_newline(self):
        """content ends with newline, old_str matches exactly."""
        result = detect_trailing_newline_mismatch(
            "def foo():\n", "def foo():\n"
        )
        assert result is None

    def test_no_mismatch__neither_ends_with_newline(self):
        """Neither old_str nor content ends with \n."""
        result = detect_trailing_newline_mismatch(
            "def foo():", "def foo():"
        )
        assert result is None

    def test_mismatch_detected_single_occurrence(self):
        """old_str ends with \n but content does not — mismatch detected (1 match)."""
        result = detect_trailing_newline_mismatch(
            "def foo():\n", "def foo():"
        )
        assert result is not None
        assert "换行" in str(result.error).lower()

    def test_mismatch_detected_multiple_occurrences(self):
        """old_str ends with \n but content does not — mismatch detected (3 matches)."""
        result = detect_trailing_newline_mismatch(
            "foo\n", "foo bar foo baz foo"
        )
        assert result is not None
        assert result.occurrences == 0  # error result
        assert "处" in str(result.error)

    def test_single_char_old_str_skipped(self):
        """old_str of length 1 with newline (e.g. just '\\n') should not crash."""
        # This falls through the len(old_str) > 1 check
        result = detect_trailing_newline_mismatch("\n", "x")
        assert result is None
