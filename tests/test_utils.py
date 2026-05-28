"""Tests for shared backend utilities in ``mambo_agents.backends.utils``."""

import pytest

from mambo_agents.backends.protocol import EditResult
from mambo_agents.backends.utils import (
    detect_trailing_newline_mismatch,
    format_tree_entries,
    format_with_line_numbers,
    human_size,
)


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
        entries = [("root", 0)]
        result = format_tree_entries(entries)
        assert result == "root"

    def test_nested(self):
        entries = [
            ("root", 0),
            ("child1", 1),
            ("child2", 1),
            ("grandchild", 2),
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
            ("a.py", 1),
            ("b.py", 1),
            ("c.py", 1),
        ]
        result = format_tree_entries(entries)
        lines = result.split("\n")
        # First two items have more siblings → ├──
        assert "├── a.py" in lines[0]
        assert "├── b.py" in lines[1]
        # Last item → └──
        assert "└── c.py" in lines[2]


# ============================================================================
# detect_trailing_newline_mismatch
# ============================================================================


class TestDetectTrailingNewlineMismatch:
    def test_no_mismatch__old_str_has_no_newline(self):
        """old_str without trailing newline → no mismatch to detect."""
        result = detect_trailing_newline_mismatch(
            "file.py", "def foo():", "def foo():\n  pass"
        )
        assert result is None

    def test_no_mismatch__content_has_newline(self):
        """content ends with newline, old_str matches exactly."""
        result = detect_trailing_newline_mismatch(
            "file.py", "def foo():\n", "def foo():\n"
        )
        assert result is None

    def test_no_mismatch__neither_ends_with_newline(self):
        """Neither old_str nor content ends with \n."""
        result = detect_trailing_newline_mismatch(
            "file.py", "def foo():", "def foo():"
        )
        assert result is None

    def test_mismatch_detected_single_occurrence(self):
        """old_str ends with \n but content does not — mismatch detected (1 match)."""
        result = detect_trailing_newline_mismatch(
            "file.py", "def foo():\n", "def foo():"
        )
        assert result is not None
        assert "newline" in (result.error or "").lower()

    def test_mismatch_detected_multiple_occurrences(self):
        """old_str ends with \n but content does not — mismatch detected (3 matches)."""
        result = detect_trailing_newline_mismatch(
            "file.py", "foo\n", "foo bar foo baz foo"
        )
        assert result is not None
        assert result.occurrences == 0  # error result
        assert "3 times" in (result.error or "")

    def test_single_char_old_str_skipped(self):
        """old_str of length 1 with newline (e.g. just '\\n') should not crash."""
        # This falls through the len(old_str) > 1 check
        result = detect_trailing_newline_mismatch("file.py", "\n", "x")
        assert result is None
