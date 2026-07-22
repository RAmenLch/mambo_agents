"""Shared helpers used across multiple backend implementations."""

from mambo_agents.backends.utils._format import (
    LINE_NUMBER_WIDTH,
    MAX_LINE_LENGTH,
    format_validation_error,
    format_with_line_numbers,
)
from mambo_agents.backends.utils._tree import TreeEntry, format_tree_entries
from mambo_agents.backends.utils._glob import fnmatch_path, translate_glob
from mambo_agents.backends.utils._edit import (
    check_path_allowed,
    detect_trailing_newline_mismatch,
    normalize_line_endings,
)

__all__ = [
    "LINE_NUMBER_WIDTH",
    "MAX_LINE_LENGTH",
    "TreeEntry",
    "check_path_allowed",
    "detect_trailing_newline_mismatch",
    "normalize_line_endings",
    "fnmatch_path",
    "format_tree_entries",
    "format_validation_error",
    "format_with_line_numbers",
    "translate_glob",
]
