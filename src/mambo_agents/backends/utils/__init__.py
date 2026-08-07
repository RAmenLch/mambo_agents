"""Shared helpers used across multiple backend implementations."""

from mambo_agents.backends.utils._decode import decode_output
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
from mambo_agents.backends.utils.multimodal import (
    EXTENSION_TO_FILE_TYPE,
    get_file_type,
    get_mime_type,
    validate_multimodal_content,
)

__all__ = [
    "EXTENSION_TO_FILE_TYPE",
    "LINE_NUMBER_WIDTH",
    "MAX_LINE_LENGTH",
    "TreeEntry",
    "decode_output",
    "check_path_allowed",
    "detect_trailing_newline_mismatch",
    "normalize_line_endings",
    "fnmatch_path",
    "format_tree_entries",
    "format_validation_error",
    "format_with_line_numbers",
    "get_file_type",
    "get_mime_type",
    "translate_glob",
    "validate_multimodal_content",
]
