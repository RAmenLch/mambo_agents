"""Shared helpers used across multiple backend implementations.

Constants, formatters, and common edit logic that would otherwise be
duplicated between ``LocalBackend``, ``StoreBackend``, and protocol.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ValidationError

from mambo_agents.backends.schemas import BackendError, EditResult, ErrorCode, VirtualPath

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LINE_NUMBER_WIDTH = 6
"""Width (characters) of the line-number column in ``read()`` output."""

MAX_LINE_LENGTH = 5000
"""Maximum characters per line before splitting into numbered chunks."""


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def format_validation_error(e: ValidationError) -> str:
    """Format a Pydantic ``ValidationError`` into a readable message for the LLM.

    Produces a bullet list of ``field: reason`` entries so the AI can
    understand exactly which argument was rejected and why, then retry
    with a corrected value.
    """
    errors = e.errors()
    lines = ["Validation error(s):"]
    for err in errors:
        loc = " -> ".join(str(p) for p in err.get("loc", ()))
        msg = err.get("msg", "unknown error")
        lines.append(f"  • {loc}: {msg}")
    return "\n".join(lines)


def format_with_line_numbers(content: str, start_line: int = 1) -> str:
    """Format content with line numbers (``cat -n`` style).

    Lines longer than :data:`MAX_LINE_LENGTH` are split into numbered
    sub-chunks (e.g. ``42.1``, ``42.2``).
    """
    lines = content.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]

    result: list[str] = []
    for i, line in enumerate(lines):
        num = i + start_line
        if len(line) <= MAX_LINE_LENGTH:
            result.append(f"{num:{LINE_NUMBER_WIDTH}d}\t{line}")
        else:
            chunk_count = (len(line) + MAX_LINE_LENGTH - 1) // MAX_LINE_LENGTH
            for ci in range(chunk_count):
                chunk = line[ci * MAX_LINE_LENGTH:(ci + 1) * MAX_LINE_LENGTH]
                marker = f"{num}.{ci}" if ci > 0 else str(num)
                result.append(f"{marker:>{LINE_NUMBER_WIDTH}}\t{chunk}")
    return "\n".join(result)


# ---------------------------------------------------------------------------
# Tree display types
# ---------------------------------------------------------------------------


class TreeEntry(BaseModel):
    """A single entry in a directory tree display.

    Attributes:
        name: Display name (e.g. ``"subdir/"``, ``"file.txt (1 KB)"``).
        depth: Nesting depth (0 = root level).
        marker: Optional status marker for directories.
            ``""`` = normal, ``"empty"`` → ``/(empty)``,
            ``"ignore"`` → ``/(ignore)``,
            ``"depth_exceeded"`` → ``/(...)``.
    """

    name: str
    depth: int
    marker: Literal["", "empty", "ignore", "depth_exceeded"] = ""


# ---------------------------------------------------------------------------
# Tree formatter
# ---------------------------------------------------------------------------


def format_tree_entries(
    entries: list[TreeEntry],
) -> str:
    """Render ``TreeEntry`` list as a visual directory tree.

    The tree-walk logic is backend-specific; this function provides the
    shared visual formatting regardless of which backend produces the entries.

    Directory markers are appended to the display name:
    - ``/(empty)`` — directory has no children
    - ``/(ignore)`` — directory in ignore_dirs (children hidden)
    - ``/(...)`` — depth limit reached, children not shown
    """
    if not entries:
        return "(empty)"

    _MARKER_SUFFIX: dict[str, str] = {
        "empty": "/(empty)",
        "ignore": "/(ignore)",
        "depth_exceeded": "/(...)",
    }

    lines: list[str] = []
    for i, entry in enumerate(entries):
        marker_suffix = _MARKER_SUFFIX.get(entry.marker, "")
        if marker_suffix:
            display = entry.name.rstrip("/") + marker_suffix
        else:
            display = entry.name
        depth = entry.depth

        # Determine connector: look ahead to see if there are siblings at the same depth
        has_more_siblings = False
        for j in range(i + 1, len(entries)):
            next_depth = entries[j].depth
            if next_depth < depth:
                break
            if next_depth == depth:
                has_more_siblings = True
                break

        connector = "├── " if has_more_siblings else "└── "
        if depth == 0:
            lines.append(display)
        else:
            prefix_parts: list[str] = []
            for level in range(1, depth):  # parent levels only; own level uses connector
                active = False
                for j in range(i + 1, len(entries)):
                    if entries[j].depth < level:
                        break
                    if entries[j].depth == level:
                        active = True
                        break
                prefix_parts.append("│   " if active else "    ")
            prefix = "".join(prefix_parts)
            lines.append(f"{prefix}{connector}{display}")

    return "\n".join(lines)


def check_path_allowed(
    path: str,
    *,
    whitelist: frozenset[VirtualPath] | None = None,
    blacklist: frozenset[VirtualPath] | None = None,
) -> bool:
    """Check whether *path* (virtual) is allowed for edit/write/delete.

    When *whitelist* is set, *path* must start with (or equal) one of
    its entries.  When *blacklist* is set, *path* must NOT start with
    (or equal) any entry.  The two are mutually exclusive and the caller
    must enforce that.

    Prefixes are normalised (trailing slash stripped) so both
    ``VirtualPath("/src")`` and ``VirtualPath("/src/")`` work identically.

    Args:
        path: Virtual absolute path to check (e.g. ``"/src/foo.py"``).
        whitelist: Allowed path prefixes (e.g. ``{VirtualPath("/src")}``).
        blacklist: Forbidden path prefixes (e.g. ``{VirtualPath("/build")}``).

    Returns:
        ``True`` if the path is permitted.
    """
    if whitelist is not None:
        return any(
            path == prefix.normalized or path.startswith(prefix.normalized + "/")
            for prefix in whitelist
        )
    if blacklist is not None:
        return not any(
            path == prefix.normalized or path.startswith(prefix.normalized + "/")
            for prefix in blacklist
        )
    return True


def detect_trailing_newline_mismatch(
    old_str: str,
    existing_content: str,
) -> EditResult | None:
    """Check whether *old_str* failed because of a trailing newline mismatch.

    LLMs often append ``\\n`` to *old_str* even when the file does not
    end with a newline.  This helper detects that case and returns a
    descriptive error so the model can retry.

    Returns ``None`` when no trailing-newline mismatch is detected (the
    caller should fall through to the generic "not found" error).
    """
    if not (
        old_str.endswith("\n")
        and len(old_str) > 1
        and existing_content.endswith(old_str.removesuffix("\n"))
    ):
        return None

    stripped = old_str.removesuffix("\n")
    stripped_count = existing_content.count(stripped)
    if stripped_count == 1:
        return EditResult(
            error=BackendError(
                code=ErrorCode.OLD_STR_NOT_FOUND,
                message="old_str 以换行符结尾，但文件不以换行符结尾。请去掉 old_str 末尾的换行符后重试",
            ),
        )
    return EditResult(
        error=BackendError(
            code=ErrorCode.MULTI_OCCURRENCES,
            message=f"old_str 以换行符结尾，去掉后匹配到 {stripped_count} 处。请去掉末尾换行符并增加上下文使匹配唯一",
        ),
    )
