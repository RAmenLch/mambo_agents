"""Shared helpers used across multiple backend implementations.

Constants, formatters, and common edit logic that would otherwise be
duplicated between ``LocalBackend``, ``StateBackend``, and protocol.
"""

from __future__ import annotations

from mambo_agents.backends.protocol import EditResult

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


def human_size(size: int) -> str:
    """Format *size* as a human-readable string (e.g. ``"1.2 MB"``)."""
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size} {unit}"
        size //= 1024
    return f"{size} TB"


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
# Edit helpers
# ---------------------------------------------------------------------------


def detect_trailing_newline_mismatch(
    file_path: str,
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
            error=(
                "old_str ends with a newline, but the file does not "
                "end with a newline. Retry with the trailing newline "
                "removed from old_str (and from new_str if it also "
                "ends with a newline)."
            ),
        )
    return EditResult(
        error=(
            f"old_str ends with a newline, but the file does not "
            f"end with a newline. With the trailing newline removed, "
            f"old_str would appear {stripped_count} times. "
            f"Retry with the trailing newline removed and add "
            f"surrounding context so the match is unique."
        ),
    )
