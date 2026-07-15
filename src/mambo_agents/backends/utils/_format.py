"""Formatters: line numbers, validation errors."""

from __future__ import annotations

from pydantic import ValidationError

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
