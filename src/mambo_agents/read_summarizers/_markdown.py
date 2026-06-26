"""Markdown file summarizer — extracts heading outline."""

from __future__ import annotations

import re
from pathlib import PurePosixPath

from mambo_agents.backends.protocol import ReadSummarizer
from mambo_agents.backends.schemas import VirtualPath

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_LINE_LIMIT = 200

_MD_SUFFIXES = frozenset({".md", ".mdx", ".markdown", ".mdown"})


def markdown_summarizer() -> ReadSummarizer:
    """Return a ``ReadSummarizer`` that analyses Markdown files.

    Extracts headings (``#`` … ``######``) with accurate line numbers
    so the LLM can navigate to specific sections.
    """

    def _summarize(file_path: VirtualPath, content: str, max_chars: int) -> str:
        suffix = PurePosixPath(file_path.value).suffix.lower()
        if suffix not in _MD_SUFFIXES:
            return _fallback(file_path, content, max_chars)
        return _summarize_md(file_path, content, max_chars)

    return _summarize


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fallback(file_path: VirtualPath, content: str, max_chars: int) -> str:
    total_lines = content.count("\n") + 1
    return (
        f"[返回结果过大（{len(content):,} 字符，{total_lines:,} 行），"
        f"超过读取上限 {max_chars:,} 字符。"
        f"请重新指定 offset + limit 参数后读取。"
        f"示例: read(file_path='{file_path}', offset=0, limit=500)]"
    )


def _summarize_md(file_path: VirtualPath, content: str, max_chars: int) -> str:
    total_lines = content.count("\n") + 1
    lines_of_content = content.split("\n")

    headings: list[tuple[int, int, str]] = []  # (level, lineno, text)
    for match in _HEADING_RE.finditer(content):
        level = len(match.group(1))
        text = match.group(2).strip()
        lineno = content[: match.start()].count("\n") + 1
        headings.append((level, lineno, text))

    outline_lines: list[str] = [
        f"[Markdown 文件过大，已生成结构大纲"
        f"（完整文件 {len(content):,} 字符，{total_lines:,} 行）]",
        "",
    ]

    if not headings:
        outline_lines.append(
            f">>> No headings found. Use read(file_path='{file_path}', offset=0, limit=100) "
            f"to read in sections"
        )
        return "\n".join(outline_lines)

    outline_lines.append(f"[Heading Outline] ({len(headings)}):")
    for level, lineno, text in headings[:_LINE_LIMIT]:
        indent = "  " * (level - 1)
        prefix = "#" * level
        teaser = text[:80] + "…" if len(text) > 80 else text
        outline_lines.append(f"  L{lineno:<5} {indent}{prefix} {teaser}")

    if len(headings) > _LINE_LIMIT:
        outline_lines.append(f"  ... {len(headings) - _LINE_LIMIT} more headings")

    outline_lines.append("")
    outline_lines.append(
        f">>> Use read(file_path='{file_path}', offset=<lineno>, limit=100) "
        f"to read sections of interest"
    )
    return "\n".join(outline_lines)
