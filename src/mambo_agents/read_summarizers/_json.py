"""JSON file summarizer — extracts top-level key structure."""

from __future__ import annotations

import json
from pathlib import PurePosixPath

from mambo_agents.backends.protocol import ReadSummarizer

_LINE_LIMIT = 200


def json_summarizer() -> ReadSummarizer:
    """Return a ``ReadSummarizer`` that analyses JSON files.

    Parses the JSON and reports top-level key structure (for objects)
    or item count with type hints (for arrays).
    """

    def _summarize(file_path: str, content: str, max_chars: int) -> str:
        suffix = PurePosixPath(file_path).suffix.lower()
        if suffix != ".json":
            return _fallback(file_path, content, max_chars)
        return _summarize_json(file_path, content, max_chars)

    return _summarize


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fallback(file_path: str, content: str, max_chars: int) -> str:
    total_lines = content.count("\n") + 1
    return (
        f"[返回结果过大（{len(content):,} 字符，{total_lines:,} 行），"
        f"超过读取上限 {max_chars:,} 字符。"
        f"请重新指定 offset + limit 参数后读取。"
        f"示例: read(file_path='{file_path}', offset=0, limit=500)]"
    )


def _summarize_json(file_path: str, content: str, max_chars: int) -> str:
    total_lines = content.count("\n") + 1
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return (
            f"[JSON 文件过大（{len(content):,} 字符，{total_lines:,} 行），"
            f"超过读取上限 {max_chars:,} 字符。"
            f"因 JSON 解析失败无法生成结构化摘要。"
            f"请重新指定 offset + limit 参数后读取。"
            f"示例: read(file_path='{file_path}', offset=0, limit=500)]"
        )

    lines: list[str] = [
        f"[JSON 文件过大，已生成结构摘要"
        f"（完整文件 {len(content):,} 字符，{total_lines:,} 行）]",
        "",
    ]

    if isinstance(data, dict):
        # Find approximate line numbers for top-level keys
        key_lines = _locate_keys(content, list(data.keys()))
        lines.append(f"[Root Object] — {len(data)} top-level keys:")
        for key, value in list(data.items())[:_LINE_LIMIT]:
            line_hint = f"L{key_lines.get(key, '?'):<5}" if key in key_lines else "       "
            val_type = _describe_value(value)
            teaser = _tease_value(value)
            line_val = f"  {line_hint} {key}: {val_type}"
            if teaser:
                line_val += f"  ({teaser})"
            lines.append(line_val)
        if len(data) > _LINE_LIMIT:
            lines.append(f"  ... {len(data) - _LINE_LIMIT} more keys")

    elif isinstance(data, list):
        total = len(data)
        lines.append(f"[Array] — {total} items")
        if total > 0:
            first_type = type(data[0]).__name__
            lines.append(f"  item type: {first_type}")
        if total <= 5:
            for i, item in enumerate(data):
                lines.append(f"  [{i}] {_describe_value(item)}  {_tease_value(item)}")
        else:
            for i in range(3):
                lines.append(f"  [{i}] {_describe_value(data[i])}  {_tease_value(data[i])}")
            lines.append(f"  ...")
            for i in range(max(total - 2, 3), total):
                lines.append(f"  [{i}] {_describe_value(data[i])}  {_tease_value(data[i])}")
    else:
        lines.append(f"[Scalar] {type(data).__name__}: {_tease_value(data)[:120]}")

    lines.append("")
    lines.append(
        f">>> Use read(file_path='{file_path}', offset=0, limit=100) "
        f"to read full content in sections"
    )
    return "\n".join(lines)


def _locate_keys(content: str, keys: list[str]) -> dict[str, int]:
    """Approximate line numbers for JSON top-level keys."""
    result: dict[str, int] = {}
    lines = content.split("\n")
    for i, line in enumerate(lines, 1):
        for key in keys:
            if key not in result and f'"{key}"' in line:
                result[key] = i
    return result


def _describe_value(value: object) -> str:
    if isinstance(value, dict):
        return f"object ({len(value)} keys)"
    if isinstance(value, list):
        return f"array ({len(value)} items)"
    if isinstance(value, str):
        return "string"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if value is None:
        return "null"
    return type(value).__name__


def _tease_value(value: object) -> str:
    if isinstance(value, str):
        return value[:60] + "…" if len(value) > 60 else value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if value is None:
        return "null"
    return ""
