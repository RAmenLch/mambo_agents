"""Composite summarizer — chains multiple ``ReadSummarizer`` callbacks together.

Each summarizer is tried in order.  The first one that produces a non-fallback
result wins.  If none match, the final fallback message is returned.
"""

from __future__ import annotations

from mambo_agents.backends.protocol import ReadSummarizer


def composite_summarizer(
    summarizers: list[ReadSummarizer],
    *,
    fallback: ReadSummarizer | None = None,
) -> ReadSummarizer:
    """Return a ``ReadSummarizer`` that tries multiple summarizers in sequence.

    Each summarizer in *summarizers* is called with the same ``(file_path,
    content, max_chars)`` arguments.  Because individual summarizers already
    check the file extension internally, only the matching one(s) will
    produce a structured summary — all others return a fallback message
    (``"[返回结果过大…]"``).

    The composite picks the **first** result that does **not** look like a
    fallback message (i.e. a result that starts with ``"[返回结果过大"``
    is considered a fallback).  If all summarizers fall back, *fallback* is
    used; if *fallback* is ``None``, the last summarizer's result is used.

    Usage::

        from mambo_agents.read_summarizers import (
            composite_summarizer,
            python_summarizer,
            java_summarizer,
        )

        multi = composite_summarizer([
            python_summarizer(),
            java_summarizer(),
        ])

        backend = LocalBackend(summarizer=multi)
    """

    def _summarize(file_path: str, content: str, max_chars: int) -> str:
        for s in summarizers:
            result = s(file_path, content, max_chars)
            # Individual summarizers return fallback messages starting with
            # "[返回结果过大…]" when the file extension doesn't match.
            if not result.startswith("[返回结果过大"):
                return result

        # All summarizers fell back — use the explicit fallback or the
        # last summarizer's result (which is a generic fallback message).
        if fallback is not None:
            return fallback(file_path, content, max_chars)

        # Return the last summarizer's result (fallback message) if no
        # explicit fallback was provided and summarizers list is non-empty.
        if summarizers:
            return summarizers[-1](file_path, content, max_chars)

        # Should never reach here if summarizers is non-empty.
        total_lines = content.count("\n") + 1
        return (
            f"[返回结果过大（{len(content):,} 字符，{total_lines:,} 行），"
            f"超过读取上限 {max_chars:,} 字符。"
            f"请重新指定 offset + limit 参数后读取。"
            f"示例: read(file_path='{file_path}', offset=0, limit=500)]"
        )

    return _summarize
