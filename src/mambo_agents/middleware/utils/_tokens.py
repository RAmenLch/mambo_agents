"""CJK-aware token counter construction for summarization."""

from __future__ import annotations

from collections.abc import Iterable
from functools import partial

from langchain.agents.middleware.summarization import TokenCounter
from langchain_core.messages.utils import (
    convert_to_messages,
    count_tokens_approximately,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default chars-per-token ratios for language-aware token estimation.
# These control how `count_tokens_approximately` converts character counts
# into approximate token counts.  English defaults to ~4 chars/token;
# CJK text (Chinese / Japanese / Korean) is denser — each character is
# typically 1–2 tokens in modern subword tokenizers.
_DEFAULT_EN_CHARS_PER_TOKEN: float = 4.0
# Conservative estimate for CJK text — most modern tokenizers encode
# ~1.5–2 CJK characters per token (e.g. o200k_base ≈ 1.8, cl100k ≈ 1.5).
_DEFAULT_CJK_CHARS_PER_TOKEN: float = 1.8

# Unicode blocks treated as "CJK" for ratio estimation.
_CJK_BLOCKS: list[tuple[int, int]] = [
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
    (0x3000, 0x303F),  # CJK Symbols & Punctuation
    (0xFF00, 0xFFEF),  # Halfwidth & Fullwidth Forms
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
    (0x2F800, 0x2FA1F),  # CJK Compatibility Ideographs Supplement
]

# Cap on characters scanned for language-ratio estimation so we don't
# linearly scan an entire massive conversation on every token-count call.
_MAX_RATIO_SCAN_CHARS: int = 50_000


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def _detect_cjk_ratio(messages: Iterable) -> float:
    """Estimate what fraction of the message content is CJK.

    Scans the first ``_MAX_RATIO_SCAN_CHARS`` characters across all messages
    and returns the ratio of CJK characters to total characters.

    Returns:
        A float in ``[0.0, 1.0]``.  Returns ``0.0`` if no characters were scanned
        (empty messages).
    """
    total = 0
    cjk = 0

    for msg in convert_to_messages(messages):
        content = msg.content
        if isinstance(content, str):
            for ch in content:
                if total >= _MAX_RATIO_SCAN_CHARS:
                    break
                total += 1
                if _is_cjk_char(ch):
                    cjk += 1
        elif isinstance(content, list):
            for block in content:
                text = block.get("text", "") if isinstance(block, dict) else ""
                for ch in text:
                    if total >= _MAX_RATIO_SCAN_CHARS:
                        break
                    total += 1
                    if _is_cjk_char(ch):
                        cjk += 1
                if total >= _MAX_RATIO_SCAN_CHARS:
                    break
        if total >= _MAX_RATIO_SCAN_CHARS:
            break

    return cjk / total if total > 0 else 0.0


def _is_cjk_char(ch: str) -> bool:
    """Return ``True`` if *ch* (single character) is in a CJK Unicode block."""
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _CJK_BLOCKS)


def _build_default_token_counter(
    chars_per_token: float | None = None,
) -> TokenCounter:
    """Build a model-agnostic token counter.

    Unlike langchain's ``_get_approximate_token_counter``, this builder:

    - Does **not** inspect the model name — avoids fragile heuristics.
    - When ``chars_per_token`` is specified, uses that value directly.
    - When ``chars_per_token`` is ``None``, auto-detects the CJK ratio
      from message content and blends ``_DEFAULT_EN_CHARS_PER_TOKEN`` with
      ``_DEFAULT_CJK_CHARS_PER_TOKEN`` accordingly.

    Args:
        chars_per_token: Explicit characters-per-token ratio.
            ``None`` means auto-detect from content.

    Returns:
        A ``TokenCounter`` callable suitable for passing to
        ``LCSummarizationMiddleware``.
    """
    if chars_per_token is not None:
        return partial(
            count_tokens_approximately,
            chars_per_token=float(chars_per_token),
            use_usage_metadata_scaling=True,
        )

    def _auto(token_iterable) -> int:
        messages = list(token_iterable)
        cjk_ratio = _detect_cjk_ratio(messages)
        effective_cpt = (
            _DEFAULT_EN_CHARS_PER_TOKEN * (1.0 - cjk_ratio)
            + _DEFAULT_CJK_CHARS_PER_TOKEN * cjk_ratio
        )
        return count_tokens_approximately(
            messages,
            chars_per_token=effective_cpt,
            use_usage_metadata_scaling=True,
        )

    return _auto
