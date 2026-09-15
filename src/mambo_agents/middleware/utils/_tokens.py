"""CJK-aware token counter construction for summarization."""

from __future__ import annotations

from collections.abc import Iterable
from functools import partial

from langchain.agents.middleware.summarization import TokenCounter
from langchain_core.messages import BaseMessage
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

# ``additional_kwargs`` keys whose text some reasoning-capable adapters
# (e.g. DeepSeek/GLM) re-send to the provider on every turn.  When present
# they must be folded into the estimate, otherwise the trigger threshold
# underestimates the real prompt.
_REASONING_FIELDS: tuple[str, ...] = ("reasoning_content", "reasoning")


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


def _extract_reasoning_text(msg: BaseMessage) -> str:
    """Collect re-sent reasoning text from ``msg.additional_kwargs``.

    Returns the concatenation of every non-empty string stored under
    :data:`_REASONING_FIELDS` (``reasoning_content`` / ``reasoning``), or an
    empty string when the message carries none.
    """
    parts = []
    for key in _REASONING_FIELDS:
        value = msg.additional_kwargs.get(key)
        if isinstance(value, str) and value:
            parts.append(value)
    return "\n".join(parts)


def _with_reasoning_in_content(messages: Iterable) -> list[BaseMessage]:
    """Fold reasoning text into ``content`` so the estimate matches the prompt.

    ``count_tokens_approximately`` only reads ``message.content`` (plus tool
    calls / role / name), so reasoning stored under ``additional_kwargs`` is
    invisible to it even though adapters such as DeepSeek re-send it.  This
    returns copies of *messages* whose reasoning text is merged into
    ``content`` (string content is appended to; list content gains a text
    block).  Messages without reasoning are passed through unchanged.
    """
    augmented: list[BaseMessage] = []
    for msg in convert_to_messages(messages):
        reasoning = _extract_reasoning_text(msg)
        if not reasoning:
            augmented.append(msg)
            continue
        content = msg.content
        if isinstance(content, str):
            merged: str | list = f"{content}\n{reasoning}" if content else reasoning
        elif isinstance(content, list):
            merged = [*content, {"type": "text", "text": reasoning}]
        else:
            merged = reasoning
        augmented.append(msg.model_copy(update={"content": merged}))
    return augmented


def _build_default_token_counter(
    chars_per_token: float | None = None,
    *,
    include_reasoning: bool = False,
) -> TokenCounter:
    """Build a model-agnostic token counter.

    Unlike langchain's ``_get_approximate_token_counter``, this builder:

    - Does **not** inspect the model name — avoids fragile heuristics.
    - When ``chars_per_token`` is specified, uses that value directly.
    - When ``chars_per_token`` is ``None``, auto-detects the CJK ratio
      from message content and blends ``_DEFAULT_EN_CHARS_PER_TOKEN`` with
      ``_DEFAULT_CJK_CHARS_PER_TOKEN`` accordingly.
    - When ``include_reasoning`` is ``True``, folds the reasoning text
      (``additional_kwargs["reasoning_content"]`` / ``["reasoning"]``) into
      the estimate so it tracks the real prompt for models that re-send the
      chain-of-thought (e.g. DeepSeek).  Leave disabled for models whose
      reasoning is not re-sent, to avoid over-counting.

    Args:
        chars_per_token: Explicit characters-per-token ratio.
            ``None`` means auto-detect from content.
        include_reasoning: Fold ``additional_kwargs`` reasoning text into the
            estimate.  Defaults to ``False`` (reasoning excluded).

    Returns:
        A ``TokenCounter`` callable suitable for passing to
        ``LCSummarizationMiddleware``.
    """
    if chars_per_token is not None:
        cpt = float(chars_per_token)
        if not include_reasoning:
            return partial(
                count_tokens_approximately,
                chars_per_token=cpt,
                use_usage_metadata_scaling=True,
            )

        def _fixed(token_iterable) -> int:
            return count_tokens_approximately(
                _with_reasoning_in_content(token_iterable),
                chars_per_token=cpt,
                use_usage_metadata_scaling=True,
            )

        return _fixed

    def _auto(token_iterable) -> int:
        messages = list(token_iterable)
        if include_reasoning:
            messages = _with_reasoning_in_content(messages)
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
