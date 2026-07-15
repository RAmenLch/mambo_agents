"""Shared middleware utilities."""

from mambo_agents.middleware.utils._tokens import (
    _build_default_token_counter,
    _CJK_BLOCKS,
    _DEFAULT_CJK_CHARS_PER_TOKEN,
    _DEFAULT_EN_CHARS_PER_TOKEN,
    _MAX_RATIO_SCAN_CHARS,
    _detect_cjk_ratio,
    _is_cjk_char,
)

__all__ = [
    "_CJK_BLOCKS",
    "_DEFAULT_CJK_CHARS_PER_TOKEN",
    "_DEFAULT_EN_CHARS_PER_TOKEN",
    "_MAX_RATIO_SCAN_CHARS",
    "_build_default_token_counter",
    "_detect_cjk_ratio",
    "_is_cjk_char",
]
