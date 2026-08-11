"""Byte-stream decoding helpers for shell-command / SSH-channel output."""

from __future__ import annotations

import locale


def decode_output(data: bytes) -> str:
    """Decode raw bytes from a subprocess or SSH channel into text.

    Strategy: modern CLIs (git, python, ripgrep, ...) emit UTF-8 by default,
    and UTF-8 strict decoding rejects most non-UTF-8 byte streams — so UTF-8
    is tried first.  Fall back to the system locale encoding (e.g. cp936 on
    Chinese Windows), then to UTF-8 with replacement as a last resort.

    Note: decoding with cp936/GBK first is unsafe — GBK is lenient enough to
    "successfully" decode most UTF-8 byte streams into mojibake without
    raising, so a UTF-8 fallback would never trigger.

    Also normalizes ``\\r\\n`` to ``\\n`` so Windows child processes don't
    produce stray carriage returns in the output.
    """
    if not data:
        return ""
    try:
        return data.decode("utf-8").replace("\r\n", "\n")
    except UnicodeDecodeError:
        pass
    try:
        return data.decode(locale.getpreferredencoding(False)).replace("\r\n", "\n")
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="replace").replace("\r\n", "\n")
