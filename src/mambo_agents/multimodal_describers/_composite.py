"""Composite multimodal describer — tries multiple describers in order.

Useful when you have several multimodal models and want a resilient fallback
(e.g. a cheaper model first, then a stronger one, then a describe-failed
message).  Mirrors :func:`mambo_agents.read_summarizers.composite_summarizer`.

A describer returning ``None`` opts out (did not handle this file type) and
the composite moves on to the next one.
"""

from __future__ import annotations

from mambo_agents.backends.protocol import MultimodalDescriber
from mambo_agents.backends.schemas import VirtualPath
from mambo_agents.multimodal_describers._base import DESCRIBE_FAILED


def composite_multimodal_describer(
    describers: list[MultimodalDescriber],
    *,
    fallback: str = DESCRIBE_FAILED,
) -> MultimodalDescriber:
    """Return a :class:`MultimodalDescriber` that tries *describers* in order.

    Each describer is called with the same ``(file_path, base64_content,
    mime_type)`` arguments.  The first describer that does not raise, does
    not return ``None``, and yields a non-empty, non-fallback description
    wins.

    If **every** describer returns ``None`` (all opted out — "not my type"),
    the composite returns ``None`` as well so the caller passes the
    multimodal result through unchanged.  ``fallback`` is only returned when
    at least one describer actually attempted to describe the file but
    failed (raised, or yielded an empty / ``DESCRIBE_FAILED`` result).
    """

    def _describe(
        file_path: VirtualPath, base64_content: str, mime_type: str,
    ) -> str | None:
        attempted = False
        for describer in describers:
            try:
                desc = describer(file_path, base64_content, mime_type)
            except Exception:
                attempted = True
                continue
            if desc is None:
                continue
            desc = desc.strip()
            if desc and desc != DESCRIBE_FAILED:
                return desc
            attempted = True
        return fallback if attempted else None

    return _describe
