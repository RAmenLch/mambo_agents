"""Pluggable ``MultimodalDescriber`` callbacks that give text-only models media understanding.

Each factory returns a ``MultimodalDescriber`` callback that receives
``(file_path, base64_content, mime_type)`` and returns either a text to
substitute for a multimodal content block, or ``None`` to leave it unchanged.
Configure one on a backend via the ``multimodal_describer`` constructor
argument (mirroring ``summarizer``).

- :func:`multimodal_describer` — describes images / video / audio / documents
  via a multimodal chat model.
- :func:`image_describer` / :func:`video_describer` / :func:`audio_describer` /
  :func:`document_describer` — per-type convenience factories (other types
  pass through).
- :func:`reject_multimodal_describer` — returns an error text for non-text
  files, with an ``allow`` set of block types that pass through instead; for
  models that neither read media nor want a multimodal block to reach their
  API (or want to reject only some media types).
- :func:`composite_multimodal_describer` — tries several describers in order.

Usage::

    from mambo_agents.multimodal_describers import multimodal_describer
    from mambo_agents.backends.local import LocalBackend
    from langchain_openai import ChatOpenAI

    vision = ChatOpenAI(model="gpt-4o")
    backend = LocalBackend(multimodal_describer=multimodal_describer(vision))

None of these are injected by default — the user selects which to use.
"""

from mambo_agents.multimodal_describers._base import (
    DESCRIBE_FAILED,
    LC_SOURCE,
    audio_describer,
    document_describer,
    image_describer,
    multimodal_describer,
    reject_multimodal_describer,
    video_describer,
)
from mambo_agents.multimodal_describers._composite import composite_multimodal_describer

__all__ = [
    "DESCRIBE_FAILED",
    "LC_SOURCE",
    "multimodal_describer",
    "image_describer",
    "video_describer",
    "audio_describer",
    "document_describer",
    "reject_multimodal_describer",
    "composite_multimodal_describer",
]
