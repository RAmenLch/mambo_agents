"""Runtime compatibility patches applied when ``mambo_agents`` is imported.

LangGraph's ``ToolNode`` stringifies tool-returned content blocks whose
``type`` is not listed in ``langchain_core.tools.base.TOOL_MESSAGE_BLOCK_TYPES``
(see ``msg_content_output`` in ``langgraph/prebuilt/tool_node.py``).  As of
langchain-core 1.4.0 that allowlist is::

    ('text', 'image_url', 'image', 'json', 'search_result',
     'custom_tool_call_output', 'document', 'file')

— ``audio`` and ``video`` are missing, so a ``read()`` of an audio/video file
(which returns ``ToolMessage(content_blocks=[{"type": "audio"|"video", ...}])``)
gets JSON-serialised into a plain string by ``ToolNode`` before it ever reaches
the model, breaking multimodal pass-through.

This module extends both consumers of the allowlist with ``"audio"`` and
``"video"``:

- ``langchain_core.tools.base.TOOL_MESSAGE_BLOCK_TYPES`` (used by
  ``_is_message_content_block`` for message-content validation);
- ``langgraph.prebuilt.tool_node.TOOL_MESSAGE_BLOCK_TYPES`` (the name bound at
  import time inside ``tool_node.py`` — patching the source module alone does
  not update it, so both attributes are patched).

The patch is idempotent: if an upstream version already includes ``audio`` /
``video`` it is left untouched, so this never overrides future upstream fixes.
"""

from __future__ import annotations

import langchain_core.tools.base as _lc_base
import langgraph.prebuilt.tool_node as _tool_node

_EXTRA_BLOCK_TYPES = ("audio", "video")

for _module in (_lc_base, _tool_node):
    _current = getattr(_module, "TOOL_MESSAGE_BLOCK_TYPES", ())
    if not any(_t in _current for _t in _EXTRA_BLOCK_TYPES):
        setattr(_module, "TOOL_MESSAGE_BLOCK_TYPES", (*_current, *_EXTRA_BLOCK_TYPES))
