"""File uploaders — pluggable "pre-upload" hooks for the ``read`` tool.

Reading a multimodal file (image / audio / video / document) normally yields
an inline base64 content block.  That base64 ends up both in the LangGraph
checkpoint and in every model request — large files blow up checkpoints and
can hit provider inline-size limits.  A ``FileUploader`` hook lets the
backend pre-upload the file and replace the inline block with a lightweight
reference (e.g. ``{"type": "file", "file_id": ...}``) before the ``read``
``ToolMessage`` is created.

Built-in implementation:

- :func:`deepseek_file_uploader` — uploads images to the DeepSeek Files API
  and caches the resulting ``file_id`` in a LangGraph store.

Configure one on any backend via the ``file_uploader`` constructor argument
(never enabled implicitly)::

    from langgraph.store.memory import InMemoryStore
    from mambo_agents.backends.local import LocalBackend
    from mambo_agents.file_uploaders import deepseek_file_uploader

    backend = LocalBackend(
        root_dir="/data",
        file_uploader=deepseek_file_uploader(store=InMemoryStore()),
    )

Custom uploaders are plain callables (sync or ``async def``) that receive
``(file_path, base64_content, mime_type)`` and return the replacement content
block(s) — or ``None`` to keep the default inline base64 block.  See
:data:`mambo_agents.backends.protocol.FileUploader` for the full contract.
"""

from mambo_agents.file_uploaders._deepseek import deepseek_file_uploader

__all__ = ["deepseek_file_uploader"]
