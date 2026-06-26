"""State schema extensions for Pregel checkpoint integration.

Defines ``FileData`` (TypedDict for file metadata), the file-channel
reducer, and ``FilesystemState`` that adds a ``files`` channel to the
base ``AgentState``.
"""

from __future__ import annotations

from typing import Annotated, NotRequired

from typing_extensions import TypedDict

from langchain.agents.middleware.types import AgentState


# ---------------------------------------------------------------------------
# FileData
# ---------------------------------------------------------------------------


class FileData(TypedDict):
    """Data structure for storing file contents with metadata.

    Uses ``TypedDict`` (not Pydantic ``BaseModel``) because LangGraph's
    ``_resolve_schema()`` merges all middleware ``state_schema`` fields
    into a single TypedDict at compile time.  Pydantic models are not
    TypedDict subclasses and cannot participate in that merge.
    """

    content: str
    """File content as a plain string (utf-8 text or base64-encoded binary)."""

    encoding: str
    """Content encoding: ``"utf-8"`` for text, ``"base64"`` for binary."""

    created_at: NotRequired[str]
    """ISO 8601 timestamp of file creation."""

    modified_at: NotRequired[str]
    """ISO 8601 timestamp of last modification."""


# ---------------------------------------------------------------------------
# Reducer — merges file updates into the ``files`` channel
# ---------------------------------------------------------------------------


def _file_data_reducer(
    left: dict[str, FileData] | None,
    right: dict[str, FileData | None],
) -> dict[str, FileData]:
    """Merge file updates into the ``files`` channel.

    ``None`` values in *right* signal file deletions (the key is removed
    from the channel).  Non-``None`` values overwrite the entry for that
    path.  All other keys are preserved unchanged.
    """
    if left is None:
        return {k: v for k, v in right.items() if v is not None}

    result = {**left}
    for key, value in right.items():
        if value is None:
            result.pop(key, None)
        else:
            result[key] = value
    return result


# ---------------------------------------------------------------------------
# FilesystemState
# ---------------------------------------------------------------------------


class FilesystemState(AgentState):
    """Agent state with a ``files`` channel for checkpoint-persistent storage.

    The ``files`` field uses a dict-merge reducer so that partial updates
    (e.g. writing a single file) only affect the changed keys — all other
    files survive unchanged across node boundaries and checkpoints.
    """

    files: Annotated[NotRequired[dict[str, FileData]], _file_data_reducer]
    """File system contents keyed by absolute path.

    **NotRequired** — omitted when no files are present, so backends that
    do not use the ``files`` channel (e.g. ``LocalBackend``) pay no cost
    except an empty ``{}`` in the checkpoint.
    """
