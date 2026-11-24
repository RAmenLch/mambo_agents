"""Chain snapshots of subagent runs, so a failed task can be continued.

A subagent is ephemeral: when it fails, everything it had done is thrown away and
the main agent either redoes the work or gives up.  This module stores **what the
subagent had already accomplished** (the messages it produced) in the LangGraph
``BaseStore`` under a chain of nodes, and hands the main agent a ``continue_id``
(``= tool_call_id`` of the failing ``task`` call) it can pass back to resume.

Design constraints that shaped this module:

* **Nodes are immutable.**  A node is written once and never modified or deleted —
  except for the one legitimately-superseded case handled by
  :func:`delete_node` (the same ``tool_call_id`` executing again).  Anything that
  needs a counter (child count, failure streak) is derived by scanning, never
  stored on a node.
* **Chains form a tree.**  ``chain`` is the ancestor path; a node may have several
  children (continuing from the same breakpoint in different directions), which is
  an explicitly supported use case.
* **Nothing here depends on LangGraph internals.**  Only ``BaseStore`` + normal
  message (de)serialization.

The state machine that decides whether a ``continue_id`` may be used lives in
:func:`validate_continuation`.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict
from langgraph.store.base import BaseStore
from pydantic import BaseModel, ConfigDict, Field

from mambo_agents.middleware.tool_call_error import is_retryable

logger = logging.getLogger(__name__)

__all__ = [
    "SUBAGENT_HISTORY_KEY",
    "ContinuationCheck",
    "ContinuationStatus",
    "SubAgentSnapshotNode",
    "adelete_node",
    "aget_chain",
    "aget_node",
    "alist_nodes",
    "aput_node",
    "build_continuation_prompt",
    "build_delta",
    "build_failure_content",
    "continuation_metadata",
    "delete_node",
    "get_chain",
    "get_node",
    "list_nodes",
    "node_from_item",
    "put_node",
    "snapshot_namespace",
    "validate_continuation",
]

#: Key under ``(thread_id, SUBAGENT_HISTORY_KEY)`` used for the node snapshots.
SUBAGENT_HISTORY_KEY = "mambo_subagent_history"

SnapshotStatus = Literal["failed", "cancelled"]
"""Terminal states a node can record.  There is deliberately no ``"sealed"``:
a node stays valid forever and may be used as a branch point again."""

ContinuationStatus = Literal[
    "available",
    "not_found",
    "type_mismatch",
    "chain_broken",
    "not_available",
    "exhausted",
    "branched_out",
    "busy",
]

_MESSAGE_LIMIT = 1000
_PAGE_SIZE = 100


class SubAgentSnapshotNode(BaseModel):
    """One link of a subagent continuation chain."""

    model_config = ConfigDict(extra="ignore")

    node_id: str = Field(description="``tool_call_id`` of the task call that produced it.")
    parent_id: str | None = Field(default=None, description="``None`` for a chain root.")
    chain: list[str] = Field(
        default_factory=list,
        description="Ordered ancestor ``node_id`` list, excluding this node.",
    )
    depth: int = Field(default=1, description="``len(chain) + 1``.")
    subagent_type: str = ""
    description: str = Field(
        default="",
        description="Instruction carried by this run (root: first task description).",
    )
    status: SnapshotStatus = "failed"
    error_type: str = ""
    error_message: str = ""
    retryable: bool = False
    available: bool = Field(
        default=True,
        description="``False`` → the ``continue_id`` is not handed back to the model.",
    )
    delta_messages: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Messages produced by this run only (``messages_to_dict`` form).",
    )
    steps: int = Field(default=0, description="Committed supersteps in this run.")
    injected_prefix_len: int = Field(
        default=0, description="Length of the history injected into this run."
    )
    created_at: str = ""

    # ------------------------------------------------------------------
    # message helpers
    # ------------------------------------------------------------------
    def messages(self) -> list[BaseMessage]:
        """Deserialize :attr:`delta_messages`."""
        if not self.delta_messages:
            return []
        try:
            return list(messages_from_dict(self.delta_messages))  # type: ignore[arg-type]
        except Exception:  # pragma: no cover - defensive
            logger.warning(
                "Failed to deserialize subagent snapshot %s; treating delta as empty",
                self.node_id,
                exc_info=True,
            )
            return []


def snapshot_namespace(thread_id: str) -> tuple[str, str]:
    """Store namespace holding the snapshots of one conversation."""
    return (thread_id, SUBAGENT_HISTORY_KEY)


# ---------------------------------------------------------------------------
# raw store access (sync + async)
# ---------------------------------------------------------------------------
def node_from_item(item: Any) -> SubAgentSnapshotNode | None:
    """Build a node from a store item, tolerating malformed values."""
    value = getattr(item, "value", item)
    if not isinstance(value, dict):
        return None
    try:
        return SubAgentSnapshotNode.model_validate(value)
    except Exception:  # pragma: no cover - defensive
        logger.warning("Malformed subagent snapshot: %r", value, exc_info=True)
        return None


def put_node(store: BaseStore, thread_id: str, node: SubAgentSnapshotNode) -> None:
    """Persist (or overwrite) a node."""
    store.put(snapshot_namespace(thread_id), node.node_id, node.model_dump())


async def aput_node(store: BaseStore, thread_id: str, node: SubAgentSnapshotNode) -> None:
    """Persist (or overwrite) a node (async)."""
    await store.aput(snapshot_namespace(thread_id), node.node_id, node.model_dump())


def get_node(
    store: BaseStore, thread_id: str, node_id: str
) -> SubAgentSnapshotNode | None:
    """Read a single node."""
    return node_from_item(store.get(snapshot_namespace(thread_id), node_id))


async def aget_node(
    store: BaseStore, thread_id: str, node_id: str
) -> SubAgentSnapshotNode | None:
    """Read a single node (async)."""
    return node_from_item(await store.aget(snapshot_namespace(thread_id), node_id))


def delete_node(store: BaseStore, thread_id: str, node_id: str) -> None:
    """Drop a node.

    Only ever used for **entry cleanup**: the same ``tool_call_id`` executing
    again supersedes whatever the previous attempt left behind.  Never used to
    prune a chain — that would break every descendant.
    """
    store.delete(snapshot_namespace(thread_id), node_id)


async def adelete_node(store: BaseStore, thread_id: str, node_id: str) -> None:
    """Drop a node (async).  See :func:`delete_node`."""
    await store.adelete(snapshot_namespace(thread_id), node_id)


def _collect(store: BaseStore, thread_id: str) -> list[SubAgentSnapshotNode]:
    namespace = snapshot_namespace(thread_id)
    out: list[SubAgentSnapshotNode] = []
    offset = 0
    while True:
        items = store.search(namespace, limit=_PAGE_SIZE, offset=offset)
        for item in items:
            node = node_from_item(item)
            if node is not None:
                out.append(node)
        if len(items) < _PAGE_SIZE:
            return out
        offset += _PAGE_SIZE


async def _acollect(store: BaseStore, thread_id: str) -> list[SubAgentSnapshotNode]:
    namespace = snapshot_namespace(thread_id)
    out: list[SubAgentSnapshotNode] = []
    offset = 0
    while True:
        items = await store.asearch(namespace, limit=_PAGE_SIZE, offset=offset)
        for item in items:
            node = node_from_item(item)
            if node is not None:
                out.append(node)
        if len(items) < _PAGE_SIZE:
            return out
        offset += _PAGE_SIZE


def list_nodes(store: BaseStore, thread_id: str) -> list[SubAgentSnapshotNode]:
    """All nodes of one conversation.

    Filtering by any field is done in Python on purpose: ``store.search(filter=...)``
    is not portable across the ``InMemory`` / ``Sqlite`` / ``Postgres`` backends.
    """
    return _collect(store, thread_id)


async def alist_nodes(store: BaseStore, thread_id: str) -> list[SubAgentSnapshotNode]:
    """All nodes of one conversation (async)."""
    return await _acollect(store, thread_id)


def get_chain(
    store: BaseStore, thread_id: str, node_id: str
) -> tuple[list[SubAgentSnapshotNode], list[str]]:
    """Return ``(nodes_in_root_to_leaf_order, missing_ancestor_ids)``."""
    node = get_node(store, thread_id, node_id)
    if node is None:
        return [], [node_id]
    nodes: list[SubAgentSnapshotNode] = []
    missing: list[str] = []
    for ancestor_id in node.chain:
        ancestor = get_node(store, thread_id, ancestor_id)
        if ancestor is None:
            missing.append(ancestor_id)
        else:
            nodes.append(ancestor)
    nodes.append(node)
    return nodes, missing


async def aget_chain(
    store: BaseStore, thread_id: str, node_id: str
) -> tuple[list[SubAgentSnapshotNode], list[str]]:
    """Return ``(nodes, missing_ancestor_ids)`` (async)."""
    node = await aget_node(store, thread_id, node_id)
    if node is None:
        return [], [node_id]
    nodes: list[SubAgentSnapshotNode] = []
    missing: list[str] = []
    for ancestor_id in node.chain:
        ancestor = await aget_node(store, thread_id, ancestor_id)
        if ancestor is None:
            missing.append(ancestor_id)
        else:
            nodes.append(ancestor)
    nodes.append(node)
    return nodes, missing


def count_children(nodes: Sequence[SubAgentSnapshotNode], node_id: str) -> int:
    """How many branches were started from *node_id*."""
    return sum(1 for node in nodes if node.parent_id == node_id)


def count_root_failures(nodes: Sequence[SubAgentSnapshotNode]) -> int:
    """How many chain roots failed in this conversation."""
    return sum(
        1 for node in nodes if node.parent_id is None and node.status == "failed"
    )


# ---------------------------------------------------------------------------
# continuation state machine
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ContinuationCheck:
    """Result of validating a ``continue_id``."""

    status: ContinuationStatus = "available"
    message: str = ""
    node: SubAgentSnapshotNode | None = None
    nodes: tuple[SubAgentSnapshotNode, ...] = ()

    @property
    def ok(self) -> bool:
        return self.status == "available"

    def history(self) -> list[BaseMessage]:
        """Concatenated delta of the whole chain, root first."""
        out: list[BaseMessage] = []
        for node in self.nodes:
            out.extend(node.messages())
        return out


def _fail(status: ContinuationStatus, message: str) -> ContinuationCheck:
    return ContinuationCheck(status=status, message=message)


def validate_continuation(
    store: BaseStore,
    thread_id: str,
    node_id: str,
    subagent_type: str,
    *,
    max_continue_attempts: int = 3,
    max_children: int = 2,
    in_flight: Collection[str] = (),
) -> ContinuationCheck:
    """Decide whether ``node_id`` may be used as a breakpoint.

    Order matters: cheap and unambiguous rejections come first, so the model gets
    the most specific reason.
    """
    if node_id in in_flight:
        return _fail("busy", "该任务正在执行中，拒绝并发续连。")

    node = get_node(store, thread_id, node_id)
    if node is None:
        return _fail(
            "not_found",
            f"未找到 continue_id={node_id!r} 对应的存档，请核对 continue_id 与 subagent_type。",
        )
    if node.subagent_type and node.subagent_type != subagent_type:
        return _fail(
            "type_mismatch",
            f"continue_id={node_id!r} 属于 subagent_type={node.subagent_type!r}，"
            f"不能用于 {subagent_type!r}。",
        )

    nodes, missing = get_chain(store, thread_id, node_id)
    if missing:
        return _fail(
            "chain_broken",
            "续连链不完整（缺失：" + ", ".join(missing) + "），"
            "请从最近的可用 continue_id 续连，或重新开始。",
        )

    if not node.available:
        return _fail(
            "not_available",
            "该断点不可续连（错误不可重试或已超限），请换策略或上报用户。",
        )
    if node.depth > max_continue_attempts:
        return _fail(
            "exhausted",
            f"已达续连上限（depth={node.depth} > {max_continue_attempts}），"
            "请换策略或上报用户。",
        )

    siblings = count_children(list_nodes(store, thread_id), node_id)
    if siblings >= max_children:
        return _fail(
            "branched_out",
            f"该断点已从不同方向续连过 {siblings} 次，仍失败请换子代理或重新开始。",
        )

    return ContinuationCheck(status="available", node=node, nodes=tuple(nodes))


async def avalidate_continuation(
    store: BaseStore,
    thread_id: str,
    node_id: str,
    subagent_type: str,
    *,
    max_continue_attempts: int = 3,
    max_children: int = 2,
    in_flight: Collection[str] = (),
) -> ContinuationCheck:
    """Async twin of :func:`validate_continuation`."""
    if node_id in in_flight:
        return _fail("busy", "该任务正在执行中，拒绝并发续连。")

    node = await aget_node(store, thread_id, node_id)
    if node is None:
        return _fail(
            "not_found",
            f"未找到 continue_id={node_id!r} 对应的存档，请核对 continue_id 与 subagent_type。",
        )
    if node.subagent_type and node.subagent_type != subagent_type:
        return _fail(
            "type_mismatch",
            f"continue_id={node_id!r} 属于 subagent_type={node.subagent_type!r}，"
            f"不能用于 {subagent_type!r}。",
        )

    nodes, missing = await aget_chain(store, thread_id, node_id)
    if missing:
        return _fail(
            "chain_broken",
            "续连链不完整（缺失：" + ", ".join(missing) + "），"
            "请从最近的可用 continue_id 续连，或重新开始。",
        )

    if not node.available:
        return _fail(
            "not_available",
            "该断点不可续连（错误不可重试或已超限），请换策略或上报用户。",
        )
    if node.depth > max_continue_attempts:
        return _fail(
            "exhausted",
            f"已达续连上限（depth={node.depth} > {max_continue_attempts}），"
            "请换策略或上报用户。",
        )

    siblings = count_children(await alist_nodes(store, thread_id), node_id)
    if siblings >= max_children:
        return _fail(
            "branched_out",
            f"该断点已从不同方向续连过 {siblings} 次，仍失败请换子代理或重新开始。",
        )

    return ContinuationCheck(status="available", node=node, nodes=tuple(nodes))


# ---------------------------------------------------------------------------
# delta & message building
# ---------------------------------------------------------------------------
def build_delta(
    injected: Sequence[BaseMessage],
    produced: Sequence[BaseMessage],
    *,
    injected_ids: Collection[str] | None = None,
) -> list[BaseMessage]:
    """Messages produced by this run only.

    Computed as an **id difference**, not a list slice: a subagent that has a
    summarization middleware (or anything using ``Overwrite``) rewrites
    ``messages``, which silently breaks ``produced[len(injected):]``.

    ``injected_ids`` must be captured **before** the run: message ids assigned
    during execution (e.g. a freshly built ``HumanMessage``) would otherwise be
    classified as "already present" and dropped from the delta.
    """
    ids = set(injected_ids) if injected_ids is not None else {
        m.id for m in injected if getattr(m, "id", None)
    }
    if ids:
        return [m for m in produced if getattr(m, "id", None) not in ids]
    identity = {id(m) for m in injected}
    remaining = [m for m in produced if id(m) not in identity]
    if len(remaining) == len(produced) and len(produced) > len(injected):
        return list(produced[len(injected) :])
    return remaining


def to_serialized(messages: Sequence[BaseMessage]) -> list[dict[str, Any]]:
    """Serialize messages for storage (keeps ids, tool calls and artifacts)."""
    return messages_to_dict(list(messages))  # type: ignore[return-value]


def build_continuation_prompt(error_type: str, description: str) -> str:
    """Instruction prepended when resuming from a breakpoint."""
    reason = error_type or "unknown error"
    tail = description or "继续完成原任务"
    return (
        f"上一次执行因 {reason} 中断。以上是已完成步骤的记录，请勿重复执行；"
        f"从断点继续完成任务：{tail}"
    )


def build_failure_content(
    *,
    subagent_type: str,
    error_type: str,
    error_message: str,
    steps: int,
    node_id: str | None,
    retryable: bool,
    available: bool,
    reason: str = "",
) -> str:
    """Model-facing text of the error ``ToolMessage``."""
    detail = (error_message or "").strip()
    lines = [
        f"Subagent '{subagent_type}' failed with {error_type}: {detail}"[: _MESSAGE_LIMIT],
        f"Progress before failure: {steps} step(s) in this run.",
    ]
    if not retryable:
        lines.append(
            "This error is not retryable. Do not retry with continue_id; report it to the user."
        )
    elif available and node_id:
        lines.append(f'Continuation saved: node_id="{node_id}"')
        lines.append("")
        lines.append("To continue without repeating completed steps, call:")
        lines.append(f'  task(subagent_type="{subagent_type}", continue_id="{node_id}")')
        lines.append(
            "If the error looks persistent (configuration / permission / repeated failures), "
            "do not keep retrying — report the error and node_id to the user."
        )
    else:
        why = reason or "no continuation is available"
        lines.append(
            f"Continuation is not available ({why}). Stop retrying and report this to the user."
        )
    return "\n".join(lines)


def continuation_metadata(
    node: SubAgentSnapshotNode | None,
    *,
    max_attempts: int,
    max_children: int,
    child_count: int = 0,
    available: bool | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    """Structured payload attached to the error ``ToolMessage``."""
    if node is None:
        return {
            "available": False,
            "reason": reason or "not_recorded",
            "max_attempts": max_attempts,
            "max_children": max_children,
        }
    return {
        "continue_id": node.node_id,
        "subagent_type": node.subagent_type,
        "status": node.status,
        "depth": node.depth,
        "max_attempts": max_attempts,
        "max_children": max_children,
        "child_count": child_count,
        "retryable": node.retryable,
        "error_type": node.error_type,
        "steps": node.steps,
        "available": node.available if available is None else available,
        "reason": reason,
    }


def classify_failure(exc: BaseException) -> tuple[str, str, bool]:
    """``(error_type, error_message, retryable)`` for an exception."""
    return type(exc).__name__, str(exc)[: _MESSAGE_LIMIT], is_retryable(exc)


def now_iso() -> str:
    """UTC timestamp used by ``created_at``."""
    return datetime.now(timezone.utc).isoformat()


def empty_check(status: ContinuationStatus, message: str) -> ContinuationCheck:
    """Convenience for callers building a rejection outside the state machine."""
    return ContinuationCheck(status=status, message=message, node=None, nodes=())


def chain_ids(nodes: Sequence[SubAgentSnapshotNode]) -> list[str]:
    """``chain`` value for a node continuing from the last of *nodes*."""
    return [node.node_id for node in nodes]
