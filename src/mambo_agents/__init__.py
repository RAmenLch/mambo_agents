"""Mambo Agents - A more robust agent framework built on top of langchain."""

# Apply runtime compatibility patches first (e.g. extend LangGraph's
# TOOL_MESSAGE_BLOCK_TYPES with "audio"/"video" so multimodal blocks are
# not stringified by ToolNode).  Must run before any tool execution.
from mambo_agents import _compat  # noqa: F401

from mambo_agents._version import __version__
from mambo_agents.backends.protocol import BackendProtocol
from mambo_agents.backends.readonly import ReadOnlyBackend
from mambo_agents.backends.store import StoreBackend
from mambo_agents.backends.hybrid_workspace import HybridWorkspaceBackend
from mambo_agents.graph import create_mambo_agent
from mambo_agents.middleware.async_subagents import (
    AsyncSubAgentMiddleware,
    AsyncTaskData,
)
from mambo_agents.middleware.tool_call_error import ToolCallErrorMiddleware
from mambo_agents.middleware.memory import (
    MamboMemoryMiddleware,
    MemoryFormatHook,
)
from mambo_agents.middleware.planning import (
    MamboPlanMiddleware,
    Plan,
    WritePlansInput,
)
from mambo_agents.middleware.review_agent import (
    FinalReviewResult,
    create_review_agent,
    run_review_sync,
)
from mambo_agents.middleware.skills import (
    SkillMetadata,
    SkillSource,
    SkillsMiddleware,
)
from mambo_agents.middleware.subagents import (
    CompiledSubAgent,
    EventGranularity,
    SubAgent,
    SubAgentMiddleware,
)
from mambo_agents.middleware.subagent_snapshot import (
    SUBAGENT_HISTORY_KEY,
    SubAgentSnapshotNode,
    aget_chain,
    aget_node,
    alist_nodes,
    get_chain,
    get_node,
    list_nodes,
)
from mambo_agents.middleware.summarization import (
    MamboSummarizationMiddleware,
    SummarizationConfig,
    SummarizationMode,
    SummaryHook,
    SummaryHookContext,
)

__all__ = [
    "AsyncSubAgentMiddleware",
    "AsyncTaskData",
    "BackendProtocol",
    "CompiledSubAgent",
    "EventGranularity",
    "FinalReviewResult",
    "MamboMemoryMiddleware",
    "MamboPlanMiddleware",
    "MamboSummarizationMiddleware",
    "MemoryFormatHook",
    "Plan",
    "ReadOnlyBackend",
    "SkillMetadata",
    "SkillSource",
    "SkillsMiddleware",
    "StoreBackend",
    "HybridWorkspaceBackend",
    "SubAgent",
    "SubAgentMiddleware",
    "SubAgentSnapshotNode",
    "SUBAGENT_HISTORY_KEY",
    "aget_chain",
    "aget_node",
    "alist_nodes",
    "get_chain",
    "get_node",
    "list_nodes",
    "SummarizationConfig",
    "SummarizationMode",
    "SummaryHook",
    "SummaryHookContext",
    "ToolCallErrorMiddleware",
    "WritePlansInput",
    "__version__",
    "create_mambo_agent",
    "create_review_agent",
    "run_review_sync",
]
