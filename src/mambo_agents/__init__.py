"""Mambo Agents - A more robust agent framework built on top of langchain."""

from mambo_agents.backends.protocol import BackendProtocol
from mambo_agents.backends.readonly import ReadOnlyBackend
from mambo_agents.backends.state import StateBackend
from mambo_agents.backends.state_schema import FileData, FilesystemState
from mambo_agents.backends.hybrid_workspace import HybridWorkspaceBackend
from mambo_agents.graph import create_mambo_agent
from mambo_agents.middleware.async_subagents import (
    AsyncSubAgentMiddleware,
    AsyncTaskData,
)
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
from mambo_agents.middleware.summarization import (
    MamboSummarizationMiddleware,
    SummarizationConfig,
    SummaryHook,
    SummaryHookContext,
)

__all__ = [
    "AsyncSubAgentMiddleware",
    "AsyncTaskData",
    "BackendProtocol",
    "CompiledSubAgent",
    "EventGranularity",
    "FileData",
    "FilesystemState",
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
    "StateBackend",
    "HybridWorkspaceBackend",
    "SubAgent",
    "SubAgentMiddleware",
    "SummarizationConfig",
    "SummaryHook",
    "SummaryHookContext",
    "WritePlansInput",
    "create_mambo_agent",
    "create_review_agent",
    "run_review_sync",
]
