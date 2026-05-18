"""Middleware for Mambo Agents."""

from mambo_agents.middleware.backend_tools import BackendToolsMiddleware
from mambo_agents.middleware.patch_tool_calls import PatchToolCallsMiddleware
from mambo_agents.middleware.planning import (
    MamboPlanMiddleware,
    Plan,
    WritePlansInput,
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
    DEFAULT_MAMBO_SUMMARY_PROMPT,
    MamboSummarizationMiddleware,
    SummarizationConfig,
    SummaryHook,
    SummaryHookContext,
)

__all__ = [
    "BackendToolsMiddleware",
    "CompiledSubAgent",
    "DEFAULT_MAMBO_SUMMARY_PROMPT",
    "EventGranularity",
    "MamboPlanMiddleware",
    "MamboSummarizationMiddleware",
    "PatchToolCallsMiddleware",
    "Plan",
    "SkillMetadata",
    "SkillSource",
    "SkillsMiddleware",
    "SubAgent",
    "SubAgentMiddleware",
    "SummarizationConfig",
    "SummaryHook",
    "SummaryHookContext",
    "WritePlansInput",
]
