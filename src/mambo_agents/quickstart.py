"""Quick-start helper — spin up a fully-featured agent with one call.

``create_powerful_agent()`` wraps :func:`~mambo_agents.graph.create_mambo_agent`
and activates the most useful optional features behind sensible defaults so you
don't have to configure everything by hand.

Features enabled by default:

* **Auto-summarisation** — long conversations are automatically compacted
  (trigger at 100 k tokens, keep the last 20 messages, persist evicted
  messages to the backend so nothing is lost).
* **Planning middleware** — the agent can create and update structured TODO
  lists via ``write_plans``.
* **General-purpose subagent** — a background subagent that shares the main
  agent's tools, great for multi-step file tasks without explicit subagent
  configuration.
* **MemorySaver checkpointer** — always wired so you can enable
  ``interrupt_on`` without extra ceremony.
* **AI-assisted security review** — when you pass ``interrupt_on=…``, tool
  calls are pre-screened by an AI (default: ``gpt-4o-mini``) before any
  human is interrupted.

Example::

    from mambo_agents.quickstart import create_powerful_agent
    from langchain_core.messages import HumanMessage

    agent = create_powerful_agent("gpt-4o")
    result = agent.invoke({"messages": [HumanMessage("Create a hello.py")]})

    # Work directly on disk:
    agent = create_powerful_agent("gpt-4o", workspace="/tmp/myproject")

    # Add human approval for dangerous operations:
    agent = create_powerful_agent(
        "gpt-4o",
        interrupt_on={"write": True, "edit": True, "delete": True},
    )
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain.agents.middleware import InterruptOnConfig
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore

from mambo_agents.backends.local import LocalBackend
from mambo_agents.backends.protocol import BackendProtocol
from mambo_agents.backends.state import StateBackend
from mambo_agents.graph import create_mambo_agent
from mambo_agents.middleware.planning import MamboPlanMiddleware
from mambo_agents.middleware.security_review import SecurityReviewConfig
from mambo_agents.middleware.skills import SkillSource
from mambo_agents.middleware.subagents import CompiledSubAgent, SubAgent
from mambo_agents.middleware.summarization import SummarizationConfig

# ---------------------------------------------------------------------------
# Power defaults (constants — importable so callers can copy & tweak)
# ---------------------------------------------------------------------------
POWER_DEFAULT_SUMMARIZATION: SummarizationConfig = {
    "trigger": ("tokens", 100_000),
    "keep": ("messages", 20),
    "offload_to_backend": True,
}
"""Default summarization config: compact when tokens exceed 100 k, keep the
last 20 messages, and persist evicted history to the backend."""

POWER_DEFAULT_SECURITY_REVIEW = SecurityReviewConfig(
    model="gpt-4o-mini",
    review_tools="all",
)
"""Default AI-assisted security review: use a cheaper model for pre-screening,
review all interrupt-on tools."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def create_powerful_agent(
    model: str | BaseChatModel,
    *,
    workspace: str | None = None,
    system_prompt: str | None = None,
    name: str | None = None,
    skills: Sequence[SkillSource] | None = None,
    tools: Sequence[BaseTool] | None = None,
    subagents: Sequence[SubAgent | CompiledSubAgent] | None = None,
    async_subagents: Sequence[SubAgent | CompiledSubAgent] | None = None,
    async_subagent_timeout: float = 3600.0,
    interrupt_on: dict[str, bool | InterruptOnConfig] | None = None,
    summarization: SummarizationConfig | bool | None = True,
    enable_planning: bool = True,
    enable_general_purpose: bool = True,
    store: BaseStore | None = None,
    **kwargs: Any,
) -> CompiledStateGraph:
    """Create a fully-featured mambo agent with sensible defaults.

    This is the one-stop entry point.  It calls
    :func:`~mambo_agents.graph.create_mambo_agent` under the hood, wiring
    together the most useful optional features so you don't have to remember
    every config knob.

    Parameters
    ----------
    model:
        Chat model name (e.g. ``"gpt-4o"``) or a ``BaseChatModel`` instance.
    workspace:
        Optional path to a directory on disk.  When provided the agent
        operates on the real filesystem via ``LocalBackend``; otherwise an
        in-memory ``StateBackend`` is used.
    system_prompt:
        Custom system prompt that replaces the default.  Keep it ``None``
        for the sensible built-in prompt.
    name:
        Optional agent name (used for LangSmith tracing).
    skills:
        Skill sources — directories containing ``SKILL.md`` files.  See
        :class:`~mambo_agents.middleware.skills.SkillsMiddleware`.
    tools:
        Extra (non-filesystem) tools to give the agent.
    subagents:
        Additional synchronous subagent specs.
    async_subagents:
        Additional async (background) subagent specs.
    async_subagent_timeout:
        Max seconds an async subagent may run before force-cancel. Default
        3600 (1 hour).
    interrupt_on:
        Per-tool interrupt config (e.g. ``{"write": True, "edit": True}``).
        When set, AI-assisted security review is automatically enabled using
        ``POWER_DEFAULT_SECURITY_REVIEW``.
    summarization:
        * ``True`` (default) — enable summarization with power defaults
          (trigger at 100 k tokens, keep 20 messages, offload to backend).
        * ``False`` / ``None`` — disable summarization entirely.
        * A ``SummarizationConfig`` dict — your own settings.
    enable_planning:
        Whether to activate ``MamboPlanMiddleware`` so the agent can use
        ``write_plans`` to track structured TODO lists.  Default ``True``.
    enable_general_purpose:
        Whether to include the built-in ``general-purpose`` subagent.
        Default ``True``.
    store:
        Optional LangGraph ``BaseStore``.
    **kwargs:
        Forwarded directly to :func:`create_mambo_agent`.

    Returns
    -------
    :class:`langgraph.graph.state.CompiledStateGraph`
        A compiled agent graph ready for ``invoke()`` / ``astream()``.

    Example
    -------
    Minimal usage::

        from mambo_agents.quickstart import create_powerful_agent
        from langchain_core.messages import HumanMessage

        agent = create_powerful_agent("gpt-4o")
        result = agent.invoke({"messages": [HumanMessage("Create a hello.py")]})

    On-disk workspace::

        agent = create_powerful_agent("gpt-4o", workspace="/tmp/project")

    With human-in-the-loop::

        agent = create_powerful_agent(
            "gpt-4o",
            interrupt_on={"write": True, "edit": True, "delete": True},
        )

    Custom summarization::

        agent = create_powerful_agent(
            "gpt-4o",
            summarization={"trigger": ("tokens", 50_000), "keep": ("messages", 10)},
        )

    All knobs off (bare minimum)::

        agent = create_powerful_agent(
            "gpt-4o",
            summarization=False,
            enable_planning=False,
            enable_general_purpose=False,
        )
    """
    # ---- Backend ----------------------------------------------------------
    backend: BackendProtocol
    if workspace is not None:
        backend = LocalBackend(root_dir=workspace)
    else:
        backend = StateBackend()

    # ---- Summarization ----------------------------------------------------
    _summarization: SummarizationConfig | None = None
    if summarization is True:
        _summarization = POWER_DEFAULT_SUMMARIZATION.copy()  # type: ignore[arg-type]
    elif isinstance(summarization, dict):
        _summarization = summarization

    # ---- Middleware -------------------------------------------------------
    _middleware: list[AgentMiddleware] = []
    if enable_planning:
        _middleware.append(MamboPlanMiddleware())

    # ---- Security review (auto-wired) ------------------------------------
    _security_review: SecurityReviewConfig | None = None
    if interrupt_on is not None:
        _security_review = POWER_DEFAULT_SECURITY_REVIEW

    # ---- Checkpointer (always on — lightweight MemorySaver) --------------
    _checkpointer = MemorySaver()

    return create_mambo_agent(
        model=model,
        backend=backend,
        system_prompt=system_prompt,
        subagents=subagents,
        include_general_purpose=enable_general_purpose,
        async_subagents=async_subagents,
        async_subagent_timeout=async_subagent_timeout,
        event_granularity="messages",
        middleware=_middleware,
        summarization=_summarization,
        skills=skills,
        tools=tools,
        interrupt_on=interrupt_on,
        security_review=_security_review,
        checkpointer=_checkpointer,
        store=store,
        name=name,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Re-export the original factory so users can import everything from here
# ---------------------------------------------------------------------------
__all__ = [
    "POWER_DEFAULT_SECURITY_REVIEW",
    "POWER_DEFAULT_SUMMARIZATION",
    "create_powerful_agent",
]
