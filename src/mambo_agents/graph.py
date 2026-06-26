"""Main graph assembly — ``create_mambo_agent()``.

Builds a langchain agent graph that uses ``BackendToolsMiddleware`` to
dynamically register file-system tools from a ``BackendProtocol``.

Optionally supports subagents via ``SubAgentMiddleware``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain.agents.factory import create_agent as _langchain_create_agent
from langchain.agents.middleware import InterruptOnConfig
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.base import BaseStore

from mambo_agents._version import __version__
from mambo_agents.backends.protocol import BackendProtocol
from mambo_agents.backends.readonly import ReadOnlyBackend
from mambo_agents.backends.schemas import VirtualPath
from mambo_agents.backends.state import StateBackend
from mambo_agents.middleware.backend_tools import (
    BackendToolsMiddleware,
    build_tool_descriptions,
)
from mambo_agents.middleware.memory import MamboMemoryMiddleware
from mambo_agents.middleware.patch_tool_calls import PatchToolCallsMiddleware
from mambo_agents.middleware.reorder_tool_messages import ReorderToolMessagesMiddleware
from mambo_agents.middleware.security_review import (
    AutoSecurityReviewMiddleware,
    SecurityReviewConfig,
)
from mambo_agents.middleware.skills import (
    SkillSource,
    SkillsMiddleware,
)
from mambo_agents.middleware.async_subagents import AsyncSubAgentMiddleware
from mambo_agents.middleware.subagents import (
    CompiledSubAgent,
    EventGranularity,
    SubAgent,
    SubAgentMiddleware,
    DEFAULT_GENERAL_PURPOSE_DESCRIPTION,
    DEFAULT_SUBAGENT_PROMPT,
    GENERAL_PURPOSE_NAME,
)
from mambo_agents.middleware.summarization import (
    DEFAULT_MAMBO_SUMMARY_PROMPT,
    DEFAULT_MAMBO_CHAINED_SUMMARY_PROMPT,
    MamboSummarizationMiddleware,
    SummarizationConfig,
    SummaryHook,
)
from mambo_agents.middleware.planning import MamboPlanMiddleware


DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant that can work with files.

You have access to file system tools to list, read, write, edit, search, and find files.
Use them to help the user accomplish their tasks efficiently.

## File System Guidelines
- Always use absolute paths starting with '/'.
- Read files before modifying them.
- Use `grep` to find specific content across files.
- Use `glob` to find files by name patterns.
- When writing new files, ensure the content is complete and correct.
"""


def create_mambo_agent(
    model: str | BaseChatModel,
    *,
    backend: BackendProtocol | None = None,
    system_prompt: str | None = None,
    subagents: Sequence[SubAgent | CompiledSubAgent] | None = None,
    include_general_purpose: bool = False,
    async_subagents: Sequence[SubAgent | CompiledSubAgent] | None = None,
    async_subagent_timeout: float = 3600.0,
    subagent_event_granularity: EventGranularity = "updates",
    middleware: Sequence[AgentMiddleware] | None = None,
    summarization: SummarizationConfig | None = None,
    skills: Sequence[SkillSource] | None = None,
    memory_sources: list[VirtualPath] | None = None,
    tools: Sequence[BaseTool] | None = None,
    interrupt_on: dict[str, bool | InterruptOnConfig] | None = None,
    security_review: SecurityReviewConfig | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
    name: str | None = None,
    **kwargs: Any,
) -> CompiledStateGraph:
    """Create an agent with file-system tools backed by *backend*.

    Args:
        model: Chat model (string name or ``BaseChatModel`` instance).
        backend: File-system backend.  Defaults to ``StateBackend()``.
        system_prompt: Custom system prompt (replaces the default).
        subagents: Optional subagent specs.  Each can be a ``SubAgent``
            dict or a pre-compiled ``CompiledSubAgent``.
        include_general_purpose: If ``True``, automatically add a
            ``"general-purpose"`` subagent (with the same model, backend
            tools, and system prompt as the main agent).  Default:
            ``False``.
        async_subagents: Optional async subagent specs.  Each runs in a
            background thread — ``async_task()`` returns immediately with
            a ``task_id``, and results are retrieved later via
            ``async_status(task_id)``.  Subagents receive a
            ``report_progress`` tool for intermediate progress reporting.
        async_subagent_timeout: Maximum seconds an async subagent may run
            before being force-cancelled.  Default: ``3600`` (1 hour).
        subagent_event_granularity: Streaming detail level for subagent custom
            events. ``"messages"`` (finest, token-level), ``"updates"``
            (default, per-node), ``"values"`` (coarsest, per-step).
        middleware: Additional middleware to include.
        summarization: Optional summarization configuration.

            When provided, a ``MamboSummarizationMiddleware`` is added
            that automatically compacts older messages when token limits
            are approached.  See
            :class:`~mambo_agents.middleware.summarization.SummarizationConfig`.

            **Example**::

                summarization={
                    "trigger": ("tokens", 100000),
                    "keep": ("messages", 20),
                }

            Default: ``None`` (summarization disabled).
        skills: Optional list of skill sources.  When provided, a
            ``SkillsMiddleware`` is added that loads skills from backend
            directories and exposes them to the agent via progressive
            disclosure.  Each source is either a bare path or a
            ``(path, label)`` tuple.

            **Example**::

                skills=[
                    "/skills/user/",
                    "/skills/project/",
                    ("/repo/.claude/skills", "Project Claude"),
                ]

            Default: ``None`` (skills disabled).
        memory_sources: Optional list of AGENTS.md file paths to load
            as agent memory.  When provided, a ``MamboMemoryMiddleware``
            is added that loads persistent context from these files and
            injects it into the system prompt.  The agent is also
            instructed to **write back** new learnings using the ``edit``
            / ``write`` tools.

            Unlike skills (on-demand), memory is always loaded and
            provides persistent, evolving context.

            **Example**::

                from mambo_agents.backends.schemas import VirtualPath
                memory_sources=[VirtualPath("/.mambo/memory/AGENTS.md")]

            Default: ``None`` (memory disabled).
        tools: Additional (non-file-system) tools.
        interrupt_on: Mapping of ``tool_name → bool or InterruptOnConfig``.
            When set, adds human-in-the-loop so that calls to the specified
            tools pause for human approval.  Requires a *checkpointer*
            (e.g. ``MemorySaver()``).

            By default the classic ``HumanInTheLoopMiddleware`` is used
            (direct human review every time).  To enable AI-assisted
            pre-screening, pass ``security_review=SecurityReviewConfig()``.

        security_review: Optional configuration to enable AI-assisted
            security pre-screening before human-in-the-loop.  When set,
            ``AutoSecurityReviewMiddleware`` replaces the default
            ``HumanInTheLoopMiddleware``.

            Two review modes are supported:

            - **llm** (default): single LLM call with structured output
              for each tool call.  Fast and cheap.
            - **agent**: a dedicated review agent with read-only backend
              tools (``ls``/``read``/``grep``/``glob``) that can inspect
              the workspace before delivering a verdict.  Backend tools
              get agent review; non-backend tools fall back to llm review.

            .. code-block:: python

                from mambo_agents.middleware import SecurityReviewConfig

                # Default — llm-mode for all tools
                security_review=SecurityReviewConfig()

                # Agent-mode — backend tools reviewed by agent
                security_review=SecurityReviewConfig(
                    review_mode="agent",
                    agent_max_steps=5,
                )

                # Custom: cheaper model + selective tool review
                security_review=SecurityReviewConfig(
                    model="gpt-4o-mini",
                    review_tools=frozenset(["edit", "delete"]),
                    system_prompt="You are an expert security auditor...",
                )

            - ``model``: review model (``None`` = reuse agent model).
            - ``system_prompt``: custom security review prompt.
            - ``review_tools``: ``"all"`` (default) or ``frozenset[str]``.
              Tools not in the set get direct HITL (no AI review).
            - ``review_mode``: ``"llm"`` (default) or ``"agent"``.
            - ``agent_max_steps``: max steps for the review agent
              (default 5, only used when ``review_mode="agent"``).

            Default: ``None`` (no AI pre-screening, classic HITL).
        checkpointer: Optional LangGraph checkpointer.  Required when using
            ``interrupt_on``.
        store: Optional LangGraph store.
        name: Optional agent name (used for LangSmith tracing).
        **kwargs: Passed through to ``langchain.agents.create_agent``.

    Returns:
        A compiled ``CompiledStateGraph`` ready for ``invoke()`` /
        ``astream()``.

    Example::

        agent = create_mambo_agent("gpt-4o")
        result = agent.invoke({"messages": [HumanMessage("Create a hello.py file")]})

    Example with pre-populated files::

        agent = create_mambo_agent(
            "gpt-4o",
            backend=StateBackend(initial_files={
                "/config.json": '{"port": 8080}',
            }),
        )
        result = agent.invoke({
            "messages": [HumanMessage("analyze config.json")],
        })

    Example with subagents and streaming::

        agent = create_mambo_agent(
            "gpt-4o",
            backend=StateBackend(),
            subagents=[
                SubAgent(
                    name="researcher",
                    description="Research topics thoroughly",
                    system_prompt="You are a researcher...",
                    model="gpt-4o",
                    tools=[],
                ),
            ],
            subagent_event_granularity="messages",
        )
        async for event in agent.astream(
            {"messages": [HumanMessage("Research Python async patterns")]},
            stream_mode=["updates", "custom"],
        ):
            print(event)

    Example with async subagents::

        agent = create_mambo_agent(
            "gpt-4o",
            backend=LocalBackend(),
            async_subagents=[
                SubAgent(
                    name="deployer",
                    description="Deploy services to Kubernetes",
                    system_prompt="You are a deployment expert...",
                    model="gpt-4o",
                    tools=[kubectl_tool, helm_tool],
                ),
            ],
            async_subagent_timeout=1800,  # 30 minutes
        )
        result = agent.invoke(
            {"messages": [HumanMessage("Deploy v2.3 to prod")]}
        )
        # Agent: "Launched task a3f4b2c1, running in background..."
        # ... later ...
        result = agent.invoke(
            {"messages": [HumanMessage("Check a3f4b2c1")]}
        )
        # Agent reads progress/result via async_status
    """
    if backend is None:
        backend = StateBackend()

    # Build middleware stack
    mw: list[AgentMiddleware] = [
        BackendToolsMiddleware(
            backend=backend,
            custom_system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT,
        ),
    ]

    # ---- Skills (opt-in) ---------------------------------------------------
    if skills is not None:
        mw.append(
            SkillsMiddleware(
                backend=backend,
                sources=skills,
            )
        )

    # ---- Memory (opt-in) ---------------------------------------------------
    if memory_sources is not None:
        mw.append(
            MamboMemoryMiddleware(
                backend=backend,
                sources=memory_sources,
            )
        )

    # ---- Summarization (opt-in) --------------------------------------------
    if summarization is not None:
        # Accept dict for convenience, convert to SummarizationConfig
        if isinstance(summarization, dict):
            summarization = SummarizationConfig(**summarization)

        # Pre-scan user middleware for summary hooks (e.g. Plan)
        _summary_hooks: list[SummaryHook] = list(summarization.summary_hooks or [])

        # Auto-detect MamboPlanMiddleware and wire its hook
        for mw_item in middleware or []:
            if isinstance(mw_item, MamboPlanMiddleware):
                _summary_hooks.append(mw_item.build_summary_hook())
                break  # One hook is sufficient

        _summary_model: str | BaseChatModel = summarization.model or model  # type: ignore[assignment]
        _summary_backend = summarization.backend or backend
        mw.append(
            MamboSummarizationMiddleware(
                model=_summary_model,
                trigger=summarization.trigger,
                keep=summarization.keep,
                summary_prompt=summarization.summary_prompt,
                chained_summary_prompt=(
                    summarization.chained_summary_prompt
                    or DEFAULT_MAMBO_CHAINED_SUMMARY_PROMPT
                ),
                trim_tokens_to_summarize=summarization.trim_tokens_to_summarize,
                token_counter=summarization.token_counter,
                chars_per_token=summarization.chars_per_token,
                offload_to_backend=summarization.offload_to_backend,
                backend=_summary_backend,
                summary_hooks=_summary_hooks or None,
            )
        )

    if middleware:
        mw.extend(middleware)

    # ---- Pre-calculate security review parameters (needed for subagents) ---
    _security_review_middleware: AutoSecurityReviewMiddleware | None = None
    if interrupt_on is not None:
        if checkpointer is None:
            raise ValueError(
                "interrupt_on requires a checkpointer (e.g. MemorySaver()). "
                "Pass `checkpointer=MemorySaver()` to create_mambo_agent."
            )
        if security_review is not None:
            # ---- AI-assisted security pre-screening ----
            _review_model: str | BaseChatModel = security_review.model or model
            _review_tools = security_review.review_tools
            _review_mode = security_review.review_mode
            _agent_max_steps = security_review.agent_max_steps
            _agent_tools_whitelist = security_review.agent_tools

            _tool_descriptions = build_tool_descriptions(
                backend, tools=tools,
            )

            _backend_tool_names: frozenset[str] = frozenset(
                ["ls", "read", "write", "edit", "grep", "glob"]
                + [t.name for t in backend.tools]
            )

            _agent_backend: ReadOnlyBackend | None = None
            if _review_mode == "agent":
                _readonly_extras = (
                    _agent_tools_whitelist if _agent_tools_whitelist is not None
                    else frozenset()
                )
                _agent_backend = ReadOnlyBackend(
                    backend, allowed_extra_tools=_readonly_extras,
                )

            _security_review_middleware = AutoSecurityReviewMiddleware(
                interrupt_on=interrupt_on,
                model=_review_model,
                review_tools=_review_tools,
                security_review_system_prompt=security_review.system_prompt,
                review_mode=_review_mode,
                agent_max_steps=_agent_max_steps,
                agent_backend=_agent_backend,
                agent_tools=_agent_tools_whitelist,
                tool_descriptions=_tool_descriptions,
                backend_tool_names=_backend_tool_names,
            )
        else:
            # ---- Classic HITL (no AI review) ----
            _security_review_middleware = AutoSecurityReviewMiddleware(
                interrupt_on=interrupt_on,
                model=model,
                review_tools=frozenset(),
            )

    # ---- Subagents ----------------------------------------------------------
    inline_subagents = list(subagents or [])

    if include_general_purpose and not any(
        s.name == GENERAL_PURPOSE_NAME
        for s in inline_subagents
    ):
        gp_middleware: list[AgentMiddleware] = [BackendToolsMiddleware(backend)]
        if _security_review_middleware is not None:
            gp_middleware.append(_security_review_middleware)

        gp_spec = SubAgent(
            name=GENERAL_PURPOSE_NAME,
            description=DEFAULT_GENERAL_PURPOSE_DESCRIPTION,
            system_prompt=DEFAULT_SUBAGENT_PROMPT,
            model=model,
            tools=list(tools or []),
            middleware=gp_middleware,
        )
        inline_subagents.insert(0, gp_spec)

    if inline_subagents:
        mw.append(
            SubAgentMiddleware(
                backend=backend,
                subagents=inline_subagents,
                event_granularity=subagent_event_granularity,
            )
        )

    # ---- Async Subagents (opt-in) -----------------------------------------
    if async_subagents is not None:
        mw.append(
            AsyncSubAgentMiddleware(
                backend=backend,
                async_subagents=list(async_subagents),
                default_timeout=async_subagent_timeout,
            )
        )

    # ---- Security review (main agent) --------------------------------------
    if _security_review_middleware is not None:
        mw.append(_security_review_middleware)

    # ---- Safety net (always on) -------------------------------------------
    # Patch first to fill dangling ToolMessages, then reorder to match
    # AIMessage.tool_calls order.  Reorder relies on complete batches
    # (len(buffered) == len(ordered_ids)), so Patch must run first.
    mw.append(PatchToolCallsMiddleware())
    mw.append(ReorderToolMessagesMiddleware())

    # ---- Default checkpointer (always on) ----------------------------------
    # Without a checkpointer, LangGraph state (files, plans, messages, etc.)
    # is lost between separate ``invoke()`` calls — even with the same
    # ``thread_id``.  This breaks multi-turn conversations and any scenario
    # where state must persist across invocations (e.g. sequential agent
    # calls that share file state).
    #
    # We default to ``InMemorySaver`` so that state persists for the
    # lifetime of the ``CompiledStateGraph`` object.  Users who need
    # durable persistence should pass their own ``checkpointer``
    # (e.g. ``SqliteSaver``, ``PostgresSaver``).
    if checkpointer is None:
        checkpointer = InMemorySaver()

    return _langchain_create_agent(
        model=model,
        system_prompt=None,  # Handled by BackendToolsMiddleware
        middleware=mw,
        tools=list(tools or []),
        checkpointer=checkpointer,
        store=store,
        **kwargs,
    ).with_config(
        {
            "recursion_limit": 9_999,
            "metadata": {
                "ls_integration": "mambo_agents",
                "versions": {"mambo_agents": __version__},
                "lc_agent_name": name,
            },
        }
    )
