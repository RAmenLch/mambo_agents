"""Security review middleware — AI-based gate before human-in-the-loop.

When ``interrupt_on`` is configured for certain tools, instead of
immediately pausing for human approval, this middleware **first** asks an
AI to review each tool call for security risks.  Only tool calls that the
AI flags as **unsafe** are escalated to human review via ``interrupt()``.

Users opt-in via the ``security_review`` parameter of ``create_mambo_agent``.

Two review modes are available:

- **llm** (default): single structured-output LLM call per tool call.
- **agent**: a dedicated review agent with optional read-only backend
  tools.  The agent can inspect the workspace before delivering a
  structured verdict via the ``submit_review_verdict`` tool.  Backend tools
  (core 6 + ``backend.tools``) use agent review; non-backend user tools
  fall back to llm review.

.. code-block:: python

    # Default — llm-mode review
    agent = create_mambo_agent("gpt-4o", interrupt_on={"write": True},
                               security_review=SecurityReviewConfig())

    # Agent-mode — backend tools reviewed by agent with read-only workspace
    agent = create_mambo_agent(
        "gpt-4o",
        interrupt_on={"write": True, "edit": True},
        security_review=SecurityReviewConfig(
            review_mode="agent",
            agent_max_steps=5,
        ),
    )

    # Custom: only review "edit" with a cheaper model
    agent = create_mambo_agent(
        "gpt-4o",
        interrupt_on={"write": True, "edit": True},
        security_review=SecurityReviewConfig(
            model="gpt-4o-mini",
            review_tools=frozenset(["edit"]),
        ),
    )

    # MCP integration: review inner MCP tools via tool_unpacker
    from mambo_agents.middleware.mcp import mcp_tool_name

    mcp = MCPMiddleware(servers=[...])

    agent = create_mambo_agent(
        "gpt-4o",
        middleware=[mcp],
        interrupt_on={
            "mcp_call_tool": True,
            mcp_tool_name("filesystem", "delete_config"): True,
        },
        security_review=SecurityReviewConfig(
            review_tools=frozenset([
                mcp_tool_name("filesystem", "delete_config"),
            ]),
            tool_unpackers=[mcp.tool_unpacker],
        ),
    )

"""


from __future__ import annotations

import time
from typing import Any, Literal

from langchain.agents.middleware.human_in_the_loop import (  # type: ignore[import-untyped]
    InterruptOnConfig,
)
from langchain.agents.middleware.types import (  # type: ignore[import-untyped]
    AgentMiddleware,
    AgentState,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph._internal._constants import CONFIG_KEY_SCRATCHPAD
from langgraph.config import get_config, get_stream_writer
from langgraph.runtime import Runtime
from langgraph.types import interrupt
from langgraph.typing import ContextT
from pydantic import BaseModel, ConfigDict, Field

from mambo_agents.backends.protocol import BackendProtocol
from mambo_agents.middleware.review_agent import (
    FinalReviewResult,
    create_review_agent,
    run_review_async,
    run_review_sync,
)
from mambo_agents.middleware.tool_unpack import ToolUnpackResult

# ---------------------------------------------------------------------------
# Interrupt protocol — source identifier
# ---------------------------------------------------------------------------

INTERRUPT_SOURCE = "mambo_security_review"
"""Identifies interrupts originating from :class:`AutoSecurityReviewMiddleware`.

Present in both the outgoing HITLRequest and the expected resume value so
that consumers can route the request and the middleware can recognize its
own replay without consuming resume values meant for other components.
"""


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class SecurityReviewConfig(BaseModel):
    """Configuration for AI-assisted security review before human-in-the-loop.

    Pass this to ``create_mambo_agent(security_review=...)`` to enable the
    feature.  Without it, ``interrupt_on`` uses classic
    ``HumanInTheLoopMiddleware`` with no AI review.

    Two review modes are supported:

    - ``"llm"`` (default): single structured-output LLM call per tool call.
    - ``"agent"``: dedicated review agent with optional read-only backend
      tools (``ls``/``read``/``grep``/``glob``) that can inspect the
      workspace.  Backend tools get agent review; non-backend user tools
      fall back to llm review.

    All fields are optional — sensible defaults are applied when omitted.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    model: str | BaseChatModel | None = Field(
        default=None,
        description=(
            "Chat model to use for security review.  ``None`` (default) "
            "means reuse the main agent model.  Accepts a model name str "
            "or a ``BaseChatModel`` instance."
        ),
    )
    system_prompt: str | None = Field(
        default=None,
        description=(
            "Custom system prompt for the security reviewer.  ``None`` "
            "uses the built-in ``DEFAULT_SECURITY_REVIEW_SYSTEM_PROMPT``."
        ),
    )
    review_tools: frozenset[str] | Literal["all"] = Field(
        default="all",
        description=(
            "Which tools to AI-review before human escalation.\n"
            "- ``'all'`` (default): every tool in ``interrupt_on`` gets AI-reviewed.\n"
            "- ``frozenset[str]``: only the listed tools get AI-reviewed; "
            "others fall through to direct HITL (no AI)."
        ),
    )
    notify_on_pass: bool = Field(
        default=True,
        description=(
            "When ``True`` (default), a ``SecurityReviewPassedEvent`` custom "
            "stream event is emitted for every tool call that passes AI review. "
            "The event carries ``tool_call_id``, ``tool_name``, ``risk_level`` "
            "and ``reason`` so consumers can bind it to the exact tool call "
            "without injecting messages into the LLM context."
        ),
    )
    description_prefix: str | None = Field(
        default=None,
        description=(
            "Prefix text shown in the interrupt description when a tool call "
            "needs human approval.  ``None`` uses the built-in default "
            "(\"Tool execution requires approval\")."
        ),
    )
    review_mode: Literal["llm", "agent"] = Field(
        default="llm",
        description=(
            "Review mode:\n"
            "- ``'llm'`` (default): single LLM call with structured output.\n"
            "- ``'agent'``: full review agent with optional read-only tools "
            "that can inspect the workspace before delivering a verdict. "
            "The agent MUST call ``submit_review_verdict`` within a limited number of steps."
        ),
    )
    agent_max_steps: int = Field(
        default=5,
        description=(
            "Max steps for the review agent (only used when "
            "``review_mode='agent'``).  The agent is forced to deliver a "
            "verdict within this limit."
        ),
    )
    agent_tools: frozenset[str] | None = Field(
        default=None,
        description=(
            "Backend tool names to expose to the review agent in agent mode. "
            "``None`` (default) is equivalent to an empty ``frozenset()``: "
            "no extra tools are exposed (the review agent still gets the "
            "core read-only tools ``ls``/``read``/``grep``/``glob``). "
            "Extra tools must be explicitly listed here.  Only read-only "
            "tools should be included — the audit backend is already read-only."
        ),
    )
    tool_unpackers: list[object] | None = Field(
        default=None,
        description=(
            "Optional list of tool-unpacker callables.  Each unpacker is a "
            "function ``(tool_name, tool_args) -> ToolUnpackResult | None`` "
            "that resolves wrapped tools (e.g., ``mcp_call_tool``) into "
            "their inner tool identity.  See ``mambo_agents.middleware.mcp`` "
            "and ``example/10_mcp_security_review.py``."
        ),
    )


class SecurityReviewPassedEvent(BaseModel):
    """Custom stream event emitted when an AI security review passes.

    This event is **not** a LangGraph message — it is written via
    ``get_stream_writer()`` so it neither enters the LLM context nor
    triggers any graph execution.  Consumers receive it when streaming
    with ``stream_mode=["custom"]``.
    """

    model_config = ConfigDict(frozen=True)

    type: Literal["security_review_passed"] = "security_review_passed"
    """Discriminator for custom event routing."""

    source: Literal["security_review"] = Field(
        default="security_review",
        description="Identifies the middleware source of this event.",
    )
    tool_call_id: str = Field(
        description="The ``id`` of the tool call that passed review.",
    )
    tool_name: str = Field(
        description="Name of the reviewed tool.",
    )
    risk_level: Literal["low", "medium", "high", "critical"] = Field(
        description="AI-assessed risk level of the tool call.",
    )
    reason: str = Field(
        description="Brief explanation for why the tool call was deemed safe.",
    )
    timestamp: float = Field(
        default_factory=time.time,
        description="Unix timestamp when the event was emitted.",
    )


class SecurityReviewFailedEvent(BaseModel):
    """Custom stream event emitted when an AI security review flags a tool
    call as unsafe (before escalating to human review via ``interrupt()``).

    Emitted **before** ``interrupt()`` so the consumer receives it in the
    same stream tick, before the graph pauses.  Consumers receive it when
    streaming with ``stream_mode=["custom"]``.
    """

    model_config = ConfigDict(frozen=True)

    type: Literal["security_review_failed"] = "security_review_failed"
    """Discriminator for custom event routing."""

    source: Literal["security_review"] = Field(
        default="security_review",
        description="Identifies the middleware source of this event.",
    )
    tool_call_id: str = Field(
        description="The ``id`` of the tool call that failed review.",
    )
    tool_name: str = Field(
        description="Name of the reviewed tool.",
    )
    risk_level: Literal["low", "medium", "high", "critical"] = Field(
        description="AI-assessed risk level of the tool call.",
    )
    reason: str = Field(
        description="Explanation for why the tool call was flagged unsafe.",
    )
    timestamp: float = Field(
        default_factory=time.time,
        description="Unix timestamp when the event was emitted.",
    )


class SecurityReviewResult(BaseModel):
    """Result of an AI security review of a single tool call."""

    model_config = ConfigDict(frozen=True)

    is_safe: bool = Field(
        description=(
            "Whether the tool call is safe to execute. "
            "True = safe (auto-approve), False = unsafe (escalate to human)."
        ),
    )
    reason: str = Field(
        description="Brief explanation for the safety decision (1-2 sentences).",
    )
    risk_level: Literal["low", "medium", "high", "critical"] = Field(
        default="low",
        description="Assessed risk level of the tool call.",
    )


class ActionRequest(BaseModel):
    """An action that requires human approval before execution.

    Carries ``tool_call_id`` so consumers can precisely match each action
    to a specific tool call from the originating ``AIMessage``, even when
    multiple tool calls of the same name are interrupted together.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="The name of the action being requested.")
    args: dict[str, Any] = Field(
        description="Key-value pairs of args needed for the action.",
    )
    tool_call_id: str = Field(
        description="The ``id`` of the originating ``ToolCall``.",
    )
    description: str | None = Field(
        default=None,
        description="Human-readable description of the action to be reviewed.",
    )


class ReviewConfig(BaseModel):
    """Review configuration for a human-in-the-loop action.

    Carries ``tool_call_id`` alongside ``action_name`` for unambiguous
    action identification.
    """

    model_config = ConfigDict(frozen=True)

    action_name: str = Field(
        description="Name of the action associated with this review configuration.",
    )
    tool_call_id: str = Field(
        description="The ``id`` of the originating ``ToolCall``.",
    )
    allowed_decisions: list[str] = Field(
        description="The decisions that are allowed for this action.",
    )
    args_schema: dict[str, Any] | None = Field(
        default=None,
        description="JSON schema for the args associated with the action.",
    )


class HITLRequest(BaseModel):
    """Request for human feedback on a sequence of tool calls.

    Each ``ActionRequest`` and ``ReviewConfig`` carries a ``tool_call_id``
    so the consumer can unambiguously associate decisions with specific
    tool calls.

    The ``source`` field identifies the interrupt originator so consumers
    can route the request correctly, and the middleware can recognize its
    own resume values during replay.  Consumers **must** include
    ``"source": "mambo_security_review"`` in their resume response.
    """

    model_config = ConfigDict(frozen=True)

    source: str = Field(
        default="mambo_security_review",
        description="Identifies this interrupt as originating from "
        "AutoSecurityReviewMiddleware.  Must be echoed back in the "
        "resume value so the middleware can recognize its own replay.",
    )
    action_requests: list[ActionRequest] = Field(
        description="A list of agent actions for human review.",
    )
    review_configs: list[ReviewConfig] = Field(
        description="Review configuration for all actions under review.",
    )


# ---------------------------------------------------------------------------
# Default prompt
# ---------------------------------------------------------------------------

DEFAULT_SECURITY_REVIEW_SYSTEM_PROMPT = """You are a security reviewer for an AI coding agent.
Your job is to review tool calls the agent wants to make and determine if they pose a security risk.

## Review Guidelines

### Generally SAFE operations (is_safe=True):
- Reading files, listing directories, searching/grepping within files
- Writing/editing files within the user's project workspace
- Creating new files in project directories
- Non-destructive git operations (status, diff, log)
- Informational/read-only system queries

### Potentially UNSAFE operations (consider is_safe=False):
- Deleting files or directories (especially outside project workspace)
- Modifying system configuration files (e.g., /etc/*, Windows Registry)
- Executing shell commands that install/uninstall software
- Commands that modify system services or scheduled tasks
- Operations that access or export credentials, API keys, or secrets
- Force pushing to git repositories
- Modifying files outside the project workspace without explicit user intent
- Network operations that send data to external servers

### Decision Rules:
- When in doubt, lean toward flagging as unsafe (is_safe=False)
- If the operation only affects the user's own project files and is non-destructive, mark as safe
- If the operation could affect system stability or security, mark as unsafe
- Consider the context: a file write to a project's config file is usually safe;
  a file write to system configuration is not

Respond with your structured assessment."""


# ---------------------------------------------------------------------------
# Helper — build review messages
# ---------------------------------------------------------------------------


def _build_review_messages(
    system_prompt: str,
    tool_call: ToolCall,
    *,
    tool_description: str | None = None,
    effective_tool_name: str | None = None,
    effective_args: dict[str, Any] | None = None,
) -> list[SystemMessage | HumanMessage]:
    """Build the system + human message pair for a security review.

    Parameters
    ----------
    system_prompt:
        The system prompt for the reviewer.
    tool_call:
        The tool call to review.
    tool_description:
        Optional human-readable description of the tool's purpose.
        When provided, injected into the human message so the reviewer
        understands the intent behind the raw name + args.
    effective_tool_name:
        When the tool call is a wrapper (e.g. ``mcp_call_tool``), this is
        the *inner* tool name displayed to the reviewer.  The outer name
        is still shown as context.
    effective_args:
        The arguments for the *inner* tool when *effective_tool_name* is set.
    """
    if effective_tool_name is not None and effective_args is not None:
        desc_block = ""
        if tool_description:
            desc_block = (
                f"\n**Tool description:** {tool_description}"
            )
        return [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=(
                    f"Please review the following tool call for security risks:\n\n"
                    f"**Wrapper tool:** `{tool_call['name']}`\n"
                    f"**Effective tool:** `{effective_tool_name}`\n"
                    f"**Arguments:**\n```json\n{effective_args}\n```"
                    f"{desc_block}"
                )
            ),
        ]

    desc_block = ""
    if tool_description:
        desc_block = (
            f"\n**Tool description:** {tool_description}"
        )

    return [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=(
                f"Please review the following tool call for security risks:\n\n"
                f"**Tool name:** `{tool_call['name']}`\n"
                f"**Arguments:**\n```json\n{tool_call['args']}\n```"
                f"{desc_block}"
            )
        ),
    ]


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class AutoSecurityReviewMiddleware(
    AgentMiddleware[AgentState[Any], ContextT, Any],
):
    """Intercept ``interrupt_on`` tools; apply AI security review where configured.

    **Per-tool routing** (controlled by ``review_tools``):

    - Tool in ``review_tools`` → AI-review first.
      Safe calls auto-approved; unsafe calls escalated to human via ``interrupt()``.
    - Tool in ``interrupt_on`` but **not** in ``review_tools`` →
      direct HITL (no AI review), same as ``HumanInTheLoopMiddleware``.

    **Replay** — detected via ``CONFIG_KEY_SCRATCHPAD``.  On replay the
    AI review is **skipped entirely**.  User decisions (which carry
    ``tool_call_id`` from the ``ActionRequest``) are matched directly to
    ``last_ai_msg.tool_calls`` by id.  Auto-approved tools (no matching
    decision) are preserved as-is.

    Parameters
    ----------
    interrupt_on:
        Mapping of tool name → bool or ``InterruptOnConfig``.  Same
        semantics as ``HumanInTheLoopMiddleware``.
    model:
        Chat model used for the security review.  Set to the agent model
        by default.
    review_tools:
        Set of tool names to AI-review.  ``None`` (or empty) means every
        tool gets direct HITL (no AI).  ``"all"`` means every
        ``interrupt_on`` tool gets AI-reviewed.  A ``frozenset`` gives
        fine-grained control.
    security_review_system_prompt:
        Custom system prompt for the security reviewer.
    description_prefix:
        Prefix used when constructing human-facing action-request
        descriptions.
    """

    # ------------------------------------------------------------------
    def __init__(
        self,
        interrupt_on: dict[str, bool | InterruptOnConfig],
        *,
        model: str | BaseChatModel,
        review_tools: frozenset[str] | Literal["all"] = "all",
        security_review_system_prompt: str | None = None,
        description_prefix: str = "Tool execution requires approval",
        notify_on_pass: bool = True,
        review_mode: Literal["llm", "agent"] = "llm",
        agent_max_steps: int = 5,
        agent_backend: BackendProtocol | None = None,
        agent_tools: frozenset[str] | None = None,
        tool_descriptions: dict[str, str] | None = None,
        backend_tool_names: frozenset[str] = frozenset(),
        tool_unpackers: list[object] | None = None,
    ) -> None:
        super().__init__()

        # ---------- resolve interrupt_on configs ----------
        resolved: dict[str, InterruptOnConfig] = {}
        for tool_name, tool_config in interrupt_on.items():
            if isinstance(tool_config, bool):
                if tool_config is True:
                    resolved[tool_name] = InterruptOnConfig(
                        allowed_decisions=["approve", "edit", "reject", "respond"],
                    )
            elif tool_config.get("allowed_decisions"):
                resolved[tool_name] = tool_config
        self._interrupt_on: dict[str, InterruptOnConfig] = resolved
        self._description_prefix = description_prefix or "Tool execution requires approval"

        # ---------- review model ----------
        if isinstance(model, str):
            from langchain.chat_models import init_chat_model

            self._review_model: BaseChatModel = init_chat_model(model)
        else:
            self._review_model = model

        self._review_system_prompt = (
            security_review_system_prompt or DEFAULT_SECURITY_REVIEW_SYSTEM_PROMPT
        )

        # ---------- which tools get AI-reviewed ----------
        # Normalize: "all" → every tool in interrupt_on; otherwise validate
        # that review_tools is a subset of interrupt_on.
        interrupt_keys = frozenset(resolved.keys())
        if review_tools == "all":
            self._review_tools: frozenset[str] = interrupt_keys
        elif isinstance(review_tools, frozenset):
            unknown = review_tools - interrupt_keys
            if unknown and not tool_unpackers:
                raise ValueError(
                    f"review_tools contains tools not in interrupt_on: "
                    f"{sorted(unknown)}. review_tools must be a subset of "
                    f"interrupt_on keys."
                )
            self._review_tools: frozenset[str] = review_tools
        else:
            self._review_tools: frozenset[str] = frozenset()

        # ---------- notify-on-pass ----------
        self._notify_on_pass: bool = notify_on_pass

        # ---------- agent-mode config ----------
        self._review_mode: Literal["llm", "agent"] = review_mode
        self._agent_max_steps: int = agent_max_steps
        self._agent_backend: BackendProtocol | None = agent_backend
        self._agent_tools: frozenset[str] | None = agent_tools
        self._tool_descriptions: dict[str, str] = tool_descriptions or {}
        self._backend_tool_names: frozenset[str] = backend_tool_names
        self._tool_unpackers: list[object] = list(tool_unpackers) if tool_unpackers else []

        # ---------- agent-mode: pre-build review agent if backend available ----------
        self._cached_review_agent: object | None = None
        if self._review_mode == "agent":
            self._init_review_agent()

    # ------------------------------------------------------------------
    # Agent-mode: build the review agent once
    # ------------------------------------------------------------------

    def _init_review_agent(self) -> None:
        """Build (or rebuild) the agent-mode review agent.

        Called from ``__init__`` when ``review_mode='agent'`` and also
        lazily if the agent hasn't been built yet.
        """
        from mambo_agents.middleware.backend_tools import build_core_tools

        tools: list[BaseTool] = []
        if self._agent_backend is not None:
            # Core read-only tools are always present — never filtered
            core_tools = build_core_tools(self._agent_backend)
            # agent_tools only controls EXTRA tools from the backend.
            # None is equivalent to an empty set: extra tools must be
            # explicitly whitelisted to be exposed.
            if self._agent_tools is not None:
                extra = [
                    t for t in self._agent_backend.tools
                    if t.name in self._agent_tools
                ]
            else:
                extra = []
            tools = core_tools + extra

        # Build system prompt with path-mapping info from the backend
        from mambo_agents.middleware.review_agent import DEFAULT_REVIEW_AGENT_SYSTEM_PROMPT

        if self._review_system_prompt != DEFAULT_SECURITY_REVIEW_SYSTEM_PROMPT:
            _prompt = self._review_system_prompt
        else:
            path_info = (
                self._agent_backend.path_mapping_info
                if self._agent_backend is not None
                else {
                    "workspace_root": "/workspace",
                    "real_root": "(未知)",
                    "virtual_prefixes": "",
                    "path_mapping": "",
                }
            )
            _prompt = DEFAULT_REVIEW_AGENT_SYSTEM_PROMPT.format(
                max_steps=self._agent_max_steps,
                final_tool_name="submit_review_verdict",
                **path_info,
            )
        if self._agent_backend is not None:
            _prompt += (
                f"\n\n## 工作区信息\n\n{self._agent_backend.description}"
            )

        self._cached_review_agent = create_review_agent(
            model=self._review_model,
            system_prompt=_prompt,
            tools=tools,
            max_steps=self._agent_max_steps,
        )

    # ------------------------------------------------------------------
    # AI security review (single tool call)
    # ------------------------------------------------------------------

    def _ai_review(self, tool_call: ToolCall) -> SecurityReviewResult:
        """Ask the AI model to review a single tool call for security.

        Two modes (controlled by ``review_mode``):

        - **llm** (default): single structured-output LLM call, no tools.
        - **agent**: full review agent with optional read-only tools that
          can inspect the workspace.  The agent MUST call
          ``submit_review_verdict`` within the configured step limit.

        .. important::
            In llm mode, the review model is invoked with an **isolated
            config** (``callbacks=[]``) so that the review messages are
            **not** captured by the main agent's stream.  Agent mode is
            inherently isolated (separate graph).

        Returns
        -------
        SecurityReviewResult
            Structured assessment with ``is_safe``, ``reason`` and ``risk_level``.
        """
        if self._review_mode == "agent" and self._should_agent_review(tool_call):
            return self._ai_review_agent(tool_call)
        return self._ai_review_llm(tool_call)

    def _should_agent_review(self, tool_call: ToolCall) -> bool:
        """Determine whether *tool_call* should use agent-mode review.

        When ``review_mode='agent'`` and ``backend_tool_names`` is
        non-empty, only backend tools get agent review; all others
        fall through to LLM review.  When ``backend_tool_names`` is
        empty (default), **all** tools get agent review.
        """
        if not self._backend_tool_names:
            return True  # No routing info → agent for everything
        return tool_call["name"] in self._backend_tool_names

    def _ai_review_llm(self, tool_call: ToolCall) -> SecurityReviewResult:
        """LLM-mode review — single structured-output call."""
        effective_name, effective_args, unpacked_desc = self._try_unpack(tool_call)
        tool_desc = unpacked_desc or self._tool_descriptions.get(tool_call["name"])
        messages = _build_review_messages(
            self._review_system_prompt, tool_call,
            tool_description=tool_desc,
            effective_tool_name=effective_name if effective_name != tool_call["name"] else None,
            effective_args=effective_args if effective_name != tool_call["name"] else None,
        )

        _isolated_config: RunnableConfig = {"callbacks": []}

        try:
            structured_model = self._review_model.with_structured_output(
                SecurityReviewResult,
                method="function_calling",
            )
            response: SecurityReviewResult = structured_model.invoke(
                messages, config=_isolated_config,
            )
            return response
        except Exception as exc:
            exc_info = f"{type(exc).__name__}: {exc}"

        raw_excerpt = ""
        try:
            raw = self._review_model.invoke(messages, config=_isolated_config)
            raw_content = raw.content if hasattr(raw, "content") else str(raw)
            if isinstance(raw_content, str) and len(raw_content) > 500:
                raw_excerpt = raw_content[:500] + "..."
            elif isinstance(raw_content, str):
                raw_excerpt = raw_content
        except Exception:
            raw_excerpt = "(unable to retrieve raw response)"

        return SecurityReviewResult(
            is_safe=False,
            reason=(
                f"Security review unavailable: structured output failed. "
                f"Exception: {exc_info} | "
                f"Raw response: {raw_excerpt}"
            ),
            risk_level="high",
        )

    def _ai_review_agent(self, tool_call: ToolCall) -> SecurityReviewResult:
        """Agent-mode review — runs a mini review agent with optional tools."""
        if self._cached_review_agent is None:
            self._init_review_agent()

        agent = self._cached_review_agent
        if agent is None:
            return SecurityReviewResult(
                is_safe=False,
                reason="Review agent unavailable — agent not initialised.",
                risk_level="high",
            )

        effective_name, effective_args, unpacked_desc = self._try_unpack(tool_call)
        tool_name = effective_name if effective_name != tool_call["name"] else tool_call["name"]
        tool_args = effective_args if effective_name != tool_call["name"] else tool_call["args"]
        tool_desc = ""
        if unpacked_desc:
            tool_desc = f"\n**Tool description:** {unpacked_desc}"
        elif self._tool_descriptions and tool_name in self._tool_descriptions:
            tool_desc = f"\n**Tool description:** {self._tool_descriptions[tool_name]}"

        review_prompt = (
            f"Please review the following tool call for security risks:\n\n"
            f"**Tool name:** `{tool_name}`\n"
            f"**Arguments:**\n```json\n{tool_args}\n```"
            f"{tool_desc}\n\n"
            f"You may use read-only tools (ls, read, grep, glob) to inspect "
            f"the workspace if needed.  When done, call `submit_review_verdict`."
        )

        try:
            result: FinalReviewResult = run_review_sync(agent, review_prompt)
            return SecurityReviewResult(
                is_safe=result.is_safe,
                reason=result.reason,
                risk_level=result.risk_level,
            )
        except Exception as exc:
            return SecurityReviewResult(
                is_safe=False,
                reason=(
                    f"Review agent failed: {type(exc).__name__}: {exc}. "
                    f"Escalating to human review."
                ),
                risk_level="high",
            )

    # ------------------------------------------------------------------
    # Async AI review methods — non-blocking for asyncio event loop
    # ------------------------------------------------------------------

    async def _ai_review_llm_async(self, tool_call: ToolCall) -> SecurityReviewResult:
        """LLM-mode review — async, does NOT block the event loop."""
        effective_name, effective_args, unpacked_desc = self._try_unpack(tool_call)
        tool_desc = unpacked_desc or self._tool_descriptions.get(tool_call["name"])
        messages = _build_review_messages(
            self._review_system_prompt, tool_call,
            tool_description=tool_desc,
            effective_tool_name=effective_name if effective_name != tool_call["name"] else None,
            effective_args=effective_args if effective_name != tool_call["name"] else None,
        )

        _isolated_config: RunnableConfig = {"callbacks": []}

        try:
            structured_model = self._review_model.with_structured_output(
                SecurityReviewResult,
                method="function_calling",
            )
            response: SecurityReviewResult = await structured_model.ainvoke(
                messages, config=_isolated_config,
            )
            return response
        except Exception as exc:
            exc_info = f"{type(exc).__name__}: {exc}"

        raw_excerpt = ""
        try:
            raw = await self._review_model.ainvoke(messages, config=_isolated_config)
            raw_content = raw.content if hasattr(raw, "content") else str(raw)
            if isinstance(raw_content, str) and len(raw_content) > 500:
                raw_excerpt = raw_content[:500] + "..."
            elif isinstance(raw_content, str):
                raw_excerpt = raw_content
        except Exception:
            raw_excerpt = "(unable to retrieve raw response)"

        return SecurityReviewResult(
            is_safe=False,
            reason=(
                f"Security review unavailable: structured output failed. "
                f"Exception: {exc_info} | "
                f"Raw response: {raw_excerpt}"
            ),
            risk_level="high",
        )

    async def _ai_review_agent_async(self, tool_call: ToolCall) -> SecurityReviewResult:
        """Agent-mode review — async, does NOT block the event loop."""
        if self._cached_review_agent is None:
            self._init_review_agent()

        agent = self._cached_review_agent
        if agent is None:
            return SecurityReviewResult(
                is_safe=False,
                reason="Review agent unavailable — agent not initialised.",
                risk_level="high",
            )

        effective_name, effective_args, unpacked_desc = self._try_unpack(tool_call)
        tool_name = effective_name if effective_name != tool_call["name"] else tool_call["name"]
        tool_args = effective_args if effective_name != tool_call["name"] else tool_call["args"]
        tool_desc = ""
        if unpacked_desc:
            tool_desc = f"\n**Tool description:** {unpacked_desc}"
        elif self._tool_descriptions and tool_name in self._tool_descriptions:
            tool_desc = f"\n**Tool description:** {self._tool_descriptions[tool_name]}"

        review_prompt = (
            f"Please review the following tool call for security risks:\n\n"
            f"**Tool name:** `{tool_name}`\n"
            f"**Arguments:**\n```json\n{tool_args}\n```"
            f"{tool_desc}\n\n"
            f"You may use read-only tools (ls, read, grep, glob) to inspect "
            f"the workspace if needed.  When done, call `submit_review_verdict`."
        )

        try:
            result: FinalReviewResult = await run_review_async(agent, review_prompt)
            return SecurityReviewResult(
                is_safe=result.is_safe,
                reason=result.reason,
                risk_level=result.risk_level,
            )
        except Exception as exc:
            return SecurityReviewResult(
                is_safe=False,
                reason=(
                    f"Review agent failed: {type(exc).__name__}: {exc}. "
                    f"Escalating to human review."
                ),
                risk_level="high",
            )

    async def _ai_review_async(self, tool_call: ToolCall) -> SecurityReviewResult:
        """Async dispatcher — routes to agent or llm review without blocking."""
        if self._review_mode == "agent" and self._should_agent_review(tool_call):
            return await self._ai_review_agent_async(tool_call)
        return await self._ai_review_llm_async(tool_call)

    # ------------------------------------------------------------------
    # Emit pass notification (custom stream event — no LLM context pollution)
    # ------------------------------------------------------------------

    @staticmethod
    def _emit_pass_notification(
        tool_call: ToolCall,
        review: SecurityReviewResult,
    ) -> None:
        """Emit a ``SecurityReviewPassedEvent`` via the stream writer.

        The event is a custom stream payload (not a LangGraph message), so
        it does **not** enter the LLM context or affect agent execution.
        Consumers receive it when streaming with ``stream_mode=["custom"]``.

        When there is no active langgraph runnable context (e.g. unit
        tests that call ``after_model`` directly), the ``RuntimeError``
        from ``get_stream_writer()`` is silently caught — the notification
        is best-effort, not critical.
        """
        try:
            writer = get_stream_writer()
        except RuntimeError:
            return  # No langgraph context (e.g. unit tests)
        event = SecurityReviewPassedEvent(
            tool_call_id=tool_call["id"],
            tool_name=tool_call["name"],
            risk_level=review.risk_level,
            reason=review.reason,
        )
        writer(event.model_dump())

    @staticmethod
    def _emit_fail_notification(
        tool_call: ToolCall,
        review: SecurityReviewResult,
    ) -> None:
        """Emit a ``SecurityReviewFailedEvent`` via the stream writer
        **before** ``interrupt()`` blocks the stream.

        See ``_emit_pass_notification`` for context-semantics notes.
        """
        try:
            writer = get_stream_writer()
        except RuntimeError:
            return  # No langgraph context (e.g. unit tests)
        event = SecurityReviewFailedEvent(
            tool_call_id=tool_call["id"],
            tool_name=tool_call["name"],
            risk_level=review.risk_level,
            reason=review.reason,
        )
        writer(event.model_dump())

    # ------------------------------------------------------------------
    # Build human-facing action / config (mirrors HITL middleware)
    # ------------------------------------------------------------------

    def _build_action_and_config(
        self,
        tool_call: ToolCall,
        config: InterruptOnConfig,
    ) -> tuple[ActionRequest, ReviewConfig]:
        """Create an ``ActionRequest`` and ``ReviewConfig`` for a tool call.

        Both carry ``tool_call_id`` so the consumer can unambiguously
        associate the HITL interrupt with a specific tool call.

        The ``description`` is kept clean — AI review results are delivered
        separately via ``SecurityReviewFailedEvent`` custom stream events.

        When a tool-unpacker recognizes the call, the effective (inner)
        tool name and arguments are used for the action request so the
        human reviewer sees the real tool identity.
        """
        effective_name, effective_args, _ = self._try_unpack(tool_call)
        tool_name = effective_name if effective_name != tool_call["name"] else tool_call["name"]
        tool_args = effective_args if effective_name != tool_call["name"] else tool_call["args"]
        tool_call_id: str = tool_call["id"]

        # --- base description (from user config or defaults) ---
        description_value = config.get("description")
        if callable(description_value):
            description = description_value(tool_call, {}, Runtime())
        elif description_value is not None:
            description = description_value
        else:
            description = (
                f"{self._description_prefix}\n\n"
                f"Tool: {tool_name}\nArgs: {tool_args}"
            )

        action_request = ActionRequest(
            name=tool_name,
            args=tool_args,
            tool_call_id=tool_call_id,
            description=description,
        )
        review_config = ReviewConfig(
            action_name=tool_name,
            tool_call_id=tool_call_id,
            allowed_decisions=config["allowed_decisions"],
        )
        return action_request, review_config

    # ------------------------------------------------------------------
    # Process a single human decision
    # ------------------------------------------------------------------

    @staticmethod
    def _process_decision(
        decision: Any,
        tool_call: ToolCall,
        config: InterruptOnConfig,
    ) -> tuple[ToolCall | None, ToolMessage | None]:
        """Process a human decision and return (revised_tool_call, tool_message)."""
        allowed_decisions = config["allowed_decisions"]

        decision_type = decision.get("type", "")
        if decision_type == "approve" and "approve" in allowed_decisions:
            return tool_call, None
        if decision_type == "edit" and "edit" in allowed_decisions:
            edited_action = decision["edited_action"]
            return (
                ToolCall(
                    type="tool_call",
                    name=edited_action["name"],
                    args=edited_action["args"],
                    id=tool_call["id"],
                ),
                None,
            )
        if decision_type == "reject" and "reject" in allowed_decisions:
            content = decision.get("message") or (
                f"User rejected: `{tool_call['name']}` (id={tool_call['id']})"
            )
            tool_message = ToolMessage(
                content=content,
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
                status="error",
            )
            return tool_call, tool_message
        if decision_type == "respond" and "respond" in allowed_decisions:
            tool_message = ToolMessage(
                content=decision["message"],
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
                status="success",
            )
            return tool_call, tool_message

        msg = (
            f"Unexpected decision: {decision}. "
            f"Type '{decision_type}' not allowed for '{tool_call['name']}'. "
            f"Expected one of {allowed_decisions}."
        )
        raise ValueError(msg)

    # ------------------------------------------------------------------
    # after_model — main interception logic
    # ------------------------------------------------------------------

    def _should_ai_review(self, tool_name: str) -> bool:
        """Return ``True`` if *tool_name* should get AI-reviewed."""
        return tool_name in self._review_tools

    def _try_unpack(
        self, tool_call: ToolCall
    ) -> tuple[str, dict[str, Any], str | None]:
        """Try to unpack a wrapped tool call via registered unpackers.

        Returns ``(effective_name, effective_args, tool_description)``.
        When no unpacker recognizes the tool call the outer name / args
        are returned unchanged.
        """
        if not self._tool_unpackers:
            return tool_call["name"], tool_call["args"], None
        for unpacker in self._tool_unpackers:
            result = unpacker(tool_call["name"], tool_call["args"])
            if result is not None:
                return (
                    result.effective_tool_name,
                    result.effective_args,
                    result.tool_description,
                )
        return tool_call["name"], tool_call["args"], None

    @staticmethod
    def _get_last_ai_message(state: AgentState[Any]) -> AIMessage | None:
        """Return the last AIMessage with tool_calls, or None."""
        messages = state.get("messages", [])
        if not messages:
            return None
        last = next(
            (msg for msg in reversed(messages) if isinstance(msg, AIMessage)),
            None,
        )
        if last is None or not last.tool_calls:
            return None
        return last

    @staticmethod
    def _detect_replay(last_ai_msg: AIMessage) -> bool:
        """Return True if this middleware's own interrupt is being replayed.

        Uses **non-consuming** read of the global resume value via
        ``get_null_resume(consume=False)`` so that interrupts issued by
        other tools (e.g. a custom ``ask_user`` tool calling
        ``interrupt()`` inside its body) are **not** consumed or disturbed.

        Only resume values carrying ``"source": "mambo_security_review"``
        are recognized as belonging to this middleware.

        Additionally cross-checks that the resume decisions actually
        belong to the current batch of tool calls by matching
        ``tool_call_id``.  This prevents a stale resume value on the
        parent scratchpad chain from being mistaken for a replay of a
        *different* tool call (e.g. a retry after rejection).
        """
        try:
            conf = get_config()["configurable"]
        except RuntimeError:
            return False
        scratchpad = conf.get(CONFIG_KEY_SCRATCHPAD) # PregelScratchpad
        if scratchpad is None:
            return False
        resume_value = scratchpad.get_null_resume(False)
        if not isinstance(resume_value, dict) and scratchpad.resume:
            for r in reversed(scratchpad.resume):
                if isinstance(r, dict) and r.get("source") == INTERRUPT_SOURCE:
                    resume_value = r
                    break
        if not isinstance(resume_value, dict):
            return False
        if resume_value.get("source") != INTERRUPT_SOURCE:
            return False
        # Verify the decisions in the resume actually match the current
        # tool calls.  Without this check a consumed-but-still-readable
        # resume value on the parent scratchpad chain would cause every
        # subsequent tool call to be silently treated as a replay.
        decision_ids = {d.get("tool_call_id") for d in resume_value.get("decisions", [])}
        current_ids = {tc["id"] for tc in last_ai_msg.tool_calls}
        return bool(decision_ids & current_ids)

    def _rebuild_tool_calls(
        self,
        tool_calls: list[ToolCall],
        decisions_by_id: dict[str, dict[str, Any]],
    ) -> tuple[list[ToolCall], list[ToolMessage]]:
        """Apply human decisions to tool_calls.

        Only tool calls with an entry in *decisions_by_id* are modified;
        all others (auto-approved, not in interrupt_on) are preserved as-is.
        """
        revised: list[ToolCall] = []
        msgs: list[ToolMessage] = []

        for tc in tool_calls:
            effective_name, _, _ = self._try_unpack(tc)
            config = self._interrupt_on.get(tc["name"]) or self._interrupt_on.get(effective_name)
            if config is None:
                revised.append(tc)
                continue
            decision = decisions_by_id.get(tc["id"])
            if decision is None:
                revised.append(tc)
                continue
            new_tc, msg = self._process_decision(decision, tc, config)
            if new_tc is not None:
                revised.append(new_tc)
            if msg is not None:
                msgs.append(msg)

        return revised, msgs

    def _handle_replay(
        self, last_ai_msg: AIMessage
    ) -> tuple[list[ToolCall], list[ToolMessage]]:
        """Replay: retrieve stored decisions by tool_call_id, skip AI review."""
        decisions: list[dict[str, Any]] = interrupt({})["decisions"]
        by_id: dict[str, dict[str, Any]] = {
            d["tool_call_id"]: d for d in decisions
        }
        return self._rebuild_tool_calls(last_ai_msg.tool_calls, by_id)

    def _handle_first_run(
        self, last_ai_msg: AIMessage
    ) -> tuple[list[ToolCall], list[ToolMessage]] | None:
        """First run: AI review, then interrupt for human on unsafe tools.

        Returns None when all tools are auto-approved (no interrupt needed).
        """
        actions_for_human: list[ActionRequest] = []
        configs_for_human: list[ReviewConfig] = []

        for tc in last_ai_msg.tool_calls:
            effective_name, effective_args, _ = self._try_unpack(tc)
            config = self._interrupt_on.get(tc["name"]) or self._interrupt_on.get(effective_name)
            if config is None:
                continue

            if self._should_ai_review(tc["name"]) or self._should_ai_review(effective_name):
                review = self._ai_review(tc)
                if review.is_safe:
                    if self._notify_on_pass:
                        self._emit_pass_notification(tc, review)
                    continue
                self._emit_fail_notification(tc, review)

            action_req, review_cfg = self._build_action_and_config(tc, config)
            actions_for_human.append(action_req)
            configs_for_human.append(review_cfg)

        if not actions_for_human:
            return None

        hitl_request = HITLRequest(
            action_requests=actions_for_human,
            review_configs=configs_for_human,
        )
        decisions: list[dict[str, Any]] = interrupt(
            hitl_request.model_dump(exclude_none=True)
        )["decisions"]

        if len(decisions) != len(actions_for_human):
            raise ValueError(
                f"Mismatch: {len(decisions)} decisions vs "
                f"{len(actions_for_human)} human-review calls."
            )

        by_id: dict[str, dict[str, Any]] = {
            actions_for_human[i].tool_call_id: d
            for i, d in enumerate(decisions)
        }
        return self._rebuild_tool_calls(last_ai_msg.tool_calls, by_id)

    async def _ahandle_first_run(
        self, last_ai_msg: AIMessage
    ) -> tuple[list[ToolCall], list[ToolMessage]] | None:
        """Async variant of :meth:`_handle_first_run`.

        Uses :meth:`_ai_review_async` so the event loop is NOT blocked
        during AI review (critical for agent mode where the review agent
        may run multi-step tool-calling loops).
        """
        actions_for_human: list[ActionRequest] = []
        configs_for_human: list[ReviewConfig] = []

        for tc in last_ai_msg.tool_calls:
            effective_name, effective_args, _ = self._try_unpack(tc)
            config = self._interrupt_on.get(tc["name"]) or self._interrupt_on.get(effective_name)
            if config is None:
                continue

            if self._should_ai_review(tc["name"]) or self._should_ai_review(effective_name):
                review = await self._ai_review_async(tc)
                if review.is_safe:
                    if self._notify_on_pass:
                        self._emit_pass_notification(tc, review)
                    continue
                self._emit_fail_notification(tc, review)

            action_req, review_cfg = self._build_action_and_config(tc, config)
            actions_for_human.append(action_req)
            configs_for_human.append(review_cfg)

        if not actions_for_human:
            return None

        hitl_request = HITLRequest(
            action_requests=actions_for_human,
            review_configs=configs_for_human,
        )
        decisions: list[dict[str, Any]] = interrupt(
            hitl_request.model_dump(exclude_none=True)
        )["decisions"]

        if len(decisions) != len(actions_for_human):
            raise ValueError(
                f"Mismatch: {len(decisions)} decisions vs "
                f"{len(actions_for_human)} human-review calls."
            )

        by_id: dict[str, dict[str, Any]] = {
            actions_for_human[i].tool_call_id: d
            for i, d in enumerate(decisions)
        }
        return self._rebuild_tool_calls(last_ai_msg.tool_calls, by_id)

    def after_model(
        self,
        state: AgentState[Any],
        runtime: Runtime[ContextT],
    ) -> dict[str, Any] | None:
        """AI security review → human-in-the-loop for unsafe tool calls.

        On first execution, tools configured for AI review are screened;
        safe calls auto-approved, unsafe escalated via ``interrupt()``.
        On replay, AI review is skipped and stored decisions are applied
        by ``tool_call_id``.
        """
        last_ai_msg = self._get_last_ai_message(state)
        if last_ai_msg is None:
            return None

        if self._detect_replay(last_ai_msg):
            revised_calls, tool_msgs = self._handle_replay(last_ai_msg)
        else:
            result = self._handle_first_run(last_ai_msg)
            if result is None:
                return None
            revised_calls, tool_msgs = result

        last_ai_msg.tool_calls = revised_calls
        return {"messages": [last_ai_msg, *tool_msgs]}

    # ------------------------------------------------------------------
    # aafter_model — async passthrough
    # ------------------------------------------------------------------

    async def aafter_model(
        self,
        state: AgentState[Any],
        runtime: Runtime[ContextT],
    ) -> dict[str, Any] | None:
        """Async variant with async AI review — does NOT block the event loop.

        Uses :meth:`_ahandle_first_run` + :meth:`_ai_review_async` so the
        review agent runs via ``astream()`` without blocking the asyncio
        event loop thread.  See :meth:`after_model` for the synchronous
        equivalent (used by ``RunnableCallable`` fallback paths).
        """
        last_ai_msg = self._get_last_ai_message(state)
        if last_ai_msg is None:
            return None

        if self._detect_replay(last_ai_msg):
            revised_calls, tool_msgs = self._handle_replay(last_ai_msg)
        else:
            result = await self._ahandle_first_run(last_ai_msg)
            if result is None:
                return None
            revised_calls, tool_msgs = result

        last_ai_msg.tool_calls = revised_calls
        return {"messages": [last_ai_msg, *tool_msgs]}
