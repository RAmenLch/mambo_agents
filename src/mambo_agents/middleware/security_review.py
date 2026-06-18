"""Security review middleware — AI-based gate before human-in-the-loop.

When ``interrupt_on`` is configured for certain tools, instead of
immediately pausing for human approval, this middleware **first** asks an
AI to review each tool call for security risks.  Only tool calls that the
AI flags as **unsafe** are escalated to human review via ``interrupt()``.

When a ``backend`` is provided, the middleware can also inspect the
**actual file content** involved—particularly useful for reviewing
scripts (e.g., Python, shell, batch) that the AI agent generates.
For ``write`` / ``edit`` operations the file path and content are
read via the backend and included in the review context.

Users opt-in via the ``security_review`` parameter of ``create_mambo_agent``.

.. code-block:: python

    # Default — no AI review (classic HITL)
    agent = create_mambo_agent("gpt-4o", interrupt_on={"write": True})

    # Opt-in AI review — all interrupt_on tools
    agent = create_mambo_agent(
        "gpt-4o",
        interrupt_on={"write": True, "edit": True},
        security_review=SecurityReviewConfig(),
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

    # With backend — reads actual file content for deeper script review
    agent = create_mambo_agent(
        "gpt-4o",
        interrupt_on={"write": True, "edit": True},
        security_review=SecurityReviewConfig(),
        backend=my_backend,
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
from langgraph._internal._constants import CONFIG_KEY_SCRATCHPAD
from langgraph.config import get_config, get_stream_writer
from langgraph.runtime import Runtime
from langgraph.types import interrupt
from langgraph.typing import ContextT
from pydantic import BaseModel, ConfigDict, Field

from mambo_agents.backends.protocol import BackendProtocol


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class SecurityReviewConfig(BaseModel):
    """Configuration for AI-assisted security review before human-in-the-loop.

    Pass this to ``create_mambo_agent(security_review=...)`` to enable the
    feature.  Without it, ``interrupt_on`` uses classic
    ``HumanInTheLoopMiddleware`` with no AI review.

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

    Replaces langchain's ``ActionRequest`` TypedDict.
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

    Replaces langchain's ``ReviewConfig`` TypedDict.
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

    Replaces langchain's ``HITLRequest`` TypedDict.
    """

    model_config = ConfigDict(frozen=True)

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
# Helper – build review messages
# ---------------------------------------------------------------------------

def _build_review_messages(
    system_prompt: str,
    tool_call: ToolCall,
    *,
    tool_description: str | None = None,
    file_content: str | None = None,
    file_content_label: str = "File content",
) -> list[SystemMessage | HumanMessage]:
    """Construct messages for the security review model call.

    When *tool_description* is provided, it is included so the AI reviewer
    understands the tool's purpose and capabilities, not just its name.

    When *file_content* is provided (via the backend), the actual script/file
    body is included so the reviewer can inspect it for security risks.
    *file_content_label* specifies how the content block is titled
    (e.g. "Current file content" for edits, "New file content" for writes).
    """
    description_block = ""
    if tool_description:
        description_block = (
            f"**Tool description:** {tool_description}\n\n"
        )

    file_block = ""
    if file_content is not None:
        # Truncate very large files to avoid blowing context
        max_chars = 12000
        if len(file_content) > max_chars:
            truncated = file_content[:max_chars] + "\n\n... [truncated]"
        else:
            truncated = file_content
        file_block = (
            f"**{file_content_label}:**\n```\n{truncated}\n```\n\n"
        )

    return [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=(
                f"Please review the following tool call for security risks:\n\n"
                f"**Tool name:** `{tool_call['name']}`\n"
                f"{description_block}"
                f"**Arguments:**\n```json\n{tool_call['args']}\n```\n"
                f"{file_block}"
            ).rstrip()
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
    backend:
        Optional backend for reading file content during review.
        When provided and a tool call targets ``write`` or ``edit``,
        the actual file content is read via the backend and included
        in the review context so the AI can audit the script body.
        ``None`` (default) reviews only the tool arguments.
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
    tool_descriptions:
        Optional mapping of tool name → description for richer review
        context.  If omitted, the reviewer only sees the tool name and
        raw arguments, not its purpose or capabilities.
    """

    # ------------------------------------------------------------------
    def __init__(
        self,
        interrupt_on: dict[str, bool | InterruptOnConfig],
        *,
        model: str | BaseChatModel,
        backend: BackendProtocol | None = None,
        review_tools: frozenset[str] | Literal["all"] = "all",
        security_review_system_prompt: str | None = None,
        description_prefix: str = "Tool execution requires approval",
        tool_descriptions: dict[str, str] | None = None,
        notify_on_pass: bool = True,
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
        self._description_prefix = description_prefix

        # ---------- review model ----------
        # IMPORTANT: the review model is invoked with an isolated config
        # (``callbacks=[]``) in ``_ai_review()`` to prevent its internal
        # LLM messages (SystemMessage, HumanMessage, AIMessage) from being
        # captured by the main agent's ``stream_mode=["messages"]`` and
        # emitted as stream events.  This isolation is applied at the
        # ``invoke()`` call-site rather than at model-construction time so
        # that we keep the same model instance (important for mocking in
        # tests and for LangSmith tracing continuity).
        if isinstance(model, str):
            from langchain.chat_models import init_chat_model

            self._review_model: BaseChatModel = init_chat_model(model)
        else:
            self._review_model = model

        self._review_system_prompt = (
            security_review_system_prompt or DEFAULT_SECURITY_REVIEW_SYSTEM_PROMPT
        )

        # ---------- backend for file-content inspection ----------
        self._backend: BackendProtocol | None = backend

        # ---------- which tools get AI-reviewed ----------
        self._review_tools: frozenset[str] | Literal["all"] = review_tools

        # ---------- tool descriptions for richer review context ----------
        self._tool_descriptions: dict[str, str] = tool_descriptions or {}

        # ---------- notify-on-pass ----------
        self._notify_on_pass: bool = notify_on_pass

    # ------------------------------------------------------------------
    # Resolve file content from backend (for write / edit tools)
    # ------------------------------------------------------------------

    _FILE_PATH_KEYS: tuple[str, ...] = ("file_path", "path")

    def _resolve_file_content(
        self,
        tool_call: ToolCall,
    ) -> tuple[str | None, str | None]:
        """Read the file that the tool call targets, if a backend is available.

        Returns
        -------
        (file_content, file_content_label)
            - For ``write``: reads the *new* content from ``tool_call['args']['content']``
              (if present) directly without a round-trip — the AI is evaluating
              the content before it hits disk.
            - For ``edit``: reads the *current* file via the backend so the
              AI can see what's already on disk alongside the proposed changes.
            - For other tools: ``(None, None)``.
        """
        if self._backend is None:
            return None, None

        args = tool_call.get("args", {})
        tool_name = tool_call["name"]

        # --- write tool: the new content is already in the arguments ---
        if tool_name in ("write", "awsrite", "awrite"):
            content = args.get("content")
            if isinstance(content, str) and content.strip():
                return content, "New file content"
            # If no inline content, try reading the target path
            for key in self._FILE_PATH_KEYS:
                path = args.get(key)
                if path:
                    break
            else:
                return None, None
            result = self._backend.read(path)
            if result.error is None and result.content:
                return result.content, "Current file content"
            return None, None

        # --- edit tool: read the current file to show what's being changed ---
        if tool_name in ("edit", "aedit"):
            for key in self._FILE_PATH_KEYS:
                path = args.get(key)
                if path:
                    break
            else:
                return None, None
            result = self._backend.read(path)
            if result.error is None and result.content:
                return result.content, "Current file content (pre-edit)"
            return None, None

        # --- other tools: no file content needed ---
        return None, None

    # ------------------------------------------------------------------
    # AI security review (single tool call)
    # ------------------------------------------------------------------

    def _ai_review(self, tool_call: ToolCall) -> SecurityReviewResult:
        """Ask the AI model to review a single tool call for security.

        When a backend is configured, this method reads the actual file
        content involved in ``write`` / ``edit`` calls so the reviewer
        can inspect scripts for malicious patterns, backdoors, etc.

        .. important::
            The review model is invoked with an **isolated config**
            (``callbacks=[]``) so that the review messages (SystemMessage,
            HumanMessage, and the model's response) are **not** captured
            by the main agent's ``stream_mode=["messages"]`` and emitted
            as stream events.  This prevents the security review internals
            from polluting the agent's message stream.

        Returns
        -------
        SecurityReviewResult
            Structured assessment with ``is_safe``, ``reason`` and ``risk_level``.
        """
        tool_desc = self._tool_descriptions.get(tool_call["name"])
        file_content, file_label = self._resolve_file_content(tool_call)
        messages = _build_review_messages(
            self._review_system_prompt,
            tool_call,
            tool_description=tool_desc,
            file_content=file_content,
            file_content_label=file_label,
        )

        # Isolate the review call from the main agent's streaming context.
        # Without this, the review model's LLM messages (SystemMessage,
        # HumanMessage, AI response) would be captured by langgraph's
        # ``stream_mode=["messages"]`` and emitted to the consumer,
        # polluting the main agent's message stream.
        _isolated_config: RunnableConfig = {"callbacks": []}

        # Attempt structured output via function-calling (tools API).
        # Chosen over ``json_schema`` because DeepSeek and other third-party
        # providers do not support OpenAI's Structured Output API
        # (``response_format: {type: "json_schema", ...}``), but universally
        # support the standard function-calling / tool-calling protocol.
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

        # Fail-closed: structured output failed — gather diagnostics then
        # escalate to human review.  We try a raw invoke to capture the
        # model's unparsed response for context.
        raw_excerpt = ""
        try:
            raw = self._review_model.invoke(messages, config=_isolated_config)
            raw_content = raw.content if hasattr(raw, "content") else str(raw)
            # Truncate to avoid a giant reason string
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
        """
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
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
    # after_model — main interception logic (mixed AI-review + direct-HITL)
    # ------------------------------------------------------------------

    def _should_ai_review(self, tool_name: str) -> bool:
        """Return ``True`` if *tool_name* should get AI-reviewed."""
        if self._review_tools == "all":
            return tool_name in self._interrupt_on
        return tool_name in self._review_tools

    def after_model(
        self,
        state: AgentState[Any],
        runtime: Runtime[ContextT],
    ) -> dict[str, Any] | None:
        """Review tool calls with AI (if configured), escalate unsafe/hitl to human.

        Writes auto-approved indices to ``_reviewed_msg_ids`` via
        ``CONFIG_KEY_SEND`` before ``interrupt()``.  Send-writes are
        persisted through ``put_writes`` and applied to the channel
        via ``apply_writes`` on replay, so ``state["_reviewed_msg_ids"]``
        carries the data on the second pass.
        """
        messages = state.get("messages", [])
        if not messages:
            return None

        last_ai_msg = next(
            (msg for msg in reversed(messages) if isinstance(msg, AIMessage)),
            None,
        )
        if last_ai_msg is None or not last_ai_msg.tool_calls:
            return None

        # ---- detect replay (interrupt scratchpad introspection) ----
        try:
            conf = get_config()["configurable"]
        except RuntimeError:
            is_replay = False  # outside runnable context (e.g. unit tests)
        else:
            scratchpad = conf.get(CONFIG_KEY_SCRATCHPAD)
            is_replay = bool(scratchpad and scratchpad.resume)

        if is_replay:
            # REPLAY — the user already reviewed each action.  Skip AI
            # review entirely.  Call ``interrupt()`` with a minimal
            # payload to get the resume value (user decisions).  Match
            # decisions to tool_calls by ``tool_call_id``.
            decisions: list[dict[str, Any]] = interrupt({})["decisions"]  # type: ignore[assignment]
            by_id = {d.get("tool_call_id"): d for d in decisions}

            revised_tool_calls: list[ToolCall] = []
            synthetic_tool_msgs: list[ToolMessage] = []

            for tool_call in last_ai_msg.tool_calls:
                config = self._interrupt_on.get(tool_call["name"])
                if config is None:
                    revised_tool_calls.append(tool_call)
                    continue
                decision = by_id.get(tool_call["id"])
                if decision is None:
                    # auto-approved by AI on first run — keep as-is
                    revised_tool_calls.append(tool_call)
                    continue
                revised_call, tool_msg = self._process_decision(
                    decision, tool_call, config,
                )
                if revised_call is not None:
                    revised_tool_calls.append(revised_call)
                if tool_msg is not None:
                    synthetic_tool_msgs.append(tool_msg)

            last_ai_msg.tool_calls = revised_tool_calls
            return {"messages": [last_ai_msg, *synthetic_tool_msgs]}

        # ---- FIRST RUN: AI security review for review_tools ----
        auto_approved: dict[int, bool] = {}
        failed_reviews: list[tuple[ToolCall, SecurityReviewResult]] = []

        for idx, tool_call in enumerate(last_ai_msg.tool_calls):
            config = self._interrupt_on.get(tool_call["name"])
            if config is None:
                continue

            if self._should_ai_review(tool_call["name"]):
                review = self._ai_review(tool_call)
                if review.is_safe:
                    auto_approved[idx] = True
                    if self._notify_on_pass:
                        self._emit_pass_notification(tool_call, review)
                    continue
                failed_reviews.append((tool_call, review))

        # Emit fail events.
        for tool_call, review in failed_reviews:
            self._emit_fail_notification(tool_call, review)

        # ---- build human-review actions ----
        actions_for_human: list[ActionRequest] = []
        configs_for_human: list[ReviewConfig] = []
        human_indices: list[int] = []

        for idx, tool_call in enumerate(last_ai_msg.tool_calls):
            config = self._interrupt_on.get(tool_call["name"])
            if config is None:
                continue
            if idx in auto_approved:
                continue

            action_req, review_cfg = self._build_action_and_config(
                tool_call, config,
            )
            actions_for_human.append(action_req)
            configs_for_human.append(review_cfg)
            human_indices.append(idx)

        if not actions_for_human:
            return None

        # ---- interrupt for human review ----
        hitl_request = HITLRequest(
            action_requests=actions_for_human,
            review_configs=configs_for_human,
        )
        decisions = interrupt(hitl_request.model_dump(exclude_none=True))["decisions"]

        if len(decisions) != len(actions_for_human):
            msg = (
                f"Mismatch: {len(decisions)} decisions vs "
                f"{len(actions_for_human)} human-review calls."
            )
            raise ValueError(msg)

        # ---- rebuild tool calls ----
        revised_tool_calls: list[ToolCall] = []
        synthetic_tool_msgs: list[ToolMessage] = []

        decision_iter = iter(decisions)
        for idx, tool_call in enumerate(last_ai_msg.tool_calls):
            if idx in auto_approved:
                revised_tool_calls.append(tool_call)
            elif idx in human_indices:
                config = self._interrupt_on[tool_call["name"]]
                decision = next(decision_iter)
                revised_call, tool_msg = self._process_decision(
                    decision, tool_call, config,
                )
                if revised_call is not None:
                    revised_tool_calls.append(revised_call)
                if tool_msg is not None:
                    synthetic_tool_msgs.append(tool_msg)
            else:
                revised_tool_calls.append(tool_call)

        last_ai_msg.tool_calls = revised_tool_calls
        return {"messages": [last_ai_msg, *synthetic_tool_msgs]}

    # ------------------------------------------------------------------
    # aafter_model — async passthrough
    # ------------------------------------------------------------------

    async def aafter_model(
        self,
        state: AgentState[Any],
        runtime: Runtime[ContextT],
    ) -> dict[str, Any] | None:
        """Async variant delegates to ``after_model`` (no additional async work)."""
        return self.after_model(state, runtime)
