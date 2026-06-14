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

from typing import Any, Literal

from langchain.agents.middleware.human_in_the_loop import (  # type: ignore[import-untyped]
    ActionRequest,
    HITLRequest,
    InterruptOnConfig,
    ReviewConfig,
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

        # Attempt structured-output (JSON-schema mode) for reliable parsing.
        try:
            structured_model = self._review_model.with_structured_output(
                SecurityReviewResult,
                method="json_schema",
            )
            response: SecurityReviewResult = structured_model.invoke(
                messages, config=_isolated_config,
            )
            return response
        except Exception:
            pass

        # Fallback: try without explicit method
        try:
            structured_model = self._review_model.with_structured_output(
                SecurityReviewResult,
            )
            response = structured_model.invoke(
                messages, config=_isolated_config,
            )
            return response
        except Exception:
            pass

        # Last-resort fallback: parse raw text
        raw = self._review_model.invoke(messages, config=_isolated_config)
        content = raw.content if hasattr(raw, "content") else str(raw)
        content_lower = content.lower()

        if "unsafe" in content_lower or "not safe" in content_lower:
            return SecurityReviewResult(
                is_safe=False,
                reason=content.strip()[:200],
                risk_level="high",
            )
        return SecurityReviewResult(
            is_safe=True,
            reason="Auto-parsed as safe from text response",
            risk_level="low",
        )

    # ------------------------------------------------------------------
    # Build human-facing action / config (mirrors HITL middleware)
    # ------------------------------------------------------------------

    def _build_action_and_config(
        self,
        tool_call: ToolCall,
        config: InterruptOnConfig,
        *,
        ai_review: SecurityReviewResult | None = None,
    ) -> tuple[ActionRequest, ReviewConfig]:
        """Create an ``ActionRequest`` and ``ReviewConfig`` for a tool call.

        When *ai_review* is provided (AI pre-screened & flagged unsafe), its
        ``reason`` and ``risk_level`` are prepended to the description so the
        human reviewer can see the AI's assessment.
        """
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

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

        # --- inject AI review assessment if available ---
        if ai_review is not None:
            ai_header = (
                f"\n\n{'─' * 50}\n"
                f"🤖 AI 安全审查: **UNSAFE**\n"
                f"   风险等级: {ai_review.risk_level.upper()}\n"
                f"   判断理由: {ai_review.reason}\n"
                f"{'─' * 50}"
            )
            description = ai_header + "\n\n" + description

        action_request = ActionRequest(
            name=tool_name,
            args=tool_args,
            description=description,
        )
        review_config = ReviewConfig(
            action_name=tool_name,
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

        Steps
        -----
        1. Locate the last ``AIMessage`` and its ``tool_calls``.
        2. Identify which calls match ``interrupt_on``.
        3. For tools in ``review_tools`` → AI review → auto-approve safe ones.
        4. For tools not in ``review_tools`` → direct HITL (no AI).
        5. Combine all unsafe/direct-HITL calls into one ``HITLRequest`` → ``interrupt()``.
        6. Process human decisions and return updated messages.
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

        # ---- buckets ----
        auto_approved: dict[int, bool] = {}  # index → ai-approved
        actions_for_human: list[ActionRequest] = []
        configs_for_human: list[ReviewConfig] = []
        human_indices: list[int] = []  # tool_call indices that need human review

        for idx, tool_call in enumerate(last_ai_msg.tool_calls):
            config = self._interrupt_on.get(tool_call["name"])
            if config is None:
                # Not in interrupt_on → pass through
                continue

            ai_review_result: SecurityReviewResult | None = None

            if self._should_ai_review(tool_call["name"]):
                # ---- AI review path ----
                review = self._ai_review(tool_call)
                if review.is_safe:
                    auto_approved[idx] = True
                    continue
                # AI flagged unsafe → save reason for human, then fall through to HITL
                ai_review_result = review

            # ---- direct HITL (or AI-flagged-unsafe) ----
            action_req, review_cfg = self._build_action_and_config(
                tool_call, config, ai_review=ai_review_result,
            )
            actions_for_human.append(action_req)
            configs_for_human.append(review_cfg)
            human_indices.append(idx)

        if not actions_for_human:
            # All either auto-approved or pass-through
            return None

        # ---- interrupt for human review ----
        hitl_request = HITLRequest(
            action_requests=actions_for_human,
            review_configs=configs_for_human,
        )
        decisions = interrupt(hitl_request)["decisions"]

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
                # Auto-approved by AI — keep as-is
                revised_tool_calls.append(tool_call)
            elif idx in human_indices:
                # Human-reviewed
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
                # Not in interrupt_on — keep as-is
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
