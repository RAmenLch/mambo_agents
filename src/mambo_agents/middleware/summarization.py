"""Summarization middleware for stateful conversation compaction.

Provides a lightweight, configurable summarization middleware that:

- Uses ``wrap_model_call`` (consistent with other mambo middleware).
- **Stateful chained summarization** — tracks a private
  ``_summarization_event`` so each compaction picks up from the correct
  cursor without re-scanning already-summarized history.
- **Preserves previous summaries** — when chained summarization occurs,
  prior summary messages are extracted and injected into the prompt as
  non-negotiable historical context that the LLM **must** retain.
- Performs **local boundary alignment** so the cutoff never slices
  through a user message.  The summary prompt's
  ``## LATEST USER INTENT`` section (injected only when the preserved
  zone contains no user messages) instructs the LLM to extract the
  current sub-task, replacing the old aggressive user-message
  retention strategy (which could block summarization entirely).
- Is **opt-in** — not automatically enabled.
- **Optional backend offload** — evicted messages are persisted to the
  configured backend at ``/conversation_history/{thread_id}.md`` before
  the summary replaces them (available when a ``BackendProtocol`` is
  supplied).

Usage::

    from mambo_agents.middleware.summarization import (
        MamboSummarizationMiddleware,
        SummarizationConfig,
    )

    agent = create_mambo_agent(
        "gpt-4o",
        summarization={
            "trigger": ("tokens", 100000),
            "keep": ("messages", 20),
        },
    )
"""

from __future__ import annotations

import asyncio
import enum
import logging
import uuid
import warnings
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Annotated, Any

from langchain.agents.middleware.summarization import (
    _DEFAULT_MESSAGES_TO_KEEP,
    _DEFAULT_TRIM_TOKEN_LIMIT,
    ContextSize,
    SummarizationMiddleware as LCSummarizationMiddleware,
    TokenCounter,
)
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
    PrivateStateAttr,
    ResponseT,
)
from langgraph.config import get_config
from langgraph.runtime import Runtime
from langgraph.typing import ContextT
from langgraph.types import Command
from langchain_core.exceptions import ContextOverflowError
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    ToolMessage,
    get_buffer_string,
)
from langchain.chat_models import BaseChatModel, init_chat_model
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import NotRequired, TypedDict

from mambo_agents.backends.protocol import BackendProtocol
from mambo_agents.backends.schemas import VirtualPath
from mambo_agents.middleware.patch_tool_calls import _unwrap_overwrite
from mambo_agents.middleware.utils._tokens import _build_default_token_counter

logger = logging.getLogger(__name__)


DEFAULT_MAMBO_SUMMARY_PROMPT = """<role>
Context Extraction Assistant
</role>

<primary_objective>
Your sole objective in this task is to extract the highest quality/most relevant context from the conversation history below.
</primary_objective>

<objective_information>
You're nearing the total number of input tokens you can accept, so you must extract the highest quality/most relevant pieces of information from your conversation history.
This context will then overwrite the conversation history presented below. Because of this, ensure the context you extract is only the most important information to continue working toward your overall goal.
</objective_information>

<instructions>
The conversation history below will be replaced with the context you extract in this step.
You want to ensure that you don't repeat any actions you've already completed, so the context you extract from the conversation history should be focused on the most important information to your overall goal.

You should structure your summary using the following sections. Each section acts as a checklist - you must populate it with relevant information or explicitly state "None" if there is nothing to report for that section:

## SESSION INTENT
What is the user's primary goal or request? What overall task are you trying to accomplish? This should be concise but complete enough to understand the purpose of the entire session.

{latest_user_intent_section}## SUMMARY
Extract and record all of the most important context from the conversation history. Include important choices, conclusions, or strategies determined during this conversation. Include the reasoning behind key decisions. Document any rejected options and why they were not pursued.

## ARTIFACTS
What artifacts, files, or resources were created, modified, or accessed during this conversation? For file modifications, list specific file paths and briefly describe the changes made to each. This section prevents silent loss of artifact information.

</instructions>

The user will message you with the full message history from which you'll extract context to create a replacement. Carefully read through it all and think deeply about what information is most important to your overall goal and should be saved:

With all of this in mind, please carefully read over the entire conversation history, and extract the most important and relevant context to replace it so that you can free up space in the conversation history.
Respond ONLY with the extracted context. Do not include any additional information, or text before or after the extracted context.

<messages>
Messages to summarize:
{messages}
</messages>"""  # noqa: E501

DEFAULT_MAMBO_CHAINED_SUMMARY_PROMPT = """<role>
Context Extraction Assistant
</role>

<primary_objective>
Your sole objective is to merge previous conversation summaries with new context into a single, comprehensive summary.
</primary_objective>

<critical_instruction>
Below are **previous summaries** from earlier parts of this conversation. Their original messages have already been permanently removed. Any facts, decisions, artifacts, or user intent documented in these summaries **MUST be preserved** in your new summary — losing them means irreversible information loss.

Treat the previous summaries as established, verified historical context. They carry equal weight to the new messages you are summarizing.
</critical_instruction>

<previous_summaries>
{previous_summaries}
</previous_summaries>

<objective_information>
You're nearing the total number of input tokens you can accept, so you must extract the highest quality/most relevant pieces of information from your conversation history.
This context will then overwrite the conversation history presented below. Because of this, ensure the context you extract is only the most important information to continue working toward your overall goal.
</objective_information>

<instructions>
The conversation history below will be replaced with the context you extract in this step.
You want to ensure that you don't repeat any actions you've already completed, so the context you extract from the conversation history should be focused on the most important information to your overall goal.

You should structure your summary using the following sections. Each section acts as a checklist - you must populate it with relevant information or explicitly state "None" if there is nothing to report for that section:

## SESSION INTENT
What is the user's primary goal or request? What overall task are you trying to accomplish? This should be concise but complete enough to understand the purpose of the entire session. **Must incorporate intent from previous summaries if present.**

{latest_user_intent_section}## SUMMARY
Extract and record all of the most important context from the conversation history. Include important choices, conclusions, or strategies determined during this conversation. Include the reasoning behind key decisions. Document any rejected options and why they were not pursued. **Must preserve all key facts, decisions, and context from previous summaries.**

## ARTIFACTS
What artifacts, files, or resources were created, modified, or accessed during this conversation? For file modifications, list specific file paths and briefly describe the changes made to each. This section prevents silent loss of artifact information. **Must include all artifacts mentioned in previous summaries.**

</instructions>

The user will message you with the full message history from which you'll extract context to create a replacement. Carefully read through it all and think deeply about what information is most important to your overall goal and should be saved:

With all of this in mind, please carefully read over the entire conversation history, and extract the most important and relevant context to replace it so that you can free up space in the conversation history.
Respond ONLY with the extracted context. Do not include any additional information, or text before or after the extracted context.

<messages>
Messages to summarize:
{messages}
</messages>"""  # noqa: E501


# ---------------------------------------------------------------------------
# Conditional summarization prompt injection
# ---------------------------------------------------------------------------

_LATEST_USER_INTENT_SECTION = """\
## LATEST USER INTENT
What is the user's most recent specific request or sub-task? This is the immediate
action the user is asking you to take right now. No user messages remain in the
conversation after summarization — this section is critical for continuing execution.

- If the latest user message is short or vague (e.g. "continue", "fix it"),
  infer the implied intent from the surrounding conversation context.
- If it is long, distill the core demand — omit background noise and redundant
  exposition.
- The goal is to capture the actionable sub-goal so that execution can continue
  seamlessly after the conversation is compacted.

"""  # Inject into the summary prompt when the preserved zone contains no user
  # messages.  Trailing double-newline ensures clean formatting when the section
  # is omitted (empty string).


# ---------------------------------------------------------------------------
# Summarization mode
# ---------------------------------------------------------------------------


class SummarizationMode(str, enum.Enum):
    """Controls when summarization is triggered during an agent run.

    Attributes:
        per_astream: Summarization happens **once** in ``before_agent``,
            before the agent execution starts.  No further checks occur
            during the entire ``astream`` / ``invoke`` cycle.  This is the
            default — it avoids redundant summarization calls when the
            agent loops through multiple model invocations (e.g. tool-use
            cycles).
        per_model_call: Summarization is checked on **every** model call
            via ``wrap_model_call``.  Useful when the agent may accumulate
            a large volume of messages within a single ``astream`` run and
            needs mid-stream compaction.
    """

    PER_ASTREAM = "per_astream"
    PER_MODEL_CALL = "per_model_call"


# ---------------------------------------------------------------------------
# Summary hook protocol (Pydantic – strict, frozen)
# ---------------------------------------------------------------------------


class SummaryHookContext(BaseModel):
    """Context passed to every summary hook when compaction occurs.

    Hooks may inspect the state and the message partitions to decide
    whether supplementary content should be appended to the summary.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    state: dict[str, Any]
    """Current full agent state (includes ``todos``, ``memory_contents``, etc.)."""

    messages_to_summarize: object
    """Messages that will be replaced by the summary (read-only)."""

    preserved_messages: object
    """Messages that will be kept after the summary message (read-only)."""


SummaryHook = Callable[[SummaryHookContext], str | None]
"""Signature of a summary hook.

Receives the full summary context and returns an optional string to append
to the summary message.  Returning ``None`` means "nothing to inject this time".
"""

# Separator injected between the AI-generated summary and hook-supplied
# content so the LLM can clearly distinguish the two sources.
_SUMMARY_HOOK_SEPARATOR = "---"


# ---------------------------------------------------------------------------
# Stateful summarization types
# ---------------------------------------------------------------------------


class SummarizationEvent(TypedDict):
    """Records a single summarization event for stateful compaction.

    Stored in the private ``_summarization_event`` field.  When chained
    summarization occurs (the N-th compaction), the event's ``cutoff_index``
    points to the absolute index in ``state["messages"]`` where the last
    summarization ended, so the next compaction knows where to pick up.

    Attributes:
        cutoff_index: Absolute index in ``state["messages"]`` — everything
            before this has been summarized into ``summary_message``.
        summary_message: The ``HumanMessage`` containing the merged summary.
        file_path: Backend path where evicted messages were offloaded,
            or ``None`` if offloading failed / was disabled.
        last_summarized_message: The last real message
            in the summarized (non-preserved) zone, or ``None`` if no such
            message exists.  Useful for consumers that need to know the
            exact boundary of the compaction window.
    """

    cutoff_index: int
    summary_message: HumanMessage
    file_path: str | None
    last_summarized_message: AnyMessage | None


class SummarizationState(AgentState):
    """Agent state extended with a private summarization tracking field.

    ``_summarization_event`` is never visible to the model — it is purely
    internal bookkeeping for chained summarization.  The key is prefixed
    with ``_`` to keep it out of the default input/output schemas.
    """

    _summarization_event: Annotated[NotRequired[SummarizationEvent | None], PrivateStateAttr]
    """Private field tracking the most recent summarization event."""


# ---------------------------------------------------------------------------
# Configuration Pydantic model
# ---------------------------------------------------------------------------


class SummarizationConfig(BaseModel):
    """Configuration for ``MamboSummarizationMiddleware`` via ``create_mambo_agent``.

    All fields are optional with sensible defaults.
    Callable fields (``token_counter``, ``summary_hooks``) require
    ``arbitrary_types_allowed=True`` since Pydantic cannot serialize callables.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    mode: SummarizationMode = Field(
        default=SummarizationMode.PER_ASTREAM,
        description=(
            "When to trigger summarization.  ``per_astream`` (default) runs once "
            "in ``before_agent``; ``per_model_call`` checks on every model call."
        ),
    )
    model: str | BaseChatModel | None = Field(
        default=None,
        description="Model to use for generating summaries. None = reuse main agent model.",
    )
    trigger: ContextSize | list[ContextSize] | None = Field(
        default=None,
        description="Threshold(s) that trigger summarization.",
    )
    keep: ContextSize = Field(
        default=("messages", 20),
        description="How many messages to retain after summarization.",
    )
    summary_prompt: str = Field(
        default=DEFAULT_MAMBO_SUMMARY_PROMPT,
        description="Prompt template with {messages} placeholder.",
    )
    chained_summary_prompt: str | None = Field(
        default=None,
        description="Prompt template for chained summarization. None = use default.",
    )
    trim_tokens_to_summarize: int | None = Field(
        default=4000,
        description="Max tokens fed to the summarization LLM call.",
    )
    token_counter: TokenCounter | None = Field(
        default=None,
        description="Custom token-counting function. None = CJK-aware auto-detect.",
    )
    chars_per_token: float | None = Field(
        default=None,
        description="Characters-per-token ratio. None = auto-detect from content.",
    )
    offload_to_backend: bool = Field(
        default=False,
        description="Enable persisting evicted messages to the backend.",
    )
    backend: BackendProtocol | None = Field(
        default=None,
        description="Backend for offloading evicted messages. None = reuse main backend.",
    )
    summary_hooks: list[SummaryHook] | None = Field(
        default=None,
        description="Optional hooks that append supplementary state to the summary.",
    )


class MamboSummarizationMiddleware(AgentMiddleware[SummarizationState, ContextT, ResponseT]):
    """Middleware that automatically compacts conversation history.

    When the total token count of messages exceeds a configurable threshold,
    older messages are summarized via an LLM call and replaced with a
    single ``HumanMessage`` containing the summary.

    Unlike the langchain ``SummarizationMiddleware``, this middleware:

    - Uses ``wrap_model_call`` and tracks compaction via a private
      ``_summarization_event`` state field — raw ``state["messages"]``
      is never mutated, but each compaction updates the cursor so the
      next one starts from the right place.
    - Injects previous summaries as required context during chained
      summarization, preventing silent loss of earlier context.
    - Performs local boundary alignment to avoid slicing through a
      user message at the cutoff edge.  The ``## LATEST USER INTENT``
      section in the summary prompt replaces the former aggressive
      user-message preservation strategy: when the preserved zone
      contains no user messages, the summarization LLM is explicitly
      instructed to extract the current sub-task intent
      (see :meth:`_adjust_cutoff_for_user_message`).
    - Token estimation is model-agnostic and CJK-aware
      (no fragile model-name heuristics).
    - **Optional backend offload** — when a ``BackendProtocol`` is provided,
      evicted messages are persisted to
      ``/conversation_history/{thread_id}.md`` before the summary replaces
      them.

    Parameters:
        model: The language model to use for generating summaries.

            Strings are resolved via ``init_chat_model()``.
        trigger: Threshold(s) that trigger summarization.
            Default: ``None`` (always checks against the ``keep`` budget).
        keep: How many messages to retain after summarization.
            Default: ``("messages", 20)``.
        summary_prompt: Prompt template with ``{messages}`` placeholder.
            Default: ``DEFAULT_MAMBO_SUMMARY_PROMPT``.
        trim_tokens_to_summarize: Max tokens fed to the summarization call.
            Default: ``4000``.
        token_counter: Token-counting callable.
            Default: CJK-aware auto-detection via ``_build_default_token_counter``.
        chars_per_token: Explicit characters-per-token ratio.
            ``None`` (default) auto-detects CJK ratio from content.
            Lower = more conservative.  Ignored when ``token_counter`` is set.
        backend: Optional ``BackendProtocol`` for offloading evicted messages.
            Default: ``None`` (no offload — evicted messages are lost).
        summary_hooks: Optional ``SummaryHook`` callables for injecting
            supplementary state into the summary message.
    """

    state_schema = SummarizationState

    def __init__(
        self,
        model: str | BaseChatModel,
        *,
        mode: SummarizationMode = SummarizationMode.PER_ASTREAM,
        trigger: ContextSize | list[ContextSize] | None = None,
        keep: ContextSize = ("messages", _DEFAULT_MESSAGES_TO_KEEP),
        summary_prompt: str = DEFAULT_MAMBO_SUMMARY_PROMPT,
        chained_summary_prompt: str = DEFAULT_MAMBO_CHAINED_SUMMARY_PROMPT,
        trim_tokens_to_summarize: int | None = _DEFAULT_TRIM_TOKEN_LIMIT,
        token_counter: TokenCounter | None = None,
        chars_per_token: float | None = None,
        offload_to_backend: bool = False,
        backend: BackendProtocol | None = None,
        summary_hooks: list[SummaryHook] | None = None,
    ) -> None:
        super().__init__()

        if offload_to_backend and backend is None:
            raise ValueError(
                "offload_to_backend=True requires a backend to be configured. "
                "Please pass a BackendProtocol instance via the 'backend' parameter, "
                "or set offload_to_backend=False to skip history persistence."
            )

        if isinstance(model, str):
            model = init_chat_model(model)

        # Resolve token_counter: explicit > chars_per_token > auto-detect
        if token_counter is None:
            token_counter = _build_default_token_counter(chars_per_token=chars_per_token)

        self._lc_helper = LCSummarizationMiddleware(
            model=model,
            trigger=trigger,
            keep=keep,
            token_counter=token_counter,
            summary_prompt=summary_prompt,
            trim_tokens_to_summarize=trim_tokens_to_summarize,
        )

        # Store both summary prompts so _create_summary can pick the right
        # one based on whether prior summaries exist.
        self._summary_prompt = summary_prompt
        self._chained_summary_prompt = chained_summary_prompt

        self._mode = mode
        self._token_counter = token_counter
        self._offload_to_backend_flag = offload_to_backend
        self._backend = backend
        self._summary_hooks: list[SummaryHook] = list(summary_hooks or [])

        # Default history path prefix — stored under a fixed virtual directory
        # on the backend (does NOT require a real filesystem root).
        self._history_path_prefix = "/.mambo/conversation_history"

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def register_hook(self, hook: SummaryHook) -> None:
        """Runtime-register an additional summary hook.

        This is useful when middleware is injected dynamically after
        ``MamboSummarizationMiddleware`` has already been constructed.
        """
        self._summary_hooks.append(hook)

    def _collect_hook_content(
        self,
        messages_to_summarize: list[AnyMessage],
        preserved_messages: list[AnyMessage],
        state: dict[str, Any],
    ) -> str:
        """Invoke all registered hooks and concatenate their outputs.

        Each hook receives a frozen :class:`SummaryHookContext` and may
        return a ``str`` to inject, or ``None`` to skip.  Hook outputs
        are separated by double newlines.

        Returns:
            Empty string if no hooks returned content, otherwise the
            concatenated section string.
        """
        if not self._summary_hooks:
            return ""

        ctx = SummaryHookContext(
            state=state,
            messages_to_summarize=messages_to_summarize,
            preserved_messages=preserved_messages,
        )
        sections: list[str] = []
        for hook in self._summary_hooks:
            result = hook(ctx)
            if result:
                sections.append(result)
        return "\n\n".join(sections) if sections else ""

    # ------------------------------------------------------------------
    # Backend offload
    # ------------------------------------------------------------------

    @staticmethod
    def _is_summary_message(msg: AnyMessage) -> bool:
        """Check if a message is a previous summarization message.

        Summary messages are ``HumanMessage`` objects with
        ``lc_source='summarization'`` in ``additional_kwargs``.  These
        should be filtered from offloads to avoid redundant storage
        during chained summarization.
        """
        if not isinstance(msg, HumanMessage):
            return False
        return msg.additional_kwargs.get("lc_source") == "summarization"

    def _filter_summary_messages(self, messages: list[AnyMessage]) -> list[AnyMessage]:
        """Filter out previous summary messages from a message list.

        When chained summarization occurs, we don't want to re-offload the
        previous summary ``HumanMessage`` since the original messages are
        already stored in the backend.
        """
        return [msg for msg in messages if not self._is_summary_message(msg)]

    def _get_thread_id(self, runtime: Any = None) -> str:
        """Extract thread_id from langgraph config, with session fallback.

        Primary path: ``get_config()`` reads ``thread_id`` from langgraph's
        ``RunnableConfig`` contextvar (the recommended API per langgraph docs).

        Fallback: when outside a runnable context (``RuntimeError``), tries
        ``getattr(runtime, "config", None)`` for backward compatibility with
        test code that injects a mock ``config`` dict.

        Args:
            runtime: The langgraph ``Runtime`` object from ``ModelRequest``,
                or ``None`` when called without runtime context.

        Returns:
            A thread ID string for use in the history file path.
        """
        try:
            config = get_config()
            thread_id = config.get("configurable", {}).get("thread_id")
            if thread_id is not None:
                return str(thread_id)
        except RuntimeError:
            # Not in a runnable context — try runtime.config as fallback
            raw_config = getattr(runtime, "config", None)
            if isinstance(raw_config, dict):
                thread_id = raw_config.get("configurable", {}).get("thread_id")
                if thread_id is not None:
                    return str(thread_id)

        generated_id = f"session_{uuid.uuid4().hex[:8]}"
        logger.debug("No thread_id found, using generated session ID: %s", generated_id)
        return generated_id

    def _get_history_path(self, runtime: Any = None) -> VirtualPath:
        """Generate path for storing conversation history.

        Returns a single file per thread that gets appended to over time.

        Args:
            runtime: Optional ``Runtime`` for extracting ``thread_id``.

        Returns:
            VirtualPath like ``'/conversation_history/{thread_id}.md'``.
        """
        thread_id = self._get_thread_id(runtime)
        return VirtualPath(f"{self._history_path_prefix}/{thread_id}.md")

    def _offload_to_backend(
        self,
        messages: list[AnyMessage],
        runtime: Any = None,
    ) -> str | None:
        """Persist messages to backend before summarization.

        Appends evicted messages to a single markdown file per thread.
        Each summarization event adds a new section with a timestamp header.

        Previous summary messages are filtered out to avoid redundant
        storage during chained summarization events.

        A ``None`` return is non-fatal; callers may proceed without the
        offloaded history.

        Args:
            messages: Messages being summarized.
            runtime: Optional ``Runtime`` for extracting ``thread_id``.

        Returns:
            The file path where history was offloaded, or ``None`` on failure.
        """
        if not self._offload_to_backend_flag:
            return None

        # At this point self._backend is guaranteed non-None (validated in __init__).
        assert self._backend is not None, (
            "Invariant broken: offload_to_backend=True but backend is None"
        )

        path = self._get_history_path(runtime)
        filtered_messages = self._filter_summary_messages(messages)

        timestamp = datetime.now(timezone.utc).isoformat()
        new_section = (
            f"## Summarized at {timestamp}\n\n"
            f"{get_buffer_string(filtered_messages)}\n\n"
        )

        # Read existing content (if any) and append.
        existing_content = ""
        try:
            responses = self._backend.download_files([path])
            if responses and responses[0].content is not None and responses[0].error is None:
                existing_content = responses[0].content.decode("utf-8")
        except Exception as e:
            logger.debug(
                "Exception reading existing history from %s (treating as new file): %s: %s",
                path,
                type(e).__name__,
                e,
            )

        combined_content = existing_content + new_section

        try:
            if existing_content:
                result = self._backend.edit(path, existing_content, combined_content)
            else:
                result = self._backend.write(path, combined_content)

            if result is None or result.error:
                logger.warning(
                    "Failed to offload conversation history to %s (%d messages): %s",
                    path,
                    len(filtered_messages),
                    "backend returned None" if result is None else result.error,
                )
                return None
        except Exception as e:
            logger.warning(
                "Exception offloading conversation history to %s (%d messages): %s: %s",
                path,
                len(filtered_messages),
                type(e).__name__,
                e,
            )
            return None
        else:
            logger.debug("Offloaded %d messages to %s", len(filtered_messages), path)
            return str(path)

    async def _aoffload_to_backend(
        self,
        messages: list[AnyMessage],
        runtime: Any = None,
    ) -> str | None:
        """Persist messages to backend before summarization (async).

        Appends evicted messages to a single markdown file per thread.
        Each summarization event adds a new section with a timestamp header.

        Previous summary messages are filtered out to avoid redundant
        storage during chained summarization events.

        A ``None`` return is non-fatal; callers may proceed without the
        offloaded history.

        Args:
            messages: Messages being summarized.
            runtime: Optional ``Runtime`` for extracting ``thread_id``.

        Returns:
            The file path where history was offloaded, or ``None`` on failure.
        """
        if not self._offload_to_backend_flag:
            return None

        # At this point self._backend is guaranteed non-None (validated in __init__).
        assert self._backend is not None, (
            "Invariant broken: offload_to_backend=True but backend is None"
        )

        path = self._get_history_path(runtime)
        filtered_messages = self._filter_summary_messages(messages)

        timestamp = datetime.now(timezone.utc).isoformat()
        new_section = (
            f"## Summarized at {timestamp}\n\n"
            f"{get_buffer_string(filtered_messages)}\n\n"
        )

        existing_content = ""
        try:
            responses = await self._backend.adownload_files([path])
            if responses and responses[0].content is not None and responses[0].error is None:
                existing_content = responses[0].content.decode("utf-8")
        except Exception as e:
            logger.debug(
                "Exception reading existing history from %s (treating as new file): %s: %s",
                path,
                type(e).__name__,
                e,
            )

        combined_content = existing_content + new_section

        try:
            if existing_content:
                result = await self._backend.aedit(path, existing_content, combined_content)
            else:
                result = await self._backend.awrite(path, combined_content)

            if result is None or result.error:
                logger.warning(
                    "Failed to offload conversation history to %s (%d messages): %s",
                    path,
                    len(filtered_messages),
                    "backend returned None" if result is None else result.error,
                )
                return None
        except Exception as e:
            logger.warning(
                "Exception offloading conversation history to %s (%d messages): %s: %s",
                path,
                len(filtered_messages),
                type(e).__name__,
                e,
            )
            return None
        else:
            logger.debug("Offloaded %d messages to %s", len(filtered_messages), path)
            return str(path)

    def _build_new_messages_with_path(self, summary: str, file_path: str | None) -> list[AnyMessage]:
        """Build the summary message with optional file path reference.

        Args:
            summary: The generated summary text.
            file_path: Path where conversation history was stored, or ``None``
                (offload was disabled or failed).

        Returns:
            List containing the summary ``HumanMessage``.
        """
        if file_path is not None:
            content = (
                "You are in the middle of a conversation that has been summarized.\n\n"
                f"The full conversation history has been saved to {file_path} "
                "should you need to refer back to it for details.\n\n"
                "A condensed summary follows:\n\n"
                f"<summary>\n{summary}\n</summary>"
            )
        else:
            content = f"Here is a summary of the conversation to date:\n\n{summary}"

        return [
            HumanMessage(
                content=content,
                additional_kwargs={"lc_source": "summarization"},
            )
        ]

    # ------------------------------------------------------------------
    # Effective-message reconstruction (stateful summarization)
    # ------------------------------------------------------------------

    def _get_effective_messages(self, request: ModelRequest) -> list[AnyMessage]:
        """Reconstruct the effective message list from raw state + event.

        When a prior summarization event exists, the effective conversation is
        the summary message followed by all messages from ``cutoff_index`` onward.
        Otherwise the raw state messages are used directly.

        Args:
            request: The model request with messages from state.

        Returns:
            The effective message list the model should see.
        """
        event = request.state.get("_summarization_event")
        return self._apply_event_to_messages(request.messages, event)

    @staticmethod
    def _apply_event_to_messages(
        messages: list[AnyMessage],
        event: SummarizationEvent | None,
    ) -> list[AnyMessage]:
        """Reconstruct effective messages from raw state + summarization event.

        Args:
            messages: Full message list from state.
            event: The ``_summarization_event`` dict, or ``None``.

        Returns:
            Effective message list: ``[summary_msg, *messages[cutoff_idx:]]``
            if an event exists, otherwise a shallow copy of ``messages``.
        """
        # Defensive unwrap: if messages leaked as Overwrite due to a
        # reducer mismatch (e.g., PlanningState bug), unwrap to the real list.
        messages = _unwrap_overwrite(messages)

        if event is None:
            return list(messages)

        try:
            summary_msg = event["summary_message"]
            cutoff_idx = event["cutoff_index"]
        except (KeyError, TypeError) as exc:
            logger.warning("Malformed _summarization_event (missing keys): %s", exc)
            return list(messages)

        if cutoff_idx > len(messages):
            logger.warning(
                "Summarization cutoff_index %d exceeds message count %d",
                cutoff_idx,
                len(messages),
            )
            return [summary_msg]

        return [summary_msg, *messages[cutoff_idx:]]

    @staticmethod
    def _compute_state_cutoff(
        event: SummarizationEvent | None,
        effective_cutoff: int,
    ) -> int:
        """Translate an effective-list cutoff index to an absolute state index.

        When a prior summarization event exists, effective index 0 is the summary
        message (not a real state message), so the ``-1`` accounts for it.  Without
        a prior event the effective and state offsets are identical.

        Args:
            event: The prior ``_summarization_event``, or ``None``.
            effective_cutoff: Cutoff index within the effective message list.

        Returns:
            The absolute cutoff index for ``state["messages"]``.
        """
        if event is None:
            return effective_cutoff
        prior_cutoff = event.get("cutoff_index")
        if not isinstance(prior_cutoff, int):
            logger.warning("Malformed _summarization_event: missing cutoff_index")
            return effective_cutoff
        return prior_cutoff + effective_cutoff - 1

    # ------------------------------------------------------------------
    # Chained summarization (preserves previous summaries)
    # ------------------------------------------------------------------

    def _build_summary_prompt(
        self,
        messages_to_summarize: list[AnyMessage],
        latest_user_intent_section: str,
    ) -> str:
        """Build the summarization prompt (including token truncation).

        Shared by :meth:`_create_summary` and :meth:`_acreate_summary`
        — the only difference between them is ``model.invoke`` vs
        ``model.ainvoke``.
        """
        prev_summaries = [m for m in messages_to_summarize if self._is_summary_message(m)]
        non_summary = [m for m in messages_to_summarize if not self._is_summary_message(m)]
        buffer = get_buffer_string(non_summary)

        if prev_summaries:
            prev_text = "\n\n".join(
                f"### Previous Summary {i + 1}\n{msg.content}"
                for i, msg in enumerate(prev_summaries)
            )
            template = self._chained_summary_prompt
            kwargs: dict[str, str] = {
                "previous_summaries": prev_text,
                "messages": buffer,
                "latest_user_intent_section": latest_user_intent_section,
            }
        else:
            template = self._summary_prompt
            kwargs = {
                "messages": buffer,
                "latest_user_intent_section": latest_user_intent_section,
            }

        prompt = template.format(**kwargs)

        # Respect trim_tokens_to_summarize: cap the total prompt tokens.
        max_tokens = self._lc_helper.trim_tokens_to_summarize
        if max_tokens is not None:
            current = self._token_counter([HumanMessage(content=prompt)])
            if current > max_tokens:
                allowed = max(0, max_tokens - self._token_counter([
                    HumanMessage(content=prompt.split("<messages>")[0]),
                ]))
                kwargs["messages"] = self._truncate_buffer_string(buffer, allowed)
                prompt = template.format(**kwargs)

        return prompt

    @staticmethod
    def _extract_response_content(response: object) -> str:
        """Extract text content from an LLM response."""
        content = response.content
        if isinstance(content, list):
            return str(content[0]) if content else ""
        return str(content) if content else ""

    def _create_summary(
        self,
        messages_to_summarize: list[AnyMessage],
        *,
        latest_user_intent_section: str = "",
    ) -> str:
        """Generate a summary using the configured prompt (sync)."""
        prompt = self._build_summary_prompt(messages_to_summarize, latest_user_intent_section)
        response = self._lc_helper.model.invoke(
            [HumanMessage(content=prompt)],
            config={"metadata": {"lc_source": "summarization"}},
        )
        return self._extract_response_content(response)

    async def _acreate_summary(
        self,
        messages_to_summarize: list[AnyMessage],
        *,
        latest_user_intent_section: str = "",
    ) -> str:
        """Async variant of :meth:`_create_summary`."""
        prompt = self._build_summary_prompt(messages_to_summarize, latest_user_intent_section)
        response = await self._lc_helper.model.ainvoke(
            [HumanMessage(content=prompt)],
            config={"metadata": {"lc_source": "summarization"}},
        )
        return self._extract_response_content(response)

    @staticmethod
    def _truncate_buffer_string(buffer: str, max_tokens: int) -> str:
        """Truncate a buffer string to approximately ``max_tokens``.

        Uses a fast character-based heuristic: :math:`chars = max\_tokens * 4`,
        and walks backward to a newline boundary for readability.  For CJK-heavy
        buffers the result may still be somewhat over-budget, but the
        summarization-model tolerance is typically wide enough.

        Args:
            buffer: The message buffer string.
            max_tokens: Target maximum token count.

        Returns:
            Truncated string ending at a newline boundary.
        """
        if max_tokens <= 0:
            return buffer
        max_chars = max_tokens * 4
        if len(buffer) <= max_chars:
            return buffer
        cutoff = buffer.rfind("\n", 0, max_chars)
        if cutoff == -1:
            cutoff = max_chars
        return buffer[:cutoff]

    # ------------------------------------------------------------------
    # before_agent (per_astream mode)
    # ------------------------------------------------------------------

    def before_agent(
        self, state: SummarizationState, runtime: Runtime[ContextT],
    ) -> dict[str, Any] | None:
        """Run summarization once before agent execution starts.

        Only active when ``self._mode == SummarizationMode.PER_ASTREAM``.
        When the token budget is exceeded, generates a summary and returns
        a ``_summarization_event`` state update so that ``wrap_model_call``
        later sees the compacted view without re-checking.

        Args:
            state: The current agent state (includes ``messages`` and
                possibly a prior ``_summarization_event``).
            runtime: The langgraph runtime context.

        Returns:
            A dict with ``_summarization_event`` if summarization occurred,
            or ``None`` if no action was needed.
        """
        if self._mode != SummarizationMode.PER_ASTREAM:
            return None
        return self._summarize_from_state(state, runtime)

    async def abefore_agent(
        self, state: SummarizationState, runtime: Runtime[ContextT],
    ) -> dict[str, Any] | None:
        """Async variant of :meth:`before_agent`."""
        if self._mode != SummarizationMode.PER_ASTREAM:
            return None
        return await self._asummarize_from_state(state, runtime)

    # ------------------------------------------------------------------
    # Shared summarization-from-state logic
    # ------------------------------------------------------------------

    def _summarize_from_state(
        self,
        state: SummarizationState,
        runtime: Runtime[ContextT],
    ) -> dict[str, Any] | None:
        """Check token budget against state messages and compact if needed.

        Reconstructs effective messages from the prior
        ``_summarization_event`` (if any), checks the token budget, and
        performs summarization when the budget is exceeded.

        Args:
            state: The full agent state.
            runtime: The langgraph runtime context.

        Returns:
            ``{"_summarization_event": new_event}`` if summarization
            occurred, or ``None``.
        """
        raw_messages = _unwrap_overwrite(state.get("messages", []))
        event = state.get("_summarization_event")
        effective = self._apply_event_to_messages(list(raw_messages), event)

        if not self._should_summarize(effective):
            return None

        cutoff_index = self._determine_cutoff_index(effective)
        if cutoff_index <= 0:
            return None

        messages_to_summarize, preserved_messages = LCSummarizationMiddleware._partition_messages(
            effective, cutoff_index
        )

        intent_section = self._resolve_intent_section(preserved_messages)

        if self._offload_to_backend_flag:
            file_path = self._offload_to_backend(messages_to_summarize, runtime)
            if file_path is None:
                msg = (
                    "Offloading conversation history to backend failed during "
                    "summarization. Older messages will not be recoverable."
                )
                logger.error(msg)
                warnings.warn(msg, stacklevel=2)
        else:
            file_path = None

        summary = self._create_summary(
            messages_to_summarize,
            latest_user_intent_section=intent_section,
        )

        new_messages, new_event = self._finalize_summarization(
            summary=summary,
            file_path=file_path,
            messages_to_summarize=messages_to_summarize,
            preserved_messages=preserved_messages,
            state=state,
            previous_event=event,
            cutoff_index=cutoff_index,
        )

        return {"_summarization_event": new_event}

    async def _asummarize_from_state(
        self,
        state: SummarizationState,
        runtime: Runtime[ContextT],
    ) -> dict[str, Any] | None:
        """Async variant of :meth:`_summarize_from_state`."""
        raw_messages = _unwrap_overwrite(state.get("messages", []))
        event = state.get("_summarization_event")
        effective = self._apply_event_to_messages(list(raw_messages), event)

        if not self._should_summarize(effective):
            return None

        cutoff_index = self._determine_cutoff_index(effective)
        if cutoff_index <= 0:
            return None

        messages_to_summarize, preserved_messages = LCSummarizationMiddleware._partition_messages(
            effective, cutoff_index
        )

        intent_section = self._resolve_intent_section(preserved_messages)

        if self._offload_to_backend_flag:
            file_path, summary = await asyncio.gather(
                self._aoffload_to_backend(messages_to_summarize, runtime),
                self._acreate_summary(
                    messages_to_summarize,
                    latest_user_intent_section=intent_section,
                ),
            )
            if file_path is None:
                msg = (
                    "Offloading conversation history to backend failed during "
                    "summarization. Older messages will not be recoverable."
                )
                logger.error(msg)
                warnings.warn(msg, stacklevel=2)
        else:
            file_path = None
            summary = await self._acreate_summary(
                messages_to_summarize,
                latest_user_intent_section=intent_section,
            )

        new_messages, new_event = self._finalize_summarization(
            summary=summary,
            file_path=file_path,
            messages_to_summarize=messages_to_summarize,
            preserved_messages=preserved_messages,
            state=state,
            previous_event=event,
            cutoff_index=cutoff_index,
        )

        return {"_summarization_event": new_event}

    # ------------------------------------------------------------------
    # Shared finalization (hook injection → event construction)
    # ------------------------------------------------------------------

    def _finalize_summarization(
        self,
        summary: str,
        file_path: str | None,
        messages_to_summarize: list[AnyMessage],
        preserved_messages: list[AnyMessage],
        state: dict[str, Any],
        previous_event: SummarizationEvent | None,
        cutoff_index: int,
    ) -> tuple[list[AnyMessage], SummarizationEvent]:
        """Build the final summary message and event.

        Shared by all four summarization paths
        (``_summarize_from_state`` / ``_asummarize_from_state`` /
        ``_summarize_and_call`` / ``_asummarize_and_call``).
        """
        hook_content = self._collect_hook_content(
            messages_to_summarize=messages_to_summarize,
            preserved_messages=preserved_messages,
            state=state,
        )

        new_messages = self._build_new_messages_with_path(summary, file_path)

        if hook_content:
            summary_msg = new_messages[0]
            base = (
                f"{summary_msg.content}"
                f"\n\n{_SUMMARY_HOOK_SEPARATOR}\n{hook_content}"
            )
            new_messages = [
                HumanMessage(
                    content=base,
                    additional_kwargs={"lc_source": "summarization"},
                )
            ]

        state_cutoff_index = self._compute_state_cutoff(previous_event, cutoff_index)
        last_msg = self._extract_last_summarized_message(messages_to_summarize)

        new_event: SummarizationEvent = {
            "cutoff_index": state_cutoff_index,
            "summary_message": new_messages[0],
            "file_path": file_path,
            "last_summarized_message": last_msg,
        }

        return new_messages, new_event

    @staticmethod
    def _resolve_intent_section(preserved_messages: list[AnyMessage]) -> str:
        """Return ``_LATEST_USER_INTENT_SECTION`` if preserved zone has no user message."""
        has_user = any(
            isinstance(msg, HumanMessage)
            and msg.additional_kwargs.get("lc_source") != "summarization"
            for msg in preserved_messages
        )
        return _LATEST_USER_INTENT_SECTION if not has_user else ""

    # ------------------------------------------------------------------
    # wrap_model_call
    # ------------------------------------------------------------------

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse | ExtendedModelResponse:
        """Potentially summarize older messages before calling the model.

        In ``per_astream`` mode (default), summarization is handled by
        :meth:`before_agent` — this method only reconstructs effective
        messages and passes them through.

        In ``per_model_call`` mode, the full summarization check runs here
        on every model invocation.

        Returns:
            ``ModelResponse`` when no summarization occurred, otherwise
            ``ExtendedModelResponse`` with ``_summarization_event`` update.
        """
        effective = self._get_effective_messages(request)

        if self._mode == SummarizationMode.PER_ASTREAM:
            # Summarization already handled in before_agent —
            # just pass through with effective messages.
            try:
                return handler(request.override(messages=effective))
            except ContextOverflowError:
                # Fall back to summarization on overflow even in per_astream mode
                return self._summarize_and_call(request, effective, handler)

        # --- per_model_call mode ---
        if not self._should_summarize(effective):
            try:
                return handler(request.override(messages=effective))
            except ContextOverflowError:
                pass  # Fall through to summarization on overflow

        return self._summarize_and_call(request, effective, handler)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse | ExtendedModelResponse:
        """Async variant of :meth:`wrap_model_call`."""
        effective = self._get_effective_messages(request)

        if self._mode == SummarizationMode.PER_ASTREAM:
            try:
                return await handler(request.override(messages=effective))
            except ContextOverflowError:
                return await self._asummarize_and_call(request, effective, handler)

        # --- per_model_call mode ---
        if not self._should_summarize(effective):
            try:
                return await handler(request.override(messages=effective))
            except ContextOverflowError:
                pass  # Fall through to summarization on overflow

        return await self._asummarize_and_call(request, effective, handler)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _should_summarize(self, messages: list[AnyMessage]) -> bool:
        """Wrap the langchain helper's ``_should_summarize``.

        Counts tokens (including system message for accuracy) and delegates
        to the trigger-condition logic in ``LCSummarizationMiddleware``.
        """
        total_tokens = self._lc_helper.token_counter(messages)
        return self._lc_helper._should_summarize(messages, total_tokens)

    def _determine_cutoff_index(self, messages: list[AnyMessage]) -> int:
        """Compute cutoff index with local user-message boundary alignment.

        First delegates to the langchain helper (which protects AI/Tool
        message pairs from being split), then performs a local boundary
        check to avoid slicing through a user message at the cutoff edge.
        """
        cutoff = self._lc_helper._determine_cutoff_index(messages)
        if cutoff <= 0:
            return cutoff
        return self._adjust_cutoff_for_user_message(messages, cutoff)

    @staticmethod
    def _extract_last_summarized_message(
        messages: list[AnyMessage],
    ) -> AnyMessage | None:
        """Extract the last real message from the summarized zone.

        Walks the message list in reverse and returns the first message
        that is **not** a summary marker (``lc_source="summarization"``).
        This covers ``HumanMessage``, ``AIMessage`` (with or without
        tool_calls), and ``ToolMessage``.

        Args:
            messages: The messages in the summarized (non-preserved) zone.

        Returns:
            The last real message in the summarized zone, or ``None``
            if the zone is empty or contains only summary markers.
        """
        for msg in reversed(messages):
            if not (
                isinstance(msg, HumanMessage)
                and msg.additional_kwargs.get("lc_source") == "summarization"
            ):
                return msg
        return None

    @staticmethod
    def _adjust_cutoff_for_user_message(
        messages: list[AnyMessage],
        cutoff_index: int,
    ) -> int:
        """Ensure the cutoff boundary does not slice through a user message.

        This only performs a local boundary check: if the message
        immediately *before* the cutoff is a non-summary HumanMessage,
        the cutoff is moved back by one to avoid splitting user context
        mid-message.

        When the preserved zone contains no user messages,
        ``_LATEST_USER_INTENT_SECTION`` is injected into the summary
        prompt so the LLM extracts the current sub-task intent.

        Previous summary messages (tagged ``lc_source="summarization"``)
        are skipped — they are not user messages.

        Args:
            messages: Full message list.
            cutoff_index: Original cutoff (from langchain helper).

        Returns:
            Adjusted ``cutoff_index`` — at most 1 less than the input.
        """
        if cutoff_index <= 0:
            return cutoff_index

        boundary_msg = messages[cutoff_index - 1]
        if isinstance(boundary_msg, HumanMessage) and (
            boundary_msg.additional_kwargs.get("lc_source") != "summarization"
        ):
            return cutoff_index - 1

        return cutoff_index

    def _summarize_and_call(
        self,
        request: ModelRequest,
        effective_messages: list[AnyMessage],
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ExtendedModelResponse:
        """Run the summarization pipeline and return an ExtendedModelResponse.

        Args:
            request: The original model request.
            effective_messages: Effective messages (already reconstructed
                from prior ``_summarization_event``).
            handler: The downstream handler.

        Returns:
            ``ExtendedModelResponse`` with a new ``_summarization_event``.
        """
        cutoff_index = self._determine_cutoff_index(effective_messages)

        if cutoff_index <= 0:
            response = handler(request.override(messages=effective_messages))
            return ExtendedModelResponse(model_response=response)

        messages_to_summarize, preserved_messages = LCSummarizationMiddleware._partition_messages(
            effective_messages, cutoff_index
        )

        intent_section = self._resolve_intent_section(preserved_messages)

        if self._offload_to_backend_flag:
            file_path = self._offload_to_backend(messages_to_summarize, request.runtime)
            if file_path is None:
                msg = (
                    "Offloading conversation history to backend failed during "
                    "summarization. Older messages will not be recoverable."
                )
                logger.error(msg)
                warnings.warn(msg, stacklevel=2)
        else:
            file_path = None

        summary = self._create_summary(
            messages_to_summarize,
            latest_user_intent_section=intent_section,
        )

        new_messages, new_event = self._finalize_summarization(
            summary=summary,
            file_path=file_path,
            messages_to_summarize=messages_to_summarize,
            preserved_messages=preserved_messages,
            state=request.state,
            previous_event=request.state.get("_summarization_event"),
            cutoff_index=cutoff_index,
        )

        modified_messages = [*new_messages, *preserved_messages]
        response = handler(request.override(messages=modified_messages))

        return ExtendedModelResponse(
            model_response=response,
            command=Command(update={"_summarization_event": new_event}),
        )

    async def _asummarize_and_call(
        self,
        request: ModelRequest,
        effective_messages: list[AnyMessage],
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ExtendedModelResponse:
        """Async variant of :meth:`_summarize_and_call`."""
        cutoff_index = self._determine_cutoff_index(effective_messages)

        if cutoff_index <= 0:
            response = await handler(request.override(messages=effective_messages))
            return ExtendedModelResponse(model_response=response)

        messages_to_summarize, preserved_messages = LCSummarizationMiddleware._partition_messages(
            effective_messages, cutoff_index
        )

        intent_section = self._resolve_intent_section(preserved_messages)

        # Offload to backend and generate summary concurrently — they are
        # independent.  Both coroutines catch all exceptions internally and
        # return None / empty string on failure, so asyncio.gather is safe.
        if self._offload_to_backend_flag:
            file_path, summary = await asyncio.gather(
                self._aoffload_to_backend(messages_to_summarize, request.runtime),
                self._acreate_summary(
                    messages_to_summarize,
                    latest_user_intent_section=intent_section,
                ),
            )
            if file_path is None:
                msg = (
                    "Offloading conversation history to backend failed during "
                    "summarization. Older messages will not be recoverable."
                )
                logger.error(msg)
                warnings.warn(msg, stacklevel=2)
        else:
            file_path = None
            summary = await self._acreate_summary(
                messages_to_summarize,
                latest_user_intent_section=intent_section,
            )

        new_messages, new_event = self._finalize_summarization(
            summary=summary,
            file_path=file_path,
            messages_to_summarize=messages_to_summarize,
            preserved_messages=preserved_messages,
            state=request.state,
            previous_event=request.state.get("_summarization_event"),
            cutoff_index=cutoff_index,
        )

        modified_messages = [*new_messages, *preserved_messages]
        response = await handler(request.override(messages=modified_messages))

        return ExtendedModelResponse(
            model_response=response,
            command=Command(update={"_summarization_event": new_event}),
        )
