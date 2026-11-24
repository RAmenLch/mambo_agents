"""Middleware for providing subagents to an agent via a ``task`` tool.

Supports streaming subagent internals as custom events so callers can
track subagent progress in real time, including parallel task support
via ``tool_call_id``.

Event granularity
-----------------

The ``event_granularity`` parameter controls how much detail is emitted
via ``stream_writer`` custom events:

* ``"messages"`` — finest: every LLM token chunk (includes
  ``AIMessageChunk`` with token-level streaming)
* ``"updates"`` — medium (default): per-node state updates (e.g. one
  event per tool call / tool result / agent turn)
* ``"values"`` — coarsest: full state snapshot after each graph step

All custom events carry a ``tool_call_id`` so parallel subagent tasks can
be distinguished by the consumer.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

from langchain.agents.factory import create_agent as _langchain_create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelRequest,
    ModelResponse,
    ResponseT,
)
from langchain.tools import BaseTool, ToolRuntime
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import StructuredTool
from langgraph.config import get_stream_writer
from langgraph.errors import GraphBubbleUp
from langgraph.types import Command, Overwrite
from pydantic import BaseModel, ConfigDict, Field

from mambo_agents.backends.protocol import BackendProtocol
from mambo_agents.middleware.subagent_snapshot import (
    SubAgentSnapshotNode,
    adelete_node,
    aget_node,
    alist_nodes,
    aput_node,
    build_continuation_prompt,
    build_delta,
    build_failure_content,
    classify_failure,
    continuation_metadata,
    count_root_failures,
    delete_node,
    get_node,
    list_nodes,
    now_iso,
    put_node,
    to_serialized,
    validate_continuation,
)
from mambo_agents.middleware.tool_call_error import ToolCallErrorMiddleware
from mambo_agents.utils.cancellation import is_cancellation_related

logger = logging.getLogger(__name__)

#: Returned when a ``task`` call has no ``tool_call_id`` — without it no
#: continuation id can be issued, and raising would just be re-wrapped by
#: ``ToolCallErrorMiddleware``.
_NO_TOOL_CALL_ID = (
    "Subagent invocation requires a tool_call_id; this call cannot be "
    "recorded or continued."
)

# ---------------------------------------------------------------------------
# Pydantic models for subagent specification
# ---------------------------------------------------------------------------


class SubAgent(BaseModel):
    """Specification for a subagent to be compiled by the middleware.

    Required fields:
        name: Unique identifier for the subagent.
            The main agent uses this name when calling the ``task()`` tool.
        description: What this subagent does.
            Be specific – the main agent uses this to decide when to delegate.
        system_prompt: Instructions for the subagent.

    Optional fields:
        tools: Tools the subagent can use. Default: empty list.
        model: Override the main agent's model.
        middleware: Additional middleware for this subagent.
        interrupt_on: Human-in-the-loop config for specific tools.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    name: str
    description: str
    system_prompt: str
    tools: Sequence[BaseTool | Callable | dict[str, Any]] = Field(default_factory=list)
    model: str | BaseChatModel | None = None
    middleware: list[AgentMiddleware] = Field(default_factory=list)
    interrupt_on: dict[str, Any] | None = None


class CompiledSubAgent(BaseModel):
    """A pre-compiled subagent.

    The runnable's state schema must include a ``'messages'`` key so the
    subagent can communicate results back to the main agent.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    name: str
    description: str
    runnable: Runnable


# ---------------------------------------------------------------------------
# Event granularity
# ---------------------------------------------------------------------------

EventGranularity = Literal["messages", "updates", "values"]
"""Streaming event granularity for subagent internals.

- ``"messages"`` — token-level LLM output (finest)
- ``"updates"`` — per-node state updates (default)
- ``"values"`` — full state snapshot per step (coarsest)
"""


class SubAgentConfig(BaseModel):
    """Top-level configuration for subagents via ``create_mambo_agent``.

    Accepts either this config object or a plain
    ``Sequence[SubAgent | CompiledSubAgent]``.  Fields that are ``None``
    fall back to the corresponding ``create_mambo_agent`` parameters
    (``include_general_purpose``, ``subagent_event_granularity``).
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    subagents: Sequence[SubAgent | CompiledSubAgent] | None = Field(
        default=None,
        description="Subagent specs (``SubAgent`` dicts or compiled runnables).",
    )
    include_general_purpose: bool | None = Field(
        default=None,
        description=(
            "Whether to add the ``\"general-purpose\"`` subagent.  ``None`` "
            "falls back to the ``include_general_purpose`` parameter."
        ),
    )
    event_granularity: EventGranularity | None = Field(
        default=None,
        description=(
            "Streaming detail level for subagent custom events.  ``None`` "
            "falls back to ``subagent_event_granularity``."
        ),
    )
    task_system_prompt: str | None = Field(
        default=None,
        description=(
            "System prompt for the ``task`` tool.  ``None`` uses the "
            "middleware default ``TASK_SYSTEM_PROMPT``."
        ),
    )
    task_description: str | None = Field(
        default=None,
        description="Optional description appended to the ``task`` tool.",
    )
    catch_errors: bool | None = Field(
        default=None,
        description=(
            "Whether a failing subagent is turned into an error ``ToolMessage`` "
            "instead of aborting the main turn.  ``None`` uses the middleware "
            "default (``True``)."
        ),
    )
    max_continue_attempts: int | None = Field(
        default=None,
        description="Maximum chain depth a ``continue_id`` may reach.",
    )
    max_children: int | None = Field(
        default=None,
        description="Maximum number of continuations started from one breakpoint.",
    )
    max_root_failures_per_thread: int | None = Field(
        default=None,
        description=(
            "Maximum number of failed chain roots kept per conversation. "
            "Only affects snapshot bookkeeping, never task execution."
        ),
    )


# ---------------------------------------------------------------------------
# State keys excluded when passing / returning subagent state
# ---------------------------------------------------------------------------

_EXCLUDED_STATE_KEYS = {
    "messages",
    "todos",
    "plans",
    "structured_response",
    "skills_metadata",
    "skills_load_errors",
    "memory_contents",
    "goal",
}


# ---------------------------------------------------------------------------
# Task tool schemas & prompts
# ---------------------------------------------------------------------------


class TaskToolSchema(BaseModel):
    """Input schema for the ``task`` tool.

    ``extra='allow'`` is critical: the langgraph ``ToolNode`` injects
    ``runtime`` and ``type`` fields after schema validation.  Without
    this, ``BaseTool._parse_input()`` drops them via Pydantic.
    """

    model_config = ConfigDict(extra="allow")

    description: str = Field(
        default="",
        description=(
            "A detailed description of the task for the subagent to perform "
            "autonomously. Include all necessary context and specify the "
            "expected output format. Required for a new task; optional when "
            "``continue_id`` is given (then it is the extra instruction)."
        ),
    )
    subagent_type: str = Field(
        description=(
            "The type of subagent to use. Must be one of the available agent "
            "types listed in the tool description."
        ),
    )
    continue_id: str | None = Field(
        default=None,
        description=(
            "Optional breakpoint id returned by a previous failure "
            "(`Continuation saved: node_id=...`). Use it to resume without "
            "redoing completed steps."
        ),
    )


TASK_TOOL_DESCRIPTION = """Launch an ephemeral subagent to handle complex, multi-step independent tasks with isolated context windows.

Available agent types and the tools they have access to:
{available_agents}

When using the Task tool, you must specify a subagent_type parameter to select which agent type to use.

## Usage notes:
1. Launch multiple agents concurrently whenever possible, to maximize performance; to do that, use a single message with multiple tool uses
2. When the agent is done, it will return a single message back to you. The result returned by the agent is not visible to the user. To show the user the result, you should send a text message back to the user with a concise summary of the result.
3. Each agent invocation starts a fresh context window. You cannot send additional messages mid-run, nor can the agent talk back outside of its final report. Therefore your prompt must contain a highly detailed task description and specify exactly what the agent should return. If a run fails, the result carries a `continue_id`: pass it back as `continue_id` to resume from that breakpoint (completed steps are not redone) instead of starting over.
4. The agent's outputs should generally be trusted
5. Clearly tell the agent whether you expect it to create content, perform analysis, or just do research (search, file reads, web fetches, etc.), since it is not aware of the user's intent
6. If the agent description mentions that it should be used proactively, then you should try your best to use it without the user having to ask for it first. Use your judgement.
7. When only the general-purpose agent is provided, you should use it for all tasks. It is great for isolating context and token usage, and completing specific, complex tasks, as it has all the same capabilities as the main agent.

### Example usage of the general-purpose agent:

<example_agent_descriptions>
"general-purpose": use this agent for general purpose tasks, it has access to all tools as the main agent.
</example_agent_descriptions>

<example>
User: "I want to conduct research on the accomplishments of Lebron James, Michael Jordan, and Kobe Bryant, and then compare them."
Assistant: *Uses the task tool in parallel to conduct isolated research on each of the three players*
Assistant: *Synthesizes the results of the three isolated research tasks and responds to the User*
<commentary>
Research is a complex, multi-step task in it of itself.
The research of each individual player is not dependent on the research of the other players.
The assistant uses the task tool to break down the complex objective into three isolated tasks.
Each research task only needs to worry about context and tokens about one player, then returns synthesized information about each player as the Tool Result.
This means each research task can dive deep and spend tokens and context deeply researching each player, but the final result is synthesized information, and saves us tokens in the long run when comparing the players to each other.
</commentary>
</example>

<example>
User: "Analyze a single large code repository for security vulnerabilities and generate a report."
Assistant: *Launches a single `task` subagent for the repository analysis*
Assistant: *Receives report and integrates results into final summary*
<commentary>
Subagent is used to isolate a large, context-heavy task, even though there is only one. This prevents the main thread from being overloaded with details.
If the user then asks followup questions, we have a concise report to reference instead of the entire history of analysis and tool calls, which is good and saves us time and money.
</commentary>
</example>

<example>
User: "Schedule two meetings for me and prepare agendas for each."
Assistant: *Calls the task tool in parallel to launch two `task` subagents (one per meeting) to prepare agendas*
Assistant: *Returns final schedules and agendas*
<commentary>
Tasks are simple individually, but subagents help silo agenda preparation.
Each subagent only needs to worry about the agenda for one meeting.
</commentary>
</example>

<example>
User: "I want to order a pizza from Dominos, order a burger from McDonald's, and order a salad from Subway."
Assistant: *Calls tools directly in parallel to order a pizza from Dominos, a burger from McDonald's, and a salad from Subway*
<commentary>
The assistant did not use the task tool because the objective is super simple and clear and only requires a few trivial tool calls.
It is better to just complete the task directly and NOT use the `task` tool.
</commentary>
</example>

### Example usage with custom agents:

<example_agent_descriptions>
"content-reviewer": use this agent after you are done creating significant content or documents
"greeting-responder": use this agent when to respond to user greetings with a friendly joke
"research-analyst": use this agent to conduct thorough research on complex topics
</example_agent_descriptions>

<example>
user: "Please write a function that checks if a number is prime"
assistant: Sure let me write a function that checks if a number is prime
assistant: First let me use the Write tool to write a function that checks if a number is prime
assistant: I'm going to use the Write tool to write the following code:
<code>
function isPrime(n) {{
  if (n <= 1) return false
  for (let i = 2; i * i <= n; i++) {{
    if (n % i === 0) return false
  }}
  return true
}}
</code>
<commentary>
Since significant content was created and the task was completed, now use the content-reviewer agent to review the work
</commentary>
assistant: Now let me use the content-reviewer agent to review the code
assistant: Uses the Task tool to launch with the content-reviewer agent
</example>

<example>
user: "Can you help me research the environmental impact of different renewable energy sources and create a comprehensive report?"
<commentary>
This is a complex research task that would benefit from using the research-analyst agent to conduct thorough analysis
</commentary>
assistant: I'll help you research the environmental impact of renewable energy sources. Let me use the research-analyst agent to conduct comprehensive research on this topic.
assistant: Uses the Task tool to launch with the research-analyst agent, providing detailed instructions about what research to conduct and what format the report should take
</example>

<example>
user: "Hello"
<commentary>
Since the user is greeting, use the greeting-responder agent to respond with a friendly joke
</commentary>
assistant: "I'm going to use the Task tool to launch with the greeting-responder agent"
</example>"""  # noqa: E501


TASK_SYSTEM_PROMPT = """## `task` (subagent spawner)

You have access to a `task` tool to launch short-lived subagents that handle isolated tasks. These agents are ephemeral — they live only for the duration of the task and return a single result.

When to use the task tool:
- When a task is complex and multi-step, and can be fully delegated in isolation
- When a task is independent of other tasks and can run in parallel
- When a task requires focused reasoning or heavy token/context usage that would bloat the orchestrator thread
- When sandboxing improves reliability (e.g. code execution, structured searches, data formatting)
- When you only care about the output of the subagent, and not the intermediate steps (ex. performing a lot of research and then returned a synthesized report, performing a series of computations or lookups to achieve a concise, relevant answer.)

Subagent lifecycle:
1. **Spawn** → Provide clear role, instructions, and expected output
2. **Run** → The subagent completes the task autonomously
3. **Return** → The subagent provides a single structured result
4. **Reconcile** → Incorporate or synthesize the result into the main thread

When NOT to use the task tool:
- If you need to see the intermediate reasoning or steps after the subagent has completed (the task tool hides them)
- If the task is trivial (a few tool calls or simple lookup)
- If delegating does not reduce token usage, complexity, or context switching
- If splitting would add latency without benefit

## Important Task Tool Usage Notes to Remember
- Whenever possible, parallelize the work that you do. This is true for both tool_calls, and for tasks. Whenever you have independent steps to complete - make tool_calls, or kick off tasks (subagents) in parallel to accomplish them faster. This saves time for the user, which is incredibly important.
- Remember to use the `task` tool to silo independent tasks within a multi-part objective.
- You should use the `task` tool whenever you have a complex task that will take multiple steps, and is independent from other tasks that the agent needs to complete. These agents are highly competent and efficient.

## Error recovery & continuing
A failing task returns a normal tool result with `status="error"`; the rest of your turn keeps working. Read the message:

- If it says `Continuation saved: node_id="<id>"`, you may resume with
  `task(subagent_type="...", continue_id="<id>")` — the subagent picks up where it
  stopped and does **not** redo completed steps. Add a `description` when you want to
  correct course (e.g. "skip the web search, use the files you already read").
- If the result says the error is **not retryable**, or `available=false` (continuation
  exhausted / too many attempts), do **not** retry the same call: change strategy,
  pick another subagent type, or report the failure to the user.
- Never fabricate a `continue_id`; only use the one returned by a previous failure."""  # noqa: E501


DEFAULT_GENERAL_PURPOSE_DESCRIPTION = (
    "General-purpose agent for researching complex questions, searching for "
    "files and content, and executing multi-step tasks. When you are searching "
    "for a keyword or file and are not confident that you will find the right "
    "match in the first few tries use this agent to perform the search for you. "
    "This agent has access to all tools as the main agent."
)

DEFAULT_SUBAGENT_PROMPT = (
    "In order to complete the objective that the user asks of you, you have "
    "access to a number of standard tools."
)

GENERAL_PURPOSE_NAME = "general-purpose"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _append_to_system_message(
    request: ModelRequest,
    extra: str,
) -> ModelRequest:
    """Append *extra* text to the system message in a ModelRequest."""
    from langchain_core.messages import SystemMessage

    existing = request.system_message
    if existing is None:
        return request.override(system_message=SystemMessage(content=extra))

    content = existing.content
    if isinstance(content, str):
        new_content = content + "\n\n" + extra
    elif isinstance(content, list):
        new_content = [*content, {"type": "text", "text": "\n\n" + extra}]
    else:
        new_content = f"{content}\n\n{extra}"
    return request.override(system_message=SystemMessage(content=new_content))


def _return_command_with_state_update(
    result: dict,
    tool_call_id: str,
) -> Command:
    """Build a Command from a subagent's final state.

    Extracts the last message as the ToolMessage content and passes
    through non-excluded state keys.
    """
    if "messages" not in result:
        raise ValueError(
            "CompiledSubAgent must return a state containing a 'messages' key. "
            "Custom StateGraphs used with CompiledSubAgent should include "
            "'messages' in their state schema to communicate results back to "
            "the main agent."
        )

    state_update = {
        k: v for k, v in result.items() if k not in _EXCLUDED_STATE_KEYS
    }

    last_msg = result["messages"][-1]
    content = last_msg.text.rstrip() if last_msg.text else ""

    return Command(
        update={
            **state_update,
            "messages": [ToolMessage(content, tool_call_id=tool_call_id)],
        }
    )


def _return_command_with_message(result: dict, message: ToolMessage) -> Command:
    """Pass through non-excluded state channels and attach *message*.

    Mirrors :func:`_return_command_with_state_update` so a failed run updates
    exactly the same channels a successful one would.
    """
    state_update = {
        k: v for k, v in (result or {}).items() if k not in _EXCLUDED_STATE_KEYS
    }
    return Command(update={**state_update, "messages": [message]})


@dataclass
class _TaskRun:
    """Everything a ``task`` call needs from validation through to reporting."""

    tool_call_id: str
    subagent_type: str
    instruction: str
    subagent: Runnable
    state: dict
    config: RunnableConfig
    injected: list[BaseMessage]
    injected_ids: set[str]
    parent: SubAgentSnapshotNode | None = None
    parent_nodes: tuple[SubAgentSnapshotNode, ...] = ()
    store: Any = None
    thread_id: str | None = None
    cleanup_pending: bool = False
    steps: int = 0


def _merge_updates_state(final: dict, chunk: dict) -> dict:
    """Merge a streaming chunk (from ``stream_mode='updates'``) into accumulated state."""
    for _node_name, state_update in chunk.items():
        for key, value in state_update.items():
            if key == "messages":
                final.setdefault("messages", []).extend(value)
            else:
                final[key] = value
    return final


# ---------------------------------------------------------------------------
# Internal spec type
# ---------------------------------------------------------------------------


class _SubagentSpec(BaseModel):
    """Internal spec for building a task tool."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    runnable: Runnable


# ---------------------------------------------------------------------------
# Task tool builder (streaming-aware)
# ---------------------------------------------------------------------------


def _build_task_tool(
    subagents: list[_SubagentSpec],
    event_granularity: EventGranularity = "updates",
    task_description: str | None = None,
    *,
    catch_errors: bool = True,
    max_continue_attempts: int = 3,
    max_children: int = 2,
    max_root_failures_per_thread: int = 10,
) -> BaseTool:
    """Create a ``task`` tool that invokes subagents with streaming events.

    Args:
        subagents: List of pre-built subagent specs.
        event_granularity: Controls what level of detail is emitted as
            custom stream events. Default: ``"updates"``.
        task_description: Custom description for the task tool.

    Returns:
        A ``StructuredTool`` that invokes subagents by type.
    """
    subagent_graphs: dict[str, Runnable] = {
        spec.name: spec.runnable for spec in subagents
    }
    subagent_description_str = "\n".join(
        f"- {s.name}: {s.description}" for s in subagents
    )

    if task_description is None:
        description = TASK_TOOL_DESCRIPTION.format(
            available_agents=subagent_description_str
        )
    elif "{available_agents}" in task_description:
        description = task_description.format(
            available_agents=subagent_description_str
        )
    else:
        description = task_description

    # We always stream with "values" to capture the final state, plus the
    # user-selected granularity for custom events.
    # "custom" is always included so that subagent middleware events
    # (e.g. AutoSecurityReviewMiddleware's pass/fail notifications) are
    # captured and nested into ``subagent_event.chunk`` so consumers get
    # everything through a single event type.
    _stream_modes: list[str] = ["values", "custom", event_granularity]

    # Deduplicate preserving order (in case user chose "values" or "custom")
    _seen: set[str] = set()
    _deduped: list[str] = []
    for m in _stream_modes:
        if m not in _seen:
            _seen.add(m)
            _deduped.append(m)
    _stream_modes = _deduped

    def _emit_custom_event(
        tool_call_id: str,
        subagent_type: str,
        chunk: object,
    ) -> None:
        """Emit a custom event via the stream writer if available.

        ``get_stream_writer()`` raises when there is no runnable context (unit
        tests, direct tool invocation).  Event emission is best-effort: a missing
        stream writer must never turn into a subagent failure.
        """
        try:
            writer = get_stream_writer()
        except Exception:  # pragma: no cover - outside a runnable context
            return
        if writer is not None:
            writer(
                {
                    "type": "subagent_event",
                    "tool_call_id": tool_call_id,
                    "subagent_type": subagent_type,
                    "granularity": event_granularity,
                    "timestamp": time.time(),
                    "chunk": chunk,
                }
            )

    # ---- continuation helpers ------------------------------------------------

    _in_flight: set[str] = set()
    _in_flight_lock = threading.Lock()

    def _claim_breakpoint(continue_id: str | None) -> bool:
        """Reserve a breakpoint atomically (check-and-add under one lock)."""
        if not continue_id:
            return True
        with _in_flight_lock:
            if continue_id in _in_flight:
                return False
            _in_flight.add(continue_id)
        return True

    def _release_breakpoint(continue_id: str | None) -> None:
        if not continue_id:
            return
        with _in_flight_lock:
            _in_flight.discard(continue_id)

    def _snapshot_target(runtime: ToolRuntime) -> tuple[Any, str | None]:
        """``(store, thread_id)`` — both ``None``/missing disables continuation."""
        store = getattr(runtime, "store", None)
        configurable = (runtime.config or {}).get("configurable") or {}
        thread_id = configurable.get("thread_id")
        return store, (str(thread_id) if thread_id else None)

    def _begin(
        description: str,
        subagent_type: str,
        continue_id: str | None,
        runtime: ToolRuntime,
    ) -> _TaskRun | str:
        """Validate the request and prepare the run; a ``str`` means rejected."""
        store, thread_id = _snapshot_target(runtime)
        continuation_enabled = store is not None and thread_id is not None

        parent: SubAgentSnapshotNode | None = None
        parent_nodes: tuple[SubAgentSnapshotNode, ...] = ()
        history: list[BaseMessage] = []

        if continue_id:
            if not continuation_enabled:
                return (
                    "Continuation is unavailable (no store configured). "
                    "Start a new task instead."
                )
            check = validate_continuation(
                store,
                thread_id,  # type: ignore[arg-type]
                continue_id,
                subagent_type,
                max_continue_attempts=max_continue_attempts,
                max_children=max_children,
                in_flight=_in_flight,
            )
            if not check.ok:
                return check.message
            parent = check.node
            parent_nodes = check.nodes
            history = check.history()
            if not _claim_breakpoint(continue_id):
                return "该任务正在执行中，拒绝并发续连。"

        instruction = description or (
            parent.description if parent is not None else ""
        )
        if not instruction:
            _release_breakpoint(continue_id)
            return (
                "Provide a `description` for a new task, or a valid "
                "`continue_id` to resume a previous one."
            )

        messages: list[BaseMessage] = list(history)
        if parent is not None:
            messages.append(
                HumanMessage(
                    content=build_continuation_prompt(parent.error_type, description)
                )
            )
        else:
            messages.append(HumanMessage(content=instruction))

        subagent = subagent_graphs[subagent_type]
        state = {
            k: v
            for k, v in runtime.state.items()
            if k not in _EXCLUDED_STATE_KEYS
        }
        state["messages"] = messages
        config: RunnableConfig = {
            "configurable": {
                **((runtime.config or {}).get("configurable") or {}),
                "ls_agent_type": "subagent",
            }
        }

        # Entry cleanup is performed by the caller (sync vs async store API) —
        # see ``_TaskRun.cleanup_pending``.

        return _TaskRun(
            tool_call_id=runtime.tool_call_id,
            subagent_type=subagent_type,
            instruction=instruction,
            subagent=subagent,
            state=state,
            config=config,
            injected=messages,
            injected_ids={m.id for m in messages if getattr(m, "id", None)},
            parent=parent,
            parent_nodes=parent_nodes,
            store=store if continuation_enabled else None,
            thread_id=thread_id if continuation_enabled else None,
            cleanup_pending=continuation_enabled,
        )

    def _new_failure_node(
        plan: _TaskRun, exc: Exception, final_state: dict
    ) -> SubAgentSnapshotNode:
        produced = list((final_state or {}).get("messages") or [])
        delta = build_delta(
            plan.injected, produced, injected_ids=plan.injected_ids
        )
        if not delta:
            delta = [HumanMessage(content=plan.instruction)]
        error_type, error_message, retryable = classify_failure(exc)
        parent = plan.parent
        chain = [*parent.chain, parent.node_id] if parent is not None else []
        return SubAgentSnapshotNode(
            node_id=plan.tool_call_id,
            parent_id=parent.node_id if parent is not None else None,
            chain=chain,
            depth=len(chain) + 1,
            subagent_type=plan.subagent_type,
            description=plan.instruction,
            status="failed",
            error_type=error_type,
            error_message=error_message,
            retryable=retryable,
            available=retryable,
            delta_messages=to_serialized(delta),
            steps=plan.steps,
            injected_prefix_len=len(plan.injected),
            created_at=now_iso(),
        )

    def _retention(
        plan: _TaskRun, node: SubAgentSnapshotNode, existing_roots: int
    ) -> tuple[bool, bool, str | None]:
        """``(should_write, available, reason)`` for a freshly built node."""
        if plan.store is None or plan.thread_id is None:
            return False, False, "snapshot_unavailable"
        if not node.retryable:
            return True, False, "not_retryable"
        if node.depth > max_continue_attempts:
            return True, False, "exhausted"
        if node.parent_id is None and existing_roots >= max_root_failures_per_thread:
            return False, False, "root_failures_exceeded"
        return True, True, None

    def _failure_result(
        plan: _TaskRun,
        node: SubAgentSnapshotNode,
        available: bool,
        reason: str | None,
        final_state: dict,
    ) -> Command:
        node.available = available
        content = build_failure_content(
            subagent_type=plan.subagent_type,
            error_type=node.error_type,
            error_message=node.error_message,
            steps=node.steps,
            node_id=node.node_id if available else None,
            retryable=node.retryable,
            available=available,
            reason=reason or "",
        )
        metadata = continuation_metadata(
            node if (plan.store is not None and available) else None,
            max_attempts=max_continue_attempts,
            max_children=max_children,
            available=available,
            reason=reason,
        )
        return _return_command_with_message(
            final_state,
            ToolMessage(
                content,
                name="task",
                tool_call_id=plan.tool_call_id,
                status="error",
                additional_kwargs={"subagent_continuation": metadata},
            ),
        )

    def _failed(plan: _TaskRun, exc: Exception, final_state: dict) -> Command:
        node = _new_failure_node(plan, exc, final_state)
        existing_roots = 0
        if plan.store is not None and plan.thread_id is not None and node.retryable:
            try:
                existing_roots = count_root_failures(
                    list_nodes(plan.store, plan.thread_id)
                )
            except Exception:  # pragma: no cover - defensive
                logger.warning("Failed to scan snapshots", exc_info=True)
        write, available, reason = _retention(plan, node, existing_roots)
        if write:
            try:
                put_node(plan.store, plan.thread_id, node)  # type: ignore[arg-type]
                if get_node(plan.store, plan.thread_id, node.node_id) is None:  # type: ignore[arg-type]
                    available, reason = False, "snapshot_write_failed"
            except Exception:  # pragma: no cover - defensive
                logger.warning("Failed to persist subagent snapshot", exc_info=True)
                available, reason = False, "snapshot_write_failed"
        return _failure_result(plan, node, available, reason, final_state)

    async def _afailed(
        plan: _TaskRun, exc: Exception, final_state: dict
    ) -> Command:
        node = _new_failure_node(plan, exc, final_state)
        existing_roots = 0
        if plan.store is not None and plan.thread_id is not None and node.retryable:
            try:
                existing_roots = count_root_failures(
                    await alist_nodes(plan.store, plan.thread_id)
                )
            except Exception:  # pragma: no cover - defensive
                logger.warning("Failed to scan snapshots", exc_info=True)
        write, available, reason = _retention(plan, node, existing_roots)
        if write:
            try:
                await aput_node(plan.store, plan.thread_id, node)  # type: ignore[arg-type]
                if await aget_node(plan.store, plan.thread_id, node.node_id) is None:  # type: ignore[arg-type]
                    available, reason = False, "snapshot_write_failed"
            except Exception:  # pragma: no cover - defensive
                logger.warning("Failed to persist subagent snapshot", exc_info=True)
                available, reason = False, "snapshot_write_failed"
        return _failure_result(plan, node, available, reason, final_state)

    # ---- sync task ----------------------------------------------------------

    def task(
        description: str,
        subagent_type: str,
        runtime: ToolRuntime,
        continue_id: str | None = None,
        **_extra: object,
    ) -> str | Command:
        if subagent_type not in subagent_graphs:
            allowed = ", ".join(f"`{k}`" for k in subagent_graphs)
            return (
                f"We cannot invoke subagent {subagent_type} because it does "
                f"not exist, the only allowed types are {allowed}"
            )
        if not runtime.tool_call_id:
            return _NO_TOOL_CALL_ID

        plan = _begin(description, subagent_type, continue_id, runtime)
        if isinstance(plan, str):
            return plan

        if plan.cleanup_pending and plan.store is not None and plan.thread_id:
            try:
                delete_node(plan.store, plan.thread_id, plan.tool_call_id)
            except Exception:  # pragma: no cover - defensive
                logger.warning("Failed to clear stale snapshot", exc_info=True)

        final_state: dict = {}
        try:
            for chunk in plan.subagent.stream(
                plan.state, plan.config, stream_mode=_stream_modes
            ):
                if isinstance(chunk, tuple) and len(chunk) == 2:
                    mode, data = chunk
                    if mode in (event_granularity, "custom"):
                        _emit_custom_event(
                            runtime.tool_call_id, subagent_type, data
                        )
                    if mode == "values":
                        final_state = cast(dict, data)
                        plan.steps += 1
                else:
                    # Single-mode output
                    _emit_custom_event(
                        runtime.tool_call_id, subagent_type, chunk
                    )
                    # In single-mode, assume "values" semantics
                    final_state = cast(dict, chunk)
                    plan.steps += 1

            # If final_state is empty (shouldn't happen), fall back to invoke
            if not final_state:
                final_state = plan.subagent.invoke(plan.state, plan.config)
                plan.steps += 1
        except GraphBubbleUp:
            raise
        except Exception as exc:
            if is_cancellation_related(exc):
                raise
            if not catch_errors:
                raise
            return _failed(plan, exc, final_state)
        finally:
            _release_breakpoint(continue_id)

        return _return_command_with_state_update(
            final_state, runtime.tool_call_id
        )

    # ---- async task ---------------------------------------------------------

    async def atask(
        description: str,
        subagent_type: str,
        runtime: ToolRuntime,
        continue_id: str | None = None,
        **_extra: object,
    ) -> str | Command:
        if subagent_type not in subagent_graphs:
            allowed = ", ".join(f"`{k}`" for k in subagent_graphs)
            return (
                f"We cannot invoke subagent {subagent_type} because it does "
                f"not exist, the only allowed types are {allowed}"
            )
        if not runtime.tool_call_id:
            return _NO_TOOL_CALL_ID

        plan = _begin(description, subagent_type, continue_id, runtime)
        if isinstance(plan, str):
            return plan

        if plan.cleanup_pending and plan.store is not None and plan.thread_id:
            try:
                await adelete_node(plan.store, plan.thread_id, plan.tool_call_id)
            except Exception:  # pragma: no cover - defensive
                logger.warning("Failed to clear stale snapshot", exc_info=True)

        final_state: dict = {}
        try:
            async for chunk in plan.subagent.astream(
                plan.state, plan.config, stream_mode=_stream_modes
            ):
                if isinstance(chunk, tuple) and len(chunk) == 2:
                    mode, data = chunk
                    if mode in (event_granularity, "custom"):
                        _emit_custom_event(
                            runtime.tool_call_id, subagent_type, data
                        )
                    if mode == "values":
                        final_state = cast(dict, data)
                        plan.steps += 1
                else:
                    # Single-mode output
                    _emit_custom_event(
                        runtime.tool_call_id, subagent_type, chunk
                    )
                    # In single-mode, assume "values" semantics
                    final_state = cast(dict, chunk)
                    plan.steps += 1

            # If streaming yielded no final state (e.g. empty subagent graph),
            # fall back to a direct invoke.
            if not final_state:
                final_state = await plan.subagent.ainvoke(
                    plan.state, plan.config
                )
                plan.steps += 1
        except GraphBubbleUp:
            raise
        except Exception as exc:
            if is_cancellation_related(exc):
                raise
            if not catch_errors:
                raise
            return await _afailed(plan, exc, final_state)
        finally:
            _release_breakpoint(continue_id)

        return _return_command_with_state_update(
            final_state, runtime.tool_call_id
        )

    return StructuredTool.from_function(
        name="task",
        func=task,
        coroutine=atask,
        description=description,
        infer_schema=False,
        args_schema=TaskToolSchema,
    )


# ---------------------------------------------------------------------------
# SubAgentMiddleware
# ---------------------------------------------------------------------------


class SubAgentMiddleware(AgentMiddleware[AgentState, ContextT, ResponseT]):
    """Middleware that provides subagents via a ``task`` tool.

    Subagents are short-lived agents that handle isolated, complex tasks.
    They run autonomously and return a single result to the main agent.

    **Streaming**: subagent internals are emitted as custom stream events.
    Use ``agent.astream(..., stream_mode=["updates", "custom"])`` to
    consume them.  Each custom event carries:

    * ``tool_call_id`` — which ``task`` call this belongs to (essential for
      distinguishing parallel subagent invocations)
    * ``subagent_type`` — the type of subagent
    * ``granularity`` — the configured event granularity
    * ``timestamp`` — when the event was emitted
    * ``chunk`` — the raw streaming payload from the subagent

    Args:
        backend: Backend for file operations.
        subagents: List of subagent specs.
        event_granularity: Streaming detail level.
            ``"messages"`` (finest), ``"updates"`` (default, medium),
            ``"values"`` (coarsest).
        system_prompt: Instructions appended to the main agent's system
            prompt about using the task tool.
        task_description: Custom description for the task tool.
    """

    def __init__(
        self,
        *,
        backend: BackendProtocol,
        subagents: Sequence[SubAgent | CompiledSubAgent],
        event_granularity: EventGranularity = "updates",
        system_prompt: str | None = TASK_SYSTEM_PROMPT,
        task_description: str | None = None,
        catch_errors: bool = True,
        max_continue_attempts: int = 3,
        max_children: int = 2,
        max_root_failures_per_thread: int = 10,
    ) -> None:
        super().__init__()

        if not subagents:
            raise ValueError("At least one subagent must be specified")

        self._backend = backend
        self._subagents = subagents
        self._event_granularity = event_granularity

        subagent_specs = self._get_subagents()
        task_tool = _build_task_tool(
            subagent_specs,
            event_granularity=event_granularity,
            task_description=task_description,
            catch_errors=catch_errors,
            max_continue_attempts=max_continue_attempts,
            max_children=max_children,
            max_root_failures_per_thread=max_root_failures_per_thread,
        )

        # Build system prompt with available agent descriptions
        if system_prompt and subagent_specs:
            agents_desc = "\n".join(
                f"- {s.name}: {s.description}" for s in subagent_specs
            )
            self._system_prompt = (
                system_prompt + "\n\nAvailable subagent types:\n" + agents_desc
            )
        else:
            self._system_prompt = system_prompt

        self.tools = [task_tool]

    # ---- subagent construction -----------------------------------------------

    def _get_subagents(self) -> list[_SubagentSpec]:
        """Create runnable agents from specs (SubAgent or CompiledSubAgent)."""
        specs: list[_SubagentSpec] = []

        for spec in self._subagents:
            if isinstance(spec, CompiledSubAgent):
                runnable = spec.runnable.with_config(
                    {
                        "metadata": {"lc_agent_name": spec.name},
                        "run_name": spec.name,
                    }
                )
                specs.append(
                    _SubagentSpec(
                        name=spec.name,
                        description=spec.description,
                        runnable=runnable,
                    )
                )
                continue

            # SubAgent — model is required if not provided
            if spec.model is None:
                raise ValueError(
                    f"SubAgent '{spec.name}' must specify 'model'"
                )

            middleware: list[AgentMiddleware] = [
                # A subagent tool failure must not abort the subagent's superstep
                # (interrupts still bubble up untouched).
                ToolCallErrorMiddleware(),
                *spec.middleware,
            ]
            if spec.interrupt_on:
                middleware.append(
                    HumanInTheLoopMiddleware(interrupt_on=spec.interrupt_on)
                )

            specs.append(
                _SubagentSpec(
                    name=spec.name,
                    description=spec.description,
                    runnable=_langchain_create_agent(
                        spec.model,
                        system_prompt=spec.system_prompt,
                        tools=spec.tools,
                        middleware=middleware,
                        name=spec.name,
                    ),
                )
            )

        return specs

    # ---- wrap_model_call / awrap_model_call ----------------------------------

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[
            [ModelRequest[ContextT]], ModelResponse[ResponseT]
        ],
    ) -> ModelResponse[ResponseT]:
        """Inject subagent usage instructions into the system prompt."""
        if self._system_prompt is not None:
            request = _append_to_system_message(request, self._system_prompt)
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[
            [ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]
        ],
    ) -> ModelResponse[ResponseT]:
        """(async) Inject subagent usage instructions into the system prompt."""
        if self._system_prompt is not None:
            request = _append_to_system_message(request, self._system_prompt)
        return await handler(request)
