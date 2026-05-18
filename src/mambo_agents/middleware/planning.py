"""Planning and task management middleware (mambo version).

Provides ``MamboPlanMiddleware`` — a Pydantic-typed re-implementation of
|langchain's ``TodoListMiddleware`` that:

- Uses frozen ``Plan`` Pydantic models instead of ``TypedDict``
- Exposes ``build_summary_hook()`` so ``MamboSummarizationMiddleware`` can
  inject current plan state when summarisation compacts conversation history
- Only injects when plans exist **and** at least one is not ``"completed"``,
  preventing hallucination from stale/empty state

The hook output is wrapped in unambiguous boundary markers so it cannot be
confused with the AI-generated summary text.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Annotated, Any, Literal, cast

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    OmitFromInput,
    ResponseT,
)
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langchain_core.tools import InjectedToolCallId, StructuredTool
from langgraph.runtime import Runtime
from langgraph.types import Command
from langgraph.typing import ContextT
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import NotRequired, TypedDict, override


# ---------------------------------------------------------------------------
# Pydantic data models (strict, frozen – no TypedDict / duck-typing)
# ---------------------------------------------------------------------------


class Plan(BaseModel):
    """A single task item with content and status."""

    model_config = ConfigDict(frozen=True)

    content: str = Field(description="The content/description of the plan item")
    status: Literal["pending", "in_progress", "completed"] = Field(
        default="pending",
        description="Current status of the plan item",
    )


class WritePlansInput(BaseModel):
    """Input schema for the ``write_plans`` tool.

    ``tool_call_id`` is injected by langchain — the LLM never sees it
    (it is stripped by ``tool_call_schema`` before dispatch).
    """

    plans: list[Plan] = Field(description="Full plan list (replaces all existing plans)")
    tool_call_id: Annotated[str, InjectedToolCallId] = Field(
        default="", description="Injected tool call ID (hidden from LLM)"
    )


# ---------------------------------------------------------------------------
# Planning state (extends AgentState with a ``plans`` key)
# ---------------------------------------------------------------------------


class PlanningState(TypedDict, total=False):
    """State schema for the planning middleware.

    ``plans`` is excluded from user-facing input so callers never need to
    pass it via ``.invoke({"plans": ...})`` — it is managed entirely by the
    ``write_plans`` tool.
    """

    messages: Annotated[list[AnyMessage], ...]  # inherited from AgentState
    plans: Annotated[NotRequired[list[Plan]], OmitFromInput]
    """Structured task list written by the agent via ``write_plans``."""


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

WRITE_PLANS_TOOL_DESCRIPTION = """Use this tool to create and manage a structured task list for your current work session. This helps you track progress, organize complex tasks, and demonstrate thoroughness to the user.

Only use this tool if it will be helpful in staying organized. If the user's request is trivial and takes less than 3 steps, it is better to NOT use this tool and just do the task directly.

## When to Use This Tool
Use this tool in these scenarios:

1. Complex multi-step tasks - When a task requires 3 or more distinct steps or actions
2. Non-trivial and complex tasks - Tasks that require careful planning or multiple operations
3. User explicitly requests plan list - When the user directly asks you to use the plan list
4. User provides multiple tasks - When users provide a list of things to be done (numbered or comma-separated)
5. The plan may need future revisions or updates based on results from the first few steps

## How to Use This Tool
1. When you start working on a task - Mark it as in_progress BEFORE beginning work.
2. After completing a task - Mark it as completed and add any new follow-up tasks discovered during implementation.
3. You can also update future tasks, such as deleting them if they are no longer necessary, or adding new tasks that are necessary. Don't change previously completed tasks.
4. You can make several updates to the plan list at once. For example, when you complete a task, you can mark the next task you need to start as in_progress.

## When NOT to Use This Tool
It is important to skip using this tool when:
1. There is only a single, straightforward task
2. The task is trivial and tracking it provides no benefit
3. The task can be completed in less than 3 trivial steps
4. The task is purely conversational or informational

## Task States and Management

1. **Task States**: Use these states to track progress:
   - pending: Task not yet started
   - in_progress: Currently working on (you can have multiple tasks in_progress at a time if they are not related to each other and can be run in parallel)
   - completed: Task finished successfully

2. **Task Management**:
   - Update task status in real-time as you work
   - Mark tasks complete IMMEDIATELY after finishing (don't batch completions)
   - Complete current tasks before starting new ones
   - Remove tasks that are no longer relevant from the list entirely
   - IMPORTANT: When you write this plan list, you should mark your first task (or tasks) as in_progress immediately!.
   - IMPORTANT: Unless all tasks are completed, you should always have at least one task in_progress to show the user that you are working on something.

3. **Task Completion Requirements**:
   - ONLY mark a task as completed when you have FULLY accomplished it
   - If you encounter errors, blockers, or cannot finish, keep the task as in_progress
   - When blocked, create a new task describing what needs to be resolved
   - Never mark a task as completed if:
     - There are unresolved issues or errors
     - Work is partial or incomplete
     - You encountered blockers that prevent completion
     - You couldn't find necessary resources or dependencies
     - Quality standards haven't been met

4. **Task Breakdown**:
   - Create specific, actionable items
   - Break complex tasks into smaller, manageable steps
   - Use clear, descriptive task names

Being proactive with task management demonstrates attentiveness and ensures you complete all requirements successfully
Remember: If you only need to make a few tool calls to complete a task, and it is clear what you need to do, it is better to just do the task directly and NOT call this tool at all."""  # noqa: E501

WRITE_PLANS_SYSTEM_PROMPT = """## `write_plans`

You have access to the `write_plans` tool to help you manage and plan complex objectives.
Use this tool for complex objectives to ensure that you are tracking each necessary step and giving the user visibility into your progress.
This tool is very helpful for planning complex objectives, and for breaking down these larger complex objectives into smaller steps.

It is critical that you mark plans as completed as soon as you are done with a step. Do not batch up multiple steps before marking them as completed.
For simple objectives that only require a few steps, it is better to just complete the objective directly and NOT use this tool.
Writing plans takes time and tokens, use it when it is helpful for managing complex many-step problems! But not for simple few-step requests.

## Important Plan List Usage Notes to Remember
- The `write_plans` tool should never be called multiple times in parallel.
- Don't be afraid to revise the plan list as you go. New information may reveal new tasks that need to be done, or old tasks that are irrelevant."""  # noqa: E501


# ---------------------------------------------------------------------------
# Tool implementations (sync + async)
# ---------------------------------------------------------------------------


def _write_plans(
    plans: list[Plan],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Create and replace the structured task list.

    The entire plan list is replaced — this is NOT an incremental update.
    """
    return Command(
        update={
            "plans": plans,
            "messages": [
                ToolMessage(
                    f"Updated plan list to {[t.model_dump() for t in plans]}",
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )


async def _awrite_plans(
    plans: list[Plan],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Async variant of ``_write_plans``."""
    return _write_plans(plans, tool_call_id=tool_call_id)


# ---------------------------------------------------------------------------
# Summary hook (for MamboSummarizationMiddleware integration)
# ---------------------------------------------------------------------------

_SUMMARY_PLAN_BOUNDARY_START = "============================================================"
_SUMMARY_PLAN_BOUNDARY_END = "============================================================"
_SUMMARY_PLAN_HEADER = "[ATTACHED STATE: **Current Task List (write_plans)**]"
_SUMMARY_PLAN_FOOTER = (
    "Above is the CURRENT state of your task list. "
    "It is authoritative — trust it over anything mentioned in the summary above. "
    "Use `write_plans` to modify it."
)

_STATUS_ICONS: dict[str, str] = {
    "pending": "⬜",
    "in_progress": "🔄",
    "completed": "✅",
}


def _build_plan_summary_hook() -> Callable[..., str | None]:
    """Factory for a summary hook that supplements Plan state.

    Returns a callable suitable for ``MamboSummarizationMiddleware``
    ``summary_hooks``.  The hook:
    - Returns ``None`` when no plans exist or all are ``"completed"``,
      to avoid injecting noise / hallucination triggers.
    - Wraps output in unambiguous boundary markers so the LLM cannot
      confuse it with the autogenerated conversation summary.
    """

    def _hook(ctx: object) -> str | None:
        """Extract and format current plan list from agent state.

        Args:
            ctx: A ``SummaryHookContext`` instance (Pydantic model).

        Returns:
            Formatted plan list section, or ``None`` if nothing to inject.
        """
        # Deferred import to avoid circular dependency at module level
        from mambo_agents.middleware.summarization import SummaryHookContext  # noqa: PLC0415

        if not isinstance(ctx, SummaryHookContext):
            return None

        raw_plans = ctx.state.get("plans")
        if not raw_plans:
            return None

        # Coerce to Plan list (may arrive as plain dicts from checkpoint restore)
        plans: list[Plan] = []
        for item in raw_plans:
            if isinstance(item, Plan):
                plans.append(item)
            elif isinstance(item, dict):
                plans.append(Plan(**item))

        if not plans:
            return None

        # Suppress when everything is already done
        if all(t.status == "completed" for t in plans):
            return None

        lines: list[str] = []
        for t in plans:
            icon = _STATUS_ICONS.get(t.status, "❓")
            lines.append(f"  {icon} [{t.status}]  {t.content}")

        return (
            f"{_SUMMARY_PLAN_BOUNDARY_START}\n"
            f"{_SUMMARY_PLAN_HEADER}\n"
            f"{_SUMMARY_PLAN_BOUNDARY_START}\n"
            + "\n".join(lines)
            + f"\n{_SUMMARY_PLAN_BOUNDARY_END}\n"
            f"{_SUMMARY_PLAN_FOOTER}\n"
            f"{_SUMMARY_PLAN_BOUNDARY_END}"
        )

    return _hook


# ---------------------------------------------------------------------------
# MamboPlanMiddleware
# ---------------------------------------------------------------------------


class MamboPlanMiddleware(AgentMiddleware[PlanningState, ContextT, ResponseT]):
    """Middleware that provides plan list management via a ``write_plans`` tool.

    Compared to langchain's ``TodoListMiddleware``:
    - Uses Pydantic ``Plan`` (frozen) instead of ``TypedDict``
    - Exposes ``build_summary_hook()`` so summarisation preserves plan state
    - Same parallel-call prevention and system prompt injection

    Parameters:
        system_prompt: Custom system prompt appended to the model's instructions.
        tool_description: Custom description for the ``write_plans`` tool.
    """

    state_schema = cast(type[PlanningState], PlanningState)

    def __init__(
        self,
        *,
        system_prompt: str = WRITE_PLANS_SYSTEM_PROMPT,
        tool_description: str = WRITE_PLANS_TOOL_DESCRIPTION,
    ) -> None:
        super().__init__()
        self._system_prompt = system_prompt
        self._tool_description = tool_description

        self.tools = [
            StructuredTool.from_function(
                name="write_plans",
                description=tool_description,
                func=_write_plans,
                coroutine=_awrite_plans,
                args_schema=WritePlansInput,
                infer_schema=False,
            )
        ]

    # ------------------------------------------------------------------
    # Summary hook factory (class-level, used by create_mambo_agent)
    # ------------------------------------------------------------------

    @staticmethod
    def build_summary_hook() -> Callable[..., str | None]:
        """Return a summary hook that injects current plan list state.

        Pass this to ``MamboSummarizationMiddleware(summary_hooks=[...])``
        so plan items are not lost when conversation history is compacted.

        Returns ``None`` (no injection) when:
        - ``plans`` key is missing or empty
        - Every plan is ``"completed"``
        """
        return _build_plan_summary_hook()

    # ------------------------------------------------------------------
    # wrap_model_call
    # ------------------------------------------------------------------

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT] | AIMessage:
        """Inject plan system prompt into the model call."""
        if request.system_message is not None:
            new_system_content = [
                *request.system_message.content_blocks,
                {"type": "text", "text": f"\n\n{self._system_prompt}"},
            ]
        else:
            new_system_content = [{"type": "text", "text": self._system_prompt}]
        new_system_message = SystemMessage(
            content=cast("list[str | dict[str, str]]", new_system_content)
        )
        return handler(request.override(system_message=new_system_message))

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT] | AIMessage:
        """(async) Inject plan system prompt into the model call."""
        if request.system_message is not None:
            new_system_content = [
                *request.system_message.content_blocks,
                {"type": "text", "text": f"\n\n{self._system_prompt}"},
            ]
        else:
            new_system_content = [{"type": "text", "text": self._system_prompt}]
        new_system_message = SystemMessage(
            content=cast("list[str | dict[str, str]]", new_system_content)
        )
        return await handler(request.override(system_message=new_system_message))

    # ------------------------------------------------------------------
    # after_model — prevent parallel write_plans calls
    # ------------------------------------------------------------------

    @override
    def after_model(
        self,
        state: PlanningState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Reject parallel ``write_plans`` calls in a single model turn.

        Since ``write_plans`` replaces the entire plan list, multiple
        parallel calls create ambiguity.  This hook returns error
        ``ToolMessage`` objects for every ``write_plans`` call when more
        than one is detected.
        """
        messages = state.get("messages", [])
        if not messages:
            return None

        last_ai_msg = next(
            (msg for msg in reversed(messages) if isinstance(msg, AIMessage)),
            None,
        )
        if not last_ai_msg or not last_ai_msg.tool_calls:
            return None

        write_plans_calls = [
            tc for tc in last_ai_msg.tool_calls if tc["name"] == "write_plans"
        ]

        if len(write_plans_calls) > 1:
            error_messages = [
                ToolMessage(
                    content=(
                        "Error: The `write_plans` tool should never be called "
                        "multiple times in parallel. Please call it only once "
                        "per model invocation to update the plan list."
                    ),
                    tool_call_id=tc["id"],
                    status="error",
                )
                for tc in write_plans_calls
            ]
            return {"messages": error_messages}

        return None

    @override
    async def aafter_model(
        self,
        state: PlanningState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """(async) Prevent parallel ``write_plans`` calls."""
        return self.after_model(state, runtime)
