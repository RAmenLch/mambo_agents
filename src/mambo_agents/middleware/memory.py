"""Memory middleware for loading agent persistent context from AGENTS.md files.

Implements the AGENTS.md specification (https://agents.md/), loading
memory/context from configurable sources and injecting into the system prompt.

## Overview

Unlike skills (which are on-demand workflows with progressive disclosure),
memory is **always loaded** and provides persistent context across turns.
The agent is also instructed to *write back* new learnings — so memory
evolves through use.

## Usage

```python
from mambo_agents.backends.state import StateBackend
from mambo_agents.middleware.memory import MamboMemoryMiddleware

middleware = MamboMemoryMiddleware(
    backend=StateBackend(),
    sources=[VirtualPath("/.mambo/memory/AGENTS.md")],
)
```

## Memory Sources

Sources are paths to AGENTS.md files that are loaded in order and
concatenated.  Multiple sources are combined — all content is included,
with later sources appearing after earlier ones in the prompt.

## File Format

AGENTS.md files are standard Markdown with no required structure.
Common sections include:
- Project overview
- Build/test commands
- Code style guidelines
- Architecture notes
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelRequest,
    ModelResponse,
    PrivateStateAttr,
    ResponseT,
)
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolRuntime

from mambo_agents.backends.protocol import BackendProtocol
from mambo_agents.backends.schemas import ErrorCode, VirtualPath

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from langchain_core.runnables import RunnableConfig
    from langgraph.runtime import Runtime

from typing import NotRequired, TypedDict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

MemoryFormatHook = Callable[[dict[str, str]], str]
"""Callback that formats loaded memory contents into a prompt string.

Args:
    contents: Dict mapping source paths to their loaded text content.

Returns:
    A formatted string to inject into the system prompt.
"""

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class MemoryState(AgentState):
    """State schema for `MamboMemoryMiddleware`.

    Attributes:
        memory_contents: Dict mapping source paths to their loaded content.
            Marked as private so it is not included in the final agent state.
    """

    memory_contents: NotRequired[Annotated[dict[str, str], PrivateStateAttr]]


class MemoryStateUpdate(TypedDict):
    """State update for `MamboMemoryMiddleware`."""

    memory_contents: dict[str, str]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

MAMBO_MEMORY_SYSTEM_PROMPT = """<agent_memory>
{agent_memory}
</agent_memory>

<memory_guidelines>
    The above <agent_memory> was loaded from files in your workspace.
    As you learn from your interactions with the user, you can persist
    new knowledge by using the `edit` or `write` tools to update the
    AGENTS.md files listed above.

    **Learning from feedback:**
    - One of your MAIN PRIORITIES is to learn from your interactions
      with the user.  These learnings can be implicit or explicit.
      This means that in the future, you will remember this important
      information.
    - When you need to remember something, updating memory must be your
      FIRST, IMMEDIATE action — before responding to the user, before
      calling other tools, before doing anything else.
    - When the user says something is better/worse, capture WHY and
      encode it as a pattern.
    - Each correction is a chance to improve permanently — do not just
      fix the immediate issue, update your instructions.
    - A great opportunity to update your memories is when the user
      interrupts a tool call and provides feedback.  Update your
      memories immediately before revising the tool call.
    - Look for the underlying principle behind corrections, not just
      the specific mistake.
    - The user might not explicitly ask you to remember something, but
      if they provide information that is useful for future use, you
      should update your memories immediately.

    **Asking for information:**
    - If you lack context to perform an action, explicitly ask the user
      for the needed information.
    - It is preferred for you to ask for information — do not assume
      anything that you do not know!
    - When the user provides information that is useful for future use,
      you should update your memories immediately.

    **When to update memories:**
    - When the user explicitly asks you to remember something (e.g.,
      "remember my email", "save this preference").
    - When the user describes your role or how you should behave (e.g.,
      "you are a web researcher", "always do X").
    - When the user gives feedback on your work — capture what was wrong
      and how to improve.
    - When the user provides context useful for future tasks, such as
      how to use tools, or which actions to take in a particular
      situation.
    - When you discover new patterns or preferences (coding styles,
      conventions, workflows).

    **When to NOT update memories:**
    - When the information is temporary or transient (e.g., "I'm
      running late", "I'm on my phone right now").
    - When the information is a one-time task request (e.g., "Find me a
      recipe", "What is 25 * 4?").
    - When the information is a simple question that does not reveal
      lasting preferences (e.g., "What day is it?", "Can you explain
      X?").
    - When the information is an acknowledgment or small talk (e.g.,
      "Sounds good!", "Hello", "Thanks for that").
    - When the information is stale or irrelevant in future
      conversations.
    - NEVER store API keys, access tokens, passwords, or any other
      credentials in any file, memory, or system prompt.
    - If the user asks where to put API keys or provides an API key, do
      NOT echo or save it.

    **Examples:**

    Example 1 (remembering user information):
    User: Can you connect to my google account?
    Agent: Sure, I will connect to your google account, what is your
           google account email?
    User: john@example.com
    Agent: Let me save this to my memory.
    Tool Call: edit(...) -> remembers that the user's google account
               email is john@example.com

    Example 2 (remembering implicit user preferences):
    User: Can you write me an example for creating a deep agent in
          LangChain?
    Agent: Sure, here is the Python example: <example code in Python>
    User: Can you do this in JavaScript?
    Agent: Let me save this to my memory.
    Tool Call: edit(...) -> remembers that the user prefers to get
               LangChain code examples in JavaScript
    Agent: Sure, here is the JavaScript example: <example code in
           JavaScript>

    Example 3 (do NOT remember transient information):
    User: I am going to play basketball tonight so I will be offline
          for a few hours.
    Agent: Okay, I will add a block to your calendar.
    Tool Call: create_calendar_event(...) -> just calls a tool, does
               NOT commit anything to memory, as it is transient
               information
</memory_guidelines>
"""


# ---------------------------------------------------------------------------
# Default formatter
# ---------------------------------------------------------------------------


def _default_format_prompt(contents: dict[str, str]) -> str:
    """Format memory with source paths and contents paired together.

    Args:
        contents: Dict mapping source paths to content.

    Returns:
        Formatted string with path+content pairs wrapped in
        ``<agent_memory>`` tags via `MAMBO_MEMORY_SYSTEM_PROMPT`.
    """
    if not contents:
        return MAMBO_MEMORY_SYSTEM_PROMPT.format(
            agent_memory="(No memory loaded)",
        )

    sections = [f"{path}\n{contents[path]}" for path in contents if contents[path]]

    if not sections:
        return MAMBO_MEMORY_SYSTEM_PROMPT.format(
            agent_memory="(No memory loaded)",
        )

    memory_body = "\n\n".join(sections)
    return MAMBO_MEMORY_SYSTEM_PROMPT.format(agent_memory=memory_body)


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class MamboMemoryMiddleware(AgentMiddleware[MemoryState, ContextT, ResponseT]):
    """Middleware for loading agent memory from AGENTS.md files.

    Loads memory content from configured sources and injects it into the
    system prompt.  The agent is also instructed to **write back** new
    learnings via the standard file tools (``edit`` / ``write``).

    Supports multiple sources that are combined together in order.

    Args:
        backend: Backend instance or factory function ``(runtime) → backend``.
        sources: List of memory file paths to load (e.g.,
            ``["/.mambo/memory/AGENTS.md"]``).
            Sources are loaded in order.
        format_prompt: Optional callback to customize how memory contents
            are formatted into the system prompt.  Defaults to
            :func:`_default_format_prompt`.

    Example:
        ```python
        from mambo_agents.backends.state import StateBackend
        from mambo_agents.middleware.memory import MamboMemoryMiddleware

        middleware = MamboMemoryMiddleware(
            backend=StateBackend(),
            sources=[VirtualPath("/.mambo/memory/AGENTS.md")],
        )
        ```
    """

    state_schema = MemoryState

    def __init__(
        self,
        *,
        backend: BackendProtocol,
        sources: list[VirtualPath],
        format_prompt: MemoryFormatHook | None = None,
    ) -> None:
        """Initialize the memory middleware.

        Args:
            backend: Backend instance or factory ``(runtime) → backend``.
                Use a factory for backends that need runtime context
                (e.g. ``StateBackend``).
            sources: List of memory file paths to load.  Sources are
                loaded in order and all content is included.
            format_prompt: Optional custom formatter.  Receives the
                ``dict[str, str]`` of path → content and returns a
                single string to inject into the system prompt.
                Default: :func:`_default_format_prompt`.
        """
        self._backend = backend
        self.sources = [VirtualPath(s) for s in sources]
        self._format_prompt: MemoryFormatHook = format_prompt or _default_format_prompt

    def _get_backend(
        self,
        state: MemoryState,
        runtime: Runtime,
        config: RunnableConfig,
    ) -> BackendProtocol:
        """Resolve backend from instance or factory.

        Args:
            state: Current agent state.
            runtime: Runtime context for factory functions.
            config: Runnable config to pass to backend factory.

        Returns:
            Resolved backend instance.
        """
        if callable(self._backend):
            tool_runtime = ToolRuntime(
                state=state,
                context=runtime.context,
                stream_writer=runtime.stream_writer,
                store=runtime.store,
                config=config,
                tool_call_id=None,
            )
            backend = self._backend(tool_runtime)
            if backend is None:
                raise AssertionError(
                    "MamboMemoryMiddleware requires a valid backend instance"
                )
            return backend
        return self._backend

    # ---- loading -----------------------------------------------------------

    def before_agent(
        self,
        state: MemoryState,
        runtime: Runtime,
        config: RunnableConfig,
    ) -> MemoryStateUpdate | None:
        """Load memory content before agent execution (synchronous).

        Loads memory from all configured sources and stores in state.
        Only loads if not already present in state.

        Args:
            state: Current agent state.
            runtime: Runtime context.
            config: Runnable config.

        Returns:
            State update with ``memory_contents`` populated.

        Raises:
            ValueError: If a source file fails to load for a reason
                other than ``file_not_found``.
        """
        if "memory_contents" in state:
            return None

        backend = self._get_backend(state, runtime, config)
        contents: dict[str, str] = {}

        results = backend.download_files(list(self.sources))
        for path, response in zip(self.sources, results, strict=True):
            if response.error is not None:
                if response.error.code == ErrorCode.NOT_FOUND:
                    continue
                msg = f"Failed to download {path}: {response.error}"
                raise ValueError(msg)
            if response.content is not None:
                contents[str(path)] = response.content.decode("utf-8")
                logger.debug("Loaded memory from: %s", path)

        return MemoryStateUpdate(memory_contents=contents)

    async def abefore_agent(
        self,
        state: MemoryState,
        runtime: Runtime,
        config: RunnableConfig,
    ) -> MemoryStateUpdate | None:
        """Load memory content before agent execution (asynchronous).

        Loads memory from all configured sources and stores in state.
        Only loads if not already present in state.

        Args:
            state: Current agent state.
            runtime: Runtime context.
            config: Runnable config.

        Returns:
            State update with ``memory_contents`` populated.

        Raises:
            ValueError: If a source file fails to load for a reason
                other than ``file_not_found``.
        """
        if "memory_contents" in state:
            return None

        backend = self._get_backend(state, runtime, config)
        contents: dict[str, str] = {}

        results = await backend.adownload_files(list(self.sources))
        for path, response in zip(self.sources, results, strict=True):
            if response.error is not None:
                if response.error.code == ErrorCode.NOT_FOUND:
                    continue
                msg = f"Failed to download {path}: {response.error}"
                raise ValueError(msg)
            if response.content is not None:
                contents[str(path)] = response.content.decode("utf-8")
                logger.debug("Loaded memory from: %s", path)

        return MemoryStateUpdate(memory_contents=contents)

    # ---- injection ---------------------------------------------------------

    def modify_request(
        self,
        request: ModelRequest[ContextT],
    ) -> ModelRequest[ContextT]:
        """Inject memory content into the system message.

        Args:
            request: Model request to modify.

        Returns:
            Modified request with memory injected into system message.
        """
        contents = request.state.get("memory_contents", {})
        memory_text = self._format_prompt(contents)

        new_system_message = _append_to_system_message(
            request.system_message, memory_text,
        )
        return request.override(system_message=new_system_message)

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """Wrap model call to inject memory into system prompt.

        Args:
            request: Model request being processed.
            handler: Handler function to call with modified request.

        Returns:
            Model response from handler.
        """
        modified_request = self.modify_request(request)
        return handler(modified_request)

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[
            [ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]
        ],
    ) -> ModelResponse[ResponseT]:
        """Async wrap model call to inject memory into system prompt.

        Args:
            request: Model request being processed.
            handler: Async handler function to call with modified request.

        Returns:
            Model response from handler.
        """
        modified_request = self.modify_request(request)
        return await handler(modified_request)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _append_to_system_message(
    existing: SystemMessage | None,
    text: str,
) -> SystemMessage:
    """Append text to a system message (or create a new one)."""
    if existing is None:
        return SystemMessage(content=text)

    content = existing.content
    if isinstance(content, str):
        return SystemMessage(content=content + "\n\n" + text)
    if isinstance(content, list):
        return SystemMessage(
            content=[*content, {"type": "text", "text": "\n\n" + text}],
        )
    return SystemMessage(content=f"{content}\n\n{text}")


__all__ = [
    "MAMBO_MEMORY_SYSTEM_PROMPT",
    "MamboMemoryMiddleware",
    "MemoryFormatHook",
    "MemoryState",
    "MemoryStateUpdate",
]
