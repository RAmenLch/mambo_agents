"""Middleware that builds core tool list from a backend.

``BackendToolsMiddleware`` creates the six mandatory file-system tools
(``ls``, ``read``, ``write``, ``edit``, ``grep``, ``glob``) by wrapping
the backend's corresponding methods, then merges in any extra tools
from ``backend.tools``.

It also intercepts tool results and evicts oversized content to the
filesystem (``/.mambo/large_tool_results/<tool_call_id>``), replacing
it with a truncated preview to prevent context-window saturation.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from typing import Annotated

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    ResponseT,
)
from langchain.tools import ToolRuntime
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import StructuredTool
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command
from langgraph.typing import ContextT
from pydantic import BaseModel, Field

from mambo_agents.backends.protocol import BackendProtocol, ReadResult


# ---------------------------------------------------------------------------
# Large result eviction — constants
# ---------------------------------------------------------------------------

_EVICTION_PREFIX = "/.mambo/large_tool_results"
"""Directory where evicted tool results are stored."""

_DEFAULT_TOOL_TOKEN_LIMIT = 20_000
"""Default token threshold; results exceeding this are evicted."""

_NUM_CHARS_PER_TOKEN = 4
"""Conservative chars-per-token ratio for size estimation."""

_PREVIEW_HEAD_LINES = 5
"""Number of lines to show from the start of evicted content."""

_PREVIEW_TAIL_LINES = 5
"""Number of lines to show from the end of evicted content."""

_TOOLS_EXCLUDED_FROM_EVICTION = ("ls", "read", "write", "edit", "grep", "glob")
"""Tools whose results should never be evicted.

- ls / glob / grep — already truncate their own output
- read — truncation would confuse pagination
- write / edit — results are never large
"""

_TOOL_RESULT_EVICTED_MSG = """Tool result too large. The full result was saved to the filesystem at: {file_path}

Preview (head and tail of content):

{preview}

You can read the full result using `read(file_path="{file_path}", offset=0, limit=200)`.
For very large results, use pagination with offset/limit."""

# ---------------------------------------------------------------------------
# Large result eviction — helpers
# ---------------------------------------------------------------------------

_SAFE_ID_RE = re.compile(r"[^a-zA-Z0-9_.-]")


def _sanitize_tool_call_id(tool_call_id: str) -> str:
    """Replace characters unsafe for file paths with underscores."""
    return _SAFE_ID_RE.sub("_", tool_call_id)


def _estimate_tokens(text: str) -> int:
    """Rough token-count estimate based on chars-per-token ratio."""
    return len(text) // _NUM_CHARS_PER_TOKEN


def _extract_text_from_tool_message(msg: ToolMessage) -> str:
    """Extract plain text from a ToolMessage.

    Handles both ``str`` content and ``list[dict]`` content blocks.
    Only ``type == "text"`` blocks contribute text; multimodal blocks
    (image, audio, video, file) are skipped.
    """
    content = msg.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


def _build_evicted_content(
    message: ToolMessage,
    replacement_text: str,
) -> str | list[dict]:
    """Build replacement content for an evicted ToolMessage.

    For plain ``str`` content, returns the replacement text directly.
    For list content with mixed block types (e.g., text + image),
    replaces all text blocks with a single text block containing the
    replacement text while keeping non-text blocks intact.

    This ensures multimodal context (images, audio, video) is not
    lost when a large tool result is evicted to the filesystem.
    """
    content = message.content
    if isinstance(content, str):
        return replacement_text
    if isinstance(content, list):
        media_blocks = [
            block for block in content
            if isinstance(block, dict) and block.get("type") != "text"
        ]
        if not media_blocks:
            return replacement_text
        return [{"type": "text", "text": replacement_text}, *media_blocks]
    return replacement_text


def _build_preview(text: str) -> str:
    """Build a head+tail preview for evicted content.

    Returns the full text when it fits within the head+tail window;
    otherwise returns head lines, a truncation notice, and tail lines
    with line numbers.
    """
    lines = text.splitlines()
    total_lines = len(lines)
    window = _PREVIEW_HEAD_LINES + _PREVIEW_TAIL_LINES

    if total_lines <= window:
        numbered = [f"{i + 1:6d}|{line}" for i, line in enumerate(lines)]
        return "\n".join(numbered)

    head = lines[:_PREVIEW_HEAD_LINES]
    tail = lines[-_PREVIEW_TAIL_LINES:]
    truncated_count = total_lines - _PREVIEW_HEAD_LINES - _PREVIEW_TAIL_LINES

    parts: list[str] = []
    for i, line in enumerate(head):
        parts.append(f"{i + 1:6d}|{line}")
    parts.append(f"... [{truncated_count} lines truncated] ...")
    for i, line in enumerate(tail):
        parts.append(f"{total_lines - _PREVIEW_TAIL_LINES + i + 1:6d}|{line}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Core tool descriptors
# ---------------------------------------------------------------------------

_CORE_TOOLS = [
    {
        "name": "ls",
        "description": (
            "List files and directories in a given path (non-recursive). "
            "Returns file names, sizes, and whether each entry is a file or directory."
        ),
        "method": "ls",
        "fields": {
            "path": (str, Field(description="Absolute path to list")),
        },
    },
    {
        "name": "read",
        "description": (
            "Read the contents of a file. "
            "For text files, returns content with line numbers. "
            "For image, audio, video, and PDF files, returns multimodal "
            "content blocks that the model can understand directly. "
            "Supports offset and limit for pagination (text only)."
        ),
        "method": "read",
        "fields": {
            "file_path": (str, Field(description="Absolute file path")),
            "offset": (int, Field(default=0, description="Line offset from start")),
            "limit": (int, Field(default=2000, description="Max lines to return")),
        },
    },
    {
        "name": "write",
        "description": (
            "Create a new file with the given content. "
            "By default, fails if the file already exists — read the existing file first, "
            "then use edit to modify it. Set overwrite=True to replace the file content entirely."
        ),
        "method": "write",
        "fields": {
            "file_path": (str, Field(description="Absolute file path")),
            "content": (str, Field(description="Content to write")),
            "overwrite": (bool, Field(default=False, description="If True, replace the file content entirely even if it already exists")),
        },
    },
    {
        "name": "edit",
        "description": (
            "Edit an existing file by replacing a string. "
            "By default the old string must appear exactly once in the file — "
            "read the file first to confirm its content. "
            "Set replace_all=True to replace all occurrences at once."
        ),
        "method": "edit",
        "fields": {
            "file_path": (str, Field(description="Absolute file path")),
            "old_str": (str, Field(description="Exact text to replace. Must be unique unless replace_all is True.")),
            "new_str": (str, Field(description="Replacement text")),
            "replace_all": (bool, Field(default=False, description="If True, replace all occurrences of old_str. If False (default), old_str must appear exactly once.")),
        },
    },
    {
        "name": "grep",
        "description": (
            "Search for a literal text pattern in files. "
            "Performs exact substring matching (NOT regex). "
            "Optionally restrict by path prefix and/or glob filename pattern."
        ),
        "method": "grep",
        "fields": {
            "pattern": (str, Field(description="Literal substring to find")),
            "path": (str, Field(default="/", description="Base directory to search")),
            "glob": (str | None, Field(default=None, description="Optional glob to filter filenames")),
        },
    },
    {
        "name": "glob",
        "description": (
            "Find files matching a glob pattern. "
            "Supports wildcards: *, ** (recursive), ?, [...]. "
        ),
        "method": "glob",
        "fields": {
            "pattern": (str, Field(description="Glob pattern to match")),
            "path": (str, Field(default="/", description="Base directory to search")),
        },
    },
]


# ---------------------------------------------------------------------------
# Async method name mapping
# ---------------------------------------------------------------------------

_ASYNC_METHOD_MAP: dict[str, str] = {
    "ls": "als",
    "read": "aread",
    "write": "awrite",
    "edit": "aedit",
    "grep": "agrep",
    "glob": "aglob",
}


class _ReadSchema(BaseModel):
    """Schema for the read tool."""

    file_path: Annotated[str, Field(description="Absolute file path")]
    offset: Annotated[int, Field(default=0, description="Line offset from start")]
    limit: Annotated[int, Field(default=2000, description="Max lines to return")]


def _build_sync_read_tool(backend: BackendProtocol) -> StructuredTool:
    """Build the read tool with proper tool_call_id support.

    Uses ``StructuredTool.from_function()`` so that the ``runtime: ToolRuntime``
    parameter is injected by LangChain's ToolNode, allowing us to access
    ``runtime.tool_call_id``.
    """

    def sync_read(
        file_path: Annotated[str, Field(description="Absolute file path")],
        offset: Annotated[int, Field(default=0, description="Line offset from start")],
        limit: Annotated[int, Field(default=2000, description="Max lines to return")],
        runtime: Annotated[ToolRuntime | None, "Injected by LangChain ToolNode"] = None,
    ) -> ToolMessage | str:
        result: ReadResult = backend.read(file_path, offset, limit)
        if result.is_multimodal and result.content is not None:
            tool_call_id = getattr(runtime, "tool_call_id", "") or ""
            return ToolMessage(
                content_blocks=[
                    {
                        "type": result.file_type,
                        "base64": result.content,
                        "mime_type": result.mime_type,
                    }
                ],
                name="read",
                tool_call_id=tool_call_id,
                additional_kwargs={
                    "read_file_path": file_path,
                    "read_file_media_type": result.mime_type,
                },
                status="success",
            )
        return str(result)

    async def async_read(
        file_path: Annotated[str, Field(description="Absolute file path")],
        offset: Annotated[int, Field(default=0, description="Line offset from start")],
        limit: Annotated[int, Field(default=2000, description="Max lines to return")],
        runtime: Annotated[ToolRuntime | None, "Injected by LangChain ToolNode"] = None,
    ) -> ToolMessage | str:
        result: ReadResult = await backend.aread(file_path, offset, limit)
        if result.is_multimodal and result.content is not None:
            tool_call_id = getattr(runtime, "tool_call_id", "") or ""
            return ToolMessage(
                content_blocks=[
                    {
                        "type": result.file_type,
                        "base64": result.content,
                        "mime_type": result.mime_type,
                    }
                ],
                name="read",
                tool_call_id=tool_call_id,
                additional_kwargs={
                    "read_file_path": file_path,
                    "read_file_media_type": result.mime_type,
                },
                status="success",
            )
        return str(result)

    return StructuredTool.from_function(
        name="read",
        description=(
            "Read the contents of a file. "
            "For text files, returns content with line numbers. "
            "For image, audio, video, and PDF files, returns multimodal "
            "content blocks that the model can understand directly. "
            "Supports offset and limit for pagination (text only)."
        ),
        func=sync_read,
        coroutine=async_read,
        args_schema=_ReadSchema,
    )


def build_core_tools(backend: BackendProtocol) -> list[StructuredTool]:
    """Build the six core ``StructuredTool`` instances from *backend*.

    The ``read`` tool is specially handled via :func:`_build_sync_read_tool`:
    when the result has a non-text ``file_type``, it returns a ``ToolMessage``
    with ``content_blocks`` (LangChain multimodal format) instead of a plain
    string, and correctly propagates ``tool_call_id`` from the ``ToolRuntime``.
    """

    def _make_func(method_name: str):
        def tool_func(**kwargs):
            m = getattr(backend, method_name)
            if method_name == "grep":
                kwargs = {k: v for k, v in kwargs.items() if v is not None}
            return str(m(**kwargs))
        return tool_func

    def _make_coro(method_name: str):
        async def tool_coro(**kwargs):
            async_m = getattr(backend, _ASYNC_METHOD_MAP[method_name])
            if method_name == "grep":
                kwargs = {k: v for k, v in kwargs.items() if v is not None}
            return str(await async_m(**kwargs))
        return tool_coro

    tools: list[StructuredTool] = [_build_sync_read_tool(backend)]

    for spec in _CORE_TOOLS:
        if spec["name"] == "read":
            continue  # already added above
        from pydantic import create_model

        tools.append(
            StructuredTool(
                name=spec["name"],
                description=spec["description"],
                args_schema=create_model(
                    f"{spec['name'].title()}Schema",
                    **spec["fields"],
                ),
                func=_make_func(spec["method"]),
                coroutine=_make_coro(spec["method"]),
            )
        )
    return tools


# ---------------------------------------------------------------------------
# BackendToolsMiddleware
# ---------------------------------------------------------------------------


class BackendToolsMiddleware(AgentMiddleware[AgentState, ContextT, ResponseT]):
    """Middleware that registers all tools: core + backend extras.

    Also intercepts oversized tool results and evicts them to the
    filesystem (``/.mambo/large_tool_results/<tool_call_id>``),
    replacing the message with a truncated preview.

    Parameters:
        backend: The backend providing file-system operations.
        custom_system_prompt: Optional text appended to the system prompt.
        tool_token_limit_before_evict:
            Token threshold above which a tool result is evicted.
            Pass ``None`` to disable eviction entirely.
            Defaults to 20000.
    """

    state_schema = AgentState

    def __init__(
        self,
        backend: BackendProtocol,
        *,
        custom_system_prompt: str | None = None,
        tool_token_limit_before_evict: int | None = _DEFAULT_TOOL_TOKEN_LIMIT,
    ) -> None:
        super().__init__()
        self.backend = backend
        self.custom_prompt = custom_system_prompt
        self._evict_limit = tool_token_limit_before_evict

        # Core tools + extra tools from backend
        self.tools = build_core_tools(backend) + list(backend.tools)

    # ------------------------------------------------------------------
    # wrap_model_call / awrap_model_call — system prompt injection
    # ------------------------------------------------------------------

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        extra = self._build_extra_prompt()
        if extra is not None:
            request = request.override(
                system_message=_append_to_system_message(
                    request.system_message, extra
                )
            )
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        extra = self._build_extra_prompt()
        if extra is not None:
            request = request.override(
                system_message=_append_to_system_message(
                    request.system_message, extra
                )
            )
        return await handler(request)

    # ------------------------------------------------------------------
    # wrap_tool_call / awrap_tool_call — large result eviction
    # ------------------------------------------------------------------

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage],
    ) -> ToolMessage:
        result = handler(request)
        return self._maybe_evict(result, request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage]],
    ) -> ToolMessage:
        result = await handler(request)
        return await self._amaybe_evict(result, request)

    # ------------------------------------------------------------------
    # Eviction logic
    # ------------------------------------------------------------------

    def _should_evict(self, tool_name: str) -> bool:
        """Return True if eviction is enabled and applicable to this tool."""
        if self._evict_limit is None:
            return False
        return tool_name not in _TOOLS_EXCLUDED_FROM_EVICTION

    def _evict(
        self,
        message: ToolMessage,
        tool_call_id: str,
    ) -> ToolMessage:
        """Write full content to filesystem, return truncated preview message.

        Args:
            message: The original ToolMessage with large content.
            tool_call_id: Sanitized tool_call_id used as the file name.

        Returns:
            A new ToolMessage whose content is the truncated preview.
            Non-text content blocks (images, audio, etc.) are preserved.
        """
        text = _extract_text_from_tool_message(message)
        if _estimate_tokens(text) <= self._evict_limit:  # type: ignore[operator]
            return message

        file_path = f"{_EVICTION_PREFIX}/{tool_call_id}"
        write_result = self.backend.write(file_path, text)
        if write_result.error:
            # If we cannot persist the content, return the original — better
            # to risk context overflow than to silently lose data.
            return message

        preview = _build_preview(text)
        replacement = _TOOL_RESULT_EVICTED_MSG.format(
            file_path=file_path,
            preview=preview,
        )
        evicted_content = _build_evicted_content(message, replacement)
        return ToolMessage(
            content=evicted_content,
            tool_call_id=message.tool_call_id,
            name=message.name,
            status=message.status,
        )

    async def _aevict(
        self,
        message: ToolMessage,
        tool_call_id: str,
    ) -> ToolMessage:
        """Async version of :meth:`_evict`.

        Non-text content blocks (images, audio, etc.) are preserved.
        """
        text = _extract_text_from_tool_message(message)
        if _estimate_tokens(text) <= self._evict_limit:  # type: ignore[operator]
            return message

        file_path = f"{_EVICTION_PREFIX}/{tool_call_id}"
        write_result = await self.backend.awrite(file_path, text)
        if write_result.error:
            return message

        preview = _build_preview(text)
        replacement = _TOOL_RESULT_EVICTED_MSG.format(
            file_path=file_path,
            preview=preview,
        )
        evicted_content = _build_evicted_content(message, replacement)
        return ToolMessage(
            content=evicted_content,
            tool_call_id=message.tool_call_id,
            name=message.name,
            status=message.status,
        )

    def _maybe_evict(
        self,
        result: ToolMessage | Command,  # type: ignore[type-arg]
        request: ToolCallRequest,
    ) -> ToolMessage | Command:  # type: ignore[type-arg]
        """Check and evict the result if needed (sync)."""
        if isinstance(result, Command):
            return self._maybe_evict_command(result)
        tool_name = request.tool_call.get("name", "")
        if not self._should_evict(tool_name):
            return result
        sane_id = _sanitize_tool_call_id(request.tool_call["id"])
        return self._evict(result, sane_id)

    def _maybe_evict_command(
        self,
        command: Command,  # type: ignore[type-arg]
    ) -> Command:  # type: ignore[type-arg]
        """Process a ``Command`` result, evicting oversized ToolMessages inside.

        Each ``ToolMessage`` in ``command.update["messages"]`` is checked
        individually; other message types are passed through unchanged.
        All messages share the same eviction threshold (``self._evict_limit``)
        since they originate from the same tool call.
        """
        update = command.update
        if update is None:
            return command
        messages: list = update.get("messages", [])
        if not messages:
            return command

        any_evicted = False
        processed_messages: list = []
        for msg in messages:
            if not isinstance(msg, ToolMessage):
                processed_messages.append(msg)
                continue
            text = _extract_text_from_tool_message(msg)
            if self._evict_limit is not None and _estimate_tokens(text) > self._evict_limit:
                sane_id = _sanitize_tool_call_id(msg.tool_call_id)
                evicted = self._evict(msg, sane_id)
                processed_messages.append(evicted)
                any_evicted = True
            else:
                processed_messages.append(msg)

        if not any_evicted:
            return command

        return Command(
            update={**update, "messages": processed_messages},
            goto=command.goto,  # type: ignore[arg-type]
        )

    async def _amaybe_evict(
        self,
        result: ToolMessage | Command,  # type: ignore[type-arg]
        request: ToolCallRequest,
    ) -> ToolMessage | Command:  # type: ignore[type-arg]
        """Check and evict the result if needed (async)."""
        if isinstance(result, Command):
            return await self._amaybe_evict_command(result)
        tool_name = request.tool_call.get("name", "")
        if not self._should_evict(tool_name):
            return result
        sane_id = _sanitize_tool_call_id(request.tool_call["id"])
        return await self._aevict(result, sane_id)

    async def _amaybe_evict_command(
        self,
        command: Command,  # type: ignore[type-arg]
    ) -> Command:  # type: ignore[type-arg]
        """Async version of :meth:`_maybe_evict_command`."""
        update = command.update
        if update is None:
            return command
        messages: list = update.get("messages", [])
        if not messages:
            return command

        any_evicted = False
        processed_messages: list = []
        for msg in messages:
            if not isinstance(msg, ToolMessage):
                processed_messages.append(msg)
                continue
            text = _extract_text_from_tool_message(msg)
            if self._evict_limit is not None and _estimate_tokens(text) > self._evict_limit:
                sane_id = _sanitize_tool_call_id(msg.tool_call_id)
                evicted = await self._aevict(msg, sane_id)
                processed_messages.append(evicted)
                any_evicted = True
            else:
                processed_messages.append(msg)

        if not any_evicted:
            return command

        return Command(
            update={**update, "messages": processed_messages},
            goto=command.goto,  # type: ignore[arg-type]
        )

    # ------------------------------------------------------------------
    # System prompt helpers
    # ------------------------------------------------------------------

    def _build_extra_prompt(self) -> str | None:
        """Build additional text to append to the system prompt."""
        extras = [t.name for t in self.backend.tools]
        text = f"## Backend\n\n{self.backend.description}"
        if extras:
            text += f"\nExtra tools: {', '.join(extras)}"
        if self.custom_prompt:
            text = self.custom_prompt + "\n\n" + text
        return text


def _append_to_system_message(
    existing: SystemMessage | None,
    extra: str,
) -> SystemMessage:
    """Append *extra* text to an existing SystemMessage, or create a new one."""
    if existing is None:
        return SystemMessage(content=extra)
    content = existing.content
    if isinstance(content, str):
        return SystemMessage(content=content + "\n\n" + extra)
    elif isinstance(content, list):
        return SystemMessage(
            content=[*content, {"type": "text", "text": "\n\n" + extra}]
        )
    return SystemMessage(content=f"{content}\n\n{extra}")
