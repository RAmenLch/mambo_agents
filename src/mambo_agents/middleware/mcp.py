"""MCP (Model Context Protocol) middleware for mambo_agents.

Provides two meta-tools — ``mcp_get_tool_description`` and ``mcp_call_tool`` —
that serve as an indexing layer over multiple MCP servers.  Instead of
exposing potentially hundreds of MCP tools directly to the model (which
would bloat the system prompt), this middleware publishes a compact tool
index and lets the model look up descriptions and execute calls on demand.

Architecture
------------

1. **Init**: ``MCPServerConfig`` list → ``MultiServerMCPClient`` → collect
   tool index (name, description, inputSchema) from each server.  Servers
   that fail to connect are marked as *unavailable* but never crash the agent.

2. **System prompt**: injects a compact MCP server + tool listing so the
   model knows what is available.

3. **mcp_get_tool_description** (no MCP connection needed): pure in-memory
   lookup returning full JSON Schema for requested tools.

4. **mcp_call_tool** (connect → execute → disconnect per call): opens a
   temporary session to the target MCP server, calls the tool, and returns
   the result.  Errors (connection failure, tool error) are returned as
   structured text — they never crash the agent run.

Usage
-----

.. code-block:: python

    from mambo_agents.middleware.mcp import MCPMiddleware, MCPServerConfig

    middleware = MCPMiddleware(
        servers=[
            MCPServerConfig(
                name="math",
                transport="stdio",
                command="python",
                args=["/path/to/math_server.py"],
            ),
            MCPServerConfig(
                name="weather",
                transport="sse",
                url="http://localhost:8000/mcp",
            ),
        ],
    )

    agent = create_mambo_agent(
        "gpt-4o",
        middleware=[middleware],
    )
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import Annotated, Any, Literal

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelRequest,
    ModelResponse,
    ResponseT,
)
from langchain.tools import ToolRuntime
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import Connection, create_session
from langgraph.prebuilt.tool_node import ToolCallRequest
from mcp import ClientSession
from mcp.types import CallToolResult, Tool as MCPTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MCP_GET_TOOL_DESCRIPTION_NAME = "mcp_get_tool_description"
_MCP_CALL_TOOL_NAME = "mcp_call_tool"

_SYSTEM_PROMPT_TEMPLATE = """
## MCP Servers

You have access to the following MCP (Model Context Protocol) servers and
their tools.  Because the tool list may be large, tools are **not** directly
callable.  Instead use two meta-tools:

- `{desc_tool}` — look up the full JSON Schema (parameters, types,
  defaults) for one or more tools before calling them.
- `{call_tool}` — execute a tool on a specific MCP server with the
  arguments you prepared.

**Available servers:**

{server_list}

**Workflow:**
1. When you need to use an MCP tool, first call `{desc_tool}` with the
   `[[server_name, tool_name], ...]` pairs you are interested in.
2. Use the returned JSON Schema to construct the correct arguments.
3. Call `{call_tool}` with the server name, tool name, and arguments.
"""

_SERVER_AVAILABLE_TEMPLATE = """### {name} (available)
{tool_list}"""

_SERVER_UNAVAILABLE_TEMPLATE = """### {name} (unavailable)
  Reason: {reason}"""

_TOOL_ITEM_TEMPLATE = "  - **{tool_name}**: {description}"

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class MCPServerConfig(BaseModel):
    """Configuration for a single MCP server.

    Mirrors ``langchain_mcp_adapters.sessions.Connection`` fields.  Extra
    fields not listed here (e.g. ``session_kwargs``, ``encoding``,
    ``sse_read_timeout``) can be passed through ``extra_kwargs``.
    """

    model_config = {"extra": "forbid"}

    name: str = Field(description="Unique name for this MCP server.")
    transport: Literal["stdio", "sse", "streamable_http", "websocket"] = "stdio"
    command: str | None = Field(
        default=None,
        description="Executable command (stdio transport).",
    )
    args: list[str] = Field(
        default_factory=list,
        description="Command-line arguments (stdio transport).",
    )
    env: dict[str, str] | None = Field(
        default=None,
        description="Environment variables (stdio transport).",
    )
    cwd: str | None = Field(
        default=None,
        description="Working directory (stdio transport).",
    )
    url: str | None = Field(
        default=None,
        description="Server URL (sse / streamable_http / websocket).",
    )
    headers: dict[str, Any] | None = Field(
        default=None,
        description="HTTP headers (sse / streamable_http).",
    )
    timeout: float | None = Field(
        default=None,
        description="HTTP timeout in seconds.",
    )

    def to_connection(self) -> Connection:
        """Convert to a ``Connection`` dict for ``MultiServerMCPClient``."""
        conn: dict[str, Any] = {"transport": self.transport}

        if self.command is not None:
            conn["command"] = self.command
        if self.args:
            conn["args"] = self.args
        if self.env is not None:
            conn["env"] = self.env
        if self.cwd is not None:
            conn["cwd"] = self.cwd
        if self.url is not None:
            conn["url"] = self.url
        if self.headers is not None:
            conn["headers"] = self.headers
        if self.timeout is not None:
            conn["timeout"] = self.timeout

        return conn  # type: ignore[return-value]


class _ToolIndexEntry(BaseModel):
    """Internal: cached metadata for one MCP tool."""

    model_config = {"frozen": True}

    server_name: str
    tool_name: str
    description: str
    input_schema: dict[str, Any]


class _ServerMeta(BaseModel):
    """Internal: metadata for one MCP server after init."""

    model_config = {"frozen": True}

    name: str
    available: bool
    tools: list[_ToolIndexEntry] = Field(default_factory=list)
    error: str | None = None


# ---------------------------------------------------------------------------
# Tool input schemas
# ---------------------------------------------------------------------------


class _GetToolDescriptionInput(BaseModel):
    """Input schema for ``mcp_get_tool_description``."""

    model_config = {"extra": "allow"}

    tool_requests: list[list[str]] = Field(
        description=(
            "List of [server_name, tool_name] pairs to look up. "
            'Example: [["math", "add"], ["weather", "forecast"]]'
        ),
    )


class _CallToolInput(BaseModel):
    """Input schema for ``mcp_call_tool``."""

    model_config = {"extra": "allow"}

    server_name: str = Field(
        description="Name of the MCP server to call.",
    )
    tool_name: str = Field(
        description="Name of the tool to execute on the server.",
    )
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments to pass to the tool as a JSON object.",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _append_to_system_message(
    existing: SystemMessage | None,
    text: str,
) -> SystemMessage:
    """Append *text* to a ``SystemMessage``, or create a new one."""
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


def _format_tool_list(tools: list[_ToolIndexEntry]) -> str:
    """Format a list of tool entries for system prompt display."""
    if not tools:
        return "  (no tools)"
    lines: list[str] = []
    for t in tools:
        desc = t.description or "(no description)"
        lines.append(_TOOL_ITEM_TEMPLATE.format(tool_name=t.tool_name, description=desc))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool index collection (init-time, sync wrapper over async)
# ---------------------------------------------------------------------------


def _collect_tool_index(
    servers: Sequence[MCPServerConfig],
) -> list[_ServerMeta]:
    """Connect to each MCP server, list its tools, build the index.

    Runs inside ``asyncio.run()`` — called once during ``__init__``.
    Each server failure is captured individually; the caller always gets a
    full list of ``_ServerMeta`` entries.
    """
    connections: dict[str, Connection] = {}
    for s in servers:
        connections[s.name] = s.to_connection()

    client = MultiServerMCPClient(connections)

    async def _collect() -> list[_ServerMeta]:
        results: list[_ServerMeta] = []
        for s in servers:
            try:
                tools = await _list_tools_via_session(client, s.name)
                entries = [
                    _ToolIndexEntry(
                        server_name=s.name,
                        tool_name=t.name,
                        description=t.description or "",
                        input_schema=t.inputSchema,
                    )
                    for t in tools
                ]
                results.append(
                    _ServerMeta(name=s.name, available=True, tools=entries),
                )
            except Exception as exc:
                logger.warning(
                    "MCP server '%s' unavailable: %s", s.name, exc,
                )
                results.append(
                    _ServerMeta(
                        name=s.name,
                        available=False,
                        error=str(exc),
                    ),
                )
        return results

    return asyncio.run(_collect())


async def _list_tools_via_session(
    client: MultiServerMCPClient,
    server_name: str,
) -> list[MCPTool]:
    """List all tools from one MCP server via a temporary session."""
    from langchain_mcp_adapters.tools import _list_all_tools

    async with client.session(server_name) as session:
        return await _list_all_tools(session)


# ---------------------------------------------------------------------------
# Tool builders
# ---------------------------------------------------------------------------


def _build_get_tool_description(
    tool_index: dict[tuple[str, str], _ToolIndexEntry],
) -> StructuredTool:
    """Build the ``mcp_get_tool_description`` meta-tool."""

    def _sync(
        tool_requests: list[list[str]],
    ) -> str:
        results: list[dict[str, Any]] = []
        for req in tool_requests:
            if len(req) != 2:
                results.append(
                    {
                        "server_name": str(req[0]) if req else "?",
                        "tool_name": str(req[1]) if len(req) > 1 else "?",
                        "error": f"Invalid request format: {req!r}. Expected [server_name, tool_name].",
                    },
                )
                continue
            key = (req[0], req[1])
            entry = tool_index.get(key)
            if entry is None:
                results.append(
                    {
                        "server_name": req[0],
                        "tool_name": req[1],
                        "error": f"Tool '{req[1]}' not found on server '{req[0]}'.",
                    },
                )
            else:
                results.append(
                    {
                        "server_name": entry.server_name,
                        "tool_name": entry.tool_name,
                        "description": entry.description,
                        "input_schema": entry.input_schema,
                    },
                )
        return json.dumps(results, ensure_ascii=False, indent=2)

    return StructuredTool.from_function(
        name=_MCP_GET_TOOL_DESCRIPTION_NAME,
        description=(
            "Get detailed descriptions and parameter schemas for MCP tools. "
            "Use this to look up a tool's input JSON Schema before calling it "
            "with mcp_call_tool. Pass a list of [server_name, tool_name] pairs."
        ),
        func=_sync,
        coroutine=_make_async(_sync),
        args_schema=_GetToolDescriptionInput,
    )


def _build_call_tool(
    servers_meta: list[_ServerMeta],
    client: MultiServerMCPClient,
) -> StructuredTool:
    """Build the ``mcp_call_tool`` meta-tool.

    Each call opens a fresh session, executes the tool, and closes the session.
    Errors (connection, tool execution) are returned as structured error text
    rather than raising exceptions.
    """
    # Quick-lookup set of available server names
    _available_names: frozenset[str] = frozenset(
        m.name for m in servers_meta if m.available
    )

    async def _call(
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str:
        # ---- validate server ----
        if server_name not in _available_names:
            # Check if it's a known-but-unavailable server
            meta = next((m for m in servers_meta if m.name == server_name), None)
            if meta is not None:
                return json.dumps(
                    {
                        "server_name": server_name,
                        "tool_name": tool_name,
                        "error": f"Server '{server_name}' is unavailable: {meta.error}",
                    },
                    ensure_ascii=False,
                )
            return json.dumps(
                {
                    "server_name": server_name,
                    "tool_name": tool_name,
                    "error": f"Unknown server '{server_name}'. Available: {sorted(_available_names)}",
                },
                ensure_ascii=False,
            )

        # ---- execute via temporary session ----
        try:
            async with client.session(server_name) as session:
                result: CallToolResult = await session.call_tool(
                    tool_name,
                    arguments,
                )
        except Exception as exc:
            logger.warning(
                "MCP call_tool failed: server=%s tool=%s error=%s",
                server_name,
                tool_name,
                exc,
            )
            return json.dumps(
                {
                    "server_name": server_name,
                    "tool_name": tool_name,
                    "error": f"Tool execution failed: {exc}",
                },
                ensure_ascii=False,
            )

        # ---- format result ----
        return _format_call_tool_result(result, server_name, tool_name)

    async def _sync_wrapper(
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str:
        """Sync wrapper that runs the async call via asyncio.run."""
        return await _call(server_name, tool_name, arguments)

    return StructuredTool.from_function(
        name=_MCP_CALL_TOOL_NAME,
        description=(
            "Execute a tool on an MCP server. "
            "Use mcp_get_tool_description first to get the required parameter schema. "
            "Arguments must be a JSON object matching the tool's inputSchema."
        ),
        func=None,  # sync not supported (needs async session)
        coroutine=_sync_wrapper,
        args_schema=_CallToolInput,
    )


def _format_call_tool_result(
    result: CallToolResult,
    server_name: str,
    tool_name: str,
) -> str:
    """Convert a ``CallToolResult`` into a JSON string for the model."""
    if result.isError:
        text_parts: list[str] = []
        for block in result.content:
            if hasattr(block, "text"):
                text_parts.append(block.text)
        return json.dumps(
            {
                "server_name": server_name,
                "tool_name": tool_name,
                "is_error": True,
                "error": "\n".join(text_parts) if text_parts else "Unknown error",
            },
            ensure_ascii=False,
        )

    # Success — extract text content
    text_parts: list[str] = []
    for block in result.content:
        if hasattr(block, "text"):
            text_parts.append(block.text)

    structured = None
    if result.structuredContent is not None:
        structured = result.structuredContent

    output: dict[str, Any] = {
        "server_name": server_name,
        "tool_name": tool_name,
        "is_error": False,
        "content": text_parts,
    }
    if structured is not None:
        output["structured_content"] = structured

    return json.dumps(output, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Async helpers
# ---------------------------------------------------------------------------


def _make_async(sync_fn: Callable[..., str]) -> Callable[..., Awaitable[str]]:
    """Wrap a sync function as an async coroutine."""

    async def _wrapper(*args: Any, **kwargs: Any) -> str:
        return sync_fn(*args, **kwargs)

    return _wrapper


# ---------------------------------------------------------------------------
# MCPMiddleware
# ---------------------------------------------------------------------------


class MCPMiddleware(AgentMiddleware[AgentState, ContextT, ResponseT]):
    """Middleware that provides MCP tools via ``mcp_get_tool_description``
    and ``mcp_call_tool`` meta-tools.

    Parameters
    ----------
    servers:
        List of ``MCPServerConfig`` objects, one per MCP server.

    Example
    -------

    .. code-block:: python

        from mambo_agents.middleware.mcp import MCPMiddleware, MCPServerConfig

        middleware = MCPMiddleware(
            servers=[
                MCPServerConfig(
                    name="filesystem",
                    transport="stdio",
                    command="npx",
                    args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                ),
            ],
        )

        agent = create_mambo_agent(
            "gpt-4o",
            middleware=[middleware],
        )
    """

    state_schema = AgentState

    def __init__(
        self,
        *,
        servers: Sequence[MCPServerConfig],
    ) -> None:
        super().__init__()

        if not servers:
            raise ValueError("At least one MCP server config is required.")

        self._servers_config = list(servers)

        # ---- Build connections & MultiServerMCPClient ----
        connections: dict[str, Connection] = {}
        for s in self._servers_config:
            connections[s.name] = s.to_connection()
        self._client = MultiServerMCPClient(connections)

        # ---- Collect tool index (init-time, may block briefly) ----
        self._servers_meta = _collect_tool_index(self._servers_config)

        # ---- Build tool lookup index ----
        self._tool_index: dict[tuple[str, str], _ToolIndexEntry] = {}
        for meta in self._servers_meta:
            for entry in meta.tools:
                self._tool_index[(entry.server_name, entry.tool_name)] = entry

        # ---- Register meta-tools ----
        self.tools: list[BaseTool] = [
            _build_get_tool_description(self._tool_index),
            _build_call_tool(self._servers_meta, self._client),
        ]

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        """Build the MCP section of the system prompt."""
        server_blocks: list[str] = []
        for meta in self._servers_meta:
            if meta.available:
                server_blocks.append(
                    _SERVER_AVAILABLE_TEMPLATE.format(
                        name=meta.name,
                        tool_list=_format_tool_list(meta.tools),
                    ),
                )
            else:
                server_blocks.append(
                    _SERVER_UNAVAILABLE_TEMPLATE.format(
                        name=meta.name,
                        reason=meta.error or "Unknown error",
                    ),
                )

        return _SYSTEM_PROMPT_TEMPLATE.format(
            desc_tool=_MCP_GET_TOOL_DESCRIPTION_NAME,
            call_tool=_MCP_CALL_TOOL_NAME,
            server_list="\n\n".join(server_blocks),
        )

    # ------------------------------------------------------------------
    # wrap_model_call / awrap_model_call
    # ------------------------------------------------------------------

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        mcp_prompt = self._build_system_prompt()
        request = request.override(
            system_message=_append_to_system_message(
                request.system_message, mcp_prompt,
            ),
        )
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[
            [ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]
        ],
    ) -> ModelResponse[ResponseT]:
        mcp_prompt = self._build_system_prompt()
        request = request.override(
            system_message=_append_to_system_message(
                request.system_message, mcp_prompt,
            ),
        )
        return await handler(request)
