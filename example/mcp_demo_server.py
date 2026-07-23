"""Minimal MCP demo server for the MCP + security_review example (10_mcp_security_review.py).

Exposes a handful of tools so the example can demonstrate targeted
security review — the agent must get human approval for ``delete_data``
while ``echo`` and ``add`` are auto-approved.

No external dependencies beyond the ``mcp`` package (pulled in by
``langchain-mcp-adapters``).

Run this script directly; it listens on stdio:

    python example/mcp_demo_server.py
"""

import asyncio
import json
import sys
from typing import Any

from mcp.server import Server, NotificationOptions
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def tool_echo(message: str) -> str:
    """Return the same message back."""
    return f"Echo: {message}"


def tool_add(a: float, b: float) -> str:
    """Add two numbers and return the result."""
    return f"{a} + {b} = {a + b}"


def tool_delete_data(table: str, key: str) -> str:
    """Simulate deleting a record from a database table."""
    return json.dumps({"deleted": True, "table": table, "key": key})


_TOOLS = [
    Tool(
        name="echo",
        description="Echo back the given message.",
        inputSchema={
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "The message to echo."},
            },
            "required": ["message"],
        },
    ),
    Tool(
        name="add",
        description="Add two numbers.",
        inputSchema={
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First number."},
                "b": {"type": "number", "description": "Second number."},
            },
            "required": ["a", "b"],
        },
    ),
    Tool(
        name="delete_data",
        description="Delete a record from a database table. DANGEROUS — cannot be undone.",
        inputSchema={
            "type": "object",
            "properties": {
                "table": {"type": "string", "description": "Table name."},
                "key": {"type": "string", "description": "Record key to delete."},
            },
            "required": ["table", "key"],
        },
    ),
]

_HANDLERS = {
    "echo": tool_echo,
    "add": tool_add,
    "delete_data": tool_delete_data,
}


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

async def main() -> None:
    server = Server(
        name="demo-server",
        version="0.1.0",
    )

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return _TOOLS

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        handler = _HANDLERS.get(name)
        if handler is None:
            raise ValueError(f"Unknown tool: {name}")
        result = handler(**arguments)
        return [TextContent(type="text", text=str(result))]

    async with stdio_server() as (reader, writer):
        await server.run(reader, writer, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
