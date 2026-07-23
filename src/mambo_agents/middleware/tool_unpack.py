"""Generic tool-call unpack protocol for wrapped tools (e.g., MCP).

When a middleware wraps inner tools behind a generic meta-tool (like
``mcp_call_tool``), downstream consumers such as
:class:`AutoSecurityReviewMiddleware` cannot inspect the *real* tool
being invoked — they only see the wrapper.

This module defines a lightweight protocol so that wrapping middlewares
can expose an *unpacker* that extracts the effective tool name, arguments
and description from a wrapped ``ToolCall``.  Security review (and other
interested middlewares) can then make targeted decisions based on the
inner tool identity — without changing the outer ``tool_call_id`` or
interrupt/replay machinery.

Usage sketch
------------

.. code-block:: python

    from mambo_agents.middleware.mcp import mcp_tool_name

    mcp = MCPMiddleware(servers=[...])

    agent = create_mambo_agent(
        "gpt-4o",
        middleware=[mcp],
        interrupt_on={
            "mcp_call_tool": True,
            mcp_tool_name("filesystem", "rm"): True,
        },
        security_review=SecurityReviewConfig(
            review_tools=frozenset([
                mcp_tool_name("filesystem", "rm"),
                "write",
            ]),
            tool_unpackers=[mcp.tool_unpacker],
        ),
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolUnpackResult:
    """Result of unpacking a wrapped (meta-)tool call into its inner tool.

    Attributes
    ----------
    effective_tool_name:
        The "real" tool name to use for security review or policy matching.
        By convention, wrapping middlewares should produce a colon-delimited
        name such as ``"mcp:<server>:<tool>"``.
    effective_args:
        The arguments for the *inner* tool (not the wrapper's own arguments).
    tool_description:
        Optional human-readable description of the inner tool's purpose.
    """

    effective_tool_name: str
    effective_args: dict[str, Any]
    tool_description: str | None = None
