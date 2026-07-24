"""Tests for MCP middleware — ``mambo_agents.middleware.mcp``.

All tests use in-process ``MultiServerMCPClient`` with mock sessions,
no external MCP server required.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import SystemMessage
from pydantic import ValidationError

from mambo_agents.middleware.mcp import (
    MCPMiddleware,
    MCPServerConfig,
    _CallToolInput,
    _GetToolDescriptionInput,
    _ServerMeta,
    _ToolIndexEntry,
    _append_to_system_message,
    _build_call_tool,
    _build_get_tool_description,
    _build_direct_tool,
    _format_call_tool_result,
    _format_tool_list,
    mcp_tool_name,
)
from mambo_agents.middleware.tool_unpack import ToolUnpackResult


# ============================================================================
# MCPServerConfig
# ============================================================================


class TestMCPServerConfig:
    """Tests for ``MCPServerConfig`` Pydantic model and ``to_connection()``."""

    def test_stdio_minimal(self):
        cfg = MCPServerConfig(
            name="test",
            transport="stdio",
            command="python",
            args=["server.py"],
        )
        conn = cfg.to_connection()
        assert conn["transport"] == "stdio"
        assert conn["command"] == "python"
        assert conn["args"] == ["server.py"]
        assert "url" not in conn

    def test_stdio_full(self):
        cfg = MCPServerConfig(
            name="full",
            transport="stdio",
            command="node",
            args=["--inspect", "server.js"],
            env={"NODE_ENV": "test"},
            cwd="/tmp",
        )
        conn = cfg.to_connection()
        assert conn["transport"] == "stdio"
        assert conn["command"] == "node"
        assert conn["args"] == ["--inspect", "server.js"]
        assert conn["env"] == {"NODE_ENV": "test"}
        assert conn["cwd"] == "/tmp"

    def test_sse(self):
        cfg = MCPServerConfig(
            name="weather",
            transport="sse",
            url="http://localhost:8000/mcp",
            headers={"Authorization": "Bearer token"},
            timeout=10.0,
        )
        conn = cfg.to_connection()
        assert conn["transport"] == "sse"
        assert conn["url"] == "http://localhost:8000/mcp"
        assert conn["headers"] == {"Authorization": "Bearer token"}
        assert conn["timeout"] == 10.0

    def test_default_transport_is_stdio(self):
        cfg = MCPServerConfig(name="d", command="cmd", args=[])
        conn = cfg.to_connection()
        assert conn["transport"] == "stdio"

    def test_name_is_required(self):
        with pytest.raises(ValidationError):
            MCPServerConfig(transport="stdio", command="cmd", args=[])  # type: ignore[arg-type]

    def test_invalid_transport_rejected(self):
        with pytest.raises(ValidationError):
            MCPServerConfig(name="x", transport="invalid", command="c", args=[])  # type: ignore[arg-type]

    def test_none_fields_excluded_from_connection(self):
        cfg = MCPServerConfig(
            name="min",
            transport="stdio",
            command="cmd",
            args=[],
            url=None,
            headers=None,
            env=None,
            cwd=None,
            timeout=None,
        )
        conn = cfg.to_connection()
        assert "url" not in conn
        assert "headers" not in conn
        assert "env" not in conn
        assert "cwd" not in conn
        assert "timeout" not in conn


# ============================================================================
# _ToolIndexEntry / _ServerMeta
# ============================================================================


class TestToolIndexEntry:
    def test_frozen(self):
        entry = _ToolIndexEntry(
            server_name="s",
            tool_name="t",
            description="desc",
            input_schema={"type": "object"},
        )
        with pytest.raises(ValidationError):
            entry.tool_name = "new"  # type: ignore[misc]

    def test_fields(self):
        entry = _ToolIndexEntry(
            server_name="math",
            tool_name="add",
            description="Add numbers",
            input_schema={"type": "object", "properties": {"a": {"type": "number"}}},
        )
        assert entry.server_name == "math"
        assert entry.tool_name == "add"
        assert entry.description == "Add numbers"
        assert entry.input_schema["type"] == "object"


class TestServerMeta:
    def test_defaults(self):
        meta = _ServerMeta(name="test", available=True)
        assert meta.tools == []
        assert meta.error is None

    def test_unavailable_with_error(self):
        meta = _ServerMeta(name="bad", available=False, error="connection refused")
        assert not meta.available
        assert meta.error == "connection refused"

    def test_with_tools(self):
        tools = [
            _ToolIndexEntry(server_name="s", tool_name="t1", description="d1", input_schema={}),
        ]
        meta = _ServerMeta(name="s", available=True, tools=tools)
        assert len(meta.tools) == 1


# ============================================================================
# Tool input schemas
# ============================================================================


class TestGetToolDescriptionInput:
    def test_valid(self):
        inp = _GetToolDescriptionInput(
            tool_requests=[["math", "add"], ["weather", "forecast"]],
        )
        assert len(inp.tool_requests) == 2
        assert inp.tool_requests[0] == ["math", "add"]

    def test_empty_list_ok(self):
        inp = _GetToolDescriptionInput(tool_requests=[])
        assert inp.tool_requests == []

    def test_missing_required(self):
        with pytest.raises(ValidationError):
            _GetToolDescriptionInput()  # type: ignore[arg-type]


class TestCallToolInput:
    def test_valid(self):
        inp = _CallToolInput(
            server_name="math",
            tool_name="add",
            arguments={"a": 1, "b": 2},
        )
        assert inp.server_name == "math"
        assert inp.arguments == {"a": 1, "b": 2}

    def test_default_arguments(self):
        inp = _CallToolInput(server_name="s", tool_name="t")
        assert inp.arguments == {}

    def test_missing_required(self):
        with pytest.raises(ValidationError):
            _CallToolInput(tool_name="t")  # type: ignore[arg-type]


# ============================================================================
# _append_to_system_message
# ============================================================================


class TestAppendToSystemMessage:
    def test_none_existing(self):
        result = _append_to_system_message(None, "hello")
        assert isinstance(result, SystemMessage)
        assert result.content == "hello"

    def test_str_content(self):
        existing = SystemMessage(content="base")
        result = _append_to_system_message(existing, "extra")
        assert result.content == "base\n\nextra"

    def test_list_content(self):
        existing = SystemMessage(content=[{"type": "text", "text": "base"}])
        result = _append_to_system_message(existing, "extra")
        assert isinstance(result.content, list)
        assert len(result.content) == 2
        assert result.content[1] == {"type": "text", "text": "\n\nextra"}


# ============================================================================
# _format_tool_list
# ============================================================================


class TestFormatToolList:
    def test_empty(self):
        assert "no tools" in _format_tool_list([])

    def test_with_tools(self):
        tools = [
            _ToolIndexEntry(server_name="s", tool_name="add", description="Add numbers", input_schema={}),
            _ToolIndexEntry(server_name="s", tool_name="sub", description="Subtract", input_schema={}),
        ]
        result = _format_tool_list(tools)
        assert "**add**: Add numbers" in result
        assert "**sub**: Subtract" in result

    def test_missing_description(self):
        tools = [
            _ToolIndexEntry(server_name="s", tool_name="t", description="", input_schema={}),
        ]
        result = _format_tool_list(tools)
        assert "(no description)" in result


# ============================================================================
# _build_get_tool_description
# ============================================================================


class TestBuildGetToolDescription:
    def test_found(self):
        index = {
            ("math", "add"): _ToolIndexEntry(
                server_name="math",
                tool_name="add",
                description="Add numbers",
                input_schema={"type": "object", "properties": {"a": {"type": "number"}}},
            ),
        }
        tool = _build_get_tool_description(index)
        result_raw = tool.func([["math", "add"]])  # type: ignore[arg-type]
        result = json.loads(result_raw)
        assert len(result) == 1
        assert result[0]["server_name"] == "math"
        assert result[0]["tool_name"] == "add"
        assert result[0]["description"] == "Add numbers"
        assert result[0]["input_schema"]["type"] == "object"
        assert "error" not in result[0]

    def test_not_found(self):
        index: dict = {}
        tool = _build_get_tool_description(index)
        result_raw = tool.func([["math", "add"]])  # type: ignore[arg-type]
        result = json.loads(result_raw)
        assert len(result) == 1
        assert "error" in result[0]
        assert "not found" in result[0]["error"]

    def test_invalid_format(self):
        index: dict = {}
        tool = _build_get_tool_description(index)
        result_raw = tool.func([["only_one"]])  # type: ignore[arg-type]
        result = json.loads(result_raw)
        assert "error" in result[0]
        assert "Invalid request format" in result[0]["error"]

    def test_multiple_requests(self):
        index = {
            ("math", "add"): _ToolIndexEntry(
                server_name="math", tool_name="add", description="Add", input_schema={},
            ),
        }
        tool = _build_get_tool_description(index)
        result_raw = tool.func([["math", "add"], ["math", "missing"]])  # type: ignore[arg-type]
        result = json.loads(result_raw)
        assert len(result) == 2
        assert "error" not in result[0]
        assert "error" in result[1]

    def test_coroutine_works(self):
        """The async wrapper should return the same result as sync."""
        index = {
            ("s", "t"): _ToolIndexEntry(
                server_name="s", tool_name="t", description="d", input_schema={},
            ),
        }
        tool = _build_get_tool_description(index)
        import asyncio

        async def run():
            return await tool.coroutine([["s", "t"]])  # type: ignore[arg-type]

        result_raw = asyncio.run(run())
        result = json.loads(result_raw)
        assert result[0]["tool_name"] == "t"


# ============================================================================
# _format_call_tool_result
# ============================================================================


class TestFormatCallToolResult:
    def _make_text_content(self, text: str):
        """Create a mock MCP TextContent object."""
        from mcp.types import TextContent

        return TextContent(type="text", text=text)

    def test_success_single_text(self):
        from mcp.types import CallToolResult

        content = [self._make_text_content("result: 42")]
        result_obj = CallToolResult(content=content)
        output = _format_call_tool_result(result_obj, "math", "add")
        data = json.loads(output)
        assert data["is_error"] is False
        assert data["content"] == ["result: 42"]

    def test_success_with_structured(self):
        from mcp.types import CallToolResult

        content = [self._make_text_content("ok")]
        result_obj = CallToolResult(
            content=content,
            structuredContent={"sum": 3},
        )
        output = _format_call_tool_result(result_obj, "math", "add")
        data = json.loads(output)
        assert data["structured_content"] == {"sum": 3}

    def test_error(self):
        from mcp.types import CallToolResult

        content = [self._make_text_content("division by zero")]
        result_obj = CallToolResult(content=content, isError=True)
        output = _format_call_tool_result(result_obj, "math", "div")
        data = json.loads(output)
        assert data["is_error"] is True
        assert "division by zero" in data["error"]

    def test_error_empty_content(self):
        from mcp.types import CallToolResult

        result_obj = CallToolResult(content=[], isError=True)
        output = _format_call_tool_result(result_obj, "s", "t")
        data = json.loads(output)
        assert data["is_error"] is True
        assert "Unknown error" in data["error"]


# ============================================================================
# _build_call_tool — logic tests via mock
# ============================================================================


class TestBuildCallTool:
    """Tests for ``_build_call_tool`` using mocked ``MultiServerMCPClient``."""

    @pytest.fixture
    def servers_meta(self):
        return [
            _ServerMeta(name="math", available=True, tools=[]),
            _ServerMeta(name="bad", available=False, error="timeout"),
        ]

    def test_unavailable_server_returns_error(self, servers_meta):
        mock_client = MagicMock()
        tool = _build_call_tool(servers_meta, mock_client)

        import asyncio

        async def run():
            return await tool.coroutine(  # type: ignore[arg-type]
                server_name="bad", tool_name="any", arguments={},
            )

        result_raw = asyncio.run(run())
        data = json.loads(result_raw)
        assert "error" in data
        assert "unavailable" in data["error"]

    def test_unknown_server_returns_error(self, servers_meta):
        mock_client = MagicMock()
        tool = _build_call_tool(servers_meta, mock_client)

        import asyncio

        async def run():
            return await tool.coroutine(  # type: ignore[arg-type]
                server_name="nonexistent", tool_name="any", arguments={},
            )

        result_raw = asyncio.run(run())
        data = json.loads(result_raw)
        assert "Unknown server" in data["error"]

    def test_call_success(self, servers_meta):
        from contextlib import asynccontextmanager

        from mcp.types import CallToolResult, TextContent

        mock_client = MagicMock()
        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(
            return_value=CallToolResult(
                content=[TextContent(type="text", text="42")],
            ),
        )

        @asynccontextmanager
        async def _mock_session(*args, **kwargs):
            yield mock_session

        mock_client.session = _mock_session

        tool = _build_call_tool(servers_meta, mock_client)

        import asyncio

        async def run():
            return await tool.coroutine(  # type: ignore[arg-type]
                server_name="math",
                tool_name="add",
                arguments={"a": 1, "b": 2},
            )

        result_raw = asyncio.run(run())
        data = json.loads(result_raw)
        assert data["is_error"] is False
        assert "42" in data["content"]

    def test_call_connection_error_returns_error_text(self, servers_meta):
        from contextlib import asynccontextmanager

        mock_client = MagicMock()

        @asynccontextmanager
        async def _failing_session(*args, **kwargs):
            raise ConnectionError("refused")
            yield  # unreachable

        mock_client.session = _failing_session

        tool = _build_call_tool(servers_meta, mock_client)

        import asyncio

        async def run():
            return await tool.coroutine(  # type: ignore[arg-type]
                server_name="math",
                tool_name="add",
                arguments={},
            )

        result_raw = asyncio.run(run())
        data = json.loads(result_raw)
        assert "error" in data
        assert "refused" in data["error"]


# ============================================================================
# MCPMiddleware init
# ============================================================================


class TestMCPMiddlewareInit:
    def test_empty_servers_raises(self):
        with pytest.raises(ValueError, match="At least one MCP server"):
            MCPMiddleware(servers=[])

    @patch("mambo_agents.middleware.mcp._collect_tool_index")
    def test_registers_two_tools(self, mock_collect):
        mock_collect.return_value = [
            _ServerMeta(
                name="test",
                available=True,
                tools=[
                    _ToolIndexEntry(
                        server_name="test",
                        tool_name="dummy",
                        description="dummy",
                        input_schema={},
                    ),
                ],
            ),
        ]
        mw = MCPMiddleware(
            servers=[MCPServerConfig(name="test", command="cmd", args=[])],
            direct_tool_threshold=0,
        )
        tool_names = {t.name for t in mw.tools}
        assert "mcp_get_tool_description" in tool_names
        assert "mcp_call_tool" in tool_names

    @patch("mambo_agents.middleware.mcp._collect_tool_index")
    def test_system_prompt_contains_available_server(self, mock_collect):
        mock_collect.return_value = [
            _ServerMeta(
                name="math",
                available=True,
                tools=[
                    _ToolIndexEntry(
                        server_name="math",
                        tool_name="add",
                        description="Add numbers",
                        input_schema={},
                    ),
                ],
            ),
        ]
        mw = MCPMiddleware(
            servers=[MCPServerConfig(name="math", command="cmd", args=[])],
        )
        prompt = mw._build_system_prompt()
        assert "math" in prompt
        assert "available" in prompt
        assert "add" in prompt
        assert "Add numbers" in prompt
        assert "mcp_get_tool_description" in prompt
        assert "mcp_call_tool" in prompt

    @patch("mambo_agents.middleware.mcp._collect_tool_index")
    def test_system_prompt_shows_unavailable_server(self, mock_collect):
        mock_collect.return_value = [
            _ServerMeta(name="bad", available=False, error="timeout"),
        ]
        mw = MCPMiddleware(
            servers=[MCPServerConfig(name="bad", command="cmd", args=[])],
        )
        prompt = mw._build_system_prompt()
        assert "bad" in prompt
        assert "unavailable" in prompt
        assert "timeout" in prompt

    @patch("mambo_agents.middleware.mcp._collect_tool_index")
    def test_wrap_model_call_injects_prompt(self, mock_collect):
        from langchain.agents.middleware.types import ModelRequest

        mock_collect.return_value = [
            _ServerMeta(name="test", available=True, tools=[]),
        ]
        mw = MCPMiddleware(
            servers=[MCPServerConfig(name="test", command="cmd", args=[])],
        )

        req = ModelRequest(
            model=MagicMock(),
            messages=[],
            system_message=SystemMessage(content="base"),
        )

        def handler(r):
            return MagicMock()

        resp = mw.wrap_model_call(req, handler)
        assert resp is not None

    @patch("mambo_agents.middleware.mcp._collect_tool_index")
    def test_tool_index_built_correctly(self, mock_collect):
        mock_collect.return_value = [
            _ServerMeta(
                name="s",
                available=True,
                tools=[
                    _ToolIndexEntry(
                        server_name="s",
                        tool_name="t1",
                        description="d1",
                        input_schema={"p": 1},
                    ),
                    _ToolIndexEntry(
                        server_name="s",
                        tool_name="t2",
                        description="d2",
                        input_schema={"p": 2},
                    ),
                ],
            ),
        ]
        mw = MCPMiddleware(
            servers=[MCPServerConfig(name="s", command="cmd", args=[])],
        )
        assert ("s", "t1") in mw._tool_index
        assert ("s", "t2") in mw._tool_index
        assert mw._tool_index[("s", "t1")].input_schema == {"p": 1}


# ============================================================================
# mcp_tool_name
# ============================================================================


class TestMcpToolName:
    def test_basic(self):
        assert mcp_tool_name("server", "tool") == "server__tool"

    def test_with_underscores(self):
        assert mcp_tool_name("my_server", "my_tool") == "my_server__my_tool"

    def test_different_servers(self):
        assert mcp_tool_name("filesystem", "delete_config") == "filesystem__delete_config"
        assert mcp_tool_name("github", "create_pr") == "github__create_pr"


# ============================================================================
# Server name validation
# ============================================================================


class TestServerNameValidation:
    @patch("mambo_agents.middleware.mcp._collect_tool_index")
    def test_valid_names(self, mock_collect):
        mock_collect.return_value = []
        for name in ["s", "my_server", "server1", "a-b", "Abc_123"]:
            MCPMiddleware(
                servers=[MCPServerConfig(name=name, command="c", args=[])],
            )

    @patch("mambo_agents.middleware.mcp._collect_tool_index")
    def test_empty_name_raises(self, mock_collect):
        mock_collect.return_value = []
        with pytest.raises(ValueError, match="must not be empty"):
            MCPMiddleware(
                servers=[MCPServerConfig(name="", command="c", args=[])],
            )

    @patch("mambo_agents.middleware.mcp._collect_tool_index")
    def test_whitespace_name_raises(self, mock_collect):
        mock_collect.return_value = []
        with pytest.raises(ValueError, match="must not be empty"):
            MCPMiddleware(
                servers=[MCPServerConfig(name="   ", command="c", args=[])],
            )

    @patch("mambo_agents.middleware.mcp._collect_tool_index")
    def test_double_underscore_raises(self, mock_collect):
        mock_collect.return_value = []
        with pytest.raises(ValueError, match="reserved substring"):
            MCPMiddleware(
                servers=[MCPServerConfig(name="a__b", command="c", args=[])],
            )

    @patch("mambo_agents.middleware.mcp._collect_tool_index")
    def test_colon_raises(self, mock_collect):
        mock_collect.return_value = []
        with pytest.raises(ValueError, match="is invalid"):
            MCPMiddleware(
                servers=[MCPServerConfig(name="a:b", command="c", args=[])],
            )

    @patch("mambo_agents.middleware.mcp._collect_tool_index")
    def test_too_long_raises(self, mock_collect):
        mock_collect.return_value = []
        long_name = "a" * 65
        with pytest.raises(ValueError, match="exceeds"):
            MCPMiddleware(
                servers=[MCPServerConfig(name=long_name, command="c", args=[])],
            )

    @patch("mambo_agents.middleware.mcp._collect_tool_index")
    def test_leading_digit_raises(self, mock_collect):
        mock_collect.return_value = []
        with pytest.raises(ValueError, match="is invalid"):
            MCPMiddleware(
                servers=[MCPServerConfig(name="0abc", command="c", args=[])],
            )


# ============================================================================
# tool_unpacker
# ============================================================================


class TestToolUnpacker:
    @patch("mambo_agents.middleware.mcp._collect_tool_index")
    def test_non_mcp_tool_returns_none(self, mock_collect):
        mock_collect.return_value = []
        mw = MCPMiddleware(
            servers=[MCPServerConfig(name="s", command="c", args=[])],
        )
        unpacker = mw.tool_unpacker
        assert unpacker("write", {"file_path": "/f"}) is None
        assert unpacker("edit", {}) is None
        assert unpacker("mcp_get_tool_description", {}) is None

    @patch("mambo_agents.middleware.mcp._collect_tool_index")
    def test_mcp_call_tool_unpacks(self, mock_collect):
        mock_collect.return_value = [
            _ServerMeta(
                name="demo",
                available=True,
                tools=[
                    _ToolIndexEntry(
                        server_name="demo",
                        tool_name="echo",
                        description="Echo message",
                        input_schema={},
                    ),
                ],
            ),
        ]
        mw = MCPMiddleware(
            servers=[MCPServerConfig(name="demo", command="c", args=[])],
        )
        unpacker = mw.tool_unpacker
        result = unpacker("mcp_call_tool", {
            "server_name": "demo",
            "tool_name": "echo",
            "arguments": {"message": "hi"},
        })
        assert result is not None
        assert result.effective_tool_name == "demo__echo"
        assert result.effective_args == {"message": "hi"}
        assert result.tool_description == "Echo message"

    @patch("mambo_agents.middleware.mcp._collect_tool_index")
    def test_mcp_call_tool_unknown_tool_no_description(self, mock_collect):
        mock_collect.return_value = []
        mw = MCPMiddleware(
            servers=[MCPServerConfig(name="demo", command="c", args=[])],
        )
        unpacker = mw.tool_unpacker
        result = unpacker("mcp_call_tool", {
            "server_name": "demo",
            "tool_name": "nonexistent",
            "arguments": {"x": 1},
        })
        assert result is not None
        assert result.effective_tool_name == "demo__nonexistent"
        assert result.effective_args == {"x": 1}
        assert result.tool_description is None

    @patch("mambo_agents.middleware.mcp._collect_tool_index")
    def test_mcp_call_tool_missing_args_defaults(self, mock_collect):
        mock_collect.return_value = []
        mw = MCPMiddleware(
            servers=[MCPServerConfig(name="s", command="c", args=[])],
        )
        unpacker = mw.tool_unpacker
        result = unpacker("mcp_call_tool", {})
        assert result is not None
        assert result.effective_tool_name == "__"
        assert result.effective_args == {}
        assert result.tool_description is None


# ============================================================================
# exclude_tools
# ============================================================================


class TestExcludeTools:
    """Tests for ``exclude_tools`` parameter."""

    @patch("mambo_agents.middleware.mcp._collect_tool_index")
    def test_excluded_tools_omitted_from_index(self, mock_collect):
        """exclude_tools is passed to _collect_tool_index for server-side filtering."""
        mock_collect.return_value = [
            _ServerMeta(
                name="s",
                available=True,
                tools=[
                    _ToolIndexEntry(server_name="s", tool_name="keep", description="k", input_schema={}),
                ],
            ),
        ]
        mw = MCPMiddleware(
            servers=[MCPServerConfig(name="s", command="c", args=[])],
            exclude_tools={"s": frozenset(["drop"])},
        )
        assert ("s", "keep") in mw._tool_index
        assert ("s", "drop") not in mw._tool_index
        # Verify exclude_tools was forwarded to _collect_tool_index
        call_kwargs = mock_collect.call_args[1]
        assert call_kwargs["exclude_tools"] == {"s": frozenset(["drop"])}

    @patch("mambo_agents.middleware.mcp._collect_tool_index")
    def test_exclude_tools_calls_collect_with_filter(self, mock_collect):
        mock_collect.return_value = []
        MCPMiddleware(
            servers=[MCPServerConfig(name="s", command="c", args=[])],
            exclude_tools={"s": frozenset(["a", "b"])},
        )
        call_kwargs = mock_collect.call_args[1]
        assert "exclude_tools" in call_kwargs
        assert call_kwargs["exclude_tools"]["s"] == frozenset(["a", "b"])


# ============================================================================
# direct_tool_threshold
# ============================================================================


class TestDirectToolThreshold:
    """Tests for ``direct_tool_threshold`` parameter."""

    @patch("mambo_agents.middleware.mcp._collect_tool_index")
    def test_direct_mode_when_below_threshold(self, mock_collect):
        mock_collect.return_value = [
            _ServerMeta(
                name="s",
                available=True,
                tools=[
                    _ToolIndexEntry(server_name="s", tool_name="echo", description="E", input_schema={"type": "object"}),
                ],
            ),
        ]
        mw = MCPMiddleware(
            servers=[MCPServerConfig(name="s", command="c", args=[])],
            direct_tool_threshold=5,
        )
        assert mw._direct_mode is True
        tool_names = {t.name for t in mw.tools}
        assert "mcp_call_tool" not in tool_names
        assert "mcp_get_tool_description" not in tool_names
        assert "s__echo" in tool_names

    @patch("mambo_agents.middleware.mcp._collect_tool_index")
    def test_wrapped_mode_when_above_threshold(self, mock_collect):
        tools = [
            _ToolIndexEntry(server_name="s", tool_name=f"t{i}", description="d", input_schema={})
            for i in range(10)
        ]
        mock_collect.return_value = [
            _ServerMeta(name="s", available=True, tools=tools),
        ]
        mw = MCPMiddleware(
            servers=[MCPServerConfig(name="s", command="c", args=[])],
            direct_tool_threshold=5,
        )
        assert mw._direct_mode is False
        tool_names = {t.name for t in mw.tools}
        assert "mcp_call_tool" in tool_names
        assert "mcp_get_tool_description" in tool_names

    @patch("mambo_agents.middleware.mcp._collect_tool_index")
    def test_direct_mode_system_prompt_not_injected(self, mock_collect):
        from langchain.agents.middleware.types import ModelRequest

        mock_collect.return_value = [
            _ServerMeta(
                name="s",
                available=True,
                tools=[
                    _ToolIndexEntry(server_name="s", tool_name="t", description="d", input_schema={}),
                ],
            ),
        ]
        mw = MCPMiddleware(
            servers=[MCPServerConfig(name="s", command="c", args=[])],
            direct_tool_threshold=5,
        )
        req = ModelRequest(
            model=MagicMock(),
            messages=[],
            system_message=SystemMessage(content="base"),
        )

        def handler(r):
            return MagicMock()

        resp = mw.wrap_model_call(req, handler)
        assert "MCP Servers" not in str(req.system_message.content)

    @patch("mambo_agents.middleware.mcp._collect_tool_index")
    def test_tool_unpacker_works_in_direct_mode(self, mock_collect):
        mock_collect.return_value = [
            _ServerMeta(
                name="demo",
                available=True,
                tools=[
                    _ToolIndexEntry(server_name="demo", tool_name="echo", description="Echo", input_schema={}),
                ],
            ),
        ]
        mw = MCPMiddleware(
            servers=[MCPServerConfig(name="demo", command="c", args=[])],
            direct_tool_threshold=5,
        )
        unpacker = mw.tool_unpacker
        result = unpacker("demo__echo", {"message": "hi"})
        assert result is not None
        assert result.effective_tool_name == "demo__echo"
        assert result.effective_args == {"message": "hi"}
        assert result.tool_description == "Echo"
