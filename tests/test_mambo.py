"""Tests for Mambo Agents – StateBackend and create_mambo_agent."""

import os

import pytest
from langchain_core.messages import HumanMessage

from mambo_agents import create_mambo_agent
from mambo_agents.backends.protocol import BackendProtocol
from mambo_agents.backends.state import StateBackend


_MODEL_NAME = "Pro/zai-org/GLM-4.7"


def _get_model():
    """Return a test ChatOpenAI model instance."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=_MODEL_NAME,
        api_key=os.environ.get("GJKEY", ""),
        base_url="https://api.siliconflow.cn/v1",
        temperature=0,
    )


# ---------------------------------------------------------------------------
# Unit tests – StateBackend
# ---------------------------------------------------------------------------


class TestStateBackend:
    def test_write_and_read(self):
        backend = StateBackend()
        r = backend.write("/hello.txt", "Hello World")
        assert r.error is None
        assert r.path == "/hello.txt"

        r2 = backend.read("/hello.txt")
        assert r2.error is None
        assert "Hello World" in (r2.content or "")

    def test_write_fails_if_exists(self):
        """Write fails if file already exists — use edit instead."""
        backend = StateBackend()
        backend.write("/a.txt", "original")
        r = backend.write("/a.txt", "modified")
        assert r.error is not None
        assert "already exists" in (r.error or "")

    def test_edit_replaces_text(self):
        """Edit replaces the matched old_str with new_str."""
        backend = StateBackend()
        backend.write("/code.py", "x = 1\ny = 2")
        r = backend.edit("/code.py", "x = 1", "100")
        assert r.error is None
        assert r.occurrences == 1
        result = backend.read("/code.py")
        assert "100" in (result.content or "")

    def test_edit_old_str_not_found(self):
        """Edit fails if old_str not in file."""
        backend = StateBackend()
        backend.write("/code.py", "hello world")
        r = backend.edit("/code.py", "not there", "replacement")
        assert r.error is not None
        assert "old_str not found" in (r.error or "")

    def test_edit_file_not_found(self):
        """Edit fails if file doesn't exist."""
        backend = StateBackend()
        r = backend.edit("/no_file.py", "something", "x")
        assert r.error is not None
        assert "file not found" in (r.error or "")

    def test_ls_shows_files(self):
        backend = StateBackend()
        backend.write("/a.py", "print(1)")
        backend.write("/b.py", "print(2)")

        result = backend.ls("/")
        assert result.entries is not None
        paths = [fi.path for fi in result.entries]
        assert "/a.py" in paths
        assert "/b.py" in paths

    def test_ls_shows_subdirs(self):
        backend = StateBackend()
        backend.write("/sub/file.txt", "hello")

        result = backend.ls("/")
        assert result.entries is not None
        paths = [fi.path for fi in result.entries]
        assert any(p.endswith("/") or "/sub/" in p for p in paths)

    def test_grep_finds_pattern(self):
        backend = StateBackend()
        backend.write("/a.py", "def foo():\n    pass")
        backend.write("/b.py", "def bar():\n    pass")

        result = backend.grep("foo", path="/")
        assert result.matches is not None
        assert any("foo" in m.text for m in result.matches)

    def test_glob_finds_files(self):
        backend = StateBackend()
        backend.write("/src/main.py", "code")
        backend.write("/src/util.py", "code")
        backend.write("/README.md", "readme")

        result = backend.glob("**/*.py", path="/src")
        assert result.matches is not None
        paths = [fi.path for fi in result.matches]
        assert len(paths) == 2
        assert "/src/main.py" in paths

    def test_tree_output_is_str(self):
        """tree() returns a plain string."""
        backend = StateBackend()
        backend.write("/a.txt", "a")
        backend.write("/sub/b.txt", "b")

        result = backend.tree("/", depth=2)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_read_not_found(self):
        backend = StateBackend()
        result = backend.read("/nonexistent.txt")
        assert result.error is not None
        assert "not found" in result.error


# ---------------------------------------------------------------------------
# Unit tests – tools property (extra tools only)
# ---------------------------------------------------------------------------


class TestTools:
    def test_state_backend_tools_extra_only(self):
        """backend.tools returns only extra tools, NOT core tools.
        
        Core tools (ls, read, write, edit, grep, glob) are built by
        BackendToolsMiddleware — they do NOT appear in backend.tools.
        """
        backend = StateBackend()
        tool_names = {t.name for t in backend.tools}
        # Only extra tool is tree
        assert "tree" in tool_names
        # Core tools should NOT be here
        assert "ls" not in tool_names
        assert "read" not in tool_names
        assert "write" not in tool_names
        assert "edit" not in tool_names
        assert "grep" not in tool_names
        assert "glob" not in tool_names

    def test_tools_are_structured(self):
        from langchain_core.tools import StructuredTool

        backend = StateBackend()
        for tool in backend.tools:
            assert isinstance(tool, StructuredTool), (
                f"{tool.name} should be StructuredTool"
            )

    def test_state_backend_size_is_from_content_length(self):
        """FileInfo.size reflects content length."""
        backend = StateBackend()
        backend.write("/test.txt", "hello")  # 5 chars
        result = backend.glob("/test.txt", path="/")
        assert result.matches is not None
        assert result.matches[0].size == 5


# ---------------------------------------------------------------------------
# Integration tests – create_mambo_agent with StateBackend
# ---------------------------------------------------------------------------


class TestCreateAgent:
    @pytest.mark.integration
    def test_basic_agent_creation(self):
        """Smoke test: creates an agent without errors."""
        model = _get_model()
        backend = StateBackend()
        agent = create_mambo_agent(model, backend=backend)
        assert agent is not None

    @pytest.mark.integration
    def test_agent_file_write_then_read(self):
        """End-to-end: agent creates a file and verifies it in backend."""
        model = _get_model()
        backend = StateBackend()
        agent = create_mambo_agent(model, backend=backend)

        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "Create a file /greeting.txt with the content "
                            "'Hello from Mambo Agents'. Reply with exactly 'DONE'."
                        )
                    )
                ]
            }
        )
        # Verify file was created in backend
        r = backend.read("/greeting.txt")
        assert "Hello from Mambo Agents" in (r.content or ""), (
            f"Expected greeting in file, got: {r.content}"
        )
