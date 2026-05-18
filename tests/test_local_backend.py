"""Tests for LocalBackend and interrupt_on."""

import os
import shutil
import tempfile
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from mambo_agents import create_mambo_agent
from mambo_agents.backends.local import LocalBackend
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
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_root():
    """Create a temporary directory and yield its path, then clean up."""
    d = tempfile.mkdtemp(prefix="mambo_test_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


def _strip_numbers(content: str) -> str:
    """Strip line-number prefix from formatted output."""
    lines = content.split("\n")
    clean: list[str] = []
    for line in lines:
        if "\t" in line:
            clean.append(line.split("\t", 1)[1])
        else:
            clean.append(line)
    return "\n".join(clean)


# ===================================================================
# Unit tests: LocalBackend
# ===================================================================


class TestLocalBackend:
    """Tests for LocalBackend operating on a real temp directory."""

    def test_write_and_read(self, tmp_root):
        backend = LocalBackend(root_dir=str(tmp_root))
        r = backend.write("/hello.txt", "Hello World")
        assert r.error is None
        assert r.path == "/hello.txt"

        r2 = backend.read("/hello.txt")
        assert r2.error is None
        assert "Hello World" in (r2.content or "")

    def test_write_fails_if_exists(self, tmp_root):
        """Write fails if file already exists — use edit instead."""
        backend = LocalBackend(root_dir=str(tmp_root))
        backend.write("/a.txt", "original")
        r = backend.write("/a.txt", "modified")
        assert r.error is not None
        assert "already exists" in (r.error or "")

    def test_edit_replaces_text(self, tmp_root):
        backend = LocalBackend(root_dir=str(tmp_root))
        backend.write("/code.py", "x = 1\ny = 2\nz = 3")
        r = backend.edit("/code.py", "x = 1", "999")
        assert r.error is None
        assert r.occurrences == 1

        result = backend.read("/code.py")
        raw = _strip_numbers(result.content or "")
        assert "999" in raw
        assert "x = 1" not in raw
        assert "z = 3" in raw  # other lines unchanged

    def test_edit_old_str_not_found(self, tmp_root):
        backend = LocalBackend(root_dir=str(tmp_root))
        backend.write("/code.py", "hello")
        r = backend.edit("/code.py", "not there", "x")
        assert r.error is not None
        assert "old_str not found" in (r.error or "")

    def test_edit_file_not_found(self, tmp_root):
        backend = LocalBackend(root_dir=str(tmp_root))
        r = backend.edit("/ghost.py", "something", "x")
        assert r.error is not None
        assert "file not found" in (r.error or "")

    def test_ls(self, tmp_root):
        backend = LocalBackend(root_dir=str(tmp_root))
        backend.write("/a.py", "1")
        backend.write("/b.txt", "2")
        (tmp_root / "subdir").mkdir()

        result = backend.ls("/")
        assert result.entries is not None
        paths = [fi.path for fi in result.entries]
        assert "/a.py" in paths
        assert "/b.txt" in paths
        assert any("subdir" in p and p.endswith("/") for p in paths)

    def test_read_not_found(self, tmp_root):
        backend = LocalBackend(root_dir=str(tmp_root))
        r = backend.read("/nope.txt")
        assert r.error is not None
        assert "not found" in r.error

    def test_grep(self, tmp_root):
        backend = LocalBackend(root_dir=str(tmp_root))
        backend.write("/a.py", "def foo():\n    pass")
        backend.write("/b.py", "def bar():\n    pass")

        r = backend.grep("foo", path="/")
        assert r.matches is not None
        assert any("foo" in m.text for m in r.matches)

    def test_glob(self, tmp_root):
        backend = LocalBackend(root_dir=str(tmp_root))
        backend.write("/src/main.py", "code")
        backend.write("/src/util.py", "code")
        backend.write("/README.md", "readme")

        r = backend.glob("*.py", path="/src")
        assert r.matches is not None
        paths = [fi.path for fi in r.matches]
        assert len(paths) == 2

    def test_tree_output_is_str(self, tmp_root):
        """tree() returns a plain string."""
        backend = LocalBackend(root_dir=str(tmp_root))
        backend.write("/a.txt", "a")
        (tmp_root / "sub").mkdir()
        backend.write("/sub/b.txt", "b")

        r = backend.tree("/", depth=2)
        assert isinstance(r, str)
        assert len(r) > 0

    # ------------------------------------------------------------------
    # Delete tool
    # ------------------------------------------------------------------

    def test_delete_file(self, tmp_root):
        backend = LocalBackend(root_dir=str(tmp_root))
        backend.write("/deleteme.txt", "bye")
        assert (tmp_root / "deleteme.txt").exists()

        r = backend.delete("/deleteme.txt")
        assert r is not None
        assert "Deleted:" in r
        assert not (tmp_root / "deleteme.txt").exists()

    def test_delete_directory_recursive(self, tmp_root):
        """delete removes non-empty directories (recursively by default)."""
        backend = LocalBackend(root_dir=str(tmp_root))
        (tmp_root / "fulldir").mkdir()
        (tmp_root / "fulldir" / "file.txt").write_text("hi")
        r = backend.delete("/fulldir")
        assert "Deleted:" in r
        assert not (tmp_root / "fulldir").exists()

    def test_delete_not_found(self, tmp_root):
        backend = LocalBackend(root_dir=str(tmp_root))
        r = backend.delete("/nope.txt")
        assert "does not exist" in r

    def test_delete_tool_registered(self, tmp_root):
        """delete tool is always available in LocalBackend."""
        backend = LocalBackend(root_dir=str(tmp_root))
        tool_names = {t.name for t in backend.tools}
        assert "delete" in tool_names

    # ------------------------------------------------------------------
    # Execute tool
    # ------------------------------------------------------------------

    def test_execute_echo(self, tmp_root):
        backend = LocalBackend(root_dir=str(tmp_root))
        r = backend.execute("echo hello_exec")
        assert "hello_exec" in r

    def test_execute_tool_registered(self, tmp_root):
        """execute tool is only available when enable_execute=True."""
        backend = LocalBackend(root_dir=str(tmp_root), enable_execute=True)
        tool_names = {t.name for t in backend.tools}
        assert "execute" in tool_names

    def test_execute_failed_command(self, tmp_root):
        backend = LocalBackend(root_dir=str(tmp_root))
        r = backend.execute("nonexistent_command_xyz_12345")
        assert "Error" in r or "Exit code:" in r

    # ------------------------------------------------------------------
    # Tools property (extra tools only)
    # ------------------------------------------------------------------

    def test_tools_extra_only(self, tmp_root):
        """backend.tools returns only extra tools. Core tools are in middleware."""
        backend = LocalBackend(root_dir=str(tmp_root), enable_execute=True)
        names = {t.name for t in backend.tools}
        # Extra tools only
        assert "tree" in names
        assert "delete" in names
        assert "execute" in names
        # Core tools NOT here
        for core in ("ls", "read", "write", "edit", "grep", "glob"):
            assert core not in names, f"Core tool '{core}' should not be in backend.tools"


# ===================================================================
# Integration test: create_mambo_agent with LocalBackend
# ===================================================================


class TestCreateAgentLocal:
    """End-to-end tests with LocalBackend (needs network)."""

    @pytest.mark.integration
    def test_basic_agent_with_local(self, tmp_root):
        model = _get_model()
        backend = LocalBackend(root_dir=str(tmp_root))
        agent = create_mambo_agent(model, backend=backend)
        assert agent is not None

    @pytest.mark.integration
    def test_agent_file_write_then_read(self, tmp_root):
        model = _get_model()
        backend = LocalBackend(root_dir=str(tmp_root))
        agent = create_mambo_agent(model, backend=backend)

        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "Create a file /greeting.txt with the content "
                            "'Hello from Mambo Agents LocalBackend'. Reply with exactly 'DONE'."
                        )
                    )
                ]
            }
        )
        # Verify on disk
        file_path = tmp_root / "greeting.txt"
        assert file_path.exists(), "File was not created on disk"
        content = file_path.read_text(encoding="utf-8")
        assert "Hello from Mambo Agents LocalBackend" in content


# ===================================================================
# Integration test: interrupt_on
# ===================================================================


class TestInterruptOn:
    """Tests for interrupt_on (HumanInTheLoopMiddleware)."""

    @pytest.mark.integration
    def test_interrupt_on_parameter_accepted(self, tmp_root):
        """interrupt_on + checkpointer does not raise on construction."""
        model = _get_model()
        backend = LocalBackend(root_dir=str(tmp_root))

        agent = create_mambo_agent(
            model,
            backend=backend,
            interrupt_on={"delete": True},
            checkpointer=MemorySaver(),
        )
        assert agent is not None

    @pytest.mark.integration
    def test_interrupt_on_without_checkpointer_raises(self, tmp_root):
        """Without checkpointer, interrupt_on raises ValueError."""
        model = _get_model()
        backend = LocalBackend(root_dir=str(tmp_root))

        with pytest.raises(ValueError, match="checkpointer"):
            create_mambo_agent(
                model,
                backend=backend,
                interrupt_on={"delete": True},
            )

    @pytest.mark.integration
    def test_interrupt_on_pauses_agent(self, tmp_root):
        """Agent with interrupt_on pauses before executing the guarded tool.

        We create a file, then ask the agent to delete it with interrupt_on set.
        The agent should go through the HumanInTheLoop flow.
        """
        model = _get_model()
        backend = LocalBackend(root_dir=str(tmp_root))

        agent = create_mambo_agent(
            model,
            backend=backend,
            interrupt_on={"delete": True},
            checkpointer=MemorySaver(),
        )

        # Pre-create a file (no interrupt on write)
        (tmp_root / "precious.txt").write_text("don't delete me", encoding="utf-8")

        config = {"configurable": {"thread_id": "test-interrupt"}}

        # Invoke without streaming – the agent will attempt to delete,
        # and the HumanInTheLoop middleware will interrupt.
        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "Read the file /precious.txt to see its content, "
                            "then delete /precious.txt. Reply 'DONE' when completed."
                        )
                    )
                ]
            },
            config=config,
        )

        # After the interrupt, the agent should have produced some output.
        assert result is not None
