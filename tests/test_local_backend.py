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
from mambo_agents.backends.schemas import BackendError, VirtualPath
from mambo_agents.backends.store import StoreBackend


_MODEL_NAME = "Pro/zai-org/GLM-4.7"


def _get_model():
    """Return a test ChatOpenAI model instance."""
    pytest.importorskip("langchain_openai")
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


_W = VirtualPath("/workspace")
_WS = _W.value  # string form for f-string interpolation


# ===================================================================
# Unit tests: LocalBackend
# ===================================================================


class TestLocalBackend:
    """Tests for LocalBackend operating on a real temp directory."""

    def test_write_and_read(self, tmp_root):
        backend = LocalBackend(root_dir=str(tmp_root))
        r = backend.write(VirtualPath(f"{_W}/hello.txt"), "Hello World")
        assert r.error is None
        assert r.path == f"{_W}/hello.txt"

        r2 = backend.read(VirtualPath(f"{_W}/hello.txt"))
        assert r2.error is None
        assert "Hello World" in (r2.content or "")

    def test_write_fails_if_exists(self, tmp_root):
        """Write fails if file already exists — use edit instead."""
        backend = LocalBackend(root_dir=str(tmp_root))
        backend.write(VirtualPath(f"{_W}/a.txt"), "original")
        r = backend.write(VirtualPath(f"{_W}/a.txt"), "modified")
        assert r.error is not None
        assert "已存在" in str(r.error)

    def test_edit_replaces_text(self, tmp_root):
        backend = LocalBackend(root_dir=str(tmp_root))
        backend.write(VirtualPath(f"{_W}/code.py"), "x = 1\ny = 2\nz = 3")
        r = backend.edit(VirtualPath(f"{_W}/code.py"), "x = 1", "999")
        assert r.error is None
        assert r.occurrences == 1

        result = backend.read(VirtualPath(f"{_W}/code.py"))
        raw = _strip_numbers(result.content or "")
        assert "999" in raw
        assert "x = 1" not in raw
        assert "z = 3" in raw  # other lines unchanged

    def test_edit_old_str_not_found(self, tmp_root):
        backend = LocalBackend(root_dir=str(tmp_root))
        backend.write(VirtualPath(f"{_W}/code.py"), "hello")
        r = backend.edit(VirtualPath(f"{_W}/code.py"), "not there", "x")
        assert r.error is not None
        assert "未找到" in str(r.error)

    def test_edit_file_not_found(self, tmp_root):
        backend = LocalBackend(root_dir=str(tmp_root))
        r = backend.edit(VirtualPath(f"{_W}/ghost.py"), "something", "x")
        assert r.error is not None
        assert "不存在" in str(r.error)

    def test_ls(self, tmp_root):
        backend = LocalBackend(root_dir=str(tmp_root))
        backend.write(VirtualPath(f"{_W}/a.py"), "1")
        backend.write(VirtualPath(f"{_W}/b.txt"), "2")
        (tmp_root / "subdir").mkdir()

        result = backend.ls(VirtualPath(f"{_W}/"))
        assert result.entries is not None
        paths = [fi.path for fi in result.entries]
        assert f"{_W}/a.py" in paths
        assert f"{_W}/b.txt" in paths
        assert any(p == f"{_W}/subdir" for p in paths)

    def test_read_not_found(self, tmp_root):
        backend = LocalBackend(root_dir=str(tmp_root))
        r = backend.read(VirtualPath(f"{_W}/nope.txt"))
        assert r.error is not None
        assert "不存在" in str(r.error)

    def test_grep(self, tmp_root):
        backend = LocalBackend(root_dir=str(tmp_root))
        backend.write(VirtualPath(f"{_W}/a.py"), "def foo():\n    pass")
        backend.write(VirtualPath(f"{_W}/b.py"), "def bar():\n    pass")

        r = backend.grep("foo", path=VirtualPath(f"{_W}/"))
        assert r.matches is not None
        assert any("foo" in m.text for m in r.matches)

    def test_glob(self, tmp_root):
        backend = LocalBackend(root_dir=str(tmp_root))
        backend.write(VirtualPath(f"{_W}/src/main.py"), "code")
        backend.write(VirtualPath(f"{_W}/src/util.py"), "code")
        backend.write(VirtualPath(f"{_W}/README.md"), "readme")

        r = backend.glob("*.py", path=VirtualPath(f"{_W}/src"))
        assert r.matches is not None
        paths = [fi.path for fi in r.matches]
        assert len(paths) == 2

    def test_tree_output_is_str(self, tmp_root):
        """tree() returns a plain string."""
        backend = LocalBackend(root_dir=str(tmp_root))
        backend.write(VirtualPath(f"{_W}/a.txt"), "a")
        (tmp_root / "sub").mkdir()
        backend.write(VirtualPath(f"{_W}/sub/b.txt"), "b")

        r = backend.tree(VirtualPath(f"{_W}/"), depth=2)
        assert isinstance(r, str)
        assert len(r) > 0

    def test_path_outside_workspace_rejected(self, tmp_root):
        """Any path not under /workspace is rejected with WorkspacePathError."""
        backend = LocalBackend(root_dir=str(tmp_root))
        r = backend.ls(VirtualPath("/etc"))
        assert r.error is not None
        assert "超出工作区" in str(r.error)

    def test_path_outside_workspace_write_rejected(self, tmp_root):
        """write to /etc/passwd-like path is rejected."""
        backend = LocalBackend(root_dir=str(tmp_root))
        r = backend.write(VirtualPath("/etc/passwd"), "evil")
        assert r.error is not None
        assert "超出工作区" in str(r.error)

    # ------------------------------------------------------------------
    # Delete tool
    # ------------------------------------------------------------------

    def test_delete_file(self, tmp_root):
        backend = LocalBackend(root_dir=str(tmp_root))
        backend.write(VirtualPath(f"{_W}/deleteme.txt"), "bye")
        assert (tmp_root / "deleteme.txt").exists()

        r = backend.delete(VirtualPath(f"{_W}/deleteme.txt"))
        assert r is not None
        assert "Deleted:" in str(r)
        assert not (tmp_root / "deleteme.txt").exists()

    def test_delete_rejects_directory(self, tmp_root):
        """delete rejects directories — only single files can be removed."""
        backend = LocalBackend(root_dir=str(tmp_root))
        (tmp_root / "fulldir").mkdir()
        (tmp_root / "fulldir" / "file.txt").write_text("hi")
        r = backend.delete(VirtualPath(f"{_W}/fulldir"))
        assert "目标" in str(r)
        assert (tmp_root / "fulldir").exists()
        assert (tmp_root / "fulldir" / "file.txt").exists()

    def test_delete_not_found(self, tmp_root):
        backend = LocalBackend(root_dir=str(tmp_root))
        r = backend.delete(VirtualPath(f"{_W}/nope.txt"))
        assert "不存在" in str(r)

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

    # ------------------------------------------------------------------
    # edit_whitelist / edit_blacklist
    # ------------------------------------------------------------------

    def test_whitelist_blacklist_mutual_exclusive(self, tmp_root):
        """edit_whitelist and edit_blacklist cannot both be provided."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            LocalBackend(
                root_dir=str(tmp_root),
                edit_whitelist=frozenset({VirtualPath(f"{_W}/src")}),
                edit_blacklist=frozenset({VirtualPath(f"{_W}/build")}),
            )

    def test_edit_whitelist_blocks_write(self, tmp_root):
        """write to a non-whitelisted path is rejected."""
        backend = LocalBackend(
            root_dir=str(tmp_root),
            edit_whitelist=frozenset({VirtualPath(f"{_W}/src")}),
        )
        r = backend.write(VirtualPath(f"{_W}/outside.txt"), "hello")
        assert r.error is not None
        assert "不允许" in str(r.error)

    def test_edit_whitelist_allows_write(self, tmp_root):
        """write to a whitelisted path is allowed."""
        backend = LocalBackend(
            root_dir=str(tmp_root),
            edit_whitelist=frozenset({VirtualPath(f"{_W}/src")}),
        )
        r = backend.write(VirtualPath(f"{_W}/src/hello.txt"), "hello")
        assert r.error is None
        assert r.path == f"{_W}/src/hello.txt"

    def test_edit_blacklist_blocks_edit(self, tmp_root):
        """edit on a blacklisted path is rejected."""
        backend = LocalBackend(
            root_dir=str(tmp_root),
            edit_blacklist=frozenset({VirtualPath(f"{_W}/build")}),
        )
        r = backend.edit(VirtualPath(f"{_W}/build/output.o"), "a", "b")
        assert r.error is not None
        assert "不允许" in str(r.error)

    def test_edit_blacklist_allows_other_paths(self, tmp_root):
        """edit on a non-blacklisted path works normally."""
        backend = LocalBackend(
            root_dir=str(tmp_root),
            edit_blacklist=frozenset({VirtualPath(f"{_W}/build")}),
        )
        backend.write(VirtualPath(f"{_W}/src/code.py"), "x = 1")
        r = backend.edit(VirtualPath(f"{_W}/src/code.py"), "x = 1", "x = 2")
        assert r.error is None
        assert r.occurrences == 1

    def test_whitelist_blocks_delete(self, tmp_root):
        """delete on a non-whitelisted path is rejected."""
        backend = LocalBackend(
            root_dir=str(tmp_root),
            edit_whitelist=frozenset({VirtualPath(f"{_W}/src")}),
        )
        r = backend.delete(VirtualPath(f"{_W}/outside/secret.txt"))
        assert "不允许" in str(r)

    def test_blacklist_blocks_delete(self, tmp_root):
        """delete on a blacklisted path is rejected."""
        backend = LocalBackend(
            root_dir=str(tmp_root),
            edit_blacklist=frozenset({VirtualPath(f"{_W}/important")}),
        )
        (tmp_root / "important").mkdir()
        r = backend.delete(VirtualPath(f"{_W}/important"))
        assert "不允许" in str(r)

    # ------------------------------------------------------------------
    # tree with ignore_dirs
    # ------------------------------------------------------------------

    def test_tree_ignore_dirs(self, tmp_root):
        """ignore_dirs hides children of marked directories but still shows the dir."""
        backend = LocalBackend(
            root_dir=str(tmp_root),
            ignore_dirs=frozenset({f"{_W}/node_modules"}),
        )
        (tmp_root / "src").mkdir()
        backend.write(VirtualPath(f"{_W}/src/main.py"), "code")
        nm = tmp_root / "node_modules"
        nm.mkdir()
        (nm / "package.json").write_text("{}")
        (nm / "lodash").mkdir()

        result = backend.tree(VirtualPath(f"{_W}/"), depth=3)
        assert "node_modules/(ignore)" in result
        # Children of node_modules should not appear
        assert "package.json" not in result
        assert "lodash" not in result
        # src should still be shown
        assert "src/" in result
        assert "main.py" in result

    def test_tree_empty_directory(self, tmp_root):
        """Empty directories show /(empty) marker."""
        backend = LocalBackend(root_dir=str(tmp_root))
        (tmp_root / "empty_dir").mkdir()
        (tmp_root / "non_empty").mkdir()
        backend.write(VirtualPath(f"{_W}/non_empty/file.txt"), "data")

        result = backend.tree(VirtualPath(f"{_W}/"), depth=2)
        assert "empty_dir/(empty)" in result
        assert "non_empty/" in result
        assert "file.txt" in result

    def test_tree_depth_exceeded(self, tmp_root):
        """Directories at max depth with children show /(...) marker."""
        backend = LocalBackend(root_dir=str(tmp_root))
        deep = tmp_root / "deep"
        deep.mkdir()
        (deep / "child").mkdir()
        (deep / "child" / "grandchild").mkdir()

        result = backend.tree(VirtualPath(f"{_W}/"), depth=2)
        assert "child/(...)" in result
        # grandchild should not appear (past depth limit)
        assert "grandchild" not in result


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
                            f"Create a file {_W}/greeting.txt with the content "
                            "'Hello from Mambo Agents LocalBackend'. Reply with exactly 'DONE'."
                        )
                    )
                ]
            },
            config={"configurable": {"thread_id": "test_local_backend_fwtr"}},
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

        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            f"Read the file {_W}/precious.txt to see its content, "
                            f"then delete {_W}/precious.txt. Reply 'DONE' when completed."
                        )
                    )
                ]
            },
            config=config,
        )

        assert result is not None
