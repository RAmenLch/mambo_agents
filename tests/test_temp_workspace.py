"""Tests for TempWorkspaceBackend — dual-backend routing & workspace isolation."""

import shutil
import tempfile
from pathlib import Path

import pytest
from langchain_core.tools import StructuredTool

from mambo_agents.backends.local import LocalBackend
from mambo_agents.backends.protocol import BackendProtocol
from mambo_agents.backends.state import StateBackend
from mambo_agents.backends.temp_workspace import TempWorkspaceBackend

from tests.test_state_backend import _simulate_graph


# ============================================================================
# Helpers
# ============================================================================


class _FakeBackend(BackendProtocol):
    """In-memory stub with a controllable tools list."""

    def __init__(self, extra_tools: list[StructuredTool] | None = None):
        super().__init__()
        self._files: dict[str, str] = {}
        self._extra_tools = extra_tools or []

    @property
    def tools(self) -> list[StructuredTool]:
        return self._extra_tools

    def ls(self, path: str):
        from mambo_agents.backends.protocol import FileInfo, LsResult

        prefix = path.rstrip("/") + "/" if path != "/" else "/"
        infos: list[FileInfo] = []
        dirs: set[str] = set()
        for fp, content in self._files.items():
            if not fp.startswith(prefix):
                continue
            rel = fp[len(prefix):]
            if "/" in rel:
                dirs.add(prefix + rel.split("/")[0] + "/")
            else:
                infos.append(FileInfo(path=fp, is_dir=False, size=len(content)))
        for d in sorted(dirs):
            infos.append(FileInfo(path=d, is_dir=True, size=0))
        infos.sort(key=lambda fi: fi.path)
        return LsResult(entries=infos)

    def read_raw(self, file_path: str, offset=0, limit=2000, include_line_numbers=False):
        from mambo_agents.backends.protocol import ReadResult

        c = self._files.get(file_path)
        if c is None:
            return ReadResult(error=f"File '{file_path}' not found")
        return ReadResult(content=c, total_lines=len(c.split("\n")))

    def write(self, file_path: str, content: str, overwrite=False):
        from mambo_agents.backends.protocol import WriteResult

        if file_path in self._files and not overwrite:
            return WriteResult(error=f"Cannot write '{file_path}': file already exists")
        self._files[file_path] = content
        return WriteResult(path=file_path)

    def edit(self, file_path: str, old_str: str, new_str: str, *, replace_all=False):
        from mambo_agents.backends.protocol import EditResult

        c = self._files.get(file_path)
        if c is None:
            return EditResult(error=f"Cannot edit '{file_path}': file not found")
        if old_str not in c:
            return EditResult(error=f"Cannot edit '{file_path}': old_str not found")
        self._files[file_path] = c.replace(old_str, new_str)
        return EditResult(path=file_path, occurrences=c.count(old_str))

    def grep(self, pattern: str, path="/", glob=None):
        from mambo_agents.backends.protocol import GrepResult, GrepMatch

        prefix = path.rstrip("/") + "/" if path != "/" else "/"
        matches: list[GrepMatch] = []
        for fp, c in sorted(self._files.items()):
            if prefix != "/" and not fp.startswith(prefix):
                continue
            for li, line in enumerate(c.split("\n"), 1):
                if pattern in line:
                    matches.append(GrepMatch(path=fp, line=li, text=line))
        return GrepResult(matches=matches)

    def glob(self, pattern: str, path="/"):
        import fnmatch as _fnmatch
        from mambo_agents.backends.protocol import FileInfo, GlobResult

        prefix = path.rstrip("/") if path != "/" else ""
        results: list[FileInfo] = []
        for fp in sorted(self._files):
            if prefix and fp != prefix and not fp.startswith(prefix + "/"):
                continue
            if _fnmatch.fnmatch(fp, pattern):
                results.append(FileInfo(path=fp, is_dir=False, size=len(self._files[fp])))
        return GlobResult(matches=results)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def tmp_root() -> Path:
    d = tempfile.mkdtemp(prefix="mambo_test_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def temp_ws(tmp_root: Path) -> TempWorkspaceBackend:
    """TempWorkspaceBackend with LocalBackend as delegate."""
    local = LocalBackend(root_dir=str(tmp_root))
    return TempWorkspaceBackend(backend=local)


# ============================================================================
# Path routing
# ============================================================================


class TestPathRouting:
    """Verify that workspace-prefixed paths hit StateBackend, others hit delegate."""

    def test_is_workspace_hits_prefix(self, temp_ws: TempWorkspaceBackend):
        assert temp_ws._is_workspace("/.mambo/")
        assert temp_ws._is_workspace("/.mambo/workspace/")
        assert temp_ws._is_workspace("/.mambo/large_tool_results/abc")
        assert temp_ws._is_workspace("/.mambo/conversation_history/t1.md")

    def test_is_workspace_misses_others(self, temp_ws: TempWorkspaceBackend):
        assert not temp_ws._is_workspace("/")
        assert not temp_ws._is_workspace("/project/")
        assert not temp_ws._is_workspace("/tmp/file.txt")
        assert not temp_ws._is_workspace("/conversation_history/x.md")

    def test_route_workspace_returns_state(self, temp_ws: TempWorkspaceBackend):
        target, path = temp_ws._route("/.mambo/workspace/f.txt")
        assert target is temp_ws._state
        assert path == "/.mambo/workspace/f.txt"

    def test_route_non_workspace_returns_backend(self, temp_ws: TempWorkspaceBackend):
        target, path = temp_ws._route("/project/main.py")
        assert target is temp_ws._backend
        assert path == "/project/main.py"


# ============================================================================
# Core operations — routing correctness
# ============================================================================


class TestCoreOpsRouting:
    """Write to both sides and verify they are isolated."""

    WORKSPACE_FILE = "/.mambo/workspace/scratch.txt"
    PROJECT_FILE = "/project/hello.txt"

    def _write_workspace(self, temp_ws: TempWorkspaceBackend, content: str):
        """Write a file under the workspace prefix via graph simulation."""
        with _simulate_graph(temp_ws._state):
            r = temp_ws.write(self.WORKSPACE_FILE, content, overwrite=True)
        assert r.error is None, r.error

    def _read_workspace(self, temp_ws: TempWorkspaceBackend) -> str:
        with _simulate_graph(temp_ws._state):
            r = temp_ws.read(self.WORKSPACE_FILE)
        assert r.error is None, r.error
        return r.content or ""

    def test_write_read_workspace(self, temp_ws: TempWorkspaceBackend):
        self._write_workspace(temp_ws, "virtual scratch")
        assert self._read_workspace(temp_ws) == "virtual scratch"

    def test_write_read_project(self, temp_ws: TempWorkspaceBackend):
        r = temp_ws.write(self.PROJECT_FILE, "real project")
        assert r.error is None, r.error
        r2 = temp_ws.read(self.PROJECT_FILE)
        assert r2.error is None
        assert "real project" in (r2.content or "")

    def test_workspace_file_not_in_backend(self, temp_ws: TempWorkspaceBackend):
        """Workspace files are invisible to the delegate backend."""
        self._write_workspace(temp_ws, "ghost")
        # Read via delegate backend directly — should not exist
        r = temp_ws._backend.read(self.WORKSPACE_FILE)
        assert r.error is not None

    def test_project_file_not_in_state(self, temp_ws: TempWorkspaceBackend):
        """Project files are invisible to the internal StateBackend."""
        temp_ws.write(self.PROJECT_FILE, "only real")
        with _simulate_graph(temp_ws._state):
            r = temp_ws._state.read(self.PROJECT_FILE)
        assert r.error is not None

    def test_edit_workspace(self, temp_ws: TempWorkspaceBackend):
        self._write_workspace(temp_ws, "line1\nline2\nline3")
        with _simulate_graph(temp_ws._state):
            r = temp_ws.edit(self.WORKSPACE_FILE, "line2", "LINE2")
        assert r.error is None, r.error
        assert "LINE2" in self._read_workspace(temp_ws)

    def test_edit_project(self, temp_ws: TempWorkspaceBackend):
        temp_ws.write(self.PROJECT_FILE, "old content here")
        r = temp_ws.edit(self.PROJECT_FILE, "old", "NEW")
        assert r.error is None, r.error
        r2 = temp_ws.read(self.PROJECT_FILE)
        assert "NEW content here" in (r2.content or "")

    def test_ls_workspace(self, temp_ws: TempWorkspaceBackend):
        self._write_workspace(temp_ws, "a")
        with _simulate_graph(temp_ws._state):
            temp_ws.write("/.mambo/workspace/other.txt", "b", overwrite=True)
            r = temp_ws.ls("/.mambo/workspace/")
        assert r.entries is not None
        paths = [fi.path for fi in r.entries]
        assert self.WORKSPACE_FILE in paths
        assert "/.mambo/workspace/other.txt" in paths

    def test_ls_project(self, temp_ws: TempWorkspaceBackend):
        temp_ws.write("/project/a.py", "1")
        temp_ws.write("/project/b.py", "2")
        r = temp_ws.ls("/project/")
        assert r.entries is not None
        paths = [fi.path for fi in r.entries]
        assert "/project/a.py" in paths
        assert "/project/b.py" in paths

    def test_grep_workspace(self, temp_ws: TempWorkspaceBackend):
        self._write_workspace(temp_ws, "needle in a haystack")
        with _simulate_graph(temp_ws._state):
            r = temp_ws.grep("needle", path="/.mambo/")
        assert r.matches is not None
        assert any("needle" in m.text for m in r.matches)

    def test_grep_project(self, temp_ws: TempWorkspaceBackend):
        temp_ws.write("/project/f.py", "def needle(): pass")
        r = temp_ws.grep("needle", path="/project/")
        assert r.matches is not None
        assert any("needle" in m.text for m in r.matches)

    def test_glob_workspace(self, temp_ws: TempWorkspaceBackend):
        self._write_workspace(temp_ws, "a")
        with _simulate_graph(temp_ws._state):
            temp_ws.write("/.mambo/workspace/b.md", "b", overwrite=True)
            r = temp_ws.glob("*.txt", path="/.mambo/workspace/")
        assert r.matches is not None
        paths = [fi.path for fi in r.matches]
        assert self.WORKSPACE_FILE in paths
        assert "/.mambo/workspace/b.md" not in paths  # .md not matched

    def test_glob_project(self, temp_ws: TempWorkspaceBackend):
        temp_ws.write("/project/x.py", "1")
        temp_ws.write("/project/y.txt", "2")
        r = temp_ws.glob("*.py", path="/project/")
        assert r.matches is not None
        paths = [fi.path for fi in r.matches]
        assert len(paths) == 1
        assert "/project/x.py" in paths

    def test_read_not_found_workspace(self, temp_ws: TempWorkspaceBackend):
        with _simulate_graph(temp_ws._state):
            r = temp_ws.read("/.mambo/workspace/nope.txt")
        assert r.error is not None

    def test_read_not_found_project(self, temp_ws: TempWorkspaceBackend):
        r = temp_ws.read("/project/nope.txt")
        assert r.error is not None


# ============================================================================
# tools delegation
# ============================================================================


def _make_dummy_tree_tool() -> StructuredTool:
    from pydantic import Field, create_model

    return StructuredTool(
        name="tree",
        description="Show tree",
        args_schema=create_model(
            "DummyTreeSchema",
            path=(str, Field(default="/")),
            depth=(int, Field(default=3)),
        ),
        func=lambda **kw: "tree output",
    )


_DUMMY_DELETE = StructuredTool(
    name="delete",
    description="Delete files",
    args_schema=None,
    func=lambda **kw: "deleted",
)


class TestToolsDelegation:
    """tools property delegates to _backend directly."""

    def test_returns_backend_tools(self):
        tree_tool = _make_dummy_tree_tool()
        fake = _FakeBackend(extra_tools=[tree_tool, _DUMMY_DELETE])
        tws = TempWorkspaceBackend(backend=fake)
        tools = tws.tools
        tool_names = {t.name for t in tools}
        assert "tree" in tool_names
        assert "delete" in tool_names

    def test_backend_with_no_extra_tools(self):
        fake = _FakeBackend()
        tws = TempWorkspaceBackend(backend=fake)
        assert tws.tools == []


# ============================================================================
# description
# ============================================================================


class TestDescription:
    def test_includes_prefix(self):
        tws = TempWorkspaceBackend(backend=_FakeBackend(), workspace_prefix="/.mambo/")
        desc = tws.description
        assert "/.mambo/" in desc
        assert "virtual" in desc.lower()

    def test_custom_description_prepended(self):
        tws = TempWorkspaceBackend(
            backend=_FakeBackend(),
            custom_description="!!! CUSTOM !!!",
        )
        assert tws.description.startswith("!!! CUSTOM !!!")


# ============================================================================
# upload / download — split by prefix
# ============================================================================


class TestUploadDownload:
    """upload_files / download_files split files by workspace prefix."""

    def test_upload_mixed_routes_correctly(self, tmp_root: Path):
        tws = TempWorkspaceBackend(backend=LocalBackend(root_dir=str(tmp_root)))

        files: list[tuple[str, bytes]] = [
            ("/.mambo/workspace/a.txt", b"workspace a"),
            ("/project/b.txt", b"project b"),
            ("/.mambo/workspace/c.txt", b"workspace c"),
            ("/project/d.txt", b"project d"),
        ]

        with _simulate_graph(tws._state):
            results = tws.upload_files(files)

        assert len(results) == 4
        assert all(r.error is None for r in results)

        # Verify workspace files in StateBackend
        with _simulate_graph(tws._state):
            r1 = tws.read("/.mambo/workspace/a.txt")
            r3 = tws.read("/.mambo/workspace/c.txt")
        assert "workspace a" in (r1.content or "")
        assert "workspace c" in (r3.content or "")

        # Verify project files in delegate
        r2 = tws.read("/project/b.txt")
        r4 = tws.read("/project/d.txt")
        assert "project b" in (r2.content or "")
        assert "project d" in (r4.content or "")

    def test_download_mixed_routes_correctly(self, tmp_root: Path):
        tws = TempWorkspaceBackend(backend=LocalBackend(root_dir=str(tmp_root)))

        # Pre-populate both sides
        with _simulate_graph(tws._state):
            tws.write("/.mambo/workspace/x.txt", "state file", overwrite=True)
        tws.write("/project/y.txt", "backend file", overwrite=True)

        with _simulate_graph(tws._state):
            results = tws.download_files([
                "/.mambo/workspace/x.txt",
                "/project/y.txt",
            ])

        assert len(results) == 2
        assert results[0].content == b"state file"
        assert results[1].content == b"backend file"
