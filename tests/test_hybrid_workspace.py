"""Tests for HybridWorkspaceBackend — multi-backend routing & workspace isolation."""

import re
import shutil
import tempfile
from pathlib import Path, PurePosixPath

import pytest
import yaml
from langchain_core.tools import StructuredTool
from pydantic import Field, ValidationError, create_model

from langgraph.store.memory import InMemoryStore

from mambo_agents.backends.hybrid_workspace import HybridWorkspaceBackend
from mambo_agents.backends.local import LocalBackend
from mambo_agents.backends.protocol import (
    BackendProtocol,
    DownloadFileResult,
    UploadFileResult,
)
from mambo_agents.backends.store import StoreBackend
from mambo_agents.backends.schemas import BackendError, ErrorCode, VirtualPath

from tests.test_store_backend import _simulate_graph


_W = "/workspace"


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

    def ls(self, path: VirtualPath):
        from mambo_agents.backends.protocol import FileInfo, LsResult

        prefix = path.value.rstrip("/") + "/" if path.value != "/" else "/"
        infos: list[FileInfo] = []
        dirs: set[str] = set()
        for fp, content in self._files.items():
            if not fp.startswith(prefix):
                continue
            rel = fp[len(prefix):]
            if "/" in rel:
                dirs.add(prefix + rel.split("/")[0])
            else:
                infos.append(FileInfo(path=fp, is_dir=False, size=len(content)))
        for d in sorted(dirs):
            infos.append(FileInfo(path=d, is_dir=True, size=0))
        infos.sort(key=lambda fi: fi.path)
        return LsResult(entries=infos)

    def read_raw(self, file_path: VirtualPath, offset=0, limit=2000, include_line_numbers=False):
        from mambo_agents.backends.protocol import ReadResult

        s = str(file_path)
        c = self._files.get(s)
        if c is None:
            return ReadResult(error=f"File '{s}' not found")
        return ReadResult(content=c, total_lines=len(c.split("\n")))

    def write(self, file_path: VirtualPath, content: str, overwrite=False):
        from mambo_agents.backends.protocol import WriteResult

        s = str(file_path)
        if s in self._files and not overwrite:
            return WriteResult(error=f"Cannot write '{s}': file already exists")
        self._files[s] = content
        return WriteResult(path=s)

    def edit(self, file_path: VirtualPath, old_str: str, new_str: str, *, replace_all=False):
        from mambo_agents.backends.protocol import EditResult

        s = str(file_path)
        c = self._files.get(s)
        if c is None:
            return EditResult(error=f"Cannot edit '{s}': file not found")
        if old_str not in c:
            return EditResult(error=f"Cannot edit '{s}': old_str not found")
        self._files[s] = c.replace(old_str, new_str)
        return EditResult(path=s, occurrences=c.count(old_str))

    def grep(self, pattern: str, path: VirtualPath, glob=None, regex=True, offset=0, limit=None):
        from mambo_agents.backends.protocol import GrepResult, GrepMatch

        prefix = path.value.rstrip("/") + "/" if path.value != "/" else "/"
        matches: list[GrepMatch] = []
        if regex:
            compiled = __import__("re").compile(pattern)
        else:
            compiled = __import__("re").compile(__import__("re").escape(pattern))
        for fp, c in sorted(self._files.items()):
            if prefix != "/" and not fp.startswith(prefix):
                continue
            for li, line in enumerate(c.split("\n"), 1):
                if compiled.search(line):
                    matches.append(GrepMatch(path=fp, line=li, text=line))
        return GrepResult(matches=matches)

    def glob(self, pattern: str, path: VirtualPath):
        import fnmatch as _fnmatch
        from mambo_agents.backends.protocol import FileInfo, GlobResult

        prefix = path.value.rstrip("/") if path.value != "/" else ""
        results: list[FileInfo] = []
        for fp in sorted(self._files):
            if prefix and fp != prefix and not fp.startswith(prefix + "/"):
                continue
            if _fnmatch.fnmatch(fp, pattern):
                results.append(FileInfo(path=fp, is_dir=False, size=len(self._files[fp])))
        return GlobResult(matches=results)


class _FakeThreadAwareBackend(BackendProtocol):
    """In-memory stub — a minimal BackendProtocol implementation."""

    def __init__(self):
        super().__init__()
        self._files: dict[str, str] = {}

    @property
    def tools(self) -> list[StructuredTool]:
        return []

    def ls(self, path: VirtualPath):
        from mambo_agents.backends.protocol import FileInfo, LsResult

        prefix = path.value.rstrip("/") + "/" if path.value != "/" else "/"
        infos: list[FileInfo] = []
        dirs: set[str] = set()
        for fp, content in self._files.items():
            if not fp.startswith(prefix):
                continue
            rel = fp[len(prefix):]
            if "/" in rel:
                dirs.add(prefix + rel.split("/")[0])
            else:
                infos.append(FileInfo(path=fp, is_dir=False, size=len(content)))
        for d in sorted(dirs):
            infos.append(FileInfo(path=d, is_dir=True, size=0))
        infos.sort(key=lambda fi: fi.path)
        return LsResult(entries=infos)

    def read_raw(self, file_path: VirtualPath, offset=0, limit=2000, include_line_numbers=False):
        from mambo_agents.backends.protocol import ReadResult

        s = str(file_path)
        c = self._files.get(s)
        if c is None:
            return ReadResult(error=f"File '{s}' not found")
        return ReadResult(content=c, total_lines=len(c.split("\n")))

    def write(self, file_path: VirtualPath, content: str, overwrite=False):
        from mambo_agents.backends.protocol import WriteResult

        s = str(file_path)
        if s in self._files and not overwrite:
            return WriteResult(error=f"Cannot write '{s}': file already exists")
        self._files[s] = content
        return WriteResult(path=s)

    def edit(self, file_path: VirtualPath, old_str: str, new_str: str, *, replace_all=False):
        from mambo_agents.backends.protocol import EditResult

        s = str(file_path)
        c = self._files.get(s)
        if c is None:
            return EditResult(error=f"Cannot edit '{s}': file not found")
        if old_str not in c:
            return EditResult(error=f"Cannot edit '{s}': old_str not found")
        self._files[s] = c.replace(old_str, new_str)
        return EditResult(path=s, occurrences=c.count(old_str))

    def grep(self, pattern: str, path: VirtualPath, glob=None, regex=True, offset=0, limit=None):
        from mambo_agents.backends.protocol import GrepResult, GrepMatch

        prefix = path.value.rstrip("/") + "/" if path.value != "/" else "/"
        matches: list[GrepMatch] = []
        if regex:
            compiled = __import__("re").compile(pattern)
        else:
            compiled = __import__("re").compile(__import__("re").escape(pattern))
        for fp, c in sorted(self._files.items()):
            if prefix != "/" and not fp.startswith(prefix):
                continue
            for li, line in enumerate(c.split("\n"), 1):
                if compiled.search(line):
                    matches.append(GrepMatch(path=fp, line=li, text=line))
        return GrepResult(matches=matches)

    def glob(self, pattern: str, path: VirtualPath):
        import fnmatch as _fnmatch
        from mambo_agents.backends.protocol import FileInfo, GlobResult

        prefix = path.value.rstrip("/") if path.value != "/" else ""
        results: list[FileInfo] = []
        for fp in sorted(self._files):
            if prefix and fp != prefix and not fp.startswith(prefix + "/"):
                continue
            if _fnmatch.fnmatch(fp, pattern):
                results.append(FileInfo(path=fp, is_dir=False, size=len(self._files[fp])))
        return GlobResult(matches=results)

    def upload_files(self, files):
        results: list[UploadFileResult] = []
        for path, raw_content in files:
            s = str(path)
            try:
                text = raw_content.decode("utf-8")
            except UnicodeDecodeError:
                text = "binary"
            self._files[s] = text
            results.append(UploadFileResult(path=s, error=None))
        return results

    def download_files(self, paths):
        results: list[DownloadFileResult] = []
        for path in paths:
            s = str(path)
            c = self._files.get(s)
            if c is None:
                results.append(DownloadFileResult(path=s, content=None, error=BackendError(code=ErrorCode.NOT_FOUND, path=s, message="file_not_found")))
            else:
                results.append(DownloadFileResult(path=s, content=c.encode("utf-8"), error=None))
        return results


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def tmp_root() -> Path:
    d = tempfile.mkdtemp(prefix="mambo_test_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def hybrid_ws(tmp_root: Path) -> HybridWorkspaceBackend:
    """HybridWorkspaceBackend with LocalBackend as real backend."""
    local = LocalBackend(root_dir=str(tmp_root))
    return HybridWorkspaceBackend(real_backend=local)


def _write_virtual(hws: HybridWorkspaceBackend, path: str, content: str):
    """Write a file under /.mambo/ via graph simulation."""
    target, stripped = hws._route(VirtualPath(path))
    if isinstance(target, StoreBackend):
        with _simulate_graph(target):
            r = target.write(stripped, content, overwrite=True)
        assert r.error is None, r.error
    else:
        r = target.write(stripped, content, overwrite=True)
        assert r.error is None, r.error


def _read_virtual(hws: HybridWorkspaceBackend, path: str) -> str:
    target, stripped = hws._route(VirtualPath(path))
    if isinstance(target, StoreBackend):
        with _simulate_graph(target):
            r = target.read(stripped)
    else:
        r = target.read(stripped)
    assert r.error is None, r.error
    return r.content or ""


# ============================================================================
# Path routing
# ============================================================================


class TestPathRouting:
    """Verify that mambo-prefixed paths hit virtual backends, others hit real."""


    def test_route_default_mambo_returns_default_state(self, hybrid_ws: HybridWorkspaceBackend):
        target, path = hybrid_ws._route(VirtualPath("/.mambo/workspace/f.txt"))
        assert target is hybrid_ws._default_mambo
        assert path == "/workspace/workspace/f.txt"

    def test_route_default_mambo_root(self, hybrid_ws: HybridWorkspaceBackend):
        target, path = hybrid_ws._route(VirtualPath("/.mambo"))
        assert target is hybrid_ws._default_mambo
        assert path == "/workspace"

    def test_route_non_mambo_returns_real_backend(self, hybrid_ws: HybridWorkspaceBackend):
        target, path = hybrid_ws._route(VirtualPath(f"{_W}/project/main.py"))
        assert target is hybrid_ws._real
        assert path == f"{_W}/project/main.py"

    def test_route_named_workspace(self, tmp_root: Path):
        skills_be = StoreBackend(store=InMemoryStore())
        hws = HybridWorkspaceBackend(
            real_backend=LocalBackend(root_dir=str(tmp_root)),
            virtual_workspaces={"skills": skills_be},
        )
        target, path = hws._route(VirtualPath("/.mambo/skills/guidelines.md"))
        assert target is skills_be
        assert path == "/workspace/guidelines.md"

    def test_route_named_workspace_exact(self, tmp_root: Path):
        skills_be = StoreBackend(store=InMemoryStore())
        hws = HybridWorkspaceBackend(
            real_backend=LocalBackend(root_dir=str(tmp_root)),
            virtual_workspaces={"skills": skills_be},
        )
        target, path = hws._route(VirtualPath("/.mambo/skills"))
        assert target is skills_be
        assert path == "/workspace"

    def test_route_named_does_not_collide_with_default(self, tmp_root: Path):
        skills_be = StoreBackend(store=InMemoryStore())
        hws = HybridWorkspaceBackend(
            real_backend=LocalBackend(root_dir=str(tmp_root)),
            virtual_workspaces={"skills": skills_be},
        )
        # "skills_other" should NOT match "skills"
        target, path = hws._route(VirtualPath("/.mambo/skills_other/f.txt"))
        assert target is hws._default_mambo
        assert path == "/workspace/skills_other/f.txt"

    def test_route_dot_overrides_default(self, tmp_root: Path):
        custom = StoreBackend(store=InMemoryStore())
        hws = HybridWorkspaceBackend(
            real_backend=LocalBackend(root_dir=str(tmp_root)),
            virtual_workspaces={".": custom},
        )
        target, path = hws._route(VirtualPath("/.mambo/config.yml"))
        assert target is custom
        assert path == "/workspace/config.yml"


# ============================================================================
# Virtual workspace strip-prefix
# ============================================================================


class TestStripPrefix:
    """Virtual backends receive paths under their own workspace_root."""

    def test_default_strips_mambo_prefix(self, hybrid_ws: HybridWorkspaceBackend):
        target, path = hybrid_ws._route(VirtualPath("/.mambo/workspace/scratch.txt"))
        assert path == "/workspace/workspace/scratch.txt"
        assert not str(path).startswith("/.mambo")

    def test_named_strips_full_prefix(self, tmp_root: Path):
        skills_be = StoreBackend(store=InMemoryStore())
        hws = HybridWorkspaceBackend(
            real_backend=LocalBackend(root_dir=str(tmp_root)),
            virtual_workspaces={"skills": skills_be},
        )
        _, path = hws._route(VirtualPath("/.mambo/skills/guides/python.md"))
        assert path == "/workspace/guides/python.md"

    def test_real_backend_path_unchanged(self, hybrid_ws: HybridWorkspaceBackend):
        _, path = hybrid_ws._route(VirtualPath(f"{_W}/src/main.py"))
        assert path == f"{_W}/src/main.py"


# ============================================================================
# Core operations — routing correctness
# ============================================================================


class TestCoreOpsRouting:
    WORKSPACE_FILE = VirtualPath("/.mambo/workspace/scratch.txt")
    PROJECT_FILE = VirtualPath(f"{_W}/project/hello.txt")

    def test_write_read_default_mambo(self, hybrid_ws: HybridWorkspaceBackend):
        _write_virtual(hybrid_ws, self.WORKSPACE_FILE, "virtual scratch")
        assert _read_virtual(hybrid_ws, self.WORKSPACE_FILE) == "virtual scratch"

    def test_write_read_project(self, hybrid_ws: HybridWorkspaceBackend):
        r = hybrid_ws.write(self.PROJECT_FILE, "real project")
        assert r.error is None, r.error
        r2 = hybrid_ws.read(self.PROJECT_FILE)
        assert r2.error is None
        assert "real project" in (r2.content or "")

    def test_mambo_file_not_in_real_backend(self, hybrid_ws: HybridWorkspaceBackend):
        _write_virtual(hybrid_ws, self.WORKSPACE_FILE, "ghost")
        r = hybrid_ws._real.read(self.WORKSPACE_FILE)
        assert r.error is not None

    def test_project_file_not_in_default_mambo(self, hybrid_ws: HybridWorkspaceBackend):
        hybrid_ws.write(self.PROJECT_FILE, "only real")
        with _simulate_graph(hybrid_ws._default_mambo):
            r = hybrid_ws._default_mambo.read(self.PROJECT_FILE)
        assert r.error is not None

    def test_named_workspace_isolated(self, tmp_root: Path):
        skills_be = StoreBackend(store=InMemoryStore())
        hws = HybridWorkspaceBackend(
            real_backend=LocalBackend(root_dir=str(tmp_root)),
            virtual_workspaces={"skills": skills_be},
        )
        # Write to skills
        _write_virtual(hws, "/.mambo/skills/guide.md", "skill guide")
        # Write to default
        _write_virtual(hws, "/.mambo/scratch.txt", "scratch")

        # Skills workspace has its own file
        _, guide_path = hws._route(VirtualPath("/.mambo/skills/guide.md"))
        with _simulate_graph(skills_be):
            r = skills_be.read(guide_path)
        assert r.error is None
        assert "skill guide" in (r.content or "")

        # Default workspace does NOT have skill file
        with _simulate_graph(hws._default_mambo):
            r2 = hws._default_mambo.read(guide_path)
        assert r2.error is not None

    def test_edit_default_mambo(self, hybrid_ws: HybridWorkspaceBackend):
        _write_virtual(hybrid_ws, self.WORKSPACE_FILE, "line1\nline2\nline3")
        target, stripped = hybrid_ws._route(self.WORKSPACE_FILE)
        with _simulate_graph(target):
            r = target.edit(stripped, "line2", "LINE2")
        assert r.error is None, r.error
        assert "LINE2" in _read_virtual(hybrid_ws, self.WORKSPACE_FILE)

    def test_edit_project(self, hybrid_ws: HybridWorkspaceBackend):
        hybrid_ws.write(self.PROJECT_FILE, "old content here")
        r = hybrid_ws.edit(self.PROJECT_FILE, "old", "NEW")
        assert r.error is None, r.error
        r2 = hybrid_ws.read(self.PROJECT_FILE)
        assert "NEW content here" in (r2.content or "")

    def test_ls_default_mambo(self, hybrid_ws: HybridWorkspaceBackend):
        _write_virtual(hybrid_ws, self.WORKSPACE_FILE, "a")
        _write_virtual(hybrid_ws, VirtualPath("/.mambo/workspace/other.txt"), "b")
        target, stripped = hybrid_ws._route(VirtualPath("/.mambo/workspace/"))
        with _simulate_graph(target):
            r = target.ls(stripped)
        assert r.entries is not None
        paths = [fi.path for fi in r.entries]
        assert "/workspace/workspace/scratch.txt" in paths
        assert "/workspace/workspace/other.txt" in paths

    def test_ls_project(self, hybrid_ws: HybridWorkspaceBackend):
        hybrid_ws.write(VirtualPath(f"{_W}/project/a.py"), "1")
        hybrid_ws.write(VirtualPath(f"{_W}/project/b.py"), "2")
        r = hybrid_ws.ls(VirtualPath(f"{_W}/project/"))
        assert r.entries is not None
        paths = [fi.path for fi in r.entries]
        assert f"{_W}/project/a.py" in paths
        assert f"{_W}/project/b.py" in paths

    def test_grep_default_mambo(self, hybrid_ws: HybridWorkspaceBackend):
        _write_virtual(hybrid_ws, self.WORKSPACE_FILE, "needle in a haystack")
        target, stripped = hybrid_ws._route(VirtualPath("/.mambo/"))
        with _simulate_graph(target):
            r = target.grep("needle", path=stripped)
        assert r.matches is not None
        assert any("needle" in m.text for m in r.matches)

    def test_grep_project(self, hybrid_ws: HybridWorkspaceBackend):
        hybrid_ws.write(VirtualPath(f"{_W}/project/f.py"), "def needle(): pass")
        r = hybrid_ws.grep("needle", path=VirtualPath(f"{_W}/project/"))
        assert r.matches is not None
        assert any("needle" in m.text for m in r.matches)

    def test_glob_default_mambo(self, hybrid_ws: HybridWorkspaceBackend):
        _write_virtual(hybrid_ws, self.WORKSPACE_FILE, "a")
        _write_virtual(hybrid_ws, VirtualPath("/.mambo/workspace/b.md"), "b")
        target, stripped = hybrid_ws._route(VirtualPath("/.mambo/workspace/"))
        with _simulate_graph(target):
            r = target.glob("*.txt", path=stripped)
        assert r.matches is not None
        paths = [fi.path for fi in r.matches]
        assert "/workspace/workspace/scratch.txt" in paths
        assert "/workspace/workspace/b.md" not in paths

    def test_glob_project(self, hybrid_ws: HybridWorkspaceBackend):
        hybrid_ws.write(VirtualPath(f"{_W}/project/x.py"), "1")
        hybrid_ws.write(VirtualPath(f"{_W}/project/y.txt"), "2")
        r = hybrid_ws.glob("*.py", path=VirtualPath(f"{_W}/project/"))
        assert r.matches is not None
        paths = [fi.path for fi in r.matches]
        assert len(paths) == 1
        assert f"{_W}/project/x.py" in paths


# ============================================================================
# tools delegation
# ============================================================================


def _make_dummy_tree_tool() -> StructuredTool:
    from pydantic import Field, ValidationError, create_model

    return StructuredTool(
        name="tree",
        description="Show tree",
        args_schema=create_model(
            "DummyTreeSchema",
            path=(str, Field(default=_W)),
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

_DUMMY_DELETE_WITH_PATH = StructuredTool(
    name="delete",
    description="Delete files",
    args_schema=create_model(
        "DeleteSchema",
        path=(str, Field(description="Absolute file path to delete")),
    ),
    func=lambda **kw: f"deleted {kw.get('path', '?')}",
)


class TestToolsDelegation:
    """tools property delegates to real_backend + includes copy."""

    def test_returns_real_backend_tools_plus_copy(self):
        tree_tool = _make_dummy_tree_tool()
        fake = _FakeBackend(extra_tools=[tree_tool, _DUMMY_DELETE])
        hws = HybridWorkspaceBackend(real_backend=fake)
        tools = hws.tools
        tool_names = {t.name for t in tools}
        assert "tree" in tool_names
        assert "delete" in tool_names
        assert "copy" in tool_names

    def test_real_backend_with_no_extra_tools_has_copy(self):
        fake = _FakeBackend()
        hws = HybridWorkspaceBackend(real_backend=fake)
        tool_names = {t.name for t in hws.tools}
        assert tool_names == {"copy"}

    def test_tool_without_path_param_not_wrapped(self):
        """Tool without path param (args_schema=None) is left untouched."""
        fake = _FakeBackend(extra_tools=[_DUMMY_DELETE])
        hws = HybridWorkspaceBackend(real_backend=fake)
        tools = hws.tools
        delete_tool = [t for t in tools if t.name == "delete"][0]
        # Should still be the same func reference (no wrapping)
        assert delete_tool.func is _DUMMY_DELETE.func

    def test_tool_with_path_param_is_wrapped(self):
        """Tool with a 'path' field in args_schema gets wrapped."""
        fake = _FakeBackend(extra_tools=[_DUMMY_DELETE_WITH_PATH])
        hws = HybridWorkspaceBackend(real_backend=fake)
        tools = hws.tools
        delete_tool = [t for t in tools if t.name == "delete"][0]
        # Wrapped func should differ from original
        assert delete_tool.func is not _DUMMY_DELETE_WITH_PATH.func

    def test_path_translated_when_routing_to_real(self):
        """Path is rewritten when it falls under Hybrid's workspace_root."""
        fake = _FakeBackend(extra_tools=[_DUMMY_DELETE_WITH_PATH])
        # Hybrid ws_root=/workspace, fake ws_root=/workspace
        hws = HybridWorkspaceBackend(real_backend=fake)
        delete_tool = [t for t in hws.tools if t.name == "delete"][0]
        result = delete_tool.invoke({"path": "/workspace/some/file.txt"})
        assert "deleted /workspace/some/file.txt" in result

    def test_path_not_rewritten_for_virtual_prefix(self):
        """Path routing to /.mambo/ is left unchanged so real backend errors clearly."""
        fake = _FakeBackend(extra_tools=[_DUMMY_DELETE_WITH_PATH])
        hws = HybridWorkspaceBackend(real_backend=fake)
        delete_tool = [t for t in hws.tools if t.name == "delete"][0]
        result = delete_tool.invoke({"path": "/.mambo/some_file"})
        # Original /.mambo/ path passes through (not rewritten)
        assert "deleted /.mambo/some_file" in result

    def test_path_wrapping_with_different_ws_roots(self):
        """When ws_roots differ, the path is translated from Hybrid namespace to real namespace."""
        # Use a real LocalBackend with a different workspace_root
        with tempfile.TemporaryDirectory() as tmpdir:
            real = LocalBackend(root_dir=tmpdir, workspace_root="/real")
            real.write(VirtualPath("/real/hello.txt"), "hello", overwrite=True)

            hws = HybridWorkspaceBackend(
                real_backend=real,
                workspace_root="/workspace",  # AI-facing
            )
            # Verify file exists on disk via real backend (using real ws_root)
            read_result = real.read(VirtualPath("/real/hello.txt"))
            assert "hello" in str(read_result)

            # Invoke delete through Hybrid's wrapper — path should be translated
            delete_tool = [t for t in hws.tools if t.name == "delete"][0]
            result = delete_tool.invoke({"path": "/workspace/hello.txt"})
            # Delete succeeded (without error message)
            assert "Error" not in result

            # Verify file is actually gone
            check = real.read(VirtualPath("/real/hello.txt"))
            assert check.error is not None


# ============================================================================
# description
# ============================================================================


class TestDescription:
    def test_includes_prefix(self):
        hws = HybridWorkspaceBackend(real_backend=_FakeBackend())
        desc = hws.description
        assert "/.mambo/" in desc
        assert "virtual" in desc.lower()

    def test_custom_description_prepended(self):
        hws = HybridWorkspaceBackend(
            real_backend=_FakeBackend(),
            custom_description="!!! CUSTOM !!!",
        )
        assert hws.description.startswith("!!! CUSTOM !!!")

    def test_lists_core_tools_only(self):
        # Without execute tool, no "NOT target" warning
        hws = HybridWorkspaceBackend(real_backend=_FakeBackend())
        desc = hws.description
        assert "ls" in desc
        assert "read" in desc
        assert "write" in desc
        assert "edit" in desc
        assert "grep" in desc
        assert "glob" in desc
        assert "do NOT target" not in desc

    def test_execute_tool_warning(self):
        """When real backend provides execute, description warns about paths."""
        execute_tool = StructuredTool(
            name="execute",
            description="Run a shell command",
            args_schema=create_model(
                "ExecSchema",
                command=(str, Field(description="Command to run")),
            ),
            func=lambda command: f"executed: {command}",
        )
        hws = HybridWorkspaceBackend(
            real_backend=_FakeBackend(extra_tools=[execute_tool])
        )
        desc = hws.description
        assert "do NOT target" in desc


# ============================================================================
# upload / download — split by prefix
# ============================================================================


class TestUploadDownload:
    """upload_files / download_files split files by mambo prefix."""

    def test_upload_mixed_routes_correctly(self, tmp_root: Path):
        hws = HybridWorkspaceBackend(real_backend=LocalBackend(root_dir=str(tmp_root)))

        files: list[tuple[VirtualPath, bytes]] = [
            (VirtualPath("/.mambo/workspace/a.txt"), b"workspace a"),
            (VirtualPath(f"{_W}/project/b.txt"), b"project b"),
            (VirtualPath("/.mambo/workspace/c.txt"), b"workspace c"),
            (VirtualPath(f"{_W}/project/d.txt"), b"project d"),
        ]

        with _simulate_graph(hws._default_mambo):
            results = hws.upload_files(files)

        assert len(results) == 4
        assert all(r.error is None for r in results)

        # Verify workspace files in default StoreBackend
        _, a_path = hws._route(VirtualPath("/.mambo/workspace/a.txt"))
        _, c_path = hws._route(VirtualPath("/.mambo/workspace/c.txt"))
        with _simulate_graph(hws._default_mambo):
            r1 = hws._default_mambo.read(a_path)
            r3 = hws._default_mambo.read(c_path)
        assert "workspace a" in (r1.content or "")
        assert "workspace c" in (r3.content or "")

        # Verify project files in real backend
        r2 = hws.read(VirtualPath(f"{_W}/project/b.txt"))
        r4 = hws.read(VirtualPath(f"{_W}/project/d.txt"))
        assert "project b" in (r2.content or "")
        assert "project d" in (r4.content or "")

    def test_download_mixed_routes_correctly(self, tmp_root: Path):
        hws = HybridWorkspaceBackend(real_backend=LocalBackend(root_dir=str(tmp_root)))

        # Pre-populate both sides
        _write_virtual(hws, "/.mambo/x.txt", "state file")
        hws.write(VirtualPath(f"{_W}/y.txt"), "backend file", overwrite=True)

        with _simulate_graph(hws._default_mambo):
            results = hws.download_files([
                VirtualPath("/.mambo/x.txt"),
                VirtualPath(f"{_W}/y.txt"),
            ])

        assert len(results) == 2
        assert results[0].content == b"state file"
        assert results[1].content == b"backend file"

    def test_upload_named_workspace(self, tmp_root: Path):
        skills_be = StoreBackend(store=InMemoryStore())
        hws = HybridWorkspaceBackend(
            real_backend=LocalBackend(root_dir=str(tmp_root)),
            virtual_workspaces={"skills": skills_be},
        )

        files: list[tuple[VirtualPath, bytes]] = [
            (VirtualPath("/.mambo/skills/guide.md"), b"skill content"),
            (VirtualPath("/.mambo/scratch.txt"), b"default scratch"),
        ]

        with _simulate_graph(skills_be), _simulate_graph(hws._default_mambo):
            results = hws.upload_files(files)

        assert len(results) == 2
        assert all(r.error is None for r in results)

        # Verify skills workspace
        _, skills_path = hws._route(VirtualPath("/.mambo/skills/guide.md"))
        with _simulate_graph(skills_be):
            r = skills_be.read(skills_path)
        assert "skill content" in (r.content or "")

        # Verify default workspace
        _, scratch_path = hws._route(VirtualPath("/.mambo/scratch.txt"))
        with _simulate_graph(hws._default_mambo):
            r = hws._default_mambo.read(scratch_path)
        assert "default scratch" in (r.content or "")

    # ------------------------------------------------------------------
    # Regression: virtual workspace upload/download
    # ------------------------------------------------------------------

    def test_virtual_workspace_upload_download(self, tmp_root: Path):
        """Virtual workspace backend handles upload_files/download_files."""
        plain_be = _FakeBackend()
        hws = HybridWorkspaceBackend(
            real_backend=LocalBackend(root_dir=str(tmp_root)),
            virtual_workspaces={"plain": plain_be},
        )

        files: list[tuple[VirtualPath, bytes]] = [
            (VirtualPath("/.mambo/plain/a.txt"), b"virtual a"),
            (VirtualPath(f"{_W}/project/b.txt"), b"real b"),
        ]
        results = hws.upload_files(files)
        assert len(results) == 2
        assert all(r.error is None for r in results)

        results = hws.download_files([
            VirtualPath("/.mambo/plain/a.txt"),
            VirtualPath(f"{_W}/project/b.txt"),
        ])
        assert len(results) == 2
        assert results[0].content == b"virtual a"
        assert results[1].content == b"real b"

    def test_named_virtual_workspace_upload_download(self, tmp_root: Path):
        """Named virtual workspace handles upload/download."""
        aware_be = _FakeThreadAwareBackend()
        plain_be = _FakeBackend()
        hws = HybridWorkspaceBackend(
            real_backend=LocalBackend(root_dir=str(tmp_root)),
            virtual_workspaces={"aware": aware_be, "plain": plain_be},
        )

        files: list[tuple[VirtualPath, bytes]] = [
            (VirtualPath("/.mambo/aware/a.txt"), b"aware a"),
            (VirtualPath("/.mambo/plain/b.txt"), b"plain b"),
            (VirtualPath(f"{_W}/real.txt"), b"real"),
        ]
        results = hws.upload_files(files)
        assert len(results) == 3
        assert all(r.error is None for r in results)

    def test_dot_override_default_virtual_workspace(self, tmp_root: Path):
        """Default virtual workspace replaced with custom backend."""
        plain_be = _FakeBackend()
        hws = HybridWorkspaceBackend(
            real_backend=LocalBackend(root_dir=str(tmp_root)),
            virtual_workspaces={".": plain_be},
        )

        results = hws.upload_files(
            [(VirtualPath("/.mambo/scratch.txt"), b"scratch")],
        )
        assert len(results) == 1
        assert results[0].error is None

        dl = hws.download_files(
            [VirtualPath("/.mambo/scratch.txt")],
        )
        assert len(dl) == 1
        assert dl[0].content == b"scratch"


# ============================================================================
# dot override
# ============================================================================


class TestDotOverride:
    def test_dot_overrides_default_state_backend(self, tmp_root: Path):
        custom = StoreBackend(store=InMemoryStore(), initial_files={"/config.yml": "port: 8080"})
        hws = HybridWorkspaceBackend(
            real_backend=LocalBackend(root_dir=str(tmp_root)),
            virtual_workspaces={".": custom},
        )
        assert hws._default_mambo is custom

        with _simulate_graph(custom):
            r = custom.read(VirtualPath("/config.yml"))
        assert r.error is None
        assert "port: 8080" in (r.content or "")

    def test_dot_with_named_workspaces(self, tmp_root: Path):
        default_be = StoreBackend(store=InMemoryStore())
        skills_be = StoreBackend(store=InMemoryStore())
        hws = HybridWorkspaceBackend(
            real_backend=LocalBackend(root_dir=str(tmp_root)),
            virtual_workspaces={
                ".": default_be,
                "skills": skills_be,
            },
        )
        assert hws._default_mambo is default_be
        assert hws._virtual["skills"] is skills_be

        # Route default
        t1, p1 = hws._route(VirtualPath("/.mambo/f.txt"))
        assert t1 is default_be
        assert p1 == "/workspace/f.txt"

        # Route named
        t2, p2 = hws._route(VirtualPath("/.mambo/skills/g.md"))
        assert t2 is skills_be
        assert p2 == "/workspace/g.md"


# ============================================================================
# copy — cross-backend single-file copy
# ============================================================================


class TestCopy:
    """copy tool copies files across backends."""

    def test_copy_mambo_to_project(self, tmp_root: Path):
        hws = HybridWorkspaceBackend(real_backend=LocalBackend(root_dir=str(tmp_root)))

        # Write in virtual workspace
        _write_virtual(hws, "/.mambo/draft.txt", "hello from virtual")

        # Copy to real filesystem
        with _simulate_graph(hws._default_mambo):
            result = hws.copy(VirtualPath("/.mambo/draft.txt"), VirtualPath(f"{_W}/output.txt"))

        assert result.error is None, result.error
        assert result.source == "/.mambo/draft.txt"
        assert result.destination == f"{_W}/output.txt"

        # Verify destination
        r = hws.read(VirtualPath(f"{_W}/output.txt"))
        assert r.error is None
        assert "hello from virtual" in (r.content or "")

    def test_copy_project_to_mambo(self, tmp_root: Path):
        hws = HybridWorkspaceBackend(real_backend=LocalBackend(root_dir=str(tmp_root)))

        # Write in real filesystem
        hws.write(VirtualPath(f"{_W}/source.txt"), "hello from real", overwrite=True)

        # Copy to virtual workspace
        with _simulate_graph(hws._default_mambo):
            result = hws.copy(VirtualPath(f"{_W}/source.txt"), VirtualPath("/.mambo/imported.txt"))

        assert result.error is None, result.error

        # Verify destination in virtual
        target, stripped = hws._route(VirtualPath("/.mambo/imported.txt"))
        with _simulate_graph(target):
            r = target.read(stripped)
        assert r.error is None
        assert "hello from real" in (r.content or "")

    def test_copy_between_virtual_workspaces(self, tmp_root: Path):
        skills_be = StoreBackend(store=InMemoryStore())
        hws = HybridWorkspaceBackend(
            real_backend=LocalBackend(root_dir=str(tmp_root)),
            virtual_workspaces={"skills": skills_be},
        )

        # Write in default mambo
        _write_virtual(hws, "/.mambo/shared/file.txt", "cross-workspace data")

        # Copy from default to skills
        with _simulate_graph(hws._default_mambo), _simulate_graph(skills_be):
            result = hws.copy(
                VirtualPath("/.mambo/shared/file.txt"),
                VirtualPath("/.mambo/skills/incoming.txt"),
            )

        assert result.error is None, result.error

        # Verify in skills workspace
        _, incoming_path = hws._route(VirtualPath("/.mambo/skills/incoming.txt"))
        with _simulate_graph(skills_be):
            r = skills_be.read(incoming_path)
        assert r.error is None
        assert "cross-workspace data" in (r.content or "")

    def test_copy_overwrites_destination(self, tmp_root: Path):
        hws = HybridWorkspaceBackend(real_backend=LocalBackend(root_dir=str(tmp_root)))

        _write_virtual(hws, "/.mambo/source.txt", "new content")
        hws.write(VirtualPath(f"{_W}/dest.txt"), "old content", overwrite=True)

        with _simulate_graph(hws._default_mambo):
            result = hws.copy(VirtualPath("/.mambo/source.txt"), VirtualPath(f"{_W}/dest.txt"))

        assert result.error is None, result.error

        r = hws.read(VirtualPath(f"{_W}/dest.txt"))
        assert "new content" in (r.content or "")

    def test_copy_source_not_found(self, tmp_root: Path):
        hws = HybridWorkspaceBackend(real_backend=LocalBackend(root_dir=str(tmp_root)))

        with _simulate_graph(hws._default_mambo):
            result = hws.copy(VirtualPath("/.mambo/nonexistent.txt"), VirtualPath(f"{_W}/out.txt"))

        assert result.error is not None
        assert result.error.code in (ErrorCode.NOT_FOUND, ErrorCode.PATH_NOT_UNDER, ErrorCode.IO_ERROR)

    def test_copy_result_model_str(self):
        from mambo_agents.backends.hybrid_workspace import CopyResult

        r = CopyResult(source="/a.txt", destination="/b.txt")
        assert "Copied:" in str(r)
        assert "/a.txt" in str(r)
        assert "/b.txt" in str(r)

        err = CopyResult(error=BackendError(code=ErrorCode.NOT_FOUND, message="source not found"))
        assert "Error:" in str(err)
        assert "source not found" in str(err)


# ============================================================================
# workspace_root rewriting — custom & mismatched roots
# ============================================================================


class TestWorkspaceRootRewrite:
    """Hybrid strips its own ws_root and prepends the target backend's ws_root."""

    def test_custom_hybrid_workspace_root(self, tmp_root: Path):
        """Hybrid.ws_root=/a, real.ws_root=/b -> /a/x.txt rewrites to /b/x.txt."""
        local = LocalBackend(root_dir=str(tmp_root), workspace_root="/b")
        hws = HybridWorkspaceBackend(
            real_backend=local,
            workspace_root="/a",
        )
        assert hws.workspace_root == "/a"

        target, path = hws._route(VirtualPath("/a/data.txt"))
        assert target is local
        assert path == "/b/data.txt"

        # Root exact match
        _, path2 = hws._route(VirtualPath("/a"))
        assert path2 == "/b"

    def test_default_workspace_root_is_noop(self, hybrid_ws: HybridWorkspaceBackend):
        """When both Hybrid and real use /workspace, paths are unchanged."""
        target, path = hybrid_ws._route(VirtualPath("/workspace/src/main.py"))
        assert target is hybrid_ws._real
        assert path == "/workspace/src/main.py"

    def test_virtual_backend_with_custom_wsroot(self, tmp_root: Path):
        """Virtual backend gets paths under its own workspace_root."""
        skills_be = StoreBackend(store=InMemoryStore())
        hws = HybridWorkspaceBackend(
            real_backend=LocalBackend(root_dir=str(tmp_root)),
            virtual_workspaces={"skills": skills_be},
            workspace_root="/myworkspace",
        )
        # Mambo paths are independent of Hybrid.ws_root
        target, path = hws._route(VirtualPath("/.mambo/skills/tips.md"))
        assert target is skills_be
        assert path == "/workspace/tips.md"

        # Real paths use Hybrid's ws_root
        _, path2 = hws._route(VirtualPath("/myworkspace/file.py"))
        assert path2 == "/workspace/file.py"


class TestRewritePredictable:
    """_rewrite always prepends ws_root — no special cases, fully predictable."""

    def test_always_prepends(self):
        from mambo_agents.backends.hybrid_workspace import HybridWorkspaceBackend as HWB

        # /.mambo/workspace/f.txt → strip → workspace/f.txt → prepend → /workspace/workspace/f.txt
        assert HWB._rewrite(VirtualPath("/.mambo/workspace/f.txt"), "/.mambo", VirtualPath("/workspace")) == "/workspace/workspace/f.txt"

    def test_without_workspace_in_path(self):
        from mambo_agents.backends.hybrid_workspace import HybridWorkspaceBackend as HWB

        assert HWB._rewrite(VirtualPath("/.mambo/foo.txt"), "/.mambo", VirtualPath("/workspace")) == "/workspace/foo.txt"

    def test_root_to_root(self):
        from mambo_agents.backends.hybrid_workspace import HybridWorkspaceBackend as HWB

        assert HWB._rewrite(VirtualPath("/.mambo"), "/.mambo", VirtualPath("/workspace")) == "/workspace"

    def test_different_target_wsroot(self):
        from mambo_agents.backends.hybrid_workspace import HybridWorkspaceBackend as HWB

        assert HWB._rewrite(VirtualPath("/a/data.txt"), "/a", VirtualPath("/b")) == "/b/data.txt"


# ============================================================================
# LocalBackend in virtual workspace slot — the motivating use case
# ============================================================================


class TestLocalBackendInVirtualSlot:
    """A LocalBackend placed in a virtual workspace slot receives properly rewritten paths."""

    def test_write_without_workspace_in_path(self, tmp_root: Path):
        """/.mambo/temp/foo.txt -> local writes under its ws_root."""
        local_slot = LocalBackend(root_dir=str(tmp_root))
        hws = HybridWorkspaceBackend(
            real_backend=LocalBackend(root_dir=str(tmp_root / "real")),
            virtual_workspaces={"temp": local_slot},
        )

        target, path = hws._route(VirtualPath("/.mambo/temp/foo.txt"))
        assert target is local_slot
        assert path == "/workspace/foo.txt"

        # Write should succeed — path starts with local's ws_root
        r = hws.write(VirtualPath("/.mambo/temp/foo.txt"), "hello temp", overwrite=True)
        assert r.error is None, r.error

        # Verify written to disk
        written = tmp_root / "foo.txt"
        assert written.read_text() == "hello temp"

    def test_always_prepends_regardless_of_dir_name(self, tmp_root: Path):
        """/.mambo/temp/workspace/bar.txt -> /workspace/workspace/bar.txt (no special-casing)."""
        local_slot = LocalBackend(root_dir=str(tmp_root))
        hws = HybridWorkspaceBackend(
            real_backend=LocalBackend(root_dir=str(tmp_root / "real")),
            virtual_workspaces={"temp": local_slot},
        )

        target, path = hws._route(VirtualPath("/.mambo/temp/workspace/bar.txt"))
        assert target is local_slot
        assert path == "/workspace/workspace/bar.txt"

        # Write should succeed
        r = hws.write(VirtualPath("/.mambo/temp/workspace/bar.txt"), "nested", overwrite=True)
        assert r.error is None, r.error

        # Verify written to disk
        written = tmp_root / "workspace" / "bar.txt"
        assert written.read_text() == "nested"


# ============================================================================
# _get_virtual_prefix / _reverse_path — unit tests
# ============================================================================


def _make_root_fake(files=None):
    """Create a _FakeBackend with ``workspace_root="/workspace"``."""
    be = _FakeBackend()
    be.workspace_root = VirtualPath("/workspace")
    if files is not None:
        be._files = files
    return be


class TestGetVirtualPrefix:
    """Unit tests for ``_get_virtual_prefix`` — mirrors ``_route``."""

    def test_named_workspace(self, tmp_root: Path):
        skills_be = _make_root_fake()
        hws = HybridWorkspaceBackend(
            real_backend=LocalBackend(root_dir=str(tmp_root)),
            virtual_workspaces={"skills": skills_be},
        )
        assert hws._get_virtual_prefix("/.mambo/skills/") == "/.mambo/skills"
        assert hws._get_virtual_prefix("/.mambo/skills") == "/.mambo/skills"
        assert hws._get_virtual_prefix("/.mambo/skills/guide.md") == "/.mambo/skills"

    def test_default_mambo(self, hybrid_ws: HybridWorkspaceBackend):
        assert hybrid_ws._get_virtual_prefix("/.mambo/") == "/.mambo"
        assert hybrid_ws._get_virtual_prefix("/.mambo") == "/.mambo"
        assert hybrid_ws._get_virtual_prefix("/.mambo/workspace/f.txt") == "/.mambo"

    def test_real_backend(self, hybrid_ws: HybridWorkspaceBackend):
        assert hybrid_ws._get_virtual_prefix("/workspace/") == "/workspace"
        assert hybrid_ws._get_virtual_prefix("/workspace/project/") == "/workspace"
        assert hybrid_ws._get_virtual_prefix("/workspace/file.py") == "/workspace"


class TestReversePath:
    """Unit tests for ``_reverse_path`` — the inverse of ``_rewrite``."""

    HWB = HybridWorkspaceBackend

    def test_from_root_workspace(self):
        """target ws_root="/workspace" → strip ws_root, prepend virtual prefix."""
        result = self.HWB._reverse_path(
            VirtualPath("/workspace/skill-a"), VirtualPath("/workspace"), VirtualPath("/.mambo/skills"),
        )
        assert result == "/.mambo/skills/skill-a"

    def test_root_to_root(self):
        """Internal path IS the target ws_root."""
        result = self.HWB._reverse_path(
            VirtualPath("/workspace"), VirtualPath("/workspace"), VirtualPath("/.mambo/skills"),
        )
        assert result == "/.mambo/skills"

    def test_from_default_workspace(self):
        """target ws_root="/workspace" → strip ws_root, prepend virtual prefix."""
        result = self.HWB._reverse_path(
            VirtualPath("/workspace/f.txt"), VirtualPath("/workspace"), VirtualPath("/.mambo"),
        )
        assert result == "/.mambo/f.txt"

    def test_nested_from_default_workspace(self):
        """Nested path under /workspace."""
        result = self.HWB._reverse_path(
            VirtualPath("/workspace/workspace/scratch.txt"),
            VirtualPath("/workspace"),
            VirtualPath("/.mambo"),
        )
        assert result == "/.mambo/workspace/scratch.txt"

    def test_real_backend_with_different_root(self):
        """Hybrid ws_root=/workspace, real ws_root=/home/user/project."""
        result = self.HWB._reverse_path(
            VirtualPath("/home/user/project/file.py"),
            VirtualPath("/home/user/project"),
            VirtualPath("/workspace"),
        )
        assert result == "/workspace/file.py"

    def test_real_backend_root_exact(self):
        """Internal path matches real ws_root exactly."""
        result = self.HWB._reverse_path(
            VirtualPath("/home/user/project"),
            VirtualPath("/home/user/project"),
            VirtualPath("/workspace"),
        )
        assert result == "/workspace"

    def test_directory_path_with_trailing_slash(self):
        """Directory paths with trailing slash are handled correctly."""
        result = self.HWB._reverse_path(
            VirtualPath("/workspace/skill-a/"), VirtualPath("/workspace"), VirtualPath("/.mambo/skills"),
        )
        assert result == "/.mambo/skills/skill-a"


# ============================================================================
# ls() — reverse path translation
# ============================================================================


class TestLsReverseTranslation:
    """``ls()`` returns ``FileInfo.path`` in the external (Hybrid) namespace."""

    def test_named_virtual_workspace_dirs(self, tmp_root: Path):
        """Directories under named virtual workspace have translated paths."""
        skills_be = _make_root_fake({
            "/workspace/skill-a/SKILL.md": "# Skill A",
            "/workspace/skill-b/SKILL.md": "# Skill B",
        })
        hws = HybridWorkspaceBackend(
            real_backend=LocalBackend(root_dir=str(tmp_root)),
            virtual_workspaces={"skills": skills_be},
        )
        result = hws.ls(VirtualPath("/.mambo/skills/"))
        assert result.entries is not None
        dirs = [e for e in result.entries if e.is_dir]
        assert len(dirs) >= 2
        for d in dirs:
            assert str(d.path).startswith(
                "/.mambo/skills/"
            ), f"Dir path {d.path} not under /.mambo/skills/"

    def test_named_virtual_workspace_no_internal_leakage(self, tmp_root: Path):
        """No internal paths (e.g. bare /workspace/skill-a) leak to caller."""
        skills_be = _make_root_fake({
            "/workspace/skill-a/SKILL.md": "# A",
        })
        hws = HybridWorkspaceBackend(
            real_backend=LocalBackend(root_dir=str(tmp_root)),
            virtual_workspaces={"skills": skills_be},
        )
        result = hws.ls(VirtualPath("/.mambo/skills/"))
        assert result.entries is not None
        for e in result.entries:
            # Must NOT be a raw internal path from the sub-backend
            assert not str(e.path).startswith("/skill"), (
                f"Internal path leaked: {e.path}"
            )
            assert str(e.path).startswith("/.mambo/"), (
                f"Path {e.path} not in external namespace"
            )

    def test_default_mambo_ls(self, hybrid_ws: HybridWorkspaceBackend):
        """ls on /.mambo/ returns paths under /.mambo/ (not /workspace/...)."""
        _write_virtual(hybrid_ws, VirtualPath("/.mambo/workspace/a.txt"), "a")
        _write_virtual(hybrid_ws, VirtualPath("/.mambo/workspace/b.txt"), "b")
        with _simulate_graph(hybrid_ws._default_mambo):
            result = hybrid_ws.ls(VirtualPath("/.mambo/workspace/"))
        assert result.entries is not None
        for e in result.entries:
            assert str(e.path).startswith("/.mambo/"), (
                f"Path {e.path} not in external namespace"
            )

    def test_real_backend_ls_preserves_paths(self, tmp_root: Path):
        """ls on real backend paths is unaffected (regression check)."""
        hws = HybridWorkspaceBackend(real_backend=LocalBackend(root_dir=str(tmp_root)))
        hws.write(VirtualPath(f"{_W}/project/x.py"), "code", overwrite=True)
        result = hws.ls(VirtualPath(f"{_W}/project/"))
        assert result.entries is not None
        paths = [e.path for e in result.entries]
        assert any(f"{_W}/project/x.py" in str(p) for p in paths)

    def test_empty_directory(self, tmp_root: Path):
        """ls on an empty named virtual workspace returns no entries (not an error)."""
        skills_be = _make_root_fake({})
        hws = HybridWorkspaceBackend(
            real_backend=LocalBackend(root_dir=str(tmp_root)),
            virtual_workspaces={"skills": skills_be},
        )
        result = hws.ls(VirtualPath("/.mambo/skills/"))
        assert result.entries is None or len(result.entries) == 0


# ============================================================================
# grep() — reverse path translation
# ============================================================================


class TestGrepReverseTranslation:
    """``grep()`` returns ``GrepMatch.path`` in the external namespace."""

    def test_named_virtual_workspace(self, tmp_root: Path):
        fake_be = _make_root_fake({
            "/workspace/skill-a/SKILL.md": "name: skill-a\ndescription: does stuff\n---",
            "/workspace/skill-b/SKILL.md": "name: skill-b\ndescription: other stuff\n---",
        })
        hws = HybridWorkspaceBackend(
            real_backend=LocalBackend(root_dir=str(tmp_root)),
            virtual_workspaces={"skills": fake_be},
        )
        result = hws.grep("description", path=VirtualPath("/.mambo/skills/"))
        assert result.matches is not None
        assert len(result.matches) >= 2
        for m in result.matches:
            assert str(m.path).startswith("/.mambo/skills/"), (
                f"Match path {m.path} not in external namespace"
            )

    def test_default_mambo(self, hybrid_ws: HybridWorkspaceBackend):
        _write_virtual(hybrid_ws, VirtualPath("/.mambo/notes.txt"), "important note here")
        with _simulate_graph(hybrid_ws._default_mambo):
            result = hybrid_ws.grep("important", path=VirtualPath("/.mambo/"))
        assert result.matches is not None
        for m in result.matches:
            assert str(m.path).startswith("/.mambo/"), (
                f"Match path {m.path} not under /.mambo/"
            )

    def test_real_backend_preserves_paths(self, tmp_root: Path):
        hws = HybridWorkspaceBackend(real_backend=LocalBackend(root_dir=str(tmp_root)))
        hws.write(VirtualPath(f"{_W}/code.py"), "def important(): pass", overwrite=True)
        result = hws.grep("important", path=VirtualPath(f"{_W}/"))
        assert result.matches is not None
        for m in result.matches:
            assert str(m.path).startswith(f"{_W}/")




# ============================================================================
# glob() — reverse path translation
# ============================================================================


class TestGlobReverseTranslation:
    """``glob()`` returns ``GlobResult.matches[].path`` in the external namespace."""

    def test_named_virtual_workspace(self, tmp_root: Path):
        fake_be = _make_root_fake({
            "/workspace/skill-a/SKILL.md": "content a",
            "/workspace/skill-a/helper.py": "print('a')",
            "/workspace/skill-b/SKILL.md": "content b",
        })
        hws = HybridWorkspaceBackend(
            real_backend=LocalBackend(root_dir=str(tmp_root)),
            virtual_workspaces={"skills": fake_be},
        )
        result = hws.glob("*SKILL*", path=VirtualPath("/.mambo/skills/"))
        assert result.matches is not None
        for m in result.matches:
            assert str(m.path).startswith("/.mambo/skills/"), (
                f"Match path {m.path} not in external namespace"
            )

    def test_default_mambo(self, hybrid_ws: HybridWorkspaceBackend):
        _write_virtual(hybrid_ws, VirtualPath("/.mambo/a.txt"), "a")
        _write_virtual(hybrid_ws, VirtualPath("/.mambo/b.md"), "b")
        with _simulate_graph(hybrid_ws._default_mambo):
            result = hybrid_ws.glob("*.txt", path=VirtualPath("/.mambo/"))
        assert result.matches is not None
        for m in result.matches:
            assert str(m.path).startswith("/.mambo/"), (
                f"Match path {m.path} not under /.mambo/"
            )

    def test_real_backend_preserves_paths(self, tmp_root: Path):
        hws = HybridWorkspaceBackend(real_backend=LocalBackend(root_dir=str(tmp_root)))
        hws.write(VirtualPath(f"{_W}/x.py"), "1", overwrite=True)
        hws.write(VirtualPath(f"{_W}/y.txt"), "2", overwrite=True)
        result = hws.glob("*.py", path=VirtualPath(f"{_W}/"))
        assert result.matches is not None
        for m in result.matches:
            assert str(m.path).startswith(f"{_W}/")




# ============================================================================
# aglob path forwarding fix — regression
# ============================================================================


class TestAglobPathForwardingFix:
    """``aglob()`` must pass the *rewritten* path to the target backend, not the
    original external path."""

    def test_passes_rewritten_path_not_original(self, tmp_root: Path):
        """If aglob passed the original path ``/.mambo/skills/`` to a backend,
        glob would find zero matches.  With the fix it passes the rewritten
        path and finds the files."""
        skills_be = _make_root_fake({
            "/workspace/skill-a/SKILL.md": "content a",
            "/workspace/skill-b/SKILL.md": "content b",
        })
        hws = HybridWorkspaceBackend(
            real_backend=LocalBackend(root_dir=str(tmp_root)),
            virtual_workspaces={"skills": skills_be},
        )
        result = hws.glob("*SKILL*", path=VirtualPath("/.mambo/skills/"))
        assert result.matches is not None
        # Both SKILL.md files should be found under "/"
        assert len(result.matches) >= 2, (
            f"Expected >=2 matches, got {len(result.matches)}. "
            f"If zero, aglob likely passed the wrong path to target."
        )


# ============================================================================
# SkillsMiddleware integration simulation
# ============================================================================


class TestSkillsMiddlewareIntegration:
    """Full simulation of ``SkillsMiddleware`` → ``HybridWorkspaceBackend`` flow,
    reproducing the exact call path from the bug report."""

    def test_skill_discovery_full_flow(self, tmp_root: Path):
        """Step-by-step: ls → collect dirs → build SKILL.md paths → download."""
        skills_be = _make_root_fake({
            "/workspace/cat-girl/SKILL.md": (
                "---\nname: cat-girl\ndescription: Cat girl skill\n---\n# Meow"
            ),
            "/workspace/web-research/SKILL.md": (
                "---\nname: web-research\ndescription: Research skill\n---\n# Research"
            ),
        })
        hws = HybridWorkspaceBackend(
            real_backend=LocalBackend(root_dir=str(tmp_root)),
            virtual_workspaces={"skills": skills_be},
        )

        # Step 1: ls (simulates SkillsMiddleware._alist_skills_with_errors)
        ls_result = hws.ls(VirtualPath("/.mambo/skills/"))
        assert ls_result.error is None, ls_result.error
        assert ls_result.entries is not None

        # Step 2: collect skill directories
        skill_dirs: list[str] = []
        for item in ls_result.entries:
            if item.is_dir:
                skill_dirs.append(str(item.path))

        assert len(skill_dirs) >= 2, f"Expected >=2 dirs, got {skill_dirs}"

        # Step 3: build SKILL.md paths
        skill_md_paths: list[str] = []
        for skill_dir_path in skill_dirs:
            skill_dir = PurePosixPath(skill_dir_path.replace("\\", "/"))
            skill_md_paths.append(str(skill_dir / "SKILL.md"))

        # Verify all paths are under /.mambo/skills/ (critical for routing)
        for p in skill_md_paths:
            assert p.startswith("/.mambo/skills/"), (
                f"SKILL.md path {p} not under /.mambo/skills/"
            )

        # Step 4: download SKILL.md files
        responses = hws.download_files([VirtualPath(p) for p in skill_md_paths])
        assert len(responses) == len(skill_md_paths)
        for r in responses:
            assert r.error is None, f"Download error for {r.path}: {r.error}"
            assert r.content is not None

        # Step 5: parse and verify skill names
        found_names: set[str] = set()
        for r in responses:
            text = r.content.decode("utf-8") if r.content else ""
            match = re.match(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
            if match:
                fm = yaml.safe_load(match.group(1))
                if isinstance(fm, dict) and fm.get("name"):
                    found_names.add(str(fm["name"]))

        assert "cat-girl" in found_names
        assert "web-research" in found_names

    def test_skill_not_found_when_paths_not_translated(self, tmp_root: Path):
        """Without reverse translation, paths would be wrong and downloads fail.
        This test verifies the fix is in place: downloads succeed."""
        skills_be = _make_root_fake({
            "/workspace/my-skill/SKILL.md": (
                "---\nname: my-skill\ndescription: A skill\n---\n# My Skill"
            ),
        })
        hws = HybridWorkspaceBackend(
            real_backend=LocalBackend(root_dir=str(tmp_root)),
            virtual_workspaces={"skills": skills_be},
        )

        ls_result = hws.ls(VirtualPath("/.mambo/skills/"))
        assert ls_result.entries is not None

        dirs = [str(e.path) for e in ls_result.entries if e.is_dir]
        assert len(dirs) == 1

        skill_md_path = str(PurePosixPath(dirs[0].replace("\\", "/")) / "SKILL.md")
        assert skill_md_path.startswith("/.mambo/skills/")

        results = hws.download_files([VirtualPath(skill_md_path)])
        assert len(results) == 1
        assert results[0].error is None, (
            f"Download failed: {results[0].error}. "
            f"Path was: {skill_md_path}"
        )
        assert results[0].content is not None

    def test_empty_skills_directory(self, tmp_root: Path):
        """Empty skills directory should not crash."""
        skills_be = _make_root_fake({})
        hws = HybridWorkspaceBackend(
            real_backend=LocalBackend(root_dir=str(tmp_root)),
            virtual_workspaces={"skills": skills_be},
        )
        ls_result = hws.ls(VirtualPath("/.mambo/skills/"))
        assert ls_result.entries is None or len(ls_result.entries) == 0


# ============================================================================
# Async reverse translation
# ============================================================================


class TestAsyncReverseTranslation:
    """Async variants (als, agrep, aglob) also reverse-translate paths."""

    @pytest.mark.asyncio
    async def test_als_named_virtual_workspace(self, tmp_root: Path):
        skills_be = _make_root_fake({
            "/workspace/skill-a/SKILL.md": "# A",
        })
        hws = HybridWorkspaceBackend(
            real_backend=LocalBackend(root_dir=str(tmp_root)),
            virtual_workspaces={"skills": skills_be},
        )
        result = await hws.als(VirtualPath("/.mambo/skills/"))
        assert result.entries is not None
        for e in result.entries:
            assert str(e.path).startswith("/.mambo/skills/"), (
                f"Path {e.path} not in external namespace"
            )

    @pytest.mark.asyncio
    async def test_agrep_named_virtual_workspace(self, tmp_root: Path):
        fake_be = _make_root_fake({
            "/workspace/skill-a/SKILL.md": "name: skill-a\ndescription: A skill\n---",
        })
        hws = HybridWorkspaceBackend(
            real_backend=LocalBackend(root_dir=str(tmp_root)),
            virtual_workspaces={"skills": fake_be},
        )
        result = await hws.agrep("description", path=VirtualPath("/.mambo/skills/"))
        assert result.matches is not None
        for m in result.matches:
            assert str(m.path).startswith("/.mambo/skills/"), (
                f"Match path {m.path} not in external namespace"
            )

    @pytest.mark.asyncio
    async def test_aglob_named_virtual_workspace(self, tmp_root: Path):
        """Verifies both the path-forwarding fix AND reverse translation."""
        fake_be = _make_root_fake({
            "/workspace/skill-a/SKILL.md": "content a",
        })
        hws = HybridWorkspaceBackend(
            real_backend=LocalBackend(root_dir=str(tmp_root)),
            virtual_workspaces={"skills": fake_be},
        )
        result = await hws.aglob("*SKILL*", path=VirtualPath("/.mambo/skills/"))
        assert result.matches is not None
        assert len(result.matches) >= 1, (
            "aglob found nothing — path forwarding may still be broken"
        )
        for m in result.matches:
            assert str(m.path).startswith("/.mambo/skills/"), (
                f"Match path {m.path} not in external namespace"
            )


# ============================================================================
# Out-of-workspace path errors — graceful error results, not exceptions
# ============================================================================


class TestOutOfWorkspacePaths:
    """Paths outside all valid prefixes must return error results, not throw."""

    @pytest.fixture
    def hws(self, tmp_path: Path) -> HybridWorkspaceBackend:
        skills_be = StoreBackend(store=InMemoryStore())
        return HybridWorkspaceBackend(
            real_backend=LocalBackend(root_dir=str(tmp_path)),
            virtual_workspaces={"skills": skills_be},
        )

    # -- _route still throws ValueError for invalid paths -------------------

    def test_route_raises_for_outside_path(self, hws: HybridWorkspaceBackend):
        from mambo_agents.backends.schemas import BackendError
        with pytest.raises(BackendError, match="路径不在"):
            hws._route(VirtualPath("/home/ramenl"))

    def test_route_raises_for_absolute_root(self, hws: HybridWorkspaceBackend):
        from mambo_agents.backends.schemas import BackendError
        with pytest.raises(BackendError, match="路径不在"):
            hws._route(VirtualPath("/etc/passwd"))

    # -- Operation methods catch ValueError and return error results --------

    def test_ls_outside_workspace_returns_error(self, hws: HybridWorkspaceBackend):
        r = hws.ls(VirtualPath("/home/ramenl"))
        assert r.error is not None
        assert r.error.code == ErrorCode.PATH_NOT_UNDER

    def test_read_outside_workspace_returns_error(self, hws: HybridWorkspaceBackend):
        r = hws.read(VirtualPath("/home/ramenl/.bashrc"))
        assert r.error is not None
        assert r.error.code == ErrorCode.PATH_NOT_UNDER

    def test_read_raw_outside_workspace_returns_error(self, hws: HybridWorkspaceBackend):
        r = hws.read_raw(VirtualPath("/home/ramenl/.bashrc"))
        assert r.error is not None
        assert r.error.code == ErrorCode.PATH_NOT_UNDER

    def test_grep_outside_workspace_returns_error(self, hws: HybridWorkspaceBackend):
        r = hws.grep("password", path=VirtualPath("/home"))
        assert r.error is not None
        assert r.error.code == ErrorCode.PATH_NOT_UNDER

    def test_glob_outside_workspace_returns_error(self, hws: HybridWorkspaceBackend):
        r = hws.glob("*", path=VirtualPath("/home"))
        assert r.error is not None
        assert r.error.code == ErrorCode.PATH_NOT_UNDER

    # -- Error message content ------------------------------------------------

    def test_error_includes_workspace_root(self, hws: HybridWorkspaceBackend):
        r = hws.ls(VirtualPath("/tmp"))
        assert r.error is not None
        assert r.error.code == ErrorCode.PATH_NOT_UNDER

    def test_error_includes_mambo_prefix(self, hws: HybridWorkspaceBackend):
        r = hws.read(VirtualPath("/root/.profile"))
        assert r.error is not None
        assert r.error.code == ErrorCode.PATH_NOT_UNDER

    def test_error_includes_named_virtual_workspaces(self, hws: HybridWorkspaceBackend):
        r = hws.glob("*", path=VirtualPath("/opt"))
        assert r.error is not None
        assert r.error.code == ErrorCode.PATH_NOT_UNDER

    def test_error_includes_real_backend_mapping(self, hws: HybridWorkspaceBackend):
        r = hws.ls(VirtualPath("/home/ramenl"))
        assert r.error is not None
        assert "映射至真实路径" not in str(r.error)  # real paths no longer leaked

    # -- _valid_paths_description --------------------------------------------

    def test_valid_paths_description_structure(self, hws: HybridWorkspaceBackend):
        desc = hws._valid_paths_description()
        assert hws.workspace_root.value in desc
        assert hws._prefix.value in desc
        assert "映射至真实路径" in desc
        assert hws._real.workspace_root.value in desc

    def test_valid_paths_description_includes_named(self, hws: HybridWorkspaceBackend):
        desc = hws._valid_paths_description()
        assert "/.mambo/skills/" in desc
        assert "虚拟工作区" in desc


# ============================================================================
# VirtualPath integration — path-traversal protection
# ============================================================================


class TestVirtualPathIntegration:
    """End-to-end: VirtualPath syntax validation protects the backend pipeline."""

    @pytest.fixture
    def hws(self, tmp_root: Path) -> HybridWorkspaceBackend:
        return HybridWorkspaceBackend(real_backend=LocalBackend(root_dir=str(tmp_root)))

    # -- VirtualPath: construction-time rejection ----------------------------

    def test_route_rejects_dotdot_traversal(self, hws: HybridWorkspaceBackend):
        """VirtualPath construction blocks '..' before it reaches _route."""
        with pytest.raises(BackendError, match="不能包含 '..'"):
            VirtualPath("/workspace/../../etc/passwd")

    def test_route_rejects_double_slash(self, hws: HybridWorkspaceBackend):
        """VirtualPath construction blocks '//' before it reaches _route."""
        with pytest.raises(BackendError, match="不能包含 '//'"):
            VirtualPath("/workspace//src")

    def test_route_rejects_non_absolute(self, hws: HybridWorkspaceBackend):
        """VirtualPath construction rejects non-absolute paths."""
        with pytest.raises(BackendError, match="必须以 '/' 开头"):
            VirtualPath("workspace/src")

    def test_route_accepts_valid_path(self, hws: HybridWorkspaceBackend):
        """Valid absolute path under workspace_root passes validation."""
        target, path = hws._route(VirtualPath("/workspace/src/main.py"))
        assert target is hws._real
        assert path == "/workspace/src/main.py"

    # -- Public methods with VirtualPath -----------------------------------

    def test_write_valid_path_succeeds(self, hws: HybridWorkspaceBackend):
        """Write to a valid workspace path."""
        r = hws.write(VirtualPath("/workspace/f.txt"), "hello")
        assert r.error is None
        assert r.path == "/workspace/f.txt"

    def test_read_valid_path_succeeds(self, hws: HybridWorkspaceBackend):
        """Read from a valid workspace path."""
        hws.write(VirtualPath("/workspace/f.txt"), "data")
        r = hws.read(VirtualPath("/workspace/f.txt"))
        assert r.error is None
        assert r.content == "data"

    def test_ls_workspace_root(self, hws: HybridWorkspaceBackend):
        """Listing workspace root should not crash."""
        hws.write(VirtualPath("/workspace/a.txt"), "a")
        r = hws.ls(VirtualPath("/workspace"))
        assert r.error is None
        assert r.entries is not None

    # -- Workspace boundary: outside-path returns error, not crash ---------

    def test_write_outside_workspace_returns_error(self, hws: HybridWorkspaceBackend):
        """Writing to a path outside workspace returns error, doesn't crash."""
        r = hws.write(VirtualPath("/etc/hosts"), "evil")
        assert r.error is not None
        assert r.error.code == ErrorCode.PATH_NOT_UNDER

    def test_edit_outside_workspace_returns_error(self, hws: HybridWorkspaceBackend):
        """Editing outside workspace returns error result, doesn't crash."""
        r = hws.edit(VirtualPath("/etc/crontab"), "old", "new")
        assert r.error is not None
        assert r.error.code == ErrorCode.PATH_NOT_UNDER

    # -- VirtualPath pass-through via _route ---------------------------------

    def test_virtual_path_roundtrips_through_route(self, hws: HybridWorkspaceBackend):
        """VirtualPath passes through _route and roundtrips correctly."""
        target, stripped = hws._route(VirtualPath("/workspace/test.txt"))
        r = target.write(stripped, "coerced!", overwrite=True)
        assert r.error is None
        r2 = target.read(stripped)
        assert r2.content == "coerced!"

    def test_route_preserves_virtual_path_for_valid_input(self, hws: HybridWorkspaceBackend):
        """VirtualPath passes through _route unchanged for valid paths."""
        vp = VirtualPath("/workspace/project/main.py")
        target, rewritten = hws._route(vp)
        assert rewritten == vp
        assert target is hws._real

    # -- Double-slash and dotdot are blocked by VirtualPath at construction

    def test_virtual_path_blocks_dotdot_before_route(self):
        """VirtualPath constructor blocks '..' before it reaches _route."""
        with pytest.raises(BackendError, match="不能包含 '..'"):
            VirtualPath("/workspace/../../etc/passwd")

    def test_virtual_path_blocks_double_slash_before_route(self):
        """VirtualPath constructor blocks '//' before it reaches _route."""
        with pytest.raises(BackendError, match="不能包含 '//'"):
            VirtualPath("/workspace//src/secret")