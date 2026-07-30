"""Dedicated tests for StoreBackend — binary handling, concurrency, upload/download,
per-thread isolation, and graph-outside operations.
"""

import base64
import contextlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest
from langgraph.store.memory import InMemoryStore

from mambo_agents.backends.store import StoreBackend
from mambo_agents.backends.schemas import ErrorCode, VirtualPath


# ============================================================================
# Test helpers
# ============================================================================


def _binary_content(size: int = 256) -> bytes:
    """Non-UTF-8 bytes for binary file testing."""
    return bytes(range(256)) * ((size + 255) // 256)


def _make_backend(**kwargs) -> StoreBackend:
    """Create a StoreBackend with an InMemoryStore for testing."""
    store = kwargs.pop("store", InMemoryStore())
    return StoreBackend(store=store, **kwargs)


@contextlib.contextmanager
def _simulate_graph(backend: StoreBackend, thread_id: str = "test"):
    """Compat shim — overrides thread_id resolution for per-thread tests.

    Without this, ``_resolve_thread_id()`` returns ``"__default__"``.
    This patches it so isolation tests can target specific threads.
    """
    orig = backend._resolve_thread_id
    backend._resolve_thread_id = lambda: thread_id
    try:
        yield
    finally:
        backend._resolve_thread_id = orig


def _files_snapshot(backend: StoreBackend, thread_id: str = "test") -> dict:
    """Read all files for a thread directly from the underlying store."""
    store = backend._store
    namespace = (thread_id, "mambo_fs")
    items = backend._search_store_paginated(store, namespace)
    return {item.key: item.value for item in items}


# ============================================================================
# Initialization
# ============================================================================


class TestStoreBackendInit:
    def test_default_construction(self):
        backend = _make_backend()
        files = _files_snapshot(backend)
        assert files == {}

    def test_initial_files_text(self):
        backend = _make_backend(initial_files={"/hello.txt": "Hello World"})
        with _simulate_graph(backend):
            files = backend._get_all_files("test")
        assert files["/hello.txt"]["content"] == "Hello World"
        assert files["/hello.txt"]["encoding"] == "utf-8"

    def test_initial_files_binary(self):
        backend = _make_backend(initial_files={"/img.png": "raw"})
        with _simulate_graph(backend):
            assert backend._get_all_files("test")["/img.png"]["encoding"] == "base64"

    def test_initial_files_mixed(self):
        backend = _make_backend(
            initial_files={
                "/a.txt": "alpha",
                "/sub/b.py": "print(1)",
                "/img.jpg": "jpeg-data",
            }
        )
        with _simulate_graph(backend):
            files = backend._get_all_files("test")
        assert files["/a.txt"]["encoding"] == "utf-8"
        assert files["/img.jpg"]["encoding"] == "base64"

    def test_initial_files_not_re_injected_after_first_access(self):
        store = InMemoryStore()
        backend = StoreBackend(store=store, initial_files={"/init.txt": "template"})
        with _simulate_graph(backend):
            files1 = backend._get_all_files("test")
            assert files1["/init.txt"]["content"] == "template"
            backend.write(VirtualPath("/init.txt"), "modified-by-agent", overwrite=True)
            files2 = backend._get_all_files("test")
            assert files2["/init.txt"]["content"] == "modified-by-agent"

    def test_initial_files_does_not_overwrite_existing(self):
        store = InMemoryStore()
        backend = StoreBackend(store=store, initial_files={"/existing.txt": "stale-init"})
        with _simulate_graph(backend):
            backend._put_file("test", "/existing.txt", "from-previous-run", "utf-8")
            backend._initialized_threads.discard("test")
            files = backend._get_all_files("test")
        assert files["/existing.txt"]["content"] == "from-previous-run"


# ============================================================================
# Write
# ============================================================================


class TestStoreBackendWrite:
    def test_write_new_file(self):
        backend = _make_backend()
        with _simulate_graph(backend):
            r = backend.write(VirtualPath("/test.txt"), "hello")
            assert r.error is None
            assert "hello" in (backend.read(VirtualPath("/test.txt")).content or "")

    def test_write_fails_if_exists(self):
        backend = _make_backend()
        with _simulate_graph(backend):
            backend.write(VirtualPath("/dup.txt"), "original")
            r = backend.write(VirtualPath("/dup.txt"), "modified")
        assert r.error is not None
        assert "已存在" in str(r.error)

    def test_write_overwrite(self):
        backend = _make_backend()
        with _simulate_graph(backend):
            backend.write(VirtualPath("/x.txt"), "v1")
            r = backend.write(VirtualPath("/x.txt"), "v2", overwrite=True)
            assert r.error is None
            assert "v2" in (backend.read(VirtualPath("/x.txt")).content or "")

    def test_binary_extension_gets_base64(self):
        backend = _make_backend()
        with _simulate_graph(backend):
            r = backend.write(VirtualPath("/img.png"), "data")
        assert r.error is not None
        assert "非文本" in str(r.error)


# ============================================================================
# Edit
# ============================================================================


class TestStoreBackendEdit:
    def test_edit_replaces(self):
        backend = _make_backend()
        with _simulate_graph(backend):
            backend.write(VirtualPath("/f.py"), "x = 1\ny = 2")
            r = backend.edit(VirtualPath("/f.py"), "x = 1", "x = 100")
        assert r.error is None
        assert r.occurrences == 1
        assert "x = 100" in _files_snapshot(backend)["/f.py"]["content"]

    def test_edit_replace_all(self):
        backend = _make_backend()
        with _simulate_graph(backend):
            backend.write(VirtualPath("/f.txt"), "A-B-A-B-A")
            r = backend.edit(VirtualPath("/f.txt"), "A", "X", replace_all=True)
        assert r.error is None
        assert r.occurrences == 3
        assert _files_snapshot(backend)["/f.txt"]["content"] == "X-B-X-B-X"

    def test_edit_multiple_without_replace_all(self):
        backend = _make_backend()
        with _simulate_graph(backend):
            backend.write(VirtualPath("/f.txt"), "A-B-A")
            r = backend.edit(VirtualPath("/f.txt"), "A", "X")
        assert "处" in str(r.error)

    def test_edit_not_found(self):
        backend = _make_backend()
        with _simulate_graph(backend):
            backend.write(VirtualPath("/f.txt"), "hello")
            r = backend.edit(VirtualPath("/f.txt"), "gone", "x")
        assert "未找到" in str(r.error)

    def test_edit_file_not_found(self):
        backend = _make_backend()
        with _simulate_graph(backend):
            r = backend.edit(VirtualPath("/no.txt"), "x", "y")
        assert "不存在" in str(r.error)

    def test_edit_binary_blocked(self):
        backend = _make_backend(initial_files={"/img.png": "AAAA"})
        with _simulate_graph(backend):
            r = backend.edit(VirtualPath("/img.png"), "AAA", "BBB")
        assert r.error is not None
        assert "非文本" in str(r.error)

    def test_edit_binary_from_initial_files_blocked(self):
        backend = _make_backend(initial_files={"/img.png": "raw"})
        with _simulate_graph(backend):
            r = backend.edit(VirtualPath("/img.png"), "raw", "new")
        assert r.error is not None
        assert "非文本" in str(r.error)

    def test_edit_trailing_newline_hint(self):
        backend = _make_backend()
        with _simulate_graph(backend):
            backend.write(VirtualPath("/f.txt"), "def foo(): pass")
            r = backend.edit(VirtualPath("/f.txt"), "def foo(): pass\n", "def bar():\n")
        assert r.error is not None
        assert "换行" in str(r.error).lower()


# ============================================================================
# Read
# ============================================================================


class TestStoreBackendRead:
    def test_read_line_numbers(self):
        backend = _make_backend()
        with _simulate_graph(backend):
            backend.write(VirtualPath("/f.py"), "a\nb\nc")
            r = backend.read(VirtualPath("/f.py"))
        assert r.total_lines == 3
        assert "a\nb\nc" in (r.content or "")

    def test_read_offset_limit(self):
        backend = _make_backend()
        with _simulate_graph(backend):
            backend.write(VirtualPath("/f.txt"), "\n".join(f"L{i}" for i in range(10)))
            r = backend.read(VirtualPath("/f.txt"), offset=2, limit=3)
        assert r.error is None
        assert "L2" in (r.content or "")
        assert "L4" in (r.content or "")

    def test_read_binary_base64(self):
        content = base64.b64encode(b"\x89PNG\r\n\x1a\n\x00").decode("ascii")
        backend = _make_backend(initial_files={"/img.png": content})
        with _simulate_graph(backend):
            r = backend.read(VirtualPath("/img.png"))
        assert r.encoding == "base64"
        assert r.file_type == "image"
        assert r.content == content

    def test_read_not_found(self):
        backend = _make_backend()
        with _simulate_graph(backend):
            r = backend.read(VirtualPath("/no.txt"))
        assert "不存在" in str(r.error)


# ============================================================================
# Ls
# ============================================================================


class TestStoreBackendLs:
    def test_ls_flat(self):
        backend = _make_backend()
        with _simulate_graph(backend):
            backend.write(VirtualPath("/workspace/a.py"), "1")
            backend.write(VirtualPath("/workspace/b.py"), "2")
            result = backend.ls(VirtualPath("/workspace"))
        paths = [fi.path for fi in result.entries]
        assert "/workspace/a.py" in paths
        assert "/workspace/b.py" in paths

    def test_ls_with_subdir(self):
        backend = _make_backend()
        with _simulate_graph(backend):
            backend.write(VirtualPath("/workspace/sub/file.txt"), "hello")
            result = backend.ls(VirtualPath("/workspace"))
        dirs = [fi for fi in result.entries if fi.is_dir]
        assert any(d.path == "/workspace/sub" for d in dirs)

    def test_ls_empty(self):
        backend = _make_backend()
        with _simulate_graph(backend):
            result = backend.ls(VirtualPath("/workspace"))
        assert result.error is None
        assert result.entries == []

    def test_ls_nonexistent_subdir(self):
        backend = _make_backend()
        with _simulate_graph(backend):
            result = backend.ls(VirtualPath("/workspace/dir"))
        assert result.error is not None
        assert "不存在" in str(result.error)


# ============================================================================
# Grep
# ============================================================================


class TestStoreBackendGrep:
    def test_grep_finds(self):
        backend = _make_backend()
        with _simulate_graph(backend):
            backend.write(VirtualPath("/workspace/a.py"), "def foo(): pass")
            backend.write(VirtualPath("/workspace/b.py"), "def bar(): pass")
            r = backend.grep("foo", path=VirtualPath("/workspace"))
        assert any("foo" in m.text for m in (r.matches or []))

    def test_grep_skips_binary(self):
        backend = _make_backend()
        with _simulate_graph(backend):
            backend.write(VirtualPath("/workspace/img.png"), "foo-visible")
            backend.write(VirtualPath("/workspace/txt.txt"), "foo-hidden")
            r = backend.grep("foo", path=VirtualPath("/workspace"))
        paths = [m.path for m in (r.matches or [])]
        assert "/workspace/txt.txt" in paths
        assert "/workspace/img.png" not in paths

    def test_grep_no_match(self):
        backend = _make_backend()
        with _simulate_graph(backend):
            backend.write(VirtualPath("/workspace/f.txt"), "hello")
            r = backend.grep("zzz", path=VirtualPath("/workspace"))
        assert len(r.matches or []) == 0


# ============================================================================
# Glob
# ============================================================================


class TestStoreBackendGlob:
    def test_glob_finds(self):
        backend = _make_backend()
        with _simulate_graph(backend):
            backend.write(VirtualPath("/workspace/src/main.py"), "1")
            backend.write(VirtualPath("/workspace/src/util.py"), "2")
            backend.write(VirtualPath("/workspace/README.md"), "3")
            r = backend.glob("**/*.py", path=VirtualPath("/workspace/src"))
        paths = [fi.path for fi in (r.matches or [])]
        assert len(paths) == 2
        assert "/workspace/src/main.py" in paths

    def test_glob_no_match(self):
        backend = _make_backend()
        with _simulate_graph(backend):
            r = backend.glob("*.py", path=VirtualPath("/workspace"))
        assert len(r.matches or []) == 0

    def test_glob_size_is_content_length(self):
        backend = _make_backend()
        with _simulate_graph(backend):
            backend.write(VirtualPath("/workspace/f.txt"), "hello")
            r = backend.glob("f.txt", path=VirtualPath("/workspace"))
        assert r.matches[0].size == 5


class TestStoreBackendGlobPatterns:
    """回归测试：glob pattern 应匹配 *相对于 path* 的路径，且支持
    ``?`` / ``[...]`` / 路径分隔符 / 精确文件名等 pathlib 兼容语义。"""

    @staticmethod
    def _setup():
        backend = _make_backend()
        ctx = _simulate_graph(backend)
        ctx.__enter__()
        backend.write(VirtualPath("/workspace/abc.txt"), "1")
        backend.write(VirtualPath("/workspace/test.txt"), "2")
        backend.write(VirtualPath("/workspace/中文无后缀"), "3")
        backend.write(VirtualPath("/workspace/sub/hello.md"), "4")
        backend.write(VirtualPath("/workspace/aXc.txt"), "5")
        return backend, ctx

    def _names(self, r):
        return sorted(fi.path.normalized.rpartition("/")[2] for fi in (r.matches or []))

    def test_exact_filename(self):
        backend, ctx = self._setup()
        try:
            assert self._names(backend.glob("abc.txt", path=VirtualPath("/workspace"))) == ["abc.txt"]
            assert self._names(backend.glob("test.txt", path=VirtualPath("/workspace"))) == ["test.txt"]
            assert self._names(backend.glob("中文无后缀", path=VirtualPath("/workspace"))) == ["中文无后缀"]
        finally:
            ctx.__exit__(None, None, None)

    def test_question_mark_single_char(self):
        backend, ctx = self._setup()
        try:
            assert self._names(backend.glob("???.*", path=VirtualPath("/workspace"))) == ["aXc.txt", "abc.txt"]
        finally:
            ctx.__exit__(None, None, None)

    def test_question_mark_in_middle(self):
        backend, ctx = self._setup()
        try:
            assert sorted(self._names(backend.glob("a?c.txt", path=VirtualPath("/workspace")))) == ["aXc.txt", "abc.txt"]
        finally:
            ctx.__exit__(None, None, None)

    def test_character_class(self):
        backend, ctx = self._setup()
        try:
            assert sorted(self._names(backend.glob("[aA]*", path=VirtualPath("/workspace")))) == ["aXc.txt", "abc.txt"]
        finally:
            ctx.__exit__(None, None, None)

    def test_subdir_pattern(self):
        backend, ctx = self._setup()
        try:
            assert self._names(backend.glob("sub/*", path=VirtualPath("/workspace"))) == ["hello.md"]
        finally:
            ctx.__exit__(None, None, None)

    def test_subdir_exact_path(self):
        backend, ctx = self._setup()
        try:
            assert self._names(backend.glob("sub/hello.md", path=VirtualPath("/workspace"))) == ["hello.md"]
        finally:
            ctx.__exit__(None, None, None)

    def test_star_does_not_cross_separator(self):
        """``*`` 不应跨 ``/`` 匹配（pathlib 语义）。"""
        backend, ctx = self._setup()
        try:
            assert sorted(self._names(backend.glob("*.txt", path=VirtualPath("/workspace")))) == ["aXc.txt", "abc.txt", "test.txt"]
            assert self._names(backend.glob("*.md", path=VirtualPath("/workspace"))) == []
        finally:
            ctx.__exit__(None, None, None)


# ============================================================================
# Upload / Download — thread_id locked at construction
# ============================================================================


class TestStoreBackendUploadDownload:

    def test_upload_text(self):
        backend = _make_backend(thread_id="t1")
        results = backend.upload_files([(VirtualPath("/workspace/upload.txt"), b"hello world")])
        assert results[0].error is None
        files = _files_snapshot(backend, "t1")
        assert files["/workspace/upload.txt"]["content"] == "hello world"
        assert files["/workspace/upload.txt"]["encoding"] == "utf-8"

    def test_upload_binary(self):
        backend = _make_backend(thread_id="t1")
        raw = _binary_content(100)
        results = backend.upload_files([(VirtualPath("/workspace/data.bin"), raw)])
        assert results[0].error is None
        fd = _files_snapshot(backend, "t1")["/workspace/data.bin"]
        assert fd["encoding"] == "base64"
        decoded = base64.b64decode(fd["content"])
        assert decoded == raw

    def test_download_text(self):
        backend = _make_backend(thread_id="t1")
        backend.upload_files([(VirtualPath("/workspace/dl.txt"), b"download me")])
        results = backend.download_files([VirtualPath("/workspace/dl.txt")])
        assert results[0].content == b"download me"

    def test_download_binary(self):
        backend = _make_backend(thread_id="t1")
        raw = _binary_content(100)
        backend.upload_files([(VirtualPath("/workspace/bin.bin"), raw)])
        results = backend.download_files([VirtualPath("/workspace/bin.bin")])
        assert results[0].content == raw

    def test_download_not_found(self):
        backend = _make_backend(thread_id="t1")
        results = backend.download_files([VirtualPath("/workspace/no.txt")])
        assert results[0].error.code == ErrorCode.NOT_FOUND

    def test_upload_multiple(self):
        backend = _make_backend(thread_id="t1")
        results = backend.upload_files([
            (VirtualPath("/workspace/a.txt"), b"alpha"),
            (VirtualPath("/workspace/b.txt"), b"beta"),
        ])
        assert all(r.error is None for r in results)
        files = _files_snapshot(backend, "t1")
        assert set(files.keys()) == {"/workspace/a.txt", "/workspace/b.txt"}

    def test_default_thread_id_fallback(self):
        """Without explicit thread_id, uses '__default__'."""
        store = InMemoryStore()
        backend = StoreBackend(store=store)
        backend.upload_files([(VirtualPath("/workspace/f.txt"), b"data")])
        snap = _files_snapshot(backend, "__default__")
        assert snap["/workspace/f.txt"]["content"] == "data"


# ============================================================================
# Per-thread isolation
# ============================================================================


class TestStoreBackendIsolation:
    def test_separate_threads_isolated(self):
        store = InMemoryStore()
        backend = StoreBackend(store=store)
        with _simulate_graph(backend, thread_id="A"):
            backend.write(VirtualPath("/workspace/secret.txt"), "A-only")
        with _simulate_graph(backend, thread_id="B"):
            files = backend._get_all_files("B")
        assert "/workspace/secret.txt" not in files

    def test_separate_threads_independent_snapshots(self):
        store = InMemoryStore()
        backend = StoreBackend(store=store)
        with _simulate_graph(backend, thread_id="A"):
            backend.write(VirtualPath("/workspace/a.txt"), "a")
        with _simulate_graph(backend, thread_id="B"):
            backend.write(VirtualPath("/workspace/b.txt"), "b")
        assert "/workspace/a.txt" in _files_snapshot(backend, "A")
        assert "/workspace/b.txt" in _files_snapshot(backend, "B")
        assert "/workspace/a.txt" not in _files_snapshot(backend, "B")
        assert "/workspace/b.txt" not in _files_snapshot(backend, "A")

    def test_initial_files_per_new_thread(self):
        store = InMemoryStore()
        backend = StoreBackend(store=store, initial_files={"/workspace/init.txt": "template"})
        with _simulate_graph(backend, thread_id="t1"):
            assert "/workspace/init.txt" in backend._get_all_files("t1")
        with _simulate_graph(backend, thread_id="t2"):
            assert "/workspace/init.txt" in backend._get_all_files("t2")

    def test_upload_per_thread_isolated(self):
        """Uploads to different threads don't leak."""
        store = InMemoryStore()
        be_a = StoreBackend(store=store, thread_id="A")
        be_b = StoreBackend(store=store, thread_id="B")
        be_a.upload_files([(VirtualPath("/workspace/a.txt"), b"a")])
        be_b.upload_files([(VirtualPath("/workspace/b.txt"), b"b")])
        assert "/workspace/a.txt" in _files_snapshot(be_a, "A")
        assert "/workspace/b.txt" in _files_snapshot(be_b, "B")
        assert "/workspace/b.txt" not in _files_snapshot(be_a, "A")


# ============================================================================
# Concurrency
# ============================================================================


class TestStoreBackendConcurrency:
    def test_concurrent_writes_no_race(self):
        num_writers = 20
        store = InMemoryStore()
        backend = StoreBackend(store=store)

        def writer(i: int):
            with _simulate_graph(backend, thread_id=f"t{i}"):
                backend.write(VirtualPath(f"/workspace/file_{i}.txt"), f"content_{i}")

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(writer, i) for i in range(num_writers)]
            for f in as_completed(futures):
                f.result()

        for i in range(num_writers):
            snap = _files_snapshot(backend, f"t{i}")
            assert f"/workspace/file_{i}.txt" in snap
            assert snap[f"/workspace/file_{i}.txt"]["content"] == f"content_{i}"

    def test_concurrent_read_write_no_race(self):
        store = InMemoryStore()
        backend = StoreBackend(store=store)
        with _simulate_graph(backend):
            backend.write(VirtualPath("/workspace/shared.txt"), "initial\n" * 10)

        def reader():
            for _ in range(100):
                with _simulate_graph(backend):
                    r = backend.read(VirtualPath("/workspace/shared.txt"))
                    if r.content:
                        assert "initial" in r.content

        def writer():
            for i in range(100):
                with _simulate_graph(backend):
                    backend.write(VirtualPath(f"/workspace/w_{i}.txt"), f"hello_{i}", overwrite=True)

        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = []
            for _ in range(3):
                futures.append(pool.submit(reader))
            for _ in range(3):
                futures.append(pool.submit(writer))
            for f in as_completed(futures):
                f.result()

    def test_concurrent_uploads_no_race(self):
        num = 30
        store = InMemoryStore()
        backend = StoreBackend(store=store, thread_id="shared")

        def uploader(start: int):
            batch = [
                (VirtualPath(f"/workspace/u_{j}.txt"), f"val_{j}".encode())
                for j in range(start, start + 10)
            ]
            return backend.upload_files(batch)

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(uploader, i * 10) for i in range(3)]
            for f in as_completed(futures):
                f.result()

        snap = _files_snapshot(backend, "shared")
        assert len(snap) == num
        for i in range(num):
            assert snap[f"/workspace/u_{i}.txt"]["content"] == f"val_{i}"


# ============================================================================
# Tools
# ============================================================================


class TestStoreBackendTools:
    def test_extra_tools_only(self):
        backend = _make_backend()
        names = {t.name for t in backend.tools}
        assert "tree" in names
        assert "ls" not in names
        assert "read" not in names

    def test_tree_is_str(self):
        backend = _make_backend()
        with _simulate_graph(backend):
            backend.write(VirtualPath("/workspace/a.txt"), "a")
            backend.write(VirtualPath("/workspace/sub/b.txt"), "b")
            result = backend.tree(VirtualPath("/workspace"), depth=2)
        assert isinstance(result, str)
        assert len(result) > 0


# ============================================================================
# StoreBackend outside graph
# ============================================================================


class TestStoreBackendStoreInjection:
    def test_no_store_lazy_creates_inmemory(self):
        backend = StoreBackend()
        r = backend.write(VirtualPath("/workspace/f.txt"), "data")
        assert r.error is None

    def test_with_thread_id_outside_graph_works(self):
        """With explicit thread_id, ops target that session."""
        store = InMemoryStore()
        backend = StoreBackend(store=store, thread_id="my-session")
        backend.upload_files([(VirtualPath("/workspace/f.txt"), b"hello")])
        result = backend.download_files([VirtualPath("/workspace/f.txt")])
        assert result[0].content == b"hello"

    def test_write_with_store_outside_graph(self):
        """Core ops work with injected store, fallback to default thread."""
        store = InMemoryStore()
        backend = StoreBackend(store=store)
        r = backend.write(VirtualPath("/workspace/f.txt"), "data")
        assert r.error is None
        snap = _files_snapshot(backend, "__default__")
        assert snap["/workspace/f.txt"]["content"] == "data"
