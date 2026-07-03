"""Dedicated tests for StateBackend — binary handling, concurrency, upload/download,
per-thread isolation, and graph-outside operations.
"""

import base64
import contextlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from mambo_agents.backends.state import StateBackend
from mambo_agents.backends.schemas import ErrorCode, VirtualPath


# ============================================================================
# Test helpers
# ============================================================================


def _binary_content(size: int = 256) -> bytes:
    """Non-UTF-8 bytes for binary file testing."""
    return bytes(range(256)) * ((size + 255) // 256)


@contextlib.contextmanager
def _simulate_graph(backend: StateBackend, thread_id: str = "test"):
    """Simulate LangGraph context for unit-testing core file operations.

    Patches ``_in_graph_context`` → ``True``, and redirects ``_read_files`` /
    ``_send_files_update`` to use ``_snapshots`` as a simulated Pregel channel.
    This exercises the business logic of write/read/edit/ls/grep/glob without
    requiring a running LangGraph Pregel loop.

    Also patches ``_resolve_thread_id`` and ``_get_config`` so that
    ``upload_files`` / ``download_files`` can use the graph-in path
    (``thread_id=None`` auto-resolved).

    **Thread-safe**: uses a refcount on ``backend._simulate_graph_nest``
    guarded by ``backend._lock`` so that concurrent readers/writers on the
    same backend instance don't race on the attribute swaps.
    """

    # ---- Mock implementations ----

    def _mock_in_graph() -> bool:
        return True

    def _mock_resolve_thread_id(_tid: str | None) -> str:
        """Always return the simulated thread_id regardless of argument."""
        return thread_id

    def _mock_get_config():
        """Return a minimal config that satisfies _resolve_thread_id."""
        return {"configurable": {"thread_id": thread_id}}

    def _mock_read() -> dict:
        """Simulates the real _read_files() pipeline:

        1. Read current snapshot (simulating ``channel_files``).
        2. If first access, inject _initial_files for paths not yet present.
        3. Flush pending uploads (always honoured).
        4. Full-replace snapshot and return shallow copy.
        """
        with backend._lock:
            snap = backend._snapshots
            # (1) Read current state (simulating channel read)
            channel_files = dict(snap.get(thread_id, {}))

            # (2) Inject initial_files for genuinely new paths (matches real
            #     _read_files: only paths NOT already in the checkpoint)
            if thread_id not in snap:
                for path, fd in backend._initial_files.items():
                    if path not in channel_files:
                        channel_files[path] = fd

            # (3) Flush pending uploads (always honoured — pop so once-only)
            if thread_id in backend._pending_uploads:
                channel_files.update(backend._pending_uploads.pop(thread_id))

            # (4) Full-replace snapshot
            snap[thread_id] = dict(channel_files)
            return dict(channel_files)

    def _mock_send(update: dict) -> None:
        """Simulates _send_files_update: writes to snapshot."""
        with backend._lock:
            if thread_id not in backend._snapshots:
                backend._snapshots[thread_id] = {}
            backend._snapshots[thread_id].update(update)

    # Thread-safe refcount so concurrent calls on the same backend instance
    # don't race on attribute swaps (unlike ``patch.object`` which uses
    # non-reentrant ``setattr``/``delattr``).
    with backend._lock:
        if not hasattr(backend, '_simulate_graph_nest'):
            backend._simulate_graph_nest = 0
        backend._simulate_graph_nest += 1

        if backend._simulate_graph_nest == 1:
            # ---- First entry: save originals & apply mocks ----
            backend._orig__in_graph_context = backend._in_graph_context
            backend._orig__resolve_thread_id = backend._resolve_thread_id
            backend._orig__get_config = backend._get_config
            backend._orig__read_files = backend._read_files
            backend._orig__send_files_update = backend._send_files_update

            backend._in_graph_context = _mock_in_graph
            backend._resolve_thread_id = _mock_resolve_thread_id
            backend._get_config = _mock_get_config
            backend._read_files = _mock_read
            backend._send_files_update = _mock_send

    try:
        yield
    finally:
        with backend._lock:
            backend._simulate_graph_nest -= 1
            if backend._simulate_graph_nest == 0:
                # ---- Last exit: restore originals ----
                backend._in_graph_context = backend._orig__in_graph_context
                backend._resolve_thread_id = backend._orig__resolve_thread_id
                backend._get_config = backend._orig__get_config
                backend._read_files = backend._orig__read_files
                backend._send_files_update = backend._orig__send_files_update
                del backend._orig__in_graph_context
                del backend._orig__resolve_thread_id
                del backend._orig__get_config
                del backend._orig__read_files
                del backend._orig__send_files_update
                del backend._simulate_graph_nest


def _files_snapshot(backend: StateBackend, thread_id: str = "test") -> dict:
    """Read the per-thread snapshot directly (bypassing graph checks)."""
    with backend._lock:
        return dict(backend._snapshots.get(thread_id, {}))


# ============================================================================
# Initialization
# ============================================================================


class TestStateBackendInit:
    def test_default_construction(self):
        backend = StateBackend()
        with _simulate_graph(backend):
            files = backend._read_files()
        assert files == {}

    def test_initial_files_text(self):
        backend = StateBackend(initial_files={"/hello.txt": "Hello World"})
        with _simulate_graph(backend):
            files = backend._read_files()
        assert files["/hello.txt"]["content"] == "Hello World"
        assert files["/hello.txt"]["encoding"] == "utf-8"

    def test_initial_files_binary(self):
        backend = StateBackend(initial_files={"/img.png": "raw"})
        with _simulate_graph(backend):
            assert backend._read_files()["/img.png"]["encoding"] == "base64"

    def test_initial_files_mixed(self):
        backend = StateBackend(
            initial_files={
                "/a.txt": "alpha",
                "/sub/b.py": "print(1)",
                "/img.jpg": "jpeg-data",
            }
        )
        with _simulate_graph(backend):
            files = backend._read_files()
        assert files["/a.txt"]["encoding"] == "utf-8"
        assert files["/img.jpg"]["encoding"] == "base64"

    def test_initial_files_not_re_injected_after_first_access(self):
        """initial_files are only injected on first _read_files() per thread.

        After the first access (snapshot created), subsequent _read_files
        must NOT re-inject initial_files — the snapshot is authoritative.
        """
        backend = StateBackend(initial_files={"/init.txt": "template"})
        with _simulate_graph(backend):
            # First access: initial_files injected
            files1 = backend._read_files()
            assert files1["/init.txt"]["content"] == "template"

            # Now overwrite the file via write (simulating graph changes)
            backend.write(VirtualPath("/init.txt"), "modified-by-agent", overwrite=True)

            # Second _read_files: should return the modified version, NOT
            # re-inject the stale initial template
            files2 = backend._read_files()
            assert files2["/init.txt"]["content"] == "modified-by-agent"

    def test_initial_files_does_not_overwrite_existing_snapshot(self):
        """initial_files never overwrite a path already present in the snapshot.

        If a thread already has snapshot data (e.g. restored from a previous
        graph execution), _read_files must NOT overwrite those files with
        initial_files values.
        """
        backend = StateBackend(initial_files={"/existing.txt": "stale-init"})
        with _simulate_graph(backend):
            # Simulate that a previous graph run wrote this file
            backend.write(VirtualPath("/existing.txt"), "from-previous-run", overwrite=True)
            # Now a "fresh" _read_files should keep the snapshot value
            files = backend._read_files()
        assert files["/existing.txt"]["content"] == "from-previous-run"


# ============================================================================
# Write
# ============================================================================


class TestStateBackendWrite:
    def test_write_new_file(self):
        backend = StateBackend()
        with _simulate_graph(backend):
            r = backend.write(VirtualPath("/test.txt"), "hello")
            assert r.error is None
            assert "hello" in (backend.read(VirtualPath("/test.txt")).content or "")

    def test_write_fails_if_exists(self):
        backend = StateBackend()
        with _simulate_graph(backend):
            backend.write(VirtualPath("/dup.txt"), "original")
            r = backend.write(VirtualPath("/dup.txt"), "modified")
        assert r.error is not None
        assert "已存在" in str(r.error)

    def test_write_overwrite(self):
        backend = StateBackend()
        with _simulate_graph(backend):
            backend.write(VirtualPath("/x.txt"), "v1")
            r = backend.write(VirtualPath("/x.txt"), "v2", overwrite=True)
            assert r.error is None
            assert "v2" in (backend.read(VirtualPath("/x.txt")).content or "")

    def test_binary_extension_gets_base64(self):
        backend = StateBackend()
        with _simulate_graph(backend):
            backend.write(VirtualPath("/img.png"), "data")
            files = backend._read_files()
        assert files["/img.png"]["encoding"] == "base64"


# ============================================================================
# Edit
# ============================================================================


class TestStateBackendEdit:
    def test_edit_replaces(self):
        backend = StateBackend()
        with _simulate_graph(backend):
            backend.write(VirtualPath("/f.py"), "x = 1\ny = 2")
            r = backend.edit(VirtualPath("/f.py"), "x = 1", "x = 100")
        assert r.error is None
        assert r.occurrences == 1
        assert "x = 100" in _files_snapshot(backend)["/f.py"]["content"]

    def test_edit_replace_all(self):
        backend = StateBackend()
        with _simulate_graph(backend):
            backend.write(VirtualPath("/f.txt"), "A-B-A-B-A")
            r = backend.edit(VirtualPath("/f.txt"), "A", "X", replace_all=True)
        assert r.error is None
        assert r.occurrences == 3
        assert _files_snapshot(backend)["/f.txt"]["content"] == "X-B-X-B-X"

    def test_edit_multiple_without_replace_all(self):
        backend = StateBackend()
        with _simulate_graph(backend):
            backend.write(VirtualPath("/f.txt"), "A-B-A")
            r = backend.edit(VirtualPath("/f.txt"), "A", "X")
        assert "处" in str(r.error)

    def test_edit_not_found(self):
        backend = StateBackend()
        with _simulate_graph(backend):
            backend.write(VirtualPath("/f.txt"), "hello")
            r = backend.edit(VirtualPath("/f.txt"), "gone", "x")
        assert "未找到" in str(r.error)

    def test_edit_file_not_found(self):
        backend = StateBackend()
        with _simulate_graph(backend):
            r = backend.edit(VirtualPath("/no.txt"), "x", "y")
        assert "不存在" in str(r.error)

    def test_edit_binary_blocked(self):
        """FIX 1: edit() blocks binary (base64) files with clear error."""
        backend = StateBackend()
        with _simulate_graph(backend):
            backend.write(VirtualPath("/img.png"), "AAAA")
            r = backend.edit(VirtualPath("/img.png"), "AAA", "BBB")
        assert r.error is not None
        assert "二进制" in str(r.error)

    def test_edit_binary_from_initial_files_blocked(self):
        """FIX 1: edit() blocked even from initial_files binary."""
        backend = StateBackend(initial_files={"/img.png": "raw"})
        with _simulate_graph(backend):
            r = backend.edit(VirtualPath("/img.png"), "raw", "new")
        assert r.error is not None
        assert "二进制" in str(r.error)

    def test_edit_trailing_newline_hint(self):
        backend = StateBackend()
        with _simulate_graph(backend):
            backend.write(VirtualPath("/f.txt"), "def foo(): pass")
            r = backend.edit(VirtualPath("/f.txt"), "def foo(): pass\n", "def bar():\n")
        assert r.error is not None
        assert "换行" in str(r.error).lower()


# ============================================================================
# Read
# ============================================================================


class TestStateBackendRead:
    def test_read_line_numbers(self):
        backend = StateBackend()
        with _simulate_graph(backend):
            backend.write(VirtualPath("/f.py"), "a\nb\nc")
            r = backend.read(VirtualPath("/f.py"))
        assert r.total_lines == 3
        assert "a\nb\nc" in (r.content or "")

    def test_read_offset_limit(self):
        backend = StateBackend()
        with _simulate_graph(backend):
            backend.write(VirtualPath("/f.txt"), "\n".join(f"L{i}" for i in range(10)))
            r = backend.read(VirtualPath("/f.txt"), offset=2, limit=3)
        assert r.error is None
        # Should have 3 lines starting from L2 (index 2)
        assert "L2" in (r.content or "")
        assert "L4" in (r.content or "")

    def test_read_binary_base64(self):
        backend = StateBackend()
        with _simulate_graph(backend):
            backend.write(VirtualPath("/img.png"), "binary-data-here")
            r = backend.read(VirtualPath("/img.png"))
        assert r.encoding == "base64"
        assert r.file_type == "image"
        assert r.content == "binary-data-here"

    def test_read_not_found(self):
        backend = StateBackend()
        with _simulate_graph(backend):
            r = backend.read(VirtualPath("/no.txt"))
        assert "不存在" in str(r.error)


# ============================================================================
# Ls
# ============================================================================


class TestStateBackendLs:
    def test_ls_flat(self):
        backend = StateBackend()
        with _simulate_graph(backend):
            backend.write(VirtualPath("/workspace/a.py"), "1")
            backend.write(VirtualPath("/workspace/b.py"), "2")
            result = backend.ls(VirtualPath("/workspace"))
        paths = [fi.path for fi in result.entries]
        assert "/workspace/a.py" in paths
        assert "/workspace/b.py" in paths

    def test_ls_with_subdir(self):
        backend = StateBackend()
        with _simulate_graph(backend):
            backend.write(VirtualPath("/workspace/sub/file.txt"), "hello")
            result = backend.ls(VirtualPath("/workspace"))
        dirs = [fi for fi in result.entries if fi.is_dir]
        assert any(d.path == "/workspace/sub" for d in dirs)

    def test_ls_empty(self):
        backend = StateBackend()
        with _simulate_graph(backend):
            # workspace_root always exists, even when empty
            result = backend.ls(VirtualPath("/workspace"))
        assert result.error is None
        assert result.entries == []

    def test_ls_nonexistent_subdir(self):
        backend = StateBackend()
        with _simulate_graph(backend):
            # Non-root directories without files should return "not found"
            result = backend.ls(VirtualPath("/workspace/dir"))
        assert result.error is not None
        assert "不存在" in str(result.error)


# ============================================================================
# Grep
# ============================================================================


class TestStateBackendGrep:
    def test_grep_finds(self):
        backend = StateBackend()
        with _simulate_graph(backend):
            backend.write(VirtualPath("/workspace/a.py"), "def foo(): pass")
            backend.write(VirtualPath("/workspace/b.py"), "def bar(): pass")
            r = backend.grep("foo", path=VirtualPath("/workspace"))
        assert any("foo" in m.text for m in (r.matches or []))

    def test_grep_skips_binary(self):
        backend = StateBackend()
        with _simulate_graph(backend):
            backend.write(VirtualPath("/workspace/img.png"), "foo-visible")
            backend.write(VirtualPath("/workspace/txt.txt"), "foo-hidden")
            r = backend.grep("foo", path=VirtualPath("/workspace"))
        paths = [m.path for m in (r.matches or [])]
        assert "/workspace/txt.txt" in paths
        assert "/workspace/img.png" not in paths  # binary files skipped

    def test_grep_no_match(self):
        backend = StateBackend()
        with _simulate_graph(backend):
            backend.write(VirtualPath("/workspace/f.txt"), "hello")
            r = backend.grep("zzz", path=VirtualPath("/workspace"))
        assert len(r.matches or []) == 0


# ============================================================================
# Glob
# ============================================================================


class TestStateBackendGlob:
    def test_glob_finds(self):
        backend = StateBackend()
        with _simulate_graph(backend):
            backend.write(VirtualPath("/workspace/src/main.py"), "1")
            backend.write(VirtualPath("/workspace/src/util.py"), "2")
            backend.write(VirtualPath("/workspace/README.md"), "3")
            r = backend.glob("**/*.py", path=VirtualPath("/workspace/src"))
        paths = [fi.path for fi in (r.matches or [])]
        assert len(paths) == 2
        assert "/workspace/src/main.py" in paths

    def test_glob_no_match(self):
        backend = StateBackend()
        with _simulate_graph(backend):
            r = backend.glob("*.py", path=VirtualPath("/workspace"))
        assert len(r.matches or []) == 0

    def test_glob_size_is_content_length(self):
        backend = StateBackend()
        with _simulate_graph(backend):
            backend.write(VirtualPath("/workspace/f.txt"), "hello")  # 5 chars
            r = backend.glob("/workspace/f.txt", path=VirtualPath("/workspace"))
        assert r.matches[0].size == 5


# ============================================================================
# Upload / Download (inside and outside graph)
# ============================================================================


class TestStateBackendUploadDownload:
    # ---- Graph-in upload/download (via _simulate_graph) ----

    def test_upload_text_in_graph(self):
        backend = StateBackend()
        with _simulate_graph(backend):
            results = backend.upload_files([(VirtualPath("/workspace/upload.txt"), b"hello world")])
            assert results[0].error is None
            files = backend._read_files()
        assert files["/workspace/upload.txt"]["content"] == "hello world"
        assert files["/workspace/upload.txt"]["encoding"] == "utf-8"

    def test_upload_binary_in_graph(self):
        backend = StateBackend()
        raw = _binary_content(100)
        with _simulate_graph(backend):
            results = backend.upload_files([(VirtualPath("/workspace/data.bin"), raw)])
            assert results[0].error is None
            fd = backend._read_files()["/workspace/data.bin"]
        assert fd["encoding"] == "base64"
        decoded = base64.b64decode(fd["content"])
        assert decoded == raw

    def test_download_text_in_graph(self):
        backend = StateBackend()
        with _simulate_graph(backend):
            backend.write(VirtualPath("/workspace/dl.txt"), "download me")
            results = backend.download_files([VirtualPath("/workspace/dl.txt")])
        assert results[0].content == b"download me"

    def test_download_binary_in_graph(self):
        backend = StateBackend()
        raw = _binary_content(100)
        with _simulate_graph(backend):
            backend.upload_files([(VirtualPath("/workspace/bin.bin"), raw)])
            results = backend.download_files([VirtualPath("/workspace/bin.bin")])
        assert results[0].content == raw

    def test_download_not_found_in_graph(self):
        backend = StateBackend()
        with _simulate_graph(backend):
            results = backend.download_files([VirtualPath("/workspace/no.txt")])
        assert results[0].error.code == ErrorCode.NOT_FOUND

    def test_upload_multiple_in_graph(self):
        backend = StateBackend()
        with _simulate_graph(backend):
            results = backend.upload_files([
                (VirtualPath("/workspace/a.txt"), b"alpha"),
                (VirtualPath("/workspace/b.txt"), b"beta"),
            ])
            assert all(r.error is None for r in results)
            files = backend._read_files()
        assert set(files.keys()) == {"/workspace/a.txt", "/workspace/b.txt"}

    # ---- Graph-outside upload/download (explicit thread_id required) ----

    def test_upload_outside_graph_requires_thread_id(self):
        """Outside graph, upload_files() with no thread_id must raise ValueError."""
        backend = StateBackend()
        with pytest.raises(ValueError, match="thread_id is required"):
            backend.upload_files([(VirtualPath("/workspace/f.txt"), b"data")])

    def test_download_outside_graph_requires_thread_id(self):
        """Outside graph, download_files() with no thread_id must raise ValueError."""
        backend = StateBackend()
        with pytest.raises(ValueError, match="thread_id is required"):
            backend.download_files([VirtualPath("/workspace/f.txt")])

    def test_upload_outside_graph_pending(self):
        """Graph-outside upload queues to per-thread pending buffer."""
        backend = StateBackend()
        backend.upload_files([(VirtualPath("/workspace/pending.txt"), b"queued")], thread_id="t1")
        # Not yet in snapshot
        assert "/workspace/pending.txt" not in _files_snapshot(backend, "t1")
        # But visible via download_files (merges snapshot + pending)
        results = backend.download_files([VirtualPath("/workspace/pending.txt")], thread_id="t1")
        assert results[0].content == b"queued"

    def test_pending_flushed_on_graph_entry(self):
        """Graph-outside uploads are flushed on first _read_files for that thread."""
        backend = StateBackend()
        backend.upload_files([(VirtualPath("/workspace/flush.txt"), b"flush-me")], thread_id="t1")
        with _simulate_graph(backend, thread_id="t1"):
            files = backend._read_files()
        assert files["/workspace/flush.txt"]["content"] == "flush-me"

    def test_download_outside_graph_reads_snapshot(self):
        """Outside graph, download reads from last-synced snapshot + pending."""
        backend = StateBackend()
        # Simulate a previous graph run that wrote a file
        with _simulate_graph(backend, thread_id="sess"):
            backend.write(VirtualPath("/workspace/output.txt"), "finished")
        # Now outside graph
        results = backend.download_files([VirtualPath("/workspace/output.txt")], thread_id="sess")
        assert results[0].content == b"finished"

    def test_download_outside_graph_nonexistent_thread(self):
        """Downloading from a never-seen thread returns empty results."""
        backend = StateBackend()
        results = backend.download_files([VirtualPath("/workspace/a.txt")], thread_id="unknown")
        assert results[0].error.code == ErrorCode.NOT_FOUND


# ============================================================================
# Per-thread isolation
# ============================================================================


class TestStateBackendIsolation:
    def test_separate_threads_isolated(self):
        """Files written in thread A must NOT leak into thread B."""
        backend = StateBackend()
        with _simulate_graph(backend, thread_id="A"):
            backend.write(VirtualPath("/workspace/secret.txt"), "A-only")
        with _simulate_graph(backend, thread_id="B"):
            files = backend._read_files()
        assert "/workspace/secret.txt" not in files

    def test_separate_threads_independent_snapshots(self):
        """Each thread has its own independent file space."""
        backend = StateBackend()
        with _simulate_graph(backend, thread_id="A"):
            backend.write(VirtualPath("/workspace/a.txt"), "a")
        with _simulate_graph(backend, thread_id="B"):
            backend.write(VirtualPath("/workspace/b.txt"), "b")
        assert "/workspace/a.txt" in _files_snapshot(backend, "A")
        assert "/workspace/b.txt" in _files_snapshot(backend, "B")
        assert "/workspace/a.txt" not in _files_snapshot(backend, "B")
        assert "/workspace/b.txt" not in _files_snapshot(backend, "A")

    def test_initial_files_per_new_thread(self):
        """Each new thread receives initial_files on first access."""
        backend = StateBackend(initial_files={"/workspace/init.txt": "template"})
        with _simulate_graph(backend, thread_id="t1"):
            assert "/workspace/init.txt" in backend._read_files()
        with _simulate_graph(backend, thread_id="t2"):
            assert "/workspace/init.txt" in backend._read_files()

    def test_pending_per_thread_isolated(self):
        """Pending uploads are per-thread — thread A's pending doesn't leak to thread B."""
        backend = StateBackend()
        backend.upload_files([(VirtualPath("/workspace/a.txt"), b"a")], thread_id="A")
        backend.upload_files([(VirtualPath("/workspace/b.txt"), b"b")], thread_id="B")
        with _simulate_graph(backend, thread_id="A"):
            files = backend._read_files()
        assert "/workspace/a.txt" in files
        assert "/workspace/b.txt" not in files  # B's pending not flushed to A

    def test_full_replace_on_reread_removes_deleted_files(self):
        """Simulates checkpoint rollback: re-read replaces, not merges."""
        backend = StateBackend()
        with _simulate_graph(backend):
            backend.write(VirtualPath("/workspace/keep.txt"), "keep")
            backend.write(VirtualPath("/workspace/gone.txt"), "gone")
            # Simulate that /gone.txt was deleted from the channel
            # (in the mock, we manually remove it from the snapshot)
            with backend._lock:
                backend._snapshots["test"] = {
                    "/workspace/keep.txt": {"content": "keep", "encoding": "utf-8"}
                }
            # _read_files should full-replace, not merge
            files = backend._read_files()
        assert "/workspace/keep.txt" in files
        assert "/workspace/gone.txt" not in files  # deleted file NOT resurrected


# ============================================================================
# Concurrency (FIX 2: threading.RLock)
# ============================================================================


class TestStateBackendConcurrency:
    def test_concurrent_writes_no_race(self):
        """FIX 2: Multiple threads writing concurrently — no lost files."""
        backend = StateBackend()
        num_writers = 20

        def writer(i: int):
            with _simulate_graph(backend, thread_id=f"t{i}"):
                backend.write(VirtualPath(f"/workspace/file_{i}.txt"), f"content_{i}")

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(writer, i) for i in range(num_writers)]
            for f in as_completed(futures):
                f.result()  # raise if any error

        for i in range(num_writers):
            snap = _files_snapshot(backend, f"t{i}")
            assert f"/workspace/file_{i}.txt" in snap
            assert snap[f"/workspace/file_{i}.txt"]["content"] == f"content_{i}"

    def test_concurrent_read_write_no_race(self):
        """FIX 2: Readers and writers interleaved — no corruption."""
        backend = StateBackend()
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
        """FIX 2: Concurrent upload_files — no lost data."""
        backend = StateBackend()
        num = 30

        def uploader(start: int):
            batch = [
                (VirtualPath(f"/workspace/u_{j}.txt"), f"val_{j}".encode())
                for j in range(start, start + 10)
            ]
            return backend.upload_files(batch, thread_id="shared")

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(uploader, i * 10) for i in range(3)]
            for f in as_completed(futures):
                f.result()

        # Verify all files exist in the pending buffer
        with backend._lock:
            pending = backend._pending_uploads.get("shared", {})
        assert len(pending) == num
        for i in range(num):
            assert pending[f"/workspace/u_{i}.txt"]["content"] == f"val_{i}"


# ============================================================================
# Tools
# ============================================================================


class TestStateBackendTools:
    def test_extra_tools_only(self):
        backend = StateBackend()
        names = {t.name for t in backend.tools}
        assert "tree" in names
        assert "ls" not in names  # core tools are built by middleware
        assert "read" not in names

    def test_tree_is_str(self):
        backend = StateBackend()
        with _simulate_graph(backend):
            backend.write(VirtualPath("/workspace/a.txt"), "a")
            backend.write(VirtualPath("/workspace/sub/b.txt"), "b")
            result = backend.tree(VirtualPath("/workspace"), depth=2)
        assert isinstance(result, str)
        assert len(result) > 0


# ============================================================================
# Graph context gating (outside graph = error)
# ============================================================================


class TestGraphContextGating:
    def test_read_files_outside_graph_raises(self):
        backend = StateBackend()
        with pytest.raises(RuntimeError, match="graph context"):
            backend._read_files()

    def test_send_files_update_outside_graph_raises(self):
        backend = StateBackend()
        with pytest.raises(RuntimeError, match="graph context"):
            backend._send_files_update({"/workspace/f.txt": {"content": "x", "encoding": "utf-8"}})

    def test_write_outside_graph_raises(self):
        backend = StateBackend()
        with pytest.raises(RuntimeError, match="graph context"):
            backend.write(VirtualPath("/workspace/f.txt"), "data")

    def test_read_outside_graph_raises(self):
        backend = StateBackend()
        with pytest.raises(RuntimeError, match="graph context"):
            backend.read(VirtualPath("/workspace/f.txt"))

    def test_ls_outside_graph_raises(self):
        backend = StateBackend()
        with pytest.raises(RuntimeError, match="graph context"):
            backend.ls(VirtualPath("/workspace"))

    def test_edit_outside_graph_raises(self):
        backend = StateBackend()
        with pytest.raises(RuntimeError, match="graph context"):
            backend.edit(VirtualPath("/workspace/f.txt"), "a", "b")
