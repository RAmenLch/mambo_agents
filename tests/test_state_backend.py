"""Dedicated tests for StateBackend — binary handling, concurrency, upload/download."""

import base64
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from mambo_agents.backends.state import StateBackend


def _binary_content(size: int = 256) -> bytes:
    """Non-UTF-8 bytes for binary file testing."""
    return bytes(range(256)) * ((size + 255) // 256)


# ============================================================================
# Initialization
# ============================================================================


class TestStateBackendInit:
    def test_default_construction(self):
        backend = StateBackend()
        assert backend._read_files() == {}

    def test_initial_files_text(self):
        backend = StateBackend(initial_files={"/hello.txt": "Hello World"})
        files = backend._read_files()
        assert files["/hello.txt"]["content"] == "Hello World"
        assert files["/hello.txt"]["encoding"] == "utf-8"

    def test_initial_files_binary(self):
        backend = StateBackend(initial_files={"/img.png": "raw"})
        assert backend._read_files()["/img.png"]["encoding"] == "base64"

    def test_initial_files_mixed(self):
        backend = StateBackend(
            initial_files={"/a.txt": "alpha", "/sub/b.py": "print(1)", "/img.jpg": "jpeg-data"}
        )
        files = backend._read_files()
        assert files["/a.txt"]["encoding"] == "utf-8"
        assert files["/img.jpg"]["encoding"] == "base64"


# ============================================================================
# Write
# ============================================================================


class TestStateBackendWrite:
    def test_write_new_file(self):
        backend = StateBackend()
        r = backend.write("/test.txt", "hello")
        assert r.error is None
        assert "hello" in (backend.read("/test.txt").content or "")

    def test_write_fails_if_exists(self):
        backend = StateBackend()
        backend.write("/dup.txt", "original")
        r = backend.write("/dup.txt", "modified")
        assert r.error is not None
        assert "already exists" in (r.error or "")

    def test_write_overwrite(self):
        backend = StateBackend()
        backend.write("/x.txt", "v1")
        r = backend.write("/x.txt", "v2", overwrite=True)
        assert r.error is None
        assert "v2" in (backend.read("/x.txt").content or "")

    def test_binary_extension_gets_base64(self):
        backend = StateBackend()
        backend.write("/img.png", "data")
        assert backend._read_files()["/img.png"]["encoding"] == "base64"


# ============================================================================
# Edit
# ============================================================================


class TestStateBackendEdit:
    def test_edit_replaces(self):
        backend = StateBackend()
        backend.write("/f.py", "x = 1\ny = 2")
        r = backend.edit("/f.py", "x = 1", "x = 100")
        assert r.error is None
        assert r.occurrences == 1
        assert "x = 100" in backend._read_files()["/f.py"]["content"]

    def test_edit_replace_all(self):
        backend = StateBackend()
        backend.write("/f.txt", "A-B-A-B-A")
        r = backend.edit("/f.txt", "A", "X", replace_all=True)
        assert r.error is None
        assert r.occurrences == 3
        assert backend._read_files()["/f.txt"]["content"] == "X-B-X-B-X"

    def test_edit_multiple_without_replace_all(self):
        backend = StateBackend()
        backend.write("/f.txt", "A-B-A")
        r = backend.edit("/f.txt", "A", "X")
        assert "appears 2 times" in (r.error or "")

    def test_edit_not_found(self):
        backend = StateBackend()
        backend.write("/f.txt", "hello")
        r = backend.edit("/f.txt", "gone", "x")
        assert "old_str not found" in (r.error or "")

    def test_edit_file_not_found(self):
        backend = StateBackend()
        r = backend.edit("/no.txt", "x", "y")
        assert "file not found" in (r.error or "")

    def test_edit_binary_blocked(self):
        """FIX 1: edit() blocks binary (base64) files with clear error."""
        backend = StateBackend()
        backend.write("/img.png", "AAAA")
        r = backend.edit("/img.png", "AAA", "BBB")
        assert r.error is not None
        assert "binary" in (r.error or "").lower()
        assert "base64" in (r.error or "").lower()

    def test_edit_binary_from_initial_files_blocked(self):
        """FIX 1: edit() blocked even from initial_files binary."""
        backend = StateBackend(initial_files={"/img.png": "raw"})
        r = backend.edit("/img.png", "raw", "new")
        assert r.error is not None
        assert "binary" in (r.error or "").lower()

    def test_edit_trailing_newline_hint(self):
        backend = StateBackend()
        backend.write("/f.txt", "def foo(): pass")
        r = backend.edit("/f.txt", "def foo(): pass\n", "def bar():\n")
        assert r.error is not None
        assert "newline" in (r.error or "").lower()


# ============================================================================
# Read
# ============================================================================


class TestStateBackendRead:
    def test_read_line_numbers(self):
        backend = StateBackend()
        backend.write("/f.py", "a\nb\nc")
        r = backend.read("/f.py")
        assert r.total_lines == 3
        assert "     1\ta" in (r.content or "")

    def test_read_offset_limit(self):
        backend = StateBackend()
        backend.write("/f.txt", "\n".join(f"L{i}" for i in range(10)))
        r = backend.read("/f.txt", offset=2, limit=3)
        assert r.error is None
        # Should have 3 lines starting from line 3
        assert "     3\tL2" in (r.content or "")

    def test_read_binary_base64(self):
        backend = StateBackend()
        backend.write("/img.png", "binary-data-here")
        r = backend.read("/img.png")
        assert r.encoding == "base64"
        assert r.file_type == "image"
        assert r.content == "binary-data-here"

    def test_read_not_found(self):
        backend = StateBackend()
        r = backend.read("/no.txt")
        assert "not found" in (r.error or "")


# ============================================================================
# Ls
# ============================================================================


class TestStateBackendLs:
    def test_ls_flat(self):
        backend = StateBackend()
        backend.write("/a.py", "1")
        backend.write("/b.py", "2")
        result = backend.ls("/")
        paths = [fi.path for fi in result.entries]
        assert "/a.py" in paths
        assert "/b.py" in paths

    def test_ls_with_subdir(self):
        backend = StateBackend()
        backend.write("/sub/file.txt", "hello")
        result = backend.ls("/")
        dirs = [fi for fi in result.entries if fi.is_dir]
        assert any("/sub/" in d.path for d in dirs)

    def test_ls_empty(self):
        backend = StateBackend()
        result = backend.ls("/")
        assert result.entries is not None
        assert len(result.entries) == 0


# ============================================================================
# Grep
# ============================================================================


class TestStateBackendGrep:
    def test_grep_finds(self):
        backend = StateBackend()
        backend.write("/a.py", "def foo(): pass")
        backend.write("/b.py", "def bar(): pass")
        r = backend.grep("foo", path="/")
        assert any("foo" in m.text for m in (r.matches or []))

    def test_grep_skips_binary(self):
        backend = StateBackend()
        backend.write("/img.png", "foo-visible")
        backend.write("/txt.txt", "foo-hidden")
        r = backend.grep("foo", path="/")
        paths = [m.path for m in (r.matches or [])]
        assert "/txt.txt" in paths
        assert "/img.png" not in paths  # binary files skipped

    def test_grep_no_match(self):
        backend = StateBackend()
        backend.write("/f.txt", "hello")
        r = backend.grep("zzz", path="/")
        assert len(r.matches or []) == 0


# ============================================================================
# Glob
# ============================================================================


class TestStateBackendGlob:
    def test_glob_finds(self):
        backend = StateBackend()
        backend.write("/src/main.py", "1")
        backend.write("/src/util.py", "2")
        backend.write("/README.md", "3")
        r = backend.glob("**/*.py", path="/src")
        paths = [fi.path for fi in (r.matches or [])]
        assert len(paths) == 2
        assert "/src/main.py" in paths

    def test_glob_no_match(self):
        backend = StateBackend()
        r = backend.glob("*.py", path="/")
        assert len(r.matches or []) == 0

    def test_glob_size_is_content_length(self):
        backend = StateBackend()
        backend.write("/f.txt", "hello")  # 5 chars
        r = backend.glob("/f.txt", path="/")
        assert r.matches[0].size == 5


# ============================================================================
# Upload / Download
# ============================================================================


class TestStateBackendUploadDownload:
    def test_upload_text(self):
        backend = StateBackend()
        results = backend.upload_files([("/upload.txt", b"hello world")])
        assert results[0].error is None
        files = backend._read_files()
        assert files["/upload.txt"]["content"] == "hello world"
        assert files["/upload.txt"]["encoding"] == "utf-8"

    def test_upload_binary(self):
        backend = StateBackend()
        raw = _binary_content(100)
        results = backend.upload_files([("/data.bin", raw)])
        assert results[0].error is None
        fd = backend._read_files()["/data.bin"]
        assert fd["encoding"] == "base64"
        decoded = base64.b64decode(fd["content"])
        assert decoded == raw

    def test_download_text(self):
        backend = StateBackend()
        backend.write("/dl.txt", "download me")
        results = backend.download_files(["/dl.txt"])
        assert results[0].content == b"download me"

    def test_download_binary(self):
        backend = StateBackend()
        raw = _binary_content(100)
        backend.upload_files([("/bin.bin", raw)])
        results = backend.download_files(["/bin.bin"])
        assert results[0].content == raw

    def test_download_not_found(self):
        backend = StateBackend()
        results = backend.download_files(["/no.txt"])
        assert results[0].error == "file_not_found"

    def test_upload_multiple(self):
        backend = StateBackend()
        results = backend.upload_files([
            ("/a.txt", b"alpha"),
            ("/b.txt", b"beta"),
        ])
        assert all(r.error is None for r in results)
        assert set(backend._read_files().keys()) == {"/a.txt", "/b.txt"}


# ============================================================================
# Concurrency (FIX 2: threading.Lock)
# ============================================================================


class TestStateBackendConcurrency:
    def test_concurrent_writes_no_race(self):
        """FIX 2: Multiple threads writing concurrently — no lost files."""
        backend = StateBackend()
        num_writers = 20

        def writer(i: int):
            backend.write(f"/file_{i}.txt", f"content_{i}")

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(writer, i) for i in range(num_writers)]
            for f in as_completed(futures):
                f.result()  # raise if any error

        files = backend._read_files()
        assert len(files) == num_writers
        for i in range(num_writers):
            assert files[f"/file_{i}.txt"]["content"] == f"content_{i}"

    def test_concurrent_read_write_no_race(self):
        """FIX 2: Readers and writers interleaved — no corruption."""
        backend = StateBackend()
        backend.write("/shared.txt", "initial\n" * 10)

        def reader():
            for _ in range(100):
                r = backend.read("/shared.txt")
                if r.content:
                    assert "initial" in r.content

        def writer():
            for i in range(100):
                backend.write(f"/w_{i}.txt", f"hello_{i}", overwrite=True)

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
            batch = [(f"/u_{j}.txt", f"val_{j}".encode()) for j in range(start, start + 10)]
            return backend.upload_files(batch)

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(uploader, i * 10) for i in range(3)]
            for f in as_completed(futures):
                f.result()

        files = backend._read_files()
        assert len(files) == num
        for i in range(num):
            assert files[f"/u_{i}.txt"]["content"] == f"val_{i}"


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
        backend.write("/a.txt", "a")
        backend.write("/sub/b.txt", "b")
        result = backend.tree("/", depth=2)
        assert isinstance(result, str)
        assert len(result) > 0
