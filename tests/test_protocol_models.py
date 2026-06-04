"""Tests for Pydantic result models and BackendProtocol in ``mambo_agents.backends.protocol``."""

import pytest
from pydantic import ValidationError

from mambo_agents.backends.protocol import (
    BackendProtocol,
    DownloadFileResult,
    EditResult,
    FileInfo,
    GlobResult,
    GrepMatch,
    GrepResult,
    LsResult,
    ReadResult,
    UploadFileResult,
    WriteResult,
)


# ============================================================================
# FileInfo
# ============================================================================


class TestFileInfo:
    def test_defaults(self):
        fi = FileInfo(path="/a.txt")
        assert fi.path == "/a.txt"
        assert fi.is_dir is False
        assert fi.size == 0
        assert fi.modified_at == ""

    def test_frozen(self):
        fi = FileInfo(path="/a.txt")
        with pytest.raises(Exception):
            fi.path = "/b.txt"  # type: ignore[misc]

    def test_serialize(self):
        fi = FileInfo(path="/a.py", size=1024, is_dir=False)
        d = fi.model_dump()
        assert d["path"] == "/a.py"
        assert d["size"] == 1024
        assert d["is_dir"] is False


# ============================================================================
# GrepMatch
# ============================================================================


class TestGrepMatch:
    def test_creation(self):
        m = GrepMatch(path="/f.py", line=42, text="def foo():")
        assert m.path == "/f.py"
        assert m.line == 42
        assert m.text == "def foo():"

    def test_frozen(self):
        m = GrepMatch(path="/f.py", line=1, text="x")
        with pytest.raises(Exception):
            m.line = 99  # type: ignore[misc]


# ============================================================================
# LsResult
# ============================================================================


class TestLsResult:
    def test_str_with_entries(self):
        r = LsResult(entries=[FileInfo(path="/a.txt", size=100)])
        s = str(r)
        assert "/a.txt" in s
        assert "100 B" in s

    def test_str_with_directories(self):
        r = LsResult(entries=[FileInfo(path="/sub", is_dir=True)])
        s = str(r)
        assert "/sub/" in s

    def test_str_empty(self):
        r = LsResult(entries=[])
        assert str(r) == "(empty directory)"

    def test_str_with_error(self):
        r = LsResult(error="Permission denied", entries=[])
        s = str(r)
        assert "Warning: Permission denied" in s

    def test_str_with_error_and_entries(self):
        """Error + entries → both shown."""
        r = LsResult(
            error="Partial listing",
            entries=[FileInfo(path="/ok.txt")],
        )
        s = str(r)
        assert "Warning: Partial listing" in s
        assert "/ok.txt" in s


# ============================================================================
# ReadResult
# ============================================================================


class TestReadResult:
    def test_defaults(self):
        r = ReadResult(content="hello")
        assert r.file_type == "text"
        assert r.mime_type == ""
        assert r.is_multimodal is False
        assert r.total_lines == 0
        assert r.encoding is None

    def test_str_with_content(self):
        r = ReadResult(content="some text", encoding="utf-8")
        assert str(r) == "some text"

    def test_str_with_error(self):
        r = ReadResult(error="File not found")
        assert str(r) == "Error: File not found"

    def test_is_multimodal_true(self):
        r = ReadResult(
            content="AAAA",
            encoding="base64",
            file_type="image",
            mime_type="image/png",
        )
        assert r.is_multimodal is True

    def test_is_multimodal_false_for_base64_text(self):
        r = ReadResult(content="AAAA", encoding="base64", file_type="text")
        assert r.is_multimodal is False


# ============================================================================
# WriteResult
# ============================================================================


class TestWriteResult:
    def test_str_success(self):
        r = WriteResult(path="/new.txt")
        assert str(r) == "File written: /new.txt"

    def test_str_error(self):
        r = WriteResult(error="File already exists", path="/new.txt")
        assert str(r) == "Error: File already exists"


# ============================================================================
# EditResult
# ============================================================================


class TestEditResult:
    def test_str_success(self):
        r = EditResult(path="/f.py", occurrences=3)
        assert str(r) == "File edited: /f.py (3 replacement(s))"

    def test_str_error(self):
        r = EditResult(error="old_str not found", path="/f.py")
        assert str(r) == "Error: old_str not found"

    def test_default_occurrences(self):
        r = EditResult(path="/f.py")
        assert r.occurrences == 0


# ============================================================================
# GrepResult
# ============================================================================


class TestGrepResult:
    def test_str_with_matches(self):
        m = GrepMatch(path="/f.py", line=3, text="def foo():")
        r = GrepResult(matches=[m])
        s = str(r)
        assert "/f.py:3: def foo():" in s

    def test_str_no_matches(self):
        r = GrepResult(matches=[])
        assert str(r) == "No matches found."

    def test_str_error(self):
        r = GrepResult(error="Search failed")
        s = str(r)
        assert "Warning: Search failed" in s


# ============================================================================
# GlobResult
# ============================================================================


class TestGlobResult:
    def test_str_with_matches(self):
        fi = FileInfo(path="/a.py")
        r = GlobResult(matches=[fi])
        assert str(r) == "/a.py"

    def test_str_no_matches(self):
        r = GlobResult(matches=[])
        assert str(r) == "No files found."

    def test_str_error(self):
        r = GlobResult(error="Invalid pattern")
        s = str(r)
        assert "Warning: Invalid pattern" in s


# ============================================================================
# UploadFileResult / DownloadFileResult
# ============================================================================


class TestUploadFileResult:
    def test_success(self):
        r = UploadFileResult(path="/upload.txt")
        assert r.path == "/upload.txt"
        assert r.error is None

    def test_error(self):
        r = UploadFileResult(path="/bad.txt", error="Disk full")
        assert r.error == "Disk full"


class TestDownloadFileResult:
    def test_success(self):
        r = DownloadFileResult(path="/down.txt", content=b"hello")
        assert r.path == "/down.txt"
        assert r.content == b"hello"
        assert r.error is None

    def test_error(self):
        r = DownloadFileResult(path="/missing.txt", error="file_not_found")
        assert r.content is None
        assert r.error == "file_not_found"


# ============================================================================
# BackendProtocol — abstract methods check
# ============================================================================


class TestBackendProtocol:
    def test_cannot_instantiate_directly(self):
        """BackendProtocol is abstract — must be subclassed."""
        with pytest.raises(TypeError):
            BackendProtocol()  # type: ignore[abstract]

    def test_subclass_without_abstract_methods_cannot_instantiate(self):
        """Subclass must implement all abstract methods."""

        class Incomplete(BackendProtocol):
            @property
            def tools(self):
                return []

        with pytest.raises(TypeError):
            Incomplete()

    def test_minimal_subclass_works(self):
        """Minimal implementation of all abstract methods works."""

        class Minimal(BackendProtocol):
            @property
            def tools(self):
                return []

            def ls(self, path):
                return LsResult(entries=[])

            def read_raw(self, file_path, offset=0, limit=2000, include_line_numbers=False):
                return ReadResult(content="mock")

            def write(self, file_path, content, overwrite=False):
                return WriteResult(path=file_path)

            def edit(self, file_path, old_str, new_str, *, replace_all=False):
                return EditResult(path=file_path)

            def grep(self, pattern, path="/", glob=None):
                return GrepResult(matches=[])

            def glob(self, pattern, path="/"):
                return GlobResult(matches=[])

        b = Minimal()
        assert b is not None
        assert b.tools == []

    def test_default_description(self):
        """Default description includes backend type name."""

        class TestBackend(BackendProtocol):
            @property
            def tools(self):
                return []

            def ls(self, path):
                return LsResult(entries=[])

            def read_raw(self, file_path, offset=0, limit=2000, include_line_numbers=False):
                return ReadResult(content="")

            def write(self, file_path, content, overwrite=False):
                return WriteResult(path=file_path)

            def edit(self, file_path, old_str, new_str, *, replace_all=False):
                return EditResult(path=file_path)

            def grep(self, pattern, path="/", glob=None):
                return GrepResult(matches=[])

            def glob(self, pattern, path="/"):
                return GlobResult(matches=[])

        b = TestBackend()
        desc = b.description
        assert "test" in desc.lower()
        assert "include_line_numbers" in desc
