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
from mambo_agents.backends.schemas import BackendError, ErrorCode


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
        assert fi.desc == ""

    def test_frozen(self):
        fi = FileInfo(path="/a.txt")
        with pytest.raises(Exception):
            fi.path = "/b.txt"  # type: ignore[misc]

    def test_serialize(self):
        fi = FileInfo(path="/a.py", size=1024, is_dir=False, desc="main script")
        d = fi.model_dump()
        assert d["path"] == "/a.py"
        assert d["size"] == 1024
        assert d["is_dir"] is False
        assert d["desc"] == "main script"


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
        assert "/a.txt(100 B)" in s

    def test_str_with_directories(self):
        r = LsResult(entries=[FileInfo(path="/sub", is_dir=True)])
        s = str(r)
        assert "/sub/" in s

    def test_str_empty(self):
        r = LsResult(entries=[])
        assert str(r) == "(empty directory)"

    def test_str_with_error(self):
        r = LsResult(error=BackendError(code=ErrorCode.INVALID, message="Permission denied"), entries=[])
        s = str(r)
        assert "Permission denied" in s

    def test_str_with_error_and_entries(self):
        """Error + entries → both shown."""
        r = LsResult(
            error=BackendError(code=ErrorCode.INVALID, message="Partial listing"),
            entries=[FileInfo(path="/ok.txt")],
        )
        s = str(r)
        assert "Partial listing" in s
        assert "/ok.txt" in s

    def test_str_file_with_desc(self):
        r = LsResult(entries=[FileInfo(path="/a.py", size=2048, desc="主入口")])
        s = str(r)
        assert s == "/a.py(2 KB)  -- 主入口"

    def test_str_dir_with_desc(self):
        r = LsResult(entries=[FileInfo(path="/lib", is_dir=True, desc="工具库")])
        s = str(r)
        assert s == "/lib/  -- 工具库"

    def test_str_desc_newline_replaced(self):
        r = LsResult(entries=[FileInfo(path="/note.txt", size=10, desc="第一行\n第二行")])
        s = str(r)
        assert s == "/note.txt(10 B)  -- 第一行 第二行"

    def test_str_empty_desc_not_shown(self):
        r = LsResult(entries=[FileInfo(path="/a.txt", size=50, desc="")])
        s = str(r)
        assert "  -- " not in s
        assert s == "/a.txt(50 B)"


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
        r = ReadResult(error=BackendError(code=ErrorCode.NOT_FOUND, message="File not found"))
        assert "File not found" in str(r)

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
        r = WriteResult(error=BackendError(code=ErrorCode.ALREADY_EXISTS, path="/new.txt", message="File already exists"), path="/new.txt")
        assert "File already exists" in str(r)


# ============================================================================
# EditResult
# ============================================================================


class TestEditResult:
    def test_str_success(self):
        r = EditResult(path="/f.py", occurrences=3)
        assert str(r) == "File edited: /f.py (3 replacement(s))"

    def test_str_error(self):
        r = EditResult(error=BackendError(code=ErrorCode.OLD_STR_NOT_FOUND, path="/f.py", message="old_str not found"), path="/f.py")
        assert "old_str not found" in str(r)

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
        r = GrepResult(error=BackendError(code=ErrorCode.INVALID, message="Search failed"))
        s = str(r)
        assert "Search failed" in s


# ============================================================================
# GlobResult
# ============================================================================


class TestGlobResult:
    def test_str_with_matches(self):
        fi = FileInfo(path="/a.py")
        r = GlobResult(matches=[fi])
        assert str(r) == "/a.py(0 B)"

    def test_str_with_size(self):
        fi = FileInfo(path="/b.py", size=4096)
        r = GlobResult(matches=[fi])
        assert str(r) == "/b.py(4 KB)"

    def test_str_with_desc(self):
        fi = FileInfo(path="/main.py", size=512, desc="启动脚本")
        r = GlobResult(matches=[fi])
        assert str(r) == "/main.py(512 B)  -- 启动脚本"

    def test_str_desc_newline_replaced(self):
        fi = FileInfo(path="/readme.md", size=100, desc="项目说明\n请阅读")
        r = GlobResult(matches=[fi])
        assert str(r) == "/readme.md(100 B)  -- 项目说明 请阅读"

    def test_str_empty_desc_not_shown(self):
        fi = FileInfo(path="/a.py", size=42, desc="")
        r = GlobResult(matches=[fi])
        assert "  -- " not in str(r)
        assert str(r) == "/a.py(42 B)"

    def test_str_multiple_matches(self):
        r = GlobResult(matches=[
            FileInfo(path="/a.py", size=100),
            FileInfo(path="/b.py", size=200),
        ])
        lines = str(r).split("\n")
        assert lines[0] == "/a.py(100 B)"
        assert lines[1] == "/b.py(200 B)"

    def test_str_no_matches(self):
        r = GlobResult(matches=[])
        assert str(r) == "No matches found."

    def test_str_error(self):
        r = GlobResult(error=BackendError(code=ErrorCode.INVALID, message="Invalid pattern"))
        s = str(r)
        assert "Invalid pattern" in s


# ============================================================================
# UploadFileResult / DownloadFileResult
# ============================================================================


class TestUploadFileResult:
    def test_success(self):
        r = UploadFileResult(path="/upload.txt")
        assert r.path == "/upload.txt"
        assert r.error is None

    def test_error(self):
        r = UploadFileResult(path="/bad.txt", error=BackendError(code=ErrorCode.IO_ERROR, path="/bad.txt", message="Disk full"))
        assert "Disk full" in str(r.error)


class TestDownloadFileResult:
    def test_success(self):
        r = DownloadFileResult(path="/down.txt", content=b"hello")
        assert r.path == "/down.txt"
        assert r.content == b"hello"
        assert r.error is None

    def test_error(self):
        r = DownloadFileResult(path="/missing.txt", error=BackendError(code=ErrorCode.NOT_FOUND, path="/missing.txt", message="file_not_found"))
        assert r.content is None
        assert r.error.code == ErrorCode.NOT_FOUND


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

            def grep(self, pattern, path="/", glob=None, **kwargs):
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

            def grep(self, pattern, path="/", glob=None, **kwargs):
                return GrepResult(matches=[])

            def glob(self, pattern, path="/"):
                return GlobResult(matches=[])

        b = TestBackend()
        desc = b.description
        assert "test" in desc.lower()
        assert "include_line_numbers" in desc
