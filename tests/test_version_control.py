"""Tests for VersionControlMiddleware and VersionStore."""

from __future__ import annotations

import hashlib
from unittest.mock import MagicMock

import pytest
from langgraph.config import get_config
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.store.memory import InMemoryStore

from mambo_agents.backends.schemas import BackendError, ErrorCode, ReadResult, VirtualPath
from mambo_agents.middleware.version_control import (
    BackupEvent,
    VersionControlConfig,
    VersionControlMiddleware,
    VersionStore,
    _extract_file_path,
)


# ============================================================================
# Helpers
# ============================================================================


def _make_mock_backend():
    """Create a MagicMock backend for testing."""
    mock = MagicMock()
    mock.write.return_value = MagicMock(error=None)
    return mock


def _make_mock_backend_with_content(content: str):
    """Create a backend that returns *content* from read_raw."""
    mock = MagicMock()
    mock.read_raw.return_value = ReadResult(
        content=content,
        encoding="utf-8",
        file_type="text",
    )
    return mock


def _make_tool_call(name: str, args: dict | None = None) -> dict:
    """Simulate a tool-call dict from an AIMessage."""
    return {"name": name, "args": args or {}, "id": "call_test"}


class _FakeConfig:
    """Stand-in for the LangGraph config obtained via ``get_config()``."""

    def __init__(
        self,
        thread_id: str = "test-thread",
        *,
        configurable: dict[str, object] | None = None,
    ):
        self._thread_id = thread_id
        self._configurable = configurable

    def get(self, key, default=None):
        if key == "configurable":
            if self._configurable is not None:
                return self._configurable
            return {"thread_id": self._thread_id}
        return default


# ============================================================================
# VersionStore
# ============================================================================


class TestVersionStore:
    """Unit tests for :class:`VersionStore`."""

    def test_add_file_to_snapshot_creates_snapshot(self):
        store = VersionStore(store=InMemoryStore())
        store.save_blob("t1", "abc123", "hello world")
        store.add_file_to_snapshot("t1", "cp_001", "/workspace/src/main.py", "abc123")

        snapshots = store.list_snapshots("t1")
        assert len(snapshots) == 1
        assert snapshots[0].checkpoint_id == "cp_001"
        assert snapshots[0].file_blobs["/workspace/src/main.py"] == "abc123"

    def test_list_snapshots_chronological_order(self):
        store = VersionStore(store=InMemoryStore())
        store.add_file_to_snapshot("t1", "cp_a", "/f1.py", "sha_a")
        store.add_file_to_snapshot("t1", "cp_b", "/f2.py", "sha_b")

        snapshots = store.list_snapshots("t1")
        assert [s.checkpoint_id for s in snapshots] == ["cp_a", "cp_b"]

    def test_get_file_retrieves_blob(self):
        store = VersionStore(store=InMemoryStore())
        store.save_blob("t1", "sha_x", "file content here")
        store.add_file_to_snapshot("t1", "cp_1", "/workspace/file.py", "sha_x")

        content = store.get_file("t1", "cp_1", "/workspace/file.py")
        assert content == "file content here"

    def test_get_file_returns_none_for_missing(self):
        store = VersionStore(store=InMemoryStore())
        assert store.get_file("t1", "cp_1", "/does/not/exist") is None

    def test_get_changed_files(self):
        store = VersionStore(store=InMemoryStore())
        store.add_file_to_snapshot("t1", "cp_1", "/a.py", "sha_a")
        store.add_file_to_snapshot("t1", "cp_1", "/b.py", "sha_b")

        files = store.get_changed_files("t1", "cp_1")
        assert set(files) == {"/a.py", "/b.py"}

    def test_save_blob_noop_when_exists(self):
        store = VersionStore(store=InMemoryStore())
        store.save_blob("t1", "sha_x", "original")
        # Write again with different content — should be no-op
        store.save_blob("t1", "sha_x", "different")
        content = store._read_blob("t1", "sha_x")
        assert content == "original"

    def test_sha256_deduplication(self):
        """Same content produces same SHA — stored once."""
        content = "identical content"
        expected_sha = hashlib.sha256(content.encode("utf-8")).hexdigest()

        store = VersionStore(store=InMemoryStore())
        store.save_blob("t1", expected_sha, content)
        store.add_file_to_snapshot("t1", "cp_1", "/a.py", expected_sha)
        store.add_file_to_snapshot("t1", "cp_1", "/b.py", expected_sha)

        # Both paths point to the same blob SHA
        snapshots = store.list_snapshots("t1")
        assert len(snapshots) == 1
        assert snapshots[0].file_blobs["/a.py"] == expected_sha
        assert snapshots[0].file_blobs["/b.py"] == expected_sha

    def test_empty_thread_returns_empty_list(self):
        store = VersionStore(store=InMemoryStore())
        assert store.list_snapshots("nonexistent") == []

    def test_persistence_across_store_instances(self):
        """Data written by one store is readable by another sharing the same BaseStore."""
        backend_store = InMemoryStore()
        s1 = VersionStore(store=backend_store)
        s1.save_blob("t1", "sha_x", "persistent data")
        s1.add_file_to_snapshot("t1", "cp_1", "/f.py", "sha_x")

        s2 = VersionStore(store=backend_store)
        assert s2.get_file("t1", "cp_1", "/f.py") == "persistent data"


# ============================================================================
# VersionStore — session-level queries
# ============================================================================


class TestSessionQueries:
    """High-level queries: get_all_changed_files, get_latest_*."""

    def test_get_all_changed_files_deduplicates(self):
        """Files changed across multiple checkpoints are merged + deduplicated."""
        store = VersionStore(store=InMemoryStore())
        store.add_file_to_snapshot("t1", "cp_a", "/a.py", "sha_a")
        store.add_file_to_snapshot("t1", "cp_a", "/b.py", "sha_b")
        store.add_file_to_snapshot("t1", "cp_b", "/a.py", "sha_a2")  # same file, later checkpoint

        all_files = store.get_all_changed_files("t1")
        assert all_files == frozenset({"/a.py", "/b.py"})

    def test_get_all_changed_files_empty_thread(self):
        store = VersionStore(store=InMemoryStore())
        assert store.get_all_changed_files("t1") == frozenset()

    def test_get_latest_snapshot(self):
        store = VersionStore(store=InMemoryStore())
        store.add_file_to_snapshot("t1", "cp_1", "/f1.py", "sha1")
        store.add_file_to_snapshot("t1", "cp_2", "/f2.py", "sha2")

        latest = store.get_latest_snapshot("t1")
        assert latest is not None
        assert latest.checkpoint_id == "cp_2"
        assert latest.file_blobs == {"/f2.py": "sha2"}

    def test_get_latest_snapshot_empty_thread(self):
        store = VersionStore(store=InMemoryStore())
        assert store.get_latest_snapshot("t1") is None

    def test_get_latest_changed_files(self):
        store = VersionStore(store=InMemoryStore())
        store.add_file_to_snapshot("t1", "cp_1", "/old.py", "sha_old")
        store.add_file_to_snapshot("t1", "cp_2", "/recent.py", "sha_recent")

        files = store.get_latest_changed_files("t1")
        assert files == ["/recent.py"]

    def test_get_latest_changed_files_empty_thread(self):
        store = VersionStore(store=InMemoryStore())
        assert store.get_latest_changed_files("t1") == []


# ============================================================================
# VersionControlMiddleware — whitelist
# ============================================================================


class TestWhitelist:
    """Tests for whitelist-mode filtering."""

    def test_is_in_whitelist_matches(self):
        backend = _make_mock_backend()
        store = VersionStore()
        mw = VersionControlMiddleware(
            store=store,
            backend=backend,
            whitelist_folders=[VirtualPath("/workspace/src"), VirtualPath("/workspace/tests")],
        )
        assert mw._is_in_whitelist(VirtualPath("/workspace/src/main.py")) is True
        assert mw._is_in_whitelist(VirtualPath("/workspace/tests/test_x.py")) is True
        assert mw._is_in_whitelist(VirtualPath("/workspace/src/deep/nested/file.py")) is True

    def test_is_in_whitelist_rejects(self):
        backend = _make_mock_backend()
        store = VersionStore()
        mw = VersionControlMiddleware(
            store=store,
            backend=backend,
            whitelist_folders=[VirtualPath("/workspace/src")],
        )
        assert mw._is_in_whitelist(VirtualPath("/workspace/README.md")) is False
        assert mw._is_in_whitelist(VirtualPath("/workspace/other/file.py")) is False

    def test_is_in_whitelist_empty_means_nothing_passes(self):
        backend = _make_mock_backend()
        store = VersionStore()
        mw = VersionControlMiddleware(
            store=store,
            backend=backend,
            whitelist_folders=[],  # empty → strict mode
        )
        assert mw._is_in_whitelist(VirtualPath("/workspace/anything.py")) is False

    def test_is_in_whitelist_trailing_slash_normalized(self):
        """Trailing slash on folder is normalized away."""
        backend = _make_mock_backend()
        store = VersionStore()
        mw = VersionControlMiddleware(
            store=store,
            backend=backend,
            whitelist_folders=[VirtualPath("/workspace/src/")],  # trailing slash
        )
        assert mw._is_in_whitelist(VirtualPath("/workspace/src/main.py")) is True

    def test_none_folders_defaults_to_empty(self):
        """None → empty whitelist → nothing passes."""
        backend = _make_mock_backend()
        store = VersionStore()
        mw = VersionControlMiddleware(
            store=store,
            backend=backend,
        )
        assert mw._is_in_whitelist(VirtualPath("/workspace/anything.py")) is False


# ============================================================================
# VersionControlMiddleware — backup
# ============================================================================


class TestBackup:
    """Tests for automatic pre-mutation backup."""

    def test_backup_file_in_whitelist(self, monkeypatch):
        """File under whitelisted folder → backed up."""
        backend = _make_mock_backend_with_content("original content")

        store = VersionStore(store=InMemoryStore())
        mw = VersionControlMiddleware(
            store=store,
            backend=backend,
            whitelist_folders=[VirtualPath("/workspace/src")],
        )
        # Simulate graph context
        cfg = _FakeConfig("t1")
        monkeypatch.setattr(
            "mambo_agents.middleware.version_control.get_config",
            lambda: cfg,
        )
        monkeypatch.setattr(
            "mambo_agents.middleware.version_control.get_stream_writer",
            lambda: None,
        )
        mw._current_parent_cp = {"t1": "cp_backup"}
        mw._backed_up = {"t1": set()}

        mw._backup_file(VirtualPath("/workspace/src/main.py"))

        # Verify blob + snapshot exist
        content = store.get_file("t1", "cp_backup", "/workspace/src/main.py")
        assert content == "original content"

    def test_backup_file_not_in_whitelist_is_skipped(self, monkeypatch):
        """File outside whitelist → NO backup, no backend read."""
        backend = _make_mock_backend()
        backend.read_raw.return_value = ReadResult(
            content="should not read", encoding="utf-8", file_type="text",
        )

        store = VersionStore(store=InMemoryStore())
        mw = VersionControlMiddleware(
            store=store,
            backend=backend,
            whitelist_folders=[VirtualPath("/workspace/src")],
        )
        cfg = _FakeConfig("t1")
        monkeypatch.setattr(
            "mambo_agents.middleware.version_control.get_config",
            lambda: cfg,
        )
        mw._current_parent_cp = {"t1": "cp_skipped"}

        mw._backup_file(VirtualPath("/workspace/other/ignored.py"))

        # Backend was never read
        backend.read_raw.assert_not_called()
        # No snapshots created
        assert store.list_snapshots("t1") == []

    def test_backup_file_noop_when_already_backed_up(self, monkeypatch):
        """Same file in same invoke → only backed up once."""
        backend = _make_mock_backend_with_content("some text")

        store = VersionStore(store=InMemoryStore())
        mw = VersionControlMiddleware(
            store=store,
            backend=backend,
            whitelist_folders=[VirtualPath("/workspace")],
        )
        cfg = _FakeConfig("t1")
        monkeypatch.setattr(
            "mambo_agents.middleware.version_control.get_config",
            lambda: cfg,
        )
        monkeypatch.setattr(
            "mambo_agents.middleware.version_control.get_stream_writer",
            lambda: None,
        )
        mw._current_parent_cp = {"t1": "cp_once"}
        mw._backed_up = {"t1": set()}

        mw._backup_file(VirtualPath("/workspace/file.py"))
        call_count = backend.read_raw.call_count
        mw._backup_file(VirtualPath("/workspace/file.py"))
        # Backend read only once
        assert backend.read_raw.call_count == call_count

    def test_backup_file_skips_on_read_error(self, monkeypatch):
        """If read_raw returns an error, no backup."""
        backend = _make_mock_backend()
        backend.read_raw.return_value = ReadResult(
            error=BackendError(code=ErrorCode.NOT_FOUND, message="file not found"),
            content=None,
        )

        store = VersionStore(store=InMemoryStore())
        mw = VersionControlMiddleware(
            store=store,
            backend=backend,
            whitelist_folders=[VirtualPath("/workspace")],
        )
        cfg = _FakeConfig("t1")
        monkeypatch.setattr(
            "mambo_agents.middleware.version_control.get_config",
            lambda: cfg,
        )
        mw._current_parent_cp = {"t1": "cp_err"}

        mw._backup_file(VirtualPath("/workspace/missing.py"))
        assert store.list_snapshots("t1") == []

    def test_backup_file_skips_on_none_content(self, monkeypatch):
        """If read_raw returns None content, no backup."""
        backend = _make_mock_backend()
        backend.read_raw.return_value = ReadResult(content=None)

        store = VersionStore(store=InMemoryStore())
        mw = VersionControlMiddleware(
            store=store,
            backend=backend,
            whitelist_folders=[VirtualPath("/workspace")],
        )
        cfg = _FakeConfig("t1")
        monkeypatch.setattr(
            "mambo_agents.middleware.version_control.get_config",
            lambda: cfg,
        )
        mw._current_parent_cp = {"t1": "cp_none"}

        mw._backup_file(VirtualPath("/workspace/empty.py"))
        assert store.list_snapshots("t1") == []


# ============================================================================
# VersionControlMiddleware — wrap_tool_call
# ============================================================================


class TestWrapToolCall:
    """Tests for wrap_tool_call hook."""

    def test_mutating_tool_triggers_backup(self, monkeypatch):
        """write tool under whitelist → backup happens."""
        backend = _make_mock_backend_with_content("before write")

        store = VersionStore(store=InMemoryStore())
        mw = VersionControlMiddleware(
            store=store,
            backend=backend,
            whitelist_folders=[VirtualPath("/workspace/src")],
        )
        cfg = _FakeConfig("t1")
        monkeypatch.setattr(
            "mambo_agents.middleware.version_control.get_config",
            lambda: cfg,
        )
        monkeypatch.setattr(
            "mambo_agents.middleware.version_control.get_stream_writer",
            lambda: None,
        )
        mw._current_parent_cp = {"t1": "cp_tool"}

        tool_call = _make_tool_call("write", {"file_path": "/workspace/src/main.py", "content": "x"})
        request = ToolCallRequest(
            tool_call=tool_call,
            tool=None,
            state={},
            runtime=MagicMock(),
        )
        mw._backed_up = {"t1": set()}

        def handler(req):
            return "write_result"

        result = mw.wrap_tool_call(request, handler)
        assert result == "write_result"

        files = store.get_changed_files("t1", "cp_tool")
        assert "/workspace/src/main.py" in files

    def test_mutating_tool_outside_whitelist_no_backup(self, monkeypatch):
        """write tool outside whitelist → no backup."""
        backend = _make_mock_backend()

        store = VersionStore(store=InMemoryStore())
        mw = VersionControlMiddleware(
            store=store,
            backend=backend,
            whitelist_folders=[VirtualPath("/workspace/src")],
        )
        cfg = _FakeConfig("t1")
        monkeypatch.setattr(
            "mambo_agents.middleware.version_control.get_config",
            lambda: cfg,
        )
        mw._current_parent_cp = {"t1": "cp_outside"}

        tool_call = _make_tool_call("write", {"file_path": "/workspace/other/README.md"})
        request = ToolCallRequest(
            tool_call=tool_call,
            tool=None,
            state={},
            runtime=MagicMock(),
        )

        def handler(req):
            return "ok"

        mw.wrap_tool_call(request, handler)
        # Backend was never read
        backend.read_raw.assert_not_called()
        assert store.list_snapshots("t1") == []

    def test_non_mutating_tool_no_backup(self, monkeypatch):
        """read tool → no backup triggered."""
        backend = _make_mock_backend()

        store = VersionStore(store=InMemoryStore())
        mw = VersionControlMiddleware(
            store=store,
            backend=backend,
            whitelist_folders=[VirtualPath("/workspace")],
        )
        cfg = _FakeConfig("t1")
        monkeypatch.setattr(
            "mambo_agents.middleware.version_control.get_config",
            lambda: cfg,
        )

        tool_call = _make_tool_call("read", {"file_path": "/workspace/file.py"})
        request = ToolCallRequest(
            tool_call=tool_call,
            tool=None,
            state={},
            runtime=MagicMock(),
        )

        def handler(req):
            return "read_result"

        mw.wrap_tool_call(request, handler)
        backend.read_raw.assert_not_called()

    def test_custom_mutating_tool_names(self, monkeypatch):
        """Custom mutating tool names are respected."""
        backend = _make_mock_backend_with_content("patch me")

        store = VersionStore(store=InMemoryStore())
        mw = VersionControlMiddleware(
            store=store,
            backend=backend,
            whitelist_folders=[VirtualPath("/workspace")],
            mutating_tool_names=["write", "edit", "delete", "patch", "rename"],
        )
        cfg = _FakeConfig("t1")
        monkeypatch.setattr(
            "mambo_agents.middleware.version_control.get_config",
            lambda: cfg,
        )
        monkeypatch.setattr(
            "mambo_agents.middleware.version_control.get_stream_writer",
            lambda: None,
        )
        mw._current_parent_cp = {"t1": "cp_custom"}
        mw._backed_up = {"t1": set()}

        tool_call = _make_tool_call("patch", {"file_path": "/workspace/file.py"})
        request = ToolCallRequest(
            tool_call=tool_call,
            tool=None,
            state={},
            runtime=MagicMock(),
        )

        def handler(req):
            return "ok"

        mw.wrap_tool_call(request, handler)
        files = store.get_changed_files("t1", "cp_custom")
        assert "/workspace/file.py" in files


# ============================================================================
# VersionControlMiddleware — before_agent
# ============================================================================


class TestBeforeAgent:
    """Tests for before_agent (checkpoint recording)."""

    def test_before_agent_records_parent_checkpoint(self, monkeypatch):
        backend = _make_mock_backend()
        store = VersionStore()

        mw = VersionControlMiddleware(
            store=store,
            backend=backend,
            whitelist_folders=[VirtualPath("/workspace")],
        )

        configurable = {
            "thread_id": "t_before",
            "version_control_ckpt_id": "cp_from_user",
        }
        cfg = _FakeConfig(thread_id="t_before", configurable=configurable)
        monkeypatch.setattr(
            "mambo_agents.middleware.version_control.get_config",
            lambda: cfg,
        )
        mw.before_agent(state={}, runtime=MagicMock())

        assert mw._current_parent_cp["t_before"] == "cp_from_user"
        assert mw._backed_up["t_before"] == set()

    def test_before_agent_default_thread_id(self, monkeypatch):
        backend = _make_mock_backend()
        store = VersionStore()

        mw = VersionControlMiddleware(
            store=store,
            backend=backend,
            whitelist_folders=[VirtualPath("/workspace")],
        )

        cfg = _FakeConfig(configurable={})
        monkeypatch.setattr(
            "mambo_agents.middleware.version_control.get_config",
            lambda: cfg,
        )
        mw.before_agent(state={}, runtime=MagicMock())
        assert mw._current_parent_cp["__default__"] == "__initial__"


# ============================================================================
# Config models
# ============================================================================


class TestVersionControlConfig:
    """Tests for :class:`VersionControlConfig`."""

    def test_defaults(self):
        cfg = VersionControlConfig()
        assert cfg.store is None
        assert cfg.whitelist_folders == []
        assert cfg.mutating_tool_names == ["write", "edit", "delete"]

    def test_custom_values(self):
        store = InMemoryStore()
        cfg = VersionControlConfig(
            store=store,
            whitelist_folders=[VirtualPath("/workspace/src")],
            mutating_tool_names=["write", "patch"],
        )
        assert cfg.store is store
        assert cfg.whitelist_folders == [VirtualPath("/workspace/src")]
        assert cfg.mutating_tool_names == ["write", "patch"]


# ============================================================================
# Module-level helpers
# ============================================================================


class TestExtractFilePath:
    """Tests for :func:`_extract_file_path`."""

    def test_extracts_file_path(self):
        tc = _make_tool_call("write", {"file_path": "/workspace/src/main.py"})
        vp = _extract_file_path(tc)
        assert vp is not None
        assert vp.value == "/workspace/src/main.py"

    def test_extracts_path_fallback(self):
        tc = _make_tool_call("delete", {"path": "/workspace/old.py"})
        vp = _extract_file_path(tc)
        assert vp is not None
        assert vp.value == "/workspace/old.py"

    def test_file_path_priority_over_path(self):
        tc = _make_tool_call("write", {
            "file_path": "/workspace/primary.py",
            "path": "/workspace/fallback.py",
        })
        vp = _extract_file_path(tc)
        assert vp is not None
        assert vp.value == "/workspace/primary.py"

    def test_no_path_returns_none(self):
        tc = _make_tool_call("read", {"limit": 100})
        assert _extract_file_path(tc) is None

    def test_invalid_path_returns_none(self):
        tc = _make_tool_call("write", {"file_path": "../escape"})
        assert _extract_file_path(tc) is None


# ============================================================================
# _resolve_parent_checkpoint_id
# ============================================================================


class TestResolveParentCheckpointId:
    """Tests for :meth:`VersionControlMiddleware._resolve_parent_checkpoint_id`."""

    def test_direct_checkpoint_id(self):
        config = {"metadata": {"checkpoint_id": "cp_direct"}, "configurable": {}}
        result = VersionControlMiddleware._resolve_parent_checkpoint_id(config)
        assert result == "cp_direct"

    def test_checkpoint_map_root(self):
        config = {
            "configurable": {
                "checkpoint_ns": "",
                "checkpoint_map": {"": "cp_from_map"},
            }
        }
        result = VersionControlMiddleware._resolve_parent_checkpoint_id(config)
        assert result == "cp_from_map"

    def test_fallback_to_initial(self):
        config = {"configurable": {}}
        result = VersionControlMiddleware._resolve_parent_checkpoint_id(config)
        assert result == "__initial__"
