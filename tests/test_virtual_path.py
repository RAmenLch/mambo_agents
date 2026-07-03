"""Tests for VirtualPath — construction, validation, business methods, and Pydantic integration."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from mambo_agents.backends.schemas import BackendError, VirtualPath, check_no_path_traversal


# ============================================================================
# Construction — valid paths
# ============================================================================


class TestVirtualPathConstruction:
    def test_basic_absolute_path(self):
        vp = VirtualPath("/workspace/src/main.py")
        assert vp.value == "/workspace/src/main.py"

    def test_preserves_trailing_slash(self):
        vp = VirtualPath("/workspace/")
        assert vp.value == "/workspace/"

    def test_preserves_multiple_segments(self):
        vp = VirtualPath("/a/b/c/d")
        assert vp.value == "/a/b/c/d"

    def test_hidden_directory_prefix(self):
        """/.mambo/ paths are valid virtual paths."""
        vp = VirtualPath("/.mambo/skills/helper.py")
        assert vp.value == "/.mambo/skills/helper.py"

    def test_accepts_existing_virtual_path(self):
        vp1 = VirtualPath("/workspace/src/main.py")
        vp2 = VirtualPath(vp1)
        assert vp2 == vp1  # value-equal, may be a different instance
        assert vp2.value == "/workspace/src/main.py"


# ============================================================================
# Construction — invalid paths
# ============================================================================


class TestVirtualPathRejection:
    def test_empty_string(self):
        with pytest.raises(BackendError, match="不能为空"):
            VirtualPath("")

    def test_whitespace_only(self):
        with pytest.raises(BackendError, match="不能为空"):
            VirtualPath("   ")

    def test_root_path_rejected(self):
        with pytest.raises(BackendError, match="不能是根目录"):
            VirtualPath("/")

    def test_non_absolute(self):
        with pytest.raises(BackendError, match="必须以 '/' 开头"):
            VirtualPath("workspace/src")

    def test_relative_with_dot(self):
        with pytest.raises(BackendError, match="必须以 '/' 开头"):
            VirtualPath("./workspace")

    @pytest.mark.parametrize(
        "attack_path",
        [
            "/workspace/../../etc/passwd",
            "/workspace/../etc/passwd",
            "/../etc/passwd",
            "/workspace/sub/../../etc/passwd",
            "/workspace/../..",
            "/a/b/../../..",
        ],
    )
    def test_path_traversal_via_dotdot(self, attack_path):
        with pytest.raises(BackendError, match="不能包含 '..'"):
            VirtualPath(attack_path)

    @pytest.mark.parametrize(
        "double_slash_path",
        [
            "/workspace//src",
            "//workspace",
            "/workspace/src//main.py",
        ],
    )
    def test_double_slash_rejected(self, double_slash_path):
        with pytest.raises(BackendError, match="不能包含 '//'"):
            VirtualPath(double_slash_path)

    def test_non_string_type(self):
        with pytest.raises((BackendError, TypeError), match="Expected str or VirtualPath"):
            VirtualPath(123)  # type: ignore[arg-type]


# ============================================================================
# Business methods
# ============================================================================


class TestIsUnder:
    def test_exact_match(self):
        vp = VirtualPath("/workspace")
        assert vp.is_under("/workspace") is True

    def test_child_path(self):
        vp = VirtualPath("/workspace/src/main.py")
        assert vp.is_under("/workspace") is True

    def test_deeply_nested(self):
        vp = VirtualPath("/workspace/a/b/c/d.py")
        assert vp.is_under("/workspace") is True

    def test_not_under(self):
        vp = VirtualPath("/etc/passwd")
        assert vp.is_under("/workspace") is False

    def test_partial_prefix_not_confused(self):
        """'/workspaces' should NOT match '/workspace'."""
        vp = VirtualPath("/workspaces/file.py")
        assert vp.is_under("/workspace") is False

    def test_empty_prefix_slash(self):
        vp = VirtualPath("/anything")
        assert vp.is_under("/") is True


class TestRelativeTo:
    def test_child_relative(self):
        vp = VirtualPath("/workspace/src/main.py")
        assert vp.relative_to("/workspace") == "src/main.py"

    def test_exact_match_returns_empty(self):
        vp = VirtualPath("/workspace")
        assert vp.relative_to("/workspace") == ""

    def test_deeply_nested(self):
        vp = VirtualPath("/workspace/a/b/c/d.py")
        assert vp.relative_to("/workspace") == "a/b/c/d.py"

    def test_not_under_raises(self):
        vp = VirtualPath("/etc/passwd")
        with pytest.raises(BackendError, match="路径不在"):
            vp.relative_to("/workspace")


class TestJoin:
    def test_single_part(self):
        vp = VirtualPath("/workspace")
        result = vp.join("src")
        assert result.value == "/workspace/src"

    def test_multiple_parts(self):
        vp = VirtualPath("/workspace")
        result = vp.join("src", "main.py")
        assert result.value == "/workspace/src/main.py"

    def test_parts_with_slashes(self):
        vp = VirtualPath("/workspace")
        result = vp.join("/src/", "/main.py")
        assert result.value == "/workspace/src/main.py"

    def test_join_returns_virtual_path(self):
        vp = VirtualPath("/workspace")
        result = vp.join("src")
        assert isinstance(result, VirtualPath)
        # join validates the result, so invalid parts raise
        with pytest.raises(BackendError):
            vp.join("../escape")  # produces /workspace/../escape → rejected


# ============================================================================
# Hash and equality
# ============================================================================


class TestHashAndEquality:
    def test_same_value_equal(self):
        vp1 = VirtualPath("/workspace/src/main.py")
        vp2 = VirtualPath("/workspace/src/main.py")
        assert vp1 == vp2

    def test_different_values_not_equal(self):
        vp1 = VirtualPath("/workspace/a.py")
        vp2 = VirtualPath("/workspace/b.py")
        assert vp1 != vp2

    def test_equals_string(self):
        vp = VirtualPath("/workspace/src/main.py")
        assert vp == "/workspace/src/main.py"
        assert vp != "/workspace/other.py"

    def test_hashable_as_dict_key(self):
        vp = VirtualPath("/workspace/file.txt")
        d: dict[VirtualPath, str] = {vp: "value"}
        assert d[vp] == "value"

    def test_not_equal_to_other_types(self):
        vp = VirtualPath("/workspace")
        assert vp != 42
        assert vp != None


# ============================================================================
# Display
# ============================================================================


class TestDisplay:
    def test_str(self):
        vp = VirtualPath("/workspace/src/main.py")
        assert str(vp) == "/workspace/src/main.py"

    def test_repr(self):
        vp = VirtualPath("/workspace/src/main.py")
        assert repr(vp) == "VirtualPath('/workspace/src/main.py')"

    def test_f_string(self):
        vp = VirtualPath("/workspace/src")
        assert f"Path: {vp}" == "Path: /workspace/src"


# ============================================================================
# Pydantic model integration (str coercion)
# ============================================================================


class TestPydanticCoercion:
    class MySchema(BaseModel):
        file_path: VirtualPath
        path: VirtualPath

    def test_str_coerced_to_virtual_path(self):
        """Pydantic should automatically coerce str → VirtualPath via model_validator(mode='before')."""
        schema = self.MySchema(
            file_path="/workspace/file.txt",
            path="/workspace/subdir",
        )
        assert isinstance(schema.file_path, VirtualPath)
        assert schema.file_path.value == "/workspace/file.txt"

    def test_str_with_traversal_rejected(self):
        """Pydantic model should reject traversal paths during validation."""
        with pytest.raises(BackendError) as exc_info:
            self.MySchema(
                file_path="/workspace/../../etc/passwd",
                path="/workspace",
            )
        assert "不能包含 '..'" in str(exc_info.value)

    def test_virtual_path_passed_directly(self):
        """Existing VirtualPath should be accepted and value-preserved."""
        vp = VirtualPath("/workspace/file.txt")
        schema = self.MySchema(file_path=vp, path=vp)
        assert schema.file_path == vp
        assert schema.file_path.value == "/workspace/file.txt"


# ============================================================================
# Immutability
# ============================================================================


class TestImmutability:
    def test_frozen(self):
        vp = VirtualPath("/workspace/file.txt")
        with pytest.raises(Exception):
            vp.value = "/etc/passwd"  # type: ignore[misc]

    def test_hashing_consistent(self):
        vp = VirtualPath("/workspace/file.txt")
        h1 = hash(vp)
        h2 = hash(vp)
        assert h1 == h2


# ============================================================================
# check_no_path_traversal — unit
# ============================================================================


class TestCheckNoPathTraversal:
    def test_normal_path_passes(self):
        check_no_path_traversal("/workspace/src/main.py")  # should not raise

    def test_dotdot_raises(self):
        with pytest.raises(BackendError, match="不能包含 '..'"):
            check_no_path_traversal("/workspace/../../etc/passwd")

    def test_double_slash_raises(self):
        with pytest.raises(BackendError, match="不能包含 '//'"):
            check_no_path_traversal("/workspace//src")

    def test_custom_name(self):
        with pytest.raises(BackendError, match="不能包含 '..'"):
            check_no_path_traversal("/../etc/passwd", name="file_path")
