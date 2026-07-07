"""Tests for skills middleware pure functions and core logic in ``mambo_agents.middleware.skills``.

All tests use ``StoreBackend`` (in-memory) — no real filesystem needed.
"""

from unittest.mock import MagicMock

import pytest
from langgraph.store.memory import InMemoryStore

from mambo_agents.backends.store import StoreBackend
from mambo_agents.backends.schemas import VirtualPath
from mambo_agents.middleware.skills import (
    MAX_SKILL_DESCRIPTION_LENGTH,
    MAX_SKILL_FILE_SIZE,
    MAX_SKILL_NAME_LENGTH,
    MAX_SKILLS_LOAD_WARNINGS,
    SkillMetadata,
    SkillsMiddleware,
    _derive_source_label,
    _format_skill_annotations,
    _parse_skill_metadata,
    _source_path,
    _truncate_skill_load_warning,
    _validate_metadata,
    _validate_module_path,
    _validate_skill_name,
    _validate_tuple_source,
    to_posix_path,
)
from tests.test_store_backend import _simulate_graph


# ============================================================================
# Source helpers
# ============================================================================


class TestSourcePath:
    def test_bare_string(self):
        assert _source_path("/skills/user/") == "/skills/user/"

    def test_tuple(self):
        assert _source_path(("/skills/project/", "Project")) == "/skills/project/"

    def test_invalid_tuple_raises(self):
        with pytest.raises(TypeError, match="Invalid skill source"):
            _source_path(("path", "label", "extra"))  # type: ignore[arg-type]

    def test_non_string_tuple_parts_raises(self):
        with pytest.raises(TypeError, match="Invalid skill source"):
            _source_path(("/path", 123))  # type: ignore[arg-type]


class TestValidateTupleSource:
    def test_valid(self):
        _validate_tuple_source(("/path", "Label"))  # should not raise

    def test_invalid_length(self):
        with pytest.raises(TypeError):
            _validate_tuple_source(("only",))

    def test_non_string_parts(self):
        with pytest.raises(TypeError):
            _validate_tuple_source((123, "label"))


class TestToPosixPath:
    def test_backslashes_converted(self):
        assert to_posix_path(r"C:\Users\test") == "C:/Users/test"

    def test_already_posix(self):
        assert to_posix_path("/skills/user") == "/skills/user"


class TestDeriveSourceLabel:
    def test_tuple_returns_label(self):
        assert _derive_source_label(("/path", "My Label")) == "My Label"

    def test_bare_path_uses_last_component(self):
        assert _derive_source_label("/skills/user/") == "User"

    def test_built_in_skills_returns_built_in(self):
        assert _derive_source_label("/bla/built_in_skills") == "Built-in"

    def test_skills_leaf_climbs(self):
        """When leaf is 'skills', label is derived from parent."""
        assert _derive_source_label("/home/user/.claude/skills") == "Claude"

    def test_empty_parts_returns_unnamed(self):
        assert _derive_source_label("/") == "Unnamed"


# ============================================================================
# Warning truncation
# ============================================================================


class TestTruncateSkillLoadWarning:
    def test_short_message_unchanged(self):
        msg = "Short error"
        assert _truncate_skill_load_warning(msg) == "Short error"

    def test_long_message_truncated(self):
        msg = "X" * 2000  # > 1000
        result = _truncate_skill_load_warning(msg)
        assert len(result) <= 1000 + 3  # truncated + suffix
        assert "... [truncated]" in result


# ============================================================================
# Validation
# ============================================================================


class TestValidateSkillName:
    def test_valid_name_matches_dir(self):
        ok, err = _validate_skill_name("web-research", "web-research")
        assert ok is True
        assert err == ""

    def test_empty_name(self):
        ok, err = _validate_skill_name("", "dir")
        assert ok is False
        assert "required" in err

    def test_too_long(self):
        name = "a" * (MAX_SKILL_NAME_LENGTH + 1)
        ok, err = _validate_skill_name(name, "dir")
        assert ok is False
        assert "64 characters" in err

    def test_starts_with_hyphen(self):
        ok, err = _validate_skill_name("-bad", "-bad")
        assert ok is False
        assert "lowercase alphanumeric" in err.lower()

    def test_double_hyphen(self):
        ok, err = _validate_skill_name("a--b", "a--b")
        assert ok is False
        assert "lowercase alphanumeric" in err.lower()

    def test_uppercase(self):
        ok, err = _validate_skill_name("Web-Research", "Web-Research")
        assert ok is False
        assert "lowercase" in err.lower()

    def test_name_mismatch_directory(self):
        ok, err = _validate_skill_name("web-research", "research")
        assert ok is False
        assert "must match directory name" in err


class TestValidateMetadata:
    def test_dict_converted_to_str_keys(self):
        raw = {"key1": 123, "key2": True}
        result = _validate_metadata(raw, "/test")
        assert result == {"key1": "123", "key2": "True"}

    def test_non_dict_returns_empty(self):
        assert _validate_metadata([1, 2, 3], "/test") == {}
        assert _validate_metadata("string", "/test") == {}

    def test_none_returns_empty(self):
        assert _validate_metadata(None, "/test") == {}


class TestValidateModulePath:
    def test_none_returns_none(self):
        assert _validate_module_path(None, "/test") is None

    def test_non_string_warns(self):
        assert _validate_module_path(123, "/test") is None

    def test_empty_string_returns_none(self):
        assert _validate_module_path("", "/test") is None
        assert _validate_module_path("  ", "/test") is None

    def test_valid_js_module(self):
        result = _validate_module_path("./helper.js", "/test")
        assert result == "helper.js"

    def test_valid_ts_module(self):
        result = _validate_module_path("index.ts", "/test")
        assert result == "index.ts"

    def test_absolute_path_rejected(self):
        result = _validate_module_path("/etc/passwd.js", "/test")
        assert result is None

    def test_path_escape_rejected(self):
        result = _validate_module_path("../secrets.js", "/test")
        assert result is None

    def test_nested_escape_rejected(self):
        result = _validate_module_path("a/../b.js", "/test")
        assert result is None

    def test_invalid_extension(self):
        result = _validate_module_path("script.py", "/test")
        assert result is None


# ============================================================================
# Skill metadata parsing
# ============================================================================


_VALID_FRONTMATTER = """---
name: web-research
description: Research topics online
license: MIT
---
# Research Skill
Instructions here...
"""

_MINIMAL_FRONTMATTER = """---
name: web-research
description: desc
---
"""

_FRONTMATTER_WITH_TOOLS = """---
name: web-research
description: Research
allowed-tools: grep read write
---
"""

_FRONTMATTER_WITH_MODULE = """---
name: custom
description: Custom skill
module: helper.js
---
"""


class TestParseSkillMetadata:
    def test_valid_frontmatter(self):
        result = _parse_skill_metadata(
            _VALID_FRONTMATTER, "/skills/user/web-research/SKILL.md", "web-research"
        )
        assert result is not None
        assert result.name == "web-research"
        assert result.description == "Research topics online"
        assert result.license == "MIT"
        assert result.path == "/skills/user/web-research/SKILL.md"

    def test_name_must_match_directory(self):
        """Name mismatch with directory → warning but still returns metadata for backwards compat."""
        result = _parse_skill_metadata(
            _MINIMAL_FRONTMATTER, "/skills/user/research/SKILL.md", "research"
        )
        # The implementation logs a warning but still returns metadata
        assert result is not None
        assert result.name == "web-research"

    def test_no_frontmatter(self):
        result = _parse_skill_metadata(
            "Just markdown, no frontmatter", "/path/SKILL.md", "dir"
        )
        assert result is None

    def test_invalid_yaml(self):
        content = "---\nname: [bad yaml\n---\n"
        result = _parse_skill_metadata(content, "/path/SKILL.md", "dir")
        assert result is None

    def test_frontmatter_not_a_dict(self):
        content = "---\n- list item\n---\n"
        result = _parse_skill_metadata(content, "/path/SKILL.md", "dir")
        assert result is None

    def test_missing_name(self):
        content = "---\ndescription: desc\n---\n"
        result = _parse_skill_metadata(content, "/path/SKILL.md", "dir")
        assert result is None

    def test_missing_description(self):
        content = "---\nname: test\n---\n"
        result = _parse_skill_metadata(content, "/path/SKILL.md", "test")
        assert result is None

    def test_content_too_large(self):
        huge = "---\nname: x\ndescription: d\n---\n" + "x" * (MAX_SKILL_FILE_SIZE + 1)
        result = _parse_skill_metadata(huge, "/path/SKILL.md", "x")
        assert result is None

    def test_description_truncated(self):
        long_desc = "D" * (MAX_SKILL_DESCRIPTION_LENGTH + 100)
        content = f"---\nname: test\ndescription: {long_desc}\n---\n"
        result = _parse_skill_metadata(content, "/path/SKILL.md", "test")
        assert result is not None
        assert len(result.description) == MAX_SKILL_DESCRIPTION_LENGTH

    def test_allowed_tools_parsed(self):
        result = _parse_skill_metadata(
            _FRONTMATTER_WITH_TOOLS, "/skills/user/web-research/SKILL.md", "web-research"
        )
        assert result is not None
        assert result.allowed_tools == ["grep", "read", "write"]

    def test_allowed_tools_non_string_ignored(self):
        content = """---
name: test
description: desc
allowed-tools: [1, 2, 3]
---
"""
        result = _parse_skill_metadata(content, "/path/SKILL.md", "test")
        assert result is not None
        assert result.allowed_tools == []

    def test_module_extracted(self):
        result = _parse_skill_metadata(
            _FRONTMATTER_WITH_MODULE, "/path/SKILL.md", "custom"
        )
        assert result is not None
        assert result.module == "helper.js"

    def test_compatibility_truncated(self):
        long_comp = "C" * 1000
        content = f"---\nname: test\ndescription: d\ncompatibility: {long_comp}\n---\n"
        result = _parse_skill_metadata(content, "/path/SKILL.md", "test")
        assert result is not None
        assert result.compatibility == "C" * 500

    def test_default_values(self):
        result = _parse_skill_metadata(
            _MINIMAL_FRONTMATTER, "/path/SKILL.md", "web-research"
        )
        assert result is not None
        assert result.license is None
        assert result.compatibility is None
        assert result.metadata == {}
        assert result.allowed_tools == []


# ============================================================================
# Formatting helpers
# ============================================================================


class TestFormatSkillAnnotations:
    def test_empty(self):
        skill = SkillMetadata(
            name="test",
            description="desc",
            path="/p/SKILL.md",
            license=None,
            compatibility=None,
            metadata={},
            allowed_tools=[],
        )
        assert _format_skill_annotations(skill) == ""

    def test_license_only(self):
        skill = SkillMetadata(
            name="test",
            description="desc",
            path="/p/SKILL.md",
            license="MIT",
            compatibility=None,
            metadata={},
            allowed_tools=[],
        )
        assert "License: MIT" in _format_skill_annotations(skill)

    def test_both(self):
        skill = SkillMetadata(
            name="test",
            description="desc",
            path="/p/SKILL.md",
            license="MIT",
            compatibility="python>=3.10",
            metadata={},
            allowed_tools=[],
        )
        result = _format_skill_annotations(skill)
        assert "License: MIT" in result
        assert "Compatibility: python>=3.10" in result


# ============================================================================
# SkillsMiddleware — initialization and formatting
# ============================================================================


class TestSkillsMiddlewareInit:
    def test_basic_init(self):
        backend = StoreBackend(store=InMemoryStore())
        mw = SkillsMiddleware(backend=backend, sources=["/skills/user/"])
        assert mw.sources == ["/skills/user/"]
        assert mw.source_labels == ["User"]
        assert mw.system_prompt_template is not None

    def test_multiple_sources(self):
        backend = StoreBackend(store=InMemoryStore())
        mw = SkillsMiddleware(
            backend=backend,
            sources=["/skills/base/", "/skills/project/"],
        )
        assert len(mw.sources) == 2
        assert len(mw.source_labels) == 2

    def test_mixed_source_types(self):
        backend = StoreBackend(store=InMemoryStore())
        mw = SkillsMiddleware(
            backend=backend,
            sources=[
                "/skills/base/",
                ("/skills/project/", "Custom Project"),
            ],
        )
        assert mw.source_labels == ["Base", "Custom Project"]


class TestFormatSkillsLocations:
    def test_single_source(self):
        backend = StoreBackend(store=InMemoryStore())
        mw = SkillsMiddleware(backend=backend, sources=["/skills/user/"])
        result = mw._format_skills_locations()
        assert "/skills/user/" in result
        assert "User" in result
        assert "higher priority" in result

    def test_multiple_sources_last_has_priority(self):
        backend = StoreBackend(store=InMemoryStore())
        mw = SkillsMiddleware(
            backend=backend,
            sources=["/skills/base/", "/skills/project/"],
        )
        result = mw._format_skills_locations()
        lines = result.split("\n")
        assert len(lines) == 2
        assert "higher priority" not in lines[0]
        assert "higher priority" in lines[1]


class TestFormatSkillsList:
    def test_empty_skills(self):
        backend = StoreBackend(store=InMemoryStore())
        mw = SkillsMiddleware(backend=backend, sources=["/skills/user/"])
        result = mw._format_skills_list([])
        assert "No skills available" in result

    def test_single_skill(self):
        backend = StoreBackend(store=InMemoryStore())
        mw = SkillsMiddleware(backend=backend, sources=["/skills/user/"])
        skill = SkillMetadata(
            name="research",
            description="Research topics",
            path="/skills/user/research/SKILL.md",
            license=None,
            compatibility=None,
            metadata={},
            allowed_tools=[],
        )
        result = mw._format_skills_list([skill])
        assert "research" in result
        assert "Research topics" in result
        assert "/skills/user/research/SKILL.md" in result

    def test_skill_with_allowed_tools(self):
        backend = StoreBackend(store=InMemoryStore())
        mw = SkillsMiddleware(backend=backend, sources=["/skills/user/"])
        skill = SkillMetadata(
            name="tool-user",
            description="Uses tools",
            path="/p/SKILL.md",
            license=None,
            compatibility=None,
            metadata={},
            allowed_tools=["grep", "read"],
        )
        result = mw._format_skills_list([skill])
        assert "grep, read" in result


class TestFormatSkillsLoadWarnings:
    def test_no_errors(self):
        backend = StoreBackend(store=InMemoryStore())
        mw = SkillsMiddleware(backend=backend, sources=[])
        assert mw._format_skills_load_warnings([]) == ""

    def test_with_errors(self):
        backend = StoreBackend(store=InMemoryStore())
        mw = SkillsMiddleware(backend=backend, sources=[])
        result = mw._format_skills_load_warnings(["Error 1", "Error 2"])
        assert "Skill Loading Warnings" in result
        assert "Error 1" in result
        assert "Error 2" in result

    def test_truncated_when_too_many(self):
        backend = StoreBackend(store=InMemoryStore())
        mw = SkillsMiddleware(backend=backend, sources=[])
        errors = [f"Error {i}" for i in range(MAX_SKILLS_LOAD_WARNINGS + 5)]
        result = mw._format_skills_load_warnings(errors)
        assert "5 additional" in result


# ============================================================================
# SkillsMiddleware — before_agent (loads skills from backend)
# ============================================================================


class TestBeforeAgent:
    def test_skips_when_already_loaded(self):
        """before_agent returns None when skills_metadata is already in state."""
        backend = StoreBackend(store=InMemoryStore())
        mw = SkillsMiddleware(backend=backend, sources=["/skills/user/"])
        state: dict = {"skills_metadata": []}
        # before_agent should early-return since skills_metadata already exists
        from langgraph.runtime import Runtime
        rt = Runtime(
            context=None, store=None, stream_writer=lambda v: None,
            previous=None, execution_info=None, server_info=None,
        )
        result = mw.before_agent(state, rt, {"configurable": {}})
        assert result is None

    def test_loads_skills_from_backend(self):
        """before_agent loads skills from backend sources."""
        backend = StoreBackend(store=InMemoryStore())
        skill_content = """---
name: test-skill
description: A test skill
---
# Test Skill
"""
        with _simulate_graph(backend, thread_id="skill_load"):
            backend.write(VirtualPath("/skills/user/test-skill/SKILL.md"), skill_content, overwrite=True)

        from langgraph.runtime import Runtime
        rt = Runtime(
            context=None, store=None, stream_writer=lambda v: None,
            previous=None, execution_info=None, server_info=None,
        )
        mw = SkillsMiddleware(backend=backend, sources=["/skills/user/"])
        state: dict = {}
        # before_agent needs graph context because it calls backend.ls()
        with _simulate_graph(backend, thread_id="skill_load_agent"):
            result = mw.before_agent(state, rt, {"configurable": {}})
        assert result is not None
        assert "skills_metadata" in result

    def test_empty_source_no_skills(self):
        """Empty source → no skills, no error."""
        backend = StoreBackend(store=InMemoryStore())
        mw = SkillsMiddleware(backend=backend, sources=[])

        from langgraph.runtime import Runtime
        rt = Runtime(
            context=None, store=None, stream_writer=lambda v: None,
            previous=None, execution_info=None, server_info=None,
        )
        result = mw.before_agent({}, rt, {"configurable": {}})
        assert result is not None
        assert "skills_metadata" in result
        assert result["skills_metadata"] == []


# ============================================================================
# modify_request / wrap_model_call
# ============================================================================


class TestModifyRequest:
    def test_injects_skills_section(self):
        """modify_request adds skills section to system message."""
        from langchain.agents.middleware.types import ModelRequest
        from langchain_core.messages import SystemMessage

        backend = StoreBackend(store=InMemoryStore())
        mw = SkillsMiddleware(backend=backend, sources=["/skills/user/"])

        req = ModelRequest(
            system_message=SystemMessage(content="base prompt"),
            messages=[],
            tools=[],
            state={
                "skills_metadata": [],
                "skills_load_errors": [],
            },
            model="test",
        )
        modified = mw.modify_request(req)
        content = str(modified.system_message.content)

        assert "Skills System" in content
        assert "/skills/user/" in content
