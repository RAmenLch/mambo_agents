"""Tests for memory middleware in ``mambo_agents.middleware.memory``.

All tests use ``StateBackend`` (in-memory) — no real filesystem needed.
"""

from __future__ import annotations

import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import SystemMessage
from langgraph.runtime import Runtime

from mambo_agents.backends.state import StateBackend
from mambo_agents.backends.schemas import VirtualPath
from mambo_agents.middleware.memory import (
    MAMBO_MEMORY_SYSTEM_PROMPT,
    MamboMemoryMiddleware,
    MemoryState,
    _append_to_system_message,
    _default_format_prompt,
)
from tests.test_state_backend import _simulate_graph

# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------

MEMORY_PATH = "/.mambo/memory/AGENTS.md"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_runtime() -> Runtime:
    return Runtime(
        context=None,
        store=None,
        stream_writer=lambda v: None,
        previous=None,
        execution_info=None,
        server_info=None,
    )


def _make_request(
    system_message: SystemMessage,
    state: dict | None = None,
    *,
    model: str = "test-model",
) -> ModelRequest:
    return ModelRequest(
        system_message=system_message,
        messages=[],
        tools=[],
        state=state or {},
        model=model,
    )


# ============================================================================
# _append_to_system_message
# ============================================================================


class TestAppendToSystemMessage:
    def test_none_existing_creates_new(self):
        result = _append_to_system_message(None, "hello")
        assert isinstance(result, SystemMessage)
        assert result.content == "hello"

    def test_string_content_concatenates(self):
        existing = SystemMessage(content="base")
        result = _append_to_system_message(existing, "extra")
        assert result.content == "base\n\nextra"

    def test_list_content_appends_block(self):
        existing = SystemMessage(content=[{"type": "text", "text": "block1"}])
        result = _append_to_system_message(existing, "block2")
        blocks = result.content
        assert isinstance(blocks, list)
        assert len(blocks) == 2
        assert blocks[0] == {"type": "text", "text": "block1"}
        assert blocks[1] == {"type": "text", "text": "\n\nblock2"}


# ============================================================================
# _default_format_prompt
# ============================================================================


class TestDefaultFormatPrompt:
    def test_empty_dict_shows_no_memory(self):
        result = _default_format_prompt({})
        assert "<agent_memory>" in result
        assert "(No memory loaded)" in result

    def test_all_empty_values_shows_no_memory(self):
        result = _default_format_prompt({"/a.md": "", "/b.md": ""})
        assert "(No memory loaded)" in result

    def test_single_source(self):
        contents = {MEMORY_PATH: "## Rules\nUse Pydantic."}
        result = _default_format_prompt(contents)
        assert MEMORY_PATH in result
        assert "## Rules" in result
        assert "Use Pydantic." in result

    def test_multiple_sources_concatenated(self):
        contents = {
            "/a.md": "# A\nrule: A",
            "/b.md": "# B\nrule: B",
        }
        result = _default_format_prompt(contents)
        assert "# A" in result
        assert "rule: A" in result
        assert "# B" in result
        assert "rule: B" in result

    def test_mixed_empty_and_populated(self):
        contents = {
            "/a.md": "",
            "/b.md": "rule: C",
        }
        result = _default_format_prompt(contents)
        assert "(No memory loaded)" not in result
        assert "/b.md" in result
        assert "rule: C" in result


# ============================================================================
# MamboMemoryMiddleware — init
# ============================================================================


class TestMamboMemoryMiddlewareInit:
    def test_basic_init(self):
        backend = StateBackend()
        mw = MamboMemoryMiddleware(backend=backend, sources=[VirtualPath(MEMORY_PATH)])
        assert mw.sources == [VirtualPath(MEMORY_PATH)]
        assert mw.state_schema is MemoryState

    def test_multiple_sources(self):
        backend = StateBackend()
        mw = MamboMemoryMiddleware(
            backend=backend,
            sources=[VirtualPath(MEMORY_PATH), VirtualPath("/.mambo/memory/team.md")],
        )
        assert len(mw.sources) == 2

    def test_custom_format_prompt(self):
        backend = StateBackend()

        def custom_fmt(contents: dict[str, str]) -> str:
            return f"CUSTOM: {len(contents)} files loaded"

        mw = MamboMemoryMiddleware(
            backend=backend,
            sources=[VirtualPath("/test/AGENTS.md")],
            format_prompt=custom_fmt,
        )
        result = mw._format_prompt({"/test/AGENTS.md": "hello"})
        assert result == "CUSTOM: 1 files loaded"

    def test_factory_backend(self):
        backend = StateBackend()
        factory_called = []

        def factory(rt):
            factory_called.append(True)
            return backend

        mw = MamboMemoryMiddleware(
            backend=factory,
            sources=[VirtualPath(MEMORY_PATH)],
        )
        rt = _make_runtime()
        result = mw._get_backend({}, rt, {"configurable": {}})
        assert result is backend
        assert len(factory_called) == 1


# ============================================================================
# MamboMemoryMiddleware — before_agent
# ============================================================================


class TestBeforeAgent:
    def test_skips_when_already_loaded(self):
        backend = StateBackend()
        mw = MamboMemoryMiddleware(backend=backend, sources=[VirtualPath(MEMORY_PATH)])
        state: dict = {"memory_contents": {MEMORY_PATH: "cached"}}
        rt = _make_runtime()
        result = mw.before_agent(state, rt, {"configurable": {}})
        assert result is None

    def test_loads_file_from_backend(self):
        memory_text = "# My Memory\n\nThis is the agent memory."
        backend = StateBackend()
        with _simulate_graph(backend, thread_id="mem_test"):
            backend.write(VirtualPath(MEMORY_PATH), memory_text, overwrite=True)

        mw = MamboMemoryMiddleware(backend=backend, sources=[VirtualPath(MEMORY_PATH)])
        rt = _make_runtime()
        with _simulate_graph(backend, thread_id="mem_test"):
            result = mw.before_agent({}, rt, {"configurable": {}})

        assert result is not None
        assert "memory_contents" in result
        contents = result["memory_contents"]
        assert MEMORY_PATH in contents
        assert contents[MEMORY_PATH] == memory_text

    def test_file_not_found_silent_skip(self):
        backend = StateBackend()
        mw = MamboMemoryMiddleware(backend=backend, sources=[VirtualPath(MEMORY_PATH)])
        rt = _make_runtime()
        with _simulate_graph(backend, thread_id="mem_not_found"):
            result = mw.before_agent({}, rt, {"configurable": {}})

        assert result is not None
        assert "memory_contents" in result
        assert result["memory_contents"] == {}

    def test_partial_sources_some_found(self):
        backend = StateBackend()
        with _simulate_graph(backend, thread_id="mem_test"):
            backend.write(VirtualPath(MEMORY_PATH), "my memory", overwrite=True)

        mw = MamboMemoryMiddleware(
            backend=backend,
            sources=[VirtualPath("/.mambo/memory/AGENTS.md"), VirtualPath(MEMORY_PATH)],
        )
        rt = _make_runtime()
        with _simulate_graph(backend, thread_id="mem_test"):
            result = mw.before_agent({}, rt, {"configurable": {}})

        assert result is not None
        contents = result["memory_contents"]
        assert MEMORY_PATH in contents
        assert contents[MEMORY_PATH] == "my memory"

    def test_empty_sources_list(self):
        backend = StateBackend()
        mw = MamboMemoryMiddleware(backend=backend, sources=[])
        rt = _make_runtime()
        with _simulate_graph(backend, thread_id="mem_test"):
            result = mw.before_agent({}, rt, {"configurable": {}})

        assert result is not None
        assert result["memory_contents"] == {}


# ============================================================================
# MamboMemoryMiddleware — modify_request
# ============================================================================


class TestModifyRequest:
    def test_injects_memory_into_system_message(self):
        backend = StateBackend()
        mw = MamboMemoryMiddleware(backend=backend, sources=[VirtualPath(MEMORY_PATH)])

        req = _make_request(
            system_message=SystemMessage(content="base prompt"),
            state={
                "memory_contents": {MEMORY_PATH: "## Rules\n- Use Pydantic"}
            },
        )
        modified = mw.modify_request(req)
        content = str(modified.system_message.content)

        assert "base prompt" in content
        assert "<agent_memory>" in content
        assert "## Rules" in content
        assert "Use Pydantic" in content
        assert "<memory_guidelines>" in content

    def test_no_memory_contents_shows_empty(self):
        backend = StateBackend()
        mw = MamboMemoryMiddleware(backend=backend, sources=[VirtualPath(MEMORY_PATH)])

        req = _make_request(
            system_message=SystemMessage(content="base"),
            state={},
        )
        modified = mw.modify_request(req)
        content = str(modified.system_message.content)

        assert "base" in content
        assert "(No memory loaded)" in content

    def test_empty_memory_contents_shows_empty(self):
        backend = StateBackend()
        mw = MamboMemoryMiddleware(backend=backend, sources=[VirtualPath(MEMORY_PATH)])

        req = _make_request(
            system_message=SystemMessage(content="base"),
            state={"memory_contents": {}},
        )
        modified = mw.modify_request(req)
        content = str(modified.system_message.content)

        assert "(No memory loaded)" in content

    def test_custom_format_prompt_used(self):
        backend = StateBackend()
        format_calls = []

        def custom_fmt(contents: dict[str, str]) -> str:
            format_calls.append(contents)
            return "[CUSTOM MEMORY FORMAT]"

        mw = MamboMemoryMiddleware(
            backend=backend,
            sources=[VirtualPath("/test/AGENTS.md")],
            format_prompt=custom_fmt,
        )

        req = _make_request(
            system_message=SystemMessage(content="base"),
            state={"memory_contents": {"/test/AGENTS.md": "data"}},
        )
        modified = mw.modify_request(req)
        content = str(modified.system_message.content)

        assert "CUSTOM MEMORY FORMAT" in content
        assert len(format_calls) == 1
        assert format_calls[0] == {"/test/AGENTS.md": "data"}


# ============================================================================
# MamboMemoryMiddleware — wrap_model_call
# ============================================================================


class TestWrapModelCall:
    def test_handles_system_message_is_none(self):
        backend = StateBackend()
        mw = MamboMemoryMiddleware(backend=backend, sources=[VirtualPath(MEMORY_PATH)])

        req = _make_request(
            system_message=None,  # type: ignore[arg-type]
            state={"memory_contents": {MEMORY_PATH: "# Memory"}},
        )
        modified = mw.modify_request(req)
        assert modified.system_message is not None
        content = str(modified.system_message.content)
        assert "# Memory" in content


# ============================================================================
# MAMBO_MEMORY_SYSTEM_PROMPT
# ============================================================================


class TestMemorySystemPrompt:
    def test_prompt_contains_required_sections(self):
        prompt = MAMBO_MEMORY_SYSTEM_PROMPT
        assert "<agent_memory>" in prompt
        assert "<memory_guidelines>" in prompt
        assert "edit" in prompt or "write" in prompt
        assert "Learning from feedback" in prompt
        assert "When to update memories" in prompt
        assert "When to NOT update memories" in prompt

    def test_format_with_placeholder(self):
        formatted = MAMBO_MEMORY_SYSTEM_PROMPT.format(agent_memory="test content")
        assert "test content" in formatted
        assert "<agent_memory>" in formatted
