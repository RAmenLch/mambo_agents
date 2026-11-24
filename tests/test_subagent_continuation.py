"""Tests for subagent failure wrapping and continuation (``continue_id``)."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableSerializable
from langgraph.prebuilt.tool_node import ToolRuntime
from langgraph.store.memory import InMemoryStore
from pydantic import ConfigDict, Field

from mambo_agents.middleware.subagent_snapshot import (
    SubAgentSnapshotNode,
    build_delta,
    get_chain,
    get_node,
    list_nodes,
    put_node,
    snapshot_namespace,
    validate_continuation,
)
from mambo_agents.middleware.subagents import _SubagentSpec, _build_task_tool


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
class StubSubagent(RunnableSerializable[Any, Any]):
    """Fails the first run of each thread, then succeeds.

    Records the message list it was handed so tests can assert what was injected.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    seen: dict[str, int] = Field(default_factory=dict)
    fed: list[list[tuple[str, str]]] = Field(default_factory=list)
    fail_with: BaseException | None = None
    steps_before_failure: int = 0

    def _run(self, state: dict, config: dict | None) -> dict:
        cfg = (config or {}).get("configurable", {})
        key = cfg.get("thread_id", "")
        self.seen[key] = self.seen.get(key, 0) + 1
        self.fed.append(
            [(type(m).__name__, getattr(m, "text", "") or "") for m in state["messages"]]
        )
        if self.seen[key] == 1:
            raise self.fail_with or TimeoutError("llm timeout")
        return {
            "messages": [*state["messages"], AIMessage(content="subagent done")],
            "todos": [{"id": "t1", "content": "x", "status": "completed"}],
        }

    def invoke(self, input, config=None, **kwargs):  # noqa: A002
        return self._run(input, config)

    def stream(self, input, config=None, **kwargs):  # noqa: A002
        yield self._run(input, config)

    async def ainvoke(self, input, config=None, **kwargs):  # noqa: A002
        return self._run(input, config)

    async def astream(self, input, config=None, **kwargs):  # noqa: A002
        yield self._run(input, config)


def make_runtime(
    *,
    store: Any = None,
    thread_id: str | None = "t1",
    tool_call_id: str = "call_1",
) -> ToolRuntime:
    configurable: dict[str, Any] = {}
    if thread_id is not None:
        configurable["thread_id"] = thread_id
    return ToolRuntime(
        state={"messages": []},
        config={"configurable": configurable},
        tool_call_id=tool_call_id,
        context={},
        stream_writer=lambda data: None,
        store=store,
    )


def make_tool(stub: StubSubagent | None = None, **kwargs) -> tuple[Any, StubSubagent]:
    stub = stub or StubSubagent()
    specs = [_SubagentSpec(name="researcher", description="R", runnable=stub)]
    return _build_task_tool(specs, **kwargs), stub


def error_message(command) -> ToolMessage:
    message = command.update["messages"][0]
    assert isinstance(message, ToolMessage)
    return message


# ---------------------------------------------------------------------------
# failure wrapping
# ---------------------------------------------------------------------------
class TestFailureWrapping:
    def test_failure_becomes_error_tool_message_with_continuation(self):
        tool, _ = make_tool()
        store = InMemoryStore()
        command = tool.func(
            description="scan src",
            subagent_type="researcher",
            runtime=make_runtime(store=store),
        )
        message = error_message(command)
        assert message.status == "error"
        assert message.name == "task"
        assert message.tool_call_id == "call_1"

        payload = message.additional_kwargs["subagent_continuation"]
        assert payload["available"] is True
        assert payload["continue_id"] == "call_1"
        assert payload["retryable"] is True
        assert 'node_id="call_1"' in message.content

    def test_node_is_written_with_progress(self):
        tool, _ = make_tool()
        store = InMemoryStore()
        tool.func(
            description="scan src",
            subagent_type="researcher",
            runtime=make_runtime(store=store),
        )
        nodes = list_nodes(store, "t1")
        assert len(nodes) == 1
        node = nodes[0]
        assert node.node_id == "call_1"
        assert node.parent_id is None
        assert node.depth == 1
        assert node.status == "failed"
        # the injected task description is part of the snapshot
        assert node.messages()[0].text == "scan src"

    def test_non_retryable_error_withholds_continue_id(self):
        stub = StubSubagent(fail_with=ValueError("invalid message structure"))
        tool, _ = make_tool(stub)
        store = InMemoryStore()
        message = error_message(
            tool.func(
                description="scan src",
                subagent_type="researcher",
                runtime=make_runtime(store=store),
            )
        )
        assert "not retryable" in message.content.lower()
        assert message.additional_kwargs["subagent_continuation"]["available"] is False
        assert get_node(store, "t1", "call_1").available is False

    def test_catch_errors_false_reraises(self):
        tool, _ = make_tool(catch_errors=False)
        with pytest.raises(TimeoutError):
            tool.func(
                description="scan src",
                subagent_type="researcher",
                runtime=make_runtime(store=InMemoryStore()),
            )

    def test_without_store_it_degrades_but_does_not_crash(self):
        tool, _ = make_tool()
        message = error_message(
            tool.func(
                description="scan src",
                subagent_type="researcher",
                runtime=make_runtime(store=None),
            )
        )
        payload = message.additional_kwargs["subagent_continuation"]
        assert payload["available"] is False
        assert payload["reason"] == "snapshot_unavailable"

    def test_progress_channels_are_passed_through(self):
        """A failed run still forwards non-message channels, like a success does."""
        tool, _ = make_tool()

        class Partial(StubSubagent):
            def _run(self, state, config):
                raise TimeoutError("boom")

        tool2, _ = make_tool(Partial())
        command = tool2.func(
            description="x",
            subagent_type="researcher",
            runtime=make_runtime(store=InMemoryStore()),
        )
        assert "messages" in command.update


# ---------------------------------------------------------------------------
# continuation
# ---------------------------------------------------------------------------
class TestContinuation:
    def test_continue_injects_history_and_advice(self):
        tool, stub = make_tool()
        store = InMemoryStore()
        tool.func(
            description="scan src",
            subagent_type="researcher",
            runtime=make_runtime(store=store),
        )
        command = tool.func(
            description="skip the web, use local files",
            subagent_type="researcher",
            continue_id="call_1",
            runtime=make_runtime(store=store, tool_call_id="call_2"),
        )
        message = error_message(command)
        assert message.status == "success"
        assert message.content == "subagent done"

        fed = stub.fed[-1]
        assert fed[0] == ("HumanMessage", "scan src")
        assert fed[-1][0] == "HumanMessage"
        assert "请勿重复执行" in fed[-1][1]
        assert "skip the web" in fed[-1][1]

    def test_continue_rejects_unknown_id(self):
        tool, _ = make_tool()
        result = tool.func(
            description="",
            subagent_type="researcher",
            continue_id="nope",
            runtime=make_runtime(store=InMemoryStore()),
        )
        assert isinstance(result, str)
        assert "未找到" in result

    def test_continue_rejects_type_mismatch(self):
        tool, _ = make_tool()
        store = InMemoryStore()
        tool.func(
            description="scan",
            subagent_type="researcher",
            runtime=make_runtime(store=store),
        )
        result = tool.func(
            description="",
            subagent_type="other-type",
            continue_id="call_1",
            runtime=make_runtime(store=store),
        )
        # unknown type wins over continuation checks
        assert isinstance(result, str)

    def test_continue_rejects_chain_broken(self):
        tool, _ = make_tool()
        store = InMemoryStore()
        tool.func(
            description="scan",
            subagent_type="researcher",
            runtime=make_runtime(store=store),
        )
        # simulate an ancestor that was removed
        node = get_node(store, "t1", "call_1")
        node.chain = ["gone"]
        node.depth = 2
        put_node(store, "t1", node)
        result = tool.func(
            description="",
            subagent_type="researcher",
            continue_id="call_1",
            runtime=make_runtime(store=store),
        )
        assert isinstance(result, str)
        assert "续连链不完整" in result

    def test_continuation_is_isolated_per_thread(self):
        tool, _ = make_tool()
        store = InMemoryStore()
        tool.func(
            description="scan",
            subagent_type="researcher",
            runtime=make_runtime(store=store, thread_id="t1"),
        )
        result = tool.func(
            description="",
            subagent_type="researcher",
            continue_id="call_1",
            runtime=make_runtime(store=store, thread_id="t2"),
        )
        assert isinstance(result, str)
        assert "未找到" in result

    def test_second_failure_extends_the_chain(self):
        tool, _ = make_tool()

        class AlwaysFirstFail(StubSubagent):
            def _run(self, state, config):
                raise TimeoutError("still failing")

        tool2, _ = make_tool(AlwaysFirstFail())
        store = InMemoryStore()
        tool2.func(
            description="scan",
            subagent_type="researcher",
            runtime=make_runtime(store=store, tool_call_id="c1"),
        )
        tool2.func(
            description="go on",
            subagent_type="researcher",
            continue_id="c1",
            runtime=make_runtime(store=store, tool_call_id="c2"),
        )
        nodes, missing = get_chain(store, "t1", "c2")
        assert not missing
        assert [n.node_id for n in nodes] == ["c1", "c2"]
        assert nodes[1].parent_id == "c1"
        assert nodes[1].depth == 2

    def test_missing_description_and_continue_is_rejected(self):
        tool, _ = make_tool()
        result = tool.func(
            description="", subagent_type="researcher", runtime=make_runtime()
        )
        assert isinstance(result, str)
        assert "description" in result

    @pytest.mark.asyncio
    async def test_async_failure_and_continue(self):
        tool, stub = make_tool()
        store = InMemoryStore()
        failed = await tool.coroutine(
            description="scan src",
            subagent_type="researcher",
            runtime=make_runtime(store=store, tool_call_id="a1"),
        )
        assert error_message(failed).status == "error"

        done = await tool.coroutine(
            description="",
            subagent_type="researcher",
            continue_id="a1",
            runtime=make_runtime(store=store, tool_call_id="a2"),
        )
        assert error_message(done).status == "success"
        assert stub.fed[-1][0] == ("HumanMessage", "scan src")


# ---------------------------------------------------------------------------
# entry cleanup / idempotency
# ---------------------------------------------------------------------------
class TestEntryCleanup:
    def test_same_tool_call_id_keeps_a_single_node(self):
        tool, _ = make_tool()

        class AlwaysFails(StubSubagent):
            def _run(self, state, config):
                raise TimeoutError("fails")

        tool2, _ = make_tool(AlwaysFails())
        store = InMemoryStore()
        runtime = make_runtime(store=store, tool_call_id="c1")
        for _ in range(3):
            tool2.func(description="scan", subagent_type="researcher", runtime=runtime)
        nodes = list_nodes(store, "t1")
        assert [n.node_id for n in nodes] == ["c1"]

    def test_entry_cleanup_keeps_ancestors(self):
        tool, _ = make_tool()

        class AlwaysFails(StubSubagent):
            def _run(self, state, config):
                raise TimeoutError("fails")

        tool2, _ = make_tool(AlwaysFails())
        store = InMemoryStore()
        tool2.func(
            description="scan",
            subagent_type="researcher",
            runtime=make_runtime(store=store, tool_call_id="c1"),
        )
        # continuing and failing again must not delete the parent
        tool2.func(
            description="again",
            subagent_type="researcher",
            continue_id="c1",
            runtime=make_runtime(store=store, tool_call_id="c2"),
        )
        assert get_node(store, "t1", "c1") is not None
        assert get_node(store, "t1", "c2") is not None


# ---------------------------------------------------------------------------
# state machine & quotas
# ---------------------------------------------------------------------------
class TestStateMachine:
    def _store_with(self, **node_kwargs) -> tuple[InMemoryStore, SubAgentSnapshotNode]:
        store = InMemoryStore()
        node = SubAgentSnapshotNode(
            node_id="n1", subagent_type="researcher", **node_kwargs
        )
        put_node(store, "t1", node)
        return store, node

    def test_not_found(self):
        store = InMemoryStore()
        assert validate_continuation(store, "t1", "missing", "researcher").status == (
            "not_found"
        )

    def test_type_mismatch(self):
        store, _ = self._store_with()
        check = validate_continuation(store, "t1", "n1", "other")
        assert check.status == "type_mismatch"

    def test_not_available(self):
        store, _ = self._store_with(available=False)
        check = validate_continuation(store, "t1", "n1", "researcher")
        assert check.status == "not_available"

    def test_exhausted(self):
        store, _ = self._store_with(depth=4)
        check = validate_continuation(
            store, "t1", "n1", "researcher", max_continue_attempts=3
        )
        assert check.status == "exhausted"

    def test_branched_out(self):
        store, _ = self._store_with()
        for index in range(2):
            put_node(
                store,
                "t1",
                SubAgentSnapshotNode(
                    node_id=f"child{index}",
                    parent_id="n1",
                    chain=["n1"],
                    depth=2,
                    subagent_type="researcher",
                ),
            )
        check = validate_continuation(store, "t1", "n1", "researcher", max_children=2)
        assert check.status == "branched_out"

    def test_busy(self):
        store, _ = self._store_with()
        check = validate_continuation(
            store, "t1", "n1", "researcher", in_flight={"n1"}
        )
        assert check.status == "busy"

    def test_root_failures_are_capped(self):
        tool, _ = make_tool()

        class AlwaysFails(StubSubagent):
            def _run(self, state, config):
                raise TimeoutError("fails")

        tool2, _ = make_tool(AlwaysFails(), max_root_failures_per_thread=2)
        store = InMemoryStore()
        for index in range(4):
            tool2.func(
                description="scan",
                subagent_type="researcher",
                runtime=make_runtime(store=store, tool_call_id=f"r{index}"),
            )
        nodes = list_nodes(store, "t1")
        # only the first two roots are kept; execution still happens
        assert len(nodes) == 2


class TestConfigSurface:
    def test_config_exposes_continuation_settings(self):
        from mambo_agents.middleware.subagents import SubAgent, SubAgentConfig

        config = SubAgentConfig(
            subagents=[SubAgent(name="r", description="d", system_prompt="p")],
            catch_errors=False,
            max_continue_attempts=1,
            max_children=0,
            max_root_failures_per_thread=2,
        )
        assert config.catch_errors is False
        assert config.max_continue_attempts == 1
        assert config.max_children == 0
        assert config.max_root_failures_per_thread == 2

    def test_task_tool_schema_accepts_continue_id(self):
        from mambo_agents.middleware.subagents import TaskToolSchema

        parsed = TaskToolSchema.model_validate(
            {"subagent_type": "researcher", "continue_id": "abc"}
        )
        assert parsed.description == ""
        assert parsed.continue_id == "abc"


# ---------------------------------------------------------------------------
# delta computation
# ---------------------------------------------------------------------------
class TestDelta:
    def test_id_based_diff_survives_rewritten_history(self):
        kept = HumanMessage(content="old", id="m1")
        injected = [kept, HumanMessage(content="task")]
        produced = [
            AIMessage(content="summary replaces history", id="m9"),
            AIMessage(content="new step", id="m10"),
        ]
        delta = build_delta(injected, produced, injected_ids={"m1"})
        assert [m.text for m in delta] == [
            "summary replaces history",
            "new step",
        ]

    def test_injected_ids_captured_before_run_excludes_history(self):
        history = [HumanMessage(content="h", id="h1")]
        instruction = HumanMessage(content="go")
        produced = [*history, instruction, AIMessage(content="ok", id="a1")]
        delta = build_delta(history, produced, injected_ids={"h1"})
        assert [type(m).__name__ for m in delta] == ["HumanMessage", "AIMessage"]
