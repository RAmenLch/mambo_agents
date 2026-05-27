"""Interactive chat CLI for manually testing Mambo Agents.

Usage::

    python -m mambo_agents.cli.chat [--model MODEL] [--workspace DIR]
                                   [--general-purpose] [--subagent NAME:DESC:PROMPT]
"""

from __future__ import annotations

import asyncio
import os
import sys

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph

from mambo_agents.backends.local import LocalBackend
from mambo_agents.backends.state import StateBackend
from mambo_agents.graph import create_mambo_agent
from mambo_agents.middleware.subagents import SubAgent

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "Pro/zai-org/GLM-4.7"
DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_model(model_name: str):
    """Build a ChatOpenAI model instance from config."""
    from langchain_openai import ChatOpenAI

    api_key = os.environ.get("GJKEY", "") or os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "") or DEFAULT_BASE_URL

    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=0.0,
        streaming=True,
    )


def _parse_args(args: list[str] | None = None) -> dict[str, object]:
    """Minimal CLI arg parser (avoids external deps).

    Returns dict with keys:
        model: str | None
        workspace: str | None
        general_purpose: bool
        subagents: list[str]  – raw "name:desc:prompt" strings
        event_granularity: str
    """
    if args is None:
        args = sys.argv[1:]
    parsed: dict[str, object] = {
        "model": None,
        "workspace": None,
        "general_purpose": False,
        "subagents": [],
        "event_granularity": "updates",
    }
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("--model", "-m"):
            i += 1
            if i < len(args):
                parsed["model"] = args[i]
        elif arg in ("--workspace", "-w"):
            i += 1
            if i < len(args):
                parsed["workspace"] = args[i]
        elif arg in ("--general-purpose", "-g"):
            parsed["general_purpose"] = True
        elif arg in ("--subagent", "-s"):
            i += 1
            if i < len(args):
                subagents: list = parsed["subagents"]  # type: ignore[assignment]
                subagents.append(args[i])
        elif arg in ("--event-granularity", "-e"):
            i += 1
            if i < len(args) and args[i] in ("messages", "updates", "values"):
                parsed["event_granularity"] = args[i]
        elif arg in ("--help", "-h"):
            _print_help()
            sys.exit(0)
        i += 1
    return parsed


def _print_help() -> None:
    print(__doc__)
    print("Options:")
    print("  --model, -m       Model name (default: Pro/zai-org/GLM-4.7)")
    print("  --workspace, -w   Working directory for LocalBackend")
    print("  --general-purpose, -g   Enable the built-in general-purpose subagent")
    print("  --subagent, -s    Add a subagent: name:description:system_prompt")
    print("  --event-granularity, -e  Subagent event detail: messages|updates|values")
    print("  --help, -h        Show this help")
    print()
    print("Subagent examples:")
    print("  -g                                   # Enable default general-purpose")
    print("  -s \"reviewer:Code reviewer:You review code for bugs\"")
    print("  -s \"writer:Content writer:You write clean documentation\"")
    print()
    print("Environment:")
    print("  GJKEY           API key (or OPENAI_API_KEY)")
    print("  OPENAI_BASE_URL API base URL")
    print()
    print("Commands:")
    print("  /exit, /q       Quit")
    print("  /clear          Reset conversation")
    print("  /help, /?       Show this help")
    print("  /system <text>  Set system prompt")


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------
_COLOR_RESET = "\033[0m"
_COLOR_DIM = "\033[2m"
_COLOR_GREEN = "\033[32m"
_COLOR_YELLOW = "\033[33m"
_COLOR_CYAN = "\033[36m"
_COLOR_MAGENTA = "\033[35m"
_COLOR_BLUE = "\033[34m"
_COLOR_BRIGHT_BLACK = "\033[90m"


async def _stream_agent(
    agent: CompiledStateGraph, new_messages: list, thread_id: str
) -> list:
    """Stream agent response, printing tokens / tool-calls in real time.

    Args:
        agent: Compiled agent graph.
        new_messages: Only the new messages to send this turn (NOT full history).
        thread_id: Thread ID for checkpoint persistence.

    Returns:
        The full message list after this turn (from checkpoint state).
    """
    config = {"configurable": {"thread_id": thread_id}}
    payload = {"messages": list(new_messages)}

    # Track tool-call IDs we've already printed to avoid duplicates
    seen_tool_calls: set[str] = set()

    async for event in agent.astream(
        payload,
        config=config,
        stream_mode=["updates", "custom"],
    ):
        if not isinstance(event, tuple) or len(event) != 2:
            continue

        mode, data = event

        if mode == "updates" and isinstance(data, dict):
            for node_name, node_output in data.items():
                if node_name == "__interrupt__":
                    print(f"\n{_COLOR_YELLOW}[HITL] Agent is paused – resume required.{_COLOR_RESET}")
                    continue

                if node_output is None:
                    continue
                messages_out = node_output if isinstance(node_output, list) else node_output.get("messages", [])
                if not isinstance(messages_out, list):
                    continue

                for msg in messages_out:
                    cls_name = type(msg).__name__

                    if cls_name in ("AIMessageChunk", "AIMessage"):
                        content = getattr(msg, "content", "") or ""
                        if content:
                            print(f"{_COLOR_GREEN}{content}{_COLOR_RESET}", end="", flush=True)

                        # Tool calls
                        tool_calls = getattr(msg, "tool_calls", None) or []
                        for tc in tool_calls:
                            tc_id = tc.get("id", "")
                            tc_name = tc.get("name", "")
                            tc_args = tc.get("args", {})
                            if tc_id and tc_id not in seen_tool_calls:
                                seen_tool_calls.add(tc_id)
                                print(f"\n{_COLOR_CYAN}[ToolCall] {tc_name}{_COLOR_RESET} {tc_args}")

                    elif cls_name in ("ToolMessage",):
                        tc_id = getattr(msg, "tool_call_id", "")
                        content_str = getattr(msg, "content", "") or ""
                        # Truncate long tool results
                        if len(content_str) > 500:
                            content_str = content_str[:500] + f"... ({len(content_str)} chars total)"
                        if tc_id not in seen_tool_calls:
                            seen_tool_calls.add(tc_id)
                        print(f"{_COLOR_DIM}[ToolResult] {tc_id}{_COLOR_RESET}")
                        if content_str:
                            print(f"{_COLOR_DIM}{content_str}{_COLOR_RESET}")

        elif mode == "custom":
            _handle_custom_event(data)

    print()  # trailing newline

    # Read back the full authoritative state from the checkpoint so the caller
    # always has the complete message history (including AIMessage, ToolMessage, etc.)
    state = await agent.aget_state(config)
    if state is not None and state.values:
        return list(state.values.get("messages", []))
    return []


# ---------------------------------------------------------------------------
# Custom event handlers
# ---------------------------------------------------------------------------

def _handle_custom_event(data: dict) -> None:
    """Handle custom stream events (subagent, summarization, etc.)."""
    event_type = data.get("type", "")
    if event_type == "subagent_event":
        _handle_subagent_event(data)


def _handle_subagent_event(data: dict) -> None:
    """Display a subagent internal event.

    Subagent output is styled in bright-black / blue to visually distinguish it
    from the main agent's green output.
    """
    subagent_type = data.get("subagent_type", "subagent")
    chunk = data.get("chunk", {})
    granularity = data.get("granularity", "updates")

    # For updates granularity: chunk is a dict like {node_name: messages_or_dict}
    # For messages granularity: chunk is a list of messages
    # For values granularity: chunk is the full state dict

    if granularity == "messages" and isinstance(chunk, list):
        for msg in chunk:
            cls_name = type(msg).__name__
            content = getattr(msg, "content", "") or ""
            if cls_name in ("AIMessageChunk", "AIMessage"):
                if content:
                    print(f"\n{_COLOR_BRIGHT_BLACK}┌─[{subagent_type}]{_COLOR_RESET} ", end="")
                    print(f"{_COLOR_BLUE}{content}{_COLOR_RESET}", end="", flush=True)
            elif cls_name in ("ToolMessage",):
                tc_id = getattr(msg, "tool_call_id", "")
                content_str = str(getattr(msg, "content", "")) or ""
                if len(content_str) > 200:
                    content_str = content_str[:200] + "..."
                print(f"\n{_COLOR_BRIGHT_BLACK}│ [{subagent_type}] ToolResult({tc_id}):{_COLOR_RESET} {_COLOR_DIM}{content_str}{_COLOR_RESET}")

    elif granularity == "updates" and isinstance(chunk, dict):
        for node_name, node_output in chunk.items():
            if node_output is None:
                continue
            messages_out = node_output if isinstance(node_output, list) else node_output.get("messages", [])
            if not isinstance(messages_out, list):
                continue
            for msg in messages_out:
                cls_name = type(msg).__name__
                if cls_name in ("AIMessageChunk", "AIMessage"):
                    content = getattr(msg, "content", "") or ""
                    if content:
                        print(f"\n{_COLOR_BRIGHT_BLACK}┌─[{subagent_type}:{node_name}]{_COLOR_RESET} ", end="")
                        print(f"{_COLOR_BLUE}{content}{_COLOR_RESET}", end="", flush=True)
                elif cls_name in ("ToolMessage",):
                    tc_id = getattr(msg, "tool_call_id", "")
                    content_str = str(getattr(msg, "content", "")) or ""
                    if len(content_str) > 200:
                        content_str = content_str[:200] + "..."
                    print(f"\n{_COLOR_BRIGHT_BLACK}│ [{subagent_type}:{node_name}] ToolResult({tc_id}):{_COLOR_RESET} {_COLOR_DIM}{content_str}{_COLOR_RESET}")

    elif granularity == "values" and isinstance(chunk, dict):
        msgs = chunk.get("messages", [])
        if isinstance(msgs, list) and msgs:
            last = msgs[-1]
            cls_name = type(last).__name__
            content = getattr(last, "content", "") or ""
            if cls_name in ("AIMessage", "AIMessageChunk"):
                print(f"\n{_COLOR_BRIGHT_BLACK}┌─[{subagent_type}:step]{_COLOR_RESET} ", end="")
                if len(content) > 300:
                    content = content[:300] + "..."
                print(f"{_COLOR_BLUE}{content}{_COLOR_RESET}", end="", flush=True)


# ---------------------------------------------------------------------------
# Subagent spec parser
# ---------------------------------------------------------------------------

def _parse_subagent(raw: str) -> SubAgent | None:
    """Parse 'name:description:system_prompt' into a SubAgent dict.

    Returns None if the format is invalid.
    """
    parts = raw.split(":", 2)
    if len(parts) != 3:
        print(f"{_COLOR_YELLOW}[Warn] Invalid subagent spec '{raw}', expected name:desc:prompt{_COLOR_RESET}")
        return None
    name, desc, prompt = parts
    name = name.strip()
    desc = desc.strip()
    prompt = prompt.strip()
    if not name or not desc or not prompt:
        print(f"{_COLOR_YELLOW}[Warn] Subagent spec has empty parts: '{raw}'{_COLOR_RESET}")
        return None
    return SubAgent(name=name, description=desc, system_prompt=prompt)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
async def interactive_chat(
    agent: CompiledStateGraph,
    *,
    system_prompt: str | None = None,
) -> None:
    """Run the interactive chat loop."""
    history: list = []
    thread_id = f"cli-{id(history)}"
    if system_prompt:
        history.append(SystemMessage(content=system_prompt))

    print(f"\n{_COLOR_MAGENTA} Mambo Agents Interactive Chat{_COLOR_RESET}")
    print(f"{_COLOR_DIM}  Type /help for commands, /exit to quit.{_COLOR_RESET}\n")

    while True:
        try:
            user_input = input(f"{_COLOR_CYAN}You> {_COLOR_RESET}")
        except (KeyboardInterrupt, EOFError):
            print(f"\n{_COLOR_DIM}Goodbye!{_COLOR_RESET}")
            break

        raw = user_input.strip()
        if not raw:
            continue

        # ---- Commands ----
        if raw.startswith("/"):
            parts = raw.split(maxsplit=1)
            cmd = parts[0].lower()
            rest = parts[1] if len(parts) > 1 else ""

            if cmd in ("/exit", "/q"):
                print(f"{_COLOR_DIM}Goodbye!{_COLOR_RESET}")
                break
            elif cmd in ("/help", "/?"):
                _print_help()
                continue
            elif cmd == "/clear":
                history = []
                if system_prompt:
                    history.append(SystemMessage(content=system_prompt))
                thread_id = f"cli-{id(history)}"
                print(f"{_COLOR_DIM}Conversation cleared.{_COLOR_RESET}")
                continue
            elif cmd == "/system":
                if not rest:
                    print(f"{_COLOR_YELLOW}Usage: /system <prompt>{_COLOR_RESET}")
                    continue
                # Replace system message
                history = [m for m in history if not isinstance(m, SystemMessage)]
                history.insert(0, SystemMessage(content=rest))
                print(f"{_COLOR_DIM}System prompt updated.{_COLOR_RESET}")
                continue
            else:
                print(f"{_COLOR_YELLOW}Unknown command: {cmd}. Type /help for commands.{_COLOR_RESET}")
                continue

        # ---- Normal chat turn ----
        user_msg = HumanMessage(content=raw)

        print(f"{_COLOR_GREEN}Agent>{_COLOR_RESET} ", end="", flush=True)
        try:
            # Only send the NEW user message — the checkpointer preserves
            # all previous messages (including AIMessages) under thread_id.
            full_history = await _stream_agent(agent, [user_msg], thread_id)
            if full_history:
                history = full_history
            else:
                # Fallback: append user message locally if state fetch failed
                history.append(user_msg)
        except Exception as exc:
            print(f"\n{_COLOR_YELLOW}[Error] {exc}{_COLOR_RESET}")
            continue


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(args: list[str] | None = None) -> None:
    """CLI entry point."""
    opts = _parse_args(args)
    model_name = (opts["model"] or DEFAULT_MODEL)  # type: ignore[assignment]
    workspace = opts["workspace"]  # type: ignore[assignment]
    general_purpose = opts["general_purpose"]  # type: ignore[assignment]
    raw_subagents = opts["subagents"]  # type: ignore[assignment]
    event_granularity = opts["event_granularity"]  # type: ignore[assignment]

    # Parse subagent specs
    subagents: list[SubAgent] = []
    for raw in raw_subagents:
        sa = _parse_subagent(raw)
        if sa:
            subagents.append(sa)

    print(f"{_COLOR_DIM}Model         : {model_name}{_COLOR_RESET}")
    if workspace:
        print(f"{_COLOR_DIM}Backend       : LocalBackend(root_dir={workspace!r}){_COLOR_RESET}")
    else:
        print(f"{_COLOR_DIM}Backend       : StateBackend (in-memory){_COLOR_RESET}")
    print(f"{_COLOR_DIM}Subagents     : general_purpose={general_purpose}, custom={len(subagents)}{_COLOR_RESET}")
    print(f"{_COLOR_DIM}Granularity   : {event_granularity}{_COLOR_RESET}")

    model = _build_model(model_name)

    if workspace:
        backend = LocalBackend(root_dir=workspace)
    else:
        backend = StateBackend()

    agent = create_mambo_agent(
        model=model,
        backend=backend,
        include_general_purpose=general_purpose,
        subagents=subagents if subagents else None,
        event_granularity=event_granularity,
        checkpointer=MemorySaver(),
    )

    asyncio.run(interactive_chat(agent))


if __name__ == "__main__":
    main()
