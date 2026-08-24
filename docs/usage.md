> 📖 **English** | [中文](usage.cn.md)

# Mambo Agents Usage Guide

## Table of Contents

1. [Installation](#1-installation)
2. [Core Concepts](#2-core-concepts)
3. [Quick Start](#3-quick-start)
4. [Agent Factory Function](#4-agent-factory-function)
5. [Let Your Agent Control the Environment](#5-let-your-agent-control-the-environment)
6. [Long Conversation Management](#6-long-conversation-management)
7. [Task Planning & Tracking](#7-task-planning--tracking)
8. [Installing Skill Packs](#8-installing-skill-packs)
9. [Let Your Agent Remember Your Preferences](#9-let-your-agent-remember-your-preferences)
10. [Security & Human Approval](#10-security--human-approval)
    - [Reviewing MCP Tools](#10x-reviewing-mcp-tools)
11. [File Change History & Rollback](#11-file-change-history--rollback)
12. [Integrating External MCP Tools](#12-integrating-external-mcp-tools)
    - [`exclude_tools` — Hiding Dangerous Tools](#12x-exclude_tools--hiding-dangerous-tools)
    - [`direct_tool_threshold` — Direct vs Wrapped Mode](#12x-direct_tool_threshold--direct-vs-wrapped-mode)
    - [Security Review Integration](#12x-security-review-integration)
    - [`mcp_tool_name()` Reference](#12x-mcp_tool_name-reference)
13. [Multi-Agent Collaboration](#13-multi-agent-collaboration)
14. [Goal-Driven Loop Control](#14-goal-driven-loop-control)
15. [Advanced Usage](#15-advanced-usage)

---

## 1. Installation

```bash
pip install mambo-agents
```

---

## 2. Core Concepts

Mambo Agents builds on LangGraph, assembling agents via the **factory function + middleware stack** pattern.

### Three-layer Architecture

| Layer | Component | Responsibility |
|-------|-----------|----------------|
| **Backend** | `BackendProtocol` | The agent's "hands": filesystem abstraction providing 6 core operations + extension tools |
| **Middleware** | `AgentMiddleware` (from langchain) | Cross-cutting concerns: mambo provides 12+ built-in middleware including summarization, planning, skills, memory, security review, version control, MCP integration, etc. |
| **Agent** | `create_mambo_agent()` | Assembles backend + middleware, returns a compiled LangGraph state graph |

---

## 3. Quick Start

### 3.1 Simplest Usage

```python
from mambo_agents import create_mambo_agent
from langchain_core.messages import HumanMessage

agent = create_mambo_agent("gpt-4o")

result = agent.invoke({
    "messages": [HumanMessage("Create a Python script that prints Hello World")]
}, config={"configurable": {"thread_id": "session-1"}})

# Check the agent's final response
print(result["messages"][-1].content)
# Example output:
# Created /hello.py:
# ```python
# print("Hello World")
# ```
```

> **Default backend:** When `backend` is not specified, the agent uses `StoreBackend` + `InMemoryStore`.
> Files are stored in session memory and disappear on process restart. For persistence (e.g. PostgreSQL)
> or real disk operations, specify `backend=LocalBackend(...)` (see next section).
>
> **Path convention:** `StoreBackend` addresses files under the `/workspace/` prefix
> (e.g. `/workspace/hello.py`); paths outside it (e.g. `/hello.py`) are not visible
> in `ls /workspace` / `glob` views. See [5.2](#52-virtual-filesystem-storebackend).

### 3.2 Working with a Local Filesystem

```python
from mambo_agents.backends.local import LocalBackend

agent = create_mambo_agent(
    "gpt-4o",
    backend=LocalBackend(root_dir="/tmp/myproject"),
)

result = agent.invoke({
    "messages": [HumanMessage("List all files in the current directory")]
}, config={"configurable": {"thread_id": "session-1"}})
print(result["messages"][-1].content)
```

### 3.3 Adding Custom Tools

Beyond built-in filesystem tools, the agent can mount any custom tools:

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get weather for a specified city"""
    return f"{city}: Sunny, 25°C"

agent = create_mambo_agent(
    "gpt-4o",
    tools=[get_weather],
)

result = agent.invoke({
    "messages": [HumanMessage("Check the weather in Beijing")]
}, config={"configurable": {"thread_id": "session-1"}})
```

### 3.4 Streaming Output

**By node (fires once per completed step):**

```python
agent = create_mambo_agent("gpt-4o")

async for event in agent.astream(
    {"messages": [HumanMessage("Analyze the project code structure")]},
    stream_mode="updates",
    config={"configurable": {"thread_id": "session-1"}},
):
    print(event)
    # Example output (fires once per completed step):
    # {'model': {'messages': [AIMessage(content='Let me first ls the directory structure...')]}}
    # {'tools': {'messages': [ToolMessage(content='...', name='ls')]}}
    # {'model': {'messages': [AIMessage(content='The project contains the following files...')]}}
```

**Token-by-token streaming (real-time LLM output):**

```python
async for event in agent.astream(
    {"messages": [HumanMessage("Explain what a decorator is")]},
    stream_mode="messages",
    config={"configurable": {"thread_id": "session-1"}},
):
    # event is a tuple of (message_chunk, metadata)
    msg_chunk, metadata = event
    if msg_chunk.content:
        print(msg_chunk.content, end="", flush=True)
```

> For more streaming modes (e.g. `custom` for receiving sub-agent progress events), see
> [13.1](#131-synchronous-sub-agents).

### 3.5 Multi-turn Conversations

The agent uses `thread_id` to distinguish different conversation sessions. Multiple calls under
the same `thread_id` share filesystem and conversation history:

```python
config = {"configurable": {"thread_id": "session-1"}}

# Turn 1
result1 = agent.invoke(
    {"messages": [HumanMessage("Create a config.json with port 8080")]},
    config=config,
)

# Turn 2 — the agent remembers the previously created file and conversation
result2 = agent.invoke(
    {"messages": [HumanMessage("Change the port to 9090")]},
    config=config,
)

# A different thread_id is a brand-new session
result3 = agent.invoke(
    {"messages": [HumanMessage("List all files")]},
    config={"configurable": {"thread_id": "session-2"}},
)
# Result is empty — session-2 is a fresh session, cannot see session-1's files
```

> **What `thread_id` does:**
> - **Conversation history isolation:** conversations with different `thread_id` are invisible to each other
> - **Filesystem isolation:** `StoreBackend` creates an independent virtual filesystem for each `thread_id`
> - **`config` is required:** every call must pass `config={"configurable": {"thread_id": "..."}}`

---

## 4. Agent Factory Function

### 4.1 `create_mambo_agent()`

Full parameter signature:

```python
def create_mambo_agent(
    model: str | BaseChatModel,
    *,
    backend: BackendProtocol | None = None,
    system_prompt: str | None = None,
    subagents: Sequence[SubAgent | CompiledSubAgent] | None = None,
    include_general_purpose: bool = False,
    async_subagents: Sequence[SubAgent | CompiledSubAgent] | None = None,
    async_subagent_timeout: float = 3600.0,
    subagent_event_granularity: EventGranularity = "updates",
    middleware: Sequence[AgentMiddleware] | None = None,
    summarization: SummarizationConfig | dict | None = None,
    skills: Sequence[SkillSource] | None = None,
    memory_sources: list[VirtualPath] | None = None,
    tools: Sequence[BaseTool] | None = None,
    interrupt_on: dict[str, bool | InterruptOnConfig] | None = None,
    security_review: SecurityReviewConfig | None = None,
    version_control: VersionControlConfig | VersionStore | bool | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
    name: str | None = None,
    **kwargs,
) -> CompiledStateGraph
```

**Required:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `str \| BaseChatModel` | LLM model name or instance |

**Commonly Used Optional Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | `BackendProtocol` | `StoreBackend()` | Filesystem backend |
| `system_prompt` | `str` | built-in default | Custom system prompt |
| `summarization` | `SummarizationConfig` or `dict` | `None` | Conversation summarization config |
| `subagents` | `list` | `None` | Synchronous sub-agent list |
| `include_general_purpose` | `bool` | `False` | Whether to add a general-purpose sub-agent |
| `async_subagents` | `list` | `None` | Async sub-agent list |
| `async_subagent_timeout` | `float` | `3600.0` | Timeout for async sub-agents (seconds) |
| `subagent_event_granularity` | `EventGranularity` | `"updates"` | Streaming granularity for sub-agent custom events |
| `skills` | `list` | `None` | Skill source paths |
| `memory_sources` | `list[VirtualPath]` | `None` | Memory file paths |
| `tools` | `list` | `None` | Extra tools |
| `interrupt_on` | `dict` | `None` | Tool approval config. `{"write": True}` for simple enable, or `{"write": {"allowed_decisions": ["approve", "reject"]}}` to limit approval options |
| `security_review` | `SecurityReviewConfig` | `None` | AI security pre-review config |
| `version_control` | `VersionControlConfig` / `VersionStore` / `bool` | `None` | Version control config |
| `checkpointer` | `BaseCheckpointSaver` | `InMemorySaver()` | Checkpoint persistence |
| `store` | `BaseStore` | `None` | LangGraph Store (for backend persistence) |

**Example:**

```python
from mambo_agents import create_mambo_agent, StoreBackend

agent = create_mambo_agent(
    "gpt-4o",
    backend=StoreBackend(initial_files={
        "/workspace/config.json": '{"port": 8080}',
    }),
    summarization={
        "trigger": ("tokens", 200000),
        "keep": ("messages", 20),
    },
    skills=["/skills/user/"],
)
```

---

## 5. Let Your Agent Control the Environment

The agent's "hands" — the `backend` parameter determines where the agent can read/write files and execute commands.

### 5.1 Core Operations

All backends must implement 6 core operations:

| Operation | Description |
|-----------|-------------|
| `ls(path)` | List directory contents (non-recursive) |
| `read(file_path, offset, limit, include_line_numbers)` | Read file content |
| `write(file_path, content, overwrite)` | Create / overwrite a file |
| `edit(file_path, old_str, new_str, replace_all)` | Replace text in a file |
| `grep(pattern, path, glob, regex, offset, limit)` | Search text content |
| `glob(pattern, path)` | Find files and directories by wildcard pattern |

Each backend can also expose extra tools via its `tools` property (e.g. `tree`, `delete`, `execute`; `HybridWorkspaceBackend` additionally provides `copy`).

> **Path conventions:** All backends use absolute paths starting with `/` (e.g. `/workspace/src/main.py`).
> Some config parameters (e.g. `memory_sources`) accept `VirtualPath` or plain `str` — the framework
> auto-converts. `VirtualPath` validates and rejects illegal patterns like `..`, `//`.
> Import: `from mambo_agents.backends.schemas import VirtualPath`.

### 5.2 Virtual Filesystem (StoreBackend)

`StoreBackend` is a virtual filesystem backed by `BaseStore`, LangGraph's key-value store interface.

```python
from mambo_agents import StoreBackend

# Default in-memory storage (disappears on process restart)
# NOTE: files must live under the workspace root "/workspace" so they show
# up in the agent's default `ls /workspace` view. Keys without the prefix
# are still readable by exact path, but invisible to ls/glob at the root.
backend = StoreBackend(
    initial_files={
        "/workspace/config.json": '{"port": 8080}',
        "/workspace/README.md": "# My Project",
    }
)

# Persistent storage (e.g. PostgreSQL)
from langgraph.store.postgres import PostgresStore

backend = StoreBackend(
    store=PostgresStore.from_conn_string("postgresql://..."),
    initial_files={"/workspace/config.json": '{"port": 8080}'},
)
```

> **Path convention:** `StoreBackend`'s `workspace_root` is fixed at `/workspace`.
> Pre-populated `initial_files` keys must start with `/workspace/` (e.g. `/workspace/config.json`),
> and agents should be instructed to create files under `/workspace/` as well — otherwise
> files are reachable only by their exact path and do **not** appear in the agent's
> default `ls /workspace` / `glob` views.

Common `BaseStore` implementations:

| Implementation | Source | Persistent | Use Case |
|---------------|--------|:---:|----------|
| `InMemoryStore` | `langgraph.store.memory` | ❌ | Dev / Testing (default) |
| `PostgresStore` | `langgraph.store.postgres` | ✅ | Production |
| Custom `BaseStore` | Implement `BaseStore` interface | Customizable | Integrate existing storage |

> The `store` parameter of `create_mambo_agent()` is `BaseStore`. If not specified, defaults to
> `InMemoryStore`. For persistence, specify `create_mambo_agent(store=PostgresStore(...))`
> and the framework automatically passes it to `StoreBackend`.

**Extra tools:** `tree`

**Characteristics:**
- Session isolation: different `thread_id` filesystems are independent (like each session having its own virtual disk)
- Works inside and outside graph context
- `thread_id` locked at construction for easy graph-outside usage:
  ```python
  be = StoreBackend(thread_id="my-session")
  be.upload_files([(path, data)])  # auto-writes to "my-session"
  ```

### 5.3 Local Disk (LocalBackend)

Direct access to real disk files, with optional shell command execution.

```python
from mambo_agents.backends.local import LocalBackend

backend = LocalBackend(
    root_dir="/home/user/project",
    timeout=120,           # Shell command timeout (seconds)
    max_output_bytes=100000,  # Max output bytes
    enable_execute=True,   # Enable shell execution (default False)
    inherit_env=True,      # Inherit system environment variables
)
```

**Extra tools:** `tree`, `delete`, `execute` (requires enabling)

**`grep` performance strategy:**
1. Prefer system `rg` (ripgrep) — orders of magnitude faster than Python traversal
2. Fall back to Python traversal (with file size guard)

> **💡 Install ripgrep for best performance:**
> - **macOS:** `brew install ripgrep`
> - **Linux:** `apt install ripgrep` / `dnf install ripgrep`
> - **Windows:** `winget install BurntSushi.ripgrep.MSVC` or `scoop install ripgrep`

> **⚠️ Security Warning:** `LocalBackend` provides direct filesystem access and shell execution. Consider using `interrupt_on` + `security_review` alongside it.

### 5.4 Remote Server (SshBackend)

Operate on remote server files via SSH/SFTP.

```python
from mambo_agents.backends.ssh import SshBackend

backend = SshBackend(
    host="192.168.1.100",
    username="deploy",
    password="secret123",         # or key_filename="/path/to/key"
    port=22,
    remote_root="/home/deploy/app",
    connect_timeout=30,
    execute_timeout=120,
    enable_execute=False,         # Enable shell execution (default False)
)
```

**Extra tools:** `tree`, `delete`, `execute` (requires enabling)

**Performance strategy:** Batch operations (`grep`, `glob`, `edit`, `tree`) execute remotely to avoid per-file SFTP round-trips. `edit` completes find-and-replace in a single remote `python3 -c` invocation.

> **💡 Install ripgrep on the remote server for best grep performance:**
> - `apt install ripgrep` / `dnf install ripgrep` / `brew install ripgrep`

### 5.5 Hybrid Workspace (HybridWorkspaceBackend)

One real backend + N virtual workspaces, all routed under `/.mambo/`.
Each virtual workspace is backed by any `BackendProtocol` implementation (typically `StoreBackend`).

```python
from mambo_agents.backends.hybrid_workspace import HybridWorkspaceBackend
from mambo_agents.backends.local import LocalBackend
from mambo_agents import StoreBackend

# Simplest: auto-create /.mambo/ default StoreBackend
backend = HybridWorkspaceBackend(
    real_backend=LocalBackend(root_dir="/tmp/project"),
)

# Multiple virtual workspaces
# NOTE: a virtual StoreBackend's workspace_root is fixed at "/workspace",
# so initial_files keys MUST use the "/workspace/" prefix to match the
# path the router delegates — otherwise pre-populated files are invisible.
backend = HybridWorkspaceBackend(
    real_backend=LocalBackend(root_dir="/tmp/project"),
    virtual_workspaces={
        "skills": StoreBackend(initial_files={"/workspace/python.md": "..."}),
        "cache": StoreBackend(),
    },
)

# Override default /.mambo/
backend = HybridWorkspaceBackend(
    real_backend=LocalBackend(root_dir="/tmp/project"),
    virtual_workspaces={
        ".": StoreBackend(initial_files={"/workspace/config.yml": "..."}),
    },
)
```

**Path routing rules:**
- `/.mambo/skills/xxx` → "skills" virtual workspace (prefix stripped, then re-prefixed with the virtual backend's `workspace_root` — `/workspace` for `StoreBackend` — before delegation)
- `/.mambo/xxx` → default StoreBackend (same rewrite as above)
- `/{workspace_root}/...` → real backend (path rewritten: strip workspace_root, prepend real backend's workspace_root)
- Other paths (e.g. `/`, `/etc`) are rejected

**Virtual workspace notes:**
- A virtual `StoreBackend` has a fixed `workspace_root` of `/workspace` (not `/`), so pre-populated `initial_files` keys must start with `/workspace/`. A key like `/python.md` would be stored under an unreachable path and the agent cannot read it.
- The virtual backend resolves its `BaseStore` from the graph execution context (`get_store()`). For graph-outside access (e.g. verifying files with `download_files()` from your own script), construct the virtual `StoreBackend` with an explicit `store=` (and pass `store=` to `HybridWorkspaceBackend` for the default workspace) so the same store instance is shared. Otherwise graph-outside calls fall back to a private `InMemoryStore` and cannot see the agent's files.
- See `example/11_hybrid_workspace.py` for a fully runnable demo.

**Extra tool:** `copy(source, destination)` — copies a single file across backends
(virtual ↔ real, or between virtual workspaces), overwriting the destination if it exists.

**Use cases:**
- Internal storage (large result eviction, conversation history dumps)
- Agent scratch files
- Sub-agent communication files
- Independent isolated spaces for multiple skills/modules

### 5.6 Read-only Mode (ReadOnlyBackend)

Wraps any `BackendProtocol`, exposing only safe read-only operations (`ls`, `read`, `grep`, `glob`).
Used internally by `AutoSecurityReviewMiddleware` in agent review mode.

```python
from mambo_agents import ReadOnlyBackend

safe = ReadOnlyBackend(backend, allowed_extra_tools=frozenset(["tree"]))
```

### 5.7 Large File Read Summaries (ReadSummarizer)

`BackendProtocol` enforces a `max_read_chars` (default 100,000 chars) upper limit.
When exceeded, content is replaced with a summary rather than simply being truncated.
Users can inject a custom `ReadSummarizer` callback that generates **file-type-aware instructive summaries**,
helping the AI navigate to the right section of a large file.

Mambo ships the `read_summarizers` sub-package with 9 pre-built summarizers:

| Summarizer | Covers | Parser |
|------------|--------|--------|
| `python_summarizer()` | `.py` | `ast` (stdlib) |
| `javascript_summarizer()` | `.js` / `.ts` / `.tsx` | tree-sitter |
| `java_summarizer()` | `.java` | tree-sitter |
| `c_summarizer()` | `.c` / `.h` | tree-sitter |
| `cpp_summarizer()` | `.cpp` / `.hpp` / `.cc` / `.hxx` | tree-sitter |
| `go_summarizer()` | `.go` | tree-sitter |
| `rust_summarizer()` | `.rs` | tree-sitter |
| `markdown_summarizer()` | `.md` / `.mdx` | regex |
| `json_summarizer()` | `.json` | `json` (stdlib) |

Each summarizer extracts a structural outline with **accurate line numbers** — e.g.
Python classes/functions, Markdown heading hierarchy, JSON top-level key structure.

```python
from mambo_agents.read_summarizers import python_summarizer
from mambo_agents.backends.local import LocalBackend

backend = LocalBackend(summarizer=python_summarizer())
```

Summarizers are **never injected by default** — users opt in per use case. Files with
unmatched suffixes fall back to the default behaviour (prompting to re-read with offset + limit).

Multiple summarizers can be combined via `composite_summarizer()`:

```python
from mambo_agents.read_summarizers import (
    composite_summarizer,
    python_summarizer,
    java_summarizer,
)

backend = LocalBackend(summarizer=composite_summarizer([
    python_summarizer(),
    java_summarizer(),
]))
```

### 5.8 Multimodal File Describers (MultimodalDescriber)

`read()` produces a raw base64 multimodal content block for images, video, audio and
documents (PDF / PPT / PPTX). Text-only models cannot consume that block directly.
The `multimodal_describers` sub-package (new in v0.4.0) provides pluggable
`MultimodalDescriber` callbacks that substitute plain text for the multimodal block —
either a natural-language description generated by a multimodal model, or an explicit
rejection message.

Configure one on any backend via the `multimodal_describer` constructor argument
(mirroring `summarizer`):

```python
from mambo_agents.multimodal_describers import multimodal_describer
from mambo_agents.backends.local import LocalBackend
from langchain_openai import ChatOpenAI

vision = ChatOpenAI(model="gpt-4o")
backend = LocalBackend(multimodal_describer=multimodal_describer(vision))
```

The sub-package ships the following factories:

| Factory | Behavior |
|---------|----------|
| `multimodal_describer(model)` | Describes images / video / audio / documents via a multimodal chat model |
| `image_describer(model)` / `video_describer(model)` / `audio_describer(model)` / `document_describer(model)` | Per-type conveniences — other types pass through |
| `reject_multimodal_describer(allow=...)` | Returns an error text for non-text files, with an `allow` set of block types that pass through |
| `composite_multimodal_describer([...])` | Tries several describers in order (first non-`None`, non-empty result wins) |

Describers are **never injected by default** — users opt in per use case. A describer
returns `None` to signal "not my type" and leave the multimodal result unchanged.

### 5.9 Backend Comparison

| Feature | StoreBackend | LocalBackend | SshBackend | HybridWorkspaceBackend |
|---------|:---:|:---:|:---:|:---:|
| Storage Location | LangGraph Store (configurable) | Local Disk | Remote Server | Hybrid |
| Session Isolation | Automatic | Manual | Manual | Automatic (/.mambo/) |
| Shell Execution | ❌ | Optional | Optional | Depends on real backend |
| Delete Operation | ❌ | ✅ | ✅ | Depends on real backend |
| Copy Operation | ❌ | ❌ | ❌ | ✅ |
| grep Acceleration | N/A | ripgrep | Remote rg/grep | Inherits delegate |
| Network Dependency | ❌ | ❌ | ✅ | Optional |
| Best For | Testing / Prototyping | Local Development | Remote Deployment | Production |

---

## 6. Long Conversation Management

Automatically compact long conversation histories to prevent LLM context window overflow.

```python
# Configure via create_mambo_agent
agent = create_mambo_agent(
    "gpt-4o",
    summarization={
        "trigger": ("tokens", 200000),  # trigger when total exceeds 200k tokens
        "keep": ("messages", 20),        # keep the last 20 messages uncompacted
        "offload_to_backend": True,      # persist evicted messages to backend
    },
)
```

**Two trigger modes:**

| Mode | Example | Description |
|------|---------|-------------|
| tokens | `("tokens", 200000)` | Cumulative token count exceeds threshold |
| messages | `("messages", 50)` | Message count exceeds threshold |

**Summarization Modes (`SummarizationMode`):**

| Mode | Description |
|------|-------------|
| `PER_ASTREAM` | (Default) Summarize once before execution starts. No further checks during the run. |
| `PER_MODEL_CALL` | Check summarization on every model call, even mid-run. |

```python
from mambo_agents import SummarizationMode

agent = create_mambo_agent(
    "gpt-4o",
    summarization={
        "mode": SummarizationMode.PER_MODEL_CALL,
        "trigger": ("tokens", 200000),
        "keep": ("messages", 20),
    },
)
```

**Chained Summaries:** When compaction fires multiple times, prior summaries are injected as
non-negotiable historical context into the summarization prompt, preventing information loss across rounds.

**Summary Hooks (`SummaryHook`):** Allow other middleware to inject additional context during
summarization. `MamboPlanMiddleware` uses this mechanism to preserve the current plan state during compaction.

**`SummarizationConfig` full fields:**

```python
from mambo_agents import SummarizationConfig, SummarizationMode

SummarizationConfig(
    mode: SummarizationMode = SummarizationMode.PER_ASTREAM,
    # SummarizationMode.PER_ASTREAM (default): summarize once before execution starts
    # SummarizationMode.PER_MODEL_CALL: check summarization on every model call

    trigger: ("tokens", 200000) | ("messages", 50) | None = None,
    # Trigger condition — compact when cumulative exceeds threshold. None = never trigger

    keep: ("messages", 20) | ("tokens", 5000) = ("messages", 20),
    # Keep the most recent messages uncompacted

    model: str | BaseChatModel | None = None,
    # None = reuse the agent's model

    trim_tokens_to_summarize: int = 4000,
    # Tokens to review when summarizing

    token_counter: Callable | None = None,
    # Custom token counter

    chars_per_token: float | None = None,
    # Custom chars/token ratio

    offload_to_backend: bool = False,
    # When True, compacted messages are persisted to backend

    backend: BackendProtocol | None = None,
    # Backend used for offload_to_backend

    summary_prompt: str = DEFAULT_MAMBO_SUMMARY_PROMPT,
    # Built-in summarization prompt (with {messages} placeholder)

    chained_summary_prompt: str | None = None,
    # Chained summary prompt (used when prior summaries exist)

    summary_hooks: list[SummaryHook] | None = None,
    # Hooks to inject additional context during summarization
)
```

### 6.1 Retrieving Summarization Events

When summarization is triggered, the middleware stores the summary event in the agent's private state `_summarization_event`. You can inspect this field by using `stream_mode="values"` to access the full state.

**`SummarizationEvent` structure:**

| Field | Type | Description |
|------|------|-------------|
| `cutoff_index` | `int` | Absolute index in `state["messages"]`. All messages **before** this index have been replaced by a single `summary_message`. |
| `summary_message` | `HumanMessage` | LLM-generated summary message with three standard sections: `SESSION INTENT` (session goal), `SUMMARY` (key decisions & conclusions), `ARTIFACTS` (created/modified files and changes). Tagged with `additional_kwargs={"lc_source": "summarization"}` for chained summarization. |
| `file_path` | `str \| None` | When `offload_to_backend=True`, evicted messages are persisted to `/.mambo/conversation_history/{thread_id}.md` on the backend. `None` means offload is disabled or failed. |
| `last_summarized_message` | `AnyMessage \| None` | The last real message in the summarized zone (excluding summary markers). Useful for understanding the exact compaction boundary. |

**Detecting summarization events:**

```python
from mambo_agents.middleware.summarization import SummarizationEvent

# Method 1: Use stream_mode="values" to get state and check _summarization_event
last_state = None
for state in agent.stream(
    {"messages": [HumanMessage("...")]},
    config=config,
    stream_mode="values",
):
    last_state = state

# Check for summarization event
sum_event: SummarizationEvent | None = last_state.get("_summarization_event")
if sum_event:
    print(f"Summarization triggered! First {sum_event['cutoff_index']} messages compacted")
    print(f"Summary content: {sum_event['summary_message'].content[:500]}...")
    print(f"Offload path: {sum_event.get('file_path')}")
```

**Real-time monitoring of new summarization events:**

```python
last_cutoff = -1

for state in agent.stream(
    {"messages": [HumanMessage("...")]},
    config=config,
    stream_mode="values",
):
    sum_event = state.get("_summarization_event")
    if sum_event and sum_event.get("cutoff_index", -1) != last_cutoff:
        last_cutoff = sum_event["cutoff_index"]
        print(f"⚡ New summarization: {sum_event['cutoff_index']} messages compacted")
        print(f"   Summary length: {len(sum_event['summary_message'].content)} chars")
```

> **Note:** `_summarization_event` is a private state field (prefixed with `_`), meant for caller observation only — it is never exposed to the LLM. During chained summarization, prior summaries are automatically injected into the new summarization prompt to prevent information loss.

---

## 7. Task Planning & Tracking

Let the agent maintain a structured TODO list and automatically track task progress.

```python
from mambo_agents.middleware.planning import MamboPlanMiddleware

agent = create_mambo_agent(
    "gpt-4o",
    middleware=[MamboPlanMiddleware()],
)
```

**Plan data model:**

```python
from mambo_agents.middleware.planning import Plan

# Each plan item has two fields:
# - content: task description
# - status: "pending" | "in_progress" | "completed"
```

When enabled, the agent gains a `write_plans` tool and will automatically break down complex
tasks into steps with tracked completion status.

**Retrieving plan state:**

```python
from mambo_agents.middleware.planning import Plan

# Method 1: Read plans from invoke result
result = agent.invoke(
    {"messages": [HumanMessage("Help me create a project skeleton...")]},
    config=config,
)
plans: list[Plan] | None = result.get("plans")
if plans:
    for p in plans:
        print(f"[{p.status}] {p.content}")

# Method 2: Observe write_plans tool calls via stream_mode="updates"
for event in agent.stream(
    {"messages": [HumanMessage("...")]},
    config=config,
    stream_mode="updates",
):
    for node_name, node_output in event.items():
        if node_name == "model":
            # AIMessage.tool_calls shows write_plans invocation and args
            for msg in node_output.get("messages", []):
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        if tc["name"] == "write_plans":
                            print(f"Plan: {tc['args']['plans']}")
        elif node_name == "tools":
            # ToolMessage contains write_plans return value
            for msg in node_output.get("messages", []):
                if msg.name == "write_plans":
                    print(f"Plan update result: {msg.content}")

# Method 3: Observe plans changes in real-time via stream_mode="values"
for state in agent.stream(
    {"messages": [HumanMessage("...")]},
    config=config,
    stream_mode="values",
):
    current_plans = state.get("plans")
    if current_plans:
        for p in current_plans:
            print(f"[{p.status}] {p.content}")
```

---

## 8. Installing Skill Packs

A progressive-disclosure skill system — skills are only loaded into the prompt when the agent needs them,
avoiding bloated context.

```python
agent = create_mambo_agent(
    "gpt-4o",
    skills=[
        "/skills/user/",                           # user-level skills
        "/skills/project/",                        # project-level skills
        ("/repo/.claude/skills", "Project Claude"),  # with custom label
    ],
)
```

**Skill file structure:**

```
/skills/user/web-research/
├── SKILL.md          # required: YAML frontmatter + markdown instructions
└── helper.py         # optional: helper files
```

**`SKILL.md` format:**

```markdown
---
name: web-research
description: Methodology for structured web research
license: MIT
---
# Web Research Skill

## Steps
1. Define research objectives
2. Search for keywords
...
```

**Skill Sources (`SkillSource`):**

| Type | Example | Label Source |
|------|---------|--------------|
| bare path | `"/skills/user/"` | Last path component, capitalized (`"/skills/user/"` → `User`; special cases: `built_in_skills` → `Built-in`, a `skills` leaf climbs one level) |
| tuple | `("/path", "My Skill")` | Custom label |

Multi-source loading: later-loaded skills override earlier ones with the same name (last wins).

---

## 9. Let Your Agent Remember Your Preferences

Loads persistent context from AGENTS.md files and instructs the AI to **write back** new learnings
during interactions. Unlike skills (on-demand), memory is always loaded and provides persistent,
evolving context across turns.

**Core mechanism:** `MamboMemoryMiddleware` injects AGENTS.md file content into the System Prompt
via the `MAMBO_MEMORY_SYSTEM_PROMPT` template during the `wrap_model_call` phase.
The final System Prompt structure is:

```
Original system_prompt (from BackendToolsMiddleware)
    ↓
<agent_memory>
AGENTS.md file content (path + body)
</agent_memory>
    ↓
<memory_guidelines>
Rules guiding the Agent on when and how to write back memories
</memory_guidelines>
```

```python
from mambo_agents.backends.schemas import VirtualPath

agent = create_mambo_agent(
    "gpt-4o",
    memory_sources=[VirtualPath("/.mambo/memory/AGENTS.md")],
)
```

**Workflow:**

```
Session start → before_agent loads AGENTS.md → wrap_model_call injects into system prompt
    ↓
During interaction → AI discovers worth-remembering info → writes back with edit/write
```

**Memory content format (AGENTS.md):**

AGENTS.md files are standard Markdown with no required structure. Common content:
- Project overview & architecture notes
- Build/test commands
- Code style guidelines
- User preferences & conventions

**AI self-learning:**

The memory prompt instructs the AI to write back to AGENTS.md when:
- User explicitly asks to remember something
- User provides reusable context (coding style, conventions, workflows)
- User gives feedback and corrections on AI's work
- **Do NOT** record: temporary info, one-off tasks, casual chat, credentials

**Viewing injection effect (intercepting the real System Prompt):**

`MamboMemoryMiddleware` injects memory into `request.system_message` via `modify_request`
during `wrap_model_call`. You can intercept the actual result by placing a lightweight
middleware **after** memory in the stack:

```python
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import SystemMessage

class SystemPromptInterceptor(AgentMiddleware):
    """Placed after memory middleware to capture the injected system prompt."""
    captured: str | None = None

    def wrap_model_call(self, request: ModelRequest, handler) -> ModelResponse:
        sm = request.system_message
        if isinstance(sm, SystemMessage) and isinstance(sm.content, str):
            self.captured = sm.content  # the complete system prompt the Agent actually receives
        return handler(request)

# Usage
interceptor = SystemPromptInterceptor()
agent = create_mambo_agent(
    "gpt-4o",
    memory_sources=[VirtualPath("/.mambo/memory/AGENTS.md")],
    middleware=[interceptor],  # placed after memory
)

agent.invoke({"messages": [...]}, config=config)
# After the call, interceptor.captured contains the full system prompt with memory injected
```

**Custom formatting:**

```python
from mambo_agents.middleware.memory import MamboMemoryMiddleware

def my_formatter(contents: dict[str, str]) -> str:
    """Customize how memory appears in the system prompt."""
    parts = []
    for path, text in contents.items():
        parts.append(f"## Source: {path}\n{text}")
    return "\n---\n".join(parts)

agent = create_mambo_agent(
    "gpt-4o",
    middleware=[
        MamboMemoryMiddleware(
            backend=StoreBackend(),
            sources=[VirtualPath("/.mambo/memory/AGENTS.md")],
            format_prompt=my_formatter,
        ),
    ],
)
```

---

## 10. Security & Human Approval

Review tool calls for safety before actual execution (or before pausing for human approval),
using a cheaper model.

Two review modes:

- **llm** (default): single structured-output LLM call per tool call — fast and cheap.
- **agent**: a dedicated review agent with read-only backend tools that can inspect the workspace
  before delivering a verdict. Backend tools (core 6 + `backend.tools`) get agent review;
  non-backend user tools fall back to llm review.

```python
# Classic HITL (no AI pre-review)
agent = create_mambo_agent(
    "gpt-4o",
    interrupt_on={"write": True, "edit": True},
)

# AI pre-review — llm mode (default)
from mambo_agents.middleware.security_review import SecurityReviewConfig

agent = create_mambo_agent(
    "gpt-4o",
    interrupt_on={"write": True, "edit": True, "delete": True},
    security_review=SecurityReviewConfig(),
)

# Agent mode — backend tools reviewed by agent with read-only workspace
agent = create_mambo_agent(
    "gpt-4o",
    interrupt_on={"write": True, "edit": True, "delete": True},
    security_review=SecurityReviewConfig(
        review_mode="agent",
        agent_max_steps=5,
    ),
)

# Custom pre-review — review only specific tools, using a separate model
agent = create_mambo_agent(
    "gpt-4o",
    interrupt_on={"write": True, "edit": True},
    security_review=SecurityReviewConfig(
        model="gpt-4o-mini",                    # review model
        review_tools=frozenset(["write"]),      # only review write
        system_prompt="You are a security audit expert...",
    ),
)
```

**Workflow:**

```
Tool Call → AI Security Review → Safe: pass through
                               → High Risk: pause → Human Approval
```

**`SecurityReviewConfig` full fields:**

```python
from mambo_agents.middleware.security_review import SecurityReviewConfig

SecurityReviewConfig(
    model: str | BaseChatModel | None = None,
    # None = reuse the agent's model
    # "gpt-4o-mini" = use a cheap model for review

    system_prompt: str | None = None,
    # None = use built-in security review prompt
    # custom = override the review prompt

    review_tools: Literal["all"] | frozenset[str] = "all",
    # "all" = review all interrupt_on tools
    # frozenset({"write", "edit"}) = only review specified tools

    notify_on_pass: bool = True,
    # When True (default), emits custom stream events for every tool call that passes AI review

    review_mode: Literal["llm", "agent"] = "llm",
    # "llm" = single LLM call per tool call (fast, default)
    # "agent" = dedicated review agent with read-only backend tools
    #           Backend tools → agent review; user tools → llm review

    agent_max_steps: int = 5,
    # Max steps for the review agent (only used in agent mode)

    agent_tools: frozenset[str] | None = None,
    # Backend tool names to expose to the review agent in agent mode
    # None = no extra tools (same as empty set); must be explicitly specified

    tool_unpackers: list[object] | None = None,
    # Tool-unpacker callables: resolve wrapped tools (e.g. mcp_call_tool)
    # into their inner tool identity.  Pass mcp.tool_unpacker for MCP tools.
)
```

**Human approval resume protocol:**

When the security review determines a tool call requires human approval, the agent pauses execution.
You must pass the approval decision via `Command(resume=...)` to resume:

```python
from langgraph.types import Command

# Resume with approval decisions
agent.invoke(
    Command(resume={
        "source": "mambo_security_review",
        "decisions": [
            {"tool_call_id": "call_abc123", "type": "approve"}
        ]
    }),
    config={"configurable": {"thread_id": "session-1"}},
)
```

**Decision types:**

| type | Description |
|------|-------------|
| `"approve"` | Approve, execute with original parameters |
| `"edit"` | Execute with modified parameters, must include `"edited_action": {"name": "...", "args": {...}}` |
| `"reject"` | Reject, do not execute, optionally include `"message"` with reason |
| `"respond"` | Reject but give feedback to agent, include `"message"` telling agent how to adjust |

> **The `source` field is required** — it lets the system identify this as a security review reply
> (rather than an interrupt from another component). If omitted, the approval decision is ignored
> and the tool executes with its original parameters.

### 10.x Reviewing MCP Tools

When using `MCPMiddleware`, MCP tools are exposed via the `mcp_call_tool` wrapper.
By default the security review middleware can only see `mcp_call_tool` — not the inner
MCP tool being invoked.  Use `mcp_tool_name()` and `tool_unpackers` to enable
targeted review of specific MCP tools:

```python
from mambo_agents.middleware.mcp import (
    MCPMiddleware, MCPServerConfig, mcp_tool_name,
)
from mambo_agents.middleware.security_review import SecurityReviewConfig

mcp = MCPMiddleware(servers=[...])

agent = create_mambo_agent(
    "gpt-4o",
    middleware=[mcp],
    interrupt_on={
        "mcp_call_tool": True,                         # catch-all
        mcp_tool_name("filesystem", "delete_config"): True,  # targeted
    },
    security_review=SecurityReviewConfig(
        review_tools=frozenset([
            mcp_tool_name("filesystem", "delete_config"),  # AI review
            # mcp_tool_name("filesystem", "read_file") omitted → direct HITL
        ]),
        tool_unpackers=[mcp.tool_unpacker],
    ),
)
```

`mcp_tool_name("filesystem", "delete_config")` returns `"filesystem__delete_config"` —
a stable name that works identically in both wrapped and direct MCP modes.
See `example/10_mcp_security_review.py` for a full runnable example.

---

## 11. File Change History & Rollback

Automatically snapshots file changes at checkpoint granularity, with **manual rollback** support.
No tools are exposed to the LLM — version data is purely for caller-side consumption (e.g. a web UI).
Rollback is triggered explicitly by the user via `restore_files()` — there is no automatic rollback.

**Design principles:**
- Storage via LangGraph `BaseStore` — blobs and indices are persisted through `BaseStore`, compatible
  with `InMemoryStore`, Postgres, or any `BaseStore` implementation
- Write-time persistence — each mutation backup writes its blob and updates the index atomically.
  Survives `astream` interruption
- Incremental — only files actually mutated by the LLM are backed up
- Content-addressed — SHA256 blobs; identical content stored once
- **Manual rollback only** — users call `restore_files()` explicitly. No automatic rollback

**Simplest usage:**

```python
agent = create_mambo_agent(
    "gpt-4o",
    backend=LocalBackend(),
    version_control=True,  # auto-enables version control
)
```

**With a custom `VersionStore`:**

```python
from langgraph.store.memory import InMemoryStore
from mambo_agents.middleware.version_control import VersionStore

store = VersionStore(store=InMemoryStore())

agent = create_mambo_agent(
    "gpt-4o",
    backend=LocalBackend(),
    version_control=store,
)
```

**With full `VersionControlConfig`:**

```python
from langgraph.store.memory import InMemoryStore
from mambo_agents.middleware.version_control import VersionControlConfig

agent = create_mambo_agent(
    "gpt-4o",
    backend=LocalBackend(),
    version_control=VersionControlConfig(
        store=InMemoryStore(),
        whitelist_folders=["/workspace/src", "/workspace/tests"],
        mutating_tool_names=["write", "edit", "delete", "patch"],
    ),
)
```

**Or pass the middleware directly:**

```python
from mambo_agents.middleware.version_control import (
    BackupEvent,
    VersionStore,
    VersionControlMiddleware,
)

store = VersionStore(store=InMemoryStore())
vc_middleware = VersionControlMiddleware(
    store=store,
    backend=local_backend,
    whitelist_folders=["/workspace/src"],
)

agent = create_mambo_agent(
    "gpt-4o",
    backend=local_backend,
    middleware=[vc_middleware],
)
```

**Receiving backup events via custom stream:**

```python
async for mode, chunk in agent.astream(
    {"messages": [...]}, config, stream_mode=["updates", "custom"],
):
    if mode == "custom":
        event = BackupEvent(**chunk)
        print(f"[backup] ckpt={event.checkpoint_id} file={event.file_path}")
```

**Manual rollback via `restore_files()`:**

```python
# Restore specific files to a previous checkpoint
vc_middleware.restore_files("thread-1", "cp_abc123", files=["/workspace/src/main.py"])

# Or restore all changed files at that checkpoint
vc_middleware.restore_files("thread-1", "cp_abc123", all=True)
```

**Caller-side query API (`VersionStore`):**

```python
store = VersionStore(store=InMemoryStore())

# All unique files changed across the entire session
all_files = store.get_all_changed_files("thread-123")

# Files changed in the latest turn
latest_files = store.get_latest_changed_files("thread-123")

# Full snapshot for the latest turn
snapshot = store.get_latest_snapshot("thread-123")
print(snapshot.checkpoint_id, snapshot.timestamp, snapshot.file_blobs)

# Per-checkpoint queries
store.list_snapshots("thread-123")
store.get_changed_files("thread-123", "cp_x")
store.get_file("thread-123", "cp_x", "/path")
```

**`VersionControlConfig` full fields:**

```python
from mambo_agents.middleware.version_control import VersionControlConfig

VersionControlConfig(
    store: BaseStore | None = None,
    # LangGraph BaseStore for version data persistence.
    # None = auto-resolved from graph execution context.

    whitelist_folders: list[VirtualPath] = [],
    # Absolute virtual paths to monitor. Empty = no files processed.

    mutating_tool_names: list[str] = ["write", "edit", "delete"],
    # Tool names that trigger pre-mutation backups.
)
```

---

## 12. Integrating External MCP Tools

Integrate MCP (Model Context Protocol) tools into the agent.

**Design:** Uses a disclosure-based approach — only two meta-tools are exposed
(`mcp_get_tool_description` and `mcp_call_tool`) instead of registering all MCP tools directly.
This keeps the system prompt compact even when MCP servers expose hundreds of tools;
the agent looks up descriptions and calls tools on demand.

```python
from mambo_agents.middleware.mcp import MCPMiddleware, MCPServerConfig

# stdio mode — spawn a local MCP server process
middleware = MCPMiddleware(
    servers=[
        MCPServerConfig(
            name="filesystem",
            transport="stdio",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        ),
        MCPServerConfig(
            name="weather",
            transport="stdio",
            command="python",
            args=["weather_server.py"],
            env={"API_KEY": "xxx"},
        ),
    ],
)

# HTTP mode — connect to a remote MCP server
middleware = MCPMiddleware(
    servers=[
        MCPServerConfig(
            name="remote-tools",
            transport="sse",
            url="https://example.com/mcp/sse",
            headers={"Authorization": "Bearer xxx"},
        ),
    ],
)

agent = create_mambo_agent(
    "gpt-4o",
    middleware=[middleware],
)
```

**`MCPServerConfig` parameters:**

| Parameter | Type | Required | Description |
|-----------|------|:---:|-------------|
| `name` | `str` | ✅ | Unique MCP server name |
| `transport` | `"stdio" \| "sse" \| "streamable_http" \| "websocket"` | ❌ | Transport type, default `"stdio"` |
| `command` | `str` | for stdio | Executable command (stdio mode) |
| `args` | `list[str]` | ❌ | Command-line arguments (stdio mode) |
| `env` | `dict[str, str]` | ❌ | Environment variables (stdio mode) |
| `cwd` | `str` | ❌ | Working directory (stdio mode) |
| `url` | `str` | for HTTP | Server URL (sse / streamable_http / websocket) |
| `headers` | `dict` | ❌ | HTTP headers (HTTP mode) |
| `timeout` | `float` | ❌ | HTTP timeout (seconds) |
| `sse_read_timeout` | `float` | ❌ | SSE read timeout (seconds) |

### 12.x `exclude_tools` — Hiding Dangerous Tools

Prevent specific MCP tools from being exposed to the LLM:

```python
mcp = MCPMiddleware(
    servers=[...],
    exclude_tools={
        "filesystem": frozenset(["send_to_external", "install_package"]),
        "github": frozenset(["force_push"]),
    },
)
```

Excluded tools are removed from the tool index before registration — they cannot
be discovered via ``mcp_get_tool_description`` and cannot be called via
``mcp_call_tool``.

### 12.x `direct_tool_threshold` — Direct vs Wrapped Mode

Control whether MCP tools are registered directly or behind the wrapper
meta-tools.  The default threshold is **15**: when the total number of MCP
tools across all servers does not exceed this number, each tool is registered as
a first-class tool named ``server__tool``.  Above the threshold, the
``mcp_call_tool`` / ``mcp_get_tool_description`` wrapper is used instead.

```python
mcp = MCPMiddleware(
    servers=[...],
    direct_tool_threshold=10,  # default 15; set to 0 to force full wrapping
)
```

In both modes ``mcp_tool_name(server, tool)`` and ``tool_unpacker`` work
identically — your ``interrupt_on`` and ``review_tools`` config does not
need to change when the threshold is adjusted.

### 12.x Security Review Integration

MCP tools can be selectively reviewed by the security review middleware.
Use `mcp_tool_name(server, tool)` to construct the name for `interrupt_on` and
`review_tools`, and pass `mcp.tool_unpacker` via `tool_unpackers`:

```python
from mambo_agents.middleware.mcp import (
    MCPMiddleware, MCPServerConfig, mcp_tool_name,
)

mcp = MCPMiddleware(servers=[...])

agent = create_mambo_agent(
    "gpt-4o",
    middleware=[mcp],
    interrupt_on={
        "mcp_call_tool": True,
        mcp_tool_name("filesystem", "delete_config"): True,
    },
    security_review=SecurityReviewConfig(
        review_tools=frozenset([
            mcp_tool_name("filesystem", "delete_config"),
        ]),
        tool_unpackers=[mcp.tool_unpacker],
    ),
)
```

### 12.x `mcp_tool_name()` Reference

``mcp_tool_name(server_name, tool_name) → str`` returns the effective tool name
used in `interrupt_on` and `review_tools`:

```python
mcp_tool_name("filesystem", "delete_config")  # → "filesystem__delete_config"
```

- **Consistent**: the same string works whether MCP is in wrapped or direct mode.
- **Safe**: server names are validated at init (no `__`, max 64 chars, alphanumeric + `_` `-`).
- See `example/10_mcp_security_review.py` and `example/mcp_demo_server.py` for
a fully runnable end-to-end demo.

---

## 13. Multi-Agent Collaboration

### 13.1 Synchronous Sub-agents

Sub-agents are **short-lived, isolated agents** invoked via the `task` tool, returning a single
result to the main agent.

```python
from mambo_agents.middleware.subagents import SubAgent

agent = create_mambo_agent(
    "gpt-4o",
    subagents=[
        SubAgent(
            name="researcher",
            description="Research topics thoroughly and return structured findings",
            system_prompt="You are a research specialist...",
            model="gpt-4o",
            tools=[search_tool, web_fetch_tool],
        ),
        SubAgent(
            name="code-reviewer",
            description="Review code for bugs, security issues, and style",
            system_prompt="You are a senior code reviewer...",
            model="gpt-4o",
            tools=[],  # read-only analysis, no tools needed
        ),
    ],
)
```

**Sub-agent specification (`SubAgent` Pydantic model):**

| Field | Required | Description |
|-------|:---:|-------------|
| `name` | ✅ | Unique identifier |
| `description` | ✅ | What it does (the agent uses this to decide when to delegate) |
| `system_prompt` | ✅ | Instructions for the sub-agent |
| `model` | ✅ | LLM model (must be explicitly provided except for general-purpose sub-agents) |
| `tools` | ❌ | Available tools (default empty) |
| `middleware` | ❌ | Extra middleware |
| `interrupt_on` | ❌ | Sub-agent level human approval |

**Pre-compiled sub-agents:**

```python
from mambo_agents import create_mambo_agent, CompiledSubAgent

# Build a sub-agent graph with create_mambo_agent
my_custom_graph = create_mambo_agent(
    "gpt-4o-mini",
    system_prompt="You are a code review expert...",
    tools=[lint_tool],
)

agent = create_mambo_agent(
    "gpt-4o",
    subagents=[
        CompiledSubAgent(
            name="code-reviewer",
            description="Review code for bugs and style issues",
            runnable=my_custom_graph,  # pre-compiled agent graph
        ),
    ],
)
```

**Event Granularity (`EventGranularity`):**

Controls the granularity of sub-agent internal events flowing through `stream_mode="custom"`,
set via the `subagent_event_granularity` parameter:

| Value | Description |
|-------|-------------|
| `"messages"` | Finest — LLM token-level streaming |
| `"updates"` | Default — per-node state updates |
| `"values"` | Coarsest — full snapshot at each graph step |

```python
# Receiving sub-agent streaming events
async for event in agent.astream(
    {"messages": [HumanMessage("Research Python async patterns")]},
    stream_mode=["updates", "custom"],  # "custom" channel for sub-agent events
):
    if event[0] == "custom":
        data = event[1]
        # data["type"]: "subagent_event"
        # data["tool_call_id"]: associated task call
        # data["subagent_type"]: sub-agent name
        # data["granularity"]: event granularity
        # data["chunk"]: sub-agent's streaming data
```

### 13.2 General-purpose Sub-agent

Setting `include_general_purpose=True` automatically creates a `general-purpose` sub-agent that
shares the main agent's model and backend tools, suitable for isolating complex multi-step subtasks:

```python
agent = create_mambo_agent(
    "gpt-4o",
    include_general_purpose=True,
)
```

Once created, the agent's system prompt includes `task` tool usage guidance, instructing it to
**auto-decide** when to delegate complex tasks to the sub-agent (e.g. parallel research,
large-scale search, etc.).

> **Note:** If you've already manually defined a sub-agent named `general-purpose` in `subagents`,
> `include_general_purpose=True` will not create a duplicate — your manual definition takes precedence.

### 13.3 Async Sub-agents

Unlike synchronous sub-agents, async sub-agents run in a background thread.
`async_task()` returns a `task_id` immediately.

```python
from mambo_agents.middleware.subagents import SubAgent

agent = create_mambo_agent(
    "gpt-4o",
    async_subagents=[
        SubAgent(
            name="deployer",
            description="Deploy services to Kubernetes",
            system_prompt="You are a deployment expert...",
            model="gpt-4o",
            tools=[kubectl_tool, helm_tool],
        ),
    ],
    async_subagent_timeout=1800,  # 30-minute timeout
)

result = agent.invoke(
    {"messages": [HumanMessage("Deploy v2.3 to production")]},
    config={"configurable": {"thread_id": "session-1"}},
)
# Agent: "Launched task a3f4b2c1, running in background..."

# Query later
result = agent.invoke(
    {"messages": [HumanMessage("Check status of a3f4b2c1")]},
    config={"configurable": {"thread_id": "session-1"}},
)
# Agent reads progress and results via async_status
```

**Async sub-agent capabilities:**

| Method | Description |
|--------|-------------|
| `async_task()` | Launch a background sub-agent, returns `task_id` immediately |
| `async_status(task_id)` | Query status: `running` (with progress), `success`, `error`, `cancelled`, `crashed` |
| `async_list(status_filter)` | List all async tasks (useful when the LLM forgets a task_id) |
| `async_cancel(task_id)` | Cancel a running task |
| `report_progress(message, percentage)` | Sub-agent self-reports progress from within |

**Crash recovery:** If a persistent checkpointer (e.g. `SqliteSaver`) is used, after a system restart,
calling `async_status()` or `async_list()` will auto-detect tasks that were in `running` state but
whose threads have been lost, marking them as `crashed`. The agent can then decide whether to restart.

> With the default `InMemorySaver`, state is not persisted, and all task records are lost on restart.

---

## 14. Goal-Driven Loop Control

**Goal-driven loop control** forces the agent loop to keep working until a goal is satisfied, a completion condition is met, or the round budget is exhausted. Without it, an LLM may stop after a single turn even when the task is unfinished. Enable it by passing `goal_loop=GoalLoopConfig(...)` to `create_mambo_agent()`.

Two modes share the same `GoalLoopMiddleware` core:

| Mode | Control | Registered tools | Typical use |
|------|---------|------------------|-------------|
| `"preset"` | User-controlled | `get_goal` only | Force the LLM to do something before finishing — e.g. "must call `show` at least once" |
| `"llm"` (default) | LLM-controlled | `create_goal` / `update_goal` / `get_goal` | Long-running tasks the LLM creates itself and drives to completion |

**How it works:** after every turn, the middleware's `after_agent` hook inspects the goal state. While the goal is `active` and no exit condition is met, it injects a synthetic `get_goal()` tool call into the last AI message and routes the graph back into the tool loop (`jump_to="tools"`). The model reads the goal, the current round and the instructions, then keeps working. The loop ends when the goal is satisfied / completed / blocked, or when the round budget runs out (status `timeout`).

### 14.1 Preset Mode — User-controlled

The goal is preset in the config (`objective`); completion is decided by `conditions` callbacks. Only `get_goal` is registered, and the preset goal is injected on the first turn:

```python
from mambo_agents import create_mambo_agent
from mambo_agents.middleware import GoalLoopConfig, tool_called_at_least

agent = create_mambo_agent(
    model,
    tools=[show_tool],
    goal_loop=GoalLoopConfig(
        mode="preset",
        objective="必须调用 show 工具向用户展示工作成果",
        conditions=[tool_called_at_least("show", 1)],
        max_rounds=4,  # at most 4 after_agent visits → at most 3 forced continuations
    ),
)
```

- `conditions` are OR-ed — any satisfied condition ends the loop. An empty list ends only when the round budget is exhausted.
- `tool_called_at_least(name, times=1, args_subset=None)` counts **model-initiated** calls of `name` in the current turn window; injected `get_goal` calls never satisfy a condition. Its dynamic progress is reported inside `get_goal` results, e.g. `工具 "show" 已调用 0/1 次(未满足)`.
- **Rounds are counted per user-turn window**: only injected `get_goal` calls after the latest `HumanMessage` count toward `max_rounds`. A new user message naturally resets the budget for the next turn.

### 14.2 LLM Mode — Autonomous Long-running Tasks

The LLM autonomously creates a goal with `create_goal` and drives it to completion. All three tools are registered:

```python
agent = create_mambo_agent(
    model,
    goal_loop=GoalLoopConfig(
        mode="llm",    # registers create_goal / update_goal / get_goal
        max_rounds=3,  # hard cap; create_goal's own round number never exceeds this
    ),
)
```

Typical lifecycle:

1. `create_goal(objective, max_goal_rounds=None)` — enter long-run mode. The round number never exceeds the middleware `max_rounds`. Simple tasks (< 3 steps) should not create a goal.
2. At the end of every round the system auto-injects `get_goal`, which returns the goal, the current round and instructions — the model keeps working.
3. `update_goal(goal_id, revision, action, ...)` ends or adjusts the loop:
   - `action="complete"` — declare completion; the loop ends and the model gives a wrap-up summary based only on facts verified in the session.
   - `action="blocked"` — declare blockage; `blocked_reason` is required, and it is accepted only after the model has worked at least `blocked_threshold` rounds (default 3). The loop then ends.
   - `action="edit"` — modify `objective` or `max_goal_rounds` in place.
   - **Read-then-write guard:** you must pass the exact `goal_id` and `revision` returned by `get_goal`; mismatched writes are rejected with an error.
4. If `rounds >= max_rounds` without completion, the goal is marked `timeout` and the model is asked to summarize progress and the unfinished parts.

### 14.3 Config Reference & Result

`GoalLoopConfig` parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mode` | `"llm"` | `"preset"` (requires `objective` + `conditions`) or `"llm"` |
| `max_rounds` | `256` | Round budget (number of `after_agent` visits) before the loop is force-stopped |
| `objective` | `None` | Preset objective — required in preset mode, forbidden in llm mode |
| `conditions` | `None` | Completion conditions (preset mode only, OR semantics) |
| `blocked_threshold` | `3` | Worked rounds required before one `blocked` declaration is accepted (llm mode only) |
| `tool_prefix` | `""` | Optional prefix for goal tool names, to avoid collisions |
| `system_prompt` | `None` | Custom system prompt (`None` = built-in per mode, `""` = disable injection) |

The goal state is exposed on the invoke result via `result.get("goal")`:

```python
goal = result.get("goal")
# {id, objective, status: "active"|"complete"|"blocked"|"timeout",
#  rounds, max_rounds, revision, blocked_reason, created_by: "preset"|"llm"}
```

The `goal` channel is managed entirely by the middleware (`GoalLoopState`) and excluded from user input — callers never pass it via `.invoke()`.

> **Full runnable demo:** `python example/18_goal_loop.py` shows both modes and prints the complete execution timeline, marking middleware-injected `get_goal` calls as `[自动注入]` and model-initiated ones as `[模型主动]`.

---

## 15. Advanced Usage

### 15.1 Custom System Prompt

```python
agent = create_mambo_agent(
    "gpt-4o",
    system_prompt="""You are a Python expert assistant.

## Coding Standards
- Always use type hints
- Follow PEP 8 style
- Include docstrings
""",
)
```

### 15.2 Adding Custom Tools

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get weather for a city"""
    return f"{city}: Sunny, 25°C"

agent = create_mambo_agent(
    "gpt-4o",
    tools=[get_weather],
)
```

### 15.3 Pre-populated Files

```python
agent = create_mambo_agent(
    "gpt-4o",
    backend=StoreBackend(initial_files={
        "/workspace/app/main.py": "def main():\n    print('hello')",
        "/workspace/app/config.yaml": "port: 8080\ndebug: true",
        "/workspace/app/requirements.txt": "fastapi==0.100.0\nuvicorn==0.23.0",
    }),
)
```

### 15.4 Custom Summarization Config

```python
agent = create_mambo_agent(
    "gpt-4o",
    summarization={
        "trigger": ("tokens", 50000),        # lower threshold, compact more often
        "keep": ("messages", 10),             # keep fewer messages
        "model": "gpt-4o-mini",               # use a cheap model for summarization
        "trim_tokens_to_summarize": 2000,     # review 2000 tokens when summarizing
        "offload_to_backend": True,           # persist compacted messages
        "summary_prompt": "Please concisely summarize the key information...",
    },
)
```

### 15.5 Checkpoint Persistence

```python
from langgraph.checkpoint.sqlite import SqliteSaver

agent = create_mambo_agent(
    "gpt-4o",
    checkpointer=SqliteSaver.from_conn_string("checkpoints.db"),
)
```

### 15.6 Skills + Sub-agents Combined

```python
from mambo_agents.middleware.subagents import SubAgent
from mambo_agents.middleware.planning import MamboPlanMiddleware

agent = create_mambo_agent(
    "gpt-4o",
    skills=["/skills/team/"],
    subagents=[
        SubAgent(
            name="analyst",
            description="Data analysis expert",
            system_prompt="You are a data analysis expert...",
            model="gpt-4o",
            tools=[pandas_tool],
        ),
    ],
    summarization={
        "trigger": ("tokens", 200000),
        "keep": ("messages", 20),
    },
    middleware=[MamboPlanMiddleware()],
)
```

### 15.7 Version Control + Memory + HITL Combined

```python
from mambo_agents.backends.schemas import VirtualPath
from mambo_agents.middleware.security_review import SecurityReviewConfig

agent = create_mambo_agent(
    "gpt-4o",
    backend=LocalBackend(root_dir="/tmp/project"),
    memory_sources=[VirtualPath("/.mambo/memory/AGENTS.md")],
    version_control=True,
    interrupt_on={"write": True, "edit": True},
    security_review=SecurityReviewConfig(model="gpt-4o-mini"),
)
```
