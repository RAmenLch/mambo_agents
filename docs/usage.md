> 📖 **English** | [中文](usage.cn.md)

# Mambo Agents Usage Guide

## Table of Contents

1. [Installation](#1-installation)
2. [Core Concepts](#2-core-concepts)
3. [Quick Start](#3-quick-start)
4. [Agent Factory Functions](#4-agent-factory-functions)
5. [Backend System](#5-backend-system)
6. [Middleware Reference](#6-middleware-reference)
7. [Sub-agent System](#7-sub-agent-system)
8. [Advanced Usage](#8-advanced-usage)
9. [API Reference](#9-api-reference)

---

## 1. Installation

```bash
pip install mambo-agents
```

Dependencies:
- Python >= 3.11
- langchain + langgraph
- paramiko >= 3.0 (required for SshBackend)
- wcmatch >= 8.0

---

## 2. Core Concepts

Mambo Agents builds on LangGraph, assembling agents via the **factory function + middleware stack** pattern.

### Three-layer Architecture

| Layer | Component | Responsibility |
|-------|-----------|----------------|
| **Backend** | `BackendProtocol` | Filesystem abstraction: 6 core operations + extension tools |
| **Middleware** | `AgentMiddleware` | Cross-cutting concerns: tool registration, summarization, planning, security review |
| **Agent** | `create_mambo_agent()` | Assembles backend + middleware, returns a compiled LangGraph state graph |

---

## 3. Quick Start

### 3.1 Simplest Usage

```python
from mambo_agents import create_mambo_agent, StateBackend
from langchain_core.messages import HumanMessage

agent = create_mambo_agent("gpt-4o")

result = agent.invoke({
    "messages": [HumanMessage("Create a Python script that prints Hello World")]
})
```

### 3.2 Working with a Local File System

```python
from mambo_agents.backends.local import LocalBackend

agent = create_mambo_agent(
    "gpt-4o",
    backend=LocalBackend(root_dir="/tmp/myproject"),
)

result = agent.invoke({
    "messages": [HumanMessage("List all files in the current directory")]
})
```

### 3.3 Human-in-the-Loop Approvals

```python
agent = create_mambo_agent(
    "gpt-4o",
    interrupt_on={
        "write": True,   # require approval before writing files
        "edit": True,    # require approval before editing files
        "delete": True,  # require approval before deleting files
    },
)
```

### 3.4 Streaming Output

```python
agent = create_mambo_agent("gpt-4o")

async for event in agent.astream(
    {"messages": [HumanMessage("Analyze the project code structure")]},
    stream_mode=["updates", "custom"],
):
    print(event)
```

When sub-agents are enabled, `stream_mode="custom"` receives real-time progress events from within those sub-agents.

### 3.5 Multi-turn Conversations

```python
config = {"configurable": {"thread_id": "session-1"}}

# Turn 1
result1 = agent.invoke(
    {"messages": [HumanMessage("Create a config.json")]},
    config=config,
)

# Turn 2 — the agent remembers the file it created
result2 = agent.invoke(
    {"messages": [HumanMessage("Add a port field to config.json")]},
    config=config,
)
```

---

## 4. Agent Factory Functions

### 4.1 `create_mambo_agent()` — Fine-grained Mode

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
    event_granularity: EventGranularity = "updates",
    middleware: Sequence[AgentMiddleware] | None = None,
    summarization: SummarizationConfig | None = None,
    skills: Sequence[SkillSource] | None = None,
    memory_sources: list[str] | None = None,
    tools: Sequence[BaseTool] | None = None,
    interrupt_on: dict[str, bool | InterruptOnConfig] | None = None,
    security_review: SecurityReviewConfig | None = None,
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
| `backend` | `BackendProtocol` | `StateBackend()` | Filesystem backend |
| `system_prompt` | `str` | built-in default | System prompt |
| `summarization` | `SummarizationConfig` | `None` | Conversation summarization config |
| `subagents` | `list` | `None` | Synchronous sub-agent specs |
| `include_general_purpose` | `bool` | `False` | Add a general-purpose sub-agent |
| `async_subagents` | `list` | `None` | Async sub-agent specs |
| `async_subagent_timeout` | `float` | `3600.0` | Timeout for async sub-agents (seconds) |
| `skills` | `list` | `None` | Skill source paths |
| `memory_sources` | `list` | `None` | Memory file paths (AGENTS.md list) |
| `tools` | `list` | `None` | Extra tools |
| `interrupt_on` | `dict` | `None` | Tool approval config |
| `security_review` | `SecurityReviewConfig` | `None` | AI security pre-review config |

**Example:**

```python
from mambo_agents import create_mambo_agent, StateBackend

agent = create_mambo_agent(
    "gpt-4o",
    backend=StateBackend(initial_files={
        "/config.json": '{"port": 8080}',
    }),
    summarization={
        "trigger": ("tokens", 200000),
        "keep": ("messages", 20),
    },
    skills=["/skills/user/"],
)
```



---

## 5. Backend System

### 5.1 BackendProtocol — Abstract Protocol

Every backend must implement 6 core operations:

| Method | Description |
|--------|-------------|
| `ls(path)` | List directory contents (non-recursive) |
| `read(file_path, offset, limit, include_line_numbers)` | Read file content |
| `write(file_path, content, overwrite)` | Create / overwrite a file |
| `edit(file_path, old_str, new_str, replace_all)` | Replace text in a file |
| `grep(pattern, path, glob)` | Search text content |
| `glob(pattern, path)` | Find files by wildcard pattern |

Each backend can also expose extra tools via its `tools` property (e.g. `tree`, `delete`, `execute`).

### 5.2 StateBackend — In-memory Filesystem

Files are stored in the LangGraph `files` state channel, automatically participating in checkpointing.

```python
from mambo_agents import StateBackend

backend = StateBackend(
    initial_files={
        "/config.json": '{"port": 8080}',
        "/README.md": "# My Project",
    }
)
```

**Extra tools:** `tree`

**Characteristics:**
- Automatic LangGraph checkpoint integration (pause / resume / rollback)
- Thread-safe (`thread_id` isolation)
- Lightweight, no disk access

### 5.3 LocalBackend — Local Filesystem

Direct access to real disk files, with optional shell command execution.

```python
from mambo_agents.backends.local import LocalBackend

backend = LocalBackend(
    root_dir="/home/user/project",
    timeout=120,              # shell command timeout (seconds)
    max_output_bytes=100000,  # max output bytes
    enable_execute=True,      # enable shell execution (default False)
    inherit_env=True,         # inherit system environment variables
)
```

**Extra tools:** `tree`, `delete`, `execute` (requires enabling)

**`grep` performance strategy:**
1. Prefer system `rg` (ripgrep) — orders of magnitude faster than Python traversal
2. Fall back to Python traversal (with file size guard)

> **⚠️ Security Warning:** `LocalBackend` provides direct filesystem access and shell execution. Consider using `interrupt_on` + `security_review` alongside it.

### 5.4 SshBackend — Remote SSH Filesystem

Operate on remote server files via SSH/SFTP.

```python
from mambo_agents.backends.ssh import SshBackend

backend = SshBackend(
    host="192.168.1.100",
    username="deploy",
    password="secret123",          # or key_filename="/path/to/key"
    port=22,
    remote_root="/home/deploy/app",
    connect_timeout=30,
    execute_timeout=120,
)
```

**Extra tools:** `tree`, `delete`, `execute`

**Performance strategy:** Batch operations (`grep`, `glob`, `edit`, `tree`) execute remotely to avoid per-file SFTP round-trips. `edit` completes find-and-replace in a single `python3 -c` invocation on the remote side.

### 5.5 HybridWorkspaceBackend — Multi-backend Routing

One real backend + N virtual workspaces, all routed under `/.mambo/`.
Each virtual workspace is backed by an independent `StateBackend` and only supports core protocol tools.

```python
from mambo_agents.backends.hybrid_workspace import HybridWorkspaceBackend
from mambo_agents.backends.local import LocalBackend
from mambo_agents import StateBackend

# Simplest: auto-create /.mambo/ default StateBackend
backend = HybridWorkspaceBackend(
    real_backend=LocalBackend(root_dir="/tmp/project"),
)

# Multiple virtual workspaces
backend = HybridWorkspaceBackend(
    real_backend=LocalBackend(root_dir="/tmp/project"),
    virtual_workspaces={
        "skills": StateBackend(initial_files={"/python.md": "..."}),
        "cache": StateBackend(),
    },
)

# Override default /.mambo/
backend = HybridWorkspaceBackend(
    real_backend=LocalBackend(root_dir="/tmp/project"),
    virtual_workspaces={
        ".": StateBackend(initial_files={"/config.yml": "..."}),
    },
)
```

**Path routing rules:**
- `/.mambo/skills/xxx` → "skills" virtual workspace (prefix stripped, passes `/xxx`)
- `/.mambo/xxx` → default StateBackend (prefix stripped, passes `/xxx`)
- Everything else → real backend

**Use cases:**
- Middleware internal storage (large result eviction, conversation history dumps)
- Agent scratch files
- Sub-agent communication files
- Independent isolated spaces for multiple skills/modules

### 5.6 Backend Comparison

| Feature | StateBackend | LocalBackend | SshBackend | HybridWorkspaceBackend |
|---------|:---:|:---:|:---:|:---:|
| Storage Location | Memory | Local Disk | Remote Server | Hybrid |
| Checkpoint Support | Automatic | Manual | Manual | Automatic (/.mambo/) |
| Shell Execution | ❌ | Optional | ✅ | ❌ |
| Delete Operation | ❌ | ✅ | ✅ | ❌ |
| grep Acceleration | N/A | ripgrep | Remote rg/grep | Inherits delegate |
| Network Dependency | ❌ | ❌ | ✅ | Optional |
| Best For | Testing / Prototyping | Local Development | Remote Deployment | Production |

### 5.7 ReadSummarizer — Large File Read Summaries

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

---

## 6. Middleware Reference

### 6.1 Middleware Stack Order

`create_mambo_agent()` assembles middleware in a fixed order:

```
1. BackendToolsMiddleware           ← registers filesystem tools (always enabled)
2. [SkillsMiddleware]               ← skill loading (when skills is not None)
3. [MamboMemoryMiddleware]          ← memory loading (when memory_sources is not None)
4. [MamboSummarizationMiddleware]   ← conversation summarization (when summarization is not None)
5. [user-defined middleware]        ← passed via middleware parameter
6. [SubAgentMiddleware]             ← synchronous sub-agents
7. [AsyncSubAgentMiddleware]        ← async sub-agents
8. [AutoSecurityReviewMiddleware | HumanInTheLoopMiddleware]  ← security review
9. PatchToolCallsMiddleware         ← fix dangling tool calls (always enabled)
10. ReorderToolMessagesMiddleware    ← reorder tool messages (always enabled)
```

### 6.2 BackendToolsMiddleware

**Capabilities:**
- Registers 6 core filesystem tools (`ls`, `read`, `write`, `edit`, `grep`, `glob`)
- Merges backend extension tools
- Auto-evicts oversized tool results to `/.mambo/large_tool_results/`

**Large Result Eviction:** When a tool result exceeds 20,000 tokens, the full content is written to the filesystem and the inline message is replaced with a preview + file path.

### 6.3 MamboSummarizationMiddleware

**Purpose:** Automatically compact long conversation histories to prevent LLM context window overflow.

```python
# Configure via create_mambo_agent
agent = create_mambo_agent(
    "gpt-4o",
    summarization={
        "trigger": ("tokens", 200000),   # trigger when total exceeds 200k tokens
        "keep": ("messages", 20),         # keep the last 20 messages uncompacted
        "offload_to_backend": True,       # persist evicted messages to backend
    },
)
```

**Two trigger modes:**

| Mode | Example | Description |
|------|---------|-------------|
| tokens | `("tokens", 200000)` | Cumulative token count exceeds threshold |
| messages | `("messages", 50)` | Message count exceeds threshold |

**Chained Summaries:** When compaction fires multiple times, prior summaries are injected as non-negotiable historical context into the summarization prompt, preventing information loss across rounds.

**Summary Hooks:** Allow other middleware to inject additional context during summarization. `MamboPlanMiddleware` uses this mechanism to preserve the current plan state during compaction.

### 6.4 MamboPlanMiddleware

**Purpose:** Provides a `write_plans` tool for the agent to maintain a structured TODO list.

```python
from mambo_agents.middleware.planning import MamboPlanMiddleware

# Via create_mambo_agent's middleware parameter
agent = create_mambo_agent(
    "gpt-4o",
    middleware=[MamboPlanMiddleware()],
)

# Or enable it via create_mambo_agent
agent = create_mambo_agent(
    "gpt-4o",
    middleware=[MamboPlanMiddleware()],
)
```

**Plan data model:**

```python
from mambo_agents import Plan

# Each plan item has two fields:
# - content: task description
# - status: "pending" | "in_progress" | "completed"
```

### 6.5 SkillsMiddleware

**Purpose:** Progressive disclosure of skills — loaded into the prompt only when the agent needs them.

```python
agent = create_mambo_agent(
    "gpt-4o",
    skills=[
        "/skills/user/",                              # user-level skills
        "/skills/project/",                           # project-level skills
        ("/repo/.claude/skills", "Project Claude"),   # with custom label
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
| bare path | `"/skills/user/"` | Last path component, uppercased |
| tuple | `("/path", "My Skill")` | Custom label |

Multi-source loading: later-loaded skills override earlier ones with the same name (last wins).

### 6.6 AutoSecurityReviewMiddleware

**Purpose:** Review tool calls for safety before actual execution (or before pausing for human approval), using a cheaper model.

Two review modes:

- **llm** (default): single structured-output LLM call per tool call — fast and cheap.
- **agent**: a dedicated review agent with read-only backend tools that can inspect the workspace before delivering a verdict via ``最终审核结果``. Backend tools (core 6 + ``backend.tools``) get agent review; non-backend user tools fall back to llm review.

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

# Agent-mode — backend tools reviewed by agent with read-only workspace
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
        model="gpt-4o-mini",                     # review model
        review_tools=frozenset(["write"]),       # only review write
        system_prompt="You are a security audit expert...",
    ),
)
```

**Workflow:**

```
Tool Call → AI Security Review → Safe: pass through
                               → High Risk: pause → Human Approval
```

### 6.7 PatchToolCallsMiddleware & ReorderToolMessagesMiddleware

These two are always enabled (no configuration needed):

- **PatchToolCallsMiddleware:** Fixes dangling tool calls in the message history (e.g. when a human interruption leaves an `AIMessage.tool_calls` without a matching `ToolMessage`)
- **ReorderToolMessagesMiddleware:** Reorders `ToolMessage` instances to match `AIMessage.tool_calls` order, preventing misinterpretation by multi-modal models

---

## 7. Sub-agent System

### 7.1 Synchronous Sub-agents

Sub-agents are **short-lived, isolated agents** invoked via the `task` tool, returning a single result to the main agent.

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
| `model` | ✅ | LLM model to use |
| `tools` | ✅ | Available tools |
| `middleware` | ❌ | Extra middleware |
| `interrupt_on` | ❌ | Sub-agent level human-in-the-loop |

**Pre-compiled sub-agents:**

```python
from mambo_agents.middleware.subagents import CompiledSubAgent

# Any runnable works (must have a 'messages' state key)
compiled = CompiledSubAgent(
    name="custom-processor",
    description="Custom processing pipeline",
    runnable=my_custom_graph,
)
```

**Event Granularity (`EventGranularity`):**

| Value | Description |
|-------|-------------|
| `"messages"` | Finest — LLM token-level streaming |
| `"updates"` | Default — per-node state updates |
| `"values"` | Coarsest — full snapshot at each graph step |

```python
# Consuming sub-agent streaming events
async for event in agent.astream(
    {"messages": [HumanMessage("Research Python async patterns")]},
    stream_mode=["updates", "custom"],
):
    if event[0] == "custom":
        custom_data = event[1]
        # custom_data includes: tool_call_id, subagent_type, chunk, timestamp
```

### 7.2 General-purpose Sub-agent

Setting `include_general_purpose=True` automatically creates a `general-purpose` sub-agent that shares the main agent's model, backend tools, and system prompt.

```python
agent = create_mambo_agent(
    "gpt-4o",
    include_general_purpose=True,
)

# The agent will auto-delegate complex multi-step tasks to this sub-agent
```

### 7.3 Async Sub-agents

Unlike sync sub-agents, async sub-agents run in a background thread. `async_task()` returns a `task_id` immediately.

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
    {"messages": [HumanMessage("Deploy v2.3 to production")]}
)
# Agent: "Launched task a3f4b2c1, running in background..."

# Query later
result = agent.invoke(
    {"messages": [HumanMessage("Check status of a3f4b2c1")]}
)
# Agent reads progress and results via async_status
```

**Async sub-agent capabilities:**

| Method | Description |
|--------|-------------|
| `async_task()` | Launch a background sub-agent, returns `task_id` immediately |
| `async_status(task_id)` | Query status: `running` (with progress), `success`, `error`, `cancelled`, `crashed` |
| `async_list(status_filter)` | List all async tasks (useful when the LLM forgets a task_id) |
| `report_progress(message, percentage)` | Sub-agent self-reports progress from within |

**Crash recovery:** On restart, tasks that were in `running` state are detected and marked as `crashed`.

---

## 8. Advanced Usage

### 8.1 Custom System Prompt

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

### 8.2 Adding Custom Tools

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

### 8.3 Pre-populated Files

```python
agent = create_mambo_agent(
    "gpt-4o",
    backend=StateBackend(initial_files={
        "/app/main.py": "def main():\n    print('hello')",
        "/app/config.yaml": "port: 8080\ndebug: true",
        "/app/requirements.txt": "fastapi==0.100.0\nuvicorn==0.23.0",
    }),
)
```

### 8.4 Custom Summarization Config

```python
agent = create_mambo_agent(
    "gpt-4o",
    summarization={
        "trigger": ("tokens", 50000),         # lower threshold, compact more often
        "keep": ("messages", 10),              # keep fewer messages
        "model": "gpt-4o-mini",                # use a cheap model for summarization
        "trim_tokens_to_summarize": 2000,      # review 2000 tokens when summarizing
        "offload_to_backend": True,            # persist compacted messages
        "summary_prompt": "Please concisely summarize the key information from the following conversation...",
    },
)
```

### 8.5 Checkpoint Persistence

```python
from langgraph.checkpoint.sqlite import SqliteSaver

agent = create_mambo_agent(
    "gpt-4o",
    checkpointer=SqliteSaver.from_conn_string("checkpoints.db"),
)
```

### 8.6 Skills + Sub-agents Combined

```python
from mambo_agents.middleware.subagents import SubAgent

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

### 8.7 Interactive CLI Testing

```bash
# Basic usage (in-memory backend)
python -m mambo_agents.cli.chat

# Specify model and working directory
python -m mambo_agents.cli.chat --model gpt-4o --workspace /tmp/test

# With general-purpose sub-agent
python -m mambo_agents.cli.chat --general-purpose

# With custom sub-agent
python -m mambo_agents.cli.chat --subagent researcher:Research specialist:You are a research specialist...
```

---

## 9. API Reference

### 9.1 Public API Exports

```python
from mambo_agents import (
    # Factory functions
    create_mambo_agent,

    # Backend
    BackendProtocol,
    StateBackend,
    HybridWorkspaceBackend,
    FileData,
    FilesystemState,

    # Synchronous sub-agents
    SubAgent,
    CompiledSubAgent,
    SubAgentMiddleware,
    EventGranularity,

    # Async sub-agents
    AsyncSubAgentMiddleware,
    AsyncTaskData,

    # Task planning
    MamboPlanMiddleware,
    Plan,
    WritePlansInput,

    # Conversation summarization
    MamboSummarizationMiddleware,
    SummarizationConfig,
    SummaryHook,
    SummaryHookContext,

    # Skills
    SkillsMiddleware,
    SkillMetadata,
    SkillSource,

    # Memory
    MamboMemoryMiddleware,
    MemoryFormatHook,
)
```

### 9.2 Backend Protocol Result Types

| Type | Purpose |
|------|---------|
| `LsResult` | Directory listing result |
| `ReadResult` | File read result (text or multimodal) |
| `WriteResult` | File write result |
| `EditResult` | File edit result |
| `GrepResult` | Text search matches |
| `GlobResult` | File glob matches |
| `FileInfo` | Single file / directory metadata |
| `GrepMatch` | Single grep match |

### 9.3 Security Review Config

```python
from mambo_agents.middleware.security_review import SecurityReviewConfig

SecurityReviewConfig(
    model: str | BaseChatModel | None = None,
    # None = reuse the agent's model
    # "gpt-4o-mini" = use a cheap model for review

    review_tools: Literal["all"] | frozenset[str] = "all",
    # "all" = review all interrupt_on tools
    # frozenset({"write", "edit"}) = only review specified tools

    system_prompt: str | None = None,
    # None = use built-in security review prompt
    # custom = override the review prompt

    review_mode: Literal["llm", "agent"] = "llm",
    # "llm" = single LLM call per tool call (fast, default)
    # "agent" = dedicated review agent with read-only backend tools
    #           Backend tools → agent review; user tools → llm review

    agent_max_steps: int = 5,
    # Max steps for the review agent (only used in agent mode)

    agent_tools: frozenset[str] | None = None,
    # Backend tool names to expose to the review agent in agent mode
    # None = all registered backend tools are available
)
```

### 9.4 HITL Interrupt / Resume Protocol

When ``AutoSecurityReviewMiddleware`` escalates tool calls for human review, it
issues a LangGraph ``interrupt()`` with the following payload structure:

```json
{
    "source": "mambo_security_review",
    "action_requests": [
        {
            "name": "write",
            "args": {"file_path": "/etc/hosts", "content": "..."},
            "tool_call_id": "call_abc123",
            "description": null
        }
    ],
    "review_configs": [
        {
            "action_name": "write",
            "tool_call_id": "call_abc123",
            "allowed_decisions": ["approve", "edit", "reject", "respond"]
        }
    ]
}
```

Your HITL infrastructure **must** include the ``"source"`` field in the resume
value when resuming the graph via ``Command(resume=...)``:

```json
{
    "source": "mambo_security_review",
    "decisions": [
        {"tool_call_id": "call_abc123", "decision": "approve"}
    ]
}
```

The ``source`` field serves two purposes:

1. **Consumer routing:** your UI can distinguish security review interrupts from
   other interrupt types (e.g. custom ``interrupt()`` calls inside tools).
2. **Replay detection:** on resume, the middleware inspects the resume value
   (non-consumingly) to decide whether it should enter the replay branch.
   Only values carrying ``"source": "mambo_security_review"`` are recognized.

> **Important:** omitting ``"source"`` causes the middleware to treat the resume
> as "not ours" and transparently pass through.  Human decisions will not be
> applied and tool calls will execute with their original arguments.

### 9.5 Summarization Config

```python
SummarizationConfig = {
    "trigger": ("tokens", 200000) | ("messages", 50) | None,
    "keep": ("messages", 20) | ("tokens", 5000),
    "model": str | BaseChatModel | None,
    "trim_tokens_to_summarize": int,        # default 4000
    "token_counter": Callable | None,
    "chars_per_token": float | None,
    "offload_to_backend": bool,             # default False
    "backend": BackendProtocol | None,
    "summary_prompt": str | None,
    "chained_summary_prompt": str | None,
    "summary_hooks": list[SummaryHook] | None,
}
```
