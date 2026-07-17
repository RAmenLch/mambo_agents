> 📖 **English** | [中文](deepagents_reference.cn.md)

# Reference Statement Regarding deepagents

## 1. Acknowledgement

**Mambo Agents** acknowledges and thanks the [deepagents](https://github.com/langchain-ai/deepagents) project (v0.5.7, by the LangChain team) for its open-source work. This project has undergone substantial refactoring and extension on top of deepagents' foundational architecture, with core inspiration drawn from its two primary design paradigms: the **Middleware Pipeline** and the **Backend Protocol** abstraction.

Specifically, we drew architectural ideas from the following deepagents components:

| Architectural Dimension | Reference Source | Description |
|-------------------------|------------------|-------------|
| Middleware Pipeline | `deepagents.graph.create_deep_agent()` | Chaining `AgentMiddleware` instances to intercept the agent lifecycle |
| Backend Protocol | `deepagents.backends.protocol.BackendProtocol` | Six core file operations: `ls` / `read` / `write` / `edit` / `grep` / `glob` |
| State Backend | `deepagents.backends.state.StateBackend` | In-memory file storage via LangGraph state channels |
| Filesystem Backend | `deepagents.backends.filesystem.FilesystemBackend` | Real filesystem wrapping |
| Skills System | `deepagents.middleware.skills.SkillsMiddleware` | Progressive disclosure via `SKILL.md` |
| Sub-agent System | `deepagents.middleware.subagents.SubAgentMiddleware` | Sync / async sub-agent delegation |
| Conversation Summarization | `deepagents.middleware.summarization.SummarizationMiddleware` | Context window compaction |
| Tool Call Patching | `deepagents.middleware.patch_tool_calls.PatchToolCallsMiddleware` | Repairing dangling tool calls |

> **Important:** This project is not a fork or branch of deepagents. It is an agent framework implemented under the guidance of deepagents' architectural ideas. Some modules (e.g. `PatchToolCallsMiddleware`, `SkillsMiddleware`) are refactored and extended from deepagents' implementations of the same name; the remaining modules are independently written.

---

## 2. Design Directions and Trade-offs

The differences below do not imply that deepagents' design is "wrong." They reflect trade-offs made by two projects with **different goals and contexts**.

### 2.1 Multi-backend and Command Execution

| Dimension | deepagents | Mambo Agents | Trade-off |
|-----------|------------|--------------|-----------|
| **Local Filesystem + Execution** | `LocalShellBackend` (`FilesystemBackend` + `SandboxBackendProtocol`) <br>• Gains `execute()` by inheriting `SandboxBackendProtocol` <br>• Runs `subprocess.run(shell=True)` directly on host, no isolation | `LocalBackend` (self-contained `execute()`) <br>• `execute()` is a method on the backend itself, no additional protocol layer required <br>• Same direct host execution, but co-located in a single class | Mambo avoids the `LocalShellBackend` / `FilesystemBackend` class split; local file operations and local shell execution are naturally unified |
| **Remote Filesystem + Execution** | `LangSmithSandbox` (extends `BaseSandbox`) <br>• Targets LangSmith's cloud container service, achieving true process isolation <br>• All file operations delegate to `execute()`, communicating via SDK | `SshBackend` (based on paramiko) <br>• `execute()` runs remotely through an SSH channel <br>• File operations directly manipulate the remote filesystem (`ls` / `read` / `write` do not go through `execute`; they have native implementations) | deepagents' remote solution is tied to the LangSmith cloud service, suitable for pre-configured containerized environments; Mambo's SSH approach is more general — any machine running sshd works |
| **Execute Architecture Philosophy** | `execute()` requires `SandboxBackendProtocol` <br>• Protocol stack: `BackendProtocol` (files only) → `SandboxBackendProtocol` (+ execute) → `BaseSandbox` (convenience wrappers) <br>• Only backends declared as "Sandbox" can execute commands | `execute()` is an inherent, optional capability of the backend <br>• Protocol stack: only `BackendProtocol`. Any backend that wants `execute` simply implements it on its own class <br>• `LocalBackend` and `SshBackend` each carry `execute`; `StoreBackend` does not — enabled on demand, no forced classification | Mambo pursues a more general, more flexible way to enable `execute`: no separate Sandbox protocol layer, `execute` is just a regular method, each backend decides for itself whether to provide it |
| **Cross-session Persistence** | ✅ `StoreBackend` (backed by LangGraph `BaseStore`) | ✅ `StoreBackend` (backed by LangGraph `BaseStore`, `thread_id` locked at construction) | Mambo's StoreBackend is self-implemented, inherits from `BackendProtocol`, locks `thread_id` at construction for session isolation |

### 2.2 Middleware Stack Differences

| Dimension | deepagents | Mambo Agents | Trade-off |
|-----------|------------|--------------|-----------|
| **Security Review** | `HumanInTheLoopMiddleware` (simple human approval) | `AutoSecurityReviewMiddleware` (AI pre-review + human fallback; supports llm and agent review modes — agent mode runs a dedicated review agent with read-only backend tools) | Mambo adds an AI auto-review layer to reduce human approval interruption frequency; agent mode allows workspace inspection before verdict |
| **Task Planning** | `TodoListMiddleware` (simple TODO management) | `MamboPlanMiddleware` (structured plans + summarization integration hooks) | Mambo's planning is deeply coupled with the summarization system, preventing compaction from losing plan state |
| **Multi-model Compatibility** | ❌ No tool message reordering | ✅ `ReorderToolMessagesMiddleware` | Some multimodal models are sensitive to tool message ordering; Mambo handles this explicitly |
| **Tool Extensibility** | `FilesystemMiddleware` (6 core tools + backend extra tools + multimodal read) | ✅ `BackendToolsMiddleware` (6 core tools + auto-injection of backend extra tools) <br>• `include_line_numbers` parameter: Mambo's `read` does not return line numbers by default; the model can choose to pass `True` when it needs to reference specific lines (deepagents unconditionally adds line numbers, with no off switch) <br>• `build_tool_descriptions()` extracts all tool descriptions into a mapping, consumable by `AutoSecurityReviewMiddleware` for understanding tool purpose | Differentiated at the basic tool layer: controllable `include_line_numbers` reduces irrelevant noise; `build_tool_descriptions()` makes tool metadata consumable by other middleware |
| **Large Result Eviction** | ✅ `FilesystemMiddleware` built-in eviction (saves to `/large_tool_results/`) <br>• Eviction timing: `wrap_model_call` (batch processing of message history before the model call) | ✅ `BackendToolsMiddleware` eviction (saves to `/.mambo/large_tool_results/`) <br>• Eviction timing: `wrap_tool_call` (immediate interception after tool returns results, rather than waiting for the next model call) | Both implement large-result eviction (including multimodal block preservation and Command multi-message scenarios); the only difference is eviction timing: Mambo acts immediately after tool return, deepagents batches before model invocation |
| **Large File Read Limit & Summarization** | ✅ `read_file` with built-in `_truncate()`: appends a static `READ_FILE_TRUNCATION_MSG` when content exceeds the character threshold (fixed template, no file-type awareness) | ✅ `BackendProtocol` with `max_read_chars` + pluggable `ReadSummarizer` callback <br>• Callback signature `(file_path, content, max_chars) -> str`, enabling differentiated instructive summaries based on file suffix (`.py`/`.json`/`.yaml`, etc.) <br>• Default summarizer guides the model to re-read with `offset`+`limit`; users inject custom callbacks for file-type-specific strategies <br>• Binary/multimodal files are never truncated <br>• Ships `read_summarizers` sub-package with 9 pre-built summarizers (Python / JS / TS / Java / C / C++ / Go / Rust / Markdown / JSON), extracting structural outlines with precise line numbers | Both have character-level read caps and truncation notices; Mambo's core difference is the **pluggable callback** system — replacing oversized content with file-type-tailored "instructive summaries" that help the model make better next-step decisions, rather than just providing a fixed truncation warning |
| **Memory System** | ✅ `MemoryMiddleware` (`AGENTS.md`) | ✅ `MamboMemoryMiddleware` (`AGENTS.md`) | Mambo references deepagents' `MemoryMiddleware` design (`before_agent` loads AGENTS.md via `backend.download_files` into state, `wrap_model_call` injects `<agent_memory>` into the system prompt, multiple source merging), and adds an optional custom `format_prompt` callback on top |
| **Profile System** | ✅ `ProviderProfile` + `HarnessProfile` (model/provider tuning) | ❌ Not implemented | Mambo currently does not focus on model-level fine-grained tuning; configuration left to the user |
| **Anthropic Caching** | ✅ `AnthropicPromptCachingMiddleware` | ❌ Not implemented | Mambo currently does not bind to provider-specific optimizations |
| **Sub-agent State Passthrough** | Sub-agent results returned as `ToolMessage` | ✅ Sub-agents return via `Command` <br>• `_return_command_with_state_update()` transparently passes non-excluded state keys (e.g. filesystem state) from the sub-agent back to the parent <br>• Three levels of streaming event granularity (`messages` / `updates` / `values`), with `subagent_event` custom events pushing sub-agent internal progress <br>• Parallel sub-agents are distinguished by `tool_call_id` in their event streams | Sub-agents not only return results but also surface their conversational state to the parent agent, allowing the parent to understand which files were created or modified during execution |

### 2.3 Architecture Design Trade-offs

| Dimension | deepagents | Mambo Agents | Trade-off |
|-----------|------------|--------------|-----------|
| **Type Safety** | Partial use of TypedDict | ✅ End-to-end Pydantic types (no Dict/Any duck typing) | Strict type control is a core Mambo coding standard |
| **Routing Backend** | `CompositeBackend` (default + routes dict, arbitrary prefix → arbitrary backend)<br>• Fully flexible routing: `/memories/` → StoreBackend, `/cache/` → StateBackend, any combination<br>• `ls("/")` transparently aggregates all routed backends<br>• `execute()` always delegates to the default backend, determined by `SandboxBackendProtocol` type check | `HybridWorkspaceBackend` (1 real + N virtual, unified `/.mambo/` prefix)<br><br>**Hard constraints:**<br>• Virtual workspace prefix is fixed at `/.mambo/`, cannot be customized to other routing paths<br>• Virtual workspaces only expose 6 core file tools<br>• System prompt explicitly informs the AI of the above constraints<br><br>**Relaxations:**<br>• Built-in `copy` tool supports cross-backend (virtual ↔ real) single-file transfer<br>• Real backend can flexibly provide tools, not limited to just `execute` | `CompositeBackend`'s arbitrary prefix routing provides maximum flexibility, but the AI may bypass the routing layer via `execute()` (e.g. `cat /memories/...`), leading to serious hallucinations; `HybridWorkspaceBackend` uses fixed prefixes + explicit tool whitelist prompts to constrain the AI, while relaxing restrictions on the real backend's tools |

---

## 3. Feature Comparison Table

A side-by-side mapping of equivalent capabilities:

### 3.1 Backend

| Feature | deepagents | Mambo Agents |
|---------|------------|--------------|
| Protocol Definition | `BackendProtocol` + `SandboxBackendProtocol` (independent execute layer) | `BackendProtocol` (execute is also a backend method; no independent protocol layer) |
| In-memory Storage | `StateBackend` | `StoreBackend` (self-implemented, backed by LangGraph `BaseStore`, `thread_id` locked at construction) |
| Local Execute | `LocalShellBackend` (`FilesystemBackend` + `SandboxBackendProtocol`, two layers of parent classes) | `LocalBackend` (single class with built-in `execute()`, `tree`, `delete`) |
| Remote Execute | `LangSmithSandbox` (cloud container service, all file operations delegated to `execute()`) | `SshBackend` (native SSH, `execute()` via SSH channel, file operations have native implementations) |
| Path Routing | `CompositeBackend` (flexible multi-backend routing; `execute()` may bypass the routing layer) | `HybridWorkspaceBackend` (1 real + N virtual route: `/.mambo/<name>/` → memory, rest → real backend; system prompt explicitly communicates workspace semantics to the AI) |
| Cross-session Storage | `StoreBackend` | `StoreBackend` (self-implemented) |

### 3.2 Middleware

| Feature | deepagents | Mambo Agents |
|---------|------------|--------------|
| File Tool Injection | `FilesystemMiddleware` (6 core + extra tools + large result eviction + multimodal read) | `BackendToolsMiddleware` (6 core + extra tools + large result eviction + multimodal read; optional `include_line_numbers`, `build_tool_descriptions()` for external consumption) |
| Large File Read Limit & Summarization | `read_file` static truncation message (fixed template, no file-type differentiation) | ✅ `max_read_chars` + `ReadSummarizer` (pluggable callback, generates differentiated instructive summaries by file suffix) |
| Skill Disclosure | `SkillsMiddleware` | `SkillsMiddleware` (reconstructed) |
| Sync Sub-agents | `SubAgentMiddleware` | `SubAgentMiddleware` (reconstructed; `subagent_event` streaming, three granularity levels, state passthrough) |
| Async Sub-agents | `AsyncSubAgentMiddleware` | `AsyncSubAgentMiddleware` (reconstructed) |
| Conversation Summarization | `SummarizationMiddleware` | `MamboSummarizationMiddleware` (extended) <br>• **Chained Summaries**: preserves prior summary content across multiple summarization rounds; uses `CHAINED_SUMMARY_PROMPT` to instruct the model to merge historical summaries rather than overwrite, preventing cross-round information loss <br>• **CJK Token Counting**: auto-detects CJK character ratio, applying a different chars-per-token ratio than English to avoid severe token under-counting for CJK text <br>• **Latest User Message Protection**: ensures the most recent user message is never summarized away (the langchain base implementation lacks this safeguard) <br>• **Optional Backend Persistence**: evicted original messages can be written via `BackendProtocol` to `/conversation_history/{thread_id}.md` |
| Dangling Tool Call Patching | `PatchToolCallsMiddleware` | `PatchToolCallsMiddleware` (preserved) |
| Tool Message Reordering | ❌ | ✅ `ReorderToolMessagesMiddleware` |
| Security Review | `HumanInTheLoopMiddleware` | ✅ `AutoSecurityReviewMiddleware` (AI pre-review + human approval; llm/agent dual-mode) |
| Task Planning | `TodoListMiddleware` | ✅ `MamboPlanMiddleware` (structured + summarization integration) |
| Memory Loading | `MemoryMiddleware` | ✅ `MamboMemoryMiddleware` |
| Tool Exclusion | `_ToolExclusionMiddleware` | ❌ |
| Anthropic Caching | `AnthropicPromptCachingMiddleware` | ❌ |

### 3.3 Entry Points

| Feature | deepagents | Mambo Agents |
|---------|------------|--------------|
| Primary Constructor | `create_deep_agent()` | `create_mambo_agent()` |
| Profile System | ✅ `ProviderProfile` + `HarnessProfile` | ❌ |
| Sub-agent Types | `SubAgent` / `CompiledSubAgent` / `AsyncSubAgent` | `SubAgent` / `CompiledSubAgent` / `AsyncSubAgent` |

---

## Closing

deepagents is an outstanding open-source project in the Agent framework space. Its **Middleware Pipeline + Backend Protocol** architecture provided a clear blueprint for Mambo Agents. Building on that foundation, Mambo Agents has pursued refactoring and extension in the following directions: strengthened security review, added remote SSH operation capability, introduced a strict type system, and addressed specific concerns around large result handling, multi-model compatibility, and plan-summarization coordination.

We remain grateful for the deepagents team's contributions to Agent infrastructure.
