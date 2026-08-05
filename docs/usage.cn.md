> 📖 [English](usage.md) | **中文**

# Mambo Agents 详细使用文档

## 目录

1. [安装](#1-安装)
2. [核心概念](#2-核心概念)
3. [快速上手](#3-快速上手)
4. [Agent 工厂函数](#4-agent-工厂函数)
5. [让 Agent 操控环境](#5-让-agent-操控环境)
6. [超长对话管理](#6-超长对话管理)
7. [任务规划与追踪](#7-任务规划与追踪)
8. [给 Agent 装技能包](#8-给-agent-装技能包)
9. [让 Agent 记住你的偏好](#9-让-agent-记住你的偏好)
10. [安全与人工审批](#10-安全与人工审批)
    - [审查 MCP 工具](#10x-审查-mcp-工具)
11. [文件修改历史与回滚](#11-文件修改历史与回滚)
12. [接入外部 MCP 工具](#12-接入外部-mcp-工具)
    - [`exclude_tools` — 隐藏危险工具](#12x-exclude_tools--隐藏危险工具)
    - [`direct_tool_threshold` — 直传-vs-包装模式](#12x-direct_tool_threshold--直传-vs-包装模式)
    - [安全审查集成](#12x-安全审查集成)
    - [`mcp_tool_name()` 参考](#12x-mcp_tool_name-参考)
13. [多 Agent 协作](#13-多-agent-协作)
14. [高级用法](#14-高级用法)

---

## 1. 安装

```bash
pip install mambo-agents
```

---

## 2. 核心概念

Mambo Agents 在 LangGraph 基础上构建，通过**工厂函数 + 中间件栈**的模式组装 Agent。

### 三层架构

| 层级 | 组件 | 职责 |
|------|------|------|
| **后端层** | `BackendProtocol` | Agent 的"手"：文件系统抽象，提供 6 个核心操作 + 扩展工具 |
| **中间件层** | `AgentMiddleware`（来自 langchain） | 横切关注点：mambo 提供摘要、规划、技能、记忆、安全审查、版本控制、MCP 集成等 12+ 种内置中间件 |
| **Agent 层** | `create_mambo_agent()` | 组装后端 + 中间件，返回编译好的 LangGraph |

---

## 3. 快速上手

### 3.1 最简单用法

```python
from mambo_agents import create_mambo_agent
from langchain_core.messages import HumanMessage

agent = create_mambo_agent("gpt-4o")

result = agent.invoke({
    "messages": [HumanMessage("创建一个 Python 脚本，打印 Hello World")]
}, config={"configurable": {"thread_id": "session-1"}})

# 查看 Agent 的最终回复
print(result["messages"][-1].content)
# 输出示例：
# 已创建 /hello.py：
# ```python
# print("Hello World")
# ```
```

> **默认后端：** 不指定 `backend` 时，Agent 使用 `StoreBackend` + `InMemoryStore`，
> 文件存储在会话内存中，进程重启后消失。如需持久化（对接 PostgreSQL 等）或
> 操作真实磁盘，请指定 `backend=LocalBackend(...)`（见下一节）。
>
> **路径约定：** `StoreBackend` 的文件统一放在 `/workspace/` 前缀下
> （如 `/workspace/hello.py`）；前缀之外的路径（如 `/hello.py`）不会出现在
> `ls /workspace` / `glob` 视图中。详见 [5.2](#52-虚拟文件系统storebackend)。

### 3.2 操作本地文件系统

```python
from mambo_agents.backends.local import LocalBackend

agent = create_mambo_agent(
    "gpt-4o",
    backend=LocalBackend(root_dir="/tmp/myproject"),
)

result = agent.invoke({
    "messages": [HumanMessage("列出当前目录的所有文件")]
}, config={"configurable": {"thread_id": "session-1"}})
print(result["messages"][-1].content)
```

### 3.3 添加自定义工具

除了内置的文件系统工具，Agent 可以挂载任意自定义工具：

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气"""
    return f"{city}: 晴朗，25°C"

agent = create_mambo_agent(
    "gpt-4o",
    tools=[get_weather],
)

result = agent.invoke({
    "messages": [HumanMessage("查一下北京的天气")]
}, config={"configurable": {"thread_id": "session-1"}})
```

### 3.4 流式输出

**按节点输出（每个步骤完成后触发一次）：**

```python
agent = create_mambo_agent("gpt-4o")

async for event in agent.astream(
    {"messages": [HumanMessage("分析项目的代码结构")]},
    stream_mode="updates",
    config={"configurable": {"thread_id": "session-1"}},
):
    print(event)
    # 输出示例（每个节点完成后触发一次）：
    # {'model': {'messages': [AIMessage(content='我先用 ls 看看目录结构...')]}}
    # {'tools': {'messages': [ToolMessage(content='...', name='ls')]}}
    # {'model': {'messages': [AIMessage(content='项目包含以下文件...')]}}
```

**逐 token 流式输出（实时显示 LLM 生成内容）：**

```python
async for event in agent.astream(
    {"messages": [HumanMessage("解释什么是装饰器")]},
    stream_mode="messages",
    config={"configurable": {"thread_id": "session-1"}},
):
    # event 是 (message_chunk, metadata) 的元组
    msg_chunk, metadata = event
    if msg_chunk.content:
        print(msg_chunk.content, end="", flush=True)
```

> 更多流式模式（如 `custom` 用于接收子代理进度事件）见
> [13.1 节](#131-同步子代理)。

### 3.5 多轮对话

Agent 通过 `thread_id` 区分不同的对话会话。同一 `thread_id` 下的多轮调用共享
文件系统和对话历史：

```python
config = {"configurable": {"thread_id": "session-1"}}

# 第一轮
result1 = agent.invoke(
    {"messages": [HumanMessage("创建一个 config.json，设置 port 为 8080")]},
    config=config,
)

# 第二轮 — Agent 记得之前创建的文件和对话内容
result2 = agent.invoke(
    {"messages": [HumanMessage("把 port 改成 9090")]},
    config=config,
)

# 换个 thread_id 就是全新的会话
result3 = agent.invoke(
    {"messages": [HumanMessage("列出所有文件")]},
    config={"configurable": {"thread_id": "session-2"}},
)
# 结果为空 — session-2 是全新会话，看不到 session-1 的文件
```

> **`thread_id` 的作用：**
> - **对话历史隔离：** 不同 `thread_id` 的对话互不可见
> - **文件系统隔离：** `StoreBackend` 会为每个 `thread_id` 创建独立的虚拟文件系统
> - **`config` 是必需的：** 每次调用必须传入 `config={"configurable": {"thread_id": "..."}}`

---

## 4. Agent 工厂函数

### 4.1 `create_mambo_agent()`

完整的参数签名：

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

**必选参数：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `model` | `str \| BaseChatModel` | LLM 模型名称或实例 |

**常用可选参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `backend` | `BackendProtocol` | `StoreBackend()` | 文件系统后端 |
| `system_prompt` | `str` | 内置默认 | 自定义系统提示词 |
| `summarization` | `SummarizationConfig` 或 `dict` | `None` | 对话摘要配置 |
| `subagents` | `list` | `None` | 同步子代理列表 |
| `include_general_purpose` | `bool` | `False` | 是否添加通用子代理 |
| `async_subagents` | `list` | `None` | 异步子代理列表 |
| `async_subagent_timeout` | `float` | `3600.0` | 异步子代理超时（秒）|
| `subagent_event_granularity` | `EventGranularity` | `"updates"` | 子代理自定义事件的流式粒度 |
| `skills` | `list` | `None` | 技能来源路径 |
| `memory_sources` | `list[VirtualPath]` | `None` | 记忆文件路径 |
| `tools` | `list` | `None` | 额外工具 |
| `interrupt_on` | `dict` | `None` | 工具审批配置。`{"write": True}` 简单启用，或 `{"write": {"allowed_decisions": ["approve", "reject"]}}` 限制审批选项 |
| `security_review` | `SecurityReviewConfig` | `None` | AI 安全预审配置 |
| `version_control` | `VersionControlConfig` / `VersionStore` / `bool` | `None` | 版本控制配置 |
| `checkpointer` | `BaseCheckpointSaver` | `InMemorySaver()` | 检查点持久化 |
| `store` | `BaseStore` | `None` | LangGraph Store（用于后端持久化）|

**示例：**

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

## 5. 让 Agent 操控环境

Agent 的"手"——通过 `backend` 参数指定后端，决定了 Agent 能在哪里读写文件、执行命令。

### 5.1 核心操作

所有后端必须实现 6 个核心操作：

| 操作 | 说明 |
|------|------|
| `ls(path)` | 列出目录内容（非递归） |
| `read(file_path, offset, limit, include_line_numbers)` | 读取文件内容 |
| `write(file_path, content, overwrite)` | 创建/覆盖文件 |
| `edit(file_path, old_str, new_str, replace_all)` | 替换文件中的文本 |
| `grep(pattern, path, glob, regex, offset, limit)` | 搜索文本内容 |
| `glob(pattern, path)` | 按通配符查找文件和目录 |

每个后端还可以通过 `tools` 属性暴露额外的工具（如 `tree`、`delete`、`execute`；`HybridWorkspaceBackend` 额外提供 `copy`）。

> **路径约定：** 所有后端使用以 `/` 开头的绝对路径（如 `/workspace/src/main.py`）。
> 部分配置参数（如 `memory_sources`）接受 `VirtualPath` 类型，也可直接传 `str`，
> 框架会自动转换。`VirtualPath` 会校验并拒绝 `..`、`//` 等非法写法，
> 导入路径：`from mambo_agents.backends.schemas import VirtualPath`。

### 5.2 虚拟文件系统（StoreBackend）

`StoreBackend` 是一个虚拟文件系统，底层数据存储在 `BaseStore` 中。
`BaseStore` 是 LangGraph 提供的键值存储接口。

```python
from mambo_agents import StoreBackend

# 默认使用内存存储（进程重启后消失）
# 注意：文件必须放在工作区根 "/workspace" 下，才能在 Agent 默认的
# `ls /workspace` 视图中可见；不带前缀的 key 虽然能按精确路径读取，
# 但在根目录的 ls/glob 视图中不可见。
backend = StoreBackend(
    initial_files={
        "/workspace/config.json": '{"port": 8080}',
        "/workspace/README.md": "# My Project",
    }
)

# 指定持久化存储（如 PostgreSQL）
from langgraph.store.postgres import PostgresStore

backend = StoreBackend(
    store=PostgresStore.from_conn_string("postgresql://..."),
    initial_files={"/workspace/config.json": '{"port": 8080}'},
)
```

> **路径约定：** `StoreBackend` 的 `workspace_root` 固定为 `/workspace`。
> 预填充的 `initial_files` key 必须以 `/workspace/` 开头（如 `/workspace/config.json`），
> 并应引导 Agent 把文件创建在 `/workspace/` 下 —— 否则文件只能按精确路径访问，
> 不会出现在 Agent 默认的 `ls /workspace` / `glob` 视图中。

常用 `BaseStore` 实现：

| 实现 | 来源 | 持久化 | 适用场景 |
|------|------|:---:|------|
| `InMemoryStore` | `langgraph.store.memory` | ❌ | 开发 / 测试（默认） |
| `PostgresStore` | `langgraph.store.postgres` | ✅ | 生产环境 |
| 自定义 `BaseStore` | 实现 `BaseStore` 接口 | 可定制 | 对接现有存储系统 |

> `create_mambo_agent()` 的 `store` 参数就是 `BaseStore`。不传时默认使用
> `InMemoryStore`。如果需要持久化，在 `create_mambo_agent(store=PostgresStore(...))`
> 指定即可，框架会自动传给 `StoreBackend`。

**额外工具：** `tree`

**特点：**
- 会话隔离：不同 `thread_id` 的文件系统互相独立（类似每个会话有自己的虚拟磁盘）
- 图内/图外均可用
- `thread_id` 锁死在构造时，图外操作简便：
  ```python
  be = StoreBackend(thread_id="my-session")
  be.upload_files([(path, data)])  # 自动写入 "my-session"
  ```

### 5.3 操作本地磁盘（LocalBackend）

直接操作真实磁盘文件，支持 Shell 命令执行。

```python
from mambo_agents.backends.local import LocalBackend

backend = LocalBackend(
    root_dir="/home/user/project",
    timeout=120,           # Shell 命令超时（秒）
    max_output_bytes=100000,  # 最大输出字节数
    enable_execute=True,   # 启用 Shell 执行（默认 False）
    inherit_env=True,      # 继承系统环境变量
)
```

**额外工具：** `tree`、`delete`、`execute`（需启用）

**`grep` 性能策略：**
1. 优先使用系统 `rg`（ripgrep）— 比 Python 遍历快数个数量级
2. 回退到 Python 遍历（带文件大小限制）

> **💡 建议安装 ripgrep 以获得最佳性能：**
> - **macOS:** `brew install ripgrep`
> - **Linux:** `apt install ripgrep` / `dnf install ripgrep`
> - **Windows:** `winget install BurntSushi.ripgrep.MSVC` 或 `scoop install ripgrep`

> **⚠️ 安全警告：** `LocalBackend` 提供直接文件系统访问和 Shell 执行能力。建议配合 `interrupt_on` + `security_review` 使用。

### 5.4 远程服务器（SshBackend）

通过 SSH/SFTP 操作远程服务器文件。

```python
from mambo_agents.backends.ssh import SshBackend

backend = SshBackend(
    host="192.168.1.100",
    username="deploy",
    password="secret123",         # 或 key_filename="/path/to/key"
    port=22,
    remote_root="/home/deploy/app",
    connect_timeout=30,
    execute_timeout=120,
    enable_execute=False,         # 启用 Shell 执行（默认 False）
)
```

**额外工具：** `tree`、`delete`、`execute`（需启用）

**性能策略：** 批量操作（`grep`、`glob`、`edit`、`tree`）在远程执行，避免逐文件 SFTP 往返。`edit` 通过远程 `python3 -c` 一次性完成查找替换。

> **💡 建议在远程服务器上安装 ripgrep 以获得最佳 grep 性能：**
> - `apt install ripgrep` / `dnf install ripgrep` / `brew install ripgrep`

### 5.5 混合工作空间（HybridWorkspaceBackend）

真实后端 + N 个虚拟 workspace，统一在 `/.mambo/` 下路由。
每个虚拟 workspace 由任意 `BackendProtocol` 实现驱动（通常是 `StoreBackend`）。

```python
from mambo_agents.backends.hybrid_workspace import HybridWorkspaceBackend
from mambo_agents.backends.local import LocalBackend
from mambo_agents import StoreBackend

# 最简用法：自动创建 /.mambo/ 默认 StoreBackend
backend = HybridWorkspaceBackend(
    real_backend=LocalBackend(root_dir="/tmp/project"),
)

# 多虚拟 workspace
# 注意：虚拟 StoreBackend 的 workspace_root 固定为 "/workspace"，
# initial_files 的 key 必须带 "/workspace/" 前缀才能与路由重写后的
# 路径匹配 —— 否则预填充的文件 Agent 读取不到。
backend = HybridWorkspaceBackend(
    real_backend=LocalBackend(root_dir="/tmp/project"),
    virtual_workspaces={
        "skills": StoreBackend(initial_files={"/workspace/python.md": "..."}),
        "cache": StoreBackend(),
    },
)

# 覆盖默认 /.mambo/
backend = HybridWorkspaceBackend(
    real_backend=LocalBackend(root_dir="/tmp/project"),
    virtual_workspaces={
        ".": StoreBackend(initial_files={"/workspace/config.yml": "..."}),
    },
)
```

**路径路由规则：**
- `/.mambo/skills/xxx` → "skills" 虚拟 workspace（strip 前缀后，再拼上虚拟后端的 `workspace_root` —— `StoreBackend` 为 `/workspace` —— 才转发给虚拟后端）
- `/.mambo/xxx` → 默认 StoreBackend（重写规则同上）
- `/{workspace_root}/...` → 真实后端（路径会被 rewrite：strip workspace_root，prepend 真实后端的 workspace_root）
- 其他路径（如 `/`、`/etc`）将被拒绝

**虚拟 workspace 注意事项：**
- 虚拟 `StoreBackend` 的 `workspace_root` 固定为 `/workspace`（而非 `/`），因此预填充的 `initial_files` key 必须以 `/workspace/` 开头；写成 `/python.md` 会存入一个不可达的路径，Agent 无法读取。
- 虚拟后端在运行时通过 `get_store()` 从图执行上下文解析 `BaseStore`。若需要在图外访问（例如在自己的脚本里用 `download_files()` 验证文件），请给虚拟 `StoreBackend` 显式传 `store=`（默认 workspace 则给 `HybridWorkspaceBackend` 传 `store=`），让图内外共享同一个 store 实例；否则图外调用会 fallback 到私有 `InMemoryStore`，看不到 Agent 写入的文件。
- 完整可运行示例见 `example/11_hybrid_workspace.py`。

**额外工具：** `copy(source, destination)` — 跨后端单文件复制
（虚拟 ↔ 真实，或虚拟 workspace 之间），目标已存在时直接覆盖。

**用途：**
- 内部存储（大结果驱逐、对话历史转储）
- Agent 草稿文件
- 子代理通信文件
- 多技能/多模块的独立隔离空间

### 5.6 只读模式（ReadOnlyBackend）

包装任意 `BackendProtocol`，仅暴露安全的只读操作（`ls`、`read`、`grep`、`glob`）。
`AutoSecurityReviewMiddleware` 在 agent 审查模式下内部使用。

```python
from mambo_agents import ReadOnlyBackend

safe = ReadOnlyBackend(backend, allowed_extra_tools=frozenset(["tree"]))
```

### 5.7 大文件读取摘要（ReadSummarizer）

内置 `max_read_chars`（默认 100,000 字符）上限控制。超限时，用摘要替换原文而非简单截断。
用户可注入自定义 `ReadSummarizer` 回调，按文件类型生成**指导性摘要**，帮助 AI 导航到大文件的正确位置。

Mambo 提供了 `read_summarizers` 子包，含 9 种文件类型的预置摘要器：

| 摘要器 | 覆盖文件 | 解析方式 |
|--------|---------|---------|
| `python_summarizer()` | `.py` | `ast`（标准库） |
| `javascript_summarizer()` | `.js` / `.ts` / `.tsx` | tree-sitter |
| `java_summarizer()` | `.java` | tree-sitter |
| `c_summarizer()` | `.c` / `.h` | tree-sitter |
| `cpp_summarizer()` | `.cpp` / `.hpp` / `.cc` / `.hxx` | tree-sitter |
| `go_summarizer()` | `.go` | tree-sitter |
| `rust_summarizer()` | `.rs` | tree-sitter |
| `markdown_summarizer()` | `.md` / `.mdx` | 正则 |
| `json_summarizer()` | `.json` | `json`（标准库） |

每个摘要器会提取文件的结构大纲和**精确行号**，例如 Python 的类/函数、Markdown 的标题层级、JSON 的顶层键结构。

```python
from mambo_agents.read_summarizers import python_summarizer
from mambo_agents.backends.local import LocalBackend

backend = LocalBackend(summarizer=python_summarizer())
```

摘要器**不会默认注入** — 用户按需选择。未匹配后缀的文件回退到默认行为（提示分段读取）。

多个摘要器可通过 `composite_summarizer()` 组合使用：

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

### 5.8 后端对比

| 特性 | StoreBackend | LocalBackend | SshBackend | HybridWorkspaceBackend |
|------|:---:|:---:|:---:|:---:|
| 存储位置 | LangGraph Store（可配置） | 本地磁盘 | 远程服务器 | 混合 |
| 会话隔离 | 自动 | 手动 | 手动 | 自动(/.mambo/) |
| Shell 执行 | ❌ | 可选 | 可选 | 取决于真实后端 |
| 删除操作 | ❌ | ✅ | ✅ | 取决于真实后端 |
| 复制操作 | ❌ | ❌ | ❌ | ✅ |
| grep 加速 | N/A | ripgrep | 远程 rg/grep | 继承委托 |
| 网络依赖 | ❌ | ❌ | ✅ | 可选 |
| 适用场景 | 测试/原型 | 本地开发 | 远程部署 | 生产环境 |

---

## 6. 超长对话管理

自动压缩长对话历史，防止超出 LLM 上下文窗口。

```python
# 通过 create_mambo_agent 配置
agent = create_mambo_agent(
    "gpt-4o",
    summarization={
        "trigger": ("tokens", 200000),  # 累计超过 200k tokens 时触发
        "keep": ("messages", 20),        # 保留最后 20 条消息不压缩
        "offload_to_backend": True,      # 被驱逐的消息持久化到后端
    },
)
```

**触发的两种方式：**

| 方式 | 示例 | 说明 |
|------|------|------|
| tokens | `("tokens", 200000)` | 累计 token 数超过阈值 |
| messages | `("messages", 50)` | 消息数量超过阈值 |

**摘要模式（`SummarizationMode`）：**

| 模式 | 说明 |
|------|------|
| `PER_ASTREAM` | （默认）在开始前执行一次摘要，之后整个运行周期不再检查 |
| `PER_MODEL_CALL` | 每次模型调用时都检查是否需要摘要，即使在运行过程中 |

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

**链式摘要：** 多次触发压缩时，之前的摘要作为"不可协商的历史上下文"注入摘要提示词，保证信息不丢失。

**摘要钩子（SummaryHook）：** 允许其他中间件在摘要时注入额外上下文。`MamboPlanMiddleware` 通过此机制在摘要发生时注入当前计划状态。

**`SummarizationConfig` 完整字段：**

```python
from mambo_agents import SummarizationConfig, SummarizationMode

SummarizationConfig(
    mode: SummarizationMode = SummarizationMode.PER_ASTREAM,
    # SummarizationMode.PER_ASTREAM（默认）：在开始前执行一次摘要
    # SummarizationMode.PER_MODEL_CALL：每次模型调用时都检查是否需要摘要

    trigger: ("tokens", 200000) | ("messages", 50) | None = None,
    # 触发条件 — 累计超过阈值时触发摘要。None = 不触发

    keep: ("messages", 20) | ("tokens", 5000) = ("messages", 20),
    # 保留最近的消息不被压缩

    model: str | BaseChatModel | None = None,
    # None = 复用 Agent 模型

    trim_tokens_to_summarize: int = 4000,
    # 摘要时回顾的 token 数

    token_counter: Callable | None = None,
    # 自定义 token 计数器

    chars_per_token: float | None = None,
    # 自定义字符/token 比例

    offload_to_backend: bool = False,
    # True 时被压缩的消息持久化到后端

    backend: BackendProtocol | None = None,
    # offload_to_backend 使用的后端

    summary_prompt: str = DEFAULT_MAMBO_SUMMARY_PROMPT,
    # 内置摘要提示词（含 {messages} 占位符）

    chained_summary_prompt: str | None = None,
    # 链式摘要提示词（之前已有摘要时使用）

    summary_hooks: list[SummaryHook] | None = None,
    # 摘要时注入额外上下文的钩子
)
```

### 6.1 获取摘要事件（SummarizationEvent）

当摘要被触发时，中间件会将摘要事件存储在 Agent 的私有状态 `_summarization_event` 中。你可以通过 `stream_mode="values"` 获取完整状态并检查该字段。

**`SummarizationEvent` 结构：**

| 字段 | 类型 | 含义 |
|------|------|------|
| `cutoff_index` | `int` | `state["messages"]` 中的绝对索引。此索引**之前**的所有消息已被摘要替换为一条 `summary_message`。 |
| `summary_message` | `HumanMessage` | LLM 生成的摘要消息，包含三个标准部分：`SESSION INTENT`（会话目标）、`SUMMARY`（关键决策与结论）、`ARTIFACTS`（创建/修改的文件及变更描述）。消息标记有 `additional_kwargs={"lc_source": "summarization"}` 用于链式摘要识别。 |
| `file_path` | `str \| None` | 当 `offload_to_backend=True` 时，被驱逐的消息会持久化到后端的 `/.mambo/conversation_history/{thread_id}.md` 路径。`None` 表示未开启 offload 或 offload 失败。 |
| `last_summarized_message` | `AnyMessage \| None` | 被摘要区域中最后一条真实消息（不含摘要标记消息）。用于了解压缩窗口的精确边界。 |

**检测摘要事件的方式：**

```python
from mambo_agents.middleware.summarization import SummarizationEvent

# 方式 1：通过 stream_mode="values" 获取 state，检查 _summarization_event
last_state = None
for state in agent.stream(
    {"messages": [HumanMessage("...")]},
    config=config,
    stream_mode="values",
):
    last_state = state

# 检查是否有摘要事件
sum_event: SummarizationEvent | None = last_state.get("_summarization_event")
if sum_event:
    print(f"摘要已触发！前 {sum_event['cutoff_index']} 条消息已被压缩")
    print(f"摘要内容：{sum_event['summary_message'].content[:500]}...")
    print(f"持久化路径：{sum_event.get('file_path')}")
```

**实时监听摘要事件（检测新触发的摘要）：**

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
        print(f"⚡ 新的摘要事件：已压缩 {sum_event['cutoff_index']} 条消息")
        print(f"   摘要长度：{len(sum_event['summary_message'].content)} 字符")
```

> **注意：** `_summarization_event` 是私有状态字段（以 `_` 开头），仅供调用方观测使用，不会暴露给 LLM。链式摘要时，之前的摘要会自动注入到新的摘要提示词中，确保信息不丢失。

---

## 7. 任务规划与追踪

让 Agent 维护结构化的 TODO 列表，自动追踪任务进度。

```python
from mambo_agents.middleware.planning import MamboPlanMiddleware

agent = create_mambo_agent(
    "gpt-4o",
    middleware=[MamboPlanMiddleware()],
)
```

**计划数据模型：**

```python
from mambo_agents.middleware.planning import Plan

# 每个计划项包含两个字段：
# - content: 任务描述
# - status: "pending" | "in_progress" | "completed"
```

启用后，Agent 会获得一个 `write_plans` 工具，在执行复杂任务时自动拆分步骤并追踪完成状态。

**获取计划状态的方式：**

```python
from mambo_agents.middleware.planning import Plan

# 方式 1：invoke 后从返回值中读取 plans
result = agent.invoke(
    {"messages": [HumanMessage("帮我创建项目骨架...")]},
    config=config,
)
plans: list[Plan] | None = result.get("plans")
if plans:
    for p in plans:
        print(f"[{p.status}] {p.content}")

# 方式 2：stream_mode="updates" 时，观察 write_plans 工具调用
for event in agent.stream(
    {"messages": [HumanMessage("...")]},
    config=config,
    stream_mode="updates",
):
    for node_name, node_output in event.items():
        if node_name == "model":
            # AIMessage.tool_calls 中可看到 write_plans 被调用及参数
            for msg in node_output.get("messages", []):
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        if tc["name"] == "write_plans":
                            print(f"计划：{tc['args']['plans']}")
        elif node_name == "tools":
            # ToolMessage 中可看到 write_plans 的返回值
            for msg in node_output.get("messages", []):
                if msg.name == "write_plans":
                    print(f"计划更新结果：{msg.content}")

# 方式 3：stream_mode="values" 时，实时观察 plans 变化
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

## 8. 给 Agent 装技能包

渐进式披露的技能系统 — 技能只在 Agent 需要时才加载到提示词中，避免一次性塞入过多上下文。

```python
agent = create_mambo_agent(
    "gpt-4o",
    skills=[
        "/skills/user/",                           # 用户级技能
        "/skills/project/",                        # 项目级技能
        ("/repo/.claude/skills", "Project Claude"),  # 带标签
    ],
)
```

**技能文件结构：**

```
/skills/user/web-research/
├── SKILL.md          # 必需：YAML frontmatter + markdown 指令
└── helper.py         # 可选：辅助文件
```

**`SKILL.md` 格式：**

```markdown
---
name: web-research
description: 结构化网页研究的方法论
license: MIT
---
# Web Research Skill

## 步骤
1. 明确研究目标
2. 搜索关键词
...
```

**技能来源（SkillSource）：**

| 类型 | 示例 | 标签来源 |
|------|------|----------|
| 裸路径 | `"/skills/user/"` | 路径最后组件首字母大写（`"/skills/user/"` → `User`；特例：`built_in_skills` → `Built-in`，`skills` 叶子上跳一级） |
| 元组 | `("/path", "我的技能")` | 自定义标签 |

多来源加载：后加载的技能覆盖同名的先前技能（后胜出）。

---

## 9. 让 Agent 记住你的偏好

从 AGENTS.md 文件加载持久上下文，并指导 AI 在交互中**学习回写**新知识。
与技能（按需加载）不同，记忆始终加载到系统提示词中，提供跨轮次的持久上下文。

**核心机制：** `MamboMemoryMiddleware` 在 `wrap_model_call` 阶段将 AGENTS.md 文件内容通过 `MAMBO_MEMORY_SYSTEM_PROMPT` 模板注入到 System Prompt 末尾。注入后的 System Prompt 结构为：

```
原始 system_prompt（来自 BackendToolsMiddleware）
    ↓
<agent_memory>
AGENTS.md 文件内容（文件路径 + 正文）
</agent_memory>
    ↓
<memory_guidelines>
指导 Agent 何时回写、如何学习的规则
</memory_guidelines>
```

```python
from mambo_agents.backends.schemas import VirtualPath

agent = create_mambo_agent(
    "gpt-4o",
    memory_sources=[VirtualPath("/.mambo/memory/AGENTS.md")],
)
```

**工作流程：**

```
启动会话 → before_agent 加载 AGENTS.md → wrap_model_call 注入 system prompt
    ↓
交互中 → AI 发现值得记住的信息 → 用 edit/write 回写 AGENTS.md
```

**记忆内容格式（AGENTS.md）：**

AGENTS.md 是标准 Markdown 文件，无必须结构。常见内容：
- 项目概述与架构说明
- 构建/测试命令
- 代码风格指南
- 用户偏好与约定

**AI 自主学习：**

记忆提示词会指导 AI 在以下情况下用 `edit`/`write` 回写 AGENTS.md：
- 用户明确要求记住某事
- 用户提供可复用的上下文（编码风格、约定、工作流）
- 用户对 AI 工作提出了反馈和纠正
- **不**记录：临时信息、一次性任务、闲聊、凭据

**查看注入效果（截取真实 System Prompt）：**

`MamboMemoryMiddleware` 在 `wrap_model_call` 阶段通过 `modify_request` 将 memory 注入到 `request.system_message`。可以写一个拦截 middleware 排在 memory 之后，从 Agent 运行时截取真实的注入结果：

```python
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import SystemMessage

class SystemPromptInterceptor(AgentMiddleware):
    """排在 memory middleware 之后，截取注入后的 system prompt。"""
    captured: str | None = None

    def wrap_model_call(self, request: ModelRequest, handler) -> ModelResponse:
        sm = request.system_message
        if isinstance(sm, SystemMessage) and isinstance(sm.content, str):
            self.captured = sm.content  # 这就是 Agent 实际收到的完整 system prompt
        return handler(request)

# 使用
interceptor = SystemPromptInterceptor()
agent = create_mambo_agent(
    "gpt-4o",
    memory_sources=[VirtualPath("/.mambo/memory/AGENTS.md")],
    middleware=[interceptor],  # 排在 memory 后面
)

agent.invoke({"messages": [...]}, config=config)
# 调用后 interceptor.captured 中即包含 memory 注入后的完整 system prompt
```

**自定义格式化：**

```python
from mambo_agents.middleware.memory import MamboMemoryMiddleware

def my_formatter(contents: dict[str, str]) -> str:
    """自定义 memory 在 system prompt 中的展示方式。"""
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

## 10. 安全与人工审批

在实际执行工具（或暂停人工审批）之前，用廉价模型审查工具调用的安全性。

两种审查模式：

- **llm**（默认）：每次工具调用单次结构化输出 LLM 调用——快速且低成本。
- **agent**：专用审查 agent，带有只读后端工具，可在提交审查结论前检查工作区。Backend 工具（核心 6 个 + `backend.tools`）使用 agent 审查；非 backend 用户工具回退到 llm 审查。

```python
# 经典 HITL（无 AI 预审）
agent = create_mambo_agent(
    "gpt-4o",
    interrupt_on={"write": True, "edit": True},
)

# AI 预审 — llm 模式（默认）
from mambo_agents.middleware.security_review import SecurityReviewConfig

agent = create_mambo_agent(
    "gpt-4o",
    interrupt_on={"write": True, "edit": True, "delete": True},
    security_review=SecurityReviewConfig(),
)

# agent 模式 — backend 工具由审查 agent 审核（带只读工作区）
agent = create_mambo_agent(
    "gpt-4o",
    interrupt_on={"write": True, "edit": True, "delete": True},
    security_review=SecurityReviewConfig(
        review_mode="agent",
        agent_max_steps=5,
    ),
)

# 自定义预审 — 仅审查特定工具，使用独立模型
agent = create_mambo_agent(
    "gpt-4o",
    interrupt_on={"write": True, "edit": True},
    security_review=SecurityReviewConfig(
        model="gpt-4o-mini",                    # 审查模型
        review_tools=frozenset(["write"]),      # 只审查 write
        system_prompt="你是安全审计专家...",
    ),
)
```

**工作流：**

```
工具调用 → AI 安全审查 → 安全：放行
                        → 高风险：暂停 → 人工审批
```

**`SecurityReviewConfig` 完整字段：**

```python
from mambo_agents.middleware.security_review import SecurityReviewConfig

SecurityReviewConfig(
    model: str | BaseChatModel | None = None,
    # None = 复用 Agent 模型
    # "gpt-4o-mini" = 使用廉价模型审查

    system_prompt: str | None = None,
    # None = 使用内置安全审查提示词
    # 自定义 = 覆盖审查提示词

    review_tools: Literal["all"] | frozenset[str] = "all",
    # "all" = 审查所有 interrupt_on 工具
    # frozenset({"write", "edit"}) = 仅审查指定工具

    notify_on_pass: bool = True,
    # True（默认）时，每个通过 AI 审查的工具调用会发出自定义流事件

    review_mode: Literal["llm", "agent"] = "llm",
    # "llm" = 每次工具调用单次 LLM 调用（快速，默认）
    # "agent" = 专用审查 agent，带只读后端工具
    #           Backend 工具 → agent 审查；用户工具 → llm 审查

    agent_max_steps: int = 5,
    # 审查 agent 的最大步数（仅在 agent 模式下使用）

    agent_tools: frozenset[str] | None = None,
    # agent 模式下暴露给审查 agent 的后端工具名列表
    # None = 无额外工具（与空集一致）；必须显式指定

    tool_unpackers: list[object] | None = None,
    # 工具解包器列表：将包装工具（如 mcp_call_tool）解析为内层工具身份
    # 用于 MCP 工具审查（传入 mcp.tool_unpacker）
)
```

**人工审批的恢复协议：**

当安全审查判定某工具调用需要人工审批时，Agent 会暂停执行。你需要通过
`Command(resume=...)` 传入审批决定来恢复运行：

```python
from langgraph.types import Command

# 恢复时传回审批决定
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

**decision 类型：**

| type | 说明 |
|------|------|
| `"approve"` | 批准，按原参数执行 |
| `"edit"` | 修改参数后执行，需附带 `"edited_action": {"name": "...", "args": {...}}` |
| `"reject"` | 拒绝，不执行，可附带 `"message"` 说明原因 |
| `"respond"` | 拒绝但给 Agent 反馈，附带 `"message"` 告知 Agent 如何调整 |

> **`source` 字段是必须的** — 它让系统能识别这是安全审查的回复（而非其他组件的中断）。
> 如果省略，审批决定会被忽略，工具将以原始参数执行。

### 10.x 审查 MCP 工具

使用 `MCPMiddleware` 时，MCP 工具通过 `mcp_call_tool` 包装暴露。安全审查中间件默认只能看到
`mcp_call_tool`，无法识别内层 MCP 工具。使用 `mcp_tool_name()` 和 `tool_unpackers` 即可
对特定 MCP 工具做针对性审查：

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
        "mcp_call_tool": True,                         # 兜底：所有 MCP 调用
        mcp_tool_name("filesystem", "delete_config"): True,  # 精确匹配
    },
    security_review=SecurityReviewConfig(
        review_tools=frozenset([
            mcp_tool_name("filesystem", "delete_config"),  # AI 预审
            # mcp_tool_name("filesystem", "read_file") 不加 → 直接人工审批
        ]),
        tool_unpackers=[mcp.tool_unpacker],
    ),
)
```

`mcp_tool_name("filesystem", "delete_config")` 返回 `"filesystem__delete_config"` —
一个稳定的名称，在包装模式和透传模式中保持一致。完整示例见 `example/10_mcp_security_review.py`。

---

## 11. 文件修改历史与回滚

以 checkpoint 为粒度自动快照文件变更，支持**手动回滚**。不向 LLM 暴露任何工具，版本数据仅供调用方（如 Web 应用）查询使用。回滚由用户显式调用 `restore_files()` 触发 — 没有自动回滚机制。

**设计原则：**
- 存储基于 LangGraph `BaseStore` — blob 和索引通过 `BaseStore` 持久化，兼容 `InMemoryStore`、Postgres 等任何 `BaseStore` 实现
- 即时持久化 — 每次变更时备份，`astream` 中断后数据不丢失
- 增量快照 — 只备份 LLM 实际变更的文件
- 内容寻址 — SHA256 存储，相同内容只存一份
- **仅手动回滚** — 用户显式调用 `restore_files()`，没有自动回滚

**最简用法：**

```python
agent = create_mambo_agent(
    "gpt-4o",
    backend=LocalBackend(),
    version_control=True,  # 自动启用版本控制
)
```

**使用自定义 `VersionStore`：**

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

**使用完整 `VersionControlConfig`：**

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

**或直接传入中间件实例：**

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

**通过 custom stream 接收备份事件：**

```python
async for mode, chunk in agent.astream(
    {"messages": [...]}, config, stream_mode=["updates", "custom"],
):
    if mode == "custom":
        event = BackupEvent(**chunk)
        print(f"[backup] ckpt={event.checkpoint_id} file={event.file_path}")
```

**手动回滚 `restore_files()`：**

```python
# 将指定文件恢复到某个 checkpoint 的状态
vc_middleware.restore_files("thread-1", "cp_abc123", files=["/workspace/src/main.py"])

# 或恢复该 checkpoint 下所有变更的文件
vc_middleware.restore_files("thread-1", "cp_abc123", all=True)
```

**调用方查询 API（`VersionStore`）：**

```python
store = VersionStore(store=InMemoryStore())

# 整个对话会话中改过哪些文件（去重）
all_files = store.get_all_changed_files("thread-123")

# 最新一轮改了什么
latest_files = store.get_latest_changed_files("thread-123")

# 获取最新一轮的完整快照
snapshot = store.get_latest_snapshot("thread-123")
print(snapshot.checkpoint_id, snapshot.timestamp, snapshot.file_blobs)

# 按 checkpoint 查询
store.list_snapshots("thread-123")
store.get_changed_files("thread-123", "cp_x")
store.get_file("thread-123", "cp_x", "/path")
```

**`VersionControlConfig` 完整字段：**

```python
from mambo_agents.middleware.version_control import VersionControlConfig

VersionControlConfig(
    store: BaseStore | None = None,
    # LangGraph BaseStore 用于版本数据持久化。
    # None = 从图执行上下文自动解析。

    whitelist_folders: list[VirtualPath] = [],
    # 要监控的虚拟路径绝对路径列表。空列表 = 不处理任何文件。

    mutating_tool_names: list[str] = ["write", "edit", "delete"],
    # 触发预变更备份的工具名称列表。
)
```

---

## 12. 接入外部 MCP 工具

将 MCP (Model Context Protocol) 工具集成到 Agent 中。

**设计特点：** 采用披露式设计，仅暴露两个元工具——`mcp_get_tool_description` 和 `mcp_call_tool`——而非将所有 MCP 工具直接注册。这样即便 MCP 服务端有上百个工具，也不会膨胀系统提示词上下文，Agent 按需查询工具描述、按需调用。

```python
from mambo_agents.middleware.mcp import MCPMiddleware, MCPServerConfig

# stdio 模式 — 启动本地 MCP 服务进程
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

# HTTP 模式 — 连接远程 MCP 服务
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

**`MCPServerConfig` 参数：**

| 参数 | 类型 | 必填 | 说明 |
|------|------|:---:|------|
| `name` | `str` | ✅ | MCP 服务唯一名称 |
| `transport` | `"stdio" \| "sse" \| "streamable_http" \| "websocket"` | ❌ | 传输方式，默认 `"stdio"` |
| `command` | `str` | stdio 必填 | 可执行命令（stdio 模式） |
| `args` | `list[str]` | ❌ | 命令行参数（stdio 模式） |
| `env` | `dict[str, str]` | ❌ | 环境变量（stdio 模式） |
| `cwd` | `str` | ❌ | 工作目录（stdio 模式） |
| `url` | `str` | HTTP 必填 | 服务地址（sse / streamable_http / websocket） |
| `headers` | `dict` | ❌ | HTTP 头（HTTP 模式） |
| `timeout` | `float` | ❌ | HTTP 超时（秒） |
| `sse_read_timeout` | `float` | ❌ | SSE 读超时（秒） |

### 12.x `exclude_tools` — 隐藏危险工具

阻止特定 MCP 工具暴露给 LLM：

```python
mcp = MCPMiddleware(
    servers=[...],
    exclude_tools={
        "filesystem": frozenset(["send_to_external", "install_package"]),
        "github": frozenset(["force_push"]),
    },
)
```

被排除的工具在注册前从 tool index 中移除 — 无法通过 ``mcp_get_tool_description``
发现，也无法通过 ``mcp_call_tool`` 调用。

### 12.x `direct_tool_threshold` — 直传 vs 包装模式

控制 MCP 工具是直接注册还是走包装 meta-tool。默认阈值为 **15**：所有 server
的 MCP 工具总数不超过此值时，每个工具注册为名为 ``server__tool`` 的一级工具。
高于阈值时使用 ``mcp_call_tool`` / ``mcp_get_tool_description`` 包装。

```python
mcp = MCPMiddleware(
    servers=[...],
    direct_tool_threshold=10,  # 默认 15；设为 0 强制全部包装
)
```

两种模式下 ``mcp_tool_name(server, tool)`` 和 ``tool_unpacker`` 行为一致 —
调整阈值时无需修改 ``interrupt_on`` 和 ``review_tools`` 配置。

### 12.x 安全审查集成

MCP 工具可以被安全审查中间件选择性审查。用 `mcp_tool_name(server, tool)` 构造
`interrupt_on` 和 `review_tools` 中的工具名，并通过 `tool_unpackers` 传入 `mcp.tool_unpacker`：

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

### 12.x `mcp_tool_name()` 参考

``mcp_tool_name(server_name, tool_name) → str`` 返回在 `interrupt_on` 和 `review_tools`
中使用的有效工具名：

```python
mcp_tool_name("filesystem", "delete_config")  # → "filesystem__delete_config"
```

- **一致性**：无论 MCP 处于包装模式还是透传模式，返回的字符串都相同。
- **安全性**：server name 在初始化时经过校验（禁止含 `__`、最长 64 字符、仅允许字母数字 + `_` `-`）。
- 完整示例见 `example/10_mcp_security_review.py` 和 `example/mcp_demo_server.py`。

---

## 13. 多 Agent 协作

### 13.1 同步子代理

子代理是**短暂的隔离 Agent**，通过 `task` 工具调用，完成后返回单一结果给主 Agent。

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
            tools=[],  # 只读分析，无需工具
        ),
    ],
)
```

**子代理规范（SubAgent Pydantic 模型）：**

| 字段 | 必需 | 说明 |
|------|:---:|------|
| `name` | ✅ | 唯一标识符 |
| `description` | ✅ | 用途描述（Agent 用它决定何时委托） |
| `system_prompt` | ✅ | 子代理的系统指令 |
| `model` | ✅ | 使用的 LLM 模型（通用子代理除外，必须显式提供）|
| `tools` | ❌ | 可用的工具列表（默认为空）|
| `middleware` | ❌ | 额外的中间件 |
| `interrupt_on` | ❌ | 子代理级别的人工审批 |

**预编译子代理：**

```python
from mambo_agents import create_mambo_agent, CompiledSubAgent

# 用 create_mambo_agent 构建一个子代理 graph
my_custom_graph = create_mambo_agent(
    "gpt-4o-mini",
    system_prompt="你是一个代码审查专家...",
    tools=[lint_tool],
)

agent = create_mambo_agent(
    "gpt-4o",
    subagents=[
        CompiledSubAgent(
            name="code-reviewer",
            description="Review code for bugs and style issues",
            runnable=my_custom_graph,  # 预编译的 agent graph
        ),
    ],
)
```

**事件粒度（EventGranularity）：**

控制子代理内部事件通过 `stream_mode="custom"` 流出时的粒度，由 `subagent_event_granularity` 参数设置：

| 值 | 说明 |
|------|------|
| `"messages"` | 最细粒 — LLM token 级别流式 |
| `"updates"` | 默认 — 按节点状态更新 |
| `"values"` | 最粗粒 — 每个图步骤的完整快照 |

```python
# 接收子代理流式事件
async for event in agent.astream(
    {"messages": [HumanMessage("研究 Python 异步模式")]},
    stream_mode=["updates", "custom"],  # "custom" 通道用于子代理事件
):
    if event[0] == "custom":
        data = event[1]
        # data["type"]: "subagent_event"
        # data["tool_call_id"]: 关联的 task 调用
        # data["subagent_type"]: 子代理名称
        # data["granularity"]: 事件粒度
        # data["chunk"]: 子代理的流式数据
```

### 13.2 通用子代理

设置 `include_general_purpose=True` 后，框架自动创建一个名为 `general-purpose` 的子代理。
它与主 Agent 使用**同一个**模型和后端工具，适合用来隔离执行复杂的多步骤子任务：

```python
agent = create_mambo_agent(
    "gpt-4o",
    include_general_purpose=True,
)
```

创建后，Agent 的系统提示词会包含 `task` 工具的调用指南，指导它**自动判断**何时将复杂任务
委托给子代理执行（如并行研究、大范围搜索等）。

> **注意：** 如果你已经在 `subagents` 中手动定义了一个名为 `general-purpose` 的子代理，
> `include_general_purpose=True` 不会重复创建，以你手动定义的为准。

### 13.3 异步子代理

与同步子代理不同，异步子代理在后台线程中运行，`async_task()` 立即返回 `task_id`。

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
    async_subagent_timeout=1800,  # 30 分钟超时
)

result = agent.invoke(
    {"messages": [HumanMessage("部署 v2.3 到生产环境")]},
    config={"configurable": {"thread_id": "session-1"}},
)
# Agent: "已启动任务 a3f4b2c1，在后台运行中..."

# 稍后查询
result = agent.invoke(
    {"messages": [HumanMessage("检查 a3f4b2c1 的状态")]},
    config={"configurable": {"thread_id": "session-1"}},
)
# Agent 通过 async_status 读取进度和结果
```

**异步子代理额外能力：**

| 方法 | 说明 |
|------|------|
| `async_task()` | 启动后端子代理，立即返回 `task_id` |
| `async_status(task_id)` | 查询状态：`running`（带进度）、`success`、`error`、`cancelled`、`crashed` |
| `async_list(status_filter)` | 列出所有任务（LLM 忘记 task_id 时使用） |
| `async_cancel(task_id)` | 取消运行中的任务 |
| `report_progress(message, percentage)` | 子代理内部报告进度 |

**崩溃恢复：** 如果使用了持久化 checkpointer（如 `SqliteSaver`），系统重启后调用
`async_status()` 或 `async_list()` 时，之前状态为 `running` 但线程已丢失的任务会被
自动标记为 `crashed`，Agent 可据此决定是否重新启动任务。

> 使用默认的 `InMemorySaver` 时状态不持久化，重启后所有任务记录丢失，无法恢复。

---

## 14. 高级用法

### 14.1 自定义系统提示词

```python
agent = create_mambo_agent(
    "gpt-4o",
    system_prompt="""你是一个 Python 专家助手。

## 编码规范
- 始终使用类型提示
- 遵循 PEP 8 风格
- 添加文档字符串
""",
)
```

### 14.2 添加自定义工具

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """获取城市天气"""
    return f"{city}: 晴朗，25°C"

agent = create_mambo_agent(
    "gpt-4o",
    tools=[get_weather],
)
```

### 14.3 预填充文件

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

### 14.4 自定义摘要配置

```python
agent = create_mambo_agent(
    "gpt-4o",
    summarization={
        "trigger": ("tokens", 50000),        # 较低阈值，更频繁压缩
        "keep": ("messages", 10),             # 较少保留消息
        "model": "gpt-4o-mini",               # 用廉价模型做摘要
        "trim_tokens_to_summarize": 2000,     # 摘要时回顾 2000 tokens
        "offload_to_backend": True,           # 被压缩的消息持久化
        "summary_prompt": "请简洁总结以下对话中的关键信息...",
    },
)
```

### 14.5 检查点持久化

```python
from langgraph.checkpoint.sqlite import SqliteSaver

agent = create_mambo_agent(
    "gpt-4o",
    checkpointer=SqliteSaver.from_conn_string("checkpoints.db"),
)
```

### 14.6 技能 + 子代理组合

```python
from mambo_agents.middleware.subagents import SubAgent
from mambo_agents.middleware.planning import MamboPlanMiddleware

agent = create_mambo_agent(
    "gpt-4o",
    skills=["/skills/team/"],
    subagents=[
        SubAgent(
            name="analyst",
            description="数据分析专家",
            system_prompt="你是数据分析专家...",
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

### 14.7 版本控制 + 记忆 + HITL 组合

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
