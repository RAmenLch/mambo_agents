> 📖 [English](usage.md) | **中文**

# Mambo Agents 详细使用文档

## 目录

1. [安装](#1-安装)
2. [核心概念](#2-核心概念)
3. [快速上手](#3-快速上手)
4. [Agent 工厂函数](#4-agent-工厂函数)
5. [后端系统](#5-后端系统)
6. [中间件详解](#6-中间件详解)
7. [子代理系统](#7-子代理系统)
8. [高级用法](#8-高级用法)
9. [API 参考](#9-api-参考)

---

## 1. 安装

```bash
pip install mambo-agents
```

依赖：
- Python >= 3.11
- langchain + langgraph
- paramiko >= 3.0（SSH 后端需要）
- wcmatch >= 8.0

---

## 2. 核心概念

Mambo Agents 在 LangGraph 基础上构建，通过**工厂函数 + 中间件栈**的模式组装 Agent。

### 三层架构

| 层级 | 组件 | 职责 |
|------|------|------|
| **后端层** | `BackendProtocol` | 文件系统抽象，提供 6 个核心操作 + 扩展工具 |
| **中间件层** | `AgentMiddleware` | 横切关注点：工具注册、摘要、规划、安全审查等 |
| **Agent 层** | `create_mambo_agent()` | 组装后端+中间件，返回编译好的 LangGraph |

---

## 3. 快速上手

### 3.1 最简单用法

```python
from mambo_agents import create_mambo_agent, StateBackend
from langchain_core.messages import HumanMessage

agent = create_mambo_agent("gpt-4o")

result = agent.invoke({
    "messages": [HumanMessage("创建一个 Python 脚本，打印 Hello World")]
})
```

### 3.2 操作本地文件系统

```python
from mambo_agents.backends.local import LocalBackend

agent = create_mambo_agent(
    "gpt-4o",
    backend=LocalBackend(root_dir="/tmp/myproject"),
)

result = agent.invoke({
    "messages": [HumanMessage("列出当前目录的所有文件")]
})
```

### 3.3 带人工审批

```python
agent = create_mambo_agent(
    "gpt-4o",
    interrupt_on={
        "write": True,   # 写文件前需要审批
        "edit": True,    # 编辑文件前需要审批
        "delete": True,  # 删除文件前需要审批
    },
)
```

### 3.4 流式输出

```python
agent = create_mambo_agent("gpt-4o")

async for event in agent.astream(
    {"messages": [HumanMessage("分析项目的代码结构")]},
    stream_mode=["updates", "custom"],
):
    print(event)
```

当启用了子代理时，`stream_mode="custom"` 可以接收子代理内部的实时进度事件。

### 3.5 多轮对话

```python
config = {"configurable": {"thread_id": "session-1"}}

# 第一轮
result1 = agent.invoke(
    {"messages": [HumanMessage("创建一个 config.json")]},
    config=config,
)

# 第二轮 — Agent 记得之前创建的文件
result2 = agent.invoke(
    {"messages": [HumanMessage("给 config.json 添加一个 port 字段")]},
    config=config,
)
```

---

## 4. Agent 工厂函数

### 4.1 `create_mambo_agent()` — 精细模式

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

**必选参数：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `model` | `str \| BaseChatModel` | LLM 模型名称或实例 |

**常用可选参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `backend` | `BackendProtocol` | `StateBackend()` | 文件系统后端 |
| `system_prompt` | `str` | 内置默认 | 系统提示词 |
| `summarization` | `SummarizationConfig` | `None` | 对话摘要配置 |
| `subagents` | `list` | `None` | 同步子代理列表 |
| `include_general_purpose` | `bool` | `False` | 是否添加通用子代理 |
| `async_subagents` | `list` | `None` | 异步子代理列表 |
| `async_subagent_timeout` | `float` | `3600.0` | 异步子代理超时（秒）|
| `skills` | `list` | `None` | 技能来源路径 |
| `memory_sources` | `list` | `None` | 记忆文件路径（AGENTS.md 列表）|
| `tools` | `list` | `None` | 额外工具 |
| `interrupt_on` | `dict` | `None` | 工具审批配置 |
| `security_review` | `SecurityReviewConfig` | `None` | AI 安全预审配置 |

**示例：**

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

## 5. 后端系统

### 5.1 BackendProtocol — 抽象协议

所有后端必须实现 6 个核心操作：

| 方法 | 说明 |
|------|------|
| `ls(path)` | 列出目录内容（非递归） |
| `read(file_path, offset, limit, include_line_numbers)` | 读取文件内容 |
| `write(file_path, content, overwrite)` | 创建/覆盖文件 |
| `edit(file_path, old_str, new_str, replace_all)` | 替换文件中的文本 |
| `grep(pattern, path, glob)` | 搜索文本内容 |
| `glob(pattern, path)` | 按通配符查找文件 |

每个后端还可以通过 `tools` 属性暴露额外的工具（如 `tree`、`delete`、`execute`）。

### 5.2 StateBackend — 内存文件系统

文件存储在 LangGraph 的 `files` 状态通道中，自动参与检查点。

```python
from mambo_agents import StateBackend

backend = StateBackend(
    initial_files={
        "/config.json": '{"port": 8080}',
        "/README.md": "# My Project",
    }
)
```

**额外工具：** `tree`

**特点：**
- 自动参与 LangGraph 检查点（支持暂停/恢复/回滚）
- 多线程安全（`thread_id` 隔离）
- 轻量级，无需磁盘访问

### 5.3 LocalBackend — 本地文件系统

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

> **⚠️ 安全警告：** `LocalBackend` 提供直接文件系统访问和 Shell 执行能力。建议配合 `interrupt_on` + `security_review` 使用。

### 5.4 SshBackend — 远程 SSH 文件系统

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
)
```

**额外工具：** `tree`、`delete`、`execute`

**性能策略：** 批量操作（`grep`、`glob`、`edit`、`tree`）在远程执行，避免逐文件 SFTP 往返。`edit` 通过远程 `python3 -c` 一次性完成查找替换。

### 5.5 HybridWorkspaceBackend — 多后端路由

真实后端 + N 个虚拟 workspace，统一在 `/.mambo/` 下路由。
每个虚拟 workspace 由独立的 `StateBackend` 驱动，只支持核心 protocol 工具。

```python
from mambo_agents.backends.hybrid_workspace import HybridWorkspaceBackend
from mambo_agents.backends.local import LocalBackend
from mambo_agents import StateBackend

# 最简用法：自动创建 /.mambo/ 默认 StateBackend
backend = HybridWorkspaceBackend(
    real_backend=LocalBackend(root_dir="/tmp/project"),
)

# 多虚拟 workspace
backend = HybridWorkspaceBackend(
    real_backend=LocalBackend(root_dir="/tmp/project"),
    virtual_workspaces={
        "skills": StateBackend(initial_files={"/python.md": "..."}),
        "cache": StateBackend(),
    },
)

# 覆盖默认 /.mambo/
backend = HybridWorkspaceBackend(
    real_backend=LocalBackend(root_dir="/tmp/project"),
    virtual_workspaces={
        ".": StateBackend(initial_files={"/config.yml": "..."}),
    },
)
```

**路径路由规则：**
- `/.mambo/skills/xxx` → "skills" 虚拟 workspace（strip 前缀后传 `/xxx`）
- `/.mambo/xxx` → 默认 StateBackend（strip 前缀后传 `/xxx`）
- 其他路径 → 真实后端

**用途：**
- 中间件内部存储（大结果驱逐、对话历史转储）
- Agent 草稿文件
- 子代理通信文件
- 多技能/多模块的独立隔离空间

### 5.6 后端对比

| 特性 | StateBackend | LocalBackend | SshBackend | HybridWorkspaceBackend |
|------|:---:|:---:|:---:|:---:|
| 存储位置 | 内存 | 本地磁盘 | 远程服务器 | 混合 |
| 检查点支持 | 自动 | 手动 | 手动 | 自动(/.mambo/) |
| Shell 执行 | ❌ | 可选 | ✅ | ❌ |
| 删除操作 | ❌ | ✅ | ✅ | ❌ |
| grep 加速 | N/A | ripgrep | 远程 rg/grep | 继承委托 |
| 网络依赖 | ❌ | ❌ | ✅ | 可选 |
| 适用场景 | 测试/原型 | 本地开发 | 远程部署 | 生产环境 |

### 5.7 ReadSummarizer — 大文件读取摘要

`BackendProtocol` 内置 `max_read_chars`（默认 100,000 字符）上限控制。超限时，用摘要替换原文而非简单截断。
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

---

## 6. 中间件详解

### 6.1 中间件栈顺序

`create_mambo_agent()` 按固定顺序组装中间件：

```
1. BackendToolsMiddleware     ← 注册文件系统工具（始终启用）
2. [SkillsMiddleware]         ← 技能加载（skills 非 None 时）
3. [MamboMemoryMiddleware]    ← 记忆加载（memory_sources 非 None 时）
4. [MamboSummarizationMiddleware]  ← 对话摘要（summarization 非 None 时）
5. [用户自定义 middleware]      ← 通过 middleware 参数传入
   ├─ VersionControlMiddleware  ← 文件版本控制
   ├─ MamboPlanMiddleware       ← 任务规划
   └─ ...
6. [SubAgentMiddleware]        ← 同步子代理
7. [AsyncSubAgentMiddleware]   ← 异步子代理
8. [AutoSecurityReviewMiddleware | HumanInTheLoopMiddleware]  ← 安全审查
9. PatchToolCallsMiddleware    ← 修复悬空工具调用（始终启用）
10. ReorderToolMessagesMiddleware ← 重排工具消息（始终启用）
```

### 6.2 BackendToolsMiddleware

**功能：**
- 注册 6 个核心文件系统工具（`ls`、`read`、`write`、`edit`、`grep`、`glob`）
- 合并后端扩展工具
- 超大工具结果自动驱逐到 `/.mambo/large_tool_results/`

**大结果驱逐：** 当工具返回超过 20000 个 token 时，完整内容被写入文件系统，上下文中的消息被替换为预览 + 文件路径。

### 6.3 MamboSummarizationMiddleware

**功能：** 自动压缩长对话历史，防止超出 LLM 上下文窗口。

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

**链式摘要：** 多次触发压缩时，之前的摘要作为"不可协商的历史上下文"注入摘要提示词，保证信息不丢失。

**摘要钩子（SummaryHook）：** 允许其他中间件在摘要时注入额外上下文。`MamboPlanMiddleware` 通过此机制在摘要发生时注入当前计划状态。

### 6.4 MamboPlanMiddleware

**功能：** 提供 `write_plans` 工具，让 Agent 维护结构化的 TODO 列表。

```python
from mambo_agents.middleware.planning import MamboPlanMiddleware

# 通过 create_mambo_agent 的 middleware 参数
agent = create_mambo_agent(
    "gpt-4o",
    middleware=[MamboPlanMiddleware()],
)

# 或通过 create_mambo_agent 启用
agent = create_mambo_agent(
    "gpt-4o",
    middleware=[MamboPlanMiddleware()],
)
```

**计划数据模型：**

```python
from mambo_agents import Plan

# 每个计划项包含三个字段：
# - content: 任务描述
# - status: "pending" | "in_progress" | "completed"
```

### 6.5 SkillsMiddleware

**功能：** 渐进式披露的技能系统 — 技能只在 Agent 需要时才加载到提示词中。

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
| 裸路径 | `"/skills/user/"` | 路径最后组件大写 |
| 元组 | `("/path", "我的技能")` | 自定义标签 |

多来源加载：后加载的技能覆盖同名的先前技能（后胜出）。

### 6.6 MamboMemoryMiddleware

**功能：** 记忆系统 — 从 AGENTS.md 文件加载持久上下文，并指导 AI 在交互中**学习回写**新知识。与技能（按需加载）不同，记忆始终加载到系统提示词中，提供跨轮次的持久上下文。

```python
agent = create_mambo_agent(
    "gpt-4o",
    memory_sources=["/.mambo/memory/AGENTS.md"],
)
```

**工作流程：**

```
before_agent → backend.download_files(sources) → memory_contents
                                                    ↓
wrap_model_call → modify_request → 注入 <agent_memory> 到系统提示
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

**自定义格式化：**

```python
from mambo_agents.middleware.memory import MamboMemoryMiddleware, MemoryFormatHook

def my_formatter(contents: dict[str, str]) -> str:
    # 自定义记忆内容如何注入系统提示
    parts = []
    for path, text in contents.items():
        parts.append(f"## Source: {path}\n{text}")
    return "\n---\n".join(parts)

agent = create_mambo_agent(
    "gpt-4o",
    middleware=[
        MamboMemoryMiddleware(
            backend=StateBackend(),
            sources=["/.mambo/memory/AGENTS.md"],
            format_prompt=my_formatter,
        ),
    ],
)
```

### 6.7 AutoSecurityReviewMiddleware

**功能：** 在实际执行工具（或暂停人工审批）之前，用廉价模型审查工具调用的安全性。

两种审查模式：

- **llm**（默认）：每次工具调用单次结构化输出 LLM 调用——快速且低成本。
- **agent**：专用审查 agent，带有只读后端工具，可在提交审查结论前检查工作区。Backend 工具（核心 6 个 + ``backend.tools``）使用 agent 审查；非 backend 用户工具回退到 llm 审查。

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

### 6.8 VersionControlMiddleware

**功能：** 以 checkpoint 为粒度自动快照文件变更，支持选择性回滚。不向 LLM 暴露任何工具，版本数据仅供调用方（如 Web 应用）查询使用。

**设计原则：**
- 存储与后端解耦 — 纯本地文件 I/O，写入 `./.mambo_versions/`
- 即时持久化 — 每次 `wrap_tool_call` 备份同时写入 blob 和 index.json
- 增量快照 — 只备份 LLM 实际变更的文件
- 内容寻址 — SHA256 存储，相同内容只存一份

**配置方式：**

```python
from mambo_agents.middleware.version_control import (
    VersionStore,
    VersionControlMiddleware,
)

store = VersionStore(storage_dir="./.mambo_versions")

agent = create_mambo_agent(
    "gpt-4o",
    backend=LocalBackend(),
    middleware=[
        VersionControlMiddleware(
            store=store,
            backend=...,
            whitelist_folders=["/workspace/src", "/workspace/tests"],
            mutating_tool_names=["write", "edit", "delete", "patch"],
        ),
    ],
)
```

**参数说明：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `store` | `VersionStore` | (必填) | 版本数据存储引擎 |
| `backend` | `BackendProtocol` | (必填) | 文件系统后端（用于读写文件内容） |
| `whitelist_folders` | `list[str]` | `[]` | **白名单模式** — 仅对此列表内的文件夹进行备份和回滚。空列表 = 不处理任何文件 |
| `mutating_tool_names` | `list[str]` | `["write", "edit", "delete"]` | 声明哪些工具是"写入/变更"工具，调用前自动触发备份 |

**白名单模式：**

`whitelist_folders` 为虚拟文件系统的绝对路径（如 `/workspace/src`）。只有位于白名单文件夹内的文件才会被备份和允许回滚。默认空列表表示严格白名单模式 — 没有文件会被处理。

```python
# 只对 src/ 和 tests/ 目录做版本控制
VersionControlMiddleware(
    store=store,
    backend=backend,
    whitelist_folders=["/workspace/src", "/workspace/tests"],
)

# 空白名单 = 不处理任何文件
VersionControlMiddleware(store=store, backend=backend)
```

**自定义变更工具：**

不同后端可能有不同的变更工具名（如 `patch`、`rename`），通过 `mutating_tool_names` 声明：

```python
VersionControlMiddleware(
    store=store,
    backend=backend,
    whitelist_folders=["/workspace"],
    mutating_tool_names=["write", "edit", "delete", "patch", "rename"],
)
```

**时间旅行回滚：**

通过 config 中的 `version_rollback` 指定要回滚的文件：

```python
config = {
    "configurable": {
        "thread_id": "session-1",
        "checkpoint_id": "cp_target",       # 回滚目标 checkpoint
        "version_rollback": {
            "files": ["/workspace/src/main.py"],  # 指定文件列表
            # 或 "all": True 将所有变更文件恢复到该 checkpoint
        },
    }
}
agent.astream({"messages": [...]}, config)
```

注意：回滚同样受白名单限制 — 不在白名单内的文件不会被恢复。

**调用方查询 API（`VersionStore`）：**

Web 应用等调用方可通过 `VersionStore` 查询版本历史：

```python
store = VersionStore(storage_dir="./.mambo_versions")

# 整个对话会话中改过哪些文件（去重）
all_files = store.get_all_changed_files("thread-123")
# → frozenset({"/workspace/src/main.py", "/workspace/tests/test.py"})

# 最新一轮改了什么
latest_files = store.get_latest_changed_files("thread-123")
# → ["/workspace/src/main.py"]

# 获取最新一轮的完整快照（含 checkpoint_id、时间戳、文件→SHA 映射）
snapshot = store.get_latest_snapshot("thread-123")
print(snapshot.checkpoint_id, snapshot.timestamp, snapshot.file_blobs)

# 按 checkpoint 查询
store.list_snapshots("thread-123")             # 所有快照（时间序）
store.get_changed_files("thread-123", "cp_x")  # 某 checkpoint 改了哪些文件
store.get_file("thread-123", "cp_x", "/path")  # 某文件在某 checkpoint 的内容
```

**配置模型（`VersionControlConfig`）：**

```python
from mambo_agents.middleware.version_control import VersionControlConfig

VersionControlConfig(
    store_dir="./.mambo_versions",              # 版本存储目录
    auto_snapshot=True,                          # 自动触发备份
    whitelist_folders=["/workspace/src"],        # 白名单文件夹
    mutating_tool_names=["write", "edit", "delete"],  # 变更工具名
)
```

### 6.9 PatchToolCallsMiddleware & ReorderToolMessagesMiddleware

这两个中间件始终启用（无需配置）：

- **PatchToolCallsMiddleware：** 修复消息历史中的悬空工具调用（例如人工中断导致 `AIMessage.tool_calls` 缺少对应的 `ToolMessage`）
- **ReorderToolMessagesMiddleware：** 将 `ToolMessage` 重新排序以匹配 `AIMessage.tool_calls` 的顺序，防止多模态模型因顺序错乱而产生误解

---

## 7. 子代理系统

### 7.1 同步子代理

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
| `model` | ✅ | 使用的 LLM 模型 |
| `tools` | ✅ | 可用的工具列表 |
| `middleware` | ❌ | 额外的中间件 |
| `interrupt_on` | ❌ | 子代理级别的人工审批 |

**预编译子代理：**

```python
from mambo_agents.middleware.subagents import CompiledSubAgent

# 可以用任意 runnable（需有 'messages' 状态键）
compiled = CompiledSubAgent(
    name="custom-processor",
    description="Custom processing pipeline",
    runnable=my_custom_graph,
)
```

**事件粒度（EventGranularity）：**

| 值 | 说明 |
|------|------|
| `"messages"` | 最细粒 — LLM token 级别流式 |
| `"updates"` | 默认 — 按节点状态更新 |
| `"values"` | 最粗粒 — 每个图步骤的完整快照 |

```python
# 消费子代理流式事件
async for event in agent.astream(
    {"messages": [HumanMessage("研究 Python 异步模式")]},
    stream_mode=["updates", "custom"],
):
    if event[0] == "custom":
        custom_data = event[1]
        # custom_data 包含: tool_call_id, subagent_type, chunk, timestamp
```

### 7.2 通用子代理（General Purpose Subagent）

设置 `include_general_purpose=True` 后，自动创建一个与主 Agent 共享相同模型、后端工具和系统提示词的 `general-purpose` 子代理。

```python
agent = create_mambo_agent(
    "gpt-4o",
    include_general_purpose=True,
)

# Agent 会自动在处理复杂多步骤任务时使用这个子代理
```

### 7.3 异步子代理

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
    {"messages": [HumanMessage("部署 v2.3 到生产环境")]}
)
# Agent: "已启动任务 a3f4b2c1，在后台运行中..."

# 稍后查询
result = agent.invoke(
    {"messages": [HumanMessage("检查 a3f4b2c1 的状态")]}
)
# Agent 通过 async_status 读取进度和结果
```

**异步子代理额外能力：**

| 方法 | 说明 |
|------|------|
| `async_task()` | 启动后端子代理，立即返回 `task_id` |
| `async_status(task_id)` | 查询状态：`running`（带进度）、`success`、`error`、`cancelled`、`crashed` |
| `async_list(status_filter)` | 列出所有任务（LLM 忘记 task_id 时使用） |
| `report_progress(message, percentage)` | 子代理内部报告进度 |

**崩溃恢复：** 系统重启后，之前状态为 `running` 的任务会被检测并标记为 `crashed`。

---

## 8. 高级用法

### 8.1 自定义系统提示词

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

### 8.2 添加自定义工具

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

### 8.3 预填充文件

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

### 8.4 自定义摘要配置

```python
agent = create_mambo_agent(
    "gpt-4o",
    summarization={
        "trigger": ("tokens", 50000),        # 较低阈值，更频繁压缩
        "keep": ("messages", 10),             # 较少保留消息
        "model": "gpt-4o-mini",               # 用廉价模型做摘要
        "trim_tokens_to_summarize": 2000,     # 摘要时回顾 2000 tokens
        "offload_to_backend": True,           # 被压缩的消息持久化
        "summary_prompt": "请简洁总结以下对话中的关键信息...",  # 自定义摘要提示词
    },
)
```

### 8.5 检查点持久化

```python
from langgraph.checkpoint.sqlite import SqliteSaver

agent = create_mambo_agent(
    "gpt-4o",
    checkpointer=SqliteSaver.from_conn_string("checkpoints.db"),
)
```

### 8.6 技能 + 子代理组合

```python
from mambo_agents.middleware.subagents import SubAgent

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

---

## 9. API 参考

### 9.1 公共 API 导出

```python
from mambo_agents import (
    # 工厂函数
    create_mambo_agent,

    # 后端
    BackendProtocol,
    StateBackend,
    HybridWorkspaceBackend,
    FileData,
    FilesystemState,

    # 同步子代理
    SubAgent,
    CompiledSubAgent,
    SubAgentMiddleware,
    EventGranularity,

    # 异步子代理
    AsyncSubAgentMiddleware,
    AsyncTaskData,

    # 任务规划
    MamboPlanMiddleware,
    Plan,
    WritePlansInput,

    # 对话摘要
    MamboSummarizationMiddleware,
    SummarizationConfig,
    SummaryHook,
    SummaryHookContext,

    # 技能
    SkillsMiddleware,
    SkillMetadata,
    SkillSource,

    # 记忆
    MamboMemoryMiddleware,
    MemoryFormatHook,

    # 版本控制
    VersionControlMiddleware,
    VersionStore,
    VersionControlConfig,
    VersionRollbackConfig,
)
```

### 9.2 后端协议结果类型

| 类型 | 用途 |
|------|------|
| `LsResult` | 目录列表结果 |
| `ReadResult` | 文件读取结果（支持文本/多模态） |
| `WriteResult` | 文件写入结果 |
| `EditResult` | 文件编辑结果 |
| `GrepResult` | 文本搜索匹配 |
| `GlobResult` | 文件通配符匹配 |
| `FileInfo` | 单个文件/目录信息 |
| `GrepMatch` | 单个 grep 匹配 |

### 9.3 安全审查配置

```python
from mambo_agents.middleware.security_review import SecurityReviewConfig

SecurityReviewConfig(
    model: str | BaseChatModel | None = None,
    # None = 复用 Agent 模型
    # "gpt-4o-mini" = 使用廉价模型审查

    review_tools: Literal["all"] | frozenset[str] = "all",
    # "all" = 审查所有 interrupt_on 工具
    # frozenset({"write", "edit"}) = 仅审查指定工具

    system_prompt: str | None = None,
    # None = 使用内置安全审查提示词
    # 自定义 = 覆盖审查提示词

    review_mode: Literal["llm", "agent"] = "llm",
    # "llm" = 每次工具调用单次 LLM 调用（快速，默认）
    # "agent" = 专用审查 agent，带只读后端工具
    #           Backend 工具 → agent 审查；用户工具 → llm 审查

    agent_max_steps: int = 5,
    # 审查 agent 的最大步数（仅在 agent 模式下使用）

    agent_tools: frozenset[str] | None = None,
    # agent 模式下暴露给审查 agent 的后端工具名列表
    # None = 所有已注册后端工具均可用
)
```

### 9.4 HITL 中断 / 恢复协议

当 ``AutoSecurityReviewMiddleware`` 将工具调用上报人工审批时，会通过
LangGraph 的 ``interrupt()`` 发出如下结构的报文：

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

你的人工审批基础设施在通过 ``Command(resume=...)`` 恢复图执行时，**必须**在
恢复报文中回传 ``"source"`` 字段：

```json
{
    "source": "mambo_security_review",
    "decisions": [
        {"tool_call_id": "call_abc123", "decision": "approve"}
    ]
}
```

``source`` 字段有两个用途：

1. **消费者路由：** UI 层可以据此将"安全审查中断"与其他类型的中断（如工具内部自
   定义的 ``interrupt()`` 调用）区分开。
2. **重放检测：** 恢复时中间件会以非消费方式读取恢复值，判断是否进入重放
   分支。只有携带 ``"source": "mambo_security_review"`` 的值才会被识别为
   本中间件的中断回复。

> **重要：** 省略 ``"source"`` 会导致中间件将恢复视为"非我方"并透明放行。
> 人类决策不会被应用，工具调用将以原始参数执行。

### 9.5 摘要配置

```python
SummarizationConfig = {
    "trigger": ("tokens", 200000) | ("messages", 50) | None,
    "keep": ("messages", 20) | ("tokens", 5000),
    "model": str | BaseChatModel | None,
    "trim_tokens_to_summarize": int,       # 默认 4000
    "token_counter": Callable | None,
    "chars_per_token": float | None,
    "offload_to_backend": bool,            # 默认 False
    "backend": BackendProtocol | None,
    "summary_prompt": str | None,
    "chained_summary_prompt": str | None,
    "summary_hooks": list[SummaryHook] | None,
}
```
