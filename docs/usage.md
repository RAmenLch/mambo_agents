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

### 两种使用模式

| 模式 | 函数 | 特点 |
|------|------|------|
| **精细模式** | `create_mambo_agent()` | 完全控制每个参数 |
| **开箱模式** | `create_powerful_agent()` | 智能默认值，一键启动全功能 Agent |

---

## 3. 快速上手

### 3.1 最简单用法

```python
from mambo_agents.quickstart import create_powerful_agent
from langchain_core.messages import HumanMessage

agent = create_powerful_agent("gpt-4o")

result = agent.invoke({
    "messages": [HumanMessage("创建一个 Python 脚本，打印 Hello World")]
})
```

### 3.2 操作本地文件系统

```python
agent = create_powerful_agent(
    "gpt-4o",
    workspace="/tmp/myproject"  # 指定本地工作目录
)

result = agent.invoke({
    "messages": [HumanMessage("列出当前目录的所有文件")]
})
```

### 3.3 带人工审批

```python
agent = create_powerful_agent(
    "gpt-4o",
    interrupt_on={
        "write": True,   # 写文件前需要审批
        "edit": True,    # 编辑文件前需要审批
        "delete": True,  # 删除文件前需要审批
    },
)
```

设置 `interrupt_on` 后，`create_powerful_agent()` 自动开启 **AI 安全预审** — 用主模型先审查工具调用，只有被 AI 标记为高风险的才会升级到人工审批。

### 3.4 流式输出

```python
agent = create_powerful_agent("gpt-4o")

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
| `skills` | `list` | `None` | 技能来源路径 |
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

### 4.2 `create_powerful_agent()` — 开箱模式

```python
def create_powerful_agent(
    model: str | BaseChatModel,
    *,
    workspace: str | None = None,
    system_prompt: str | None = None,
    name: str | None = None,
    skills: Sequence[SkillSource] | None = None,
    tools: Sequence[BaseTool] | None = None,
    subagents: Sequence[SubAgent | CompiledSubAgent] | None = None,
    async_subagents: Sequence[SubAgent | CompiledSubAgent] | None = None,
    interrupt_on: dict[str, bool | InterruptOnConfig] | None = None,
    summarization: SummarizationConfig | bool | None = True,
    enable_planning: bool = True,
    enable_general_purpose: bool = True,
    store: BaseStore | None = None,
    **kwargs,
) -> CompiledStateGraph
```

**默认开启的特性：**

| 特性 | 默认配置 |
|------|----------|
| 对话摘要 | 200k tokens 触发，保留最后 20 条消息，后端持久化 |
| 任务规划 | `write_plans` 工具可用 |
| 通用子代理 | 共享主 Agent 工具的后端子代理 |
| AI 安全审查 | 设置 `interrupt_on` 时自动启用（复用主模型） |

**关闭所有可选特性：**

```python
agent = create_powerful_agent(
    "gpt-4o",
    summarization=False,
    enable_planning=False,
    enable_general_purpose=False,
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

### 5.5 TempWorkspaceBackend — 双后端路由

将 `/.mambo/` 路径路由到 `StateBackend`（虚拟机内），其余路径路由到真实后端。

```python
from mambo_agents.backends.temp_workspace import TempWorkspaceBackend
from mambo_agents.backends.local import LocalBackend

backend = TempWorkspaceBackend(
    backend=LocalBackend(root_dir="/tmp/project"),
)
```

**用途：**
- 中间件内部存储（大结果驱逐、对话历史转储）
- Agent 草稿文件
- 子代理通信文件

### 5.6 后端对比

| 特性 | StateBackend | LocalBackend | SshBackend | TempWorkspaceBackend |
|------|:---:|:---:|:---:|:---:|
| 存储位置 | 内存 | 本地磁盘 | 远程服务器 | 混合 |
| 检查点支持 | 自动 | 手动 | 手动 | 自动(/.mambo/) |
| Shell 执行 | ❌ | 可选 | ✅ | ❌ |
| 删除操作 | ❌ | ✅ | ✅ | ❌ |
| grep 加速 | N/A | ripgrep | 远程 rg/grep | 继承委托 |
| 网络依赖 | ❌ | ❌ | ✅ | 可选 |
| 适用场景 | 测试/原型 | 本地开发 | 远程部署 | 生产环境 |

---

## 6. 中间件详解

### 6.1 中间件栈顺序

`create_mambo_agent()` 按固定顺序组装中间件：

```
1. BackendToolsMiddleware     ← 注册文件系统工具（始终启用）
2. [SkillsMiddleware]         ← 技能加载（skills 非 None 时）
3. [MamboSummarizationMiddleware]  ← 对话摘要（summarization 非 None 时）
4. [用户自定义 middleware]      ← 通过 middleware 参数传入
5. [SubAgentMiddleware]        ← 同步子代理
6. [AsyncSubAgentMiddleware]   ← 异步子代理
7. [AutoSecurityReviewMiddleware | HumanInTheLoopMiddleware]  ← 安全审查
8. PatchToolCallsMiddleware    ← 修复悬空工具调用（始终启用）
9. ReorderToolMessagesMiddleware ← 重排工具消息（始终启用）
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

# 或通过 create_powerful_agent（默认开启）
agent = create_powerful_agent("gpt-4o", enable_planning=True)
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

### 6.6 AutoSecurityReviewMiddleware

**功能：** 在实际执行工具（或暂停人工审批）之前，用廉价模型审查工具调用的安全性。

```python
# 经典 HITL（无 AI 预审）
agent = create_mambo_agent(
    "gpt-4o",
    interrupt_on={"write": True, "edit": True},
)

# AI 预审 — 所有 interrupt_on 工具
from mambo_agents.middleware.security_review import SecurityReviewConfig

agent = create_mambo_agent(
    "gpt-4o",
    interrupt_on={"write": True, "edit": True, "delete": True},
    security_review=SecurityReviewConfig(),
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

### 6.7 PatchToolCallsMiddleware & ReorderToolMessagesMiddleware

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
        {
            "name": "researcher",
            "description": "Research topics thoroughly and return structured findings",
            "system_prompt": "You are a research specialist...",
            "model": "gpt-4o",
            "tools": [search_tool, web_fetch_tool],
        },
        {
            "name": "code-reviewer",
            "description": "Review code for bugs, security issues, and style",
            "system_prompt": "You are a senior code reviewer...",
            "model": "gpt-4o",
            "tools": [],  # 只读分析，无需工具
        },
    ],
)
```

**子代理规范（SubAgent TypedDict）：**

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
compiled: CompiledSubAgent = {
    "name": "custom-processor",
    "description": "Custom processing pipeline",
    "runnable": my_custom_graph,
}
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
agent = create_mambo_agent(
    "gpt-4o",
    async_subagents=[
        {
            "name": "deployer",
            "description": "Deploy services to Kubernetes",
            "system_prompt": "You are a deployment expert...",
            "model": "gpt-4o",
            "tools": [kubectl_tool, helm_tool],
        },
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
agent = create_mambo_agent(
    "gpt-4o",
    skills=["/skills/team/"],
    subagents=[
        {
            "name": "analyst",
            "description": "数据分析专家",
            "system_prompt": "你是数据分析专家...",
            "model": "gpt-4o",
            "tools": [pandas_tool],
        },
    ],
    summarization={
        "trigger": ("tokens", 200000),
        "keep": ("messages", 20),
    },
    middleware=[MamboPlanMiddleware()],
)
```

### 8.7 使用 CLI 交互式测试

```bash
# 基本使用（内存后端）
python -m mambo_agents.cli.chat

# 指定模型和工作目录
python -m mambo_agents.cli.chat --model gpt-4o --workspace /tmp/test

# 带通用子代理
python -m mambo_agents.cli.chat --general-purpose

# 带自定义子代理
python -m mambo_agents.cli.chat --subagent researcher:研究专家:你是一个研究专家...
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
    TempWorkspaceBackend,
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
)
```

### 9.4 摘要配置

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
