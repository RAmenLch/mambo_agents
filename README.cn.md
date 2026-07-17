

# Mambo Agents

**基于 LangChain/LangGraph 的 AI Agent 框架** — 提供多后端文件系统、子代理并行调度、对话摘要、安全审查、技能系统等开箱即用的能力。

> 📖 [English](README.md) | **中文**

> 本项目参考了 [deepagents](https://github.com/langchain-ai/deepagents) 的架构设计，在其基础上进行了独立重构与扩展。详见 [与 deepagents 的对比](docs/deepagents_reference.cn.md)。



## 核心特性

- **[让 Agent 操控环境](docs/usage.cn.md#5-让-agent-操控环境)** — 多后端文件系统（虚拟/本地/SSH/混合），支持 Shell 命令执行
- **[超长对话管理](docs/usage.cn.md#6-超长对话管理)** — 自动压缩对话历史，链式摘要防止信息丢失
- **[任务规划与追踪](docs/usage.cn.md#7-任务规划与追踪)** — 结构化 TODO 列表，自动追踪完成进度
- **[给 Agent 装技能包](docs/usage.cn.md#8-给-agent-装技能包)** — 渐进式披露，按需加载，多来源覆盖
- **[让 Agent 记住你的偏好](docs/usage.cn.md#9-让-agent-记住你的偏好)** — AGENTS.md 持久上下文，AI 自主学习回写
- **[安全与人工审批](docs/usage.cn.md#10-安全与人工审批)** — 廉价模型预审 + 高风险人工审批，灵活的中断恢复协议
- **[文件修改历史与回滚](docs/usage.cn.md#11-文件修改历史与回滚)** — checkpoint 级增量快照，手动回滚任意版本
- **[接入外部 MCP 工具](docs/usage.cn.md#12-接入外部-mcp-工具)** — 披露式设计，按需查询，不膨胀上下文
- **[多 Agent 协作](docs/usage.cn.md#13-多-agent-协作)** — 同步/异步子代理，并行调度，隔离上下文窗口
## 快速开始

```bash
pip install mambo-agents
```

```python
from mambo_agents import create_mambo_agent, StoreBackend
from langchain_core.messages import HumanMessage

agent = create_mambo_agent(
    "gpt-4o",
    backend=StoreBackend(),
    include_general_purpose=True,
)
result = agent.invoke({"messages": [HumanMessage("创建一个 hello.py 文件")]})
```

## 架构

```
┌──────────────────────────────────────────────────────────┐
│                    create_mambo_agent()                   │
│                                                          │
│  ┌─────────────┐  ┌──────────┐  ┌────────────────────┐  │
│  │   Backend   │  │  Model   │  │    Middleware Stack │  │
│  │  Protocol   │  │ (LLM)    │  │                    │  │
│  │             │  │          │  │ BackendTools       │  │
│  │ State       │  │          │  │ Skills             │  │
│  │ Local       │  │          │  │ Memory             │  │
│  │ SSH         │  │          │  │ Summarization      │  │
│  │ TempWs      │  │          │  │ Planning           │  │
│  └─────────────┘  └──────────┘  │ SubAgents          │  │
│                                 │ AsyncSubAgents     │  │
│                                 │ SecurityReview     │  │
│                                 │ Patch + Reorder    │  │
│                                 └────────────────────┘  │
│                                                          │
│  ┌─────────────────────────────────────────────────────┐ │
│  │              LangGraph CompiledGraph                │ │
│  │  invoke() · astream() · astream_events()            │ │
│  └─────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

## 文档

| 中文 | English |
|------|---------|
| [详细使用文档](docs/usage.cn.md) | [Usage Guide](docs/usage.md) |
| [与 deepagents 的对比](docs/deepagents_reference.cn.md) | [Comparison with deepagents](docs/deepagents_reference.md) |

## 相关项目

- [MamboChat](https://github.com/RAmenLch/mambochat) — 基于 Mambo Agents 的完善功能 WebUI 项目

## 许可证

MIT License
