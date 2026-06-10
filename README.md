# Mambo Agents

**基于 LangChain/LangGraph 的 AI Agent 框架** — 提供多后端文件系统、子代理并行调度、对话摘要、安全审查、技能系统等开箱即用的能力。

> 本项目参考了 [deepagents](https://github.com/langchain-ai/deepagents) 的架构设计，在其基础上进行了独立重构与扩展。详见 [与 deepagents 的对比](docs/deepagents_reference.md)。

## 核心特性

- **多后端文件系统** — `StateBackend`(内存)、`LocalBackend`(本地磁盘)、`SshBackend`(远程SSH)、`HybridWorkspaceBackend`(混合路由)，统一通过 `BackendProtocol` 抽象
- **子代理系统** — 同步/异步子代理，支持并行调度、流式事件、隔离上下文窗口
- **对话摘要** — 自动压缩长对话历史，防止上下文窗口溢出，支持链式摘要与后端持久化
- **任务规划** — `MamboPlanMiddleware` 提供结构化 TODO 列表，与摘要系统深度集成
- **AI 安全审查** — 工具调用前用廉价模型预审，高风险操作才升级到人工审核
- **技能系统** — 渐进式披露的技能加载，支持多层来源覆盖
- **开箱即用** — `create_powerful_agent()` 一行代码启动全功能 Agent

## 快速开始

```bash
pip install mambo-agents
```

```python
from mambo_agents.quickstart import create_powerful_agent
from langchain_core.messages import HumanMessage

# 一行创建全功能Agent（自动开启摘要、规划、通用子代理）
agent = create_powerful_agent("gpt-4o")
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
│  │ Local       │  │          │  │ Summarization      │  │
│  │ SSH         │  │          │  │ Planning           │  │
│  │ TempWs      │  │          │  │ SubAgents          │  │
│  └─────────────┘  └──────────┘  │ AsyncSubAgents     │  │
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
| [详细使用文档](docs/usage.md) | [Usage Guide](docs/usage.en.md) |
| [与 deepagents 的对比](docs/deepagents_reference.md) | [Comparison with deepagents](docs/deepagents_reference.en.md) |

[English README](README.en.md)

## 许可证

MIT License
