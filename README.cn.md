

# Mambo Agents

**基于 LangChain/LangGraph 的 AI Agent 框架** — 提供多后端文件系统、子代理并行调度、对话摘要、安全审查、技能系统等开箱即用的能力。

> 📖 [English](README.md) | **中文**

> 本项目参考了 [deepagents](https://github.com/langchain-ai/deepagents) 的架构设计，在其基础上进行了独立重构与扩展。详见 [与 deepagents 的对比](docs/deepagents_reference.cn.md)。



## 核心特性

- **多后端文件系统** — `StateBackend`(内存)、`LocalBackend`(本地磁盘)、`SshBackend`(远程SSH)、`HybridWorkspaceBackend`(混合路由)，统一通过 `BackendProtocol` 抽象
- **子代理系统** — 同步/异步子代理，支持并行调度、流式事件、隔离上下文窗口
- **对话摘要** — 自动压缩长对话历史，防止上下文窗口溢出，支持链式摘要与后端持久化
- **任务规划** — `MamboPlanMiddleware` 提供结构化 TODO 列表，与摘要系统深度集成
- **AI 安全审查** — 工具调用前用廉价模型预审，高风险操作才升级到人工审核
- **技能系统** — 渐进式披露的技能加载，支持多层来源覆盖
- **记忆系统** — 从 `AGENTS.md` 加载持久上下文，支持 AI 自主学习和回写
## 快速开始

```bash
pip install mambo-agents
```

```python
from mambo_agents import create_mambo_agent, StateBackend
from langchain_core.messages import HumanMessage

agent = create_mambo_agent(
    "gpt-4o",
    backend=StateBackend(),
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
