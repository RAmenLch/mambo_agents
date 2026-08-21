

# Mambo Agents

**AI Agent framework built on LangChain/LangGraph** — multi-backend filesystem, sub-agent parallel scheduling, conversation summarization, security review, skills system, and more, out of the box.  
**基于 LangChain/LangGraph 的 AI Agent 框架** — 提供多后端文件系统、子代理并行调度、对话摘要、安全审查、技能系统等开箱即用的能力。  
> 📖 **English** | [中文](README.cn.md)

> This project draws architectural inspiration from [deepagents](https://github.com/langchain-ai/deepagents), with independent refactoring and extensions. See [comparison with deepagents](docs/deepagents_reference.md).

## Key Features

- **[Let Your Agent Control the Environment](docs/usage.md#5-let-your-agent-control-the-environment)** — multi-backend filesystem (virtual/local/SSH/hybrid), with shell command execution
- **[Long Conversation Management](docs/usage.md#6-long-conversation-management)** — automatic conversation history compaction with chained summaries
- **[Task Planning & Tracking](docs/usage.md#7-task-planning--tracking)** — structured TODO lists with automatic progress tracking
- **[Installing Skill Packs](docs/usage.md#8-installing-skill-packs)** — progressive disclosure, on-demand loading, multi-source overlay
- **[Let Your Agent Remember Your Preferences](docs/usage.md#9-let-your-agent-remember-your-preferences)** — AGENTS.md persistent context, AI self-learning write-back
- **[Security & Human Approval](docs/usage.md#10-security--human-approval)** — cheap model pre-review + human approval for high-risk operations
- **[File Change History & Rollback](docs/usage.md#11-file-change-history--rollback)** — checkpoint-level incremental snapshots, manual rollback
- **[Integrating External MCP Tools](docs/usage.md#12-integrating-external-mcp-tools)** — disclosure-based design, on-demand lookup
- **[Multi-Agent Collaboration](docs/usage.md#13-multi-agent-collaboration)** — sync/async sub-agents, parallel scheduling, isolated context windows
- **[Goal-Driven Loop Control](docs/usage.md#14-goal-driven-loop-control)** — preset goals with completion conditions, or LLM-autonomous long-running tasks with auto-continuation
## Quick Start

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
result = agent.invoke({"messages": [HumanMessage("Create a hello.py file")]}, config={"configurable": {"thread_id": "session-1"}})
```

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    create_mambo_agent()                   │
│                                                          │
│  ┌─────────────┐  ┌──────────┐  ┌────────────────────┐  │
│  │   Backend   │  │  Model   │  │    Middleware Stack │  │
│  │  Protocol   │  │ (LLM)    │  │                    │  │
│  │  Store      │  │          │  │ BackendTools       │  │
│  │  Local      │  │          │  │ Skills             │  │
│  │  SSH        │  │          │  │ Memory             │  │
│  │  Hybrid     │  │          │  │ VersionControl     │  │
│  │  ReadOnly   │  │          │  │ Summarization      │  │
│  │             │  │          │  │ Planning           │  │
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

## Docs

- [Detailed Usage Guide](docs/usage.md) — API reference, configuration, advanced usage
- [Comparison with deepagents](docs/deepagents_reference.md) — architectural differences, feature mapping, design trade-offs

## Related Projects

- [MamboChat](https://github.com/RAmenLch/mambochat) — Full-featured Web UI built on Mambo Agents

## License

MIT License
