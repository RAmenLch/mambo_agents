# Mambo Agents

**AI Agent framework built on LangChain/LangGraph** — multi-backend filesystem, sub-agent parallel scheduling, conversation summarization, security review, skills system, and more, out of the box.

> This project draws architectural inspiration from [deepagents](https://github.com/langchain-ai/deepagents), with independent refactoring and extensions. See [comparison with deepagents](docs/deepagents_reference.en.md).

## Key Features

- **Multi-backend Filesystem** — `StateBackend` (in-memory), `LocalBackend` (local disk), `SshBackend` (remote SSH), `HybridWorkspaceBackend` (hybrid routing), unified through `BackendProtocol`
- **Sub-agent System** — sync/async sub-agents with parallel scheduling, streaming events, and isolated context windows
- **Conversation Summarization** — automatic long-history compaction with chained summaries and optional backend persistence
- **Task Planning** — `MamboPlanMiddleware` provides structured TODO lists, deeply integrated with the summarization system
- **AI Security Review** — pre-approve tool calls with a cheap model, escalating only high-risk operations to human review
- **Skills System** — progressive disclosure of skills, with multi-source overlay support
- **Plug-and-Play** — `create_powerful_agent()` launches a full-featured agent in one line

## Quick Start

```bash
pip install mambo-agents
```

```python
from mambo_agents.quickstart import create_powerful_agent
from langchain_core.messages import HumanMessage

# One line to create a full-featured agent (summarization, planning, general-purpose sub-agent enabled)
agent = create_powerful_agent("gpt-4o")
result = agent.invoke({"messages": [HumanMessage("Create a hello.py file")]})
```

## Architecture

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

## Docs

- [Detailed Usage Guide](docs/usage.en.md) — API reference, configuration, advanced usage
- [Comparison with deepagents](docs/deepagents_reference.en.md) — architectural differences, feature mapping, design trade-offs

## License

MIT License
