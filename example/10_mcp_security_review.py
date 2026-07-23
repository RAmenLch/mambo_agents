"""
============================================================
Mambo Agents - 示例 10：MCP 工具 + 安全审查（可运行）
============================================================

需要先启动 MCP demo server（本目录下的 mcp_demo_server.py），然后运行本脚本。

本示例演示：
  1. MCPMiddleware 连接一个真实 MCP server
  2. security_review 解包 mcp_call_tool，针对内层工具做审查
  3. delete_data → AI 审查不通过 → 人工审批 → 批准 → 执行
  4. echo / add → AI 审查通过 → 自动放行（通过 custom stream 事件可见）

运行方式：
  python example/10_mcp_security_review.py

依赖：mcp（由 langchain-mcp-adapters 自动安装）
============================================================
"""

import asyncio
import json
import os
import sys
import textwrap

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from mambo_agents import create_mambo_agent
from mambo_agents.middleware.mcp import (
    MCPMiddleware,
    MCPServerConfig,
    mcp_tool_name,
)
from mambo_agents.middleware.security_review import SecurityReviewConfig
from deepseek_chat_model import ChatDeepSeek

load_dotenv()

# ---------------------------------------------------------------------------
# MCP server 路径
# ---------------------------------------------------------------------------

_DEMO_SERVER = os.path.join(os.path.dirname(__file__), "mcp_demo_server.py")


# ---------------------------------------------------------------------------
# 辅助：用 astream 跑完整流程，自动处理 interrupt
# ---------------------------------------------------------------------------

async def run_with_auto_approve(agent, prompt: str, thread_id: str):
    """异步跑一次 agent，遇到 interrupt 自动批准所有决策。"""
    final_messages = []
    interrupt_seen = False

    config = {"configurable": {"thread_id": thread_id}}

    async for mode, chunk in agent.astream(
        {"messages": [HumanMessage(content=prompt)]},
        config,
        stream_mode=["updates", "custom"],
    ):
        if mode == "custom":
            event_type = chunk.get("type", "")
            if event_type == "security_review_passed":
                print(f"  ✅ AI 审查通过: {chunk['tool_name']} (风险: {chunk['risk_level']})")
                print(f"     原因: {chunk['reason']}")
            elif event_type == "security_review_failed":
                print(f"  ⚠️  AI 审查不通过: {chunk['tool_name']} (风险: {chunk['risk_level']})")
                print(f"     原因: {chunk['reason']}")
            else:
                prefix = chunk.get("source", "") or chunk.get("type", "")
                print(f"  [custom] {prefix}: {json.dumps(chunk, ensure_ascii=False)[:200]}")
        elif mode == "updates":
            for node, update in chunk.items():
                if node == "__interrupt__":
                    interrupt_seen = True
                    raw = update[0] if isinstance(update, (list, tuple)) else update
                    hitl = raw.value if hasattr(raw, "value") else raw
                    print_interrupt(hitl)
                    decisions = build_approve_decisions(hitl)
                    print("  🔄 自动批准，继续执行...\n")
                    async for mode2, chunk2 in agent.astream(
                        Command(resume={"source": "mambo_security_review", "decisions": decisions}),
                        config,
                        stream_mode=["updates", "custom"],
                    ):
                        if mode2 == "updates":
                            for _node2, update2 in chunk2.items():
                                if isinstance(update2, dict) and "messages" in update2:
                                    final_messages.extend(update2["messages"])
                elif isinstance(update, dict) and "messages" in update:
                    final_messages.extend(update["messages"])

    return final_messages, interrupt_seen


def print_interrupt(hitl):
    """打印 interrupt 信息。"""
    requests = hitl.get("action_requests", [])
    print(f"\n  ⏸️  暂停 — 需要人工审批 ({len(requests)} 个操作):")
    for ar in requests:
        print(f"    - {ar['name']}({json.dumps(ar['args'], ensure_ascii=False)[:100]})")


def build_approve_decisions(hitl):
    """根据 action_requests 构建批准决策列表。"""
    decisions = []
    for ar in hitl.get("action_requests", []):
        decisions.append({
            "tool_call_id": ar["tool_call_id"],
            "type": "approve",
        })
    return decisions


# ---------------------------------------------------------------------------
# 示例 1：delete_data 触发 AI 审查 → 不通过 → 人工审批 → 自动批准
# ---------------------------------------------------------------------------

async def main_unsafe_tool_review():
    print("=" * 60)
    print("示例 1：delete_data → AI 审查不通过 → 人工审批 → 批准执行")
    print("=" * 60)

    model = ChatDeepSeek(model="deepseek-v4-flash")

    mcp = MCPMiddleware(
        servers=[
            MCPServerConfig(
                name="demo",
                transport="stdio",
                command=sys.executable,
                args=[_DEMO_SERVER],
            ),
        ],
    )

    agent = create_mambo_agent(
        model,
        middleware=[mcp],
        interrupt_on={
            "mcp_call_tool": True,
            mcp_tool_name("demo", "delete_data"): True,
            mcp_tool_name("demo", "echo"): True,
            mcp_tool_name("demo", "add"): True,
        },
        security_review=SecurityReviewConfig(
            model=model,
            review_tools=frozenset([
                mcp_tool_name("demo", "delete_data"),
                mcp_tool_name("demo", "echo"),
                mcp_tool_name("demo", "add"),
            ]),
            tool_unpackers=[mcp.tool_unpacker],
        ),
    )

    prompt = (
        "用 MCP demo server 的 delete_data 工具删除 users 表中 key=42 的记录。"
        "直接执行，不要反复确认。"
    )

    print(f"\n📝 用户: {prompt}\n")
    messages, interrupted = await run_with_auto_approve(agent, prompt, "session-1")

    last = messages[-1]
    content = last.content if hasattr(last, "content") else str(last)
    print(f"\n📤 Agent 最终回复:\n{textwrap.shorten(content, width=500)}\n")
    print(f"是否触发 interrupt: {'是' if interrupted else '否'}  ← 预期: 是\n")


# ---------------------------------------------------------------------------
# 示例 2：echo / add → AI 审查通过 → 自动放行
# ---------------------------------------------------------------------------

async def main_safe_tool_pass():
    print("=" * 60)
    print("示例 2：echo + add → AI 审查通过 → 自动放行，不触发 interrupt")
    print("=" * 60)

    model = ChatDeepSeek(model="deepseek-v4-flash")

    mcp = MCPMiddleware(
        servers=[
            MCPServerConfig(
                name="demo",
                transport="stdio",
                command=sys.executable,
                args=[_DEMO_SERVER],
            ),
        ],
    )

    agent = create_mambo_agent(
        model,
        middleware=[mcp],
        interrupt_on={
            "mcp_call_tool": True,
            mcp_tool_name("demo", "delete_data"): True,
            mcp_tool_name("demo", "echo"): True,
            mcp_tool_name("demo", "add"): True,
        },
        security_review=SecurityReviewConfig(
            model=model,
            review_tools=frozenset([
                mcp_tool_name("demo", "echo"),
                mcp_tool_name("demo", "add"),
                mcp_tool_name("demo", "delete_data"),
            ]),
            tool_unpackers=[mcp.tool_unpacker],
        ),
    )

    prompt = textwrap.dedent("""\
        请用 MCP demo server 做两件事：
        1. 用 echo 工具传消息 "hello from mambo"
        2. 用 add 工具计算 123 + 456
        不要调用 delete_data。
    """)

    print(f"\n📝 用户: {prompt}\n")
    messages, interrupted = await run_with_auto_approve(agent, prompt, "session-2")

    last = messages[-1]
    content = last.content if hasattr(last, "content") else str(last)
    print(f"\n📤 Agent 最终回复:\n{textwrap.shorten(content, width=500)}\n")
    print(f"是否触发 interrupt: {'是' if interrupted else '否'}  ← 预期: 否\n")


# ---------------------------------------------------------------------------
# 示例 3：混合 — 跳过 AI 审查，直接人工
# ---------------------------------------------------------------------------

async def main_skip_ai_review():
    print("=" * 60)
    print("示例 3：部分工具跳过 AI 审查，直接人工审批")
    print("=" * 60)

    model = ChatDeepSeek(model="deepseek-v4-flash")

    mcp = MCPMiddleware(
        servers=[
            MCPServerConfig(
                name="demo",
                transport="stdio",
                command=sys.executable,
                args=[_DEMO_SERVER],
            ),
        ],
    )

    agent = create_mambo_agent(
        model,
        middleware=[mcp],
        interrupt_on={
            mcp_tool_name("demo", "delete_data"): True,
        },
        security_review=SecurityReviewConfig(
            model=model,
            review_tools=frozenset([
                mcp_tool_name("demo", "echo"),
                mcp_tool_name("demo", "add"),
                # delete_data 不在此列 → 不进 AI，直接人工审批
            ]),
            tool_unpackers=[mcp.tool_unpacker],
        ),
    )

    prompt = "请用 MCP demo server 的 delete_data 工具删除 users 表中 key=7"

    print(f"\n📝 用户: {prompt}\n")
    messages, interrupted = await run_with_auto_approve(agent, prompt, "session-3")

    last = messages[-1]
    content = last.content if hasattr(last, "content") else str(last)
    print(f"\n📤 Agent 最终回复:\n{textwrap.shorten(content, width=500)}\n")
    print(f"是否触发 interrupt: {'是' if interrupted else '否'}  ← 预期: 是（但无 AI 预审）\n")


# ---------------------------------------------------------------------------
# 示例 4：exclude_tools — 剔除危险工具
# ---------------------------------------------------------------------------

async def main_exclude_tools():
    print("=" * 60)
    print("示例 4：exclude_tools — 从工具列表中剔除 delete_data")
    print("=" * 60)

    model = ChatDeepSeek(model="deepseek-v4-flash")

    mcp = MCPMiddleware(
        servers=[
            MCPServerConfig(
                name="demo",
                transport="stdio",
                command=sys.executable,
                args=[_DEMO_SERVER],
            ),
        ],
        exclude_tools={"demo": frozenset(["delete_data"])},
    )

    agent = create_mambo_agent(
        model,
        middleware=[mcp],
        interrupt_on={"mcp_call_tool": True},
        security_review=SecurityReviewConfig(
            model=model,
            review_tools=frozenset([
                mcp_tool_name("demo", "echo"),
                mcp_tool_name("demo", "add"),
            ]),
            tool_unpackers=[mcp.tool_unpacker],
        ),
    )

    prompt = (
        "用 MCP demo server 的 delete_data 工具删除 users 表中 key=1 的记录。"
        "如果工具不可用，用 echo 告诉我当前有哪些工具。"
    )

    print(f"\n📝 用户: {prompt}\n")
    messages, interrupted = await run_with_auto_approve(agent, prompt, "session-4")

    last = messages[-1]
    content = last.content if hasattr(last, "content") else str(last)
    print(f"\n📤 Agent 最终回复:\n{textwrap.shorten(content, width=500)}\n")
    print(f"是否触发 interrupt: {'是' if interrupted else '否'}  ← 预期: 否（delete_data 不可用）\n")


# ---------------------------------------------------------------------------
# 示例 5：direct_tool_threshold — 透传模式
# ---------------------------------------------------------------------------

async def main_direct_mode():
    print("=" * 60)
    print("示例 5：direct_tool_threshold — 强制透传模式（工具直接暴露给 LLM）")
    print("=" * 60)

    model = ChatDeepSeek(model="deepseek-v4-flash")

    mcp = MCPMiddleware(
        servers=[
            MCPServerConfig(
                name="demo",
                transport="stdio",
                command=sys.executable,
                args=[_DEMO_SERVER],
            ),
        ],
        direct_tool_threshold=100,  # 3 个工具 < 100 → 走透传
    )

    agent = create_mambo_agent(
        model,
        middleware=[mcp],
        interrupt_on={
            "demo__delete_data": True,
            "demo__echo": True,
        },
        security_review=SecurityReviewConfig(
            model=model,
            review_tools=frozenset([
                "demo__echo",
                "demo__add",
            ]),
            tool_unpackers=[mcp.tool_unpacker],
        ),
    )

    prompt = "用 demo__echo 工具发送消息 'hello direct mode'"

    print(f"\n📝 用户: {prompt}\n")
    print("  ℹ️  注意：透传模式下 LLM 直接看到工具名为 demo__echo 等\n")
    messages, interrupted = await run_with_auto_approve(agent, prompt, "session-5")

    last = messages[-1]
    content = last.content if hasattr(last, "content") else str(last)
    print(f"\n📤 Agent 最终回复:\n{textwrap.shorten(content, width=500)}\n")
    print(f"是否触发 interrupt: {'是' if interrupted else '否'}  ← 预期: 否（AI 审查通过）\n")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

async def main():
    print("\n" + "=" * 60)
    print("MCP + security_review 集成示例")
    print(f"MCP demo server: {_DEMO_SERVER}")
    print("=" * 60 + "\n")

    await main_unsafe_tool_review()
    await main_safe_tool_pass()
    await main_skip_ai_review()
    await main_exclude_tools()
    await main_direct_mode()


if __name__ == "__main__":
    asyncio.run(main())
