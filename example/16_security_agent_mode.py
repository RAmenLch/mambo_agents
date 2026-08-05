"""
============================================================
Mambo Agents - 示例 16：安全审查 Agent 模式 + 人工审批协议
============================================================

演示 SecurityReviewConfig 的 review_mode="agent"：
  审查 Agent 拥有只读工作区工具（ls/read/grep/glob + 白名单扩展工具），
  在执行前检查工作区内容再给出裁决。

同时演示 Command(resume=...) 人工审批恢复协议的全部 4 种决策类型：
  - approve  批准，按原参数执行
  - edit     修改参数后执行（需携带 edited_action）
  - reject   拒绝，不执行
  - respond  拒绝但给 Agent 反馈，让它调整后重试

运行前请先配置 .env 文件中的 API Key。
运行方式：
  python example/16_security_agent_mode.py
============================================================
"""

import os
import tempfile
import textwrap

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from mambo_agents import StoreBackend, create_mambo_agent
from mambo_agents.backends.local import LocalBackend
from mambo_agents.middleware.security_review import SecurityReviewConfig
from deepseek_chat_model import ChatDeepSeek

load_dotenv()


def run_with_decision(agent, prompt: str, thread_id: str, decision: str):
    """跑一次 agent，遇到 interrupt 时按指定的决策类型恢复。"""
    config = {"configurable": {"thread_id": thread_id}}
    interrupted = False

    for event in agent.stream(
        {"messages": [HumanMessage(content=prompt)]},
        config,
        stream_mode="updates",
    ):
        for node, update in event.items():
            if node == "__interrupt__":
                interrupted = True
                raw = update[0] if isinstance(update, (list, tuple)) else update
                hitl = raw.value if hasattr(raw, "value") else raw
                print_interrupt(hitl)
                decisions = build_decisions(hitl, decision)
                print(f"  🔄 恢复决策: {decision}\n")
                for resume_event in agent.stream(
                    Command(resume={
                        "source": "mambo_security_review",
                        "decisions": decisions,
                    }),
                    config,
                    stream_mode="updates",
                ):
                    for _node2, update2 in resume_event.items():
                        if isinstance(update2, dict) and "messages" in update2:
                            for msg in update2["messages"]:
                                if msg.type == "ai" and msg.content:
                                    print(f"  💬 {msg.content[:200]}")
                            print()

    return interrupted


def print_interrupt(hitl):
    """打印 interrupt 信息（工具名 + 参数）。"""
    requests = hitl.get("action_requests", [])
    print(f"\n  ⏸️  暂停 — 需要人工审批 ({len(requests)} 个操作):")
    for ar in requests:
        args = textwrap.shorten(str(ar["args"]), width=100)
        print(f"    - {ar['name']}({args})  [tool_call_id={ar['tool_call_id'][:8]}...]")


def build_decisions(hitl, decision: str) -> list[dict]:
    """根据决策类型构建 resume decisions。"""
    decisions = []
    for ar in hitl.get("action_requests", []):
        if decision == "approve":
            decisions.append({"tool_call_id": ar["tool_call_id"], "type": "approve"})
        elif decision == "reject":
            decisions.append({
                "tool_call_id": ar["tool_call_id"],
                "type": "reject",
                "message": "用户拒绝本次操作",
            })
        elif decision == "respond":
            decisions.append({
                "tool_call_id": ar["tool_call_id"],
                "type": "respond",
                "message": "文件路径不对，请写入 /tmp 目录下的文件",
            })
        elif decision == "edit":
            # 演示 edit：修改 write 的内容参数后再执行
            edited_args = dict(ar["args"])
            if "content" in edited_args:
                edited_args["content"] = "# 用户修改后的内容\nprint('edited by human')\n"
            decisions.append({
                "tool_call_id": ar["tool_call_id"],
                "type": "edit",
                "edited_action": {"name": ar["name"], "args": edited_args},
            })
    return decisions


def demo_agent_mode_approve():
    """review_mode='agent' + 批准执行。"""
    print("=" * 60)
    print("示例 1：agent 审查模式 - 审查 Agent 预检后人工批准")
    print("=" * 60)

    model = ChatDeepSeek(model="deepseek-v4-flash")

    agent = create_mambo_agent(
        model,
        interrupt_on={"write": True, "edit": True},
        security_review=SecurityReviewConfig(
            model=model,
            review_mode="agent",        # 审查 Agent 带只读工具
            agent_max_steps=5,
        ),
    )

    interrupted = run_with_decision(
        agent,
        "创建一个 /app.py，内容为 print('hello agent review')",
        "session-1",
        decision="approve",
    )
    print(f"是否触发 interrupt: {'是' if interrupted else '否'}  ← 预期: 是（高风险需人工）")
    print()


def demo_agent_mode_reject():
    """拒绝执行。"""
    print("=" * 60)
    print("示例 2：人工拒绝（reject）- 工具不执行")
    print("=" * 60)

    model = ChatDeepSeek(model="deepseek-v4-flash")

    # 用 LocalBackend（提供 delete 工具），预置一个真实存在的文件，
    # 确保 Agent 会真正发起 delete 调用
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "config.py"), "w", encoding="utf-8") as f:
            f.write("PORT = 8080\n")
        backend = LocalBackend(root_dir=tmpdir)

        agent = create_mambo_agent(
            model,
            backend=backend,
            interrupt_on={"write": True, "edit": True, "delete": True},
            security_review=SecurityReviewConfig(
                model=model,
                review_mode="agent",
            ),
        )

        interrupted = run_with_decision(
            agent,
            "调用 delete 工具删除 /workspace/config.py 文件。"
            "直接调用 delete 工具执行，不要先做其他检查。",
            "session-2",
            decision="reject",
        )
        print(f"是否触发 interrupt: {'是' if interrupted else '否'}  ← 预期: 是")
    print()


def demo_agent_mode_respond():
    """拒绝但给反馈，Agent 调整后重试。"""
    print("=" * 60)
    print("示例 3：respond - 拒绝并反馈，Agent 调整重试")
    print("=" * 60)

    model = ChatDeepSeek(model="deepseek-v4-flash")

    agent = create_mambo_agent(
        model,
        interrupt_on={"write": True, "edit": True},
        security_review=SecurityReviewConfig(
            model=model,
            review_mode="agent",
        ),
    )

    interrupted = run_with_decision(
        agent,
        "调用 write 工具创建 /app.py 文件，内容为 print('respond demo')。"
        "直接调用 write 工具，不要先做其他检查。",
        "session-3",
        decision="respond",
    )
    print(f"是否触发 interrupt: {'是' if interrupted else '否'}  ← 预期: 是")
    print()


def demo_agent_mode_edit():
    """修改参数后执行。"""
    print("=" * 60)
    print("示例 4：edit - 人工修改参数后执行")
    print("=" * 60)

    model = ChatDeepSeek(model="deepseek-v4-flash")

    agent = create_mambo_agent(
        model,
        interrupt_on={"write": True, "edit": True},
        security_review=SecurityReviewConfig(
            model=model,
            review_mode="agent",
        ),
    )

    interrupted = run_with_decision(
        agent,
        "创建一个 /app.py，内容为 print('original')",
        "session-4",
        decision="edit",
    )
    print(f"是否触发 interrupt: {'是' if interrupted else '否'}  ← 预期: 是")
    print()


if __name__ == "__main__":
    demo_agent_mode_approve()
    demo_agent_mode_reject()
    demo_agent_mode_respond()
    demo_agent_mode_edit()
