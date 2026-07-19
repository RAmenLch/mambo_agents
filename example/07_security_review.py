"""
============================================================
Mambo Agents - 示例 07：安全与人工审批（Security & HITL）
============================================================

演示安全审查和人工审批功能 - 在执行工具前用 AI 审查安全性。

运行前请先配置 .env 文件中的 API Key。
运行方式：
  python example/07_security_review.py
============================================================
"""

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from mambo_agents import create_mambo_agent
from mambo_agents.middleware.security_review import SecurityReviewConfig
from deepseek_chat_model import ChatDeepSeek

load_dotenv()


def main_classic_hitl():
    """经典人工审批模式 - 无 AI 预审"""
    print("=" * 60)
    print("示例 1：经典人工审批（无 AI 预审）")
    print("=" * 60)

    model = ChatDeepSeek(model="deepseek-v4-flash")

    # 对 write 和 edit 操作启用人工审批
    # 框架会自动配置 MemorySaver 作为默认 checkpointer，
    # 用于中断/恢复执行状态
    agent = create_mambo_agent(
        model,
        interrupt_on={"write": True, "edit": True},
    )

    result = agent.invoke({
        "messages": [HumanMessage(
            "创建一个 /safe_script.py，打印 'Hello Safe World'"
        )]
    }, config={"configurable": {"thread_id": "session-1"}})

    # 这里可能会在 write 时暂停，等待人工审批
    # 实际上在脚本中演示时，我们只展示配置方式
    print("Agent 已配置人工审批：write=True, edit=True")
    print("当 Agent 尝试写文件时，会暂停等待人工审批。")
    print(f"Agent 回复: {result['messages'][-1].content[:300]}...")
    print()


def main_ai_review():
    """AI 安全预审模式"""
    print("=" * 60)
    print("示例 2：AI 安全预审（llm 模式）")
    print("=" * 60)

    model = ChatDeepSeek(model="deepseek-v4-flash")

    # 启用 AI 安全预审 + 人工审批
    agent = create_mambo_agent(
        model,
        interrupt_on={"write": True, "edit": True, "delete": True},
        security_review=SecurityReviewConfig(
            model=model,              # 用同样的模型做审查（实际可用更便宜的模型）
            review_tools="all",       # 审查所有工具
        ),
    )

    result = agent.invoke({
        "messages": [HumanMessage(
            "创建 /review_test.py，内容为 print('test')"
        )]
    }, config={"configurable": {"thread_id": "session-1"}})

    print("Agent 已配置 AI 安全预审 + 人工审批")
    print("工具调用流程：工具调用 → AI 安全审查 → 安全：放行 / 高风险：暂停 → 人工审批")
    print(f"Agent 回复: {result['messages'][-1].content[:300]}...")
    print()


def main_custom_review():
    """自定义审查 - 只审查特定工具"""
    print("=" * 60)
    print("示例 3：自定义审查 - 只审查 write 工具")
    print("=" * 60)

    model = ChatDeepSeek(model="deepseek-v4-flash")

    agent = create_mambo_agent(
        model,
        interrupt_on={"write": True, "edit": True},
        security_review=SecurityReviewConfig(
            model=model,
            review_tools=frozenset(["write"]),  # 只审查 write
            system_prompt="你是一个安全审计专家，请审查工具调用的安全性。",
        ),
    )

    result = agent.invoke({
        "messages": [HumanMessage(
            "查看当前目录的文件列表，然后创建 /hello.py"
        )]
    }, config={"configurable": {"thread_id": "session-1"}})

    print("Agent 已配置：只审查 write 操作，edit 操作直接审批")
    print(f"Agent 回复: {result['messages'][-1].content[:300]}...")
    print()


if __name__ == "__main__":
    main_classic_hitl()
    main_ai_review()
    main_custom_review()
