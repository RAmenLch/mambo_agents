"""
============================================================
Mambo Agents - 示例 08：子代理协作（Sub-Agents）
============================================================

演示多 Agent 协作 - 同步子代理（SubAgent）。

运行前请先配置 .env 文件中的 API Key。
运行方式：
  python example/08_subagents.py
============================================================
"""

import asyncio

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from mambo_agents import CompiledSubAgent, create_mambo_agent
from mambo_agents.middleware.subagents import SubAgent
from deepseek_chat_model import ChatDeepSeek

load_dotenv()


def main_sync_subagents():
    """同步子代理示例"""
    print("=" * 60)
    print("示例 1：同步子代理 - 多个专业子代理协作")
    print("=" * 60)

    model = ChatDeepSeek(model="deepseek-v4-flash")

    agent = create_mambo_agent(
        model,
        subagents=[
            SubAgent(
                name="python-expert",
                description="Python 编程专家，擅长编写高质量 Python 代码",
                system_prompt=(
                    "你是一个 Python 编程专家。"
                    "编写代码时始终包含类型提示、文档字符串，并遵循 PEP 8 风格。"
                ),
                model=model,
            ),
            SubAgent(
                name="code-reviewer",
                description="代码审查专家，检查代码的 bug、安全问题、代码风格",
                system_prompt=(
                    "你是一个高级代码审查专家。"
                    "检查代码中的潜在 bug、安全漏洞和代码风格问题。"
                ),
                model=model,
            ),
        ],
    )

    result = agent.invoke({
        "messages": [HumanMessage(
            "我需要一个 Python 函数，用来计算斐波那契数列。"
            "请让 python-expert 来编写代码，"
            "然后让 code-reviewer 来审查它。"
        )]
    }, config={"configurable": {"thread_id": "session-1"}})

    print(result["messages"][-1].content[:800], "...")
    print()


def main_general_purpose():
    """通用子代理示例"""
    print("=" * 60)
    print("示例 2：通用子代理 - 自动创建 general-purpose")
    print("=" * 60)

    model = ChatDeepSeek(model="deepseek-v4-flash")

    agent = create_mambo_agent(
        model,
        include_general_purpose=True,
    )

    result = agent.invoke({
        "messages": [HumanMessage(
            "帮我做两个任务：\n"
            "1. 研究一下 Python 的 dataclass 用法\n"
            "2. 研究一下 Python 的 enum 用法\n"
            "你可以用 task 工具来并行执行这两个任务。"
        )]
    }, config={"configurable": {"thread_id": "session-1"}})

    print(result["messages"][-1].content[:800], "...")
    print()


def main_compiled_subagent():
    """预编译子代理示例"""
    print("=" * 60)
    print("示例 3：预编译子代理 - 用 create_mambo_agent 构建子代理")
    print("=" * 60)

    model = ChatDeepSeek(model="deepseek-v4-flash")

    # 先用 create_mambo_agent 构建一个独立的子代理 graph
    my_custom_graph = create_mambo_agent(
        model,
        system_prompt=(
            "你是一个代码审查专家。"
            "检查代码中的潜在 bug、安全漏洞和代码风格问题。"
            "给出具体的改进建议。"
        ),
    )

    agent = create_mambo_agent(
        model,
        subagents=[
            CompiledSubAgent(
                name="code-reviewer",
                description="代码审查专家，检查代码的 bug 和风格问题",
                runnable=my_custom_graph,
            ),
        ],
    )

    result = agent.invoke({
        "messages": [HumanMessage(
            "创建一个 /fib.py 文件，内容是实现斐波那契数列函数，"
            "然后用 code-reviewer 审查它。"
        )]
    }, config={"configurable": {"thread_id": "session-1"}})

    print(result["messages"][-1].content[:800], "...")
    print()


async def main_subagent_streaming():
    """子代理流式事件"""
    print("=" * 60)
    print("示例 4：接收子代理的流式事件")
    print("=" * 60)

    model = ChatDeepSeek(model="deepseek-v4-flash")

    agent = create_mambo_agent(
        model,
        subagents=[
            SubAgent(
                name="researcher",
                description="研究 Python 异步模式",
                system_prompt="你是 Python 异步编程专家。",
                model=model,
            ),
        ],
    )

    async for event in agent.astream(
        {"messages": [HumanMessage("研究一下 Python 的 asyncio 最佳实践")]},
        stream_mode=["updates", "custom"],
        config={"configurable": {"thread_id": "session-1"}},
    ):
        if event[0] == "custom":
            data = event[1]
            print(f"[子代理事件] type={data.get('type')}, "
                  f"subagent={data.get('subagent_type')}")
        else:
            print(f"[主代理事件] {list(event[1].keys())}")

    print()


if __name__ == "__main__":
    main_sync_subagents()
    main_general_purpose()
    main_compiled_subagent()
    asyncio.run(main_subagent_streaming())
