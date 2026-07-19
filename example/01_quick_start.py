"""
============================================================
Mambo Agents - 示例 01：快速上手（Quick Start）
============================================================

最简单的用法：创建一个 Agent，让它帮你写代码。

运行前请先配置 .env 文件中的 API Key：
  - 如果使用 OpenAI 模型：设置 OPENAI_API_KEY
  - 如果使用 DeepSeek 模型：设置 DEEPSEEK_API_KEY

运行方式：
  python example/01_quick_start.py
============================================================
"""

import asyncio

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from mambo_agents import create_mambo_agent
from deepseek_chat_model import ChatDeepSeek

# 加载 .env 文件中的环境变量
load_dotenv()


def main_sync():
    """同步调用示例"""
    print("=" * 60)
    print("示例 1：同步调用 - 最简单的 Agent")
    print("=" * 60)

    # 使用 DeepSeek 模型
    model = ChatDeepSeek(model="deepseek-v4-flash")

    # 创建 Agent，使用默认的虚拟文件系统（StoreBackend + InMemoryStore）
    agent = create_mambo_agent(model)

    # 向 Agent 发送任务
    result = agent.invoke({
        "messages": [
            HumanMessage("创建一个 Python 脚本 /hello.py，打印 'Hello World'")
        ]
    }, config={"configurable": {"thread_id": "session-0"}})

    # 查看 Agent 的最终回复
    print(result["messages"][-1].content)
    print()


async def main_async():
    """异步流式调用示例"""
    print("=" * 60)
    print("示例 2：异步流式输出 - 按节点输出")
    print("=" * 60)

    model = ChatDeepSeek(model="deepseek-v4-flash")
    agent = create_mambo_agent(model)

    async for event in agent.astream(
        {"messages": [HumanMessage("列出当前目录的所有文件，然后创建一个 README.md")]},
        stream_mode="updates",
        config={"configurable": {"thread_id": "session-0"}},
    ):
        print(event)

    print()


async def main_token_stream():
    """逐 token 流式输出示例"""
    print("=" * 60)
    print("示例 3：逐 token 流式输出 - 实时显示 LLM 生成内容")
    print("=" * 60)

    model = ChatDeepSeek(model="deepseek-v4-flash")
    agent = create_mambo_agent(model)

    async for msg_chunk, metadata in agent.astream(
        {"messages": [HumanMessage("解释什么是 Python 装饰器")]},
        stream_mode="messages",
        config={"configurable": {"thread_id": "session-0"}},
    ):
        if msg_chunk.content:
            print(msg_chunk.content, end="", flush=True)

    print("\n")


async def main_multi_turn():
    """多轮对话示例"""
    print("=" * 60)
    print("示例 4：多轮对话 - 使用 thread_id 保持会话")
    print("=" * 60)

    model = ChatDeepSeek(model="deepseek-v4-flash")
    agent = create_mambo_agent(model)

    # 使用同一个 thread_id 实现多轮对话
    config = {"configurable": {"thread_id": "session-1"}}

    # 第一轮：创建文件
    result1 = agent.invoke(
        {"messages": [HumanMessage("创建一个 /config.json，设置 port 为 8080")]},
        config=config,
    )
    print("[第一轮]", result1["messages"][-1].content[:200], "...")
    print()

    # 第二轮：Agent 记得之前的上下文
    result2 = agent.invoke(
        {"messages": [HumanMessage("把 port 改成 9090")]},
        config=config,
    )
    print("[第二轮]", result2["messages"][-1].content[:200], "...")
    print()

    # 换个 thread_id 就是全新的会话
    result3 = agent.invoke(
        {"messages": [HumanMessage("列出所有文件")]},
        config={"configurable": {"thread_id": "session-2"}},
    )
    print("[新会话]", result3["messages"][-1].content[:200], "...")
    print()


if __name__ == "__main__":
    # 运行同步示例
    main_sync()

    # 运行异步示例
    asyncio.run(main_async())
    asyncio.run(main_token_stream())
    asyncio.run(main_multi_turn())
