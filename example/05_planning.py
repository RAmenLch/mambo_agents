"""
============================================================
Mambo Agents - 示例 05：任务规划与追踪（Planning）
============================================================

演示任务规划功能 - Agent 维护结构化的 TODO 列表，自动追踪进度。

运行前请先配置 .env 文件中的 API Key。
运行方式：
  python example/05_planning.py
============================================================
"""

import json

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from mambo_agents import create_mambo_agent
from mambo_agents.middleware.planning import MamboPlanMiddleware, Plan
from deepseek_chat_model import ChatDeepSeek

load_dotenv()


def demo_basic_planning():
    """演示基本的任务规划功能（invoke 方式）。"""
    print("=" * 60)
    print("Part 1：基本任务规划")
    print("=" * 60)

    model = ChatDeepSeek(model="deepseek-v4-flash")

    # 启用规划中间件
    agent = create_mambo_agent(
        model,
        middleware=[MamboPlanMiddleware()],
    )

    config = {"configurable": {"thread_id": "session-1"}}

    # 发送一个复杂任务，Agent 会自动拆分步骤
    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    "帮我创建一个 Python 项目骨架，需要包含以下内容：\n"
                    "1. 一个 /app/main.py 入口文件\n"
                    "2. 一个 /app/config.py 配置文件\n"
                    "3. 一个 /app/utils.py 工具函数文件\n"
                    "4. 一个 /README.md 项目说明\n"
                    "请用 write_plans 工具制定计划，然后逐步完成。"
                )
            ]
        },
        config=config,
    )

    print(result["messages"][-1].content[:800], "...")
    print()

    # 从最终 state 中获取 plans
    plans: list[Plan] | None = result.get("plans")
    if plans:
        print("最终计划列表：")
        for p in plans:
            status_icon = {"pending": "⬜", "in_progress": "🔄", "completed": "✅"}[
                p.status
            ]
            print(f"  {status_icon} [{p.status}] {p.content}")
    else:
        print("(Agent 未使用 write_plans 工具)")

    print()


def demo_stream_planning():
    """演示通过 stream_mode='updates' 观察 write_plans 工具调用和计划更新。

    stream_mode='updates' 按节点输出事件，每个节点完成后触发一次：
    - 'model' 节点：AIMessage（含 tool_calls，可看到 write_plans 被调用）
    - 'tools' 节点：ToolMessage（含 write_plans 的返回值，即完整的计划列表）
    """
    print("=" * 60)
    print("Part 2：流式观察 write_plans 工具调用")
    print("=" * 60)

    model = ChatDeepSeek(model="deepseek-v4-flash")

    agent = create_mambo_agent(
        model,
        middleware=[MamboPlanMiddleware()],
    )

    config = {"configurable": {"thread_id": "session-2"}}

    print("流式事件输出：")
    print("-" * 40)

    for event in agent.stream(
        {
            "messages": [
                HumanMessage(
                    "帮我创建一个简单的项目，包含以下文件：\n"
                    "1. /src/main.py\n"
                    "2. /tests/test_main.py\n"
                    "请用 write_plans 制定计划并逐步完成。"
                )
            ]
        },
        config=config,
        stream_mode="updates",
    ):
        # event 是一个 dict，key 是节点名，value 是该节点的输出
        for node_name, node_output in event.items():
            if node_name == "model":
                # model 节点的输出包含 AIMessage
                messages = node_output.get("messages", [])
                for msg in messages:
                    if isinstance(msg, AIMessage):
                        # 检查是否调用了 write_plans 工具
                        if msg.tool_calls:
                            for tc in msg.tool_calls:
                                if tc["name"] == "write_plans":
                                    print(f"\n🔧 [model] 调用 write_plans 工具：")
                                    # tc["args"] 包含 {"plans": [{"content": "...", "status": "..."}, ...]}
                                    args = tc["args"]
                                    if "plans" in args:
                                        for plan_item in args["plans"]:
                                            icon = {
                                                "pending": "⬜",
                                                "in_progress": "🔄",
                                                "completed": "✅",
                                            }.get(plan_item.get("status", ""), "❓")
                                            print(
                                                f"     {icon} [{plan_item.get('status', '?')}] "
                                                f"{plan_item.get('content', '?')}"
                                            )
                                else:
                                    print(f"\n🔧 [model] 调用工具：{tc['name']}")
                                    print(f"     args: {json.dumps(tc['args'], ensure_ascii=False)[:150]}")
                        else:
                            # 普通文本回复
                            content = msg.content
                            if isinstance(content, str) and content.strip():
                                print(f"\n💬 [model] {content[:150]}...")

            elif node_name == "tools":
                # tools 节点的输出包含 ToolMessage
                messages = node_output.get("messages", [])
                for msg in messages:
                    if isinstance(msg, ToolMessage):
                        if msg.name == "write_plans":
                            # write_plans 的 ToolMessage 内容包含完整的计划列表
                            print(f"\n✅ [tools] write_plans 返回结果：")
                            # 内容格式： "Updated plan list to [{'content': ..., 'status': ...}, ...]"
                            print(f"     {msg.content}")
                        else:
                            # 其他工具调用结果（如 write, read 等），截断显示
                            content = str(msg.content)[:200]
                            print(f"\n✅ [tools] {msg.name}: {content}...")

    print("\n" + "-" * 40)
    print("（流式事件演示结束）")
    print()


def demo_plans_state():
    """演示从 state 中读取 plans 字段，了解 Plan 数据结构。

    Plan 数据模型（Pydantic）：
    - content (str): 任务描述
    - status (Literal["pending", "in_progress", "completed"]): 任务状态

    可以通过 result["plans"] 获取最终的 plans 列表。
    """
    print("=" * 60)
    print("Part 3：Plans 数据结构说明")
    print("=" * 60)

    model = ChatDeepSeek(model="deepseek-v4-flash")

    agent = create_mambo_agent(
        model,
        middleware=[MamboPlanMiddleware()],
    )

    config = {"configurable": {"thread_id": "session-3"}}

    # 使用 stream_mode="values" 观察 plans 的实时变化
    print("通过 stream_mode='values' 观察 plans 字段的更新：")
    print("-" * 40)

    last_plans = None
    for state in agent.stream(
        {
            "messages": [
                HumanMessage(
                    "帮我做三件事：\n"
                    "1. 创建 /README.md 文件\n"
                    "2. 创建 /config.yaml 文件\n"
                    "3. 验证两个文件都已创建\n"
                    "请使用 write_plans 工具制定计划并执行。"
                )
            ]
        },
        config=config,
        stream_mode="values",
    ):
        current_plans = state.get("plans")
        if current_plans != last_plans and current_plans:
            last_plans = current_plans
            print(f"\n📋 Plans 更新：")
            for p in current_plans:
                icon = {"pending": "⬜", "in_progress": "🔄", "completed": "✅"}.get(
                    p.status if isinstance(p, Plan) else p["status"], "❓"
                )
                content = p.content if isinstance(p, Plan) else p["content"]
                status = p.status if isinstance(p, Plan) else p["status"]
                print(f"  {icon} [{status}] {content}")

    print("\n" + "-" * 40)
    print("\nPlan 字段说明：")
    print("  • content (str): 任务描述")
    print("  • status (str): 任务状态，可选值：")
    print("      - pending: 未开始")
    print("      - in_progress: 进行中")
    print("      - completed: 已完成")
    print()
    print("获取方式：")
    print("  1. result['plans'] — invoke 后从返回值中读取")
    print("  2. state.get('plans') — stream_mode='values' 时从 state 中读取")
    print("  3. stream_mode='updates' 时，tools 节点的 ToolMessage 内容中包含 plans 列表")
    print()


def main():
    demo_basic_planning()
    demo_stream_planning()
    demo_plans_state()


if __name__ == "__main__":
    main()
