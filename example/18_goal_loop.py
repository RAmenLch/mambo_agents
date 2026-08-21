"""
============================================================
Mambo Agents - 示例 18：目标驱动循环（Goal Loop）
============================================================

演示 GoalLoopMiddleware 的两种形态：

1. 用户控制：预设目标 + 完成条件，强制 LLM 必须完成某事
   （例如：最少调用一次 show 工具才允许结束）
2. LLM 控制：create_goal / update_goal / get_goal，
   LLM 自主创建长程任务并一路干到底

每个示例末尾都会打印「goal 工具调用轨迹」，
方便确认 get_goal / create_goal / update_goal 是否真的被调用、
以及自动注入的续跑轮次。

运行前请先配置 .env 文件中的 API Key。
运行方式：
  python example/18_goal_loop.py
============================================================
"""

from dotenv import load_dotenv
import json
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from mambo_agents import create_mambo_agent
from mambo_agents.backends.store import StoreBackend
from mambo_agents.middleware import GoalLoopConfig, tool_called_at_least
from deepseek_chat_model import ChatDeepSeek

load_dotenv()

INJECT_PREFIX = "goal-loop-"


def show(text: str) -> str:
    """向用户展示一段文本内容。"""
    return f"[展示] {text}"


show_tool = StructuredTool.from_function(func=show, name="show")


def _indent(text, prefix: str = "    ") -> str:
    """多行文本统一缩进,保持结构清晰。"""
    return "\n".join(f"{prefix}{line}" for line in str(text).splitlines())


def _format_content(content: str) -> str:
    """若内容是合法 JSON 则格式化多行显示,否则原样返回。"""
    try:
        return json.dumps(json.loads(content), ensure_ascii=False, indent=2)
    except (ValueError, TypeError):
        return content


def print_goal_trace(messages) -> None:
    """按时间线打印完整执行轨迹:模型输出与工具调用交错显示。

    顺序即真实执行顺序:
      用户消息 → 模型输出 → 工具调用 → 工具返回 → 模型输出 → ...
    自动注入的 get_goal(强制续跑)以 [自动注入] 标记。
    工具返回内容完整打印,不截断。
    """
    print()
    print("═" * 60)
    print("  执行轨迹(按真实时间线)")
    print("═" * 60)
    for m in messages:
        if isinstance(m, HumanMessage):
            print(f"\n▶ [用户]")
            print(_indent(m.content))
        elif isinstance(m, AIMessage):
            if m.content:
                print(f"\n▶ [模型]")
                print(_indent(m.content))
            for tc in m.tool_calls:
                tag = "[自动注入]" if str(tc["id"]).startswith(INJECT_PREFIX) else "[模型主动]"
                print(f"\n▶ {tag} {tc['name']}")
                print(f"    参数: {json.dumps(tc['args'], ensure_ascii=False)}")
        elif isinstance(m, ToolMessage) and m.name:
            content = m.content if isinstance(m.content, str) else str(m.content)
            print(f"  ↳ 返回({m.name}):")
            print(_indent(_format_content(content)))


def count_injected(messages) -> int:
    """统计自动注入的 get_goal 续跑次数。"""
    return sum(
        1
        for m in messages
        if isinstance(m, AIMessage)
        and any(str(tc.get("id", "")).startswith(INJECT_PREFIX) for tc in m.tool_calls)
    )


def demo_user_controlled():
    """形态 1：用户控制 —— 强制 LLM 最少调用一次 show 工具"""
    print("=" * 60)
    print("示例 1：用户控制（强制最少调用一次 show 工具）")
    print("=" * 60)

    model = ChatDeepSeek(model="deepseek-v4-flash")

    agent = create_mambo_agent(
        model,
        backend=StoreBackend(initial_files={
            "/tasks.md": (
                "# 今日任务\n"
                "- 完成项目 A 的代码审查\n"
                "- 修复模块 B 的登录 bug\n"
                "- 为模块 C 补充单元测试\n"
            ),
        }),
        tools=[show_tool],
        goal_loop=GoalLoopConfig(
            mode="preset",
            objective="必须调用 show 工具向用户展示工作成果",
            conditions=[tool_called_at_least("show", 1)],
            max_rounds=4,  # 最多到达 after_agent 4 次（即最多强制续跑 3 次）
        ),
    )

    # 提示词引导"先干活、后展示"的两段式流程:
    # 第一轮模型正常阅读文件、分析并回复(但不调用 show)→ 收尾后条件未满足,
    # 自动注入 get_goal 强制续跑 → 第二轮模型看到目标要求,调用 show 展示 → 结束。
    result = agent.invoke(
        {
            "messages": [HumanMessage(
                "请阅读 /tasks.md 中的今日任务并完成分析。\n"
                "请先阅读文件、完成分析并回复分析结论"
                "目前现在正在测试get_goal工具,请你在 get_goal 返回 第 1/4 轮 时,绝对不要自行show(那怕它要求你使用show),并直接结束对话"
                "如果你发现get_goal工具返回的是2/4轮,此时你就可以使用show了"
            )]
        },
        config={"configurable": {"thread_id": "goal-demo-1"}},
    )

    print_goal_trace(result["messages"])
    show_calls = [
        m for m in result["messages"]
        if isinstance(m, ToolMessage) and m.name == "show"
    ]
    print(f"自动注入续跑次数: {count_injected(result['messages'])}")
    print(f"show 工具执行次数: {len(show_calls)}")
    print(f"最终 goal 状态: {result.get('goal', {}).get('status')}")
    print(f"Agent 最终回复: {result['messages'][-1].content[:200]}...")
    print()


def demo_llm_controlled():
    """形态 2：LLM 控制 —— 长程任务自动续跑"""
    print("=" * 60)
    print("示例 2：LLM 控制（自主创建目标并长程执行）")
    print("=" * 60)

    model = ChatDeepSeek(model="deepseek-v4-flash")

    agent = create_mambo_agent(
        model,
        goal_loop=GoalLoopConfig(
            mode="llm",   # 注册 create_goal / update_goal / get_goal
            max_rounds=3,  # 最高轮数上限（create_goal 的轮数会被 clamp 到此值）
        ),
    )

    # 提示词显式要求模型先 create_goal 进入长程模式，
    # 并在完成后用 update_goal(complete) 收尾 —— 展示完整生命周期。
    result = agent.invoke(
        {
            "messages": [HumanMessage(
                "帮我写一个完整的贪吃蛇游戏,要有计分、难度递增和最高分存档。\n"
                "这是一个多阶段的大任务:请先调用 create_goal 创建长程目标,在3个轮次后解决"
                "在4个轮次中你必须不断完善你的代码,你必须等到第 3/3 轮才可标记解决action='complete'"
                "全部完成后再调用 update_goal(action='complete') 结束。"
            )]
        },
        config={"configurable": {"thread_id": "goal-demo-2"}},
    )

    print_goal_trace(result["messages"])
    goal = result.get("goal")
    if goal is None:
        print("未创建目标（模型没有调用 create_goal，任务在一轮内直接完成）")
    else:
        print(f"最终 goal: status={goal['status']}, "
              f"rounds={goal['rounds']}/{goal['max_rounds']}, "
              f"objective={goal['objective'][:40]}")
    print(f"自动注入续跑次数: {count_injected(result['messages'])}")
    print(f"Agent 最终回复: {result['messages'][-1].content[:200]}...")
    print()


if __name__ == "__main__":
    demo_user_controlled()
    demo_llm_controlled()
