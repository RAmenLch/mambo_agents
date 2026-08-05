"""
============================================================
Mambo Agents - 示例 15：异步子代理（Async Sub-Agents）
============================================================

演示 AsyncSubAgentMiddleware - 后台线程运行子代理：
  - async_task(description, subagent_type)  → 立即返回 task_id
  - async_status(task_id)                   → 查询状态/进度/结果
  - async_list(status_filter)               → 列出所有任务
  - async_cancel(task_id)                   → 取消运行中的任务
  - report_progress(message, percentage)    → 子代理内部自报进度

状态值：running / success / error / cancelled / crashed

运行前请先配置 .env 文件中的 API Key。
运行方式：
  python example/15_async_subagents.py
============================================================
"""

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from mambo_agents import create_mambo_agent
from mambo_agents.middleware.subagents import SubAgent
from deepseek_chat_model import ChatDeepSeek

load_dotenv()


def main_async_task_and_status():
    """演示 async_task 启动后台任务 + async_status 查询状态。"""
    print("=" * 60)
    print("示例 1：async_task + async_status - 后台任务与状态查询")
    print("=" * 60)

    model = ChatDeepSeek(model="deepseek-v4-flash")

    agent = create_mambo_agent(
        model,
        async_subagents=[
            SubAgent(
                name="researcher",
                description="后台研究助手，负责长时间运行的研究任务",
                system_prompt=(
                    "你是一个研究助手。认真分析收到的任务描述，"
                    "输出结构化的研究结论。"
                ),
                model=model,
            ),
        ],
        async_subagent_timeout=600,  # 10 分钟超时
    )

    config = {"configurable": {"thread_id": "session-1"}}

    # 第 1 轮：启动后台任务
    result = agent.invoke({
        "messages": [HumanMessage(
            "用 async_task 启动一个后台任务：研究 Python 3.13 的新特性，"
            "子代理类型用 researcher。启动后告诉我 task_id。"
        )]
    }, config=config)
    print("[启动任务]", result["messages"][-1].content[:400], "...")
    print()

    # 第 2 轮：查询任务状态（同一会话，Agent 记得 task_id）
    result = agent.invoke({
        "messages": [HumanMessage(
            "用 async_status 查询刚才那个后台任务的状态，"
            "如果还在 running 就告诉我进度。"
        )]
    }, config=config)
    print("[查询状态]", result["messages"][-1].content[:400], "...")
    print()


def main_async_list_and_cancel():
    """演示 async_list 列出任务 + async_cancel 取消任务。"""
    print("=" * 60)
    print("示例 2：async_list + async_cancel - 任务列表与取消")
    print("=" * 60)

    model = ChatDeepSeek(model="deepseek-v4-flash")

    agent = create_mambo_agent(
        model,
        async_subagents=[
            SubAgent(
                name="researcher",
                description="后台研究助手",
                system_prompt="你是一个研究助手。",
                model=model,
            ),
        ],
    )

    config = {"configurable": {"thread_id": "session-2"}}

    # 启动一个任务，然后立即取消它
    result = agent.invoke({
        "messages": [HumanMessage(
            "用 async_task 启动一个后台任务：分析 1000 个 GitHub 仓库的代码质量，"
            "子代理类型用 researcher。启动后立刻用 async_cancel 取消它，"
            "然后用 async_list 列出所有任务确认状态。"
        )]
    }, config=config)
    print("[启动并取消]", result["messages"][-1].content[:500], "...")
    print()


def main_async_status_enum():
    """说明 AsyncTaskData 状态字段结构。"""
    print("=" * 60)
    print("示例 3：AsyncTaskData 字段说明")
    print("=" * 60)
    print("""
async_status / async_list 返回的每个任务包含以下字段：
  task_id            str      任务唯一标识
  agent_name         str      子代理名称
  status             str      running / success / error / cancelled / crashed
  progress_message   str|None 子代理通过 report_progress 自报的进度描述
  percentage         float    完成度 0.0 ~ 1.0
  result             str|None 任务成功后的最终结果
  error_message      str|None 失败原因
  created_at         str      创建时间
  finished_at        str|None 结束时间
  cancelled_at       str|None 取消时间

崩溃恢复：使用持久化 checkpointer（如 SqliteSaver）时，进程重启后调用
async_status / async_list 会自动把丢失线程的 running 任务标记为 crashed。
默认 InMemorySaver 下重启后任务记录全部丢失。
""")


if __name__ == "__main__":
    main_async_task_and_status()
    main_async_list_and_cancel()
    main_async_status_enum()
