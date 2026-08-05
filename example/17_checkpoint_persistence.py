"""
============================================================
Mambo Agents - 示例 17：Checkpoint 持久化（SqliteSaver）
============================================================

演示 checkpointer 参数 - 用 SqliteSaver 把**对话历史**持久化到 SQLite 文件：
  - 同一 thread_id 的对话历史跨进程保留
  - 模拟"进程重启"：创建多个独立的 agent（共享同一个 db 文件），
    后续 agent 能回忆起之前 agent 的会话内容

重要区别（见 usage.md 5.2 / 14.5）：
  - checkpointer（SqliteSaver）持久化对话历史（消息状态）
  - 文件系统状态存放在 LangGraph Store 中：默认 InMemoryStore 不跨进程，
    需要跨进程保留文件时请传入持久化 store（如 PostgresStore），
    或使用 LocalBackend（直接落盘）

对比：默认 InMemorySaver 只在单个 CompiledStateGraph 生命周期内有效，
进程重启后对话历史全部丢失。

运行前请先配置 .env 文件中的 API Key。
运行方式：
  python example/17_checkpoint_persistence.py
============================================================
"""

import os
import tempfile

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from mambo_agents import create_mambo_agent
from deepseek_chat_model import ChatDeepSeek

load_dotenv()


def main():
    print("=" * 60)
    print("示例：SqliteSaver 持久化 - 对话历史跨进程保留")
    print("=" * 60)

    model = ChatDeepSeek(model="deepseek-v4-flash")

    # 使用临时 db 文件；实际项目中可换成固定路径（如 "checkpoints.db"）
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "checkpoints.db")
        config = {"configurable": {"thread_id": "session-1"}}

        # ── "进程 A"：第一个 agent 实例 ──
        print("\n--- 进程 A：创建文件并写入配置 ---")
        with SqliteSaver.from_conn_string(db_path) as checkpointer:
            agent_a = create_mambo_agent(
                model,
                checkpointer=checkpointer,
            )
            result_a = agent_a.invoke({
                "messages": [HumanMessage(
                    "创建一个 /config.json，设置 port 为 8080"
                )]
            }, config=config)
            print("[A]", result_a["messages"][-1].content[:300], "...")
            # 此时 agent_a 超出作用域销毁 —— 模拟进程 A 退出

        # ── "进程 B"：第二个 agent 实例（同一 db 文件）──
        print("\n--- 进程 B：新实例，同一 thread_id ---")
        with SqliteSaver.from_conn_string(db_path) as checkpointer:
            agent_b = create_mambo_agent(
                model,
                checkpointer=checkpointer,
            )
            # B 应该能回忆起 A 的对话内容（对话历史已持久化）
            result_b = agent_b.invoke({
                "messages": [HumanMessage(
                    "请回忆我们上一轮会话做了什么：创建了什么文件？port 设置成了多少？"
                    "直接根据对话历史回答，不要操作文件系统。"
                )],
            }, config=config)
            print("[B]", result_b["messages"][-1].content[:400], "...")

        # ── "进程 C"：再次重启，验证历史持续累积 ──
        print("\n--- 进程 C：再次重启，验证对话历史持续累积 ---")
        with SqliteSaver.from_conn_string(db_path) as checkpointer:
            agent_c = create_mambo_agent(
                model,
                checkpointer=checkpointer,
            )
            result_d = agent_c.invoke({
                "messages": [HumanMessage(
                    "我们一共聊过几轮？每一轮做了什么？请根据对话历史回答。"
                )],
            }, config=config)
            print("[C]", result_d["messages"][-1].content[:400], "...")
            # 预期：C 能完整复述 A、B 两轮的对话内容

        print("\n提示：")
        print("  1. 默认 InMemorySaver 下，agent 销毁后对话历史全部丢失；")
        print("     SqliteSaver 等持久化 checkpointer 可跨进程保留对话历史。")
        print("  2. 注意：文件系统状态存放在 LangGraph Store（默认 InMemoryStore）中，")
        print("     不随 checkpointer 持久化；需要文件也跨进程保留时，")
        print("     请传入持久化 store（如 PostgresStore）或使用 LocalBackend 落盘。")


if __name__ == "__main__":
    main()
