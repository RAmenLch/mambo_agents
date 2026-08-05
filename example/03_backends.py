"""
============================================================
Mambo Agents - 示例 03：后端文件系统（Backends）
============================================================

演示不同的文件系统后端：
  - StoreBackend：虚拟文件系统（默认）
  - LocalBackend：本地磁盘操作

运行前请先配置 .env 文件中的 API Key。
运行方式：
  python example/03_backends.py
============================================================
"""

import asyncio
import tempfile

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from mambo_agents import StoreBackend, create_mambo_agent
from mambo_agents.backends.local import LocalBackend
from deepseek_chat_model import ChatDeepSeek
import os
load_dotenv()


def main_store_backend():
    """虚拟文件系统 - 预填充初始文件"""
    print("=" * 60)
    print("示例 1：StoreBackend - 虚拟文件系统（预填充文件）")
    print("=" * 60)

    model = ChatDeepSeek(model="deepseek-v4-flash")

    # 创建带初始文件的虚拟文件系统
    # 注意：文件必须放在 /workspace/ 前缀下，才能在 Agent 的
    # ls /workspace 视图中可见（StoreBackend 的 workspace_root 固定为 /workspace）
    backend = StoreBackend(initial_files={
        "/workspace/app/main.py": (
            "def greet(name: str) -> str:\n"
            '    return f"Hello, {name}!"\n'
            "\n"
            'if __name__ == "__main__":\n'
            '    print(greet("World"))\n'
        ),
        "/workspace/app/config.yaml": (
            "server:\n"
            "  port: 8080\n"
            "  host: 0.0.0.0\n"
            "database:\n"
            "  url: localhost:5432\n"
        ),
        "/workspace/app/requirements.txt": (
            "fastapi==0.100.0\n"
            "uvicorn==0.23.0\n"
        ),
    })

    agent = create_mambo_agent(model, backend=backend)

    result = agent.invoke({
        "messages": [HumanMessage(
            "查看 /workspace/app/config.yaml 的内容，然后把 server.port 改成 9090"
        )]
    }, config={"configurable": {"thread_id": "session-1"}})
    print(result["messages"][-1].content[:500], "...")
    print()


def main_local_backend():
    """本地磁盘操作"""
    print("=" * 60)
    print("示例 2：LocalBackend - 操作本地文件系统")
    print("=" * 60)

    model = ChatDeepSeek(model="deepseek-v4-flash")

    # 使用临时目录，避免污染实际文件系统
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = LocalBackend(root_dir=tmpdir)

        agent = create_mambo_agent(model, backend=backend)

        result = agent.invoke({
            "messages": [HumanMessage(
                f"在 {tmpdir} 中创建一个 hello.py 文件，"
                "内容是打印 'Hello from LocalBackend!'"
            )]
        }, config={"configurable": {"thread_id": "session-1"}})
        print(result["messages"][-1].content[:500], "...")
        print()

        # 验证文件确实写入了磁盘
        hello_path = os.path.join(tmpdir, "hello.py")
        if os.path.exists(hello_path):
            with open(hello_path) as f:
                print(f"[磁盘上的实际文件内容] {hello_path}")
                print(f.read())
        else:
            print(f"文件未找到: {hello_path}")


async def main_astream():
    """异步流式 - StoreBackend"""
    print("=" * 60)
    print("示例 3：异步流式 + StoreBackend")
    print("=" * 60)

    model = ChatDeepSeek(model="deepseek-v4-flash")

    backend = StoreBackend(initial_files={
        "/workspace/project/README.md": "# My Project\n\nA sample project.\n",
    })
    agent = create_mambo_agent(model, backend=backend)

    async for event in agent.astream(
        {"messages": [HumanMessage(
            "查看 /workspace/project/README.md，然后添加一个 '## Features' 章节"
        )]},
        stream_mode="updates",
        config={"configurable": {"thread_id": "session-1"}},
    ):
        print(event)

    print()


if __name__ == "__main__":
    main_store_backend()
    main_local_backend()
    asyncio.run(main_astream())
