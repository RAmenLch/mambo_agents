"""
============================================================
Mambo Agents - 示例 09：高级用法（Advanced Usage）
============================================================

演示综合高级用法：
  - 自定义系统提示词
  - 预填充文件
  - 版本控制
  - 组合使用多种功能

运行前请先配置 .env 文件中的 API Key。
运行方式：
  python example/09_advanced.py
============================================================
"""

import tempfile

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from mambo_agents import StoreBackend, create_mambo_agent
from mambo_agents.backends.local import LocalBackend
from mambo_agents.middleware.security_review import SecurityReviewConfig
from deepseek_chat_model import ChatDeepSeek
from deepseek_chat_model import ChatDeepSeek

load_dotenv()


def main_custom_system_prompt():
    """自定义系统提示词"""
    print("=" * 60)
    print("示例 1：自定义系统提示词")
    print("=" * 60)

    model = ChatDeepSeek(model="deepseek-v4-flash")

    agent = create_mambo_agent(
        model,
        system_prompt="""你是一个 Python 专家助手。

## 编码规范
- 始终使用类型提示
- 遵循 PEP 8 风格
- 使用 docstring 文档字符串
- 优先使用 pathlib 处理路径

## 回复格式
- 代码块使用 ```python 标记
- 用中文回复用户
""",
    )

    result = agent.invoke({
        "messages": [HumanMessage("创建一个 /utils.py，包含一个读取 JSON 文件的函数")]
    }, config={"configurable": {"thread_id": "session-1"}})

    print(result["messages"][-1].content[:500], "...")
    print()


def main_prepopulated_files():
    """预填充文件"""
    print("=" * 60)
    print("示例 2：预填充初始文件")
    print("=" * 60)

    model = ChatDeepSeek(model="deepseek-v4-flash")

    backend = StoreBackend(initial_files={
        "/app/main.py": (
            '"""Main application entry point."""\n'
            "from fastapi import FastAPI\n\n"
            "app = FastAPI()\n\n"
            '@app.get("/")\n'
            "def root() -> dict:\n"
            '    return {"message": "Hello World"}\n'
        ),
        "/app/config.py": (
            '"""Application configuration."""\n'
            "PORT = 8080\n"
            "DEBUG = True\n"
        ),
        "/requirements.txt": (
            "fastapi==0.100.0\n"
            "uvicorn==0.23.0\n"
        ),
    })

    agent = create_mambo_agent(model, backend=backend)

    result = agent.invoke({
        "messages": [HumanMessage(
            "查看 /app/main.py，然后在 /app/config.py 中添加 DATABASE_URL 配置项"
        )]
    }, config={"configurable": {"thread_id": "session-1"}})

    print(result["messages"][-1].content[:500], "...")
    print()


def main_version_control():
    """版本控制示例"""
    print("=" * 60)
    print("示例 3：版本控制 - 自动快照文件变更")
    print("=" * 60)

    model = ChatDeepSeek(model="deepseek-v4-flash")

    with tempfile.TemporaryDirectory() as tmpdir:
        backend = LocalBackend(root_dir=tmpdir)

        agent = create_mambo_agent(
            model,
            backend=backend,
            version_control=True,  # 启用版本控制
        )

        result = agent.invoke({
            "messages": [HumanMessage(
                "创建 /version_demo.py，内容为: print('v1')"
            )]
        }, config={"configurable": {"thread_id": "session-1"}})

        print("[v1 创建]", result["messages"][-1].content[:300], "...")

        # 修改文件
        config = {"configurable": {"thread_id": "vc-demo"}}
        result = agent.invoke(
            {"messages": [HumanMessage(
                "把 /version_demo.py 的内容改为 print('v2')"
            )]},
            config=config,
        )

        print("[v2 修改]", result["messages"][-1].content[:300], "...")
        print("版本控制已启用，每次文件变更都会自动备份。")
        print("可通过 VersionStore 查询历史版本和手动回滚。")


def main_combined():
    """组合使用 - 记忆 + 安全审查 + 版本控制"""
    print("=" * 60)
    print("示例 4：组合使用 - 记忆 + 安全审查 + 版本控制")
    print("=" * 60)

    model = ChatDeepSeek(model="deepseek-v4-flash")

    with tempfile.TemporaryDirectory() as tmpdir:
        backend = LocalBackend(root_dir=tmpdir)

        agent = create_mambo_agent(
            model,
            backend=backend,
            system_prompt="你是 Python 专家，代码必须包含类型提示和 docstring。",
            version_control=True,
            interrupt_on={"write": True, "edit": True},
            security_review=SecurityReviewConfig(model=model),
        )

        result = agent.invoke({
            "messages": [HumanMessage(
                "创建 /secure_demo.py，包含一个安全的加法函数"
            )]
        }, config={"configurable": {"thread_id": "session-1"}})

        print(result["messages"][-1].content[:500], "...")
        print()
        print("已启用：系统提示词 + 版本控制 + 人工审批 + AI 安全预审")


if __name__ == "__main__":
    main_custom_system_prompt()
    main_prepopulated_files()
    main_version_control()
    main_combined()
