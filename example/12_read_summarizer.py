"""
============================================================
Mambo Agents - 示例 12：大文件读取摘要（ReadSummarizer）
============================================================

演示 ReadSummarizer - 当读取的文件超过 max_read_chars 上限时，
用文件类型感知的结构化摘要替代整段内容，帮助 AI 定位大文件中
需要精读的章节（配合 offset + limit 二次精读）。

Mambo 内置 9 种 summarizer：
  python / javascript / java / c / cpp / go / rust / markdown / json

运行前请先配置 .env 文件中的 API Key。
运行方式：
  python example/12_read_summarizer.py
============================================================
"""

import os
import tempfile

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from mambo_agents import create_mambo_agent
from mambo_agents.backends.local import LocalBackend
from mambo_agents.backends.schemas import VirtualPath
from mambo_agents.read_summarizers import (
    composite_summarizer,
    json_summarizer,
    markdown_summarizer,
    python_summarizer,
)
from deepseek_chat_model import ChatDeepSeek

load_dotenv()

# 构造一个超过 max_read_chars 的大 Python 文件
_BIG_PYTHON_SOURCE = '''"""Demo module with many definitions."""

import os
import sys
from dataclasses import dataclass


@dataclass
class Config:
    """Application configuration."""
    host: str = "0.0.0.0"
    port: int = 8080
    debug: bool = False


def load_config(path: str) -> Config:
    """Load configuration from a file."""
    return Config()


def save_config(config: Config, path: str) -> None:
    """Persist configuration to disk."""
    pass


def validate_port(port: int) -> bool:
    """Check that a port number is valid."""
    return 0 < port < 65536


def start_server(config: Config) -> None:
    """Start the HTTP server."""
    print(f"listening on {config.host}:{config.port}")


def stop_server() -> None:
    """Stop the HTTP server gracefully."""
    pass


def restart_server(config: Config) -> None:
    """Stop then start the server again."""
    stop_server()
    start_server(config)


def main() -> None:
    cfg = load_config("config.yaml")
    start_server(cfg)


if __name__ == "__main__":
    main()
''' * 30  # 重复 30 次，确保超过 max_read_chars


def demo_direct_call():
    """直接调用 summarizer 回调，观察摘要输出格式。"""
    print("=" * 60)
    print("Part 1：直接调用 python_summarizer()")
    print("=" * 60)

    summarizer = python_summarizer()
    summary = summarizer(
        VirtualPath("/app/main.py"),
        _BIG_PYTHON_SOURCE,
        max_chars=2000,
    )
    print(summary)
    print()


def demo_agent_with_summarizer():
    """通过 LocalBackend(summarizer=...) 集成到 Agent。

    将 max_read_chars 调小（2000 字符），使大文件读取必然触发摘要，
    观察 Agent 如何依据摘要定位并精读指定函数。
    """
    print("=" * 60)
    print("Part 2：Agent 集成 - 读取大文件时自动获得结构化摘要")
    print("=" * 60)

    model = ChatDeepSeek(model="deepseek-v4-flash")

    with tempfile.TemporaryDirectory() as tmpdir:
        # 把大文件写入临时目录
        with open(os.path.join(tmpdir, "main.py"), "w", encoding="utf-8") as f:
            f.write(_BIG_PYTHON_SOURCE)

        backend = LocalBackend(
            root_dir=tmpdir,
            max_read_chars=2000,          # 调小上限，确保触发摘要
            summarizer=python_summarizer(),
        )

        agent = create_mambo_agent(model, backend=backend)

        result = agent.invoke({
            "messages": [HumanMessage(
                "读取 /workspace/main.py，找到 start_server 函数的实现，"
                "告诉我它打印什么内容"
            )]
        }, config={"configurable": {"thread_id": "session-1"}})

        print(result["messages"][-1].content[:600], "...")
        print()


def demo_composite_summarizer():
    """演示 composite_summarizer 组合多种语言的摘要器。"""
    print("=" * 60)
    print("Part 3：composite_summarizer - 组合多种语言摘要器")
    print("=" * 60)

    summarizer = composite_summarizer([
        python_summarizer(),
        json_summarizer(),
        markdown_summarizer(),
    ])

    with tempfile.TemporaryDirectory() as tmpdir:
        json_content = '{"server": {"host": "0.0.0.0", "port": 8080}, "database": {"url": "postgres://localhost:5432", "pool": {"min": 1, "max": 10}}, "logging": {"level": "info", "file": "/var/log/app.log"}}'
        md_content = "# Project\n\n## Features\n- Fast\n- Safe\n\n## Usage\n\n### Install\npip install x\n\n### Run\npython main.py\n\n## FAQ\n\n### Q1\nA1\n\n### Q2\nA2\n"

        with open(os.path.join(tmpdir, "config.json"), "w", encoding="utf-8") as f:
            f.write(json_content)
        with open(os.path.join(tmpdir, "README.md"), "w", encoding="utf-8") as f:
            f.write(md_content)

        backend = LocalBackend(
            root_dir=tmpdir,
            max_read_chars=100,
            summarizer=summarizer,
        )

        print("JSON 文件摘要：")
        print("-" * 40)
        print(summarizer(VirtualPath("/workspace/config.json"), json_content, 100))
        print()
        print("Markdown 文件摘要：")
        print("-" * 40)
        print(summarizer(VirtualPath("/workspace/README.md"), md_content, 100))
        print()

        # 验证 backend 已挂载组合摘要器（读取超限文件返回摘要而非报错）
        agent = create_mambo_agent(
            ChatDeepSeek(model="deepseek-v4-flash"),
            backend=backend,
        )
        result = agent.invoke({
            "messages": [HumanMessage(
                "读取 /workspace/README.md，用一句话概括文档主题"
            )]
        }, config={"configurable": {"thread_id": "session-2"}})
        print(result["messages"][-1].content[:400], "...")
        print()


if __name__ == "__main__":
    demo_direct_call()
    demo_agent_with_summarizer()
    demo_composite_summarizer()
