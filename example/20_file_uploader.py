"""
============================================================
Mambo Agents - 示例 20：文件预上传钩子（FileUploader）
============================================================

read 工具读取图片等多模态文件时，默认会把 base64 直接放进 ToolMessage：
  - 它会进入 LangGraph checkpoint 和每一次模型请求，大文件造成膨胀 / 超限
    （如 DeepSeek 单图内联 32 MiB 上限）。
配置 file_uploader 钩子后，文件先上传（本示例用 DeepSeek Files API），
消息里只保留 {"type": "file", "file_id": ...} 的轻量引用块。

本示例：
  - 测试图片：docs/mambo.png
  - 上传器：deepseek_file_uploader（store 缓存，同一文件不会重复上传）
  - 对话模型：deepseek-v4-flash-vision-exp（示例 ChatDeepSeek 已适配 file 块）

运行前请先配置 .env 文件中的 DEEPSEEK_API_KEY。
运行方式（在项目根目录执行）：
  python example/20_file_uploader.py
============================================================
"""

import asyncio
import base64
import os
import shutil
import tempfile

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.store.memory import InMemoryStore

from mambo_agents import create_mambo_agent
from mambo_agents.backends.local import LocalBackend
from mambo_agents.backends.schemas import VirtualPath
from mambo_agents.file_uploaders import deepseek_file_uploader
from deepseek_chat_model import ChatDeepSeek

load_dotenv()

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_MAMBO_PNG = os.path.join(_PROJECT_ROOT, "docs", "mambo.png")


def _load_mambo_png() -> str:
    """读取 docs/mambo.png 并返回 base64 内容。"""
    with open(_MAMBO_PNG, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def demo_direct_call():
    """直接调用上传钩子：查看返回块，并验证第二次调用命中缓存。"""
    print("=" * 60)
    print("Part 1：直接调用 deepseek_file_uploader（含缓存校验）")
    print("=" * 60)

    uploader = deepseek_file_uploader(store=InMemoryStore())
    file_path = VirtualPath("/workspace/mambo.png")

    async def run():
        blocks1 = await uploader(file_path, _load_mambo_png(), "image/png")
        blocks2 = await uploader(file_path, _load_mambo_png(), "image/png")
        print("第一次返回块:", blocks1)
        print("第二次返回块:", blocks2)
        print("两次 file_id 一致（命中 store/内存缓存，未重复上传）:", blocks1 == blocks2)

    asyncio.run(run())
    print()


def demo_agent_with_uploader():
    """Agent 集成：read 图片后消息里只有 file_id 引用，checkpoint 不再存 base64。"""
    print("=" * 60)
    print("Part 2：Agent 集成 - 读图仅保留 file_id 引用")
    print("=" * 60)

    model = ChatDeepSeek(model="deepseek-v4-flash-vision-exp")

    with tempfile.TemporaryDirectory() as tmpdir:
        shutil.copy(_MAMBO_PNG, os.path.join(tmpdir, "mambo.png"))

        backend = LocalBackend(
            root_dir=tmpdir,
            file_uploader=deepseek_file_uploader(store=InMemoryStore()),
        )
        agent = create_mambo_agent(model, backend=backend)

        result = agent.invoke(
            {"messages": [HumanMessage("读取 /workspace/mambo.png，说明这张图里有什么")]},
            config={"configurable": {"thread_id": "session-1"}},
        )

        for message in result["messages"]:
            if isinstance(message, ToolMessage):
                print("read 工具消息 content_blocks:", message.content_blocks)

        print()
        print("模型回答:", str(result["messages"][-1].content)[:300], "...")
        print()
        print('说明：read 消息中只有 {"type": "file", "file_id": ...}，')
        print("base64 不再进入 checkpoint；模型请求通过 file_id 引用已上传的图片。")


if __name__ == "__main__":
    demo_direct_call()
    demo_agent_with_uploader()
