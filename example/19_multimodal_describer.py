"""
============================================================
Mambo Agents - 示例 19：多模态描述（MultimodalDescriber）
============================================================

演示 MultimodalDescriber - 当纯文本模型读取图片/视频/音频/文档时，
原本得到的是一段 base64 多模态内容块，纯文本模型无法理解。配置
MultimodalDescriber 后，会用多模态模型生成精确的文字描述，
让纯文本模型也能“看”媒体文件。

本示例：
  - 测试图片：docs/mambo.png
  - 识图模型：deepseek-v4-flash-vision-exp（ChatDeepSeek）
  - 文本模型：deepseek-v4-flash（ChatDeepSeek）

运行前请先配置 .env 文件中的 DEEPSEEK_API_KEY。
运行方式（在项目根目录执行）：
  python example/19_multimodal_describer.py
============================================================
"""

import base64
import os
import shutil
import tempfile

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from mambo_agents import create_mambo_agent
from mambo_agents.backends.local import LocalBackend
from mambo_agents.backends.schemas import VirtualPath
from mambo_agents.multimodal_describers import multimodal_describer
from deepseek_chat_model import ChatDeepSeek

load_dotenv()

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_MAMBO_PNG = os.path.join(_PROJECT_ROOT, "docs", "mambo.png")


def _load_mambo_png() -> str:
    """读取 docs/mambo.png 并返回 base64 内容。"""
    with open(_MAMBO_PNG, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def demo_direct_call():
    """直接调用 multimodal_describer 回调，观察精确描述输出。"""
    print("=" * 60)
    print("Part 1：直接调用 multimodal_describer() 识图 docs/mambo.png")
    print("=" * 60)

    vision_model = ChatDeepSeek(model="deepseek-v4-flash-vision-exp")
    describer = multimodal_describer(vision_model, max_chars=500)

    description = describer(
        VirtualPath("/workspace/mambo.png"),
        _load_mambo_png(),
        "image/png",
    )
    print(description)
    print()


def demo_agent_with_describer():
    """通过 LocalBackend(multimodal_describer=...) 集成到 Agent。

    纯文本模型读取图片时，得到的是多模态模型生成的精确文字描述。
    """
    print("=" * 60)
    print("Part 2：Agent 集成 - 纯文本模型读图")
    print("=" * 60)

    text_model = ChatDeepSeek(model="deepseek-v4-flash")
    vision_model = ChatDeepSeek(model="deepseek-v4-flash-vision-exp")

    with tempfile.TemporaryDirectory() as tmpdir:
        # 把测试图片复制进 backend 工作区
        shutil.copy(_MAMBO_PNG, os.path.join(tmpdir, "mambo.png"))

        backend = LocalBackend(
            root_dir=tmpdir,
            multimodal_describer=multimodal_describer(vision_model),
        )

        agent = create_mambo_agent(text_model, backend=backend)

        result = agent.invoke({
            "messages": [HumanMessage(
                "读取 /workspace/mambo.png，精确描述这张图里有什么"
            )]
        }, config={"configurable": {"thread_id": "session-1"}})

        print(result["messages"][-1].content[:600], "...")
        print()


if __name__ == "__main__":
    demo_direct_call()
    demo_agent_with_describer()
