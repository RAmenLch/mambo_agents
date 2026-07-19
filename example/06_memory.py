"""
============================================================
Mambo Agents - 示例 06：记忆系统（Memory）
============================================================

演示 Agent 记忆功能 - 从 AGENTS.md 加载持久上下文，
并在交互中学习回写新知识。

Memory 中间件的工作原理：
  1. before_agent 阶段：从 backend 加载 AGENTS.md 文件内容
  2. wrap_model_call 阶段：将内容注入到 System Prompt 中
  3. Agent 可自行通过 write/edit 工具回写新的记忆

运行前请先配置 .env 文件中的 API Key。
运行方式：
  python example/06_memory.py
============================================================
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from dotenv import load_dotenv
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import HumanMessage, SystemMessage

from mambo_agents import StoreBackend, create_mambo_agent
from mambo_agents.backends.schemas import VirtualPath
from deepseek_chat_model import ChatDeepSeek

load_dotenv()


# ---------------------------------------------------------------------------
# 辅助 middleware：截取 memory 注入后的 system prompt
# ---------------------------------------------------------------------------


class SystemPromptInterceptor(AgentMiddleware):
    """拦截 middleware，放在 memory middleware 之后，截取注入后的 system prompt。

    MamboMemoryMiddleware 在 wrap_model_call 中通过 modify_request 将 memory 注入
    到 request.system_message。这个拦截器排在它后面，因此拿到的 system_message
    已经包含了注入后的 memory 内容。
    """

    captured: str | None = None
    """最近一次截取的 system prompt 文本。"""

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        sm = request.system_message
        if isinstance(sm, SystemMessage):
            content = sm.content
            if isinstance(content, str):
                self.captured = content
            elif isinstance(content, list):
                self.captured = "\n".join(
                    b["text"] if isinstance(b, dict) else str(b) for b in content
                )
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        sm = request.system_message
        if isinstance(sm, SystemMessage):
            content = sm.content
            if isinstance(content, str):
                self.captured = content
            elif isinstance(content, list):
                self.captured = "\n".join(
                    b["text"] if isinstance(b, dict) else str(b) for b in content
                )
        return await handler(request)


def demo_memory_injection():
    """演示 memory 文件内容如何被真实注入到 system prompt 中。

    通过一个自定义 SystemPromptInterceptor middleware（排在 memory middleware
    之后）截取 wrap_model_call 中的 request.system_message，这就是 Agent 实际
    收到并执行的 system prompt——包含 memory 注入后的完整内容。
    """
    print("=" * 60)
    print("Part 1：截取 Memory 注入后的 System Prompt")
    print("=" * 60)

    model = ChatDeepSeek(model="deepseek-v4-flash")

    memory_content = (
        "# Project AGENTS.md\n\n"
        "## 项目信息\n"
        "- 项目名称: MyApp\n"
        "- 编程语言: Python 3.12+\n"
        "- 包管理工具: uv\n\n"
        "## 编码规范\n"
        "- 始终使用类型提示\n"
        "- 遵循 PEP 8 风格\n"
        "- 使用 f-string 进行字符串格式化\n\n"
        "## 测试命令\n"
        "- 运行测试: pytest tests/\n"
        "- 检查覆盖率: pytest --cov=src/\n"
    )

    backend = StoreBackend(initial_files={
        "/.mambo/memory/AGENTS.md": memory_content,
    })

    interceptor = SystemPromptInterceptor()

    agent = create_mambo_agent(
        model,
        backend=backend,
        memory_sources=[VirtualPath("/.mambo/memory/AGENTS.md")],
        middleware=[interceptor],  # 排在 memory 后面，截取注入后的 system prompt
    )

    # 展示 memory 文件原始内容
    print("\n📄 原始 AGENTS.md 文件内容：")
    print("-" * 40)
    print(memory_content)
    print("-" * 40)

    # 调用 Agent，拦截器会截取到注入后的 system prompt
    config = {"configurable": {"thread_id": "memory-demo-1"}}
    result = agent.invoke(
        {"messages": [HumanMessage(
            "帮我创建一个 /app/calculator.py 文件，"
            "包含一个 add 函数和一个 multiply 函数"
        )]},
        config=config,
    )

    # 展示 Agent 运行时真实的 system prompt（由拦截器截取）
    captured = interceptor.captured
    if captured:
        # 找出 memory 注入的起始位置（<agent_memory> 标记）
        marker = "<agent_memory>"
        idx = captured.find(marker)
        if idx != -1:
            print(f"\n🔗 Agent 实际收到的 System Prompt：")
            print("-" * 40)
            # 显示注入点之前的部分（原始 system prompt 尾部）
            before = captured[:idx].strip()
            # 只显示原始 prompt 的最后 300 字符
            if len(before) > 300:
                print("...(原始 system prompt 头部省略)...")
                print(before[-300:])
            else:
                print(before)
            print()
            # 显示 memory 注入部分
            print(f"{'='*40}")
            print("▼ 以下是 Memory 中间件注入的内容 ▼")
            print(f"{'='*40}")
            # 只显示注入内容的前 800 字符
            injected = captured[idx:]
            if len(injected) > 800:
                print(injected[:800])
                print(f"...(共 {len(injected)} 字符，后续省略)")
            else:
                print(injected)
            print(f"{'='*40}")
            print(f"▲ Memory 注入内容结束（共 {len(injected)} 字符）▲")
            print(f"{'='*40}")
        else:
            print("\n🔗 Agent 实际收到的 System Prompt（完整）：")
            print("-" * 40)
            print(captured[:1000])
            if len(captured) > 1000:
                print(f"...(共 {len(captured)} 字符，后续省略)")
    else:
        print("\n⚠️ 未能截取到 system prompt")

    print(f"\n💬 Agent 回复（基于 memory 中的编码规范）：")
    print(result["messages"][-1].content[:500], "...")
    print()


def demo_memory_learning():
    """演示 Agent 根据记忆自主学习回写。

    第二轮调用时，Agent 记得之前的编码规范和上下文。
    """
    print("=" * 60)
    print("Part 2：Agent 记忆持久化与自主学习")
    print("=" * 60)

    model = ChatDeepSeek(model="deepseek-v4-flash")

    memory_content = (
        "# Project AGENTS.md\n\n"
        "## 编码规范\n"
        "- 始终使用类型提示\n"
        "- 遵循 PEP 8 风格\n\n"
        "## 测试命令\n"
        "- 运行测试: pytest tests/\n"
    )

    backend = StoreBackend(initial_files={
        "/.mambo/memory/AGENTS.md": memory_content,
    })

    agent = create_mambo_agent(
        model,
        backend=backend,
        memory_sources=[VirtualPath("/.mambo/memory/AGENTS.md")],
    )

    config = {"configurable": {"thread_id": "memory-demo-2"}}

    # 第一轮：Agent 会读取记忆并遵循编码规范
    result = agent.invoke(
        {"messages": [HumanMessage(
            "帮我创建一个 /app/calculator.py 文件，"
            "包含一个 add 函数和一个 multiply 函数"
        )]},
        config=config,
    )
    print("[第一轮]", result["messages"][-1].content[:500], "...")
    print()

    # 第二轮：Agent 记得之前的编码规范（memory 始终在 system prompt 中）
    result = agent.invoke(
        {"messages": [HumanMessage("再创建一个 /app/subtract.py，包含 subtract 函数")]},
        config=config,
    )
    print("[第二轮]", result["messages"][-1].content[:500], "...")
    print()

    # 读取更新后的 memory 文件（Agent 可能在交互中回写了新内容）
    print("📄 当前 memory 文件内容：")
    print("-" * 40)
    files = backend.download_files([VirtualPath("/.mambo/memory/AGENTS.md")])
    if files and files[0].content:
        print(files[0].content.decode("utf-8"))
    print("-" * 40)
    print()


def demo_custom_format_prompt():
    """演示自定义 memory 格式化函数。

    通过 format_prompt 参数可以自定义 memory 内容在 system prompt 中的
    展示方式。同样使用 SystemPromptInterceptor 截取实际注入效果。
    """
    print("=" * 60)
    print("Part 3：自定义 Memory 格式化（format_prompt）")
    print("=" * 60)

    from mambo_agents.middleware.memory import MamboMemoryMiddleware

    model = ChatDeepSeek(model="deepseek-v4-flash")

    memory_content = (
        "项目: MyApp\n"
        "语言: Python 3.12+\n"
        "规范: 类型提示 + PEP 8\n"
    )

    backend = StoreBackend(initial_files={
        "/.mambo/memory/AGENTS.md": memory_content,
    })

    def my_format_prompt(contents: dict[str, str]) -> str:
        """自定义格式化：将 memory 内容以简洁的格式注入 system prompt。"""
        parts = []
        for path, text in contents.items():
            parts.append(f"[MEMORY] 来源: {path}\n---\n{text}\n---")
        if not parts:
            return "(无记忆内容)"
        return "以下是已加载的 Agent 记忆：\n\n" + "\n\n".join(parts)

    interceptor = SystemPromptInterceptor()

    agent = create_mambo_agent(
        model,
        backend=backend,
        middleware=[
            MamboMemoryMiddleware(
                backend=backend,
                sources=[VirtualPath("/.mambo/memory/AGENTS.md")],
                format_prompt=my_format_prompt,
            ),
            interceptor,  # 排在 memory 后面，截取注入后的 system prompt
        ],
    )

    config = {"configurable": {"thread_id": "memory-demo-3"}}
    result = agent.invoke(
        {"messages": [HumanMessage("介绍一下本项目的编码规范")]},
        config=config,
    )

    # 展示自定义格式化后，Agent 运行时真实的 system prompt
    captured = interceptor.captured
    if captured:
        print("\n📋 自定义 format_prompt 注入后的 System Prompt 片段：")
        print("-" * 40)
        marker = "[MEMORY]"
        idx = captured.find(marker)
        if idx != -1:
            print("...(原始 system prompt 省略)...")
            print(captured[idx:idx + 500])
            print(f"...(memory 注入部分共 {len(captured) - idx} 字符)")
        else:
            print(captured[:600])

    print(f"\n💬 Agent 回复：")
    print(result["messages"][-1].content[:500], "...")
    print()


def main():
    demo_memory_injection()
    demo_memory_learning()
    demo_custom_format_prompt()


if __name__ == "__main__":
    main()
