"""
============================================================
Mambo Agents - 示例 13：技能包（Skills）
============================================================

演示 SkillsMiddleware - 渐进式披露的技能系统：
技能只在 Agent 需要时才加载进 prompt，避免上下文膨胀。

技能文件结构（放在 backend 上）：
  /skills/user/web-research/
  ├── SKILL.md          # 必需：YAML frontmatter + markdown 指令
  └── helper.py         # 可选：辅助文件

运行前请先配置 .env 文件中的 API Key。
运行方式：
  python example/13_skills.py
============================================================
"""

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from mambo_agents import StoreBackend, create_mambo_agent
from deepseek_chat_model import ChatDeepSeek

load_dotenv()

# ---------------------------------------------------------------------------
# 技能内容定义（写入 StoreBackend 虚拟文件系统）
# ---------------------------------------------------------------------------

SKILLS_FILES = {
    # 技能 1：web-research —— 裸路径来源，标签由目录名推导（"Web Research"）
    "/skills/user/web-research/SKILL.md": """---
name: web-research
description: 结构化网络调研方法论，用于系统性收集和分析信息
license: MIT
---
# Web Research Skill

## Steps
1. 明确调研目标与关键词
2. 分步骤搜索并记录来源
3. 交叉验证关键结论
4. 输出带来源引用的结构化报告

## Output Format
- 使用 Markdown 标题分层
- 每个结论后附 `[来源: ...]`
""",
    "/skills/user/web-research/helper.py": (
        "# 可选辅助文件，技能加载时随 SKILL.md 一并提供\n"
        "def normalize_query(q: str) -> str:\n"
        "    return q.strip().lower()\n"
    ),

    # 技能 2：code-review —— 通过 tuple 来源自定义标签
    "/skills/project/code-review/SKILL.md": """---
name: code-review
description: Python 代码审查清单，检查 bug、安全与风格问题
---
# Code Review Skill

## Checklist
1. 检查未处理异常与资源泄漏
2. 检查 SQL 注入 / 命令注入风险
3. 检查类型提示与文档字符串
4. 检查命名与 PEP 8 风格
""",
}


def demo_basic_skills():
    """演示裸路径来源 + 派生标签。"""
    print("=" * 60)
    print("示例 1：基本技能 - 裸路径来源（/skills/user/）")
    print("=" * 60)

    model = ChatDeepSeek(model="deepseek-v4-flash")

    backend = StoreBackend(initial_files=SKILLS_FILES)

    agent = create_mambo_agent(
        model,
        backend=backend,
        skills=["/skills/user/"],
    )

    result = agent.invoke({
        "messages": [HumanMessage(
            "我要调研 Python 3.12 的新特性，请按照你掌握的调研技能"
            "给出一个调研计划"
        )]
    }, config={"configurable": {"thread_id": "session-1"}})

    print(result["messages"][-1].content[:600], "...")
    print()


def demo_tuple_source():
    """演示 tuple 来源：自定义标签（/skills/project/, "Project Skills"）。"""
    print("=" * 60)
    print("示例 2：tuple 来源 - 自定义技能标签")
    print("=" * 60)

    model = ChatDeepSeek(model="deepseek-v4-flash")

    backend = StoreBackend(initial_files=SKILLS_FILES)

    agent = create_mambo_agent(
        model,
        backend=backend,
        skills=[
            ("/skills/project/", "Project Skills"),
        ],
    )

    result = agent.invoke({
        "messages": [HumanMessage(
            "帮我审查下面这段代码，找出安全问题：\n"
            "def query(user_input):\n"
            "    return db.execute(f'SELECT * FROM users WHERE name = {user_input}')"
        )]
    }, config={"configurable": {"thread_id": "session-2"}})

    print(result["messages"][-1].content[:600], "...")
    print()


def demo_multiple_sources():
    """演示多来源加载：用户级 + 项目级技能同时挂载。"""
    print("=" * 60)
    print("示例 3：多来源加载 - 用户级 + 项目级技能")
    print("=" * 60)

    model = ChatDeepSeek(model="deepseek-v4-flash")

    backend = StoreBackend(initial_files=SKILLS_FILES)

    agent = create_mambo_agent(
        model,
        backend=backend,
        skills=[
            "/skills/user/",
            ("/skills/project/", "Project Skills"),
        ],
    )

    # 两个技能同时可用，Agent 会根据任务自行选择合适的技能
    result = agent.invoke({
        "messages": [HumanMessage(
            "两个任务：\n"
            "1. 给出调研 Python 异步编程的步骤计划\n"
            "2. 审查这段代码是否有注入风险：\n"
            "   os.system(f'rm -rf {path}')"
        )]
    }, config={"configurable": {"thread_id": "session-3"}})

    print(result["messages"][-1].content[:800], "...")
    print()


if __name__ == "__main__":
    demo_basic_skills()
    demo_tuple_source()
    demo_multiple_sources()
