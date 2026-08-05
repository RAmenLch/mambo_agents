"""
============================================================
Mambo Agents - 示例 11：混合工作区（HybridWorkspaceBackend）
============================================================

演示 HybridWorkspaceBackend：
  - 一个真实后端（LocalBackend）+ N 个虚拟工作区（StoreBackend）
  - 虚拟工作区统一挂在 /.mambo/ 前缀下
  - copy 工具跨后端复制文件（虚拟 ↔ 真实，或虚拟 ↔ 虚拟）

路径路由规则：
  - /.mambo/skills/xxx  → "skills" 虚拟工作区（前缀剥离，传 /xxx）
  - /.mambo/xxx         → 默认 StoreBackend（前缀剥离，传 /xxx）
  - /workspace/...      → 真实后端（路径重写后落到本地磁盘）
  - 其他路径（如 /、/etc）被拒绝

运行前请先配置 .env 文件中的 API Key。
运行方式：
  python example/11_hybrid_workspace.py
============================================================
"""

import os
import tempfile

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.store.memory import InMemoryStore

from mambo_agents import HybridWorkspaceBackend, StoreBackend, create_mambo_agent
from mambo_agents.backends.local import LocalBackend
from mambo_agents.backends.schemas import VirtualPath
from deepseek_chat_model import ChatDeepSeek

load_dotenv()


def main_basic_routing():
    """演示基本路径路由：真实磁盘 + 两个虚拟工作区。"""
    print("=" * 60)
    print("示例 1：基本路由 - 真实磁盘 + /.mambo/ 虚拟工作区")
    print("=" * 60)

    model = ChatDeepSeek(model="deepseek-v4-flash")

    with tempfile.TemporaryDirectory() as tmpdir:
        # 共享 store：保证外部验证与 Agent 运行时读写同一份虚拟文件。
        # 注意：虚拟工作区路由重写后路径带 /workspace 前缀（StoreBackend
        # 的 workspace_root 固定为 /workspace），因此 initial_files 的
        # key 也要用 /workspace/ 前缀才能匹配。
        shared_store = InMemoryStore()

        # 保留虚拟后端引用，便于运行后验证文件落点
        skills_backend = StoreBackend(
            store=shared_store,
            initial_files={
                "/workspace/python.md": "# Python 技巧\n\n- 使用 f-string\n- 遵循 PEP 8\n",
            },
        )
        cache_backend = StoreBackend(store=shared_store)

        backend = HybridWorkspaceBackend(
            real_backend=LocalBackend(root_dir=tmpdir),
            virtual_workspaces={
                # 命名工作区 → /.mambo/skills/
                "skills": skills_backend,
                # 空的工作区 → /.mambo/cache/
                "cache": cache_backend,
            },
            store=shared_store,  # 默认 /.mambo/ 工作区也共享同一 store
        )

        agent = create_mambo_agent(model, backend=backend)

        result = agent.invoke({
            "messages": [HumanMessage(
                "做三件事：\n"
                "1. 在 /workspace/ 下创建一个 report.md，内容为 'Real disk file'\n"
                "2. 在 /.mambo/skills/ 下创建一个 notes.md，内容为 'Virtual workspace file'\n"
                "3. 用 ls 分别列出 /workspace/ 和 /.mambo/skills/ 的内容"
            )]
        }, config={"configurable": {"thread_id": "session-1"}})

        print(result["messages"][-1].content[:600], "...")
        print()

        # ── 验证 ──
        # 真实文件应出现在本地磁盘临时目录中
        disk_path = os.path.join(tmpdir, "report.md")
        print(f"[验证] 真实文件落盘: {os.path.exists(disk_path)} → {disk_path}")
        if os.path.exists(disk_path):
            with open(disk_path) as f:
                print(f"        内容: {f.read().strip()}")

        # 虚拟文件存在虚拟 StoreBackend 中（通过 download_files 读取）
        files = skills_backend.download_files([VirtualPath("/workspace/notes.md")])
        if files and files[0].error is None:
            print(f"[验证] 虚拟文件存在于 skills 工作区: {files[0].content.decode()!r}")
        else:
            print(f"[验证] 虚拟文件未找到: {files[0].error}")
        print()


def main_copy_across_backends():
    """演示 copy 工具：虚拟 ↔ 真实跨后端复制。"""
    print("=" * 60)
    print("示例 2：copy 跨后端复制（虚拟 → 真实）")
    print("=" * 60)

    model = ChatDeepSeek(model="deepseek-v4-flash")

    with tempfile.TemporaryDirectory() as tmpdir:
        shared_store = InMemoryStore()
        cache_backend = StoreBackend(
            store=shared_store,
            initial_files={
                "/workspace/config.json": '{"port": 8080}',
            },
        )

        backend = HybridWorkspaceBackend(
            real_backend=LocalBackend(root_dir=tmpdir),
            virtual_workspaces={
                "cache": cache_backend,
            },
            store=shared_store,
        )

        agent = create_mambo_agent(model, backend=backend)

        result = agent.invoke({
            "messages": [HumanMessage(
                "把 /.mambo/cache/config.json 复制到 /workspace/config.json，"
                "然后在 /workspace/config.json 里把 port 改成 9090"
            )]
        }, config={"configurable": {"thread_id": "session-2"}})

        print(result["messages"][-1].content[:600], "...")
        print()

        # ── 验证 ──
        disk_path = os.path.join(tmpdir, "config.json")
        if os.path.exists(disk_path):
            with open(disk_path) as f:
                print(f"[验证] 磁盘上的 config.json: {f.read().strip()}")
        else:
            print("[验证] 磁盘上的 config.json 不存在")
        print()


def main_override_default_mambo():
    """演示用 "." 键覆盖默认 /.mambo/ 工作区。"""
    print("=" * 60)
    print("示例 3：覆盖默认 /.mambo/ 工作区（key='.'）")
    print("=" * 60)

    model = ChatDeepSeek(model="deepseek-v4-flash")

    with tempfile.TemporaryDirectory() as tmpdir:
        shared_store = InMemoryStore()
        default_backend = StoreBackend(
            store=shared_store,
            initial_files={
                "/workspace/config.yml": "debug: true\n",
            },
        )

        backend = HybridWorkspaceBackend(
            real_backend=LocalBackend(root_dir=tmpdir),
            virtual_workspaces={
                ".": default_backend,
            },
            store=shared_store,
        )

        agent = create_mambo_agent(model, backend=backend)

        result = agent.invoke({
            "messages": [HumanMessage(
                "读取 /.mambo/config.yml 的内容，然后在 /.mambo/ 下新建一个 "
                "scratch.txt 记录你看到的配置"
            )]
        }, config={"configurable": {"thread_id": "session-3"}})

        print(result["messages"][-1].content[:600], "...")
        print()

        # ── 验证：默认工作区中的文件 ──
        files = default_backend.download_files([VirtualPath("/workspace/scratch.txt")])
        if files and files[0].error is None:
            print(f"[验证] /.mambo/scratch.txt 存在: {files[0].content.decode()[:100]!r}")
        else:
            print(f"[验证] /.mambo/scratch.txt 未找到: {files[0].error}")
        print()


if __name__ == "__main__":
    main_basic_routing()
    main_copy_across_backends()
    main_override_default_mambo()
