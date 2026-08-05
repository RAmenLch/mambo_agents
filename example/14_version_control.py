"""
============================================================
Mambo Agents - 示例 14：版本控制高级 API（Version Control）
============================================================

演示版本控制中间件的完整用法：
  1. 通过 middleware 直接挂载 VersionControlMiddleware（自建 VersionStore）
  2. 通过 stream_mode="custom" 接收 BackupEvent 备份事件
  3. VersionStore 查询 API：list_snapshots / get_changed_files /
     get_all_changed_files / get_latest_snapshot / get_latest_changed_files
  4. restore_files() 手动回滚到指定 checkpoint

注意：version_control 默认只备份白名单（whitelist_folders）内的文件。
通过 create_mambo_agent(version_control=True) 的便捷形式会自动白名单
整个 workspace_root；本示例演示显式配置白名单的完整形式。

运行前请先配置 .env 文件中的 API Key。
运行方式：
  python example/14_version_control.py
============================================================
"""

import os
import tempfile

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.store.memory import InMemoryStore

from mambo_agents import create_mambo_agent
from mambo_agents.backends.local import LocalBackend
from mambo_agents.backends.schemas import VirtualPath
from mambo_agents.middleware.version_control import (
    BackupEvent,
    VersionControlMiddleware,
    VersionStore,
)
from deepseek_chat_model import ChatDeepSeek

load_dotenv()

THREAD_ID = "vc-demo"


def main():
    print("=" * 60)
    print("示例：版本控制完整流程 - 备份事件 / 查询 / 回滚")
    print("=" * 60)

    model = ChatDeepSeek(model="deepseek-v4-flash")

    with tempfile.TemporaryDirectory() as tmpdir:
        backend = LocalBackend(root_dir=tmpdir)

        # ── 1. 自建 VersionStore + VersionControlMiddleware ──
        store = VersionStore(store=InMemoryStore())
        vc_middleware = VersionControlMiddleware(
            store=store,
            backend=backend,
            whitelist_folders=[VirtualPath("/workspace")],  # 只监控 workspace 下的文件
        )

        agent = create_mambo_agent(
            model,
            backend=backend,
            middleware=[vc_middleware],
        )

        config = {"configurable": {"thread_id": THREAD_ID}}

        # ── 2. 第一轮：创建文件 v1 ──
        # 注意：备份发生在"修改已有文件"之前（记录修改前的内容）。
        # 首次创建新文件时无旧内容可备份，因此不会产生 backup 事件。
        print("\n--- 第 1 轮：创建 /app.py (v1) ---")
        backup_events: list[BackupEvent] = []

        for mode, chunk in agent.stream(
            {"messages": [HumanMessage("创建 /workspace/app.py，内容为 print('v1')")]},
            config,
            stream_mode=["updates", "custom"],
        ):
            if mode == "custom":
                event = BackupEvent(**chunk)
                backup_events.append(event)
                print(f"  [backup] ckpt={event.checkpoint_id} file={event.file_path} sha={event.sha256[:8]}")

        print(f"  本轮 backup 事件数: {len(backup_events)}（创建新文件无备份，符合预期）")

        # ── 3. 第二轮：修改为 v2 ──
        print("\n--- 第 2 轮：修改 /app.py (v2) ---")
        backup_events.clear()

        for mode, chunk in agent.stream(
            {"messages": [HumanMessage("把 /workspace/app.py 的内容改为 print('v2')")]},
            config,
            stream_mode=["updates", "custom"],
        ):
            if mode == "custom":
                event = BackupEvent(**chunk)
                backup_events.append(event)
                print(f"  [backup] ckpt={event.checkpoint_id} file={event.file_path} sha={event.sha256[:8]}")

        cp_v2_write = backup_events[-1].checkpoint_id if backup_events else None
        print(f"  checkpoint（备份的是修改前内容 v1）: {cp_v2_write}")

        # ── 4. 第三轮：修改为 v3 ──
        print("\n--- 第 3 轮：修改 /app.py (v3) ---")
        backup_events.clear()

        for mode, chunk in agent.stream(
            {"messages": [HumanMessage("把 /workspace/app.py 的内容改为 print('v3')")]},
            config,
            stream_mode=["updates", "custom"],
        ):
            if mode == "custom":
                event = BackupEvent(**chunk)
                backup_events.append(event)
                print(f"  [backup] ckpt={event.checkpoint_id} file={event.file_path} sha={event.sha256[:8]}")

        cp_v3_write = backup_events[-1].checkpoint_id if backup_events else None
        print(f"  checkpoint（备份的是修改前内容 v2）: {cp_v3_write}")

        app_path = os.path.join(tmpdir, "app.py")

        def show_disk_state(label: str):
            with open(app_path) as f:
                print(f"  {label}: {f.read().strip()!r}")

        # ── 5. VersionStore 查询 API ──
        print("\n--- VersionStore 查询 API ---")
        snapshots = store.list_snapshots(THREAD_ID)
        print(f"  list_snapshots: 共 {len(snapshots)} 个快照")
        for snap in snapshots:
            print(f"    - ckpt={snap.checkpoint_id[:16]}... ts={snap.timestamp}")

        all_files = store.get_all_changed_files(THREAD_ID)
        print(f"  get_all_changed_files: {sorted(all_files)}")

        latest_files = store.get_latest_changed_files(THREAD_ID)
        print(f"  get_latest_changed_files: {latest_files}")

        latest_snap = store.get_latest_snapshot(THREAD_ID)
        if latest_snap:
            print(f"  get_latest_snapshot: ckpt={latest_snap.checkpoint_id[:16]}...")

        changed_v3 = store.get_changed_files(THREAD_ID, cp_v3_write)
        print(f"  get_changed_files(v3 写入轮): {changed_v3}")

        # ── 6. restore_files 回滚 ──
        # 语义：checkpoint 记录的是"该次修改前"的文件内容，
        # 因此回滚到第 3 轮 checkpoint → 得到 v2；回滚到第 2 轮 checkpoint → 得到 v1。
        print("\n--- restore_files 回滚 ---")
        show_disk_state("回滚前")

        # 回滚到第 3 轮 checkpoint（内容恢复为 v2）
        results = vc_middleware.restore_files(THREAD_ID, cp_v3_write, all=True)
        for r in results:
            print(f"  restore: {r.path} → success={r.success} {r.message or ''}")
        show_disk_state("回滚到第 3 轮 checkpoint 后")

        # 精确回滚单个文件到第 2 轮 checkpoint（内容恢复为 v1）
        results = vc_middleware.restore_files(
            THREAD_ID, cp_v2_write, files=["/workspace/app.py"]
        )
        for r in results:
            print(f"  restore: {r.path} → success={r.success} {r.message or ''}")
        show_disk_state("回滚到第 2 轮 checkpoint 后")

        print("\n提示：回滚是手动触发的（restore_files），不会自动发生。")


if __name__ == "__main__":
    main()
