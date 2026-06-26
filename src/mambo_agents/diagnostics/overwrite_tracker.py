"""Overwrite 行为追踪器。

通过环境变量 ``MAMBO_DIAG_OVERWRITE=1`` 启用。
日志默认写入当前工作目录的 ``mambo_overwrite_diag.log``，
可通过 ``MAMBO_DIAG_LOG=/path/to/file.log`` 自定义。

日志格式（每行一条 JSON，便于后期 grep/jq 分析）::

    {
        "ts": "2026-06-26T15:10:00.123456",
        "thread_id": "abc123",
        "checkpoint_ns": "",
        "event": "overwrite_produced",
        "source": "patch_tool_calls.before_agent",
        "details": {...}
    }

事件类型:
- ``overwrite_produced`` — 中间件返回了 Overwrite
- ``state_messages_type_mismatch`` — state["messages"] 类型异常（非 list/Overwrite）
- ``request_messages_is_overwrite`` — ModelRequest.messages 为 Overwrite 对象（崩溃前兆）
- ``subagent_config_inherited`` — 子代理继承了父代理的 checkpoint 配置
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any

# ---------------------------------------------------------------------------
# 模块级 logger（独立文件，不传播到 root logger）
# ---------------------------------------------------------------------------

_logger_name = "mambo_agents.diag.overwrite"
_logger = logging.getLogger(_logger_name)
_logger.propagate = False  # 不污染 root logger / stdout
_logger.setLevel(logging.DEBUG)

_initialized = False
_init_lock = threading.Lock()


def _ensure_initialized() -> None:
    """延迟初始化 handler（避免在 import 阶段创建空文件）"""
    global _initialized
    if _initialized:
        return
    with _init_lock:
        if _initialized:
            return
        log_path = os.environ.get(
            "MAMBO_DIAG_LOG",
            os.path.join(os.getcwd(), "mambo_overwrite_diag.log"),
        )
        handler = logging.FileHandler(log_path, encoding="utf-8", delay=True)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter("%(message)s"))
        _logger.addHandler(handler)
        _initialized = True


# ---------------------------------------------------------------------------
# 公共 API
# ---------------------------------------------------------------------------


def is_enabled() -> bool:
    """检查诊断是否启用。

    默认开启；设置 MAMBO_DIAG_OVERWRITE=0 可关闭。
    """
    return os.environ.get("MAMBO_DIAG_OVERWRITE", "").strip() not in ("0", "false", "no", "off")


def get_tracker() -> OverwriteTracker:
    """获取全局 OverwriteTracker 单例。"""
    return _tracker


class OverwriteTracker:
    """追踪 Overwrite 的生成、消费和泄漏。

    典型用法是在中间件中惰性调用，不影响正常执行路径::

        from mambo_agents.diagnostics import get_tracker, is_enabled

        if is_enabled():
            get_tracker().log_overwrite_produced(
                source="reorder_tool_messages.before_model",
                runtime=runtime,
                extra={"messages_count": 42},
            )
    """

    def _log(self, event: str, thread_id: str, checkpoint_ns: str, source: str, **extra: Any) -> None:
        _ensure_initialized()
        record: dict[str, Any] = {
            "ts": _now(),
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "event": event,
            "source": source,
            "details": extra,
        }
        _logger.info(json.dumps(record, ensure_ascii=False, default=str))

    # ---- helpers ----

    @staticmethod
    def extract_ids(runtime: Any) -> tuple[str, str]:
        """从 runtime 对象中提取 (thread_id, checkpoint_ns)。"""
        try:
            configurable = runtime.config.get("configurable", {})
            thread_id = str(configurable.get("thread_id", "?"))
            checkpoint_ns = str(configurable.get("checkpoint_ns", ""))
            return thread_id, checkpoint_ns
        except Exception:
            return "?", "?"

    @staticmethod
    def describe_messages(value: Any) -> dict[str, Any]:
        """获取 value 的类型和概览信息，用于诊断日志。"""
        info: dict[str, Any] = {"type": type(value).__name__}
        if hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
            info["len"] = len(value)
        if type(value).__name__ == "Overwrite":
            try:
                inner = value.value
                info["inner_type"] = type(inner).__name__
                if hasattr(inner, "__len__") and not isinstance(inner, (str, bytes)):
                    info["inner_len"] = len(inner)
            except Exception:
                info["inner_type"] = "?"
        return info

    # ---- 事件记录方法 ----

    def log_overwrite_produced(
        self,
        source: str,
        runtime: Any,
        **extra: Any,
    ) -> None:
        """记录中间件返回 Overwrite 的事件。"""
        thread_id, checkpoint_ns = self.extract_ids(runtime)
        self._log("overwrite_produced", thread_id, checkpoint_ns, source, **extra)

    def log_state_messages_type_mismatch(
        self,
        source: str,
        runtime: Any,
        value: Any,
    ) -> None:
        """记录 state["messages"] 类型异常。"""
        thread_id, checkpoint_ns = self.extract_ids(runtime)
        self._log(
            "state_messages_type_mismatch",
            thread_id,
            checkpoint_ns,
            source,
            messages_info=self.describe_messages(value),
        )

    def log_request_messages_is_overwrite(
        self,
        source: str,
        runtime: Any,
        value: Any,
    ) -> None:
        """记录 ModelRequest.messages 是 Overwrite 对象（崩溃前兆）。"""
        thread_id, checkpoint_ns = self.extract_ids(runtime)
        self._log(
            "request_messages_is_overwrite",
            thread_id,
            checkpoint_ns,
            source,
            messages_info=self.describe_messages(value),
        )

    def log_subagent_config_inherited(
        self,
        source: str,
        runtime: Any,
        subagent_type: str,
    ) -> None:
        """记录子代理继承了父代理的 checkpoint 配置。"""
        thread_id, checkpoint_ns = self.extract_ids(runtime)
        self._log(
            "subagent_config_inherited",
            thread_id,
            checkpoint_ns,
            source,
            subagent_type=subagent_type,
            inherited_checkpoint_ns=checkpoint_ns,
            inherited_thread_id=thread_id,
        )


# ---------------------------------------------------------------------------
# 全局单例
# ---------------------------------------------------------------------------

_tracker = OverwriteTracker()


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

def _now() -> str:
    """ISO 8601 毫秒时间戳。"""
    return time.strftime("%Y-%m-%dT%H:%M:%S.", time.localtime()) + f"{time.time() % 1:.6f}"[2:]
