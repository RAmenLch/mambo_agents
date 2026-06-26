"""诊断工具

提供独立于标准日志的输出通道，用于追踪 Overwrite 相关的运行时行为。
所有日志写入独立的文件，不污染 stdout/stderr。
"""

from mambo_agents.diagnostics.overwrite_tracker import (
    OverwriteTracker,
    get_tracker,
    is_enabled,
)

__all__ = ["OverwriteTracker", "get_tracker", "is_enabled"]
