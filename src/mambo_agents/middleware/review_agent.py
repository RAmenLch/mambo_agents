"""Agent-level security review — constructs a mini review-agent and monitors it.

Unlike the simple LLM call in :meth:`AutoSecurityReviewMiddleware._ai_review`,
this module creates a full LangChain agent that can optionally use read-only
tools (``ls``, ``read``, ``grep``, ``glob``) to inspect the workspace before
delivering a structured verdict via the **最终审核结果** tool.

The agent is constrained:
- MUST call ``最终审核结果`` within a limited number of steps.
- SHOULD avoid unnecessary tool calls to save time / tokens.
"""

from __future__ import annotations

import uuid
from typing import Any, Literal

from langchain.agents.factory import create_agent as _langchain_create_agent
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.callbacks.manager import CallbackManager
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Structured result
# ---------------------------------------------------------------------------

class FinalReviewResult(BaseModel):
    """Result returned by the review agent via the ``最终审核结果`` tool.

    Compatible with :class:`~mambo_agents.middleware.security_review.SecurityReviewResult`.
    """

    is_safe: bool = Field(
        description=(
            "Whether the tool call is safe to execute. "
            "True = safe (auto-approve), False = unsafe (escalate to human)."
        ),
    )
    reason: str = Field(
        description="Brief explanation for the safety decision (1-2 sentences).",
    )
    risk_level: Literal["low", "medium", "high", "critical"] = Field(
        default="low",
        description="Assessed risk level of the tool call.",
    )


# ---------------------------------------------------------------------------
# 最终审核结果 tool
# ---------------------------------------------------------------------------

_FINAL_TOOL_NAME = "submit_review_verdict"
"""Internal function name (must match ``^[a-zA-Z0-9_-]+$`` for OpenAI API).

The tool is presented to the review agent as the **最终审核结果** tool."""


def _build_final_result_tool() -> StructuredTool:
    """Build the ``submit_review_verdict`` (最终审核结果) tool."""

    def finalize(
        is_safe: bool,
        reason: str,
        risk_level: Literal["low", "medium", "high", "critical"] = "low",
    ) -> str:
        """Submit the final security review verdict (最终审核结果).

        You MUST call this tool exactly once before finishing — otherwise
        the review is considered incomplete and will be rejected.
        """
        return (
            f"审核结果已提交: is_safe={is_safe}, risk_level={risk_level}, "
            f"reason={reason}"
        )

    return StructuredTool.from_function(
        name=_FINAL_TOOL_NAME,
        description=(
            "最终审核结果 — 提交最终安全审核结果。你必须在完成审核后调用此工具，"
            "否则审核无效。调用后立即停止所有后续操作。"
            "Submit the final security review verdict — REQUIRED."
        ),
        func=finalize,
        args_schema=FinalReviewResult,
    )


# ---------------------------------------------------------------------------
# Default system prompt for the review agent
# ---------------------------------------------------------------------------

DEFAULT_REVIEW_AGENT_SYSTEM_PROMPT = """你是一名安全审核专家，负责审查 AI 编码 agent 的工具调用是否安全。

## 核心规则

1. **最终结论必须通过调用 `{final_tool_name}` 工具提交**，仅输出文字分析而不调用该工具视为审核失败。
2. **必须在 {max_steps} 步之内调用 `{final_tool_name}`**，否则审核失败。
3. 可以输出文字进行分析推理，但**分析完毕后必须立即调用 `{final_tool_name}`** 提交结论。
4. **尽量不调用文件系统工具** —— 仅在确实需要检查工作区文件才能判断安全性时才使用工具。

## 路径体系

本项目使用**虚拟路径**与**真实路径**双层体系，你需要理解二者的区别：

### 你使用的工具全部操作在虚拟路径上
- 你拥有的只读工具（ls、read、grep、glob）全部接受**虚拟路径**参数。
- 你审查的 tool call 中，`file_path` / `path` 等参数也是**虚拟路径**。
- 常见虚拟路径前缀：`{workspace_root}/`（项目工作区）{virtual_prefixes}

### 虚拟路径 ↔ 真实路径映射
- 虚拟路径 `{workspace_root}/` 映射到真实路径 `{real_root}/`。{path_mapping}

### 审查涉及真实路径的 tool call 时的注意事项
- 当你审查 `execute` 等使用真实路径的 tool call 时，如果需要检查其执行的脚本内容，应使用 **read 工具读取脚本的虚拟路径**（脚本在虚拟文件系统上可访问）。
- **脚本内容中出现的路径是真实路径**（因为脚本最终在真实系统上运行）。你需要对照上述映射关系判断这些真实路径是否落在项目工作区范围内。
- 对真实系统中的敏感路径（如 `/etc/`、`/root/`、`C:\\Windows\\`、`~/.ssh/` 等）保持高度警惕。

## 安全判断指南

### 通常 SAFE（is_safe=True）：
- 读取文件、列出目录、搜索/grep 文件内容
- 在用户项目工作区内写入/编辑文件
- 在项目目录创建新文件
- 非破坏性的版本控制操作（status、diff、log）
- 信息查询类系统操作

### 通常 UNSAFE（is_safe=False）：
- 删除文件或目录（尤其是工作区外的）
- 修改系统配置文件（/etc/*、Windows Registry 等）
- 执行安装/卸载软件的 shell 命令
- 修改系统服务或定时任务的命令
- 访问或导出凭据、API 密钥、机密信息的操作
- 修改 git 仓库
- 修改项目工作区外的文件
- 发送数据到外部服务器的网络操作
- 执行自定义脚本中包含UNSAFE的行为[适用于使用工具阅读脚本内容]

## 决策原则
- 存疑时倾向标记为 unsafe
- 仅影响用户自己项目文件且非破坏性的 → safe
- 可能影响系统稳定性或安全的 → unsafe

完成后立即调用 `{final_tool_name}` 提交结果。"""


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_review_agent(
    model: str | BaseChatModel,
    *,
    system_prompt: str | None = None,
    tools: list[StructuredTool] | None = None,
    middleware: list[AgentMiddleware] | None = None,
    max_steps: int = 5,
) -> CompiledStateGraph:
    """Create a mini agent whose sole purpose is security review.

    The agent is given a ``最终审核结果`` tool that it **must** call to
    deliver its structured verdict.  Optionally it receives read-only
    file-system tools so it can inspect the workspace before deciding.

    Parameters
    ----------
    model:
        Chat model for the review agent (should be cheap/fast).
    system_prompt:
        Custom system prompt.  ``{max_steps}`` and ``{final_tool_name}``
        are interpolated when ``None`` is passed.
    tools:
        Additional tools the review agent may use (e.g. read-only
        backend tools).  Defaults to empty list.
    middleware:
        Additional middleware for the review agent.
    max_steps:
        Maximum agent steps before forced termination (recursion_limit).

    Returns
    -------
    CompiledStateGraph
        A compiled LangGraph agent ready for ``invoke()`` / ``astream()``.
    """
    final_tool = _build_final_result_tool()
    all_tools = [final_tool] + (list(tools) if tools else [])

    if system_prompt is None:
        system_prompt = DEFAULT_REVIEW_AGENT_SYSTEM_PROMPT.format(
            max_steps=max_steps,
            final_tool_name=_FINAL_TOOL_NAME,
            workspace_root="/workspace",
            real_root="(未知)",
            virtual_prefixes="",
            path_mapping="",
        )

    return _langchain_create_agent(
        model=model,
        system_prompt=system_prompt,
        tools=all_tools,
        middleware=list(middleware) if middleware else [],
    ).with_config(
        {"recursion_limit": max_steps * 2 + 3}
    )


# ---------------------------------------------------------------------------
# Monitoring runner — early exit on 最终审核结果
# ---------------------------------------------------------------------------

class _ReviewIncompleteError(Exception):
    """Raised when the review agent finishes without calling 最终审核结果."""


def run_review_sync(
    agent: CompiledStateGraph,
    review_prompt: str,
) -> FinalReviewResult:
    """Run the review agent and extract the ``最终审核结果`` verdict.

    Uses ``stream()`` with an **isolated config** so the review agent
    does not inherit the parent graph's checkpointer or runtime context.
    Streaming allows **early exit** as soon as the verdict tool is
    called, saving tokens.

    Parameters
    ----------
    agent:
        A compiled review agent (from :func:`create_review_agent`).
    review_prompt:
        The review task description (tool call details to audit).

    Returns
    -------
    FinalReviewResult
        The structured verdict.

    Raises
    ------
    _ReviewIncompleteError
        If the agent finishes without ever calling ``最终审核结果``.
    """
    # Isolate from parent graph's config (checkpointer, thread_id,
    # callbacks) so review agent messages don't leak into the main
    # agent's message stream.
    #
    # IMPORTANT: Do NOT use "callbacks": [] — langgraph's ensure_config
    # (langgraph/_internal/_config.py) uses _is_not_empty() which
    # treats [] as empty (len([])==0), so the parent CallbackManager
    # (including StreamMessagesHandler) leaks through.  An empty
    # CallbackManager bypasses this filter since it's not a list/dict.
    _isolated_config: dict[str, Any] = {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "callbacks": CallbackManager(handlers=[]),
    }

    final_result: FinalReviewResult | None = None

    for event in agent.stream(
        {"messages": [HumanMessage(content=review_prompt)]},
        stream_mode=["updates"],
        config=_isolated_config,
    ):
        # "updates" mode yields either a bare dict {node: {channel: [...]}}
        # or a (mode, payload) tuple when called as a subgraph.
        if isinstance(event, tuple) and len(event) == 2:
            _, payload = event
        else:
            payload = event

        for node_output in (payload or {}).values():
            if not isinstance(node_output, dict):
                continue
            messages = node_output.get("messages") or []
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for tc in msg.tool_calls:
                        if tc.get("name") == _FINAL_TOOL_NAME:
                            args = tc.get("args", {})
                            try:
                                final_result = FinalReviewResult(**args)
                            except Exception:
                                final_result = _parse_legacy_result(args)
                            break
                    if final_result is not None:
                        break
            if final_result is not None:
                break
        if final_result is not None:
            break

    if final_result is None:
        raise _ReviewIncompleteError(
            "Review agent completed without calling submit_review_verdict. "
            "The audit is inconclusive — escalating to human review."
        )

    return final_result


async def run_review_async(
    agent: CompiledStateGraph,
    review_prompt: str,
) -> FinalReviewResult:
    """Run the review agent **asynchronously** — safe to call from an async context.

    Uses ``astream()`` instead of ``stream()`` so the asyncio event loop is
    **not** blocked while the review agent runs.  The event format and early-exit
    logic are identical to :func:`run_review_sync`.

    Parameters
    ----------
    agent:
        A compiled review agent (from :func:`create_review_agent`).
    review_prompt:
        The review task description (tool call details to audit).

    Returns
    -------
    FinalReviewResult
        The structured verdict.

    Raises
    ------
    _ReviewIncompleteError
        If the agent finishes without ever calling ``submit_review_verdict``.
    """
    _isolated_config: dict[str, Any] = {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "callbacks": CallbackManager(handlers=[]),
    }

    final_result: FinalReviewResult | None = None

    async for event in agent.astream(
        {"messages": [HumanMessage(content=review_prompt)]},
        stream_mode=["updates"],
        config=_isolated_config,
    ):
        if isinstance(event, tuple) and len(event) == 2:
            _, payload = event
        else:
            payload = event

        for node_output in (payload or {}).values():
            if not isinstance(node_output, dict):
                continue
            messages = node_output.get("messages") or []
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for tc in msg.tool_calls:
                        if tc.get("name") == _FINAL_TOOL_NAME:
                            args = tc.get("args", {})
                            try:
                                final_result = FinalReviewResult(**args)
                            except Exception:
                                final_result = _parse_legacy_result(args)
                            break
                    if final_result is not None:
                        break
            if final_result is not None:
                break
        if final_result is not None:
            break

    if final_result is None:
        raise _ReviewIncompleteError(
            "Review agent completed without calling submit_review_verdict. "
            "The audit is inconclusive — escalating to human review."
        )

    return final_result


def _parse_legacy_result(args: dict[str, Any]) -> FinalReviewResult:
    """Fallback parser for malformed tool call args."""
    return FinalReviewResult(
        is_safe=bool(args.get("is_safe", False)),
        reason=str(args.get("reason", "Unable to parse review result")),
        risk_level=args.get("risk_level", "high"),
    )
