import base64
from typing import Any, Dict, List, Optional, Tuple, Union

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
)
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.utils import from_env, secret_from_env
from langchain_openai import ChatOpenAI
from pydantic import Field, SecretStr, ConfigDict


class ChatDeepSeek(ChatOpenAI):
    """
    自定义 DeepSeek Chat 模型（支持 V4）。
    功能说明：
    1. 思考模式(Reasoning)下工具调用(Tool Call)时，回传 reasoning_content。
    2. 支持 DeepSeek V4 的 thinking 参数和 reasoning_effort 参数。
       - thinking: {"type": "enabled"/"disabled"} 控制思考模式开关
       - reasoning_effort: "high"/"max" 控制思考强度
    3. 不对 tool/assistant 消息的 content 做扁平化：DeepSeek 现版本已接受 list content
       （含纯文本块与 image_url 图片块，实测通过），扁平化会剥离 tool 消息中的
       image_url 块，导致 vision 模型（deepseek-v4-flash-vision-exp）无法识图。
    4. 规整媒体块：把 langchain 包装出的嵌套 file 块（{"type":"file","file":{...}}）
       拍平为 DeepSeek 只认的扁平 file_id / file_data 形态；video / audio 等不支持的
       媒体块降级为文本占位。配合 mambo_agents 的 file_uploader 钩子使用 ——
       read 工具消息里只保留 {"type": "file", "file_id": ...} 轻量引用块。
    """

    # 默认配置
    openai_api_base: str | None = Field(
        default_factory=from_env("DEEPSEEK_API_BASE", default="https://api.deepseek.com"),
        alias="base_url"
    )
    openai_api_key: SecretStr | None = Field(
        default_factory=secret_from_env("DEEPSEEK_API_KEY", default=None),
        alias="api_key"
    )
    model_name: str = Field(default="deepseek-v4-pro", alias="model")

    # DeepSeek V4 思考模式参数
    thinking: Optional[Dict[str, str]] = Field(
        default=None,
        description='DeepSeek V4 思考模式开关，格式：{"type": "enabled"} 或 {"type": "disabled"}。'
    )
    reasoning_effort: Optional[str] = Field(
        default=None,
        description='DeepSeek V4 推理强度："high" 或 "max"。'
    )

    model_config = ConfigDict(populate_by_name=True)

    @property
    def _llm_type(self) -> str:
        return "deepseek-chat-custom"

    def _get_request_payload(
            self,
            input_: List[BaseMessage],
            *args,
            **kwargs
    ) -> Dict:
        """
        重写构建请求体的方法。
        """
        # 1. 先降级 DeepSeek 不支持、且 langchain 转换阶段会直接抛错的内容块（如 video），
        #    再获取 OpenAI 格式的标准 payload。索引与消息类型保持不变，故下方仍可用 input_[i]。
        messages = self._sanitize_messages(input_)
        payload = super()._get_request_payload(messages, *args, **kwargs)

        # 2. 注入 DeepSeek V4 思考模式参数
        # reasoning_effort 是 OpenAI SDK 支持的标准参数（o1/o3 系列也使用）
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        # thinking 是 DeepSeek 扩展参数，必须通过 extra_body 传递，不能直接放在请求体中
        if self.thinking is not None:
            extra_body = payload.get("extra_body", {}) or {}
            extra_body.update({"thinking": self.thinking})
            payload["extra_body"] = extra_body

        # 3. 思考模式与 tool_choice 互斥处理
        # deepseek 思考模式下，tool_choice 指定具体函数名
        # （如 {"type": "function", "function": {"name": "SecurityReviewResult"}}）
        # 会触发 400 报错 "Thinking mode does not support this tool_choice"。
        # 此处在请求体层面自动关闭思考，保证 structured output / 安全审核等场景正常工作。
        tool_choice = payload.get("tool_choice")
        if tool_choice and not isinstance(tool_choice, str):
            extra_body = payload.get("extra_body", {}) or {}
            thinking_cfg = extra_body.get("thinking")
            # thinking_cfg 为 None 时 deepseek 默认开启思考，同样需要显式关闭
            is_thinking_on = thinking_cfg is None or thinking_cfg.get("type") == "enabled"
            if is_thinking_on:
                extra_body["thinking"] = {"type": "disabled"}
                payload["extra_body"] = extra_body

        # 4. 遍历 payload 中的消息进行修复
        for i, payload_msg in enumerate(payload["messages"]):

            # --- 修复: 回传 reasoning_content (解决 400 错误) ---
            # 找到对应的 LangChain 原始消息
            if i < len(input_):
                lc_msg = input_[i]
                if isinstance(lc_msg, AIMessage):
                    # 检查 additional_kwargs 中是否有 reasoning_content
                    reasoning = lc_msg.additional_kwargs.get("reasoning_content")
                    if reasoning:
                        # 显式将 reasoning_content 加回发送给 API 的字典中
                        payload_msg["reasoning_content"] = reasoning

        # 5. 规整媒体块：嵌套 file 块 -> 扁平 file_id / file_data；不支持的媒体块 -> 文本占位
        self._normalize_media_blocks(payload)

        return payload

    # ------------------------------------------------------------------
    # 媒体块规整（配合 mambo_agents 的 file_uploader 钩子）
    #
    # read 工具产出 {"type": "file", "file_id": ...} 轻量引用块后，langchain 的
    # _format_message_content 会把它包装成嵌套 {"type": "file", "file": {...}}；
    # 而 DeepSeek 只认**扁平**的 file_id / file_data。此处把嵌套块拍平，
    # 并把 DeepSeek 不支持的媒体块（非图片 file、音频等）降级为文本占位，
    # 避免整个请求失败。注意：只能替换 payload 列表中的元素，不要原地修改块 dict，
    # 否则会污染原始消息对象 / checkpoint。
    # ------------------------------------------------------------------

    def _sanitize_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """把 DeepSeek 无法处理的内容块降级为文本占位，返回（必要时新建的）消息列表。

        langchain 的 ``_format_message_content`` 对 ``video`` 块会直接抛 ``ValueError``；
        ``text-plain`` 等其它不支持类型同样在这里统一降级。
        """
        raising_types = {"video", "text-plain"}
        out: List[BaseMessage] = []
        changed = False
        for msg in messages or []:
            content = getattr(msg, "content", None)
            if not isinstance(content, list):
                out.append(msg)
                continue
            new_content = []
            modified = False
            for block in content:
                if isinstance(block, dict) and block.get("type") in raising_types:
                    mime = block.get("mime_type") or "未知类型"
                    new_content.append(
                        {"type": "text", "text": f"[附件文件（{mime}）不受当前模型支持，已省略]"}
                    )
                    modified = True
                else:
                    new_content.append(block)
            if modified:
                changed = True
                out.append(msg.model_copy(update={"content": new_content}))
            else:
                out.append(msg)
        return out if changed else messages

    @staticmethod
    def _iter_content_lists(payload: Dict[str, Any]):
        """产出 payload 中每条消息的 content 列表（可原地替换其元素）。"""
        for msg in (payload or {}).get("messages", []) or []:
            if isinstance(msg, dict) and isinstance(msg.get("content"), list):
                yield msg["content"]

    def _normalize_media_blocks(self, payload: Dict[str, Any]) -> None:
        """把 payload 中的媒体块规整为 DeepSeek 可接受的形态。"""
        for content in self._iter_content_lists(payload):
            for idx, block in enumerate(content):
                if not isinstance(block, dict):
                    continue
                new_block = self._rewrite_media_block(block)
                if new_block is not None:
                    content[idx] = new_block

    def _rewrite_media_block(self, block: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """返回替换后的块；None 表示保持原样。"""
        block_type = block.get("type")

        if block_type == "input_audio":
            fmt = (block.get("input_audio") or {}).get("format") or "音频"
            return {"type": "text", "text": f"[附件音频（{fmt}）不受当前模型支持，已省略]"}

        if block_type != "file":
            return None

        # 已是扁平 file_id：保持（mambo_agents 的 file_uploader 产出的就是这种块）
        if block.get("file_id"):
            return None
        file_data = block.get("file_data")
        filename = block.get("filename")
        if file_data is None:
            inner = block.get("file")
            if not isinstance(inner, dict):
                return None
            if inner.get("file_id"):
                return {"type": "file", "file_id": inner["file_id"]}
            file_data = inner.get("file_data")
            filename = filename or inner.get("filename")
        if file_data is None:
            return None
        return self._flat_file_block(file_data, filename)

    def _flat_file_block(
        self, file_data: str, filename: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """扁平 file_data：图片保持内联；非图片转文本占位。"""
        if self._decode_image_data_url(file_data) is None:
            mime = self._mime_from_data_url(file_data) or "未知类型"
            return {"type": "text", "text": f"[附件文件（{mime}）不受当前模型支持，已省略]"}
        flat: Dict[str, Any] = {"type": "file", "file_data": file_data}
        if filename:
            flat["filename"] = filename
        return flat

    @staticmethod
    def _decode_image_data_url(url: Any) -> Optional[Tuple[str, bytes]]:
        """将 ``data:image/...;base64,...`` 解码为 ``(mime, raw_bytes)``，非图片返回 None。"""
        if not isinstance(url, str) or not url.startswith("data:image/"):
            return None
        try:
            header, b64 = url.split(",", 1)
        except ValueError:
            return None
        if ";base64" not in header:
            return None
        mime = header[len("data:"):].split(";", 1)[0]
        try:
            raw = base64.b64decode(b64)
        except Exception:
            return None
        return mime, raw

    @staticmethod
    def _mime_from_data_url(data_url: Any) -> Optional[str]:
        """从 ``data:<mime>;base64,...`` 提取 MIME 类型。"""
        if not isinstance(data_url, str) or not data_url.startswith("data:"):
            return None
        header = data_url.split(",", 1)[0]
        return header[len("data:"):].split(";", 1)[0] or None

    def _create_chat_result(
            self, response: Union[Dict, Any], generation_info: Optional[Dict] = None
    ) -> ChatResult:
        """
        处理非流式响应：从 API 响应中提取 reasoning_content 并存入 additional_kwargs。
        """
        rtn = super()._create_chat_result(response, generation_info)

        # 尝试从 response 中提取 reasoning_content
        choices = getattr(response, "choices", [])
        if not choices and isinstance(response, dict):
            choices = response.get("choices", [])

        if choices:
            choice = choices[0]
            message = getattr(choice, "message", None) or choice.get("message", {})

            # 获取 reasoning_content
            reasoning_content = None
            if hasattr(message, "reasoning_content"):
                reasoning_content = message.reasoning_content
            elif isinstance(message, dict):
                reasoning_content = message.get("reasoning_content")

            # 如果存在，存入 additional_kwargs
            if reasoning_content:
                rtn.generations[0].message.additional_kwargs["reasoning_content"] = reasoning_content

        return rtn

    def _convert_chunk_to_generation_chunk(
            self,
            chunk: Dict,
            default_chunk_class: Any,
            base_generation_info: Optional[Dict]
    ) -> Optional[ChatGenerationChunk]:
        """
        处理流式响应：从 Chunk 中提取 reasoning_content 的增量。
        """
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )

        if not generation_chunk:
            return None

        # 尝试从 chunk delta 中提取 reasoning_content
        choices = chunk.get("choices", [])
        if choices:
            delta = choices[0].get("delta", {})
            reasoning_content = delta.get("reasoning_content")

            if reasoning_content:
                # 将 reasoning_content 放入 additional_kwargs
                generation_chunk.message.additional_kwargs["reasoning_content"] = reasoning_content

        return generation_chunk
