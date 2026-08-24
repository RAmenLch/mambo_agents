"""Multimodal describers — substitute text for non-text file reads.

A text-only model cannot consume the raw base64 multimodal content block that
``read()`` produces for images, video, audio, or documents.  This module builds
``MultimodalDescriber`` callbacks that either:

- describe / transcribe / summarise the file via a multimodal chat model
  (:func:`multimodal_describer` plus the per-type ``video_describer``,
  ``audio_describer`` and ``document_describer``), or
- reject any multimodal file with an explicit error text
  (:func:`reject_multimodal_describer`).

A describer returns ``None`` to signal "not my type — leave the multimodal
result unchanged".
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any, Iterable

from langchain_core.messages import HumanMessage

from mambo_agents.backends.protocol import MultimodalDescriber
from mambo_agents.backends.schemas import VirtualPath

DESCRIBE_FAILED = "(无法生成多模态描述)"
"""Fallback text returned when the multimodal model yields no usable caption."""

LC_SOURCE = "multimodal_describer"
"""Marker attached to describer model calls so they are not mis-consumed by
user-facing traces / history logic (mirrors ``summarization``'s ``lc_source``)."""

_DEFAULT_PROMPT = (
    "请精确描述这个多模态文件的内容：如实说明主体、场景、关键细节以及出现的文字；"
    "不要臆测，不要概括。如果内容模糊或无法辨认，请如实说明。"
)
_IMAGE_PROMPT = (
    "请精确描述这张图片：如实说明主体、场景、颜色、动作、可见文字与细节；"
    "不要臆测，不要概括。如果图片模糊、过暗或无法辨认，请如实说明。"
)
_VIDEO_PROMPT = (
    "请精确描述这个视频的内容：如实说明场景、动作、主要对象、出现的文字以及事件顺序；"
    "不要臆测，不要概括。如果无法观看，请如实说明。"
)
_AUDIO_PROMPT = (
    "请精确转述这段音频的内容：如实记录其中的语音、文字或音乐信息；"
    "不要臆测，不要概括。如果无法听清，请如实说明。"
)
_DOCUMENT_PROMPT = (
    "请全面如实的转述这份文档的内容"
    "不要臆测，不要概括。如果内容无法识别，请如实说明。"
)

_DEFAULT_REJECT_TEMPLATE = (
    "[无法读取：{path} 是一个多模态/二进制文件（{mime_type}），"
    "纯文本模型无法理解其内容。如需处理请改用支持多模态的模型，或忽略该文件。]"
)


def _block_type(mime_type: str) -> str | None:
    """Map a MIME type to LangChain's unified multimodal block type."""
    mime = (mime_type or "").lower()
    if mime.startswith("image/"):
        return "image"
    if mime.startswith("video/"):
        return "video"
    if mime.startswith("audio/"):
        return "audio"
    if mime.startswith("application/"):
        return "file"  # PDF / PPT / PPTX and other documents
    return None


def _make_describer(
    model: Any,
    allowed_types: frozenset[str],
    *,
    prompt: str,
    max_chars: int,
) -> MultimodalDescriber:
    """Build a describer that sends unified multimodal blocks for *allowed_types*."""

    def _describe(file_path: VirtualPath, base64_content: str, mime_type: str) -> str | None:
        block_type = _block_type(mime_type)
        if block_type is None or block_type not in allowed_types:
            return None
        block: dict[str, str] = {
            "type": block_type,
            "base64": base64_content,
            "mime_type": mime_type,
        }
        if block_type == "file":
            block["filename"] = PurePosixPath(file_path.value).name
        message = model.invoke(
            [HumanMessage(content=[block, {"type": "text", "text": prompt}])],
            config={"metadata": {"lc_source": LC_SOURCE}},
        )
        text = _coerce_text(message.content)
        if len(text) > max_chars:
            text = text[:max_chars] + "…"
        return text.strip() or DESCRIBE_FAILED

    return _describe


def multimodal_describer(
    model: Any,
    *,
    prompt: str = _DEFAULT_PROMPT,
    max_chars: int = 2000,
) -> MultimodalDescriber:
    """Return a :class:`MultimodalDescriber` that describes any media file.

    Handles images, video, audio and documents by sending LangChain's unified
    content block (``image`` / ``video`` / ``audio`` / ``file``) — the same
    shapes the ``read`` tool emits — which the model provider translates
    automatically.

    Args:
        model: A LangChain ``BaseChatModel`` with multimodal capability
            (e.g. ``ChatAnthropic``, ``ChatOpenAI``, ``ChatGoogleGenerativeAI``).
        prompt: Instruction sent alongside the media file.
        max_chars: Truncate the returned text to this many characters.
    """
    return _make_describer(
        model, frozenset({"image", "video", "audio", "file"}),
        prompt=prompt, max_chars=max_chars,
    )


def image_describer(
    model: Any,
    *,
    prompt: str = _IMAGE_PROMPT,
    max_chars: int = 2000,
) -> MultimodalDescriber:
    """Return a :class:`MultimodalDescriber` that describes **image** files only."""
    return _make_describer(
        model, frozenset({"image"}),
        prompt=prompt, max_chars=max_chars,
    )


def video_describer(
    model: Any,
    *,
    prompt: str = _VIDEO_PROMPT,
    max_chars: int = 2000,
) -> MultimodalDescriber:
    """Return a :class:`MultimodalDescriber` that describes **video** files only."""
    return _make_describer(
        model, frozenset({"video"}),
        prompt=prompt, max_chars=max_chars,
    )


def audio_describer(
    model: Any,
    *,
    prompt: str = _AUDIO_PROMPT,
    max_chars: int = 2000,
) -> MultimodalDescriber:
    """Return a :class:`MultimodalDescriber` that transcribes **audio** files only."""
    return _make_describer(
        model, frozenset({"audio"}),
        prompt=prompt, max_chars=max_chars,
    )


def document_describer(
    model: Any,
    *,
    prompt: str = _DOCUMENT_PROMPT,
    max_chars: int = 2000,
) -> MultimodalDescriber:
    """Return a :class:`MultimodalDescriber` that summarises **documents** only.

    Handles PDF / PPT / PPTX (MIME types starting with ``application/``).
    """
    return _make_describer(
        model, frozenset({"file"}),
        prompt=prompt, max_chars=max_chars,
    )


def reject_multimodal_describer(
    *,
    allow: Iterable[str] = (),
    message_template: str = _DEFAULT_REJECT_TEMPLATE,
) -> MultimodalDescriber:
    """Return a :class:`MultimodalDescriber` that rejects multimodal files.

    Every non-text file (image, video, audio, document) read through a
    backend configured with this describer yields the rejection text instead
    of the raw base64 content block, unless its type is in *allow* — those
    pass through (return ``None``) and keep the original multimodal result.

    *allow* accepts LangChain unified block types: ``"image"``, ``"video"``,
    ``"audio"``, ``"file"``.  For example, a vision-capable model that wants
    to read images natively but defend against video would use
    ``allow={"image"}`` — images stay multimodal blocks for the model to
    read directly, while video / audio / documents get rejected.

    Args:
        allow: Block types that must NOT be rejected (pass through).
        message_template: Template with ``{path}`` and ``{mime_type}``
            placeholders used to build the error text.
    """
    allowed = frozenset(allow)

    def _describe(file_path: VirtualPath, base64_content: str, mime_type: str) -> str | None:
        block_type = _block_type(mime_type)
        if block_type is not None and block_type in allowed:
            return None
        mime = mime_type or "未知类型"
        return message_template.format(path=file_path.value, mime_type=mime)

    return _describe


def _coerce_text(content: Any) -> str:
    """Coerce an ``AIMessage`` content value to a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return " ".join(parts)
    if content is None:
        return ""
    return str(content)
