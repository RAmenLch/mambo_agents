"""Multimodal file type classification and content validation.

Provides extension-to-type mapping, MIME type guessing, and integrity
validation for multimodal files (images, audio, video, documents) before
they are handed to LLM APIs.
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import PurePosixPath

from mambo_agents.backends.schemas import BackendError, ErrorCode, FileType, ReadResult, VirtualPath


# ============================================================================
# Extension → file type mapping
# ============================================================================

EXTENSION_TO_FILE_TYPE: dict[str, FileType] = {
    # Images (https://ai.google.dev/gemini-api/docs/image-understanding)
    ".png": "image",
    ".jpeg": "image",
    ".jpg": "image",
    ".webp": "image",
    ".gif": "image",
    ".heic": "image",
    ".heif": "image",
    # Video (https://ai.google.dev/gemini-api/docs/video-understanding)
    ".mp4": "video",
    ".mpeg": "video",
    ".mov": "video",
    ".avi": "video",
    ".flv": "video",
    ".mpg": "video",
    ".webm": "video",
    ".wmv": "video",
    ".3gpp": "video",
    # Audio (https://ai.google.dev/gemini-api/docs/audio)
    ".wav": "audio",
    ".mp3": "audio",
    ".aiff": "audio",
    ".aac": "audio",
    ".ogg": "audio",
    ".flac": "audio",
    # Documents
    ".pdf": "file",
    ".ppt": "file",
    ".pptx": "file",
}


def get_file_type(path: str | VirtualPath) -> FileType:
    """Classify a file by its extension.

    Returns:
        One of ``"text"``, ``"image"``, ``"audio"``, ``"video"``, or ``"file"``.
        Defaults to ``"text"`` for unrecognized extensions.
    """
    s = path.value if isinstance(path, VirtualPath) else path
    return EXTENSION_TO_FILE_TYPE.get(PurePosixPath(s).suffix.lower(), "text")


def get_mime_type(path: str | VirtualPath) -> str:
    """Guess the IANA media type for a file path."""
    s = path.value if isinstance(path, VirtualPath) else path
    return (
        mimetypes.guess_type("file" + PurePosixPath(s).suffix)[0]
        or "application/octet-stream"
    )


# ============================================================================
# Magic‑byte signatures for multimodal content validation
# ============================================================================

# Each entry is a list of *alternatives* (OR match — any one alternative is
# sufficient).  Each alternative is a list of ``(offset, bytes)`` pairs that
# must **all** match (AND match).
#
# See the per‑extension comments for authoritative format references.

_SIGNATURES: dict[str, list[list[tuple[int, bytes]]]] = {
    # -- Images ---------------------------------------------------------------
    ".png": [
        [(0, b"\x89PNG\r\n\x1a\n")],
    ],
    ".jpeg": [
        [(0, b"\xff\xd8\xff")],
    ],
    ".jpg": [
        [(0, b"\xff\xd8\xff")],
    ],
    ".webp": [
        [(0, b"RIFF"), (8, b"WEBP")],
    ],
    ".gif": [
        [(0, b"GIF87a")],
        [(0, b"GIF89a")],
    ],
    ".heic": [
        [(4, b"ftyp")],
    ],
    ".heif": [
        [(4, b"ftyp")],
    ],
    # -- Video ----------------------------------------------------------------
    ".mp4": [
        [(4, b"ftyp")],
    ],
    ".mpeg": [
        [(0, b"\x00\x00\x01\xba")],
        [(0, b"\x00\x00\x01\xb3")],
    ],
    ".mov": [
        [(4, b"ftyp")],
    ],
    ".avi": [
        [(0, b"RIFF"), (8, b"AVI ")],
    ],
    ".flv": [
        [(0, b"FLV\x01")],
    ],
    ".mpg": [
        [(0, b"\x00\x00\x01\xba")],
        [(0, b"\x00\x00\x01\xb3")],
    ],
    ".webm": [
        [(0, b"\x1a\x45\xdf\xa3")],
    ],
    ".wmv": [
        [(0, b"\x30\x26\xb2\x75\x8e\x66\xcf\x11\xa6\xd9\x00\xaa\x00\x62\xce\x6c")],
    ],
    ".3gpp": [
        [(4, b"ftyp")],
    ],
    # -- Audio ----------------------------------------------------------------
    ".wav": [
        [(0, b"RIFF"), (8, b"WAVE")],
    ],
    ".mp3": [
        [(0, b"\xff\xfb")],
        [(0, b"\xff\xf3")],
        [(0, b"\xff\xf2")],
        [(0, b"ID3")],
    ],
    ".aiff": [
        [(0, b"FORM"), (8, b"AIFF")],
    ],
    ".aac": [
        [(0, b"\xff\xf1")],
        [(0, b"\xff\xf9")],
    ],
    ".ogg": [
        [(0, b"OggS")],
    ],
    ".flac": [
        [(0, b"fLaC")],
    ],
    # -- Documents ------------------------------------------------------------
    ".pdf": [
        [(0, b"%PDF")],
    ],
    ".ppt": [
        [(0, b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1")],
    ],
    ".pptx": [
        [(0, b"PK\x03\x04")],
    ],
}


def _check_signature(raw: bytes, alternatives: list[list[tuple[int, bytes]]]) -> bool:
    """Return ``True`` if *raw* matches at least one alternative."""
    for alternative in alternatives:
        if all(
            len(raw) >= offset + len(expected) and raw[offset:offset + len(expected)] == expected
            for offset, expected in alternative
        ):
            return True
    return False


def validate_multimodal_content(
    result: ReadResult,
    file_path: VirtualPath,
) -> ReadResult:
    """Validate that a multimodal ``ReadResult`` contains intact content.

    Returns the original *result* if the content is valid, or an error
    ``ReadResult`` if the base64 cannot be decoded or the binary data
    does not match the expected file‑type signature.
    """
    if not result.is_multimodal or result.error is not None or result.content is None:
        return result

    try:
        raw = base64.b64decode(result.content, validate=True)
    except Exception:
        return ReadResult(error=BackendError(
            code=ErrorCode.INVALID,
            path=file_path,
            message="多模态文件读取失败：base64 解码错误，文件可能已损坏",
        ))

    if not raw:
        return ReadResult(error=BackendError(
            code=ErrorCode.INVALID,
            path=file_path,
            message="多模态文件读取失败：文件内容为空",
        ))

    ext = PurePosixPath(file_path.value).suffix.lower()
    alternatives = _SIGNATURES.get(ext)

    if alternatives is not None and not _check_signature(raw, alternatives):
        return ReadResult(error=BackendError(
            code=ErrorCode.INVALID,
            path=file_path,
            message=f"多模态文件读取失败：{result.file_type} 文件格式无效或已损坏",
        ))

    return result
