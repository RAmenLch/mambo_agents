"""Edit helpers: path allowlist/blocklist, trailing-newline detection."""

from __future__ import annotations

from mambo_agents.backends.schemas import BackendError, EditResult, ErrorCode, VirtualPath


def check_path_allowed(
    path: str,
    *,
    whitelist: frozenset[VirtualPath] | None = None,
    blacklist: frozenset[VirtualPath] | None = None,
) -> bool:
    """Check whether *path* (virtual) is allowed for edit/write/delete.

    When *whitelist* is set, *path* must start with (or equal) one of
    its entries.  When *blacklist* is set, *path* must NOT start with
    (or equal) any entry.  The two are mutually exclusive and the caller
    must enforce that.

    Prefixes are normalised (trailing slash stripped) so both
    ``VirtualPath("/src")`` and ``VirtualPath("/src/")`` work identically.

    Args:
        path: Virtual absolute path to check (e.g. ``"/src/foo.py"``).
        whitelist: Allowed path prefixes (e.g. ``{VirtualPath("/src")}``).
        blacklist: Forbidden path prefixes (e.g. ``{VirtualPath("/build")}``).

    Returns:
        ``True`` if the path is permitted.
    """
    if whitelist is not None:
        return any(
            path == prefix.normalized or path.startswith(prefix.normalized + "/")
            for prefix in whitelist
        )
    if blacklist is not None:
        return not any(
            path == prefix.normalized or path.startswith(prefix.normalized + "/")
            for prefix in blacklist
        )
    return True


def detect_trailing_newline_mismatch(
    old_str: str,
    existing_content: str,
) -> EditResult | None:
    """Check whether *old_str* failed because of a trailing newline mismatch.

    LLMs often append ``\\n`` to *old_str* even when the file does not
    end with a newline.  This helper detects that case and returns a
    descriptive error so the model can retry.

    Returns ``None`` when no trailing-newline mismatch is detected (the
    caller should fall through to the generic "not found" error).
    """
    if not (
        old_str.endswith("\n")
        and len(old_str) > 1
        and existing_content.endswith(old_str.removesuffix("\n"))
    ):
        return None

    stripped = old_str.removesuffix("\n")
    stripped_count = existing_content.count(stripped)
    if stripped_count == 1:
        return EditResult(
            error=BackendError(
                code=ErrorCode.OLD_STR_NOT_FOUND,
                message="old_str 以换行符结尾，但文件不以换行符结尾。请去掉 old_str 末尾的换行符后重试",
            ),
        )
    return EditResult(
        error=BackendError(
            code=ErrorCode.MULTI_OCCURRENCES,
            message=f"old_str 以换行符结尾，去掉后匹配到 {stripped_count} 处。请去掉末尾换行符并增加上下文使匹配唯一",
        ),
    )
