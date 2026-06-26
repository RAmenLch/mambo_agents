"""Python source file summarizer — extracts class/function outline via ``ast``."""

from __future__ import annotations

import ast
from pathlib import PurePosixPath

from mambo_agents.backends.protocol import ReadSummarizer
from mambo_agents.backends.schemas import VirtualPath

_LINE_LIMIT = 200


def python_summarizer() -> ReadSummarizer:
    """Return a ``ReadSummarizer`` that analyses Python source files.

    Extracts class and function definitions with accurate line numbers
    via the stdlib ``ast`` module.  Non-``.py`` files fall back to the
    generic behaviour (prompt to re-read with offset + limit).
    """

    def _summarize(file_path: VirtualPath, content: str, max_chars: int) -> str:
        suffix = PurePosixPath(file_path.value).suffix.lower()
        if suffix != ".py":
            return _fallback(file_path, content, max_chars)
        return _summarize_python(file_path, content, max_chars)

    return _summarize


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fallback(file_path: VirtualPath, content: str, max_chars: int) -> str:
    total_lines = content.count("\n") + 1
    return (
        f"[返回结果过大（{len(content):,} 字符，{total_lines:,} 行），"
        f"超过读取上限 {max_chars:,} 字符。"
        f"请重新指定 offset + limit 参数后读取。"
        f"示例: read(file_path='{file_path}', offset=0, limit=500)]"
    )


def _summarize_python(file_path: VirtualPath, content: str, max_chars: int) -> str:
    total_lines = content.count("\n") + 1
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return (
            f"[Python 文件过大（{len(content):,} 字符，{total_lines:,} 行），"
            f"超过读取上限 {max_chars:,} 字符。"
            f"因语法错误无法解析结构。"
            f"请重新指定 offset + limit 参数后读取。"
            f"示例: read(file_path='{file_path}', offset=0, limit=500)]"
        )

    classes: list[tuple[int, str]] = []
    functions: list[tuple[int, str]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            bases = _class_bases(node)
            label = f"{node.name}({', '.join(bases)})" if bases else node.name
            if node.decorator_list:
                decos = _decorator_names(node.decorator_list)
                label = f"{'@' + ', @'.join(decos)}  {label}"
            classes.append((node.lineno, label))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
            args_str = _function_args(node)
            label = f"{prefix}{node.name}({args_str})"
            if node.decorator_list:
                decos = _decorator_names(node.decorator_list)
                label = f"{'@' + ', @'.join(decos)}  {label}"
            functions.append((node.lineno, label))

    # Build output — re-use the same compact format for all summaries.
    lines: list[str] = [
        f"[Python 文件过大，已生成结构摘要"
        f"（完整文件 {len(content):,} 字符，{total_lines:,} 行）]",
        "",
    ]

    if classes:
        lines.append(f"[Classes] ({len(classes)}):")
        for lineno, label in classes[: _LINE_LIMIT]:
            lines.append(f"  L{lineno:<5} {label}")
        if len(classes) > _LINE_LIMIT:
            lines.append(f"  ... {len(classes) - _LINE_LIMIT} more classes")
        lines.append("")

    if functions:
        lines.append(f"[Functions] ({len(functions)}):")
        for lineno, label in functions[: _LINE_LIMIT]:
            lines.append(f"  L{lineno:<5} {label}")
        if len(functions) > _LINE_LIMIT:
            lines.append(f"  ... {len(functions) - _LINE_LIMIT} more functions")
        lines.append("")

    lines.append(
        f">>> Use read(file_path='{file_path}', offset=<lineno>, limit=100) "
        f"to read sections of interest"
    )
    return "\n".join(lines)


def _class_bases(node: ast.ClassDef) -> list[str]:
    names: list[str] = []
    for base in node.bases:
        if isinstance(base, ast.Name):
            names.append(base.id)
        elif isinstance(base, ast.Attribute):
            names.append(ast.unparse(base))
    return names


def _decorator_names(decorator_list: list[ast.expr]) -> list[str]:
    names: list[str] = []
    for deco in decorator_list:
        if isinstance(deco, ast.Name):
            names.append(deco.id)
        elif isinstance(deco, ast.Attribute):
            names.append(ast.unparse(deco))
        elif isinstance(deco, ast.Call):
            if isinstance(deco.func, ast.Name):
                names.append(deco.func.id)
            elif isinstance(deco.func, ast.Attribute):
                names.append(ast.unparse(deco.func))
    return names


def _function_args(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    parts: list[str] = []
    for arg in node.args.args:
        name = arg.arg
        if arg.annotation:
            name += f": {ast.unparse(arg.annotation)}"
        parts.append(name)
    if node.args.vararg:
        parts.append(f"*{node.args.vararg.arg}")
    if node.args.kwonlyargs:
        for kw in node.args.kwonlyargs:
            name = kw.arg
            if kw.annotation:
                name += f": {ast.unparse(kw.annotation)}"
            parts.append(name)
    if node.args.kwarg:
        parts.append(f"**{node.args.kwarg.arg}")
    return ", ".join(parts)
