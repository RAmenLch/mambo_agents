"""Java source file summarizer via ``tree-sitter``."""

from __future__ import annotations

from pathlib import PurePosixPath

from tree_sitter import Language, Node, Parser
import tree_sitter_java

from mambo_agents.backends.protocol import ReadSummarizer

_LANG = Language(tree_sitter_java.language())
_PARSER = Parser(_LANG)
_LINE_LIMIT = 200


def java_summarizer() -> ReadSummarizer:
    """Return a ``ReadSummarizer`` that analyses Java source files."""

    def _summarize(file_path: str, content: str, max_chars: int) -> str:
        suffix = PurePosixPath(file_path).suffix.lower()
        if suffix != ".java":
            return _fallback(file_path, content, max_chars)
        return _summarize_java(file_path, content, max_chars)

    return _summarize


def _fallback(file_path: str, content: str, max_chars: int) -> str:
    total_lines = content.count("\n") + 1
    return (
        f"[返回结果过大（{len(content):,} 字符，{total_lines:,} 行），"
        f"超过读取上限 {max_chars:,} 字符。"
        f"请重新指定 offset + limit 参数后读取。"
        f"示例: read(file_path='{file_path}', offset=0, limit=500)]"
    )


def _summarize_java(file_path: str, content: str, max_chars: int) -> str:
    total_lines = content.count("\n") + 1
    code = content.encode("utf-8")
    tree = _PARSER.parse(code)

    classes: list[tuple[int, str]] = []
    methods: list[tuple[int, str]] = []

    _walk_java(tree.root_node, code, classes, methods)

    lines: list[str] = [
        f"[Java 文件过大，已生成结构摘要"
        f"（完整文件 {len(content):,} 字符，{total_lines:,} 行）]",
        "",
    ]

    if classes:
        lines.append(f"[Classes / Interfaces / Enums] ({len(classes)}):")
        for lineno, label in classes[:_LINE_LIMIT]:
            lines.append(f"  L{lineno:<5} {label}")
        if len(classes) > _LINE_LIMIT:
            lines.append(f"  ... {len(classes) - _LINE_LIMIT} more")
        lines.append("")

    if methods:
        lines.append(f"[Methods / Constructors] ({len(methods)}):")
        for lineno, label in methods[:_LINE_LIMIT]:
            lines.append(f"  L{lineno:<5} {label}")
        if len(methods) > _LINE_LIMIT:
            lines.append(f"  ... {len(methods) - _LINE_LIMIT} more")
        lines.append("")

    lines.append(
        f">>> Use read(file_path='{file_path}', offset=<lineno>, limit=100) "
        f"to read sections of interest"
    )
    return "\n".join(lines)


def _walk_java(
    node: Node,
    source: bytes,
    classes: list[tuple[int, str]],
    methods: list[tuple[int, str]],
) -> None:
    kind = node.type

    if kind == "class_declaration":
        name = _child_name(node, source)
        if name:
            classes.append((node.start_point[0] + 1, f"class {name}"))
        _walk_java_children(node, source, classes, methods)
        return

    if kind == "interface_declaration":
        name = _child_name(node, source)
        if name:
            classes.append((node.start_point[0] + 1, f"interface {name}"))
        return

    if kind == "enum_declaration":
        name = _child_name(node, source)
        if name:
            classes.append((node.start_point[0] + 1, f"enum {name}"))
        return

    if kind == "method_declaration":
        name = _child_name(node, source)
        if name:
            ret = _child_modifiers(node, source)
            label = f"{ret}{name}()" if ret else f"{name}()"
            methods.append((node.start_point[0] + 1, label))
        return

    if kind == "constructor_declaration":
        name = _child_name(node, source)
        if name:
            methods.append((node.start_point[0] + 1, f"{name}() [ctor]"))
        return

    if kind in (
        "block", "formal_parameters", "annotation_type_declaration",
        "field_declaration", "local_variable_declaration",
    ):
        return

    _walk_java_children(node, source, classes, methods)


def _walk_java_children(
    node: Node,
    source: bytes,
    classes: list[tuple[int, str]],
    methods: list[tuple[int, str]],
) -> None:
    for child in node.children:
        _walk_java(child, source, classes, methods)


def _child_name(node: Node, source: bytes) -> str | None:
    for child in node.children:
        if child.type == "identifier":
            return source[child.start_byte : child.end_byte].decode("utf-8")
    return None


def _child_modifiers(node: Node, source: bytes) -> str:
    """Return a modifiers prefix like 'static void ' etc."""
    parts: list[str] = []
    for child in node.children:
        if child.type in ("modifiers",):
            text = source[child.start_byte : child.end_byte].decode("utf-8")
            parts.append(text.strip() + " ")
        elif child.type in (
            "void_type", "integral_type", "floating_point_type",
            "boolean_type", "generic_type", "array_type", "type_identifier",
        ):
            parts.append(source[child.start_byte : child.end_byte].decode("utf-8") + " ")
    return "".join(parts)
