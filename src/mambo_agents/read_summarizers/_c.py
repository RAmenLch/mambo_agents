"""C source file summarizer via ``tree-sitter``."""

from __future__ import annotations

from pathlib import PurePosixPath

from tree_sitter import Language, Node, Parser
import tree_sitter_c

from mambo_agents.backends.protocol import ReadSummarizer
from mambo_agents.backends.schemas import VirtualPath

_LANG = Language(tree_sitter_c.language())
_PARSER = Parser(_LANG)
_LINE_LIMIT = 200


def c_summarizer() -> ReadSummarizer:
    """Return a ``ReadSummarizer`` that analyses C source files."""

    def _summarize(file_path: VirtualPath, content: str, max_chars: int) -> str:
        suffix = PurePosixPath(file_path.value).suffix.lower()
        if suffix not in (".c", ".h"):
            return _fallback(file_path, content, max_chars)
        return _summarize_c(file_path, content, max_chars)

    return _summarize


def _fallback(file_path: VirtualPath, content: str, max_chars: int) -> str:
    total_lines = content.count("\n") + 1
    return (
        f"[返回结果过大（{len(content):,} 字符，{total_lines:,} 行），"
        f"超过读取上限 {max_chars:,} 字符。"
        f"请重新指定 offset + limit 参数后读取。"
        f"示例: read(file_path='{file_path}', offset=0, limit=500)]"
    )


def _summarize_c(file_path: VirtualPath, content: str, max_chars: int) -> str:
    total_lines = content.count("\n") + 1
    code = content.encode("utf-8")
    tree = _PARSER.parse(code)

    structs: list[tuple[int, str]] = []
    functions: list[tuple[int, str]] = []

    _walk_c(tree.root_node, code, structs, functions)

    lines: list[str] = [
        f"[C 文件过大，已生成结构摘要"
        f"（完整文件 {len(content):,} 字符，{total_lines:,} 行）]",
        "",
    ]

    if structs:
        lines.append(f"[Structs / Unions / Enums] ({len(structs)}):")
        for lineno, label in structs[:_LINE_LIMIT]:
            lines.append(f"  L{lineno:<5} {label}")
        if len(structs) > _LINE_LIMIT:
            lines.append(f"  ... {len(structs) - _LINE_LIMIT} more")
        lines.append("")

    if functions:
        lines.append(f"[Functions] ({len(functions)}):")
        for lineno, label in functions[:_LINE_LIMIT]:
            lines.append(f"  L{lineno:<5} {label}")
        if len(functions) > _LINE_LIMIT:
            lines.append(f"  ... {len(functions) - _LINE_LIMIT} more")
        lines.append("")

    lines.append(
        f">>> Use read(file_path='{file_path}', offset=<lineno>, limit=100) "
        f"to read sections of interest"
    )
    return "\n".join(lines)


def _walk_c(
    node: Node,
    source: bytes,
    structs: list[tuple[int, str]],
    functions: list[tuple[int, str]],
) -> None:
    kind = node.type

    if kind == "function_definition":
        declarator = node.child_by_field_name("declarator")
        tp = _type_text(node.child_by_field_name("type"), source)
        name = _declarator_name(declarator, source) if declarator else "?"
        label = f"{tp} {name}(...)" if tp else f"{name}(...)"
        functions.append((node.start_point[0] + 1, label))
        return

    if kind == "struct_specifier":
        name = _child_name(node, source)
        label = f"struct {name}" if name else "struct { ... }"
        structs.append((node.start_point[0] + 1, label))
        return

    if kind == "union_specifier":
        name = _child_name(node, source)
        label = f"union {name}" if name else "union { ... }"
        structs.append((node.start_point[0] + 1, label))
        return

    if kind == "enum_specifier":
        name = _child_name(node, source)
        label = f"enum {name}" if name else "enum { ... }"
        structs.append((node.start_point[0] + 1, label))
        return

    if kind in (
        "compound_statement", "field_declaration_list",
        "enumerator_list", "parameter_list",
        "declaration",
        "preproc_def", "preproc_function_def",
        "preproc_if", "preproc_ifdef", "preproc_include",
    ):
        return

    for child in node.children:
        _walk_c(child, source, structs, functions)


def _child_name(node: Node, source: bytes) -> str | None:
    for child in node.children:
        if child.type in ("type_identifier", "identifier", "field_identifier"):
            return source[child.start_byte : child.end_byte].decode("utf-8")
    return None


def _declarator_name(node: Node | None, source: bytes) -> str:
    if node is None:
        return "?"
    # Walk down nested declarators to find the identifier
    for child in node.children:
        if child.type == "identifier":
            return source[child.start_byte : child.end_byte].decode("utf-8")
        if child.type in (
            "function_declarator", "pointer_declarator",
            "array_declarator", "parenthesized_declarator",
        ):
            return _declarator_name(child, source)
    return "?"


def _type_text(node: Node | None, source: bytes) -> str:
    if node is None:
        return ""
    text: list[str] = []
    for child in node.children:
        if child.type in (
            "primitive_type", "sized_type_specifier",
            "type_identifier", "struct_specifier",
            "enum_specifier", "union_specifier",
        ):
            text.append(source[child.start_byte : child.end_byte].decode("utf-8"))
        elif child.type == "type_qualifier":
            text.append(source[child.start_byte : child.end_byte].decode("utf-8"))
    return " ".join(text)
