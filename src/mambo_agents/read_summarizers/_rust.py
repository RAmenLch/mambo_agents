"""Rust source file summarizer via ``tree-sitter``."""

from __future__ import annotations

from pathlib import PurePosixPath

from tree_sitter import Language, Node, Parser
import tree_sitter_rust

from mambo_agents.backends.protocol import ReadSummarizer

_LANG = Language(tree_sitter_rust.language())
_PARSER = Parser(_LANG)
_LINE_LIMIT = 200


def rust_summarizer() -> ReadSummarizer:
    """Return a ``ReadSummarizer`` that analyses Rust source files."""

    def _summarize(file_path: str, content: str, max_chars: int) -> str:
        suffix = PurePosixPath(file_path).suffix.lower()
        if suffix != ".rs":
            return _fallback(file_path, content, max_chars)
        return _summarize_rust(file_path, content, max_chars)

    return _summarize


def _fallback(file_path: str, content: str, max_chars: int) -> str:
    total_lines = content.count("\n") + 1
    return (
        f"[返回结果过大（{len(content):,} 字符，{total_lines:,} 行），"
        f"超过读取上限 {max_chars:,} 字符。"
        f"请重新指定 offset + limit 参数后读取。"
        f"示例: read(file_path='{file_path}', offset=0, limit=500)]"
    )


def _summarize_rust(file_path: str, content: str, max_chars: int) -> str:
    total_lines = content.count("\n") + 1
    code = content.encode("utf-8")
    tree = _PARSER.parse(code)

    types: list[tuple[int, str]] = []
    functions: list[tuple[int, str]] = []

    _walk_rust(tree.root_node, code, types, functions)

    lines: list[str] = [
        f"[Rust 文件过大，已生成结构摘要"
        f"（完整文件 {len(content):,} 字符，{total_lines:,} 行）]",
        "",
    ]

    if types:
        lines.append(f"[Structs / Enums / Traits / Impls / Modules] ({len(types)}):")
        for lineno, label in types[:_LINE_LIMIT]:
            lines.append(f"  L{lineno:<5} {label}")
        if len(types) > _LINE_LIMIT:
            lines.append(f"  ... {len(types) - _LINE_LIMIT} more")
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


def _walk_rust(
    node: Node,
    source: bytes,
    types: list[tuple[int, str]],
    functions: list[tuple[int, str]],
) -> None:
    kind = node.type

    if kind == "function_item":
        name = _child_name(node, source)
        params = _parameters_text(node, source)
        ret = _return_text(node, source)
        label = f"fn {name}({params})" if name else f"fn({params})"
        if ret:
            label += f" -> {ret}"
        functions.append((node.start_point[0] + 1, label))
        return

    if kind == "struct_item":
        name = _child_name(node, source)
        label = f"struct {name}" if name else "struct { ... }"
        types.append((node.start_point[0] + 1, label))
        return

    if kind == "enum_item":
        name = _child_name(node, source)
        label = f"enum {name}" if name else "enum { ... }"
        types.append((node.start_point[0] + 1, label))
        return

    if kind == "trait_item":
        name = _child_name(node, source)
        label = f"trait {name}" if name else "trait { ... }"
        types.append((node.start_point[0] + 1, label))
        return

    if kind == "impl_item":
        tp = _impl_type_text(node, source)
        trait = _impl_trait_text(node, source)
        if trait:
            label = f"impl {trait} for {tp}" if tp else f"impl {trait}"
        else:
            label = f"impl {tp}" if tp else "impl { ... }"
        types.append((node.start_point[0] + 1, label))
        # Walk children to find methods inside impl blocks
        for child in node.children:
            _walk_rust(child, source, types, functions)
        return

    if kind == "mod_item":
        name = _child_name(node, source)
        label = f"mod {name}" if name else "mod { ... }"
        types.append((node.start_point[0] + 1, label))
        return

    if kind == "const_item":
        name = _child_name(node, source)
        if name:
            types.append((node.start_point[0] + 1, f"const {name}"))
        return

    if kind == "static_item":
        name = _child_name(node, source)
        if name:
            types.append((node.start_point[0] + 1, f"static {name}"))
        return

    if kind in (
        "block", "declaration_list", "parameters",
        "field_declaration_list", "enum_variant_list",
        "where_clause", "use_declaration",
        "macro_definition", "macro_invocation",
        "attribute_item", "inner_attribute_item",
    ):
        return

    for child in node.children:
        _walk_rust(child, source, types, functions)


def _child_name(node: Node, source: bytes) -> str | None:
    name_node = node.child_by_field_name("name")
    if name_node:
        return source[name_node.start_byte : name_node.end_byte].decode("utf-8")
    return None


def _parameters_text(node: Node, source: bytes) -> str:
    params_node = node.child_by_field_name("parameters")
    if params_node:
        inner = source[params_node.start_byte + 1 : params_node.end_byte - 1].decode("utf-8").strip()
        return inner[:80] + "…" if len(inner) > 80 else inner
    return ""


def _return_text(node: Node, source: bytes) -> str:
    ret_node = node.child_by_field_name("return_type")
    if ret_node:
        inner = source[ret_node.start_byte + 2 : ret_node.end_byte].decode("utf-8").strip()
        return inner[:60] + "…" if len(inner) > 60 else inner
    return ""


def _impl_type_text(node: Node, source: bytes) -> str:
    tp_node = node.child_by_field_name("type")
    if tp_node:
        return source[tp_node.start_byte : tp_node.end_byte].decode("utf-8").strip()
    return ""


def _impl_trait_text(node: Node, source: bytes) -> str | None:
    trait_node = node.child_by_field_name("trait")
    if trait_node:
        return source[trait_node.start_byte : trait_node.end_byte].decode("utf-8").strip()
    return None
