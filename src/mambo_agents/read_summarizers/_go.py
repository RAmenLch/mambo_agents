"""Go source file summarizer via ``tree-sitter``."""

from __future__ import annotations

from pathlib import PurePosixPath

from tree_sitter import Language, Node, Parser
import tree_sitter_go

from mambo_agents.backends.protocol import ReadSummarizer

_LANG = Language(tree_sitter_go.language())
_PARSER = Parser(_LANG)
_LINE_LIMIT = 200


def go_summarizer() -> ReadSummarizer:
    """Return a ``ReadSummarizer`` that analyses Go source files."""

    def _summarize(file_path: str, content: str, max_chars: int) -> str:
        suffix = PurePosixPath(file_path).suffix.lower()
        if suffix != ".go":
            return _fallback(file_path, content, max_chars)
        return _summarize_go(file_path, content, max_chars)

    return _summarize


def _fallback(file_path: str, content: str, max_chars: int) -> str:
    total_lines = content.count("\n") + 1
    return (
        f"[返回结果过大（{len(content):,} 字符，{total_lines:,} 行），"
        f"超过读取上限 {max_chars:,} 字符。"
        f"请重新指定 offset + limit 参数后读取。"
        f"示例: read(file_path='{file_path}', offset=0, limit=500)]"
    )


def _summarize_go(file_path: str, content: str, max_chars: int) -> str:
    total_lines = content.count("\n") + 1
    code = content.encode("utf-8")
    tree = _PARSER.parse(code)

    types: list[tuple[int, str]] = []
    functions: list[tuple[int, str]] = []

    _walk_go(tree.root_node, code, types, functions)

    lines: list[str] = [
        f"[Go 文件过大，已生成结构摘要"
        f"（完整文件 {len(content):,} 字符，{total_lines:,} 行）]",
        "",
    ]

    if types:
        lines.append(f"[Types / Interfaces] ({len(types)}):")
        for lineno, label in types[:_LINE_LIMIT]:
            lines.append(f"  L{lineno:<5} {label}")
        if len(types) > _LINE_LIMIT:
            lines.append(f"  ... {len(types) - _LINE_LIMIT} more")
        lines.append("")

    if functions:
        lines.append(f"[Functions / Methods] ({len(functions)}):")
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


def _walk_go(
    node: Node,
    source: bytes,
    types: list[tuple[int, str]],
    functions: list[tuple[int, str]],
) -> None:
    kind = node.type

    if kind == "function_declaration":
        name = _child_name(node, source)
        params = _signature_params(node, source)
        returns = _return_types(node, source)
        label = f"func {name}({params})" if name else f"func({params})"
        if returns:
            label += f" {returns}"
        functions.append((node.start_point[0] + 1, label))
        return

    if kind == "method_declaration":
        name = _field_name(node, source)
        receiver = _receiver_text(node, source)
        params = _signature_params(node, source)
        returns = _return_types(node, source)
        label = f"func ({receiver}) {name}({params})"
        if returns:
            label += f" {returns}"
        functions.append((node.start_point[0] + 1, label))
        return

    if kind == "type_declaration":
        for child in node.children:
            if child.type == "type_spec":
                name = _child_name(child, source)
                tp = _type_spec_type(child, source)
                label = f"type {name} {tp}" if name and tp else f"type {name}" if name else "type ..."
                types.append((child.start_point[0] + 1, label))
        return

    if kind in (
        "block", "parameter_list", "field_declaration_list",
        "var_declaration", "const_declaration",
        "import_declaration", "short_var_declaration",
        "expression_list", "literal_value",
    ):
        return

    for child in node.children:
        _walk_go(child, source, types, functions)


def _child_name(node: Node, source: bytes) -> str | None:
    name_node = node.child_by_field_name("name")
    if name_node:
        return source[name_node.start_byte : name_node.end_byte].decode("utf-8")
    return None


def _field_name(node: Node, source: bytes) -> str:
    field = node.child_by_field_name("name")
    if field:
        return source[field.start_byte : field.end_byte].decode("utf-8")
    return "?"


def _signature_params(node: Node, source: bytes) -> str:
    params_node = node.child_by_field_name("parameters")
    if params_node:
        inner = source[params_node.start_byte + 1 : params_node.end_byte - 1].decode("utf-8").strip()
        return inner[:80] + "…" if len(inner) > 80 else inner
    return ""


def _return_types(node: Node, source: bytes) -> str:
    result_node = node.child_by_field_name("result")
    if result_node:
        inner = source[result_node.start_byte : result_node.end_byte].decode("utf-8").strip()
        inner = inner.removeprefix("(").removesuffix(")").strip()
        return inner[:60] + "…" if len(inner) > 60 else inner
    return ""


def _receiver_text(node: Node, source: bytes) -> str:
    recv = node.child_by_field_name("receiver")
    if recv:
        return source[recv.start_byte + 1 : recv.end_byte - 1].decode("utf-8").strip()
    return "?"


def _type_spec_type(node: Node, source: bytes) -> str:
    tp_node = node.child_by_field_name("type")
    if tp_node is None:
        return ""
    tp = source[tp_node.start_byte : tp_node.end_byte].decode("utf-8").strip()
    return tp[:80] + "…" if len(tp) > 80 else tp
