"""C++ source file summarizer via ``tree-sitter``."""

from __future__ import annotations

from pathlib import PurePosixPath

from tree_sitter import Language, Node, Parser
import tree_sitter_cpp

from mambo_agents.backends.protocol import ReadSummarizer
from mambo_agents.backends.schemas import VirtualPath

_LANG = Language(tree_sitter_cpp.language())
_PARSER = Parser(_LANG)
_LINE_LIMIT = 200

_CPP_SUFFIXES = frozenset({".cpp", ".cc", ".cxx", ".c++", ".hpp", ".hh", ".hxx", ".h++"})


def cpp_summarizer() -> ReadSummarizer:
    """Return a ``ReadSummarizer`` that analyses C++ source files."""

    def _summarize(file_path: VirtualPath, content: str, max_chars: int) -> str:
        suffix = PurePosixPath(file_path.value).suffix.lower()
        if suffix not in _CPP_SUFFIXES:
            return _fallback(file_path, content, max_chars)
        return _summarize_cpp(file_path, content, max_chars)

    return _summarize


def _fallback(file_path: VirtualPath, content: str, max_chars: int) -> str:
    total_lines = content.count("\n") + 1
    return (
        f"[返回结果过大（{len(content):,} 字符，{total_lines:,} 行），"
        f"超过读取上限 {max_chars:,} 字符。"
        f"请重新指定 offset + limit 参数后读取。"
        f"示例: read(file_path='{file_path}', offset=0, limit=500)]"
    )


def _summarize_cpp(file_path: VirtualPath, content: str, max_chars: int) -> str:
    total_lines = content.count("\n") + 1
    code = content.encode("utf-8")
    tree = _PARSER.parse(code)

    classes: list[tuple[int, str]] = []
    functions: list[tuple[int, str]] = []

    _walk_cpp(tree.root_node, code, classes, functions)

    lines: list[str] = [
        f"[C++ 文件过大，已生成结构摘要"
        f"（完整文件 {len(content):,} 字符，{total_lines:,} 行）]",
        "",
    ]

    if classes:
        lines.append(f"[Classes / Structs / Enums / Namespaces] ({len(classes)}):")
        for lineno, label in classes[:_LINE_LIMIT]:
            lines.append(f"  L{lineno:<5} {label}")
        if len(classes) > _LINE_LIMIT:
            lines.append(f"  ... {len(classes) - _LINE_LIMIT} more")
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


def _walk_cpp(
    node: Node,
    source: bytes,
    classes: list[tuple[int, str]],
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

    if kind == "class_specifier":
        name = _child_name(node, source)
        bases = _class_bases(node, source)
        label = f"class {name}" if name else "class { ... }"
        if bases:
            label += f" : {', '.join(bases)}"
        classes.append((node.start_point[0] + 1, label))
        return

    if kind == "struct_specifier":
        name = _child_name(node, source)
        bases = _class_bases(node, source)
        label = f"struct {name}" if name else "struct { ... }"
        if bases:
            label += f" : {', '.join(bases)}"
        classes.append((node.start_point[0] + 1, label))
        return

    if kind == "enum_specifier":
        name = _child_name(node, source)
        label = f"enum {name}" if name else "enum { ... }"
        classes.append((node.start_point[0] + 1, label))
        return

    if kind == "namespace_definition":
        name = _child_name(node, source) or "{anonymous}"
        classes.append((node.start_point[0] + 1, f"namespace {name}"))
        for child in node.children:
            _walk_cpp(child, source, classes, functions)
        return

    if kind == "template_declaration":
        # Record the template params and unwrap to the inner declaration
        tparams = _template_params(node, source)
        for child in node.children:
            if child.type in (
                "function_definition", "class_specifier",
                "struct_specifier", "declaration",
            ):
                _walk_cpp_with_template(child, source, classes, functions, tparams)
        return

    if kind in (
        "compound_statement", "field_declaration_list",
        "enumerator_list", "parameter_list",
        "declaration", "type_definition",
        "access_specifier", "base_class_clause",
    ):
        return

    for child in node.children:
        _walk_cpp(child, source, classes, functions)


def _walk_cpp_with_template(
    node: Node,
    source: bytes,
    classes: list[tuple[int, str]],
    functions: list[tuple[int, str]],
    tparams: str,
) -> None:
    kind = node.type
    if kind == "function_definition":
        declarator = node.child_by_field_name("declarator")
        tp = _type_text(node.child_by_field_name("type"), source)
        name = _declarator_name(declarator, source) if declarator else "?"
        label = f"template<{tparams}> {tp} {name}(...)" if tp else f"template<{tparams}> {name}(...)"
        functions.append((node.start_point[0] + 1, label))
    elif kind in ("class_specifier", "struct_specifier"):
        name = _child_name(node, source)
        kind_label = "class" if kind == "class_specifier" else "struct"
        label = f"template<{tparams}> {kind_label} {name}" if name else f"template<{tparams}> {kind_label}"
        classes.append((node.start_point[0] + 1, label))
    else:
        _walk_cpp(node, source, classes, functions)


def _child_name(node: Node, source: bytes) -> str | None:
    name_node = node.child_by_field_name("name")
    if name_node:
        return source[name_node.start_byte : name_node.end_byte].decode("utf-8")
    return None


def _declarator_name(node: Node | None, source: bytes) -> str:
    if node is None:
        return "?"
    for child in node.children:
        if child.type in ("identifier", "field_identifier", "operator_name", "destructor_name"):
            return source[child.start_byte : child.end_byte].decode("utf-8")
        if child.type in (
            "function_declarator", "pointer_declarator",
            "array_declarator", "parenthesized_declarator",
            "reference_declarator",
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
            "type_identifier", "qualified_identifier",
            "template_type", "auto",
        ):
            text.append(source[child.start_byte : child.end_byte].decode("utf-8"))
        elif child.type == "type_qualifier":
            text.append(source[child.start_byte : child.end_byte].decode("utf-8"))
    return " ".join(text)


def _class_bases(node: Node, source: bytes) -> list[str]:
    bases: list[str] = []
    for child in node.children:
        if child.type == "base_class_clause":
            for bc in child.children:
                if bc.type in ("type_identifier", "qualified_identifier"):
                    bases.append(source[bc.start_byte : bc.end_byte].decode("utf-8"))
                elif bc.type == "template_type":
                    bases.append(source[bc.start_byte : bc.end_byte].decode("utf-8"))
    return bases


def _template_params(node: Node, source: bytes) -> str:
    for child in node.children:
        if child.type == "template_parameter_list":
            return source[child.start_byte : child.end_byte].decode("utf-8")
    return "..."
