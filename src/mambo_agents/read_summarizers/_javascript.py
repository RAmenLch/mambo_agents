"""JavaScript / TypeScript source file summarizer via ``tree-sitter``."""

from __future__ import annotations

from pathlib import PurePosixPath

from tree_sitter import Language, Node, Parser
import tree_sitter_javascript
import tree_sitter_typescript

from mambo_agents.backends.protocol import ReadSummarizer
from mambo_agents.backends.schemas import VirtualPath

_JS_LANG = Language(tree_sitter_javascript.language())
_TS_LANG = Language(tree_sitter_typescript.language_typescript())
_TSX_LANG = Language(tree_sitter_typescript.language_tsx())

_LINE_LIMIT = 200


def javascript_summarizer() -> ReadSummarizer:
    """Return a ``ReadSummarizer`` that analyses JavaScript / TypeScript files.

    Uses ``tree-sitter`` for accurate AST-based extraction of classes,
    function declarations, arrow functions, and exported declarations.
    ``.js`` → JavaScript; ``.ts`` / ``.tsx`` → TypeScript.
    """

    _parsers: dict[str, Parser] = {
        ".js": Parser(_JS_LANG),
        ".ts": Parser(_TS_LANG),
        ".tsx": Parser(_TSX_LANG),
    }

    def _summarize(file_path: VirtualPath, content: str, max_chars: int) -> str:
        suffix = PurePosixPath(file_path.value).suffix.lower()
        parser = _parsers.get(suffix)
        if parser is None:
            return _fallback(file_path, content, max_chars)
        return _summarize_js(file_path, content, max_chars, parser, suffix)

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


def _summarize_js(
    file_path: VirtualPath, content: str, max_chars: int, parser: Parser, suffix: str
) -> str:
    total_lines = content.count("\n") + 1
    code = content.encode("utf-8")
    tree = parser.parse(code)

    classes: list[tuple[int, str]] = []
    functions: list[tuple[int, str]] = []
    variables: list[tuple[int, str]] = []

    _walk_js_node(
        tree.root_node,
        code,
        classes,
        functions,
        variables,
    )

    name = {".js": "JavaScript", ".ts": "TypeScript", ".tsx": "TSX"}.get(
        suffix, suffix
    )
    lines: list[str] = [
        f"[{name} 文件过大，已生成结构摘要"
        f"（完整文件 {len(content):,} 字符，{total_lines:,} 行）]",
        "",
    ]

    if classes:
        lines.append(f"[Classes / Interfaces] ({len(classes)}):")
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

    if variables:
        lines.append(f"[Top-level Declarations] ({len(variables)}):")
        for lineno, label in variables[:_LINE_LIMIT]:
            lines.append(f"  L{lineno:<5} {label}")
        if len(variables) > _LINE_LIMIT:
            lines.append(f"  ... {len(variables) - _LINE_LIMIT} more")
        lines.append("")

    lines.append(
        f">>> Use read(file_path='{file_path}', offset=<lineno>, limit=100) "
        f"to read sections of interest"
    )
    return "\n".join(lines)


def _walk_js_node(
    node: Node,
    source: bytes,
    classes: list[tuple[int, str]],
    functions: list[tuple[int, str]],
    variables: list[tuple[int, str]],
) -> None:
    """Recursively walk ``tree-sitter`` node and collect declarations.

    Only processes top-level and exported declarations (does not dive
    into nested function bodies).
    """
    kind = node.type

    # class_declaration
    if kind == "class_declaration":
        name = _child_text(node, "identifier", source)
        if name:
            classes.append((node.start_point[0] + 1, f"class {name}"))
        return  # don't recurse into class body separately

    # function_declaration / generator_function_declaration
    if kind in ("function_declaration", "generator_function_declaration"):
        name = _child_text(node, "identifier", source)
        if name:
            prefix = "async " if _has_child(node, "async") else ""
            functions.append((node.start_point[0] + 1, f"{prefix}function {name}()"))
        return

    # arrow_function assigned to a variable/constant
    if kind in (
        "lexical_declaration",
        "variable_declaration",
    ):
        for child in node.children:
            if child.type == "variable_declarator":
                _handle_variable_declarator(child, source, variables)
        return

    # export_statement — peel and recurse into the exported thing
    if kind == "export_statement":
        for child in node.children:
            if child.type not in ("export", "default"):
                _walk_js_node(child, source, classes, functions, variables)
        return

    # method_definition inside class body — skip for now (handled at class level)
    if kind in (
        "method_definition",
        "pair",
        "object",
        "statement_block",
        "class_body",
        "formal_parameters",
        "property_signature",
        "method_signature",
        "call_signature",
        "construct_signature",
        "index_signature",
        "arrow_function",
        "function",
    ):
        return

    # interface / type alias (TypeScript)
    if kind == "interface_declaration":
        name = _child_text(node, "type_identifier", source)
        if name:
            classes.append((node.start_point[0] + 1, f"interface {name}"))
        return

    if kind == "type_alias_declaration":
        name = _child_text(node, "type_identifier", source)
        if name:
            classes.append((node.start_point[0] + 1, f"type {name}"))
        return

    if kind == "enum_declaration":
        name = _child_text(node, "identifier", source)
        if name:
            classes.append((node.start_point[0] + 1, f"enum {name}"))
        return

    # Recurse into non-leaf nodes
    for child in node.children:
        _walk_js_node(child, source, classes, functions, variables)


def _handle_variable_declarator(
    node: Node, source: bytes, variables: list[tuple[int, str]]
) -> None:
    """Extract variable name and detect arrow-function / function assignments."""
    name = _child_text(node, "identifier", source)
    if not name:
        # destructuring pattern — skip the name
        pattern = node.child_by_field_name("name")
        if pattern:
            name = source[pattern.start_byte : pattern.end_byte].decode("utf-8")

    value = node.child_by_field_name("value")
    if value is None:
        if name:
            variables.append((node.start_point[0] + 1, name))
        return

    if value.type == "arrow_function":
        prefix = "async " if _has_child(value, "async") else ""
        variables.append((node.start_point[0] + 1, f"{prefix}{name} = (...) => {{ ... }}"))
    elif value.type == "function":
        prefix = "async " if _has_child(value, "async") else ""
        variables.append((node.start_point[0] + 1, f"{prefix}{name} = function() {{ ... }}"))
    elif value.type == "class":
        variables.append((node.start_point[0] + 1, f"{name} = class {{ ... }}"))
    else:
        variables.append((node.start_point[0] + 1, name))


def _child_text(node: Node, child_type: str, source: bytes) -> str | None:
    child = node.child_by_field_name("name")
    if child is None:
        # Fall back to first child of given type
        for c in node.children:
            if c.type == child_type:
                child = c
                break
    if child is None:
        return None
    return source[child.start_byte : child.end_byte].decode("utf-8")


def _has_child(node: Node, child_type: str) -> bool:
    return any(c.type == child_type for c in node.children)
