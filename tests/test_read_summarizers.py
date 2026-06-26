"""Tests for the ``read_summarizers`` module."""

from __future__ import annotations

import pytest

from mambo_agents.backends.schemas import VirtualPath
from mambo_agents.read_summarizers import (
    c_summarizer,
    cpp_summarizer,
    go_summarizer,
    java_summarizer,
    javascript_summarizer,
    json_summarizer,
    markdown_summarizer,
    python_summarizer,
    rust_summarizer,
)


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------

_ALL_SUMMARIZERS = [
    ("python", python_summarizer),
    ("javascript", javascript_summarizer),
    ("markdown", markdown_summarizer),
    ("json", json_summarizer),
    ("java", java_summarizer),
    ("c", c_summarizer),
    ("cpp", cpp_summarizer),
    ("go", go_summarizer),
    ("rust", rust_summarizer),
]

MAX_CHARS = 1000


def _build_class(capsys_size: int | None = None) -> str:
    """Build a class definition string, optionally very large."""
    inner = "\n    # some member\n" * 50
    if capsys_size is not None:
        inner = "\n".join(f"    x_{i} = {i}" for i in range(capsys_size))
    return f"class Capsys:\n{inner}"


# ---------------------------------------------------------------------------
# Factory & basic callable tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,factory", _ALL_SUMMARIZERS)
def test_factory_returns_callable(name: str, factory) -> None:
    """Each factory returns a callable."""
    s = factory()
    assert callable(s), f"{name} factory did not return a callable"


@pytest.mark.parametrize("name,factory", _ALL_SUMMARIZERS)
def test_callable_accepts_three_args(name: str, factory) -> None:
    """The returned callable accepts (file_path, content, max_chars)."""
    s = factory()
    result = s(VirtualPath("/test.txt"), "hello world", 100)
    assert isinstance(result, str), f"{name} did not return a str"


@pytest.mark.parametrize("name,factory", _ALL_SUMMARIZERS)
def test_fallback_on_unmatched_suffix(name: str, factory) -> None:
    """Unmatched file suffix → fallback (mentions offset + limit)."""
    s = factory()
    result = s(VirtualPath("/file.xyz"), _build_class(200), MAX_CHARS)
    assert "offset" in result.lower(), f"{name} fallback missing offset hint"
    assert "limit" in result.lower(), f"{name} fallback missing limit hint"


# ---------------------------------------------------------------------------
# Python summarizer
# ---------------------------------------------------------------------------


def test_python_extracts_classes_and_functions() -> None:
    py_src = """\
import os

class Foo:
    def method_a(self) -> None:
        pass

    async def method_b(self) -> None:
        pass

def top_level(x: int) -> int:
    return x
"""
    s = python_summarizer()
    result = s(VirtualPath("/test.py"), py_src, MAX_CHARS)

    assert "L" in result
    assert "Foo" in result
    assert "method_a" in result
    assert "method_b" in result
    assert "top_level" in result
    assert "L4" in result  # class Foo line
    assert "L10" in result  # top_level line


def test_python_handles_syntax_error() -> None:
    s = python_summarizer()
    result = s(VirtualPath("/test.py"), "def broken(:", MAX_CHARS)
    assert "语法错误" in result or "无法解析" in result


# ---------------------------------------------------------------------------
# JavaScript summarizer
# ---------------------------------------------------------------------------


def test_js_extracts_class_and_functions() -> None:
    js_src = """\
class Calculator {
    add(a, b) { return a + b; }
}

export async function fetchData(url) {
    return await fetch(url);
}

const square = (x) => x * x;
"""
    s = javascript_summarizer()
    result = s(VirtualPath("/test.js"), js_src, MAX_CHARS)

    assert "class Calculator" in result
    assert "fetchData" in result
    assert "square" in result


def test_ts_uses_typescript_parser() -> None:
    ts_src = """\
interface User {
    name: string;
}

class Admin implements User {
    name: string = "";
}
"""
    s = javascript_summarizer()
    result = s(VirtualPath("/test.ts"), ts_src, MAX_CHARS)
    assert "interface User" in result
    assert "class Admin" in result


# ---------------------------------------------------------------------------
# Java summarizer
# ---------------------------------------------------------------------------


def test_java_extracts_class_and_interface() -> None:
    java_src = """\
public class App {
    public static void main(String[] args) {
        System.out.println("Hello");
    }
}

interface Processor {
    void process();
}
"""
    s = java_summarizer()
    result = s(VirtualPath("/test.java"), java_src, MAX_CHARS)
    assert "class App" in result
    assert "interface Processor" in result
    assert "main" in result


# ---------------------------------------------------------------------------
# C summarizer
# ---------------------------------------------------------------------------


def test_c_extracts_struct_and_functions() -> None:
    c_src = """\
struct Node {
    int val;
};

int add(int a, int b) {
    return a + b;
}

enum Result { OK, FAIL };
"""
    s = c_summarizer()
    result = s(VirtualPath("/test.c"), c_src, MAX_CHARS)
    assert "struct Node" in result
    assert "add" in result
    assert "enum Result" in result


def test_c_handles_h_files() -> None:
    s = c_summarizer()
    result = s(VirtualPath("/test.h"), "int add(int, int);", MAX_CHARS)
    assert isinstance(result, str)  # does not crash


# ---------------------------------------------------------------------------
# C++ summarizer
# ---------------------------------------------------------------------------


def test_cpp_extracts_class_and_namespace() -> None:
    cpp_src = """\
class Animal {
public:
    virtual void speak() = 0;
};

namespace zoo {
    void feed(Animal* a) {}
}
"""
    s = cpp_summarizer()
    result = s(VirtualPath("/test.cpp"), cpp_src, MAX_CHARS)
    assert "class Animal" in result
    assert "namespace zoo" in result
    assert "feed" in result


# ---------------------------------------------------------------------------
# Go summarizer
# ---------------------------------------------------------------------------


def test_go_extracts_type_and_func() -> None:
    go_src = """\
package main

type Runner interface {
    Run() error
}

type Config struct {
    Port int
}

func NewConfig() *Config {
    return &Config{Port: 8080}
}

func (c *Config) Start() {}
"""
    s = go_summarizer()
    result = s(VirtualPath("/test.go"), go_src, MAX_CHARS)
    assert "Runner" in result
    assert "Config" in result
    assert "NewConfig" in result
    assert "Start" in result
    # receiver should be mentioned
    assert "Config" in result.split("Start")[0]


# ---------------------------------------------------------------------------
# Rust summarizer
# ---------------------------------------------------------------------------


def test_rust_extracts_struct_trait_impl() -> None:
    rust_src = """\
pub trait Named {
    fn name(&self) -> &str;
}

pub struct Person {
    name: String,
}

impl Person {
    pub fn new(name: &str) -> Self { Person { name: name.into() } }
}
"""
    s = rust_summarizer()
    result = s(VirtualPath("/test.rs"), rust_src, MAX_CHARS)
    assert "trait Named" in result
    assert "struct Person" in result
    assert "impl Person" in result


# ---------------------------------------------------------------------------
# Markdown summarizer
# ---------------------------------------------------------------------------


def test_md_extracts_headings() -> None:
    md_src = """\
# Title
text
## Section 1
more text
### Sub 1.1
## Section 2
"""
    s = markdown_summarizer()
    result = s(VirtualPath("/test.md"), md_src, MAX_CHARS)
    assert "# Title" in result
    assert "## Section 1" in result
    assert "### Sub 1.1" in result
    assert "## Section 2" in result


def test_md_no_headings() -> None:
    s = markdown_summarizer()
    result = s(VirtualPath("/test.md"), "just some text\nno headings\n", MAX_CHARS)
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# JSON summarizer
# ---------------------------------------------------------------------------


def test_json_extracts_keys() -> None:
    json_str = '{"name": "app", "version": 2, "deps": ["a", "b"], "config": null}'
    s = json_summarizer()
    result = s(VirtualPath("/test.json"), json_str, MAX_CHARS)
    for key in ("name", "version", "deps", "config"):
        assert key in result, f"missing key {key}"


def test_json_handles_array() -> None:
    s = json_summarizer()
    result = s(VirtualPath("/test.json"), "[1, 2, 3, 4, 5, 6, 7, 8]", MAX_CHARS)
    assert "Array" in result or "array" in result


def test_json_handles_parse_error() -> None:
    s = json_summarizer()
    result = s(VirtualPath("/test.json"), "{bad json", MAX_CHARS)
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,factory", _ALL_SUMMARIZERS)
def test_empty_file_does_not_crash(name: str, factory) -> None:
    s = factory()
    result = s(VirtualPath("/test.txt"), "", MAX_CHARS)
    assert isinstance(result, str)


@pytest.mark.parametrize("name,factory", _ALL_SUMMARIZERS)
def test_content_shorter_than_limit_is_handled(name: str, factory) -> None:
    """Summarizer can still produce a result even below the limit
    (caller decides whether to invoke it)."""
    s = factory()
    # Force a matching file — use .py for python, .java for java, etc.
    # For simplicity, just ensure no exception is raised.
    result = s(VirtualPath("/test.txt"), "small content", 1_000_000)
    assert isinstance(result, str)


def test_line_numbers_are_positive() -> None:
    """All line numbers referenced in a summary should be >= 1."""
    code = (
        "class A:\n"
        + "    pass\n" * 100
    )
    s = python_summarizer()
    result = s(VirtualPath("/test.py"), code, 500)
    # Extract L-prefixed numbers
    import re
    nums = [int(m.group(1)) for m in re.finditer(r"L(\d+)", result)]
    assert len(nums) > 0, "no line numbers found"
    assert all(n >= 1 for n in nums), f"invalid line numbers: {nums}"
