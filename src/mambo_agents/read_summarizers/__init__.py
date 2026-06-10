"""Pluggable ``ReadSummarizer`` callbacks for file-type-aware oversized-read summaries.

Each factory function returns a ``ReadSummarizer`` callable that analyses
oversized file content and produces an instructive summary with accurate
line numbers, helping the LLM decide where to re-read with ``offset``+``limit``.

Usage (single)::

    from mambo_agents.read_summarizers import python_summarizer
    from mambo_agents.backends import LocalBackend

    backend = LocalBackend(summarizer=python_summarizer())

Usage (multiple languages via composite)::

    from mambo_agents.read_summarizers import (
        composite_summarizer,
        python_summarizer,
        java_summarizer,
    )
    from mambo_agents.backends import LocalBackend

    multi = composite_summarizer([python_summarizer(), java_summarizer()])
    backend = LocalBackend(summarizer=multi)

None of these are injected by default — the user selects which ones to use.
"""

from mambo_agents.read_summarizers._python import python_summarizer
from mambo_agents.read_summarizers._javascript import javascript_summarizer
from mambo_agents.read_summarizers._markdown import markdown_summarizer
from mambo_agents.read_summarizers._json import json_summarizer
from mambo_agents.read_summarizers._java import java_summarizer
from mambo_agents.read_summarizers._c import c_summarizer
from mambo_agents.read_summarizers._cpp import cpp_summarizer
from mambo_agents.read_summarizers._go import go_summarizer
from mambo_agents.read_summarizers._rust import rust_summarizer
from mambo_agents.read_summarizers._composite import composite_summarizer

__all__ = [
    "python_summarizer",
    "javascript_summarizer",
    "markdown_summarizer",
    "json_summarizer",
    "java_summarizer",
    "c_summarizer",
    "cpp_summarizer",
    "go_summarizer",
    "rust_summarizer",
    "composite_summarizer",
]
