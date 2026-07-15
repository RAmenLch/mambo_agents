"""POSIX glob → regex translation (shared by grep glob filtering across backends)."""

from __future__ import annotations


def translate_glob(pattern: str) -> "re.Pattern[str]":
    """Compile a glob pattern into a regex (pathlib-compatible, posix-style).

    Semantics (matched against the path *relative* to the search root):

    * ``**`` matches any number of path segments (including zero)
    * ``*`` matches any chars except the path separator ``/``
    * ``?`` matches a single char except ``/``
    * ``[...]`` / ``[!...]`` character classes
    """
    import re

    i, n = 0, len(pattern)
    res: list[str] = []
    while i < n:
        c = pattern[i]
        i += 1
        if c == "*":
            if i < n and pattern[i] == "*":
                i += 1
                if i < n and pattern[i] == "/":
                    i += 1
                    res.append("(?:.*/)?")
                else:
                    res.append(".*")
            else:
                res.append("[^/]*")
        elif c == "?":
            res.append("[^/]")
        elif c == "[":
            j = i
            if j < n and pattern[j] in ("!", "^"):
                j += 1
            if j < n and pattern[j] == "]":
                j += 1
            while j < n and pattern[j] != "]":
                j += 1
            if j >= n:
                res.append(re.escape("["))
            else:
                stuff = pattern[i:j].replace("\\", "\\\\")
                i = j + 1
                if stuff.startswith("!"):
                    stuff = "^" + stuff[1:]
                res.append("[" + stuff + "]")
        else:
            res.append(re.escape(c))
    return re.compile("(?s:" + "".join(res) + r")\Z")


def fnmatch_path(path: str, glob_pattern: str) -> bool:
    """Test whether *path* matches a POSIX-style *glob_pattern*.

    *path* must use ``/`` separators.  Returns ``True`` if the
    entire path matches the pattern.
    """
    return bool(translate_glob(glob_pattern).match(path))
