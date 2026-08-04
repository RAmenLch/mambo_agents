"""Tree display: TreeEntry model + formatter."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class TreeEntry(BaseModel):
    """A single entry in a directory tree display.

    Attributes:
        name: Display name (e.g. ``"subdir/"``, ``"file.txt (1 KB)"``).
        depth: Nesting depth (0 = root level).
        marker: Optional status marker for directories.
            ``""`` = normal, ``"empty"`` → ``/(empty)``,
            ``"ignore"`` → ``/(ignore)``,
            ``"depth_exceeded"`` → ``/(...)``.
    """

    name: str
    depth: int
    marker: Literal["", "empty", "ignore", "depth_exceeded"] = ""


def format_tree_entries(
    entries: list[TreeEntry],
) -> str:
    """Render ``TreeEntry`` list as a visual directory tree.

    The tree-walk logic is backend-specific; this function provides the
    shared visual formatting regardless of which backend produces the entries.

    Directory markers are appended to the display name:
    - ``/(empty)`` — directory has no children
    - ``/(ignore)`` — directory whose name is in ignore_dirs (children hidden)
    - ``/(...)`` — depth limit reached, children not shown
    """
    if not entries:
        return "(empty)"

    _MARKER_SUFFIX: dict[str, str] = {
        "empty": "/(empty)",
        "ignore": "/(ignore)",
        "depth_exceeded": "/(...)",
    }

    lines: list[str] = []
    for i, entry in enumerate(entries):
        marker_suffix = _MARKER_SUFFIX.get(entry.marker, "")
        if marker_suffix:
            display = entry.name.rstrip("/") + marker_suffix
        else:
            display = entry.name
        depth = entry.depth

        # Determine connector: look ahead to see if there are siblings at the same depth
        has_more_siblings = False
        for j in range(i + 1, len(entries)):
            next_depth = entries[j].depth
            if next_depth < depth:
                break
            if next_depth == depth:
                has_more_siblings = True
                break

        connector = "├── " if has_more_siblings else "└── "
        if depth == 0:
            lines.append(display)
        else:
            prefix_parts: list[str] = []
            for level in range(1, depth):  # parent levels only; own level uses connector
                active = False
                for j in range(i + 1, len(entries)):
                    if entries[j].depth < level:
                        break
                    if entries[j].depth == level:
                        active = True
                        break
                prefix_parts.append("│   " if active else "    ")
            prefix = "".join(prefix_parts)
            lines.append(f"{prefix}{connector}{display}")

    return "\n".join(lines)
