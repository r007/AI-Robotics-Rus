#!/usr/bin/env python3
"""mdBook preprocessor: group per-framework code blocks into tabs.

The book shows the same code for MXNet, PyTorch, TensorFlow and PaddlePaddle.
In the Markdown sources each variant is a bold label line followed by a code
block:

    **MXNet:**

    ```python
    ...
    ```

    **PyTorch:**

    ```python
    ...
    ```

Two or more such pairs in a row become one group of panels. Together with
theme/framework-tabs.css and theme/framework-tabs.js the reader sees one
framework at a time and the choice is remembered. Without JavaScript every
panel stays visible with its label, exactly as in the sources.

Labels followed by prose instead of code are left as they are: the sources do
not mark where such a variant ends.

Registered in book.toml:

    [preprocessor.framework-tabs]
    command = "python3 ../../scripts/mdbook_framework_tabs.py"
"""

from __future__ import annotations

import html
import json
import re
import sys

FRAMEWORK = r"(?:MXNet|PyTorch|TensorFlow|PaddlePaddle)"
LABEL = re.compile(rf"^\*\*(?P<names>{FRAMEWORK}(?:, {FRAMEWORK})*):\*\*\s*$")
FENCE = re.compile(r"^(?:```|~~~)")


def next_nonblank(lines: list[str], index: int) -> int:
    while index < len(lines) and not lines[index].strip():
        index += 1
    return index


def read_fences(lines: list[str], index: int) -> tuple[list[str], int] | None:
    """Read one or more consecutive fenced blocks starting at index."""
    if index >= len(lines) or not FENCE.match(lines[index]):
        return None
    block: list[str] = []
    while index < len(lines) and FENCE.match(lines[index]):
        start = index
        index += 1
        while index < len(lines) and not FENCE.match(lines[index]):
            index += 1
        if index >= len(lines):
            return None  # unterminated fence: leave the text alone
        index += 1
        block.extend(lines[start:index])
        after = next_nonblank(lines, index)
        if after < len(lines) and FENCE.match(lines[after]):
            block.append("")
            index = after
        else:
            break
    return block, index


def read_group(lines: list[str], index: int):
    panels = []
    while index < len(lines):
        match = LABEL.match(lines[index])
        if not match:
            break
        fences = read_fences(lines, next_nonblank(lines, index + 1))
        if fences is None:
            break
        code, end = fences
        panels.append((match.group("names"), code))
        index = next_nonblank(lines, end)
    return panels, index


def render(panels) -> list[str]:
    out = ['<div class="fw-tabs">', ""]
    for names, code in panels:
        data = html.escape(names.replace(", ", " "), quote=True)
        out += [
            f'<div class="fw-panel" data-fw="{data}">',
            f'<p class="fw-label"><strong>{html.escape(names)}:</strong></p>',
            "",
            *code,
            "",
            "</div>",
            "",
        ]
    out += ["</div>", ""]
    return out


def convert(text: str) -> str:
    lines = text.split("\n")
    out: list[str] = []
    index, in_fence = 0, False
    while index < len(lines):
        line = lines[index]
        if FENCE.match(line):
            in_fence = not in_fence
        elif not in_fence and LABEL.match(line):
            panels, end = read_group(lines, index)
            if len(panels) >= 2:
                out.extend(render(panels))
                index = end
                continue
        out.append(line)
        index += 1
    return "\n".join(out)


def walk(items: list) -> None:
    for item in items:
        chapter = item.get("Chapter") if isinstance(item, dict) else None
        if chapter:
            chapter["content"] = convert(chapter["content"])
            walk(chapter.get("sub_items", []))


def main(argv: list[str]) -> int:
    if len(argv) >= 3 and argv[1] == "supports":
        return 0 if argv[2] == "html" else 1
    _context, book = json.load(sys.stdin)
    walk(book.get("items") or book.get("sections") or [])
    json.dump(book, sys.stdout, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
