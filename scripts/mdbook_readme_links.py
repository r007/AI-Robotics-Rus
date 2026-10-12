#!/usr/bin/env python3
"""mdBook preprocessor: point links to README.md at index.md.

mdBook renders a folder's README.md as index.html, but it rewrites only the
sidebar entries. A link such as [text](dir/README.md) in a page body becomes
dir/README.html, which does not exist. The books keep README.md links in their
sources so that they also work on GitHub; this preprocessor makes them work in
the built book.

Registered in each book.toml:

    [preprocessor.readme-links]
    command = "python3 ../../scripts/mdbook_readme_links.py"
"""

from __future__ import annotations

import json
import re
import sys

# [text](target) and ![alt](target), with an optional "title"
MD_LINK = re.compile(r'(\]\()((?:[^)\s]*/)?README\.md)(#[^)\s]*)?((?:\s+"[^"]*")?\))')
# <a href="target"> and <img src="target">
HTML_ATTR = re.compile(r'''(\b(?:href|src)=["'])((?:[^"']*/)?README\.md)(#[^"']*)?(["'])''')
EXTERNAL = re.compile(r"^(?:[a-zA-Z][a-zA-Z0-9+.-]*:|//)")
# Fenced code blocks and inline code are left alone.
PROTECTED = re.compile(r"(^(?:```|~~~).*?^(?:```|~~~)[^\n]*$|`[^`\n]+`)", re.M | re.S)


def fix_text(text: str) -> str:
    def md(match: re.Match[str]) -> str:
        target = match.group(2)
        if EXTERNAL.match(target):
            return match.group(0)
        return f"{match.group(1)}{target[: -len('README.md')]}index.md{match.group(3) or ''}{match.group(4)}"

    def html(match: re.Match[str]) -> str:
        target = match.group(2)
        if EXTERNAL.match(target):
            return match.group(0)
        return f"{match.group(1)}{target[: -len('README.md')]}index.html{match.group(3) or ''}{match.group(4)}"

    pieces = PROTECTED.split(text)
    for index in range(0, len(pieces), 2):
        pieces[index] = HTML_ATTR.sub(html, MD_LINK.sub(md, pieces[index]))
    return "".join(pieces)


def walk(items: list) -> None:
    for item in items:
        chapter = item.get("Chapter") if isinstance(item, dict) else None
        if chapter:
            chapter["content"] = fix_text(chapter["content"])
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
