#!/usr/bin/env python3
"""Check relative Markdown links inside a book's src/ tree.

Usage:
    python3 scripts/check_links.py books/embodied-ai/src

Links to http(s), mailto and pure #anchors are skipped, and anchors are not
checked. A link to a directory (ending in /) resolves to its README.md.
Exit status is 1 when any relative target is missing.
"""

from __future__ import annotations

import re
import sys
import urllib.parse
from pathlib import Path

LINK = re.compile(r'\]\(([^)\s]+)(?:\s+"[^"]*")?\)')
FENCE = re.compile(r"^(```|~~~).*?^\1", re.M | re.S)
CODE_SPAN = re.compile(r"`[^`\n]*`")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    root = Path(argv[1]).resolve()
    checked = 0
    broken: list[tuple[Path, str]] = []
    for md in sorted(root.rglob("*.md")):
        text = md.read_text(encoding="utf-8")
        text = CODE_SPAN.sub("", FENCE.sub("", text))
        for match in LINK.finditer(text):
            target = match.group(1)
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            path = urllib.parse.unquote(target.split("#", 1)[0])
            if not path:
                continue
            checked += 1
            dest = md.parent / path
            if path.endswith("/"):
                dest = dest / "README.md"
            if not dest.exists():
                broken.append((md.relative_to(root), target))
    for source, target in broken:
        print(f"{source}: broken link -> {target}")
    print(f"checked {checked} relative links, broken {len(broken)}")
    return 1 if broken else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
