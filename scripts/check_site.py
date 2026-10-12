#!/usr/bin/env python3
"""Check internal links and images in a built site.

Usage:
    python3 scripts/check_site.py [site]       # default: site/

Every relative href/src in every HTML file must point to an existing file or
to a folder that has index.html. External links, mailto: links and pure #anchors
are skipped. print.html pages are skipped: they repeat links from other pages. 404.html pages are
skipped: mdBook writes them with absolute site-url paths. HTML files that are not
mdBook pages (saved copies of other sites inside a book) are skipped too.
Exit status is 1 when any target is missing.
"""

from __future__ import annotations

import re
import sys
import urllib.parse
from pathlib import Path

ATTR = re.compile(r'\b(?:href|src)="([^"#][^"]*)"')
SCHEME = re.compile(r"^(?:[a-zA-Z][a-zA-Z0-9+.-]*:|//)")


def main(argv: list[str]) -> int:
    root = Path(argv[1] if len(argv) > 1 else "site").resolve()
    if not root.is_dir():
        print(f"error: {root} is not a directory")
        return 2
    checked = 0
    broken: dict[str, set[str]] = {}
    for page in sorted(root.rglob("*.html")):
        if page.name in ("print.html", "404.html"):
            continue
        text = page.read_text(encoding="utf-8", errors="replace")
        if "mdbook-" not in text:
            continue
        # code samples may show HTML with src="..."; they are not links
        text = re.sub(r"<pre\b.*?</pre>|<code\b.*?</code>", "", text, flags=re.S)
        for match in ATTR.finditer(text):
            target = match.group(1)
            if SCHEME.match(target):
                continue
            path = urllib.parse.unquote(target.split("#", 1)[0].split("?", 1)[0])
            if not path:
                continue
            checked += 1
            dest = (root / path.lstrip("/")) if path.startswith("/") else (page.parent / path)
            if dest.is_dir():
                dest = dest / "index.html"
            if not dest.exists():
                broken.setdefault(str(page.relative_to(root)), set()).add(target)
    for page, targets in sorted(broken.items()):
        for target in sorted(targets):
            print(f"{page}: missing {target}")
    total = sum(len(targets) for targets in broken.values())
    print(f"checked {checked} links, {total} broken")
    return 1 if broken else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
