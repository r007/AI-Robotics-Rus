#!/usr/bin/env python3
"""Check that every folder in books/ has the layout the site and the directory page need.

Each book is a folder in books/ with these files:

    book.toml       mdBook metadata: [book] title, description, authors, language, src
    meta.toml       directory metadata: order, status, status_label, summary, license, [source]
    NOTICE.md       authorship, source and license of the text
    src/README.md   first page of the book
    src/SUMMARY.md  table of contents

If meta.toml says summary = "generated", src/SUMMARY.md must match what
scripts/make_summary.py produces. Books with summary = "manual" keep their own.

Usage:
    python3 scripts/check_books.py
Exit status is 1 when a book breaks the layout.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from make_summary import build

ROOT = Path(__file__).resolve().parent.parent
BOOKS = ROOT / "books"

REQUIRED_FILES = ("book.toml", "meta.toml", "NOTICE.md", "src/README.md", "src/SUMMARY.md")
BOOK_KEYS = ("title", "description", "authors", "language")
META_KEYS = ("order", "status", "status_label", "summary", "license", "source")
SOURCE_KEYS = ("title", "author", "license")
STATUSES = ("in-progress", "draft", "complete")
SUMMARY_MODES = ("generated", "manual")


def load(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def check_book(book: Path) -> list[str]:
    problems = [f"missing {name}" for name in REQUIRED_FILES if not (book / name).is_file()]
    if problems:
        return problems

    config = load(book / "book.toml").get("book", {})
    problems += [f"book.toml: [book] has no {key}" for key in BOOK_KEYS if key not in config]
    if config.get("src", "src") != "src":
        problems.append('book.toml: src must be "src"')

    meta = load(book / "meta.toml")
    problems += [f"meta.toml: missing {key}" for key in META_KEYS if key not in meta]
    if not isinstance(meta.get("order"), int):
        problems.append("meta.toml: order must be an integer")
    if meta.get("status") not in STATUSES:
        problems.append(f"meta.toml: status must be one of {', '.join(STATUSES)}")
    mode = meta.get("summary")
    if mode not in SUMMARY_MODES:
        problems.append(f"meta.toml: summary must be one of {', '.join(SUMMARY_MODES)}")
    source = meta.get("source", {})
    problems += [f"meta.toml: [source] has no {key}" for key in SOURCE_KEYS if key not in source]

    if mode == "generated":
        expected = build(book / "src")
        actual = (book / "src" / "SUMMARY.md").read_text(encoding="utf-8")
        if actual != expected:
            problems.append(f"src/SUMMARY.md is out of date; run scripts/make_summary.py books/{book.name}")
    return problems


def main() -> int:
    failed = 0
    seen_orders: dict[object, str] = {}
    for book in sorted(path for path in BOOKS.iterdir() if path.is_dir()):
        problems = check_book(book)
        if (book / "meta.toml").is_file():
            order = load(book / "meta.toml").get("order")
            if order in seen_orders:
                problems.append(f"order {order} is also used by {seen_orders[order]}")
            seen_orders[order] = book.name
        if problems:
            failed += 1
            for problem in problems:
                print(f"{book.name}: {problem}")
        else:
            print(f"{book.name}: ok")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
