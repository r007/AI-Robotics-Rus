#!/usr/bin/env python3
"""Merge every book under books/ into one mdBook and build it into site/.

A book is a directory in books/ that has a book.toml. The build:

1. links each book's src/ tree into build/src/<book>/ (hard links, falling
   back to copies), so the books keep their own files and nothing is moved;
2. merges each book's SUMMARY.md into build/src/SUMMARY.md, rewriting the
   links to point at the book's folder;
3. copies directory/ into build/src/, so its README.md is the front page;
4. runs `mdbook build` with the root book.toml, writing to site/.

Usage:
    python3 scripts/build_site.py              # assemble and build
    python3 scripts/build_site.py --assemble   # only assemble build/src
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BOOKS = ROOT / "books"
DIRECTORY = ROOT / "directory"
BUILD = ROOT / "build"
SRC = BUILD / "src"
ASSETS = BUILD / "assets"

# The mermaid runtime lives with the embodied-ai book and is shared by all books.
MERMAID_SOURCE = BOOKS / "embodied-ai"
MERMAID_FILES = ["mermaid.min.js", "mermaid-init.js"]

# Markdown link target: [text](target). External, anchor and absolute targets are left alone.
LINK = re.compile(r"\]\(([^)\s]+)\)")
EXTERNAL = re.compile(r"^(?:[a-zA-Z][a-zA-Z0-9+.-]*:|#|/)")


def read_book_config(book_dir: Path) -> dict:
    with (book_dir / "book.toml").open("rb") as handle:
        return tomllib.load(handle)["book"]


def is_skipped(relative: Path) -> bool:
    return any(part.startswith(".") or part == "__pycache__" for part in relative.parts)


def link_tree(src: Path, dst: Path, skip_names: set[str], transform=None) -> int:
    """Link every file from src into dst. Markdown goes through transform when given."""
    count = 0
    for path in sorted(src.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(src)
        if is_skipped(relative) or path.name in skip_names:
            continue
        target = dst / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if transform and path.suffix == ".md":
            target.write_text(transform(path.read_text(encoding="utf-8")), encoding="utf-8")
        else:
            try:
                os.link(path, target)
            except OSError:
                shutil.copy2(path, target)
        count += 1
    return count


def rewrite_links(text: str, prefix: str) -> str:
    def replace(match: re.Match[str]) -> str:
        target = match.group(1)
        if EXTERNAL.match(target):
            return match.group(0)
        return f"]({prefix}/{target})"

    return LINK.sub(replace, text)


def point_readmes_at_index(text: str) -> str:
    """mdBook renders a folder's README.md as index.html, so links must point there."""

    def replace(match: re.Match[str]) -> str:
        target = match.group(1)
        if EXTERNAL.match(target):
            return match.group(0)
        return f"]({re.sub(r'README\.md(?=#|$)', 'index.html', target)})"

    return LINK.sub(replace, text)


def book_summary(book_dir: Path, slug: str, title: str) -> str:
    """Return one book's SUMMARY entries under a part heading, with links rebased."""
    summary = (book_dir / "src" / "SUMMARY.md").read_text(encoding="utf-8").strip().splitlines()
    # The book's own "# Содержание" title would clash with the merged summary's structure.
    if summary and summary[0].startswith("# "):
        summary = summary[1:]
    body = rewrite_links("\n".join(summary).strip(), slug)
    return f"# {title}\n\n{body}\n"


def assemble() -> list[str]:
    if BUILD.exists():
        shutil.rmtree(BUILD)
    SRC.mkdir(parents=True)

    books = sorted(
        path for path in BOOKS.iterdir() if path.is_dir() and (path / "book.toml").is_file()
    )
    parts = ["[Каталог книг](README.md)", ""]
    slugs = []
    for book_dir in books:
        config = read_book_config(book_dir)
        slug = book_dir.name
        src = book_dir / config.get("src", "src")
        if not (src / "SUMMARY.md").is_file():
            sys.exit(f"error: {src / 'SUMMARY.md'} is missing")
        files = link_tree(src, SRC / slug, skip_names={"SUMMARY.md"}, transform=point_readmes_at_index)
        parts.append(book_summary(book_dir, slug, config["title"]))
        slugs.append(slug)
        print(f"{slug}: {files} files")

    if DIRECTORY.is_dir():
        link_tree(DIRECTORY, SRC, skip_names=set(), transform=point_readmes_at_index)
    else:
        sys.exit(f"error: {DIRECTORY} is missing")

    (SRC / "SUMMARY.md").write_text("\n".join(parts).rstrip() + "\n", encoding="utf-8")

    ASSETS.mkdir(parents=True)
    for name in MERMAID_FILES:
        shutil.copy2(MERMAID_SOURCE / name, ASSETS / name)
    return slugs


def build() -> None:
    subprocess.run(["mdbook", "build"], cwd=ROOT, check=True)


def main(argv: list[str]) -> int:
    slugs = assemble()
    print(f"assembled {len(slugs)} books into {SRC}")
    if "--assemble" not in argv:
        build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
