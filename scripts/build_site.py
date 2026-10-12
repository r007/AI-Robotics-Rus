#!/usr/bin/env python3
"""Build the site: every book as its own mdBook plus the directory page.

A book is a folder in books/ that has a book.toml and a meta.toml. The build:

1. runs `mdbook build books/<slug>` for every book, writing site/<slug>/;
2. renders directory/index.html into site/index.html: one card per book,
   filled from the book's book.toml and meta.toml;
3. copies covers/<slug>.svg into site/covers/.

Each book is also a normal mdBook, so `mdbook serve books/<slug>` works on its own.

Usage:
    python3 scripts/build_site.py                      # all books and the directory page
    python3 scripts/build_site.py --book embodied-ai   # only the named book(s), then the directory page
    python3 scripts/build_site.py --directory-only     # only the directory page (no mdBook needed)
    python3 scripts/build_site.py --out /tmp/site      # another output folder
"""

from __future__ import annotations

import argparse
import html
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BOOKS = ROOT / "books"
DIRECTORY = ROOT / "directory"
COVERS = ROOT / "covers"

# Short labels for the "С чего начать" block, by book slug.
START_HINTS = {
    "dive-into-deep-learning": "Вы только начинаете с машинного обучения и хотите понять нейронные сети с кодом.",
    "embodied-ai": "Вас интересуют роботы, модели VLA и симуляция, и нужен маршрут от основ до практики.",
    "physical-ai-systems": "Вы строите системы, в которых обученная модель управляет железом, и вас волнует безопасность.",
}


@dataclass
class Book:
    slug: str
    title: str
    tagline: str
    why: str
    order: int
    status: str
    status_label: str
    license: str
    accent: str
    tags: list[str]
    learn: list[str]
    source_title: str
    source_author: str
    source_url: str

    @property
    def path(self) -> Path:
        return BOOKS / self.slug


def load_books() -> list[Book]:
    books = []
    for folder in sorted(BOOKS.iterdir()):
        if not (folder / "book.toml").is_file():
            continue
        with (folder / "book.toml").open("rb") as handle:
            config = tomllib.load(handle)["book"]
        with (folder / "meta.toml").open("rb") as handle:
            meta = tomllib.load(handle)
        source = meta["source"]
        books.append(
            Book(
                slug=folder.name,
                title=config["title"],
                tagline=meta["tagline"],
                why=meta["why"],
                order=meta["order"],
                status=meta["status"],
                status_label=meta["status_label"],
                license=meta["license"],
                accent=meta.get("accent", "#ea580c"),
                tags=list(meta["tags"]),
                learn=list(meta["learn"]),
                source_title=source["title"],
                source_author=source["author"],
                source_url=source.get("url", ""),
            )
        )
    return sorted(books, key=lambda book: book.order)


def esc(value: str) -> str:
    return html.escape(value, quote=True)


def render_card(book: Book) -> str:
    learn = "\n".join(f"              <li>{esc(item)}</li>" for item in book.learn)
    tags = "".join(f"<li>{esc(tag)}</li>" for tag in book.tags)
    source = esc(book.source_title)
    if book.source_url:
        source = f'<a href="{esc(book.source_url)}">{source}</a>'
    original = ""
    if book.source_url:
        original = f'\n              <a class="btn" href="{esc(book.source_url)}">Оригинал</a>'
    return f"""        <article class="card" style="--accent: {esc(book.accent)}">
          <a class="cover" href="{book.slug}/" tabindex="-1" aria-hidden="true"><img src="covers/{book.slug}.svg" alt="" width="800" height="450" loading="lazy"></a>
          <div class="body">
            <p class="status">{esc(book.status_label)}</p>
            <h3><a href="{book.slug}/">{esc(book.title)}</a></h3>
            <p class="tagline">{esc(book.tagline)}</p>
            <p class="why"><strong>Чем ценна.</strong> {esc(book.why)}</p>
            <p class="learn-title">Вы узнаете</p>
            <ul class="learn">
{learn}
            </ul>
            <ul class="tags">{tags}</ul>
            <dl class="meta">
              <div><dt>Оригинал: </dt><dd>{source}, {esc(book.source_author)}</dd></div>
              <div><dt>Лицензия: </dt><dd>{esc(book.license)}</dd></div>
            </dl>
            <div class="actions">
              <a class="btn primary" href="{book.slug}/">Читать онлайн</a>{original}
            </div>
          </div>
        </article>"""


def render_start(book: Book) -> str:
    hint = START_HINTS.get(book.slug, book.tagline)
    return (
        f'        <li style="--accent: {esc(book.accent)}">'
        f'<a href="{book.slug}/"><strong>{esc(book.title)}</strong></a>'
        f"<span>{esc(hint)}</span></li>"
    )


def render_directory(books: list[Book], out: Path) -> None:
    template = (DIRECTORY / "index.html").read_text(encoding="utf-8")
    licenses = "Лицензии книг: " + "; ".join(
        f"{esc(book.title)} — {esc(book.license)}" for book in books
    ) + "."
    page = (
        template.replace("{{START}}", "\n".join(render_start(book) for book in books))
        .replace("{{CARDS}}", "\n".join(render_card(book) for book in books))
        .replace("{{LICENSES}}", licenses)
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(page, encoding="utf-8")
    covers_out = out / "covers"
    covers_out.mkdir(exist_ok=True)
    for book in books:
        cover = COVERS / f"{book.slug}.svg"
        if not cover.is_file():
            sys.exit(f"error: cover {cover.relative_to(ROOT)} is missing")
        shutil.copy2(cover, covers_out / cover.name)
    # GitHub Pages must serve the book folders as they are.
    (out / ".nojekyll").write_text("", encoding="utf-8")
    print(f"directory page: {out / 'index.html'}")


def require_tools(book: Book) -> None:
    needed = ["mdbook", "mdbook-katex"]
    if "mermaid" in (book.path / "book.toml").read_text(encoding="utf-8"):
        needed.append("mdbook-mermaid")
    missing = [tool for tool in needed if shutil.which(tool) is None]
    if missing:
        sys.exit(f"error: {', '.join(missing)} not found; see README.md for the required versions")


def build_book(book: Book, out: Path) -> None:
    require_tools(book)
    destination = out / book.slug
    if destination.exists():
        shutil.rmtree(destination)
    print(f"building {book.slug} ...", flush=True)
    subprocess.run(
        ["mdbook", "build", str(book.path), "--dest-dir", str(destination)],
        cwd=ROOT,
        check=True,
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--book", action="append", default=[], metavar="SLUG", help="build only this book (repeatable)")
    parser.add_argument("--directory-only", action="store_true", help="render only the directory page")
    parser.add_argument("--out", type=Path, default=ROOT / "site", help="output folder (default: site/)")
    args = parser.parse_args(argv[1:])

    books = load_books()
    known = {book.slug for book in books}
    unknown = [slug for slug in args.book if slug not in known]
    if unknown:
        sys.exit(f"error: unknown book(s): {', '.join(unknown)}; known: {', '.join(sorted(known))}")

    out = args.out.resolve()
    if not args.directory_only:
        for book in books:
            if not args.book or book.slug in args.book:
                build_book(book, out)
    render_directory(books, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
