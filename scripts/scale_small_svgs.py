#!/usr/bin/env python3
"""Enlarge tiny matplotlib figures so they are readable in mdBook.

The "margin" figures of the Quarto original (``*_margin_*.svg`` and a few
figures with other names) were drawn for a narrow side column and carry an
absolute size of about one inch (72-130 pt, i.e. 100-170 px). mdBook has no
margin column and shows an <img> at its intrinsic size, so such a figure is a
postage stamp with 5 px labels.

The script rewrites ``width`` and ``height`` on the root <svg> so that one
viewBox unit is SCALE px wide; labels then come out at body-text size. The
viewBox and the drawing are not touched. Files that already scale with the
column (``width="100%"`` or no width at all) and files that are big enough are
skipped, so a second run changes nothing.

Usage:
    python3 scripts/scale_small_svgs.py                          # the three MLSys books
    python3 scripts/scale_small_svgs.py books/intro-ml-systems   # given folders or files
    python3 scripts/scale_small_svgs.py --check                  # report only, exit 1 if anything would change
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Books whose figures come from the Quarto original (matplotlib output with absolute sizes).
# dive-into-deep-learning is left out on purpose: its diagrams are drawn at their native size.
DEFAULT_BOOKS = ("agentic-ml-systems", "intro-ml-systems", "physical-ai-systems")

SCALE = 4.0  # px per viewBox unit (1 unit = 1 pt in matplotlib, so this is 3x the old size)
MIN_WIDTH = 300.0  # px; wider figures are left as they are

UNITS_PX = {"": 1.0, "px": 1.0, "pt": 96 / 72, "pc": 16.0, "in": 96.0, "cm": 96 / 2.54, "mm": 96 / 25.4}

ROOT_TAG = re.compile(r"<svg\b[^>]*>", re.S)
LENGTH = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([a-z]*)\s*$")


def to_px(value: str | None) -> float | None:
    """Absolute CSS length in px, or None for a missing, relative (%, em) or unparsable one."""
    if value is None:
        return None
    match = LENGTH.match(value)
    if not match or match.group(2) not in UNITS_PX:
        return None
    return float(match.group(1)) * UNITS_PX[match.group(2)]


def attr(tag: str, name: str) -> str | None:
    match = re.search(rf'\s{name}="([^"]*)"', tag)
    return match.group(1) if match else None


def fmt(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def rescale(text: str) -> str | None:
    """Return the file text with a larger root size, or None if it needs no change."""
    match = ROOT_TAG.search(text)
    if not match:
        return None
    tag = match.group(0)
    width, height = to_px(attr(tag, "width")), to_px(attr(tag, "height"))
    box = (attr(tag, "viewBox") or "").replace(",", " ").split()
    if width is None or height is None or len(box) != 4:
        return None
    try:
        box_w, box_h = float(box[2]), float(box[3])
    except ValueError:
        return None
    # the 0.01 slack absorbs the two-decimal rounding of an already scaled file
    if box_w <= 0 or box_h <= 0 or width >= MIN_WIDTH or width / box_w >= SCALE - 0.01:
        return None
    new_tag = re.sub(r'(\swidth=")[^"]*(")', rf"\g<1>{fmt(box_w * SCALE)}\g<2>", tag, count=1)
    new_tag = re.sub(r'(\sheight=")[^"]*(")', rf"\g<1>{fmt(box_h * SCALE)}\g<2>", new_tag, count=1)
    return text[: match.start()] + new_tag + text[match.end() :]


def svg_files(targets: list[Path]):
    for target in targets:
        if target.is_file():
            yield target
        else:
            yield from sorted(target.rglob("*.svg"))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", type=Path, help="book folders or SVG files (default: the three MLSys books)")
    parser.add_argument("--check", action="store_true", help="do not write; exit 1 if any file would change")
    args = parser.parse_args(argv)

    targets = args.paths or [ROOT / "books" / slug / "src" for slug in DEFAULT_BOOKS]
    changed = 0
    for path in svg_files(targets):
        text = path.read_bytes().decode("utf-8")  # bytes, not read_text: keep line endings as they are
        new_text = rescale(text)
        if new_text is None:
            continue
        changed += 1
        print(f"{'would scale' if args.check else 'scaled'}: {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
        if not args.check:
            path.write_bytes(new_text.encode("utf-8"))
    print(f"{changed} file(s) {'need scaling' if args.check else 'changed'}")
    return 1 if args.check and changed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
