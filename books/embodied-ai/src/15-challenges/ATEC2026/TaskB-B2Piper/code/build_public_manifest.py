#!/usr/bin/env python3
"""Build a deterministic manifest for a public competition release."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXCLUDED = {".git", "__pycache__", ".pytest_cache"}
MAX_HASH_BYTES = 100 * 1024 * 1024


def sha256(path: Path) -> str | None:
    if path.stat().st_size > MAX_HASH_BYTES:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(part in EXCLUDED for part in path.parts):
            continue
        relative = path.relative_to(root).as_posix()
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    manifest = {
        "format": "every-embodied-public-manifest-v1",
        "root": root.name,
        "files": files,
    }
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(files)} files to {args.output}")


if __name__ == "__main__":
    main()
