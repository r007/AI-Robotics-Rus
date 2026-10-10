#!/usr/bin/env python3
"""Create a deterministic SHA256 manifest for a public release directory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


FORBIDDEN_SUFFIXES = {
    ".ckpt",
    ".h5",
    ".hdf5",
    ".pkl",
    ".pth",
    ".safetensors",
    ".mp4",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")

    files = []
    rejected = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if path.is_symlink() or path.suffix.lower() in FORBIDDEN_SUFFIXES:
            rejected.append(str(relative))
            continue
        files.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
        )

    payload = {
        "status": "PASS" if not rejected else "REJECT",
        "root_label": root.name,
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "files": files,
        "rejected_paths": rejected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "file_count": len(files), "rejected": len(rejected)}))
    return 0 if not rejected else 1


if __name__ == "__main__":
    raise SystemExit(main())
