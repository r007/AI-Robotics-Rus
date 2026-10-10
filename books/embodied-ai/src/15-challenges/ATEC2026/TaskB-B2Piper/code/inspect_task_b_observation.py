#!/usr/bin/env python3
"""Inspect a redacted Task B observation JSON without importing the simulator.

Input: a JSON file containing an observation-like mapping.
Output: a stable tree of keys, Python types, and list shapes.
The tool intentionally does not read simulator ground truth or model checkpoints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def shape(value: Any) -> list[int]:
    result: list[int] = []
    current = value
    while isinstance(current, list):
        result.append(len(current))
        if not current:
            break
        current = current[0]
    return result


def walk(value: Any, path: str = "obs") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(value, dict):
        rows.append({"path": path, "type": "dict", "keys": sorted(value)})
        for key in sorted(value):
            rows.extend(walk(value[key], f"{path}.{key}"))
    elif isinstance(value, list):
        rows.append({"path": path, "type": "list", "shape": shape(value)})
        if value and isinstance(value[0], (dict, list)):
            rows.extend(walk(value[0], f"{path}[0]"))
    else:
        rows.append({"path": path, "type": type(value).__name__})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    print(json.dumps(walk(payload), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
