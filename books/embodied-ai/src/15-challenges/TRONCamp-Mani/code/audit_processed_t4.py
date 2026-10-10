#!/usr/bin/env python3
"""Strictly audit an ACT-style processed HDF5 dataset.

This public utility deliberately does not know any private workspace path and
does not repair data. It requires h5py and numpy in the user's own environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


CAMERAS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--expected", type=int, default=550)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--raw-manifest-sha256",
        help="Optional expected SHA256 of the separately stored raw manifest.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    expected_names = {f"episode_{index}.hdf5" for index in range(args.expected)}
    paths = sorted(args.dataset_dir.glob("episode_*.hdf5"))
    actual_names = {path.name for path in paths}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise SystemExit(f"file-set mismatch: missing={missing[:10]} extra={extra[:10]}")

    rows = []
    max_next_error = 0.0
    for index in range(args.expected):
        path = args.dataset_dir / f"episode_{index}.hdf5"
        with h5py.File(path, "r") as root:
            action = np.asarray(root["/action"])
            qpos = np.asarray(root["/observations/qpos"])
            if action.ndim != 2 or qpos.ndim != 2:
                raise SystemExit(f"rank error: {path.name}")
            if action.shape != qpos.shape or action.shape[1] != 16:
                raise SystemExit(f"shape error: {path.name}: {action.shape} {qpos.shape}")
            if not np.isfinite(action).all() or not np.isfinite(qpos).all():
                raise SystemExit(f"non-finite action/qpos: {path.name}")

            left_dim = np.asarray(root["/observations/left_arm_dim"])
            right_dim = np.asarray(root["/observations/right_arm_dim"])
            if not np.all(left_dim == 7) or not np.all(right_dim == 7):
                raise SystemExit(f"arm metadata error: {path.name}")

            images = root["/observations/images"]
            cameras = sorted(images.keys())
            if cameras != CAMERAS:
                raise SystemExit(f"camera error in {path.name}: {cameras}")
            lengths = {images[name].shape[0] for name in CAMERAS}
            if lengths != {action.shape[0]}:
                raise SystemExit(f"camera length error: {path.name}: {lengths}")

            next_error = (
                float(np.max(np.abs(action[:-1] - qpos[1:])))
                if len(action) > 1
                else 0.0
            )
            if next_error != 0.0:
                raise SystemExit(f"next-state alignment error in {path.name}: {next_error}")
            max_next_error = max(max_next_error, next_error)
            rows.append(
                {
                    "episode": index,
                    "frames": int(action.shape[0]),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "next_state_max_error": next_error,
                }
            )

    payload = {
        "status": "PASS",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_label": "user_supplied_processed_act_dataset",
        "totals": {
            "episodes": len(rows),
            "bytes": sum(row["bytes"] for row in rows),
            "min_frames": min(row["frames"] for row in rows),
            "max_frames": max(row["frames"] for row in rows),
            "state_dim": 16,
            "action_dim": 16,
            "camera_names": CAMERAS,
            "next_state_max_error": max_next_error,
        },
        "raw_manifest_sha256": args.raw_manifest_sha256,
        "episodes": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", **payload["totals"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
