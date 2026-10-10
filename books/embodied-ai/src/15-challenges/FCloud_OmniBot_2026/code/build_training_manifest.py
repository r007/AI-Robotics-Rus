#!/usr/bin/env python3
"""Build a JSONL manifest from logged episodes for later fine-tuning conversion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "notes" / "generated" / "training_manifest.jsonl"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def episode_record(episode_dir: Path) -> dict:
    meta = _json(episode_dir / "meta.json")
    summary = _json(episode_dir / "summary.json")
    readiness = _json(episode_dir / "submission_readiness.json")
    step_count = 0
    steps_path = episode_dir / "steps.jsonl"
    if steps_path.exists():
        step_count = sum(1 for line in steps_path.read_text(encoding="utf-8").splitlines() if line.strip())
    return {
        "episode_dir": str(episode_dir.resolve()),
        "episode_id": meta.get("episode_id", episode_dir.name),
        "task_name": meta.get("task_name"),
        "track": meta.get("track"),
        "robot": meta.get("robot"),
        "policy": meta.get("policy"),
        "step_count": step_count,
        "ready_for_submission": readiness.get("ready", False),
        "summary_status": summary.get("status"),
        "has_score_estimate": (episode_dir / "score_estimate.json").exists(),
        "has_video": bool(list(episode_dir.glob("*.mp4"))),
        "use_for_training": bool(meta.get("use_for_training", False)),
        "not_for_submission": bool(meta.get("not_for_submission", False)),
        "exclude_reason": meta.get("exclude_reason", "not_official_real_episode_until_reviewed"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("episode_roots", nargs="*", type=Path, default=[ROOT / "logs"])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    episode_dirs: list[Path] = []
    for root in args.episode_roots:
        if (root / "meta.json").exists():
            episode_dirs.append(root)
        elif root.exists():
            episode_dirs.extend(path for path in root.rglob("meta.json") if path.parent.is_dir())
    records = [episode_record(path.parent if path.name == "meta.json" else path) for path in episode_dirs]
    records.sort(key=lambda item: str(item["episode_dir"]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(args.out)
    print(f"episodes={len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
