#!/usr/bin/env python3
"""Check whether collected episodes are ready for model fine-tuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "notes" / "generated" / "training_manifest.jsonl"
NON_TRAINING_MARKERS = ("fake", "dryrun", "dry-run", "smoke", "placeholder", "diagnostic", "probe")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _load_manifest(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _has_marker(record: dict) -> bool:
    text = " ".join(str(record.get(key, "")).lower() for key in ("episode_dir", "task_name", "robot", "policy", "summary_status"))
    return any(marker in text for marker in NON_TRAINING_MARKERS)


def _score(record: dict) -> tuple[int | None, int | None]:
    score_path = Path(record["episode_dir"]) / "score_estimate.json"
    data = _load_json(score_path)
    score = data.get("score")
    max_score = data.get("max_score")
    return (score if isinstance(score, int) else None, max_score if isinstance(max_score, int) else None)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--min-episodes", type=int, default=20)
    parser.add_argument("--min-score", type=int, default=10)
    parser.add_argument("--out", type=Path, default=ROOT / "notes" / "generated" / "training_readiness.json")
    args = parser.parse_args()

    records = _load_manifest(args.manifest)
    candidates = []
    rejected = []
    for record in records:
        score, max_score = _score(record)
        ok = (
            bool(record.get("use_for_training", False))
            and not bool(record.get("not_for_submission", False))
            and not _has_marker(record)
            and score is not None
            and max_score == 12
            and score >= args.min_score
            and bool(record.get("has_video", False))
        )
        annotated = {**record, "score": score, "max_score": max_score}
        if ok:
            candidates.append(annotated)
        else:
            rejected.append(annotated)

    report = {
        "manifest": str(args.manifest),
        "ready": len(candidates) >= args.min_episodes,
        "candidate_count": len(candidates),
        "required_candidate_count": args.min_episodes,
        "min_score": args.min_score,
        "candidates": candidates,
        "rejected_count": len(rejected),
        "rejected_sample": rejected[:20],
        "next_action": "collect real official Isaac episodes with video and score >= threshold" if len(candidates) < args.min_episodes else "start fine-tuning",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
