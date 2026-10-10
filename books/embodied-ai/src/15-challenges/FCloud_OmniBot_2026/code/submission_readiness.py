#!/usr/bin/env python3
"""Check whether an episode directory has the artifacts required for submission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_FILES = ("meta.json", "steps.jsonl", "summary.json")
NON_SUBMITTABLE_MARKERS = ("fake", "dryrun", "dry-run", "smoke", "placeholder", "diagnostic", "probe")


def count_step_images(episode_dir: Path) -> int:
    return len(list(episode_dir.glob("step_*_head.png")))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def has_non_submittable_marker(*values: str) -> bool:
    text = " ".join(value.lower() for value in values if value)
    return any(marker in text for marker in NON_SUBMITTABLE_MARKERS)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("episode_dir", type=Path)
    parser.add_argument("--require-video", action="store_true")
    parser.add_argument("--allow-smoke", action="store_true", help="Allow smoke/dry-run episodes for internal package tests.")
    parser.add_argument("--min-tabletop-score", type=int, default=10)
    args = parser.parse_args()

    episode_dir = args.episode_dir.resolve()
    checks: list[tuple[str, bool, str]] = []
    checks.append(("episode_dir_exists", episode_dir.is_dir(), str(episode_dir)))
    for filename in REQUIRED_FILES:
        checks.append((f"has_{filename}", (episode_dir / filename).exists(), filename))

    meta = load_json(episode_dir / "meta.json")
    summary = load_json(episode_dir / "summary.json")
    is_smoke_or_dryrun = has_non_submittable_marker(
        episode_dir.name,
        str(meta.get("task_name", "")),
        str(meta.get("policy", "")),
        str(meta.get("robot", "")),
        str(summary.get("status", "")),
    ) or bool(meta.get("not_for_submission", False))
    checks.append(
        (
            "not_smoke_or_dryrun",
            args.allow_smoke or not is_smoke_or_dryrun,
            "internal smoke/dry-run is not a leaderboard submission",
        )
    )

    steps_path = episode_dir / "steps.jsonl"
    step_count = 0
    if steps_path.exists():
        step_count = sum(1 for line in steps_path.read_text(encoding="utf-8").splitlines() if line.strip())
    checks.append(("has_step_records", step_count > 0, f"steps={step_count}"))
    checks.append(("has_step_images", count_step_images(episode_dir) >= step_count > 0, f"head_images={count_step_images(episode_dir)}"))

    video_files = list(episode_dir.glob("*.mp4"))
    checks.append(("has_video", bool(video_files) or not args.require_video, ",".join(p.name for p in video_files) or "none"))

    score_files = [episode_dir / "score_estimate.json", episode_dir / "summary.json"]
    score_data = {}
    for path in score_files:
        score_data.update(load_json(path))
    checks.append(("has_score_or_summary", bool(score_data), "score_estimate.json or summary.json"))
    is_tabletop = "tabletop" in str(meta.get("task_name", "")).lower() or "tabletop" in episode_dir.name.lower()
    if is_tabletop:
        tabletop_score = score_data.get("score")
        tabletop_max = score_data.get("max_score")
        checks.append(
            (
                "tabletop_score_threshold",
                isinstance(tabletop_score, int)
                and isinstance(tabletop_max, int)
                and tabletop_max == 12
                and tabletop_score >= args.min_tabletop_score,
                f"score={tabletop_score}/{tabletop_max}, required>={args.min_tabletop_score}/12",
            )
        )

    report = {
        "episode_dir": str(episode_dir),
        "ready": all(ok for _, ok, _ in checks),
        "checks": [{"name": name, "ok": ok, "detail": detail} for name, ok, detail in checks],
    }
    out = episode_dir / "submission_readiness.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
