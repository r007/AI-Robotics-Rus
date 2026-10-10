#!/usr/bin/env python3
"""Run the local FCloud OmniBot tabletop self-evaluation.

This is the canonical local/material-submission path for OmniBot tabletop
sorting. It scores final object transforms against the public +/-5 cm /
+/-10 deg rule and then runs the submission-readiness gate. It is not a
GenManip runner.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET_PLAN = ROOT / "notes" / "generated" / "tabletop_target_plan.json"

sys.path.insert(0, str(ROOT / "src"))

from omnibot_agent.scoring import write_score  # noqa: E402
from omnibot_agent.tabletop_eval import evaluate_tabletop_transforms  # noqa: E402
from omnibot_agent.tabletop_eval import transforms_from_json  # noqa: E402


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _object_count(final_transforms: dict[str, Any]) -> int:
    objects = final_transforms.get("objects", final_transforms)
    if isinstance(objects, dict):
        return len(objects)
    if isinstance(objects, list):
        return len(objects)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("episode_dir", type=Path)
    parser.add_argument("--target-plan", type=Path, default=DEFAULT_TARGET_PLAN)
    parser.add_argument("--elapsed-s", type=float, default=None)
    parser.add_argument("--used-replicator-randomization", action="store_true")
    parser.add_argument("--transformer-visible-count", type=int, default=None)
    parser.add_argument("--require-video", action="store_true")
    parser.add_argument("--min-tabletop-score", type=int, default=10)
    parser.add_argument("--strict", action="store_true", help="Return non-zero when readiness fails.")
    args = parser.parse_args()

    episode_dir = args.episode_dir.resolve()
    final_path = episode_dir / "final_transforms.json"
    if not final_path.exists():
        print(f"Missing {final_path}", file=sys.stderr)
        return 2

    final_json = _load_json(final_path)
    target_plan = _load_json(args.target_plan)
    meta = _load_json(episode_dir / "meta.json")
    summary = _load_json(episode_dir / "summary.json")

    elapsed_s = args.elapsed_s
    if elapsed_s is None:
        elapsed_s = float(summary.get("elapsed_s", meta.get("elapsed_s", 600.0)))

    used_randomization = bool(
        args.used_replicator_randomization
        or summary.get("used_replicator_randomization")
        or meta.get("used_replicator_randomization")
    )
    visible_count = args.transformer_visible_count
    if visible_count is None:
        visible_count = int(summary.get("transformer_visible_count", _object_count(final_json)))

    score = evaluate_tabletop_transforms(
        target_plan=target_plan,
        final_transforms=transforms_from_json(final_json),
        elapsed_s=elapsed_s,
        used_replicator_randomization=used_randomization,
        transformer_visible_count=visible_count,
    )
    score_path = episode_dir / "score_estimate.json"
    write_score(score_path, score)

    readiness_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "submission_readiness.py"),
        str(episode_dir),
        "--min-tabletop-score",
        str(args.min_tabletop_score),
    ]
    if args.require_video:
        readiness_cmd.append("--require-video")
    readiness = subprocess.run(readiness_cmd, cwd=str(ROOT), check=False)

    print(
        json.dumps(
            {
                "episode_dir": str(episode_dir),
                "score_estimate": str(score_path),
                "score": score.score,
                "max_score": score.max_score,
                "used_replicator_randomization": used_randomization,
                "transformer_visible_count": visible_count,
                "readiness": str(episode_dir / "submission_readiness.json"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return readiness.returncode if args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
