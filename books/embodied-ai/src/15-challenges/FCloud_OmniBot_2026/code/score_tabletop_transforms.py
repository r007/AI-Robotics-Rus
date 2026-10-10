#!/usr/bin/env python3
"""Score final tabletop object transforms against the target plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omnibot_agent.scoring import write_score  # noqa: E402
from omnibot_agent.tabletop_eval import evaluate_tabletop_transforms  # noqa: E402
from omnibot_agent.tabletop_eval import transforms_from_json  # noqa: E402


DEFAULT_TARGET_PLAN = ROOT / "notes" / "generated" / "tabletop_target_plan.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("final_transforms_json", type=Path)
    parser.add_argument("--target-plan", type=Path, default=DEFAULT_TARGET_PLAN)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--elapsed-s", type=float, default=600.0)
    parser.add_argument("--used-replicator-randomization", action="store_true")
    parser.add_argument("--transformer-visible-count", type=int, default=0)
    args = parser.parse_args()

    target_plan = json.loads(args.target_plan.read_text(encoding="utf-8"))
    final_transforms = transforms_from_json(
        json.loads(args.final_transforms_json.read_text(encoding="utf-8"))
    )
    score = evaluate_tabletop_transforms(
        target_plan=target_plan,
        final_transforms=final_transforms,
        elapsed_s=args.elapsed_s,
        used_replicator_randomization=args.used_replicator_randomization,
        transformer_visible_count=args.transformer_visible_count,
    )
    out = args.out or args.final_transforms_json.with_name("score_estimate.json")
    write_score(out, score)
    print(out)
    print(f"score={score.score}/{score.max_score}")
    return 0 if score.submission_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
