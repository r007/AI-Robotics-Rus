#!/usr/bin/env python3
"""Dry-run a tabletop episode with fake observations and competition logging."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omnibot_agent.episode_logger import EpisodeLogger  # noqa: E402
from omnibot_agent.pi05_adapter import OmniBotObservation  # noqa: E402
from omnibot_agent.policy_client import FakePolicy  # noqa: E402
from omnibot_agent.safety import ActionSafetyLimits  # noqa: E402
from omnibot_agent.safety import clip_action_chunk  # noqa: E402


def image(seed: int) -> np.ndarray:
    yy, xx = np.mgrid[0:224, 0:224]
    return np.stack([(xx + seed) % 256, (yy + 2 * seed) % 256, (xx + yy + 3 * seed) % 256], axis=-1).astype(np.uint8)


def main() -> int:
    logger = EpisodeLogger.create(
        ROOT / "logs" / "tabletop_dryrun",
        task_name="tabletop_sorting",
        track="monthly",
        robot="omnibot-placeholder",
        policy="fake-pi05-interface",
    )
    policy = FakePolicy()
    limits = ActionSafetyLimits()

    for step in range(3):
        obs = OmniBotObservation(
            head_image=image(step),
            left_hand_image=image(step + 10),
            right_hand_image=image(step + 20),
            joint_state=np.zeros(12, dtype=np.float32),
            gripper_state=np.zeros(4, dtype=np.float32),
            prompt="sort the tabletop objects into the target areas",
        )
        raw_actions = policy.infer(obs)
        safe_actions = clip_action_chunk(raw_actions, obs.joint_state, limits)
        logger.log_step(
            step_index=step,
            observation=obs,
            raw_actions=raw_actions,
            safe_actions=safe_actions,
            decision={"phase": "dryrun", "target": "placeholder"},
            score_estimate={"placed_objects": 0, "collisions": 0},
        )

    logger.finalize({"status": "dryrun_complete", "steps": 3})
    print(logger.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
