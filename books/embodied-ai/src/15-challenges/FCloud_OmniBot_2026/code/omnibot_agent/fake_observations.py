"""Deterministic fake observations for pipeline smoke tests."""

from __future__ import annotations

import numpy as np

from omnibot_agent.pi05_adapter import OmniBotObservation


def gradient_image(seed: int, height: int = 224, width: int = 224) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    return np.stack(
        [
            (xx + 17 * seed) % 256,
            (yy * 2 + 31 * seed) % 256,
            ((xx // 2 + yy // 3) + 47 * seed) % 256,
        ],
        axis=-1,
    ).astype(np.uint8)


def make_fake_omnibot_observation(step: int, prompt: str) -> OmniBotObservation:
    return OmniBotObservation(
        head_image=gradient_image(step + 1),
        left_hand_image=gradient_image(step + 11),
        right_hand_image=gradient_image(step + 21),
        joint_state=np.zeros(12, dtype=np.float32),
        gripper_state=np.zeros(4, dtype=np.float32),
        prompt=prompt,
    )
