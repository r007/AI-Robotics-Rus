"""Safety filters for model-produced action chunks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from omnibot_agent.pi05_adapter import EBenchActionChunk


@dataclass(frozen=True)
class ActionSafetyLimits:
    joint_abs_limit: float = 2.8
    joint_step_limit: float = 0.08
    gripper_min: float = -1.0
    gripper_max: float = 1.0
    base_xy_step_limit: float = 0.06
    base_yaw_step_limit: float = 0.12


def clip_action_chunk(chunk: EBenchActionChunk, current_joint: np.ndarray, limits: ActionSafetyLimits) -> EBenchActionChunk:
    """Clip action chunk before it is sent to a simulator or robot.

    The filter is conservative by design. It assumes model outputs are desired
    positions or small deltas and enforces per-step movement bounds around the
    current joint state.
    """

    current_joint = np.asarray(current_joint, dtype=np.float32).reshape(1, -1)
    if current_joint.shape[1] != chunk.joint.shape[1]:
        raise ValueError(f"current_joint dim {current_joint.shape[1]} != action joint dim {chunk.joint.shape[1]}")

    lower = np.maximum(current_joint - limits.joint_step_limit, -limits.joint_abs_limit)
    upper = np.minimum(current_joint + limits.joint_step_limit, limits.joint_abs_limit)
    safe_joint = np.clip(chunk.joint, lower, upper)
    safe_gripper = np.clip(chunk.gripper, limits.gripper_min, limits.gripper_max)

    safe_base = chunk.base.copy()
    if safe_base.shape[1] >= 2:
        safe_base[:, :2] = np.clip(safe_base[:, :2], -limits.base_xy_step_limit, limits.base_xy_step_limit)
    if safe_base.shape[1] >= 3:
        safe_base[:, 2] = np.clip(safe_base[:, 2], -limits.base_yaw_step_limit, limits.base_yaw_step_limit)

    return EBenchActionChunk(joint=safe_joint, gripper=safe_gripper, base=safe_base)
