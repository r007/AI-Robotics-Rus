"""Adapters from GenieSim-style G1 IK joints to official OmniBot G1 joints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


GENIE_TO_OFFICIAL = {
    "idx05_left_arm_joint1": "left_shoulder_pitch_joint",
    "idx06_left_arm_joint2": "left_shoulder_roll_joint",
    "idx07_left_arm_joint3": "left_shoulder_yaw_joint",
    "idx08_left_arm_joint4": "left_elbow_joint",
    "idx09_left_arm_joint5": "left_wrist_roll_joint",
    "idx10_left_arm_joint6": "left_wrist_pitch_joint",
    "idx11_left_arm_joint7": "left_wrist_yaw_joint",
    "idx12_right_arm_joint1": "right_shoulder_pitch_joint",
    "idx13_right_arm_joint2": "right_shoulder_roll_joint",
    "idx14_right_arm_joint3": "right_shoulder_yaw_joint",
    "idx15_right_arm_joint4": "right_elbow_joint",
    "idx16_right_arm_joint5": "right_wrist_roll_joint",
    "idx17_right_arm_joint6": "right_wrist_pitch_joint",
    "idx18_right_arm_joint7": "right_wrist_yaw_joint",
}


@dataclass(frozen=True)
class JointAdapterReport:
    mapped_count: int
    missing_source: tuple[str, ...]
    missing_target: tuple[str, ...]


def apply_genie_joint_solution(
    *,
    official_dof_names: list[str],
    current_positions: np.ndarray,
    genie_joint_names: list[str],
    genie_joint_positions: np.ndarray,
    mapping: Mapping[str, str] = GENIE_TO_OFFICIAL,
) -> tuple[np.ndarray, JointAdapterReport]:
    """Return official G1 joint targets with a GenieSim IK solution overlaid."""

    targets = np.asarray(current_positions, dtype=np.float32).reshape(-1).copy()
    if targets.shape[0] != len(official_dof_names):
        raise ValueError("current_positions length must match official_dof_names")
    source_positions = np.asarray(genie_joint_positions, dtype=np.float32).reshape(-1)
    if source_positions.shape[0] != len(genie_joint_names):
        raise ValueError("genie_joint_positions length must match genie_joint_names")

    source_index = {name: idx for idx, name in enumerate(genie_joint_names)}
    target_index = {name: idx for idx, name in enumerate(official_dof_names)}
    missing_source = []
    missing_target = []
    mapped = 0
    for source_name, target_name in mapping.items():
        if source_name not in source_index:
            missing_source.append(source_name)
            continue
        if target_name not in target_index:
            missing_target.append(target_name)
            continue
        targets[target_index[target_name]] = source_positions[source_index[source_name]]
        mapped += 1
    return targets, JointAdapterReport(
        mapped_count=mapped,
        missing_source=tuple(missing_source),
        missing_target=tuple(missing_target),
    )

