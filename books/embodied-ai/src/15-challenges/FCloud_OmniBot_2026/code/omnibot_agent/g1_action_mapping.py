"""Conservative mapping from pi0.5 EBench actions to the OmniBot G1 asset."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass

import numpy as np


LEFT_ARM_6DOF = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
)

RIGHT_ARM_6DOF = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
)

LEFT_HAND_GROUPS = (
    ("left_hand_index_0_joint", "left_hand_index_1_joint"),
    ("left_hand_middle_0_joint", "left_hand_middle_1_joint"),
    ("left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint"),
)

RIGHT_HAND_GROUPS = (
    ("right_hand_index_0_joint", "right_hand_index_1_joint"),
    ("right_hand_middle_0_joint", "right_hand_middle_1_joint"),
    ("right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint"),
)

UNUSED_G1_ARM_JOINTS = (
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)


@dataclass(frozen=True)
class G1ActionMapping:
    pi05_joint_dim: int
    pi05_gripper_dim: int
    pi05_base_dim: int
    joint_index_to_g1_joint: dict[int, str]
    gripper_index_to_g1_groups: dict[int, tuple[str, ...]]
    base_index_meaning: dict[int, str]
    unused_g1_joints: tuple[str, ...]
    caution: str

    def to_json(self) -> dict:
        return asdict(self)


def default_mapping() -> G1ActionMapping:
    """Return the first executable mapping for G1 tabletop policy adaptation.

    The mapping intentionally uses six major arm joints per side to match the
    inherited pi0.5 EBench 12-joint action head. Wrist yaw is held by the
    scripted controller until a G1-native action head is trained.
    """

    arm_joints = LEFT_ARM_6DOF + RIGHT_ARM_6DOF
    gripper_groups = {
        0: LEFT_HAND_GROUPS[0] + LEFT_HAND_GROUPS[1],
        1: LEFT_HAND_GROUPS[2],
        2: RIGHT_HAND_GROUPS[0] + RIGHT_HAND_GROUPS[1],
        3: RIGHT_HAND_GROUPS[2],
    }
    return G1ActionMapping(
        pi05_joint_dim=12,
        pi05_gripper_dim=4,
        pi05_base_dim=3,
        joint_index_to_g1_joint={idx: name for idx, name in enumerate(arm_joints)},
        gripper_index_to_g1_groups=gripper_groups,
        base_index_meaning={0: "base_dx_or_forward_velocity", 1: "base_dy_or_lateral_velocity", 2: "base_dyaw"},
        unused_g1_joints=UNUSED_G1_ARM_JOINTS,
        caution=(
            "This is a conservative bridge from the existing EBench pi0.5 action "
            "head to G1. For final leaderboard performance, prefer a G1-native "
            "action head or a scripted IK controller that includes wrist yaw."
        ),
    )


def pi05_action_to_g1_targets(
    *,
    dof_names: list[str],
    current_positions: np.ndarray,
    joint_action: np.ndarray,
    gripper_action: np.ndarray,
    joint_scale: float = 1.0,
    gripper_scale: float = 1.0,
) -> np.ndarray:
    """Map one pi0.5 action vector into a full G1 DOF position target.

    Unknown/missing DOFs are left at their current positions. The action values
    are interpreted as deltas around the current position because the inherited
    pi0.5 policy head was trained with delta-style joint actions.
    """

    mapping = default_mapping()
    targets = np.asarray(current_positions, dtype=np.float32).reshape(-1).copy()
    joint_action = np.asarray(joint_action, dtype=np.float32).reshape(-1)
    gripper_action = np.asarray(gripper_action, dtype=np.float32).reshape(-1)
    if joint_action.shape[0] < mapping.pi05_joint_dim:
        raise ValueError(f"joint_action needs {mapping.pi05_joint_dim} values, got {joint_action.shape[0]}")
    if gripper_action.shape[0] < mapping.pi05_gripper_dim:
        raise ValueError(f"gripper_action needs {mapping.pi05_gripper_dim} values, got {gripper_action.shape[0]}")

    dof_index = {name: idx for idx, name in enumerate(dof_names)}
    for action_index, joint_name in mapping.joint_index_to_g1_joint.items():
        idx = dof_index.get(joint_name)
        if idx is not None:
            targets[idx] = targets[idx] + float(joint_action[int(action_index)]) * joint_scale

    for action_index, joint_names in mapping.gripper_index_to_g1_groups.items():
        value = float(gripper_action[int(action_index)]) * gripper_scale
        for joint_name in joint_names:
            idx = dof_index.get(joint_name)
            if idx is not None:
                targets[idx] = targets[idx] + value
    return targets
