"""Adapters for GenManip/OmniBot online observations and actions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from omnibot_agent.pi05_adapter import EBenchActionChunk
from omnibot_agent.pi05_adapter import OmniBotObservation


HEAD_IMAGE_KEY = "video.overlook_camera_view"
LEFT_HAND_IMAGE_KEY = "video.left_camera_view"
RIGHT_HAND_IMAGE_KEY = "video.right_camera_view"
JOINT_STATE_KEY = "state.joints"
GRIPPER_STATE_KEY = "state.gripper"
BASE_STATE_KEY = "state.base"
INSTRUCTION_KEY = "instruction"


@dataclass(frozen=True)
class GenManipActionConfig:
    """Action mapping expected by the GenManip mobile dual-arm API."""

    control_type: str = "joint_position"
    is_rel: bool = False
    base_is_rel: bool = True
    base_mode: str = "chunk_delta"
    max_base_step: float | None = 0.08


def worker_obs_payload(worker_obs: dict[str, Any]) -> dict[str, Any]:
    """Accept either a full worker record or the nested obs payload."""

    nested = worker_obs.get("obs")
    if isinstance(nested, dict):
        return nested
    return worker_obs


def _require_array(payload: dict[str, Any], key: str) -> np.ndarray:
    if key not in payload:
        keys = ", ".join(sorted(str(k) for k in payload))
        raise KeyError(f"missing GenManip observation key {key!r}; available keys: {keys}")
    return np.asarray(payload[key])


def _image_uint8_hwc(image: np.ndarray, *, key: str) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"{key} must be an image array, got shape {image.shape}")
    if image.shape[0] in {1, 3, 4} and image.shape[-1] not in {3, 4}:
        image = np.transpose(image, (1, 2, 0))
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.shape[-1] != 3:
        raise ValueError(f"{key} must have 3 RGB channels, got shape {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0) * 255.0
    return np.asarray(image, dtype=np.uint8)


def observation_from_genmanip(
    worker_obs: dict[str, Any],
    *,
    prompt_override: str | None = None,
) -> OmniBotObservation:
    """Build the pi0.5-facing observation from a GenManip worker observation."""

    payload = worker_obs_payload(worker_obs)
    prompt = prompt_override or str(payload.get(INSTRUCTION_KEY) or "sort the tabletop objects into the target areas")
    joint_state = np.asarray(_require_array(payload, JOINT_STATE_KEY), dtype=np.float32).reshape(-1)
    gripper_state = np.asarray(_require_array(payload, GRIPPER_STATE_KEY), dtype=np.float32).reshape(-1)
    if joint_state.shape != (12,):
        raise ValueError(f"{JOINT_STATE_KEY} must have shape (12,), got {joint_state.shape}")
    if gripper_state.shape != (4,):
        raise ValueError(f"{GRIPPER_STATE_KEY} must have shape (4,), got {gripper_state.shape}")
    return OmniBotObservation(
        head_image=_image_uint8_hwc(_require_array(payload, HEAD_IMAGE_KEY), key=HEAD_IMAGE_KEY),
        left_hand_image=_image_uint8_hwc(_require_array(payload, LEFT_HAND_IMAGE_KEY), key=LEFT_HAND_IMAGE_KEY),
        right_hand_image=_image_uint8_hwc(_require_array(payload, RIGHT_HAND_IMAGE_KEY), key=RIGHT_HAND_IMAGE_KEY),
        joint_state=joint_state,
        gripper_state=gripper_state,
        prompt=prompt,
    )


def base_state_from_genmanip(worker_obs: dict[str, Any]) -> np.ndarray:
    payload = worker_obs_payload(worker_obs)
    value = payload.get(BASE_STATE_KEY)
    if value is None:
        return np.zeros(3, dtype=np.float32)
    base = np.asarray(value, dtype=np.float32).reshape(-1)
    if base.shape != (3,):
        raise ValueError(f"{BASE_STATE_KEY} must have shape (3,), got {base.shape}")
    return base


def ordered_dual_arm_action(joint: np.ndarray, gripper: np.ndarray) -> np.ndarray:
    """Map pi0.5 joint/gripper slices to GenManip dual-arm joint_position order."""

    joint = np.asarray(joint, dtype=np.float32).reshape(12)
    gripper = np.asarray(gripper, dtype=np.float32).reshape(4)
    return np.concatenate(
        [
            joint[:6],
            gripper[:2],
            joint[6:12],
            gripper[2:4],
        ]
    ).astype(np.float32)


def _base_motion(base_chunk: np.ndarray, index: int, previous: np.ndarray, config: GenManipActionConfig) -> np.ndarray:
    base = np.asarray(base_chunk[index], dtype=np.float32).reshape(3)
    if config.base_mode == "chunk_delta":
        motion = base - previous
    elif config.base_mode == "step_delta":
        motion = base
    elif config.base_mode == "zero":
        motion = np.zeros(3, dtype=np.float32)
    else:
        raise ValueError(f"unsupported base_mode={config.base_mode!r}")
    if config.max_base_step is not None:
        limit = float(config.max_base_step)
        motion = np.clip(motion, -limit, limit)
    return motion.astype(np.float32)


def genmanip_action_steps(
    action_chunk: EBenchActionChunk,
    *,
    horizon: int,
    config: GenManipActionConfig | None = None,
) -> list[dict[str, Any]]:
    """Convert an EBench action chunk to GenManip single-worker actions."""

    config = config or GenManipActionConfig()
    horizon = min(horizon, len(action_chunk.joint), len(action_chunk.gripper), len(action_chunk.base))
    if horizon <= 0:
        raise ValueError("horizon must select at least one action")

    previous_base = np.zeros(3, dtype=np.float32)
    actions: list[dict[str, Any]] = []
    for index in range(horizon):
        base_motion = _base_motion(action_chunk.base, index, previous_base, config)
        if config.base_mode == "chunk_delta":
            previous_base = np.asarray(action_chunk.base[index], dtype=np.float32).reshape(3)
        actions.append(
            {
                "action": ordered_dual_arm_action(action_chunk.joint[index], action_chunk.gripper[index]),
                "base_motion": base_motion,
                "control_type": config.control_type,
                "is_rel": config.is_rel,
                "base_is_rel": config.base_is_rel,
            }
        )
    return actions


def worker_major_step_chunk(worker_actions: dict[str, list[dict[str, Any]]]) -> list[dict[str, dict[str, Any]]]:
    """Convert per-worker action lists into EvalClient step_chunk format."""

    max_len = max((len(actions) for actions in worker_actions.values()), default=0)
    if max_len <= 0:
        raise ValueError("worker_actions must contain at least one action")
    chunk: list[dict[str, dict[str, Any]]] = []
    for index in range(max_len):
        step_action: dict[str, dict[str, Any]] = {}
        for worker_id, actions in worker_actions.items():
            if index < len(actions):
                step_action[str(worker_id)] = actions[index]
        if step_action:
            chunk.append(step_action)
    return chunk


def save_policy_npz(path: Path, *, observation: OmniBotObservation, action_chunk: EBenchActionChunk) -> None:
    """Persist model input/output arrays for later LeRobot conversion."""

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        head_image=observation.head_image,
        left_hand_image=observation.left_hand_image,
        right_hand_image=observation.right_hand_image,
        state_joint=np.asarray(observation.joint_state, dtype=np.float32),
        state_gripper=np.asarray(observation.gripper_state, dtype=np.float32),
        action_joint=np.asarray(action_chunk.joint, dtype=np.float32),
        action_gripper=np.asarray(action_chunk.gripper, dtype=np.float32),
        action_base=np.asarray(action_chunk.base, dtype=np.float32),
    )
