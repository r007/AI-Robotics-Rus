"""Adapters between OmniBot observations and the pi0.5 EBench policy interface."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


EBENCH_IMAGE_KEYS = ("images/head", "images/hand_left", "images/hand_right")
EBENCH_STATE_JOINT_DIM = 12
EBENCH_STATE_GRIPPER_DIM = 4
EBENCH_ACTION_JOINT_DIM = 12
EBENCH_ACTION_GRIPPER_DIM = 4
EBENCH_ACTION_BASE_DIM = 3
EBENCH_ACTION_DIM = EBENCH_ACTION_JOINT_DIM + EBENCH_ACTION_GRIPPER_DIM + EBENCH_ACTION_BASE_DIM


@dataclass(frozen=True)
class OmniBotObservation:
    """Minimal model-facing observation.

    All images must be uint8 HWC arrays captured from robot-mounted cameras.
    Do not fill these fields from privileged USD object poses.
    """

    head_image: np.ndarray
    left_hand_image: np.ndarray
    right_hand_image: np.ndarray
    joint_state: np.ndarray
    gripper_state: np.ndarray
    prompt: str


@dataclass(frozen=True)
class EBenchActionChunk:
    joint: np.ndarray
    gripper: np.ndarray
    base: np.ndarray


def _image_uint8_hwc(image: np.ndarray, *, name: str) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"{name} must be HWC/CHW image, got shape {image.shape}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.transpose(image, (1, 2, 0))
    if image.shape[-1] != 3:
        raise ValueError(f"{name} must have 3 channels, got shape {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0) * 255.0
    return np.asarray(image, dtype=np.uint8)


def pack_ebench_request(obs: OmniBotObservation) -> dict[str, Any]:
    joint = np.asarray(obs.joint_state, dtype=np.float32).reshape(-1)
    gripper = np.asarray(obs.gripper_state, dtype=np.float32).reshape(-1)
    if joint.shape != (EBENCH_STATE_JOINT_DIM,):
        raise ValueError(f"joint_state must have shape ({EBENCH_STATE_JOINT_DIM},), got {joint.shape}")
    if gripper.shape != (EBENCH_STATE_GRIPPER_DIM,):
        raise ValueError(f"gripper_state must have shape ({EBENCH_STATE_GRIPPER_DIM},), got {gripper.shape}")
    return {
        "states/joint": joint,
        "states/gripper": gripper,
        "images/head": _image_uint8_hwc(obs.head_image, name="head_image"),
        "images/hand_left": _image_uint8_hwc(obs.left_hand_image, name="left_hand_image"),
        "images/hand_right": _image_uint8_hwc(obs.right_hand_image, name="right_hand_image"),
        "prompt": obs.prompt,
    }


def split_ebench_actions(actions: np.ndarray) -> EBenchActionChunk:
    actions = np.asarray(actions)
    if actions.ndim != 2 or actions.shape[1] < EBENCH_ACTION_DIM:
        raise ValueError(f"actions must be [T,{EBENCH_ACTION_DIM}] or wider, got {actions.shape}")
    trimmed = actions[:, :EBENCH_ACTION_DIM]
    joint_end = EBENCH_ACTION_JOINT_DIM
    gripper_end = joint_end + EBENCH_ACTION_GRIPPER_DIM
    return EBenchActionChunk(
        joint=trimmed[:, :joint_end],
        gripper=trimmed[:, joint_end:gripper_end],
        base=trimmed[:, gripper_end:],
    )


def save_policy_step_log(
    out_dir: Path,
    *,
    step_index: int,
    request: dict[str, Any],
    response: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"step_{step_index:04d}"
    for key in EBENCH_IMAGE_KEYS:
        image = request.get(key)
        if isinstance(image, np.ndarray):
            Image.fromarray(image).save(out_dir / f"{prefix}_{key.replace('/', '_')}.png")

    actions = response.get("actions")
    action_chunk = split_ebench_actions(actions) if isinstance(actions, np.ndarray) else None
    record = {
        "step_index": step_index,
        "prompt": request.get("prompt"),
        "state_joint_shape": list(np.asarray(request.get("states/joint")).shape),
        "state_gripper_shape": list(np.asarray(request.get("states/gripper")).shape),
        "action_shape": list(actions.shape) if isinstance(actions, np.ndarray) else None,
        "action_joint_shape": list(action_chunk.joint.shape) if action_chunk else None,
        "action_gripper_shape": list(action_chunk.gripper.shape) if action_chunk else None,
        "action_base_shape": list(action_chunk.base.shape) if action_chunk else None,
        "extra": extra or {},
    }
    with (out_dir / "steps.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
