"""Competition-style episode logging."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from PIL import Image

from omnibot_agent.pi05_adapter import EBenchActionChunk
from omnibot_agent.pi05_adapter import OmniBotObservation


@dataclass(frozen=True)
class EpisodeMeta:
    episode_id: str
    task_name: str
    track: str
    robot: str
    policy: str
    started_at: str


class EpisodeLogger:
    def __init__(self, root: Path, meta: EpisodeMeta) -> None:
        self.root = root / meta.episode_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.meta = meta
        (self.root / "meta.json").write_text(json.dumps(asdict(meta), indent=2, ensure_ascii=False), encoding="utf-8")
        self.steps_path = self.root / "steps.jsonl"

    @classmethod
    def create(
        cls,
        root: Path,
        *,
        task_name: str,
        track: str,
        robot: str,
        policy: str,
        episode_id: str | None = None,
    ) -> "EpisodeLogger":
        stamp = time.strftime("%Y%m%d_%H%M%S")
        meta = EpisodeMeta(
            episode_id=episode_id or f"{track}_{task_name}_{stamp}",
            task_name=task_name,
            track=track,
            robot=robot,
            policy=policy,
            started_at=stamp,
        )
        return cls(root, meta)

    def log_step(
        self,
        *,
        step_index: int,
        observation: OmniBotObservation,
        raw_actions: EBenchActionChunk,
        safe_actions: EBenchActionChunk,
        decision: dict[str, Any] | None = None,
        score_estimate: dict[str, Any] | None = None,
        save_policy_npz: bool = False,
    ) -> None:
        prefix = f"step_{step_index:04d}"
        self._save_image(f"{prefix}_head.png", observation.head_image)
        self._save_image(f"{prefix}_hand_left.png", observation.left_hand_image)
        self._save_image(f"{prefix}_hand_right.png", observation.right_hand_image)

        policy_npz = None
        if save_policy_npz:
            policy_npz = f"{prefix}_policy.npz"
            np.savez_compressed(
                self.root / policy_npz,
                head_image=np.asarray(observation.head_image, dtype=np.uint8),
                left_hand_image=np.asarray(observation.left_hand_image, dtype=np.uint8),
                right_hand_image=np.asarray(observation.right_hand_image, dtype=np.uint8),
                state_joint=np.asarray(observation.joint_state, dtype=np.float32),
                state_gripper=np.asarray(observation.gripper_state, dtype=np.float32),
                action_joint=np.asarray(safe_actions.joint, dtype=np.float32),
                action_gripper=np.asarray(safe_actions.gripper, dtype=np.float32),
                action_base=np.asarray(safe_actions.base, dtype=np.float32),
            )

        record = {
            "step_index": step_index,
            "prompt": observation.prompt,
            "policy_npz": policy_npz,
            "instruction": observation.prompt,
            "sent_action_count": 1,
            "joint_state_shape": list(np.asarray(observation.joint_state).shape),
            "gripper_state_shape": list(np.asarray(observation.gripper_state).shape),
            "raw_action": self._chunk_summary(raw_actions),
            "safe_action": self._chunk_summary(safe_actions),
            "decision": decision or {},
            "score_estimate": score_estimate or {},
        }
        with self.steps_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        if save_policy_npz:
            with (self.root / "policy_steps.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def finalize(self, summary: dict[str, Any]) -> None:
        (self.root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    def _save_image(self, filename: str, image: np.ndarray) -> None:
        Image.fromarray(np.asarray(image, dtype=np.uint8)).save(self.root / filename)

    @staticmethod
    def _array_summary(array: np.ndarray) -> dict[str, Any]:
        array = np.asarray(array)
        return {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "min": float(array.min()) if array.size else None,
            "max": float(array.max()) if array.size else None,
            "mean": float(array.mean()) if array.size else None,
        }

    @classmethod
    def _chunk_summary(cls, chunk: EBenchActionChunk) -> dict[str, Any]:
        return {
            "joint": cls._array_summary(chunk.joint),
            "gripper": cls._array_summary(chunk.gripper),
            "base": cls._array_summary(chunk.base),
        }
