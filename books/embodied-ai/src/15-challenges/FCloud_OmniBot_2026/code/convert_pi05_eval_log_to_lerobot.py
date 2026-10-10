#!/usr/bin/env python3
"""Convert pi0.5 logged episodes to a LeRobot dataset.

Default mode is strict and only accepts reviewed, high-scoring real episodes.
Use --allow-bootstrap only to smoke-test the OpenPI fine-tuning machinery; those
datasets are not suitable for leaderboard training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEROBOT_ROOT = Path("<SHARED_ROOT>/datasets/omnibot_challenge/lerobot")
NON_TRAINING_MARKERS = ("fake", "dryrun", "dry-run", "smoke", "placeholder", "diagnostic", "probe")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _load_policy_steps(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _score_ok(episode_dir: Path, min_score: int) -> bool:
    score = _load_json(episode_dir / "score_estimate.json")
    return score.get("max_score") == 12 and isinstance(score.get("score"), int) and score["score"] >= min_score


def _has_training_marker(episode_dir: Path, meta: dict) -> bool:
    text = " ".join(
        [
            str(episode_dir).lower(),
            str(meta.get("task_name", "")).lower(),
            str(meta.get("policy", "")).lower(),
            str(meta.get("source", "")).lower(),
        ]
    )
    return any(marker in text for marker in NON_TRAINING_MARKERS)


def _eligible_episode(
    episode_dir: Path,
    *,
    min_score: int,
    allow_unreviewed: bool,
    allow_chunked: bool,
    allow_bootstrap: bool,
) -> tuple[bool, str]:
    meta = _load_json(episode_dir / "meta.json")
    if not meta:
        return False, "missing_meta"
    if meta.get("not_for_submission", False) and not allow_bootstrap:
        return False, "not_for_submission"
    if _has_training_marker(episode_dir, meta) and not allow_bootstrap:
        return False, "non_training_marker"
    if not allow_unreviewed and not allow_bootstrap and not meta.get("use_for_training", False):
        return False, "not_reviewed_for_training"
    if not allow_bootstrap and not _score_ok(episode_dir, min_score):
        return False, "score_below_threshold_or_missing"
    if not allow_bootstrap and not list(episode_dir.glob("*.mp4")):
        return False, "missing_video"
    steps = _load_policy_steps(episode_dir / "policy_steps.jsonl")
    if not steps:
        return False, "missing_policy_steps"
    if not allow_chunked and any(int(step.get("sent_action_count", 1)) != 1 for step in steps):
        return False, "chunked_actions_need_chunk_size_1_for_clean_bc"
    return True, "ok"


def _features() -> dict:
    return {
        "video.overlook_camera_view": {
            "dtype": "image",
            "shape": (224, 224, 3),
            "names": ["height", "width", "channel"],
        },
        "video.left_camera_view": {
            "dtype": "image",
            "shape": (224, 224, 3),
            "names": ["height", "width", "channel"],
        },
        "video.right_camera_view": {
            "dtype": "image",
            "shape": (224, 224, 3),
            "names": ["height", "width", "channel"],
        },
        "state.joints": {
            "dtype": "float32",
            "shape": (12,),
            "names": ["joint"],
        },
        "state.gripper": {
            "dtype": "float32",
            "shape": (4,),
            "names": ["gripper"],
        },
        "action.joints": {
            "dtype": "float32",
            "shape": (12,),
            "names": ["joint_action"],
        },
        "action.gripper": {
            "dtype": "float32",
            "shape": (4,),
            "names": ["gripper_action"],
        },
        "action.base": {
            "dtype": "float32",
            "shape": (3,),
            "names": ["base_action"],
        },
    }


def _resize_to_224(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.uint8)
    if image.shape[:2] == (224, 224):
        return image
    from PIL import Image

    return np.asarray(Image.fromarray(image).resize((224, 224), Image.Resampling.BILINEAR), dtype=np.uint8)


def _add_episode(dataset, episode_dir: Path) -> int:
    steps = _load_policy_steps(episode_dir / "policy_steps.jsonl")
    frame_count = 0
    for step in steps:
        npz_path = episode_dir / step["policy_npz"]
        if not npz_path.exists():
            raise FileNotFoundError(f"missing policy npz: {npz_path}")
        with np.load(npz_path) as data:
            dataset.add_frame(
                {
                    "video.overlook_camera_view": _resize_to_224(data["head_image"]),
                    "video.left_camera_view": _resize_to_224(data["left_hand_image"]),
                    "video.right_camera_view": _resize_to_224(data["right_hand_image"]),
                    "state.joints": np.asarray(data["state_joint"], dtype=np.float32),
                    "state.gripper": np.asarray(data["state_gripper"], dtype=np.float32),
                    "action.joints": np.asarray(data["action_joint"][0], dtype=np.float32),
                    "action.gripper": np.asarray(data["action_gripper"][0], dtype=np.float32),
                    "action.base": np.asarray(data["action_base"][0], dtype=np.float32),
                    "task": step.get("instruction") or "sort the tabletop objects into the target areas",
                }
            )
            frame_count += 1
    dataset.save_episode()
    return frame_count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("episode_dirs", nargs="*", type=Path)
    parser.add_argument("--scan-root", type=Path, default=ROOT / "logs" / "genmanip_online_pi05")
    parser.add_argument("--repo-id", default="local/omnibot_tabletop_pi05")
    parser.add_argument("--lerobot-root", type=Path, default=DEFAULT_LEROBOT_ROOT)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--min-score", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-unreviewed", action="store_true")
    parser.add_argument("--allow-chunked", action="store_true")
    parser.add_argument(
        "--allow-bootstrap",
        action="store_true",
        help="Permit smoke/bootstrap episodes to verify training startup only; not for leaderboard training.",
    )
    args = parser.parse_args()

    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise RuntimeError("Run this with the openpi-pi05 environment.") from exc

    episode_dirs = list(args.episode_dirs)
    if not episode_dirs and args.scan_root.exists():
        episode_dirs = [path.parent for path in args.scan_root.rglob("policy_steps.jsonl")]
    episode_dirs = sorted({path.resolve() for path in episode_dirs})

    accepted: list[Path] = []
    rejected: list[dict] = []
    for episode_dir in episode_dirs:
        ok, reason = _eligible_episode(
            episode_dir,
            min_score=args.min_score,
            allow_unreviewed=args.allow_unreviewed,
            allow_chunked=args.allow_chunked,
            allow_bootstrap=args.allow_bootstrap,
        )
        if ok:
            accepted.append(episode_dir)
        else:
            rejected.append({"episode_dir": str(episode_dir), "reason": reason})

    dataset_root = args.lerobot_root / args.repo_id
    if dataset_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{dataset_root} exists; pass --overwrite to replace it")
        shutil.rmtree(dataset_root)

    if not accepted:
        report = {
            "created": False,
            "accepted_count": 0,
            "rejected_count": len(rejected),
            "rejected_sample": rejected[:20],
            "next_action": "collect and review real official episodes, then set meta.use_for_training=true",
        }
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 1

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=dataset_root,
        fps=args.fps,
        robot_type="omnibot_g1_mobile_dual_arm",
        features=_features(),
        use_videos=True,
        image_writer_threads=8,
        image_writer_processes=2,
    )
    total_frames = 0
    for episode_dir in accepted:
        total_frames += _add_episode(dataset, episode_dir)

    report = {
        "created": True,
        "repo_id": args.repo_id,
        "dataset_root": str(dataset_root),
        "accepted_count": len(accepted),
        "total_frames": total_frames,
        "rejected_count": len(rejected),
        "rejected_sample": rejected[:20],
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
