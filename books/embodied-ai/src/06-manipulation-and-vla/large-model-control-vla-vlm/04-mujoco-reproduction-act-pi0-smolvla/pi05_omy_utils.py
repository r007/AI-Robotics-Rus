"""OMY single-mug dataset preparation and pi05 training interfaces (LeRobot 0.6.1)."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

LEROBOT_VERSION = "0.6.1"
CAMERAS = {
    "observation.image": "observation.images.image",
    "observation.wrist_image": "observation.images.image2",
}
STATE_NAMES = ["x", "y", "z", "roll", "pitch", "yaw"]
ACTION_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]


def check_version():
    import importlib.metadata

    actual = importlib.metadata.version("lerobot")
    if actual != LEROBOT_VERSION:
        raise RuntimeError(f"Use the separate LeRobot {LEROBOT_VERSION} kernel, got {actual}")


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _vector(value, size, name):
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name}: expected {size} finite values, got {array.shape}")
    return array


def _image(value, root, shape):
    """Decode embedded HF Image bytes or a path inside the source dataset."""
    if not isinstance(value, dict) or not (value.get("bytes") or value.get("path")):
        raise ValueError("Expected a complete image feature; check Git LFS download")
    if value.get("bytes"):
        source = io.BytesIO(value["bytes"])
    else:
        source = (root / value["path"]).resolve()
        if not source.is_relative_to(root):
            raise ValueError("Image path must stay inside the source dataset")
    with Image.open(source) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    if array.shape != tuple(shape):
        raise ValueError(f"Unexpected image shape: {array.shape}, expected {shape}")
    return array


def read_v21(root):
    """Read and audit chapter-1 data, returning episodes in canonical frame order.

    State semantics (EEF pose) and command-label provenance must be checked by the
    operator. Neither can be inferred from the six/seven numerical dimensions.
    The source directory is opened read-only throughout.
    """
    root = Path(root).resolve()
    info = _json(root / "meta/info.json")
    if info.get("codebase_version") != "v2.1":
        raise ValueError("This converter accepts chapter-1 LeRobot v2.1 data only")
    if info.get("fps") != 20:
        raise ValueError("This OMY workflow expects chapter-1 collection at 20 Hz")
    features = info["features"]
    for key, dim in {"observation.state": 6, "action": 7, "obj_init": 6}.items():
        if features.get(key, {}).get("shape") != [dim]:
            raise ValueError(f"{key}: expected shape [{dim}]; use the chapter-1 OMY collector")
    for key in CAMERAS:
        if features.get(key, {}).get("dtype") != "image":
            raise ValueError(f"{key}: expected embedded image data, not a video/LFS pointer")
        if features[key].get("shape") != [256, 256, 3]:
            raise ValueError(f"{key}: expected chapter-1 256x256 RGB images")
    tasks = {int(t["task_index"]): t["task"] for t in _jsonl(root / "meta/tasks.jsonl")}
    if not tasks or any(not isinstance(t, str) or not t.strip() for t in tasks.values()):
        raise ValueError("Missing language task in meta/tasks.jsonl")
    episodes = _jsonl(root / "meta/episodes.jsonl")
    episodes.sort(key=lambda e: e["episode_index"])
    if [e["episode_index"] for e in episodes] != list(range(info["total_episodes"])):
        raise ValueError("Episode metadata is incomplete or non-contiguous")
    result, global_index = [], 0
    for episode in episodes:
        ep = episode["episode_index"]
        rel = info["data_path"].format(episode_chunk=ep // info["chunks_size"], episode_index=ep)
        rows = pq.read_table(root / rel).to_pylist()
        rows.sort(key=lambda row: row["frame_index"])
        if not rows or len(rows) != episode["length"]:
            raise ValueError(f"Episode {ep}: incomplete Parquet data")
        for frame, row in enumerate(rows):
            if (row["episode_index"], row["frame_index"], row["index"]) != (ep, frame, global_index):
                raise ValueError(f"Episode {ep}: invalid frame/global indices")
            if not np.isclose(row["timestamp"], frame / info["fps"], atol=1e-4):
                raise ValueError(f"Episode {ep}: timestamps do not match 20 Hz")
            row["observation.state"] = _vector(row["observation.state"], 6, "EEF state")
            row["action"] = _vector(row["action"], 7, "absolute joint command")
            row["obj_init"] = _vector(row["obj_init"], 6, "mug and plate initial positions")
            if not 0 <= row["action"][-1] <= 1:
                raise ValueError(f"Episode {ep}: gripper command is outside [0, 1]")
            if not np.allclose(row["obj_init"], rows[0]["obj_init"], atol=1e-5):
                raise ValueError(f"Episode {ep}: object reset metadata changed mid-episode")
            if int(row["task_index"]) not in tasks:
                raise ValueError(f"Episode {ep}: unknown task_index")
            row["task"] = tasks[int(row["task_index"])]
            for key in CAMERAS:
                # Audit every image before creating any destination files.
                _image(row[key], root, features[key]["shape"])
            global_index += 1
        if len({row["task"] for row in rows}) != 1:
            raise ValueError(f"Episode {ep}: this single-task workflow requires one task per episode")
        result.append(rows)
    if global_index != info["total_frames"]:
        raise ValueError("total_frames does not match the source data")
    return info, result


def split_episodes(count, seed=42):
    """Hold out whole episodes; one/two demos are suitable only for interface checks."""
    if count < 3:
        return list(range(count)), []
    order = np.random.default_rng(seed).permutation(count).tolist()
    n_eval = max(1, round(count * 0.2))
    return sorted(order[n_eval:]), sorted(order[:n_eval])


def _numeric_stats(rows, key):
    array = np.stack([row[key] for row in rows]).astype(np.float64)
    return {
        "min": array.min(0).tolist(), "max": array.max(0).tolist(),
        "mean": array.mean(0).tolist(), "std": array.std(0).tolist(),
        "q01": np.quantile(array, 0.01, axis=0).tolist(),
        "q99": np.quantile(array, 0.99, axis=0).tolist(), "count": [len(rows)],
    }


def prepare_dataset(source, destination, repo_id, *, labels_verified=False, split_seed=42):
    """Convert into a NEW v3 directory and compute training-only quantile statistics."""
    check_version()
    if not labels_verified:
        raise ValueError("Replay newly collected command-labelled data and set labels_verified=True; old labels cannot be repaired by conversion")
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination == source or destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("Source and destination must be separate directories")
    if destination.exists():
        raise FileExistsError(f"Destination already exists; choose a new path: {destination}")
    info, episodes = read_v21(source)
    if not episodes:
        raise ValueError("No complete episodes found")
    train, heldout = split_episodes(len(episodes), split_seed)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = {
        new: {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channels"]}
        for new in CAMERAS.values()
    }
    features.update({
        "observation.state": {"dtype": "float32", "shape": (6,), "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": (7,), "names": ACTION_NAMES},
        "obj_init": {"dtype": "float32", "shape": (6,), "names": ["mug_x", "mug_y", "mug_z", "plate_x", "plate_y", "plate_z"]},
    })
    writer = LeRobotDataset.create(repo_id=repo_id, root=destination, fps=info["fps"],
                                   features=features, robot_type="omy", use_videos=False)
    try:
        for rows in episodes:
            for row in rows:
                frame = {new: _image(row[old], source, features[new]["shape"]) for old, new in CAMERAS.items()}
                frame.update({key: row[key].copy() for key in ("observation.state", "action", "obj_init")})
                frame["task"] = row["task"]
                writer.add_frame(frame)
            writer.save_episode()
    finally:
        writer.finalize()
    stats_path = destination / "meta/stats.json"
    stats = _json(stats_path)
    training_rows = [row for ep in train for row in episodes[ep]]
    for key in ("observation.state", "action"):
        stats[key] = _numeric_stats(training_rows, key)
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {
        "source": str(source), "lerobot_version": LEROBOT_VERSION, "fps": info["fps"],
        "state_kind": "eef_pose", "action_kind": "absolute_joint_targets",
        "labels_verified": True, "split_seed": split_seed,
        "train_episodes": train, "heldout_episodes": heldout,
        "normalization_episodes": train, "total_frames": info["total_frames"],
        "repo_id": repo_id,
    }
    (destination / "omy_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def verify_alignment(root, chunk_size=50):
    """Check the actual LeRobot reader's action chunks at every episode boundary."""
    check_version()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(root).resolve()
    manifest = _json(root / "omy_manifest.json")
    info = _json(root / "meta/info.json")
    dataset = LeRobotDataset(manifest["repo_id"], root=root,
        delta_timestamps={"action": [i / info["fps"] for i in range(chunk_size)]})
    # Physical Parquet row order must agree with global indices used by action slicing.
    rows = []
    for path in sorted((root / "data").rglob("*.parquet")):
        rows.extend(pq.read_table(path).select(["index", "episode_index", "frame_index", "action"]).to_pylist())
    if [row["index"] for row in rows] != list(range(len(rows))):
        raise AssertionError("Non-canonical physical row order")
    for ep in range(info["total_episodes"]):
        indices = [i for i, row in enumerate(rows) if row["episode_index"] == ep]
        start, end = indices[0], indices[-1]
        for idx in set([start, end, max(start, end - chunk_size + 1)]):
            sample = dataset[idx]
            expected = np.stack([rows[min(idx + k, end)]["action"] for k in range(chunk_size)])
            np.testing.assert_allclose(sample["action"].numpy(), expected, atol=1e-6)
            expected_pad = np.arange(chunk_size) + idx > end
            np.testing.assert_array_equal(sample["action_is_pad"].numpy(), expected_pad)
            if tuple(sample["observation.state"].shape) != (6,):
                raise AssertionError("State feature must stay six-dimensional")
    return {"frames": len(rows), "episodes": info["total_episodes"], "chunk_size": chunk_size}


def training_command(root, output, *, steps=10000, batch_size=1, chunk_size=50,
                     n_action_steps=5, pretrained="lerobot/pi05_base"):
    """Build the official CLI; keep absolute joint actions and EEF states distinct."""
    root, output = Path(root).resolve(), Path(output).resolve()
    manifest = _json(root / "omy_manifest.json")
    if output.exists():
        raise FileExistsError(f"Choose a new training output directory: {output}")
    if not 0 < n_action_steps <= chunk_size or steps <= 0 or batch_size <= 0:
        raise ValueError("Invalid training/chunk parameters")
    return [sys.executable, "-m", "pi05_omy_train",
        f"--dataset.repo_id={manifest['repo_id']}", f"--dataset.root={root}",
        f"--dataset.episodes={json.dumps(manifest['train_episodes'])}",
        "--dataset.eval_split=0.0", "--policy.type=pi05",
        f"--policy.pretrained_path={pretrained}", "--policy.push_to_hub=false",
        "--policy.device=cuda", "--policy.dtype=bfloat16",
        "--policy.use_relative_actions=false", "--policy.train_expert_only=true",
        "--policy.freeze_vision_encoder=true", "--policy.gradient_checkpointing=true",
        "--policy.compile_model=false", f"--policy.chunk_size={chunk_size}",
        f"--policy.n_action_steps={n_action_steps}", f"--steps={steps}",
        f"--batch_size={batch_size}", "--num_workers=0", "--env_eval_freq=0",
        "--eval_steps=0", "--wandb.enable=false", "--log_freq=1" if steps < 10 else "--log_freq=50",
        f"--save_freq={min(steps, 2000)}", "--seed=42", f"--output_dir={output}",
        "--job_name=pi05_omy"]


def load_policy(checkpoint, device="cuda"):
    """Load model AND saved processors, preserving the training quantile statistics."""
    check_version()
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.policies.factory import make_pre_post_processors
    from pi05_omy_train import install_strict_loader

    install_strict_loader()

    checkpoint = Path(checkpoint).resolve()
    if not (checkpoint / "config.json").is_file():
        raise FileNotFoundError(f"Expected a fine-tuned pretrained_model directory: {checkpoint}")
    policy = PI05Policy.from_pretrained(str(checkpoint))
    policy.to(device).eval()
    policy.config.device = device
    if policy.config.use_relative_actions:
        raise ValueError("OMY EEF states cannot be subtracted from joint target actions")
    if policy.config.input_features["observation.state"].shape != (6,):
        raise ValueError("This evaluator expects chapter-1 EEF states")
    if policy.config.output_features["action"].shape != (7,):
        raise ValueError("This evaluator expects OMY joint1..6 plus gripper")
    pre, post = make_pre_post_processors(policy.config, pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": device}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}})
    return policy, pre, post


def rollout(policy, preprocessor, postprocessor, env, *, task, fps=20, max_steps=600):
    """Run synchronous policy actions at the dataset rate in simulated time.

    Policy latency slows wall-clock playback; physics does not run between
    observations and actions. The environment's command limits are logged.
    """
    import time
    import torch

    if env.action_type != "joint_angle":
        raise ValueError("Set action_type='joint_angle' for absolute OMY joint commands")
    substeps = round(1 / (fps * env.env.dt))
    if substeps < 1 or not np.isclose(substeps * env.env.dt, 1 / fps, atol=1e-6):
        raise ValueError("Physics timestep must divide the dataset control interval")
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()
    predicted, applied, latency, frames = [], [], [], []
    success = False
    for _ in range(max_steps):
        agent, wrist = env.grab_image()
        def image_tensor(rgb):
            rgb = np.asarray(Image.fromarray(rgb).resize((256, 256)), dtype=np.uint8).copy()
            return torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        observation = {"observation.images.image": image_tensor(agent),
                       "observation.images.image2": image_tensor(wrist),
                       "observation.state": torch.from_numpy(env.get_ee_pose()), "task": task}
        tic = time.perf_counter()
        with torch.inference_mode():
            action = postprocessor(policy.select_action(preprocessor(observation)))
        command = action.detach().cpu().numpy().reshape(-1)
        command = _vector(command, 7, "predicted joint command")
        latency.append(time.perf_counter() - tic)
        predicted.append(command.copy())
        env.step(command)
        applied.append(env.get_commanded_joint_action().copy())
        for _ in range(substeps):
            env.step_env()
        frames.append(np.asarray(agent).copy())
        if env.check_success():
            success = True
            break
    return {"success": bool(success), "steps": len(predicted), "fps": fps,
            "sim_seconds": len(predicted) / fps, "inference_seconds": latency,
            "predicted_actions": predicted, "applied_actions": applied, "frames": frames}
