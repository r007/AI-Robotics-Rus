"""Integration checks for v2.1 migration using the actual LeRobot v3 writer/reader."""

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from pi05_omy_utils import (CAMERAS, prepare_dataset, read_v21, rollout,
                           split_episodes, training_command, verify_alignment)


def make_source(root):
    (root / "meta").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    features = {key: {"dtype": "image", "shape": [256, 256, 3]} for key in CAMERAS}
    for key, dim in [("observation.state", 6), ("action", 7), ("obj_init", 6)]:
        features[key] = {"dtype": "float32", "shape": [dim]}
    info = {"codebase_version": "v2.1", "fps": 20, "features": features,
            "total_episodes": 3, "total_frames": 9, "chunks_size": 1000,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"}
    (root / "meta/info.json").write_text(json.dumps(info), encoding="utf-8")
    (root / "meta/tasks.jsonl").write_text('{"task_index": 0, "task": "Put mug cup on the plate"}\n', encoding="utf-8")
    # Metadata and Parquet rows are intentionally out of order.
    (root / "meta/episodes.jsonl").write_text("\n".join(json.dumps({"episode_index": e, "length": 3}) for e in [2, 0, 1]), encoding="utf-8")
    for ep in [2, 0, 1]:
        rows = []
        for f in [2, 0, 1]:
            buf = io.BytesIO()
            Image.new("RGB", (256, 256), (ep * 60, f * 60, 10)).save(buf, format="PNG")
            row = {"episode_index": ep, "frame_index": f, "index": ep * 3 + f,
                   "timestamp": f / 20, "task_index": 0,
                   "observation.state": [float(ep * 10 + f)] * 6,
                   "action": [float(ep * 10 + f)] * 6 + [f / 2],
                   "obj_init": [0.3, 0.1, 0.82, 0.4, -0.1, 0.82]}
            row.update({key: {"bytes": buf.getvalue(), "path": None} for key in CAMERAS})
            rows.append(row)
        pq.write_table(pa.Table.from_pylist(rows), root / f"data/chunk-000/episode_{ep:06d}.parquet")


class MigrationTests(unittest.TestCase):
    def test_conversion_preserves_values_images_and_boundaries(self):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        with tempfile.TemporaryDirectory() as tmp:
            source, dest = Path(tmp) / "source", Path(tmp) / "v3"
            make_source(source)
            original = {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()}
            manifest = prepare_dataset(source, dest, "local/omy-test", labels_verified=True)
            self.assertEqual(verify_alignment(dest, chunk_size=5), {"frames": 9, "episodes": 3, "chunk_size": 5})
            dataset = LeRobotDataset("local/omy-test", root=dest)
            for idx in range(9):
                ep, frame = divmod(idx, 3)
                sample = dataset[idx]
                np.testing.assert_allclose(sample["observation.state"].numpy(), [ep * 10 + frame] * 6)
                np.testing.assert_allclose(sample["action"].numpy(), [ep * 10 + frame] * 6 + [frame / 2])
                np.testing.assert_allclose(sample["observation.images.image"][:, 0, 0].numpy() * 255, [ep * 60, frame * 60, 10], atol=1e-4)
                self.assertEqual(sample["task"], "Put mug cup on the plate")
            stats = json.loads((dest / "meta/stats.json").read_text(encoding="utf-8"))
            values = [ep * 10 + frame for ep in manifest["train_episodes"] for frame in range(3)]
            self.assertAlmostEqual(stats["action"]["q01"][0], np.quantile(values, 0.01))
            self.assertEqual(stats["action"]["count"], [6])
            self.assertFalse(set(manifest["train_episodes"]) & set(manifest["heldout_episodes"]))
            self.assertEqual(original, {p.relative_to(source): p.read_bytes() for p in source.rglob("*") if p.is_file()})
            with self.assertRaises(FileExistsError):
                prepare_dataset(source, dest, "local/omy-test", labels_verified=True)
            cmd = training_command(dest, Path(tmp) / "train", steps=2)
            self.assertEqual(cmd[1:3], ["-m", "pi05_omy_train"])
            self.assertIn("--policy.use_relative_actions=false", cmd)
            self.assertIn("--steps=2", cmd)

            # Parse exactly the generated flags and load the official training dataset.
            import draccus
            from lerobot.configs.train import TrainPipelineConfig
            from lerobot.policies.pi05.configuration_pi05 import PI05Config
            from lerobot.datasets.factory import make_train_eval_datasets
            from lerobot.utils.feature_utils import dataset_to_policy_features
            cfg = draccus.parse(TrainPipelineConfig, args=cmd[3:])
            cfg.validate()
            train_dataset, eval_dataset = make_train_eval_datasets(cfg)
            self.assertIsNone(eval_dataset)
            self.assertEqual(len(train_dataset), 6)
            actual = dataset_to_policy_features(train_dataset.meta.features)
            self.assertEqual(actual["observation.state"].shape, (6,))
            self.assertEqual(actual["action"].shape, (7,))
            np.testing.assert_allclose(np.asarray(train_dataset.meta.stats["action"]["q01"]), stats["action"]["q01"])

    def test_rejects_unverified_labels_wrong_shape_and_timestamps(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            make_source(source)
            with self.assertRaises(ValueError):
                prepare_dataset(source, Path(tmp) / "new", "local/test")
            p = source / "data/chunk-000/episode_000000.parquet"
            rows = pq.read_table(p).to_pylist()
            rows[0]["action"] = [0.0] * 6
            pq.write_table(pa.Table.from_pylist(rows), p)
            with self.assertRaisesRegex(ValueError, "expected 7"):
                read_v21(source)
            rows[0]["action"] = [0.0] * 7
            rows[0]["timestamp"] = 10.0
            pq.write_table(pa.Table.from_pylist(rows), p)
            with self.assertRaisesRegex(ValueError, "timestamps"):
                read_v21(source)

    def test_no_holdout_for_one_demo(self):
        self.assertEqual(split_episodes(1), ([0], []))

    def test_rollout_uses_eef_state_absolute_commands_and_20hz_physics(self):
        import torch

        class Policy:
            def reset(self):
                pass
            def select_action(self, obs):
                self.observation = obs
                return torch.tensor([[0.1] * 6 + [0.7]])
        class Processor:
            def reset(self):
                pass
            def __call__(self, value):
                return value
        class Env:
            action_type = "joint_angle"
            env = type("Physics", (), {"dt": 0.002})()
            ticks = 0
            def grab_image(self):
                rgb = np.full((256, 256, 3), 128, dtype=np.uint8)
                return rgb, rgb
            def get_ee_pose(self):
                return np.arange(6, dtype=np.float32)
            def step(self, action):
                self.action = action
            def get_commanded_joint_action(self):
                return self.action.copy()
            def step_env(self):
                self.ticks += 1
            def check_success(self):
                return self.ticks >= 50
        env, policy = Env(), Policy()
        result = rollout(policy, Processor(), Processor(), env, task="test", max_steps=10)
        self.assertTrue(result["success"])
        self.assertEqual(result["steps"], 2)
        self.assertEqual(env.ticks, 50)
        np.testing.assert_array_equal(policy.observation["observation.state"].numpy(), np.arange(6))
        np.testing.assert_allclose(result["applied_actions"][0], [0.1] * 6 + [0.7])


class PretrainedLoaderTests(unittest.TestCase):
    def test_complete_load_is_strict_and_installed_once(self):
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from pi05_omy_train import install_strict_loader

        sentinel, calls = object(), []
        def original(cls, *args, **kwargs):
            calls.append(kwargs["strict"])
            print("All keys loaded successfully!")
            return sentinel
        with patch.object(PI05Policy, "from_pretrained", classmethod(original)):
            install_strict_loader()
            wrapped = PI05Policy.from_pretrained.__func__
            install_strict_loader()
            self.assertIs(PI05Policy.from_pretrained.__func__, wrapped)
            self.assertIs(PI05Policy.from_pretrained("local/model", strict=False), sentinel)
        self.assertEqual(calls, [True])

    def test_silent_random_initialization_fallback_is_rejected(self):
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        from pi05_omy_train import install_strict_loader

        for message in ["Returning model without loading pretrained weights",
                        "Warning: Could not load state dict"]:
            with self.subTest(message=message):
                def original(cls, *args, **kwargs):
                    print(message)
                    return object()
                with patch.object(PI05Policy, "from_pretrained", classmethod(original)):
                    install_strict_loader()
                    with self.assertRaisesRegex(RuntimeError, "not completely loaded"):
                        PI05Policy.from_pretrained("local/model")


if __name__ == "__main__":
    unittest.main()
