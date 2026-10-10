"""Record a finite, close-up rollout from a local Microduck checkpoint."""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from rsl_rl.runners import OnPolicyRunner

TASK_ID = "Mjlab-Velocity-Flat-MicroDuck"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--speed", type=float, default=0.25)
    parser.add_argument("--motion", choices=("walk", "dance"), default="walk")
    parser.add_argument("--camera-distance", type=float, default=0.65)
    parser.add_argument("--camera-elevation", type=float, default=-8.0)
    parser.add_argument("--camera-azimuth", type=float, default=90.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def set_dance_commands(env: ManagerBasedRlEnv, step: int) -> None:
    """Apply a bounded 12-second velocity and head-pose choreography."""
    t = step * env.step_dt
    phase = t % 12.0

    twist = env.command_manager.get_term("twist")
    command = twist.vel_command_b
    command.zero_()
    if 2.0 <= phase < 4.0:
        command[:, 1] = 0.24 * math.sin(math.pi * (phase - 2.0))
    elif 4.0 <= phase < 6.0:
        command[:, 2] = 0.85 * math.sin(math.pi * (phase - 4.0))
    elif 6.0 <= phase < 8.0:
        command[:, 0] = 0.35 * math.sin(math.pi * (phase - 6.0))
    elif 8.0 <= phase < 10.0:
        command[:, 0] = 0.25
        command[:, 2] = 0.65
    elif phase >= 10.0:
        command[:, 1] = 0.18 * math.sin(2.0 * math.pi * (phase - 10.0))
        command[:, 2] = 0.55 * math.sin(math.pi * (phase - 10.0))
    twist.vel_command_w.copy_(command)

    head = env.command_manager.get_command("head_pose")
    head[:, 0] = 0.25 * math.sin(math.pi * t)
    head[:, 1] = -0.20 * math.sin(math.pi * t)
    head[:, 2] = 0.75 * math.sin(2.0 * math.pi * t / 3.0)
    head[:, 3] = 0.16 * math.sin(2.0 * math.pi * t / 1.5)


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    configure_torch_backends()
    env_cfg = load_env_cfg(TASK_ID, play=True)
    agent_cfg = load_rl_cfg(TASK_ID)

    env_cfg.seed = args.seed
    env_cfg.scene.num_envs = 1
    env_cfg.viewer.width = args.width
    env_cfg.viewer.height = args.height
    env_cfg.viewer.distance = args.camera_distance
    env_cfg.viewer.elevation = args.camera_elevation
    env_cfg.viewer.azimuth = args.camera_azimuth

    # Keep the showcase focused on velocity tracking rather than push recovery.
    env_cfg.events.pop("push_robot", None)
    twist = env_cfg.commands["twist"]
    twist.heading_command = False
    twist.rel_standing_envs = 0.0
    twist.rel_heading_envs = 0.0
    twist.rel_forward_envs = 1.0 if args.motion == "walk" else 0.0
    twist.rel_turn_in_place_envs = 0.0
    twist.resampling_time_range = (1000.0, 1000.0)
    twist.ranges.lin_vel_x = (args.speed, args.speed)
    twist.ranges.lin_vel_y = (0.0, 0.0)
    twist.ranges.ang_vel_z = (0.0, 0.0)
    twist.ranges.heading = None
    if args.motion == "dance":
        env_cfg.commands["head_pose"].ranges = ((0.0, 0.0),) * 4
        env_cfg.commands["body_pose"].ranges = ((0.0, 0.0),) * 6

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base_env = ManagerBasedRlEnv(
        cfg=env_cfg, device=args.device, render_mode="rgb_array"
    )
    video_env = VideoRecorder(
        base_env,
        video_folder=output_dir,
        step_trigger=lambda step: step == 0,
        video_length=args.frames,
        name_prefix=f"microduck-{args.motion}",
        disable_logger=False,
    )
    env = RslRlVecEnvWrapper(video_env, clip_actions=agent_cfg.clip_actions)

    runner_cls = load_runner_cls(TASK_ID) or OnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=args.device)
    runner.load(
        str(checkpoint),
        load_cfg={"actor": True},
        strict=True,
        map_location=args.device,
    )
    policy = runner.get_inference_policy(device=args.device)

    for step in range(args.frames):
        with torch.inference_mode():
            if args.motion == "dance":
                set_dance_commands(base_env, step)
            observations = env.get_observations()
            actions = policy(observations)
            env.step(actions)

    env.close()
    print(f"Recorded {args.frames} frames in {output_dir}")


if __name__ == "__main__":
    main()
