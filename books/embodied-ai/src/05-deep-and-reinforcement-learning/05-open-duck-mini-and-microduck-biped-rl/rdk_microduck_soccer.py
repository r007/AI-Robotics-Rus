#!/usr/bin/env python3
"""Render MuJoCo on Ubuntu while RDK X5 runs walking and kick policies."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from rdk_microduck_video import (
    CONTROL_HZ,
    DECIMATION,
    DEFAULT_POSE,
    Mp4Writer,
    compile_with_bam,
    make_duck_states,
)
from rdk_policy_client import RdkPolicyClient


BALL_RADIUS = 0.035
KICK_DURATION_S = 3.0
REACQUIRE_DELAY_S = 0.9
MIN_SUCCESSFUL_KICK_M = 0.025

SOCCER_SCENE_XML = """
<mujoco model="microduck-rdk-soccer">
  <visual>
    <global offwidth="1920" offheight="1080"/>
    <headlight diffuse="0.72 0.72 0.72" ambient="0.38 0.38 0.38" specular="0.12 0.12 0.12"/>
    <rgba haze="0.12 0.18 0.24 1"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.42 0.58 0.72" rgb2="0.04 0.07 0.10" width="512" height="3072"/>
    <texture type="2d" name="groundtex" builtin="checker" mark="edge"
      rgb1="0.08 0.11 0.14" rgb2="0.18 0.22 0.26" markrgb="0.90 0.42 0.08"
      width="512" height="512"/>
    <material name="groundmat" texture="groundtex" texuniform="true" texrepeat="18 18" reflectance="0.12"/>
  </asset>
  <worldbody>
    <light pos="0 -2 4" dir="0 0.4 -1" directional="true"/>
    <light pos="-2 2 2" dir="0.7 -0.5 -1" directional="true" diffuse="0.35 0.38 0.42"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="groundmat"/>
    <body name="ball" pos="0.42 -0.11 0.035">
      <freejoint name="ball_free"/>
      <geom name="ball_geom" type="sphere" size="0.035" mass="0.015"
        rgba="0.96 0.42 0.05 1" friction="0.5 0.005 0.0001"/>
    </body>
  </worldbody>
</mujoco>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--microduck-repo", type=Path, required=True)
    parser.add_argument("--policy-host", required=True, help="RDK X5 address")
    parser.add_argument("--policy-port", type=int, default=8766)
    parser.add_argument("--policy-timeout", type=float, default=2.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--max-kicks", type=int, default=3)
    parser.add_argument("--max-duration", type=float, default=60.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera-distance", type=float, default=0.85)
    return parser.parse_args()


def build_soccer_spec(robot_xml: Path) -> mujoco.MjSpec:
    parent = mujoco.MjSpec.from_string(SOCCER_SCENE_XML)
    robot = mujoco.MjSpec.from_file(str(robot_xml))
    frame = parent.worldbody.add_frame(pos=[0.0, 0.0, 0.0])
    parent.attach(robot, prefix="duck0_", frame=frame)
    return parent


def yaw_from_quaternion(quaternion: np.ndarray) -> float:
    w, x, y, z = quaternion
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def ball_in_robot_frame(robot_xy: np.ndarray, robot_yaw: float, ball_xy: np.ndarray) -> np.ndarray:
    delta = ball_xy - robot_xy
    cosine = math.cos(robot_yaw)
    sine = math.sin(robot_yaw)
    return np.array(
        [cosine * delta[0] + sine * delta[1], -sine * delta[0] + cosine * delta[1]],
        dtype=np.float32,
    )


def bounded_command(error: float, gain: float, minimum: float, maximum: float) -> float:
    """Map a nonzero pose error to a command above the walking policy dead zone."""
    if abs(error) < 1.0e-6:
        return 0.0
    magnitude = float(np.clip(abs(error) * gain, minimum, maximum))
    return math.copysign(magnitude, error)


@dataclass
class SoccerController:
    max_kicks: int
    mode: str = "searching"
    foot: str = "right"
    kicks: int = 0
    attempts: int = 0
    timer_s: float = 0.0
    kick_start_ball: np.ndarray | None = None
    kick_results: list[dict] = field(default_factory=list)

    def select(self, ball_local: np.ndarray, ball_world: np.ndarray, dt: float) -> tuple[str, np.ndarray]:
        if self.mode == "kicking":
            self.timer_s += dt
            if self.timer_s >= KICK_DURATION_S:
                assert self.kick_start_ball is not None
                displacement = float(np.linalg.norm(ball_world - self.kick_start_ball))
                successful = displacement >= MIN_SUCCESSFUL_KICK_M
                self.attempts += 1
                self.kick_results.append(
                    {
                        "attempt": self.attempts,
                        "policy": f"kick_{self.foot}",
                        "ball_world_start_m": self.kick_start_ball.tolist(),
                        "ball_world_end_m": ball_world.tolist(),
                        "ball_displacement_m": displacement,
                        "success": successful,
                    }
                )
                if successful:
                    self.kicks += 1
                self.mode = "finished" if self.kicks >= self.max_kicks else "reacquire"
                self.timer_s = 0.0
            return f"kick_{self.foot}", np.zeros(13, dtype=np.float32)

        if self.mode == "finished":
            return "walking", np.zeros(13, dtype=np.float32)

        if self.mode == "reacquire":
            self.timer_s += dt
            if self.timer_s < REACQUIRE_DELAY_S:
                return "walking", np.zeros(13, dtype=np.float32)
            self.mode = "searching"
            self.timer_s = 0.0

        x, y = (float(ball_local[0]), float(ball_local[1]))
        distance = math.hypot(x, y)
        bearing = math.atan2(y, x)
        command = np.zeros(13, dtype=np.float32)

        if x <= 0.0 or abs(bearing) > 0.22:
            self.mode = "aligning"
            # The released walking checkpoint turns more reliably on an arc
            # than from a zero-speed yaw command.
            command[0] = 0.16 if x > 0.0 else 0.0
            command[2] = float(np.clip(2.2 * bearing, -0.8, 0.8))
            return "walking", command

        if distance > 0.22:
            self.mode = "approaching"
            command[0] = float(np.clip(1.2 * (distance - 0.13), 0.08, 0.30))
            command[2] = float(np.clip(1.8 * bearing, -0.55, 0.55))
            return "walking", command

        self.foot = "right" if y <= 0.0 else "left"
        target_y = -0.043 if self.foot == "right" else 0.043
        x_error = x - 0.095
        y_error = y - target_y
        if abs(x_error) <= 0.012 and abs(y_error) <= 0.018:
            self.mode = "kicking"
            self.timer_s = 0.0
            self.kick_start_ball = ball_world.copy()
            return f"kick_{self.foot}", np.zeros(13, dtype=np.float32)

        self.mode = "positioning-foot"
        command[0] = bounded_command(x_error, 1.5, 0.12, 0.18)
        command[1] = bounded_command(y_error, 1.8, 0.10, 0.16)
        return "walking", command


def main() -> None:
    args = parse_args()
    if args.max_kicks <= 0 or args.max_duration <= 0 or args.fps <= 0:
        raise ValueError("max-kicks, max-duration, and fps must be positive")
    if args.width % 2 or args.height % 2:
        raise ValueError("H.264 dimensions must be even")

    # Keep mapped-drive paths intact on Windows. MuJoCo's XML loader can fail
    # when a resolved native path contains non-ASCII directory names.
    repo = args.microduck_repo.expanduser().absolute()
    robot_xml = repo / "src/mjlab_microduck/robot/microduck/robot_groundcontact.xml"
    if not robot_xml.is_file():
        raise FileNotFoundError(robot_xml)
    output = args.output.expanduser().absolute()

    required = ("walking", "kick_left", "kick_right")
    client = RdkPolicyClient(
        args.policy_host,
        args.policy_port,
        args.policy_timeout,
        required_policies=required,
    )
    for name in required:
        policy = client.policies[name]
        if int(policy["observations"]) != 61 or int(policy["actions"]) != 14:
            raise RuntimeError(f"{name}: expected [1,61] -> [1,14], got {policy}")

    model, data, bam_controller = compile_with_bam(build_soccer_spec(robot_xml))
    duck = make_duck_states(model, data, 1, 1)[0]
    ball_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
    if ball_joint < 0:
        raise RuntimeError("ball_free joint is missing")
    ball_qpos = int(model.jnt_qposadr[ball_joint])
    ball_dof = int(model.jnt_dofadr[ball_joint])
    initial_ball = np.array([0.42, -0.11], dtype=np.float64)
    data.qpos[ball_qpos : ball_qpos + 7] = [*initial_ball, BALL_RADIUS, 1.0, 0.0, 0.0, 0.0]
    data.qvel[ball_dof : ball_dof + 6] = 0.0
    if hasattr(bam_controller, "reset"):
        bam_controller.reset(data.qpos)
    else:
        bam_controller.last_ts = data.time
    bam_controller.q_target[duck.actuator_ids] = DEFAULT_POSE
    mujoco.mj_forward(model, data)

    controller = SoccerController(args.max_kicks)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth = 145.0
    camera.elevation = -24.0
    camera.distance = args.camera_distance
    writer = Mp4Writer(output, args.width, args.height, args.fps)
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    font = ImageFont.truetype(str(font_path), 22) if font_path.is_file() else ImageFont.load_default()
    expected_frames = int(round(args.max_duration * args.fps))
    rendered_frames = 0
    control_steps = int(round(args.max_duration * CONTROL_HZ))
    finished_hold_steps = 0
    rendering_host = "Ubuntu MuJoCo" if platform.system() == "Linux" else "Windows validation MuJoCo"
    last_robot_xy = np.zeros(2, dtype=np.float64)
    last_ball_world = initial_ball.copy()
    last_ball_local = initial_ball.copy()
    started_ns = time.perf_counter_ns()

    try:
        for control_step in range(control_steps):
            robot_xy = data.qpos[duck.free_qpos_address : duck.free_qpos_address + 2].copy()
            robot_yaw = yaw_from_quaternion(data.qpos[duck.free_qpos_address + 3 : duck.free_qpos_address + 7])
            ball_world = data.qpos[ball_qpos : ball_qpos + 2].copy()
            ball_local = ball_in_robot_frame(robot_xy, robot_yaw, ball_world)
            last_robot_xy = robot_xy
            last_ball_world = ball_world
            last_ball_local = ball_local
            policy_name, command = controller.select(ball_local, ball_world, 1.0 / CONTROL_HZ)
            observation = duck.observation(model, data, command)
            action = client.infer(policy_name, observation)[0]
            duck.last_action = action
            bam_controller.q_target[duck.actuator_ids] = DEFAULT_POSE + action

            for _ in range(DECIMATION):
                bam_controller.update()
                mujoco.mj_step(model, data)

            frames_due = min(
                expected_frames,
                int(math.floor((control_step + 1) * args.fps / CONTROL_HZ + 1.0e-9)),
            )
            if rendered_frames < frames_due:
                camera.lookat[:] = [float(robot_xy[0]), float(robot_xy[1]), 0.11]
                renderer.update_scene(data, camera=camera)
                frame = renderer.render().copy()
                image = Image.fromarray(frame)
                drawing = ImageDraw.Draw(image)
                drawing.rounded_rectangle((14, 12, 720, 82), radius=5, fill=(15, 22, 28))
                drawing.text((24, 17), f"RDK X5 POLICY BRAIN | {rendering_host}", font=font, fill=(245, 245, 245))
                status = (
                    f"{controller.mode} | {policy_name} | kick {controller.kicks}/{args.max_kicks} "
                    f"| try {controller.attempts} | ball {float(np.linalg.norm(ball_local)):.2f} m"
                )
                drawing.text((24, 49), status, font=font, fill=(255, 145, 45))
                writer.write(np.asarray(image))
                rendered_frames += 1

            if controller.mode == "finished":
                finished_hold_steps += 1
                if finished_hold_steps >= int(1.5 * CONTROL_HZ):
                    break
    finally:
        writer.close()
        renderer.close()
        client.close()

    wall_seconds = (time.perf_counter_ns() - started_ns) / 1e9
    report = {
        "architecture": {
            "policy_brain": "RDK X5",
            "physics_and_rendering": rendering_host,
            "local_onnx_fallback": False,
            "protocol": "MDP2",
        },
        "rdk_catalog": client.catalog,
        "rdk_latency": client.latency_report(),
        "simulation": {
            "control_hz": CONTROL_HZ,
            "wall_seconds": wall_seconds,
            "rendered_frames": rendered_frames,
            "ball_placement_count": 1,
            "initial_ball_world_m": initial_ball.tolist(),
            "requested_kicks": args.max_kicks,
            "completed_kicks": controller.kicks,
            "kick_attempts": controller.attempts,
            "minimum_successful_kick_m": MIN_SUCCESSFUL_KICK_M,
            "kick_results": controller.kick_results,
            "final_controller_mode": controller.mode,
            "final_robot_world_m": last_robot_xy.tolist(),
            "final_ball_world_m": last_ball_world.tolist(),
            "final_ball_robot_frame_m": last_ball_local.tolist(),
        },
        "video": {
            "path": str(output),
            "fps": args.fps,
            "resolution": [args.width, args.height],
            "bytes": output.stat().st_size,
        },
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    report_path = (args.report or output.with_suffix(".json")).expanduser().absolute()
    report_path.write_text(rendered + "\n", encoding="utf-8")
    if controller.kicks != args.max_kicks:
        raise RuntimeError(
            f"only completed {controller.kicks}/{args.max_kicks} kicks before timeout; "
            f"see {report_path}"
        )
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
