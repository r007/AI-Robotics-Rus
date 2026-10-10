#!/usr/bin/env python3
"""Run Microduck ONNX policies in MuJoCo and record an RDK-side MP4."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import socket
import struct
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

import mujoco
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont


DEFAULT_POSE = np.array(
    [
        0.0,
        -0.0873,
        -0.4579,
        -0.0049,
        0.4530,
        0.3491,
        0.3491,
        0.0,
        0.0,
        0.0,
        0.0873,
        0.4579,
        0.0049,
        -0.4530,
    ],
    dtype=np.float32,
)
EXPECTED_OBSERVATIONS = 61
EXPECTED_ACTIONS = 14
TIMESTEP = 0.005
DECIMATION = 4
CONTROL_HZ = 1.0 / (TIMESTEP * DECIMATION)
POLICY_MAGIC = b"MDP1"
POLICY_HELLO = struct.Struct("!4sHH32s")
POLICY_COUNT = struct.Struct("!I")

SCENE_XML = """
<mujoco model="microduck-formation">
  <visual>
    <global offwidth="1920" offheight="1080"/>
    <headlight diffuse="0.72 0.72 0.72" ambient="0.38 0.38 0.38" specular="0.12 0.12 0.12"/>
    <rgba haze="0.12 0.18 0.24 1"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.42 0.58 0.72" rgb2="0.04 0.07 0.10" width="512" height="3072"/>
    <texture type="2d" name="groundtex" builtin="checker" mark="edge"
      rgb1="0.12 0.16 0.20" rgb2="0.25 0.31 0.36" markrgb="0.85 0.85 0.85"
      width="512" height="512"/>
    <material name="groundmat" texture="groundtex" texuniform="true" texrepeat="18 18" reflectance="0.16"/>
  </asset>
  <worldbody>
    <light pos="0 -2 4" dir="0 0.4 -1" directional="true"/>
    <light pos="-2 2 2" dir="0.7 -0.5 -1" directional="true" diffuse="0.35 0.38 0.42"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="groundmat"/>
  </worldbody>
</mujoco>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--microduck-repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, help="Local ONNX path; unnecessary with --policy-host")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--cols", type=int, default=1)
    parser.add_argument("--spacing-x", type=float, default=0.32)
    parser.add_argument("--spacing-y", type=float, default=0.30)
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--speed", type=float, default=0.30)
    parser.add_argument("--yaw-rate", type=float, default=0.0)
    parser.add_argument("--camera-azimuth", type=float, default=145.0)
    parser.add_argument("--camera-elevation", type=float, default=-22.0)
    parser.add_argument("--camera-distance", type=float)
    parser.add_argument("--no-bam", action="store_true")
    parser.add_argument("--policy-host", help="RDK policy server address; omit for local ONNX")
    parser.add_argument("--policy-port", type=int, default=8765)
    parser.add_argument("--policy-timeout", type=float, default=1.0)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_spec(robot_xml: Path, rows: int, cols: int, spacing_x: float, spacing_y: float) -> mujoco.MjSpec:
    parent = mujoco.MjSpec.from_string(SCENE_XML)
    for row in range(rows):
        for col in range(cols):
            index = row * cols + col
            x = -row * spacing_x
            y = (col - (cols - 1) / 2.0) * spacing_y
            child = mujoco.MjSpec.from_file(str(robot_xml))
            frame = parent.worldbody.add_frame(pos=[x, y, 0.0])
            parent.attach(child, prefix=f"duck{index}_", frame=frame)
    return parent


def compile_with_bam(spec: mujoco.MjSpec):
    from bam.model import load_model
    from bam.mujoco import MujocoController

    bam_model = load_model(motor_name="xl330", model="m6")
    bam_model.actuator.kp = 200.0
    bam_model.actuator.vin = 7.4
    bam_model.actuator.max_current = None
    force_limit = bam_model.actuator.vin * bam_model.kt.value / bam_model.R.value

    actuator_names: list[str] = []
    for actuator in spec.actuators:
        target = actuator.target
        target_name = target.name if hasattr(target, "name") else str(target)
        if target_name.startswith("passive_"):
            continue
        actuator.set_to_motor()
        actuator.forcelimited = True
        actuator.forcerange = (-force_limit, force_limit)
        actuator.ctrllimited = False
        actuator.gear = [1.0, 0, 0, 0, 0, 0]
        actuator_names.append(actuator.name)
        for joint in spec.joints:
            if joint.name == target_name:
                try:
                    joint.damping = 0.0
                except TypeError:
                    joint.damping = np.zeros((3, 1))
                joint.frictionloss = 0.0
                joint.solref_friction = (-5.0e4, -2.0e2)
                joint.solimp_friction = (0.99, 0.9999, 0.001, 0.5, 2.0)
                break

    model = spec.compile()
    model.opt.timestep = TIMESTEP
    data = mujoco.MjData(model)
    controller_parameters = inspect.signature(MujocoController).parameters
    voltage_drop = (
        {"vin_drop_gain": 0.1}
        if "vin_drop_gain" in controller_parameters
        else {"vin_drop_resistance": 0.1 * bam_model.kt.value}
    )
    controller = MujocoController(
        bam_model,
        actuator_names,
        model,
        data,
        vin_min=6.0,
        **voltage_drop,
    )
    return model, data, controller


def compile_position_actuators(spec: mujoco.MjSpec):
    model = spec.compile()
    model.opt.timestep = TIMESTEP
    return model, mujoco.MjData(model), None


def rotate_inverse(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    w = quaternion[0]
    xyz = quaternion[1:4]
    cross = np.cross(xyz, vector) * 2.0
    return vector - w * cross + np.cross(xyz, cross)


@dataclass
class DuckState:
    prefix: str
    actuator_ids: np.ndarray
    qpos_indices: np.ndarray
    qvel_indices: np.ndarray
    free_qpos_address: int
    trunk_body_id: int
    gyro_sensor_address: int
    last_action: np.ndarray
    start_xy: np.ndarray
    minimum_height: float = math.inf

    def observation(self, model: mujoco.MjModel, data: mujoco.MjData, command: np.ndarray) -> np.ndarray:
        angular_velocity = data.sensordata[self.gyro_sensor_address : self.gyro_sensor_address + 3]
        quaternion = data.xquat[self.trunk_body_id].astype(np.float32)
        gravity = rotate_inverse(quaternion, np.array([0.0, 0.0, -1.0], dtype=np.float32))
        joint_position = data.qpos[self.qpos_indices].astype(np.float32) - DEFAULT_POSE
        joint_velocity = data.qvel[self.qvel_indices].astype(np.float32)
        observation = np.concatenate(
            [angular_velocity, gravity, joint_position, joint_velocity, self.last_action, command]
        ).astype(np.float32)
        if observation.shape != (EXPECTED_OBSERVATIONS,) or not np.isfinite(observation).all():
            raise RuntimeError(f"{self.prefix} produced an invalid observation: {observation.shape}")
        return observation


def make_duck_states(model: mujoco.MjModel, data: mujoco.MjData, rows: int, cols: int) -> list[DuckState]:
    states: list[DuckState] = []
    for index in range(rows * cols):
        prefix = f"duck{index}_"
        actuator_ids = np.array(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, prefix + name)
                for name in (
                    "left_hip_yaw",
                    "left_hip_roll",
                    "left_hip_pitch",
                    "left_knee",
                    "left_ankle",
                    "neck_pitch",
                    "head_pitch",
                    "head_yaw",
                    "head_roll",
                    "right_hip_yaw",
                    "right_hip_roll",
                    "right_hip_pitch",
                    "right_knee",
                    "right_ankle",
                )
            ],
            dtype=np.int32,
        )
        if np.any(actuator_ids < 0):
            raise RuntimeError(f"Missing actuators for {prefix}")
        joint_ids = model.actuator_trnid[actuator_ids, 0]
        qpos_indices = model.jnt_qposadr[joint_ids].astype(np.int32)
        qvel_indices = model.jnt_dofadr[joint_ids].astype(np.int32)
        free_joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, prefix + "trunk_base_freejoint"
        )
        trunk_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + "trunk_base")
        gyro_sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, prefix + "imu_ang_vel")
        gyro_address = int(model.sensor_adr[gyro_sensor_id])
        free_address = int(model.jnt_qposadr[free_joint_id])
        data.qpos[qpos_indices] = DEFAULT_POSE
        data.qpos[free_address + 2] = 0.125
        data.qpos[free_address + 3 : free_address + 7] = [1.0, 0.0, 0.0, 0.0]
        states.append(
            DuckState(
                prefix=prefix,
                actuator_ids=actuator_ids,
                qpos_indices=qpos_indices,
                qvel_indices=qvel_indices,
                free_qpos_address=free_address,
                trunk_body_id=trunk_body_id,
                gyro_sensor_address=gyro_address,
                last_action=np.zeros(EXPECTED_ACTIONS, dtype=np.float32),
                start_xy=data.qpos[free_address : free_address + 2].copy(),
            )
        )
    mujoco.mj_forward(model, data)
    return states


class Mp4Writer:
    def __init__(self, path: Path, width: int, height: int, fps: int):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s:v",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg stdin is unavailable")
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        return_code = self.process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg exited with status {return_code}")


def receive_exact(connection: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise EOFError("RDK policy server disconnected")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class LocalPolicy:
    def __init__(self, model_path: Path):
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.model_input = self.session.get_inputs()[0]
        self.model_output = self.session.get_outputs()[0]
        if self.model_input.shape[-1] != EXPECTED_OBSERVATIONS or self.model_output.shape[-1] != EXPECTED_ACTIONS:
            raise RuntimeError(f"Unexpected ONNX contract: {self.model_input.shape} -> {self.model_output.shape}")
        self.model_hash = file_sha256(model_path)
        self.description = "local ONNX Runtime"

    def infer(self, observations: np.ndarray) -> np.ndarray:
        actions = np.empty((len(observations), EXPECTED_ACTIONS), dtype=np.float32)
        for index, observation in enumerate(observations):
            actions[index] = self.session.run(
                [self.model_output.name],
                {self.model_input.name: observation.reshape(1, EXPECTED_OBSERVATIONS)},
            )[0].reshape(EXPECTED_ACTIONS)
        return actions

    def close(self) -> None:
        return None


class RemotePolicy:
    def __init__(self, host: str, port: int, timeout: float):
        self.connection = socket.create_connection((host, port), timeout=timeout)
        self.connection.settimeout(timeout)
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        magic, observations, actions, model_hash = POLICY_HELLO.unpack(
            receive_exact(self.connection, POLICY_HELLO.size)
        )
        if magic != POLICY_MAGIC or observations != EXPECTED_OBSERVATIONS or actions != EXPECTED_ACTIONS:
            raise RuntimeError(
                f"Unexpected RDK policy handshake: {magic!r}, {observations} -> {actions}"
            )
        self.model_hash = model_hash.hex()
        self.description = f"RDK TCP policy server {host}:{port}"

    def infer(self, observations: np.ndarray) -> np.ndarray:
        observations = np.asarray(observations, dtype="<f4")
        batch_size = len(observations)
        self.connection.sendall(POLICY_COUNT.pack(batch_size) + observations.tobytes())
        returned_batch = POLICY_COUNT.unpack(receive_exact(self.connection, POLICY_COUNT.size))[0]
        if returned_batch != batch_size:
            raise RuntimeError(f"RDK returned batch {returned_batch}, expected {batch_size}")
        payload = receive_exact(self.connection, batch_size * EXPECTED_ACTIONS * 4)
        return np.frombuffer(payload, dtype="<f4").reshape(batch_size, EXPECTED_ACTIONS).copy()

    def close(self) -> None:
        try:
            self.connection.sendall(POLICY_COUNT.pack(0))
        finally:
            self.connection.close()


def main() -> None:
    args = parse_args()
    if args.rows <= 0 or args.cols <= 0 or args.duration <= 0 or args.fps <= 0:
        raise ValueError("rows, cols, duration, and fps must be positive")
    if args.width % 2 or args.height % 2:
        raise ValueError("H.264 output width and height must be even")

    repo = args.microduck_repo.expanduser().resolve()
    robot_xml = repo / "src/mjlab_microduck/robot/microduck/robot_groundcontact.xml"
    model_path = args.model.expanduser().resolve() if args.model else None
    output_path = args.output.expanduser().resolve()
    if not robot_xml.is_file():
        raise FileNotFoundError(robot_xml)
    if model_path is not None and not model_path.is_file():
        raise FileNotFoundError(model_path)
    if not args.policy_host and model_path is None:
        raise ValueError("--model is required when --policy-host is not set")

    policy = (
        RemotePolicy(args.policy_host, args.policy_port, args.policy_timeout)
        if args.policy_host
        else LocalPolicy(model_path)
    )

    spec = build_spec(robot_xml, args.rows, args.cols, args.spacing_x, args.spacing_y)
    if args.no_bam:
        model, data, bam_controller = compile_position_actuators(spec)
        actuator_mode = "MuJoCo position actuators"
    else:
        model, data, bam_controller = compile_with_bam(spec)
        actuator_mode = "BAM M6 XL330"
    ducks = make_duck_states(model, data, args.rows, args.cols)

    if bam_controller is not None:
        if hasattr(bam_controller, "reset"):
            bam_controller.reset(data.qpos)
        else:
            bam_controller.last_ts = data.time
        for duck in ducks:
            bam_controller.q_target[duck.actuator_ids] = DEFAULT_POSE
    else:
        for duck in ducks:
            data.ctrl[duck.actuator_ids] = DEFAULT_POSE
    mujoco.mj_forward(model, data)

    command = np.zeros(13, dtype=np.float32)
    command[:3] = [args.speed, 0.0, args.yaw_rate]
    control_steps = int(round(args.duration * CONTROL_HZ))
    expected_frames = int(round(args.duration * args.fps))
    policy_latencies_ns: list[int] = []

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth = args.camera_azimuth
    camera.elevation = args.camera_elevation
    camera.distance = args.camera_distance or max(0.8, 0.72 + 0.33 * max(args.rows, args.cols))
    writer = Mp4Writer(output_path, args.width, args.height, args.fps)
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    font = ImageFont.truetype(str(font_path), 22) if font_path.is_file() else ImageFont.load_default()

    rendered_frames = 0
    run_start_ns = time.perf_counter_ns()
    try:
        for control_step in range(control_steps):
            observations = np.stack([duck.observation(model, data, command) for duck in ducks])
            policy_start_ns = time.perf_counter_ns()
            actions = policy.infer(observations)
            policy_latencies_ns.append(time.perf_counter_ns() - policy_start_ns)
            if actions.shape != (len(ducks), EXPECTED_ACTIONS) or not np.isfinite(actions).all():
                raise RuntimeError(f"Policy returned invalid actions: {actions.shape}")
            for duck, action in zip(ducks, actions):
                if action.shape != (EXPECTED_ACTIONS,) or not np.isfinite(action).all():
                    raise RuntimeError(f"{duck.prefix} produced an invalid action")
                duck.last_action = action
                targets = DEFAULT_POSE + action
                if bam_controller is not None:
                    bam_controller.q_target[duck.actuator_ids] = targets
                else:
                    data.ctrl[duck.actuator_ids] = targets

            for _ in range(DECIMATION):
                if bam_controller is not None:
                    bam_controller.update()
                mujoco.mj_step(model, data)

            for duck in ducks:
                height = float(data.qpos[duck.free_qpos_address + 2])
                duck.minimum_height = min(duck.minimum_height, height)

            frames_due = min(
                expected_frames,
                int(math.floor((control_step + 1) * args.fps / CONTROL_HZ + 1.0e-9)),
            )
            if rendered_frames < frames_due:
                xy = np.array(
                    [data.qpos[duck.free_qpos_address : duck.free_qpos_address + 2] for duck in ducks]
                )
                center = xy.mean(axis=0)
                camera.lookat[:] = [float(center[0]), float(center[1]), 0.11]
                renderer.update_scene(data, camera=camera)
                frame = renderer.render().copy()
                title = f"RDK X5 | Microduck ONNX | {args.rows}x{args.cols} | {CONTROL_HZ:.0f} Hz"
                image = Image.fromarray(frame)
                drawing = ImageDraw.Draw(image)
                drawing.rounded_rectangle((14, 12, 530, 48), radius=5, fill=(15, 22, 28))
                drawing.text((24, 17), title, font=font, fill=(245, 245, 245))
                frame = np.asarray(image)
                writer.write(frame)
                rendered_frames += 1
    finally:
        writer.close()
        renderer.close()
        policy.close()

    wall_seconds = (time.perf_counter_ns() - run_start_ns) / 1_000_000_000.0
    latency_ms = np.asarray(policy_latencies_ns, dtype=np.float64) / 1_000_000.0
    duck_results = []
    for duck in ducks:
        end_xy = data.qpos[duck.free_qpos_address : duck.free_qpos_address + 2].copy()
        displacement = end_xy - duck.start_xy
        duck_results.append(
            {
                "name": duck.prefix.rstrip("_"),
                "start_xy_m": duck.start_xy.tolist(),
                "end_xy_m": end_xy.tolist(),
                "displacement_xy_m": displacement.tolist(),
                "forward_speed_mps": float(displacement[0] / args.duration),
                "minimum_trunk_height_m": duck.minimum_height,
            }
        )

    report = {
        "platform": "RDK X5",
        "safe_mode": "MuJoCo simulation only; no servo commands",
        "mujoco_version": mujoco.__version__,
        "onnxruntime_version": ort.__version__,
        "policy_backend": policy.description,
        "onnx_sha256": policy.model_hash,
        "onnx_contract": [[1, EXPECTED_OBSERVATIONS], [1, EXPECTED_ACTIONS]],
        "actuator_model": actuator_mode,
        "formation": {"rows": args.rows, "cols": args.cols, "ducks": len(ducks)},
        "simulation": {
            "duration_s": args.duration,
            "control_hz": CONTROL_HZ,
            "control_steps": control_steps,
            "wall_seconds": wall_seconds,
            "realtime_factor": args.duration / wall_seconds,
        },
        "policy_batch_or_roundtrip_latency_ms": {
            "mean": float(latency_ms.mean()),
            "p50": float(np.percentile(latency_ms, 50)),
            "p90": float(np.percentile(latency_ms, 90)),
            "p99": float(np.percentile(latency_ms, 99)),
            "max": float(latency_ms.max()),
        },
        "video": {
            "path": str(output_path),
            "frames": rendered_frames,
            "fps": args.fps,
            "resolution": [args.width, args.height],
            "bytes": output_path.stat().st_size,
        },
        "ducks": duck_results,
    }
    rendered_report = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered_report)
    report_path = args.report or output_path.with_suffix(".json")
    report_path = report_path.expanduser().resolve()
    report_path.write_text(rendered_report + "\n", encoding="utf-8")
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
