"""Minimal Pinocchio phase bridge for official Unitree wholebody control."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("<PROJECT_ROOT>")
G1_URDF = Path("<DATA_ROOT>/refs/xr_teleoperate/assets/g1/g1_body29_hand14.urdf")
ARM_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def parse_actuated_joint_names(urdf_path: Path = G1_URDF) -> list[str]:
    root = ET.parse(urdf_path).getroot()
    return [
        joint.attrib["name"]
        for joint in root.findall("joint")
        if joint.attrib.get("type") not in {"fixed", "floating"} and joint.attrib.get("name")
    ]


def quat_to_rot(q_wxyz: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    w, x, y, z = [float(v) for v in q_wxyz]
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def pose_matrix(pos: np.ndarray, quat_wxyz: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = quat_to_rot(quat_wxyz)
    out[:3, 3] = np.asarray(pos, dtype=np.float64)
    return out


def world_to_base_target(robot: Any, target_pos_w: np.ndarray, target_rot_w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    names = robot.data.body_names
    pelvis_idx = names.index("pelvis")
    pelvis = robot.data.body_link_state_w[0, pelvis_idx, :7].detach().cpu().numpy().astype(np.float64)
    T_w_b = pose_matrix(pelvis[:3], pelvis[3:7])
    R_b_w = T_w_b[:3, :3].T
    pos_b = R_b_w @ (np.asarray(target_pos_w, dtype=np.float64) - T_w_b[:3, 3])
    rot_b = R_b_w @ np.asarray(target_rot_w, dtype=np.float64)
    return pos_b, rot_b


def current_urdf_q(robot: Any, urdf_names: list[str]) -> np.ndarray:
    sim_names = robot.data.joint_names
    sim_index = {name: i for i, name in enumerate(sim_names)}
    q_sim = robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float64)
    values: list[float] = []
    missing: list[str] = []
    for name in urdf_names:
        if name in sim_index:
            values.append(float(q_sim[sim_index[name]]))
        elif "_hand_" in name:
            values.append(0.0)
        else:
            missing.append(name)
    if missing:
        raise RuntimeError(f"URDF joints not in Isaac robot: {missing}")
    return np.asarray(values, dtype=np.float64)


def current_arm_motor_positions(robot: Any) -> list[float]:
    sim_names = robot.data.joint_names
    sim_index = {name: i for i, name in enumerate(sim_names)}
    q_sim = robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float64)
    motor_positions = [0.0] * 29
    for i, name in enumerate(ARM_NAMES):
        motor_positions[15 + i] = float(q_sim[sim_index[name]])
    return motor_positions


def q_to_motor_positions(q: np.ndarray, urdf_names: list[str]) -> list[float]:
    q_map = {name: float(q[i]) for i, name in enumerate(urdf_names)}
    motor_positions = [0.0] * 29
    for i, name in enumerate(ARM_NAMES):
        motor_positions[15 + i] = q_map[name]
    return motor_positions


def current_wrist_rot(robot: Any, arm: str = "left") -> np.ndarray:
    idx = robot.data.body_names.index(f"{arm}_wrist_yaw_link")
    state = robot.data.body_link_state_w[0, idx, :7].detach().cpu().numpy().astype(np.float64)
    return quat_to_rot(state[3:7])


def tcp_offset_in_wrist(robot: Any, arm: str = "left") -> np.ndarray:
    names = robot.data.body_names
    wrist_idx = names.index(f"{arm}_wrist_yaw_link")
    tip1_idx = names.index(f"{arm}_hand_Link1_3")
    tip2_idx = names.index(f"{arm}_hand_Link2_3")
    poses = robot.data.body_link_state_w[0, :, :7].detach().cpu().numpy().astype(np.float64)
    wrist_pos = poses[wrist_idx, :3]
    wrist_rot = quat_to_rot(poses[wrist_idx, 3:7])
    tcp_pos = 0.5 * (poses[tip1_idx, :3] + poses[tip2_idx, :3])
    return wrist_rot.T @ (tcp_pos - wrist_pos)


def tcp_to_wrist_target(
    robot: Any,
    tcp_pos_w: np.ndarray,
    wrist_rot_w: np.ndarray | None = None,
    arm: str = "left",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wrist_rot = current_wrist_rot(robot, arm) if wrist_rot_w is None else np.asarray(wrist_rot_w, dtype=np.float64)
    offset = tcp_offset_in_wrist(robot, arm)
    wrist_pos_w = np.asarray(tcp_pos_w, dtype=np.float64) - wrist_rot @ offset
    return wrist_pos_w, wrist_rot, offset


def tcp_position_w(robot: Any, arm: str = "left") -> np.ndarray:
    names = robot.data.body_names
    poses = robot.data.body_link_state_w[0, :, :7].detach().cpu().numpy().astype(np.float64)
    tip1 = poses[names.index(f"{arm}_hand_Link1_3"), :3]
    tip2 = poses[names.index(f"{arm}_hand_Link2_3"), :3]
    return 0.5 * (tip1 + tip2)


def solve_arm_phase(
    robot: Any,
    *,
    target_tcp_w: np.ndarray,
    wrist_rot_w: np.ndarray | None,
    out_dir: Path,
    label: str,
    steps: int = 40,
    arm: str = "left",
    primary_side: str = "left",
    backend: str = "ark",
) -> tuple[list[np.ndarray], dict[str, Any], list[str]]:
    urdf_names = parse_actuated_joint_names(G1_URDF)
    q = current_urdf_q(robot, urdf_names)
    names = robot.data.body_names
    left_idx = names.index("left_wrist_yaw_link")
    right_idx = names.index("right_wrist_yaw_link")
    left_cur = robot.data.body_link_state_w[0, left_idx, :7].detach().cpu().numpy().astype(np.float64)
    right_cur = robot.data.body_link_state_w[0, right_idx, :7].detach().cpu().numpy().astype(np.float64)

    target_wrist_pos_w, target_wrist_rot_w, tcp_offset = tcp_to_wrist_target(
        robot,
        tcp_pos_w=np.asarray(target_tcp_w, dtype=np.float64),
        wrist_rot_w=wrist_rot_w,
        arm=arm,
    )
    left_target_pos_w = left_cur[:3]
    left_target_rot_w = quat_to_rot(left_cur[3:7])
    right_target_pos_w = right_cur[:3]
    right_target_rot_w = quat_to_rot(right_cur[3:7])
    if arm == "left":
        left_target_pos_w = target_wrist_pos_w
        left_target_rot_w = target_wrist_rot_w
    else:
        right_target_pos_w = target_wrist_pos_w
        right_target_rot_w = target_wrist_rot_w
        primary_side = "right"
    left_pos_b, left_rot_b = world_to_base_target(robot, left_target_pos_w, left_target_rot_w)
    right_pos_b, right_rot_b = world_to_base_target(robot, right_target_pos_w, right_target_rot_w)
    request = {
        "urdf_path": str(G1_URDF),
        "q": q.tolist(),
        "left_target": {"pos": left_pos_b.tolist(), "rot": np.asarray(left_rot_b, dtype=float).tolist()},
        "right_target": {"pos": right_pos_b.tolist(), "rot": np.asarray(right_rot_b, dtype=float).tolist()},
        "primary_side": primary_side,
        "steps": int(steps),
        "inner_iters": 5,
        "gain": 0.4,
        "damping": 1e-3,
        "max_step": 0.08,
    }
    req_path = out_dir / f"{label}_{backend}_request.json"
    plan_path = out_dir / f"{label}_{backend}_plan.json"
    req_path.write_text(json.dumps(request, indent=2), encoding="utf-8")
    if backend == "pinocchio":
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "solve_g1_pinocchio_ik.py"),
            "--input",
            str(req_path),
            "--output",
            str(plan_path),
        ]
    elif backend == "official":
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "solve_g1_unitree_official_ik.py"),
            "--input",
            str(req_path),
            "--output",
            str(plan_path),
        ]
    elif backend == "ark":
        cmd = [
            str(ROOT / "scripts" / "solve_g1_ark_ik.py"),
            "--input",
            str(req_path),
            "--output",
            str(plan_path),
        ]
    else:
        raise RuntimeError(f"unsupported offline IK backend: {backend}")
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    proc = subprocess.run(cmd, env=env, text=True, capture_output=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"offline IK failed rc={proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    result = json.loads(plan_path.read_text(encoding="utf-8"))
    waypoints = [np.asarray(row, dtype=np.float64) for row in result["plan"]]
    info = {
        "solver": result.get("solver", f"{backend}_subprocess"),
        "backend": backend,
        "primary_side": primary_side,
        "arm": arm,
        "final_left_pos_error_m": result.get("final_position_error_m"),
        "target_tcp_w": np.asarray(target_tcp_w, dtype=float).tolist(),
        "target_wrist_w": np.asarray(target_wrist_pos_w, dtype=float).tolist(),
        "target_pos_b": left_pos_b.tolist(),
        "right_target_pos_b": right_pos_b.tolist(),
        "plan_path": str(plan_path),
        "request_path": str(req_path),
        "tcp_offset_wrist": tcp_offset.astype(float).tolist(),
    }
    return waypoints, info, urdf_names


def solve_left_arm_phase(
    robot: Any,
    *,
    target_tcp_w: np.ndarray,
    wrist_rot_w: np.ndarray | None,
    out_dir: Path,
    label: str,
    steps: int = 40,
    primary_side: str = "left",
    backend: str = "ark",
) -> tuple[list[np.ndarray], dict[str, Any], list[str]]:
    return solve_arm_phase(
        robot,
        target_tcp_w=target_tcp_w,
        wrist_rot_w=wrist_rot_w,
        out_dir=out_dir,
        label=label,
        steps=steps,
        arm="left",
        primary_side=primary_side,
        backend=backend,
    )


def execute_waypoints(
    env: Any,
    provider: Any,
    gripper_dds: Any,
    waypoints: list[np.ndarray],
    urdf_names: list[str],
    *,
    stride: int = 1,
    arm: str = "left",
) -> list[dict[str, Any]]:
    robot = env.scene["robot"]
    obj = env.scene["object"]
    records: list[dict[str, Any]] = []
    for idx, q in enumerate(waypoints[:: max(1, int(stride))]):
        provider.robot_dds.positions = q_to_motor_positions(q, urdf_names)
        provider.get_action(env)
        if idx % 5 == 0 or idx == len(waypoints) - 1:
            records.append(
                {
                    "idx": idx,
                    "object_pos_w": obj.data.root_pos_w[0].detach().cpu().numpy().astype(float).tolist(),
                    "root_pos_w": robot.data.root_pos_w[0].detach().cpu().numpy().astype(float).tolist(),
                    "gripper_left": float(getattr(gripper_dds, "left", 0.0)),
                    "active_tcp_w": tcp_position_w(robot, arm).astype(float).tolist(),
                }
            )
    return records
