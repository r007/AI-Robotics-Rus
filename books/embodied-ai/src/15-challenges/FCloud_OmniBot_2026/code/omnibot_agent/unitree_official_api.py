"""Official Unitree IsaacLab backend implementing the donor-style control API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from omnibot_agent.fcloud_control_api import BasePose
from omnibot_agent.fcloud_control_api import FCloudControlApi
from omnibot_agent.fcloud_control_api import GraspPose
from omnibot_agent.fcloud_control_api import ObjectEstimate
from omnibot_agent.fcloud_control_api import PickResult
from omnibot_agent.fcloud_control_api import PlaceResult
from omnibot_agent.fcloud_control_api import TabletopTaskSpec
from omnibot_agent.fcloud_control_api import TargetRegion
from omnibot_agent.fcloud_control_api import VerificationResult
from omnibot_agent.official_pinocchio_phase import current_wrist_rot
from omnibot_agent.official_pinocchio_phase import execute_waypoints as execute_pinocchio_waypoints
from omnibot_agent.official_pinocchio_phase import solve_arm_phase
from omnibot_agent.official_pinocchio_phase import tcp_position_w
from omnibot_agent.pi05_adapter import OmniBotObservation


OBS_ARM_STATE_JOINTS = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
)
GRIPPER_JOINTS = (
    "left_hand_Joint1_1",
    "left_hand_Joint2_1",
    "right_hand_Joint1_1",
    "right_hand_Joint2_1",
)
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
CAMERA_MAP = {
    "head": "front_camera",
    "hand_left": "left_wrist_camera",
    "hand_right": "right_wrist_camera",
}
LEFT_ARM_PRESETS: dict[str, tuple[float, ...]] = {
    "official_hold": (
        0.09090921,
        0.00761812,
        0.05730354,
        -0.42793712,
        0.16531947,
        -0.11228485,
        0.06922878,
    ),
    "official_action": (
        0.03544275,
        0.01278301,
        0.05234933,
        -0.4675626,
        0.17536025,
        -0.12697633,
        0.07976142,
    ),
    "reach_extend": (
        0.337,
        0.562,
        -1.008,
        0.618,
        -0.856,
        -0.580,
        -0.829,
    ),
    "push_low": (
        0.20,
        0.52,
        -0.72,
        0.95,
        -0.15,
        -0.45,
        -0.20,
    ),
}
RIGHT_ARM_PRESETS: dict[str, tuple[float, ...]] = {
    "right_neutral": (
        -0.28,
        -0.02,
        0.06,
        -0.42,
        -0.12,
        0.10,
        0.02,
    ),
}
NAV_COMMAND_PROFILES: dict[str, tuple[float, float, float, float] | None] = {
    "dynamic_oracle": None,
    "straight": (0.45, 0.0, 0.0, 0.8),
    "left_small": (0.45, 0.08, 0.0, 0.8),
    "right_small": (0.45, -0.08, 0.0, 0.8),
    "left_yaw": (0.42, 0.04, 0.15, 0.8),
    "right_yaw": (0.42, -0.04, -0.15, 0.8),
}


@dataclass(frozen=True)
class OfficialUnitreeApiConfig:
    nav_steps: int = 220
    nav_profile_name: str = "left_yaw"
    nav_cutover_threshold_m: float = 0.26
    nav_cutover_patience: int = 24
    settle_steps: int = 60
    arm_steps: int = 120
    close_steps: int = 60
    lift_steps: int = 90
    release_steps: int = 50
    desired_offset_xy: tuple[float, float] = (-0.55, -0.20)
    left_arm_preset_name: str = "official_hold"
    right_arm_preset_name: str = "right_neutral"
    pick_mode: str = "preset_relservo"
    relservo_keep_ratio: float = 0.20
    relservo_z_offset_m: float = 0.02
    relservo_target_bias_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    relservo_steps: int = 36
    relservo_stride: int = 2
    relservo_settle_steps: int = 18
    relservo_seat_delta_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    relservo_seat_steps: int = 24
    relservo_seat_follow_object_gain: float = 0.0
    relservo_seat_follow_object_max_delta_m: float = 0.04
    post_close_settle_steps: int = 12
    post_close_seat_delta_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    post_close_seat_steps: int = 12
    ik_backend: str = "ark"
    gripper_open_value: float = -0.02
    gripper_close_value: float = 0.024
    lift_success_z_gain_m: float = 0.02
    lift_phase_delta_m: float = 0.05
    lift_phase_max_delta_m: float = 0.035


class OfficialUnitreeOracleApi(FCloudControlApi):
    """Backend bound to the official Unitree wholebody env.

    The current implementation is intentionally honest:
    - navigation and manipulation are real provider-driven motions;
    - object estimation is still GT/oracle inside the simulator;
    - this is a donor/bootstrap backend, not leaderboard-valid perception.
    """

    def __init__(
        self,
        *,
        env: Any,
        provider: Any,
        motor_positions: list[float],
        gripper_dds: Any,
        config: OfficialUnitreeApiConfig = OfficialUnitreeApiConfig(),
    ) -> None:
        self.env = env
        self.provider = provider
        self.motor_positions = motor_positions
        self.gripper_dds = gripper_dds
        self.config = config
        self._last_prompt = ""

    def get_robot_observation(self) -> OmniBotObservation:
        frame = self._capture_triplet()
        joint_state, gripper_state = self._state_from_robot()
        return OmniBotObservation(
            head_image=frame["head"],
            left_hand_image=frame["hand_left"],
            right_hand_image=frame["hand_right"],
            joint_state=joint_state,
            gripper_state=gripper_state,
            prompt=self._last_prompt,
        )

    def get_current_base_pose(self) -> BasePose:
        robot = self.env.scene["robot"]
        root_pos = robot.data.root_pos_w[0].detach().cpu().numpy().astype(float)
        yaw = 0.0
        return BasePose(xyz=(float(root_pos[0]), float(root_pos[1]), float(root_pos[2])), yaw_rad=yaw)

    def compute_table_approach_pose(self, task: TabletopTaskSpec) -> BasePose:
        estimate = self._object_estimate_from_env(task.object_name)
        base_xyz = (
            float(estimate.xyz[0] + self.config.desired_offset_xy[0]),
            float(estimate.xyz[1] + self.config.desired_offset_xy[1]),
            self.get_current_base_pose().xyz[2],
        )
        return BasePose(xyz=base_xyz, yaw_rad=0.0)

    def approach_table(self, base_pose: BasePose) -> dict[str, Any]:
        robot = self.env.scene["robot"]
        obj = self.env.scene["object"]
        desired_offset = np.asarray(self.config.desired_offset_xy, dtype=float)
        best_dist = float("inf")
        best_step = -1
        stagnant = 0
        for step in range(self.config.nav_steps):
            root_xy = robot.data.root_pos_w[0, :2].detach().cpu().numpy().astype(float)
            object_xy = obj.data.root_pos_w[0, :2].detach().cpu().numpy().astype(float)
            command = self._resolve_nav_command(
                self.config.nav_profile_name,
                root_xy,
                object_xy,
                desired_offset,
            )
            self.provider.run_command_dds.command = command
            self.provider.get_action(self.env)
            root_xy_after = robot.data.root_pos_w[0, :2].detach().cpu().numpy().astype(float)
            object_xy_after = obj.data.root_pos_w[0, :2].detach().cpu().numpy().astype(float)
            dist = float(np.linalg.norm(root_xy_after - object_xy_after))
            if dist < best_dist:
                best_dist = dist
                best_step = step
                stagnant = 0
            else:
                stagnant += 1
            if best_dist <= self.config.nav_cutover_threshold_m and stagnant >= self.config.nav_cutover_patience:
                break
        self.provider.run_command_dds.command = [0.0, 0.0, 0.0, 0.8]
        self._run_ticks(self.config.settle_steps)
        return {
            "nav_profile": self.config.nav_profile_name,
            "best_distance_xy_m": best_dist,
            "best_step": best_step,
            "stop_stagnant_steps": stagnant,
            "target_base_pose": {"xyz": list(base_pose.xyz), "yaw_rad": base_pose.yaw_rad},
        }

    def estimate_object_pose(self, object_name: str) -> ObjectEstimate:
        return self._object_estimate_from_env(object_name)

    def sample_grasp_pose(self, estimate: ObjectEstimate, preferred_hand: str | None = None) -> GraspPose:
        hand = preferred_hand or ("right" if estimate.xyz[0] >= -4.25 else "left")
        tcp_xyz = estimate.xyz
        pregrasp_xyz = (estimate.xyz[0], estimate.xyz[1], estimate.xyz[2] + 0.18)
        quat = (1.0, 0.0, 0.0, 0.0)
        return GraspPose(
            tcp_xyz=tcp_xyz,
            tcp_quat_wxyz=quat,
            pregrasp_xyz=pregrasp_xyz,
            pregrasp_quat_wxyz=quat,
            hand=hand,
            source="official_unitree_oracle_preset",
            score=1.0,
        )

    def execute_pick(self, grasp: GraspPose, object_name: str) -> PickResult:
        obj = self.env.scene["object"]
        robot = self.env.scene["robot"]
        start = obj.data.root_pos_w[0].detach().cpu().numpy().astype(float)

        left = list(LEFT_ARM_PRESETS[self.config.left_arm_preset_name])
        right = list(RIGHT_ARM_PRESETS[self.config.right_arm_preset_name])
        self.gripper_dds.left = self.config.gripper_open_value
        self.gripper_dds.right = self.config.gripper_open_value
        self._set_arm_positions(left, right)
        self._run_ticks(self.config.arm_steps)

        phase_debug: dict[str, Any] = {}
        if self.config.pick_mode == "preset_relservo":
            current_tcp = tcp_position_w(robot, grasp.hand)
            current_rot = current_wrist_rot(robot, grasp.hand)
            delta = current_tcp - start[:3]
            bias = np.asarray(self.config.relservo_target_bias_xyz, dtype=float)
            target_tcp = start[:3] + float(self.config.relservo_keep_ratio) * delta
            target_tcp = target_tcp + bias
            target_tcp[2] = float(start[2] + bias[2] + self.config.relservo_z_offset_m)
            debug_dir = self._ensure_debug_dir()
            try:
                waypoints, ik_info, urdf_names = solve_arm_phase(
                    robot,
                    target_tcp_w=target_tcp,
                    wrist_rot_w=current_rot,
                    out_dir=debug_dir,
                    label="approach_relservo",
                    steps=self.config.relservo_steps,
                    arm=grasp.hand,
                    backend=self.config.ik_backend,
                )
                phase_records = execute_pinocchio_waypoints(
                    self.env,
                    self.provider,
                    self.gripper_dds,
                    waypoints,
                    urdf_names,
                    stride=self.config.relservo_stride,
                    arm=grasp.hand,
                )
                self._run_ticks(self.config.relservo_settle_steps)
                phase_debug["approach_relservo"] = {
                    "ik": ik_info,
                    "records": phase_records,
                    "current_tcp_w": current_tcp.astype(float).tolist(),
                    "target_tcp_w": target_tcp.astype(float).tolist(),
                    "delta_tcp_object_w": delta.astype(float).tolist(),
                    "settled_tcp_w": tcp_position_w(robot, grasp.hand).astype(float).tolist(),
                }
            except Exception as exc:  # noqa: BLE001
                phase_debug["approach_relservo_error"] = repr(exc)

        seat_delta = np.asarray(self.config.relservo_seat_delta_xyz, dtype=float)
        seat_follow = np.zeros(3, dtype=float)
        if self.config.pick_mode == "preset_relservo" and self.config.relservo_seat_follow_object_gain > 1e-9:
            obj_now = obj.data.root_pos_w[0].detach().cpu().numpy().astype(float)
            geom_now = self._hand_geometry(grasp.hand, obj_now[:3])
            seat_follow = (
                np.asarray(geom_now["tcp_to_object_w"], dtype=float) * float(self.config.relservo_seat_follow_object_gain)
            )
            norm = float(np.linalg.norm(seat_follow))
            max_norm = float(self.config.relservo_seat_follow_object_max_delta_m)
            if norm > max_norm > 0.0:
                seat_follow = seat_follow / norm * max_norm
            phase_debug["pre_close_follow_correction"] = {
                "gain": float(self.config.relservo_seat_follow_object_gain),
                "max_delta_m": max_norm,
                "tcp_to_object_w": geom_now["tcp_to_object_w"],
                "applied_delta_w": seat_follow.astype(float).tolist(),
            }
        seat_total = seat_delta + seat_follow
        if self.config.pick_mode == "preset_relservo" and np.linalg.norm(seat_total) > 1e-9:
            seated_tcp = tcp_position_w(robot, grasp.hand) + seat_total
            debug_dir = self._ensure_debug_dir()
            try:
                waypoints, ik_info, urdf_names = solve_arm_phase(
                    robot,
                    target_tcp_w=seated_tcp,
                    wrist_rot_w=current_wrist_rot(robot, grasp.hand),
                    out_dir=debug_dir,
                    label="seat_relservo",
                    steps=max(8, self.config.relservo_seat_steps),
                    arm=grasp.hand,
                    backend=self.config.ik_backend,
                )
                phase_records = execute_pinocchio_waypoints(
                    self.env,
                    self.provider,
                    self.gripper_dds,
                    waypoints,
                    urdf_names,
                    stride=self.config.relservo_stride,
                    arm=grasp.hand,
                )
                self._run_ticks(max(4, self.config.relservo_settle_steps // 2))
                phase_debug["seat_relservo"] = {
                    "ik": ik_info,
                    "records": phase_records,
                    "seat_delta_xyz": seat_delta.astype(float).tolist(),
                    "seat_follow_delta_xyz": seat_follow.astype(float).tolist(),
                    "seat_total_delta_xyz": seat_total.astype(float).tolist(),
                    "settled_tcp_w": tcp_position_w(robot, grasp.hand).astype(float).tolist(),
                }
            except Exception as exc:  # noqa: BLE001
                phase_debug["seat_relservo_error"] = repr(exc)

        phase_debug["pre_close_geometry"] = self._hand_geometry(grasp.hand, start[:3])

        if grasp.hand == "left":
            self.gripper_dds.left = self.config.gripper_close_value
        else:
            self.gripper_dds.right = self.config.gripper_close_value
        self._run_ticks(self.config.close_steps)
        if self.config.post_close_settle_steps > 0:
            self._run_ticks(self.config.post_close_settle_steps)
        obj_after_close = obj.data.root_pos_w[0].detach().cpu().numpy().astype(float)
        phase_debug["post_close_geometry"] = self._hand_geometry(grasp.hand, obj_after_close[:3])

        post_close_seat_delta = np.asarray(self.config.post_close_seat_delta_xyz, dtype=float)
        if self.config.pick_mode == "preset_relservo" and np.linalg.norm(post_close_seat_delta) > 1e-9:
            seated_tcp = tcp_position_w(robot, grasp.hand) + post_close_seat_delta
            debug_dir = self._ensure_debug_dir()
            try:
                waypoints, ik_info, urdf_names = solve_arm_phase(
                    robot,
                    target_tcp_w=seated_tcp,
                    wrist_rot_w=current_wrist_rot(robot, grasp.hand),
                    out_dir=debug_dir,
                    label="post_close_seat_relservo",
                    steps=max(6, self.config.post_close_seat_steps),
                    arm=grasp.hand,
                    backend=self.config.ik_backend,
                )
                phase_records = execute_pinocchio_waypoints(
                    self.env,
                    self.provider,
                    self.gripper_dds,
                    waypoints,
                    urdf_names,
                    stride=self.config.relservo_stride,
                    arm=grasp.hand,
                )
                self._run_ticks(max(4, self.config.relservo_settle_steps // 2))
                obj_after_post_close_seat = obj.data.root_pos_w[0].detach().cpu().numpy().astype(float)
                phase_debug["post_close_seat_relservo"] = {
                    "ik": ik_info,
                    "records": phase_records,
                    "seat_delta_xyz": post_close_seat_delta.astype(float).tolist(),
                    "settled_tcp_w": tcp_position_w(robot, grasp.hand).astype(float).tolist(),
                    "post_seat_geometry": self._hand_geometry(grasp.hand, obj_after_post_close_seat[:3]),
                }
            except Exception as exc:  # noqa: BLE001
                phase_debug["post_close_seat_relservo_error"] = repr(exc)

        if self.config.pick_mode == "preset_relservo":
            closed_tcp = tcp_position_w(robot, grasp.hand)
            lift_tcp = closed_tcp.copy()
            lift_tcp[2] += float(min(self.config.lift_phase_delta_m, self.config.lift_phase_max_delta_m))
            debug_dir = self._ensure_debug_dir()
            try:
                waypoints, ik_info, urdf_names = solve_arm_phase(
                    robot,
                    target_tcp_w=lift_tcp,
                    wrist_rot_w=current_wrist_rot(robot, grasp.hand),
                    out_dir=debug_dir,
                    label="lift_relservo",
                    steps=max(20, self.config.relservo_steps // 2),
                    arm=grasp.hand,
                    backend=self.config.ik_backend,
                )
                phase_records = execute_pinocchio_waypoints(
                    self.env,
                    self.provider,
                    self.gripper_dds,
                    waypoints,
                    urdf_names,
                    stride=self.config.relservo_stride,
                    arm=grasp.hand,
                )
                self._run_ticks(max(6, self.config.relservo_settle_steps // 2))
                phase_debug["lift_relservo"] = {
                    "ik": ik_info,
                    "records": phase_records,
                    "closed_tcp_w": closed_tcp.astype(float).tolist(),
                    "lift_tcp_w": lift_tcp.astype(float).tolist(),
                    "settled_tcp_w": tcp_position_w(robot, grasp.hand).astype(float).tolist(),
                }
                self._run_ticks(max(8, self.config.lift_steps // 3))
            except Exception as exc:  # noqa: BLE001
                phase_debug["lift_relservo_error"] = repr(exc)
                self._fallback_lift(left, right, grasp.hand)
        else:
            self._fallback_lift(left, right, grasp.hand)
        obj_after_lift = obj.data.root_pos_w[0].detach().cpu().numpy().astype(float)
        phase_debug["post_lift_geometry"] = self._hand_geometry(grasp.hand, obj_after_lift[:3])

        final = obj.data.root_pos_w[0].detach().cpu().numpy().astype(float)
        z_gain = float(final[2] - start[2])
        xy_delta = float(np.linalg.norm(final[:2] - start[:2]))
        retained = z_gain >= self.config.lift_success_z_gain_m
        return PickResult(
            success=retained,
            hand=grasp.hand,
            contact_observed=bool(xy_delta > 0.005 or z_gain > 0.005),
            retained_after_lift=retained,
            debug={
                "left_arm_preset_name": self.config.left_arm_preset_name,
                "right_arm_preset_name": self.config.right_arm_preset_name,
                "object_start": start.tolist(),
                "object_final": final.tolist(),
                "object_z_gain_m": z_gain,
                "object_xy_delta_m": xy_delta,
                "pick_mode": self.config.pick_mode,
                "phase_debug": phase_debug,
            },
        )

    def execute_place(self, target_region: TargetRegion, hand: str) -> PlaceResult:
        if hand == "left":
            self.gripper_dds.left = self.config.gripper_open_value
        else:
            self.gripper_dds.right = self.config.gripper_open_value
        self._run_ticks(self.config.release_steps)

        obj = self.env.scene["object"]
        final = obj.data.root_pos_w[0].detach().cpu().numpy().astype(float)
        target = np.asarray(target_region.xyz, dtype=float)
        xyz_err = float(np.max(np.abs(final[:3] - target[:3])))
        in_target = xyz_err <= target_region.tolerance_xyz_m
        return PlaceResult(
            success=in_target,
            released=True,
            in_target_region=in_target,
            debug={
                "object_final": final.tolist(),
                "target_xyz": list(target_region.xyz),
                "xyz_error_m": xyz_err,
            },
        )

    def verify_result(self, task: TabletopTaskSpec) -> VerificationResult:
        obj = self.env.scene["object"]
        final = obj.data.root_pos_w[0].detach().cpu().numpy().astype(float)
        target = np.asarray(task.target_region.xyz, dtype=float)
        xyz_err = float(np.max(np.abs(final[:3] - target[:3])))
        success = xyz_err <= task.target_region.tolerance_xyz_m
        score_estimate = {
            "object_name": task.object_name,
            "target_region": task.target_region.name,
            "xyz_error_m": xyz_err,
            "within_xyz_tolerance": success,
        }
        return VerificationResult(
            success=success,
            score_estimate=score_estimate,
            debug={"object_final": final.tolist(), "target_xyz": list(task.target_region.xyz)},
        )

    def set_prompt(self, prompt: str) -> None:
        self._last_prompt = prompt

    def _object_estimate_from_env(self, object_name: str) -> ObjectEstimate:
        obj = self.env.scene["object"]
        pos = obj.data.root_pos_w[0].detach().cpu().numpy().astype(float)
        quat_xyzw = obj.data.root_quat_w[0].detach().cpu().numpy().astype(float)
        quat_wxyz = (float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2]))
        return ObjectEstimate(
            name=object_name,
            xyz=(float(pos[0]), float(pos[1]), float(pos[2])),
            quat_wxyz=quat_wxyz,
            source="official_env_gt_bootstrap",
            confidence=1.0,
        )

    def _capture_triplet(self) -> dict[str, np.ndarray]:
        frame: dict[str, np.ndarray] = {}
        for out_name, sensor_name in CAMERA_MAP.items():
            sensor = self.env.scene[sensor_name]
            sensor.update(0.02, force_recompute=True)
            rgb = sensor.data.output.get("rgb")
            if rgb is None:
                raise RuntimeError(f"missing_rgb:{sensor_name}")
            arr = rgb[0].detach().cpu().numpy()
            if arr.shape[-1] == 4:
                arr = arr[..., :3]
            frame[out_name] = np.clip(arr, 0, 255).astype(np.uint8)
        return frame

    def _state_from_robot(self) -> tuple[np.ndarray, np.ndarray]:
        robot = self.env.scene["robot"]
        joint_names = list(robot.data.joint_names)
        joint_index = {name: i for i, name in enumerate(joint_names)}
        q = robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float32)
        arm_state = np.zeros(len(OBS_ARM_STATE_JOINTS), dtype=np.float32)
        for i, name in enumerate(OBS_ARM_STATE_JOINTS):
            arm_state[i] = float(q[joint_index[name]])
        gripper_state = np.zeros(len(GRIPPER_JOINTS), dtype=np.float32)
        for i, name in enumerate(GRIPPER_JOINTS):
            gripper_state[i] = float(q[joint_index[name]])
        return arm_state, gripper_state

    def _set_arm_positions(self, left: list[float], right: list[float]) -> None:
        for i, value in enumerate(left + right):
            self.motor_positions[15 + i] = float(value)
        self.provider.robot_dds.positions = list(self.motor_positions)

    def _resolve_nav_command(
        self,
        nav_profile_name: str,
        root_xy: np.ndarray,
        object_xy: np.ndarray,
        desired_offset: np.ndarray,
    ) -> list[float]:
        if nav_profile_name == "dynamic_oracle":
            target_root = object_xy + desired_offset
            err = target_root - root_xy
            return [
                float(np.clip(0.9 * err[0], -0.45, 0.45)),
                float(np.clip(0.9 * err[1], -0.25, 0.25)),
                0.0,
                0.8,
            ]
        profile = NAV_COMMAND_PROFILES[nav_profile_name]
        if profile is None:
            raise RuntimeError(f"unexpected_none_profile:{nav_profile_name}")
        return [float(v) for v in profile]

    def _run_ticks(self, steps: int) -> None:
        for _ in range(steps):
            self.provider.get_action(self.env)

    def _hand_geometry(self, hand: str, object_pos_w: np.ndarray) -> dict[str, Any]:
        robot = self.env.scene["robot"]
        names = robot.data.body_names
        poses = robot.data.body_link_state_w[0, :, :7].detach().cpu().numpy().astype(float)
        tip1 = poses[names.index(f"{hand}_hand_Link1_3"), :3]
        tip2 = poses[names.index(f"{hand}_hand_Link2_3"), :3]
        wrist_idx = names.index(f"{hand}_wrist_yaw_link")
        wrist = poses[wrist_idx, :3]
        wrist_rot = current_wrist_rot(robot, hand)
        tcp = 0.5 * (tip1 + tip2)
        jaw_vec_w = tip2 - tip1
        jaw_span = float(np.linalg.norm(jaw_vec_w))
        obj = np.asarray(object_pos_w, dtype=float)
        tcp_to_obj = obj - tcp
        obj_in_wrist = wrist_rot.T @ (obj - wrist)
        tcp_in_wrist = wrist_rot.T @ (tcp - wrist)
        return {
            "tip1_w": tip1.tolist(),
            "tip2_w": tip2.tolist(),
            "tcp_w": tcp.tolist(),
            "wrist_w": wrist.tolist(),
            "jaw_span_m": jaw_span,
            "jaw_vec_w": jaw_vec_w.tolist(),
            "object_w": obj.tolist(),
            "tcp_to_object_w": tcp_to_obj.tolist(),
            "tcp_to_object_dist_m": float(np.linalg.norm(tcp_to_obj)),
            "tip1_to_object_dist_m": float(np.linalg.norm(obj - tip1)),
            "tip2_to_object_dist_m": float(np.linalg.norm(obj - tip2)),
            "object_in_wrist_frame": obj_in_wrist.tolist(),
            "tcp_in_wrist_frame": tcp_in_wrist.tolist(),
        }

    def _fallback_lift(self, left: list[float], right: list[float], hand: str) -> None:
        lift_left = list(left)
        lift_right = list(right)
        if hand == "left":
            lift_left[0] -= 0.18
            lift_left[3] -= 0.20
            lift_left[5] -= 0.18
        else:
            lift_right[0] += 0.18
            lift_right[3] += 0.20
            lift_right[5] += 0.18
        self._set_arm_positions(lift_left, lift_right)
        self._run_ticks(self.config.lift_steps)

    def _ensure_debug_dir(self) -> Path:
        out_dir = Path("<PROJECT_ROOT>/logs/unitree_control_api_debug")
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir
