import json
import os
import torch
from typing import Any

class AlgSolution:

    ACTION_SCALE = 0.5
    EE_BODY_NAME_CANDIDATES = ("gripper_base", "piper_gripper_base")
    ARM_JOINT_NAME_CANDIDATES = (
        ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
        ["arm_joint1", "arm_joint2", "arm_joint3", "arm_joint4", "arm_joint5", "arm_joint6"],
    )

    def __init__(self):
        policy_path = os.path.dirname(os.path.abspath(__file__)) + '/policy.pt'
        self.device = 'cuda'

        self.policy = torch.jit.load(policy_path, map_location=self.device)
        self.policy.eval()

        self.leg_action_dim = 12
        self.wheel_action_dim = 4
        self.arm_action_dim = 8

        self.leg_joint_indices = list(range(12))
        self.wheel_joint_indices = list(range(12, 16))

        self.train_to_env_action_scale = torch.tensor(
            [
                0.25, 0.5, 0.5,
                0.25, 0.5, 0.5,
                0.25, 0.5, 0.5,
                0.25, 0.5, 0.5,
            ],
            device=self.device,
            dtype=torch.float32,
        ).view(1, -1)

        self.env_to_train_action_scale = torch.tensor(
            [
                4.0, 2.0, 2.0,
                4.0, 2.0, 2.0,
                4.0, 2.0, 2.0,
                4.0, 2.0, 2.0,
            ],
            device=self.device,
            dtype=torch.float32,
        ).view(1, -1)

        default_schedule = (
            "0:1.5:-0.4:0,"
            "2:1.2:0.0:0,"
            "5:1.2:-0.4:0,"
            "7.4:1.2:0.25:0,"
            "8.8:1.2:0.0:0,"
            "10.5:1.1:-0.2:0,"
            "12.5:1.1:0.2:0"
        )
        self.command_schedule = self._parse_command_schedule(
            os.environ.get("ATEC_SCORE_SCHEDULE", default_schedule)
        )
        self.current_score = 0.0

        self.arm_default_action = torch.zeros(
            (1, self.arm_action_dim),
            device=self.device,
            dtype=torch.float32,
        )
        self.wheel_action = float(os.environ.get("ATEC_WHEEL_ACTION", "0.0"))
        self.smooth_alpha = float(os.environ.get("ATEC_SMOOTH_ALPHA", "1.0"))
        self.prev_action_env = None
        self.step_count = 0
        self.best_score = float("-inf")
        self.best_score_step = 0
        self.giveup_min_score = float(os.environ.get("ATEC_GIVEUP_MIN_SCORE", "999.0"))
        self.giveup_stall_steps = int(os.environ.get("ATEC_GIVEUP_STALL_STEPS", "2500"))
        self.giveup_target_score = float(os.environ.get("ATEC_GIVEUP_TARGET_SCORE", "999.0"))
        self.controller_mode = os.environ.get("ATEC_CONTROLLER_MODE", "schedule").strip().lower()
        self.manual_cmd_file = os.environ.get("ATEC_MANUAL_CMD_FILE", "/tmp/atec_l0_manual_cmd.json")
        self.manual_cmd_max = self._parse_float_list(
            os.environ.get("ATEC_MANUAL_CMD_MAX", "1.8,0.8,1.0"),
            3,
        ) or [1.8, 0.8, 1.0]
        self.manual_cmd_default = self._parse_float_list(
            os.environ.get("ATEC_MANUAL_CMD_DEFAULT", "0.0,0.0,0.0"),
            3,
        ) or [0.0, 0.0, 0.0]
        self.manual_zero_action_stop = os.environ.get("ATEC_MANUAL_ZERO_ACTION_STOP", "1") != "0"
        self.manual_stop_eps = float(os.environ.get("ATEC_MANUAL_STOP_EPS", "1e-4"))
        self.manual_rough_assist = os.environ.get("ATEC_MANUAL_ROUGH_ASSIST", "1") != "0"
        self.manual_rough_score_min = float(os.environ.get("ATEC_MANUAL_ROUGH_SCORE_MIN", "1.95"))
        self.manual_rough_score_max = float(os.environ.get("ATEC_MANUAL_ROUGH_SCORE_MAX", "5.20"))
        self.manual_rough_vx_min = float(os.environ.get("ATEC_MANUAL_ROUGH_VX_MIN", "0.85"))
        self.manual_rough_vx_max = float(os.environ.get("ATEC_MANUAL_ROUGH_VX_MAX", "1.08"))
        self.manual_rough_vy_clip = float(os.environ.get("ATEC_MANUAL_ROUGH_VY_CLIP", "0.12"))
        self.manual_rough_vy_bias = float(os.environ.get("ATEC_MANUAL_ROUGH_VY_BIAS", "0.0"))
        self.manual_rough_yaw_clip = float(os.environ.get("ATEC_MANUAL_ROUGH_YAW_CLIP", "0.12"))
        self.rough_option_moe = os.environ.get("ATEC_ROUGH_OPTION_MOE", "1") != "0"
        self.rough_stall_steps = int(os.environ.get("ATEC_ROUGH_STALL_STEPS", "120"))
        self.rough_progress_eps = float(os.environ.get("ATEC_ROUGH_PROGRESS_EPS", "0.035"))
        self.rough_escalate_steps = int(os.environ.get("ATEC_ROUGH_ESCALATE_STEPS", "260"))
        self.rough_escape_period = int(os.environ.get("ATEC_ROUGH_ESCAPE_PERIOD", "45"))
        self.rough_escape_vx = float(os.environ.get("ATEC_ROUGH_ESCAPE_VX", "0.68"))
        self.rough_escape_vy = float(os.environ.get("ATEC_ROUGH_ESCAPE_VY", "0.08"))
        self.rough_escape_yaw = float(os.environ.get("ATEC_ROUGH_ESCAPE_YAW", "0.34"))
        self.rough_escape_yaw_strong = float(os.environ.get("ATEC_ROUGH_ESCAPE_YAW_STRONG", "0.46"))
        self.rough_score_anchor = 0.0
        self.rough_step_anchor = 0
        self.last_option_id = 0
        self.last_velocity_command = None


    def get_action_spec(self) -> dict[str, dict[str, Any]] | None:
        return None

    def _parse_float_list(self, raw: str, expected_len: int) -> list[float] | None:
        try:
            vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
        except Exception:
            return None
        if len(vals) != expected_len:
            return None
        return vals

    @staticmethod
    def _clip(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))

    def _parse_command_schedule(self, raw: str) -> list[tuple[float, torch.Tensor]]:
        stages = []
        for item in raw.split(","):
            item = item.strip()
            if not item:
                continue
            parts = item.split(":")
            if len(parts) != 4:
                continue
            score, vx, vy, wz = (float(x) for x in parts)
            cmd = torch.tensor([vx, vy, wz], device=self.device, dtype=torch.float32).view(1, 3)
            stages.append((score, cmd))
        if not stages:
            stages.append((0.0, torch.tensor([1.2, -0.2, 0.0], device=self.device).view(1, 3)))
        stages.sort(key=lambda x: x[0])
        return stages

    def _apply_rough_option_moe(self, cmd: list[float]) -> list[float]:
        if (
            not self.rough_option_moe
            or not (self.manual_rough_score_min <= self.current_score < self.manual_rough_score_max)
            or cmd[0] <= self.manual_stop_eps
        ):
            self.last_option_id = 0
            self.rough_score_anchor = self.current_score
            self.rough_step_anchor = self.step_count
            return cmd

        if self.current_score >= self.rough_score_anchor + self.rough_progress_eps:
            self.rough_score_anchor = self.current_score
            self.rough_step_anchor = self.step_count

        stall_steps = self.step_count - self.rough_step_anchor
        if stall_steps >= self.rough_stall_steps:
            phase = 1.0 if ((stall_steps // max(1, self.rough_escape_period)) % 2 == 0) else -1.0
            yaw_mag = self.rough_escape_yaw_strong if stall_steps >= self.rough_escalate_steps else self.rough_escape_yaw
            self.last_option_id = 2
            return [
                self._clip(self.rough_escape_vx, 0.40, self.manual_cmd_max[0]),
                self._clip(phase * self.rough_escape_vy, -self.manual_cmd_max[1], self.manual_cmd_max[1]),
                self._clip(phase * yaw_mag, -self.manual_cmd_max[2], self.manual_cmd_max[2]),
            ]

        self.last_option_id = 1
        return [
            self._clip(cmd[0], self.manual_rough_vx_min, self.manual_rough_vx_max),
            self._clip(cmd[1] + self.manual_rough_vy_bias, -self.manual_rough_vy_clip, self.manual_rough_vy_clip),
            self._clip(cmd[2], -self.manual_rough_yaw_clip, self.manual_rough_yaw_clip),
        ]

    def _resolve_joint_ids(self, candidates: tuple[list[str], ...]) -> list[int]:
        last_error = None
        for names in candidates:
            try:
                ids, found_names = self.robot.find_joints(names)
            except ValueError as err:
                last_error = err
                continue
            if len(ids) == len(names):
                if candidates is self.ARM_JOINT_NAME_CANDIDATES:
                    self.arm_joint_names = list(found_names)
                return list(ids)
        raise ValueError(
            f"Cannot resolve required joints from candidates: {candidates}. Last error: {last_error}"
        )

    def _resolve_ee_body_name(self) -> str:
        last_error = None
        for name in self.EE_BODY_NAME_CANDIDATES:
            try:
                body_ids, _ = self.robot.find_bodies(name)
            except ValueError as err:
                last_error = err
                continue
            if len(body_ids) == 1:
                return name
        raise ValueError(
            f"Cannot resolve EE body from candidates: {self.EE_BODY_NAME_CANDIDATES}. Last error: {last_error}"
        )

    def _ensure_cartesian_targets(self):
        self.cartesian_ctrl.reset()

    def _compute_arm_overlay_action(self) -> torch.Tensor:
        self._ensure_cartesian_targets()

        arm_jpos_des = self.cartesian_ctrl.compute_base(
            self.ee_pos_target_b,
            self.ee_quat_target_b,
        )

        full_target = self.robot.data.joint_pos.clone()
        full_target[:, self.arm_ids] = arm_jpos_des
        full_target[:, self.gripper_ids] = self.gripper_open_pos.repeat(full_target.shape[0], 1)

        return (full_target - self.default_joint_pos) / self.ACTION_SCALE

    def _get_velocity_commands(self, proprio: torch.Tensor) -> torch.Tensor:
        """Return fixed velocity commands for policy input."""
        num_envs = proprio.shape[0]

        if self.controller_mode == "manual_file":
            cmd = list(self.manual_cmd_default)
            try:
                with open(self.manual_cmd_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    cmd = [
                        float(data.get("vx", cmd[0])),
                        float(data.get("vy", cmd[1])),
                        float(data.get("yaw", data.get("wz", cmd[2]))),
                    ]
                elif isinstance(data, list) and len(data) >= 3:
                    cmd = [float(data[0]), float(data[1]), float(data[2])]
            except Exception:
                pass
            clipped = [
                self._clip(cmd[0], -self.manual_cmd_max[0], self.manual_cmd_max[0]),
                self._clip(cmd[1], -self.manual_cmd_max[1], self.manual_cmd_max[1]),
                self._clip(cmd[2], -self.manual_cmd_max[2], self.manual_cmd_max[2]),
            ]
            if self.manual_rough_assist:
                clipped = self._apply_rough_option_moe(clipped)
            out = torch.tensor(clipped, device=self.device, dtype=proprio.dtype).view(1, 3)
            if num_envs > 1:
                out = out.repeat(num_envs, 1)
            self.last_velocity_command = out.clone()
            return out

        base_cmd = self.command_schedule[0][1]
        for score, cmd_i in self.command_schedule:
            if self.current_score >= score:
                base_cmd = cmd_i
            else:
                break
        cmd = base_cmd.to(dtype=proprio.dtype, device=self.device)
        if num_envs > 1:
            cmd = cmd.repeat(num_envs, 1)
        if self.manual_rough_assist:
            cmd_list = [float(cmd[0, 0]), float(cmd[0, 1]), float(cmd[0, 2])]
            cmd_list = self._apply_rough_option_moe(cmd_list)
            cmd = torch.tensor(cmd_list, device=self.device, dtype=proprio.dtype).view(1, 3)
            if num_envs > 1:
                cmd = cmd.repeat(num_envs, 1)
        self.last_velocity_command = cmd.clone()
        return cmd

    def _extract_policy_obs(self, obs, action_dim) -> torch.Tensor:
        proprio = obs["proprio"].to(self.device)

        expected_dim = 3 + 3 + 3 + 3 + action_dim + action_dim + action_dim

        idx = 0
        _base_lin_vel = proprio[:, idx:idx + 3]
        idx += 3

        base_ang_vel = proprio[:, idx:idx + 3]
        idx += 3

        _velocity_commands_env = proprio[:, idx:idx + 3]
        idx += 3

        projected_gravity = proprio[:, idx:idx + 3]
        idx += 3

        joint_pos_all = proprio[:, idx:idx + action_dim]
        idx += action_dim

        joint_vel_all = proprio[:, idx:idx + action_dim]
        idx += action_dim

        actions_all = proprio[:, idx:idx + action_dim]

        joint_pos_leg = joint_pos_all[:, self.leg_joint_indices]
        joint_vel_leg = joint_vel_all[:, self.leg_joint_indices]
        actions_env_leg = actions_all[:, self.leg_joint_indices]

        actions_train_leg = actions_env_leg * self.env_to_train_action_scale.to(dtype=proprio.dtype)
        velocity_commands = self._get_velocity_commands(proprio)

        policy_obs = torch.cat(
            [
                base_ang_vel * 0.25,
                projected_gravity,
                velocity_commands,
                joint_pos_leg,
                joint_vel_leg * 0.05,
                actions_train_leg,
            ],
            dim=-1,
        )

        return policy_obs

    def _map_policy_action_to_env_action(self, action_train: torch.Tensor, action_dim: int) -> torch.Tensor:
        """Map training-time 12D leg action to current env full-body action."""
        if action_train.shape[-1] != self.leg_action_dim:
            raise ValueError(
                f"Policy output dim mismatch: got {action_train.shape[-1]}, expected {self.leg_action_dim}"
            )

        num_envs = action_train.shape[0]
        leg_action_env = action_train * self.train_to_env_action_scale

        action_env = torch.zeros(
            (num_envs, action_dim),
            device=self.device,
            dtype=torch.float32,
        )

        action_env[:, self.leg_joint_indices] = leg_action_env
        if action_dim >= self.leg_action_dim + self.wheel_action_dim + self.arm_action_dim:
            action_env[:, self.wheel_joint_indices] = self.wheel_action
            arm_start = self.leg_action_dim + self.wheel_action_dim
        else:
            arm_start = self.leg_action_dim
        action_env[:, arm_start:arm_start + self.arm_action_dim] = self.arm_default_action.repeat(num_envs, 1)

        return action_env
    
    def predicts(self, obs, current_score):
        """Run policy inference and return current-env full-body action."""
        self.current_score = float(current_score)
        self.step_count += 1
        if current_score >= self.giveup_target_score:
            return {'action': [], 'giveup': True}
        if current_score > self.best_score + 1e-4:
            self.best_score = current_score
            self.best_score_step = self.step_count
        elif (
            current_score >= self.giveup_min_score
            and self.step_count - self.best_score_step > self.giveup_stall_steps
        ):
            return {'action': [], 'giveup': True}

        proprio = obs["proprio"].to(self.device)
        action_dim = (int(proprio.shape[-1]) - 12) // 3
        policy_obs = self._extract_policy_obs(obs, action_dim)
        if (
            self.controller_mode == "manual_file"
            and self.manual_zero_action_stop
            and self.last_velocity_command is not None
            and torch.max(torch.abs(self.last_velocity_command)) <= self.manual_stop_eps
        ):
            action_env = torch.zeros((proprio.shape[0], action_dim), device=self.device, dtype=torch.float32)
            self.prev_action_env = action_env.clone()
            return {'action': action_env.cpu().numpy().tolist(), 'giveup': False}

        with torch.inference_mode():
            action_train = self.policy(policy_obs)

        if not isinstance(action_train, torch.Tensor):
            action_train = torch.as_tensor(
                action_train, device=self.device, dtype=torch.float32
            )

        action_train = action_train.to(device=self.device, dtype=torch.float32)

        if action_train.ndim == 1:
            action_train = action_train.unsqueeze(0)

        action_env = self._map_policy_action_to_env_action(action_train, action_dim)
        if self.smooth_alpha < 1.0:
            if self.prev_action_env is None or self.prev_action_env.shape != action_env.shape:
                self.prev_action_env = action_env.clone()
            else:
                action_env = self.smooth_alpha * action_env + (1.0 - self.smooth_alpha) * self.prev_action_env
                self.prev_action_env = action_env.clone()
        action_env = action_env.cpu().numpy().tolist()
        return {'action': action_env, 'giveup': False}
