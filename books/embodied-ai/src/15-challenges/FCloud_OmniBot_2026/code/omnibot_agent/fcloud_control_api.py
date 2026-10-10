"""Lightweight FCloud/Unitree control API modeled after CaP-X.

This module does not bind to a specific simulator runtime yet. It defines the
interfaces and stage runner we want the official IsaacLab / Unitree scripts to
converge on, so we can keep the donor structure stable while swapping the
backend implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any
from typing import Protocol
from typing import Sequence

import numpy as np

from omnibot_agent.episode_logger import EpisodeLogger
from omnibot_agent.pi05_adapter import EBenchActionChunk
from omnibot_agent.pi05_adapter import OmniBotObservation


@dataclass(frozen=True)
class BasePose:
    xyz: tuple[float, float, float]
    yaw_rad: float


@dataclass(frozen=True)
class ObjectEstimate:
    name: str
    xyz: tuple[float, float, float]
    quat_wxyz: tuple[float, float, float, float] | None = None
    source: str = "unknown"
    confidence: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TargetRegion:
    name: str
    xyz: tuple[float, float, float]
    quat_wxyz: tuple[float, float, float, float] | None = None
    tolerance_xyz_m: float = 0.05
    tolerance_rpy_deg: float = 10.0


@dataclass(frozen=True)
class GraspPose:
    tcp_xyz: tuple[float, float, float]
    tcp_quat_wxyz: tuple[float, float, float, float]
    pregrasp_xyz: tuple[float, float, float]
    pregrasp_quat_wxyz: tuple[float, float, float, float]
    hand: str
    source: str
    score: float | None = None


@dataclass(frozen=True)
class PickResult:
    success: bool
    hand: str
    contact_observed: bool
    retained_after_lift: bool
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlaceResult:
    success: bool
    released: bool
    in_target_region: bool
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VerificationResult:
    success: bool
    score_estimate: dict[str, Any]
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TabletopTaskSpec:
    object_name: str
    prompt: str
    target_region: TargetRegion
    preferred_hand: str | None = None


class FCloudControlApi(Protocol):
    """Backend-facing API that mirrors CaP-X style stage decomposition."""

    def get_robot_observation(self) -> OmniBotObservation:
        """Return robot-mounted images and compact robot state."""

    def get_current_base_pose(self) -> BasePose:
        """Return the current robot base pose in the world frame."""

    def compute_table_approach_pose(self, task: TabletopTaskSpec) -> BasePose:
        """Return a base pose that should expose a usable tabletop view."""

    def approach_table(self, base_pose: BasePose) -> dict[str, Any]:
        """Execute the navigation / wholebody approach stage."""

    def estimate_object_pose(self, object_name: str) -> ObjectEstimate:
        """Estimate the target object's pose from robot-mounted sensing."""

    def sample_grasp_pose(self, estimate: ObjectEstimate, preferred_hand: str | None = None) -> GraspPose:
        """Return a grasp candidate for the current object estimate."""

    def execute_pick(self, grasp: GraspPose, object_name: str) -> PickResult:
        """Run the physical pick stage."""

    def execute_place(self, target_region: TargetRegion, hand: str) -> PlaceResult:
        """Run the physical place stage."""

    def verify_result(self, task: TabletopTaskSpec) -> VerificationResult:
        """Verify final result from robot-visible evidence and/or local scoring."""


def null_action_chunk() -> EBenchActionChunk:
    return EBenchActionChunk(
        joint=np.zeros((1, 12), dtype=np.float32),
        gripper=np.zeros((1, 4), dtype=np.float32),
        base=np.zeros((1, 3), dtype=np.float32),
    )


class TabletopTaskRunner:
    """Stage runner that enforces the donor-style execution order."""

    def __init__(self, api: FCloudControlApi, logger: EpisodeLogger | None = None) -> None:
        self.api = api
        self.logger = logger

    def run_task(self, task: TabletopTaskSpec) -> dict[str, Any]:
        step_index = 0
        run: dict[str, Any] = {
            "object_name": task.object_name,
            "target_region": task.target_region.name,
            "prompt": task.prompt,
            "status": "started",
        }

        approach_pose = self.api.compute_table_approach_pose(task)
        run["approach_pose"] = {
            "xyz": list(approach_pose.xyz),
            "yaw_rad": approach_pose.yaw_rad,
        }
        nav_info = self.api.approach_table(approach_pose)
        run["approach_info"] = nav_info
        step_index = self._log_phase(step_index, "approach_table", task.prompt, {"approach_info": nav_info})

        estimate = self.api.estimate_object_pose(task.object_name)
        run["object_estimate"] = {
            "name": estimate.name,
            "xyz": list(estimate.xyz),
            "quat_wxyz": list(estimate.quat_wxyz) if estimate.quat_wxyz is not None else None,
            "source": estimate.source,
            "confidence": estimate.confidence,
            "extra": estimate.extra,
        }
        step_index = self._log_phase(step_index, "estimate_object_pose", task.prompt, run["object_estimate"])

        grasp = self.api.sample_grasp_pose(estimate, preferred_hand=task.preferred_hand)
        run["grasp_pose"] = {
            "tcp_xyz": list(grasp.tcp_xyz),
            "tcp_quat_wxyz": list(grasp.tcp_quat_wxyz),
            "pregrasp_xyz": list(grasp.pregrasp_xyz),
            "pregrasp_quat_wxyz": list(grasp.pregrasp_quat_wxyz),
            "hand": grasp.hand,
            "source": grasp.source,
            "score": grasp.score,
        }
        step_index = self._log_phase(step_index, "sample_grasp_pose", task.prompt, run["grasp_pose"])

        pick = self.api.execute_pick(grasp, task.object_name)
        run["pick_result"] = {
            "success": pick.success,
            "hand": pick.hand,
            "contact_observed": pick.contact_observed,
            "retained_after_lift": pick.retained_after_lift,
            "debug": pick.debug,
        }
        step_index = self._log_phase(step_index, "execute_pick", task.prompt, run["pick_result"])

        if not pick.success:
            run["status"] = "pick_failed"
            return run

        place = self.api.execute_place(task.target_region, hand=pick.hand)
        run["place_result"] = {
            "success": place.success,
            "released": place.released,
            "in_target_region": place.in_target_region,
            "debug": place.debug,
        }
        step_index = self._log_phase(step_index, "execute_place", task.prompt, run["place_result"])

        verify = self.api.verify_result(task)
        run["verification"] = {
            "success": verify.success,
            "score_estimate": verify.score_estimate,
            "debug": verify.debug,
        }
        run["status"] = "success" if verify.success else "verify_failed"
        self._log_phase(step_index, "verify_result", task.prompt, run["verification"])
        return run

    def _log_phase(
        self,
        step_index: int,
        phase: str,
        prompt: str,
        decision: dict[str, Any],
    ) -> int:
        if self.logger is None:
            return step_index + 1
        observation = self.api.get_robot_observation()
        log_obs = OmniBotObservation(
            head_image=np.asarray(observation.head_image, dtype=np.uint8),
            left_hand_image=np.asarray(observation.left_hand_image, dtype=np.uint8),
            right_hand_image=np.asarray(observation.right_hand_image, dtype=np.uint8),
            joint_state=np.asarray(observation.joint_state, dtype=np.float32),
            gripper_state=np.asarray(observation.gripper_state, dtype=np.float32),
            prompt=prompt,
        )
        self.logger.log_step(
            step_index=step_index,
            observation=log_obs,
            raw_actions=null_action_chunk(),
            safe_actions=null_action_chunk(),
            decision={"phase": phase, **decision},
            score_estimate=decision.get("score_estimate"),
            save_policy_npz=False,
        )
        return step_index + 1


def build_tabletop_tasks_from_plan(
    target_plan: dict[str, Any],
    *,
    prompt_template: str = "Pick up {object_name} and place it into {target_role}.",
) -> list[TabletopTaskSpec]:
    tasks: list[TabletopTaskSpec] = []
    for target in target_plan.get("targets", []):
        object_name = str(target["object_name"])
        target_role = str(target["target_role"])
        target_xyz = tuple(float(v) for v in target["target_xyz"])
        target_rpy = target.get("target_rpy_deg", [0.0, 0.0, 0.0])
        preferred_hand = target.get("hand")
        tasks.append(
            TabletopTaskSpec(
                object_name=object_name,
                prompt=prompt_template.format(object_name=object_name, target_role=target_role),
                preferred_hand=str(preferred_hand) if preferred_hand is not None else None,
                target_region=TargetRegion(
                    name=target_role,
                    xyz=target_xyz,
                    quat_wxyz=None,
                    tolerance_xyz_m=float(target.get("thresholds", {}).get("xyz_m", 0.05)),
                    tolerance_rpy_deg=float(target.get("thresholds", {}).get("rpy_deg", 10.0)),
                ),
            )
        )
    return tasks
