"""Heuristic tabletop expert plans for OmniBot data collection.

The plan is intentionally geometric: it does not claim that a trajectory has
been executed.  It turns the public target plan into pick/lift/place waypoints
that an Isaac IK or motion-generation controller can consume.
"""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from typing import Any


LEFT_PALM_PATH = "/World/g1/left_wrist_yaw_link/left_hand_palm_link"
RIGHT_PALM_PATH = "/World/g1/right_wrist_yaw_link/right_hand_palm_link"


@dataclass(frozen=True)
class ExpertWaypoint:
    phase: str
    xyz: tuple[float, float, float]
    gripper: str
    hold_steps: int


@dataclass(frozen=True)
class ExpertPickPlace:
    object_name: str
    object_path: str
    target_role: str
    hand: str
    palm_path: str
    base_stance_xyz: tuple[float, float, float]
    gripper_open_m: float
    gripper_closed_m: float
    waypoints: tuple[ExpertWaypoint, ...]
    caution: str | None = None


def _xyz(values: list[float] | tuple[float, float, float]) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError(f"Expected xyz length 3, got {values!r}")
    return (float(values[0]), float(values[1]), float(values[2]))


def _choose_hand(initial_xyz: tuple[float, float, float], target_xyz: tuple[float, float, float]) -> str:
    # G1 starts north-west of the island in this scene. Keep most cross-table
    # transfers on the nearer arm; use right hand for far/sink-side objects.
    mean_x = 0.5 * (initial_xyz[0] + target_xyz[0])
    return "right" if mean_x >= 5.45 else "left"


def _gripper_opening(object_name: str) -> tuple[float, float]:
    lower = object_name.lower()
    if "banana" in lower:
        return (0.075, 0.035)
    if "mug" in lower:
        return (0.085, 0.045)
    if "book" in lower or "album" in lower:
        return (0.09, 0.05)
    if "can" in lower:
        return (0.065, 0.028)
    return (0.075, 0.03)


def _base_stance(initial_xyz: tuple[float, float, float], target_xyz: tuple[float, float, float]) -> tuple[float, float, float]:
    # Face the island from the robot side. The z is the current G1 base height
    # observed in the official scene inventory.
    x = max(4.55, min(6.05, 0.5 * (initial_xyz[0] + target_xyz[0])))
    y = max(4.70, min(5.10, max(initial_xyz[1], target_xyz[1]) + 0.65))
    return (x, y, 0.8481320433582218)


def make_pick_place(target: dict[str, Any]) -> ExpertPickPlace:
    name = str(target["object_name"])
    initial = _xyz(target["initial_xyz"])
    target_xyz = _xyz(target["target_xyz"])
    hand = _choose_hand(initial, target_xyz)
    open_m, closed_m = _gripper_opening(name)
    role = str(target["target_role"])

    grasp_z = initial[2] + 0.035
    place_z = target_xyz[2] + 0.045
    caution = None
    if role != "tabletop_sort_grid":
        caution = "special target region; confirm drawer/sink interaction with official evaluator"

    waypoints = (
        ExpertWaypoint("pregrasp", (initial[0], initial[1], initial[2] + 0.24), "open", 10),
        ExpertWaypoint("grasp", (initial[0], initial[1], grasp_z), "open", 8),
        ExpertWaypoint("close", (initial[0], initial[1], grasp_z), "closed", 10),
        ExpertWaypoint("lift", (initial[0], initial[1], initial[2] + 0.30), "closed", 12),
        ExpertWaypoint("preplace", (target_xyz[0], target_xyz[1], target_xyz[2] + 0.26), "closed", 12),
        ExpertWaypoint("place", (target_xyz[0], target_xyz[1], place_z), "closed", 8),
        ExpertWaypoint("release", (target_xyz[0], target_xyz[1], place_z), "open", 8),
        ExpertWaypoint("retreat", (target_xyz[0], target_xyz[1], target_xyz[2] + 0.26), "open", 10),
    )

    return ExpertPickPlace(
        object_name=name,
        object_path=str(target["object_path"]),
        target_role=role,
        hand=hand,
        palm_path=RIGHT_PALM_PATH if hand == "right" else LEFT_PALM_PATH,
        base_stance_xyz=_base_stance(initial, target_xyz),
        gripper_open_m=open_m,
        gripper_closed_m=closed_m,
        waypoints=waypoints,
        caution=caution,
    )


def build_expert_plan(target_plan: dict[str, Any]) -> dict[str, Any]:
    tasks = [make_pick_place(target) for target in target_plan["targets"]]
    return {
        "source_target_plan_status": target_plan.get("status"),
        "status": "ik_ready_plan_not_executed",
        "robot_prim": "/World/g1",
        "left_palm_path": LEFT_PALM_PATH,
        "right_palm_path": RIGHT_PALM_PATH,
        "task_count": len(tasks),
        "estimated_steps": sum(sum(wp.hold_steps for wp in task.waypoints) for task in tasks),
        "tasks": [asdict(task) for task in tasks],
        "next_controller_requirements": [
            "Use Isaac motion_generation/Lula or cuRobo to solve palm_path waypoints.",
            "Apply gripper open/closed commands at each waypoint.",
            "Record robot-mounted RGB observations and final object transforms.",
            "Only mark generated episodes trainable after real execution scores >= 10/12.",
        ],
    }

