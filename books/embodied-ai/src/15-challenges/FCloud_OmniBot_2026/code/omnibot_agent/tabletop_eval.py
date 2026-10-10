"""Tabletop transform evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from omnibot_agent.scoring import TabletopObjectResult
from omnibot_agent.scoring import TabletopScore


@dataclass(frozen=True)
class ObjectTransform:
    name: str
    xyz: tuple[float, float, float]
    rpy_deg: tuple[float, float, float]


def _angle_error_deg(actual: float, target: float) -> float:
    return abs((actual - target + 180.0) % 360.0 - 180.0)


def _max_abs_delta(actual: tuple[float, ...], target: tuple[float, ...]) -> float:
    return max(abs(a - b) for a, b in zip(actual, target, strict=True))


def _max_angle_delta(actual: tuple[float, ...], target: tuple[float, ...]) -> float:
    return max(_angle_error_deg(a, b) for a, b in zip(actual, target, strict=True))


def _as_tuple3(values: Any) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError(f"Expected 3 values, got {len(values)}")
    return (float(values[0]), float(values[1]), float(values[2]))


def transforms_from_json(data: dict[str, Any]) -> dict[str, ObjectTransform]:
    raw_objects = data.get("objects", data)
    if isinstance(raw_objects, dict):
        items = [{"name": name, **value} for name, value in raw_objects.items()]
    else:
        items = raw_objects
    transforms: dict[str, ObjectTransform] = {}
    for item in items:
        name = str(item.get("name") or item.get("object_name"))
        transforms[name] = ObjectTransform(
            name=name,
            xyz=_as_tuple3(item["xyz"]),
            rpy_deg=_as_tuple3(item.get("rpy_deg", [0.0, 0.0, 0.0])),
        )
    return transforms


def evaluate_tabletop_transforms(
    *,
    target_plan: dict[str, Any],
    final_transforms: dict[str, ObjectTransform],
    elapsed_s: float,
    used_replicator_randomization: bool,
    transformer_visible_count: int,
) -> TabletopScore:
    results: list[TabletopObjectResult] = []
    for target in target_plan["targets"]:
        name = str(target["object_name"])
        final = final_transforms.get(name)
        thresholds = target.get("thresholds", {})
        xyz_threshold = float(thresholds.get("xyz_m", 0.05))
        rpy_threshold = float(thresholds.get("rpy_deg", 10.0))
        if final is None:
            results.append(
                TabletopObjectResult(
                    name=name,
                    in_target_region=False,
                    xyz_error_m=math.inf,
                    rpy_error_deg=math.inf,
                )
            )
            continue
        target_xyz = _as_tuple3(target["target_xyz"])
        target_rpy = _as_tuple3(target.get("target_rpy_deg", [0.0, 0.0, 0.0]))
        xyz_error = _max_abs_delta(final.xyz, target_xyz)
        rpy_error = _max_angle_delta(final.rpy_deg, target_rpy)
        results.append(
            TabletopObjectResult(
                name=name,
                in_target_region=xyz_error <= xyz_threshold and rpy_error <= rpy_threshold,
                xyz_error_m=xyz_error,
                rpy_error_deg=rpy_error,
            )
        )
    return TabletopScore(
        objects=results,
        elapsed_s=elapsed_s,
        used_replicator_randomization=used_replicator_randomization,
        transformer_visible_count=transformer_visible_count,
    )
