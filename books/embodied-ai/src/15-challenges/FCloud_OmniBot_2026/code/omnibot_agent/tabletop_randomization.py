"""Replicator-compatible tabletop randomization helpers."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
import random


@dataclass(frozen=True)
class RandomizedObjectPose:
    name: str
    path: str
    xyz: tuple[float, float, float]
    rz_deg: float


@dataclass(frozen=True)
class TabletopRandomization:
    seed: int
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    z_by_object: dict[str, float]
    poses: list[RandomizedObjectPose]

    def to_json(self) -> dict:
        return {
            "seed": self.seed,
            "x_range": list(self.x_range),
            "y_range": list(self.y_range),
            "poses": [asdict(pose) for pose in self.poses],
        }


def generate_tabletop_randomization(
    objects: list[dict],
    *,
    seed: int,
    x_range: tuple[float, float] = (4.65, 6.30),
    y_range: tuple[float, float] = (3.72, 4.30),
    min_xy_distance: float = 0.18,
    max_tries: int = 1000,
) -> TabletopRandomization:
    rng = random.Random(seed)
    placed_xy: list[tuple[float, float]] = []
    poses: list[RandomizedObjectPose] = []
    z_by_object: dict[str, float] = {}
    for item in objects:
        name = str(item["name"])
        path = str(item["path"])
        z = float(item["world_translation"][2])
        z_by_object[name] = z
        for _ in range(max_tries):
            x = rng.uniform(*x_range)
            y = rng.uniform(*y_range)
            if all((x - px) ** 2 + (y - py) ** 2 >= min_xy_distance**2 for px, py in placed_xy):
                placed_xy.append((x, y))
                poses.append(
                    RandomizedObjectPose(
                        name=name,
                        path=path,
                        xyz=(round(x, 4), round(y, 4), z),
                        rz_deg=round(rng.uniform(0.0, 360.0), 3),
                    )
                )
                break
        else:
            raise RuntimeError(f"Could not place {name} after {max_tries} attempts")
    return TabletopRandomization(
        seed=seed,
        x_range=x_range,
        y_range=y_range,
        z_by_object=z_by_object,
        poses=poses,
    )
