"""Readers for the generated OmniBot official scene inventory."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INVENTORY = PROJECT_ROOT / "notes" / "generated" / "official_scene_inventory.json"

# The official rules mention 12 tabletop scoring objects. The current USD has
# 13 children below /World/items; treat the floor lamp as a non-tabletop fixture
# unless the official evaluator proves otherwise.
DEFAULT_NON_SCORING_OBJECT_NAMES = frozenset({"_09_ArcFloorLamp"})


@dataclass(frozen=True)
class ScenePrim:
    path: str
    name: str
    type_name: str
    active: bool
    loaded: bool
    world_translation: tuple[float, float, float] | None
    world_rotation_quat_xyzw: tuple[float, float, float, float] | None
    references: tuple[str, ...]
    payloads: tuple[str, ...]


@dataclass(frozen=True)
class HumanCommand:
    line: int
    raw: str
    subject: str | None = None
    command: str | None = None
    xyz: tuple[float, float, float] | None = None
    duration_s: float | None = None


def _tuple_or_none(values: list[float] | None, length: int) -> tuple[float, ...] | None:
    if values is None:
        return None
    if len(values) != length:
        raise ValueError(f"Expected {length} values, got {len(values)}")
    return tuple(float(value) for value in values)


def _prim_from_dict(item: dict[str, Any]) -> ScenePrim:
    return ScenePrim(
        path=str(item["path"]),
        name=str(item["name"]),
        type_name=str(item.get("type_name", "")),
        active=bool(item.get("active", False)),
        loaded=bool(item.get("loaded", False)),
        world_translation=_tuple_or_none(item.get("world_translation"), 3),  # type: ignore[arg-type]
        world_rotation_quat_xyzw=_tuple_or_none(item.get("world_rotation_quat_xyzw"), 4),  # type: ignore[arg-type]
        references=tuple(str(path) for path in item.get("references", [])),
        payloads=tuple(str(path) for path in item.get("payloads", [])),
    )


def _human_command_from_dict(item: dict[str, Any]) -> HumanCommand:
    xyz = item.get("xyz")
    return HumanCommand(
        line=int(item["line"]),
        raw=str(item["raw"]),
        subject=item.get("subject"),
        command=item.get("command"),
        xyz=tuple(float(value) for value in xyz) if xyz is not None else None,
        duration_s=float(item["duration_s"]) if "duration_s" in item else None,
    )


class SceneInventory:
    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data
        self.stage_path = str(data["stage_path"])
        self.objects = [_prim_from_dict(item) for item in data.get("objects", [])]
        self.navigation_checkpoints = [
            _prim_from_dict(item) for item in data.get("navigation_checkpoints", [])
        ]
        self.robots = [_prim_from_dict(item) for item in data.get("robots", [])]
        self.characters = [_prim_from_dict(item) for item in data.get("characters", [])]
        self.human_commands = [
            _human_command_from_dict(item) for item in data.get("human_commands", [])
        ]

    @classmethod
    def load(cls, path: Path = DEFAULT_INVENTORY) -> "SceneInventory":
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def scoring_objects(
        self, non_scoring_names: set[str] | frozenset[str] = DEFAULT_NON_SCORING_OBJECT_NAMES
    ) -> list[ScenePrim]:
        return [item for item in self.objects if item.name not in non_scoring_names]

    def object_by_name(self, name: str) -> ScenePrim:
        for item in self.objects:
            if item.name == name:
                return item
        raise KeyError(name)

    def checkpoint_order(self) -> list[ScenePrim]:
        def key(item: ScenePrim) -> tuple[int, str]:
            if item.name == "disk":
                return (0, item.name)
            if item.name.startswith("disk_"):
                return (int(item.name.split("_", 1)[1]), item.name)
            return (999, item.name)

        return sorted(self.navigation_checkpoints, key=key)

    def human_goto_points(self) -> list[tuple[float, float, float]]:
        return [cmd.xyz for cmd in self.human_commands if cmd.command == "GoTo" and cmd.xyz is not None]
