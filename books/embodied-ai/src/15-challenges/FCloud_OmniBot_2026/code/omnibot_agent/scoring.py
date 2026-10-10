"""Local score estimators for OmniBot challenge tracks.

These estimators are intentionally conservative and mirror the public rules.
They are not official judges; they help us decide whether a run is worth
submitting and generate transparent materials for review.
"""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TabletopObjectResult:
    name: str
    in_target_region: bool
    xyz_error_m: float
    rpy_error_deg: float

    @property
    def placed(self) -> bool:
        return self.in_target_region and self.xyz_error_m <= 0.05 and self.rpy_error_deg <= 10.0


@dataclass(frozen=True)
class TabletopScore:
    objects: list[TabletopObjectResult]
    elapsed_s: float
    used_replicator_randomization: bool
    transformer_visible_count: int

    @property
    def score(self) -> int:
        return sum(obj.placed for obj in self.objects)

    @property
    def max_score(self) -> int:
        return len(self.objects)

    @property
    def submission_ready(self) -> bool:
        return (
            self.max_score == 12
            and self.used_replicator_randomization
            and self.transformer_visible_count >= 12
            and self.score >= 10
        )


@dataclass(frozen=True)
class NavigationScore:
    waypoint_hits: int
    returned_to_start: bool
    collisions: int
    elapsed_s: float
    humanoid_multiplier: float = 1.0

    @property
    def raw_score(self) -> int:
        return self.waypoint_hits + int(self.returned_to_start) - self.collisions

    @property
    def adjusted_score(self) -> float:
        return max(0, self.raw_score) * self.humanoid_multiplier


@dataclass(frozen=True)
class VqaScore:
    correct_answers: int
    total_questions: int
    elapsed_s: float

    @property
    def score(self) -> int:
        return self.correct_answers

    @property
    def accuracy(self) -> float:
        return self.correct_answers / self.total_questions if self.total_questions else 0.0


def summarize_score(score: TabletopScore | NavigationScore | VqaScore) -> dict[str, Any]:
    data = asdict(score)
    if isinstance(score, TabletopScore):
        data.update(score=score.score, max_score=score.max_score, submission_ready=score.submission_ready)
    elif isinstance(score, NavigationScore):
        data.update(raw_score=score.raw_score, adjusted_score=score.adjusted_score)
    elif isinstance(score, VqaScore):
        data.update(score=score.score, accuracy=score.accuracy)
    return data


def write_score(path: Path, score: TabletopScore | NavigationScore | VqaScore) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summarize_score(score), indent=2, ensure_ascii=False), encoding="utf-8")
