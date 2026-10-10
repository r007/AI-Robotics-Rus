"""Small, dependency-free reference adapter for the ATEC Task D contract.

The adapter illustrates the boundary between a 16-D locomotion policy and a
24-D B2wPiper submission action. It is not a complete controller and does not
start Isaac Sim.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence


TASKD_STAGE_THRESHOLDS = (1.9, 15.0, 21.0)


def _as_float_list(values: Iterable[float], expected: int, name: str) -> List[float]:
    result = [float(value) for value in values]
    if len(result) != expected:
        raise ValueError(f"{name} must contain {expected} values, got {len(result)}")
    return result


def adapt_taskd_16d_to_official(
    locomotion_action: Sequence[float],
    arm_action: Sequence[float] | None = None,
) -> List[float]:
    """Map 12 leg + 4 wheel policy outputs to the official 24-D action.

    Args:
        locomotion_action: 16 values ordered as 12 leg commands followed by
            4 wheel commands.
        arm_action: Optional 8-value Piper command. Zero is used when omitted.

    Returns:
        A 24-value list ordered as legs, wheels, and Piper arm commands.
    """

    base = _as_float_list(locomotion_action, 16, "locomotion_action")
    arm = [0.0] * 8 if arm_action is None else _as_float_list(arm_action, 8, "arm_action")
    return base + arm


def taskd_stage_from_score(score: float) -> str:
    """Return a coarse Task D stage from score thresholds.

    The thresholds mirror the public Logic-TARS reference and must be checked
    against the active official scorer before use in an experiment.
    """

    value = float(score)
    if value < TASKD_STAGE_THRESHOLDS[0]:
        return "approach_box"
    if value < TASKD_STAGE_THRESHOLDS[1]:
        return "push_box"
    if value < TASKD_STAGE_THRESHOLDS[2]:
        return "nav_platform"
    return "climb_finish"


if __name__ == "__main__":
    action = adapt_taskd_16d_to_official(range(16))
    print("official_action_dim:", len(action))
    print("stages:", [taskd_stage_from_score(score) for score in (0.0, 2.0, 16.0, 22.0)])
