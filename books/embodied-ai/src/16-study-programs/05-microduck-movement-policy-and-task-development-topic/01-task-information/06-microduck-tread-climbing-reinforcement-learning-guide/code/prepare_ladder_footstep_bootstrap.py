"""Expand a MicroDuck checkpoint for the V3 foothold-planning task.

The V3 actor and critic append target rungs, phase, swing-foot, foot-position,
contact and curriculum observations to the standard locomotion contract.  The
new columns are zero-initialised (normalizer variance/std is one), so a
walking checkpoint remains a safe warm start while PPO learns the new task.

Unlike the original V2 helper, this script infers the source dimensions and
also handles checkpoints whose optimizer state is empty, which is common for
exported or early MicroDuck checkpoints.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import torch


# V3 adds six explicit support-transfer values to both observation groups:
# contact, root/support height gap, root x error, root z error, uprightness,
# and planar speed.  Keep these constants in one place so a walking
# checkpoint can be expanded before a resumed training run.
TARGET_ACTOR_OBS = 224
TARGET_CRITIC_OBS = 239


def _append_stat(state: dict, key: str, extra: int, fill: float) -> None:
    value = state[key]
    if value.shape[-1] + extra < 0:
        raise ValueError(f"cannot shrink {key}: shape={tuple(value.shape)}")
    if extra == 0:
        return
    tail = torch.full(
        (*value.shape[:-1], extra),
        fill_value=fill,
        dtype=value.dtype,
        device=value.device,
    )
    state[key] = torch.cat((value, tail), dim=-1)


def _expand_model_state(state: dict, target_obs: int) -> tuple[int, int]:
    mean = state["obs_normalizer._mean"]
    old_obs = int(mean.shape[-1])
    extra = target_obs - old_obs
    if extra < 0:
        raise ValueError(
            f"source observation width {old_obs} exceeds V3 target {target_obs}"
        )

    _append_stat(state, "obs_normalizer._mean", extra, 0.0)
    _append_stat(state, "obs_normalizer._var", extra, 1.0)
    _append_stat(state, "obs_normalizer._std", extra, 1.0)

    weight = state["mlp.0.weight"]
    if weight.shape[-1] != old_obs:
        raise ValueError(
            "first layer and observation normalizer disagree: "
            f"weight={tuple(weight.shape)}, obs={old_obs}"
        )
    if extra:
        state["mlp.0.weight"] = torch.cat(
            (
                weight,
                torch.zeros(
                    (weight.shape[0], extra),
                    dtype=weight.dtype,
                    device=weight.device,
                ),
            ),
            dim=1,
        )
    return old_obs, extra


def _expand_optimizer_state(
    optimizer: dict,
    actor_old: int,
    actor_extra: int,
    critic_old: int,
    critic_extra: int,
) -> int:
    """Pad first-layer Adam slots without relying on parameter IDs.

    rsl_rl checkpoint parameter IDs vary between releases.  Matching the
    known first-layer tensor shapes is stable across those releases and leaves
    scalar ``step`` entries untouched.
    """
    state_map = optimizer.get("state", {})
    changed = 0
    first_layers = {
        (actor_old, actor_extra),
        (critic_old, critic_extra),
    }
    for slots in state_map.values():
        for key in ("exp_avg", "exp_avg_sq"):
            value = slots.get(key)
            if not isinstance(value, torch.Tensor) or value.ndim != 2:
                continue
            match = next(
                (
                    (old, extra)
                    for old, extra in first_layers
                    if extra > 0 and value.shape[-1] == old
                ),
                None,
            )
            if match is None:
                continue
            _, extra = match
            slots[key] = torch.cat(
                (
                    value,
                    torch.zeros(
                        (value.shape[0], extra),
                        dtype=value.dtype,
                        device=value.device,
                    ),
                ),
                dim=1,
            )
            changed += 1
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--actor-obs", type=int, default=TARGET_ACTOR_OBS)
    parser.add_argument("--critic-obs", type=int, default=TARGET_CRITIC_OBS)
    parser.add_argument(
        "--action-std",
        type=float,
        default=None,
        help="optional initial actor action standard deviation for exploration",
    )
    args = parser.parse_args()
    if args.action_std is not None and args.action_std <= 0.0:
        raise ValueError("--action-std must be positive")

    checkpoint = torch.load(args.source, map_location="cpu", weights_only=False)
    checkpoint["actor_state_dict"] = deepcopy(checkpoint["actor_state_dict"])
    checkpoint["critic_state_dict"] = deepcopy(checkpoint["critic_state_dict"])
    checkpoint["optimizer_state_dict"] = deepcopy(
        checkpoint.get("optimizer_state_dict", {"state": {}, "param_groups": []})
    )

    actor_old, actor_extra = _expand_model_state(
        checkpoint["actor_state_dict"], args.actor_obs
    )
    critic_old, critic_extra = _expand_model_state(
        checkpoint["critic_state_dict"], args.critic_obs
    )
    optimizer_slots = _expand_optimizer_state(
        checkpoint["optimizer_state_dict"],
        actor_old,
        actor_extra,
        critic_old,
        critic_extra,
    )

    checkpoint["iter"] = 0
    checkpoint.setdefault("infos", {})
    # mjlab restores this counter when a checkpoint is loaded.  Carrying the
    # walking run's value here would silently skip the foothold curriculum and
    # start V3 at the full fourteen-rung stage.
    checkpoint["infos"]["env_state"] = {"common_step_counter": 0}
    checkpoint["infos"]["ladder_footstep_bootstrap"] = {
        "source": str(args.source),
        "actor_source_obs": actor_old,
        "actor_target_obs": args.actor_obs,
        "critic_source_obs": critic_old,
        "critic_target_obs": args.critic_obs,
        "new_columns_zero_initialized": True,
        "optimizer_first_layer_slots_padded": optimizer_slots,
    }
    if args.action_std is not None:
        checkpoint["actor_state_dict"]["distribution.std_param"].fill_(args.action_std)
        checkpoint["infos"]["ladder_footstep_bootstrap"]["action_std"] = args.action_std
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(
        f"wrote {args.output} "
        f"(actor {actor_old}->{args.actor_obs}, "
        f"critic {critic_old}->{args.critic_obs})"
    )


if __name__ == "__main__":
    main()
