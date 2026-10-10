"""Expand a 61D MicroDuck checkpoint to the ladder task's 196D actor input.

The first 61 features are the standard MicroDuck locomotion contract.  The
ladder task appends a 135-ray forward height scan.  Zero-initialising the new
columns preserves the walking policy at the start of fine-tuning while leaving
the scan weights trainable.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import torch


EXTRA_ACTOR = 135
EXTRA_CRITIC = 135


def _append_obs_stat(state: dict, key: str, extra: int, fill: float) -> None:
    value = state[key]
    tail = torch.full(
        (*value.shape[:-1], extra), fill_value=fill, dtype=value.dtype, device=value.device
    )
    state[key] = torch.cat((value, tail), dim=-1)


def _expand_model_state(state: dict, extra: int, first_layer_key: str) -> None:
    for key, fill in (
        ("obs_normalizer._mean", 0.0),
        ("obs_normalizer._var", 1.0),
        ("obs_normalizer._std", 1.0),
    ):
        _append_obs_stat(state, key, extra, fill)
    weight = state["mlp.0.weight"]
    state["mlp.0.weight"] = torch.cat(
        (weight, torch.zeros((weight.shape[0], extra), dtype=weight.dtype)), dim=1
    )


def _expand_optimizer_state(optimizer: dict) -> None:
    # Adam states 1 and 9 correspond to actor/critic first-layer weights in
    # rsl_rl's stable parameter ordering.  Pad their new columns with zeros.
    for state_id, extra in ((1, EXTRA_ACTOR), (9, EXTRA_CRITIC)):
        for key in ("exp_avg", "exp_avg_sq"):
            value = optimizer["state"][state_id][key]
            tail = torch.zeros((value.shape[0], extra), dtype=value.dtype)
            optimizer["state"][state_id][key] = torch.cat((value, tail), dim=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    checkpoint = torch.load(args.source, map_location="cpu", weights_only=False)
    checkpoint["actor_state_dict"] = deepcopy(checkpoint["actor_state_dict"])
    checkpoint["critic_state_dict"] = deepcopy(checkpoint["critic_state_dict"])
    checkpoint["optimizer_state_dict"] = deepcopy(checkpoint["optimizer_state_dict"])
    _expand_model_state(
        checkpoint["actor_state_dict"], EXTRA_ACTOR, "mlp.0.weight"
    )
    _expand_model_state(
        checkpoint["critic_state_dict"], EXTRA_CRITIC, "mlp.0.weight"
    )
    _expand_optimizer_state(checkpoint["optimizer_state_dict"])
    checkpoint["optimizer_state_dict"]["param_groups"][0]["lr"] = 3e-4
    checkpoint["iter"] = 0
    checkpoint.setdefault("infos", {})
    checkpoint["infos"]["ladder_bootstrap"] = {
        "source": str(args.source),
        "actor_input": 61 + EXTRA_ACTOR,
        "critic_input": 76 + EXTRA_CRITIC,
        "scan_columns_zero_initialized": EXTRA_ACTOR,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

