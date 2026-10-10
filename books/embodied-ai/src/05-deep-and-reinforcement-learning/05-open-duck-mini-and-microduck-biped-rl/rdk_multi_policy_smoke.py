#!/usr/bin/env python3
"""Verify that walking and ball-kick actions are inferred by an RDK X5."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from rdk_policy_client import RdkPolicyClient


POLICIES = ("walking", "kick_left", "kick_right")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="RDK X5 IPv4 address")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")

    outputs: dict[str, dict[str, float | int | list[int]]] = {}
    with RdkPolicyClient(
        args.host,
        args.port,
        timeout=args.timeout,
        required_policies=POLICIES,
    ) as client:
        rng = np.random.default_rng(20260905)
        for name in POLICIES:
            observation_size = int(client.policies[name]["observations"])
            action_size = int(client.policies[name]["actions"])
            minimum = float("inf")
            maximum = float("-inf")
            checksum = 0.0
            for _ in range(args.iterations):
                observation = rng.normal(0.0, 0.05, (1, observation_size)).astype(
                    np.float32
                )
                actions = client.infer(name, observation)
                if actions.shape != (1, action_size):
                    raise RuntimeError(
                        f"{name}: expected action shape (1,{action_size}), got {actions.shape}"
                    )
                minimum = min(minimum, float(actions.min()))
                maximum = max(maximum, float(actions.max()))
                checksum += float(actions.sum())
            outputs[name] = {
                "output_shape": [1, action_size],
                "action_min": minimum,
                "action_max": maximum,
                "aggregate_checksum": checksum,
            }
        report = {
            "verified_at_utc": datetime.now(timezone.utc).isoformat(),
            "rdk_endpoint": f"{args.host}:{args.port}",
            "inference_location": "RDK X5 CPUExecutionProvider",
            "local_fallback": False,
            "catalog": client.catalog,
            "outputs": outputs,
            "round_trip_latency": client.latency_report(),
        }

    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
