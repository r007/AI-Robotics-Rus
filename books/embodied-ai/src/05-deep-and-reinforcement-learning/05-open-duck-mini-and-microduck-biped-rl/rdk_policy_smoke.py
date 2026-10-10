#!/usr/bin/env python3
"""Validate a Microduck ONNX policy on RDK without touching robot hardware."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


EXPECTED_OBSERVATIONS = 61
EXPECTED_ACTIONS = 14


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Path to the ONNX policy")
    parser.add_argument("--iterations", type=int, default=2000, help="Unpaced benchmark steps")
    parser.add_argument("--warmup", type=int, default=100, help="Warm-up inference steps")
    parser.add_argument("--paced-steps", type=int, default=500, help="Real-time loop steps")
    parser.add_argument("--control-hz", type=float, default=50.0, help="Target control frequency")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-output", type=Path, help="Optional path for the JSON report")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def latency_stats(latencies_ns: list[int]) -> dict[str, float]:
    values_ms = np.asarray(latencies_ns, dtype=np.float64) / 1_000_000.0
    return {
        "mean_ms": float(values_ms.mean()),
        "p50_ms": float(np.percentile(values_ms, 50)),
        "p90_ms": float(np.percentile(values_ms, 90)),
        "p99_ms": float(np.percentile(values_ms, 99)),
        "max_ms": float(values_ms.max()),
    }


def validate_contract(session: ort.InferenceSession) -> tuple[str, str]:
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise RuntimeError(f"Expected one input and one output, got {len(inputs)} and {len(outputs)}")

    model_input = inputs[0]
    model_output = outputs[0]
    if model_input.type != "tensor(float)" or model_output.type != "tensor(float)":
        raise RuntimeError(f"Expected float tensors, got {model_input.type} and {model_output.type}")
    if model_input.shape[-1] != EXPECTED_OBSERVATIONS:
        raise RuntimeError(f"Expected {EXPECTED_OBSERVATIONS} observations, got {model_input.shape}")
    if model_output.shape[-1] != EXPECTED_ACTIONS:
        raise RuntimeError(f"Expected {EXPECTED_ACTIONS} actions, got {model_output.shape}")
    return model_input.name, model_output.name


def infer(
    session: ort.InferenceSession,
    input_name: str,
    output_name: str,
    observation: np.ndarray,
) -> np.ndarray:
    action = session.run([output_name], {input_name: observation})[0]
    if action.shape != (1, EXPECTED_ACTIONS):
        raise RuntimeError(f"Unexpected action shape: {action.shape}")
    if not np.isfinite(action).all():
        raise RuntimeError("Policy produced NaN or Inf")
    return action


def main() -> None:
    args = parse_args()
    if args.iterations <= 0 or args.warmup < 0 or args.paced_steps <= 0:
        raise ValueError("iterations and paced-steps must be positive; warmup must be non-negative")
    if args.control_hz <= 0:
        raise ValueError("control-hz must be positive")

    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    print("SAFE MODE: policy inference only; no IMU, servo, or motor commands are used.")
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name, output_name = validate_contract(session)

    rng = np.random.default_rng(args.seed)
    observations = np.clip(
        rng.normal(0.0, 0.1, size=(max(args.iterations, args.paced_steps), EXPECTED_OBSERVATIONS)),
        -1.0,
        1.0,
    ).astype(np.float32)

    for index in range(args.warmup):
        observation = observations[index % len(observations)][None, :]
        infer(session, input_name, output_name, observation)

    benchmark_latencies: list[int] = []
    action_min = float("inf")
    action_max = float("-inf")
    benchmark_start = time.perf_counter_ns()
    for index in range(args.iterations):
        observation = observations[index][None, :]
        step_start = time.perf_counter_ns()
        action = infer(session, input_name, output_name, observation)
        benchmark_latencies.append(time.perf_counter_ns() - step_start)
        action_min = min(action_min, float(action.min()))
        action_max = max(action_max, float(action.max()))
    benchmark_elapsed_s = (time.perf_counter_ns() - benchmark_start) / 1_000_000_000.0

    period_ns = int(1_000_000_000 / args.control_hz)
    paced_latencies: list[int] = []
    deadline_misses = 0
    paced_start = time.perf_counter_ns()
    for index in range(args.paced_steps):
        release_ns = paced_start + index * period_ns
        remaining_ns = release_ns - time.perf_counter_ns()
        if remaining_ns > 0:
            time.sleep(remaining_ns / 1_000_000_000.0)

        step_start = time.perf_counter_ns()
        observation = observations[index][None, :]
        infer(session, input_name, output_name, observation)
        step_end = time.perf_counter_ns()
        paced_latencies.append(step_end - step_start)
        if step_end > release_ns + period_ns:
            deadline_misses += 1
    paced_elapsed_s = (time.perf_counter_ns() - paced_start) / 1_000_000_000.0

    report = {
        "safe_mode": True,
        "model": str(model_path),
        "sha256": sha256(model_path),
        "providers": session.get_providers(),
        "contract": {
            "input_name": input_name,
            "input_shape": session.get_inputs()[0].shape,
            "output_name": output_name,
            "output_shape": session.get_outputs()[0].shape,
        },
        "unpaced": {
            "iterations": args.iterations,
            "throughput_hz": args.iterations / benchmark_elapsed_s,
            "latency": latency_stats(benchmark_latencies),
        },
        "paced": {
            "target_hz": args.control_hz,
            "steps": args.paced_steps,
            "effective_hz": args.paced_steps / paced_elapsed_s,
            "deadline_misses": deadline_misses,
            "deadline_miss_rate": deadline_misses / args.paced_steps,
            "latency": latency_stats(paced_latencies),
        },
        "action_range": [action_min, action_max],
    }

    rendered = json.dumps(report, ensure_ascii=True, indent=2)
    print(rendered)
    if args.json_output:
        output_path = args.json_output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"Report written to {output_path}")


if __name__ == "__main__":
    main()
