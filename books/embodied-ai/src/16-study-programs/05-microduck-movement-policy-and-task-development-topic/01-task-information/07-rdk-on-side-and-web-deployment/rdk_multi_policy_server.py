#!/usr/bin/env python3
"""Serve multiple named Microduck ONNX policies from one RDK X5 process."""

from __future__ import annotations

import argparse
import hashlib
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

from rdk_policy_protocol import (
    REQUEST_HEADER,
    RESPONSE_HEADER,
    STATUS_OK,
    encode_policy_name,
    receive_exact,
    send_catalog,
    send_error,
)


@dataclass(frozen=True)
class Policy:
    name: str
    path: Path
    session: ort.InferenceSession
    input_name: str
    output_name: str
    observations: int
    actions: int
    sha256: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        action="append",
        required=True,
        metavar="NAME=MODEL.onnx",
        help="Named policy; repeat for walking, kick_left, kick_right, and future skills",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--max-batch", type=int, default=32)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_policy_specs(specs: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"policy must use NAME=PATH syntax: {spec!r}")
        name, raw_path = spec.split("=", 1)
        name = name.strip()
        encode_policy_name(name)
        if name in parsed:
            raise ValueError(f"duplicate policy name: {name}")
        parsed[name] = Path(raw_path).expanduser().resolve()
    return parsed


def load_policy(name: str, path: Path) -> Policy:
    if not path.is_file():
        raise FileNotFoundError(path)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise RuntimeError(f"{name}: expected one input and one output")
    model_input = inputs[0]
    model_output = outputs[0]
    if model_input.type != "tensor(float)" or model_output.type != "tensor(float)":
        raise RuntimeError(f"{name}: only float32 policies are supported")
    if len(model_input.shape) != 2 or model_input.shape[0] != 1:
        raise RuntimeError(f"{name}: expected static [1,N] input, got {model_input.shape}")
    if len(model_output.shape) != 2 or model_output.shape[0] != 1:
        raise RuntimeError(f"{name}: expected static [1,N] output, got {model_output.shape}")
    observations = int(model_input.shape[1])
    actions = int(model_output.shape[1])
    return Policy(
        name=name,
        path=path,
        session=session,
        input_name=model_input.name,
        output_name=model_output.name,
        observations=observations,
        actions=actions,
        sha256=file_sha256(path),
    )


def build_catalog(policies: dict[str, Policy]) -> dict:
    return {
        "protocol": "MDP2",
        "runtime": {"provider": "CPUExecutionProvider", "onnxruntime": ort.__version__},
        "policies": {
            name: {
                "observations": policy.observations,
                "actions": policy.actions,
                "sha256": policy.sha256,
            }
            for name, policy in sorted(policies.items())
        },
    }


def infer(policy: Policy, observations: np.ndarray) -> np.ndarray:
    actions = np.empty((len(observations), policy.actions), dtype=np.float32)
    for index, observation in enumerate(observations):
        actions[index] = policy.session.run(
            [policy.output_name],
            {policy.input_name: observation.reshape(1, policy.observations)},
        )[0].reshape(policy.actions)
    if not np.isfinite(actions).all():
        raise RuntimeError(f"{policy.name}: policy produced NaN or Inf")
    return actions


def serve_client(
    connection: socket.socket,
    policies: dict[str, Policy],
    catalog: dict,
    max_batch: int,
    inference_lock: threading.Lock,
) -> None:
    send_catalog(connection, catalog)
    request_count = 0
    inference_ns: dict[str, int] = {name: 0 for name in policies}
    action_count: dict[str, int] = {name: 0 for name in policies}
    while True:
        name_size, batch_size = REQUEST_HEADER.unpack(
            receive_exact(connection, REQUEST_HEADER.size)
        )
        if name_size == 0 and batch_size == 0:
            return
        if name_size <= 0 or name_size > 255:
            raise ValueError(f"invalid policy name length: {name_size}")
        name = receive_exact(connection, name_size).decode("utf-8")
        policy = policies.get(name)
        if policy is None:
            send_error(connection, f"unknown policy: {name}")
            continue
        if batch_size <= 0 or batch_size > max_batch:
            send_error(connection, f"invalid batch size: {batch_size}")
            continue
        payload = receive_exact(connection, batch_size * policy.observations * 4)
        observations = np.frombuffer(payload, dtype="<f4").reshape(
            batch_size, policy.observations
        )
        if not np.isfinite(observations).all():
            send_error(connection, f"{name}: received NaN or Inf observation")
            continue
        started_ns = time.perf_counter_ns()
        with inference_lock:
            actions = infer(policy, observations)
        inference_ns[name] += time.perf_counter_ns() - started_ns
        action_count[name] += batch_size
        connection.sendall(
            RESPONSE_HEADER.pack(STATUS_OK, batch_size)
            + actions.astype("<f4", copy=False).tobytes()
        )
        request_count += 1
        if request_count % 250 == 0:
            summary = " ".join(
                f"{policy_name}={inference_ns[policy_name] / max(action_count[policy_name], 1) / 1e6:.3f}ms"
                for policy_name in sorted(policies)
                if action_count[policy_name]
            )
            print(f"requests={request_count} mean_per_action: {summary}", flush=True)


def handle_client(
    connection: socket.socket,
    address: tuple[str, int],
    policies: dict[str, Policy],
    catalog: dict,
    max_batch: int,
    inference_lock: threading.Lock,
) -> None:
    print(f"client connected: {address[0]}:{address[1]}", flush=True)
    with connection:
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            serve_client(connection, policies, catalog, max_batch, inference_lock)
        except (EOFError, ConnectionError) as error:
            print(f"client closed: {error}", flush=True)
        except Exception as error:
            print(f"client error: {type(error).__name__}: {error}", flush=True)


def main() -> None:
    args = parse_args()
    if args.max_batch <= 0:
        raise ValueError("--max-batch must be positive")
    paths = parse_policy_specs(args.policy)
    policies = {name: load_policy(name, path) for name, path in paths.items()}
    catalog = build_catalog(policies)
    print("RDK X5 multi-policy brain:", flush=True)
    for name, policy in sorted(policies.items()):
        print(
            f"  {name}: [1,{policy.observations}] -> [1,{policy.actions}] "
            f"sha256={policy.sha256}",
            flush=True,
        )
    print(f"listening on {args.host}:{args.port}", flush=True)
    inference_lock = threading.Lock()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, args.port))
        server.listen(4)
        while True:
            connection, address = server.accept()
            threading.Thread(
                target=handle_client,
                args=(
                    connection,
                    address,
                    policies,
                    catalog,
                    args.max_batch,
                    inference_lock,
                ),
                daemon=True,
            ).start()


if __name__ == "__main__":
    main()
