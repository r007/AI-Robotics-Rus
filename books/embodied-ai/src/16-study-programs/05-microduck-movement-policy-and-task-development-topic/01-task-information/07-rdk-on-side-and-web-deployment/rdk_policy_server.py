#!/usr/bin/env python3
"""Serve a Microduck ONNX policy from an RDK board over a persistent TCP link."""

from __future__ import annotations

import argparse
import hashlib
import socket
import struct
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


MAGIC = b"MDP1"
OBSERVATIONS = 61
ACTIONS = 14
HELLO = struct.Struct("!4sHH32s")
COUNT = struct.Struct("!I")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--max-batch", type=int, default=32)
    return parser.parse_args()


def sha256(path: Path) -> bytes:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.digest()


def receive_exact(connection: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise EOFError("client disconnected")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def serve_client(
    connection: socket.socket,
    session: ort.InferenceSession,
    model_hash: bytes,
    max_batch: int,
) -> None:
    model_input = session.get_inputs()[0]
    model_output = session.get_outputs()[0]
    connection.sendall(HELLO.pack(MAGIC, OBSERVATIONS, ACTIONS, model_hash))
    request_count = 0
    action_count = 0
    inference_ns = 0
    started_ns = time.perf_counter_ns()

    while True:
        batch_size = COUNT.unpack(receive_exact(connection, COUNT.size))[0]
        if batch_size == 0:
            break
        if batch_size > max_batch:
            raise ValueError(f"batch {batch_size} exceeds --max-batch {max_batch}")

        payload = receive_exact(connection, batch_size * OBSERVATIONS * 4)
        observations = np.frombuffer(payload, dtype="<f4").reshape(batch_size, OBSERVATIONS)
        if not np.isfinite(observations).all():
            raise ValueError("received NaN or Inf observation")

        actions = np.empty((batch_size, ACTIONS), dtype=np.float32)
        batch_start_ns = time.perf_counter_ns()
        for index, observation in enumerate(observations):
            actions[index] = session.run(
                [model_output.name],
                {model_input.name: observation.reshape(1, OBSERVATIONS)},
            )[0].reshape(ACTIONS)
        inference_ns += time.perf_counter_ns() - batch_start_ns
        if not np.isfinite(actions).all():
            raise RuntimeError("policy produced NaN or Inf action")

        connection.sendall(COUNT.pack(batch_size) + actions.astype("<f4", copy=False).tobytes())
        request_count += 1
        action_count += batch_size
        if request_count % 250 == 0:
            elapsed_s = (time.perf_counter_ns() - started_ns) / 1_000_000_000.0
            mean_ms = inference_ns / max(action_count, 1) / 1_000_000.0
            print(
                f"requests={request_count} actions={action_count} "
                f"mean_inference_ms={mean_ms:.3f} elapsed_s={elapsed_s:.1f}",
                flush=True,
            )


def main() -> None:
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    model_input = session.get_inputs()[0]
    model_output = session.get_outputs()[0]
    if model_input.shape != [1, OBSERVATIONS] or model_output.shape != [1, ACTIONS]:
        raise RuntimeError(f"Unexpected ONNX contract: {model_input.shape} -> {model_output.shape}")

    model_hash = sha256(model_path)
    print(
        f"RDK policy server: {args.host}:{args.port}, "
        f"contract=[1,{OBSERVATIONS}]->[1,{ACTIONS}], sha256={model_hash.hex()}",
        flush=True,
    )
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, args.port))
        server.listen(2)
        while True:
            connection, address = server.accept()
            print(f"client connected: {address[0]}:{address[1]}", flush=True)
            with connection:
                connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                try:
                    serve_client(connection, session, model_hash, args.max_batch)
                except (EOFError, ConnectionError) as error:
                    print(f"client closed: {error}", flush=True)
                except Exception as error:
                    print(f"client error: {type(error).__name__}: {error}", flush=True)


if __name__ == "__main__":
    main()
