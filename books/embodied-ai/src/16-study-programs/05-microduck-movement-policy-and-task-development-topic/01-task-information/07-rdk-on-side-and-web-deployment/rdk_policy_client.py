#!/usr/bin/env python3
"""Strict MDP2 client: every action must come from the connected RDK board."""

from __future__ import annotations

import socket
import time
from collections import defaultdict

import numpy as np

from rdk_policy_protocol import (
    REQUEST_HEADER,
    RESPONSE_HEADER,
    STATUS_ERROR,
    STATUS_OK,
    encode_policy_name,
    receive_catalog,
    receive_exact,
)


class RdkPolicyClient:
    def __init__(
        self,
        host: str,
        port: int = 8765,
        timeout: float = 1.0,
        required_policies: tuple[str, ...] = (),
    ) -> None:
        self.connection = socket.create_connection((host, port), timeout=timeout)
        self.connection.settimeout(timeout)
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.catalog = receive_catalog(self.connection)
        self.policies = self.catalog.get("policies", {})
        missing = sorted(set(required_policies) - set(self.policies))
        if missing:
            self.connection.close()
            raise RuntimeError(f"RDK does not provide required policies: {missing}")
        self.host = host
        self.port = port
        self.request_counts: dict[str, int] = defaultdict(int)
        self.latencies_ns: dict[str, list[int]] = defaultdict(list)

    def infer(self, policy_name: str, observations: np.ndarray) -> np.ndarray:
        policy = self.policies.get(policy_name)
        if policy is None:
            raise KeyError(f"policy is not in RDK catalog: {policy_name}")
        expected_observations = int(policy["observations"])
        expected_actions = int(policy["actions"])
        observations = np.asarray(observations, dtype="<f4")
        if observations.ndim == 1:
            observations = observations[None, :]
        if observations.ndim != 2 or observations.shape[1] != expected_observations:
            raise ValueError(
                f"{policy_name}: expected observations [N,{expected_observations}], "
                f"got {observations.shape}"
            )
        if not np.isfinite(observations).all():
            raise ValueError(f"{policy_name}: observation contains NaN or Inf")
        name = encode_policy_name(policy_name)
        started_ns = time.perf_counter_ns()
        self.connection.sendall(
            REQUEST_HEADER.pack(len(name), len(observations))
            + name
            + observations.tobytes()
        )
        status, count = RESPONSE_HEADER.unpack(
            receive_exact(self.connection, RESPONSE_HEADER.size)
        )
        if status == STATUS_ERROR:
            raise RuntimeError(receive_exact(self.connection, count).decode("utf-8"))
        if status != STATUS_OK or count != len(observations):
            raise RuntimeError(f"invalid RDK response: status={status}, count={count}")
        payload = receive_exact(self.connection, count * expected_actions * 4)
        actions = np.frombuffer(payload, dtype="<f4").reshape(count, expected_actions).copy()
        if not np.isfinite(actions).all():
            raise RuntimeError(f"{policy_name}: RDK returned NaN or Inf")
        self.request_counts[policy_name] += 1
        self.latencies_ns[policy_name].append(time.perf_counter_ns() - started_ns)
        return actions

    def latency_report(self) -> dict[str, dict[str, float | int]]:
        report: dict[str, dict[str, float | int]] = {}
        for name, values in sorted(self.latencies_ns.items()):
            milliseconds = np.asarray(values, dtype=np.float64) / 1_000_000.0
            report[name] = {
                "requests": self.request_counts[name],
                "mean_ms": float(milliseconds.mean()),
                "p50_ms": float(np.percentile(milliseconds, 50)),
                "p99_ms": float(np.percentile(milliseconds, 99)),
                "max_ms": float(milliseconds.max()),
            }
        return report

    def close(self) -> None:
        if self.connection.fileno() < 0:
            return
        try:
            self.connection.sendall(REQUEST_HEADER.pack(0, 0))
        except OSError:
            pass
        finally:
            self.connection.close()

    def __enter__(self) -> "RdkPolicyClient":
        return self

    def __exit__(self, *_args) -> None:
        self.close()
