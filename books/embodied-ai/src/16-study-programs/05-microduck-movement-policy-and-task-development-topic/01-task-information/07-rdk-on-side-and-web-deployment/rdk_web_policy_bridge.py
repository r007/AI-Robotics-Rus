#!/usr/bin/env python3
"""Expose the RDK X5 MDP2 policy service to a browser over local HTTP."""

from __future__ import annotations

import argparse
import json
import mimetypes
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np

from rdk_policy_client import RdkPolicyClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rdk-host", required=True)
    parser.add_argument("--rdk-port", type=int, default=8766)
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=8767)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument(
        "--static-root",
        type=Path,
        help="Optional built web directory; serves the simulator and API from one origin",
    )
    return parser.parse_args()


class PolicyBridge:
    def __init__(self, host: str, port: int, timeout: float) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.lock = threading.Lock()
        self.client = self._connect()

    def _connect(self) -> RdkPolicyClient:
        return RdkPolicyClient(self.host, self.port, timeout=self.timeout)

    def infer(self, policy: str, observation: np.ndarray) -> tuple[np.ndarray, float]:
        started = time.perf_counter_ns()
        with self.lock:
            try:
                action = self.client.infer(policy, observation)
            except (ConnectionError, EOFError, OSError):
                self.client.close()
                self.client = self._connect()
                action = self.client.infer(policy, observation)
        return action, (time.perf_counter_ns() - started) / 1_000_000.0

    def close(self) -> None:
        self.client.close()


def make_handler(bridge: PolicyBridge, static_root: Path | None):
    class Handler(BaseHTTPRequestHandler):
        server_version = "RdkX5PolicyBridge/1.0"

        def log_message(self, message: str, *args) -> None:
            print(f"{self.address_string()} - {message % args}", flush=True)

        def end_headers(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def send_json(self, status: int, value: dict) -> None:
            body = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self.end_headers()

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/health":
                self.send_json(
                    200,
                    {
                        "ok": True,
                        "brain": "RDK X5",
                        "local_onnx_fallback": False,
                        "endpoint": f"{bridge.client.host}:{bridge.client.port}",
                    },
                )
                return
            if path == "/catalog":
                self.send_json(200, bridge.client.catalog)
                return
            if static_root is not None:
                relative = unquote(path).lstrip("/") or "index.html"
                candidate = (static_root / relative).resolve()
                try:
                    candidate.relative_to(static_root)
                except ValueError:
                    self.send_json(403, {"error": "path escapes static root"})
                    return
                if candidate.is_dir():
                    candidate = candidate / "index.html"
                if not candidate.is_file() and "." not in Path(relative).name:
                    candidate = static_root / "index.html"
                if candidate.is_file():
                    body = candidate.read_bytes()
                    mime_type = {
                        ".js": "text/javascript; charset=utf-8",
                        ".mjs": "text/javascript; charset=utf-8",
                        ".css": "text/css; charset=utf-8",
                        ".wasm": "application/wasm",
                    }.get(candidate.suffix.lower()) or mimetypes.guess_type(candidate.name)[0]
                    mime_type = mime_type or "application/octet-stream"
                    self.send_response(200)
                    self.send_header("Content-Type", mime_type)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
            self.send_json(404, {"error": "not found"})

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            prefix = "/infer/"
            if not path.startswith(prefix):
                self.send_json(404, {"error": "not found"})
                return
            policy = unquote(path[len(prefix) :])
            contract = bridge.client.policies.get(policy)
            if contract is None:
                self.send_json(404, {"error": f"unknown RDK policy: {policy}"})
                return
            expected_bytes = int(contract["observations"]) * 4
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self.send_json(400, {"error": "invalid Content-Length"})
                return
            if content_length != expected_bytes:
                self.send_json(
                    400,
                    {"error": f"expected {expected_bytes} observation bytes, got {content_length}"},
                )
                return
            body = self.rfile.read(content_length)
            observation = np.frombuffer(body, dtype="<f4").reshape(1, -1).copy()
            try:
                action, latency_ms = bridge.infer(policy, observation)
            except Exception as error:
                self.send_json(502, {"error": f"RDK inference failed: {error}"})
                return
            payload = action.astype("<f4", copy=False).tobytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("X-RDK-Roundtrip-Ms", f"{latency_ms:.3f}")
            self.send_header("X-Policy-Name", policy)
            self.end_headers()
            self.wfile.write(payload)

    return Handler


def main() -> None:
    args = parse_args()
    static_root = args.static_root.expanduser().resolve() if args.static_root else None
    if static_root is not None and not (static_root / "index.html").is_file():
        raise FileNotFoundError(f"static root does not contain index.html: {static_root}")
    bridge = PolicyBridge(args.rdk_host, args.rdk_port, args.timeout)
    server = ThreadingHTTPServer(
        (args.listen_host, args.listen_port),
        make_handler(bridge, static_root),
    )
    print(
        f"browser bridge http://{args.listen_host}:{args.listen_port} -> "
        f"RDK X5 {args.rdk_host}:{args.rdk_port}; local fallback disabled",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        bridge.close()


if __name__ == "__main__":
    main()
