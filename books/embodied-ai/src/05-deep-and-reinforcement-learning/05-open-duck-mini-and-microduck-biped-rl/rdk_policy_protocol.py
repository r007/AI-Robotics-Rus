#!/usr/bin/env python3
"""Binary protocol shared by the RDK policy server and Ubuntu simulator."""

from __future__ import annotations

import json
import socket
import struct
from typing import Any


MAGIC = b"MDP2"
CATALOG_HEADER = struct.Struct("!4sI")
REQUEST_HEADER = struct.Struct("!HI")
RESPONSE_HEADER = struct.Struct("!BI")
STATUS_OK = 0
STATUS_ERROR = 1
MAX_CATALOG_BYTES = 1024 * 1024
MAX_POLICY_NAME_BYTES = 255
MAX_ERROR_BYTES = 64 * 1024


def receive_exact(connection: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise EOFError("policy connection closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_catalog(connection: socket.socket, catalog: dict[str, Any]) -> None:
    payload = json.dumps(catalog, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(payload) > MAX_CATALOG_BYTES:
        raise ValueError("policy catalog is too large")
    connection.sendall(CATALOG_HEADER.pack(MAGIC, len(payload)) + payload)


def receive_catalog(connection: socket.socket) -> dict[str, Any]:
    magic, size = CATALOG_HEADER.unpack(receive_exact(connection, CATALOG_HEADER.size))
    if magic != MAGIC:
        raise RuntimeError(f"unexpected policy protocol magic: {magic!r}")
    if size <= 0 or size > MAX_CATALOG_BYTES:
        raise RuntimeError(f"invalid policy catalog size: {size}")
    catalog = json.loads(receive_exact(connection, size).decode("utf-8"))
    if catalog.get("protocol") != MAGIC.decode("ascii"):
        raise RuntimeError("policy catalog protocol does not match handshake")
    return catalog


def encode_policy_name(name: str) -> bytes:
    payload = name.encode("utf-8")
    if not payload or len(payload) > MAX_POLICY_NAME_BYTES:
        raise ValueError(f"invalid policy name length: {len(payload)}")
    return payload


def send_error(connection: socket.socket, message: str) -> None:
    payload = message.encode("utf-8")[:MAX_ERROR_BYTES]
    connection.sendall(RESPONSE_HEADER.pack(STATUS_ERROR, len(payload)) + payload)
