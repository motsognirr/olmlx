"""Distributed inference sideband coordination.

Only imported when EXPERIMENTAL_DISTRIBUTED=true. Uses simple TCP sockets
with length-prefixed JSON messages for coordinator↔worker communication.
"""

from __future__ import annotations

import json
import logging
import socket
import struct
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class InferenceRequest:
    """Sideband message from coordinator to workers."""

    prompt_tokens: list[int]
    max_tokens: int
    gen_kwargs: dict[str, Any]
    action: str  # "generate" or "shutdown"

    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "max_tokens": self.max_tokens,
            "gen_kwargs": self.gen_kwargs,
            "action": self.action,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> InferenceRequest:
        return cls(
            prompt_tokens=d["prompt_tokens"],
            max_tokens=d["max_tokens"],
            gen_kwargs=d["gen_kwargs"],
            action=d["action"],
        )

    @classmethod
    def from_json(cls, data: str | bytes) -> InferenceRequest:
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        return cls.from_dict(json.loads(data))


def _send_message(sock: socket.socket, msg: dict) -> None:
    """Send a length-prefixed JSON message over a socket."""
    payload = json.dumps(msg).encode("utf-8")
    header = struct.pack("!I", len(payload))
    sock.sendall(header + payload)


def _recv_message(sock: socket.socket) -> dict | None:
    """Receive a length-prefixed JSON message. Returns None on closed connection."""
    header = _recv_exact(sock, 4)
    if header is None:
        return None
    (length,) = struct.unpack("!I", header)
    payload = _recv_exact(sock, length)
    if payload is None:
        return None
    return json.loads(payload.decode("utf-8"))


def _recv_exact(sock: socket.socket, n: int) -> bytes | None:
    """Receive exactly n bytes. Returns None on closed connection."""
    data = bytearray()
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            return None
        data.extend(chunk)
    return bytes(data)


class DistributedCoordinator:
    """Rank 0 sideband server that broadcasts inference params to workers."""

    def __init__(
        self, world_size: int, port: int = 32400, bind: str = "0.0.0.0"
    ) -> None:
        self._world_size = world_size
        self._workers: list[socket.socket] = []
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind((bind, port))
        self._server.listen(world_size - 1)
        self._port = self._server.getsockname()[1]
        logger.info(
            "Distributed coordinator listening on %s:%d (expecting %d workers)",
            bind,
            self._port,
            world_size - 1,
        )

    @property
    def port(self) -> int:
        return self._port

    def wait_for_workers(self, timeout: float = 30.0) -> None:
        """Block until all N-1 workers connect and report ready.

        Workers must send a {"action": "ready"} message after loading and
        sharding their model. This prevents the coordinator from broadcasting
        inference before workers are actually ready.
        """
        expected = self._world_size - 1
        start = time.monotonic()
        pending: list[socket.socket] = []

        # Phase 1: accept TCP connections
        while len(pending) < expected:
            remaining = timeout - (time.monotonic() - start)
            if remaining <= 0:
                raise TimeoutError(
                    f"Timed out waiting for workers to connect "
                    f"({len(pending)}/{expected} connected)"
                )
            self._server.settimeout(remaining)
            try:
                conn, addr = self._server.accept()
                pending.append(conn)
                logger.info(
                    "Worker connected from %s (%d/%d)",
                    addr,
                    len(pending),
                    expected,
                )
            except socket.timeout:
                raise TimeoutError(
                    f"Timed out waiting for workers to connect "
                    f"({len(pending)}/{expected} connected)"
                )

        # Phase 2: wait for "ready" message from each worker
        for i, conn in enumerate(pending):
            remaining = timeout - (time.monotonic() - start)
            if remaining <= 0:
                raise TimeoutError(
                    f"Timed out waiting for workers to report ready "
                    f"({len(self._workers)}/{expected} ready)"
                )
            conn.settimeout(remaining)
            try:
                msg = _recv_message(conn)
                if msg is None or msg.get("action") != "ready":
                    raise TimeoutError(
                        f"Worker {i + 1} sent unexpected message instead of ready: {msg}"
                    )
                self._workers.append(conn)
                conn.settimeout(None)
                logger.info(
                    "Worker %d/%d ready",
                    len(self._workers),
                    expected,
                )
            except socket.timeout:
                raise TimeoutError(
                    f"Timed out waiting for workers to report ready "
                    f"({len(self._workers)}/{expected} ready)"
                )
        self._server.settimeout(None)

    def broadcast_inference(
        self,
        prompt_tokens: list[int],
        max_tokens: int,
        gen_kwargs: dict[str, Any],
    ) -> None:
        """Send inference parameters to all connected workers."""
        msg = {
            "prompt_tokens": prompt_tokens,
            "max_tokens": max_tokens,
            "gen_kwargs": gen_kwargs,
            "action": "generate",
        }
        for worker in self._workers:
            _send_message(worker, msg)

    def broadcast_shutdown(self) -> None:
        """Tell all workers to exit."""
        msg = {
            "prompt_tokens": [],
            "max_tokens": 0,
            "gen_kwargs": {},
            "action": "shutdown",
        }
        for worker in self._workers:
            try:
                _send_message(worker, msg)
            except Exception:
                logger.warning("Failed to send shutdown to worker", exc_info=True)

    def close(self) -> None:
        """Clean up all sockets."""
        for worker in self._workers:
            try:
                worker.close()
            except Exception:
                pass
        self._workers.clear()
        try:
            self._server.close()
        except Exception:
            pass


class DistributedWorker:
    """Non-rank-0 sideband client that receives inference params from coordinator."""

    def __init__(self, coordinator_host: str, port: int) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.connect((coordinator_host, port))
        logger.info("Connected to coordinator at %s:%d", coordinator_host, port)

    def send_ready(self) -> None:
        """Signal to coordinator that this worker has loaded and sharded its model."""
        _send_message(self._sock, {"action": "ready"})

    def wait_for_inference(self) -> InferenceRequest | None:
        """Block until next broadcast. Returns None on shutdown."""
        msg = _recv_message(self._sock)
        if msg is None:
            return None
        req = InferenceRequest.from_dict(msg)
        if req.action == "shutdown":
            return None
        return req

    def close(self) -> None:
        try:
            self._sock.close()
        except Exception:
            pass
