"""Distributed inference sideband coordination.

Only imported when OLMLX_EXPERIMENTAL_DISTRIBUTED=true. Uses simple TCP sockets
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

_MAX_MESSAGE_BYTES = 64 * 1024 * 1024  # 64 MiB hard cap


@dataclass
class InferenceRequest:
    """Sideband message from coordinator to workers."""

    prompt_tokens: list[int]
    prompt_text: str
    max_tokens: int
    gen_kwargs: dict[str, Any]
    action: str  # "generate" or "shutdown"

    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "prompt_text": self.prompt_text,
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
            prompt_text=d.get("prompt_text", ""),
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
    if length > _MAX_MESSAGE_BYTES:
        raise ValueError(f"Sideband message too large: {length} bytes")
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
    """Rank 0 sideband server that broadcasts inference params to workers.

    Security: the shared secret is sent in plaintext over TCP. This provides
    authentication against accidental connections, not confidentiality.
    Deploy on a trusted network or use SSH tunnels for encryption.
    """

    def __init__(
        self,
        world_size: int,
        port: int = 32400,
        bind: str = "0.0.0.0",
        secret: str | None = None,
    ) -> None:
        self._world_size = world_size
        self._secret = secret
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
                for s in pending:
                    s.close()
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
                for s in pending:
                    s.close()
                raise TimeoutError(
                    f"Timed out waiting for workers to connect "
                    f"({len(pending)}/{expected} connected)"
                )

        # Phase 2: wait for "ready" message from each worker
        for i, conn in enumerate(pending):
            remaining = timeout - (time.monotonic() - start)
            if remaining <= 0:
                for s in pending[i:]:
                    s.close()
                raise TimeoutError(
                    f"Timed out waiting for workers to report ready "
                    f"({len(self._workers)}/{expected} ready)"
                )
            conn.settimeout(remaining)
            try:
                msg = _recv_message(conn)
                if msg is None or msg.get("action") != "ready":
                    for s in pending[i:]:
                        s.close()
                    raise RuntimeError(
                        f"Worker {i + 1} sent unexpected message instead of ready: {msg}"
                    )
                # Validate shared secret if configured
                if self._secret is not None:
                    if msg.get("secret") != self._secret:
                        for s in pending[i:]:
                            s.close()
                        raise RuntimeError(
                            f"Worker {i + 1} provided invalid secret — "
                            f"rejecting connection"
                        )
                self._workers.append(conn)
                conn.settimeout(None)
                logger.info(
                    "Worker %d/%d ready",
                    len(self._workers),
                    expected,
                )
            except socket.timeout:
                for s in pending[i:]:
                    s.close()
                raise TimeoutError(
                    f"Timed out waiting for workers to report ready "
                    f"({len(self._workers)}/{expected} ready)"
                )
        # Issue 12: Close server socket — no more connections expected
        self._server.close()
        logger.info("All workers connected, server socket closed")

    def broadcast_inference(
        self,
        prompt_tokens: list[int],
        prompt_text: str,
        max_tokens: int,
        gen_kwargs: dict[str, Any],
    ) -> None:
        """Send inference parameters to all connected workers.

        Raises RuntimeError if any worker send fails — partial broadcasts
        leave the cluster in an unrecoverable state (all_sum deadlock).

        Thread safety: callers must serialize access. In olmlx this is
        guaranteed by _inference_lock — only one inference runs at a time.
        """
        msg = {
            "prompt_tokens": prompt_tokens,
            "prompt_text": prompt_text,
            "max_tokens": max_tokens,
            "gen_kwargs": gen_kwargs,
            "action": "generate",
        }
        for i, worker in enumerate(self._workers):
            try:
                _send_message(worker, msg)
            except Exception:
                logger.error(
                    "Failed to broadcast to worker %d/%d — cluster is degraded",
                    i + 1,
                    len(self._workers),
                    exc_info=True,
                )
                # Try to send shutdown to workers that already received the
                # generate message so they don't hang on all_sum indefinitely.
                shutdown_msg = {
                    "prompt_tokens": [],
                    "max_tokens": 0,
                    "gen_kwargs": {},
                    "action": "shutdown",
                }
                for j in range(i):
                    try:
                        _send_message(self._workers[j], shutdown_msg)
                    except Exception:
                        pass
                # Clean up all worker sockets — cluster is unrecoverable
                num_workers = len(self._workers)
                for w in self._workers:
                    try:
                        w.close()
                    except Exception:
                        pass
                self._workers.clear()
                raise RuntimeError(
                    f"Distributed broadcast failed: worker {i + 1}/{num_workers} unreachable. "
                    f"The cluster cannot recover — restart all nodes."
                )

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

    def __init__(self, coordinator_host: str, port: int, timeout: float = 30.0) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(timeout)
        self._sock.connect((coordinator_host, port))
        self._sock.settimeout(None)
        logger.info("Connected to coordinator at %s:%d", coordinator_host, port)

    def send_ready(self, secret: str | None = None) -> None:
        """Signal to coordinator that this worker has loaded and sharded its model."""
        msg: dict[str, Any] = {"action": "ready"}
        if secret is not None:
            msg["secret"] = secret
        _send_message(self._sock, msg)

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
