"""SQLite persistence for autonomous agent runs (issue #446).

A single SQLite database (``~/.olmlx/agent.db`` by default) holds run
lifecycle rows, per-iteration checkpoints (full message history for resume),
an event log (for SSE replay), and — added in later phases — FTS5 memory and
learned skills.

All runtime reads/writes are offloaded to a worker thread via
``asyncio.to_thread`` (mirroring the prompt-cache disk pattern), so the event
loop never blocks on disk I/O. One connection is shared across those threads
(``check_same_thread=False``) and serialized with a ``threading.Lock`` — SQLite
itself is not safe for concurrent use of a single connection. Schema creation
and the startup interrupted-run scan run synchronously in ``__init__`` /
``mark_interrupted_runs`` because they happen once outside the request path.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

#: Run lifecycle states. ``interrupted`` is assigned at startup to any run left
#: ``running`` by a crash/restart (resumable via ``/resume``).
RUN_STATUSES = frozenset(
    {
        "queued",
        "running",
        "paused",
        "finished",
        "failed",
        "cancelled",
        "interrupted",
    }
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id          TEXT PRIMARY KEY,
    goal        TEXT NOT NULL,
    status      TEXT NOT NULL,
    model       TEXT NOT NULL DEFAULT '',
    config      TEXT NOT NULL DEFAULT '{}',
    parent_id   TEXT,
    depth       INTEGER NOT NULL DEFAULT 0,
    iterations  INTEGER NOT NULL DEFAULT 0,
    tokens      INTEGER NOT NULL DEFAULT 0,
    result      TEXT,
    error       TEXT,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_parent ON runs(parent_id);

CREATE TABLE IF NOT EXISTS checkpoints (
    run_id      TEXT NOT NULL,
    seq         INTEGER NOT NULL,
    messages    TEXT NOT NULL,
    iterations  INTEGER NOT NULL,
    tokens      INTEGER NOT NULL,
    created_at  REAL NOT NULL,
    PRIMARY KEY (run_id, seq)
);

CREATE TABLE IF NOT EXISTS events (
    run_id      TEXT NOT NULL,
    seq         INTEGER NOT NULL,
    type        TEXT NOT NULL,
    data        TEXT NOT NULL,
    created_at  REAL NOT NULL,
    PRIMARY KEY (run_id, seq)
);
"""

_RUN_COLUMNS = (
    "id",
    "goal",
    "status",
    "model",
    "config",
    "parent_id",
    "depth",
    "iterations",
    "tokens",
    "result",
    "error",
    "created_at",
    "updated_at",
)

#: Columns ``update_run`` accepts. ``updated_at`` is always stamped.
_UPDATABLE = frozenset(
    {"goal", "status", "model", "iterations", "tokens", "result", "error", "config"}
)


class AgentStore:
    """SQLite-backed store for agent runs, checkpoints, and events."""

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self.db_path), check_same_thread=False, isolation_level=None
        )
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- run lifecycle ---------------------------------------------------

    async def create_run(
        self,
        *,
        run_id: str,
        goal: str,
        model: str,
        config: dict[str, Any],
        parent_id: str | None = None,
        depth: int = 0,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self._create_run, run_id, goal, model, config, parent_id, depth
        )

    def _create_run(
        self,
        run_id: str,
        goal: str,
        model: str,
        config: dict[str, Any],
        parent_id: str | None,
        depth: int,
    ) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO runs (id, goal, status, model, config, parent_id, "
                "depth, iterations, tokens, created_at, updated_at) "
                "VALUES (?, ?, 'queued', ?, ?, ?, ?, 0, 0, ?, ?)",
                (
                    run_id,
                    goal,
                    model,
                    json.dumps(config),
                    parent_id,
                    depth,
                    now,
                    now,
                ),
            )
        return {
            "id": run_id,
            "goal": goal,
            "status": "queued",
            "model": model,
            "config": config,
            "parent_id": parent_id,
            "depth": depth,
            "iterations": 0,
            "tokens": 0,
            "result": None,
            "error": None,
            "created_at": now,
            "updated_at": now,
        }

    async def get_run(self, run_id: str) -> dict[str, Any] | None:
        return await asyncio.to_thread(self._get_run, run_id)

    def _get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
        return _row_to_run(row) if row else None

    async def list_runs(self) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._list_runs)

    def _list_runs(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM runs ORDER BY created_at DESC, id DESC"
            ).fetchall()
        return [_row_to_run(r) for r in rows]

    async def list_children(self, parent_id: str) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._list_children, parent_id)

    def _list_children(self, parent_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM runs WHERE parent_id = ? ORDER BY created_at ASC, id ASC",
                (parent_id,),
            ).fetchall()
        return [_row_to_run(r) for r in rows]

    async def update_run(self, run_id: str, **fields: Any) -> None:
        await asyncio.to_thread(self._update_run, run_id, fields)

    def _update_run(self, run_id: str, fields: dict[str, Any]) -> None:
        unknown = set(fields) - _UPDATABLE
        if unknown:
            raise ValueError(f"update_run: unknown columns {sorted(unknown)}")
        if "status" in fields and fields["status"] not in RUN_STATUSES:
            raise ValueError(f"update_run: invalid status {fields['status']!r}")
        sets = []
        values: list[Any] = []
        for key, value in fields.items():
            sets.append(f"{key} = ?")
            values.append(json.dumps(value) if key == "config" else value)
        sets.append("updated_at = ?")
        values.append(time.time())
        values.append(run_id)
        with self._lock:
            self._conn.execute(
                f"UPDATE runs SET {', '.join(sets)} WHERE id = ?", values
            )

    # -- checkpoints -----------------------------------------------------

    async def append_checkpoint(
        self, run_id: str, messages: list[dict], iterations: int, tokens: int
    ) -> int:
        return await asyncio.to_thread(
            self._append_checkpoint, run_id, messages, iterations, tokens
        )

    def _append_checkpoint(
        self, run_id: str, messages: list[dict], iterations: int, tokens: int
    ) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(seq) + 1, 0) AS seq FROM checkpoints "
                "WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            seq = int(row["seq"])
            self._conn.execute(
                "INSERT INTO checkpoints (run_id, seq, messages, iterations, "
                "tokens, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (run_id, seq, json.dumps(messages), iterations, tokens, time.time()),
            )
        return seq

    async def latest_checkpoint(self, run_id: str) -> dict[str, Any] | None:
        return await asyncio.to_thread(self._latest_checkpoint, run_id)

    def _latest_checkpoint(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM checkpoints WHERE run_id = ? ORDER BY seq DESC LIMIT 1",
                (run_id,),
            ).fetchone()
        if not row:
            return None
        return {
            "seq": row["seq"],
            "messages": json.loads(row["messages"]),
            "iterations": row["iterations"],
            "tokens": row["tokens"],
            "created_at": row["created_at"],
        }

    # -- events (SSE replay) --------------------------------------------

    async def append_event(self, run_id: str, event: dict[str, Any]) -> int:
        return await asyncio.to_thread(self._append_event, run_id, event)

    def _append_event(self, run_id: str, event: dict[str, Any]) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(seq) + 1, 0) AS seq FROM events WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            seq = int(row["seq"])
            self._conn.execute(
                "INSERT INTO events (run_id, seq, type, data, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    run_id,
                    seq,
                    str(event.get("type", "")),
                    json.dumps(event),
                    time.time(),
                ),
            )
        return seq

    async def get_events(
        self, run_id: str, after_seq: int = -1
    ) -> list[dict[str, Any]]:
        """Return events with ``seq > after_seq`` (default ``-1`` = all)."""
        return await asyncio.to_thread(self._get_events, run_id, after_seq)

    def _get_events(self, run_id: str, after_seq: int) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT seq, type, data, created_at FROM events "
                "WHERE run_id = ? AND seq > ? ORDER BY seq ASC",
                (run_id, after_seq),
            ).fetchall()
        return [
            {
                "seq": r["seq"],
                "type": r["type"],
                "data": json.loads(r["data"]),
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    # -- startup recovery -----------------------------------------------

    async def mark_interrupted_runs(self) -> list[str]:
        """Mark every ``running`` run ``interrupted``; return their ids.

        Called at startup so a crash mid-run leaves resumable state rather than
        a phantom ``running`` row with no live task behind it.
        """
        return await asyncio.to_thread(self._mark_interrupted_runs)

    def _mark_interrupted_runs(self) -> list[str]:
        now = time.time()
        with self._lock:
            rows = self._conn.execute(
                "SELECT id FROM runs WHERE status = 'running' ORDER BY id ASC"
            ).fetchall()
            ids = [r["id"] for r in rows]
            if ids:
                self._conn.execute(
                    "UPDATE runs SET status = 'interrupted', updated_at = ? "
                    "WHERE status = 'running'",
                    (now,),
                )
        return ids


def _row_to_run(row: sqlite3.Row) -> dict[str, Any]:
    run = {key: row[key] for key in _RUN_COLUMNS}
    run["config"] = json.loads(run["config"]) if run["config"] else {}
    return run
