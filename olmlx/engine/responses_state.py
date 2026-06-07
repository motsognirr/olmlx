"""In-memory LRU store for Responses-API state (previous_response_id).

Single-user, localhost-only: a bounded OrderedDict keyed by response_id.
Not persisted; lost on restart. Bounded by OLMLX_RESPONSES_STORE_MAX so it is
not counted against the memory limit (text-only entries).
"""

from __future__ import annotations

from collections import OrderedDict
from threading import Lock
from typing import Any

from olmlx.config import settings


class ResponsesStore:
    def __init__(self, max_entries: int | None = None) -> None:
        self._max = (
            max_entries if max_entries is not None else settings.responses_store_max
        )
        self._store: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = Lock()

    def put(self, response_id: str, entry: dict[str, Any]) -> None:
        with self._lock:
            self._store[response_id] = entry
            self._store.move_to_end(response_id)
            while len(self._store) > self._max:
                self._store.popitem(last=False)

    def get(self, response_id: str) -> dict[str, Any] | None:
        with self._lock:
            entry = self._store.get(response_id)
            if entry is not None:
                self._store.move_to_end(response_id)
            return entry

    def delete(self, response_id: str) -> bool:
        with self._lock:
            return self._store.pop(response_id, None) is not None

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


_store = ResponsesStore()


def get_store() -> ResponsesStore:
    """Return the process-wide Responses store singleton."""
    return _store
