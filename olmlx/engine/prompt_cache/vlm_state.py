"""Per-cache_id LRU of mlx_vlm PromptCacheState objects for VLM prompt caching.

mlx_vlm's ``stream_generate`` owns the hard parts of cross-turn KV reuse:
prefix matching against a stored token sequence, trimming the cache to the
common prefix, detecting whether the new tokens still contain image
placeholders (so vision features are only recomputed when needed), and
updating the state in place after generation.  This store just bounds how
many ``PromptCacheState`` lineages are retained, keyed by the request's
``cache_id``.

Mirrors the speculative ``_SpecCacheStore`` ergonomics: all inference is
serialized under the inference lock, so no internal locking is needed;
``capacity == 0`` is the kill switch (``OLMLX_VLM_PROMPT_CACHE_SLOTS=0``).

Disk spill (#491): when an entry is evicted from the in-memory LRU and disk
spill is enabled (``OLMLX_VLM_PROMPT_CACHE_DISK=1``), its KV state is written
to a per-model ``.safetensors`` file so a later request for the same
``cache_id`` can restore it instead of re-prefilling. Mirrors the text-path
design (``prompt_cache/store.py``): ``mlx_lm``'s ``save_prompt_cache`` /
``load_prompt_cache`` serialize the per-layer KV (``PromptCacheState`` carries
only ``cache`` + ``token_ids``, so ``token_ids`` rides in the safetensors
metadata sidecar), disk I/O is offloaded with ``asyncio.to_thread`` via the
``async_get`` / ``async_insert`` methods, and a byte-capped oldest-mtime
cleanup bounds disk use. Remaining v1 limits: no radix-takeover, no KV-quant.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

import mlx.core as mx

from olmlx.engine.prompt_cache.checkpoint import flatten_cache_state
from olmlx.utils.loop_affinity import assert_loop_thread

try:
    from mlx_lm.models.cache import load_prompt_cache, save_prompt_cache
except ImportError:  # pragma: no cover - mlx_lm always present in practice
    load_prompt_cache = None  # type: ignore[assignment]
    save_prompt_cache = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def _safe_name(name: str) -> str:
    # Append a short hash of the raw name so distinct ids that sanitize to the
    # same string (e.g. "agent/1" and "agent_1") don't collide onto one file
    # and silently overwrite each other's spill (#634).
    safe = re.sub(r"[^\w\-.]", "_", name) or "_default"
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:12]
    return f"{safe}-{digest}"


class VlmPromptCacheStore:
    def __init__(
        self,
        capacity: int,
        *,
        disk_path: Path | None = None,
        model_name: str = "",
        disk_max_bytes: int | None = None,
    ) -> None:
        self._capacity = max(int(capacity), 0)
        # Insertion-ordered dict as an LRU: first key is least-recently-used.
        self._entries: dict[str, Any] = {}
        # Bumped by every bulk drop (``clear``) so an ``async_get`` whose disk
        # restore was in flight across the drop can tell its result is stale
        # and skip the re-insert (#618; mirrors PromptCacheStore).
        self._evict_generation = 0
        # Cumulative reuse counters surfaced on /api/ps (acceptance criterion 1).
        self._hits = 0
        self._misses = 0
        self._tokens_reused = 0

        # Disk spill (#491). Enabled only when a path is configured AND the
        # mlx_lm serialization helpers imported.
        self._disk_path = disk_path
        self._model_name = model_name
        self._disk_max_bytes = disk_max_bytes
        self._disk_enabled = (
            disk_path is not None
            and save_prompt_cache is not None
            and load_prompt_cache is not None
        )

    @property
    def capacity(self) -> int:
        return self._capacity

    def enabled(self) -> bool:
        return self._capacity > 0

    def clear(self) -> None:
        """Drop retained states. Counters are cumulative and NOT reset here, so
        /api/ps reuse totals survive memory-pressure flushes.

        Deliberately NOT loop-affine (#463): like PromptCacheStore.clear, the
        production caller is the worker-thread close path in
        ``_close_loaded_model``, which owns the store exclusively by then.

        Disk files are left in place: they are a cold tier keyed by cache_id,
        not request state, and the byte cap bounds them. A model unload drops
        the in-memory tier; the disk tier is reclaimed lazily by the cap.
        """
        self._entries.clear()
        # Signal any in-flight async_get that its restore is now stale (#618).
        self._evict_generation += 1

    def note_hit(self, reused_tokens: int) -> None:
        self._hits += 1
        self._tokens_reused += max(int(reused_tokens), 0)

    def note_miss(self) -> None:
        self._misses += 1

    def metrics(self) -> dict[str, int]:
        return {
            "vlm_cache_hits": self._hits,
            "vlm_cache_misses": self._misses,
            "vlm_cache_tokens_reused": self._tokens_reused,
        }

    def get(self, cache_id: str) -> Any | None:
        """Return the PromptCacheState for ``cache_id`` and promote it to
        most-recently-used, or ``None`` on miss / when disabled.

        Memory tier only — callers wanting the disk tier use ``async_get``."""
        assert_loop_thread("VlmPromptCacheStore.get")
        if not self.enabled():
            return None
        state = self._entries.pop(cache_id, None)
        if state is None:
            return None
        self._entries[cache_id] = state  # re-insert at MRU end
        return state

    def insert(self, cache_id: str, state: Any) -> None:
        """Store ``state`` as most-recently-used, dropping evicted entries.

        Memory tier only — callers wanting evictees spilled to disk use
        ``async_insert``. No-op when disabled."""
        for _id, _state in self._insert_capture(cache_id, state):
            pass

    def _insert_capture(self, cache_id: str, state: Any) -> list[tuple[str, Any]]:
        """Memory insert that returns the entries evicted past capacity (instead
        of dropping them), so the async wrappers can spill them to disk."""
        assert_loop_thread("VlmPromptCacheStore.insert")
        if not self.enabled():
            return []
        self._entries.pop(cache_id, None)  # refresh position if already present
        self._entries[cache_id] = state
        evicted: list[tuple[str, Any]] = []
        while len(self._entries) > self._capacity:
            oldest = next(iter(self._entries))
            evicted.append((oldest, self._entries.pop(oldest)))
        return evicted

    # ----- disk tier (#491) --------------------------------------------------

    async def async_get(self, cache_id: str) -> Any | None:
        """Memory-first lookup; on a memory miss, restore from disk (offloaded).

        A disk hit is promoted back into the memory LRU; any entry that gets
        evicted by that promotion is spilled to disk in a worker thread."""
        mem = self.get(cache_id)
        if mem is not None:
            return mem
        if not self._disk_enabled or not self.enabled():
            return None
        gen_before = self._evict_generation
        loaded = await asyncio.to_thread(self._load_from_disk, cache_id)
        if loaded is None:
            return None
        # If a bulk clear() (memory-pressure flush) landed during the await,
        # the eviction was intentional — don't re-insert the stale disk copy
        # and undo the flush. Leave the disk file intact for a later restore.
        if self._evict_generation != gen_before:
            return None
        # Another coroutine may have inserted a fresher state for this id
        # during the await; keep it (get() refreshes it to MRU) and discard
        # the stale disk load rather than let _insert_capture's pop() clobber
        # it.
        existing = self.get(cache_id)
        if existing is not None:
            return existing
        # Promote the restored entry; spill whatever it evicts.
        for over_id, over_state in self._insert_capture(cache_id, loaded):
            await self._spill(over_id, over_state)
        return loaded

    async def async_insert(self, cache_id: str, state: Any) -> None:
        """Insert into memory; spill any evicted entry to disk (offloaded)."""
        for over_id, over_state in self._insert_capture(cache_id, state):
            await self._spill(over_id, over_state)

    async def _spill(self, cache_id: str, state: Any) -> None:
        """Eager-eval the KV state on the loop thread (so no lazy Metal-stream
        graph crosses into the worker thread, #284), then write it to disk in a
        worker thread."""
        # The eager-eval below must run on the loop thread (the offloaded
        # _save_to_disk then sees concrete arrays); guard for consistency with
        # get/insert/_insert_capture.
        assert_loop_thread("VlmPromptCacheStore._spill")
        if not self._disk_enabled:
            return
        cache = getattr(state, "cache", None)
        if not cache:
            return  # never-filled state (inserted empty, evicted before use)
        try:
            states = flatten_cache_state(cache)
            if states:
                mx.eval(states)
        except Exception:
            logger.warning(
                "VLM disk spill: eager-eval failed for '%s'; skipping save",
                cache_id,
                exc_info=True,
            )
            return
        await asyncio.to_thread(self._save_to_disk, cache_id, state)

    def _disk_dir(self) -> Path:
        assert self._disk_path is not None  # _disk_enabled guards callers
        return self._disk_path / _safe_name(self._model_name)

    def _disk_file_path(self, cache_id: str) -> Path:
        return self._disk_dir() / f"{_safe_name(cache_id)}.safetensors"

    def _save_to_disk(self, cache_id: str, state: Any) -> None:
        """Serialize one PromptCacheState to disk. Best-effort: a save failure
        (non-serializable cache, full disk) must never fail the request."""
        cache = getattr(state, "cache", None)
        if not self._disk_enabled or not cache:
            return
        try:
            self._disk_dir().mkdir(parents=True, exist_ok=True)
            file_path = self._disk_file_path(cache_id)
            token_ids = getattr(state, "token_ids", None) or []
            save_prompt_cache(
                str(file_path), cache, {"token_ids": json.dumps(token_ids)}
            )
            self._cleanup_disk()
        except Exception:
            logger.warning(
                "VLM disk spill: failed to save '%s'", cache_id, exc_info=True
            )

    def _load_from_disk(self, cache_id: str) -> Any | None:
        """Reconstruct a PromptCacheState from disk, or None if absent/unreadable."""
        if not self._disk_enabled:
            return None
        file_path = self._disk_file_path(cache_id)
        if not file_path.exists():
            return None
        try:
            from mlx_vlm.generate import PromptCacheState

            cache, metadata = load_prompt_cache(str(file_path), return_metadata=True)
            token_ids = json.loads(metadata.get("token_ids", "[]"))
            state = PromptCacheState()
            state.update(token_ids, cache)
            return state
        except Exception:
            logger.warning(
                "VLM disk spill: failed to load '%s'", cache_id, exc_info=True
            )
            return None

    def _cleanup_disk(self) -> None:
        """Evict oldest-mtime disk files until total size is under the cap."""
        if not self._disk_enabled or self._disk_max_bytes is None:
            return
        try:
            files = sorted(
                self._disk_dir().glob("*.safetensors"),
                key=lambda p: p.stat().st_mtime,
            )
            total = sum(p.stat().st_size for p in files)
            for p in files:
                if total <= self._disk_max_bytes:
                    break
                size = p.stat().st_size
                p.unlink(missing_ok=True)
                total -= size
        except Exception:
            logger.warning("VLM disk spill: cleanup failed", exc_info=True)
