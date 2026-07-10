"""LRU-based prompt cache store with optional disk offload.

Issue #365.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
from collections import OrderedDict
from pathlib import Path
from typing import Any

from olmlx.engine.prompt_cache.metrics import CacheMetrics
from olmlx.engine.prompt_cache.radix import PrefixCacheIndex
from olmlx.engine.prompt_cache.state import CachedPromptState
from olmlx.utils import tracing as _tracing
from olmlx.utils.loop_affinity import assert_loop_thread

try:
    from mlx_lm.models.cache import (
        load_prompt_cache,
        save_prompt_cache,
    )
except ImportError:  # pragma: no cover
    save_prompt_cache = None  # type: ignore[assignment]
    load_prompt_cache = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Layer classes the byte estimator failed to size (issue #465). Warned once
# per class so bytes_in_ram undercounting is visible without log spam.
_UNSIZED_LAYER_CLASSES: set[type] = set()

# Sentinel distinguishing "no .state attribute" from ".state is None":
# presence of the attribute means the layer matches the ArraysCache-style
# sizing strategy even when the state is currently empty.
_MISSING: Any = object()


def _shed_transient_buffers(state: CachedPromptState) -> None:
    """Free recoverable, transient side buffers before a cache is stored.

    Called from ``_set_in_memory`` — the single insertion chokepoint for every
    store path (``set`` / ``async_set`` / ``insert_checkpoint`` / disk restore
    / ``takeover``), so the "stored caches carry no transient buffer" invariant
    holds universally rather than at individual call sites.

    Only ``TurboQuantKVCache`` is affected: it holds a full-precision dequant
    side buffer (~4-8x the packed footprint at 4-bit) that is recoverable from
    the packed indices+norms and rebuilt lazily on the next
    ``update_and_fetch``. Left resident, it makes ``_estimate_state_bytes`` see
    a quantised cache at ~8x its persistable size, so a single long
    conversation trips the ``ram_budget_bytes`` soft-eviction and is silently
    dropped around ~30k tokens — forcing a full re-prefill every turn after.
    Duck-typed: layers without ``release_dequant_buffers`` (plain KVCache,
    Spectral, Shard, ArraysCache, non-cache placeholders) are skipped.
    """
    for layer in state.cache or ():
        release = getattr(layer, "release_dequant_buffers", None)
        if callable(release):
            release()


def _estimate_state_bytes(state: CachedPromptState) -> int:
    """Best-effort byte estimate of a cached state.

    Walks the per-layer cache list and sums ``nbytes`` of any mlx array
    encountered. Falls back to ``layer.__dict__`` plus ``layer.state`` for
    layer types that don't expose ``.keys`` / ``.values`` directly — most
    importantly ``ArraysCache`` (used by Qwen3.5 / Nemotron-H / Jamba
    hybrid layers), which would otherwise report 0 bytes and let the RAM
    budget overrun silently for the model families the checkpoint path
    primarily targets. Walking ``__dict__`` plus the ``.state`` view
    (deduped by ``id``) covers both surfaces.

    ``TurboQuantKVCache`` holds a full-precision dequantisation side buffer
    (``_key_dequant`` / ``_value_dequant``, ~4-8x the packed footprint at
    4-bit) that ``.state`` deliberately excludes. It is shed via
    ``release_dequant_buffers`` at the store insertion chokepoint (see
    ``_shed_transient_buffers``, called from ``_set_in_memory``), so a stored
    entry walks to its packed size here and the ``ram_budget_bytes``
    soft-eviction guard reflects the persistable footprint rather than the
    transient buffer. A live cache that still carries the buffer is counted
    honestly by the ``__dict__`` walk regardless.

    Unknown layer types still contribute 0, but a layer whose class
    matches *none* of the sizing strategies (no ``.keys``/``.values``
    attributes, no ``.state``, no mlx arrays reachable from ``__dict__``)
    logs a one-time warning per class (issue #465) — otherwise
    ``bytes_in_ram`` undercounts silently and the RAM budget never fires.
    Legitimately-empty caches of known layouts (a fresh ``KVCache`` whose
    ``keys``/``values`` are still ``None``, an ``ArraysCache`` whose
    ``.state`` is a list of ``None``) do not warn: the attribute presence
    itself marks the layout as recognized.
    """
    total = 0
    for layer in state.cache or ():
        # Fast path: KVCache-style layers with .keys / .values attributes.
        seen_attr = False
        for attr in ("keys", "values"):
            arr = getattr(layer, attr, None)
            nbytes = getattr(arr, "nbytes", None)
            if isinstance(nbytes, int):
                total += nbytes
                seen_attr = True
        if seen_attr:
            continue
        # Fallback path: walk every mlx array owned by the layer. Walking
        # ``__dict__`` catches side buffers (e.g. TurboQuant's
        # ``_key_dequant``) that aren't surfaced through ``.state``;
        # walking ``.state`` covers layer types whose backing storage
        # lives in a sub-container exposed only via the property
        # (``ArraysCache``). Container types (tuple / list / dict) are
        # traversed; ``id()`` dedup prevents double-counting the same
        # underlying array when it appears in both views.
        seen_ids: set[int] = set()
        stack: list[Any] = []
        if hasattr(layer, "__dict__"):
            stack.extend(vars(layer).values())
        # ``_MISSING`` (not ``None``) so a present-but-empty ``.state``
        # still counts as a strategy match below. Note ``getattr`` with a
        # default also swallows a *property* that raises AttributeError —
        # e.g. a fresh KVCache, whose ``.state`` getter touches
        # ``self.keys.shape`` while ``keys`` is still None.
        state_attr = getattr(layer, "state", _MISSING)
        if state_attr is not _MISSING:
            stack.append(state_attr)
        while stack:
            item = stack.pop()
            if item is None:
                continue
            if isinstance(item, (tuple, list)):
                stack.extend(item)
                continue
            if isinstance(item, dict):
                stack.extend(item.values())
                continue
            nbytes = getattr(item, "nbytes", None)
            # Guard against non-array objects that happen to have an
            # ``nbytes`` attribute by additionally requiring ``ndim``
            # (which mx.array has, but ints / Dtype singletons / etc. do
            # not).
            ndim = getattr(item, "ndim", None)
            if isinstance(nbytes, int) and ndim is not None:
                key = id(item)
                if key in seen_ids:
                    continue
                seen_ids.add(key)
                total += nbytes
        # No sizing strategy matched this layer's class: warn once per
        # class (issue #465). ``seen_ids`` non-empty means the __dict__ /
        # .state walk found at least one array, i.e. the layer was sized.
        if (
            not seen_ids
            and state_attr is _MISSING
            and not hasattr(layer, "keys")
            and not hasattr(layer, "values")
        ):
            layer_cls = type(layer)
            if layer_cls not in _UNSIZED_LAYER_CLASSES:
                _UNSIZED_LAYER_CLASSES.add(layer_cls)
                logger.warning(
                    "prompt cache byte estimator does not understand %s; "
                    "RAM budget accounting will undercount",
                    layer_cls.__name__,
                )
    return total


class PromptCacheStore:
    """LRU store for per-agent KV caches with optional disk offload.

    When disk_path and model_name are provided, evicted caches are saved to
    disk instead of deleted.  On cache miss, the disk is checked before
    returning None.
    """

    def __init__(
        self,
        max_slots: int,
        disk_path: Path | None = None,
        model_name: str = "",
        disk_max_bytes: int | None = None,
        ram_budget_bytes: int | None = None,
    ) -> None:
        self._max_slots = max_slots
        self._entries: OrderedDict[str, CachedPromptState] = OrderedDict()
        self._disk_path = disk_path
        self._model_name = model_name.replace("/", "--")
        self._disk_max_bytes = disk_max_bytes
        self._ram_budget_bytes = ram_budget_bytes
        self._evict_generation = 0  # bumped by async_evict_all_to_disk
        self._radix = PrefixCacheIndex()
        self.metrics = CacheMetrics()

    @property
    def _disk_enabled(self) -> bool:
        return (
            self._disk_path is not None
            and self._model_name != ""
            and save_prompt_cache is not None
            and load_prompt_cache is not None
        )

    def _disk_dir(self) -> Path:
        """Return the model-specific disk cache directory."""
        assert self._disk_path is not None
        return self._disk_path / self._model_name

    def _disk_file_path(self, cache_id: str) -> Path:
        """Return the disk file path for a given cache_id."""
        safe_name = re.sub(r"[^\w\-.]", "_", cache_id) or "_default"
        return self._disk_dir() / f"{safe_name}.safetensors"

    def _save_to_disk(self, cache_id: str, state: CachedPromptState) -> None:
        """Save a cache entry to disk."""
        with _tracing.span("cache.disk_write", cache_id=cache_id) as _sp:
            if not self._disk_enabled:
                return
            # TurboQuant caches use a custom format incompatible with safetensors
            if state.cache:
                from olmlx.engine.model_manager import _is_serializable_cache

                if not _is_serializable_cache(state.cache):
                    logger.debug(
                        "Skipping disk save for '%s': cache uses non-serializable format (TurboQuant)",
                        cache_id,
                    )
                    return
            try:
                disk_dir = self._disk_dir()
                disk_dir.mkdir(parents=True, exist_ok=True)
                file_path = self._disk_file_path(cache_id)
                metadata = {
                    "tokens": json.dumps(state.tokens),
                    "cache_type": state.cache_type,
                    "is_checkpoint": "1" if state.is_checkpoint else "0",
                }
                save_prompt_cache(str(file_path), state.cache, metadata)
                if file_path.exists():
                    _sp.set_attribute("bytes", file_path.stat().st_size)
                logger.info(
                    "Saved evicted cache '%s' to disk (%s)",
                    cache_id,
                    file_path,
                )
                self._cleanup_disk()
            except Exception:
                logger.warning(
                    "Failed to save cache '%s' to disk",
                    cache_id,
                    exc_info=True,
                )

    def _load_from_disk(self, cache_id: str) -> CachedPromptState | None:
        """Try to load a cache entry from disk. Returns None if not found."""
        with _tracing.span("cache.disk_read", cache_id=cache_id) as _sp:
            _sp.set_attribute("hit", False)
            if not self._disk_enabled:
                return None
            file_path = self._disk_file_path(cache_id)
            if not file_path.exists():
                return None
            try:
                cache, metadata = load_prompt_cache(
                    str(file_path), return_metadata=True
                )
                tokens = json.loads(metadata.get("tokens", "[]"))
                # Pre-PR entries lack cache_type / is_checkpoint metadata;
                # fall through to the CachedPromptState defaults in that
                # case so old disk caches still round-trip correctly.
                cache_type = metadata.get("cache_type", "assistant")
                is_checkpoint = metadata.get("is_checkpoint", "0") == "1"
                state = CachedPromptState(
                    tokens=tokens,
                    cache=cache,
                    cache_type=cache_type,  # type: ignore[arg-type]
                    is_checkpoint=is_checkpoint,
                )
                # Store in memory via set() — respects max_slots AND
                # _enforce_ram_budget. Delete the evicted entry to release
                # GPU buffers promptly.
                evicted = self.set(cache_id, state)
                if evicted is not None:
                    del evicted
                # Remove disk file only if the entry is still in RAM. If
                # the budget evicted it back to disk, the file at the same
                # path was just rewritten by _save_to_disk inside set() —
                # removing it here would lose the entry from both tiers.
                if cache_id in self._entries:
                    file_path.unlink(missing_ok=True)
                    self._refresh_disk_bytes()
                logger.info(
                    "Restored cache '%s' from disk (%d tokens)",
                    cache_id,
                    len(tokens),
                )
                _sp.set_attribute("hit", True)
                return state
            except Exception:
                logger.warning(
                    "Failed to load cache '%s' from disk",
                    cache_id,
                    exc_info=True,
                )
                # Remove corrupt file
                file_path.unlink(missing_ok=True)
                self._refresh_disk_bytes()
                return None

    def _touch_or_resave(
        self, cache_id: str, state: CachedPromptState, file_path: Path
    ) -> None:
        """Refresh the surviving disk copy of a budget-bounced restore.

        Called from async_get's spill loop when the RAM budget evicts the
        entry that was just restored from ``file_path``. Rewriting the
        same bytes would be pure churn (issue #466), but the file's mtime
        must move: ``_cleanup_disk`` evicts oldest-mtime files under the
        disk cap, so a stale timestamp would make the just-used entry the
        first one deleted. If a sibling evictee's save already triggered a
        cleanup that removed the file, fall back to a real save so the
        entry isn't lost from both tiers.
        """
        try:
            os.utime(file_path)
        except FileNotFoundError:
            self._save_to_disk(cache_id, state)
        except OSError:
            # Best-effort, like _save_to_disk's internal handling — a
            # failed mtime refresh must not fail the in-flight request.
            logger.warning(
                "Failed to refresh disk cache mtime for '%s'",
                cache_id,
                exc_info=True,
            )

    def _unlink_and_refresh(self, file_path: Path) -> None:
        """Unlink a disk cache file and refresh ``bytes_on_disk``.

        Convenience helper for the async paths so both the unlink and
        the directory-scan refresh happen in a single thread offload —
        avoids blocking the event loop with the sync ``_refresh_disk_bytes``.
        """
        file_path.unlink(missing_ok=True)
        self._refresh_disk_bytes()

    def _refresh_disk_bytes(self) -> None:
        """Recompute ``bytes_on_disk`` from a directory scan.

        Cheap for the per-model cache count (typically slot-cap-sized).
        Called from every disk-state-changing path so the metric stays
        consistent across saves, removes, and restores.
        """
        if self._disk_path is None:
            return
        disk_dir = self._disk_dir()
        if not disk_dir.exists():
            self.metrics.bytes_on_disk = 0
            return
        self.metrics.bytes_on_disk = sum(
            f.stat().st_size for f in disk_dir.glob("*.safetensors")
        )

    def _cleanup_disk(self) -> None:
        """Refresh bytes_on_disk and, if a size cap is set, remove
        oldest disk cache files until the total fits.
        """
        if self._disk_path is None:
            return
        disk_dir = self._disk_dir()
        if not disk_dir.exists():
            self.metrics.bytes_on_disk = 0
            return
        # Single stat pass: collect (path, size, mtime) to avoid double-stat
        file_info = []
        for f in disk_dir.glob("*.safetensors"):
            st = f.stat()
            file_info.append((f, st.st_size, st.st_mtime))
        file_info.sort(key=lambda x: x[2])  # sort by mtime
        total = sum(size for _, size, _ in file_info)
        if self._disk_max_bytes is not None:
            while total > self._disk_max_bytes and file_info:
                path, size, _ = file_info.pop(0)
                total -= size
                path.unlink(missing_ok=True)
                self.metrics.evictions_disk += 1
                logger.info("Disk cache cleanup: removed %s", path)
        # Always record the post-cleanup total so /api/ps surfaces an
        # accurate disk footprint even when no size cap is configured.
        self.metrics.bytes_on_disk = total

    def peek(self, cache_id: str) -> CachedPromptState | None:
        """Read-only check for a cache entry without LRU side effects."""
        return self._entries.get(cache_id)

    def get(self, cache_id: str) -> CachedPromptState | None:
        """Get a cache entry, promoting it to MRU.

        Checks memory first, then disk.  Returns None if not found in either.
        """
        # _entries/_radix have no internal locking: every mutating entry point
        # (including LRU promotion here) must run on the loop thread (#463).
        assert_loop_thread("PromptCacheStore.get")
        state = self._entries.get(cache_id)
        if state is not None:
            self._entries.move_to_end(cache_id)
            self.metrics.cache_id_hits += 1
            return state
        # Memory miss — try disk
        loaded = self._load_from_disk(cache_id)
        if loaded is not None:
            self.metrics.cache_id_hits += 1
        else:
            self.metrics.cache_id_misses += 1
        return loaded

    def take(self, cache_id: str) -> CachedPromptState | None:
        """Remove and return the in-memory entry (move semantics).

        Memory tier only — the disk fallback lives in ``async_take``.
        Used by the batched path (batching plan Phase 2): its consumers
        are not serialized by the inference lock, so the exclusive
        path's get-then-remove sequence would leave an await window in
        which a concurrent request could grab the same mutable cache
        object. The pop here is synchronous on the loop thread.

        Metrics-neutral: the radix path that calls this has already
        counted a radix hit via ``find_by_prefix``, and ``async_take``
        owns hit/miss accounting for the cache_id path.
        """
        assert_loop_thread("PromptCacheStore.take")
        state = self._entries.pop(cache_id, None)
        if state is None:
            return None
        self._radix.remove(state.tokens, cache_id)
        self.metrics.bytes_in_ram -= _estimate_state_bytes(state)
        return state

    def _set_in_memory(
        self, cache_id: str, state: CachedPromptState
    ) -> tuple[str | None, CachedPromptState | None]:
        """Update in-memory entries.

        Returns (evicted_id, evicted_or_displaced):
        - On cache-ID collision: (None, displaced_state_or_None)
        - On LRU eviction: (evicted_id, evicted_state)
        - No eviction needed: (None, None)
        """
        # Redundant for current callers (every public entry point asserts),
        # but this is the insertion chokepoint for _entries/_radix — a
        # future caller that bypasses the public surface still trips the
        # contract here (#463).
        assert_loop_thread("PromptCacheStore._set_in_memory")
        _shed_transient_buffers(state)
        if cache_id in self._entries:
            self._entries.move_to_end(cache_id)
            old = self._entries[cache_id]
            self.metrics.bytes_in_ram -= _estimate_state_bytes(old)
            self._radix.remove(old.tokens, cache_id)
            self._entries[cache_id] = state
            self._radix.insert(state.tokens, cache_id)
            self.metrics.bytes_in_ram += _estimate_state_bytes(state)
            displaced = old if old.cache is not state.cache else None
            return None, displaced
        evicted: CachedPromptState | None = None
        evicted_id: str | None = None
        if len(self._entries) >= self._max_slots:
            popped_id, popped_state = self._entries.popitem(last=False)
            self._radix.remove(popped_state.tokens, popped_id)
            self.metrics.bytes_in_ram -= _estimate_state_bytes(popped_state)
            self.metrics.evictions_ram += 1
            evicted_id, evicted = popped_id, popped_state
        self._entries[cache_id] = state
        self._radix.insert(state.tokens, cache_id)
        self.metrics.bytes_in_ram += _estimate_state_bytes(state)
        return evicted_id, evicted

    def _enforce_ram_budget(self) -> list[tuple[str, CachedPromptState]]:
        """Evict LRU entries until bytes_in_ram <= ram_budget_bytes.

        Returns the list of (cache_id, state) tuples that were evicted,
        for the caller to spill to disk if applicable.
        """
        evicted: list[tuple[str, CachedPromptState]] = []
        if self._ram_budget_bytes is None:
            return evicted
        while self._entries and self.metrics.bytes_in_ram > self._ram_budget_bytes:
            cache_id, state = self._entries.popitem(last=False)
            self._radix.remove(state.tokens, cache_id)
            self.metrics.bytes_in_ram -= _estimate_state_bytes(state)
            self.metrics.evictions_ram += 1
            evicted.append((cache_id, state))
        return evicted

    def set(self, cache_id: str, state: CachedPromptState) -> CachedPromptState | None:
        """Set a cache entry, evicting LRU if at capacity.

        Returns the displaced CachedPromptState when its GPU resources need
        cleanup (different .cache object), or None when no cleanup is needed.
        Evicted entries are saved to disk if disk offload is enabled.
        """
        assert_loop_thread("PromptCacheStore.set")
        evicted_id, evicted = self._set_in_memory(cache_id, state)
        if evicted_id is not None and evicted is not None:
            self._save_to_disk(evicted_id, evicted)
        # Byte-budget overflow → additional evictions
        for over_id, over_state in self._enforce_ram_budget():
            self._save_to_disk(over_id, over_state)
        return evicted

    def remove(self, cache_id: str) -> None:
        """Remove a specific cache entry from memory and disk."""
        assert_loop_thread("PromptCacheStore.remove")
        existing = self._entries.pop(cache_id, None)
        if existing is not None:
            self._radix.remove(existing.tokens, cache_id)
            self.metrics.bytes_in_ram -= _estimate_state_bytes(existing)
        if self._disk_enabled:
            self._disk_file_path(cache_id).unlink(missing_ok=True)
            self._refresh_disk_bytes()

    def clear_disk(self) -> int:
        """Remove all on-disk entries for this store's model namespace.

        Returns the number of files removed.  Used at probe time for
        non-persistable models to clean up stale pre-PR data that would
        otherwise sit on disk indefinitely (issue #284).  Caller must
        also call ``clear()`` if they want to drop the in-memory entries.
        """
        if not self._disk_enabled or self._disk_path is None:
            return 0
        disk_dir = self._disk_dir()
        if not disk_dir.exists():
            return 0
        removed = 0
        for f in disk_dir.glob("*.safetensors"):
            try:
                f.unlink()
                removed += 1
            except OSError:
                logger.debug("Failed to remove stale disk cache %s", f, exc_info=True)
        # Recompute rather than zero so any files that survived an
        # unlink failure are still reflected in the metric.
        self._refresh_disk_bytes()
        return removed

    def evict_all_to_disk(self) -> None:
        """Save all in-memory entries to disk, then clear memory.

        Used during memory pressure to free GPU memory while preserving
        cache state on disk for later restoration.
        """
        assert_loop_thread("PromptCacheStore.evict_all_to_disk")
        self._evict_all_to_disk_sync()

    def _evict_all_to_disk_sync(self) -> None:
        """Sync implementation of evict_all_to_disk."""
        if self._disk_enabled:
            for cache_id, state in list(self._entries.items()):
                self._save_to_disk(cache_id, state)
        self.metrics.evictions_ram += len(self._entries)
        self._entries.clear()
        self._radix = PrefixCacheIndex()
        self.metrics.bytes_in_ram = 0
        self._evict_generation += 1

    def _save_entries_to_disk(
        self, entries: list[tuple[str, CachedPromptState]]
    ) -> None:
        """Save a snapshot of entries to disk (safe to call from a worker thread)."""
        for cache_id, state in entries:
            self._save_to_disk(cache_id, state)

    def _read_from_disk(
        self, cache_id: str
    ) -> tuple[CachedPromptState | None, Path | None]:
        """Read a cache entry from disk without mutating _entries.

        Returns (state, file_path) so the caller can delete the file after
        a successful insertion into _entries.  Returns (None, None) on miss
        or failure.
        """
        with _tracing.span("cache.disk_read", cache_id=cache_id) as _sp:
            _sp.set_attribute("hit", False)
            if not self._disk_enabled:
                return None, None
            file_path = self._disk_file_path(cache_id)
            if not file_path.exists():
                return None, None
            try:
                cache, metadata = load_prompt_cache(
                    str(file_path), return_metadata=True
                )
                tokens = json.loads(metadata.get("tokens", "[]"))
                cache_type = metadata.get("cache_type", "assistant")
                is_checkpoint = metadata.get("is_checkpoint", "0") == "1"
                state = CachedPromptState(
                    tokens=tokens,
                    cache=cache,
                    cache_type=cache_type,  # type: ignore[arg-type]
                    is_checkpoint=is_checkpoint,
                )
                logger.info(
                    "Restored cache '%s' from disk (%d tokens)",
                    cache_id,
                    len(tokens),
                )
                _sp.set_attribute("hit", True)
                return state, file_path
            except Exception:
                logger.warning(
                    "Failed to load cache '%s' from disk",
                    cache_id,
                    exc_info=True,
                )
                file_path.unlink(missing_ok=True)
                self._refresh_disk_bytes()
                return None, None

    # -- Async wrappers that offload disk I/O to threads ----------------
    #
    # All _entries mutations happen on the event loop.  Only pure disk I/O
    # (read/write safetensors, unlink, stat) is dispatched to a worker thread.
    #
    # Cross-thread array safety (mlx thread-local streams, >=0.31.2, #499):
    # _save_to_disk's save_prompt_cache(...) call builds AND evaluates each
    # layer's `.state` slice entirely within the worker thread it runs on,
    # so that part is self-contained. What must hold *before* a
    # CachedPromptState crosses into to_thread is that its cache arrays are
    # already materialized (no lazy graph bound to the request's original
    # worker thread) — guaranteed upstream by snapshot_cache_for_persistence
    # (explicit eager_eval, or the prefill drive's own mx.eval right before
    # the snapshot). _set_in_memory/_shed_transient_buffers/_estimate_state_
    # bytes/takeover (all event-loop-side) are pure bookkeeping and never
    # build or evaluate array ops, so they don't perturb this.

    async def _to_thread_traced(self, fn: Any, *fn_args: Any) -> Any:
        """Run ``fn`` in a worker thread with the request's OTel context
        re-attached, so the disk span ``fn`` opens nests under the request
        trace.

        When tracing is off ``current_context()`` is ``None`` and we offload
        ``fn`` directly — identical call shape and zero extra allocation versus
        a bare ``asyncio.to_thread``."""
        ctx = _tracing.current_context()
        if ctx is None:
            return await asyncio.to_thread(fn, *fn_args)

        def _runner() -> Any:
            token = _tracing.attach_context(ctx)
            try:
                return fn(*fn_args)
            finally:
                _tracing.detach_context(token)

        return await asyncio.to_thread(_runner)

    async def async_get(self, cache_id: str) -> CachedPromptState | None:
        """Async version of get(). Memory lookup is sync; disk fallback runs in a thread."""
        assert_loop_thread("PromptCacheStore.async_get")
        state = self._entries.get(cache_id)
        if state is not None:
            self._entries.move_to_end(cache_id)
            self.metrics.cache_id_hits += 1
            return state
        # Memory miss — read from disk in a thread, then insert on the event loop.
        gen_before = self._evict_generation
        loaded, disk_path = await self._to_thread_traced(self._read_from_disk, cache_id)
        if loaded is None:
            self.metrics.cache_id_misses += 1
            return None
        # If entries were bulk-evicted during the await (memory pressure),
        # don't re-insert — the eviction was intentional.  Leave the disk
        # file intact so it can be restored later.
        if self._evict_generation != gen_before:
            self.metrics.cache_id_misses += 1
            return None
        # Another coroutine may have populated this cache_id during the await;
        # if so, keep the fresher in-memory entry and discard the stale disk load.
        if cache_id in self._entries:
            self._entries.move_to_end(cache_id)
            # Capture before any await — entries could be cleared during unlink
            existing = self._entries[cache_id]
            if disk_path is not None:
                await asyncio.to_thread(self._unlink_and_refresh, disk_path)
            self.metrics.cache_id_hits += 1
            return existing
        evicted_id, evicted = self._set_in_memory(cache_id, loaded)
        # Save evicted entry to disk first to avoid a window where
        # evicted_id is in neither memory nor disk.
        if evicted_id is not None and evicted is not None:
            await self._to_thread_traced(self._save_to_disk, evicted_id, evicted)
        # The disk restore counts against the RAM byte budget; enforce
        # it now so a chain of disk hits can't quietly blow the budget
        # until the next write.
        for over_id, over_state in self._enforce_ram_budget():
            if over_id == cache_id and disk_path is not None:
                # The just-restored entry bounced straight back out under
                # budget pressure. Its original disk file is the surviving
                # copy (the unlink below is guarded on residency), so
                # rewriting the same bytes would be pure disk churn — touch
                # it instead so the disk-cap LRU sees it as fresh, and only
                # re-save if a sibling evictee's save already cleaned it up.
                await self._to_thread_traced(
                    self._touch_or_resave, cache_id, over_state, disk_path
                )
                continue
            await self._to_thread_traced(self._save_to_disk, over_id, over_state)
        # Only unlink the original disk file if the restored entry is
        # still in RAM. If the budget evicted it back to disk, the file
        # at the same path was just rewritten by _save_to_disk — removing
        # it here would lose the entry from both tiers.
        if disk_path is not None and cache_id in self._entries:
            await asyncio.to_thread(self._unlink_and_refresh, disk_path)
        self.metrics.cache_id_hits += 1
        return loaded

    async def async_take(self, cache_id: str) -> CachedPromptState | None:
        """``take()`` with a disk-tier fallback (read + unlink in a thread).

        The entry leaves the store entirely. Concurrent disk takes of the
        same id are safe: each caller deserializes its own state object,
        and the unlink is idempotent.
        """
        assert_loop_thread("PromptCacheStore.async_take")
        state = self.take(cache_id)
        if state is not None:
            if self._disk_enabled:
                # Defensive: a stale disk copy must not resurrect the
                # entry after this move.
                await asyncio.to_thread(
                    self._unlink_and_refresh, self._disk_file_path(cache_id)
                )
            self.metrics.cache_id_hits += 1
            return state
        if not self._disk_enabled:
            self.metrics.cache_id_misses += 1
            return None
        gen_before = self._evict_generation
        loaded, disk_path = await self._to_thread_traced(self._read_from_disk, cache_id)
        if loaded is None:
            self.metrics.cache_id_misses += 1
            return None
        # A bulk eviction during the await (memory pressure) may have just
        # rewritten this id's file with a *fresh* in-memory state that now
        # lives only on disk; unlinking would destroy its sole copy. Keep
        # the file in that case — the worst outcome of skipping the unlink
        # is a benign stale-read later, never data loss. (Same generation
        # guard as async_get's re-insert path.)
        if disk_path is not None and self._evict_generation == gen_before:
            await asyncio.to_thread(self._unlink_and_refresh, disk_path)
        self.metrics.cache_id_hits += 1
        return loaded

    async def async_set(
        self, cache_id: str, state: CachedPromptState
    ) -> CachedPromptState | None:
        """Async version of set(). Memory ops are sync; disk save runs in a thread."""
        assert_loop_thread("PromptCacheStore.async_set")
        evicted_id, evicted = self._set_in_memory(cache_id, state)
        if evicted_id is not None and evicted is not None:
            await self._to_thread_traced(self._save_to_disk, evicted_id, evicted)
        for over_id, over_state in self._enforce_ram_budget():
            await self._to_thread_traced(self._save_to_disk, over_id, over_state)
        return evicted

    async def async_evict_all_to_disk(self) -> None:
        """Async version of evict_all_to_disk(). Offloads disk I/O to a thread.

        Snapshots and clears _entries on the event loop first, then saves
        the snapshot to disk in a worker thread — no thread touches _entries.

        Note: there is a brief window between clearing memory and completing
        the disk writes where entries are in neither location.  Any async_get
        during this window will return None (cache miss).  This is acceptable
        for a single-user server where this only runs under memory pressure.
        """
        assert_loop_thread("PromptCacheStore.async_evict_all_to_disk")
        if self._disk_enabled:
            snapshot = list(self._entries.items())
        else:
            snapshot = []
        self.metrics.evictions_ram += len(self._entries)
        self._entries.clear()
        self._radix = PrefixCacheIndex()
        self.metrics.bytes_in_ram = 0
        self._evict_generation += 1
        if snapshot:
            await asyncio.to_thread(self._save_entries_to_disk, snapshot)

    def clear(self) -> None:
        """Remove all cache entries and disk cache files.

        Deliberately NOT loop-affine (#463): the only production caller is
        ``_close_loaded_model``, which runs on a worker thread via
        ``asyncio.to_thread`` during unload/eviction/expiry — by then the
        LoadedModel is popped from ``_loaded``, so the closer owns this store
        exclusively and no loop-side caller can race it.

        Note this does blocking disk I/O (``shutil.rmtree``) — do not call it
        from the event loop; loop-side cleanup goes through ``remove`` /
        ``async_evict_all_to_disk`` instead.
        """
        self._entries.clear()
        self._radix = PrefixCacheIndex()
        self.metrics.bytes_in_ram = 0
        if self._disk_path is not None and self._model_name:
            disk_dir = self._disk_dir()
            if disk_dir.exists():
                shutil.rmtree(disk_dir, ignore_errors=True)
        self.metrics.bytes_on_disk = 0

    def find_by_prefix(
        self,
        tokens: list[int],
        min_prefix_tokens: int,
    ) -> tuple[str, CachedPromptState, int] | None:
        """Longest-prefix lookup against in-memory entries.

        Returns (old_cache_id, state, prefix_len) or None if no reachable
        terminal meets `min_prefix_tokens`. The trie also surfaces sibling
        entries whose stored sequence diverges past the shared prefix.
        """
        cache_id, depth = self._radix.find_longest_prefix(
            tokens, min_depth=min_prefix_tokens
        )
        # find_longest_prefix already returns (None, 0) for below-threshold
        # descents, so a non-None cache_id always satisfies the threshold.
        if cache_id is None:
            self.metrics.radix_misses += 1
            return None
        state = self._entries.get(cache_id)
        if state is None:
            # Trie out of sync — defensive.
            self.metrics.radix_misses += 1
            return None
        self.metrics.radix_hits += 1
        return cache_id, state, depth

    # -- Multi-checkpoint API ----------------------------------------------
    #
    # Companion to the cache_id-keyed `get`/`set`/`takeover` API. Used by the
    # message-boundary checkpoint path: many checkpoint entries can coexist,
    # keyed by their token sequences (the cache_id is derived from the tokens).
    # Lookup returns the deepest STRICT-PREFIX match — no trim required, so
    # this path is safe for non-trimmable caches (RotatingKVCache after wrap,
    # ArraysCache via eager-eval; see snapshot_cache_for_persistence).

    @staticmethod
    def _checkpoint_cache_id(tokens: list[int]) -> str:
        """Stable cache_id derived from token list. Two checkpoints over
        identical tokens collide here intentionally — re-insertion replaces.
        """
        # Hash-based keying; collisions across distinct sequences would only
        # cause a false "displaced" eviction which the store handles.
        h = hashlib.sha1(
            b"".join(t.to_bytes(4, "little", signed=False) for t in tokens),
            usedforsecurity=False,
        ).hexdigest()
        return f"ckpt:{h}"

    def insert_checkpoint(self, state: CachedPromptState) -> None:
        """Add a checkpoint state, keyed by its tokens.

        Re-inserting with the same tokens replaces the prior entry.
        """
        cid = self._checkpoint_cache_id(state.tokens)
        evicted = self.set(cid, state)
        if evicted is not None:
            del evicted

    def fetch_nearest(
        self, tokens: list[int]
    ) -> tuple[CachedPromptState, list[int]] | None:
        """Find the deepest stored entry whose tokens are a strict prefix
        of ``tokens``; return (state, suffix) where ``suffix`` is the
        portion of ``tokens`` not covered by ``state.tokens``.

        Returns None when no terminal in the index is a strict prefix.

        Note: in-memory only. After ``async_evict_all_to_disk`` clears
        ``self._radix`` under memory pressure, checkpoint entries that
        spilled to disk are unreachable through this method until a
        warm-start writes a fresh entry whose path overlaps. Adding a
        disk-tier prefix index is tracked separately — see the issue
        thread on PR #397.
        """
        assert_loop_thread("PromptCacheStore.fetch_nearest")
        cid, depth = self._radix.find_strict_prefix(tokens, min_depth=1)
        if cid is None or depth == 0:
            self.metrics.radix_misses += 1
            return None
        state = self._entries.get(cid)
        if state is None:
            # Trie out of sync — defensive. Strict-prefix's descent path
            # is exactly ``tokens[:depth]`` (the stored entry's tokens are
            # a proper prefix of the query), so we can drop the stale
            # terminal here. Otherwise every future query that hits the
            # same subtree would short-circuit on the same stale cid and
            # produce indefinite radix misses for an otherwise-recoverable
            # prefix.
            self._radix.remove(tokens[:depth], cid)
            self.metrics.radix_misses += 1
            return None
        self._entries.move_to_end(cid)
        self.metrics.radix_hits += 1
        suffix = tokens[depth:]
        return state, suffix

    def takeover(
        self,
        old_cache_id: str,
        new_cache_id: str,
    ) -> CachedPromptState | None:
        """Re-key an existing in-memory entry. No KV copy.

        The trie terminal moves from old_cache_id to new_cache_id; the
        entry is promoted to MRU under the new key. Returns the state,
        or None if old_cache_id is no longer present.
        """
        assert_loop_thread("PromptCacheStore.takeover")
        state = self._entries.pop(old_cache_id, None)
        if state is None:
            return None
        self._radix.remove(state.tokens, old_cache_id)
        # Update bytes_in_ram book-keeping: the pop above didn't decrement
        # since we bypassed remove(); _set_in_memory will increment when it
        # re-inserts the same state. Decrement here for symmetry so the
        # two calls cancel out exactly.
        self.metrics.bytes_in_ram -= _estimate_state_bytes(state)
        evicted_id, evicted = self._set_in_memory(new_cache_id, state)
        # The pop above freed a slot, so the subsequent insert cannot
        # trigger an LRU eviction. A non-None evicted_id here would mean
        # blocking disk I/O on the event loop — assert it can't happen
        # rather than silently stalling.
        assert evicted_id is None, (
            "takeover triggered an unexpected LRU eviction; this would "
            "block the event loop with sync disk I/O"
        )
        return state

    def __len__(self) -> int:
        return len(self._entries)
