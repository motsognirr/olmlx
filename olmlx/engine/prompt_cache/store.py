"""LRU-based prompt cache store with optional disk offload.

Issue #365.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
from collections import OrderedDict
from pathlib import Path

from olmlx.engine.prompt_cache.state import CachedPromptState

try:
    from mlx_lm.models.cache import (
        load_prompt_cache,
        save_prompt_cache,
    )
except ImportError:  # pragma: no cover
    save_prompt_cache = None  # type: ignore[assignment]
    load_prompt_cache = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


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
    ) -> None:
        self._max_slots = max_slots
        self._entries: OrderedDict[str, CachedPromptState] = OrderedDict()
        self._disk_path = disk_path
        self._model_name = model_name.replace("/", "--")
        self._disk_max_bytes = disk_max_bytes
        self._evict_generation = 0  # bumped by async_evict_all_to_disk

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
            metadata = {"tokens": json.dumps(state.tokens)}
            save_prompt_cache(str(file_path), state.cache, metadata)
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
        if not self._disk_enabled:
            return None
        file_path = self._disk_file_path(cache_id)
        if not file_path.exists():
            return None
        try:
            cache, metadata = load_prompt_cache(str(file_path), return_metadata=True)
            tokens = json.loads(metadata.get("tokens", "[]"))
            state = CachedPromptState(tokens=tokens, cache=cache)
            # Store in memory via set() to respect max_slots capacity.
            # Delete the evicted entry to release GPU buffers promptly.
            evicted = self.set(cache_id, state)
            if evicted is not None:
                del evicted
            # Remove disk file only after set() succeeded
            file_path.unlink(missing_ok=True)
            logger.info(
                "Restored cache '%s' from disk (%d tokens)",
                cache_id,
                len(tokens),
            )
            return state
        except Exception:
            logger.warning(
                "Failed to load cache '%s' from disk",
                cache_id,
                exc_info=True,
            )
            # Remove corrupt file
            file_path.unlink(missing_ok=True)
            return None

    def _cleanup_disk(self) -> None:
        """Remove oldest disk cache files if total size exceeds the limit."""
        if self._disk_max_bytes is None or self._disk_path is None:
            return
        disk_dir = self._disk_dir()
        if not disk_dir.exists():
            return
        # Single stat pass: collect (path, size, mtime) to avoid double-stat
        file_info = []
        for f in disk_dir.glob("*.safetensors"):
            st = f.stat()
            file_info.append((f, st.st_size, st.st_mtime))
        file_info.sort(key=lambda x: x[2])  # sort by mtime
        total = sum(size for _, size, _ in file_info)
        while total > self._disk_max_bytes and file_info:
            path, size, _ = file_info.pop(0)
            total -= size
            path.unlink(missing_ok=True)
            logger.info("Disk cache cleanup: removed %s", path)

    def peek(self, cache_id: str) -> CachedPromptState | None:
        """Read-only check for a cache entry without LRU side effects."""
        return self._entries.get(cache_id)

    def get(self, cache_id: str) -> CachedPromptState | None:
        """Get a cache entry, promoting it to MRU.

        Checks memory first, then disk.  Returns None if not found in either.
        """
        state = self._entries.get(cache_id)
        if state is not None:
            self._entries.move_to_end(cache_id)
            return state
        # Memory miss — try disk
        return self._load_from_disk(cache_id)

    def _set_in_memory(
        self, cache_id: str, state: CachedPromptState
    ) -> tuple[str | None, CachedPromptState | None]:
        """Update in-memory entries.

        Returns (evicted_id, evicted_or_displaced):
        - On cache-ID collision: (None, displaced_state_or_None)
        - On LRU eviction: (evicted_id, evicted_state)
        - No eviction needed: (None, None)
        """
        if cache_id in self._entries:
            self._entries.move_to_end(cache_id)
            old = self._entries[cache_id]
            self._entries[cache_id] = state
            displaced = old if old.cache is not state.cache else None
            return None, displaced
        evicted: CachedPromptState | None = None
        evicted_id: str | None = None
        if len(self._entries) >= self._max_slots:
            evicted_id, evicted = self._entries.popitem(last=False)
        self._entries[cache_id] = state
        return evicted_id, evicted

    def set(self, cache_id: str, state: CachedPromptState) -> CachedPromptState | None:
        """Set a cache entry, evicting LRU if at capacity.

        Returns the displaced CachedPromptState when its GPU resources need
        cleanup (different .cache object), or None when no cleanup is needed.
        Evicted entries are saved to disk if disk offload is enabled.
        """
        evicted_id, evicted = self._set_in_memory(cache_id, state)
        if evicted_id is not None and evicted is not None:
            self._save_to_disk(evicted_id, evicted)
        return evicted

    def remove(self, cache_id: str) -> None:
        """Remove a specific cache entry from memory and disk."""
        self._entries.pop(cache_id, None)
        if self._disk_enabled:
            self._disk_file_path(cache_id).unlink(missing_ok=True)

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
        return removed

    def evict_all_to_disk(self) -> None:
        """Save all in-memory entries to disk, then clear memory.

        Used during memory pressure to free GPU memory while preserving
        cache state on disk for later restoration.
        """
        self._evict_all_to_disk_sync()

    def _evict_all_to_disk_sync(self) -> None:
        """Sync implementation of evict_all_to_disk."""
        if self._disk_enabled:
            for cache_id, state in list(self._entries.items()):
                self._save_to_disk(cache_id, state)
        self._entries.clear()
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
        if not self._disk_enabled:
            return None, None
        file_path = self._disk_file_path(cache_id)
        if not file_path.exists():
            return None, None
        try:
            cache, metadata = load_prompt_cache(str(file_path), return_metadata=True)
            tokens = json.loads(metadata.get("tokens", "[]"))
            state = CachedPromptState(tokens=tokens, cache=cache)
            logger.info(
                "Restored cache '%s' from disk (%d tokens)",
                cache_id,
                len(tokens),
            )
            return state, file_path
        except Exception:
            logger.warning(
                "Failed to load cache '%s' from disk",
                cache_id,
                exc_info=True,
            )
            file_path.unlink(missing_ok=True)
            return None, None

    # -- Async wrappers that offload disk I/O to threads ----------------
    #
    # All _entries mutations happen on the event loop.  Only pure disk I/O
    # (read/write safetensors, unlink, stat) is dispatched to a worker thread.

    async def async_get(self, cache_id: str) -> CachedPromptState | None:
        """Async version of get(). Memory lookup is sync; disk fallback runs in a thread."""
        state = self._entries.get(cache_id)
        if state is not None:
            self._entries.move_to_end(cache_id)
            return state
        # Memory miss — read from disk in a thread, then insert on the event loop.
        gen_before = self._evict_generation
        loaded, disk_path = await asyncio.to_thread(self._read_from_disk, cache_id)
        if loaded is None:
            return None
        # If entries were bulk-evicted during the await (memory pressure),
        # don't re-insert — the eviction was intentional.  Leave the disk
        # file intact so it can be restored later.
        if self._evict_generation != gen_before:
            return None
        # Another coroutine may have populated this cache_id during the await;
        # if so, keep the fresher in-memory entry and discard the stale disk load.
        if cache_id in self._entries:
            self._entries.move_to_end(cache_id)
            # Capture before any await — entries could be cleared during unlink
            existing = self._entries[cache_id]
            if disk_path is not None:
                await asyncio.to_thread(disk_path.unlink, True)
            return existing
        evicted_id, evicted = self._set_in_memory(cache_id, loaded)
        # Save evicted entry to disk first to avoid a window where
        # evicted_id is in neither memory nor disk.
        if evicted_id is not None and evicted is not None:
            await asyncio.to_thread(self._save_to_disk, evicted_id, evicted)
        # Delete disk file only after successful insertion and eviction save
        if disk_path is not None:
            await asyncio.to_thread(disk_path.unlink, True)
        return loaded

    async def async_set(
        self, cache_id: str, state: CachedPromptState
    ) -> CachedPromptState | None:
        """Async version of set(). Memory ops are sync; disk save runs in a thread."""
        evicted_id, evicted = self._set_in_memory(cache_id, state)
        if evicted_id is not None and evicted is not None:
            await asyncio.to_thread(self._save_to_disk, evicted_id, evicted)
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
        if self._disk_enabled:
            snapshot = list(self._entries.items())
        else:
            snapshot = []
        self._entries.clear()
        self._evict_generation += 1
        if snapshot:
            await asyncio.to_thread(self._save_entries_to_disk, snapshot)

    def clear(self) -> None:
        """Remove all cache entries and disk cache files."""
        self._entries.clear()
        if self._disk_path is not None and self._model_name:
            disk_dir = self._disk_dir()
            if disk_dir.exists():
                shutil.rmtree(disk_dir, ignore_errors=True)

    def __len__(self) -> int:
        return len(self._entries)
