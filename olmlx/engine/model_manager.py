from __future__ import annotations

import asyncio
import gc
import importlib
import json
import logging
import re
import shutil
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlx.core as mx

from olmlx.config import SyncMode, experimental as global_experimental
from olmlx.config import resolve_experimental, settings
from olmlx.engine.registry import ModelRegistry, SpeculativeConfig
from olmlx.utils import memory as memory_utils
from olmlx.engine.template_caps import TemplateCaps, detect_caps

if TYPE_CHECKING:
    from olmlx.models.store import ModelStore

from olmlx.models.store import _strip_ollama_tag

logger = logging.getLogger(__name__)


try:
    from mlx_lm.models.cache import (
        load_prompt_cache,
        save_prompt_cache,
    )
except ImportError:  # pragma: no cover
    save_prompt_cache = None  # type: ignore[assignment]
    load_prompt_cache = None  # type: ignore[assignment]

# Exceptions from mlx-lm.load() that indicate the model simply isn't
# compatible with mlx-lm and should be retried with mlx-vlm.  Errors like
# ImportError, MemoryError, and RuntimeError indicate real problems that
# should propagate immediately.
_FALLBACK_EXCEPTIONS = (
    ValueError,
    KeyError,
    FileNotFoundError,
    OSError,
    json.JSONDecodeError,
)


def _sanitize_model_config_in_place(load_path) -> None:
    """Fix known on-disk config.json issues that block transformers loading.

    Currently handles: ``layer_types`` longer than ``num_hidden_layers``.
    Step-3.5 ships with ``num_hidden_layers=45`` but ``len(layer_types)=48``
    (the trailing 3 entries describe MTP layers whose weights mlx-lm's
    sanitize() drops). Newer transformers' ``validate_layer_type`` rejects
    the mismatch. Truncating ``layer_types`` to match is consistent with
    what mlx-lm does for the weights and is idempotent on subsequent loads.
    """
    config_file = Path(load_path) / "config.json"
    if not config_file.exists():
        return
    try:
        cfg = json.loads(config_file.read_text())
    except json.JSONDecodeError:
        return

    nhl = cfg.get("num_hidden_layers")
    layer_types = cfg.get("layer_types")
    if (
        isinstance(layer_types, list)
        and isinstance(nhl, int)
        and nhl > 0
        and len(layer_types) > nhl
    ):
        logger.info(
            "Truncating layer_types from %d to %d entries in %s "
            "(num_hidden_layers); excess entries describe layers mlx-lm drops "
            "(e.g. MTP).",
            len(layer_types),
            nhl,
            config_file,
        )
        cfg["layer_types"] = layer_types[:nhl]
        try:
            config_file.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
        except OSError:
            logger.warning(
                "Failed to write sanitized layer_types to %s "
                "(read-only?); model may fail to load if transformers validates "
                "the mismatch.",
                config_file,
            )


def _load_with_model_type_fallback(mlx_lm, load_path, **kwargs):
    """Load model + tokenizer, remapping unrecognised model_type if needed.

    Some very new models (e.g. DeepSeek V3.2 with model_type "deepseek_v32")
    aren't in the installed transformers' CONFIG_MAPPING yet, causing
    ``PreTrainedConfig.__post_init__`` to crash during tokenizer loading.

    The model architecture (loaded by mlx-lm's own registry) is fine — only
    the tokenizer loading via ``AutoTokenizer`` / ``PreTrainedConfig`` fails.
    So we load the model first with the real config, then patch config.json
    temporarily to load only the tokenizer.
    """
    _sanitize_model_config_in_place(load_path)
    kwargs.setdefault("tokenizer_config", {"trust_remote_code": True})
    try:
        return mlx_lm.load(str(load_path), **kwargs)
    except (AttributeError, ValueError, KeyError) as exc:
        config_file = Path(load_path) / "config.json"
        if not config_file.exists():
            raise
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        original_text = config_file.read_text()
        cfg = json.loads(original_text)
        mt = cfg.get("model_type", "")
        # Strip last digit: deepseek_v32 → deepseek_v3
        fallback = re.sub(r"\d+$", lambda m: m.group()[:-1], mt) if mt else ""
        if not fallback or fallback == mt or fallback not in CONFIG_MAPPING:
            raise

        logger.info(
            "Loading model with real config, tokenizer with %r -> %r (%s)",
            mt,
            fallback,
            exc,
        )
        # Load model with the real config (mlx-lm knows the architecture).
        model, model_cfg = mlx_lm.utils.load_model(
            Path(load_path),
            **{k: v for k, v in kwargs.items() if k in ("lazy", "model_config")},
        )
        # Temporarily patch config.json for tokenizer loading only.
        try:
            cfg["model_type"] = fallback
            config_file.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
            tokenizer = mlx_lm.utils.load_tokenizer(
                Path(load_path),
                kwargs.get("tokenizer_config"),
                eos_token_ids=model_cfg.get("eos_token_id", None),
            )
        finally:
            config_file.write_text(original_text)
        return model, tokenizer


def _is_serializable_cache(cache: list) -> bool:
    """Check if a cache list can be serialized with mlx-lm's save_prompt_cache."""
    from olmlx.engine.spectralquant_cache import SpectralQuantKVCache
    from olmlx.engine.turboquant_cache import TurboQuantKVCache

    return not any(
        isinstance(c, (TurboQuantKVCache, SpectralQuantKVCache)) for c in cache
    )


# mlx-lm cache classes whose `trim(n)` reliably removes exactly n tokens from
# any offset (no silent under-delivery).  Class-name allowlist is authoritative
# because `is_trimmable()` on a fresh (offset==0) cache cannot detect rotate
# or chunk problems that only manifest once the buffer fills.
#
# Verified trim behaviour against mlx-lm's cache.py:
# - KVCache.trim(n) → min(offset, n); trims back to 0 cleanly (line 378).
# - QuantizedKVCache.trim(n) → same clamping (line 309).
# - ConcatenateKVCache.trim(n) → same clamping (line 214); used by afm7.
#
# Deliberately excluded:
# - RotatingKVCache: is_trimmable() returns False once the ring buffer fills.
# - ChunkedKVCache: trim() silently clamps to (offset - start_position);
#   once prompt > chunk_size, trims beyond the current chunk under-deliver.
# - ArraysCache / CacheList: no usable trim semantics for our purposes.
# - BatchKVCache / BatchRotatingKVCache: batch path; untested here.
#
# Probe always inspects bare mlx-lm classes (not quantization wrappers like
# TurboQuant/Spectral), so those wrappers do not need to appear here even
# though they implement trim correctly — they're simply never the layer
# types we see.  See ml-explore/mlx-lm#980 for upstream hybrid-trim work.
_TRIMMABLE_CACHE_CLASSES = frozenset(
    {
        "KVCache",
        "QuantizedKVCache",
        "ConcatenateKVCache",
    }
)


def _cache_supports_trim(cache_list: list) -> bool:
    """True iff every layer is in the known-trimmable allowlist.

    Used at model load time to decide whether `trim_prompt_cache()` is worth
    attempting on this model.  Hybrid sliding-window models (Gemma 4,
    Qwen3-Next, etc.) include `RotatingKVCache` layers and return False —
    such models still benefit from strict-extension cache reuse but cannot
    be trimmed back on prompt divergence.
    """
    return all(type(layer).__name__ in _TRIMMABLE_CACHE_CLASSES for layer in cache_list)


# Cache layer classes whose stored state CAN be safely reused on a later
# request.  Allowlist (not denylist) so that any new mlx-lm cache class
# defaults to non-persistable: a false negative here just means a missed
# cache hit, while a false positive (treating an unknown SSM-style cache
# as persistable) crashes mlx-lm during the next prefill with "RuntimeError:
# There is no Stream(gpu, N) in current thread".  Issue #284.
#
# ArraysCache (gated-delta SSM state used by Qwen3.5, Qwen3-Next) is the
# motivating exclusion: its stored arrays carry a lazy graph that
# references a Metal stream from the previous worker thread; re-evaluating
# them in a fresh worker thread fails.  Within-request cache reuse during
# a single generation works fine; only cross-request persistence is unsafe.
#
# Verified safe against mlx-lm by inspection of mlx_lm/models/cache.py:
# the failure mode in #284 is that ArraysCache stores arrays produced by
# the ``gated_delta_kernel`` (mx.fast.metal_kernel) whose lazy graph
# carries a Metal stream reference from the generating worker thread,
# and re-evaluating that graph in a different worker thread raises
# "There is no Stream(gpu, N)".  The classes in the allowlist below
# all store keys/values produced by stock matmul/attention ops only —
# no metal_kernel outputs, no generator-thread-bound state — so their
# arrays are reusable across worker threads:
# - KVCache: bare keys + values, mx.concatenate at update boundaries.
# - QuantizedKVCache: same, with mx.quantize/dequantize wrappers.
# - ConcatenateKVCache: same, used by AFM-7 family.
# - RotatingKVCache: ring buffer over fixed window (Gemma 4); writes
#   in place via mx.assign at modular offsets.
# - ChunkedKVCache: chunked layout (afm7); same semantics as KVCache
#   bounded by chunk_size.
# If a future mlx-lm release adds metal_kernel-style state to any of
# these, that class must be removed from the allowlist.
#
# Deliberately excluded:
# - ArraysCache: gated-delta SSM state — see issue #284.
# - CacheList: wraps other caches, would need recursion to classify safely.
# - BatchKVCache / BatchRotatingKVCache: batch path, untested in olmlx
#   (single-user server) — disable persistence rather than risk a crash.
_PERSISTABLE_CACHE_CLASSES = frozenset(
    {
        "KVCache",
        "QuantizedKVCache",
        "ConcatenateKVCache",
        "RotatingKVCache",
        "ChunkedKVCache",
    }
)


def _cache_supports_persistence(cache_list: list) -> bool:
    """True iff every layer is a cache type known to be safe for
    cross-request persistence.

    See ``_PERSISTABLE_CACHE_CLASSES`` for the allowlist.  Models that fail
    this check still get within-request cache reuse — just not the
    cross-request strict-extension reuse the prompt cache store normally
    provides.

    Exact class-name match (no MRO walk).  With allowlist semantics an MRO
    walk would invert the safety guarantee: a future ``BadSSMCache(KVCache)``
    that has unsafe state but inherits from an allowlisted class would
    silently pass.  Subclasses must be added to the allowlist explicitly.
    A false-negative just costs a cache miss; a false-positive crashes
    mlx-lm with a Metal stream error on the next request.

    Empty ``cache_list`` returns False (no evidence of safety).  Unlike the
    trim probe — where a false-positive falls back gracefully — a stray
    True here would crash the next request.
    """
    if not cache_list:
        return False
    for layer in cache_list:
        if type(layer).__name__ not in _PERSISTABLE_CACHE_CLASSES:
            return False
    return True


class ModelLoadTimeoutError(TimeoutError):
    """Raised when model loading exceeds OLMLX_MODEL_LOAD_TIMEOUT."""


class SpectralCalibrationMissingError(FileNotFoundError):
    """Raised when SpectralQuant is configured but calibration data is absent."""


@dataclass
class CachedPromptState:
    """KV cache state from a previous generation, for prompt cache reuse."""

    tokens: list[int]  # Full sequence: prompt + generated tokens
    cache: list[Any]  # Per-layer KV cache objects (mutated in-place by generate_step)


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
        if state.cache and not _is_serializable_cache(state.cache):
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


@dataclass
class LoadedModel:
    name: str
    hf_path: str
    model: Any
    tokenizer: Any  # tokenizer (mlx-lm) or processor (mlx-vlm)
    is_vlm: bool = False
    is_distributed: bool = False
    is_flash: bool = False
    is_flash_moe: bool = False
    speculative_decoder: Any = None
    weight_store: Any = None
    template_caps: TemplateCaps = field(default_factory=TemplateCaps)
    loaded_at: float = field(default_factory=time.time)
    expires_at: float | None = None
    size_bytes: int = 0
    active_refs: int = 0
    _active_refs_lock: threading.Lock = field(
        default_factory=threading.Lock, compare=False, repr=False
    )
    prompt_cache_store: PromptCacheStore = field(default=None)  # type: ignore[assignment]
    kv_cache_quant: str | None = None
    # False for hybrid sliding-window models (RotatingKVCache layers).
    # Set by the loader; default True covers direct construction in tests.
    supports_cache_trim: bool = True
    # False for hybrid SSM-style models (ArraysCache layers, e.g. Qwen3.5,
    # Qwen3-Next).  When False, prompt cache state is not persisted across
    # requests because cross-request reuse crashes mlx-lm with a Metal
    # stream error.  Set by the loader's _probe_cache_capabilities call.
    # Issue #284.
    #
    # Defaults to False (unsafe-by-default) — unlike supports_cache_trim,
    # a false-positive here crashes the next request rather than wasting a
    # trim_prompt_cache() call.  Direct LoadedModel construction must
    # explicitly opt in if the model's cache layout is known to be safe.
    supports_cache_persistence: bool = False
    spectral_calibration_dir: Any = None  # Path | None, typed as Any to avoid import
    default_options: dict = field(default_factory=dict)
    inference_queue_timeout: float | None = None
    inference_timeout: float | None = None
    sync_mode: SyncMode | None = None

    def __post_init__(self):
        if self.prompt_cache_store is None:
            disk_path = (
                Path(settings.prompt_cache_disk_path).expanduser()
                if settings.prompt_cache_disk
                else None
            )
            disk_max_bytes = (
                int(settings.prompt_cache_disk_max_gb * 1024**3)
                if settings.prompt_cache_disk
                else None
            )
            self.prompt_cache_store = PromptCacheStore(
                max_slots=settings.prompt_cache_max_slots,
                disk_path=disk_path,
                model_name=self.name,
                disk_max_bytes=disk_max_bytes,
            )

    @property
    def is_speculative(self) -> bool:
        return self.speculative_decoder is not None

    @property
    def text_tokenizer(self) -> Any:
        """Return the underlying text tokenizer, unwrapping VLM processor if needed.

        mlx-vlm's load() returns a processor whose .tokenizer attribute is
        the actual HuggingFace tokenizer with the chat template.
        """
        tok = self.tokenizer
        if self.is_vlm and hasattr(tok, "tokenizer"):
            return tok.tokenizer
        return tok


def parse_keep_alive(value: str | int) -> float | None:
    """Parse keep_alive to seconds. Returns None for never-expire (-1)."""
    if isinstance(value, (int, float)):
        if value < 0:
            return None
        return float(value)
    value = str(value).strip()
    if value == "-1":
        return None
    if value == "0":
        return 0.0
    # Bare integer string → treat as seconds (consistent with Ollama API)
    if value.isdigit():
        return float(int(value))
    match = re.match(r"^(\d+)(s|m|h)$", value)
    if not match:
        logger.warning("Invalid keep_alive format: %r, defaulting to 5m", value)
        return 300.0  # default 5m
    num, unit = int(match.group(1)), match.group(2)
    multipliers = {"s": 1, "m": 60, "h": 3600}
    return float(num * multipliers[unit])


class ModelManager:
    """Manages loading/unloading of MLX models with LRU eviction."""

    def __init__(
        self,
        registry: ModelRegistry,
        store: ModelStore | None = None,
        distributed_group: Any = None,
        distributed_strategy: str = "tensor",
        distributed_layer_counts: list[int] | None = None,
    ):
        self.registry = registry
        self.store = store
        self._distributed_group = distributed_group
        self._distributed_strategy = distributed_strategy
        self._distributed_layer_counts = distributed_layer_counts
        self._loaded: dict[str, LoadedModel] = {}
        self._lock = asyncio.Lock()
        self._expiry_task: asyncio.Task | None = None
        self._pending_cleanups: dict[str, asyncio.Task] = {}
        self._pending_load_tasks: dict[str, asyncio.Task] = {}

    def start_expiry_checker(self):
        self._expiry_task = asyncio.create_task(self._check_expiry_loop())

    async def stop(self):
        if self._expiry_task:
            self._expiry_task.cancel()
            try:
                await self._expiry_task
            except asyncio.CancelledError:
                pass
        for task in self._pending_cleanups.values():
            task.cancel()
        if self._pending_cleanups:
            await asyncio.gather(
                *self._pending_cleanups.values(), return_exceptions=True
            )
        self._pending_cleanups.clear()
        # Drain orphaned load tasks so their exceptions are retrieved.
        # The underlying threads can't be interrupted (Python limitation)
        # so use a bounded timeout to avoid blocking shutdown indefinitely.
        if self._pending_load_tasks:
            drained = True
            try:
                await asyncio.wait_for(
                    asyncio.gather(
                        *self._pending_load_tasks.values(), return_exceptions=True
                    ),
                    timeout=5.0,
                )
            except asyncio.TimeoutError:
                drained = False
                logger.warning(
                    "Timed out waiting for %d orphaned load thread(s) to finish; "
                    "they will be abandoned on process exit",
                    len(self._pending_load_tasks),
                )
            # Only flush when all threads finished — if the drain timed out,
            # threads are still allocating and mx.clear_cache() is unsafe.
            if drained:
                gc.collect()
                mx.clear_cache()
        self._pending_load_tasks.clear()
        self._loaded.clear()

    def _resolve_keep_alive(self, keep_alive: str | None) -> float | None:
        """Parse keep_alive, falling back to the global default."""
        return parse_keep_alive(
            keep_alive if keep_alive is not None else settings.default_keep_alive
        )

    def _evict_lru_if_needed(self) -> None:
        """Evict least-recently-used models until below max_loaded_models.

        Must be called while holding self._lock.
        Raises RuntimeError if all loaded models are in active use.
        """
        while len(self._loaded) >= settings.max_loaded_models:
            evictable = {k: v for k, v in self._loaded.items() if v.active_refs == 0}
            if not evictable:
                raise RuntimeError(
                    "All loaded models are in use, cannot evict to load a new model"
                )
            oldest_name = min(evictable, key=lambda k: evictable[k].loaded_at)
            logger.info("Evicting model %s", oldest_name)
            evicted = self._loaded.pop(oldest_name)
            # Release draft model Metal memory promptly
            evicted.speculative_decoder = None
            del evicted

        # Flush Metal allocator cache so that buffers from evicted models
        # don't inflate the mem_before measurement below.  Skip when
        # any deferred cleanup is pending — a different model's
        # background thread may still be allocating Metal memory.
        if not self._pending_cleanups:
            gc.collect()
            mx.clear_cache()

    async def ensure_loaded(
        self, name: str, keep_alive: str | None = None
    ) -> LoadedModel:
        """Ensure a model is loaded and return it."""
        normalized = self.registry.normalize_name(name)
        if normalized != name:
            logger.info("Normalized model name '%s' -> '%s'", name, normalized)

        while True:
            # If a previous load timed out and the background thread is still
            # running, wait for its deferred cleanup to finish BEFORE acquiring
            # the lock — the cleanup may take minutes (waiting for a background
            # download thread), and holding the lock would block all model ops.
            cleanup = self._pending_cleanups.get(normalized)
            if cleanup is not None:
                logger.info(
                    "Waiting for deferred cleanup of '%s' before retry",
                    normalized,
                )
                try:
                    await cleanup
                except BaseException:
                    logger.warning(
                        "Deferred cleanup for '%s' failed; proceeding with retry",
                        normalized,
                        exc_info=True,
                    )

            async with self._lock:
                # A cleanup may have been scheduled while we were waiting
                # for the lock (another caller timed out).  Release the lock
                # and loop back to await it outside.
                if normalized in self._pending_cleanups:
                    continue

                if normalized in self._loaded:
                    lm = self._loaded[normalized]
                    # Refresh expiry
                    ka = self._resolve_keep_alive(keep_alive)
                    if ka is not None:
                        lm.expires_at = time.time() + ka
                    else:
                        lm.expires_at = None
                    return lm

                # Resolve before eviction — reject unknown models without
                # disturbing already-loaded models or their KV caches.
                model_config = self.registry.resolve(name)
                if model_config is None:
                    suggestions = self.registry.search(name, max_results=3)
                    msg = f"Model '{name}' not found."
                    if suggestions:
                        names = ", ".join(s[0] for s in suggestions)
                        msg += f" Did you mean: {names}?"
                    msg += (
                        f"\nAdd it to {settings.models_config} or use a HuggingFace path "
                        f"like 'mlx-community/Qwen2.5-3B-Instruct-4bit'"
                    )
                    raise ValueError(msg)

                hf_path = model_config.hf_path

                # Auto-register direct HF paths so future requests find them.
                # Skip if name contains ":" (already resolved, not a plain HF path).
                if "/" in name and ":" not in name:
                    self.registry.add_mapping(name, hf_path, model_config=model_config)

                # Strip Ollama-style tag from HF path.
                hf_path = _strip_ollama_tag(hf_path)

                # Resolve per-model experimental overrides
                model_exp = resolve_experimental(
                    global_experimental, model_config.experimental
                )
                # Resolve per-model speculative settings (now top-level, not
                # under ``experimental``). Falls back to global Settings.
                spec_config = model_config.resolved_speculative()
                kv_cache_quant = model_config.resolved_kv_cache_quant()

                self._evict_lru_if_needed()

                logger.info("Loading model %s from %s", normalized, hf_path)
                mem_before = memory_utils.get_metal_memory()

                # Initialize before try so the except handler can always
                # clean up, whether _load_model or the post-load check fails.
                model = tokenizer = None
                load_task = lm = None
                try:
                    coro = asyncio.to_thread(
                        self._load_model_and_shard, hf_path, model_exp, spec_config
                    )
                    timeout = settings.model_load_timeout
                    is_distributed = False
                    if timeout is not None:
                        load_task = asyncio.create_task(coro)
                        try:
                            (
                                model,
                                tokenizer,
                                is_vlm,
                                caps,
                                is_distributed,
                                _spec_decoder,
                            ) = await asyncio.wait_for(
                                asyncio.shield(load_task), timeout=timeout
                            )
                        except (asyncio.TimeoutError, asyncio.CancelledError) as exc:
                            # The background thread continues running — Python
                            # cannot interrupt native threads.  Schedule GPU
                            # cleanup for when it finishes to prevent Metal
                            # memory leaks from orphaned model weights.
                            # This handles both explicit timeouts AND external
                            # cancellations (e.g. client disconnect), since
                            # asyncio.shield protects load_task from being
                            # cancelled but CancelledError still propagates
                            # to the caller.
                            self._schedule_deferred_cleanup(load_task, normalized)
                            if isinstance(exc, asyncio.TimeoutError):
                                raise ModelLoadTimeoutError(
                                    f"Loading model '{normalized}' timed out after {timeout}s. "
                                    f"Increase OLMLX_MODEL_LOAD_TIMEOUT or unset it to disable."
                                )
                            raise
                    else:
                        (
                            model,
                            tokenizer,
                            is_vlm,
                            caps,
                            is_distributed,
                            _spec_decoder,
                        ) = await coro

                    # Check if the model fits safely in memory.  On Apple Silicon
                    # the GPU shares system RAM — if total Metal memory exceeds the
                    # configured fraction of system RAM, generation will almost
                    # certainly OOM and crash the process (Metal abort, not catchable
                    # in Python).  Reject early with a clear error instead.
                    #
                    # Note: this is a post-hoc check — the model is already loaded.
                    # If loading itself triggers an OOM the process will still crash.
                    # In practice, loading succeeds; it is the KV cache allocation
                    # during generation that causes the abort.
                    mem_after = memory_utils.get_metal_memory()
                    total = memory_utils.get_system_memory_bytes()
                    if total > 0 and mem_after > int(
                        total * settings.memory_limit_fraction
                    ):
                        limit = int(total * settings.memory_limit_fraction)
                        model_mb = max(0, (mem_after - mem_before)) // (1024 * 1024)
                        total_mb = total // (1024 * 1024)
                        limit_mb = limit // (1024 * 1024)
                        logger.error(
                            "Model %s uses %d MB, pushing Metal memory to %d MB "
                            "which exceeds the limit of %d MB "
                            "(%.0f%% of %d MB system RAM). Refusing to load.",
                            normalized,
                            model_mb,
                            mem_after // (1024 * 1024),
                            limit_mb,
                            settings.memory_limit_fraction * 100,
                            total_mb,
                        )
                        raise MemoryError(
                            f"Model '{normalized}' requires ~{model_mb} MB but the memory limit "
                            f"is {limit_mb} MB ({settings.memory_limit_fraction:.0%} of "
                            f"{total_mb} MB system RAM). Use a smaller/more quantized model, "
                            f"or increase OLMLX_MEMORY_LIMIT_FRACTION (current: "
                            f"{settings.memory_limit_fraction})."
                        )

                    # Memory check passed — register the model.
                    # Use per-model keep_alive as fallback when request
                    # doesn't specify one.
                    effective_keep_alive = keep_alive
                    if (
                        effective_keep_alive is None
                        and model_config.keep_alive is not None
                    ):
                        effective_keep_alive = model_config.keep_alive
                    ka = self._resolve_keep_alive(effective_keep_alive)
                    expires = time.time() + ka if ka is not None else None

                    logger.info(
                        "Model %s caps: tools=%s, thinking=%s, thinking_tags=%s",
                        normalized,
                        caps.supports_tools,
                        caps.supports_enable_thinking,
                        caps.has_thinking_tags,
                    )

                    # Detect if flash mode was used
                    is_flash = False
                    is_flash_moe = False
                    _weight_store = None
                    try:
                        from olmlx.engine.flash.flash_model import (
                            FlashModelWrapper,
                        )

                        is_flash = isinstance(model, FlashModelWrapper)
                    except ImportError:
                        pass
                    try:
                        from olmlx.engine.flash.flash_moe_model import (
                            FlashMoeModelWrapper,
                        )

                        if isinstance(model, FlashMoeModelWrapper):
                            is_flash_moe = True
                            _weight_store = getattr(model, "_weight_store", None)
                    except ImportError:
                        pass

                    lm = LoadedModel(
                        name=normalized,
                        hf_path=hf_path,
                        model=model,
                        tokenizer=tokenizer,
                        is_vlm=is_vlm,
                        is_distributed=is_distributed,
                        is_flash=is_flash,
                        is_flash_moe=is_flash_moe,
                        speculative_decoder=_spec_decoder,
                        weight_store=_weight_store,
                        template_caps=caps,
                        expires_at=expires,
                        kv_cache_quant=kv_cache_quant,
                        spectral_calibration_dir=self._find_spectral_dir(
                            hf_path, kv_cache_quant
                        ),
                        default_options=dict(model_config.options),
                        inference_queue_timeout=model_config.inference_queue_timeout,
                        inference_timeout=model_config.inference_timeout,
                        sync_mode=model_config.sync_mode,
                    )
                    self._probe_cache_capabilities(lm)
                    self._loaded[normalized] = lm
                    return lm
                except BaseException:
                    # Drop references and flush Metal allocator so the memory
                    # is actually reclaimed before we raise.  Also clean up
                    # lm if it was already constructed (exception between
                    # LoadedModel() and return), and pop from _loaded to
                    # prevent a zombie entry holding GPU memory.
                    if lm is not None:
                        self._loaded.pop(normalized, None)
                        del lm
                    # When _load_model fails, model/tokenizer are still None —
                    # only bother deleting if they hold actual GPU resources.
                    if model is not None:
                        del model, tokenizer
                    # Release load_task's stored result tuple so the model
                    # weights can actually be freed by gc.collect below.
                    if load_task is not None:
                        del load_task
                    # Skip immediate cache flush when a deferred cleanup is
                    # pending — the background thread is still running and
                    # mx.clear_cache() is not safe to call concurrently with
                    # active Metal allocations.  The deferred cleanup handles
                    # it after the thread finishes.
                    if normalized not in self._pending_cleanups:
                        gc.collect()
                        mx.clear_cache()
                    raise

    def _schedule_deferred_cleanup(
        self, load_task: asyncio.Task, model_name: str
    ) -> None:
        """Schedule GPU cleanup for when a timed-out load thread finishes.

        asyncio.to_thread() threads cannot be interrupted, so after a timeout
        the thread continues running and may allocate GPU memory.  This method
        schedules a task that waits for the thread to complete, then flushes
        the Metal allocator to reclaim the memory.  The task is tracked in
        _pending_cleanups so that a retry for the same model waits for
        cleanup to finish before starting a new load.
        """

        async def _cleanup() -> None:
            cancelled = False
            try:
                await load_task
            except asyncio.CancelledError:
                # The background thread keeps running (Python can't
                # interrupt native threads).  Don't call load_task.cancel()
                # — it can't stop the thread, and the task's result/exception
                # will be drained by stop() via _pending_load_tasks.
                cancelled = True
                raise
            except BaseException:
                pass
            finally:
                # Flush Metal allocator before removing the entry.
                # gc/clear must complete before pop — otherwise a
                # concurrent retry could start its own pre-load gc/clear
                # while ours is still running, which is unsafe for the
                # Metal allocator.  If gc/clear itself fails, still pop
                # so the model slot isn't permanently bricked.
                #
                # Skip gc/clear when cancelled — the background thread is
                # still active and mx.clear_cache() is not safe to call
                # concurrently with Metal allocations.
                try:
                    if not cancelled:
                        gc.collect()
                        mx.clear_cache()
                finally:
                    self._pending_cleanups.pop(model_name, None)
                    # Only pop load_task when not cancelled — stop()
                    # needs the reference to drain orphaned threads.
                    if not cancelled:
                        self._pending_load_tasks.pop(model_name, None)
            logger.info(
                "Deferred GPU cleanup after timeout of '%s' completed", model_name
            )

        self._pending_cleanups[model_name] = asyncio.create_task(_cleanup())
        self._pending_load_tasks[model_name] = load_task

    def get_loaded(self) -> list[LoadedModel]:
        return list(self._loaded.values())

    def invalidate_prompt_cache(self, model_name: str, cache_id: str) -> None:
        """Remove a prompt cache entry for the given model and cache_id."""
        normalized = self.registry.normalize_name(model_name)
        lm = self._loaded.get(normalized)
        if lm is not None:
            lm.prompt_cache_store.remove(cache_id)

    def unload(self, name: str) -> bool:
        """Unload a model. Returns True if unloaded, False if not loaded.

        Raises RuntimeError if model has active requests.
        """
        normalized = self.registry.normalize_name(name)
        lm = self._loaded.get(normalized)
        if lm is None:
            return False
        if lm.active_refs > 0:
            raise RuntimeError(
                f"Model '{normalized}' has {lm.active_refs} active request(s)"
            )
        lm = self._loaded.pop(normalized)
        # Close prefetcher *before* weight store: prefetcher tasks submit into
        # the weight store's pool, so weight store must outlive the prefetcher.
        if hasattr(lm.model, "prefetcher") and lm.model.prefetcher is not None:
            lm.model.prefetcher.close()
        if lm.weight_store is not None:
            lm.weight_store.close()
        # Release draft model Metal memory promptly
        lm.speculative_decoder = None
        return True

    # Config keys that indicate a vision-language model
    _VLM_CONFIG_KEYS = frozenset(
        {
            "vision_config",
            "visual",
            "vision_tower",
            "vision_model",
            "image_token_id",
            "vision_feature_layer",
            "mm_vision_tower",
            "visual_config",
            "vit_config",
        }
    )

    def _detect_model_kind(self, hf_path: str) -> str:
        """Return 'text', 'vlm', or 'unknown' by checking config.json against installed libraries."""
        config = None
        # Check local store first
        if self.store is not None:
            local_config = self.store.local_path(hf_path) / "config.json"
            if local_config.exists():
                try:
                    with open(local_config) as f:
                        config = json.load(f)
                except Exception:
                    pass
        # Fall back to HF hub
        if config is None:
            try:
                from huggingface_hub import hf_hub_download

                config_path = hf_hub_download(hf_path, "config.json")
                with open(config_path) as f:
                    config = json.load(f)
            except Exception:
                return "unknown"

        model_type = config.get("model_type", "").lower()
        if not model_type:
            return "unknown"

        # Check for vision-related config keys — these indicate a VLM regardless
        # of whether the base model_type also exists in mlx-lm
        has_vision_keys = bool(self._VLM_CONFIG_KEYS & config.keys())
        if has_vision_keys:
            # Issue #284: hybrid SSM+attention VLMs (Qwen3.5, Qwen3_5_moe)
            # crash in mlx-vlm with "There is no Stream(gpu, N) in current
            # thread" on text-only inference.  These models ship with a
            # dedicated text-only module in mlx-lm — route through it.
            # Discriminator: text_config.layer_types contains
            # "linear_attention", which also signals the gated-delta
            # ArraysCache layout that triggers the persistence bug.
            #
            # Heuristic — known limitations:
            # - Over-fires: a future VLM that uses "linear_attention" but
            #   loads correctly through mlx-vlm would be silently rerouted
            #   to the text path, losing vision.
            # - Under-fires: a new hybrid SSM model that names its layers
            #   differently (e.g. "gated_delta", "mamba") wouldn't trigger
            #   this guard and would still crash through mlx-vlm.
            # Update this matcher (or replace with a model_type allowlist)
            # as new hybrid families appear.
            text_cfg = config.get("text_config")
            if isinstance(text_cfg, dict):
                layer_types = text_cfg.get("layer_types")
                if isinstance(layer_types, list) and "linear_attention" in layer_types:
                    # Resolve the mlx-lm module name in the try/except so the
                    # ImportError/AttributeError handler only catches errors
                    # from that resolution — not from find_spec or the
                    # raise-on-no-module path below — to keep error
                    # attribution unambiguous.
                    text_model_type = text_cfg.get("model_type", model_type)
                    try:
                        from mlx_lm.utils import MODEL_REMAPPING as LM_REMAP

                        mapped_lm = LM_REMAP.get(text_model_type, text_model_type)
                    except (ImportError, AttributeError) as exc:
                        # mlx-lm absent or restructured.  Catching
                        # AttributeError too because if MODEL_REMAPPING
                        # imports as something that isn't a dict, the
                        # ``LM_REMAP.get(...)`` call would raise
                        # AttributeError uncaught — defeating the
                        # discriminator the same way an ImportError
                        # would.  Either way: raise at load time rather
                        # than defer to a known-broken loader.
                        raise ValueError(
                            f"Model '{hf_path}' (model_type '{model_type}') "
                            f"uses hybrid linear-attention layers (issue "
                            f"#284) but mlx_lm.utils.MODEL_REMAPPING is "
                            f"unavailable or not a mapping. Loading "
                            f"through mlx-vlm would crash with a Metal "
                            f"stream error."
                        ) from exc

                    # Hybrid VLMs (Qwen3.5) carry the model architecture in
                    # text_config.model_type; the top-level model_type may be
                    # VLM-specific (e.g. "qwen3_5_vl") with no mlx-lm module,
                    # while text_config.model_type ("qwen3_5") does.  We
                    # already resolved using text_model_type above.
                    if (
                        importlib.util.find_spec(f"mlx_lm.models.{mapped_lm}")
                        is not None
                    ):
                        # WARNING because vision capability is permanently
                        # lost for this model load — image inputs would
                        # produce confusing errors with no other signal.
                        # Log both the top-level VLM type and the mlx-lm
                        # text-path type that the routing decision was
                        # made on, since they typically differ for
                        # hybrid VLMs (e.g. "qwen3_5_vl" → "qwen3_5").
                        logger.warning(
                            "Routing hybrid linear-attention VLM '%s' "
                            "through mlx-lm text path '%s' (issue #284). "
                            "Vision capability is disabled for this load.",
                            model_type,
                            text_model_type,
                        )
                        return "text"
                    # mlx-lm has no module for this model_type.  The mlx-vlm
                    # fallback chain is the known-crashing path for these
                    # models; raise at detection time so the user sees a
                    # clear load-time error rather than a silent Metal
                    # stream crash mid-inference.
                    text_mt_present = "model_type" in text_cfg
                    raise ValueError(
                        f"Cannot load '{hf_path}': it has hybrid "
                        f"linear-attention layers (model_type "
                        f"'{model_type}', "
                        + (
                            f"text_config.model_type '{text_model_type}'"
                            if text_mt_present
                            else f"text_config.model_type missing — fell "
                            f"back to top-level '{text_model_type}'"
                        )
                        + f") but mlx-lm has no module named "
                        f"'mlx_lm.models.{mapped_lm}'.  Loading through "
                        f"mlx-vlm would crash with a Metal stream error "
                        f"(issue #284).  Upgrade mlx-lm to a version "
                        f"that ships the matching text-only module"
                        + (
                            ""
                            if text_mt_present
                            else " (or add 'model_type' to text_config "
                            "in the model's config.json if it points to "
                            "a different mlx-lm architecture)"
                        )
                        + ", or — if a future mlx-vlm release has fixed "
                        "this for the model — relax the discriminator "
                        "in olmlx/engine/model_manager.py "
                        "_detect_model_kind."
                    )

            # Verify mlx-vlm can handle it
            try:
                from mlx_vlm.utils import MODEL_REMAPPING as VLM_REMAP

                mapped = VLM_REMAP.get(model_type, model_type)
                spec = importlib.util.find_spec(f"mlx_vlm.models.{mapped}")
                if spec is not None:
                    return "vlm"
            except ImportError:
                pass
            # Has vision keys but mlx-vlm can't handle it — check mlx-lm
            try:
                from mlx_lm.utils import MODEL_REMAPPING as LM_REMAP

                mapped = LM_REMAP.get(model_type, model_type)
                spec = importlib.util.find_spec(f"mlx_lm.models.{mapped}")
                if spec is not None:
                    logger.info(
                        "Config has vision keys but model_type '%s' not in mlx-vlm; "
                        "mlx-lm supports it, using text",
                        model_type,
                    )
                    return "text"
            except ImportError:
                pass
            # Neither library explicitly supports it — try both via fallback
            logger.info(
                "Config has vision keys but model_type '%s' not in mlx-vlm or mlx-lm",
                model_type,
            )
            return "unknown"

        # No vision keys — check mlx-lm
        try:
            from mlx_lm.utils import MODEL_REMAPPING as LM_REMAP

            mapped = LM_REMAP.get(model_type, model_type)
            spec = importlib.util.find_spec(f"mlx_lm.models.{mapped}")
            if spec is not None:
                return "text"
        except ImportError:
            pass

        # Fallback: check mlx-vlm even without vision keys
        try:
            from mlx_vlm.utils import MODEL_REMAPPING as VLM_REMAP

            mapped = VLM_REMAP.get(model_type, model_type)
            spec = importlib.util.find_spec(f"mlx_vlm.models.{mapped}")
            if spec is not None:
                return "vlm"
        except ImportError:
            pass

        return "unknown"

    @staticmethod
    def _load_chat_template(tokenizer: Any, load_path: str, hf_path: str = "") -> None:
        """Load chat_template from file if the tokenizer doesn't have one.

        Checks local files first (chat_template.jinja, chat_template.json),
        then tries downloading from HF hub.
        """
        if getattr(tokenizer, "chat_template", None):
            return
        model_dir = Path(load_path)
        jinja = model_dir / "chat_template.jinja"
        json_file = model_dir / "chat_template.json"
        if jinja.exists():
            tokenizer.chat_template = jinja.read_text()
        elif json_file.exists():
            try:
                data = json.loads(json_file.read_text())
                tokenizer.chat_template = data["chat_template"]
            except (json.JSONDecodeError, KeyError):
                pass
        elif hf_path:
            # Try downloading from HF hub — first from the repo itself,
            # then from the base_model listed in the model card (and its -it variant).
            try:
                from huggingface_hub import hf_hub_download, model_info

                try:
                    path = hf_hub_download(hf_path, "chat_template.jinja")
                    tokenizer.chat_template = Path(path).read_text()
                    return
                except Exception as exc:
                    logger.debug(
                        "chat_template.jinja not found in %s: %s", hf_path, exc
                    )
                # Primary repo didn't have it — check base_model from model card
                try:
                    info = model_info(hf_path)
                    base = (
                        info.card_data.base_model
                        if info.card_data and hasattr(info.card_data, "base_model")
                        else None
                    )
                    if isinstance(base, list):
                        base = base[0] if base else None
                    if base and base != hf_path:
                        candidates = [base]
                        if not base.endswith("-it"):
                            candidates.append(f"{base}-it")
                        for candidate in candidates:
                            try:
                                path = hf_hub_download(candidate, "chat_template.jinja")
                                tokenizer.chat_template = Path(path).read_text()
                                return
                            except Exception as exc:
                                logger.debug(
                                    "chat_template.jinja not found in %s: %s",
                                    candidate,
                                    exc,
                                )
                                continue
                except Exception as exc:
                    logger.warning(
                        "Failed to fetch model info for %s: %s", hf_path, exc
                    )
            except ImportError:
                pass

    def _try_lm_then_vlm(
        self, load_path: str, label: str
    ) -> tuple[Any, Any, bool, TemplateCaps]:
        """Try loading with mlx-lm first, fall back to mlx-vlm on failure."""
        try:
            import mlx_lm

            model, tokenizer = _load_with_model_type_fallback(mlx_lm, load_path)
            caps = detect_caps(tokenizer)
            return model, tokenizer, False, caps
        except _FALLBACK_EXCEPTIONS as exc:
            logger.warning("mlx-lm failed for %s (%s), trying mlx-vlm", label, exc)
            import mlx_vlm

            model, processor = mlx_vlm.load(load_path)
            tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
            self._load_chat_template(tok, load_path, label)
            caps = detect_caps(tok)
            return model, processor, True, caps

    @staticmethod
    def _vlm_fallback_load(
        load_path: str, hf_path: str, *, lazy: bool
    ) -> tuple[Any, Any, Any]:
        """Load a VLM via mlx-vlm and extract the language model + tokenizer.

        Returns (language_model, tokenizer, vlm_model).
        """
        import mlx_vlm

        logger.info("mlx-lm failed for %s, falling back to mlx-vlm", hf_path)
        vlm_model, processor = mlx_vlm.load(load_path, lazy=lazy)
        model = vlm_model.language_model
        tokenizer = (
            processor.tokenizer if hasattr(processor, "tokenizer") else processor
        )
        return model, tokenizer, vlm_model

    def _flash_dir(self, hf_path: str) -> Path | None:
        """Return the flash-prepared directory for a model, if it exists."""
        if self.store is None:
            return None
        flash_path = self.store.local_path(hf_path) / "flash"
        if flash_path.exists() and (flash_path / "flash_layout.json").exists():
            return flash_path
        return None

    def _probe_cache_capabilities(self, lm: LoadedModel) -> None:
        """Probe how the model's prompt cache can be safely reused, recording
        the result on ``lm.supports_cache_trim`` and ``lm.supports_cache_persistence``.

        Hybrid sliding-window models (Gemma 4, Qwen3-Next) include
        `RotatingKVCache` layers which become non-trimmable once the
        rotating buffer fills.  Detecting this once at load time lets the
        request path skip a doomed `trim_prompt_cache()` call (which
        would silently return 0 and force a full prefill).

        Hybrid SSM-style models (Qwen3.5, Qwen3-Next gated-delta layers) use
        ``ArraysCache`` whose stored state cannot be reused across requests
        without crashing mlx-lm during the next prefill — see issue #284.
        Such models keep within-request cache reuse but skip the cross-
        request store/load path.

        Always probes the bare mlx-lm cache — trim behaviour is determined
        by the underlying layer classes, not the quantization wrapper, and
        TurboQuant/Spectral factories would otherwise pull calibration data
        off disk on every model load only to throw the result away.
        """
        try:
            from mlx_lm.models.cache import make_prompt_cache
        except ImportError:
            return  # mlx-lm prompt cache unavailable; nothing to probe

        cache_model = (
            getattr(lm.model, "language_model", lm.model) if lm.is_vlm else lm.model
        )
        probe_cache: list | None = None
        probe_succeeded = False
        try:
            probe_cache = make_prompt_cache(cache_model)
            # Stage both results before assigning so a hypothetical raise
            # in either probe doesn't leave one flag set and the other
            # falling into the except handler's defaults.  Currently both
            # probes are pure string-set lookups that can't raise, but the
            # pattern is cheap and removes the ordering dependency.
            trim_ok = _cache_supports_trim(probe_cache)
            persist_ok = _cache_supports_persistence(probe_cache)
            lm.supports_cache_trim = trim_ok
            lm.supports_cache_persistence = persist_ok
            probe_succeeded = True
        except Exception:
            # Best-effort probe; on failure assume trim works so the
            # request path's existing partial-trim fallback handles it.
            # Persistence has no equivalent fallback — store + reload of an
            # ArraysCache crashes the next request (issue #284) — so default
            # to False on probe failure.  Worst case we lose cross-request
            # cache reuse for a model that didn't need protection.
            # Logged at WARNING because the consequence — cross-request
            # cache reuse silently disabled for this model — would be hard
            # to diagnose from DEBUG output alone.
            logger.warning(
                "Cache probe raised an exception for %s; defaulting to "
                "trimmable, non-persistable. Cross-request prompt cache "
                "reuse is disabled for this model.",
                lm.name,
                exc_info=True,
            )
            lm.supports_cache_trim = True
            lm.supports_cache_persistence = False
        finally:
            if probe_cache is not None:
                del probe_cache
                gc.collect()
                mx.clear_cache()

        if not lm.supports_cache_trim:
            logger.info(
                "Model %s uses a non-trimmable hybrid cache (e.g. RotatingKVCache); "
                "prompt cache will only be reused for strict-extension turns.",
                lm.name,
            )
        # Only log the layout reason when the probe actually inspected
        # the cache.  The probe-failure path already emits its own WARNING
        # above with the real cause.  Message is generic ("hybrid
        # SSM/ArraysCache or unclassified") because an empty cache_list
        # also returns False from the persistence check — the message
        # would otherwise misattribute the disable to ArraysCache when
        # the layout had nothing to do with it.
        if not lm.supports_cache_persistence and probe_succeeded:
            logger.info(
                "Model %s uses a non-persistable cache (hybrid SSM/"
                "ArraysCache or unclassified); prompt cache will not be "
                "stored across requests (issue #284).",
                lm.name,
            )
        # One-pass cleanup of any stale pre-PR on-disk entries for this
        # model.  Gated on BOTH probe_succeeded and not-persistable so we
        # only wipe disk when we KNOW the model is non-persistable.  On a
        # probe-failure path, supports_cache_persistence is False as a
        # safe default but the model may be genuinely persistable —
        # deleting its disk cache because of a transient probe failure
        # (e.g. import error during make_prompt_cache, OOM) would
        # destroy valid cached prefixes.  Stale pre-PR files on a probe-
        # failure model are handled by the existing disk-size eviction.
        if probe_succeeded and not lm.supports_cache_persistence:
            try:
                removed = lm.prompt_cache_store.clear_disk()
                if removed:
                    logger.info(
                        "Removed %d stale on-disk prompt cache file(s) for "
                        "non-persistable model %s",
                        removed,
                        lm.name,
                    )
            except Exception:
                logger.debug(
                    "clear_disk failed for %s; non-persistable cleanup "
                    "deferred to disk-size eviction",
                    lm.name,
                    exc_info=True,
                )

    def _find_spectral_dir(
        self, hf_path: str, kv_cache_quant: str | None
    ) -> Path | None:
        """Return the spectral calibration directory if spectral quant is configured."""
        if kv_cache_quant is None or not kv_cache_quant.startswith("spectral:"):
            return None
        if self.store is None:
            return None
        spectral_path = self.store.local_path(hf_path) / "spectral"
        if spectral_path.exists() and (spectral_path / "spectral_config.json").exists():
            return spectral_path
        raise SpectralCalibrationMissingError(
            f"SpectralQuant configured ({kv_cache_quant}) but no calibration data "
            f"found at {spectral_path}. Run 'olmlx spectral prepare <model>' first."
        )

    def _resolve_draft_path(self, hf_path: str) -> str:
        """Download a draft model if needed and return the local path."""
        if self.store is not None:
            local_dir = self.store.ensure_downloaded(hf_path)
            return str(local_dir)
        return hf_path

    @staticmethod
    def _check_vocab_match(target: Any, draft: Any) -> None:
        """Raise ValueError if target and draft vocab sizes differ."""
        target_vocab = getattr(getattr(target, "args", None), "vocab_size", None)
        draft_vocab = getattr(getattr(draft, "args", None), "vocab_size", None)
        if target_vocab is None or draft_vocab is None:
            logger.warning(
                "Could not verify vocab compatibility: target_vocab=%s draft_vocab=%s",
                target_vocab,
                draft_vocab,
            )
            return
        if target_vocab != draft_vocab:
            raise ValueError(
                f"Draft model vocab_size ({draft_vocab}) does not match "
                f"target model vocab_size ({target_vocab}). "
                f"Speculative decoding requires matching vocabularies."
            )

    def _load_dflash_decoder(
        self, target_model: Any, hf_path: str, model_exp: Any
    ) -> Any:
        """Load a dflash draft model and create a DFlashDecoder."""
        from olmlx.engine.dflash.adapters import get_adapter
        from olmlx.engine.dflash.decoder import DFlashDecoder
        from olmlx.engine.dflash.draft_model import DFlashDraftModel, DraftConfig

        if not model_exp.dflash_draft_model:
            raise ValueError(
                "dflash requires dflash_draft_model to be set "
                "(OLMLX_EXPERIMENTAL_DFLASH_DRAFT_MODEL)"
            )

        logger.info("Loading dflash draft model %s", model_exp.dflash_draft_model)
        load_path = self._resolve_draft_path(model_exp.dflash_draft_model)

        config_file = Path(load_path) / "config.json"
        if not config_file.exists():
            raise FileNotFoundError(
                f"DFlash draft model config not found at {config_file}"
            )

        draft_cfg_dict = json.loads(config_file.read_text())
        _required = [
            "hidden_size",
            "num_attention_heads",
            "num_layers",
            "target_layer_ids",
            "vocab_size",
        ]
        missing = [k for k in _required if k not in draft_cfg_dict]
        if missing:
            raise ValueError(
                f"DFlash draft config at {config_file} is missing "
                f"required keys: {missing}"
            )
        target_hidden_size = getattr(
            getattr(target_model, "args", None), "hidden_size", None
        )
        draft_config = DraftConfig(
            hidden_size=draft_cfg_dict["hidden_size"],
            num_attention_heads=draft_cfg_dict["num_attention_heads"],
            num_layers=draft_cfg_dict["num_layers"],
            target_layer_ids=draft_cfg_dict["target_layer_ids"],
            vocab_size=draft_cfg_dict["vocab_size"],
            target_hidden_size=target_hidden_size or draft_cfg_dict.get("hidden_size"),
        )

        draft_model = DFlashDraftModel(draft_config)
        draft_dir = Path(load_path)
        weight_files = sorted(draft_dir.glob("model*.safetensors"))
        if not weight_files:
            raise FileNotFoundError(
                f"DFlash draft model weights not found in {draft_dir}. "
                "A pre-trained dflash draft model is required."
            )
        for wf in weight_files:
            draft_model.load_weights(str(wf), strict=False)
        logger.info(
            "Loaded dflash draft weights from %s (%d file(s))",
            draft_dir,
            len(weight_files),
        )

        # Detect model type for adapter selection
        target_config_file = None
        if self.store is not None:
            target_local = self.store.local_path(hf_path)
            target_config_file = target_local / "config.json"
        if target_config_file is not None and target_config_file.exists():
            target_cfg = json.loads(target_config_file.read_text())
            model_type = target_cfg.get("model_type", "")
        else:
            model_type = draft_cfg_dict.get("target_model_type", "")

        if not model_type:
            raise ValueError(
                "Cannot determine target model_type for dflash adapter selection. "
                "Set 'target_model_type' in the draft model's config.json or "
                "ensure the target model's config.json contains 'model_type'."
            )

        adapter = get_adapter(model_type)

        return DFlashDecoder(
            target_model=target_model,
            draft_model=draft_model,
            adapter=adapter,
            draft_config=draft_config,
            block_size=model_exp.dflash_block_size,
        )

    def _load_speculative_decoder(
        self,
        target_model: Any,
        hf_path: str,
        spec_config: SpeculativeConfig,
        *,
        is_vlm: bool = False,
    ) -> Any:
        """Load a draft model and create a SpeculativeDecoder.

        For VLM targets (``is_vlm=True``), the decoder runs on the unwrapped
        language model (``target_model.language_model``) so the draft only
        needs to match the text decoder's vocabulary and the speculative loop
        can call the language model directly with token inputs and a KV cache.
        """
        from olmlx.engine.speculative import SpeculativeDecoder

        # Hard guard rather than assert — assert is elided under
        # ``python -O``, and this invariant must hold in production too.
        if not spec_config.enabled:
            raise RuntimeError(
                "_load_speculative_decoder called with spec_config.enabled=False"
            )
        draft_model_path = spec_config.draft_model
        num_tokens = spec_config.num_tokens
        if not draft_model_path:
            raise ValueError(
                "speculative requires speculative_draft_model to be set "
                "(OLMLX_SPECULATIVE_DRAFT_MODEL or per-model "
                "'speculative_draft_model' in models.json)"
            )

        logger.info(
            "Loading draft model %s for speculative decoding",
            draft_model_path,
        )
        load_path = self._resolve_draft_path(draft_model_path)

        import mlx_lm

        draft_model, _draft_tokenizer = _load_with_model_type_fallback(
            mlx_lm, load_path, lazy=False
        )

        if is_vlm:
            spec_target = getattr(target_model, "language_model", None)
            if spec_target is None:
                raise ValueError(
                    "VLM model does not expose .language_model; speculative "
                    "decoding requires direct access to the text decoder"
                )
        else:
            spec_target = target_model
        self._check_vocab_match(spec_target, draft_model)

        return SpeculativeDecoder(
            draft_model=draft_model,
            target_model=spec_target,
            num_speculative_tokens=num_tokens,
        )

    def _is_flash_enabled(self, model_exp: Any) -> bool:
        return model_exp.flash

    def _load_flash_model(
        self, hf_path: str, load_path: str, flash_dir: Path, *, model_exp: Any
    ) -> tuple[Any, Any, bool, TemplateCaps, Any]:
        """Load a model in flash mode (LLM in a Flash).

        1. Load model normally via mlx-lm
        2. Create FlashWeightStore, PredictorBank, WindowManager
        3. Wrap model with FlashModelWrapper (replaces FFN layers)
        """
        from olmlx.engine.flash.flash_model import FlashConfig, FlashModelWrapper
        from olmlx.engine.flash.predictor import LookaheadBank, PredictorBank
        from olmlx.engine.flash.weight_store import FlashWeightStore

        logger.info("Loading model %s in flash mode from %s", hf_path, flash_dir)

        # Load the full model first (we'll free FFN weights after wrapping).
        # VL models (e.g. Qwen3.5) have vision tower weights in safetensors
        # that the text-only model class doesn't use — retry with strict=False.
        from olmlx.engine.flash.prepare import load_model_with_strict_fallback

        is_vlm = False
        try:
            model, tokenizer = load_model_with_strict_fallback(load_path, lazy=False)
        except _FALLBACK_EXCEPTIONS:
            # lazy=True: avoid materializing vision encoder weights that are
            # discarded after extracting language_model; Flash loads from SSD.
            model, tokenizer, _ = self._vlm_fallback_load(load_path, hf_path, lazy=True)
            is_vlm = True
        caps = detect_caps(tokenizer)

        # Load predictor bank
        predictor_path = flash_dir / "predictors"
        predictor_bank = PredictorBank.load(predictor_path)

        # Read flash layout for dimensions
        layout_config = json.loads((flash_dir / "flash_layout.json").read_text())

        flash_config = FlashConfig(
            hidden_size=layout_config["hidden_size"],
            intermediate_size=layout_config["intermediate_size"],
            num_layers=layout_config["num_layers"],
            sparsity_threshold=model_exp.flash_sparsity_threshold,
            min_active_neurons=model_exp.flash_min_active_neurons,
            max_active_neurons=model_exp.flash_max_active_neurons,
            window_size=model_exp.flash_window_size,
            io_threads=model_exp.flash_io_threads,
            cache_budget_neurons=model_exp.flash_cache_budget_neurons,
            memory_budget_fraction=model_exp.flash_memory_budget_fraction,
            prefetch=model_exp.flash_prefetch,
            prefetch_confidence_threshold=model_exp.flash_prefetch_confidence_threshold,
            prefetch_min_neurons=model_exp.flash_prefetch_min_neurons,
            prefetch_max_neurons=model_exp.flash_prefetch_max_neurons,
            prefetch_io_threads=model_exp.flash_prefetch_io_threads,
        )

        weight_store = FlashWeightStore(
            flash_dir,
            num_io_threads=flash_config.io_threads,
            cache_budget_neurons=flash_config.cache_budget_neurons,
            bypass_cache=model_exp.flash_bypass_os_cache,
            use_preallocated_buffer=model_exp.flash_preallocated_buffer,
        )

        # Load lookahead predictors if available (for speculative prefetching)
        lookahead_bank = None
        lookahead_path = flash_dir / "lookahead_predictors"
        if model_exp.flash_prefetch and lookahead_path.exists():
            try:
                lookahead_bank = LookaheadBank.load(lookahead_path)
                logger.info("Loaded lookahead predictor bank from %s", lookahead_path)
            except Exception:
                logger.warning(
                    "Failed to load lookahead predictors, falling back to sparsity predictor",
                    exc_info=True,
                )

        # Wrap model — this replaces FFN layers and frees original weights
        wrapped = FlashModelWrapper(
            model, predictor_bank, weight_store, flash_config, lookahead_bank
        )

        if model_exp.flash_speculative:
            from olmlx.engine.flash.speculative import SpeculativeFlashDecoder

            if not model_exp.flash_speculative_draft_model:
                raise ValueError(
                    "flash_speculative requires flash_speculative_draft_model to be set "
                    "(OLMLX_EXPERIMENTAL_FLASH_SPECULATIVE_DRAFT_MODEL)"
                )

            logger.info(
                "Loading draft model %s for speculative decoding",
                model_exp.flash_speculative_draft_model,
            )
            draft_model, draft_tokenizer = load_model_with_strict_fallback(
                model_exp.flash_speculative_draft_model, lazy=False
            )

            # Verify vocabulary compatibility — a mismatch causes silent token ID errors
            target_vocab = getattr(getattr(wrapped, "args", None), "vocab_size", None)
            draft_vocab = getattr(
                getattr(draft_model, "args", None), "vocab_size", None
            )
            if (
                target_vocab is not None
                and draft_vocab is not None
                and target_vocab != draft_vocab
            ):
                raise ValueError(
                    f"Draft model vocab_size ({draft_vocab}) does not match "
                    f"target model vocab_size ({target_vocab}). "
                    f"Speculative decoding requires matching vocabularies."
                )

            decoder = SpeculativeFlashDecoder(
                draft_model=draft_model,
                target_model=wrapped,
                num_speculative_tokens=model_exp.flash_speculative_tokens,
                prefetcher=wrapped.prefetcher,
            )
            return wrapped, tokenizer, is_vlm, caps, decoder

        return wrapped, tokenizer, is_vlm, caps, None

    def _flash_moe_dir(self, hf_path: str) -> Path | None:
        """Return the flash-MoE directory for a model, if it exists."""
        if self.store is None:
            return None
        flash_moe_path = self.store.local_path(hf_path) / "flash_moe"
        if (
            flash_moe_path.exists()
            and (flash_moe_path / "flash_moe_layout.json").exists()
        ):
            return flash_moe_path
        return None

    def _is_flash_moe_enabled(self, model_exp: Any) -> bool:
        return model_exp.flash_moe

    def _load_flash_moe_model(
        self,
        hf_path: str,
        load_path: str,
        flash_moe_dir: Path,
        *,
        model_exp: Any,
    ) -> tuple[Any, Any, bool, TemplateCaps]:
        """Load a model in Flash-MoE mode.

        1. Load model with lazy=True to avoid materializing expert weights
        2. Create FlashMoeWeightStore
        3. Wrap model with FlashMoeModelWrapper (replaces SwitchGLU)
        4. Eval only non-expert params
        """
        from olmlx.engine.flash.flash_moe_model import FlashMoeModelWrapper
        from olmlx.engine.flash.moe_weight_store import FlashMoeWeightStore

        import mlx_lm

        logger.info(
            "Loading model %s in Flash-MoE mode from %s", hf_path, flash_moe_dir
        )

        is_vlm = False
        try:
            model, tokenizer = _load_with_model_type_fallback(
                mlx_lm, load_path, lazy=True
            )
        except _FALLBACK_EXCEPTIONS:
            model, tokenizer, _ = self._vlm_fallback_load(load_path, hf_path, lazy=True)
            is_vlm = True
        caps = detect_caps(tokenizer)

        # Read flash_moe_config for architecture info
        moe_config = json.loads((flash_moe_dir / "flash_moe_config.json").read_text())

        store = FlashMoeWeightStore(
            flash_moe_dir,
            num_io_threads=model_exp.flash_moe_io_threads,
            cache_budget_experts=model_exp.flash_moe_cache_budget_experts,
        )

        try:
            wrapped = FlashMoeModelWrapper(
                model,
                store,
                moe_layer_indices=moe_config["moe_layer_indices"],
                hidden_size=moe_config["hidden_size"],
                intermediate_size=moe_config["intermediate_size"],
                num_experts=moe_config["num_experts"],
                num_experts_per_tok=moe_config["num_experts_per_tok"],
            )
            # Materialize only non-expert weights
            mx.eval(wrapped.parameters())
        except Exception:
            store.close()
            raise

        return wrapped, tokenizer, is_vlm, caps

    def _load_model(
        self,
        hf_path: str,
        *,
        model_exp: Any = None,
        spec_config: SpeculativeConfig | None = None,
    ) -> tuple[Any, Any, bool, TemplateCaps, Any]:
        """Load a model, using config.json inspection to choose the right library.

        *model_exp* is the resolved ExperimentalSettings for this model.
        Falls back to global defaults if not provided.

        *spec_config* is the resolved ``SpeculativeConfig`` (per-model
        overrides applied). Falls back to global ``Settings`` values
        when ``None``.

        Returns (model, tokenizer, is_vlm, caps, speculative_decoder).
        """
        if model_exp is None:
            from olmlx.config import experimental

            model_exp = experimental
        if spec_config is None:
            # Delegate to the canonical resolution path so this fallback
            # can't drift from ``ModelConfig.resolved_speculative()``.
            # ``ModelConfig`` requires an ``hf_path`` but the value is
            # not consulted by ``resolved_speculative`` — it reads only
            # from ``settings`` for an unconfigured ModelConfig.
            from olmlx.engine.registry import ModelConfig

            spec_config = ModelConfig(hf_path=hf_path).resolved_speculative()
        spec_enabled = spec_config.enabled

        # Ensure model is downloaded to the store
        load_path: str = hf_path
        if self.store is not None:
            local_dir = self.store.ensure_downloaded(hf_path)
            load_path = str(local_dir)

        # Check for flash-MoE-prepared model
        if self._is_flash_moe_enabled(model_exp):
            flash_moe_dir = self._flash_moe_dir(hf_path)
            if flash_moe_dir is not None:
                if spec_enabled:
                    logger.warning(
                        "OLMLX_SPECULATIVE is enabled but %s is loaded via "
                        "Flash-MoE, which does not support standalone "
                        "speculative decoding; the setting will be ignored.",
                        hf_path,
                    )
                return (
                    *self._load_flash_moe_model(
                        hf_path, load_path, flash_moe_dir, model_exp=model_exp
                    ),
                    None,
                )

        # Check for flash-prepared model
        if self._is_flash_enabled(model_exp):
            flash_dir = self._flash_dir(hf_path)
            if flash_dir is not None:
                if spec_enabled:
                    logger.warning(
                        "OLMLX_SPECULATIVE is enabled but %s is loaded via "
                        "Flash, which uses OLMLX_EXPERIMENTAL_FLASH_SPECULATIVE "
                        "for speculative decoding; the OLMLX_SPECULATIVE "
                        "setting will be ignored.",
                        hf_path,
                    )
                return self._load_flash_model(
                    hf_path, load_path, flash_dir, model_exp=model_exp
                )

        kind = self._detect_model_kind(hf_path)
        logger.info("Detected model kind for %s: %s", hf_path, kind)

        if kind == "vlm":
            # VLM detected — load with mlx-vlm directly
            try:
                import mlx_vlm

                model, processor = mlx_vlm.load(load_path)
                tok = (
                    processor.tokenizer
                    if hasattr(processor, "tokenizer")
                    else processor
                )
                self._load_chat_template(tok, load_path, hf_path)
                caps = detect_caps(tok)
                if model_exp.dflash:
                    raise ValueError(
                        "dflash is not supported on VLM targets; "
                        "remove dflash from models.json or unset "
                        "OLMLX_EXPERIMENTAL_DFLASH"
                    )
                if model_exp.flash_speculative:
                    raise ValueError(
                        "flash_speculative is not supported on VLM targets; "
                        "remove flash_speculative from models.json (or unset "
                        "OLMLX_EXPERIMENTAL_FLASH_SPECULATIVE) and use "
                        "speculative instead"
                    )
                if spec_enabled:
                    decoder = self._load_speculative_decoder(
                        model, hf_path, spec_config, is_vlm=True
                    )
                    return model, processor, True, caps, decoder
                return model, processor, True, caps, None
            except OSError as exc:
                # AutoProcessor may fail (e.g. missing preprocessor_config.json
                # in quantized repos). Fall through to lm-then-vlm fallback.
                logger.warning(
                    "mlx-vlm.load failed for %s (%s), trying fallback", hf_path, exc
                )

        # Text or unknown — try mlx-lm first, fall back to mlx-vlm
        model, tokenizer, is_vlm, caps = self._try_lm_then_vlm(load_path, hf_path)

        enabled = [
            name
            for name, flag in [
                ("dflash", model_exp.dflash),
                ("speculative", spec_enabled),
                ("flash_speculative", model_exp.flash_speculative),
            ]
            if flag
        ]
        if len(enabled) > 1:
            raise ValueError(
                f"Only one speculative decoder can be active at a time; got: {enabled}"
            )

        if model_exp.dflash:
            decoder = self._load_dflash_decoder(model, hf_path, model_exp)
            return model, tokenizer, is_vlm, caps, decoder

        if spec_enabled:
            decoder = self._load_speculative_decoder(model, hf_path, spec_config)
            return model, tokenizer, is_vlm, caps, decoder

        return model, tokenizer, is_vlm, caps, None

    def _load_model_and_shard(
        self,
        hf_path: str,
        model_exp: Any = None,
        spec_config: SpeculativeConfig | None = None,
    ) -> tuple[Any, Any, bool, TemplateCaps, bool, Any]:
        """Load a model and optionally shard it for distributed inference.

        *model_exp* is the resolved ExperimentalSettings for this model
        (global defaults merged with per-model overrides).

        *spec_config* is the resolved speculative config tuple
        ``(enabled, draft_model, num_tokens)``.

        Returns (model, tokenizer, is_vlm, caps, is_distributed, speculative_decoder).
        """
        model, tokenizer, is_vlm, caps, speculative_decoder = self._load_model(
            hf_path, model_exp=model_exp, spec_config=spec_config
        )
        is_distributed = False

        if self._distributed_group is not None:
            if is_vlm:
                del model, tokenizer
                gc.collect()
                mx.clear_cache()
                raise ValueError(
                    f"VLM models are not supported in distributed mode. "
                    f"Model {hf_path} is a vision-language model which cannot "
                    f"be sharded correctly across workers."
                )

            if self._distributed_strategy == "pipeline":
                from olmlx.engine.pipeline import apply_pipeline

                try:
                    apply_pipeline(
                        model,
                        self._distributed_group,
                        layer_counts=self._distributed_layer_counts,
                    )
                except ValueError:
                    del model, tokenizer
                    gc.collect()
                    mx.clear_cache()
                    raise
                # Materialize parameters — pipeline nullified unowned layers
                # so only owned weights are evaluated (no cross-rank ops).
                mx.eval(model.parameters())
                is_distributed = True
                logger.info(
                    "Model %s pipeline-sharded for distributed inference", hf_path
                )
            elif self._distributed_strategy == "tensor":
                if not hasattr(model, "shard"):
                    del model, tokenizer
                    gc.collect()
                    mx.clear_cache()
                    raise ValueError(
                        f"Model {hf_path} does not support distributed inference "
                        f"(no shard() method). Supported architectures include: "
                        f"llama, qwen2, qwen3, deepseek_v2, deepseek_v3, etc."
                    )
                model.shard(self._distributed_group)
                # Materialize all lazy weight slices BEFORE any forward pass.
                # model.shard() creates lazy array slices; if they're first
                # evaluated during inference (with all_sum), the combined Metal
                # command buffer can exceed the ~10s GPU timeout for large
                # models (32B+).  Evaluating here is purely local (no all_sum).
                mx.eval(model.parameters())
                is_distributed = True
                logger.info("Model %s sharded for distributed inference", hf_path)
            else:
                del model, tokenizer
                gc.collect()
                mx.clear_cache()
                raise ValueError(
                    f"Unknown distributed strategy {self._distributed_strategy!r}. "
                    f"Supported: 'tensor', 'pipeline'."
                )

        return model, tokenizer, is_vlm, caps, is_distributed, speculative_decoder

    async def _expire_stale(self):
        """Unload models whose keep-alive has expired (active_refs == 0)."""
        now = time.time()
        async with self._lock:
            # Models with active_refs > 0 are skipped — they are currently
            # serving requests.  Even if a model slips through with
            # active_refs == 0 between ensure_loaded() and _inference_ref(),
            # the caller still holds a Python reference, so the model/
            # tokenizer stay alive; only the _loaded dict entry is removed.
            expired = [
                name
                for name, lm in self._loaded.items()
                if lm.expires_at is not None
                and lm.expires_at <= now
                and lm.active_refs == 0
            ]
            for name in expired:
                logger.info("Unloading expired model %s", name)
                del self._loaded[name]

    async def _check_expiry_loop(self):
        while True:
            await asyncio.sleep(30)
            await self._expire_stale()
