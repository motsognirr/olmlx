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

from olmlx.config import FlashMoeConfig, SyncMode, experimental as global_experimental
from olmlx.config import resolve_experimental, settings
from olmlx.engine.registry import ModelRegistry, ResolvedFlashConfig, SpeculativeConfig
from olmlx.utils import memory as memory_utils
from olmlx.engine.template_caps import TemplateCaps, detect_caps

if TYPE_CHECKING:
    from olmlx.models.store import ModelStore

from olmlx.models.store import _dir_size, _strip_ollama_tag

logger = logging.getLogger(__name__)

#: Default max tokens collected per head during SpectralQuant calibration.
#: Duplicated from spectralquant_calibrate (which imports numpy/mlx_lm
#: eagerly) to avoid pulling those imports into lightweight CLI paths.
_SPECTRAL_DEFAULT_MAX_TOKENS_PER_HEAD = 8192
_SPECTRAL_DEFAULT_NUM_SAMPLES = 256


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


def _ensure_tokenizer_eos_in_stops(tokenizer: Any) -> None:
    """Add the tokenizer's own ``eos_token_id`` to its stop-token set.

    Workaround for repos (e.g. ``mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit``,
    issue #308) whose ``config.json`` declares ``eos_token_id`` as a token
    different from the chat template's real end-of-turn token (the
    ``eos_token`` field in ``tokenizer_config.json``). mlx-lm's ``load()``
    feeds only the config.json value into ``TokenizerWrapper.eos_token_ids``,
    so generation does not stop at the template's actual EOT and the EOT
    string leaks into the decoded response.
    """
    # ``add_eos_token`` is the TokenizerWrapper marker — keep the gate on it
    # so plain HF tokenizers and mlx-vlm processors are skipped — but mutate
    # the ``eos_token_ids`` set directly to bypass the wrapper's stringly-typed
    # API (``add_eos_token(str)`` does ``int(token)`` first then a vocab
    # lookup; we already have the integer so set-mutation is unambiguous).
    if not callable(getattr(tokenizer, "add_eos_token", None)):
        return
    stops = getattr(tokenizer, "eos_token_ids", None)
    if not isinstance(stops, set):
        # Symmetric with the missing-_tokenizer DEBUG log below: an mlx-lm
        # change from set to list/frozenset/dict for eos_token_ids would
        # otherwise silently disable the workaround.
        logger.debug(
            "TokenizerWrapper.eos_token_ids is %s (not set) on %s; eos "
            "stop-set augmentation skipped (issue #308).",
            type(stops).__name__,
            type(tokenizer).__name__,
        )
        return
    inner_tok = getattr(tokenizer, "_tokenizer", None)
    if inner_tok is None:
        # Past the ``add_eos_token`` gate but no ``_tokenizer`` attribute —
        # mlx-lm likely renamed the field. Log at debug so the regression is
        # discoverable in DEBUG-enabled environments without spamming the
        # warning channel for a possibly-deliberate variant.
        logger.debug(
            "TokenizerWrapper._tokenizer not accessible on %s; eos stop-set "
            "augmentation skipped (issue #308).",
            type(tokenizer).__name__,
        )
        return
    inner_eos = getattr(inner_tok, "eos_token_id", None)
    if isinstance(inner_eos, list):
        # Stock HF tokenizers always surface a single int here; defensive
        # against custom trust_remote_code=True tokenizers that override
        # ``eos_token_id`` to return list[int].
        stops.update(t for t in inner_eos if isinstance(t, int))
        return
    if inner_eos is None:
        # Tokenizer has no EOS configured — legitimate for some HF tokenizers
        # (e.g. base/non-instruction-tuned variants). Silent no-op.
        return
    if not isinstance(inner_eos, int):
        # mlx-lm renamed ``_tokenizer`` or the inner HF tokenizer surfaces an
        # unexpected type. ``warning``, not ``debug``: this branch indicates
        # the #308 workaround has silently regressed because mlx-lm changed
        # its internals, and we recover by no-op'ing — operators need to see
        # the signal in default logging configs, not only under DEBUG.
        logger.warning(
            "Inner eos_token_id has unexpected type %s on %s; skipping "
            "eos stop-set augmentation (issue #308 workaround may have "
            "regressed against mlx-lm internals).",
            type(inner_eos).__name__,
            type(tokenizer).__name__,
        )
        return
    if inner_eos in stops:
        return
    stops.add(inner_eos)


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
        model, tokenizer = mlx_lm.load(str(load_path), **kwargs)
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
    # Outside try/except: an AttributeError from the EOS helper must not
    # trigger the model-type remapping fallback with a misleading error.
    _ensure_tokenizer_eos_in_stops(tokenizer)
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

    Pure check against ``_TRIMMABLE_CACHE_CLASSES``; no knowledge of the
    persistence flag.  Used at model load time to decide whether
    ``trim_prompt_cache()`` is worth attempting on this model.  Hybrid
    sliding-window models (Gemma 4, Qwen3-Next, etc.) include
    ``RotatingKVCache`` layers and return False.  The downstream
    consequence — non-trimmable layouts also lose cross-request
    persistence (#343) — is applied by ``_probe_cache_capabilities``,
    not here.
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
#   in place via mx.assign at modular offsets.  Non-trimmable, so
#   ``_probe_cache_capabilities`` folds this to effective persist=False
#   (issue #343) — listed here only because the bare layout itself is
#   Metal-safe to store.
# - ChunkedKVCache: chunked layout (afm7); same semantics as KVCache
#   bounded by chunk_size.  Also non-trimmable, same #343 fold applies.
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


class SpectralCalibrationMissingError(Exception):
    """Raised when SpectralQuant is configured but calibration data is absent."""


class ActiveRequestsError(RuntimeError):
    """Raised by ``ModelManager.unload`` when a model has in-flight requests.

    Subclasses ``RuntimeError`` so legacy ``except RuntimeError:`` keeps
    working, but the dedicated type lets HTTP routers narrow the 409 path
    to exactly this condition. Without it, an unrelated ``RuntimeError``
    from ``_close_loaded_model`` would be misreported as 409.
    """


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
    is_whisper: bool = False
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
    #: Weight quantization string (e.g. "hqq:4") applied at load time.
    #: ``None`` means no weight quantization was applied.
    weight_quant: str | None = None
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


def _resolve_attention_causal(dflash_cfg: dict) -> bool:
    """Detect legacy draft checkpoints that were trained with causal attention.

    DFlash draft attention switched from causal to bidirectional (mask=None)
    in v2. Checkpoints carrying ``dflash_attention_version`` >= 2 use
    bidirectional; version 1 or missing defaults to causal with a warning
    so operators know to re-train.
    """
    version = dflash_cfg.get("dflash_attention_version", 2)
    # Accept int, float, and string: JSON doesn't distinguish ``2``
    # from ``2.0`` at the wire level, and a hand-edited config might
    # store ``\"2\"``.  Convert to int so fractional values like
    # ``1.5`` are treated as v1 rather than silently misclassified —
    # version bumps are integers; fractional values are misconfigs.
    # Default to 2 (bidirectional) when the key is absent — matches
    # both z-lab pre-trained drafts and current training output.
    try:
        version_int = int(float(version))
    except (TypeError, ValueError):
        version_int = 2
    if version_int >= 2:
        return False
    logger.warning(
        "DFlash draft checkpoint was trained with causal attention "
        "(dflash_attention_version=%r → %d < 2). Re-training with the "
        "current code is recommended — running an old checkpoint "
        "produces a distribution mismatch that degrades acceptance rate.",
        version,
        version_int,
    )
    return True


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
        self._flush_lock = asyncio.Lock()
        self._flush_thread_lock = threading.Lock()

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
                await self._flush_metal()
        self._pending_load_tasks.clear()
        self._loaded.clear()

    def _resolve_keep_alive(self, keep_alive: int | str | None) -> float | None:
        """Parse keep_alive, falling back to the global default."""
        return parse_keep_alive(
            keep_alive if keep_alive is not None else settings.default_keep_alive
        )

    @staticmethod
    def _close_loaded_model(lm: "LoadedModel") -> None:
        """Release resources held by a LoadedModel.

        Used by every lifecycle exit (explicit unload, LRU eviction, keep-alive
        expiry). Without this, Flash models leak ThreadPoolExecutor workers and
        per-layer file descriptors on eviction/expiry — issue #178.

        Order matters: prefetcher tasks submit into the weight store's pool, so
        the prefetcher must shut down before the weight store. Closing the
        speculative decoder (not just dropping the reference) is required when
        it owns a ``GDNStateCapture`` for a hybrid linear-attention
        target/draft: the class-level monkey-patch holds a strong reference to
        the capture instance via its closure, so ``__del__`` never fires and
        the patch lock stays held until ``close()`` is called explicitly.

        Each close() is attempted independently and logged at point of
        failure. On any error, always raises an ``ExceptionGroup`` carrying
        every failure — even when only one resource failed — so callers see
        a stable exception type. Without this, code writing
        ``except RuntimeError:`` around the single-failure case would
        silently miss a two-failure case. Use ``except*`` (PEP 654) to
        handle individual member types.

        Python's nested-try/finally would otherwise silently discard
        earlier exceptions when a later finally also raised.
        """
        errors: list[BaseException] = []
        if getattr(lm.model, "prefetcher", None) is not None:
            try:
                lm.model.prefetcher.close()
            except Exception as exc:
                logger.exception("Error closing prefetcher for %s", lm.name)
                errors.append(exc)
        if lm.weight_store is not None:
            try:
                lm.weight_store.close()
                # Null only on success — if close() raised, the store may
                # be partially closed (executor mid-drain). Preserving the
                # reference lets a future cleanup path (or test) inspect
                # the failed object instead of orphaning it. Same reasoning
                # for speculative_decoder below.
                lm.weight_store = None
            except Exception as exc:
                logger.exception("Error closing weight store for %s", lm.name)
                errors.append(exc)
        if lm.speculative_decoder is not None:
            try:
                lm.speculative_decoder.close()
                lm.speculative_decoder = None
            except Exception as exc:
                logger.exception("Error closing speculative decoder for %s", lm.name)
                errors.append(exc)
        # Free GPU memory held by prompt caches.  CachedPromptState entries
        # hold per-layer KV cache buffers (deepest GPU consumer besides
        # weights).  Uses clear() which also deletes on-disk entries for this
        # model — correct for a full eviction since any previously-offloaded
        # caches are permanently stale.
        #
        # Model and tokenizer are NOT nulled here — _close_evictees handles
        # that on the event loop after the worker thread joins, so concurrent
        # inference callers holding a LoadedModel reference are not exposed to
        # the null mid-request.
        if lm.prompt_cache_store is not None:
            try:
                lm.prompt_cache_store.clear()
                # Null on success (consistent with weight_store /
                # speculative_decoder) so re-entry from the
                # _close_evictees drain skips a redundant clear().
                lm.prompt_cache_store = None
            except Exception as exc:
                logger.exception("Error clearing prompt cache for %s", lm.name)
                errors.append(exc)
        # ``lm.model.prefetcher`` is intentionally not nulled — it lives on
        # the FlashModelWrapper, not on the LM bookkeeping, and the wrapper
        # goes away with the LM.
        # Drop xgrammar compile cache entries keyed on this tokenizer's id
        # while the tokenizer is still alive. After this returns,
        # ``_close_evictees`` nulls the LoadedModel's tokenizer reference
        # and CPython may recycle the address for a future tokenizer;
        # stale id-keyed entries would then return a CompiledGrammar built
        # for the wrong vocab. Must use ``lm.text_tokenizer`` (the HF
        # tokenizer, post-VLM-unwrap) to match what
        # ``_install_grammar_processor`` keyed the cache on.
        #
        # Whisper models (issue #366) return ``tokenizer=None``; guard the
        # grammar drop so it doesn't choke on the missing tokenizer.
        if not lm.is_whisper:
            try:
                from olmlx.engine import grammar as _grammar

                _grammar.drop_for_tokenizer(lm.text_tokenizer)
            except Exception as exc:
                logger.exception("Error dropping grammar cache for %s", lm.name)
                errors.append(exc)
        # Whisper models (issue #366) are injected into mlx_whisper's
        # module-level ModelHolder by generate_transcription so transcribe()
        # reuses the managed weights. ModelManager owns that lifetime, so on
        # close we must drop the holder's strong reference — otherwise the
        # float16 GPU weights survive eviction/expiry until the next
        # transcription overwrites them. Guard on identity so closing one
        # whisper model doesn't clear a different one still referenced.
        if lm.is_whisper:
            try:
                import importlib

                # NOTE: import the submodule via importlib — mlx_whisper's
                # __init__ rebinds the ``transcribe`` attribute to the
                # function, shadowing the submodule for ``import ... as``.
                whisper_transcribe = importlib.import_module("mlx_whisper.transcribe")
                holder = whisper_transcribe.ModelHolder
                if holder.model is lm.model:
                    holder.model = None
                    holder.model_path = None
            except Exception as exc:
                logger.exception("Error clearing whisper ModelHolder for %s", lm.name)
                errors.append(exc)
        if errors:
            raise ExceptionGroup(f"Errors closing resources for {lm.name}", errors)

    @staticmethod
    def _flush_metal_impl() -> None:
        gc.collect()
        mx.synchronize()
        mx.clear_cache()

    def _flush_metal_sync(self) -> None:
        """Thread-safe wrapper for :meth:`_flush_metal_impl`.

        Acquires ``_flush_thread_lock`` so that a thread-pool caller
        (``_load_model_and_shard``) and an event-loop caller
        (``_flush_metal`` via ``asyncio.to_thread``) cannot run
        ``mx.clear_cache()`` concurrently — the MLX allocator is not
        safe under concurrent clear from separate threads."""

        with self._flush_thread_lock:
            self._flush_metal_impl()

    async def _flush_metal(self) -> None:
        """Async wrapper for :meth:`_flush_metal_sync`.

        Serialises via ``_flush_lock`` so that concurrent callers from the
        event loop (e.g. eviction flush + deferred cleanup flush arriving
        together) do not all drain the Metal queue in parallel.  Offloads
        the actual drain to a worker thread so the event loop stays
        responsive while the GPU command queue drains.
        """
        async with self._flush_lock:
            await asyncio.to_thread(self._flush_metal_sync)

    def _pop_lru_evictees(self) -> list[LoadedModel]:
        """Pop LRU evictees from ``_loaded`` until below max_loaded_models.

        Must be called while holding ``self._lock``. Returns the popped
        models for the caller to close via :meth:`_close_evictees` after
        the lock is released. Raises ``RuntimeError`` if all loaded
        models are in active use.

        The pop / close split exists because ``_close_loaded_model`` calls
        ``executor.shutdown(wait=True)`` on the prefetcher (16 threads)
        and weight store (32 threads). Doing that join while holding
        ``self._lock`` stalls every other ``ensure_loaded`` caller —
        including one that just wants to return an already-loaded model
        — for the full duration of the join. Issue #315.
        """
        evictees: list[LoadedModel] = []
        while len(self._loaded) >= settings.max_loaded_models:
            evictable = {k: v for k, v in self._loaded.items() if v.active_refs == 0}
            if not evictable:
                raise RuntimeError(
                    "All loaded models are in use, cannot evict to load a new model"
                )
            oldest_name = min(evictable, key=lambda k: evictable[k].loaded_at)
            logger.info("Evicting model %s", oldest_name)
            evictees.append(self._loaded.pop(oldest_name))
        return evictees

    def _pop_one_idle_lru(self, exclude: str | None = None) -> LoadedModel | None:
        """Pop the oldest idle (``active_refs == 0``) model, or None if none.

        Unlike :meth:`_pop_lru_evictees`, this is driven by memory pressure
        rather than the ``max_loaded_models`` count, and never raises when
        every model is active — it simply returns None so the caller can
        stop evicting and proceed (a model in active use cannot be freed).

        ``exclude`` names a model that must never be popped — the pressure
        loop passes the model it is about to load so that a concurrent caller
        which loaded the same model during the lock-release window (and whose
        ``active_refs`` is still 0 until inference starts) is not closed out
        from under it, which would also make this caller load a duplicate.

        Must be called while holding ``self._lock``. The caller closes the
        returned model via :meth:`_close_evictees` after releasing the lock
        (same pop / close split as ``_pop_lru_evictees`` — issue #315).
        """
        idle = {
            k: v for k, v in self._loaded.items() if v.active_refs == 0 and k != exclude
        }
        if not idle:
            return None
        oldest_name = min(idle, key=lambda k: idle[k].loaded_at)
        return self._loaded.pop(oldest_name)

    async def _close_evictees(self, evictees: list[LoadedModel]) -> None:
        """Close popped evictees off the event loop. MUST NOT hold ``self._lock``.

        Mirrors :meth:`_expire_stale`'s close loop: each close runs on a
        worker thread via ``asyncio.to_thread`` so the event loop keeps
        servicing other coroutines (including other ``ensure_loaded``
        callers) while the prefetcher / weight store pools drain.

        Pops from the list as we go so no live reference (including the
        loop variable) survives into the caller's ``gc.collect()`` —
        otherwise the Metal buffers attached to the LoadedModel can't be
        reclaimed. Same reason :meth:`_expire_stale` uses ``del lm`` in
        its pop loop.

        On any abnormal exit (``CancelledError`` from client disconnect,
        unexpected close failure, ``KeyboardInterrupt`` etc.) drain the
        rest of the list synchronously. The popped models are already
        gone from ``_loaded`` — nothing else will close their
        prefetcher / weight store pools (48 threads + per-layer fds
        each) if we drop them on the floor here. Sync close is fine on
        the abnormal path: we'd rather briefly stall the loop than
        permanently leak. The recursive ``_close_loaded_model`` close
        still absorbs ``ExceptionGroup`` per-model.

        Normal-path ``ExceptionGroup`` absorption matches
        ``_close_loaded_model``'s documented contract (always raises
        ExceptionGroup on any error) — same shape as the catch in
        :meth:`unload`. An unexpected non-ExceptionGroup exception goes
        through the abnormal-path drain and re-raises.
        """
        try:
            while evictees:
                evicted = evictees.pop()
                try:
                    await asyncio.to_thread(self._close_loaded_model, evicted)
                except ExceptionGroup:
                    pass  # already logged per-resource inside _close_loaded_model
                except BaseException:
                    # Abnormal exit. Put the just-popped evictee back so the
                    # finally drain catches it. ``_close_loaded_model`` is
                    # idempotent (nulls weight_store / speculative_decoder
                    # sentinels and ThreadPoolExecutor.shutdown(wait=True)
                    # accepts repeated calls), so a re-close from drain
                    # racing with the background thread is safe — just
                    # noisy in logs if both run to completion.
                    evictees.append(evicted)
                    raise
                # Null model and tokenizer HERE on the event loop (NOT in
                # the worker thread) to avoid a race: between
                # ensure_loaded() returning and the caller accessing
                # lm.model, the worker thread could set it to None and
                # crash the caller.  See the _expire_stale contract: "the
                # caller still holds a Python reference, so the model/
                # tokenizer stay alive."  No await between null and del
                # — back on the event loop after the thread join — so
                # no other coroutine can observe the nulled field.
                evicted.model = None
                evicted.tokenizer = None
                del evicted
        finally:
            # On normal exit ``evictees`` is empty and this is a no-op. On
            # any abnormal exit (CancelledError from client disconnect,
            # unexpected close failure, KeyboardInterrupt) drain the rest
            # synchronously so the 48-thread + per-layer-fd resources
            # don't leak — the popped models are already gone from
            # ``_loaded`` and nothing else will close them.
            while evictees:
                lm = evictees.pop()
                try:
                    self._close_loaded_model(lm)
                except ExceptionGroup:
                    pass  # already logged per-resource inside _close_loaded_model
                lm.model = None
                lm.tokenizer = None
                del lm

    async def ensure_loaded(
        self, name: str, keep_alive: int | str | None = None
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

            evictees: list[LoadedModel] = []
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
                # Resolve per-model Flash primary-knob settings (also
                # promoted to top-level). Advanced/tuning Flash fields
                # still live under ``experimental`` and ride along on
                # ``model_exp``.
                flash_config = model_config.resolved_flash()
                kv_cache_quant = model_config.resolved_kv_cache_quant()
                weight_quant_str = model_config.resolved_weight_quant()
                # Whisper models (issue #366) have no LLM KV cache — never
                # apply KV-cache quantization / spectral calibration to them.
                _model_kind = self._detect_model_kind(hf_path)
                if _model_kind == "whisper":
                    kv_cache_quant = None
                flash_moe_config = model_config.resolved_flash_moe()

                # Pop LRU evictees under the lock; close them outside the lock
                # below so other ``ensure_loaded`` callers — especially ones
                # asking for an already-loaded model — aren't stalled for the
                # 48-thread executor.shutdown join. Issue #315.
                evictees = self._pop_lru_evictees()

            # Close popped evictees off the event loop with the lock released.
            if evictees:
                await self._close_evictees(evictees)

            # Pre-load Metal allocator flush. Runs even with no evictees so
            # the ``mem_before`` measurement in the second lock section sees
            # a clean baseline. Skip when a deferred cleanup is pending — a
            # background thread for a different model may still be allocating
            # Metal memory, and ``mx.clear_cache()`` is not safe to call
            # concurrently with active allocations.
            #
            # The ``_pending_cleanups`` read is lock-free. Safe because asyncio
            # is cooperative: there is no ``await`` between the dict check and
            # the ``mx.clear_cache()`` call, so no other coroutine (including
            # the one that would schedule a deferred cleanup via
            # ``_schedule_deferred_cleanup``) can interleave between the two.
            if not self._pending_cleanups:
                try:
                    await self._flush_metal()
                except Exception:
                    logger.exception(
                        "Metal flush failed before load; released "
                        "memory will be drained on the next flush."
                    )

            # Pre-load memory hygiene: if Metal memory is still under
            # pressure after eviction + close + gc, evict prompt caches
            # from all remaining loaded models and do another cleanup pass.
            # Runs OUTSIDE the lock and offloads disk I/O to worker threads
            # so concurrent ensure_loaded callers — including ones asking
            # for already-loaded models — are not stalled (issue #315).
            # Without this hygiene pass, loading a model on top of residual
            # GPU allocations from a prior model pushes Metal into swap,
            # causing the sub-token-per-second thrashing documented in
            # issue #223.
            # Trigger the pressure / eviction pass against the same effective
            # budget the admission check uses (limit - headroom), not the raw
            # limit. Otherwise, with headroom configured, an idle model can
            # sit below the raw limit but above the effective budget — the
            # hygiene pass would skip eviction and the load would then be
            # rejected even though evicting that idle model first would fit.
            budget = settings.effective_load_budget_fraction
            if memory_utils.is_memory_pressure_high(budget):
                logger.warning(
                    "Metal memory under pressure before loading %s; "
                    "flushing prompt caches to free GPU memory",
                    normalized,
                )
                # Snapshot to a list before the await loop — iterating
                # a live dict view across await yields would risk
                # RuntimeError if another coroutine modifies _loaded
                # during the I/O (issue #315 lock-split contract).
                #
                # Narrow race: _expire_stale (background loop, every 30s)
                # may concurrently pop and close a model that is in our
                # snapshot.  In CPython the GIL prevents a segfault, and
                # the concurrent close makes our async_evict_all_to_disk()
                # a harmless no-op (the store is already cleared).  Worst
                # case a few cache entries are not flushed to disk, but
                # GPU memory is freed either way.
                for other_lm in list(self._loaded.values()):
                    # Capture the store reference before the await to
                    # prevent TOCTOU: _expire_stale (background task,
                    # every 30s) may concurrently close this model,
                    # setting ``other_lm.prompt_cache_store = None``
                    # during the yield.  If that happens, re-evaluating
                    # the field would raise AttributeError.
                    store = other_lm.prompt_cache_store
                    if store is not None:
                        try:
                            await store.async_evict_all_to_disk()
                        except Exception:
                            # _close_loaded_model (running in a worker
                            # thread) may have concurrently called
                            # store.clear() — the store is empty, the
                            # GPU buffers are already freed.
                            logger.debug(
                                "Concurrent close cleared prompt cache "
                                "for %s during hygiene flush; skipping",
                                other_lm.name,
                            )
                # Skip gc/clear when a deferred cleanup is pending:
                # mx.clear_cache() is not safe to call concurrently with
                # active Metal allocations from a background thread (see
                # the matching guard at the _ensure_loaded pre-load path).
                if not self._pending_cleanups:
                    await self._flush_metal()

                # Flushing prompt caches frees KV buffers but leaves the
                # resident models' *weights* in Metal.  When several models
                # are kept resident (max_loaded_models > 1), those weights are
                # what push Metal into swap once a new large model loads on
                # top — the sub-token/sec thrash in issue #223.  Count-based
                # eviction (_pop_lru_evictees) won't touch them because the
                # count is still under the limit.  So if pressure persists,
                # evict idle (active_refs == 0) models LRU-first until it
                # clears or nothing idle remains — the "exclusive load" the
                # issue asks for.  Models in active use are left alone (they
                # cannot be freed); the load then proceeds with a warning.
                pressure = memory_utils.is_memory_pressure_high(budget)
                while pressure:
                    async with self._lock:
                        # Never evict ``normalized`` itself: a concurrent
                        # caller may have loaded it during the lock-release
                        # window (active_refs still 0), and closing it would
                        # corrupt that caller and force a duplicate load here.
                        idle = self._pop_one_idle_lru(exclude=normalized)
                    if idle is None:
                        break
                    logger.warning(
                        "Metal pressure persists; evicting idle model %s to "
                        "make room for %s",
                        idle.name,
                        normalized,
                    )
                    await self._close_evictees([idle])
                    if not self._pending_cleanups:
                        await self._flush_metal()
                    pressure = memory_utils.is_memory_pressure_high(budget)

                if pressure:
                    logger.warning(
                        "Metal pressure persists after hygiene flush; "
                        "proceeding to load %s anyway — generation may "
                        "be slow if Metal swaps",
                        normalized,
                    )

            async with self._lock:
                # State may have changed while the lock was released for the
                # close. Re-validate before proceeding with the load.
                if normalized in self._pending_cleanups:
                    continue

                if normalized in self._loaded:
                    # Another caller loaded the same model during our close.
                    lm = self._loaded[normalized]
                    ka = self._resolve_keep_alive(keep_alive)
                    lm.expires_at = (time.time() + ka) if ka is not None else None
                    return lm

                if len(self._loaded) >= settings.max_loaded_models:
                    # Another caller filled the slot we just emptied. Loop
                    # back to the top and re-evict before retrying.
                    continue

                logger.info("Loading model %s from %s", normalized, hf_path)
                mem_before = memory_utils.get_metal_memory()

                # Initialize before try so the except handler can always
                # clean up, whether _load_model or the post-load check fails.
                model = tokenizer = None
                load_task = lm = None
                try:
                    coro = asyncio.to_thread(
                        self._load_model_and_shard,
                        hf_path,
                        model_exp,
                        spec_config,
                        flash_config,
                        flash_moe_config,
                        weight_quant_str,
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
                    # Reserve headroom below the limit for the KV cache and
                    # activations that allocate on top of the weights during
                    # decode — a model whose weights land just under the limit
                    # can still swap mid-generation (issue #223).  Default
                    # headroom 0.0 reproduces the legacy weights-only check.
                    # Same budget the pre-load eviction trigger uses (above).
                    effective_fraction = settings.effective_load_budget_fraction
                    if total > 0 and mem_after > int(total * effective_fraction):
                        limit = int(total * effective_fraction)
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
                            effective_fraction * 100,
                            total_mb,
                        )
                        headroom_note = (
                            f" (includes a {settings.inference_headroom_fraction:.0%} "
                            f"inference headroom reserve; lower "
                            f"OLMLX_INFERENCE_HEADROOM_FRACTION to reclaim it)"
                            if settings.inference_headroom_fraction > 0
                            else ""
                        )
                        raise MemoryError(
                            f"Model '{normalized}' requires ~{model_mb} MB but the memory limit "
                            f"is {limit_mb} MB ({effective_fraction:.0%} of "
                            f"{total_mb} MB system RAM){headroom_note}. Use a smaller/more "
                            f"quantized model, or increase OLMLX_MEMORY_LIMIT_FRACTION "
                            f"(current: {settings.memory_limit_fraction})."
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

                        is_flash_moe = isinstance(model, FlashMoeModelWrapper)
                    except ImportError:
                        pass
                    # Both Flash wrappers store the underlying weight store as
                    # ``_weight_store`` so eviction / keep-alive expiry can
                    # close it. Non-Flash models leave this at None.
                    _weight_store = getattr(model, "_weight_store", None)

                    # Read on-disk size from the manifest (backfilled by
                    # _load_model), falling back to _dir_size for
                    # pre-existing manifests that may lack the field.
                    _model_size = 0
                    if self.store is not None:
                        _local_dir = self.store.local_path(hf_path)
                        if _local_dir.exists():
                            _manifest_path = _local_dir / "manifest.json"
                            if _manifest_path.exists():
                                try:
                                    import json

                                    _model_size = json.loads(
                                        _manifest_path.read_text()
                                    ).get("size", 0)
                                except Exception:
                                    pass
                            if _model_size == 0:
                                _model_size = await asyncio.to_thread(
                                    _dir_size, _local_dir
                                )

                    # _find_spectral_dir may trigger multi-minute calibration.
                    # Run it in a thread so it does not block the event loop.
                    _spectral_dir = await asyncio.to_thread(
                        self._find_spectral_dir, hf_path, kv_cache_quant
                    )
                    lm = LoadedModel(
                        name=normalized,
                        hf_path=hf_path,
                        model=model,
                        tokenizer=tokenizer,
                        is_vlm=is_vlm,
                        is_distributed=is_distributed,
                        is_flash=is_flash,
                        is_flash_moe=is_flash_moe,
                        is_whisper=(_model_kind == "whisper"),
                        speculative_decoder=_spec_decoder,
                        weight_store=_weight_store,
                        template_caps=caps,
                        expires_at=expires,
                        size_bytes=_model_size,
                        kv_cache_quant=kv_cache_quant,
                        weight_quant=weight_quant_str,
                        spectral_calibration_dir=_spectral_dir,
                        default_options=dict(model_config.options),
                        inference_queue_timeout=model_config.inference_queue_timeout,
                        inference_timeout=model_config.inference_timeout,
                        sync_mode=model_config.sync_mode,
                    )
                    # Register before probe so concurrent callers see the
                    # model while the probe's async Metal flush releases the
                    # lock.  _probe_cache_capabilities sets all cache flags
                    # synchronously before its first await (the finally-block
                    # Metal flush), so no coroutine can observe an incomplete
                    # probe state between registration and the yield point.
                    self._loaded[normalized] = lm
                    await self._probe_cache_capabilities(lm)
                    return lm
                except BaseException:
                    # Drop references and flush Metal allocator so the memory
                    # is actually reclaimed before we raise.  Also clean up
                    # lm if it was already constructed (exception between
                    # LoadedModel() and return).  ``lm`` may be registered
                    # in ``_loaded`` at this point (it is inserted before
                    # the async probe).  If a concurrent caller retrieved
                    # ``lm`` during the probe's async yield and started
                    # inference (``active_refs > 0``), leave it in
                    # ``_loaded`` — eviction/expiry will clean it up when
                    # refs drop to zero.  Otherwise ``pop(normalized, None)``
                    # handles both the was-only and already-registered cases.
                    if lm is not None:
                        if lm.active_refs == 0:
                            self._loaded.pop(normalized, None)
                            del lm
                        else:
                            logger.warning(
                                "Load of '%s' cancelled after registration "
                                "with %d active request(s); model left in "
                                "_loaded for expiry/eviction to clean up.",
                                normalized,
                                lm.active_refs,
                            )
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
                    # mx.clear_cache() (global Metal allocator) is not safe
                    # to call concurrently with active Metal allocations.
                    # The deferred cleanup handles it after the thread
                    # finishes.  Guarded on the *global* dict (not
                    # per-model) because a background thread for *any*
                    # model allocating Metal memory makes a concurrent
                    # clear unsafe (matching _expire_stale and unload).
                    if not self._pending_cleanups:
                        await self._flush_metal()
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
                        await self._flush_metal()
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

    async def unload(self, name: str) -> bool:
        """Unload a model. Returns True if unloaded, False if not loaded.

        Raises ActiveRequestsError if model has in-flight requests.

        Async because ``_close_loaded_model`` calls
        ``executor.shutdown(wait=True)`` on the prefetcher (16 threads)
        and weight store (32 threads); doing that join on the event loop
        thread stalls every concurrent coroutine for the duration of the
        join. ``asyncio.to_thread`` offloads it so other endpoints keep
        responding while a Flash model unloads. Same fix shape as the
        eviction / expiry paths. Issue #315.

        Close failures from ``_close_loaded_model`` are absorbed: the
        model is already gone from ``_loaded`` (won't accept new
        requests), so the user-visible unload semantics are satisfied
        even if some background threads leaked. The per-resource log
        lines inside ``_close_loaded_model`` record what failed.
        Surfacing the ExceptionGroup as a 500 instead would leave the
        client unable to distinguish "close failed, model is gone" from
        an unrelated 500 — and either way, the model is gone.
        """
        normalized = self.registry.normalize_name(name)
        lm = self._loaded.get(normalized)
        if lm is None:
            return False
        if lm.active_refs > 0:
            raise ActiveRequestsError(
                f"Model '{normalized}' has {lm.active_refs} active request(s)"
            )
        lm = self._loaded.pop(normalized)
        try:
            await asyncio.to_thread(self._close_loaded_model, lm)
        except ExceptionGroup:
            pass  # already logged per-resource inside _close_loaded_model
        # Drop the local reference so the Metal buffers can be reclaimed
        # before the function returns. Matches the pattern in
        # ``_close_evictees`` and ``_expire_stale``; without it a caller
        # that polls ``get_metal_memory()`` immediately after unload may
        # see the model's allocations still resident in the pool.
        del lm
        # Return freed Metal memory to the OS.  Mirrors the flush
        # in ``_expire_stale`` and ``ensure_loaded``.  Guarded on the
        # global ``_pending_cleanups`` dict because ``mx.clear_cache()``
        # flushes the entire Metal allocator — if ANY background thread
        # is still allocating Metal memory (even for a different model),
        # the concurrent clear is unsafe.
        if not self._pending_cleanups:
            try:
                await self._flush_metal()
            except Exception:
                logger.exception(
                    "Metal flush failed for unload of '%s'; model already "
                    "unloaded, Metal memory will be reclaimed on next flush.",
                    normalized,
                )
        else:
            logger.debug(
                "Skipping Metal flush for unload of '%s': deferred cleanup "
                "still in flight (global Metal clear unsafe). The deferred "
                "cleanup task will flush when it completes.",
                normalized,
            )
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

        # Whisper STT (issue #366). Check before the empty-model_type return:
        # mlx-community whisper repos ship a non-HF config.json carrying
        # mlx_whisper.whisper.ModelDimensions fields, and load_model pops
        # "model_type", so model_type is often absent. The dims keys are the
        # robust discriminator; the model_type check covers HF-style configs.
        if model_type == "whisper" or (
            "n_mels" in config and "n_audio_state" in config
        ):
            return "whisper"

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
                    #
                    # Inverse case (Qwen3.6+): ``text_config.model_type`` is
                    # the ``_text``-suffixed name (``qwen3_5_moe_text``)
                    # with no mlx-lm module, but the top-level type
                    # (``qwen3_5_moe``) does have one. Fall back to the
                    # top-level type before raising — ``mlx_lm.load()``
                    # (which the engine actually invokes downstream)
                    # resolves via top-level model_type and handles these
                    # configs successfully.
                    mapped_top = LM_REMAP.get(model_type, model_type)
                    # Resolve module-availability once for both the
                    # fallback condition and the primary check below to
                    # avoid duplicate ``find_spec`` import-system walks.
                    mapped_lm_present = (
                        importlib.util.find_spec(f"mlx_lm.models.{mapped_lm}")
                        is not None
                    )
                    # Belt-and-suspenders: re-assert the
                    # ``linear_attention`` discriminant from the outer
                    # guard so the warning's "hybrid linear-attention"
                    # label is verifiable from this block alone. A
                    # future refactor that hoists this fallback out of
                    # the outer guard could otherwise silently route a
                    # full-attention VLM with a missing ``_text``-
                    # suffixed inner type to mlx-lm and lose vision.
                    if (
                        not mapped_lm_present
                        and mapped_top != mapped_lm
                        and importlib.util.find_spec(f"mlx_lm.models.{mapped_top}")
                        is not None
                        # Redundant with the outer ``"linear_attention" in
                        # layer_types`` guard at line 1262; kept so this
                        # block stays self-contained against future
                        # refactoring.
                        and "linear_attention" in layer_types
                    ):
                        logger.warning(
                            "Routing hybrid linear-attention VLM '%s' through "
                            "mlx-lm text path '%s' (issue #284) — text_config "
                            "model_type '%s' has no matching mlx-lm module, "
                            "falling back to top-level model_type. Vision "
                            "capability is disabled for this load.",
                            model_type,
                            mapped_top,
                            text_model_type,
                        )
                        return "text"
                    if mapped_lm_present:
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
                    # Fallback: some hybrid model variants flip the convention
                    # — top-level model_type is the bare arch ("qwen3_5") and
                    # text_config.model_type is a derivative tag with no
                    # mlx-lm module ("qwen3_5_text"). mlx-lm's own loader
                    # uses the top-level model_type for module resolution
                    # (mlx_lm.utils._get_classes), so when the text path
                    # lookup misses we should try the top-level before
                    # giving up.
                    if text_model_type != model_type:
                        mapped_top = LM_REMAP.get(model_type, model_type)
                        if (
                            importlib.util.find_spec(f"mlx_lm.models.{mapped_top}")
                            is not None
                        ):
                            logger.warning(
                                "Routing hybrid linear-attention VLM '%s' "
                                "through mlx-lm text path '%s' (issue #284, "
                                "via top-level fallback because "
                                "text_config.model_type '%s' has no mlx-lm "
                                "module). Vision capability disabled.",
                                model_type,
                                mapped_top,
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

    async def _probe_cache_capabilities(self, lm: LoadedModel) -> None:
        """Probe how the model's prompt cache can be safely reused, recording
        the result on ``lm.supports_cache_trim`` and ``lm.supports_cache_persistence``.

        **Caller contract**: the caller must register ``lm`` in ``_loaded``
        BEFORE calling this method (so concurrent ``ensure_loaded`` callers
        see the model). This is safe because every ``lm.*`` flag assignment
        in this method completes synchronously, before the first ``await``
        (the ``finally`` block's Metal flush).  Do not insert an ``await``
        before the ``lm.supports_cache_* = ...`` lines without updating the
        caller-side registration guard accordingly.

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
        if lm.is_whisper:
            # Whisper has no LLM-style prompt cache; nothing to probe.
            return
        try:
            from mlx_lm.models.cache import make_prompt_cache
        except ImportError:
            return  # mlx-lm prompt cache unavailable; nothing to probe

        cache_model = (
            getattr(lm.model, "language_model", lm.model) if lm.is_vlm else lm.model
        )
        probe_cache: list | None = None
        probe_succeeded = False
        # Hoisted so the logging dispatch below can read the raw allowlist
        # result (#343 vs #284 attribution).  Dispatch is gated on
        # probe_succeeded, so the False default is harmless on exception.
        persist_layout_ok = False
        try:
            probe_cache = make_prompt_cache(cache_model)
            # Stage both results before assigning so a hypothetical raise
            # in either probe doesn't leave one flag set and the other
            # falling into the except handler's defaults.  Currently both
            # probes are pure string-set lookups that can't raise, but the
            # pattern is cheap and removes the ordering dependency.
            # ``persist_layout_ok`` is a third intermediate computed here
            # and consumed in the logging dispatch outside the try, but
            # its read site is gated on ``probe_succeeded`` — if any
            # fallible logic is later inserted between the two probes,
            # an exception there leaves ``persist_layout_ok = False`` with
            # ``probe_succeeded = False`` and the dispatch is skipped.
            trim_ok = _cache_supports_trim(probe_cache)
            # Raw layout-persistence result from the allowlist, kept
            # separately from the post-#343 effective flag so the
            # logging dispatch below can distinguish the two non-
            # persistable sub-cases (#343 RotatingKVCache vs #284
            # ArraysCache) and emit the correct attribution.
            persist_layout_ok = _cache_supports_persistence(probe_cache)
            # Issue #343: non-trimmable cache layouts (RotatingKVCache,
            # ChunkedKVCache) can never realign their stored prompt +
            # generated state with the next request's tokens.  In real
            # chat flow the client-supplied assistant message retokenizes
            # differently from what the model actually emitted (whitespace,
            # EOS handling, early-stop, chat-template stop sequences), so
            # the trim check at lookup time always demands a non-zero
            # rollback that these caches can't perform — and the cached
            # entry is discarded.  Storing under those conditions is pure
            # overhead with no realistic recovery path.  Treat the same as
            # ArraysCache (#284): disable persistence at probe time and
            # short-circuit the store/load path entirely.  Trim implies
            # persist; persist without trim does not, post-#343.
            #
            # Today ``trim_ok ⇒ persist_layout_ok`` holds because
            # _TRIMMABLE_CACHE_CLASSES ⊂ _PERSISTABLE_CACHE_CLASSES, but
            # the two-term form is kept on purpose so the formula self-
            # corrects if that subset relation ever breaks (e.g. a
            # future cache class added to the trim allowlist but not the
            # persist allowlist).  Do not simplify to ``persist_ok =
            # trim_ok``: that would silently over-persist any such class
            # at the next mlx-lm release.
            persist_ok = persist_layout_ok and trim_ok
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
                # All cache flags (supports_cache_trim,
                # supports_cache_persistence) are set synchronously above
                # — this is the method's first await.  Do not insert an
                # await before the flag assignments without also verifying
                # that concurrent ensure_loaded callers, which may observe
                # the registered LoadedModel during this yield, see
                # fully-configured flags.
                #
                # Guarded on _pending_cleanups (matching all other
                # _flush_metal sites): if a deferred cleanup is still
                # running, its own _flush_metal will drain the released
                # probe memory when it completes.
                if not self._pending_cleanups:
                    await self._flush_metal()

        # Single load-time log site for "cross-request reuse disabled."
        # Distinguishes the two disable reasons by inspecting the raw
        # layout-persistence result captured above:
        #
        #   layout-persistable + non-trimmable → #343 (RotatingKVCache,
        #       ChunkedKVCache); the layout itself could persist but the
        #       fold in _probe forces it off because stored state can't
        #       realign across requests.
        #
        #   layout non-persistable              → #284 (ArraysCache /
        #       hybrid SSM, or an unclassified cache type that's not on
        #       either allowlist).
        #
        # The two reasons are mutually exclusive given the current
        # allowlists (_TRIMMABLE_CACHE_CLASSES ⊂ _PERSISTABLE_CACHE_CLASSES),
        # so this is exactly one line per affected model with correct
        # attribution.  Probe-failure path already WARNED above and is
        # skipped here.
        if probe_succeeded and not lm.supports_cache_persistence:
            if persist_layout_ok:
                logger.info(
                    "Model %s uses a non-trimmable hybrid sliding-window "
                    "cache (RotatingKVCache/ChunkedKVCache); cross-request "
                    "prompt cache reuse is disabled (issue #343).",
                    lm.name,
                )
            else:
                logger.info(
                    "Model %s uses a non-persistable cache (hybrid SSM/"
                    "ArraysCache or unclassified); cross-request prompt "
                    "cache reuse is disabled (issue #284).",
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

        # Parse and validate bit width before checking for calibration data.
        # Defence-in-depth: config.py's validator already gates all real
        # entrypoints; this protects direct callers (e.g. tests) that bypass
        # config loading and would otherwise hit a cryptic downstream error.
        try:
            configured_bits = int(kv_cache_quant.split(":", 1)[1])
        except (ValueError, IndexError):
            raise ValueError(
                f"Invalid SpectralQuant config {kv_cache_quant!r}; "
                f"expected spectral:2 or spectral:4"
            )
        if configured_bits not in (2, 4):
            raise ValueError(
                f"Invalid SpectralQuant bit width {kv_cache_quant!r}; expected 2 or 4"
            )

        spectral_path = self.store.local_path(hf_path) / "spectral"
        if spectral_path.exists() and (spectral_path / "spectral_config.json").exists():
            try:
                config = json.loads(
                    (spectral_path / "spectral_config.json").read_text()
                )
            except (json.JSONDecodeError, OSError) as exc:
                recalibrate_cmd = f"olmlx spectral prepare {hf_path}"
                if configured_bits != 4:
                    recalibrate_cmd += f" --avg-bits {configured_bits}"
                raise SpectralCalibrationMissingError(
                    f"SpectralQuant configured ({kv_cache_quant}) but calibration "
                    f"file at {spectral_path}/spectral_config.json is unreadable "
                    f"({exc}). Re-run '{recalibrate_cmd}'."
                )
            cal_bits = config.get("meta", {}).get("avg_bits")
            if cal_bits is not None and cal_bits != configured_bits:
                calibrate_cmd = (
                    f"olmlx spectral prepare {hf_path} --avg-bits {configured_bits}"
                )
                raise SpectralCalibrationMissingError(
                    f"SpectralQuant configured ({kv_cache_quant}) but calibration "
                    f"data at {spectral_path} was generated with "
                    f"--avg-bits {cal_bits}. Run '{calibrate_cmd}' "
                    f"to re-calibrate at {configured_bits}-bit, "
                    f"or set OLMLX_KV_CACHE_QUANT=spectral:{cal_bits} "
                    f"to use the existing calibration."
                )
            return spectral_path

        # Auto-calibrate if enabled
        if settings.kv_cache_auto_calibrate:
            return self._auto_calibrate_spectral(hf_path, kv_cache_quant)

        other_bits = 4 if configured_bits == 2 else 2

        calibrate_cmd = f"olmlx spectral prepare {hf_path}"
        if configured_bits != 4:  # 4 is calibrate_model's default avg_bits
            calibrate_cmd += f" --avg-bits {configured_bits}"
        raise SpectralCalibrationMissingError(
            f"SpectralQuant configured ({kv_cache_quant}) but no calibration data "
            f"found at {spectral_path}. Run '{calibrate_cmd}' "
            f"to calibrate. Calibration collects KV vectors from "
            f"~{_SPECTRAL_DEFAULT_NUM_SAMPLES} text samples (C4 by default) "
            f"and computes per-layer eigendecompositions for non-uniform "
            f"bit allocation ({_SPECTRAL_DEFAULT_MAX_TOKENS_PER_HEAD} "
            f"tokens/head). "
            f"Use --avg-bits {other_bits} for {other_bits}-bit mode, "
            f"--samples N to change sample count, "
            f"--calibration-dataset synthetic to override dataset. "
            f"Or set OLMLX_KV_CACHE_AUTO_CALIBRATE=true to auto-calibrate."
        )

    def _auto_calibrate_spectral(self, hf_path: str, kv_cache_quant: str) -> Path:
        """Run spectral calibration automatically with default settings."""
        import logging

        from olmlx.engine.spectralquant_calibrate import calibrate_model

        method, bits_str = kv_cache_quant.split(":")
        assert method == "spectral", (
            f"_auto_calibrate_spectral called with non-spectral quant: "
            f"{kv_cache_quant!r}"
        )
        avg_bits = int(bits_str)
        local_dir = self.store.local_path(hf_path)
        logger = logging.getLogger(__name__)
        logger.info(
            "Auto-calibrating spectral quant (%s-bit) for %s "
            "(this may take several minutes)...",
            avg_bits,
            hf_path,
        )
        try:
            output_dir = calibrate_model(
                model_path=str(local_dir),
                num_samples=64,
                calibration_dataset="c4",
                avg_bits=avg_bits,
                max_tokens_per_head=2048,
            )
        except Exception as exc:
            raise SpectralCalibrationMissingError(
                f"Auto-calibration failed for {hf_path}: {exc}. "
                f"Run 'olmlx spectral prepare {hf_path}' manually."
            ) from exc
        spectral_path = Path(output_dir)
        if spectral_path.exists() and (spectral_path / "spectral_config.json").exists():
            logger.info("Auto-calibration complete for %s", hf_path)
            return spectral_path
        raise SpectralCalibrationMissingError(
            f"Auto-calibration completed but spectral data not found at "
            f"{spectral_path}. Run 'olmlx spectral prepare {hf_path}' manually."
        )

    def _resolve_draft_path(self, hf_path: str) -> str:
        """Download a draft model if needed and return the local path.

        Accepts either a HuggingFace repo id (``"namespace/repo_name"``)
        or an *absolute* filesystem path to a local draft directory.
        Local-path short-circuiting is gated on ``is_absolute()`` to
        avoid a false positive where a valid HF repo id (e.g.
        ``"my-org/dflash-draft"``) happens to match a directory under
        the server's CWD; that would silently swap the operator's
        intended remote artifact for whatever the working directory
        contains. Without short-circuiting, feeding an absolute path
        through ``ensure_downloaded`` raises ``HFValidationError``
        ("Repo id must be in the form 'repo_name' or
        'namespace/repo_name'").
        """
        candidate = Path(hf_path).expanduser()
        if candidate.is_absolute():
            # Absolute paths are unambiguous local references — they
            # cannot be HF repo ids. If the directory is missing, fall
            # through to ``ensure_downloaded`` would surface as an
            # ``HFValidationError`` ("Repo id must be in the form
            # 'repo_name' or 'namespace/repo_name'") which is actively
            # misleading for someone who passed e.g.
            # ``/Users/.../dflash`` and made a typo or pointed at a
            # path before training finished. Raise a clear
            # ``FileNotFoundError`` with the actual path instead.
            if not candidate.is_dir():
                raise FileNotFoundError(f"Draft model directory not found: {candidate}")
            return str(candidate)
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
        self,
        target_model: Any,
        spec_config: SpeculativeConfig,
    ) -> Any:
        """Load a dflash draft model and create a DFlashDecoder.

        Universal target support: no per-architecture adapter is required —
        the decoder hooks the target's selected layers in place via
        ``_patch_model``. The draft borrows ``embed_tokens`` and
        ``lm_head`` from the target via ``draft.bind(target_model)``.
        """
        from olmlx.engine.dflash.decoder import DFlashDecoder
        from olmlx.engine.dflash.draft_model import DFlashDraftModel, DraftConfig

        if not spec_config.enabled:
            raise RuntimeError(
                "_load_dflash_decoder called with spec_config.enabled=False"
            )
        if not spec_config.draft_model:
            raise ValueError(
                "speculative_strategy='dflash' requires speculative_draft_model "
                "to be set (OLMLX_SPECULATIVE_DRAFT_MODEL or per-model "
                "'speculative_draft_model' in models.json)"
            )

        logger.info("Loading dflash draft model %s", spec_config.draft_model)
        load_path = self._resolve_draft_path(spec_config.draft_model)

        config_file = Path(load_path) / "config.json"
        if not config_file.exists():
            raise FileNotFoundError(
                f"DFlash draft model config not found at {config_file}"
            )

        draft_cfg_dict = json.loads(config_file.read_text())
        dflash_cfg = draft_cfg_dict.get("dflash_config")
        if not isinstance(dflash_cfg, dict):
            raise ValueError(
                f"DFlash draft config at {config_file} is missing the "
                "'dflash_config' object (must contain 'target_layer_ids' "
                "and 'mask_token_id'). This loader expects the upstream "
                "z-lab DFlash schema."
            )
        _required_top = [
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "intermediate_size",
            "vocab_size",
            "rms_norm_eps",
            "rope_theta",
            "max_position_embeddings",
            "block_size",
        ]
        missing = [k for k in _required_top if k not in draft_cfg_dict]
        _required_dflash = ["target_layer_ids", "mask_token_id"]
        missing += [
            f"dflash_config.{k}" for k in _required_dflash if k not in dflash_cfg
        ]
        if missing:
            raise ValueError(
                f"DFlash draft config at {config_file} is missing "
                f"required keys: {missing}"
            )

        layer_types_raw = (
            draft_cfg_dict.get("layer_types")
            or ["full_attention"] * draft_cfg_dict["num_hidden_layers"]
        )

        draft_config = DraftConfig(
            hidden_size=draft_cfg_dict["hidden_size"],
            num_hidden_layers=draft_cfg_dict["num_hidden_layers"],
            num_attention_heads=draft_cfg_dict["num_attention_heads"],
            num_key_value_heads=draft_cfg_dict["num_key_value_heads"],
            head_dim=draft_cfg_dict["head_dim"],
            intermediate_size=draft_cfg_dict["intermediate_size"],
            vocab_size=draft_cfg_dict["vocab_size"],
            rms_norm_eps=draft_cfg_dict["rms_norm_eps"],
            rope_theta=draft_cfg_dict["rope_theta"],
            max_position_embeddings=draft_cfg_dict["max_position_embeddings"],
            block_size=draft_cfg_dict["block_size"],
            num_target_layers=len(dflash_cfg["target_layer_ids"]),
            target_layer_ids=list(dflash_cfg["target_layer_ids"]),
            mask_token_id=int(dflash_cfg["mask_token_id"]),
            rope_scaling=draft_cfg_dict.get("rope_scaling"),
            layer_types=tuple(layer_types_raw),
            sliding_window=draft_cfg_dict.get("sliding_window"),
            final_logit_softcapping=draft_cfg_dict.get("final_logit_softcapping"),
            attention_causal=_resolve_attention_causal(dflash_cfg),
        )

        draft_model = DFlashDraftModel(draft_config)
        draft_dir = Path(load_path)
        # Prefer the conventional ``model*.safetensors`` (HF/mlx-lm
        # convention, also covers sharded ``model-00001-of-N``). Only
        # fall back to ``*.safetensors`` if no conventional file is
        # present, so a co-located ``adapter_model.safetensors`` (LoRA)
        # or tokenizer projection file can't silently overwrite draft
        # weights via shared key names under ``strict=False``.
        weight_files = sorted(draft_dir.glob("model*.safetensors"))
        if not weight_files:
            weight_files = sorted(draft_dir.glob("*.safetensors"))
        if not weight_files:
            raise FileNotFoundError(
                f"DFlash draft model weights not found in {draft_dir}. "
                "A pre-trained dflash draft model is required."
            )
        weights: list[tuple[str, Any]] = []
        for wf in weight_files:
            weights.extend(mx.load(str(wf)).items())
        # ``strict=False`` permits missing keys for ``embed_tokens`` and
        # ``lm_head`` — those are bound from the target via
        # ``DFlashDraftModel.bind()`` and are intentionally absent from
        # the draft safetensors.
        draft_model.load_weights(weights, strict=False)
        logger.info(
            "Loaded dflash draft weights from %s (%d file(s))",
            draft_dir,
            len(weight_files),
        )

        # Vocab-size check: ``DFlashDraftModel.bind`` borrows the
        # target's ``embed_tokens`` / ``lm_head``, so a mismatch between
        # the draft's pre-trained vocab and the target produces an
        # ``mx.array`` shape error at the first draft forward pass —
        # surface it here at load time with a clear message instead.
        # ``DFlashDraftModel`` doesn't expose ``args.vocab_size``
        # (config lives on ``draft_config``), so we read the two
        # values directly rather than via ``_check_vocab_match``. Probe
        # the same locations ``_get_layers`` walks (top-level,
        # ``.model``, ``.language_model``) so VLM/wrapped targets that
        # don't expose ``args`` at the outer level still get the check.
        target_vocab: int | None = None
        for chain in ((), ("model",), ("language_model",), ("language_model", "model")):
            obj: Any = target_model
            for attr in chain:
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is None:
                continue
            args = getattr(obj, "args", None) or getattr(obj, "config", None)
            v = getattr(args, "vocab_size", None) if args is not None else None
            if v is not None:
                target_vocab = int(v)
                break
        if target_vocab is None:
            logger.warning(
                "Could not determine target vocab_size for DFlash draft "
                "compatibility check (target has no .args/.config at the "
                "probed locations). A mismatch will surface as an mx.array "
                "shape error at the first draft forward pass."
            )
        elif target_vocab != draft_config.vocab_size:
            raise ValueError(
                f"DFlash draft vocab_size ({draft_config.vocab_size}) does "
                f"not match target vocab_size ({target_vocab}). The draft "
                "must be trained against a target with the same vocabulary."
            )

        # ``draft_config.block_size`` is treated as the *draft token
        # count* directly (matching the convention #287 ships with —
        # the value used verbatim by ``SpeculativeDecoder``). Local
        # ``olmlx dflash prepare`` writes the same convention to disk
        # so a checkpoint trained here loads back without translation.
        # ``None`` (no user override) inherits the draft's pre-trained
        # block.
        block_size = (
            spec_config.num_tokens
            if spec_config.num_tokens is not None
            else draft_config.block_size
        )
        # Going *above* the trained draft count runs the draft on block
        # lengths it has never seen; the positional encoding and
        # block-diffusion training are bound to the trained length.
        # Warn (don't fail) — users may experiment.
        if (
            spec_config.num_tokens is not None
            and spec_config.num_tokens > draft_config.block_size
        ):
            logger.warning(
                "speculative_tokens=%d exceeds the draft's pre-trained "
                "block_size=%d; output quality may degrade. Omit "
                "speculative_tokens (or pass <= %d) to stay within the "
                "trained block length.",
                spec_config.num_tokens,
                draft_config.block_size,
                draft_config.block_size,
            )
        return DFlashDecoder(
            target_model=target_model,
            draft_model=draft_model,
            draft_config=draft_config,
            block_size=block_size,
        )

    def _load_eagle_decoder(
        self,
        target_model: Any,
        spec_config: SpeculativeConfig,
    ) -> Any:
        """Load an EAGLE draft model and create an EagleDecoder.

        Mirrors ``_load_dflash_decoder`` but consumes the EAGLE saved
        schema (flat target dims plus a top-level ``eagle_config``
        block carrying ``block_size`` and ``target_layer_id``). The
        EAGLE draft has no ``mask_token_id`` or ``target_layer_ids``
        — EAGLE conditions on a single target hidden state per step,
        and ``olmlx eagle prepare`` records the chosen layer in
        ``eagle_config.target_layer_id`` (the deepest layer of the
        precomputed shard ladder). When that field is absent
        (pre-fix checkpoints), the decoder falls back to
        ``num_layers - 1`` and a ``logger.warning`` surfaces the
        misconfiguration with a nudge to retrain.
        """
        from olmlx.engine.eagle.decoder import EagleDecoder
        from olmlx.engine.eagle.draft_model import EagleConfig, EagleDraftModel

        if not spec_config.enabled:
            raise RuntimeError(
                "_load_eagle_decoder called with spec_config.enabled=False"
            )
        if not spec_config.draft_model:
            raise ValueError(
                "speculative_strategy='eagle' requires speculative_draft_model "
                "to be set (OLMLX_SPECULATIVE_DRAFT_MODEL or per-model "
                "'speculative_draft_model' in models.json)"
            )

        logger.info("Loading EAGLE draft model %s", spec_config.draft_model)
        load_path = self._resolve_draft_path(spec_config.draft_model)

        config_file = Path(load_path) / "config.json"
        if not config_file.exists():
            raise FileNotFoundError(
                f"EAGLE draft model config not found at {config_file}"
            )

        draft_cfg_dict = json.loads(config_file.read_text())
        eagle_cfg = draft_cfg_dict.get("eagle_config")
        if not isinstance(eagle_cfg, dict):
            raise ValueError(
                f"EAGLE draft config at {config_file} is missing the "
                "'eagle_config' object (must contain 'block_size'). The "
                "saved checkpoint may be a DFlash draft — pass "
                "speculative_strategy='dflash' instead, or retrain via "
                "`olmlx eagle prepare`."
            )
        _required_top = [
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "intermediate_size",
            "vocab_size",
            "rms_norm_eps",
            "rope_theta",
            "max_position_embeddings",
        ]
        missing = [k for k in _required_top if k not in draft_cfg_dict]
        if "block_size" not in eagle_cfg:
            missing.append("eagle_config.block_size")
        if missing:
            raise ValueError(
                f"EAGLE draft config at {config_file} is missing required "
                f"keys: {missing}"
            )

        draft_config = EagleConfig(
            hidden_size=draft_cfg_dict["hidden_size"],
            num_hidden_layers=draft_cfg_dict["num_hidden_layers"],
            num_attention_heads=draft_cfg_dict["num_attention_heads"],
            num_key_value_heads=draft_cfg_dict["num_key_value_heads"],
            head_dim=draft_cfg_dict["head_dim"],
            intermediate_size=draft_cfg_dict["intermediate_size"],
            vocab_size=draft_cfg_dict["vocab_size"],
            rms_norm_eps=draft_cfg_dict["rms_norm_eps"],
            rope_theta=draft_cfg_dict["rope_theta"],
            max_position_embeddings=draft_cfg_dict["max_position_embeddings"],
            block_size=int(eagle_cfg["block_size"]),
            rope_scaling=draft_cfg_dict.get("rope_scaling"),
        )

        draft_model = EagleDraftModel(draft_config)
        draft_dir = Path(load_path)
        # Same conventional-then-fallback search ``_load_dflash_decoder``
        # uses; comment there explains the precedence rationale.
        weight_files = sorted(draft_dir.glob("model*.safetensors"))
        if not weight_files:
            weight_files = sorted(draft_dir.glob("*.safetensors"))
        if not weight_files:
            raise FileNotFoundError(
                f"EAGLE draft model weights not found in {draft_dir}. "
                "Train one via `olmlx eagle prepare <target>`."
            )
        weights: list[tuple[str, Any]] = []
        for wf in weight_files:
            weights.extend(mx.load(str(wf)).items())
        # ``strict=False`` permits the absent ``embed_tokens`` /
        # ``lm_head`` (re-bound from target on every prefill).
        draft_model.load_weights(weights, strict=False)
        logger.info(
            "Loaded EAGLE draft weights from %s (%d file(s))",
            draft_dir,
            len(weight_files),
        )

        # Vocab-size + hidden-size cross-checks, mirroring the DFlash
        # loader so a cross-target draft surfaces here rather than at
        # the first forward pass.
        #
        # ``hidden_size`` matters because EAGLE's input projection is
        # shape ``(2 * hidden_size, hidden_size)`` — it concatenates
        # the target's hidden (shape ``hidden_size``) with the
        # embedding (shape ``hidden_size``). A draft trained against
        # Qwen3.5-7B (hidden=3584) loaded against Qwen3.5-27B
        # (hidden=5120) would pass vocab and crash with a cryptic
        # shape error inside ``input_proj`` on the first prefill.
        target_vocab: int | None = None
        target_hidden: int | None = None
        for chain in ((), ("model",), ("language_model",), ("language_model", "model")):
            obj: Any = target_model
            for attr in chain:
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is None:
                continue
            args = getattr(obj, "args", None) or getattr(obj, "config", None)
            if args is not None:
                if target_vocab is None:
                    v = getattr(args, "vocab_size", None)
                    if v is not None:
                        target_vocab = int(v)
                if target_hidden is None:
                    h = getattr(args, "hidden_size", None)
                    if h is not None:
                        target_hidden = int(h)
            if target_vocab is not None and target_hidden is not None:
                break
        if target_vocab is None:
            logger.warning(
                "Could not determine target vocab_size for EAGLE draft "
                "compatibility check. A mismatch will surface as an mx.array "
                "shape error at the first draft forward pass."
            )
        elif target_vocab != draft_config.vocab_size:
            raise ValueError(
                f"EAGLE draft vocab_size ({draft_config.vocab_size}) does "
                f"not match target vocab_size ({target_vocab}). The draft "
                "must be trained against a target with the same vocabulary."
            )
        if target_hidden is None:
            logger.warning(
                "Could not determine target hidden_size for EAGLE draft "
                "compatibility check. A mismatch will surface as an mx.array "
                "shape error inside the draft's input_proj on the first "
                "prefill."
            )
        elif target_hidden != draft_config.hidden_size:
            raise ValueError(
                f"EAGLE draft hidden_size ({draft_config.hidden_size}) does "
                f"not match target hidden_size ({target_hidden}). The draft's "
                "input_proj is shaped (2 * hidden_size, hidden_size); a "
                "mismatch would crash inside input_proj at the first "
                "prefill. Retrain the draft against the current target."
            )

        block_size = (
            spec_config.num_tokens
            if spec_config.num_tokens is not None
            else draft_config.block_size
        )
        # ``target_layer_id`` (optional, recorded by ``olmlx eagle prepare``
        # at training time) tells the decoder which target layer to hook.
        # MUST match the layer the draft was trained against — feeding
        # the draft hiddens from a different layer at inference produces
        # ~5% acceptance even for an otherwise well-converged draft, since
        # mid-network and post-final-norm hiddens have very different
        # distributions. ``None`` falls back to the decoder's default
        # (last layer) — appropriate for older checkpoints from before
        # this field was recorded.
        target_layer_id_raw = eagle_cfg.get("target_layer_id")
        target_layer_id = (
            int(target_layer_id_raw) if target_layer_id_raw is not None else None
        )
        if target_layer_id is None:
            # Pre-fix checkpoints (trained before
            # ``olmlx eagle prepare`` persisted the layer ID) silently
            # fall back to ``len(layers) - 1``. If the precompute
            # captured a mid-network layer (e.g. 50 of 64), this is the
            # exact configuration that collapsed bench acceptance to
            # ~5% in the original Phase F bench — the operator gets a
            # working-looking but mis-routed checkpoint. Surface it.
            logger.warning(
                "EAGLE draft at %s has no 'target_layer_id' in its "
                "config (likely a pre-fix checkpoint). The decoder will "
                "fall back to the target's last layer; if the draft was "
                "actually trained against a mid-network layer, bench "
                "acceptance will be significantly degraded. Retrain "
                "with `olmlx eagle prepare` against the current target "
                "to get the field persisted into the saved config.",
                config_file,
            )
        return EagleDecoder(
            target_model=target_model,
            draft_model=draft_model,
            block_size=block_size,
            target_layer_id=target_layer_id,
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
        # ``None`` means "no user override"; classic speculative decoding
        # uses 4 as its strategy default.
        num_tokens = spec_config.num_tokens if spec_config.num_tokens is not None else 4
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
            tree_width=settings.tree_width if settings.tree_speculative else 1,
            tree_max_nodes=settings.tree_max_nodes,
        )

    def _load_pld_decoder(
        self,
        target_model: Any,
        spec_config: SpeculativeConfig,
        *,
        is_vlm: bool = False,
    ) -> Any:
        """Construct a PromptLookupDecoder (no draft model required).

        For VLM targets, decoder runs on the unwrapped language model so
        the prompt-cache state is the same one mlx-vlm's generate would
        touch. All PLD knobs (max-draft, ngram range, lookup window) are
        read from ``spec_config`` so per-model ``models.json`` overrides
        compose with the global ``OLMLX_SPECULATIVE_PLD_*`` env vars
        (``ModelConfig.resolved_speculative`` handles the fallback chain).
        """
        from olmlx.engine.speculative import PromptLookupDecoder

        if not spec_config.enabled:
            raise RuntimeError(
                "_load_pld_decoder called with spec_config.enabled=False"
            )
        # PLD default max-draft is 10 (Saxena's reference value); classic
        # speculative defaults to 4 but the regime is different — PLD's
        # per-step compute is dominated by the target forward whose cost
        # scales sub-linearly with draft length up to a point.
        num_tokens = (
            spec_config.num_tokens if spec_config.num_tokens is not None else 10
        )
        if spec_config.draft_model:
            logger.warning(
                "speculative_strategy='pld' ignores speculative_draft_model "
                "(%s) — PLD has no draft model.",
                spec_config.draft_model,
            )

        if is_vlm:
            pld_target = getattr(target_model, "language_model", None)
            if pld_target is None:
                raise ValueError(
                    "VLM model does not expose .language_model; PLD "
                    "requires direct access to the text decoder"
                )
        else:
            pld_target = target_model

        # ``resolved_speculative`` populates these from the global
        # Settings defaults (3, 1, 8192) when no per-model override is
        # present, so they are never None at this point in normal use.
        # Use explicit ``raise`` rather than ``assert`` so the misuse
        # also surfaces under ``python -O`` (which would otherwise
        # strip the check and let ``None`` fall through to
        # ``PromptLookupDecoder.__init__`` with a confusing ``TypeError``
        # on the ``<`` comparison there).
        if spec_config.pld_max_ngram is None:
            raise ValueError(
                "_load_pld_decoder: spec_config.pld_max_ngram is None; "
                "caller must go through ModelConfig.resolved_speculative()"
            )
        if spec_config.pld_min_ngram is None:
            raise ValueError(
                "_load_pld_decoder: spec_config.pld_min_ngram is None; "
                "caller must go through ModelConfig.resolved_speculative()"
            )
        if spec_config.pld_lookup_window is None:
            raise ValueError(
                "_load_pld_decoder: spec_config.pld_lookup_window is None; "
                "caller must go through ModelConfig.resolved_speculative()"
            )
        logger.info(
            "Constructing PLD decoder (max_draft=%d, ngram=%d..%d, lookup_window=%d)",
            num_tokens,
            spec_config.pld_min_ngram,
            spec_config.pld_max_ngram,
            spec_config.pld_lookup_window,
        )
        return PromptLookupDecoder(
            target_model=pld_target,
            num_speculative_tokens=num_tokens,
            max_ngram_size=spec_config.pld_max_ngram,
            min_ngram_size=spec_config.pld_min_ngram,
            lookup_window=spec_config.pld_lookup_window,
        )

    def _load_self_speculative_decoder(
        self,
        target_model: Any,
        spec_config: SpeculativeConfig,
    ) -> Any:
        """Create a SelfSpeculativeDecoder using the target's own early layers.

        No external draft model is loaded. ``spec_config.layers_skip``
        determines how many layers the draft skips (defaulting to
        ``L // 4`` when ``None``).
        """
        from olmlx.engine.gdn_rollback import get_model_layers
        from olmlx.engine.self_speculative import SelfSpeculativeDecoder

        num_tokens = spec_config.num_tokens if spec_config.num_tokens is not None else 4

        total_layers = len(get_model_layers(target_model))
        if spec_config.layers_skip is not None:
            layers_skip = spec_config.layers_skip
        else:
            layers_skip = max(total_layers // 4, 1)
        num_early_layers = total_layers - layers_skip
        if num_early_layers < 1:
            num_early_layers = 1
            layers_skip = total_layers - 1

        logger.info(
            "Self-speculative: draft uses %d/%d layers (skip=%d, λ=%d)",
            num_early_layers,
            total_layers,
            layers_skip,
            num_tokens,
        )

        return SelfSpeculativeDecoder(
            target_model=target_model,
            num_early_layers=num_early_layers,
            num_speculative_tokens=num_tokens,
        )

    def _is_flash_enabled(self, flash_config: ResolvedFlashConfig) -> bool:
        return flash_config.enabled

    def _load_flash_model(
        self,
        hf_path: str,
        load_path: str,
        flash_dir: Path,
        *,
        model_exp: Any,
        flash_config: ResolvedFlashConfig,
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

        runtime_flash_config = FlashConfig(
            hidden_size=layout_config["hidden_size"],
            intermediate_size=layout_config["intermediate_size"],
            num_layers=layout_config["num_layers"],
            sparsity_threshold=flash_config.sparsity_threshold,
            min_active_neurons=flash_config.min_active_neurons,
            max_active_neurons=flash_config.max_active_neurons,
            window_size=model_exp.flash_window_size,
            io_threads=model_exp.flash_io_threads,
            cache_budget_neurons=model_exp.flash_cache_budget_neurons,
            memory_budget_fraction=flash_config.memory_budget_fraction,
            prefetch=flash_config.prefetch,
            prefetch_confidence_threshold=model_exp.flash_prefetch_confidence_threshold,
            prefetch_min_neurons=model_exp.flash_prefetch_min_neurons,
            prefetch_max_neurons=model_exp.flash_prefetch_max_neurons,
            prefetch_io_threads=model_exp.flash_prefetch_io_threads,
        )

        weight_store = FlashWeightStore(
            flash_dir,
            num_io_threads=runtime_flash_config.io_threads,
            cache_budget_neurons=runtime_flash_config.cache_budget_neurons,
            bypass_cache=model_exp.flash_bypass_os_cache,
            use_preallocated_buffer=model_exp.flash_preallocated_buffer,
        )

        # Load lookahead predictors if available (for speculative prefetching)
        lookahead_bank = None
        lookahead_path = flash_dir / "lookahead_predictors"
        if flash_config.prefetch and lookahead_path.exists():
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
            model, predictor_bank, weight_store, runtime_flash_config, lookahead_bank
        )

        if flash_config.flash_speculative:
            from olmlx.engine.flash.speculative import SpeculativeFlashDecoder

            if not flash_config.flash_speculative_draft_model:
                raise ValueError(
                    "flash_speculative requires flash_speculative_draft_model to be set "
                    "(OLMLX_FLASH_SPECULATIVE_DRAFT_MODEL)"
                )

            logger.info(
                "Loading draft model %s for speculative decoding",
                flash_config.flash_speculative_draft_model,
            )
            draft_model, draft_tokenizer = load_model_with_strict_fallback(
                flash_config.flash_speculative_draft_model, lazy=False
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
                num_speculative_tokens=flash_config.flash_speculative_tokens,
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

    def _is_flash_moe_enabled(self, flash_moe_config: FlashMoeConfig) -> bool:
        return flash_moe_config.enabled

    def _load_flash_moe_model(
        self,
        hf_path: str,
        load_path: str,
        flash_moe_dir: Path,
        *,
        flash_moe_config: FlashMoeConfig,
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
            num_io_threads=flash_moe_config.io_threads,
            cache_budget_experts=flash_moe_config.cache_budget_experts,
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
        flash_config: ResolvedFlashConfig | None = None,
        flash_moe_config: FlashMoeConfig | None = None,
        weight_quant_str: str | None = None,
    ) -> tuple[Any, Any, bool, TemplateCaps, Any]:
        """Load a model, using config.json inspection to choose the right library.

        *model_exp* is the resolved ExperimentalSettings for this model.
        Falls back to global defaults if not provided.

        *spec_config* is the resolved ``SpeculativeConfig`` (per-model
        overrides applied). Falls back to global ``Settings`` values
        when ``None``.

        *flash_config* is the resolved Flash primary-knob config
        (``ModelConfig.resolved_flash()``). Falls back to a fresh
        resolution from global ``Settings`` values when ``None``.

        *flash_moe_config* is the resolved ``FlashMoeConfig`` (per-model
        overrides applied). Falls back to a fresh resolution from
        global ``Settings`` values when ``None``.

        *weight_quant_str* is the resolved weight quantization string
        (e.g. ``"hqq:4"``). ``None`` means no weight quantization.

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
        if flash_config is None:
            from olmlx.engine.registry import ModelConfig

            flash_config = ModelConfig(hf_path=hf_path).resolved_flash()
        if flash_moe_config is None:
            from olmlx.engine.registry import ModelConfig

            flash_moe_config = ModelConfig(hf_path=hf_path).resolved_flash_moe()
        spec_enabled = spec_config.enabled

        # Ensure model is downloaded to the store
        load_path: str = hf_path
        if self.store is not None:
            local_dir = self.store.ensure_downloaded(hf_path)
            load_path = str(local_dir)
            # Backfill manifest.json when missing so show() and list_local()
            # don't need to derive metadata on every access.
            if local_dir.exists() and not (local_dir / "manifest.json").exists():
                from olmlx.models.store import _derive_manifest

                # Find the Ollama short name for this hf_path, falling back
                # to the hf_path itself if no registry entry maps to it.
                manifest_name = self.registry.normalize_name(hf_path)
                for short_name, mc in self.registry.list_models().items():
                    if mc.hf_path == hf_path:
                        manifest_name = short_name
                        break

                manifest = _derive_manifest(
                    local_dir,
                    manifest_name,
                    hf_path,
                )
                manifest.save(local_dir / "manifest.json")

        # Check for flash-MoE-prepared model
        if self._is_flash_moe_enabled(flash_moe_config):
            flash_moe_dir = self._flash_moe_dir(hf_path)
            if flash_moe_dir is not None:
                if spec_enabled and spec_config.strategy in ("dflash", "eagle"):
                    # Both feature-conditioned strategies route through
                    # ``_load_dflash_decoder`` / ``_load_eagle_decoder``
                    # which expect a dense target; the Flash-MoE wrapper
                    # changes the forward path enough that the hidden-
                    # state hooks and GDN-rollback assumptions don't
                    # apply. Without this guard, ``eagle`` would silently
                    # fall through to ``_load_speculative_decoder`` and
                    # run classic speculative instead — a confusing
                    # behavioural mismatch with the user's setting.
                    raise ValueError(
                        f"speculative_strategy={spec_config.strategy!r} is "
                        "not supported on Flash-MoE targets. Use "
                        "speculative_strategy='classic' or "
                        "'self_speculative', or remove the speculative "
                        "settings."
                    )
                if flash_config.flash_speculative:
                    raise ValueError(
                        "flash_speculative is not supported on Flash-MoE "
                        "targets. Remove flash_speculative from models.json "
                        "(or unset OLMLX_FLASH_SPECULATIVE) and "
                        "use speculative instead."
                    )
                if weight_quant_str:
                    logger.warning(
                        "weight_quant=%s is set but %s is loaded via "
                        "Flash-MoE; weight quantization is not applied on "
                        "the Flash-MoE code path.",
                        weight_quant_str,
                        hf_path,
                    )
                model, tokenizer, is_vlm, caps = self._load_flash_moe_model(
                    hf_path, load_path, flash_moe_dir, flash_moe_config=flash_moe_config
                )
                if spec_enabled:
                    if spec_config.strategy == "pld":
                        decoder = self._load_pld_decoder(
                            model, spec_config, is_vlm=is_vlm
                        )
                    elif spec_config.strategy == "self_speculative":
                        decoder = self._load_self_speculative_decoder(
                            model, spec_config
                        )
                    else:
                        decoder = self._load_speculative_decoder(
                            model, hf_path, spec_config, is_vlm=is_vlm
                        )
                    return model, tokenizer, is_vlm, caps, decoder
                return model, tokenizer, is_vlm, caps, None

        # Check for flash-prepared model
        if self._is_flash_enabled(flash_config):
            flash_dir = self._flash_dir(hf_path)
            if flash_dir is not None:
                # PLD has no draft model and can ride on top of a Flash
                # wrapper without conflicting with the Flash speculative
                # path — wire it through after the wrap. Other strategies
                # collide with ``flash_speculative`` and are warned/ignored
                # to preserve the prior behaviour.
                pld_requested = spec_enabled and spec_config.strategy == "pld"
                # Non-PLD strategies still get ignored on Flash regardless
                # of whether ``flash_speculative`` is also set — the user's
                # ``OLMLX_SPECULATIVE`` choice doesn't take effect either
                # way, so the warning must fire. ``flash_speculative`` is
                # an orthogonal concern (it picks the *flash-side*
                # speculative implementation; classic/dflash/eagle are
                # never honoured on a Flash target).
                if spec_enabled and not pld_requested:
                    logger.warning(
                        "OLMLX_SPECULATIVE is enabled but %s is loaded via "
                        "Flash, which uses OLMLX_FLASH_SPECULATIVE "
                        "for speculative decoding; the OLMLX_SPECULATIVE "
                        "setting will be ignored.",
                        hf_path,
                    )
                if pld_requested and flash_config.flash_speculative:
                    raise ValueError(
                        "Cannot enable both flash_speculative and "
                        "speculative_strategy='pld' on the same model. "
                        "Pick one."
                    )
                if weight_quant_str:
                    logger.warning(
                        "weight_quant=%s is set but %s is loaded via "
                        "Flash; weight quantization is not applied on "
                        "the Flash code path.",
                        weight_quant_str,
                        hf_path,
                    )
                model, tokenizer, is_vlm, caps, decoder = self._load_flash_model(
                    hf_path,
                    load_path,
                    flash_dir,
                    model_exp=model_exp,
                    flash_config=flash_config,
                )
                if pld_requested:
                    decoder = self._load_pld_decoder(model, spec_config, is_vlm=is_vlm)
                return model, tokenizer, is_vlm, caps, decoder

        kind = self._detect_model_kind(hf_path)
        logger.info("Detected model kind for %s: %s", hf_path, kind)

        if kind == "whisper":
            # Whisper STT (issue #366). Load via mlx-whisper's loader and
            # return no tokenizer/caps/speculative — the transcription path
            # drives mlx_whisper.transcribe() directly.
            import mlx.core as mx
            import mlx_whisper.load_models as whisper_loader

            model = whisper_loader.load_model(load_path, dtype=mx.float16)
            return model, None, False, TemplateCaps(), None

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

                # Apply HQQ weight quantization if configured
                self._maybe_quantize_model(model, True, weight_quant_str, hf_path)

                if spec_enabled and spec_config.strategy in ("dflash", "eagle"):
                    raise ValueError(
                        f"speculative_strategy={spec_config.strategy!r} is not "
                        "supported on VLM targets. Use "
                        "speculative_strategy='classic' or "
                        "'self_speculative', or remove the speculative "
                        "settings."
                    )
                if flash_config.flash_speculative:
                    raise ValueError(
                        "flash_speculative is not supported on VLM targets; "
                        "remove flash_speculative from models.json (or unset "
                        "OLMLX_FLASH_SPECULATIVE) and use "
                        "speculative instead"
                    )
                if spec_enabled:
                    if spec_config.strategy == "pld":
                        decoder = self._load_pld_decoder(
                            model, spec_config, is_vlm=True
                        )
                    elif spec_config.strategy == "self_speculative":
                        spec_target = getattr(model, "language_model", model)
                        decoder = self._load_self_speculative_decoder(
                            spec_target, spec_config
                        )
                    else:
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

        # Apply HQQ weight quantization if configured.
        # The text path (including hybrid VLMs routed through mlx-lm)
        # loads a plain text model — pass is_vlm=False so we quantize
        # the model directly without looking for .language_model.
        self._maybe_quantize_model(model, False, weight_quant_str, hf_path)

        if spec_enabled and flash_config.flash_speculative:
            raise ValueError(
                "Only one speculative decoder can be active at a time; "
                "got both 'speculative' and 'flash_speculative'."
            )

        if spec_enabled:
            if spec_config.strategy == "dflash":
                decoder = self._load_dflash_decoder(model, spec_config)
            elif spec_config.strategy == "eagle":
                decoder = self._load_eagle_decoder(model, spec_config)
            elif spec_config.strategy == "pld":
                decoder = self._load_pld_decoder(model, spec_config)
            elif spec_config.strategy == "self_speculative":
                decoder = self._load_self_speculative_decoder(model, spec_config)
            else:
                decoder = self._load_speculative_decoder(model, hf_path, spec_config)
            return model, tokenizer, is_vlm, caps, decoder

        return model, tokenizer, is_vlm, caps, None

    @staticmethod
    def _maybe_quantize_model(
        model: Any,
        is_vlm: bool,
        weight_quant_str: str | None,
        hf_path: str,
    ) -> None:
        """Apply HQQ weight quantization to the model if configured.

        For VLM models, quantizes the language model portion only.
        The vision component is left untouched.
        """
        if not weight_quant_str:
            return
        from olmlx.engine.hqq.quantize import HQQConfig, quantize_model

        cfg = HQQConfig.from_string(weight_quant_str)
        if cfg is None:
            return

        if is_vlm:
            if hasattr(model, "language_model"):
                target = model.language_model
            else:
                logger.warning(
                    "weight_quant=%s configured for VLM %s but model has no "
                    "language_model attribute; quantizing the full model",
                    weight_quant_str,
                    hf_path,
                )
                target = model
        else:
            target = model

        logger.info(
            "Applying HQQ weight quantization (%d-bit, group_size=%d) to %s",
            cfg.bits,
            cfg.group_size,
            hf_path,
        )
        quantize_model(target, cfg)
        mx.eval(target.parameters())

    def _load_model_and_shard(
        self,
        hf_path: str,
        model_exp: Any = None,
        spec_config: SpeculativeConfig | None = None,
        flash_config: ResolvedFlashConfig | None = None,
        flash_moe_config: FlashMoeConfig | None = None,
        weight_quant_str: str | None = None,
    ) -> tuple[Any, Any, bool, TemplateCaps, bool, Any]:
        """Load a model and optionally shard it for distributed inference.

        *model_exp* is the resolved ExperimentalSettings for this model
        (global defaults merged with per-model overrides).

        *spec_config* is the resolved speculative config tuple
        ``(enabled, draft_model, num_tokens)``.

        *flash_config* is the resolved Flash primary-knob config.

        *weight_quant_str* is the resolved weight quantization string
        (e.g. ``"hqq:4"``). ``None`` means no weight quantization.

        Returns (model, tokenizer, is_vlm, caps, is_distributed, speculative_decoder).
        """
        model, tokenizer, is_vlm, caps, speculative_decoder = self._load_model(
            hf_path,
            model_exp=model_exp,
            spec_config=spec_config,
            flash_config=flash_config,
            flash_moe_config=flash_moe_config,
            weight_quant_str=weight_quant_str,
        )
        is_distributed = False

        if self._distributed_group is not None:
            if is_vlm:
                del model, tokenizer
                self._flush_metal_sync()
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
                    self._flush_metal_sync()
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
                    self._flush_metal_sync()
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
                self._flush_metal_sync()
                raise AssertionError(
                    "unreachable: distributed_strategy is Literal['tensor']; "
                    "pydantic rejects any other value at config parse"
                )

        return model, tokenizer, is_vlm, caps, is_distributed, speculative_decoder

    async def _expire_stale(self):
        """Unload models whose keep-alive has expired (active_refs == 0)."""
        now = time.time()
        # Pop expired entries under the lock, then close them after releasing
        # it. _close_loaded_model calls executor.shutdown(wait=True) which
        # blocks the event loop until the pool drains; holding self._lock
        # during that would stall every concurrent ensure_loaded() caller.
        # Models with active_refs > 0 are skipped — they are currently
        # serving requests. Even if a model slips through with
        # active_refs == 0 between ensure_loaded() and _inference_ref(),
        # the caller still holds a Python reference, so the model/
        # tokenizer stay alive; only the _loaded dict entry is removed.
        async with self._lock:
            expired_lms: list[LoadedModel] = []
            for name, lm in list(self._loaded.items()):
                if (
                    lm.expires_at is not None
                    and lm.expires_at <= now
                    and lm.active_refs == 0
                ):
                    logger.info("Unloading expired model %s", name)
                    expired_lms.append(self._loaded.pop(name))

        # Close outside the lock AND off the event loop thread.
        # ``_close_loaded_model`` calls ``executor.shutdown(wait=True)`` on
        # the prefetcher (16 threads) and weight store (32 threads); that
        # join is synchronous, so running it on the event loop would stall
        # every concurrent coroutine until the pools drained. ``to_thread``
        # offloads the join. Per-model isolation: one failing close() must
        # not skip the remaining expired models — _close_loaded_model
        # already logs per-resource, so callers absorb silently.
        # Pop from the list so no live reference (including the loop
        # variable) survives into the gc.collect() below — otherwise the
        # Metal buffers attached to the LoadedModel can't be reclaimed and
        # mx.clear_cache() runs against nothing. Same reason
        # _evict_lru_if_needed calls ``del evicted``.
        flush = bool(expired_lms) and not self._pending_cleanups
        while expired_lms:
            lm = expired_lms.pop()
            try:
                await asyncio.to_thread(self._close_loaded_model, lm)
            except ExceptionGroup:
                pass  # already logged per-resource inside _close_loaded_model
            del lm

        # Flush Metal allocator cache so freed buffers don't inflate the
        # next ensure_loaded() memory check. Mirrors _evict_lru_if_needed.
        # Skip when any deferred cleanup is pending — a background thread
        # for a different model may still be allocating Metal memory.
        if flush:
            await self._flush_metal()

    async def _check_expiry_loop(self):
        while True:
            await asyncio.sleep(30)
            # Guard the while True: any unhandled exception here would
            # permanently kill the background expiry task and leak models
            # indefinitely. Per-model errors are already absorbed inside
            # ``_expire_stale``; this catch is a belt-and-braces for the
            # surrounding async machinery (CancelledError still propagates).
            try:
                await self._expire_stale()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Expiry check failed")
