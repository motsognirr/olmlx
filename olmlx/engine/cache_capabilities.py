"""KV-cache capability predicates (extracted from model_manager.py)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING


from olmlx.engine.prompt_cache import CachedPromptState, PromptCacheStore  # noqa: F401

# Speculative-decoder draft loaders live in a focused mixin module (#454).
# ``_resolve_attention_causal`` is re-exported for back-compat with tests that
# import it from here.
from olmlx.engine.speculative_loaders import (
    _resolve_attention_causal as _resolve_attention_causal,
)

if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


def _is_serializable_cache(cache: list) -> bool:
    """Check if a cache list can be serialized with mlx-lm's save_prompt_cache."""
    from olmlx.engine.shardquant_cache import ShardKVCache
    from olmlx.engine.spectralquant_cache import SpectralQuantKVCache
    from olmlx.engine.turboquant_cache import TurboQuantKVCache

    return not any(
        isinstance(c, (TurboQuantKVCache, SpectralQuantKVCache, ShardKVCache))
        for c in cache
    )


# KV-quant config string prefixes whose backing cache class cannot be
# deep-copied by ``snapshot_cache_for_persistence``.  Currently empty:
# - ``turboquant:`` is handled by ``TurboQuantKVCache.__deepcopy__`` which
#   shares the ``mx.Dtype`` singleton by reference and eager-evals the
#   private dequant side buffers that ``flatten_cache_state`` doesn't see.
# - ``spectral:`` deepcopies cleanly via the default walk (no ``mx.Dtype``
#   attribute, no out-of-state side buffers).
# - ``shard:`` likewise deepcopies via the default walk: ShardKVCache keeps
#   no ``mx.Dtype`` attribute and exposes every mutable array (sink, window,
#   packed middle, norms) through ``state``, so the flatten + ``mx.eval``
#   pass materializes its in-place-write graphs before the copy.
# Kept as the registration point for any future kv-quant cache class
# that proves unsafe — mirrors ``_EXCLUDED_MIXED_LAYER_PAIRS``.  Disk-save
# remains gated separately by ``_is_serializable_cache`` because that path
# uses safetensors and the packed indices/codebook layout has no
# upstream-compatible serialisation.
_KV_QUANT_PREFIXES_BLOCKING_SNAPSHOT: tuple[str, ...] = ()


def _kv_quant_blocks_snapshot(kv_cache_quant: str | None) -> bool:
    """True iff ``kv_cache_quant`` selects a quantization wrapper whose
    layer state can't be deep-copied (the checkpoint path requirement).

    Prefix-based because the probe constructs the bare (unquantized) cache
    by design — see the comment on ``_probe_cache_capabilities`` — so we
    can't check via ``isinstance`` like ``_is_serializable_cache`` does.
    """
    if kv_cache_quant is None:
        return False
    return any(
        kv_cache_quant.startswith(prefix)
        for prefix in _KV_QUANT_PREFIXES_BLOCKING_SNAPSHOT
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


# Layer classes that are safe to persist via the message-boundary
# CHECKPOINT path (not the flat path).  Strict superset of
# _PERSISTABLE_CACHE_CLASSES: RotatingKVCache is safe because the
# checkpoint mechanism avoids trim entirely (#343); ArraysCache is
# safe when the snapshot helper materialises the lazy gated_delta_kernel
# graph via mx.eval before storage (#284).
_PERSISTABLE_CACHE_CLASSES_WITH_CHECKPOINT = frozenset(
    {
        "KVCache",
        "QuantizedKVCache",
        "ConcatenateKVCache",
        "RotatingKVCache",
        "ChunkedKVCache",
        "ArraysCache",
    }
)

# Layer-class compositions that are individually safe via the checkpoint
# path but combine into a layout we can't currently snapshot consistently.
# Empty since #396 was lifted: the Qwen3-Next mix (RotatingKVCache +
# ArraysCache) is handled by ``snapshot_cache_for_persistence`` — each
# layer is deepcopied independently and ArraysCache lazy state is
# materialized via ``mx.eval`` before the deepcopy, so the joint snapshot
# captures both sub-states at the same logical token offset and survives
# a continued forward pass after restore.  Kept as the registration point
# for any future composition that proves unsafe.
_EXCLUDED_MIXED_LAYER_PAIRS: frozenset[frozenset[str]] = frozenset()


def _layer_layout_is_mixed_excluded(cache_list: list) -> bool:
    """True iff the layer-class set is in the excluded mixed-pair list."""
    if not cache_list:
        return False
    classes = {type(layer).__name__ for layer in cache_list}
    for excluded in _EXCLUDED_MIXED_LAYER_PAIRS:
        if excluded.issubset(classes):
            return True
    return False


def _cache_supports_checkpoint_persistence(cache_list: list) -> bool:
    """True iff every layer is checkpoint-persistable AND the layout is
    not in the excluded mixed-pair list.
    """
    if not cache_list:
        return False
    if _layer_layout_is_mixed_excluded(cache_list):
        return False
    return all(
        type(layer).__name__ in _PERSISTABLE_CACHE_CLASSES_WITH_CHECKPOINT
        for layer in cache_list
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
