"""Model-agnostic speculative decoding.

Uses a small draft model for fast candidate generation, then verifies
candidates with the target model in a single forward pass.

This base class contains the core algorithm without Flash-specific
dependencies. See engine/flash/speculative.py for the Flash-aware
subclass that adds prefetching and neuron window sizing.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn

try:
    from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
except ImportError:
    make_prompt_cache = None  # type: ignore[assignment]
    trim_prompt_cache = None  # type: ignore[assignment]

from olmlx.engine.gdn_rollback import (
    GDNBuffer,
    GDNStateCapture,
    find_gdn_class,
)
from olmlx.engine.prompt_cache.checkpoint import snapshot_cache_for_persistence
from olmlx.engine.spec_decoder_base import (
    SpecDecoderBase,
    # Canonical home moved to spec_decoder_base (#467); re-exported here
    # for remaining importers (tests; the decoders verify via the base's
    # ``_verify_greedy`` instead).
    _trim_recent_cache,
    verify_draft_greedy as verify_draft_greedy,
)
from olmlx.utils import tracing as _tracing

logger = logging.getLogger(__name__)


# Sub-chunk size for prefill forward passes. Matches mlx-lm's
# ``generate_step`` ``prefill_step_size`` default so the speculative decoder
# bounds activation memory the same way mlx-lm's native prefill does. A single
# forward over a long prompt OOMs Metal — the attention-score intermediate for
# a ~38k-token prefill exceeds the ~41 GB single-buffer limit (observed on
# Qwen3.6-35B-A3B-6bit). Speculative decoders own their own prefill (prompt
# caching is gated off for them), so they don't inherit the prompt-cache path's
# segmented-prefill chunking — this is the equivalent bound for that path.
_PREFILL_CHUNK = 2048


class PrefillCancelled(Exception):
    """Raised by the prefill helpers when ``cancel_event`` is set mid-prefill.

    A long speculative prefill (e.g. a ~69k-token agentic prompt) is otherwise
    non-interruptible: the decode loop checks ``cancel_event`` between steps,
    but ``prefill()`` ran to completion regardless, pinning the GPU and the
    inference lock for minutes after a client disconnect (and 503-ing the next
    request via deferred cleanup). Checking ``cancel_event`` at each sub-chunk
    boundary and raising this lets ``speculative_stream_generate`` exit cleanly.
    """


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    """Length of the longest shared leading run of ``a`` and ``b``.

    Pure-Python (small inputs: a handful of cross-request cache lineages,
    not the full prompt every step). Mirrors mlx-lm's ``common_prefix_len``
    semantics without pulling its array-oriented import into this module.
    """
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


@dataclass
class _SpecCacheEntry:
    """One persisted speculative-cache lineage.

    ``tokens`` is the token sequence the snapshot represents (it always ends
    at a message boundary). ``payload`` is decoder-specific snapshot state —
    a ``(target_snap, draft_snap)`` pair for :class:`SpeculativeDecoder`, or a
    single target snapshot for :class:`PromptLookupDecoder`. The store treats
    it as opaque; only the owning decoder interprets it.
    """

    tokens: list[int]
    payload: Any = field(default=None)


class _SpecCacheStore:
    """Tiny LRU of speculative KV snapshots keyed by longest token-prefix.

    Held on a single per-model decoder instance; all inference is serialized
    under the inference lock, so the store needs no internal locking. Lookup
    is a longest-common-prefix linear scan over the (capacity-bounded) entry
    set — no ``cache_id`` dependency, which makes reuse robust to unstable
    client-supplied ids (the same problem the non-speculative path's radix
    index solves; at 2–4 slots a trie is unnecessary).

    ``capacity == 0`` disables reuse entirely (``enabled()`` is False and
    ``insert`` is a no-op) — the ``OLMLX_SPECULATIVE_CACHE_SLOTS=0`` kill
    switch that restores the previous fresh-prefill-every-turn behavior.
    """

    def __init__(self, capacity: int) -> None:
        self._capacity = max(int(capacity), 0)
        # LRU order: index 0 is the least-recently-used entry (evicted first),
        # the last index is most-recently-used.
        self._entries: list[_SpecCacheEntry] = []
        # Guard backing _serialized() — never blocked on, only try-acquired.
        self._access_guard = threading.Lock()

    @contextmanager
    def _serialized(self) -> Iterator[None]:
        """Enforce the no-concurrent-access contract explicitly (#463).

        Unlike the loop-bound prompt-cache stores, this store legitimately
        runs on the decode worker thread — its safety comes from the
        inference lock serializing all access, not from loop affinity.  A
        non-blocking acquire turns an overlapping access (a refactor that
        races the decode thread against another caller) into an immediate
        RuntimeError instead of silent LRU-list corruption.
        """
        # Deliberately a plain Lock, not RLock: same-thread reentry (a future
        # find → ... → find/clear chain mutating mid-scan) is just as unsafe
        # as cross-thread overlap, so it should raise too, not silently pass.
        if not self._access_guard.acquire(blocking=False):
            raise RuntimeError(
                "_SpecCacheStore accessed concurrently or re-entered "
                "mid-operation; accesses must be serialized under the "
                "inference lock (issue #463)"
            )
        try:
            yield
        finally:
            self._access_guard.release()

    @property
    def capacity(self) -> int:
        return self._capacity

    def enabled(self) -> bool:
        return self._capacity > 0

    def clear(self) -> None:
        with self._serialized():
            self._entries = []

    def find(self, tokens: list[int]) -> tuple[_SpecCacheEntry, int] | None:
        """Return ``(entry, common_prefix_len)`` for the entry whose stored
        tokens share the longest leading run with ``tokens``, promoting that
        entry to most-recently-used. ``None`` when the store is empty/disabled
        or no entry shares any prefix.

        Note: promotion happens on any prefix overlap, even one the caller
        later rejects (e.g. a non-trimmable partial-prefix hit that falls back
        to fresh prefill). At the small default slot counts this near-miss
        promotion is immaterial; revisit if slots are raised substantially.
        """
        with self._serialized():
            if not self.enabled() or not self._entries or not tokens:
                return None
            best: _SpecCacheEntry | None = None
            best_common = 0
            for entry in self._entries:
                common = _common_prefix_len(entry.tokens, tokens)
                if common > best_common:
                    best_common = common
                    best = entry
            if best is None or best_common == 0:
                return None
            self._entries.remove(best)
            self._entries.append(best)
            return best, best_common

    def insert(self, tokens: list[int], payload: Any) -> None:
        """Store ``payload`` under ``tokens`` as the most-recently-used entry,
        replacing any existing entry with identical tokens, and evict the
        least-recently-used entries past ``capacity``. No-op when disabled.
        """
        with self._serialized():
            if not self.enabled():
                return
            toks = list(tokens)
            # Refresh: drop any prior entry for the same lineage so a re-prefill
            # of the same prompt doesn't consume two slots.
            self._entries = [e for e in self._entries if e.tokens != toks]
            self._entries.append(_SpecCacheEntry(tokens=toks, payload=payload))
            while len(self._entries) > self._capacity:
                self._entries.pop(0)


# Cache-layer class names used to classify a working cache for reuse.
# Kept local (not imported from model_manager, which imports this module) to
# avoid a circular import; mirrors model_manager's allowlists.
_TRIMMABLE_CACHE_NAMES = frozenset(
    {"KVCache", "QuantizedKVCache", "ConcatenateKVCache"}
)
_LAZY_STATE_CACHE_NAMES = frozenset({"ArraysCache"})


def _cache_is_trimmable(cache: list) -> bool:
    """True iff every layer is in the known-trimmable allowlist (so
    ``trim_prompt_cache`` can realign a reused snapshot to a shorter prefix)."""
    return bool(cache) and all(
        type(layer).__name__ in _TRIMMABLE_CACHE_NAMES for layer in cache
    )


def _cache_has_lazy_state(cache: list) -> bool:
    """True iff any layer carries a lazy ``gated_delta_kernel`` graph
    (``ArraysCache``) that must be ``mx.eval``-materialized before a snapshot
    crosses the request/worker-thread boundary (#284 hazard family)."""
    return any(type(layer).__name__ in _LAZY_STATE_CACHE_NAMES for layer in cache)


def _is_pure_rotating_cache(cache: list) -> bool:
    """True iff the cache is a sliding-window layout (``RotatingKVCache``) with
    no ``ArraysCache`` layers. These (gpt-oss, Step-3.5, Gemma 3) are fragile
    under interior-boundary prefill splits, so speculative cache reuse skips
    them and keeps the legacy fresh-prefill path."""
    names = {type(layer).__name__ for layer in cache}
    return "RotatingKVCache" in names and "ArraysCache" not in names


def _spec_reuse_decision(
    is_trimmable: bool,
    entry_tokens: list[int],
    common: int,
    prompt_len: int,
) -> tuple[bool, int]:
    """Decide whether a stored snapshot is reusable and to what depth.

    Returns ``(reuse, already_covered)``:

    - **Trimmable** caches can roll back to any shared-prefix depth, so reuse
      up to ``common`` — backing up one position on an exact full match
      because the prefill needs at least one trailing token to forward for the
      seeding logit. A single-token prompt yields nothing reusable.
    - **Non-trimmable** (hybrid ``ArraysCache``) caches can only *extend* a
      stored state, never trim it: reuse only when the stored tokens are a
      strict full prefix of the new prompt (``common == len(entry_tokens)``)
      and there is at least one new token to forward. An exact full match is
      discarded (no way to back up one position) — fresh prefill instead.
    """
    if is_trimmable:
        already_covered = min(common, prompt_len - 1)
        return (already_covered > 0, max(already_covered, 0))
    # Non-trimmable: strict full-prefix extension only.
    if common != len(entry_tokens):
        return (False, 0)
    already_covered = common
    if already_covered <= 0 or already_covered >= prompt_len:
        return (False, 0)
    return (True, already_covered)


def _logits(out: Any) -> mx.array:
    # mlx-vlm's language_model returns LanguageModelOutput(logits=...);
    # mlx-lm models return a raw mx.array.
    return cast(mx.array, getattr(out, "logits", out))


def _eval_cache(cache: list) -> None:
    """Materialise KV cache state without evaluating logits.

    Handles KVCache (.keys/.values as mx.array), quantised caches that wrap
    packed data in a list/tuple, and ArraysCache-style caches (.state
    returning a list of tensors).
    """
    arrs = []
    for c in cache:
        k = getattr(c, "keys", None)
        v = getattr(c, "values", None)
        if isinstance(k, mx.array):
            arrs.append(k)
        elif isinstance(k, (list, tuple)):
            arrs.extend(a for a in k if isinstance(a, mx.array))
        if isinstance(v, mx.array):
            arrs.append(v)
        elif isinstance(v, (list, tuple)):
            arrs.extend(a for a in v if isinstance(a, mx.array))
        if k is None and v is None:
            # ArraysCache and similar: state is a list/tuple of arrays
            state = getattr(c, "state", None)
            if isinstance(state, (list, tuple)):
                arrs.extend(a for a in state if isinstance(a, mx.array))
        # TurboQuantKVCache maintains a dequantized side buffer in
        # _key_dequant / _value_dequant that update_and_fetch returns
        # views over. The buffer is not part of .state, so probe it
        # explicitly to force the dequant graph here instead of letting
        # it fuse into pass 2. (SpectralQuantKVCache dequantizes fresh on
        # each call and is already covered by the .state branch above.)
        for attr in ("_key_dequant", "_value_dequant"):
            buf = getattr(c, attr, None)
            if isinstance(buf, mx.array):
                arrs.append(buf)
    if arrs:
        mx.eval(*arrs)
    elif cache:
        # A non-empty cache produced no arrays to evaluate (unrecognised
        # cache type). MLX's lazy graph chains pass-1's cache mutations
        # into pass-2's forward regardless, so tokens stay correct — but
        # the OOM this helper was designed to prevent will resurface
        # (pass-2's eval pulls pass-1's lm_head graph through). Warn so
        # the gap is visible; do not raise, since hard-failing here would
        # break speculative decoding for every new mlx-lm cache type
        # before its probe lands in this function.
        logger.error(
            "_eval_cache: no mx.array entries found in %d cache objects "
            "(types: %s); the OOM-avoidance graph break is a no-op for "
            "this cache type — add an explicit probe here if Metal OOMs "
            "during prefill on long prompts.",
            len(cache),
            sorted({type(c).__name__ for c in cache}),
        )


def _chunked_prefill(
    model: Any,
    tokens: mx.array,
    cache: list,
    cancel_event: threading.Event | None = None,
) -> None:
    """Run ``model`` over ``tokens`` (shape (1, T)) to populate ``cache``,
    in sub-chunks of at most ``_PREFILL_CHUNK`` tokens.

    Logits are discarded; cache state is materialised (and the Metal buffer
    cache cleared) after every sub-chunk so peak activation memory stays at
    roughly one sub-chunk's worth instead of the whole prompt's. Mirrors
    mlx-lm's prefill loop — without this, a single forward over a long prompt
    OOMs Metal (the attention-score intermediate exceeds the ~41 GB
    single-buffer limit). Standard KV-cached attention is chunking-invariant,
    so the resulting cache state is identical to a single forward.

    If ``cancel_event`` is set, raises :class:`PrefillCancelled` at the next
    sub-chunk boundary (checked before each forward), bounding the post-cancel
    GPU work to at most one sub-chunk instead of the whole prompt.
    """
    n = tokens.shape[1]
    pos = 0
    while pos < n:
        if cancel_event is not None and cancel_event.is_set():
            raise PrefillCancelled()
        stop = min(pos + _PREFILL_CHUNK, n)
        model(tokens[:, pos:stop], cache=cache)
        _eval_cache(cache)
        mx.clear_cache()
        pos = stop


def _prefill_last_logit(
    model: Any,
    prompt: mx.array,
    cache: list,
    cancel_event: threading.Event | None = None,
) -> mx.array:
    """Return the logit for the last prompt position without materialising
    the full [batch, seq_len, vocab] tensor.

    For prompts longer than 1 token, runs two passes:
    - Pass 1: prefix[:-1] fills the KV cache; the model output is discarded
      so MLX's lazy evaluation never computes lm_head on seq_len-1 positions.
    - Pass 2: last token produces a [1, 1, vocab] logit; safe to evaluate.

    This avoids the Metal OOM that occurs when seq_len × vocab_size exceeds
    the ~41 GB Metal buffer limit (e.g. Gemma 4 31B, vocab=262 144, long ctx).

    The returned logit is lazy. Cache state is materialised between passes
    (graph-break for OOM avoidance) but the caller is expected to evaluate
    the returned logit, which transitively forces pass-2's cache state.
    """
    if prompt.shape[1] <= 1:
        if cancel_event is not None and cancel_event.is_set():
            raise PrefillCancelled()
        return _logits(model(prompt, cache=cache))[0, -1, :]
    prefix, last = prompt[:, :-1], prompt[:, -1:]
    # Pass 1: fill the KV cache from the prefix in bounded sub-chunks. Output
    # is discarded (lm_head never materialised over seq_len-1 positions) and
    # cache state is materialised between passes for OOM avoidance.
    _chunked_prefill(model, prefix, cache, cancel_event=cancel_event)
    # Cancel may have fired on the final prefix sub-chunk; skip the trailing
    # forward too so post-cancel work stays bounded to one sub-chunk.
    if cancel_event is not None and cancel_event.is_set():
        raise PrefillCancelled()
    return _logits(model(last, cache=cache))[0, 0, :]


def _drive_spec_prefill(
    *,
    flat: list[int],
    boundaries: list[int],
    already_covered: int,
    lanes: list[tuple[Any, list]],
    cancel_event: threading.Event | None,
    on_boundary: Any,
) -> mx.array:
    """Forward each ``(model, cache)`` lane over ``flat[already_covered:]`` and
    return ``lanes[0]`` (the target)'s last-position logit (lazy).

    Snapshots happen via ``on_boundary(deepest_boundary)`` — invoked once, after
    every lane has been filled up to the deepest interior message boundary but
    before the final span — so the caller can persist all caches at a clean
    boundary depth for the next turn. When no interior boundary lies in
    ``(already_covered, len(flat))`` the whole span runs in one pass and
    ``on_boundary`` is not called.

    Runs on the **default stream** (no ``mx.stream`` wrapper) — the same stream
    ``step()`` decodes on. ``inference.py``'s ``_drive_segmented_prefill`` is
    *replicated* rather than reused because it forces ``generation_stream``, and
    splitting prefill/decode across streams reintroduces the GatedDeltaNet /
    MoE-routing corruption on Qwen3-Next-family targets (#284 / #396 family).
    Under mlx's thread-local streams (>=0.31.2, #499) "the default stream"
    means this worker thread's own default stream — still correct here
    because the whole decoder (this prefill drive and every later ``step()``)
    runs start-to-finish on one worker thread; the invariant this docstring
    protects (prefill and decode never straddle two streams) would break the
    same way it always did if that thread affinity were ever lost, just via
    a "no Stream in current thread" crash instead of silent corruption.
    Each sub-chunk is bounded at ``_PREFILL_CHUNK`` and ``cancel_event`` is
    checked at every boundary so a client disconnect interrupts within one
    sub-chunk.
    """
    n = len(flat)
    deepest: int | None = None
    for b in boundaries:
        if already_covered < b < n:
            deepest = b

    def _arr(start: int, end: int) -> mx.array:
        return mx.array(flat[start:end], dtype=mx.int32)[None, :]

    def _fill(model: Any, cache: list, start: int, end: int) -> None:
        pos = start
        while pos < end:
            if cancel_event is not None and cancel_event.is_set():
                raise PrefillCancelled()
            stop = min(pos + _PREFILL_CHUNK, end)
            model(_arr(pos, stop), cache=cache)
            _eval_cache(cache)
            mx.clear_cache()
            pos = stop

    def _target_last_logit(start: int, end: int) -> mx.array:
        # Fill [start, end-1] (logits discarded — keeps lm_head out of the eval
        # graph over the prefix), then forward the final token for the seeding
        # logit (``_prefill_last_logit`` semantics).
        target_model, target_cache = lanes[0]
        _fill(target_model, target_cache, start, end - 1)
        if cancel_event is not None and cancel_event.is_set():
            raise PrefillCancelled()
        return _logits(target_model(_arr(end - 1, end), cache=target_cache))[0, -1, :]

    if deepest is None:
        for model, cache in lanes[1:]:
            _fill(model, cache, already_covered, n)
        return _target_last_logit(already_covered, n)

    # Chunk 1: uncovered prefix up to the deepest interior boundary (all lanes).
    for model, cache in lanes:
        _fill(model, cache, already_covered, deepest)
    on_boundary(deepest)
    # Chunk 2: boundary to the end. Non-target lanes fill KV; the target
    # captures the seeding logit at the final position.
    for model, cache in lanes[1:]:
        _fill(model, cache, deepest, n)
    return _target_last_logit(deepest, n)


class SpeculativeDecoder(SpecDecoderBase):
    """Speculative decoding with a draft model.

    The draft model generates lambda candidate tokens autoregressively.
    The target model verifies all candidates in one forward pass.
    Accepted tokens are returned; on rejection, the target's preferred
    token replaces the first rejected position.

    Supports two modes:
    - Stateless: ``generate_step(prompt)`` — no cross-step caching
    - Cached: ``prefill(prompt)`` then ``step()`` — persistent KV caches

    Not thread-safe: one decoder instance must serve one request at a time.
    """

    def __init__(
        self,
        draft_model: nn.Module,
        target_model: nn.Module,
        num_speculative_tokens: int = 4,
        acceptance_rate_ema: float = 0.9,
        tree_width: int = 1,
        tree_max_nodes: int = 8,
        cache_slots: int = 0,
    ):
        super().__init__()
        if trim_prompt_cache is None:
            raise RuntimeError(
                "trim_prompt_cache is unavailable (mlx-lm import missing); "
                "speculative decoding requires it for correct cache trimming"
            )
        self._draft = draft_model
        self._target = target_model
        self._lambda = num_speculative_tokens
        self._alpha = 0.5  # initial acceptance rate estimate
        self._alpha_ema = acceptance_rate_ema
        self._tree_width = max(tree_width, 1)
        self._tree_max_nodes = max(tree_max_nodes, 3)
        self._use_tree = self._tree_width >= 2

        # Cross-request KV reuse (#421): persists target+draft snapshots keyed
        # by token prefix so each agent turn prefills only the new suffix.
        # ``cache_slots == 0`` disables it (fresh prefill every turn).
        self._cache_store = _SpecCacheStore(cache_slots)
        #: Tokens reused from a stored snapshot on the most recent prefill
        #: (0 when fresh). Exposed for tests / diagnostics.
        self._last_reused_tokens: int = 0

        # Persistent KV cache state (populated by prefill/step)
        self._target_cache: list | None = None
        self._draft_cache: list | None = None
        self._cache_seq_len: int = 0
        self._last_target_logit: mx.array | None = None
        self._pending_token: int | None = None

        # GDN rollback: install a class-level patch on ``GatedDeltaNet``
        # if either model has hybrid linear-attention layers. The single
        # capture instance routes per-call writes to whichever buffer is
        # currently active (target vs draft), so target and draft can
        # share the same ``GDNStateCapture`` when they use the same GDN
        # class (the usual case — e.g. Qwen3.5 target + Qwen3.5 draft).
        # ``find_gdn_class`` returns ``None`` for non-hybrid models.
        # If target and draft use *different* GDN classes (unusual but
        # not impossible — e.g. Qwen3-Coder-Next target + Qwen3.5 draft
        # if they ever ship distinct subclasses), we raise rather than
        # patching two classes silently — one capture per class would
        # require lifting the class-level patch lock.
        self._gdn_capture: GDNStateCapture | None = None
        self._target_gdn_buffer: GDNBuffer | None = None
        self._draft_gdn_buffer: GDNBuffer | None = None
        target_gdn_cls = find_gdn_class(target_model)
        draft_gdn_cls = find_gdn_class(draft_model)
        if target_gdn_cls is not None or draft_gdn_cls is not None:
            if (
                target_gdn_cls is not None
                and draft_gdn_cls is not None
                and target_gdn_cls is not draft_gdn_cls
            ):
                raise NotImplementedError(
                    "Target and draft use different GatedDeltaNet classes "
                    f"({target_gdn_cls.__module__}.{target_gdn_cls.__name__} "
                    f"vs {draft_gdn_cls.__module__}.{draft_gdn_cls.__name__}). "
                    "Classic speculative GDN rollback currently shares one "
                    "class-level patch between target and draft; supporting "
                    "two distinct classes would require lifting the patch "
                    "lock. File an olmlx issue if you hit this."
                )
            gdn_cls = target_gdn_cls if target_gdn_cls is not None else draft_gdn_cls
            # ``assert`` is stripped under ``python -O``; the outer ``if``
            # guarantees at least one of the two is non-None, but a
            # future refactor of that check could silently violate the
            # invariant. Use ``RuntimeError`` so the failure surfaces
            # in production builds too.
            if gdn_cls is None:
                raise RuntimeError(
                    "SpeculativeDecoder GDN-setup invariant violated: "
                    "outer ``if`` branch entered but both target and "
                    "draft GDN classes are None. Please file an olmlx bug."
                )
            self._gdn_capture = GDNStateCapture(gdn_cls)
            # ``create_buffer`` walks the model and can raise (e.g. orphaned
            # GDN modules outside ``get_model_layers``). Close the capture
            # explicitly to release the patch lock — relying on ``__del__``
            # to clean up a partially-constructed decoder leaks the lock
            # until CPython's refcount GC fires, which is fragile under
            # asyncio teardown and blocks any subsequent hybrid load.
            try:
                if target_gdn_cls is not None:
                    self._target_gdn_buffer = self._gdn_capture.create_buffer(
                        target_model
                    )
                if draft_gdn_cls is not None:
                    self._draft_gdn_buffer = self._gdn_capture.create_buffer(
                        draft_model
                    )
            except Exception:
                self._gdn_capture.close()
                self._gdn_capture = None
                self._target_gdn_buffer = None
                self._draft_gdn_buffer = None
                raise

    def close(self) -> None:
        """Release the GDN class-level monkey-patch (idempotent).

        Overrides the base ``close()`` (which is just ``reset()``)
        because the classic decoder's GDN capture and snapshot store are
        decoder-lifetime, not request-lifetime. Decoders should be
        ``close()``-d explicitly when no longer used; the ``__del__``
        finaliser is best-effort because the patched ``__call__`` holds
        a strong reference to ``GDNStateCapture`` through its closure,
        breaking the GC cycle that would otherwise run our finaliser.

        The ``finally`` matters: ``ModelManager`` preserves the decoder
        reference when ``close()`` raises, so a failing capture close
        must not skip dropping the (multi-GB) snapshot store and working
        KV caches that reference would otherwise pin.
        """
        try:
            if self._gdn_capture is not None:
                self._gdn_capture.close()
                self._gdn_capture = None
                self._target_gdn_buffer = None
                self._draft_gdn_buffer = None
        finally:
            # Drop persisted snapshots so a model unload frees their
            # (multi-GB) KV immediately rather than waiting on refcount GC.
            self._cache_store.clear()
            # Release the last request's working KV caches eagerly too —
            # the uniform base lifecycle (close ⊇ reset). Never raises.
            self.reset()

    def _update_acceptance_rate(self, num_accepted: int) -> int:
        """Update the rolling acceptance rate via EMA and return accepted-draft count."""
        assert num_accepted >= 1, (
            "_update_acceptance_rate: _verify() must return at least 1 token"
        )
        num_accepted_draft = min(num_accepted - 1, self._lambda)
        acceptance = num_accepted_draft / max(self._lambda, 1)
        self._alpha = self._alpha_ema * self._alpha + (1 - self._alpha_ema) * acceptance
        return num_accepted_draft

    # ------------------------------------------------------------------
    # Cached API: prefill() + step()
    # ------------------------------------------------------------------

    def _reset_state(self) -> None:
        # Detach the decoder-lifetime GDN capture from any buffer. The base
        # ``step()`` runs this on a mid-step exception, but the capture (and
        # its class-level ``GatedDeltaNet.__call__`` patch) outlives the
        # request; if the buffer stays routed, every subsequent GDN call —
        # including *other* hybrid models' non-speculative inference sharing
        # the patched class — keeps appending into the stale buffer, growing
        # memory without bound until the next ``prefill`` (#633).
        if self._gdn_capture is not None:
            self._gdn_capture.use_buffer(None)
        self._target_cache = None
        self._draft_cache = None
        self._cache_seq_len = 0
        self._last_target_logit = None
        self._pending_token = None
        self._last_reused_tokens = 0
        # NB: ``self._cache_store`` is *not* cleared here — it persists across
        # requests (that is the whole point of #421). Only ``close()`` drops it.

    def _stats_extra(self) -> dict[str, Any]:
        return {
            "ema_acceptance_rate": self._alpha,
            "lambda": self._lambda,
        }

    def _prefill_impl(
        self,
        prompt: mx.array,
        *,
        segmented: Any = None,
        cancel_event: threading.Event | None = None,
    ) -> int:
        """Process the prompt through both models, populating KV caches.

        Args:
            prompt: (1, seq_len) input token IDs.
            segmented: optional :class:`SegmentedPrompt` carrying message
                boundaries. When provided and ``OLMLX_SPECULATIVE_CACHE_SLOTS``
                is non-zero, enables cross-request KV reuse (#421): a stored
                snapshot whose tokens form a prefix of ``prompt`` is reused so
                only the new suffix is prefilled, and a fresh snapshot is taken
                at the deepest interior message boundary for the next turn.
            cancel_event: if set during prefill, raises :class:`PrefillCancelled`
                at the next sub-chunk boundary so a client disconnect interrupts
                a long prefill promptly instead of pinning the GPU.

        Returns:
            The first generated token (from target model's greedy argmax).
        """
        if make_prompt_cache is None or trim_prompt_cache is None:
            raise RuntimeError(
                "mlx_lm.models.cache not available; cannot use cached speculative decoding"
            )

        self._target_cache = make_prompt_cache(self._target)
        self._draft_cache = make_prompt_cache(self._draft)

        # Observed on mlx-vlm 0.4.4 with Qwen3_5: the language model caches
        # `_position_ids` and `_rope_deltas` on the module instance across
        # calls.  Left over from a prior request they produce broadcast
        # mismatches when a new prompt has a different length.  Reset them
        # at the start of each prefill.  If a future VLM caches analogous
        # state under a different attribute name, add it here.
        for attr in ("_position_ids", "_rope_deltas"):
            if hasattr(self._target, attr):
                setattr(self._target, attr, None)

        # Suppress GDN capture during prefill: no rollback is needed
        # for the prompt forward, and recording it would just bloat the
        # buffers (which are sized for one step's worth of captures).
        if self._gdn_capture is not None:
            self._gdn_capture.use_buffer(None)

        # #421 reuse path: only when the store is enabled, the caller
        # supplied message boundaries, and the cache is not a fragile
        # pure-sliding-window layout (whose windowed attention an interior
        # prefill split corrupts — those keep the legacy fresh prefill).
        use_reuse = (
            self._cache_store.enabled()
            and segmented is not None
            and not _is_pure_rotating_cache(self._target_cache)
        )

        if use_reuse:
            first_token = self._prefill_with_reuse(
                prompt, segmented, cancel_event=cancel_event
            )
        else:
            self._last_target_logit = _prefill_last_logit(
                self._target, prompt, self._target_cache, cancel_event=cancel_event
            )
            mx.eval(self._last_target_logit)
            # Populate draft cache; logits not needed. Sub-chunked like the
            # target prefill above so a long prompt doesn't OOM Metal.
            _chunked_prefill(
                self._draft, prompt, self._draft_cache, cancel_event=cancel_event
            )
            self._cache_seq_len = prompt.shape[1]
            first_token = int(mx.argmax(self._last_target_logit).item())

        self._pending_token = first_token
        return first_token

    def _prefill_with_reuse(
        self,
        prompt: mx.array,
        segmented: Any,
        *,
        cancel_event: threading.Event | None,
    ) -> int:
        """Reuse-aware prefill: look up a stored snapshot, continue it for the
        new suffix, and snapshot the deepest interior boundary for next turn.

        Caches (``self._target_cache`` / ``self._draft_cache``) are fresh on
        entry; on a reuse hit they are replaced with deepcopies of the stored
        snapshots (so ``step()``'s in-place mutations never touch the store).
        """
        flat = prompt[0].tolist()
        n = len(flat)

        # Cache layer types are invariant for a given model, so classifying the
        # fresh caches is valid for the (same-type) restored snapshot below.
        is_trimmable = _cache_is_trimmable(self._target_cache) and _cache_is_trimmable(
            self._draft_cache
        )
        already_covered = 0
        hit = self._cache_store.find(flat)
        if hit is not None:
            entry, common = hit
            reuse, covered = _spec_reuse_decision(is_trimmable, entry.tokens, common, n)
            if reuse:
                target_snap, draft_snap = entry.payload
                # Deepcopy the stored snapshots into the working caches so this
                # request's mutations never touch the store. This is a copy of
                # a copy (the store already holds deepcopies) — the unavoidable
                # cost of copy-on-reuse isolation; see config docstring.
                self._target_cache = snapshot_cache_for_persistence(
                    target_snap, eager_eval=_cache_has_lazy_state(target_snap)
                )
                self._draft_cache = snapshot_cache_for_persistence(
                    draft_snap, eager_eval=_cache_has_lazy_state(draft_snap)
                )
                if is_trimmable:
                    trim = len(entry.tokens) - covered
                    if trim > 0:
                        trim_prompt_cache(self._target_cache, trim)
                        trim_prompt_cache(self._draft_cache, trim)
                already_covered = covered

        self._last_target_logit = self._drive_segmented_prefill(
            flat, segmented, already_covered, cancel_event=cancel_event
        )
        mx.eval(self._last_target_logit)
        self._cache_seq_len = n
        self._last_reused_tokens = already_covered
        return int(mx.argmax(self._last_target_logit).item())

    def _drive_segmented_prefill(
        self,
        flat: list[int],
        segmented: Any,
        already_covered: int,
        *,
        cancel_event: threading.Event | None,
    ) -> mx.array:
        """Forward target+draft over ``flat[already_covered:]`` and return the
        target's last-position logit (lazy), snapshotting both caches at the
        deepest interior message boundary for the next turn. See
        :func:`_drive_spec_prefill` for the stream / chunking invariants."""
        boundaries = segmented.boundary_offsets()
        # Guard: boundaries are only meaningful if the segmentation tokenizes
        # to exactly this prompt. A mismatch (re-tokenization drift) means no
        # snapshot this turn rather than a misaligned one.
        if segmented.flatten() != flat:
            boundaries = []
        return _drive_spec_prefill(
            flat=flat,
            boundaries=boundaries,
            already_covered=already_covered,
            lanes=[
                (self._target, self._target_cache),
                (self._draft, self._draft_cache),
            ],
            cancel_event=cancel_event,
            on_boundary=lambda boundary: self._store_snapshot(flat[:boundary]),
        )

    def _store_snapshot(self, tokens: list[int]) -> None:
        """Deep-copy + materialize the current caches and persist them under
        ``tokens`` (which ends at a message boundary)."""
        target_snap = snapshot_cache_for_persistence(
            self._target_cache, eager_eval=_cache_has_lazy_state(self._target_cache)
        )
        draft_snap = snapshot_cache_for_persistence(
            self._draft_cache, eager_eval=_cache_has_lazy_state(self._draft_cache)
        )
        self._cache_store.insert(tokens, (target_snap, draft_snap))

    def _step_impl(self) -> tuple[list[int], int]:
        """One speculative decoding step using persistent KV caches.

        Must call ``prefill()`` first to populate caches.

        When ``self._use_tree`` is True, builds a tree of draft
        alternatives and verifies them against the target in one forward
        pass using a sparse attention mask.

        Returns:
            (accepted_tokens, num_draft_generated).
        """
        assert self._target_cache is not None, "Call prefill() before step()"
        assert self._draft_cache is not None, "Call prefill() before step()"
        assert self._pending_token is not None, "Call prefill() before step()"

        pending_token = self._pending_token

        if self._use_tree:
            return self._step_tree(pending_token)
        return self._step_linear(pending_token)

    def _step_linear(self, pending_token: int) -> tuple[list[int], int]:
        """Linear speculative decoding (the existing algorithm)."""
        assert self._target_cache is not None
        assert self._draft_cache is not None

        # 1. Draft: feed pending token, then generate lambda candidates.
        if self._gdn_capture is not None:
            if self._draft_gdn_buffer is not None:
                self._draft_gdn_buffer.clear()
            self._gdn_capture.use_buffer(self._draft_gdn_buffer)
        draft_tokens, draft_ctx = self._draft_generate_cached(
            pending_token, self._lambda
        )

        self._after_draft(draft_ctx)

        # 2. Target: feed [pending, D1, ..., D_lambda] in one pass.
        if self._gdn_capture is not None:
            if self._target_gdn_buffer is not None:
                self._target_gdn_buffer.clear()
            self._gdn_capture.use_buffer(self._target_gdn_buffer)
        all_tokens = mx.array([[pending_token] + draft_tokens])
        target_out = _logits(self._target(all_tokens, cache=self._target_cache))
        mx.eval(target_out)

        if self._gdn_capture is not None:
            self._gdn_capture.use_buffer(None)

        verification_logits = target_out[0]  # (lambda+1, vocab)

        # 3. Verify
        with _tracing.span("spec.verify", strategy=self._strategy_label):
            accepted = self._verify(draft_tokens, verification_logits)
        num_accepted = len(accepted)

        self._after_verify(num_accepted)

        # 4. Roll back caches to keep only the accepted prefix.
        #
        # Target was fed (λ+1) tokens [pending, D_1..D_λ]; keep
        # ``num_accepted`` of them, so trim by (λ+1) - num_accepted.
        # Draft was fed λ tokens autoregressively
        # [pending, D_1..D_{λ-1}]; keep ``num_accepted`` of those if
        # partial acceptance (= a-1 draft tokens between pending and
        # the correction/bonus), so trim by λ - num_accepted.
        trim_target = max(self._lambda + 1 - num_accepted, 0)
        trim_draft = max(self._lambda - num_accepted, 0)

        if trim_target > 0:
            if self._target_gdn_buffer is not None and self._gdn_capture is not None:
                # ``rollback_single`` takes ``accepted`` as the number of
                # *additional* tokens beyond the first to keep
                # (n = accepted + 1). We want to keep ``num_accepted``
                # total, so accepted_arg = num_accepted - 1.
                self._gdn_capture.rollback_single(
                    self._target_gdn_buffer,
                    self._target_cache,
                    accepted=num_accepted - 1,
                    trim=trim_target,
                )
            else:
                # Rotating-aware trim: mlx-lm's ``trim_prompt_cache`` is
                # all-or-nothing and silently no-ops on sliding-window
                # (``RotatingKVCache``) targets once the window fills,
                # leaving rejected draft tokens resident (#605).
                _trim_recent_cache(self._target_cache, trim_target)

        if trim_draft > 0:
            if self._draft_gdn_buffer is not None and self._gdn_capture is not None:
                self._gdn_capture.rollback_autoregressive(
                    self._draft_gdn_buffer,
                    self._draft_cache,
                    num_steps=self._lambda,
                    num_keep_steps=num_accepted,
                    trim=trim_draft,
                )
            else:
                _trim_recent_cache(self._draft_cache, trim_draft)

        # On full acceptance, align draft cache with target cache.
        # ``use_buffer(None)`` was already called after the target
        # forward, so the align step's draft forward — which DOES
        # invoke ``GatedDeltaNet.__call__`` on a hybrid draft — won't
        # write to either buffer.
        if num_accepted > self._lambda:
            last_draft = mx.array([[draft_tokens[-1]]])
            align_logits = _logits(self._draft(last_draft, cache=self._draft_cache))
            mx.eval(align_logits)

        # 5. Update state.
        # The next pending token is definitionally ``accepted[-1]``:
        # ``verify_draft_greedy`` already argmax'd ``verification_logits[
        # num_accepted - 1]`` to produce it (a correction on partial accept,
        # the bonus on full accept). Re-deriving it here forced a second
        # ``mx.eval`` + ``.item()`` Metal round-trip per decode step for an
        # identical result; take the shortcut PLD already documents.
        self._cache_seq_len += num_accepted
        assert num_accepted >= 1, "step(): _verify() must return at least 1 token"
        self._pending_token = int(accepted[-1])

        num_accepted_draft = self._update_acceptance_rate(num_accepted)

        self._stats_steps += 1
        self._stats_proposed += self._lambda
        self._stats_accepted_draft += num_accepted_draft

        return accepted, self._lambda

    def _step_tree(self, pending_token: int) -> tuple[list[int], int]:
        """Tree-based speculative decoding step.

        Produces a tree of draft alternatives, verifies them against the
        target in one forward pass with a sparse attention mask, and rolls
        back/re-builds caches to keep only the accepted prefix.

        Known limitation: tree nodes are fed as a flat sequence, so sibling
        nodes inherit sequential RoPE positions from their flat indices rather
        than their tree depth.  This causes siblings at depth *d* to receive
        positional encodings intended for later positions, which may degrade
        acceptance rates.  A future PR should inject per-node position IDs
        based on ``tree.depths``.
        """
        from olmlx.engine.tree_speculative import (
            _patch_target_for_tree_forward,
            _restore_target,
            build_comb_tree,
            build_tree_attention_mask,
            verify_tree_greedy,
        )

        assert self._target_cache is not None
        assert self._draft_cache is not None

        # 1. Draft: generate primary path + alternatives per step
        if self._gdn_capture is not None:
            if self._draft_gdn_buffer is not None:
                self._draft_gdn_buffer.clear()
            self._gdn_capture.use_buffer(self._draft_gdn_buffer)
        primary_tokens, alt_per_step, draft_ctx = self._draft_generate_tree(
            pending_token, self._lambda
        )

        self._after_draft(draft_ctx)

        # 2. Build the tree, respecting the max-nodes cap.
        tree = build_comb_tree(
            pending_token=pending_token,
            primary_tokens=primary_tokens,
            alt_tokens_per_step=alt_per_step,
            max_nodes=self._tree_max_nodes,
        )

        # 3. Build sparse attention mask + patch target.
        # The tree mask only covers the tree nodes; the KV cache already
        # holds the prompt + previous generation tokens at positions
        # [0, cache_len).  Every tree node must attend to the full cached
        # prefix, so we pad the mask with zeros on the key-length axis.
        tree_mask = build_tree_attention_mask(tree)
        cache_len = self._target_cache[0].offset
        prefix_mask = mx.zeros((1, 1, tree.num_nodes, cache_len), dtype=mx.float32)
        full_mask = mx.concatenate([prefix_mask, tree_mask], axis=-1)

        if self._gdn_capture is not None:
            if self._target_gdn_buffer is not None:
                self._target_gdn_buffer.clear()
            self._gdn_capture.use_buffer(self._target_gdn_buffer)

        orig_call = _patch_target_for_tree_forward(self._target, full_mask)
        try:
            tree_tokens = mx.array([tree.tokens])
            target_out = _logits(self._target(tree_tokens, cache=self._target_cache))
            mx.eval(target_out)
        finally:
            _restore_target(self._target, orig_call)

        if self._gdn_capture is not None:
            self._gdn_capture.use_buffer(None)

        verification_logits = target_out[0]  # (num_tree_nodes, vocab)

        # 4. Verify tree
        accepted, used_sibling = verify_tree_greedy(tree, verification_logits)
        num_accepted = len(accepted)

        self._after_verify(num_accepted)

        # 5. Roll back / rebuild caches.
        #
        # The tree forward appended ``num_tree_nodes`` entries to the
        # target cache.  If the accepted path follows the primary branch
        # (the common case), the positions are contiguous and we can
        # simply trim from the end.  If a sibling was accepted, the
        # positions are not contiguous; we rebuild from the pre-step state.
        tree_total = tree.num_nodes

        if used_sibling:
            # Rebuild: trim all tree nodes *except* the root (pending_token
            # P), whose entry from the previous step is still valid in the
            # cache.  Then feed only the accepted prefix tokens (excluding
            # the bonus, which becomes the next _pending_token).
            trim_amt = tree_total - 1
            if self._target_gdn_buffer is not None and self._gdn_capture is not None:
                self._gdn_capture.rollback_single(
                    self._target_gdn_buffer,
                    self._target_cache,
                    accepted=0,
                    trim=trim_amt,
                )
            elif trim_prompt_cache is not None:
                trim_prompt_cache(self._target_cache, trim_amt)
            # Feed accepted tokens up to (but not including) the bonus.
            # Re-enable GDN capture for the rebuild so linear-attention
            # state stays consistent with the KV cache.
            if self._target_gdn_buffer is not None and self._gdn_capture is not None:
                self._target_gdn_buffer.clear()
                self._gdn_capture.use_buffer(self._target_gdn_buffer)
            rebuild_tokens = accepted[:-1]  # [D1, ..., sibling]
            rebuild_arr = mx.array([rebuild_tokens])
            rebuild_out = _logits(self._target(rebuild_arr, cache=self._target_cache))
            mx.eval(rebuild_out)
            if self._target_gdn_buffer is not None and self._gdn_capture is not None:
                self._gdn_capture.use_buffer(None)
            rebuild_logits = rebuild_out[0]
            # The logit at the sibling's position predicts the bonus.
            self._last_target_logit = rebuild_logits[len(rebuild_tokens) - 1]
        else:
            # Primary path: trim from the end.
            trim_target = max(tree_total - num_accepted, 0)
            if trim_target > 0:
                if (
                    self._target_gdn_buffer is not None
                    and self._gdn_capture is not None
                ):
                    self._gdn_capture.rollback_single(
                        self._target_gdn_buffer,
                        self._target_cache,
                        accepted=num_accepted - 1,
                        trim=trim_target,
                    )
                elif trim_prompt_cache is not None:
                    trim_prompt_cache(self._target_cache, trim_target)
            last_accepted_tree_idx = tree.primary_branch[num_accepted - 1]
            self._last_target_logit = verification_logits[last_accepted_tree_idx]

        mx.eval(self._last_target_logit)

        # 6. Align draft cache.  The draft generated λ primary-path tokens
        # autoregressively.  When a sibling was accepted the accepted path
        # includes the sibling + bonus but the draft only ran the primary;
        # we keep only the prefix of primary steps that are still valid.
        num_draft_keep = (num_accepted - 2) if used_sibling else num_accepted
        num_draft_keep = max(num_draft_keep, 0)

        if self._draft_gdn_buffer is not None and self._gdn_capture is not None:
            num_keep_steps = min(num_draft_keep, self._lambda)
            trim_draft_steps = self._lambda - num_keep_steps
            if trim_draft_steps > 0:
                self._gdn_capture.rollback_autoregressive(
                    self._draft_gdn_buffer,
                    self._draft_cache,
                    num_steps=self._lambda,
                    num_keep_steps=num_keep_steps,
                    trim=trim_draft_steps,
                )
        else:
            trim_draft = max(self._lambda - num_draft_keep, 0)
            if trim_draft > 0 and trim_prompt_cache is not None:
                trim_prompt_cache(self._draft_cache, trim_draft)

        if num_accepted > self._lambda and not used_sibling:
            last_primary = mx.array([[primary_tokens[-1]]])
            align_logits = _logits(self._draft(last_primary, cache=self._draft_cache))
            mx.eval(align_logits)

        # When a sibling was accepted, the draft cache reflects only the
        # primary path up to the branch point — it is missing the sibling
        # token.  Feed the sibling so the draft cache is aligned with the
        # accepted prefix before the next step.
        if used_sibling:
            sib_token = accepted[-2]  # sibling is second-to-last in accepted
            sib_arr = mx.array([[sib_token]])
            mx.eval(_logits(self._draft(sib_arr, cache=self._draft_cache)))

        # 7. Update state
        self._cache_seq_len += num_accepted
        assert num_accepted >= 1, "step(): _verify() must return at least 1 token"
        self._pending_token = int(mx.argmax(self._last_target_logit).item())

        num_accepted_draft = self._update_acceptance_rate(num_accepted)

        self._stats_steps += 1
        self._stats_proposed += tree.num_nodes - 1  # all non-root nodes
        self._stats_accepted_draft += num_accepted_draft

        return accepted, tree.num_nodes - 1

    def _draft_generate_cached(
        self, pending_token: int, n: int
    ) -> tuple[list[int], list[mx.array]]:
        """Generate n candidate tokens using the persistent draft cache.

        Returns:
            (tokens, context) — context is empty in the base class;
            subclasses may return captured hidden states.
        """
        assert self._draft_cache is not None

        next_token = pending_token
        tokens: list[int] = []

        for _ in range(n):
            inp = mx.array([[next_token]])
            logits = _logits(self._draft(inp, cache=self._draft_cache))
            next_logits = logits[:, -1, :]
            mx.eval(next_logits)
            next_token = int(mx.argmax(next_logits, axis=-1).item())
            tokens.append(next_token)

        return tokens, []

    def _draft_generate_tree(
        self, pending_token: int, n: int
    ) -> tuple[list[int], list[list[int]], list[mx.array]]:
        """Generate *n* primary tokens + top-K alternatives per step.

        Only called when ``self._use_tree`` is True.  Returns the primary
        path tokens and, for each step, the top-K candidate tokens (the
        first is the primary, the rest are siblings).

        Returns:
            (primary_tokens, alt_tokens_per_step, context).
        """
        assert self._draft_cache is not None
        from olmlx.engine.tree_speculative import extract_top_k_from_logits

        k = self._tree_width
        next_token = pending_token
        primary_tokens: list[int] = []
        alt_tokens_per_step: list[list[int]] = []

        for _ in range(n):
            inp = mx.array([[next_token]])
            logits = _logits(self._draft(inp, cache=self._draft_cache))
            next_logits = logits[:, -1, :]
            candidates = extract_top_k_from_logits(next_logits, k)
            primary = candidates[0]
            primary_tokens.append(primary)
            alt_tokens_per_step.append(candidates[1:])  # siblings (may be empty)
            next_token = primary

        return primary_tokens, alt_tokens_per_step, []

    def _after_draft(self, draft_ctx: list[mx.array]) -> None:
        """Hook called after draft generation. Override for prefetch submission."""

    def _after_verify(self, num_accepted: int) -> None:
        """Hook called after verification. Override for prefetch cancellation."""

    # ------------------------------------------------------------------
    # Stateless API
    # ------------------------------------------------------------------

    def _draft_generate(self, prompt: mx.array, n: int) -> list[int]:
        """Generate n candidate tokens with fresh KV cache (stateless)."""
        if n <= 0:
            return []
        if make_prompt_cache is None:
            # Caller (generate_step) already raises with the same wording;
            # this guard exists because asserts are stripped under -O.
            raise RuntimeError(
                "mlx_lm.models.cache.make_prompt_cache is not available; "
                "upgrade mlx-lm to a version that exports it."
            )
        try:
            cache = make_prompt_cache(self._draft)
        except (TypeError, AttributeError) as exc:
            raise RuntimeError(
                f"make_prompt_cache failed for draft model "
                f"{type(self._draft).__name__!r}: {exc}. The draft model may "
                "not be compatible with mlx-lm's KV-cache API."
            ) from exc
        # Same reset prefill()/generate_step apply to the target: VLM drafts
        # (mlx-vlm 0.4.4 Qwen3_5 etc.) cache _position_ids/_rope_deltas on
        # the module instance across calls, and pass 1 of _prefill_last_logit
        # below has cache_offset==0, which would consume a stale slice of
        # _position_ids from a previous request.
        for attr in ("_position_ids", "_rope_deltas"):
            if hasattr(self._draft, attr):
                setattr(self._draft, attr, None)
        tokens: list[int] = []

        # Same OOM path as the target: a large-vocab draft (e.g. same model
        # family as target, 262k vocab) on a long prompt would materialise
        # [1, seq_len, vocab] inside lm_head before the [:, -1, :] slice.
        # Route through _prefill_last_logit to keep the prefix lm_head out
        # of the eval graph.
        first_logit = _prefill_last_logit(self._draft, prompt, cache)
        mx.eval(first_logit)
        next_token = int(mx.argmax(first_logit).item())
        tokens.append(next_token)

        for _ in range(n - 1):
            inp = mx.array([[next_token]])
            logits = _logits(self._draft(inp, cache=cache))
            next_logits = logits[:, -1, :]
            mx.eval(next_logits)
            next_token = int(mx.argmax(next_logits, axis=-1).item())
            tokens.append(next_token)

        return tokens

    def _verify(
        self,
        draft_tokens: list[int],
        target_logits: mx.array,
    ) -> list[int]:
        """Verify draft tokens against target model logits (greedy)."""
        return verify_draft_greedy(draft_tokens, target_logits)

    def generate_step(
        self,
        prompt: mx.array,
    ) -> tuple[list[int], int]:
        """One speculative decoding step (stateless, no cross-step caching).

        Uses a temporary KV cache internally so the target forward over the
        long prompt does not materialise the full [batch, seq_len, vocab]
        logit matrix (same Metal OOM that ``_prefill_last_logit`` avoids).
        """
        if make_prompt_cache is None:
            raise RuntimeError(
                "mlx_lm.models.cache.make_prompt_cache is not available "
                "(import failed at module load). Upgrade mlx-lm to a "
                "version that exports it (or use the cached prefill+step "
                "API instead, which has the same requirement). The "
                "previous cache-less path was removed because it OOMed "
                "on Metal for large-vocab models on long prompts."
            )

        draft_tokens = self._draft_generate(prompt, self._lambda)

        # See prefill() for why this reset is needed (mlx-vlm 0.4.4 VLMs).
        for attr in ("_position_ids", "_rope_deltas"):
            if hasattr(self._target, attr):
                setattr(self._target, attr, None)

        # Two-pass split: prefill the prompt into a temporary cache (yields
        # the first verification logit), then feed [pending, D1..D_lambda]
        # for the remaining lambda logits. Total materialised logit shape is
        # [1, lambda+1, vocab] instead of [1, seq_len+lambda, vocab].
        target_cache = make_prompt_cache(self._target)
        first_logit = _prefill_last_logit(self._target, prompt, target_cache)
        draft_ids = mx.array([draft_tokens])
        draft_out = _logits(self._target(draft_ids, cache=target_cache))
        target_logits = mx.concatenate([first_logit[None, :], draft_out[0]], axis=0)
        mx.eval(target_logits)

        accepted = self._verify(draft_tokens, target_logits)

        num_accepted_draft = self._update_acceptance_rate(len(accepted))
        self._stats_steps += 1
        self._stats_proposed += self._lambda
        self._stats_accepted_draft += num_accepted_draft
        return accepted, self._lambda


class PromptLookupDecoder(SpecDecoderBase):
    """Prompt-lookup decoding (PLD): zero-cost draft via n-gram lookup.

    Reference: https://github.com/apoorvumang/prompt-lookup-decoding

    The "draft" comes from searching the prompt+generated history for
    occurrences of the most recent n-gram suffix; the tokens immediately
    after the match become the draft candidates. No draft model, no
    training, no per-token Metal sync — just a small CPU n-gram scan
    between forward passes.

    Works on any target, including Flash-MoE (which blocks DFlash/EAGLE
    because those rely on target hidden-state hooks). Particularly
    effective on code edits, structured output (JSON), and any task with
    high contextual repetition.

    Implements the same ``prefill`` / ``step`` / ``reset`` protocol as
    ``SpeculativeDecoder`` so the streaming adapter is shared. Not
    thread-safe: one decoder instance must serve one request at a time.
    """

    def __init__(
        self,
        target_model: nn.Module,
        num_speculative_tokens: int = 10,
        max_ngram_size: int = 3,
        min_ngram_size: int = 1,
        lookup_window: int | None = 8192,
        acceptance_rate_ema: float = 0.9,
        cache_slots: int = 0,
    ):
        super().__init__()
        if trim_prompt_cache is None or make_prompt_cache is None:
            raise RuntimeError(
                "mlx_lm.models.cache imports failed (trim_prompt_cache / "
                "make_prompt_cache unavailable); PLD requires both — fail "
                "fast at construction rather than crashing on first prefill"
            )
        if num_speculative_tokens < 1:
            raise ValueError(
                f"num_speculative_tokens must be >= 1, got {num_speculative_tokens}"
            )
        if min_ngram_size < 1 or max_ngram_size < min_ngram_size:
            raise ValueError(
                f"Invalid n-gram range: min={min_ngram_size}, "
                f"max={max_ngram_size}. Must satisfy "
                f"1 <= min_ngram_size <= max_ngram_size."
            )
        if lookup_window is not None and lookup_window < max_ngram_size:
            raise ValueError(
                f"lookup_window ({lookup_window}) must be >= max_ngram_size "
                f"({max_ngram_size}) or None. A window smaller than the "
                f"largest n-gram makes that n-gram unmatchable."
            )
        self._target = target_model
        self._lambda = num_speculative_tokens
        self._max_ngram = max_ngram_size
        self._min_ngram = min_ngram_size
        #: Cap on how many of the most recent ``_tokens`` entries the
        #: n-gram scan walks. ``None`` disables the cap. The pending
        #: token is always included regardless of this window.
        self._lookup_window = lookup_window

        # Cross-request KV reuse (#421): persists target snapshots keyed by
        # token prefix. ``cache_slots == 0`` disables it (fresh prefill).
        self._cache_store = _SpecCacheStore(cache_slots)
        #: Tokens reused from a stored snapshot on the most recent prefill.
        self._last_reused_tokens: int = 0
        # Init to 0.0 (not 0.5 as in ``SpeculativeDecoder``): PLD's
        # acceptance rate before any step has run is honestly 0 (nothing
        # drafted, nothing accepted), and a 0.5 warm-up would mislead
        # ``stats_summary`` consumers (bench, streaming layer) into
        # reporting 50% for the first several steps.
        self._alpha = 0.0
        self._alpha_ema = acceptance_rate_ema

        # Persistent state populated by prefill/step
        self._target_cache: list | None = None
        self._cache_seq_len: int = 0
        self._pending_token: int | None = None
        # Full token history (prompt + cached generated tokens) used as
        # the lookup table. The currently-pending token is tracked
        # separately in ``_pending_token`` and is virtually appended to
        # this list during n-gram search.
        self._tokens: list[int] = []

        # GDN rollback for hybrid linear-attention targets
        # (Qwen3.5/3.6 GatedDeltaNet). PLD has no draft model, so we
        # only patch the target — no shared-class invariant to enforce.
        self._gdn_capture: GDNStateCapture | None = None
        self._target_gdn_buffer: GDNBuffer | None = None
        target_gdn_cls = find_gdn_class(target_model)
        if target_gdn_cls is not None:
            self._gdn_capture = GDNStateCapture(target_gdn_cls)
            try:
                self._target_gdn_buffer = self._gdn_capture.create_buffer(target_model)
            except Exception:
                self._gdn_capture.close()
                self._gdn_capture = None
                self._target_gdn_buffer = None
                raise

    def close(self) -> None:
        """Release the GDN class-level monkey-patch (idempotent).

        Overrides the base ``close()`` because PLD's GDN capture and
        snapshot store are decoder-lifetime (created in ``__init__``),
        not request-lifetime. The ``finally`` mirrors the classic
        decoder: a failing capture close must not skip dropping the
        snapshot store and working KV cache, since ``ModelManager``
        preserves the decoder reference when ``close()`` raises.
        """
        try:
            if self._gdn_capture is not None:
                self._gdn_capture.close()
                self._gdn_capture = None
                self._target_gdn_buffer = None
        finally:
            # Drop persisted snapshots so a model unload frees their KV
            # promptly, and release the last request's working cache too
            # (the uniform base lifecycle: close ⊇ reset). Never raises.
            self._cache_store.clear()
            self.reset()

    def _reset_state(self) -> None:
        # Detach the decoder-lifetime GDN capture from any buffer on reset —
        # otherwise a mid-step exception leaves the class-level patch routed
        # to a stale buffer that every later GDN call keeps growing (#633).
        # See the classic decoder's ``_reset_state`` for the full rationale.
        if self._gdn_capture is not None:
            self._gdn_capture.use_buffer(None)
        self._target_cache = None
        self._cache_seq_len = 0
        self._pending_token = None
        self._tokens = []
        self._last_reused_tokens = 0
        # Reset EMA too; ``stats_summary`` is documented as "PLD's
        # acceptance rate before any step is honestly 0", and that
        # invariant must hold per-request, not just per-decoder.
        self._alpha = 0.0
        # NB: ``self._cache_store`` persists across requests — not cleared here.

    def _stats_extra(self) -> dict[str, Any]:
        return {
            "ema_acceptance_rate": self._alpha,
            "lambda": self._lambda,
        }

    def _prefill_impl(
        self,
        prompt: mx.array,
        *,
        segmented: Any = None,
        cancel_event: threading.Event | None = None,
    ) -> int:
        """Process the prompt through the target, populating its KV cache.

        Args:
            prompt: (1, seq_len) input token IDs.
            segmented: optional :class:`SegmentedPrompt`. When provided and
                ``OLMLX_SPECULATIVE_CACHE_SLOTS`` is non-zero, enables
                cross-request KV reuse (#421) — only the new suffix is
                prefilled and a snapshot is taken at the deepest interior
                message boundary for the next turn.
            cancel_event: if set during prefill, raises :class:`PrefillCancelled`
                at the next sub-chunk boundary so a client disconnect interrupts
                a long prefill promptly instead of pinning the GPU.

        Returns:
            The first generated token (target's greedy argmax).
        """
        # ``__init__`` already enforces ``make_prompt_cache`` /
        # ``trim_prompt_cache`` non-None at construction, so any
        # ``PromptLookupDecoder`` that reaches this point has them.
        # No need to re-check.
        self._target_cache = make_prompt_cache(self._target)

        # Same VLM cache-reset rationale as ``SpeculativeDecoder.prefill``.
        for attr in ("_position_ids", "_rope_deltas"):
            if hasattr(self._target, attr):
                setattr(self._target, attr, None)

        # No rollback needed for the prompt forward.
        if self._gdn_capture is not None:
            self._gdn_capture.use_buffer(None)

        use_reuse = (
            self._cache_store.enabled()
            and segmented is not None
            and not _is_pure_rotating_cache(self._target_cache)
        )
        if use_reuse:
            last_logit = self._prefill_with_reuse(
                prompt, segmented, cancel_event=cancel_event
            )
        else:
            last_logit = _prefill_last_logit(
                self._target, prompt, self._target_cache, cancel_event=cancel_event
            )
            self._cache_seq_len = prompt.shape[1]
        mx.eval(last_logit)

        # Seed the lookup table with the FULL prompt tokens (regardless of how
        # much was reused — the n-gram scan must see the whole history, not
        # just the freshly-prefilled suffix). The pending (first generated)
        # token lives in ``_pending_token`` until the next ``step()`` puts it
        # into the cache. Cap the seed to the lookup window so we don't pay an
        # O(prompt_len) Metal-to-Python copy for tokens that will never
        # participate in the scan (the cached prefix is still in the KV cache;
        # this list only feeds the lookup heuristic).
        if self._lookup_window is not None and prompt.shape[1] > self._lookup_window:
            self._tokens = prompt[0, -self._lookup_window :].tolist()
        else:
            self._tokens = prompt[0].tolist()

        first_token = int(mx.argmax(last_logit).item())
        self._pending_token = first_token
        return first_token

    def _prefill_with_reuse(
        self,
        prompt: mx.array,
        segmented: Any,
        *,
        cancel_event: threading.Event | None,
    ) -> mx.array:
        """Reuse-aware target-only prefill (#421). Mirrors
        :meth:`SpeculativeDecoder._prefill_with_reuse` without the draft lane.
        Returns the target's last-position logit (lazy)."""
        flat = prompt[0].tolist()
        n = len(flat)
        # Cache layer types are invariant per model, so classifying the fresh
        # cache is valid for the (same-type) restored snapshot below.
        is_trimmable = _cache_is_trimmable(self._target_cache)

        already_covered = 0
        hit = self._cache_store.find(flat)
        if hit is not None:
            entry, common = hit
            reuse, covered = _spec_reuse_decision(is_trimmable, entry.tokens, common, n)
            if reuse:
                # Deepcopy the stored snapshot (a copy of a copy) into the
                # working cache so this request's mutations never touch the
                # store — copy-on-reuse isolation; see config docstring.
                self._target_cache = snapshot_cache_for_persistence(
                    entry.payload, eager_eval=_cache_has_lazy_state(entry.payload)
                )
                if is_trimmable:
                    trim = len(entry.tokens) - covered
                    if trim > 0:
                        trim_prompt_cache(self._target_cache, trim)
                already_covered = covered

        boundaries = segmented.boundary_offsets()
        if segmented.flatten() != flat:
            boundaries = []
        logit = _drive_spec_prefill(
            flat=flat,
            boundaries=boundaries,
            already_covered=already_covered,
            lanes=[(self._target, self._target_cache)],
            cancel_event=cancel_event,
            on_boundary=lambda boundary: self._store_snapshot(flat[:boundary]),
        )
        self._cache_seq_len = n
        self._last_reused_tokens = already_covered
        return logit

    def _store_snapshot(self, tokens: list[int]) -> None:
        """Deep-copy + materialize the target cache and persist it under
        ``tokens`` (which ends at a message boundary)."""
        target_snap = snapshot_cache_for_persistence(
            self._target_cache, eager_eval=_cache_has_lazy_state(self._target_cache)
        )
        self._cache_store.insert(tokens, target_snap)

    def _step_impl(self) -> tuple[list[int], int]:
        """One PLD step using the persistent target KV cache.

        Must call ``prefill()`` first. Returns
        ``(accepted_tokens, num_draft_proposed)`` — the second value is the
        *actual* draft length (0..lambda), not the configured maximum.
        """
        # Explicit ``raise`` (not ``assert``) so the misuse still
        # surfaces under ``python -O`` — otherwise the stripped
        # asserts let ``None`` fall through and produce a confusing
        # ``TypeError`` from the target forward a few lines below.
        if self._target_cache is None or self._pending_token is None:
            raise RuntimeError(
                "PromptLookupDecoder.step() called before prefill(); "
                "call prefill(prompt) first"
            )

        pending_token = self._pending_token

        # 1. Build the draft via n-gram lookup. May be empty when no
        # match is found — that turns the step into a single-token
        # target forward, equivalent to plain greedy decoding for this
        # step.
        draft_tokens = self._lookup_draft()
        num_drafted = len(draft_tokens)

        # 2. Target forward on [pending, D_1..D_num_drafted].
        if self._gdn_capture is not None:
            if self._target_gdn_buffer is not None:
                self._target_gdn_buffer.clear()
            self._gdn_capture.use_buffer(self._target_gdn_buffer)
        all_tokens = mx.array([[pending_token] + draft_tokens])
        target_out = _logits(self._target(all_tokens, cache=self._target_cache))
        mx.eval(target_out)
        if self._gdn_capture is not None:
            self._gdn_capture.use_buffer(None)

        verification_logits = target_out[0]  # (num_drafted+1, vocab)

        # 3. Greedy verification — identical to SpeculativeDecoder.
        accepted = self._verify_greedy(draft_tokens, verification_logits)
        num_accepted = len(accepted)

        # 4. Trim target cache to keep only the accepted prefix.
        # Target was fed (num_drafted + 1) tokens; keep num_accepted of
        # them, so trim by (num_drafted + 1) - num_accepted.
        trim_target = max(num_drafted + 1 - num_accepted, 0)
        if trim_target > 0:
            if self._target_gdn_buffer is not None and self._gdn_capture is not None:
                self._gdn_capture.rollback_single(
                    self._target_gdn_buffer,
                    self._target_cache,
                    accepted=num_accepted - 1,
                    trim=trim_target,
                )
            else:
                # Rotating-aware trim (see classic decoder / #605):
                # mlx-lm's ``trim_prompt_cache`` no-ops on filled
                # sliding-window caches, corrupting SWA targets.
                _trim_recent_cache(self._target_cache, trim_target)

        # 5. Update state. The cache now contains the original prompt
        # plus the pending token plus the first (num_accepted - 1)
        # accepted tokens; the final accepted token (the bonus from
        # verify_draft_greedy) becomes the new pending and is *not* in
        # the cache yet. Unlike ``SpeculativeDecoder.step``, we derive
        # the next pending from ``accepted[-1]`` directly — verify
        # already did the argmax — so we don't store or eval the
        # corresponding logit (it would force a Metal round-trip for a
        # tensor that's immediately discarded).
        self._tokens.append(pending_token)
        if num_accepted > 1:
            self._tokens.extend(accepted[:-1])
        # Keep memory O(lookup_window) regardless of conversation
        # length: anything beyond the window can never be matched, so
        # prune in-place to drop both the Python objects and any
        # downstream slicing cost in ``_lookup_draft``. ``del`` on a
        # slice is the cheap mutation here — avoids an alloc/copy.
        if self._lookup_window is not None and len(self._tokens) > self._lookup_window:
            del self._tokens[: len(self._tokens) - self._lookup_window]
        self._cache_seq_len += num_accepted
        self._pending_token = int(accepted[-1])

        # 6. Stats. EMA only updates when something was proposed —
        # otherwise the step would push acceptance rate toward 0 even
        # though nothing was rejected.
        num_accepted_draft = min(num_accepted - 1, num_drafted)
        if num_drafted > 0:
            acceptance = num_accepted_draft / num_drafted
            self._alpha = (
                self._alpha_ema * self._alpha + (1 - self._alpha_ema) * acceptance
            )

        self._stats_steps += 1
        self._stats_proposed += num_drafted
        self._stats_accepted_draft += num_accepted_draft

        return accepted, num_drafted

    def _lookup_draft(self) -> list[int]:
        """Find up to ``self._lambda`` draft tokens via n-gram lookup.

        Iterates n-gram sizes from ``max_ngram`` down to ``min_ngram``.
        For each size, walks the sequence backwards looking for the most
        recent match of the trailing n-gram (excluding the trailing
        occurrence itself) and returns the tokens immediately following
        that match. Returns ``[]`` if no n-gram of any size matches.

        The search corpus is the most recent ``_lookup_window`` tokens
        of ``self._tokens`` plus ``self._pending_token`` — the pending
        token is always included so drafts can match across the
        prompt/generation boundary. The window cap exists because the
        scan is pure Python and grows with history length: an
        unbounded history at 128k tokens would add ~30–80 ms per step
        on Apple Silicon, rivalling the target forward.
        """
        pending = self._pending_token
        if pending is None:
            # Same defensive philosophy as ``step()``: explicit raise
            # so the misuse survives ``python -O``. _lookup_draft is
            # private and only reached via ``step()``, but the
            # consistency makes the contract clear.
            raise RuntimeError(
                "_lookup_draft called with _pending_token=None; call prefill() first"
            )
        # Materialise the search corpus once. The alternative — a
        # closure that branches between history[i] and pending — pays
        # a Python frame on every access, and ``_lookup_draft`` does
        # O(window × max_ngram) accesses (≈24k at the defaults).
        # Concatenation costs a single ~window-sized list alloc per
        # step, which is cheaper than the per-call overhead. No need
        # to re-cap by ``_lookup_window`` here: ``prefill()`` caps
        # the initial seed and ``step()`` trims at the end of each
        # call, so ``self._tokens`` is already <= window every time
        # we reach this point.
        seq = self._tokens + [pending]
        L = len(seq)

        if L < 2:
            return []

        for ngram_size in range(self._max_ngram, self._min_ngram - 1, -1):
            if L < ngram_size + 1:
                continue
            query_start = L - ngram_size
            query = seq[query_start:L]
            # Walk backwards through possible match start positions,
            # excluding the trailing query's own position.
            for start in range(query_start - 1, -1, -1):
                if seq[start : start + ngram_size] != query:
                    continue
                # Cap ``draft_end`` at ``L - 1`` to exclude the pending
                # token itself from the draft. The closest-possible
                # match (``start = query_start - 1``) would otherwise
                # produce ``draft = [pending]`` — asking the target to
                # predict the same token twice in a row. That's almost
                # always rejected and wastes a draft slot's worth of
                # acceptance-rate accounting. Excluding pending makes
                # this match yield an empty draft; we fall through to
                # an earlier match position or a smaller n-gram size.
                draft_start = start + ngram_size
                draft_end = min(draft_start + self._lambda, L - 1)
                if draft_end > draft_start:
                    return seq[draft_start:draft_end]
                # Empty draft (this match position only proposed
                # pending). Continue to earlier match positions at the
                # current n-gram size, then to smaller n-gram sizes.
        return []
