"""EAGLE decoder: prefill / step / reset protocol over an EAGLE draft.

Mirrors the same ``SpeculativeDecoder`` protocol used by classic
speculative decoding and DFlash, so the existing
``speculative_stream_generate`` bridge works unchanged. The crucial
difference is that EAGLE's draft is **autoregressive in feature
space** — it consumes ``(token_prev, h_prev)`` and produces ``(logits,
h_new)``, recursing for ``block_size`` draft tokens before each verify.

Cache management
----------------

Two regimes, picked at prefill time based on
``can_trim_prompt_cache``:

- **Trim-able caches** (standard attention; most non-hybrid LMs):
  ``trim_prompt_cache`` lops the rejected suffix off both target and
  draft caches at the end of each ``step()``.

- **Non-trim-able caches** (``GatedDeltaNet`` linear-attention layers
  in Qwen3.5/3.6 hybrids): we install DFlash's ``_GDNStateCapture``
  to monkey-patch ``GatedDeltaNet.__call__`` and snapshot the
  recurrent state per layer. After a partial-acceptance verify,
  ``rollback`` replays ``gated_delta_update`` on the accepted prefix
  to restore the correct state. The draft cache is always trim-able
  (the draft is a small standard-attention transformer), so we trim
  it directly.

Layer hooking is shared with DFlash via ``_patch_model`` /
``_get_layers`` / ``_LayerHook`` so we don't reimplement the universal
target-layer-output capture pattern.
"""

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from olmlx.engine.dflash.decoder import (
    _find_gdn_class,
    _GDNStateCapture,
    _get_layers,
    _HAS_GDN,
    _patch_model,
    _unpatch_model,
)
from olmlx.engine.eagle.draft_model import EagleDraftModel
from olmlx.engine.speculative import verify_draft_greedy

try:
    from mlx_lm.models.cache import (
        can_trim_prompt_cache,
        make_prompt_cache,
        trim_prompt_cache,
    )
except ImportError:  # pragma: no cover - mlx-lm always installed in production
    make_prompt_cache = None  # type: ignore[assignment]
    trim_prompt_cache = None  # type: ignore[assignment]
    can_trim_prompt_cache = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def _logits(out: Any) -> mx.array:
    # mlx-vlm wraps logits in a dataclass; mlx-lm returns the raw array.
    return getattr(out, "logits", out)


class EagleDecoder:
    """Autoregressive draft + verify protocol for EAGLE.

    State transitions
    -----------------

    ``prefill(prompt_tokens)``:
        1. Patch the target's chosen layer with a capture hook.
        2. Run the target on the full prompt; record its last-layer
           hidden states and final logits.
        3. Sample a first token from the target's logits at the
           prompt-tail position. Hold that token + the corresponding
           hidden as the seed for ``step()``.

    ``step()``:
        1. Run the draft autoregressively for ``block_size`` iterations,
           seeding from the held ``(token, hidden)`` and feeding each
           subsequent step its own ``(sampled_token, h_new)``. Collect
           ``block_size`` candidate tokens.
        2. Run the target on
           ``[seed_token, *candidates]`` (length ``block_size + 1``),
           capturing logits and the target's last-layer hidden via the
           hook installed in prefill.
        3. ``verify_draft_greedy`` over the captured logits → list of
           accepted tokens.
        4. Trim both caches back to the accepted prefix; rotate
           ``(seed_token, seed_hidden)`` to the last accepted token
           and the target's hidden at that position.
        5. Return the accepted tokens.

    ``reset()``:
        Clear caches, unpatch the target, unbind the draft.
    """

    def __init__(
        self,
        target_model: nn.Module,
        draft_model: EagleDraftModel,
        block_size: int = 4,
        target_layer_id: int | None = None,
    ):
        if make_prompt_cache is None:
            raise RuntimeError(
                "mlx_lm.models.cache is unavailable; EagleDecoder requires it "
                "for prompt-cache management."
            )
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1; got {block_size}")
        self._target = target_model
        self._draft = draft_model
        self._block_size = block_size
        # Default to the deepest (last) layer, i.e. ``len(layers) - 1``.
        # In practice trained checkpoints always supply
        # ``target_layer_id`` (recorded by ``olmlx eagle prepare`` from
        # the precomputed shard ladder), so this default only fires for
        # hand-constructed ``EagleDecoder`` instances and for legacy
        # checkpoints from before the field was persisted. Mismatching
        # this against the layer the draft was trained on collapses
        # bench acceptance to ~5% — operators wanting a non-default
        # layer must pass ``target_layer_id`` explicitly.
        num_target_layers = len(_get_layers(target_model))
        if target_layer_id is None:
            target_layer_id = num_target_layers - 1
        # Bounds-check explicitly. ``_patch_model`` indexes the layers
        # list with this value; an out-of-range index would surface as
        # ``IndexError`` at the first prefill, far from the load site.
        # Catch the cross-target-size mismatch case (e.g. checkpoint
        # trained on a 64-layer target, loaded against a 32-layer
        # target) here with an actionable error.
        if not 0 <= target_layer_id < num_target_layers:
            raise ValueError(
                f"target_layer_id={target_layer_id} is out of range for a "
                f"target with {num_target_layers} layers (valid indices: "
                f"0..{num_target_layers - 1}). The EAGLE draft was likely "
                f"trained against a target of a different depth; retrain "
                f"with `olmlx eagle prepare` against the current target."
            )
        self._target_layer_id = target_layer_id
        self._hidden_storage: list[Any] = [None]

        # Per-request state populated by ``prefill``.
        self._target_cache: list | None = None
        self._draft_cache: list | None = None
        self._seed_token: int | None = None
        self._seed_hidden: mx.array | None = None
        self._patched: bool = False
        self._bound: bool = False
        # GDN state capture for non-trim-able targets. Created in
        # ``prefill`` only when the target cache is non-trim-able
        # (Qwen3.5/3.6 hybrid linear-attention case). ``None`` for
        # standard-attention targets that take the trim path.
        self._capture: _GDNStateCapture | None = None
        self._target_can_trim: bool = True

        # Stats (reset on prefill).
        self._stats_steps = 0
        self._stats_proposed = 0
        self._stats_accepted_draft = 0

    # ----- lifecycle -------------------------------------------------------

    def reset(self) -> None:
        # Close the GDN capture *first* — it holds ``_GDN_PATCH_LOCK``,
        # and if any of the steps below raises we still want the lock
        # released so subsequent decoder instances can use the patch.
        if self._capture is not None:
            try:
                self._capture.close()
            finally:
                self._capture = None
        if self._patched:
            try:
                _unpatch_model(self._target)
            finally:
                self._patched = False
        if self._bound:
            try:
                self._draft.unbind()
            finally:
                self._bound = False
        self._target_cache = None
        self._draft_cache = None
        self._seed_token = None
        self._seed_hidden = None
        self._target_can_trim = True
        self._hidden_storage = [None]
        self._stats_steps = 0
        self._stats_proposed = 0
        self._stats_accepted_draft = 0

    def __del__(self) -> None:
        # Belt-and-braces: if the decoder is dropped without explicit
        # ``reset()`` (cancelled request, exception unwinding past the
        # streaming bridge), the GDN patch lock would otherwise leak.
        # Mirrors DFlashDecoder's finalizer.
        try:
            self.reset()
        except Exception:
            pass

    def stats_summary(self) -> dict[str, Any]:
        steps = self._stats_steps
        proposed = self._stats_proposed
        accepted_draft = self._stats_accepted_draft
        return {
            "steps": steps,
            "proposed": proposed,
            "accepted_draft": accepted_draft,
            "acceptance_rate": accepted_draft / proposed if proposed else 0.0,
            "avg_tokens_per_step": (accepted_draft + steps) / steps if steps else 0.0,
            "block_size": self._block_size,
        }

    # ----- prefill ---------------------------------------------------------

    def prefill(self, prompt: mx.array) -> int:
        """Run the target on the prompt; return the first sampled token.

        ``prompt`` shape: ``(1, seq_len)`` int tokens. After prefill the
        decoder holds the target's last-layer hidden at position
        ``seq_len - 1`` and the greedily-sampled first token.
        """
        self.reset()

        # Hook the chosen target layer so its output is captured into
        # ``_hidden_storage[0]`` on every target forward.
        _patch_model(self._target, [self._target_layer_id], self._hidden_storage)
        self._patched = True
        self._draft.bind(self._target)
        self._bound = True

        # Build fresh caches for both models.
        self._target_cache = make_prompt_cache(self._target)
        self._draft_cache = self._draft.make_cache()

        # (Classic ``SpeculativeDecoder.prefill`` resets ``_position_ids``
        # / ``_rope_deltas`` on the target here for mlx-vlm targets,
        # but EAGLE blocks VLM targets at load time in
        # ``_load_eagle_decoder`` — those attrs are unreachable from
        # this path and we don't replicate the reset.)

        # Pick the cache-trimming regime: trim_prompt_cache for
        # standard attention, or _GDNStateCapture for hybrid linear-
        # attention targets whose caches are non-trim-able.
        self._target_can_trim = can_trim_prompt_cache is None or can_trim_prompt_cache(
            self._target_cache
        )
        if not self._target_can_trim:
            if not _HAS_GDN:
                self.reset()
                raise RuntimeError(
                    "Target model has non-trim-able KV cache (likely "
                    "GatedDeltaNet linear-attention layers) but "
                    "mlx_lm.models.gated_delta is unavailable. Cannot "
                    "perform EAGLE rejection rollback."
                )
            # Defensive guard: the non-trim branch assumes the
            # non-trim caches are GDN/SSM state owned by linear-
            # attention layers. If the target has no GDN modules at
            # all, then some *other* cache type returned
            # ``is_trimmable() == False`` — e.g. a future quantised
            # KV cache that decides not to support trim, or a model
            # mixing pure-attention layers with an unfamiliar
            # stateful cache. The rollback path would either
            # silently mis-trim (if the orphan check inside
            # ``_GDNStateCapture.__init__`` doesn't catch it) or
            # crash cryptically later. Reject up front with a
            # clear message that names the configuration mismatch.
            #
            # Today's quant caches (``TurboQuantKVCache``,
            # ``SpectralQuantKVCache``) both return
            # ``is_trimmable() == True`` so they take the trim
            # path and never reach this branch — this is a
            # forward-compatibility guard, not a current-bug fix.
            if _find_gdn_class(self._target) is None:
                self.reset()
                raise RuntimeError(
                    "Target reports a non-trim-able KV cache but no "
                    "``GatedDeltaNet`` submodule is present. EAGLE's "
                    "non-trim rollback path is GDN-specific; some other "
                    "cache type (custom quant cache that doesn't "
                    "support trim, mixed-architecture model with an "
                    "unfamiliar stateful cache) must be at play. "
                    "Disable EAGLE for this target or replace the "
                    "non-trim cache with a trim-able variant."
                )
        # Wrap the GDN capture install + the prompt forward in
        # try/reset so any exception (lock contention or unexpected
        # state inside ``_GDNStateCapture.__init__``; OOM / Metal
        # stream / shape mismatch inside the forward) doesn't leave
        # the model patched and (for GDN targets) the
        # ``_GDN_PATCH_LOCK`` held. Without this, ``_patched=True``
        # and the patch lock are retained until the next
        # ``prefill()`` self-heals via ``reset()`` (called at its
        # top) or until ``__del__`` fires — a window during which
        # another in-flight request inspecting the target's layers
        # sees the monkey-patched ``__call__`` and (for GDN targets)
        # blocks on the lock. The capture install in particular is
        # not covered by the "after the forward" wrapper because
        # ``__init__`` acquires the lock — if that itself raises,
        # the lock state has to be cleaned up here.
        try:
            if not self._target_can_trim:
                # Install the patch — must happen *before* the prompt
                # forward so the patched ``__call__`` records the prompt's
                # GDN state, which subsequent rollbacks may need to replay
                # against. ``_GDNStateCapture.__init__`` acquires
                # ``_GDN_PATCH_LOCK``; ``reset()`` releases it via
                # ``capture.close()``.
                self._capture = _GDNStateCapture(self._target)

            # Run the target on the prompt and capture its last-layer hidden.
            target_out = self._target(prompt, cache=self._target_cache)
            target_logits = _logits(target_out)
            captured = self._hidden_storage[0]
            if captured is None:
                raise RuntimeError(
                    "EagleDecoder prefill: target forward did not populate "
                    f"the hidden capture slot for layer "
                    f"{self._target_layer_id}. Either ``_patch_model`` is "
                    "incompatible with this target structure, or the chosen "
                    "layer wasn't visited."
                )

            # Greedy sample at the prompt-tail position.
            last_logits = target_logits[:, -1:, :]
            seed_token = int(mx.argmax(last_logits, axis=-1).item())
            # Hidden at the prompt-tail position becomes the conditioning
            # for the first draft step.
            self._seed_hidden = captured[:, -1:, :]
        except Exception:
            # Best-effort cleanup. ``reset()`` itself swallows nested
            # exceptions from each step (unpatch, unbind, capture
            # close) so the caller sees the original error, not the
            # cleanup error.
            self.reset()
            raise
        self._seed_token = seed_token
        return seed_token

    # ----- step ------------------------------------------------------------

    def step(self) -> tuple[list[int], int]:
        """Generate ``block_size`` draft candidates, verify against the
        target, return ``(accepted_tokens, num_accepted_draft)``.

        ``num_accepted_draft`` is the number of *draft* tokens accepted
        (between 0 and ``block_size``). The returned ``accepted_tokens``
        list always has length ``num_accepted_draft + 1``: zero or more
        accepted draft tokens followed by the target's preferred token
        at the first mismatch (or the target's bonus token if all
        drafts were accepted).
        """
        if self._target_cache is None or self._draft_cache is None:
            raise RuntimeError(
                "EagleDecoder.step() called before prefill(); call "
                "prefill(prompt) to populate caches first."
            )
        if self._seed_token is None or self._seed_hidden is None:
            raise RuntimeError("EagleDecoder.step(): seed state is unset")

        # Mirror ``prefill``'s try/reset wrapper. A mid-step exception
        # in the draft loop (MLX Metal error during ``.item()``) or
        # the verify forward (OOM, shape mismatch) would otherwise
        # leave both KV caches in a partially-modified state and the
        # GDN capture's ``_gdn_inputs`` list with leftover entries
        # from the failed verify — the next ``step()`` would either
        # consume corrupt caches or trip the cardinality check
        # inside ``rollback`` in a confusing way. ``reset()`` here
        # forces the caller to re-``prefill`` to recover, which is
        # the right semantic for a hard inference failure.
        try:
            return self._step_impl()
        except Exception:
            self.reset()
            raise

    def _step_impl(self) -> tuple[list[int], int]:
        """Inner body of ``step()``. Wrapped by ``step()`` in a
        try/reset so any exception forces a clean tear-down. Split
        into a separate method for readability — the exception-safety
        boundary stays at the single ``step()`` call site instead of
        being inlined around every forward.
        """
        # ---- Draft phase: produce block_size candidates autoregressively.
        #
        # Note: the ``.item()`` below forces an MLX eval each draft
        # step because the *next* draft iteration needs the integer
        # token id to build its input. That's one sync per draft
        # position — ``block_size=4`` means 4 sync points per verify.
        # This is inherent to autoregressive feature-space drafting
        # (you can't speculate position k+1 without knowing the token
        # at position k), so doubling ``block_size`` doubles sync
        # count, not just compute. Worth keeping in mind when tuning.
        # The classic-speculative path avoids this by letting the
        # draft LM hold the integer token in its own KV cache.
        draft_tokens: list[int] = []
        cur_token = self._seed_token
        cur_hidden = self._seed_hidden
        for _ in range(self._block_size):
            tok_in = mx.array([[cur_token]], dtype=mx.int32)
            logits, h_new = self._draft(
                token_ids=tok_in, h_prev=cur_hidden, cache=self._draft_cache
            )
            # ``logits`` is typed ``mx.array | None`` because the draft's
            # ``compute_logits=False`` path returns None. We pass the
            # default ``compute_logits=True`` here so logits must be an
            # array, but pyright can't narrow across the union and a
            # future refactor that flipped the flag would crash on the
            # ``logits[:, -1, :]`` slice with a confusing ``NoneType``
            # error rather than at the call site. Match the explicit-
            # raise pattern used in ``_eagle_loss``.
            if logits is None:
                raise RuntimeError(
                    "EagleDraftModel returned None logits in the draft "
                    "loop despite compute_logits=True (default). This is "
                    "an olmlx bug — please file an issue."
                )
            sampled = int(mx.argmax(logits[:, -1, :], axis=-1).item())
            draft_tokens.append(sampled)
            cur_token = sampled
            cur_hidden = h_new

        # ---- Verify phase: target forward on [seed_token, *drafts].
        # Length is block_size + 1; target's logit at position i is its
        # prediction for position i+1, so logits[0] predicts what
        # *should* follow seed_token (i.e. should match draft[0]),
        # logits[1] should match draft[1], etc.
        # Clear the GDN capture's per-step state so this verify's
        # snapshots don't pile on top of last step's. ``rollback``
        # replays positionally over ``self._capture._gdn_inputs``, so
        # leftover entries would be applied to the wrong layer.
        if self._capture is not None:
            self._capture.clear()
        # Reset the hidden-capture slot to ``None`` before the verify
        # forward. Without this, a silent hook failure on step 2+ would
        # leave last step's hidden in place and the ``is None`` guard
        # below would never fire — the decoder would happily seed the
        # next step with a stale hidden one verify out of date. The
        # guard only catches step-1 failures (right after ``prefill``
        # resets storage), which doesn't cover the actual common
        # mode (transient hook detach mid-stream).
        self._hidden_storage[0] = None
        verify_input = mx.array([[self._seed_token, *draft_tokens]], dtype=mx.int32)
        target_out = self._target(verify_input, cache=self._target_cache)
        target_logits = _logits(target_out)
        captured = self._hidden_storage[0]
        if captured is None:
            raise RuntimeError(
                "EagleDecoder.step: target verify did not populate the "
                "hidden capture slot."
            )

        # ``verify_draft_greedy`` expects the per-position-prediction
        # logits, i.e. shape (n+1, vocab) where n = len(draft_tokens).
        # ``target_logits`` is (B=1, n+1, vocab); squeeze batch.
        flat_logits = target_logits[0]
        accepted = verify_draft_greedy(draft_tokens, flat_logits)

        # ---- Cache trimming + state rotation.
        # We accepted ``num_accepted = len(accepted)`` tokens. The
        # caches grew by block_size+1 (target) and block_size (draft)
        # during this step. Trim them back to the accepted prefix.
        num_accepted = len(accepted)
        # Target and draft caches trim by the same amount.
        # - Target grew by block_size + 1; we keep num_accepted; trim
        #   away (block_size + 1 - num_accepted).
        # - Draft grew by block_size; we keep num_accepted - 1 (the
        #   draft emits drafts, not the seed); trim away
        #   (block_size - (num_accepted - 1)) = (block_size + 1 -
        #   num_accepted).
        # Both algebra paths land at the same value. Keep one
        # variable so a future change to either side has to update
        # the trim formula in one spot — two identically-computed
        # variables would be a maintenance trap (e.g. EAGLE-2 tree
        # speculation could change the draft accounting while target
        # stays the same).
        trim = (self._block_size + 1) - num_accepted
        if trim > 0:
            if self._target_can_trim:
                if trim_prompt_cache is not None:
                    trim_prompt_cache(self._target_cache, trim)
            else:
                # Hybrid linear-attention path. ``_capture`` was created
                # in prefill; same control-flow invariant guards as
                # DFlash's ``step``.
                if self._capture is None:
                    raise RuntimeError(
                        "EagleDecoder internal invariant violated: target "
                        "cache is non-trim-able but no GDN capture was "
                        "installed. This is an olmlx bug."
                    )
                # ``rollback(cache, accepted, trim)`` internally slices
                # the captured GDN inputs to ``q[:, :accepted + 1]``
                # — i.e. the first ``accepted + 1`` positions of *this
                # step's* capture. The capture is populated by the
                # verify forward over ``verify_input = [seed_token,
                # *draft_tokens]`` (length ``block_size + 1``), so its
                # position 0 is the seed_token regardless of whether
                # the seed came from prefill (step 1) or from the
                # previous step's accepted tail (step 2+).
                #
                # We want to keep ``num_accepted`` positions of this
                # step's verify in the cache. Passing
                # ``accepted = num_accepted - 1`` yields the slice
                # ``q[:, :num_accepted]`` = positions 0..num_accepted-1
                # of [seed, draft_1, ..., draft_{bs}], which is
                # exactly the prefix that stays in the cache after
                # ``target_trim``. The DFlash caller uses the same
                # arithmetic for the same reason — its verify input
                # is ``[pending, MASK*bs]`` where ``pending`` plays
                # the same position-0 role as our ``seed_token``.
                self._capture.rollback(self._target_cache, num_accepted - 1, trim)
        # Draft cache trim — the draft is always a standard-attention
        # transformer with trim-able KVCache, so this path is the same
        # regardless of target architecture.
        if trim_prompt_cache is not None and trim > 0:
            trim_prompt_cache(self._draft_cache, trim)

        # New seed: the last accepted token, with target's hidden at
        # the corresponding position. captured spans positions
        # [0..block_size]; the last accepted-draft position is
        # num_accepted - 1.
        self._seed_token = accepted[-1]
        self._seed_hidden = captured[:, num_accepted - 1 : num_accepted, :]

        # Stats.
        self._stats_steps += 1
        self._stats_proposed += self._block_size
        self._stats_accepted_draft += max(num_accepted - 1, 0)

        return accepted, max(num_accepted - 1, 0)
