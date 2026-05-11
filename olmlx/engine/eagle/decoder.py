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
        if target_layer_id is None:
            num = len(_get_layers(target_model))
            target_layer_id = num - 1
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

        # Reset transient mlx-vlm state if present (mirrors classic
        # SpeculativeDecoder).
        for attr in ("_position_ids", "_rope_deltas"):
            if hasattr(self._target, attr):
                setattr(self._target, attr, None)

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
            self.reset()
            raise RuntimeError(
                "EagleDecoder prefill: target forward did not populate the "
                f"hidden capture slot for layer {self._target_layer_id}. "
                "Either ``_patch_model`` is incompatible with this target "
                "structure, or the chosen layer wasn't visited."
            )

        # Greedy sample at the prompt-tail position.
        last_logits = target_logits[:, -1:, :]
        seed_token = int(mx.argmax(last_logits, axis=-1).item())
        # Hidden at the prompt-tail position becomes the conditioning
        # for the first draft step.
        self._seed_hidden = captured[:, -1:, :]
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

        # ---- Draft phase: produce block_size candidates autoregressively.
        draft_tokens: list[int] = []
        cur_token = self._seed_token
        cur_hidden = self._seed_hidden
        for _ in range(self._block_size):
            tok_in = mx.array([[cur_token]], dtype=mx.int32)
            logits, h_new = self._draft(
                token_ids=tok_in, h_prev=cur_hidden, cache=self._draft_cache
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
        # Target cache grew by block_size + 1; we keep num_accepted
        # tokens; trim away (block_size + 1 - num_accepted).
        target_trim = (self._block_size + 1) - num_accepted
        # Draft cache grew by block_size; we keep num_accepted - 1
        # (the draft only emits drafts, not the seed). Trim away
        # (block_size - (num_accepted - 1)) = (block_size + 1 - num_accepted).
        draft_trim = (self._block_size + 1) - num_accepted
        if target_trim > 0:
            if self._target_can_trim:
                if trim_prompt_cache is not None:
                    trim_prompt_cache(self._target_cache, target_trim)
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
                # ``rollback(cache, accepted, trim)``: ``accepted`` is
                # the count of *draft* tokens accepted (excluding the
                # seed and the bonus position) — for EAGLE that's
                # ``num_accepted - 1`` because the seed is included in
                # the verify input but is also already in the cache
                # from the previous step's accepted tail.
                #
                # Invariant the ``num_accepted - 1`` accounting relies
                # on: when ``step()`` begins, the GDN recurrent state
                # in ``self._target_cache`` *already incorporates* the
                # seed_token's update — the previous step's verify
                # ran with ``verify_input = [old_seed, *prev_drafts]``,
                # the trim path replayed
                # ``gated_delta_update(q[:, :num_accepted_prev], ...)``
                # which includes the slot we now call ``seed_token``,
                # and ``_seed_token = accepted[-1]`` (line ~404) names
                # that already-incorporated tail. So the *new* GDN
                # positions this step contributes are exactly the
                # accepted drafts (1..num_accepted - 1), not the seed
                # at position 0 — replaying with ``accepted + 1 = n``
                # where ``accepted = num_accepted - 1`` slices
                # ``q[:, :num_accepted]`` of this step's capture, which
                # is the right prefix: it stops *before* the rejected
                # draft and yields the GDN state that matches the
                # trimmed token sequence. If a future change moves
                # the seed-token's GDN update out of the previous
                # step's replay (e.g. by emitting the seed *outside*
                # the verify input), the ``accepted - 1`` accounting
                # here must be revisited.
                self._capture.rollback(
                    self._target_cache, num_accepted - 1, target_trim
                )
        # Draft cache trim — the draft is always a standard-attention
        # transformer with trim-able KVCache, so this path is the same
        # regardless of target architecture.
        if trim_prompt_cache is not None and draft_trim > 0:
            trim_prompt_cache(self._draft_cache, draft_trim)

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
