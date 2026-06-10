"""MTP decoder: prefill / step / reset protocol over an MTP draft.

Mirrors the same ``SpeculativeDecoder`` protocol used by classic
speculative decoding and DFlash, so the existing
``speculative_stream_generate`` bridge works unchanged. The crucial
difference is that MTP's draft is **autoregressive in feature
space** — it consumes ``(token_prev, h_prev)`` and produces ``(logits,
h_new)``, recursing for ``block_size`` draft tokens before each verify.

Unlike EAGLE, the MTP draft chains the **pre-final-norm hidden** of the
target: ``h_prev`` is the layer output before ``model.norm`` is applied,
matching the DeepSeek/Qwen MTP convention.

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
import threading
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from olmlx.engine.gdn_rollback import (
    _HAS_GDN,
    find_gdn_class as _find_gdn_class,
    get_model_layers as _get_layers,
)
from olmlx.engine.mtp.draft_model import MTPDraftModel
from olmlx.engine.spec_decoder_base import SpecDecoderBase
from olmlx.engine.speculative import (
    PrefillCancelled,
    _chunked_prefill,
)

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


class MTPDecoder(SpecDecoderBase):
    """Autoregressive draft + verify protocol for MTP.

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
        draft_model: MTPDraftModel,
        block_size: int = 3,
        target_layer_id: int | None = None,
    ):
        super().__init__()
        if make_prompt_cache is None:
            raise RuntimeError(
                "mlx_lm.models.cache is unavailable; MTPDecoder requires it "
                "for prompt-cache management."
            )
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1; got {block_size}")
        self._target = target_model
        self._draft = draft_model
        self._block_size = block_size
        # Default to the deepest (last) layer, i.e. ``len(layers) - 1``.
        # The shipped MTP head always consumes the target's final-layer
        # (pre-``model.norm``) hidden — there is no training step that
        # could pin it elsewhere — so this default is the right and only
        # value in normal use. ``target_layer_id`` stays overridable for
        # tests and future heads, but mismatching it against the layer
        # the head expects collapses acceptance toward zero.
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
                f"0..{num_target_layers - 1}). The MTP draft was likely "
                f"trained against a target of a different depth; retrain "
                f"with `olmlx mtp prepare` against the current target."
            )
        self._target_layer_id = target_layer_id
        self._hidden_storage: list[Any] = [None]

        # Per-request state populated by ``prefill``. The hook/capture
        # lifecycle fields (``_patched``/``_bound``/``_capture``/
        # ``_capture_buffer``) and the stats counters live on
        # ``SpecDecoderBase``; the GDN capture is created in ``prefill``
        # only when the target cache is non-trim-able (Qwen3.5/3.6
        # hybrid linear-attention case).
        self._target_cache: list | None = None
        self._draft_cache: list | None = None
        self._seed_token: int | None = None
        self._seed_hidden: mx.array | None = None
        self._target_can_trim: bool = True

    # ----- lifecycle -------------------------------------------------------

    def _reset_state(self) -> None:
        self._target_cache = None
        self._draft_cache = None
        self._seed_token = None
        self._seed_hidden = None
        self._target_can_trim = True
        self._hidden_storage = [None]

    def _stats_extra(self) -> dict[str, Any]:
        return {"block_size": self._block_size}

    # ----- prefill ---------------------------------------------------------

    def _prefill_impl(
        self,
        prompt: mx.array,
        *,
        segmented: Any = None,
        cancel_event: threading.Event | None = None,
    ) -> int:
        """Run the target on the prompt; return the first sampled token.

        ``prompt`` shape: ``(1, seq_len)`` int tokens. After prefill the
        decoder holds the target's last-layer hidden at position
        ``seq_len - 1`` and the greedily-sampled first token.

        ``segmented`` is accepted for the canonical ``SpecDecoderBase``
        signature but ignored (MTP has no snapshot store — always fresh
        prefill). ``cancel_event`` is honored at each pass-1 sub-chunk
        boundary (and before the pass-2 single-token forward): a client
        disconnect during a long agentic prefill raises
        :class:`PrefillCancelled`, bounding the post-cancel GPU work to
        one sub-chunk. ``speculative_stream_generate`` catches it for a
        clean, token-less stream exit.
        """
        # Hook the chosen target layer so its output is captured into
        # ``_hidden_storage[0]`` on every target forward.
        self._install_layer_hooks([self._target_layer_id], self._hidden_storage)
        self._bind_draft()

        # Build fresh caches for both models.
        self._target_cache = make_prompt_cache(self._target)
        self._draft_cache = self._draft.make_cache()

        # (Classic ``SpeculativeDecoder.prefill`` resets ``_position_ids``
        # / ``_rope_deltas`` on the target here for mlx-vlm targets,
        # but MTP blocks VLM targets at load time in
        # ``_load_mtp_decoder`` — those attrs are unreachable from
        # this path and we don't replicate the reset.)

        # Pick the cache-trimming regime: trim_prompt_cache for
        # standard attention, or _GDNStateCapture for hybrid linear-
        # attention targets whose caches are non-trim-able.
        self._target_can_trim = can_trim_prompt_cache is None or can_trim_prompt_cache(
            self._target_cache
        )
        if not self._target_can_trim:
            if not _HAS_GDN:
                raise RuntimeError(
                    "Target model has non-trim-able KV cache (likely "
                    "GatedDeltaNet linear-attention layers) but "
                    "mlx_lm.models.gated_delta is unavailable. Cannot "
                    "perform MTP rejection rollback."
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
                raise RuntimeError(
                    "Target reports a non-trim-able KV cache but no "
                    "``GatedDeltaNet`` submodule is present. MTP's "
                    "non-trim rollback path is GDN-specific; some other "
                    "cache type (custom quant cache that doesn't "
                    "support trim, mixed-architecture model with an "
                    "unfamiliar stateful cache) must be at play. "
                    "Disable MTP for this target or replace the "
                    "non-trim cache with a trim-able variant."
                )
        # Any exception from here on (lock contention or unexpected state
        # inside ``GDNStateCapture.__init__``; OOM / Metal stream / shape
        # mismatch inside the forward) is handled by ``SpecDecoderBase.
        # prefill``'s try/reset, so the model is never left patched and
        # the ``_GDN_PATCH_LOCK`` never stays held.
        if not self._target_can_trim:
            # Install the patch — must happen *before* the prompt forward
            # so the forwards run the patched ``__call__`` that maintains
            # the GDN conv/delta cache state. ``GDNStateCapture.__init__``
            # acquires ``_GDN_PATCH_LOCK``; ``reset()`` releases it via
            # ``capture.close()``. ``_install_gdn_capture`` locates the GDN
            # class, constructs the capture and a buffer pre-populated with
            # the target's GDN modules in forward-pass order. The buffer is
            # left *inactive* during prefill (see below) — the prompt
            # forward's snapshots are never consumed: ``step()`` clears the
            # buffer before every verify (``_step_impl``), and rollback
            # replays only over the current step's verify window.
            self._install_gdn_capture()

        # Run the target on the prompt and capture its last-layer hidden.
        # Two-pass: prefix fills the KV cache without materialising the
        # full [batch, seq_len, vocab] logit; final single-token pass
        # produces a [1, 1, vocab] logit and refreshes the capture slot.
        #
        # Pass 1 is sub-chunked via ``_chunked_prefill`` (mirrors the
        # classic/PLD decoders): a single forward over a long prefix forces
        # ``lm_head`` over every position — a [1, seq-1, vocab] tensor that
        # exceeds Metal's ~41 GB single-buffer limit on agentic prompts and
        # 500s the request (#360). Chunking bounds peak activation memory to
        # one sub-chunk and evals/clears the cache between chunks. Standard
        # KV-cached attention (and GDN linear-attention recurrence) is
        # chunking-invariant, so the resulting cache state is identical to a
        # single forward. ``_chunked_prefill`` also honours ``cancel_event``
        # at each sub-chunk boundary.
        #
        # GDN capture is suppressed across the chunked prefix
        # (``use_buffer(None)``, mirroring ``SpeculativeDecoder``):
        # ``_capturing_gdn_call`` *appends* per forward, so leaving it
        # active would pile every sub-chunk's q/k/v/conv tensors into the
        # buffer and hold them live across all chunks — re-bloating exactly
        # the memory the chunking just bounded, on the hybrid GDN targets
        # (Qwen3.6) that are MTP's primary use case. It is re-enabled before
        # pass-2 so the buffer is active when ``step()``'s verify runs
        # (``step()`` never re-enables it; it only clears between verifies).
        # No try/finally is needed to restore the buffer on error: if
        # ``_chunked_prefill`` raises (cancel, OOM), the enclosing
        # ``except`` runs ``reset()`` which closes the capture outright, so
        # the transiently-suppressed buffer is never observed — a failed
        # prefill forces the caller to re-``prefill`` before any ``step()``.
        if prompt.shape[1] > 1:
            if self._capture is not None:
                self._capture.use_buffer(None)
            _chunked_prefill(
                self._target,
                prompt[:, :-1],
                self._target_cache,
                cancel_event=cancel_event,
            )
            if self._capture is not None:
                self._capture.use_buffer(self._capture_buffer)
            # Discard the pass-1 capture slot so the pass-2 None-check below
            # is an active guard rather than dead code.
            self._hidden_storage[0] = None
            if cancel_event is not None and cancel_event.is_set():
                raise PrefillCancelled()
            target_out = self._target(prompt[:, -1:], cache=self._target_cache)
        else:
            if self._capture is not None:
                self._capture.use_buffer(self._capture_buffer)
            if cancel_event is not None and cancel_event.is_set():
                raise PrefillCancelled()
            target_out = self._target(prompt, cache=self._target_cache)
        target_logits = _logits(target_out)
        captured = self._hidden_storage[0]
        if captured is None:
            raise RuntimeError(
                "MTPDecoder prefill: target forward did not populate "
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
        self._seed_token = seed_token
        return seed_token

    # ----- step ------------------------------------------------------------

    def _step_impl(self) -> tuple[list[int], int]:
        """Generate ``block_size`` draft candidates, verify against the
        target, return ``(accepted_tokens, num_accepted_draft)``.

        ``num_accepted_draft`` is the number of *draft* tokens accepted
        (between 0 and ``block_size``). The returned ``accepted_tokens``
        list always has length ``num_accepted_draft + 1``: zero or more
        accepted draft tokens followed by the target's preferred token
        at the first mismatch (or the target's bonus token if all
        drafts were accepted).

        ``SpecDecoderBase.step`` wraps this in try/reset: a mid-step
        exception in the draft loop (MLX Metal error during ``.item()``)
        or the verify forward (OOM, shape mismatch) would otherwise
        leave both KV caches partially modified and the GDN capture's
        ``_gdn_inputs`` list with leftover entries from the failed
        verify. The forced ``reset()`` makes the caller re-``prefill``
        to recover — the right semantic for a hard inference failure.
        """
        if self._target_cache is None or self._draft_cache is None:
            raise RuntimeError(
                "MTPDecoder.step() called before prefill(); call "
                "prefill(prompt) to populate caches first."
            )
        if self._seed_token is None or self._seed_hidden is None:
            raise RuntimeError("MTPDecoder.step(): seed state is unset")

        # ---- Draft phase: produce block_size candidates autoregressively.
        #
        # Note: the ``.item()`` below forces an MLX eval each draft
        # step because the *next* draft iteration needs the integer
        # token id to build its input. That's one sync per draft
        # position — ``block_size=3`` means 3 sync points per verify.
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
            # error rather than at the call site, so raise explicitly.
            if logits is None:
                raise RuntimeError(
                    "MTPDraftModel returned None logits in the draft "
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
        # snapshots don't pile on top of last step's. ``rollback_single``
        # replays positionally over the buffer, so leftover entries
        # would be applied to the wrong layer.
        if self._capture_buffer is not None:
            self._capture_buffer.clear()
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
                "MTPDecoder.step: target verify did not populate the "
                "hidden capture slot."
            )

        # ``verify_draft_greedy`` expects the per-position-prediction
        # logits, i.e. shape (n+1, vocab) where n = len(draft_tokens).
        # ``target_logits`` is (B=1, n+1, vocab); squeeze batch.
        flat_logits = target_logits[0]
        accepted = self._verify_greedy(draft_tokens, flat_logits)

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
                if self._capture is None or self._capture_buffer is None:
                    raise RuntimeError(
                        "MTPDecoder internal invariant violated: target "
                        "cache is non-trim-able but no GDN capture was "
                        "installed. This is an olmlx bug."
                    )
                # ``rollback_single(buffer, cache, accepted, trim)``
                # internally slices the captured GDN inputs to
                # ``q[:, :accepted + 1]`` — i.e. the first
                # ``accepted + 1`` positions of *this step's* capture.
                # The capture is populated by the verify forward over
                # ``verify_input = [seed_token, *draft_tokens]``
                # (length ``block_size + 1``), so its position 0 is the
                # seed_token regardless of whether the seed came from
                # prefill (step 1) or from the previous step's accepted
                # tail (step 2+).
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
                self._capture.rollback_single(
                    self._capture_buffer,
                    self._target_cache,
                    accepted=num_accepted - 1,
                    trim=trim,
                )
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
