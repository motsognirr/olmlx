"""DFlash block-diffusion speculative decoder.

Implements the same ``prefill``/``step``/``reset`` protocol as
``SpeculativeDecoder`` so the existing ``speculative_stream_generate``
streaming bridge works unchanged.

Universal target support is achieved by monkey-patching the target's
selected layers with ``_LayerHook`` (see ``_patch_model``) — this works
for any model whose layers list lives at one of three known locations
(``model.layers`` / ``model.model.layers`` / ``model.language_model.layers``).
No per-architecture adapter code is needed.

For target architectures whose KV cache cannot be trimmed in-place
(notably Qwen3.5 / Qwen3-Coder-Next with ``GatedDeltaNet`` linear-attention
layers), ``GDNStateCapture`` from :mod:`olmlx.engine.gdn_rollback`
monkey-patches ``GatedDeltaNet.__call__`` to snapshot the conv + GDN
state per draft step and replays ``gated_delta_update`` on the accepted
prefix to restore the correct state on rejection.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import (
    RotatingKVCache,
    can_trim_prompt_cache,
    make_prompt_cache,
)

from olmlx.engine.dflash.draft_model import DFlashDraftModel, DraftConfig
from olmlx.engine.gdn_rollback import (
    # Re-export: dflash/prepare.py, tests, and scripts import ``_get_layers``
    # from this module.
    get_model_layers as _get_layers,  # noqa: F401
)
from olmlx.engine.spec_decoder_base import (
    SpecDecoderBase,
    # Canonical home moved to spec_decoder_base (#467); re-exported here
    # because cli, prepare, eagle/mtp tests and scripts import them from
    # this module.
    _LayerHook as _LayerHook,
    _patch_model as _patch_model,
    _unpatch_model as _unpatch_model,
)
from olmlx.engine.speculative import _eval_cache


# ``_trim_recent_cache`` reaches into ``RotatingKVCache._temporal_order`` and
# ``._idx`` to reorder + slice the rotating buffer. These are private to mlx-lm
# and may be renamed without a semver bump. Probe at import time so an
# incompatible mlx-lm release fails fast (rather than mid-generation when
# DFlash first hits a sliding-window draft cache). ``_temporal_order`` is a
# method (class-level), but ``_idx`` is set in ``__init__`` (instance-level),
# so it must be probed via a sentinel instance — and its semantic (integer
# write-position counter that advances with ``update_and_fetch``) must hold
# too, otherwise the trim writes a corrupt value back.
def _probe_rotating_kv_privates() -> bool:
    if not hasattr(RotatingKVCache, "_temporal_order"):
        return False
    try:
        c = RotatingKVCache(max_size=4, keep=0)
        if not hasattr(c, "_idx"):
            return False
        if not isinstance(c._idx, int) or c._idx != 0:
            return False
        # One ``update_and_fetch`` should advance ``_idx`` by exactly the
        # number of tokens written. If the semantic changed (e.g. it
        # became a ring-buffer wrap counter or a bool flag), the trim
        # math in ``_trim_recent_cache`` would silently corrupt state.
        k = v = mx.zeros((1, 1, 1, 4))
        c.update_and_fetch(k, v)
        return isinstance(c._idx, int) and c._idx == 1
    except Exception:
        return False


_HAS_ROTATING_KV_PRIVATES = _probe_rotating_kv_privates()

logger = logging.getLogger(__name__)


def _trim_recent_cache(cache: list[Any], num_tokens: int) -> None:
    """Trim trailing ``num_tokens`` from each layer's KV cache.

    Special-cases ``RotatingKVCache`` (must reorder before slicing).
    Skips entries with no ``trim`` method — the GDN rollback path
    handles those separately.
    """
    if num_tokens <= 0:
        return
    for c in cache:
        n = min(getattr(c, "offset", num_tokens), num_tokens)
        if n <= 0:
            continue
        if isinstance(c, RotatingKVCache) and c.keys is not None:
            if not _HAS_ROTATING_KV_PRIVATES:
                raise RuntimeError(
                    "DFlash trims target or draft KV caches with rotating "
                    "windows via the private mlx-lm API "
                    "``RotatingKVCache._temporal_order`` / ``._idx``. The "
                    "installed mlx-lm version no longer exposes them — pin "
                    "a compatible version or file an olmlx bug to update "
                    "the private-API access pattern."
                )
            # ``cache.offset`` is the *absolute* token counter mlx-lm
            # passes to RoPE as the next-write base position — not the
            # buffer size. A previous version set ``c.offset =
            # actual_stored - n``, which threw away the absolute count
            # and applied wrong RoPE positions to all subsequent
            # writes for sliding-window targets that had already
            # rotated. Decrementing ``c.offset`` by exactly ``n``
            # preserves the absolute-position semantic. ``c._idx`` is
            # the in-buffer write pointer; setting it to the new buffer
            # length leaves ``update_and_fetch`` to grow / rotate the
            # buffer normally on the next write.
            actual_stored = min(c.offset, c.keys.shape[2])
            n = min(n, actual_stored)
            c.keys = c._temporal_order(c.keys)
            c.values = c._temporal_order(c.values)
            c.keys = c.keys[..., :-n, :]
            c.values = c.values[..., :-n, :]
            c.offset = c.offset - n
            c._idx = c.keys.shape[2]
        elif hasattr(c, "trim"):
            c.trim(n)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


class DFlashDecoder(SpecDecoderBase):
    """Block-diffusion speculative decoder.

    Each ``step()`` builds a masked block ``[pending_token, MASK,
    MASK, ...]`` of length ``block_size + 1``, runs one parallel draft
    forward pass to produce ``block_size`` candidate tokens, then runs
    one verification forward pass through the target. Greedy verification
    accepts the longest matching prefix; the bonus token comes from the
    target. On rejection the target cache (or GDN state for hybrid
    models) is rolled back.

    ``block_size`` is the number of *draft* tokens per step (matches
    ``SpeculativeDecoder``'s ``num_speculative_tokens``). The total
    block length passed through the draft is ``block_size + 1`` because
    the pending token occupies position 0 (sliced off via ``logits_start=1``).
    """

    def __init__(
        self,
        target_model: nn.Module,
        draft_model: DFlashDraftModel,
        draft_config: DraftConfig,
        block_size: int = 16,
    ):
        super().__init__()
        self._target = target_model
        self._draft = draft_model
        self._config = draft_config
        self._block_size = block_size

        # State (populated by prefill()). The hook/capture lifecycle
        # fields (``_patched``/``_bound``/``_capture``/``_capture_buffer``)
        # and the stats counters live on ``SpecDecoderBase``.
        self._target_cache: list[Any] | None = None
        self._draft_cache: list[Any] | None = None
        self._target_can_trim: bool = True
        self._hidden: mx.array | None = None
        self._pending_token: int | None = None
        self._prompt_size: int = 0
        # Per-layer hidden state captured by ``_LayerHook``. Owned by the
        # decoder (not the target ``nn.Module``) so captured ``mx.array``s
        # never appear in ``target.parameters()``.
        self._hidden_capture: list[Any] = []
        # Counts the *generated* tokens (matching upstream's ``n``);
        # used to compute draft-cache trim amounts.
        self._n_generated: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _reset_state(self) -> None:
        self._target_cache = None
        self._draft_cache = None
        self._target_can_trim = True
        self._hidden = None
        self._pending_token = None
        self._prompt_size = 0
        self._hidden_capture = []
        self._n_generated = 0

    def _stats_extra(self) -> dict[str, Any]:
        return {"lambda": self._block_size}

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def _prefill_impl(
        self,
        prompt: mx.array,
        *,
        segmented: Any = None,
        cancel_event: threading.Event | None = None,
    ) -> int:
        """Process the prompt through the target, capturing hidden states.

        Returns the first generated token (target greedy argmax).

        ``segmented`` and ``cancel_event`` are accepted for the canonical
        ``SpecDecoderBase`` signature but not honored: DFlash has no
        snapshot store, prefills the target in a single forward (no
        sub-chunk loop to check between), and is an experimental strategy
        off the default path.
        """
        target_layer_ids = list(self._config.target_layer_ids)
        # Build the target cache before patching: ``make_prompt_cache``
        # walks ``model.layers`` to pick a per-layer cache type (sliding
        # vs. full attention) by probing the layer object. Today it uses
        # ``hasattr``, which ``_LayerHook.__getattr__`` proxies through,
        # but a future ``isinstance`` check would silently get the wrong
        # cache type for patched layers. Doing the cache build first
        # decouples cache selection from the patch.
        self._target_cache = make_prompt_cache(self._target)
        self._hidden_capture = [None] * len(target_layer_ids)
        self._install_layer_hooks(target_layer_ids, self._hidden_capture)
        self._bind_draft()

        # Call the draft's own ``make_cache`` directly. ``make_prompt_cache``
        # would defer to it via ``hasattr(model, "make_cache")`` today, but
        # going through the public method keeps the per-layer
        # ``RotatingKVCache`` / ``KVCache`` selection (driven by
        # ``DraftConfig.layer_types``) explicit at the call site rather
        # than relying on mlx-lm's dispatch.
        self._draft_cache = self._draft.make_cache()
        self._target_can_trim = can_trim_prompt_cache(self._target_cache)
        if not self._target_can_trim:
            # ``GDNStateCapture.for_model`` raises with a clear message
            # if mlx_lm.models.gated_delta is unavailable or if the
            # target has no ``GatedDeltaNet`` submodule, so a separate
            # pre-check would just duplicate the same error.
            self._install_gdn_capture()
            self._capture.use_buffer(self._capture_buffer)

        # Two-pass prefill avoids materialising the full [batch, N, vocab] logit
        # (the OOM path for large-vocab models on long contexts).
        #
        # Pass 1 — prefix (positions 0..N-2): fills the KV cache and captures
        # hidden states for ALL prefix positions. The model output is discarded,
        # so lm_head on N-1 positions is never evaluated by MLX's lazy scheduler.
        # Pass 2 — last token: produces a [1, 1, vocab] logit + captures the
        # last position's hidden state.
        #
        # DFlash's draft conditions on self._hidden[1, N, …] (all prompt hiddens),
        # so we concatenate captured_prefix + captured_last along the time axis.
        _err_msg = (
            "Target forward did not populate all configured "
            "target_layer_ids — check that the layer indices exist on "
            f"this model (got {target_layer_ids})."
        )
        if prompt.shape[1] > 1:
            prefix, last = prompt[:, :-1], prompt[:, -1:]
            self._target(prefix, cache=self._target_cache)  # output discarded
            captured_prefix = list(self._hidden_capture)
            if any(h is None for h in captured_prefix):
                raise RuntimeError(_err_msg)
            # Force pass-1 hiddens before dropping slot references, so
            # correctness does not depend on _eval_cache transitively
            # materialising the same graph.
            mx.eval(*captured_prefix)
            # Reset capture slots so the pass-2 None-check is independent.
            self._hidden_capture[:] = [None] * len(self._hidden_capture)
            _eval_cache(self._target_cache)
            target_out = self._target(last, cache=self._target_cache)
            captured_last = list(self._hidden_capture)
            if any(h is None for h in captured_last):
                raise RuntimeError(_err_msg)
            captured = [
                mx.concatenate([p, q], axis=1)
                for p, q in zip(captured_prefix, captured_last)
            ]
        else:
            target_out = self._target(prompt, cache=self._target_cache)
            captured = list(self._hidden_capture)
            if any(h is None for h in captured):
                raise RuntimeError(_err_msg)
        logits = _logits(target_out)
        self._hidden = mx.concatenate(captured, axis=-1)
        last_logit = logits[:, -1, :]
        mx.eval(last_logit, self._hidden)

        self._prompt_size = int(prompt.shape[1])
        first_token = int(mx.argmax(last_logit, axis=-1).item())
        self._pending_token = first_token
        self._n_generated = 1
        return first_token

    def _step_impl(self) -> tuple[list[int], int]:
        """One block-diffusion speculative step. ``SpecDecoderBase.step``
        wraps this in try/reset so a mid-step exception (Metal error,
        OOM, rollback failure) can never leave corrupt caches behind
        (#460)."""
        # ``raise`` rather than ``assert`` — these are API-contract
        # guards, not debug checks, and ``assert`` is stripped under
        # ``python -O``.
        if (
            self._target_cache is None
            or self._draft_cache is None
            or self._pending_token is None
            or self._hidden is None
        ):
            raise RuntimeError("Call prefill() before step()")

        pending = self._pending_token
        bs_total = self._block_size + 1  # block length including pending token
        mask_id = int(self._config.mask_token_id)

        # 1. Draft a block in one parallel forward pass.
        block = mx.array([[pending] + [mask_id] * self._block_size])
        draft_logits = self._draft(
            block, self._hidden, self._draft_cache, logits_start=1
        )
        # Invariant (full-attention layers only): the draft cache stores
        # exactly the context tokens corresponding to ``prompt_size +
        # n_generated - 1`` target positions. The draft processes
        # ``S = _hidden.shape[1]`` ctx tokens per step (= previous
        # step's ``num_accepted``), so a full-attention layer's offset
        # advances in lockstep with ``_n_generated``. Sliding-window
        # layers truncate ``x_ctx`` to ``window - 1`` before
        # ``update_and_fetch`` and grow by ``min(S, window-1)`` instead,
        # so the invariant doesn't apply there. We only check the first
        # full-attention layer if one exists; the assertion catches
        # bookkeeping bugs without firing falsely on sliding drafts.
        ft_idx = next(
            (
                i
                for i, t in enumerate(self._config.layer_types)
                if t == "full_attention"
            ),
            None,
        )
        if ft_idx is not None:
            draft_offset = self._draft_cache[ft_idx].offset
            target_offset = self._prompt_size + self._n_generated - 1
            # Sliding-window draft layers do NOT desync this check, even
            # on full rejection (gh#453 conjectured they would). When a
            # sliding layer's context exceeds its window,
            # ``DFlashAttention.__call__`` truncates ``x_ctx`` to ``keep
            # = window - 1`` tokens and pre-advances ``cache.offset +=
            # skip`` for the dropped positions. The subsequent
            # ``RotatingKVCache.update_and_fetch`` then advances offset
            # by the *post-truncation* count (``keep``), so the net
            # advance is ``skip + keep == S`` — identical to a
            # full-attention layer. The skipped positions are real
            # (evicted from the window), so the pre-advance is correct
            # and is deliberately NOT rolled back on rejection: undoing
            # it would apply wrong RoPE positions to later context
            # writes. This check runs on a full-attention layer (which
            # never skips), whose offset therefore always equals
            # ``prompt_size + n_generated - 1`` by construction. The
            # guard remains a fail-stop against future bookkeeping bugs.
            # Locked by ``test_sliding_draft_offset_stays_in_lockstep_
            # on_full_rejection``.
            if draft_offset != target_offset:
                msg = (
                    f"DFlash internal invariant violated: draft cache offset "
                    f"({draft_offset}) at full-attention layer {ft_idx} does "
                    f"not match expected target offset ({target_offset}). "
                    "This is an olmlx bug — please report at "
                    "https://github.com/motsognirr/olmlx/issues with the "
                    "model name and prompt details."
                )
                logger.error(msg)
                raise RuntimeError(msg)
        draft_tokens_arr = mx.argmax(draft_logits, axis=-1)
        mx.eval(draft_tokens_arr)
        # ``mx.array.tolist()`` is typed as ``list_or_scalar``; for a 1-D
        # array of ints it always returns ``list[int]`` at runtime.
        draft_tokens: list[int] = draft_tokens_arr[0].tolist()  # type: ignore[assignment]

        # 2. Verify with the target in one parallel forward pass.
        if self._capture_buffer is not None:
            self._capture_buffer.clear()
        # Zero the capture slots before the verification forward. Each
        # hook fires once per ``__call__`` and overwrites its slot, but
        # a hook that *stops* firing mid-generation (e.g. layer skipped
        # by a future model change) would otherwise leave the previous
        # step's tensor in the slot — the ``None``-check below would
        # pass and we would silently feed stale hiddens to the next
        # draft step. ``prefill`` allocates the list fresh; ``step``
        # has to clear it explicitly. Slice-assign so the same list
        # object stays referenced by the installed hooks.
        self._hidden_capture[:] = [None] * len(self._hidden_capture)
        verify_input = mx.array([[pending] + draft_tokens])
        target_out = self._target(verify_input, cache=self._target_cache)
        logits = _logits(target_out)
        captured = list(self._hidden_capture)
        # Same guard as ``prefill``: an unfired hook here would surface
        # downstream as a cryptic ``mx.concatenate`` type error rather
        # than an actionable message.
        if any(h is None for h in captured):
            raise RuntimeError(
                "Target verification forward did not populate all "
                f"configured target_layer_ids "
                f"({list(self._config.target_layer_ids)}); a layer hook may "
                "have been removed mid-generation."
            )
        new_hidden = mx.concatenate(captured, axis=-1)
        mx.eval(logits, new_hidden)

        # 3. Greedy verification.
        verification_logits = logits[0]  # (block_size + 1, vocab)
        accepted = self._verify_greedy(draft_tokens, verification_logits)
        num_accepted = len(accepted)  # 1..block_size+1
        # accepted_drafts is the count BEFORE the bonus position
        # (excludes the target's correction or all-accepted bonus).
        accepted_drafts = num_accepted - 1

        # 4. Roll back caches: remove the unused tail of the verify block.
        trim = bs_total - num_accepted  # 0..block_size
        if trim > 0:
            if self._target_can_trim:
                _trim_recent_cache(self._target_cache, trim)
            else:
                # Same control-flow invariant as the prefill setup
                # (``_target_can_trim is False`` ⇒ ``_capture`` was
                # created), but use a hard error since ``assert`` is
                # stripped under ``python -O``.
                if self._capture is None or self._capture_buffer is None:
                    raise RuntimeError(
                        "DFlash internal invariant violated: target cache "
                        "is non-trimmable but no GDN capture was installed."
                    )
                self._capture.rollback_single(
                    self._capture_buffer,
                    self._target_cache,
                    accepted_drafts,
                    trim,
                )

        # 5. Slice hidden state to the accepted prefix; this becomes the
        # new "ctx" length for the next draft call. Length = num_accepted
        # which spans positions [pending_token, accepted_drafts...].
        self._hidden = new_hidden[:, :num_accepted, :]

        # 6. Update state.
        self._pending_token = accepted[-1]
        self._n_generated += num_accepted

        # Diagnostics.
        self._stats_steps += 1
        self._stats_proposed += self._block_size
        self._stats_accepted_draft += accepted_drafts

        return accepted, self._block_size


def _logits(out: Any) -> mx.array:
    """Unwrap mlx-vlm ``LanguageModelOutput`` if needed."""
    return getattr(out, "logits", out)
