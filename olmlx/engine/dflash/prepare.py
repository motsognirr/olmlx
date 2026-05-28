"""Train a DFlash draft model for a given target.

Pipeline mirrors ``engine/flash/prepare.py``:

1. Load the target via mlx-lm; freeze it.
2. Build a ``DraftConfig`` from the target's config + CLI overrides.
3. Construct ``DFlashDraftModel`` and ``bind()`` it to the target.
4. Stream training batches from a HuggingFace dataset (default:
   UltraChat) via ``training_data.stream_training_batches``.
5. For each batch: run target (no grad) to capture hidden states via
   the same ``_patch_model`` used at inference; pick a random pivot
   ``p`` shared across the batch; build a masked draft window
   ``[tokens[p], MASK*block_size]``; compute cross-entropy on the
   ``block_size`` masked positions vs. the original tokens.
6. AdamW + cosine schedule. Save to ``<model_dir>/dflash/{config.json,
   model-00001-of-00001.safetensors}`` in the upstream-compatible
   schema so ``_load_dflash_decoder`` can consume the result without
   any further translation.

Trains on **one random window per sequence per step**. That
under-uses each target forward pass (one window vs. potentially
``L - block_size`` windows) but keeps the loss/grad surface tiny and
debuggable.

Two acceleration paths are available:

- ``distill=True`` adds a Hinton-style KL term against the target's
  logits at the masked positions: ``(1 - alpha) * CE + alpha * T^2 *
  KL``. The target forward already runs to capture hiddens, so
  capturing logits is free.
- ``use_precomputed=<dir>`` reads precomputed
  ``(input_ids, target_hidden)`` shards from disk (produced by
  ``engine/dflash/precompute.py``) instead of running the target each
  step. Mutually exclusive with ``distill`` because precomputed shards
  do not store vocab-size logits.
"""

from __future__ import annotations

import json
import logging
import math
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils as mx_utils

from olmlx.engine.dflash.decoder import _get_layers, _patch_model, _unpatch_model
from olmlx.engine.dflash.draft_model import DFlashDraftModel, DraftConfig
from olmlx.engine.dflash.training_data import stream_training_batches

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# Paper-reported configuration (arxiv:2602.06036): 5 draft layers, 5
# target hidden states, block_size=16. The pre-#317 defaults of 4/4/4
# were ad-hoc and have not been bench-validated since the paper was
# published. See gh#317 (Gap 3).
DEFAULT_NUM_HIDDEN_LAYERS = 5
DEFAULT_NUM_TARGET_LAYERS = 5
DEFAULT_BLOCK_SIZE = 16  # number of draft tokens per step (== MASK count)
DEFAULT_STEPS = 2000
DEFAULT_BATCH_SIZE = 4
DEFAULT_SEQ_LEN = 2048
DEFAULT_LR = 5e-4
DEFAULT_WARMUP_FRAC = 0.05


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_target_config(model_path: Path) -> dict[str, Any]:
    cfg_path = model_path / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Target config not found at {cfg_path}")
    return json.loads(cfg_path.read_text())


def _text_config(target_cfg: dict[str, Any]) -> dict[str, Any]:
    """Return the text-tower portion of a target config.

    Multimodal targets (e.g. Qwen3.6's ``Qwen3_5MoeForConditionalGeneration``)
    nest text-tower fields under ``text_config`` and reserve the top level for
    cross-modal metadata (``vision_config``, image/video token ids, etc.). The
    DFlash draft only models text, so it should consume the nested block when
    present and fall through to the flat config otherwise.

    Defensive against unrelated ``text_config`` shapes: descend only when the
    nested block actually carries text-tower fields (``hidden_size`` is the
    canonical marker). A non-VLM model that happens to use the
    ``text_config`` key for an unrelated purpose would otherwise regress
    from a working flat lookup to a ``KeyError`` on the descent.
    """
    nested = target_cfg.get("text_config")
    if isinstance(nested, dict) and "hidden_size" in nested:
        return nested
    return target_cfg


def _evenly_spaced(num_layers: int, k: int) -> list[int]:
    """Pick ``k`` target-layer indices following the upstream DFlash recipe.

    Returns ``k`` indices evenly distributed across ``[1, num_layers - 3]``
    (inclusive), early-biased to match upstream
    ``build_target_layer_ids`` in z-lab/dflash. For ``num_layers=32,
    k=5`` this gives ``[1, 8, 15, 22, 29]``. The range avoids layer 0
    (the draft already sees that signal through ``embed_tokens``) and
    the final two layers (the bound ``lm_head`` already carries that
    information). See gh#317 (Gap 4).

    Falls back to a centred 0..num_layers-1 spread when the upstream
    range ``[1, num_layers - 3]`` is too small to fit ``k`` unique
    indices (i.e. ``num_layers - 3 < k``); note this fallback **can
    return layer 0**, in contradiction with the no-layer-0 invariant
    of the non-degenerate path — the trade is intentional so small
    synthetic targets in unit tests still produce ``k`` distinct
    indices. Returns ``list(range(num_layers))`` when ``k >=
    num_layers`` and ``[]`` when ``k <= 0``.
    """
    if k <= 0:
        return []
    if k >= num_layers:
        return list(range(num_layers))

    end = num_layers - 3
    if end < k:
        # Degenerate: the upstream [1, N-3] range is too small to fit
        # ``k`` unique indices (e.g. ``num_layers=4, k=2`` has range
        # size 1, ``num_layers=5, k=3`` has range size 2). Fall back
        # to a centred spread across the full layer range so small
        # synthetic targets used in unit tests still produce sensible
        # (and unique) indices.
        if k == 1:
            return [num_layers // 2]
        fallback_step = (num_layers - 1) / (k - 1)
        return sorted({int(round(i * fallback_step)) for i in range(k)})

    if k == 1:
        return [1]
    step = (end - 1) / (k - 1)
    # Dedupe + sort: rounding can collide for small ``num_layers``
    # close to ``k`` (e.g. ``num_layers=5, k=4`` would otherwise hit
    # duplicates). Duplicates would double-wrap the repeated layer in
    # ``_LayerHook`` and ``_unpatch_model`` only strips one level —
    # the dangling hook would corrupt subsequent captures.
    result = sorted({int(round(1 + i * step)) for i in range(k)})
    if len(result) < k:
        # Operator surprise: ``--num-target-layers`` won't match the
        # final ``num_target_layers`` baked into the saved
        # ``config.json``. Surface the rounding collision so
        # debugging is straightforward.
        logger.warning(
            "_evenly_spaced: requested %d layers from a %d-layer target "
            "but rounding collisions reduced the result to %d unique "
            "indices: %s. The saved config will reflect the deduplicated "
            "count.",
            k,
            num_layers,
            len(result),
            result,
        )
    return result


def _resolve_target_layer_ids(
    requested: list[int] | None,
    num_target_layers: int | None,
    target_num_layers: int,
) -> list[int]:
    """Pick the target_layer_ids list.

    Precedence: explicit list > evenly-spaced derivation from the
    ``num_target_layers`` count > 4 evenly-spaced layers.
    """
    if requested:
        for lid in requested:
            if not 0 <= lid < target_num_layers:
                raise ValueError(
                    f"target_layer_ids contains {lid}, out of range "
                    f"[0, {target_num_layers})"
                )
        # Duplicates would double-wrap a layer in ``_LayerHook`` and leak
        # the wrapper through ``_unpatch_model`` (which only strips one
        # level), corrupting hidden-state captures.
        if len(set(requested)) != len(requested):
            raise ValueError(
                f"target_layer_ids contains duplicate indices: {requested}"
            )
        return sorted(requested)
    k = num_target_layers or DEFAULT_NUM_TARGET_LAYERS
    return _evenly_spaced(target_num_layers, k)


def _build_draft_config(
    target_cfg: dict[str, Any],
    *,
    target_layer_ids: list[int],
    num_hidden_layers: int,
    block_size: int,
    mask_token_id: int,
) -> DraftConfig:
    """Derive a DraftConfig from the target's config.json.

    Falls back to sensible defaults for fields the target doesn't
    expose. The draft inherits hidden_size, head_dim, GQA shape,
    rope_theta, and vocab_size from the target so weights are
    dimensionally compatible at inference time.
    """

    # Multimodal targets put text-tower fields under ``text_config``; descend
    # into that block before reading any field.
    text_cfg = _text_config(target_cfg)

    # ``or`` would short-circuit on a degenerate-but-valid ``0.0`` /
    # ``0`` value (e.g. ``rms_norm_eps=0.0``); use ``is not None``
    # ternaries to fall back only on genuinely missing keys, matching
    # CLAUDE.md's stated convention.
    def _get(key: str, default: Any) -> Any:
        v = text_cfg.get(key)
        return v if v is not None else default

    hidden_size = int(text_cfg["hidden_size"])
    # ``num_attention_heads`` has no safe default: the prior fallback
    # ``hidden_size // 64`` assumed 64-dim heads (correct for some
    # Gemma variants but wrong for the dominant 128-dim convention of
    # Qwen3 / Llama 3 / Mistral, which would silently get 2× too many
    # heads and a mis-sized ``head_dim`` derived from it). Modern
    # config.json files virtually always include this field; raise on
    # the missing case rather than silently producing a draft
    # architecture incompatible with the target.
    raw_num_heads = text_cfg.get("num_attention_heads")
    if raw_num_heads is None:
        raise ValueError(
            "target config.json is missing 'num_attention_heads'. "
            "There is no safe default — the head count drives "
            "head_dim derivation and a wrong value silently produces "
            "an architecturally-mismatched draft. Add the field to "
            "the target config or open an olmlx issue if your model "
            "encodes it differently."
        )
    num_attention_heads = int(raw_num_heads)
    num_kv_heads = int(_get("num_key_value_heads", num_attention_heads))
    head_dim = int(_get("head_dim", hidden_size // num_attention_heads))
    intermediate_size = int(_get("intermediate_size", hidden_size * 4))
    rms_norm_eps = float(_get("rms_norm_eps", 1e-6))
    # Newer config schemas (Qwen3.5+, Qwen3.6) drop the flat ``rope_theta`` in
    # favor of a nested ``rope_parameters`` block. The default 10000.0 is
    # off by 1000× from the long-context bases these targets use, so a
    # silent fallback would produce a draft whose RoPE frequencies are
    # incompatible with the positions the target was trained on. Prefer the
    # flat field when present; otherwise descend.
    # Cascade at the ``rope_theta`` level rather than the dict level: a
    # partial ``rope_parameters`` block in ``text_config`` (e.g.
    # ``{"rope_type": "yarn"}`` with no theta) is truthy and would
    # short-circuit the ``or``, swallowing a top-level block that carries
    # the correct value. Read ``rope_theta`` from each source in priority
    # order instead.
    rope_params_inner = text_cfg.get("rope_parameters")
    rope_params_outer = (
        target_cfg.get("rope_parameters") if text_cfg is not target_cfg else None
    )
    if text_cfg.get("rope_theta") is not None:
        rope_theta = float(text_cfg["rope_theta"])
    elif (
        isinstance(rope_params_inner, dict)
        and rope_params_inner.get("rope_theta") is not None
    ):
        rope_theta = float(rope_params_inner["rope_theta"])
    elif (
        isinstance(rope_params_outer, dict)
        and rope_params_outer.get("rope_theta") is not None
    ):
        rope_theta = float(rope_params_outer["rope_theta"])
    else:
        # ``logger.error`` rather than just ``warning``: the fallback
        # value is off by 1000× for modern long-context targets
        # (Qwen3.5+, Qwen3.6 use 10_000_000), producing a draft whose
        # RoPE frequencies are incompatible with the positions the
        # target was trained on. The near-zero acceptance rate only
        # shows up at inference time, far from the cause, so surface
        # this loudly.
        rope_theta = 10000.0
        logger.error(
            "No 'rope_theta' found at the top level or under "
            "'rope_parameters' in the target config — falling back to "
            "10000.0. Long-context targets (Qwen3.5+, Qwen3.6) typically "
            "use ~10_000_000; verify the target's config.json."
        )
    max_position_embeddings = int(_get("max_position_embeddings", 4096))

    # Stored on disk as the draft-token count directly (the same
    # convention #287's ``_load_dflash_decoder`` consumes verbatim).
    # If a future investigation confirms z-lab uses a different
    # convention, both the writer here and the loader in model_manager
    # would need to flip together.
    return DraftConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        intermediate_size=intermediate_size,
        vocab_size=int(text_cfg["vocab_size"]),
        rms_norm_eps=rms_norm_eps,
        rope_theta=rope_theta,
        max_position_embeddings=max_position_embeddings,
        block_size=block_size,
        num_target_layers=len(target_layer_ids),
        target_layer_ids=list(target_layer_ids),
        mask_token_id=mask_token_id,
        # Fall back to the top-level when ``rope_scaling`` isn't
        # mirrored inside ``text_config``. ``or`` is safe because
        # ``rope_scaling`` is a dict — falsy only when ``None`` or
        # ``{}`` (and ``{}`` carries no useful information either).
        rope_scaling=text_cfg.get("rope_scaling") or target_cfg.get("rope_scaling"),
    )


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------


def _capture_target_outputs(
    target: nn.Module,
    inputs: mx.array,
    cache: list[Any] | None,
    storage: list[Any],
    *,
    capture_logits: bool,
) -> tuple[mx.array, mx.array | None]:
    """Run target on ``inputs`` and return ``(hidden, logits | None)``.

    Assumes ``_patch_model`` has been installed with the same *storage*
    list this function reads from. ``hidden`` has shape
    ``(B, L, num_target_layers * hidden_size)``. ``logits`` is captured
    only when ``capture_logits=True`` (KL distillation needs them); the
    pure-CE path skips it to avoid materializing the vocab-size tensor.
    Both outputs are detached via ``mx.stop_gradient`` — the target is
    frozen during draft training.
    """
    # Reset slots before each forward — the ``any(h is None ...)`` guard
    # below only catches hooks that *never* fired, not hooks that fired
    # on a previous step but were skipped this step. Slice-assign keeps
    # the same list object the installed hooks reference.
    storage[:] = [None] * len(storage)
    out = target(inputs, cache=cache)
    captured = list(storage)
    if any(h is None for h in captured):
        raise RuntimeError(
            "Target forward did not populate all configured target_layer_ids"
        )
    hidden = mx.stop_gradient(mx.concatenate(captured, axis=-1))
    logits: mx.array | None = None
    if capture_logits:
        # mlx-vlm wraps logits in a dataclass; mlx-lm returns the raw array.
        raw = getattr(out, "logits", out)
        logits = mx.stop_gradient(raw)
    return hidden, logits


def _draft_loss(
    draft: DFlashDraftModel,
    block_input: mx.array,
    target_hidden: mx.array,
    targets: mx.array,
    cache: list[Any],
    target_logits_window: mx.array | None = None,
    distill_alpha: float = 0.0,
    distill_temp: float = 1.0,
    pad_token_id: int | None = None,
    position_decay_gamma: float | None = None,
) -> mx.array:
    """Loss on the masked positions: cross-entropy + optional KL distillation.

    ``block_input`` has shape ``(B, block_size + 1)`` — position 0 is
    the visible pending token, positions 1..block_size are MASK.
    ``logits_start=1`` slices the position-0 logit out so draft logits
    has shape ``(B, block_size, vocab)``. ``targets`` has shape
    ``(B, block_size)`` and contains the original (unmasked) tokens.

    When ``target_logits_window`` is provided (also shape ``(B,
    block_size, vocab)``), the loss is
    ``(1 - alpha) * CE + alpha * T^2 * KL(target || draft)`` per the
    Hinton-style distillation recipe. The ``T^2`` factor restores the
    gradient magnitude lost to the temperature softening so distillation
    stays comparable to CE in scale.

    When ``pad_token_id`` is provided, positions where ``targets ==
    pad_token_id`` are zero-weighted in both CE and KL reductions and
    the divisor switches from total-position-count to non-pad-count.
    Without this, batches whose pivot lands in the padding region of a
    right-padded sequence trivially solve to ``CE ≈ 0`` (the
    ``bind()``-tied lm_head predicts the input token, which equals pad,
    which equals the target after MASK==PAD aliasing) — contaminating
    the running average without contributing any gradient. The mask
    makes such batches deliver an honest no-op step instead.

    When ``position_decay_gamma`` is provided (positive float), apply
    the paper's per-position weighting ``w_k = exp(-(k-1)/gamma)`` for
    ``k = 1..block_size`` (so position 0 has weight 1.0, decaying to
    ``exp(-(N-1)/gamma)`` at position N-1). This emphasises early
    positions because acceptance length compounds — a wrong token at
    position 1 wastes the remaining positions 2..N regardless of how
    correct they would have been. With ``None`` (or ``<= 0``) the
    reduction stays a uniform mean (legacy behaviour). See gh#317
    (Gap 2).
    """
    draft_logits = draft(block_input, target_hidden, cache, logits_start=1)
    log_probs = nn.log_softmax(draft_logits, axis=-1)
    nll = -mx.take_along_axis(log_probs, targets[..., None], axis=-1).squeeze(-1)

    # ``pos_weights`` has shape ``(block_size,)`` and broadcasts against
    # the ``(B, block_size)`` per-position NLL / KL arrays. When
    # ``position_decay_gamma`` is unset, fall back to ones so the
    # weighted-mean reduction collapses to the legacy uniform mean.
    if position_decay_gamma is not None and position_decay_gamma > 0:
        block_size = nll.shape[-1]
        pos = mx.arange(block_size, dtype=nll.dtype)
        pos_weights = mx.exp(-pos / float(position_decay_gamma))
    else:
        pos_weights = None

    # Initialise upfront so the type checker can see ``denom`` is in
    # scope when the KL branch reaches for it.
    valid: mx.array | None = None
    denom: mx.array | None = None
    if pad_token_id is not None:
        valid = (targets != pad_token_id).astype(nll.dtype)
        # Combine pad mask with per-position weights when both are
        # active. The pad path's invariant — sum(weights) >= 0 across
        # all-pad batches collapses to exactly 0.0 — is preserved
        # because positive ``pos_weights`` only scale a zero mask.
        combined = valid if pos_weights is None else valid * pos_weights
        # ``valid.sum().clip(1.0, ...)`` keeps the divisor from hitting
        # zero when the entire window is pad. Combined with the
        # zero-weighted nll the result is an exact 0.0 — the optimizer
        # update then has zero gradient (flowing back through 0/1).
        denom = mx.maximum(combined.sum(), mx.array(1.0, dtype=nll.dtype))
        ce = (nll * combined).sum() / denom
    elif pos_weights is not None:
        # Pure position-weighted mean: sum(w * nll) / sum(w). ``denom``
        # stays ``None`` here — the KL branch below recomputes its own
        # denominator using the same weights.
        ce = (nll * pos_weights).sum() / (pos_weights.sum() * nll.shape[0])
    else:
        ce = mx.mean(nll)

    if target_logits_window is None or distill_alpha <= 0.0:
        return ce

    # KL(p || q) where p is target softmax(./T), q is draft softmax(./T).
    # Computed via target_log_probs - draft_log_probs weighted by p; the
    # T^2 multiplier is the standard distillation correction.
    t = float(distill_temp)
    target_log_probs = nn.log_softmax(target_logits_window / t, axis=-1)
    target_probs = mx.exp(target_log_probs)
    draft_log_probs_t = nn.log_softmax(draft_logits / t, axis=-1)
    kl = mx.sum(target_probs * (target_log_probs - draft_log_probs_t), axis=-1)
    if valid is not None:
        # ``valid`` and ``denom`` are co-assigned in the
        # ``pad_token_id is not None`` branch above, so checking
        # ``valid`` alone implies ``denom`` is non-None. Reuse the
        # combined-weights ``denom`` via dtype cast.
        denom_kl = cast(mx.array, denom)
        combined_kl = valid if pos_weights is None else valid * pos_weights
        kl_loss = (kl * combined_kl).sum() / denom_kl.astype(kl.dtype) * (t * t)
    elif pos_weights is not None:
        kl_loss = (kl * pos_weights).sum() / (pos_weights.sum() * kl.shape[0]) * (t * t)
    else:
        kl_loss = mx.mean(kl) * (t * t)

    return (1.0 - distill_alpha) * ce + distill_alpha * kl_loss


def _select_pivot(
    input_ids: mx.array,
    pad_token_id: int,
    block_size: int,
) -> int | None:
    """Pick a pivot inside the right-padded prefix shared by every batch row.

    Returns ``None`` when no row has at least ``2 * block_size + 1`` real
    tokens in its prefix — the caller should skip the batch in that case
    rather than forcing a degenerate pivot. The pivot range is
    ``[block_size, min_real_len - block_size - 1]`` (inclusive) so that
    every row has both a real ``pending`` token at position ``p`` and
    real targets at positions ``p+1..p+block_size``.

    Scans from the right to find the trailing-pad boundary (reversed
    argmax for the first non-pad position) rather than counting
    non-pad tokens globally. A non-pad count is only equal to the
    prefix length when ``pad_token_id`` never appears as real content
    — and with ``mask_token_id == pad_token_id == eos_token_id`` (the
    Qwen3.x default) multi-turn sequences carry EOS markers mid-stream
    that are content, not padding. A naive non-pad count miscounts
    those as padding and shrinks the pivot range, silently skipping
    batches with valid windows. A naive left-to-right
    ``argmax(ids == pad)`` returns the first mid-stream EOS instead
    of the trailing-pad boundary, which is even more wrong for that
    case.

    Edge case: when a row's *final* real token equals ``pad_token_id``
    (e.g. an end-of-conversation EOS that happens to alias the loader's
    pad), the trailing-pad scan absorbs it into the pad count, reporting
    ``real_lens`` as one shorter than the true content boundary. The
    direction of the error is safe (more conservative pivot range, no
    invalid windows) but loses one valid pivot slot per such row. We
    accept that conservatism rather than carry per-row metadata about
    "true" sequence length through the loader.

    Costs one CPU sync per call (``min().item()``) — unavoidable because
    Python's ``random.randint`` requires a host int. We accept that
    rather than calling ``mx.random.randint(...).item()`` in the hot
    loop, which would also sync but additionally drain the lazy graph.
    This is an intentional trade-off vs. the original design that
    avoided MLX syncs entirely; the sync cost is amortised by the
    dominant target-forward-pass time and the correctness gain from
    precise trailing-pad detection.
    """
    seq_len = input_ids.shape[1]
    # Find the right-padded prefix boundary by counting *trailing* pads,
    # not the position of the first pad anywhere. The latter conflates
    # mid-stream EOS markers (real content) with right-pad: with
    # ``mask_token_id == pad_token_id == eos_token_id`` (the Qwen3.x
    # default) a multi-turn row carries EOS at every turn boundary,
    # and using ``argmax`` on the pad mask would return the first
    # mid-stream EOS — collapsing the apparent prefix to a tiny
    # window. Reversing along the sequence axis and locating the first
    # non-pad gives the trailing-pad count directly.
    reversed_ids = input_ids[:, ::-1]
    not_pad_rev = reversed_ids != pad_token_id
    has_real = not_pad_rev.any(axis=1)
    first_real_rev = not_pad_rev.argmax(axis=1)
    # ``argmax`` on booleans returns ``uint32``; cast to ``int32`` for
    # explicit type agreement with the ``seq_len`` fallback below.
    # MLX's type promotion handles the mixed ``mx.where`` today, but the
    # promotion rules are not part of the stable API contract.
    first_real_rev = first_real_rev.astype(mx.int32)
    # All-pad rows: ``any(axis=1) == False``, so ``argmax`` returns 0
    # (which is meaningless); fall back to ``seq_len`` so trailing pads
    # = whole length and ``real_lens`` = 0.
    trailing_pads = mx.where(
        has_real, first_real_rev, mx.array(seq_len, dtype=mx.int32)
    )
    real_lens = mx.array(seq_len, dtype=mx.int32) - trailing_pads
    min_real = int(real_lens.min().item())
    if min_real < 2 * block_size + 1:
        return None
    return random.randint(block_size, min_real - block_size - 1)


def _select_pivots(
    input_ids: mx.array,
    pad_token_id: int,
    block_size: int,
    num_windows: int,
) -> list[int] | None:
    """Pick up to ``num_windows`` non-overlapping pivots in the shared
    unpadded prefix via slot-and-jitter placement.

    Returns ``None`` only when no window fits at all (matching
    ``_select_pivot``'s ``None`` semantics — the caller treats this as
    "skip the batch"). Otherwise returns a list of length
    ``1..num_windows``; a length below ``num_windows`` means the valid
    range was too small for that many non-overlapping slots, and the
    caller proceeds with the windows it received (no skip).

    Slot-and-jitter: divide the valid pivot range into ``num_windows``
    equal-width slots and sample one pivot per slot from the slot's left
    portion ``[slot_lo, slot_hi - block_size]``, leaving a ``block_size``
    buffer at the right of each slot. That buffer is what guarantees
    non-overlap between adjacent windows: each pivot's
    ``[p, p+block_size]`` span fits inside its slot, so adjacent pivots
    are at least ``block_size + 1`` apart. Slots narrower than
    ``block_size + 1`` would make the sampling range empty, so we cap
    ``num_windows`` to ``max_fit = range_size // (block_size + 1)``.

    ``num_windows == 1`` delegates to ``_select_pivot`` so the K=1 path
    is bit-exact with the legacy single-window code under a fixed RNG
    seed (same single ``random.randint`` call with the same bounds).
    This also preserves the monkey-patch behaviour the existing
    ``test_target_hidden_slice_excludes_pending_position`` test relies
    on.
    """
    if num_windows <= 0:
        raise ValueError(f"num_windows must be >= 1, got {num_windows}")
    if num_windows == 1:
        p = _select_pivot(input_ids, pad_token_id, block_size)
        return None if p is None else [p]

    # Replicate the trailing-pad detection from _select_pivot. We can't
    # call _select_pivot directly for K > 1 because we need access to
    # ``min_real`` to lay out the slots; factoring it out would
    # complicate the K=1 RNG-equivalence guarantee, so we duplicate the
    # cheap reversal + argmax instead.
    seq_len = input_ids.shape[1]
    reversed_ids = input_ids[:, ::-1]
    not_pad_rev = reversed_ids != pad_token_id
    has_real = not_pad_rev.any(axis=1)
    first_real_rev = not_pad_rev.argmax(axis=1).astype(mx.int32)
    trailing_pads = mx.where(
        has_real, first_real_rev, mx.array(seq_len, dtype=mx.int32)
    )
    real_lens = mx.array(seq_len, dtype=mx.int32) - trailing_pads
    min_real = int(real_lens.min().item())
    if min_real < 2 * block_size + 1:
        return None

    lo = block_size
    hi = min_real - block_size - 1  # inclusive
    range_size = hi - lo + 1

    # Cap num_windows by what actually fits non-overlapping. Each
    # window's [p, p+block_size] span has length block_size+1, so two
    # adjacent pivots need to be at least that far apart.
    max_fit = max(1, range_size // (block_size + 1))
    k = min(num_windows, max_fit)

    if k == 1:
        # Range only big enough for one window even though the operator
        # asked for more. Delegate again so the K=1 RNG path matches the
        # single-pivot case exactly.
        return [random.randint(lo, hi)]

    # Equal-width slots over the inclusive range [lo, hi]. Using float
    # division then int-truncating the per-slot boundaries keeps the
    # slot edges close to ``range_size / k`` apart for small K — the
    # alternative ``range_size // k`` integer step would systematically
    # bias the last slot wider.
    slot_width = range_size / k
    pivots: list[int] = []
    for i in range(k):
        slot_lo = lo + int(i * slot_width)
        slot_hi_exclusive = lo + int((i + 1) * slot_width)
        slot_hi = slot_hi_exclusive - 1  # inclusive
        # Sample from ``[slot_lo, slot_hi - block_size]`` so the
        # pivot's ``[p, p+block_size]`` span fits inside its slot.
        # That guarantees adjacent pivots are at least
        # ``block_size + 1`` apart: ``p_i + block_size <= slot_hi_i <
        # slot_lo_{i+1} <= p_{i+1}``. ``max_fit`` guarantees
        # ``slot_width >= block_size + 1``, so ``slot_hi - block_size
        # >= slot_lo`` and the sampling range is non-empty (collapses
        # to a single point at the lower bound when the slot is
        # exactly the minimum width).
        sample_hi = slot_hi - block_size
        pivots.append(random.randint(slot_lo, sample_hi))
    return pivots


def _cosine_lr(step: int, total: int, peak: float, warmup: int) -> float:
    """Linear warmup followed by cosine decay to 10% of peak."""
    if step < warmup:
        return peak * (step + 1) / max(warmup, 1)
    if step >= total:
        return peak * 0.1
    progress = (step - warmup) / max(total - warmup, 1)
    return peak * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def prepare_dflash_draft(
    model_path: str | Path,
    *,
    dataset: str | None = None,
    dataset_split: str | None = None,
    steps: int = DEFAULT_STEPS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seq_len: int = DEFAULT_SEQ_LEN,
    block_size: int = DEFAULT_BLOCK_SIZE,
    num_hidden_layers: int = DEFAULT_NUM_HIDDEN_LAYERS,
    target_layer_ids: list[int] | None = None,
    num_target_layers: int | None = None,
    lr: float = DEFAULT_LR,
    mask_token_id: int | None = None,
    output_dir: str | Path | None = None,
    progress_callback: Callable[[str, float], None] | None = None,
    log_every: int = 50,
    distill: bool = False,
    distill_alpha: float = 0.5,
    distill_temp: float = 2.0,
    position_decay_gamma: float | None = None,
    train_windows_per_step: int = 1,
    use_precomputed: str | Path | None = None,
    _target_loader: Callable[[str], tuple[Any, Any]] | None = None,
    _batch_iterator: Any = None,
) -> Path:
    """Train a DFlash draft model and write it to disk.

    ``distill``: enable Hinton-style KL distillation against the target
    logits at the masked positions. Loss becomes
    ``(1 - alpha) * CE + alpha * T^2 * KL``. Requires running the
    target online — incompatible with ``use_precomputed`` (precomputed
    shards store hiddens but not vocab-size logits).

    ``position_decay_gamma``: per-position loss weighting decay
    constant ``γ`` in ``w_k = exp(-(k-1)/γ)`` (k=1..block_size). When
    ``None`` (the default), the weighting is disabled and the
    reduction is a uniform mean over positions — matching legacy
    behaviour bit-for-bit. The issue's suggested starting value to
    sweep is ``block_size / 2``; the paper does not publicly pin a
    single value. ``0`` or a negative value also disables the
    weighting. See gh#317 (Gap 2).

    ``train_windows_per_step``: number of non-overlapping masked
    windows to train on per batch (per optimizer step). Default ``1``
    reproduces the legacy single-window behaviour bit-for-bit. ``K > 1``
    amortises the dominant per-step cost (the target forward) across
    multiple draft-loss windows: the target runs once, then K windows
    are sliced from its hidden states (and, when ``distill=True``, its
    logits). When the batch's shared unpadded prefix is too short to
    fit K non-overlapping windows, fewer are used — K is a target, not
    a guarantee. See gh#382.

    ``use_precomputed``: read precomputed (input_ids, hidden) shards
    from this directory instead of running the target each step. Skips
    target instantiation entirely.

    ``_target_loader`` and ``_batch_iterator`` are injection hooks for
    tests so the trainer can run without downloading a multi-GB target
    and without hitting the network. In normal use the trainer
    defaults to ``mlx_lm.load`` and ``stream_training_batches``.
    """
    if distill and use_precomputed is not None:
        raise ValueError(
            "--distill requires running the target online for vocab-size "
            "logits and is incompatible with --use-precomputed (which "
            "stores hidden states only). Re-run with one or the other."
        )
    if not 0.0 <= distill_alpha <= 1.0:
        raise ValueError(f"distill_alpha must be in [0, 1], got {distill_alpha}")
    if distill_temp <= 0:
        raise ValueError(f"distill_temp must be > 0, got {distill_temp}")
    # ``position_decay_gamma`` defaults to ``None`` (disabled) — the
    # uniform-mean reduction matches legacy behaviour bit-for-bit. The
    # issue (gh#317 Gap 2) flags this as a hyperparameter to sweep,
    # not a definitively-correct default; operators opt in via the
    # ``--position-decay-gamma`` CLI flag (suggested starting point:
    # ``block_size / 2``). ``0`` or a negative value explicitly
    # disables and is normalised to ``None`` so the loss branches stay
    # simple.
    if position_decay_gamma is not None and position_decay_gamma <= 0:
        position_decay_gamma = None
    if train_windows_per_step < 1:
        # An empty windows list would divide by zero in the mean-over-K
        # reduction; negative values are nonsensical.
        raise ValueError(
            f"train_windows_per_step must be >= 1, got {train_windows_per_step}"
        )
    if block_size < 1:
        # ``block_size == 0`` builds zero-length mask/target tensors,
        # ``_draft_loss`` then returns 0 with no gradient, and the
        # optimizer sees zeros every step — silently producing a
        # worthless checkpoint after the full ``--steps`` run.
        raise ValueError(f"block_size must be >= 1, got {block_size}")
    if steps < 1:
        # ``steps == 0`` means the training loop's ``if step >= steps:
        # break`` fires immediately on the first iteration; the function
        # then falls through to the checkpoint-save block and writes a
        # random-initialized draft as if it had been trained. Reject.
        raise ValueError(f"steps must be >= 1, got {steps}")
    model_path = Path(model_path)
    target_cfg = _read_target_config(model_path)

    if _target_loader is None:
        from mlx_lm import load as _mlx_lm_load

        # ``mlx_lm.load`` returns a 2-tuple in current versions; older
        # variants returned 3-tuples. Slice to the first two so either
        # works.
        loaded = _mlx_lm_load(str(model_path))
        target, tokenizer = loaded[0], loaded[1]
    else:
        target, tokenizer = _target_loader(str(model_path))

    # Freeze target params — frozen tensors still flow through forward
    # but ``nn.value_and_grad(draft, ...)`` only differentiates draft
    # parameters, so the target is implicitly frozen by virtue of not
    # being inside the closure. We additionally call ``.freeze()`` to
    # prevent dropout/training-mode side effects.
    target.eval()  # type: ignore[attr-defined]
    if hasattr(target, "freeze"):
        target.freeze()

    # Reuse ``_get_layers`` so the layer-count probe stays in sync with
    # ``_patch_model`` — covers VLM targets (``language_model.layers``).
    try:
        target_num_layers = len(_get_layers(target))
    except AttributeError as exc:
        raise ValueError(
            f"Cannot determine layer count for target model "
            f"{type(target).__name__}: {exc}"
        ) from exc

    layer_ids = _resolve_target_layer_ids(
        target_layer_ids, num_target_layers, target_num_layers
    )
    logger.info("DFlash target_layer_ids = %s", layer_ids)

    # Two related but semantically distinct pad-token notions:
    #
    # ``pad_for_pivot`` mirrors the data loader's right-pad token
    # (see ``training_data.py``). The loader pads short sequences with
    # ``tokenizer.pad_token_id`` and falls back to ``eos_token_id`` if
    # that's missing, so the pivot-selection helper needs to recognise
    # both — only with this match does ``_select_pivot`` correctly find
    # the trailing-pad boundary on right-padded data.
    #
    # ``pad_for_loss`` is the value flowed into ``_draft_loss`` to
    # zero-weight padding *targets* in CE/KL. Here we deliberately
    # *do not* fall back to ``eos_token_id``: in multi-turn
    # instruction-tuning data each turn ends with EOS as a real
    # content token, and zero-weighting those would teach the draft to
    # never predict EOS at turn boundaries. Since ``_select_pivot``
    # already restricts the pivot to the unpadded prefix, pad targets
    # can no longer reach ``_draft_loss`` via the trained path; the
    # mask is defensive belt-and-suspenders for genuine pad tokens
    # only.
    #
    # **Qwen3.x aliasing**: ``tokenizer.pad_token_id`` is sometimes set
    # to the *same value* as ``eos_token_id`` (e.g. Qwen3.x's
    # ``<|endoftext|>`` for both). Using that shared id as
    # ``pad_for_loss`` masks every mid-stream EOS turn separator —
    # exactly the failure mode the comment above warns about. Disable
    # the loss-mask (set to ``None``) when the two ids collide so the
    # loss falls through to the unmasked mean reduction. The
    # ``pad_for_pivot`` value still uses the loader's actual pad token
    # so trailing-pad detection works.
    _tok_pad = getattr(tokenizer, "pad_token_id", None)
    _tok_eos = getattr(tokenizer, "eos_token_id", None)
    pad_for_pivot: int | None
    if _tok_pad is not None:
        pad_for_pivot = int(_tok_pad)
    elif _tok_eos is not None:
        pad_for_pivot = int(_tok_eos)
    else:
        pad_for_pivot = None
    pad_for_loss: int | None = (
        int(_tok_pad)
        if (
            _tok_pad is not None
            and (_tok_eos is None or int(_tok_pad) != int(_tok_eos))
        )
        else None
    )
    # When ``pad_for_loss is None`` (Qwen3.x aliasing, or no pad token
    # at all), ``_draft_loss`` falls through to the unmasked mean
    # reduction.  ``_select_pivot`` is the sole guard: it restricts
    # every pivot to the unpadded prefix, so pad targets cannot reach
    # the loss in the first place.  See ``_select_pivot``'s docstring
    # for the one conservative edge case (trailing real-EOS absorbed
    # into the pad count).

    if mask_token_id is None:
        # ``or`` would short-circuit on token ID 0 (a valid pad id for
        # Llama 2 / Mistral / Qwen 1.x), silently picking EOS as the
        # mask id and misaligning training vs. inference.
        _pad = getattr(tokenizer, "pad_token_id", None)
        _eos = getattr(tokenizer, "eos_token_id", None)
        if _pad is not None:
            mask_token_id = _pad
        elif _eos is not None:
            mask_token_id = _eos
        else:
            # Token 0 is not a safe fallback — for many tokenizers it's
            # ``<bos>`` or ``<unk>``, which the model has strong priors
            # about; using it as the MASK token at training time
            # confuses the loss and degrades inference quality. Refuse
            # the run rather than silently saving a poorly-trained
            # checkpoint.
            raise ValueError(
                "Could not derive a mask token id: tokenizer has no "
                "pad_token_id and no eos_token_id, and token 0 is not "
                "a safe default (often <bos>/<unk>). Pass an explicit "
                "mask_token_id (CLI: --mask-token-id N) for this target."
            )

    draft_config = _build_draft_config(
        target_cfg,
        target_layer_ids=layer_ids,
        num_hidden_layers=num_hidden_layers,
        block_size=block_size,
        mask_token_id=int(mask_token_id),
    )

    draft = DFlashDraftModel(draft_config)
    # Defer ``draft.bind(target)`` until inside the try/finally below so
    # any exception in batch-iterator validation (e.g. missing precompute
    # shards) cannot leave the draft holding live references to the
    # target's ``embed_tokens`` / ``lm_head`` weights.
    mx.eval(draft.parameters())

    # Optimizer + LR schedule.
    optimizer = optim.AdamW(learning_rate=lr)
    warmup = max(int(steps * DEFAULT_WARMUP_FRAC), 1)

    # Multi-window loss closure. ``windows`` is a list of
    # (block_input, target_hidden, targets, target_logits_window)
    # tuples — one per pivot. Each window gets a fresh draft cache;
    # the closure sums per-window losses and divides by K. At K=1 the
    # list has one element and ``sum_of_one / 1`` reduces to the
    # legacy per-window loss exactly.
    def loss_fn_multi(
        model: DFlashDraftModel,
        windows: list[tuple[mx.array, mx.array, mx.array, mx.array | None]],
    ) -> mx.array:
        total = mx.array(0.0)
        for block_input, target_hidden, targets, target_logits_window in windows:
            cache = model.make_cache()
            total = total + _draft_loss(
                model,
                block_input,
                target_hidden,
                targets,
                cache,
                target_logits_window=target_logits_window,
                distill_alpha=distill_alpha if distill else 0.0,
                distill_temp=distill_temp,
                pad_token_id=pad_for_loss,
                position_decay_gamma=position_decay_gamma,
            )
        return total / len(windows)

    loss_and_grad_multi = nn.value_and_grad(draft, loss_fn_multi)

    def _step(
        windows: list[tuple[mx.array, mx.array, mx.array, mx.array | None]],
    ) -> mx.array:
        loss, grads = loss_and_grad_multi(draft, windows)
        optimizer.update(draft, grads)
        return loss

    # Streaming data loop — each iteration yields either a raw
    # ``input_ids`` array (online target path) or a
    # ``(input_ids, hidden)`` tuple (precomputed-shards path). The
    # ``_batch_iterator`` test hook bypasses both real data sources.
    if _batch_iterator is not None:
        batches = _batch_iterator
    elif use_precomputed is not None:
        from olmlx.engine.dflash.precompute import (
            iter_precomputed_shards,
            read_precomputed_index,
        )

        # Validate shard shape against the current training config so
        # mismatches surface as a clear error here rather than as an
        # obscure shape crash inside ``_draft_loss``.
        meta = read_precomputed_index(use_precomputed)
        expected_concat_hidden = len(layer_ids) * int(
            _text_config(target_cfg)["hidden_size"]
        )
        mismatches = []
        if meta["batch_size"] != batch_size:
            mismatches.append(
                f"batch_size: shard={meta['batch_size']} requested={batch_size}"
            )
        if meta["seq_len"] != seq_len:
            mismatches.append(f"seq_len: shard={meta['seq_len']} requested={seq_len}")
        if meta["concat_hidden_size"] != expected_concat_hidden:
            mismatches.append(
                f"concat_hidden_size: shard={meta['concat_hidden_size']} expected="
                f"{expected_concat_hidden} (num_target_layers * target hidden_size)"
            )
        # Two precompute runs with the same *number* of target layers
        # but different *indices* both produce identical
        # ``concat_hidden_size`` values, so the shape check above can't
        # tell them apart. Compare ``target_layer_ids`` exactly to
        # surface the mismatch — without this the draft would silently
        # consume hiddens from the wrong layers and only show degraded
        # acceptance rates at inference time.
        shard_layer_ids = list(meta.get("target_layer_ids", []))
        if shard_layer_ids != list(layer_ids):
            mismatches.append(
                f"target_layer_ids: shard={shard_layer_ids} requested={list(layer_ids)}"
            )
        if mismatches:
            raise ValueError(
                "Precompute shard shape does not match current training config: "
                + "; ".join(mismatches)
            )
        batches = iter_precomputed_shards(use_precomputed, max_examples=steps)
    else:
        batches = stream_training_batches(
            tokenizer,
            dataset=dataset or "HuggingFaceH4/ultrachat_200k",
            split=dataset_split or "train_sft",
            batch_size=batch_size,
            seq_len=seq_len,
            max_examples=steps,
        )

    # Caller-owned hidden-state storage passed into both the patch and
    # ``_capture_target_outputs`` — keeps captured ``mx.array``s out of
    # ``target.parameters()`` (mlx tracks ``list``-typed attributes as
    # parameters once they hold arrays). Allocated *after* batch-iterator
    # setup so an exception there (e.g. ``read_precomputed_index``
    # raising on a missing shard directory) cannot leave the target
    # permanently patched.
    hidden_capture: list[Any] = [None] * len(layer_ids)

    losses: list[float] = []
    try:
        # ``_patch_model`` is the *first* statement inside the try so a
        # partial-hook-install failure (e.g. on an unexpected layer
        # structure) still triggers ``_unpatch_model`` in the finally
        # block. Bind the draft inside the try too so ``unbind()`` runs
        # even if the bind itself or the first training step raises.
        _patch_model(target, layer_ids, hidden_capture)
        draft.bind(target)
        # ``step`` from ``enumerate(batches)`` advances on *every* iteration,
        # including the ones we ``continue`` past via ``_select_pivot is None``.
        # Track real (non-skipped) gradient steps separately so the LR schedule,
        # progress-bar fraction, log-every cadence, and termination condition
        # reflect actual training progress rather than batch iteration count.
        # Without this counter, skipped pad-only batches silently retire slots
        # of the operator's ``steps`` budget and produce an under-trained
        # checkpoint.
        real_step = 0
        # Guard against an infinite-iterator + all-padding scenario.
        # HuggingFace streaming datasets are typically infinite, and a
        # misconfigured ``block_size`` (or a dataset of uniformly very
        # short sequences) can produce a stream where every batch is
        # rejected by ``_select_pivot``. Without this guard the loop
        # would spin forever — never advancing ``real_step``, never
        # reaching the under-training warning after the loop. Cap the
        # consecutive-skip count at a multiple of the requested
        # ``steps`` so a few rough patches don't trigger a false
        # positive on a real run.
        consecutive_skips = 0
        # Cap consecutive skips with a hard ceiling so degenerate
        # datasets fail fast instead of chewing through thousands of
        # CPU syncs. Each skipped batch costs one sync via
        # ``_select_pivot``. ``min(…, 500)`` bounds the worst-case
        # stall while preserving the ``2*n + 50`` budget for short
        # runs where it's proportional and inexpensive. The ceiling
        # of 500 is chosen as ~10× the largest reasonable batch-window
        # of consecutive short sequences in real datasets; a run that
        # hits it is almost certainly misconfigured (block_size too
        # large for the dataset, or wrong pad token).
        max_consecutive_skips = min(steps * 2 + 50, 500)
        for batch in batches:
            if real_step >= steps:
                break
            if consecutive_skips >= max_consecutive_skips:
                # ``logger.error + break`` rather than ``raise RuntimeError``:
                # the latter skips the checkpoint save below (which runs
                # after the try/finally block), silently discarding all
                # progress from real gradient steps already completed.
                # Breaking lets the post-loop warnings and checkpoint save
                # still execute, preserving partial progress.
                logger.error(
                    "DFlash training aborted: %d consecutive batches "
                    "skipped without a real gradient update before "
                    "reaching %d/%d steps. Every batch had at least one "
                    "row shorter than 2*block_size + 1 = %d real tokens. "
                    "Likely causes: a dataset of uniformly short sequences, "
                    "a misconfigured --block-size, or a tokenizer whose "
                    "pad token coincides with the loader's actual pad. "
                    "Inspect the dataset or lower --block-size.",
                    consecutive_skips,
                    real_step,
                    steps,
                    2 * block_size + 1,
                )
                break

            optimizer.learning_rate = _cosine_lr(real_step, steps, lr, warmup)

            # Resolve ``input_ids`` and (if precomputed) ``target_hidden_full``
            # from the batch *without* running the target forward yet — the
            # target forward is the dominant cost on a frozen 35B target, and
            # ``_select_pivot`` below may reject the batch entirely. Running
            # the target before the pivot check would burn a full forward pass
            # for every pad-only batch.
            if isinstance(batch, tuple):
                # Tuple batches come from precomputed shards or a
                # ``_batch_iterator`` test hook; neither carries logits,
                # so distillation can't run. The
                # ``use_precomputed``-aware guard at the top of
                # ``prepare_dflash_draft`` rejects the production
                # combination, but a test passing
                # ``distill=True, _batch_iterator=tuples`` would
                # otherwise silently degrade to CE-only without any
                # KL signal — surface that here.
                if distill:
                    raise RuntimeError(
                        "distill=True requires online target logits, but the "
                        "current batch is a (input_ids, hidden) tuple (from "
                        "precomputed shards or a tuple-yielding "
                        "_batch_iterator). Pass raw input_ids batches or "
                        "drop --distill."
                    )
                input_ids, precomputed_hidden = batch
            else:
                input_ids = batch
                precomputed_hidden = None

            # Pick up to ``train_windows_per_step`` non-overlapping
            # pivots. Two regimes mirror the legacy single-pivot path:
            #
            # - When the loader's pad token is known
            #   (``pad_for_pivot is not None``), restrict every pivot
            #   to the shared unpadded prefix via ``_select_pivots``.
            #   At K=1 this delegates to ``_select_pivot`` so the RNG
            #   draw and the monkey-patch hook surface stay
            #   bit-exactly equivalent to the legacy path.
            # - When ``pad_for_pivot is None`` (test fixtures with no
            #   padding), use an inline slot-and-jitter sampler so the
            #   no-padding test path does not start syncing through
            #   the MLX-side trailing-pad detection.
            seq = input_ids.shape[1]
            if pad_for_pivot is None:
                lo = block_size
                hi_inclusive = seq - block_size - 1
                if hi_inclusive < lo:
                    raise ValueError(
                        f"seq_len={seq} too small for block_size={block_size}; "
                        f"need at least 2*block_size + 1 tokens per sequence"
                    )
                range_size = hi_inclusive - lo + 1
                max_fit = max(1, range_size // (block_size + 1))
                k = min(train_windows_per_step, max_fit)
                if k == 1:
                    # K=1 (default) uses the same random.randint call
                    # the legacy single-window path used — same range,
                    # same RNG draw under a fixed seed.
                    pivots = [random.randint(lo, hi_inclusive)]
                else:
                    # Slot-and-jitter with a block_size buffer at the
                    # right of each slot so adjacent pivots are at
                    # least block_size + 1 apart (see _select_pivots
                    # docstring for the proof). The max_fit cap above
                    # ensures slot_width >= block_size + 1, so the
                    # sampling range [slot_lo, slot_hi - block_size]
                    # is non-empty.
                    slot_width = range_size / k
                    pivots = []
                    for i in range(k):
                        slot_lo = lo + int(i * slot_width)
                        slot_hi = lo + int((i + 1) * slot_width) - 1
                        pivots.append(random.randint(slot_lo, slot_hi - block_size))
            else:
                pivot_list = _select_pivots(
                    input_ids,
                    pad_for_pivot,
                    block_size,
                    train_windows_per_step,
                )
                if pivot_list is None:
                    # Every row was shorter than 2*block_size + 1 real
                    # tokens; no window fits. Skip the batch.
                    logger.debug(
                        "skipping all-padding batch before real step %d "
                        "(no row has %d+ real tokens)",
                        real_step + 1,
                        2 * block_size + 1,
                    )
                    consecutive_skips += 1
                    continue
                pivots = pivot_list
                if len(pivots) < train_windows_per_step:
                    # Shared unpadded prefix was too short for the
                    # full K. Train on what we got — this is NOT a
                    # skip; a real gradient update still happens.
                    logger.debug(
                        "step %d: shared prefix too short for %d windows; using %d",
                        real_step + 1,
                        train_windows_per_step,
                        len(pivots),
                    )
            consecutive_skips = 0

            # Pivots accepted: run the target forward (online) or
            # consume the precomputed hidden state. One target forward
            # per batch — its output is shared by all K windows below.
            target_logits_full: mx.array | None = None
            if precomputed_hidden is not None:
                target_hidden_full = precomputed_hidden
            else:
                target_hidden_full, target_logits_full = _capture_target_outputs(
                    target,
                    input_ids,
                    cache=None,
                    storage=hidden_capture,
                    capture_logits=distill,
                )

            # Build the per-window list. Each window slices its own
            # block_input / targets / target_hidden / (optionally)
            # target_logits_window from the shared target forward. The
            # MASK block is identical across windows (same shape, same
            # mask_token_id, same dtype), so build it once outside the
            # loop.
            mask_block = mx.full(
                (input_ids.shape[0], block_size),
                int(draft_config.mask_token_id),
                dtype=input_ids.dtype,
            )
            windows: list[tuple[mx.array, mx.array, mx.array, mx.array | None]] = []
            for p in pivots:
                pending = input_ids[:, p : p + 1]  # (B, 1)
                block_input = mx.concatenate([pending, mask_block], axis=1)
                targets = input_ids[:, p + 1 : p + 1 + block_size]
                # Slice ctx to positions 0..p-1 so the draft sees the
                # same hidden-state distribution at training and
                # inference time (gh#317 Gap 1).
                target_hidden = target_hidden_full[:, :p, :]

                target_logits_window: mx.array | None = None
                if distill and target_logits_full is not None:
                    target_logits_window = target_logits_full[:, p : p + block_size, :]

                windows.append(
                    (block_input, target_hidden, targets, target_logits_window)
                )

            loss = _step(windows)
            mx.eval(loss, draft.parameters(), optimizer.state)
            losses.append(float(loss.item()))
            real_step += 1

            if real_step % log_every == 0 or real_step == 1:
                avg = sum(losses[-log_every:]) / max(min(log_every, len(losses)), 1)
                logger.info(
                    "step %d/%d  loss=%.4f  avg(%d)=%.4f  lr=%.2e",
                    real_step,
                    steps,
                    losses[-1],
                    log_every,
                    avg,
                    optimizer.learning_rate,
                )
            if progress_callback:
                progress_callback(
                    f"Training step {real_step}/{steps} loss={losses[-1]:.4f}",
                    real_step / steps,
                )
    finally:
        _unpatch_model(target)
        draft.unbind()

    # If the batch stream ran out before the operator's ``steps`` budget
    # of real gradient updates was hit, the saved checkpoint is silently
    # under-trained (or, in the all-pad-skip degenerate case, untrained
    # entirely). Surface the discrepancy on the way out so the operator
    # knows to investigate the dataset / block_size combination rather
    # than discovering the problem only at inference time when
    # acceptance is poor.
    if real_step == 0:
        logger.warning(
            "No real gradient steps completed for %s — every batch was "
            "skipped (each row had fewer than %d real tokens). The saved "
            "checkpoint is essentially the random init. Check the dataset "
            "and --block-size (currently %d).",
            model_path,
            2 * block_size + 1,
            block_size,
        )
    elif real_step < steps:
        logger.warning(
            "Only %d of %d requested steps completed for %s; the batch "
            "stream was exhausted before the budget was hit. Saved "
            "checkpoint is under-trained relative to --steps.",
            real_step,
            steps,
            model_path,
        )

    # Save: <output>/{config.json, model-00001-of-00001.safetensors}.
    if output_dir is None:
        output_dir = model_path / "dflash"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_dict = _draft_config_to_disk(draft_config)
    (output_dir / "config.json").write_text(json.dumps(config_dict, indent=2))

    weights = dict(mx_utils.tree_flatten(draft.parameters()))
    # Drop the bound embed/lm_head references — they live on the target
    # and are re-bound on load. ``DFlashDraftModel`` stores them as
    # plain attributes, but if a future change made them ``Module``
    # children they would leak into the saved weights here.
    weights = {
        k: v
        for k, v in weights.items()
        if not k.startswith("embed_tokens.") and not k.startswith("lm_head.")
    }
    mx.save_safetensors(str(output_dir / "model-00001-of-00001.safetensors"), weights)

    logger.info(
        "Saved DFlash draft to %s (%d weight tensors)", output_dir, len(weights)
    )
    return output_dir


def _draft_config_to_disk(cfg: DraftConfig) -> dict[str, Any]:
    """Serialize ``DraftConfig`` to the upstream-compatible JSON schema.

    Mirrors what ``_load_dflash_decoder`` parses on the way in:
    top-level scalars + nested ``dflash_config`` block carrying
    ``target_layer_ids`` and ``mask_token_id``.
    """
    out: dict[str, Any] = {
        "hidden_size": cfg.hidden_size,
        "num_hidden_layers": cfg.num_hidden_layers,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads,
        "head_dim": cfg.head_dim,
        "intermediate_size": cfg.intermediate_size,
        "vocab_size": cfg.vocab_size,
        "rms_norm_eps": cfg.rms_norm_eps,
        "rope_theta": cfg.rope_theta,
        "max_position_embeddings": cfg.max_position_embeddings,
        "block_size": cfg.block_size,
        "num_target_layers": cfg.num_target_layers,
        "layer_types": list(cfg.layer_types),
        "dflash_config": {
            "target_layer_ids": list(cfg.target_layer_ids),
            "mask_token_id": cfg.mask_token_id,
            "dflash_attention_version": 1 if cfg.attention_causal else 2,
        },
    }
    if cfg.rope_scaling is not None:
        out["rope_scaling"] = cfg.rope_scaling
    if cfg.sliding_window is not None:
        out["sliding_window"] = cfg.sliding_window
    if cfg.final_logit_softcapping is not None:
        out["final_logit_softcapping"] = cfg.final_logit_softcapping
    return out
