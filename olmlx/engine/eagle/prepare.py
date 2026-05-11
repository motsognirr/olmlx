"""Train an EAGLE draft model for a given target.

Pipeline (precompute-only mode for Phase D):

1. Load the target via ``mlx_lm.load`` (only to ``bind()`` its
   ``embed_tokens``/``lm_head`` into the draft — the target itself is
   not run during training).
2. Build an ``EagleConfig`` from the target's ``config.json``.
3. Construct an ``EagleDraftModel``, ``bind()`` to the target, freeze
   the target.
4. Stream precomputed ``(input_ids, target_hidden)`` shards (produced
   by ``olmlx dflash precompute`` — the same shards EAGLE and DFlash
   share). Each batch is a teacher-forcing step:
       inputs at position t = (input_ids[t], h_target[t-1])
       label  at position t = input_ids[t+1]
   CE loss across the sequence (excluding the final position which has
   no label).
5. AdamW + cosine LR schedule.
6. Save to ``<output>/{config.json, model-00001-of-00001.safetensors}``
   with an ``eagle_config`` marker so the inference loader (Phase E)
   can distinguish from DFlash drafts.

EAGLE has no notion of MASK token, no block-diffusion window, and no
per-batch pivot — every position in the sequence supplies a training
signal.

Online (run-target-each-step) training is not supported in Phase D.
The target forward dominates per-step cost on dense Apple-Silicon
inference, and the precompute path delivers ~25x speedup. We always
require ``--use-precomputed``.
"""

from __future__ import annotations

import json
import logging
import math
import random
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils as mx_utils

from olmlx.engine.eagle.draft_model import EagleConfig, EagleDraftModel

logger = logging.getLogger(__name__)


# Defaults
DEFAULT_NUM_HIDDEN_LAYERS = 1
DEFAULT_BLOCK_SIZE = 4
DEFAULT_STEPS = 2000
DEFAULT_BATCH_SIZE = 4
DEFAULT_SEQ_LEN = 2048
DEFAULT_LR = 5e-4
DEFAULT_WARMUP_FRAC = 0.05
# How many positions per sequence to score with ``lm_head`` during
# training. The full sequence is run through self-attention so each
# scored position sees its true context — only the final vocab-size
# projection is subsampled. 256 is a sweet spot for 250k-vocab targets:
# ~10x faster than full scoring, variance across an epoch is dwarfed by
# normal training noise, and convergence matches full scoring within
# logging resolution.
DEFAULT_SAMPLE_POSITIONS = 256


# ---------------------------------------------------------------------------
# Config helpers (self-contained — don't depend on dflash/prepare)
# ---------------------------------------------------------------------------


def _read_target_config(model_path: Path) -> dict[str, Any]:
    cfg_path = model_path / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Target config not found at {cfg_path}")
    return json.loads(cfg_path.read_text())


def _text_config(target_cfg: dict[str, Any]) -> dict[str, Any]:
    """Return the text-tower portion of a target config.

    Multimodal targets nest text-tower fields under ``text_config``.
    Descend only when the nested block actually carries text-tower
    fields (``hidden_size`` is the canonical marker) — a non-VLM that
    happens to use ``text_config`` for an unrelated purpose would
    otherwise regress from a working flat lookup to a ``KeyError``.
    """
    nested = target_cfg.get("text_config")
    if isinstance(nested, dict) and "hidden_size" in nested:
        return nested
    return target_cfg


def _build_eagle_config(
    target_cfg: dict[str, Any],
    *,
    num_hidden_layers: int,
    block_size: int,
) -> EagleConfig:
    """Derive an ``EagleConfig`` from the target's config.json."""
    text_cfg = _text_config(target_cfg)

    def _get(key: str, default: Any) -> Any:
        v = text_cfg.get(key)
        return v if v is not None else default

    hidden_size = int(text_cfg["hidden_size"])
    raw_num_heads = text_cfg.get("num_attention_heads")
    if raw_num_heads is None:
        raise ValueError(
            "target config.json is missing 'num_attention_heads'. There is "
            "no safe default — the head count drives head_dim derivation."
        )
    num_attention_heads = int(raw_num_heads)
    num_kv_heads = int(_get("num_key_value_heads", num_attention_heads))
    head_dim = int(_get("head_dim", hidden_size // num_attention_heads))
    intermediate_size = int(_get("intermediate_size", hidden_size * 4))
    rms_norm_eps = float(_get("rms_norm_eps", 1e-6))

    rope_params = text_cfg.get("rope_parameters") or target_cfg.get("rope_parameters")
    if text_cfg.get("rope_theta") is not None:
        rope_theta = float(text_cfg["rope_theta"])
    elif isinstance(rope_params, dict) and rope_params.get("rope_theta") is not None:
        rope_theta = float(rope_params["rope_theta"])
    else:
        # Default 10000.0 is off by 1000× from the long-context bases
        # modern Qwen3.5/3.6 targets use; surface the fallback so the
        # operator notices.
        rope_theta = 10000.0
        logger.warning(
            "No 'rope_theta' found at the top level or under "
            "'rope_parameters' in the target config — falling back to "
            "10000.0. Long-context targets (Qwen3.5+, Qwen3.6) typically "
            "use ~10_000_000; verify the target's config.json."
        )
    max_position_embeddings = int(_get("max_position_embeddings", 4096))

    return EagleConfig(
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
        # Prefer ``text_cfg`` when present; only fall back to the
        # top-level when ``text_cfg`` lacks the key entirely. ``or``
        # would treat an explicitly empty dict ``{}`` as "missing" and
        # silently fall through to the top-level value — same class of
        # foot-gun the ``rope_theta`` path explicitly defends against.
        rope_scaling=(
            text_cfg["rope_scaling"]
            if "rope_scaling" in text_cfg
            else target_cfg.get("rope_scaling")
        ),
    )


# ---------------------------------------------------------------------------
# Loss + LR schedule
# ---------------------------------------------------------------------------


def _eagle_loss(
    draft: EagleDraftModel,
    target_hidden: mx.array,
    input_ids: mx.array,
    *,
    sample_positions: int | None = None,
) -> mx.array:
    """Autoregressive next-token CE under teacher forcing.

    EAGLE alignment: at training position ``t`` the draft sees
    ``(h_{t-1}, token_t)`` and predicts ``token_{t+1}``. The hidden
    corresponds to the position *before* the current token was seen
    — this is what the published recipe expects and what the inference
    decoder feeds (after prefill, ``seed_hidden = h_{P-1}`` and
    ``seed_token = token_P``; after each verify, the captured hidden
    at slot ``num_accepted - 1`` is the hidden *before* the new seed
    token was incorporated). Aligning at the same index instead would
    make the draft see ``h_t`` (which already contains ``token_t``)
    and learn a redundant mapping that never matches the inference
    pairing — observed empirically as ~1% acceptance rate at bench.

    Indexing: ``t`` ranges over ``1..L-2`` (need ``t+1 <= L-1`` for
    the label and ``t-1 >= 0`` for the hidden). Each batch of length
    ``L`` therefore yields ``L-2`` training positions.

    ``target_hidden`` shape ``(B, L, H)`` is the *last layer* of the
    target's forward — caller must slice to a single layer before
    calling.

    ``sample_positions``: if set, randomly subsample this many
    positions per sequence and compute loss only there. The full
    draft forward (self-attention over all positions) still runs so
    each sampled position sees its true context, but we skip the
    expensive ``lm_head`` projection at the un-sampled positions.
    On a 250k-vocab × 2048-position × batch=4 setup the full logits
    tensor is ~4 GB per forward; sampling 256 positions cuts that to
    ~512 MB and is ~10x faster end-to-end. Per-position CE is i.i.d.
    so sampling is an unbiased estimator of the full-sequence mean.
    Pass ``None`` to compute loss at every position (production
    behavior; tests rely on it).
    """
    tokens = input_ids[:, 1:-1]
    h = target_hidden[:, :-2, :]
    labels = input_ids[:, 2:]

    if sample_positions is None:
        # Full path: lm_head over every position. Cheap on tests with
        # tiny vocabs; expensive on real targets.
        logits, _h_new = draft(token_ids=tokens, h_prev=h)
        # ``logits`` is annotated ``mx.array | None`` because the
        # subsampled path passes ``compute_logits=False``. In this
        # branch we pass the default ``compute_logits=True``, so it
        # must be a real array. Make the type narrowing explicit so a
        # future refactor that flips ``compute_logits`` here crashes
        # at the call site rather than mid-softmax.
        assert logits is not None
        log_probs = nn.log_softmax(logits, axis=-1)
        nll = -mx.take_along_axis(log_probs, labels[..., None], axis=-1).squeeze(-1)
        return mx.mean(nll)

    # Subsampled path: skip lm_head in the draft forward, then apply it
    # manually at the sampled positions only.
    if draft.lm_head is None:
        raise RuntimeError(
            "_eagle_loss(sample_positions=...) requires the draft to be "
            "bound to a target so lm_head is available."
        )
    _none, h_new = draft(token_ids=tokens, h_prev=h, compute_logits=False)
    L = tokens.shape[1]
    # Pick ``min(sample_positions, L)`` distinct positions per batch
    # row. Same indices across batch rows is fine — independent
    # sampling per row would yield ragged tensors and the variance
    # gain is negligible at our batch sizes.
    k = min(sample_positions, L)
    idx = mx.array(
        sorted(random.sample(range(L), k)),
        dtype=mx.int32,
    )
    h_sub = h_new[:, idx, :]
    labels_sub = labels[:, idx]
    logits_sub = draft.lm_head(h_sub)
    log_probs = nn.log_softmax(logits_sub, axis=-1)
    nll = -mx.take_along_axis(log_probs, labels_sub[..., None], axis=-1).squeeze(-1)
    return mx.mean(nll)


def _cosine_lr(step: int, total: int, peak: float, warmup: int) -> float:
    """Linear warmup then cosine decay to 10% of peak.

    Caller invariant: ``0 <= step < total`` (the training loop breaks
    at ``step >= total`` before calling this), so we don't need a
    ``step >= total`` guard. ``progress`` is bounded by ``1`` for
    the same reason.
    """
    if step < warmup:
        return peak * (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return peak * (0.1 + 0.9 * cosine)


# ---------------------------------------------------------------------------
# Save/load helpers
# ---------------------------------------------------------------------------


def _config_to_disk(
    cfg: EagleConfig,
    *,
    target_layer_id: int | None = None,
) -> dict[str, Any]:
    eagle_block: dict[str, Any] = {"block_size": cfg.block_size}
    if target_layer_id is not None:
        # The decoder must hook the same target layer the draft was
        # trained against; otherwise it sees a hidden from a different
        # distribution at inference (~5% acceptance vs ~50% expected).
        # Persist it here so ``_load_eagle_decoder`` can wire it in.
        eagle_block["target_layer_id"] = int(target_layer_id)
    return {
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
        "rope_scaling": cfg.rope_scaling,
        "eagle_config": eagle_block,
    }


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------


def prepare_eagle_draft(
    model_path: str | Path,
    *,
    use_precomputed: str | Path | None = None,
    steps: int = DEFAULT_STEPS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seq_len: int = DEFAULT_SEQ_LEN,
    block_size: int = DEFAULT_BLOCK_SIZE,
    num_hidden_layers: int = DEFAULT_NUM_HIDDEN_LAYERS,
    lr: float = DEFAULT_LR,
    sample_positions: int | None = DEFAULT_SAMPLE_POSITIONS,
    seed: int = 0,
    output_dir: str | Path | None = None,
    progress_callback: Callable[[str, float], None] | None = None,
    log_every: int = 50,
    _target_loader: Callable[[str], tuple[Any, Any]] | None = None,
    _batch_iterator: Iterable[tuple[mx.array, mx.array]] | None = None,
) -> Path:
    """Train an EAGLE draft model and write it to disk.

    ``use_precomputed`` points at a directory of shards produced by
    ``olmlx dflash precompute`` (EAGLE and DFlash share the same
    shard format; EAGLE consumes only the deepest captured layer).

    ``seed`` makes the training run reproducible. We seed *both* mlx's
    PRNG (drives weight init, dropout, etc.) and Python's stdlib
    ``random`` (drives the position subsample inside
    ``_eagle_loss(sample_positions=...)`` via ``random.sample``).
    Seeding only ``mx.random`` would leave the position draw
    non-deterministic and identical hyperparameter runs would converge
    to slightly different checkpoints, making regressions hard to
    bisect.

    ``_target_loader`` and ``_batch_iterator`` are injection hooks for
    tests so the trainer can run without downloading a multi-GB target
    or hitting disk. In normal use the trainer defaults to
    ``mlx_lm.load`` and ``iter_precomputed_shards``.
    """
    mx.random.seed(seed)
    random.seed(seed)
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1; got {block_size}")
    if steps < 1:
        raise ValueError(f"steps must be >= 1; got {steps}")
    if seq_len < 3:
        # ``_eagle_loss`` slices ``input_ids[:, 1:-1]`` (length seq_len-2)
        # under the EAGLE pairing (h_{t-1}, token_t) -> token_{t+1}.
        # seq_len < 3 leaves zero training positions and ``mx.mean`` over
        # an empty CE vector returns NaN, which would silently corrupt
        # the optimizer state. Reject up front with a clear message.
        raise ValueError(
            f"seq_len must be >= 3 for EAGLE training (loss needs at least "
            f"one position with both a preceding hidden and a successor "
            f"label after slicing); got {seq_len}"
        )

    model_path = Path(model_path)
    target_cfg = _read_target_config(model_path)

    if _target_loader is None:
        from mlx_lm import load as _mlx_lm_load

        loaded = _mlx_lm_load(str(model_path))
        target, _tokenizer = loaded[0], loaded[1]
    else:
        target, _tokenizer = _target_loader(str(model_path))

    target.eval()  # type: ignore[attr-defined]
    if hasattr(target, "freeze"):
        target.freeze()

    eagle_cfg = _build_eagle_config(
        target_cfg,
        num_hidden_layers=num_hidden_layers,
        block_size=block_size,
    )

    draft = EagleDraftModel(eagle_cfg)
    mx.eval(draft.parameters())

    # Optimizer + LR schedule.
    optimizer = optim.AdamW(learning_rate=lr)
    warmup = max(int(steps * DEFAULT_WARMUP_FRAC), 1)

    def loss_fn(
        model: EagleDraftModel,
        h: mx.array,
        ids: mx.array,
    ) -> mx.array:
        return _eagle_loss(model, h, ids, sample_positions=sample_positions)

    loss_and_grad = nn.value_and_grad(draft, loss_fn)

    def _step(h: mx.array, ids: mx.array) -> mx.array:
        loss, grads = loss_and_grad(draft, h, ids)
        optimizer.update(draft, grads)
        return loss

    # Set up batch source.
    eagle_target_layer_id: int | None = None
    if _batch_iterator is not None:
        # ``_batch_iterator`` is a test-only injection hook. Combining
        # it with ``use_precomputed`` would silently take the
        # iterator path and *not* derive ``eagle_target_layer_id``
        # from the shard index, producing a saved config without
        # ``target_layer_id`` — a misconfigured checkpoint that
        # would later collapse bench acceptance to the level
        # observed before the target_layer_id fix landed. Forbid
        # the combination so future callers can't trip on it.
        if use_precomputed is not None:
            raise ValueError(
                "prepare_eagle_draft: ``_batch_iterator`` (test injection) "
                "and ``use_precomputed`` are mutually exclusive — passing "
                "both would produce a saved checkpoint without "
                "``target_layer_id``, leading to a layer-mismatch at "
                "inference. Pick one path."
            )
        batches = _batch_iterator
    else:
        if use_precomputed is None:
            raise ValueError(
                "Phase D requires --use-precomputed: pass the directory of "
                "shards produced by `olmlx dflash precompute`. Online "
                "(run-target-each-step) training is not implemented."
            )
        from olmlx.engine.dflash.precompute import (
            iter_precomputed_shards,
            read_precomputed_index,
        )

        meta = read_precomputed_index(use_precomputed)
        shard_concat_size = int(meta["concat_hidden_size"])
        # EAGLE only needs the *deepest* layer of the shard's
        # concatenated hidden ladder. Slice accordingly per batch.
        target_hidden_size = eagle_cfg.hidden_size
        if shard_concat_size % target_hidden_size != 0:
            raise ValueError(
                f"Precomputed shard concat_hidden_size ({shard_concat_size}) is "
                f"not a multiple of target hidden_size ({target_hidden_size}); "
                "shards may have been produced with a different target."
            )
        # The shards' batch_size and seq_len are baked in at precompute
        # time and override the caller's ``batch_size`` / ``seq_len``
        # — neither is consulted again in this branch. Surface that
        # silently-ignored config rather than letting an operator
        # believe ``--batch-size 8`` had any effect.
        shard_batch_size = int(meta["batch_size"])
        shard_seq_len = int(meta["seq_len"])
        if shard_batch_size != batch_size:
            raise ValueError(
                f"Precomputed shards were written with batch_size="
                f"{shard_batch_size} but ``--batch-size {batch_size}`` was "
                "requested. The shard batch shape is the effective batch "
                "size during training; rerun `olmlx dflash precompute` "
                f"with --batch-size {batch_size}, or pass --batch-size "
                f"{shard_batch_size} to acknowledge the shard layout."
            )
        if shard_seq_len != seq_len:
            raise ValueError(
                f"Precomputed shards were written with seq_len="
                f"{shard_seq_len} but ``--seq-len {seq_len}`` was "
                "requested. The shard sequence length is the effective "
                "training context; rerun `olmlx dflash precompute` "
                f"with --seq-len {seq_len}, or pass --seq-len "
                f"{shard_seq_len} to acknowledge the shard layout."
            )
        # Record the *layer index* the deepest captured hidden came
        # from. dflash precompute usually stores hiddens from a few
        # mid-network layers (e.g. [13, 25, 38, 50] for a 64-layer
        # target), and the deepest one — index ``target_layer_ids[-1]``
        # in the shard ladder — is what we slice into ``h_last`` and
        # train against. The inference decoder MUST then hook *that*
        # same layer; otherwise it feeds the draft hiddens from a
        # different distribution (e.g. layer 63's post-final-norm
        # output) and acceptance collapses (~5% observed). Persist
        # the layer index in the EAGLE config so the loader can wire
        # it through.
        captured_layer_ids = list(meta.get("target_layer_ids") or [])
        if not captured_layer_ids:
            raise ValueError(
                "Precomputed shard index.json is missing 'target_layer_ids'. "
                "EAGLE inference needs the captured layer index to hook the "
                "matching layer at runtime; rerun `olmlx dflash precompute` "
                "with a recent olmlx version that records this field."
            )
        # The slice ``h_concat[:, :, -target_hidden_size:]`` below assumes
        # layers are concatenated in *ascending* order with the deepest
        # layer at the end of the feature axis. DFlash's precompute today
        # preserves this order (it concatenates ``storage`` in the same
        # order ``target_layer_ids`` was passed), but if a future shard
        # writer reorders or shuffles, training would silently consume
        # the wrong layer's hiddens and persist a misleading
        # ``target_layer_id``. Validate explicitly.
        if captured_layer_ids != sorted(captured_layer_ids):
            raise ValueError(
                f"Precomputed shard 'target_layer_ids' must be sorted "
                f"ascending so the slice ``h_concat[:, :, -hidden_size:]`` "
                f"yields the deepest captured layer; got {captured_layer_ids}. "
                "Rerun `olmlx dflash precompute` with sorted --target-layer-ids."
            )
        eagle_target_layer_id = int(captured_layer_ids[-1])
        logger.info(
            "EAGLE will train on (and bind to at inference) target layer %d "
            "based on precomputed shard ladder %s",
            eagle_target_layer_id,
            captured_layer_ids,
        )

        def _slice_iter():
            for ids, h_concat in iter_precomputed_shards(
                use_precomputed, max_examples=steps
            ):
                # Take the LAST (deepest) layer's hidden along the
                # concatenated feature axis.
                h_last = h_concat[:, :, -target_hidden_size:]
                yield ids, h_last

        batches = _slice_iter()

    losses: list[float] = []
    try:
        draft.bind(target)
        for step, batch in enumerate(batches):
            if step >= steps:
                break
            optimizer.learning_rate = _cosine_lr(step, steps, lr, warmup)
            input_ids, h_last = batch
            loss = _step(h_last, input_ids)
            mx.eval(loss, draft.parameters(), optimizer.state)
            losses.append(float(loss.item()))

            if (step + 1) % log_every == 0 or step == 0:
                window = losses[-log_every:]
                avg = sum(window) / len(window)
                logger.info(
                    "step %d/%d  loss=%.4f  avg(%d)=%.4f  lr=%.2e",
                    step + 1,
                    steps,
                    losses[-1],
                    log_every,
                    avg,
                    optimizer.learning_rate,
                )
            if progress_callback:
                progress_callback(
                    f"Training step {step + 1}/{steps} loss={losses[-1]:.4f}",
                    (step + 1) / steps,
                )
    finally:
        draft.unbind()

    # Save checkpoint.
    if output_dir is None:
        output_dir = model_path / "eagle"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg_dict = _config_to_disk(eagle_cfg, target_layer_id=eagle_target_layer_id)
    (output_dir / "config.json").write_text(json.dumps(cfg_dict, indent=2))

    weights = dict(mx_utils.tree_flatten(draft.parameters()))
    # Drop the bound embed/lm_head references — they live on the target
    # and are re-bound on load. The ``object.__setattr__`` guard in
    # ``EagleDraftModel.bind`` already keeps them out of
    # ``draft.parameters()``, so this filter is belt-and-suspenders.
    weights = {
        k: v
        for k, v in weights.items()
        if not k.startswith("embed_tokens.") and not k.startswith("lm_head.")
    }
    mx.save_safetensors(str(output_dir / "model-00001-of-00001.safetensors"), weights)
    logger.info("Saved EAGLE draft to %s (%d weight tensors)", output_dir, len(weights))
    return output_dir
