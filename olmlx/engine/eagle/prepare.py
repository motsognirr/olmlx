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
        rope_scaling=text_cfg.get("rope_scaling") or target_cfg.get("rope_scaling"),
    )


# ---------------------------------------------------------------------------
# Loss + LR schedule
# ---------------------------------------------------------------------------


def _eagle_loss(
    draft: EagleDraftModel,
    target_hidden: mx.array,
    input_ids: mx.array,
) -> mx.array:
    """Autoregressive next-token CE under teacher forcing.

    At each position ``t`` the draft sees the target's hidden ``h_t``
    and the token at position ``t`` (``input_ids[t]``); it must
    predict ``input_ids[t+1]``. The last position has no label so we
    drop it.

    ``target_hidden`` shape ``(B, L, H)`` is the *last layer* of the
    target's forward — caller must slice to a single layer before
    calling.
    """
    # Inputs span positions 0..L-2; labels are positions 1..L-1.
    tokens = input_ids[:, :-1]
    h = target_hidden[:, :-1, :]
    labels = input_ids[:, 1:]
    logits, _h_new = draft(token_ids=tokens, h_prev=h)
    log_probs = nn.log_softmax(logits, axis=-1)
    nll = -mx.take_along_axis(log_probs, labels[..., None], axis=-1).squeeze(-1)
    return mx.mean(nll)


def _cosine_lr(step: int, total: int, peak: float, warmup: int) -> float:
    """Linear warmup then cosine decay to 10% of peak."""
    if step < warmup:
        return peak * (step + 1) / max(warmup, 1)
    if step >= total:
        return peak * 0.1
    progress = (step - warmup) / max(total - warmup, 1)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return peak * (0.1 + 0.9 * cosine)


# ---------------------------------------------------------------------------
# Save/load helpers
# ---------------------------------------------------------------------------


def _config_to_disk(cfg: EagleConfig) -> dict[str, Any]:
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
        "eagle_config": {
            "block_size": cfg.block_size,
        },
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

    ``_target_loader`` and ``_batch_iterator`` are injection hooks for
    tests so the trainer can run without downloading a multi-GB target
    or hitting disk. In normal use the trainer defaults to
    ``mlx_lm.load`` and ``iter_precomputed_shards``.
    """
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1; got {block_size}")
    if steps < 1:
        raise ValueError(f"steps must be >= 1; got {steps}")

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
        return _eagle_loss(model, h, ids)

    loss_and_grad = nn.value_and_grad(draft, loss_fn)

    def _step(h: mx.array, ids: mx.array) -> mx.array:
        loss, grads = loss_and_grad(draft, h, ids)
        optimizer.update(draft, grads)
        return loss

    # Set up batch source.
    if _batch_iterator is not None:
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

    cfg_dict = _config_to_disk(eagle_cfg)
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
