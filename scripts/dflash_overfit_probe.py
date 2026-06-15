"""Overfit test: can the DFlash draft memorize a single training batch?

Trains the draft on ONE fixed batch and a few fixed pivots for N steps.
If per-position accuracy on the trained windows does not approach 1.0,
the architecture/objective/gradient path has a bug. If it overfits
cleanly, the training pipeline works and poor real-run acceptance is a
data/scale/domain problem.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load

from olmlx.engine.dflash.prepare import (
    _build_draft_config,
    _draft_loss,
    _read_target_config,
    _resolve_target_layer_ids,
)
from olmlx.engine.dflash.draft_model import DFlashDraftModel
from olmlx.engine.spec_decoder_base import _patch_model, _unpatch_model

TEXT = (
    "The transformer architecture has revolutionized natural language "
    "processing. At its core, a transformer uses self-attention to weigh "
    "the relevance of every token to every other token in a sequence. "
    "Each layer consists of a multi-head attention block followed by a "
    "feed-forward network, with residual connections and layer "
    "normalization throughout. During training, the model learns to "
    "predict the next token given all previous tokens. "
) * 8


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--block-size", type=int, default=16)
    args = ap.parse_args()

    model_path = Path(args.target).expanduser()
    target, tok = load(str(model_path))
    target.eval()
    target.freeze()

    target_cfg = _read_target_config(model_path)
    from olmlx.engine.gdn_rollback import get_model_layers

    n_layers = len(get_model_layers(target))
    layer_ids = _resolve_target_layer_ids(None, 5, n_layers)
    print(f"target_layer_ids={layer_ids}")

    mask_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    cfg = _build_draft_config(
        target_cfg,
        target_layer_ids=layer_ids,
        num_hidden_layers=5,
        block_size=args.block_size,
        mask_token_id=int(mask_id),
    )
    draft = DFlashDraftModel(cfg)
    mx.eval(draft.parameters())

    ids = tok.encode(TEXT, add_special_tokens=True)[:512]
    input_ids = mx.array([ids])
    print(f"seq len = {len(ids)}")

    storage: list = [None] * len(layer_ids)
    _patch_model(target, layer_ids, storage)
    draft.bind(target)
    optimizer = optim.AdamW(learning_rate=args.lr)

    storage[:] = [None] * len(storage)
    target(input_ids, cache=None)
    hidden_full = mx.stop_gradient(mx.concatenate(list(storage), axis=-1))
    mx.eval(hidden_full)

    bs = args.block_size
    pivots = [64, 128, 200, 300]
    mask_block = mx.full((1, bs), int(cfg.mask_token_id), dtype=input_ids.dtype)

    def loss_fn(model, p):
        pending = input_ids[:, p : p + 1]
        block_input = mx.concatenate([pending, mask_block], axis=1)
        targets = input_ids[:, p + 1 : p + 1 + bs]
        ctx = hidden_full[:, :p, :]
        cache = model.make_cache()
        return _draft_loss(model, block_input, ctx, targets, cache)

    lng = nn.value_and_grad(draft, loss_fn)

    try:
        for step in range(args.steps):
            p = pivots[step % len(pivots)]
            loss, grads = lng(draft, p)
            optimizer.update(draft, grads)
            mx.eval(loss, draft.parameters(), optimizer.state)
            if step % 50 == 0 or step == args.steps - 1:
                print(f"step {step}: loss={float(loss.item()):.4f}")

        # eval on trained pivots
        for p in pivots:
            pending = input_ids[:, p : p + 1]
            block_input = mx.concatenate([pending, mask_block], axis=1)
            ctx = hidden_full[:, :p, :]
            cache = draft.make_cache()
            logits = draft(block_input, ctx, cache, logits_start=1)
            pred = mx.argmax(logits, axis=-1)
            mx.eval(pred)
            pred_l = pred[0].tolist()
            tgt = ids[p + 1 : p + 1 + bs]
            acc = sum(int(a == b) for a, b in zip(pred_l, tgt)) / bs
            print(f"pivot {p}: window accuracy = {acc:.2f}")
        # eval on UNSEEN pivots (generalization within the same text)
        for p in [96, 160, 250]:
            pending = input_ids[:, p : p + 1]
            block_input = mx.concatenate([pending, mask_block], axis=1)
            ctx = hidden_full[:, :p, :]
            cache = draft.make_cache()
            logits = draft(block_input, ctx, cache, logits_start=1)
            pred = mx.argmax(logits, axis=-1)
            mx.eval(pred)
            pred_l = pred[0].tolist()
            tgt = ids[p + 1 : p + 1 + bs]
            acc = sum(int(a == b) for a, b in zip(pred_l, tgt)) / bs
            print(f"UNSEEN pivot {p}: window accuracy = {acc:.2f}")
    finally:
        draft.unbind()
        _unpatch_model(target)


if __name__ == "__main__":
    main()
