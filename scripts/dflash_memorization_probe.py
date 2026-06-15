"""Discriminate position-keyed memorization from content learning.

a) Eval per-position accuracy on a SEEN training sequence (from the
   training JSONL) at several pivots.
b) Same sequence SHIFTED right by k tokens (prepend filler): content-
   based drafts keep accuracy; position-keyed memorization collapses.
c) Print actual vs predicted tokens on a held-out window.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

from olmlx.engine.spec_decoder_base import _patch_model, _unpatch_model

sys.path.insert(0, str(Path(__file__).parent))
from dflash_acceptance_probe import load_draft  # noqa: E402


def window_acc(
    target,
    draft,
    config,
    ids: list[int],
    pivots: list[int],
    verbose_pivot: int | None = None,
    tok=None,
):
    storage: list = [None] * len(config.target_layer_ids)
    _patch_model(target, list(config.target_layer_ids), storage)
    try:
        storage[:] = [None] * len(storage)
        target(mx.array([ids]), cache=None)
        hidden_full = mx.concatenate(list(storage), axis=-1)
        mx.eval(hidden_full)
        draft.bind(target)
        bs = config.block_size
        mask_id = int(config.mask_token_id)
        results = []
        for p in pivots:
            block = mx.array([[ids[p]] + [mask_id] * bs])
            ctx = hidden_full[:, :p, :]
            cache = draft.make_cache()
            logits = draft(block, ctx, cache, logits_start=1)
            pred = mx.argmax(logits, axis=-1)
            mx.eval(pred)
            pred_l = pred[0].tolist()
            tgt = ids[p + 1 : p + 1 + bs]
            acc = sum(int(a == b) for a, b in zip(pred_l, tgt)) / len(tgt)
            results.append(acc)
            if verbose_pivot is not None and p == verbose_pivot and tok:
                print(f"    pivot {p} target: {tok.decode(tgt)!r}")
                print(f"    pivot {p} pred  : {tok.decode(pred_l)!r}")
        draft.unbind()
        return results
    finally:
        _unpatch_model(target)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--data", required=True, help="training JSONL")
    args = ap.parse_args()

    target, tok = load(str(Path(args.target).expanduser()))
    draft, config = load_draft(Path(args.draft).expanduser())

    with Path(args.data).open() as f:
        seen = json.loads(f.readline())["ids"]
    seen = seen[:512]
    pivots = [60, 100, 150, 200, 300, 400]

    accs = window_acc(target, draft, config, seen, pivots, verbose_pivot=150, tok=tok)
    print("a) SEEN training sequence:")
    print("   " + " ".join(f"p@{p}={a:.2f}" for p, a in zip(pivots, accs)))

    # b) shift by 7 tokens: prepend filler ids after BOS region.
    filler = tok.encode("Hello there, friend. ", add_special_tokens=False)
    k = len(filler)
    shifted = filler + seen
    accs_s = window_acc(target, draft, config, shifted, [p + k for p in pivots])
    print(f"b) SAME sequence shifted by {k} tokens:")
    print("   " + " ".join(f"p@{p + k}={a:.2f}" for p, a in zip(pivots, accs_s)))


if __name__ == "__main__":
    main()
