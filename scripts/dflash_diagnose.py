"""Diagnose DFlash acceptance: training-mode vs inference-mode draft accuracy.

Training-mode: replicate _draft_loss's exact setup on a real generation
transcript (full-sequence target forward, ctx = hidden[:, :p], block =
[tok_p, MASK*bs]) and measure per-position draft top-1 accuracy.

Inference-mode: drive DFlashDecoder on the same prompt and measure
per-position acceptance.

If training-mode position-1 accuracy >> inference-mode, the decoder
plumbing mismatches training. If both are low, training is the problem.

Also re-checks the first greedy-baseline divergence: prints the target's
top-2 logit margin at the divergence point (tiny margin → batched-vs-
single-token kernel numerics, not a cache bug).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from olmlx.engine.dflash.decoder import DFlashDecoder
from olmlx.engine.spec_decoder_base import _patch_model, _unpatch_model

import sys

sys.path.insert(0, str(Path(__file__).parent))
from dflash_acceptance_probe import load_draft  # noqa: E402

PROMPT = "Explain how a transformer language model works in simple terms."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--gen-tokens", type=int, default=128)
    args = ap.parse_args()

    target, tok = load(str(Path(args.target).expanduser()))
    draft, config = load_draft(Path(args.draft).expanduser())
    bs = config.block_size

    messages = [{"role": "user", "content": PROMPT}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tok.encode(text, add_special_tokens=False)
    n_prompt = len(prompt_ids)

    # --- greedy transcript ---
    cache = make_prompt_cache(target)
    ids = mx.array([prompt_ids])
    logits = target(ids, cache=cache)
    logits = getattr(logits, "logits", logits)
    cur = int(mx.argmax(logits[:, -1, :]).item())
    gen = [cur]
    while len(gen) < args.gen_tokens:
        logits = target(mx.array([[gen[-1]]]), cache=cache)
        logits = getattr(logits, "logits", logits)
        gen.append(int(mx.argmax(logits[:, -1, :]).item()))
    full = prompt_ids + gen
    print(f"prompt={n_prompt} tokens, generated={len(gen)} tokens")

    # --- training-mode eval ---
    storage: list = [None] * len(config.target_layer_ids)
    _patch_model(target, list(config.target_layer_ids), storage)
    try:
        storage[:] = [None] * len(storage)
        full_arr = mx.array([full])
        target(full_arr, cache=None)
        hidden_full = mx.concatenate(list(storage), axis=-1)
        mx.eval(hidden_full)

        draft.bind(target)
        mask_id = int(config.mask_token_id)
        pos_hits = [0] * bs
        pos_total = 0
        # pivots across the generated region (same regime inference sees)
        pivots = list(range(n_prompt - 1, len(full) - bs - 1))
        for p in pivots:
            block = mx.array([[full[p]] + [mask_id] * bs])
            ctx = hidden_full[:, :p, :]
            dcache = draft.make_cache()
            dlogits = draft(block, ctx, dcache, logits_start=1)
            pred = mx.argmax(dlogits, axis=-1)
            mx.eval(pred)
            pred_l = pred[0].tolist()
            for j in range(bs):
                if full[p + 1 + j] == pred_l[j]:
                    pos_hits[j] += 1
            pos_total += 1
        print(
            "\n--- training-mode per-position top-1 accuracy "
            f"({pos_total} pivots, ctx=full-seq forward) ---"
        )
        print(
            "  "
            + " ".join(f"p{j + 1}={pos_hits[j] / pos_total:.2f}" for j in range(bs))
        )
        draft.unbind()
    finally:
        _unpatch_model(target)

    # --- inference-mode eval ---
    decoder = DFlashDecoder(
        target_model=target,
        draft_model=draft,
        draft_config=config,
        block_size=bs,
    )
    first = decoder.prefill(mx.array([prompt_ids]))
    out = [first]
    pos1_hits = 0
    steps = 0
    per_step_accept = []
    while len(out) < args.gen_tokens:
        accepted, _ = decoder.step()
        per_step_accept.append(len(accepted) - 1)
        if len(accepted) > 1:
            pos1_hits += 1
        out.extend(accepted)
        steps += 1
    stats = decoder.stats_summary()
    decoder.reset()
    print(f"\n--- inference-mode ({steps} steps) ---")
    print(f"  position-1 hit rate: {pos1_hits / steps:.2f}")
    print(f"  acceptance: {stats['acceptance_rate']:.3f}")
    print(f"  accepted-per-step: {per_step_accept[:30]}")

    # --- divergence margin check ---
    div = next((i for i, (a, b) in enumerate(zip(gen, out)) if a != b), None)
    if div is None:
        print(
            f"\nno divergence from greedy baseline in {min(len(gen), len(out))} tokens"
        )
    else:
        prefix = prompt_ids + gen[:div]
        cache2 = make_prompt_cache(target)
        lg = target(mx.array([prefix]), cache=cache2)
        lg = getattr(lg, "logits", lg)
        last = lg[0, -1, :]
        top2 = mx.argpartition(-last, 2)[:2]
        mx.eval(top2)
        t0, t1 = int(top2[0].item()), int(top2[1].item())
        v0, v1 = float(last[t0].item()), float(last[t1].item())
        print(
            f"\nfirst divergence at generated index {div}: "
            f"baseline={gen[div]} spec={out[div]}"
        )
        print(
            f"  fresh-prefill top-2: {t0} ({v0:.4f}) vs {t1} ({v1:.4f}), "
            f"margin={abs(v0 - v1):.4f}"
        )
        print(
            f"  baseline token in top-2: {gen[div] in (t0, t1)}, "
            f"spec token in top-2: {out[div] in (t0, t1)}"
        )


if __name__ == "__main__":
    main()
