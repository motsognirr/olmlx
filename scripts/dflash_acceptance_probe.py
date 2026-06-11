"""Drive DFlashDecoder directly to measure acceptance rate and verify
exactness against plain greedy decoding.

Usage:
    uv run python scripts/dflash_acceptance_probe.py \
        --target ~/.olmlx/models/mlx-community_Qwen3-0.6B-4bit \
        --draft /tmp/dflash-qwen3-0.6b-smoke \
        --max-tokens 200

For each prompt: runs (1) plain greedy decoding with the bare target and
(2) DFlash speculative decoding, then compares the two token sequences.
Greedy speculative decoding is exactness-preserving, so any divergence
beyond rare numerical argmax ties indicates a decoder bug.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from olmlx.engine.dflash.decoder import DFlashDecoder
from olmlx.engine.dflash.draft_model import DFlashDraftModel, DraftConfig
from olmlx.engine.speculative_loaders import _resolve_attention_causal

PROMPTS = [
    "Explain how a transformer language model works in simple terms.",
    "Write a Python function that returns the n-th Fibonacci number, "
    "with a short docstring.",
    "List five facts about the planet Mars.",
    "Summarize the plot of Romeo and Juliet in one paragraph.",
]


def load_draft(draft_dir: Path) -> tuple[DFlashDraftModel, DraftConfig]:
    cfg_dict = json.loads((draft_dir / "config.json").read_text())
    dflash_cfg = cfg_dict["dflash_config"]
    layer_types_raw = (
        cfg_dict.get("layer_types")
        or ["full_attention"] * cfg_dict["num_hidden_layers"]
    )
    config = DraftConfig(
        hidden_size=cfg_dict["hidden_size"],
        num_hidden_layers=cfg_dict["num_hidden_layers"],
        num_attention_heads=cfg_dict["num_attention_heads"],
        num_key_value_heads=cfg_dict["num_key_value_heads"],
        head_dim=cfg_dict["head_dim"],
        intermediate_size=cfg_dict["intermediate_size"],
        vocab_size=cfg_dict["vocab_size"],
        rms_norm_eps=cfg_dict["rms_norm_eps"],
        rope_theta=cfg_dict["rope_theta"],
        max_position_embeddings=cfg_dict["max_position_embeddings"],
        block_size=cfg_dict["block_size"],
        num_target_layers=len(dflash_cfg["target_layer_ids"]),
        target_layer_ids=list(dflash_cfg["target_layer_ids"]),
        mask_token_id=int(dflash_cfg["mask_token_id"]),
        rope_scaling=cfg_dict.get("rope_scaling"),
        layer_types=tuple(layer_types_raw),
        sliding_window=cfg_dict.get("sliding_window"),
        final_logit_softcapping=cfg_dict.get("final_logit_softcapping"),
        attention_causal=_resolve_attention_causal(dflash_cfg),
    )
    draft = DFlashDraftModel(config)
    weight_files = sorted(draft_dir.glob("model*.safetensors"))
    weights: list[tuple[str, mx.array]] = []
    for wf in weight_files:
        weights.extend(mx.load(str(wf)).items())
    draft.load_weights(weights, strict=False)
    return draft, config


def greedy_baseline(target, prompt_ids: mx.array, n: int) -> tuple[list[int], float]:
    """Plain greedy decode; returns (tokens, decode_seconds)."""
    cache = make_prompt_cache(target)
    logits = target(prompt_ids, cache=cache)
    logits = getattr(logits, "logits", logits)
    tok = int(mx.argmax(logits[:, -1, :], axis=-1).item())
    t0 = time.perf_counter()
    out = [tok]
    while len(out) < n:
        logits = target(mx.array([[out[-1]]]), cache=cache)
        logits = getattr(logits, "logits", logits)
        tok = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        out.append(tok)
    return out, time.perf_counter() - t0


def speculative_run(decoder: DFlashDecoder, prompt_ids: mx.array, n: int):
    first = decoder.prefill(prompt_ids)
    out = [first]
    steps = 0
    t0 = time.perf_counter()
    while len(out) < n:
        accepted, _ = decoder.step()
        out.extend(accepted)
        steps += 1
    dt = time.perf_counter() - t0
    stats = decoder.stats_summary()
    return out[:n], stats, steps, dt


def classify_divergence(
    target, prompt_ids: mx.array, base: list[int], match_len: int
) -> str:
    """At the first divergence, check the target's top-2 margin on a fresh
    full-prefix forward. Batched-vs-single-token kernel numerics flip
    near-ties; a large margin indicates a real decoder bug."""
    prefix = mx.concatenate(
        [prompt_ids, mx.array([base[:match_len]], dtype=prompt_ids.dtype)], axis=1
    )
    cache = make_prompt_cache(target)
    lg = target(prefix, cache=cache)
    lg = getattr(lg, "logits", lg)
    last = lg[0, -1, :]
    top2 = mx.argpartition(-last, 2)[:2]
    mx.eval(top2)
    t0, t1 = int(top2[0].item()), int(top2[1].item())
    margin = abs(float(last[t0].item()) - float(last[t1].item()))
    return f"margin={margin:.4f} top2={{{t0},{t1}}} -> " + (
        "numerics near-tie" if margin <= 0.5 else "SUSPICIOUS (large margin)"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--block-size", type=int, default=None)
    args = ap.parse_args()

    print(f"loading target {args.target} ...", flush=True)
    target, tok = load(str(Path(args.target).expanduser()))
    draft, config = load_draft(Path(args.draft).expanduser())
    block_size = args.block_size or config.block_size

    total_proposed = 0
    total_accepted = 0
    base_tok = base_time = spec_tok = spec_time = 0.0
    all_exact = True
    for prompt in PROMPTS:
        messages = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        ids = mx.array([tok.encode(text, add_special_tokens=False)])

        base, base_dt = greedy_baseline(target, ids, args.max_tokens)

        decoder = DFlashDecoder(
            target_model=target,
            draft_model=draft,
            draft_config=config,
            block_size=block_size,
        )
        spec, stats, steps, dt = speculative_run(decoder, ids, args.max_tokens)
        decoder.reset()

        match_len = 0
        for a, b in zip(base, spec):
            if a != b:
                break
            match_len += 1
        exact = match_len == len(base)
        all_exact = all_exact and exact

        total_proposed += stats["proposed"]
        total_accepted += stats["accepted_draft"]
        base_tok += len(base)
        base_time += base_dt
        spec_tok += len(spec)
        spec_time += dt
        rate = stats["accepted_draft"] / max(stats["proposed"], 1)
        print(f"\n--- prompt: {prompt[:60]}...")
        print(
            f"  acceptance={rate:.3f} "
            f"({stats['accepted_draft']}/{stats['proposed']} draft tokens, "
            f"{steps} steps, avg {stats['avg_tokens_per_step']:.2f} tok/step)"
        )
        print(
            f"  dflash {len(spec) / dt:.1f} tok/s vs greedy "
            f"{len(base) / base_dt:.1f} tok/s "
            f"(speedup {(len(spec) / dt) / (len(base) / base_dt):.2f}x)"
        )
        print(
            f"  exact-match with greedy baseline: {exact} "
            f"(matched {match_len}/{len(base)} tokens)"
        )
        if not exact:
            lo = max(0, match_len - 5)
            hi = match_len + 5
            print(f"  baseline[{lo}:{hi}] = {base[lo:hi]}")
            print(f"  spec    [{lo}:{hi}] = {spec[lo:hi]}")
            print(f"  baseline text: {tok.decode(base[lo:hi])!r}")
            print(f"  spec text    : {tok.decode(spec[lo:hi])!r}")
            print(f"  divergence: {classify_divergence(target, ids, base, match_len)}")
        print(f"  spec output: {tok.decode(spec)[:200]!r}")

    overall = total_accepted / max(total_proposed, 1)
    print(
        f"\n=== overall acceptance: {overall:.3f} "
        f"({total_accepted}/{total_proposed}) ==="
    )
    spec_rate = spec_tok / max(spec_time, 1e-9)
    base_rate = base_tok / max(base_time, 1e-9)
    print(
        f"=== dflash {spec_rate:.1f} tok/s vs greedy {base_rate:.1f} tok/s: "
        f"speedup {spec_rate / base_rate:.2f}x ==="
    )
    print(f"=== all prompts exact-match: {all_exact} ===")


if __name__ == "__main__":
    main()
