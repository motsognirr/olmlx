"""A/B bench: shard vs spectral vs turboquant KV-cache quantization (#377).

Measures, per mode, on a real model:
  - reconstruction cosine similarity of K and V vs the unquantized cache
    (overall, and middle-region-only for shard — sink/window are exact)
  - decode throughput (tok/s) and prompt throughput via mlx_lm stream_generate
  - resident cache bytes (every mx.array attribute on every layer cache,
    which charges TurboQuant for its persistent dequant side buffers and
    credits shard's transient strategy)

Calibration artifacts are created in the model dir's standard spectral/ and
shard/ locations on first run and reused afterwards.

Usage:
  uv run python scripts/shard_ab_bench.py \
      --model mlx-community/Qwen3-0.6B-4bit --bits 4 \
      --prompt-tokens 1024 --decode-tokens 128 --samples 32
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

PARAGRAPH = (
    "The unified memory architecture of Apple Silicon changes the calculus "
    "of local inference: weights, activations, and the key-value cache all "
    "compete for the same physical pool, so cache compression converts "
    "directly into longer usable context. Quantizing the cache trades "
    "per-step dequantization compute for resident bytes, and the right "
    "trade depends on whether the workload is bounded by memory capacity "
    "or by decode latency. "
)


def _model_dir(model: str) -> Path:
    from olmlx.config import settings
    from olmlx.models.store import _safe_dir_name

    p = settings.models_dir / _safe_dir_name(model)
    if (p / "config.json").exists():
        return p
    if (Path(model) / "config.json").exists():
        return Path(model)
    raise SystemExit(f"model {model} not found at {p}; pull it first")


def _ensure_calibrations(model_dir: Path, bits: int, samples: int) -> None:
    if not (model_dir / "spectral" / "spectral_config.json").exists():
        from olmlx.engine.spectralquant_calibrate import calibrate_model

        print(f"[calibrate] spectral ({samples} samples)...")
        calibrate_model(
            str(model_dir),
            num_samples=samples,
            calibration_dataset="synthetic",
            avg_bits=bits,
        )
    if not (model_dir / "shard" / "shard_config.json").exists():
        from olmlx.engine.shardquant_calibrate import calibrate_model_shard

        print(f"[calibrate] shard ({samples} samples)...")
        calibrate_model_shard(
            str(model_dir),
            num_samples=samples,
            calibration_dataset="synthetic",
            bits=bits,
        )


def _make_cache(mode: str, model, model_dir: Path, bits: int) -> list:
    if mode == "fp16":
        from mlx_lm.models.cache import make_prompt_cache

        return make_prompt_cache(model)
    if mode == "turboquant":
        from olmlx.engine.turboquant_cache import make_turboquant_cache

        return make_turboquant_cache(model, bits=bits)
    if mode == "spectral":
        from olmlx.engine.spectralquant_cache import make_spectral_cache

        return make_spectral_cache(model, model_dir / "spectral", avg_bits=bits)
    if mode == "shard":
        from olmlx.engine.shardquant_cache import make_shard_cache

        return make_shard_cache(model, model_dir / "shard", bits=bits)
    raise ValueError(mode)


def _cosine(a: mx.array, b: mx.array) -> float:
    # Cast in MLX first: numpy's buffer protocol can't read bfloat16.
    an = np.array(a.astype(mx.float32))
    bn = np.array(b.astype(mx.float32))
    num = np.sum(an * bn, axis=-1)
    den = np.linalg.norm(an, axis=-1) * np.linalg.norm(bn, axis=-1) + 1e-9
    return float((num / den).mean())


def _resident_bytes(cache: list) -> int:
    total = 0
    for c in cache:
        for v in c.__dict__.values():
            if isinstance(v, mx.array):
                total += v.nbytes
    return total


def _reconstruction(model, model_dir: Path, modes: list[str], bits: int, kv_ref):
    """kv_ref: list of (K, V) per attention layer from an fp16 prefill."""
    from olmlx.engine.shardquant_cache import ShardKVCache

    results: dict[str, dict[str, float]] = {}
    for mode in modes:
        if mode == "fp16":
            continue
        cache = _make_cache(mode, model, model_dir, bits)
        cos_k, cos_v, mid_k, mid_v = [], [], [], []
        for layer_cache, ref in zip(cache, kv_ref):
            if ref is None:
                continue
            k_true, v_true = ref
            k_out, v_out = layer_cache.update_and_fetch(k_true, v_true)
            mx.eval(k_out, v_out)
            cos_k.append(_cosine(k_out, k_true))
            cos_v.append(_cosine(v_out, v_true))
            if isinstance(layer_cache, ShardKVCache):
                s, w = layer_cache.sink_size, layer_cache.window_size
                if k_true.shape[2] > s + w:
                    mid_k.append(_cosine(k_out[..., s:-w, :], k_true[..., s:-w, :]))
                    mid_v.append(_cosine(v_out[..., s:-w, :], v_true[..., s:-w, :]))
        results[mode] = {
            "cos_k": float(np.mean(cos_k)),
            "cos_v": float(np.mean(cos_v)),
        }
        if mid_k:
            results[mode]["cos_k_mid"] = float(np.mean(mid_k))
            results[mode]["cos_v_mid"] = float(np.mean(mid_v))
        del cache
        mx.clear_cache()
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="mlx-community/Qwen3-0.6B-4bit")
    ap.add_argument("--bits", type=int, default=4, choices=[2, 4, 8])
    ap.add_argument("--prompt-tokens", type=int, default=1024)
    ap.add_argument("--decode-tokens", type=int, default=128)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument(
        "--modes",
        default="fp16,turboquant,spectral,shard",
        help="comma-separated subset of fp16,turboquant,spectral,shard",
    )
    args = ap.parse_args()
    modes = args.modes.split(",")

    from mlx_lm import load, stream_generate
    from mlx_lm.models.cache import KVCache, make_prompt_cache

    model_dir = _model_dir(args.model)
    _ensure_calibrations(model_dir, args.bits, args.samples)

    print(f"[load] {model_dir}")
    model, tokenizer = load(str(model_dir))

    # Build a prompt of ~prompt_tokens tokens.
    text = PARAGRAPH * (args.prompt_tokens // 40 + 2)
    ids = tokenizer.encode(text)[: args.prompt_tokens]
    prompt = tokenizer.decode(ids)
    messages = [{"role": "user", "content": f"Summarize:\n{prompt}"}]
    try:
        chat_prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        chat_prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )

    # Reference K/V from an fp16 prefill (one entry per layer; None for
    # non-KVCache layers).
    ref_cache = make_prompt_cache(model)
    model(mx.array([chat_prompt]), cache=ref_cache)
    mx.eval([c.state for c in ref_cache if hasattr(c, "state")])
    kv_ref = []
    for c in ref_cache:
        st = c.state if isinstance(c, KVCache) else None
        kv_ref.append((st[0], st[1]) if st is not None and len(st) >= 2 else None)
    del ref_cache
    mx.clear_cache()

    print("\n== Reconstruction (cosine vs fp16 cache) ==")
    recon = _reconstruction(model, model_dir, modes, args.bits, kv_ref)
    for mode, r in recon.items():
        extra = (
            f"  mid-only K={r['cos_k_mid']:.4f} V={r['cos_v_mid']:.4f}"
            if "cos_k_mid" in r
            else ""
        )
        print(f"  {mode:>10}: K={r['cos_k']:.4f} V={r['cos_v']:.4f}{extra}")
    del kv_ref
    mx.clear_cache()

    print(
        f"\n== Generation ({len(chat_prompt)} prompt tokens, "
        f"{args.decode_tokens} decode) =="
    )
    rows = []
    for mode in modes:
        cache = _make_cache(mode, model, model_dir, args.bits)
        last = None
        t0 = time.perf_counter()
        text_out = ""
        for resp in stream_generate(
            model,
            tokenizer,
            chat_prompt,
            max_tokens=args.decode_tokens,
            prompt_cache=cache,
        ):
            text_out += resp.text
            last = resp
        wall = time.perf_counter() - t0
        resident = _resident_bytes(cache)
        rows.append(
            (
                mode,
                last.prompt_tps,
                last.generation_tps,
                resident / 1e6,
                wall,
                text_out[:60].replace("\n", " "),
            )
        )
        del cache
        mx.clear_cache()

    print(
        f"  {'mode':>10}  {'prefill t/s':>11}  {'decode t/s':>10}  "
        f"{'cache MB':>8}  {'wall s':>6}  sample"
    )
    base_mb = next((r[3] for r in rows if r[0] == "fp16"), None)
    for mode, ptps, gtps, mb, wall, sample in rows:
        ratio = f" ({base_mb / mb:.1f}x)" if base_mb and mode != "fp16" else ""
        print(
            f"  {mode:>10}  {ptps:>11.1f}  {gtps:>10.1f}  "
            f"{mb:>8.1f}{ratio:<7}  {wall:>6.1f}  {sample!r}"
        )


if __name__ == "__main__":
    main()
