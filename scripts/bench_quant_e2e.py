"""End-to-end tokens/sec benchmark for TurboQuant KV-cache compression.

Loads an mlx-lm model, runs prefill + decode using TurboQuantKVCache, and
times the run twice — once with the production (mx.compile-wrapped) quantize
kernels, once with the inlined pre-compile baseline. Reports prefill
throughput, decode throughput, and total wall time.

The baseline run is realized by monkey-patching
``olmlx.engine.turboquant.turboquant_quantize`` /
``turboquant_dequantize`` for the duration of that run, so the same
``TurboQuantKVCache`` plumbing (and the same model forward) is exercised
under both implementations.

Usage:
    uv run python scripts/bench_quant_e2e.py \\
        --model ~/.olmlx/models/mlx-community_Qwen2.5-0.5B-Instruct-4bit \\
        --bits 4 --prompt-tokens 512 --gen-tokens 128
"""

from __future__ import annotations

import argparse
import time
from contextlib import contextmanager
from pathlib import Path

import mlx.core as mx
import mlx_lm
from mlx_lm.utils import load

from olmlx.engine import turboquant as tq
from olmlx.engine.turboquant_cache import make_turboquant_cache


# ---- Inlined pre-compile baseline ----------------------------------------


def _baseline_quantize(x, rotation, bits):
    head_dim = x.shape[-1]
    centroids = mx.array(tq.GAUSSIAN_CODEBOOKS[bits], dtype=mx.float32) / mx.sqrt(
        mx.array(float(head_dim))
    )
    norms = mx.sqrt(mx.sum(x.astype(mx.float32) ** 2, axis=-1, keepdims=True))
    x_norm = (
        x.astype(mx.float32) / mx.maximum(norms, mx.array(1e-8, dtype=mx.float32))
    ).astype(x.dtype)
    y = x_norm @ rotation.matrix.T
    best_idx = mx.zeros(y.shape, dtype=mx.uint8)
    best_dist = mx.full(y.shape, float("inf"))
    for ci in range(len(centroids)):
        d = mx.abs(y - centroids[ci])
        better = d < best_dist
        best_idx = mx.where(better, mx.array(ci, dtype=mx.uint8), best_idx)
        best_dist = mx.minimum(best_dist, d)
    return tq.pack_indices(best_idx, bits), norms


def _baseline_dequantize(packed, norms, rotation, bits, dtype=None):
    head_dim = rotation.matrix.shape[0]
    centroids = mx.array(tq.GAUSSIAN_CODEBOOKS[bits], dtype=mx.float32) / mx.sqrt(
        mx.array(float(head_dim))
    )
    indices = tq.unpack_indices(packed, bits, head_dim)
    y_hat = centroids[indices.astype(mx.uint32)]
    x_hat = y_hat @ rotation.matrix
    result = x_hat * norms.astype(x_hat.dtype)
    return result.astype(dtype) if dtype is not None else result


@contextmanager
def use_baseline():
    """Swap in the pre-compile baseline implementations for the duration."""
    real_q = tq.turboquant_quantize
    real_d = tq.turboquant_dequantize
    tq.turboquant_quantize = _baseline_quantize
    tq.turboquant_dequantize = _baseline_dequantize
    # turboquant_cache imports the names at module load — patch there too.
    from olmlx.engine import turboquant_cache as tqc

    real_qc = tqc.turboquant_quantize
    real_dc = tqc.turboquant_dequantize
    tqc.turboquant_quantize = _baseline_quantize
    tqc.turboquant_dequantize = _baseline_dequantize
    try:
        yield
    finally:
        tq.turboquant_quantize = real_q
        tq.turboquant_dequantize = real_d
        tqc.turboquant_quantize = real_qc
        tqc.turboquant_dequantize = real_dc


# ---- Bench --------------------------------------------------------------


def _build_prompt(tokenizer, target_tokens: int) -> str:
    """Generate a synthetic prompt of approximately ``target_tokens`` tokens."""
    base = (
        "The quick brown fox jumps over the lazy dog. Lorem ipsum dolor sit amet, "
        "consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et "
        "dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation. "
    )
    text = ""
    while len(tokenizer.encode(text)) < target_tokens:
        text += base
    # Trim back to roughly the requested length
    ids = tokenizer.encode(text)[:target_tokens]
    return tokenizer.decode(ids)


def _run(
    model,
    tokenizer,
    prompt: str,
    gen_tokens: int,
    bits: int,
    label: str,
) -> dict:
    cache = make_turboquant_cache(model, bits=bits)

    # Prefill: tokens that go in via the prompt.
    prompt_ids = tokenizer.encode(prompt)
    n_prompt = len(prompt_ids)
    print(f"\n[{label}] prefill={n_prompt} tokens, decode={gen_tokens} tokens")

    prefill_t0 = time.perf_counter()
    decode_t0 = None
    last_t = None
    n_decoded = 0
    first_token_latency = None

    for resp in mlx_lm.stream_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=gen_tokens,
        prompt_cache=cache,
    ):
        now = time.perf_counter()
        if first_token_latency is None:
            first_token_latency = now - prefill_t0
            decode_t0 = now
        n_decoded += 1
        last_t = now

    total = (last_t or time.perf_counter()) - prefill_t0
    decode_elapsed = (last_t - decode_t0) if (decode_t0 and last_t) else 0.0
    # Prefill = time from start to first token; decode tps = remaining tokens / decode time.
    prefill_tps = n_prompt / first_token_latency if first_token_latency else 0.0
    decode_tps = (n_decoded - 1) / decode_elapsed if decode_elapsed > 0 else 0.0

    print(
        f"  prefill: {first_token_latency * 1000:8.1f} ms  ({prefill_tps:7.1f} tok/s)\n"
        f"  decode : {decode_elapsed * 1000:8.1f} ms over {n_decoded - 1} tokens"
        f"  ({decode_tps:7.1f} tok/s)\n"
        f"  total  : {total * 1000:8.1f} ms"
    )
    return {
        "prefill_ms": first_token_latency * 1000 if first_token_latency else 0.0,
        "decode_ms": decode_elapsed * 1000,
        "prefill_tps": prefill_tps,
        "decode_tps": decode_tps,
        "n_prompt": n_prompt,
        "n_decoded": n_decoded,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to mlx-lm model dir")
    ap.add_argument("--bits", type=int, default=4, choices=(2, 4))
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--gen-tokens", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=2, help="runs per implementation")
    args = ap.parse_args()

    model_path = Path(args.model).expanduser().resolve()
    print(f"Loading model from {model_path}")
    model, tokenizer = load(str(model_path))

    prompt = _build_prompt(tokenizer, args.prompt_tokens)

    print("\n=== Warmup (compiled) ===")
    _run(model, tokenizer, prompt, gen_tokens=8, bits=args.bits, label="warmup")
    with use_baseline():
        print("\n=== Warmup (baseline) ===")
        _run(model, tokenizer, prompt, gen_tokens=8, bits=args.bits, label="warmup")

    runs_compiled = []
    runs_baseline = []
    for i in range(args.repeats):
        runs_compiled.append(
            _run(
                model,
                tokenizer,
                prompt,
                args.gen_tokens,
                args.bits,
                f"compiled #{i + 1}",
            )
        )
    with use_baseline():
        for i in range(args.repeats):
            runs_baseline.append(
                _run(
                    model,
                    tokenizer,
                    prompt,
                    args.gen_tokens,
                    args.bits,
                    f"baseline #{i + 1}",
                )
            )

    def mean(rows, key):
        return sum(r[key] for r in rows) / len(rows)

    print("\n=== Mean across repeats ===")
    for label, rows in (("baseline", runs_baseline), ("compiled", runs_compiled)):
        print(
            f"  {label:9s}  prefill={mean(rows, 'prefill_ms'):7.1f} ms  "
            f"({mean(rows, 'prefill_tps'):6.1f} tok/s)  "
            f"decode={mean(rows, 'decode_ms'):7.1f} ms  "
            f"({mean(rows, 'decode_tps'):6.1f} tok/s)"
        )

    base_decode_tps = mean(runs_baseline, "decode_tps")
    comp_decode_tps = mean(runs_compiled, "decode_tps")
    base_prefill_tps = mean(runs_baseline, "prefill_tps")
    comp_prefill_tps = mean(runs_compiled, "prefill_tps")
    print(
        f"\n  decode  speedup: {comp_decode_tps / base_decode_tps:5.2f}× "
        f"({base_decode_tps:.1f} → {comp_decode_tps:.1f} tok/s)\n"
        f"  prefill speedup: {comp_prefill_tps / base_prefill_tps:5.2f}× "
        f"({base_prefill_tps:.1f} → {comp_prefill_tps:.1f} tok/s)"
    )


if __name__ == "__main__":
    main()
