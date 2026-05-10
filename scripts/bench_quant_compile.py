"""Microbenchmark TurboQuant / SpectralQuant quantize/dequantize hot paths.

Compares the production (mx.compile-wrapped) implementations against an
inlined baseline that mirrors the pre-compile code path. Reports per-call
mean ms, speedup ratio, and projected per-token / per-prefill impact at a
realistic 32-layer model.

Usage:
    uv run python scripts/bench_quant_compile.py [--prefill-len N] [--iters N]
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable

import mlx.core as mx
import numpy as np

from olmlx.engine.turboquant import (
    GAUSSIAN_CODEBOOKS,
    TurboQuantRotation,
    pack_indices,
    turboquant_dequantize,
    turboquant_quantize,
    unpack_indices,
)
from olmlx.engine.spectralquant import (
    SpectralRotation,
    fit_codebook,
    pack_indices as sq_pack_indices,
    spectral_dequantize,
    spectral_quantize,
    unpack_indices as sq_unpack_indices,
)


# ----- Inlined baseline (pre-compile) implementations -----------------------


def _tq_quantize_baseline(
    x: mx.array, rotation: TurboQuantRotation, bits: int
) -> tuple[mx.array, mx.array]:
    head_dim = x.shape[-1]
    centroids = mx.array(GAUSSIAN_CODEBOOKS[bits], dtype=mx.float32) / mx.sqrt(
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
    return pack_indices(best_idx, bits), norms


def _tq_dequantize_baseline(
    packed: mx.array,
    norms: mx.array,
    rotation: TurboQuantRotation,
    bits: int,
    dtype: mx.Dtype,
) -> mx.array:
    head_dim = rotation.matrix.shape[0]
    centroids = mx.array(GAUSSIAN_CODEBOOKS[bits], dtype=mx.float32) / mx.sqrt(
        mx.array(float(head_dim))
    )
    indices = unpack_indices(packed, bits, head_dim)
    y_hat = centroids[indices.astype(mx.uint32)]
    x_hat = y_hat @ rotation.matrix
    result = x_hat * norms.astype(x_hat.dtype)
    return result.astype(dtype)


def _sq_quantize_baseline(
    x: mx.array,
    rotation: SpectralRotation,
    cb_sem: mx.array,
    cb_tail: mx.array,
    d_eff: int,
    bits_high: int,
    bits_low: int,
) -> tuple[mx.array, mx.array, mx.array]:
    norms = mx.sqrt(mx.sum(x.astype(mx.float32) ** 2, axis=-1, keepdims=True))
    x_norm = (
        x.astype(mx.float32) / mx.maximum(norms, mx.array(1e-8, dtype=mx.float32))
    ).astype(x.dtype)
    y = rotation.rotate(x_norm)
    y_sem = y[..., :d_eff]
    y_tail = y[..., d_eff:]

    def _q(data, codebook, bits):
        dists = mx.abs(data[..., None] - codebook)
        best_idx = mx.argmin(dists, axis=-1).astype(mx.uint8)
        return sq_pack_indices(best_idx, bits)

    return _q(y_sem, cb_sem, bits_high), _q(y_tail, cb_tail, bits_low), norms


def _sq_dequantize_baseline(
    packed_sem: mx.array,
    packed_tail: mx.array,
    norms: mx.array,
    rotation: SpectralRotation,
    cb_sem: mx.array,
    cb_tail: mx.array,
    d_eff: int,
    bits_high: int,
    bits_low: int,
    dtype: mx.Dtype,
) -> mx.array:
    head_dim = rotation.V.shape[0]
    d_tail = head_dim - d_eff
    idx_sem = sq_unpack_indices(packed_sem, bits_high, d_eff)
    y_sem = cb_sem[idx_sem.astype(mx.uint32)]
    idx_tail = sq_unpack_indices(packed_tail, bits_low, d_tail)
    y_tail = cb_tail[idx_tail.astype(mx.uint32)]
    y_hat = mx.concatenate([y_sem, y_tail], axis=-1)
    x_hat = rotation.unrotate(y_hat)
    result = x_hat * norms.astype(x_hat.dtype)
    return result.astype(dtype)


# ----- Bench harness --------------------------------------------------------


def _bench(fn: Callable[[], object], iters: int, warmup: int = 20) -> float:
    for _ in range(warmup):
        out = fn()
        if isinstance(out, tuple):
            mx.eval(*out)
        else:
            mx.eval(out)
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn()
        if isinstance(out, tuple):
            mx.eval(*out)
        else:
            mx.eval(out)
    return (time.perf_counter() - t0) / iters * 1000.0


def _print_row(name: str, base_ms: float, comp_ms: float) -> None:
    speedup = base_ms / comp_ms if comp_ms > 0 else float("inf")
    delta = base_ms - comp_ms
    print(
        f"  {name:38s}  base={base_ms:8.4f}  compiled={comp_ms:8.4f}"
        f"  Δ={delta:+8.4f} ms  ({speedup:5.2f}×)"
    )


def bench_turboquant(
    bits: int,
    head_dim: int,
    n_heads: int,
    seq_lens: list[int],
    iters: int,
    layers: int,
) -> None:
    print(f"\n=== TurboQuant {bits}-bit  head_dim={head_dim}  n_heads={n_heads} ===")
    rot = TurboQuantRotation(head_dim=head_dim, seed=0)

    for seq_len in seq_lens:
        print(f"\n[seq_len={seq_len}]")
        x = mx.random.normal(shape=(1, n_heads, seq_len, head_dim), dtype=mx.float16)
        mx.eval(x)

        # Quantize
        idx, nrm = turboquant_quantize(x, rot, bits)
        mx.eval(idx, nrm)
        b_q = _bench(lambda: _tq_quantize_baseline(x, rot, bits), iters)
        c_q = _bench(lambda: turboquant_quantize(x, rot, bits), iters)
        _print_row("quantize", b_q, c_q)

        # Dequantize
        b_d = _bench(
            lambda: _tq_dequantize_baseline(idx, nrm, rot, bits, mx.float16), iters
        )
        c_d = _bench(
            lambda: turboquant_dequantize(idx, nrm, rot, bits, dtype=mx.float16),
            iters,
        )
        _print_row("dequantize", b_d, c_d)

        per_token_base = (b_q + b_d) * layers
        per_token_comp = (c_q + c_d) * layers
        print(
            f"  → projected at {layers} layers: "
            f"base={per_token_base:7.2f} ms  compiled={per_token_comp:7.2f} ms  "
            f"Δ={per_token_base - per_token_comp:+7.2f} ms"
        )


def bench_spectralquant(
    bits_high: int,
    bits_low: int,
    head_dim: int,
    n_heads: int,
    seq_lens: list[int],
    iters: int,
    layers: int,
) -> None:
    print(
        f"\n=== SpectralQuant {bits_high}/{bits_low}-bit  "
        f"head_dim={head_dim}  n_heads={n_heads} ==="
    )
    rng = np.random.RandomState(0)
    a = rng.randn(head_dim, head_dim).astype(np.float32)
    q, _ = np.linalg.qr(a)
    rot = SpectralRotation(V=mx.array(q))
    d_eff = head_dim // 2

    cal = mx.random.normal(shape=(4096,), dtype=mx.float32) / mx.sqrt(
        mx.array(float(head_dim))
    )
    cb_sem = fit_codebook(cal, bits_high, max_iter=20)
    cb_tail = fit_codebook(cal, bits_low, max_iter=20)
    mx.eval(cb_sem, cb_tail)

    for seq_len in seq_lens:
        print(f"\n[seq_len={seq_len}]")
        x = mx.random.normal(shape=(1, n_heads, seq_len, head_dim), dtype=mx.float16)
        mx.eval(x)

        ps, pt, nrm = spectral_quantize(
            x, rot, cb_sem, cb_tail, d_eff, bits_high, bits_low
        )
        mx.eval(ps, pt, nrm)

        b_q = _bench(
            lambda: _sq_quantize_baseline(
                x, rot, cb_sem, cb_tail, d_eff, bits_high, bits_low
            ),
            iters,
        )
        c_q = _bench(
            lambda: spectral_quantize(
                x, rot, cb_sem, cb_tail, d_eff, bits_high, bits_low
            ),
            iters,
        )
        _print_row("quantize", b_q, c_q)

        b_d = _bench(
            lambda: _sq_dequantize_baseline(
                ps, pt, nrm, rot, cb_sem, cb_tail, d_eff, bits_high, bits_low,
                mx.float16,
            ),
            iters,
        )
        c_d = _bench(
            lambda: spectral_dequantize(
                ps, pt, nrm, rot, cb_sem, cb_tail, d_eff, bits_high, bits_low,
                dtype=mx.float16,
            ),
            iters,
        )
        _print_row("dequantize", b_d, c_d)

        per_token_base = (b_q + b_d) * layers
        per_token_comp = (c_q + c_d) * layers
        print(
            f"  → projected at {layers} layers: "
            f"base={per_token_base:7.2f} ms  compiled={per_token_comp:7.2f} ms  "
            f"Δ={per_token_base - per_token_comp:+7.2f} ms"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=32)
    ap.add_argument("--prefill-len", type=int, default=2048)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--layers", type=int, default=32, help="for projection only")
    ap.add_argument("--skip-spectral", action="store_true")
    args = ap.parse_args()

    seq_lens = [1, args.prefill_len]

    for bits in (4, 2):
        bench_turboquant(
            bits, args.head_dim, args.n_heads, seq_lens, args.iters, args.layers
        )

    if not args.skip_spectral:
        for bits_pair in [(4, 2), (8, 1)]:
            bench_spectralquant(
                bits_pair[0],
                bits_pair[1],
                args.head_dim,
                args.n_heads,
                seq_lens,
                args.iters,
                args.layers,
            )


if __name__ == "__main__":
    main()
