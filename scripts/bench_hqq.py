"""Benchmark HQQ vs MLX native quantization.

Compares memory usage and reconstruction quality between HQQ-4 and
MLX native int4 affine quantization.

Usage:
    uv run python scripts/bench_hqq.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import mlx.core as mx
from mlx.core import random

from olmlx.engine.hqq.quantize import HQQConfig, hqq_quantize_weight


@dataclass
class Result:
    label: str
    mae: float
    elapsed_ms: float
    compression_ratio: float


def bench_one(
    w: mx.array,
    bits: int,
    group_size: int,
    label: str,
    n_iters: int | None = None,
) -> Result:
    """Benchmark one quantization config."""
    t0 = time.perf_counter()

    if n_iters is not None:
        # HQQ path
        cfg = HQQConfig(bits=bits, group_size=group_size, n_iters=n_iters)
        packed, scales, biases = hqq_quantize_weight(w, cfg)
        restored = mx.dequantize(packed, scales, biases, group_size, bits)
    else:
        # MLX native path
        packed, scales, biases = mx.quantize(w, group_size=group_size, bits=bits)
        restored = mx.dequantize(packed, scales, biases, group_size, bits)

    mx.eval(restored)
    elapsed = (time.perf_counter() - t0) * 1000

    mae = mx.mean(mx.abs(w - restored)).item()

    original_bytes = w.size * w.itemsize
    quantized_bytes = (
        packed.size * 4 + scales.size * scales.itemsize + biases.size * biases.itemsize
    )
    ratio = original_bytes / quantized_bytes

    return Result(label=label, mae=mae, elapsed_ms=elapsed, compression_ratio=ratio)


def main() -> None:
    random.seed(42)

    shapes = [
        (4096, 4096),    # large FFN layer
        (4096, 1024),    # attention projection
        (8192, 2048),    # MoE expert
    ]

    print("HQQ vs MLX Native Quantization Benchmark")
    print("=" * 70)
    print(f"{'Shape':>20}  {'Method':>12}  {'MAE':>10}  {'Time':>10}  {'Ratio':>8}")
    print("-" * 70)

    for shape in shapes:
        w = random.normal(shape=shape) * 0.02
        mx.eval(w)

        for bits, gs in [(4, 64), (8, 128)]:
            # HQQ
            hqq_result = bench_one(w, bits, gs, f"HQQ-{bits}bit", n_iters=3)
            # MLX native
            mx_result = bench_one(w, bits, gs, f"MLX-{bits}bit", n_iters=None)

            shape_str = f"{shape[0]}x{shape[1]}"
            print(
                f"{shape_str:>20}  "
                f"{hqq_result.label:>12}  "
                f"{hqq_result.mae:10.6f}  "
                f"{hqq_result.elapsed_ms:8.1f}ms  "
                f"{hqq_result.compression_ratio:6.1f}x"
            )
            print(
                f"{'':>20}  "
                f"{mx_result.label:>12}  "
                f"{mx_result.mae:10.6f}  "
                f"{mx_result.elapsed_ms:8.1f}ms  "
                f"{mx_result.compression_ratio:6.1f}x"
            )
            print()

    print("=" * 70)
    print("MAE = Mean Absolute Error (lower is better)")
    print("Ratio = Original bytes / Quantized bytes")
    print("HQQ cost includes 3 half-quadratic iterations per group.")


if __name__ == "__main__":
    main()
