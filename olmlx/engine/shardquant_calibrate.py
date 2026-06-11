"""Shard calibration: per-head no-RoPE PCA for K, product-VQ for V (#377).

Reuses the KV-collection scaffolding from spectralquant_calibrate (the
collected chunks each start at sample position 0, which is what makes the
chunk-wise de-rope below exact) and spectral's covariance / eigh / d_eff /
Lloyd-Max helpers. The shard-specific analysis is:

- K: de-rope each chunk -> per-head covariance + eigendecomposition ->
  per-layer rank = max participation ratio over heads -> one pooled
  Lloyd-Max codebook over the kept rotated coefficients.
- V: pooled across heads -> normalize -> fixed orthonormal rotation
  (Hadamard / seeded QR) -> per-position k-means PQ codebooks (256 entries,
  group size 8 // bits).

RoPE frequencies are persisted so runtime re-rope is guaranteed identical
to calibration-time de-rope.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from olmlx.engine.shardquant import (
    detect_rope_spec,
    fit_vq_codebooks,
    make_v_rotation,
    rope_transform,
)
from olmlx.engine.spectralquant import fit_codebook
from olmlx.engine.spectralquant_calibrate import (
    _load_calibration_model,
    collect_kv_vectors,
    compute_covariance,
    eigendecompose,
)

logger = logging.getLogger(__name__)

_SHARD_DEFAULT_MAX_TOKENS_PER_HEAD = 8192
_SHARD_DEFAULT_NUM_SAMPLES = 256

#: Fraction of eigenvalue energy the kept rank must capture.  Replaces the
#: participation ratio, which collapses to ~2 under a single dominant
#: eigenvalue and discards a heavy tail that still carries real signal
#: (measured as a 0.64-vs-0.99 K-cosine gap by scripts/shard_ab_bench.py).
#: 0.999 because truncation — not 4-bit quantization — dominates the
#: reconstruction error on Qwen3-family keys (rank sweep in #377): at 0.99
#: the kept rank lands ~50/128 and cosine plateaus at ~0.93, while the
#: quantizer alone reaches ~0.99.  Lower it per-calibration via --k-energy
#: to trade K quality for bytes.
_SHARD_RANK_ENERGY = 0.999


def _rank_from_eigenvalues(
    eigenvalues: mx.array, energy: float = _SHARD_RANK_ENERGY
) -> int:
    """Smallest rank whose leading eigenvalues capture ``energy``."""
    ev = np.array(eigenvalues.astype(mx.float32))
    total = float(ev.sum())
    if total <= 0.0:
        return len(ev)
    cum = np.cumsum(ev) / total
    return int(np.searchsorted(cum, energy) + 1)


#: layer_idx -> calibration entry
ShardCalibration = dict[int, dict[str, Any]]


def save_shard_calibration(
    calibration: ShardCalibration, meta: dict[str, Any], output_dir: Path
) -> None:
    """Write shard_config.json + calibration.safetensors."""
    import safetensors.numpy

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config: dict[str, Any] = {"meta": meta, "layers": {}}
    tensors: dict[str, np.ndarray] = {}

    for layer, entry in calibration.items():
        prefix = f"layer_{layer}"
        config["layers"][str(layer)] = {
            "k_rank": entry["k_rank"],
            "rope_dims": entry["rope_dims"],
            "rope_traditional": entry["rope_traditional"],
        }
        tensors[f"{prefix}_k_basis"] = np.array(entry["k_basis"])
        tensors[f"{prefix}_k_codebook"] = np.array(entry["k_codebook"])
        tensors[f"{prefix}_v_rotation"] = np.array(entry["v_rotation"])
        tensors[f"{prefix}_v_codebooks"] = np.array(entry["v_codebooks"])
        if entry.get("k_mean") is not None:
            tensors[f"{prefix}_k_mean"] = np.array(entry["k_mean"])
        if entry["rope_freqs"] is not None:
            tensors[f"{prefix}_rope_freqs"] = np.array(entry["rope_freqs"])

    (output_dir / "shard_config.json").write_text(json.dumps(config, indent=2))
    safetensors.numpy.save_file(tensors, str(output_dir / "calibration.safetensors"))


def load_shard_calibration(
    calibration_dir: Path,
) -> tuple[ShardCalibration, dict[str, Any]]:
    """Load shard calibration; returns (per-layer entries, meta)."""
    import safetensors.numpy

    calibration_dir = Path(calibration_dir)
    config = json.loads((calibration_dir / "shard_config.json").read_text())
    tensors = safetensors.numpy.load_file(
        str(calibration_dir / "calibration.safetensors")
    )

    result: ShardCalibration = {}
    for layer_str, layer_cfg in config["layers"].items():
        layer = int(layer_str)
        prefix = f"layer_{layer}"
        freqs_np = tensors.get(f"{prefix}_rope_freqs")
        # Tier-1 artifacts have no k_mean (un-centered pipeline) and a
        # shared 1-D codebook; both load as-is for backward compatibility.
        mean_np = tensors.get(f"{prefix}_k_mean")
        result[layer] = {
            "k_basis": mx.array(tensors[f"{prefix}_k_basis"]),
            "k_rank": layer_cfg["k_rank"],
            "k_codebook": mx.array(tensors[f"{prefix}_k_codebook"]),
            "k_mean": mx.array(mean_np) if mean_np is not None else None,
            "v_rotation": mx.array(tensors[f"{prefix}_v_rotation"]),
            "v_codebooks": mx.array(tensors[f"{prefix}_v_codebooks"]),
            "rope_freqs": mx.array(freqs_np) if freqs_np is not None else None,
            "rope_dims": layer_cfg["rope_dims"],
            "rope_traditional": layer_cfg["rope_traditional"],
        }
    return result, config.get("meta", {})


def _find_attention_module(model: Any, inner: Any, layer_idx: int) -> Any | None:
    """Best-effort lookup of layer layer_idx's attention module."""
    for src in (inner, model):
        layers = getattr(src, "layers", None)
        if layers is None or layer_idx >= len(layers):
            continue
        attn = getattr(layers[layer_idx], "self_attn", None)
        if attn is not None:
            return attn
    return None


def _derope_chunks(chunks: list[mx.array], rope_spec) -> mx.array:
    """De-rope per-sample chunks (each starts at position 0) and concat."""
    if rope_spec is None:
        return mx.concatenate(chunks, axis=0)
    out = []
    for c in chunks:
        # (seq, D): rope_transform works on any (..., S, D)
        out.append(rope_transform(c, rope_spec, 0, inverse=True))
    return mx.concatenate(out, axis=0)


def calibrate_model_shard(
    model_path: str,
    output_dir: Path | None = None,
    num_samples: int = _SHARD_DEFAULT_NUM_SAMPLES,
    calibration_dataset: str | None = None,
    bits: int = 4,
    max_tokens_per_head: int = _SHARD_DEFAULT_MAX_TOKENS_PER_HEAD,
    progress_callback: Any | None = None,
    k_energy: float = _SHARD_RANK_ENERGY,
) -> Path:
    """Run shard calibration on a model. Returns the shard directory."""
    import gc
    import time

    if bits not in (2, 4, 8):
        raise ValueError(f"shard calibration supports bits in {{2,4,8}}, got {bits}")
    if not 0.0 < k_energy <= 1.0:
        raise ValueError(f"k_energy must be in (0, 1], got {k_energy}")
    group_size = 8 // bits

    if output_dir is None:
        output_dir = Path(model_path) / "shard"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if progress_callback:
        progress_callback("Loading model", 0.0)

    model, tokenizer, inner, head_dim, n_kv_heads, num_layers = _load_calibration_model(
        model_path
    )

    if progress_callback:
        progress_callback("Generating calibration data", 0.05)

    from olmlx.engine.flash.prepare import (
        _get_c4_calibration_data,
        _get_calibration_data,
    )

    if calibration_dataset == "synthetic":
        texts = _get_calibration_data(num_samples)
    else:
        texts = _get_c4_calibration_data(num_samples)

    if progress_callback:
        progress_callback("Collecting KV vectors", 0.1)

    kv_collectors = collect_kv_vectors(
        model,
        tokenizer,
        inner,
        num_layers=num_layers,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        texts=texts,
        max_tokens_per_head=max_tokens_per_head,
        progress_callback=progress_callback,
    )

    if progress_callback:
        progress_callback("Analyzing", 0.5)

    if head_dim % group_size != 0:
        raise ValueError(
            f"head_dim={head_dim} not divisible by VQ group size {group_size} "
            f"(bits={bits}); shard quantization unsupported for this model."
        )

    calibration: ShardCalibration = {}
    for layer_idx in range(num_layers):
        head_chunks = kv_collectors[layer_idx]
        if not any(head_chunks[h]["key"] for h in range(n_kv_heads)):
            logger.debug("No KV data for layer %d, skipping", layer_idx)
            continue

        attn = _find_attention_module(model, inner, layer_idx)
        rope_spec = detect_rope_spec(attn) if attn is not None else None
        if rope_spec is None:
            logger.debug(
                "Layer %d: no recognizable RoPE; calibrating in roped basis",
                layer_idx,
            )

        # --- K: per-head no-RoPE PCA -----------------------------------
        # Pipeline (matches shard_compress_keys exactly): unit-normalize,
        # subtract the per-head mean, eigendecompose the centered
        # covariance, keep the smallest rank capturing _SHARD_RANK_ENERGY,
        # fit one Lloyd-Max codebook per head on its own centered
        # coefficients (a shared codebook lets a wide-scale head dominate
        # the fit and crush the others).
        bases = []
        means = []
        ranks = []
        coeff_per_head = []
        for h in range(n_kv_heads):
            chunks = head_chunks[h]["key"]
            if not chunks:
                break
            k_nope = _derope_chunks(chunks, rope_spec).astype(mx.float32)
            norms = mx.maximum(
                mx.sqrt(mx.sum(k_nope * k_nope, axis=-1, keepdims=True)),
                mx.array(1e-8),
            )
            k_unit = k_nope / norms
            mean_h = mx.mean(k_unit, axis=0)
            centered = k_unit - mean_h
            eigenvalues, eigenvectors = eigendecompose(compute_covariance(centered))
            basis = eigenvectors.T  # rows = eigenvectors, descending variance
            bases.append(basis)
            means.append(mean_h)
            ranks.append(_rank_from_eigenvalues(eigenvalues, k_energy))
            coeff_per_head.append(centered @ basis.T)
        if len(bases) < n_kv_heads:
            logger.debug("Layer %d: incomplete per-head K data, skipping", layer_idx)
            continue
        k_rank = max(ranks)
        k_codebook = mx.stack(
            [fit_codebook(c[:, :k_rank].reshape(-1), bits=bits) for c in coeff_per_head]
        )  # (H, 2**bits)

        # --- V: pooled rotation + PQ ------------------------------------
        v_chunks = []
        for h in range(n_kv_heads):
            v_chunks.extend(head_chunks[h]["value"])
        v_data = mx.concatenate(v_chunks, axis=0).astype(mx.float32)
        v_norms = mx.maximum(
            mx.sqrt(mx.sum(v_data * v_data, axis=-1, keepdims=True)),
            mx.array(1e-8),
        )
        v_rotation = make_v_rotation(head_dim, seed=layer_idx)
        v_rotated = (v_data / v_norms) @ v_rotation.T
        v_codebooks = fit_vq_codebooks(
            np.array(v_rotated), group_size=group_size, seed=layer_idx
        )

        calibration[layer_idx] = {
            "k_basis": mx.stack(bases),  # (H, D, D)
            "k_rank": int(k_rank),
            "k_codebook": k_codebook,
            "k_mean": mx.stack(means),  # (H, D)
            "v_rotation": v_rotation,
            "v_codebooks": v_codebooks,
            "rope_freqs": rope_spec.freqs if rope_spec is not None else None,
            "rope_dims": rope_spec.dims if rope_spec is not None else None,
            "rope_traditional": (
                rope_spec.traditional if rope_spec is not None else False
            ),
        }
        if progress_callback:
            frac = 0.5 + (layer_idx + 1) / num_layers * 0.4
            progress_callback(f"Calibrated layer {layer_idx + 1}/{num_layers}", frac)

    del kv_collectors
    gc.collect()
    mx.clear_cache()

    if progress_callback:
        progress_callback("Saving calibration", 0.9)

    meta = {
        "num_layers": num_layers,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "bits": bits,
        "v_group_size": group_size,
        "num_samples": num_samples,
        "max_tokens_per_head": max_tokens_per_head,
        "calibration_dataset": calibration_dataset or "c4",
        "calibrated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    save_shard_calibration(calibration, meta, output_dir)

    if progress_callback:
        progress_callback("Done", 1.0)

    logger.info("Shard calibration complete: %s", output_dir)
    return output_dir
