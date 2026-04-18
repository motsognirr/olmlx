"""SpectralQuant calibration: eigenspectral analysis of KV cache vectors.

Collects key/value vectors during a calibration pass, computes per-head
covariance matrices, eigendecomposes them, and derives the spectral
rotation matrices and non-uniform codebooks needed for SpectralQuant
compression.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from olmlx.engine.spectralquant import allocate_bits, fit_codebook

logger = logging.getLogger(__name__)


def _resolve_config_holder(inner: Any, model: Any) -> Any:
    # Some architectures (e.g. Qwen3Next) expose `.args` only on the top-level
    # model, not on the backbone returned by `_get_backbone`.
    return inner if hasattr(inner, "args") else model


def _is_attention_cache_state(state: Any) -> bool:
    # Hybrid models (e.g. Qwen3Next) mix attention caches with SSM caches.
    # Only the attention layers hold 4D KV tensors we can calibrate on.
    # Note: this is a shape-based heuristic keyed on state[0]. For Qwen3Next,
    # SSM entries expose the 3D conv state at index 0, so they are correctly
    # rejected; a future hybrid architecture whose non-attention cache puts a
    # 4D tensor first would bypass this guard.
    if not state or len(state) < 2:
        return False
    keys = state[0]
    return hasattr(keys, "ndim") and keys.ndim == 4


def _resolve_cache_owner(inner: Any, model: Any) -> Any:
    # `make_prompt_cache` defers to `make_cache()` on the passed object. For
    # hybrid models (e.g. Qwen3Next SSM+attention), only the top-level model
    # exposes `make_cache` with the correct per-layer cache types — the bare
    # backbone would yield uniform KVCaches and break SSM layers.
    if hasattr(model, "make_cache"):
        return model
    return inner


def compute_covariance(data: mx.array) -> mx.array:
    """Compute centered sample covariance matrix.

    Args:
        data: (N, D) matrix of vectors.

    Returns:
        (D, D) covariance matrix in float32.
    """
    data = data.astype(mx.float32)
    n = data.shape[0]
    mean = mx.mean(data, axis=0, keepdims=True)
    data_c = data - mean
    return (data_c.T @ data_c) / n


def eigendecompose(cov: mx.array) -> tuple[mx.array, mx.array]:
    """Eigendecompose a symmetric covariance matrix.

    Args:
        cov: (D, D) symmetric matrix.

    Returns:
        (eigenvalues, eigenvectors) sorted descending by eigenvalue.
        eigenvalues: (D,), eigenvectors: (D, D) — columns are eigenvectors.
    """
    eigenvalues, eigenvectors = mx.linalg.eigh(cov, stream=mx.cpu)
    mx.eval(eigenvalues, eigenvectors)

    # eigh returns ascending order — reverse to descending
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]

    # Clamp negative eigenvalues (numerical noise)
    eigenvalues = mx.maximum(eigenvalues, mx.array(0.0))

    return eigenvalues, eigenvectors


def compute_d_eff(eigenvalues: mx.array) -> int:
    """Compute effective dimensionality via participation ratio.

    d_eff = (sum(lambda))^2 / sum(lambda^2)

    This measures how many dimensions carry significant signal.

    Args:
        eigenvalues: Sorted eigenvalues (descending).

    Returns:
        Effective dimensionality (integer, at least 1).
    """
    ev = eigenvalues.astype(mx.float32)
    total = mx.sum(ev)
    total_sq = mx.sum(ev * ev)

    if float(total_sq) < 1e-12:
        return len(eigenvalues)

    d_eff = float(total * total / total_sq)
    # Round and clamp to [1, dim]
    d_eff = max(1, min(round(d_eff), len(eigenvalues)))
    return d_eff


def calibrate_head(
    kv_data: mx.array,
    avg_bits: int = 4,
) -> dict[str, Any]:
    """Calibrate spectral quant for a single attention head.

    Args:
        kv_data: (N, head_dim) key or value vectors from calibration.
        avg_bits: Target average bits per dimension.

    Returns:
        Dict with keys: eigenvectors, d_eff, codebook_sem, codebook_tail,
        bits_high, bits_low.
    """
    head_dim = kv_data.shape[-1]

    # Step 1: Covariance and eigendecomposition
    cov = compute_covariance(kv_data)
    eigenvalues, eigenvectors = eigendecompose(cov)

    # Step 2: Effective dimensionality
    d_eff = compute_d_eff(eigenvalues)

    # Step 3: Bit allocation
    bits_high, bits_low = allocate_bits(d_eff, head_dim, avg_bits)

    # Step 4: Normalize data (matching spectral_quantize which normalizes to
    # unit sphere before rotating), then rotate into spectral basis.
    data_f32 = kv_data.astype(mx.float32)
    norms = mx.sqrt(mx.sum(data_f32**2, axis=-1, keepdims=True))
    data_norm = data_f32 / mx.maximum(norms, mx.array(1e-8))
    rotated = data_norm @ eigenvectors.T

    sem_data = rotated[:, :d_eff].reshape(-1)
    tail_data = rotated[:, d_eff:].reshape(-1)

    codebook_sem = fit_codebook(sem_data, bits=bits_high)
    codebook_tail = (
        fit_codebook(tail_data, bits=bits_low) if d_eff < head_dim else mx.array([0.0])
    )

    return {
        "eigenvectors": eigenvectors,  # (head_dim, head_dim), columns = eigvecs
        "d_eff": d_eff,
        "codebook_sem": codebook_sem,
        "codebook_tail": codebook_tail,
        "bits_high": bits_high,
        "bits_low": bits_low,
    }


# ---------------------------------------------------------------------------
# Calibration data persistence
# ---------------------------------------------------------------------------

# Key format: (layer_idx, head_idx, "key"|"value")
CalibrationKey = tuple[int, int, str]
CalibrationData = dict[CalibrationKey, dict[str, Any]]


def save_calibration(calibration: CalibrationData, output_dir: Path) -> None:
    """Save calibration data to disk.

    Writes:
      - spectral_config.json: metadata (d_eff, bit allocations per head)
      - calibration.safetensors: eigenvectors + codebooks
    """
    import safetensors.numpy

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config: dict[str, Any] = {"heads": {}}
    tensors: dict[str, np.ndarray] = {}

    for (layer, head, kind), data in calibration.items():
        prefix = f"layer_{layer}_head_{head}_{kind}"
        config["heads"][prefix] = {
            "d_eff": data["d_eff"],
            "bits_high": data["bits_high"],
            "bits_low": data["bits_low"],
        }
        tensors[f"{prefix}_eigvecs"] = np.array(data["eigenvectors"])
        tensors[f"{prefix}_codebook_sem"] = np.array(data["codebook_sem"])
        tensors[f"{prefix}_codebook_tail"] = np.array(data["codebook_tail"])

    (output_dir / "spectral_config.json").write_text(json.dumps(config, indent=2))
    safetensors.numpy.save_file(tensors, str(output_dir / "calibration.safetensors"))


def load_calibration(calibration_dir: Path) -> CalibrationData:
    """Load calibration data from disk.

    Returns:
        Dict mapping (layer, head, kind) → calibration result.
    """
    import safetensors.numpy

    calibration_dir = Path(calibration_dir)
    config = json.loads((calibration_dir / "spectral_config.json").read_text())
    tensors = safetensors.numpy.load_file(
        str(calibration_dir / "calibration.safetensors")
    )

    result: CalibrationData = {}
    for prefix, meta in config["heads"].items():
        # Parse prefix: "layer_{i}_head_{j}_{kind}"
        parts = prefix.split("_")
        layer = int(parts[1])
        head = int(parts[3])
        kind = parts[4]  # "key" or "value"

        result[(layer, head, kind)] = {
            "eigenvectors": mx.array(tensors[f"{prefix}_eigvecs"]),
            "d_eff": meta["d_eff"],
            "codebook_sem": mx.array(tensors[f"{prefix}_codebook_sem"]),
            "codebook_tail": mx.array(tensors[f"{prefix}_codebook_tail"]),
            "bits_high": meta["bits_high"],
            "bits_low": meta["bits_low"],
        }

    return result


# ---------------------------------------------------------------------------
# Full model calibration pipeline
# ---------------------------------------------------------------------------


def calibrate_model(
    model_path: str,
    output_dir: Path | None = None,
    num_samples: int = 256,
    calibration_dataset: str | None = None,
    avg_bits: int = 4,
    max_tokens_per_head: int = 8192,
    progress_callback: Any | None = None,
) -> Path:
    """Run spectral calibration on a model.

    Loads the model, runs calibration text through it to collect K/V vectors
    from attention layers, then performs eigenspectral analysis per head.

    Args:
        model_path: HF model path or local directory.
        output_dir: Where to write calibration files. Defaults to model_dir/spectral.
        num_samples: Number of calibration text samples.
        calibration_dataset: "c4", "synthetic", or None (defaults to c4).
        avg_bits: Target average bits per dimension.
        max_tokens_per_head: Max tokens to collect per head for covariance.
        progress_callback: Called with (description, fraction).

    Returns:
        Path to the spectral calibration directory.
    """
    import gc
    import time

    from olmlx.engine.flash.prepare import (
        _get_backbone,
        _get_c4_calibration_data,
        _get_calibration_data,
        _encode_tokens,
        load_model_with_strict_fallback,
    )
    from olmlx.engine.turboquant_cache import _detect_head_dim

    if output_dir is None:
        output_dir = Path(model_path) / "spectral"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if progress_callback:
        progress_callback("Loading model", 0.0)

    # Load model
    try:
        model, tokenizer = load_model_with_strict_fallback(model_path, lazy=False)
    except ValueError:
        import mlx_vlm

        model, processor = mlx_vlm.load(model_path, lazy=False)
        tokenizer = (
            processor.tokenizer if hasattr(processor, "tokenizer") else processor
        )

    inner = _get_backbone(model)
    layers = inner.layers
    num_layers = len(layers)
    cfg_holder = _resolve_config_holder(inner, model)
    head_dim = _detect_head_dim(cfg_holder)

    # Determine number of KV heads
    n_kv_heads = getattr(cfg_holder.args, "num_key_value_heads", None)
    if n_kv_heads is None:
        n_kv_heads = getattr(cfg_holder.args, "num_attention_heads", 1)

    if progress_callback:
        progress_callback("Generating calibration data", 0.05)

    # Generate calibration data
    if calibration_dataset == "synthetic":
        texts = _get_calibration_data(num_samples)
    else:
        texts = _get_c4_calibration_data(num_samples)

    if progress_callback:
        progress_callback("Collecting KV vectors", 0.1)

    # Collect K/V vectors per (layer, head)
    # kv_collectors[layer][head]["key"|"value"] = list of mx.array chunks
    kv_collectors: dict[int, dict[int, dict[str, list[mx.array]]]] = {}
    for i in range(num_layers):
        kv_collectors[i] = {}
        for h in range(n_kv_heads):
            kv_collectors[i][h] = {"key": [], "value": []}

    tokens_collected = {
        (i, h, k): 0
        for i in range(num_layers)
        for h in range(n_kv_heads)
        for k in ("key", "value")
    }

    # Collect post-RoPE K/V vectors by running each sample with a fresh
    # KV cache, then extracting the cached tensors.  This captures keys and
    # values *after* rotary positional embeddings — the actual distribution
    # that gets stored in the KV cache at inference time.
    from mlx_lm.models.cache import make_prompt_cache

    cache_model = _resolve_cache_owner(inner, model)
    first_exc: Exception | None = None
    for sample_idx, text in enumerate(texts):
        tokens = _encode_tokens(tokenizer, text)
        if len(tokens) > 512:
            tokens = tokens[:512]
        input_ids = mx.array([tokens])

        # Create a fresh cache and run the forward pass
        prompt_cache = make_prompt_cache(cache_model)
        try:
            model(input_ids, cache=prompt_cache)
        except Exception as exc:
            if first_exc is None:
                first_exc = exc
            logger.debug("Skipping sample %d: %s", sample_idx, exc)
            continue
        mx.eval([c.state for c in prompt_cache if hasattr(c, "state")])

        # Extract K/V from each layer's cache
        for layer_idx in range(min(num_layers, len(prompt_cache))):
            cache_entry = prompt_cache[layer_idx]
            state = cache_entry.state if hasattr(cache_entry, "state") else None
            if not _is_attention_cache_state(state):
                continue
            # KVCache.state returns [keys, values] with shape
            # (1, n_kv_heads, seq_len, head_dim)
            cached_keys = state[0]  # (1, n_kv_heads, seq, head_dim)
            cached_values = state[1]  # (1, n_kv_heads, seq, head_dim)

            for h in range(min(n_kv_heads, cached_keys.shape[1])):
                if tokens_collected[(layer_idx, h, "key")] < max_tokens_per_head:
                    k_h = cached_keys[0, h, :, :]  # (seq, head_dim)
                    remaining = (
                        max_tokens_per_head - tokens_collected[(layer_idx, h, "key")]
                    )
                    k_h = k_h[:remaining]
                    kv_collectors[layer_idx][h]["key"].append(k_h)
                    tokens_collected[(layer_idx, h, "key")] += k_h.shape[0]

                if tokens_collected[(layer_idx, h, "value")] < max_tokens_per_head:
                    v_h = cached_values[0, h, :, :]  # (seq, head_dim)
                    remaining = (
                        max_tokens_per_head - tokens_collected[(layer_idx, h, "value")]
                    )
                    v_h = v_h[:remaining]
                    kv_collectors[layer_idx][h]["value"].append(v_h)
                    tokens_collected[(layer_idx, h, "value")] += v_h.shape[0]

        del prompt_cache
        if progress_callback:
            frac = 0.1 + (sample_idx + 1) / len(texts) * 0.4
            progress_callback(f"Collected {sample_idx + 1}/{len(texts)} samples", frac)

    # Guard: if no KV vectors were collected, fail early with a clear message
    total_collected = sum(tokens_collected.values())
    if total_collected == 0:
        detail = (
            f" First forward-pass error: {type(first_exc).__name__}: {first_exc}"
            if first_exc
            else ""
        )
        raise RuntimeError(
            "No KV vectors were collected during calibration. "
            "All forward passes failed — check that the model loads correctly." + detail
        )

    if progress_callback:
        progress_callback("Running eigenspectral analysis", 0.5)

    # Calibrate per layer by aggregating all heads.
    # The KV cache operates on all heads simultaneously (shape: B, n_heads,
    # seq, head_dim), so a single rotation per layer is applied to every head.
    # Aggregating across heads produces a rotation that captures the shared
    # eigenstructure rather than being tuned to one head's statistics.
    calibration: CalibrationData = {}
    total_items = num_layers * 2  # key + value per layer
    done = 0

    for layer_idx in range(num_layers):
        for kind in ("key", "value"):
            # Concatenate all heads' data for this layer+kind
            all_chunks = []
            for head_idx in range(n_kv_heads):
                all_chunks.extend(kv_collectors[layer_idx][head_idx][kind])
            if not all_chunks:
                logger.warning(
                    "No KV data for layer %d %s, skipping",
                    layer_idx,
                    kind,
                )
                continue

            kv_data = mx.concatenate(all_chunks, axis=0)
            result = calibrate_head(kv_data, avg_bits=avg_bits)
            calibration[(layer_idx, 0, kind)] = result

            done += 1
            if progress_callback:
                frac = 0.5 + done / total_items * 0.4
                progress_callback(f"Calibrated {done}/{total_items} layer-kinds", frac)

    # Free collectors
    del kv_collectors
    gc.collect()
    mx.clear_cache()

    if progress_callback:
        progress_callback("Saving calibration", 0.9)

    # Save
    save_calibration(calibration, output_dir)

    # Write additional metadata
    meta = {
        "num_layers": num_layers,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "avg_bits": avg_bits,
        "num_samples": num_samples,
        "max_tokens_per_head": max_tokens_per_head,
        "calibration_dataset": calibration_dataset or "c4",
        "calibrated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    # Merge into spectral_config.json
    config_path = output_dir / "spectral_config.json"
    config = json.loads(config_path.read_text())
    config["meta"] = meta
    config_path.write_text(json.dumps(config, indent=2))

    if progress_callback:
        progress_callback("Done", 1.0)

    logger.info("Spectral calibration complete: %s", output_dir)
    return output_dir
