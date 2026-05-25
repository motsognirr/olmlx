"""HQQ (Half-Quadratic Quantization) for MLX weights.

Data-free quantization: no calibration set needed. Solves a half-quadratic
optimisation per weight matrix to find better scale/bias parameters than
naive min/max affine quantisation (which is what ``mx.quantize`` does).
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import logging

logger = logging.getLogger(__name__)


@dataclass
class HQQConfig:
    """HQQ quantization configuration.

    Attributes:
        bits: Quantization bit width (4 or 8).
        group_size: Group size along the last dimension.
        n_iters: Number of half-quadratic iterations.
        skip_patterns: Module name patterns to skip during ``quantize_model``.
    """

    bits: int
    group_size: int | None = None
    n_iters: int = 3
    skip_patterns: tuple[str, ...] = ("lm_head", "embed_tokens")

    def __post_init__(self) -> None:
        if self.group_size is None:
            self.group_size = 64 if self.bits == 4 else 128

    @classmethod
    def from_string(cls, s: str | None) -> HQQConfig | None:
        """Parse a config string like ``"hqq:4"`` or ``"hqq:4:128"``."""
        if s is None:
            return None
        parts = s.split(":")
        bits = int(parts[1])
        group_size = None
        if len(parts) >= 3:
            group_size = int(parts[2])
        return cls(bits=bits, group_size=group_size)


def _hqq_solve_group(
    w_grp: mx.array,
    n_iters: int,
    n_levels: int,
) -> tuple[mx.array, mx.array, mx.array]:
    """Run HQQ optimization on a weight group.

    w_grp: shape [..., group_size] — float weights for one group.
    Returns (quantized_indices uint32, scale float, bias float) per group.
    """
    w_min = w_grp.min(axis=-1, keepdims=True)
    w_max = w_grp.max(axis=-1, keepdims=True)
    beta = w_min
    delta = w_max - w_min
    eps = mx.array(1e-9, dtype=w_grp.dtype)
    scale = mx.maximum(delta, eps) / mx.array(float(n_levels), dtype=w_grp.dtype)

    for _ in range(n_iters):
        q = mx.round((w_grp - beta) / mx.maximum(scale, eps))
        q = mx.clip(q, 0, n_levels)
        q_mean = q.mean(axis=-1, keepdims=True)
        w_mean = w_grp.mean(axis=-1, keepdims=True)
        qc = q - q_mean
        wc = w_grp - w_mean
        cov = (qc * wc).sum(axis=-1, keepdims=True)
        var = (qc * qc).sum(axis=-1, keepdims=True)
        scale = cov / mx.maximum(var, eps)
        beta = w_mean - scale * q_mean

    q = mx.round((w_grp - beta) / mx.maximum(scale, eps))
    q = mx.clip(q, 0, n_levels)
    q = q.astype(mx.uint32)
    return q, scale.astype(w_grp.dtype), beta.astype(w_grp.dtype)


def _pack_indices(q: mx.array, bits: int) -> mx.array:
    """Pack uint32 indices into bit-packed uint32 along the last axis.

    q: shape [..., K] where K is divisible by (32 // bits).
    Returns: shape [..., K // (32 // bits)] uint32.
    """
    elems_per_pack = 32 // bits
    last_dim = q.shape[-1]
    q_shaped = q.reshape(*q.shape[:-1], last_dim // elems_per_pack, elems_per_pack)
    packed = mx.zeros(q_shaped.shape[:-1], dtype=mx.uint32)
    for j in range(elems_per_pack):
        shift = mx.array(j * bits, dtype=mx.uint32)
        packed = packed | (q_shaped[..., j] << shift)
    return packed.reshape(*q.shape[:-1], last_dim // elems_per_pack)


class HQQLinear(nn.Module):
    """A drop-in replacement for ``nn.Linear`` with HQQ-quantized weights.

    Dequantizes on-the-fly during the forward pass using MLX's native
    ``mx.dequantize``, which is fast on Apple Silicon.
    """

    def __init__(
        self,
        weight: mx.array,
        scales: mx.array,
        biases: mx.array,
        bias: mx.array | None,
        group_size: int,
        bits: int,
    ):
        super().__init__()
        self._quant_weight = weight
        self._scales = scales
        self._biases = biases
        if bias is not None:
            self._bias = bias
        self._group_size = group_size
        self._bits = bits

    def __call__(self, x: mx.array) -> mx.array:
        w = mx.dequantize(
            self._quant_weight,
            self._scales,
            self._biases,
            self._group_size,
            self._bits,
        )
        y = x @ w.T
        if "_bias" in self:
            y = y + self._bias
        return y


def hqq_quantize_weight(
    w: mx.array,
    cfg: HQQConfig,
) -> tuple[mx.array, mx.array, mx.array]:
    """Quantize a 2D weight matrix using HQQ.

    Args:
        w: Weight matrix of shape ``[out_features, in_features]``.
        cfg: HQQ configuration.

    Returns:
        ``(packed, scales, biases)`` in MLX's affine quantized format.
        ``packed`` has the same shape as the output of ``mx.quantize``
        (packed uint32), ``scales`` and ``biases`` have shape
        ``[out_features, in_features // group_size]``.
    """
    bits = cfg.bits
    group_size = cfg.group_size
    n_levels = (1 << bits) - 1
    out_features, in_features = w.shape
    n_groups = in_features // group_size

    if n_groups == 0:
        raise ValueError(
            f"Cannot quantize weight of shape ({out_features}, {in_features}) "
            f"with group_size={group_size}: in_features must be >= group_size"
        )

    w_grp = w.reshape(out_features, n_groups, group_size)
    q, scales, biases = _hqq_solve_group(w_grp, cfg.n_iters, n_levels)

    q = q.reshape(out_features, in_features)
    packed = _pack_indices(q, bits)
    scales = scales.reshape(out_features, n_groups)
    biases = biases.reshape(out_features, n_groups)

    return packed, scales, biases


def hqq_quantize_linear(
    layer: nn.Linear,
    cfg: HQQConfig,
) -> HQQLinear:
    """Convert an ``nn.Linear`` layer to an ``HQQLinear`` with quantized weights.

    Returns the new ``HQQLinear`` module. The caller should replace the
    ``nn.Linear`` instance with the returned module in its parent container.
    """
    w = layer.weight
    bias = layer.bias if "bias" in layer else None
    packed, scales, q_biases = hqq_quantize_weight(w, cfg)
    return HQQLinear(
        weight=packed,
        scales=scales,
        biases=q_biases,
        bias=bias,
        group_size=cfg.group_size,
        bits=cfg.bits,
    )


def _replace_module(
    parent: nn.Module,
    name: str,
    new_module: nn.Module,
) -> None:
    """Replace a child module in *parent* by full dotted *name*."""
    parts = name.split(".")
    obj = parent
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], new_module)


def quantize_model(
    model: nn.Module,
    cfg: HQQConfig,
) -> None:
    """Walk model modules and replace all ``nn.Linear`` with ``HQQLinear``.

    Skips modules whose name contains any of ``cfg.skip_patterns``
    (e.g. ``lm_head``, ``embed_tokens``) and layers whose input
    dimension is smaller than ``cfg.group_size``.
    """
    replacements: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(p in name for p in cfg.skip_patterns):
                continue
            if module.weight.shape[1] < cfg.group_size:
                logger.debug(
                    "Skipping %s: in_features=%d < group_size=%d",
                    name,
                    module.weight.shape[1],
                    cfg.group_size,
                )
                continue
            replacements.append((name, module))

    for name, linear in replacements:
        hqq_layer = hqq_quantize_linear(linear, cfg)
        _replace_module(model, name, hqq_layer)
