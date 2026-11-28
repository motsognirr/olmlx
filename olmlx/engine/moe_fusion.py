"""Load-time fusion of MoE expert gate/up projections.

mlx-lm's ``SwitchGLU`` evaluates the routed experts' ``gate_proj`` and
``up_proj`` as two ``gather_qmm`` (or ``gather_mm``) calls over the same input.
Stacking both weight tensors along the output axis (axis 1 of
``[experts, out, in]``) turns them into one call. Every output row is still the
same independent dot product, so the result is bit-identical, not merely close.

On Qwen3.6-35B-A3B this was +2.7% decode at 4-bit and +1.2% at 8-bit on an
M4 Pro. Prefill did not change measurably. The gain comes from running one
larger expert kernel per layer instead of two small ones.

Only the exact ``SwitchGLU`` class is fused, because a subclass may override
``__call__``. Projection pairs that differ in type, quantization or array
layout are left alone. ``unfuse_switch_glu_layers`` reverses the change with
zero-copy views; LoRA adapters need it because they address
``switch_mlp.gate_proj`` / ``up_proj`` by name.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

# model_manager imports this module, and olmlx.engine.inference must still
# import (degraded) without mlx-lm. Without it there is no SwitchGLU to fuse.
try:
    from mlx_lm.models.switch_layers import (
        QuantizedSwitchLinear,
        SwitchGLU,
        SwitchLinear,
    )
except ImportError:  # pragma: no cover - exercised by a subprocess test
    SwitchGLU = None
    _PROJ_TYPES: tuple[type, ...] = ()
else:
    _PROJ_TYPES = (SwitchLinear, QuantizedSwitchLinear)

# Private mlx-lm helpers, imported separately: if a release renames them,
# fall back to copies of the mlx-lm 0.31.3 versions rather than letting the
# ImportError above switch fusion off for every model. Sorting only groups
# routes by expert; it never changes the output.
try:
    from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort
except ImportError:

    def _gather_sort(x, indices):
        *_, M = indices.shape
        indices = indices.flatten()
        order = mx.argsort(indices)
        inv_order = mx.argsort(order)
        return x.flatten(0, -3)[order // M], indices[order], inv_order

    def _scatter_unsort(x, inv_order, shape=None):
        x = x[inv_order]
        if shape is not None:
            x = mx.unflatten(x, 0, shape)
        return x


# Per-instance bookkeeping, excluded when comparing gate/up projections.
_PRIVATE_ATTRS = ("_no_grad", "_training")


class FusedSwitchGLU(nn.Module):
    """``SwitchGLU`` with ``gate_proj`` and ``up_proj`` stacked into one projection.

    ``gate_up_proj`` holds the gate rows first, then the up rows, along the
    output axis. ``__call__`` mirrors ``SwitchGLU.__call__`` (mlx-lm) step for
    step, with the two expert matmuls replaced by one plus a split.
    """

    def __init__(self, gate_up_proj, down_proj, activation, hidden_dims: int):
        super().__init__()
        self.gate_up_proj = gate_up_proj
        self.down_proj = down_proj
        self.activation = activation
        self.hidden_dims = hidden_dims

    def __call__(self, x, indices) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))

        # Same threshold as SwitchGLU: sort routes by expert for long inputs.
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        if self.training:
            idx = mx.stop_gradient(idx)
        x_gate, x_up = mx.split(
            self.gate_up_proj(x, idx, sorted_indices=do_sort),
            [self.hidden_dims],
            axis=-1,
        )
        x = self.down_proj(
            self.activation(x_up, x_gate),
            idx,
            sorted_indices=do_sort,
        )

        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)

        return x.squeeze(-2)


def _scalar_attrs(proj: nn.Module) -> dict[str, Any]:
    return {k: v for k, v in proj.__dict__.items() if k not in _PRIVATE_ATTRS}


def _out_dims(proj: nn.Module) -> int | None:
    w = proj.get("weight")
    return w.shape[1] if isinstance(w, mx.array) and w.ndim == 3 else None


def _fusable(glu: Any) -> bool:
    """True when *glu* is a plain ``SwitchGLU`` whose gate/up can be stacked."""
    if SwitchGLU is None or type(glu) is not SwitchGLU:
        return False
    gate, up = glu.get("gate_proj"), glu.get("up_proj")
    if gate is None or up is None:
        return False
    if type(gate) is not type(up) or type(gate) not in _PROJ_TYPES:
        return False
    out = _out_dims(gate)
    if out is None or out != _out_dims(up):
        return False
    if _scalar_attrs(gate) != _scalar_attrs(up):
        return False  # bits / group_size / mode / ... differ
    if set(gate.keys()) != set(up.keys()):
        return False
    for key, g in gate.items():
        u = up[key]
        # Every array must carry the output dim on axis 1 (weight, scales,
        # biases, bias); anything else is an unknown layout, so don't guess.
        if not (isinstance(g, mx.array) and isinstance(u, mx.array)):
            return False
        if g.ndim < 2 or g.dtype != u.dtype or g.shape[1] != out:
            return False
        if g.shape[:1] + g.shape[2:] != u.shape[:1] + u.shape[2:]:
            return False
    return True


def _new_proj_like(src: nn.Module, arrays: dict[str, mx.array]) -> nn.Module:
    """A copy of projection *src* holding *arrays* instead of its own arrays.

    Uses ``structural_copy`` so there is one definition of which per-module
    state is shared and which is copied (#362).
    """
    # Lazy: loaded_model sits above this module in model_manager's imports.
    from olmlx.engine.loaded_model import structural_copy

    proj = structural_copy(src)
    for k, v in arrays.items():
        proj[k] = v
    return proj


def _fuse(glu: nn.Module) -> FusedSwitchGLU:
    gate, up = glu.gate_proj, glu.up_proj
    arrays = {k: mx.concatenate([gate[k], up[k]], axis=1) for k in gate.keys()}
    # Materialize on this (load) thread: a lazy concatenate stays bound to the
    # thread's stream and crashes the first forward on the generation worker.
    mx.eval(arrays)
    fused = FusedSwitchGLU(
        _new_proj_like(gate, arrays),
        glu.down_proj,
        glu.activation,
        gate["weight"].shape[1],
    )
    fused.train(glu.training)
    return fused


def _unfuse(fused: FusedSwitchGLU) -> nn.Module:
    # A FusedSwitchGLU only exists if mlx-lm imported, so SwitchGLU is set.
    assert SwitchGLU is not None
    gu = fused.gate_up_proj
    h = fused.hidden_dims
    # Basic slices are views of the fused buffers, so restoring the separate
    # projections allocates nothing; gather_qmm accepts the batch strides.
    gate_arrays = {k: v[:, :h] for k, v in gu.items()}
    up_arrays = {k: v[:, h:] for k, v in gu.items()}
    mx.eval(gate_arrays, up_arrays)
    glu = SwitchGLU.__new__(SwitchGLU)
    nn.Module.__init__(glu)
    # Item assignment is what Module.__setattr__ does for module values.
    glu["gate_proj"] = _new_proj_like(gu, gate_arrays)
    glu["up_proj"] = _new_proj_like(gu, up_arrays)
    glu["down_proj"] = fused.down_proj
    glu.activation = fused.activation
    glu.train(fused.training)
    return glu


def _replace_children(model: Any, match, convert, after_each=None) -> int:
    if not isinstance(model, nn.Module):
        return 0
    # Record (parent, key) slots and keep no reference to the children:
    # named_modules() lists every module, so iterating it while replacing
    # would keep each original block alive until the end of the loop.
    slots = [
        (module, key)
        for _, module in model.named_modules()
        for key, child in module.items()
        if match(child)
    ]
    for module, key in slots:
        setattr(module, key, convert(module[key]))
        if after_each is not None:
            after_each()
    return len(slots)


def fuse_switch_glu_layers(model: Any, *, clear_cache=mx.clear_cache) -> int:
    """Fuse every eligible ``SwitchGLU`` in *model* in place; return how many.

    Run on the thread that loaded the model, after all weight transforms.
    Blocks are fused one at a time and each original is freed before the
    next, so peak memory grows by about one block's gate+up weights.

    *clear_cache* runs after each block. The freed originals go back to MLX's
    buffer cache, can't serve the next block's twice-as-wide buffers, and
    would otherwise stay there, counted by ``get_metal_memory()`` (active +
    cache). ``mx.clear_cache`` is global, so the model manager passes a
    version serialized with its other Metal cache clears.
    """
    return _replace_children(model, _fusable, _fuse, after_each=clear_cache)


def unfuse_switch_glu_layers(model: Any) -> int:
    """Restore separate ``gate_proj``/``up_proj`` as views; return how many."""
    return _replace_children(model, lambda m: isinstance(m, FusedSwitchGLU), _unfuse)
