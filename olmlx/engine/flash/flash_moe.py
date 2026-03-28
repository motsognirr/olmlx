"""Flash-aware MoE module for SSD-based expert offloading.

FlashMoE replaces DeepseekV3MoE's SwitchGLU with on-demand expert loading
from SSD. The router (gate) and shared experts stay in RAM; only the
routed expert weights are loaded per-token from the FlashMoeWeightStore.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from olmlx.engine.flash.moe_weight_store import FlashMoeWeightStore


class FlashMoE(nn.Module):
    """Drop-in replacement for MoE expert dispatch that loads weights from SSD.

    Takes pre-computed router outputs (inds, scores) and dispatches to
    experts loaded on-demand from FlashMoeWeightStore.
    """

    def __init__(
        self,
        layer_idx: int,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        weight_store: FlashMoeWeightStore,
        activation: nn.Module | None = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.weight_store = weight_store
        self._activation = activation

    def _apply_gated_activation(self, up_out: mx.array, gate_out: mx.array) -> mx.array:
        """Apply gated activation (SwiGLU). Activation is a 2-arg callable."""
        if self._activation is not None:
            return self._activation(up_out, gate_out)
        return nn.silu(gate_out) * up_out

    def _apply_ungated_activation(self, x: mx.array) -> mx.array:
        """Apply non-gated activation (e.g. relu2). Activation is a 1-arg callable."""
        if self._activation is None:
            raise ValueError(
                "FlashMoE: non-gated (fc1/fc2) experts require an explicit activation "
                "function — pass activation=relu2 (or your model's activation) to FlashMoE.__init__."
            )
        return self._activation(x)

    def __call__(
        self,
        x: mx.array,
        inds: mx.array,
        scores: mx.array,
    ) -> mx.array:
        """Sparse MoE forward pass with SSD-loaded experts.

        Args:
            x: Input hidden states, shape (B, L, hidden_size).
            inds: Router-selected expert indices, shape (B, L, K).
            scores: Router scores for selected experts, shape (B, L, K).

        Returns:
            Output hidden states, shape (B, L, hidden_size).
        """
        orig_shape = x.shape
        B, L, H = orig_shape
        K = inds.shape[-1]

        # Collect unique expert indices across entire batch
        mx.eval(inds)
        flat_inds = inds.reshape(-1).tolist()
        unique_experts = sorted(set(flat_inds))

        # Load only needed experts from SSD
        loaded = self.weight_store.load_experts(self.layer_idx, unique_experts)
        idx_map = loaded.expert_index_map  # global -> local

        # Remap global indices to local positions in stacked arrays
        remap = mx.array(
            [idx_map[int(i)] for i in flat_inds],
            dtype=mx.uint32,
        ).reshape(B, L, K)

        # Compute expert outputs using loaded weights.
        # Gated (gate_proj+up_proj+down_proj): gate_out, up_out → silu(gate)*up → down
        # Non-gated (fc1+fc2): up_out → activation → down
        #
        # We use gather_mm for efficient batched expert dispatch.

        # Expand x for gather_mm: (B, L, 1, 1, H)
        x_expanded = mx.expand_dims(x, (-2, -3))
        gated = loaded.gate_weight is not None

        if loaded.is_quantized:
            qmm_kwargs = dict(
                transpose=True,
                group_size=loaded.group_size,
                bits=loaded.bits,
                mode=loaded.quant_mode,
            )
            if gated:
                gate_out = mx.gather_qmm(
                    x_expanded,
                    loaded.gate_weight,
                    loaded.gate_scales,
                    loaded.gate_biases,
                    rhs_indices=remap,
                    **qmm_kwargs,
                )
                if loaded.gate_bias is not None:
                    gate_out = gate_out + loaded.gate_bias[remap][..., None, :]
            up_out = mx.gather_qmm(
                x_expanded,
                loaded.up_weight,
                loaded.up_scales,
                loaded.up_biases,
                rhs_indices=remap,
                **qmm_kwargs,
            )
            if loaded.up_bias is not None:
                up_out = up_out + loaded.up_bias[remap][..., None, :]
            activated = (
                self._apply_gated_activation(up_out, gate_out)
                if gated
                else self._apply_ungated_activation(up_out)
            )
            expert_out = mx.gather_qmm(
                activated,
                loaded.down_weight,
                loaded.down_scales,
                loaded.down_biases,
                rhs_indices=remap,
                **qmm_kwargs,
            )
            if loaded.down_bias is not None:
                expert_out = expert_out + loaded.down_bias[remap][..., None, :]
        else:
            if gated:
                gate_out = mx.gather_mm(
                    x_expanded,
                    loaded.gate_weight.swapaxes(-1, -2),
                    rhs_indices=remap,
                )
                if loaded.gate_bias is not None:
                    gate_out = gate_out + loaded.gate_bias[remap][..., None, :]
            up_out = mx.gather_mm(
                x_expanded,
                loaded.up_weight.swapaxes(-1, -2),
                rhs_indices=remap,
            )
            if loaded.up_bias is not None:
                up_out = up_out + loaded.up_bias[remap][..., None, :]
            activated = (
                self._apply_gated_activation(up_out, gate_out)
                if gated
                else self._apply_ungated_activation(up_out)
            )
            expert_out = mx.gather_mm(
                activated,
                loaded.down_weight.swapaxes(-1, -2),
                rhs_indices=remap,
            )
            if loaded.down_bias is not None:
                expert_out = expert_out + loaded.down_bias[remap][..., None, :]

        # expert_out shape: (B, L, K, 1, H) — squeeze the extra dim
        expert_out = expert_out.squeeze(-2)

        # Weight by router scores and sum across experts
        output = (expert_out * scores[..., None]).sum(axis=-2)

        return output.astype(x.dtype)
