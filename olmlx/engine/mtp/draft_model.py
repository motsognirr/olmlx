"""MTP draft head: one full-attention Qwen3.6 layer + MTP front-end.

The head consumes ``(token_{i+1}, h_i)`` where ``h_i`` is the target's
last-layer (pre-``model.norm``) hidden, and produces ``(logits, h_new)``.
``h_new`` is the layer output BEFORE the head's own ``norm`` — it is fed
back as ``h_prev`` for the next autoregressive draft step (DeepSeek/Qwen
MTP convention). ``norm`` is applied only to compute logits via the
target's borrowed ``lm_head``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mlx_lm.models.qwen3_5 import TextModelArgs as _Qwen35TextArgs


@dataclass
class MTPConfig:
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    vocab_size: int
    max_position_embeddings: int
    full_attention_interval: int
    block_size: int
    rope_parameters: dict[str, Any] | None = None
    tie_word_embeddings: bool = False
    # MoE (0 => dense)
    num_experts: int = 0
    num_experts_per_tok: int = 0
    moe_intermediate_size: int = 0
    shared_expert_intermediate_size: int = 0
    norm_topk_prob: bool = True
    decoder_sparse_step: int = 1
    # Quantization (None => not quantized)
    quant_group_size: int | None = None
    quant_bits: int | None = None

    @property
    def is_moe(self) -> bool:
        return self.num_experts > 0

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "MTPConfig":
        text = config.get("text_config", config)
        quant = config.get("quantization") or config.get("quantization_config") or {}
        return cls(
            hidden_size=text["hidden_size"],
            intermediate_size=text["intermediate_size"],
            num_attention_heads=text["num_attention_heads"],
            num_key_value_heads=text["num_key_value_heads"],
            head_dim=text.get(
                "head_dim", text["hidden_size"] // text["num_attention_heads"]
            ),
            rms_norm_eps=text.get("rms_norm_eps", 1e-6),
            vocab_size=text["vocab_size"],
            max_position_embeddings=text.get("max_position_embeddings", 262144),
            full_attention_interval=text.get("full_attention_interval", 4),
            block_size=config.get("block_size", 1),
            rope_parameters=text.get("rope_parameters"),
            tie_word_embeddings=text.get("tie_word_embeddings", False),
            num_experts=text.get("num_experts", 0),
            num_experts_per_tok=text.get("num_experts_per_tok", 0),
            moe_intermediate_size=text.get("moe_intermediate_size", 0),
            shared_expert_intermediate_size=text.get(
                "shared_expert_intermediate_size", 0
            ),
            norm_topk_prob=text.get("norm_topk_prob", True),
            decoder_sparse_step=text.get("decoder_sparse_step", 1),
            quant_group_size=quant.get("group_size"),
            quant_bits=quant.get("bits"),
        )

    def to_qwen35_text_args(self) -> _Qwen35TextArgs:
        """Build the mlx-lm ``TextModelArgs`` the reused ``DecoderLayer``
        expects. ``num_hidden_layers`` is forced to ``full_attention_interval``
        so layer_idx ``full_attention_interval - 1`` is a FULL-attention
        layer (``(idx+1) % interval == 0``)."""
        return _Qwen35TextArgs(
            model_type="qwen3_5_text",
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.full_attention_interval,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            rms_norm_eps=self.rms_norm_eps,
            vocab_size=self.vocab_size,
            max_position_embeddings=self.max_position_embeddings,
            full_attention_interval=self.full_attention_interval,
            tie_word_embeddings=self.tie_word_embeddings,
            num_experts=self.num_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            moe_intermediate_size=self.moe_intermediate_size,
            shared_expert_intermediate_size=self.shared_expert_intermediate_size,
            norm_topk_prob=self.norm_topk_prob,
            decoder_sparse_step=self.decoder_sparse_step,
            rope_parameters=self.rope_parameters
            or {
                "type": "default",
                "mrope_section": [11, 11, 10],
                "rope_theta": 10000000,
                "partial_rotary_factor": 0.25,
            },
        )
