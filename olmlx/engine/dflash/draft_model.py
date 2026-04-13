"""DFlash draft model architecture.

The draft model takes hidden states from specific target layers as input
and produces logits for candidate tokens. Uses a cross-attention-like
mechanism where queries attend to concatenated target + current hidden states.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class DraftConfig:
    """Configuration for the dflash draft model."""

    hidden_size: int
    num_attention_heads: int
    num_layers: int
    target_layer_ids: list[int]
    vocab_size: int
    target_hidden_size: int | None = None  # defaults to hidden_size if None
    head_dim: int | None = None  # defaults to hidden_size // num_attention_heads

    def __post_init__(self):
        if self.target_hidden_size is None:
            self.target_hidden_size = self.hidden_size
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads


class DFlashAttention(nn.Module):
    """Cross-attention: queries attend to concatenation of target + current states."""

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

    def __call__(self, x: mx.array, context: mx.array | None = None) -> mx.array:
        B, L, _ = x.shape

        q = (
            self.q_proj(x)
            .reshape(B, L, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )

        # Keys/values from concatenated [context, x] if context provided
        kv_input = mx.concatenate([context, x], axis=1) if context is not None else x
        k = (
            self.k_proj(kv_input)
            .reshape(B, -1, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.v_proj(kv_input)
            .reshape(B, -1, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )

        # No causal mask: at inference, _draft_block passes single tokens
        # (L=1), so masking is a no-op. The architecture is non-causal by
        # design — queries attend bidirectionally to [context, x].
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        weights = mx.softmax(scores, axis=-1)
        out = (weights @ v).transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.o_proj(out)


class DFlashDecoderLayer(nn.Module):
    """Single draft decoder layer: attention + MLP with residual connections."""

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int):
        super().__init__()
        self.attention = DFlashAttention(hidden_size, num_heads, head_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4, bias=False),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size, bias=False),
        )
        self.input_layernorm = nn.RMSNorm(hidden_size)
        self.post_attention_layernorm = nn.RMSNorm(hidden_size)

    def __call__(self, x: mx.array, context: mx.array | None = None) -> mx.array:
        h = x + self.attention(self.input_layernorm(x), context=context)
        return h + self.mlp(self.post_attention_layernorm(h))


class DFlashDraftModel(nn.Module):
    """Draft model for dflash speculative decoding.

    Takes current token embeddings and target hidden states as input,
    produces logits for candidate tokens.
    """

    def __init__(self, config: DraftConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Project target hidden states to draft hidden_size if they differ
        if config.target_hidden_size != config.hidden_size:
            self.target_proj = nn.Linear(
                config.target_hidden_size, config.hidden_size, bias=False
            )
        else:
            self.target_proj = None

        self.layers = [
            DFlashDecoderLayer(
                config.hidden_size, config.num_attention_heads, config.head_dim
            )
            for _ in range(config.num_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        input_ids: mx.array,
        target_hidden_states: dict[int, mx.array],
    ) -> mx.array:
        """Forward pass.

        Args:
            input_ids: (B, L) token IDs.
            target_hidden_states: {layer_idx: (B, L, target_hidden_size)}

        Returns:
            (B, L, vocab_size) logits.
        """
        context = self._build_context(target_hidden_states)
        return self.forward_with_context(input_ids, context)

    def forward_with_context(
        self,
        input_ids: mx.array,
        context: mx.array | None,
    ) -> mx.array:
        """Forward pass with pre-computed context (avoids recomputing per token)."""
        h = self.embed_tokens(input_ids)

        for layer in self.layers:
            h = layer(h, context=context)

        h = self.norm(h)
        return self.lm_head(h)

    def _build_context(
        self, target_hidden_states: dict[int, mx.array]
    ) -> mx.array | None:
        """Combine target hidden states into a context tensor."""
        if not target_hidden_states:
            return None

        # Simple mean pooling across target layers. This collapses
        # layer-specific signals but keeps the draft model architecture
        # simple. Pre-trained draft models are trained against this
        # aggregation; per-layer projections can be added if needed.
        states = list(target_hidden_states.values())
        combined = states[0]
        for s in states[1:]:
            combined = combined + s
        combined = combined / len(states)

        if self.target_proj is not None:
            combined = self.target_proj(combined)

        return combined
