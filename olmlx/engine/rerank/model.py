from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from olmlx.engine.rerank.config import RerankerConfig


def roberta_position_ids(input_ids: mx.array, pad_token_id: int) -> mx.array:
    """Position ids with the RoBERTa offset.

    Mirrors transformers' ``create_position_ids_from_input_ids``:
    ``position_ids = cumsum(mask) * mask + pad_token_id`` where
    ``mask = (input_ids != pad_token_id)``. The first real token therefore
    gets position ``pad_token_id + 1`` (== 2 for XLM-RoBERTa), which is why
    the position-embedding table is sized ``max_seq + pad_token_id + 1``.
    """
    mask = (input_ids != pad_token_id).astype(mx.int32)
    incremental = mx.cumsum(mask, axis=1) * mask
    return incremental + pad_token_id


class XLMRobertaEmbeddings(nn.Module):
    def __init__(self, config: RerankerConfig):
        super().__init__()
        self.pad_token_id = config.pad_token_id
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
        # PascalCase matches the HF checkpoint weight key (embeddings.LayerNorm).
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(self, input_ids: mx.array) -> mx.array:
        pos_ids = roberta_position_ids(input_ids, self.pad_token_id)
        words = self.word_embeddings(input_ids)
        positions = self.position_embeddings(pos_ids)
        # token_type_ids are all zero (type_vocab_size == 1): embed token-type 0
        # once as a [1, hidden] row and let it broadcast across batch/sequence,
        # avoiding a full [batch, seq, hidden] gather.
        token_type = self.token_type_embeddings(mx.array([0]))
        return self.LayerNorm(words + positions + token_type)


class XLMRobertaSelfAttention(nn.Module):
    def __init__(self, config: RerankerConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        h = config.hidden_size
        self.query = nn.Linear(h, h)
        self.key = nn.Linear(h, h)
        self.value = nn.Linear(h, h)

    def __call__(self, x: mx.array, additive_mask: mx.array) -> mx.array:
        b, s, _ = x.shape
        q = (
            self.query(x)
            .reshape(b, s, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        k = (
            self.key(x)
            .reshape(b, s, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.value(x)
            .reshape(b, s, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        scores = scores + additive_mask  # [b, 1, 1, s] broadcast
        weights = mx.softmax(scores, axis=-1)
        out = weights @ v  # [b, heads, s, head_dim]
        return out.transpose(0, 2, 1, 3).reshape(b, s, -1)


class XLMRobertaLayer(nn.Module):
    def __init__(self, config: RerankerConfig):
        super().__init__()
        h = config.hidden_size
        self.attention_self = XLMRobertaSelfAttention(config)
        self.attention_output_dense = nn.Linear(h, h)
        self.attention_output_norm = nn.LayerNorm(h, eps=config.layer_norm_eps)
        self.intermediate_dense = nn.Linear(h, config.intermediate_size)
        self.output_dense = nn.Linear(config.intermediate_size, h)
        self.output_norm = nn.LayerNorm(h, eps=config.layer_norm_eps)

    def __call__(self, x: mx.array, additive_mask: mx.array) -> mx.array:
        attn = self.attention_self(x, additive_mask)
        x = self.attention_output_norm(self.attention_output_dense(attn) + x)
        inter = nn.gelu(self.intermediate_dense(x))
        x = self.output_norm(self.output_dense(inter) + x)
        return x


class XLMRobertaClassificationHead(nn.Module):
    def __init__(self, config: RerankerConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def __call__(self, features: mx.array) -> mx.array:
        x = features[:, 0, :]  # first token (<s> / CLS)
        x = mx.tanh(self.dense(x))
        return self.out_proj(x)


class XLMRobertaCrossEncoder(nn.Module):
    def __init__(self, config: RerankerConfig):
        super().__init__()
        self.config = config
        self.embeddings = XLMRobertaEmbeddings(config)
        self.layers = [XLMRobertaLayer(config) for _ in range(config.num_hidden_layers)]
        self.classifier = XLMRobertaClassificationHead(config)

    def __call__(self, input_ids: mx.array, attention_mask: mx.array) -> mx.array:
        # additive mask: keep -> 0, pad -> -inf, shaped [b, 1, 1, s]
        additive = (1.0 - attention_mask.astype(mx.float32))[:, None, None, :] * -1e9
        x = self.embeddings(input_ids)
        for layer in self.layers:
            x = layer(x, additive)
        return self.classifier(x)
