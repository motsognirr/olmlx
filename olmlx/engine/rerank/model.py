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
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(self, input_ids: mx.array) -> mx.array:
        pos_ids = roberta_position_ids(input_ids, self.pad_token_id)
        words = self.word_embeddings(input_ids)
        positions = self.position_embeddings(pos_ids)
        # token_type_ids are all zero (type_vocab_size == 1); add row-0 bias.
        token_type = self.token_type_embeddings(mx.zeros_like(input_ids))
        return self.LayerNorm(words + positions + token_type)
