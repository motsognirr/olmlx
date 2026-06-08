from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RerankerConfig:
    """Sizing parameters parsed from an XLM-RoBERTa cross-encoder config.json."""

    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    max_position_embeddings: int
    vocab_size: int
    type_vocab_size: int
    layer_norm_eps: float
    pad_token_id: int
    num_labels: int
    hidden_act: str = "gelu"

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RerankerConfig:
        num_labels = raw.get("num_labels")
        if num_labels is None:
            id2label = raw.get("id2label") or {"0": "LABEL_0"}
            num_labels = len(id2label)
        return cls(
            hidden_size=int(raw["hidden_size"]),
            num_hidden_layers=int(raw["num_hidden_layers"]),
            num_attention_heads=int(raw["num_attention_heads"]),
            intermediate_size=int(raw["intermediate_size"]),
            max_position_embeddings=int(raw["max_position_embeddings"]),
            vocab_size=int(raw["vocab_size"]),
            type_vocab_size=int(raw.get("type_vocab_size", 1)),
            layer_norm_eps=float(raw.get("layer_norm_eps", 1e-5)),
            pad_token_id=int(raw.get("pad_token_id", 1)),
            num_labels=int(num_labels),
            hidden_act=str(raw.get("hidden_act", "gelu")),
        )
