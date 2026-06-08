"""Native MLX cross-encoder reranker (XLM-RoBERTa family). Issue #369."""

from olmlx.engine.rerank.config import RerankerConfig
from olmlx.engine.rerank.model import XLMRobertaCrossEncoder
from olmlx.engine.rerank.weights import detect_layout, load_cross_encoder

__all__ = [
    "RerankerConfig",
    "XLMRobertaCrossEncoder",
    "detect_layout",
    "load_cross_encoder",
]
