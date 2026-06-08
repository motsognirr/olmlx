import mlx.core as mx

from olmlx.engine.rerank.config import RerankerConfig
from olmlx.engine.rerank.model import roberta_position_ids


def test_rerankerconfig_from_dict_bge():
    raw = {
        "architectures": ["XLMRobertaForSequenceClassification"],
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "intermediate_size": 4096,
        "max_position_embeddings": 8194,
        "vocab_size": 250002,
        "type_vocab_size": 1,
        "layer_norm_eps": 1e-5,
        "pad_token_id": 1,
        "hidden_act": "gelu",
        "id2label": {"0": "LABEL_0"},
    }
    cfg = RerankerConfig.from_dict(raw)
    assert cfg.hidden_size == 1024
    assert cfg.num_hidden_layers == 24
    assert cfg.num_attention_heads == 16
    assert cfg.intermediate_size == 4096
    assert cfg.max_position_embeddings == 8194
    assert cfg.vocab_size == 250002
    assert cfg.type_vocab_size == 1
    assert cfg.pad_token_id == 1
    assert cfg.num_labels == 1
    assert cfg.head_dim == 64
    assert cfg.hidden_act == "gelu"


def test_rerankerconfig_num_labels_from_num_labels_key():
    raw = {
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "max_position_embeddings": 1026,
        "vocab_size": 250002,
        "type_vocab_size": 1,
        "layer_norm_eps": 1e-5,
        "pad_token_id": 1,
        "num_labels": 1,
    }
    cfg = RerankerConfig.from_dict(raw)
    assert cfg.num_labels == 1
    assert cfg.head_dim == 64
    assert cfg.hidden_act == "gelu"  # default applied when key absent


def test_roberta_position_ids_offset_no_padding():
    # pad_token_id=1; a fully-real sequence -> positions start at 2
    input_ids = mx.array([[5, 6, 7, 8]])
    pos = roberta_position_ids(input_ids, pad_token_id=1)
    assert pos.tolist() == [[2, 3, 4, 5]]


def test_roberta_position_ids_offset_with_padding():
    # trailing pad tokens (id=1) keep position == pad_token_id (1)
    input_ids = mx.array([[5, 6, 1, 1]])
    pos = roberta_position_ids(input_ids, pad_token_id=1)
    assert pos.tolist() == [[2, 3, 1, 1]]
