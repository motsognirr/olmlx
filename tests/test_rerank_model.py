import mlx.core as mx

from olmlx.engine.rerank.config import RerankerConfig
from olmlx.engine.rerank.model import (
    XLMRobertaCrossEncoder,
    XLMRobertaEmbeddings,
    roberta_position_ids,
)


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


def _tiny_config() -> RerankerConfig:
    return RerankerConfig(
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=32,
        vocab_size=50,
        type_vocab_size=1,
        layer_norm_eps=1e-5,
        pad_token_id=1,
        num_labels=1,
    )


def test_embeddings_output_shape():
    cfg = _tiny_config()
    emb = XLMRobertaEmbeddings(cfg)
    input_ids = mx.array([[5, 6, 7, 2], [5, 6, 1, 1]])
    out = emb(input_ids)
    mx.eval(out)
    assert out.shape == (2, 4, cfg.hidden_size)


def test_cross_encoder_forward_shape():
    cfg = _tiny_config()
    model = XLMRobertaCrossEncoder(cfg)
    input_ids = mx.array([[5, 6, 7, 2], [5, 6, 1, 1]])  # 2nd row padded
    attention_mask = mx.array([[1, 1, 1, 1], [1, 1, 0, 0]])
    logits = model(input_ids, attention_mask)
    mx.eval(logits)
    assert logits.shape == (2, 1)
    assert not bool(mx.any(mx.isnan(logits)))  # masking must not produce NaN


def test_cross_encoder_padding_invariance():
    # Scoring a sequence must not change when extra pad tokens are appended,
    # because the attention mask zeroes them out.
    cfg = _tiny_config()
    model = XLMRobertaCrossEncoder(cfg)
    short_ids = mx.array([[5, 6, 7, 2]])
    short_mask = mx.array([[1, 1, 1, 1]])
    padded_ids = mx.array([[5, 6, 7, 2, 1, 1]])
    padded_mask = mx.array([[1, 1, 1, 1, 0, 0]])
    a = model(short_ids, short_mask)
    b = model(padded_ids, padded_mask)
    mx.eval(a, b)
    assert abs(float(a[0, 0]) - float(b[0, 0])) < 1e-5
