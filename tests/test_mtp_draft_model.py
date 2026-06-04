import mlx.core as mx
import mlx.nn as nn

from olmlx.engine.mtp.draft_model import MTPConfig, MTPDraftModel
from tests.test_mtp_config import _DENSE_CFG


def _tiny_cfg():
    # Shrink the dense config so the test builds instantly.
    cfg = MTPConfig.from_dict(_DENSE_CFG)
    cfg.hidden_size = 128
    cfg.intermediate_size = 256
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 2
    cfg.head_dim = 32
    cfg.vocab_size = 512
    return cfg


def test_mtp_draft_forward_shapes():
    cfg = _tiny_cfg()
    draft = MTPDraftModel(cfg)
    embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
    lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    draft.bind_via_modules(embed, lm_head)
    mx.eval(draft.parameters())

    tok = mx.array([[5]], dtype=mx.int32)
    h_prev = mx.zeros((1, 1, cfg.hidden_size))
    cache = draft.make_cache()
    logits, h_new = draft(tok, h_prev, cache=cache)
    assert logits.shape == (1, 1, cfg.vocab_size)
    assert h_new.shape == (1, 1, cfg.hidden_size)


def test_mtp_draft_requires_bind():
    cfg = _tiny_cfg()
    draft = MTPDraftModel(cfg)
    try:
        draft(mx.array([[1]], dtype=mx.int32), mx.zeros((1, 1, cfg.hidden_size)))
        assert False, "expected RuntimeError without bind()"
    except RuntimeError:
        pass
