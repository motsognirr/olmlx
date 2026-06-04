import pytest
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
    with pytest.raises(RuntimeError):
        draft(mx.array([[1]], dtype=mx.int32), mx.zeros((1, 1, cfg.hidden_size)))


def test_mtp_draft_compute_logits_false_returns_none():
    cfg = _tiny_cfg()
    draft = MTPDraftModel(cfg)
    embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
    lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    draft.bind_via_modules(embed, lm_head)
    mx.eval(draft.parameters())
    logits, h_new = draft(
        mx.array([[3]], dtype=mx.int32),
        mx.zeros((1, 1, cfg.hidden_size)),
        cache=draft.make_cache(),
        compute_logits=False,
    )
    assert logits is None
    assert h_new.shape == (1, 1, cfg.hidden_size)


def test_mtp_h_new_is_pre_norm():
    # h_new must be the pre-norm layer output, NOT norm(x). Use a nonzero
    # h_prev so the layer output is non-trivial.
    cfg = _tiny_cfg()
    draft = MTPDraftModel(cfg)
    draft.bind_via_modules(
        nn.Embedding(cfg.vocab_size, cfg.hidden_size),
        nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False),
    )
    mx.eval(draft.parameters())
    h_prev = mx.random.normal((1, 1, cfg.hidden_size))
    _, h_new = draft(mx.array([[7]], dtype=mx.int32), h_prev, cache=draft.make_cache())
    # norm(h_new) must differ from h_new (else h_new was already normed).
    diff = float(mx.max(mx.abs(draft.norm(h_new) - h_new)).item())
    assert diff > 1e-4, "h_new appears already-normalised; pre-norm chaining broken"


def test_mtp_layer_is_full_attention():
    cfg = _tiny_cfg()
    draft = MTPDraftModel(cfg)
    layer = draft.layers[0]
    assert getattr(layer, "is_linear", None) is False
    assert hasattr(layer, "self_attn") and not hasattr(layer, "linear_attn")


def test_mtp_concat_order_changes_output():
    # Build two independent models with opposite concat_hidden_first settings.
    # Give them identical weights and verify that the concat order affects output.
    cfg_a = _tiny_cfg()
    cfg_a.concat_hidden_first = True
    a = MTPDraftModel(cfg_a)

    cfg_b = _tiny_cfg()
    cfg_b.concat_hidden_first = False
    b = MTPDraftModel(cfg_b)

    # Copy weights from a into b so only concat order differs.
    b.update(a.parameters())

    embed = nn.Embedding(cfg_a.vocab_size, cfg_a.hidden_size)
    lm_head = nn.Linear(cfg_a.hidden_size, cfg_a.vocab_size, bias=False)
    a.bind_via_modules(embed, lm_head)
    b.bind_via_modules(embed, lm_head)
    mx.eval(a.parameters(), b.parameters())

    tok = mx.array([[2]], dtype=mx.int32)
    h = mx.random.normal((1, 1, cfg_a.hidden_size))
    la, _ = a(tok, h, cache=a.make_cache())
    lb, _ = b(tok, h, cache=b.make_cache())
    assert float(mx.max(mx.abs(la - lb)).item()) > 1e-4
