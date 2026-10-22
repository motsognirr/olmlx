"""Vendored Flint architecture (olmlx.engine.backports) and its Flash-MoE router."""

import mlx.core as mx
import mlx.nn as nn

from olmlx.engine.backports import install_mlx_lm_backports

_TINY_CONFIG = {
    "model_type": "flint",
    "hidden_size": 32,
    "num_hidden_layers": 4,
    "intermediate_size": 64,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 16,
    "shared_expert_intermediate_size": 16,
    "mlp_only_layers": [0],
    "layer_types": [
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ],
    "sliding_window": 4,
    "rms_norm_eps": 1e-6,
    "vocab_size": 64,
}


def _tiny_model():
    install_mlx_lm_backports()
    from mlx_lm.utils import _get_classes

    model_cls, args_cls = _get_classes(_TINY_CONFIG)
    model = model_cls(args_cls.from_dict(_TINY_CONFIG))
    mx.eval(model.parameters())
    return model


def test_registration_resolves_flint_model_type():
    import importlib.util

    install_mlx_lm_backports()
    assert importlib.util.find_spec("mlx_lm.models.flint") is not None


def test_registration_is_a_real_submodule():
    install_mlx_lm_backports()
    from mlx_lm.models import flint  # needs the package attribute, not just sys.modules

    assert hasattr(flint, "Model")


def test_tiny_flint_forward_with_cache():
    model = _tiny_model()
    cache = model.make_cache()
    prompt = mx.array([[1, 2, 3, 4, 5, 6]])
    logits = model(prompt, cache=cache)
    assert logits.shape == (1, 6, 64)
    # Decode past the sliding window to exercise RotatingKVCache.
    step = model(mx.array([[7]]), cache=cache)
    assert step.shape == (1, 1, 64)


def test_flash_moe_flint_matches_sparse_block():
    from olmlx.engine.flash.flash_moe_model import _FlashMoEFlint, _is_flint_moe

    model = _tiny_model()
    block = model.layers[1].mlp
    assert _is_flint_moe(block)
    assert not _is_flint_moe(nn.Linear(2, 2))

    # Dense stand-in for FlashMoE: same dispatch, weights from RAM.
    def dense_flash(x, inds, scores):
        return (block.switch_mlp(x, inds) * scores[..., None]).sum(axis=-2)

    replacement = _FlashMoEFlint(block, dense_flash)
    x = mx.random.normal((1, 5, 32))
    expected = block(x)
    actual = replacement(x)
    assert mx.allclose(actual, expected, atol=1e-4).item()


def test_failed_backport_is_logged_and_retried(monkeypatch, caplog):
    import olmlx.engine.backports as bp

    monkeypatch.setattr(bp, "_installed", False)
    monkeypatch.setattr(
        bp, "_BACKPORTS", {"nope_arch": "olmlx.engine.backports.does_not_exist"}
    )
    with caplog.at_level("WARNING", logger="olmlx.engine.backports"):
        bp.install_mlx_lm_backports()  # must not raise
    assert "nope_arch" in caplog.text
    assert bp._installed is False  # not latched: the next load retries


def test_backport_non_import_error_is_contained(monkeypatch, caplog):
    """API drift surfacing as e.g. AttributeError must not break other loads."""
    import importlib

    import olmlx.engine.backports as bp

    monkeypatch.setattr(bp, "_installed", False)
    monkeypatch.setattr(bp, "_warned", set())
    monkeypatch.setattr(bp, "_BACKPORTS", {"drift_arch": "x.y"})

    def boom(name):
        raise AttributeError("mlx.nn has no attribute 'shard_linear'")

    monkeypatch.setattr(importlib, "import_module", boom)
    with caplog.at_level("DEBUG", logger="olmlx.engine.backports"):
        bp.install_mlx_lm_backports()
        bp.install_mlx_lm_backports()
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1  # warned once, retry logged at debug


def test_flash_moe_qwen3_branch_rejects_shared_expert():
    import pytest

    from olmlx.engine.flash.flash_moe_model import _FlashMoEQwen3

    class _LinearGatedWithShared(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = nn.Linear(4, 4, bias=False)
            self.shared_expert = nn.Linear(4, 4)
            self.top_k = 2
            self.norm_topk_prob = True

    with pytest.raises(NotImplementedError, match="shared_expert"):
        _FlashMoEQwen3(_LinearGatedWithShared(), lambda *a: None)


def test_flash_moe_qwen3_branch_rejects_share_expert_spelling():
    import pytest

    from olmlx.engine.flash.flash_moe_model import _FlashMoEQwen3

    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = nn.Linear(4, 4, bias=False)
            self.share_expert = nn.Linear(4, 4)
            self.top_k = 2
            self.norm_topk_prob = True

    with pytest.raises(NotImplementedError, match="share_expert"):
        _FlashMoEQwen3(_Block(), lambda *a: None)
