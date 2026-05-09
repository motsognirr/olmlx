"""Tests for the EAGLE-style autoregressive speculative draft.

Covers:

- ``EagleConfig`` dataclass validation
- ``EagleDraftModel`` construction, forward pass, and weight shapes
- ``bind`` / ``unbind`` for sharing target's ``embed_tokens`` and
  ``lm_head`` (mirrors the DFlash pattern)
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest


class TestEagleConfig:
    """Field-by-field validation for the draft config dataclass."""

    def test_minimal_fields_construct_cleanly(self):
        from olmlx.engine.eagle.draft_model import EagleConfig

        cfg = EagleConfig(
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=32,
            intermediate_size=128,
            vocab_size=1024,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            max_position_embeddings=2048,
            block_size=4,
        )
        assert cfg.hidden_size == 64
        assert cfg.num_hidden_layers == 1
        assert cfg.block_size == 4

    def test_block_size_must_be_positive(self):
        from olmlx.engine.eagle.draft_model import EagleConfig

        with pytest.raises(ValueError, match="block_size"):
            EagleConfig(
                hidden_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=32,
                intermediate_size=128,
                vocab_size=1024,
                rms_norm_eps=1e-6,
                rope_theta=10000.0,
                max_position_embeddings=2048,
                block_size=0,
            )

    def test_kv_head_count_must_divide_attention_heads(self):
        # GQA invariant: num_attention_heads must be a multiple of
        # num_key_value_heads. Same constraint mlx-lm enforces.
        from olmlx.engine.eagle.draft_model import EagleConfig

        with pytest.raises(ValueError, match="num_key_value_heads"):
            EagleConfig(
                hidden_size=64,
                num_hidden_layers=1,
                num_attention_heads=3,  # 3 not divisible by 2
                num_key_value_heads=2,
                head_dim=32,
                intermediate_size=128,
                vocab_size=1024,
                rms_norm_eps=1e-6,
                rope_theta=10000.0,
                max_position_embeddings=2048,
                block_size=4,
            )


class TestEagleDraftModelConstruction:
    """The draft model builds from a config and exposes the expected
    parameter shapes. No target binding yet — that's a separate test
    class."""

    def _cfg(self, **overrides):
        from olmlx.engine.eagle.draft_model import EagleConfig

        defaults = dict(
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=32,
            intermediate_size=128,
            vocab_size=1024,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            max_position_embeddings=2048,
            block_size=4,
        )
        defaults.update(overrides)
        return EagleConfig(**defaults)

    def test_input_projection_shape(self):
        # The projection takes ``concat([h_target, embed(token)])`` of
        # size ``2 * hidden_size`` and produces ``hidden_size``.
        from olmlx.engine.eagle.draft_model import EagleDraftModel

        m = EagleDraftModel(self._cfg(hidden_size=64))
        # mlx.nn.Linear stores weights as (out_features, in_features).
        assert m.input_proj.weight.shape == (64, 128)

    def test_one_decoder_layer_by_default(self):
        from olmlx.engine.eagle.draft_model import EagleDraftModel

        m = EagleDraftModel(self._cfg(num_hidden_layers=1))
        assert len(m.layers) == 1

    def test_supports_two_layer_drafts(self):
        # EAGLE-2 uses two layers; the code must support num_layers > 1
        # without architecture changes.
        from olmlx.engine.eagle.draft_model import EagleDraftModel

        m = EagleDraftModel(self._cfg(num_hidden_layers=2))
        assert len(m.layers) == 2

    def test_norm_layer_present(self):
        from olmlx.engine.eagle.draft_model import EagleDraftModel

        m = EagleDraftModel(self._cfg())
        assert isinstance(m.norm, nn.RMSNorm)


class TestEagleDraftModelForward:
    """Forward pass shape and value checks against a synthetic target."""

    def _cfg_and_model(self, **overrides):
        from olmlx.engine.eagle.draft_model import EagleConfig, EagleDraftModel

        defaults = dict(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            intermediate_size=64,
            vocab_size=128,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            max_position_embeddings=512,
            block_size=4,
        )
        defaults.update(overrides)
        return EagleDraftModel(EagleConfig(**defaults))

    def _bind_dummy(self, model):
        # Provide an embed_tokens + lm_head so forward works.
        cfg = model.args
        embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        lm = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        model.bind_via_modules(embed, lm)
        return model

    def test_unbound_forward_raises(self):
        # Without binding ``embed_tokens`` and ``lm_head``, forward
        # cannot run — surface that rather than silently producing
        # garbage.
        m = self._cfg_and_model()
        with pytest.raises(RuntimeError, match="bind"):
            m(
                token_ids=mx.array([[5]]),
                h_prev=mx.zeros((1, 1, 32)),
            )

    def test_forward_shape(self):
        # Single-step draft: input is one previous token + one previous
        # hidden state. Output is logits over vocab + new hidden.
        m = self._bind_dummy(self._cfg_and_model())
        token_ids = mx.array([[5, 9]])  # batch=1, seq_len=2
        h_prev = mx.zeros((1, 2, 32))
        logits, h_new = m(token_ids=token_ids, h_prev=h_prev)
        assert logits.shape == (1, 2, 128)
        assert h_new.shape == (1, 2, 32)

    def test_bind_borrows_target_weights(self):
        # ``bind(target)`` must borrow the target's ``embed_tokens`` and
        # ``lm_head`` modules (not copy weights). Verify identity.
        m = self._cfg_and_model()
        target_embed = nn.Embedding(m.args.vocab_size, m.args.hidden_size)
        target_lm = nn.Linear(m.args.hidden_size, m.args.vocab_size, bias=False)

        class _FakeInner:
            embed_tokens = target_embed

        class _FakeTarget:
            model = _FakeInner()
            lm_head = target_lm

        m.bind(_FakeTarget())
        assert m.embed_tokens is target_embed
        assert m.lm_head is target_lm

        m.unbind()
        assert m.embed_tokens is None
        assert m.lm_head is None
