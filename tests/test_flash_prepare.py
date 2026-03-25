"""Tests for streaming flash inference preparation."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import mlx.core as mx
import mlx.nn as nn
import pytest


# --- Fake model components that mimic mlx-lm's structure ---

HIDDEN = 32
INTER = 64
VOCAB = 100
NUM_LAYERS = 2
NUM_HEADS = 4
HEAD_DIM = HIDDEN // NUM_HEADS


class FakeMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(HIDDEN, INTER, bias=False)
        self.up_proj = nn.Linear(HIDDEN, INTER, bias=False)
        self.down_proj = nn.Linear(INTER, HIDDEN, bias=False)

    def __call__(self, x):
        gate = nn.silu(self.gate_proj(x))
        return self.down_proj(gate * self.up_proj(x))


class FakeAttention(nn.Module):
    """Minimal attention that just passes through (no RoPE, no cache)."""

    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.k_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.v_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.o_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.scale = HEAD_DIM**-0.5

    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, NUM_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, NUM_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, NUM_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, HIDDEN)
        return self.o_proj(out)


class FakeTransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = FakeAttention()
        self.mlp = FakeMLP()
        self.input_layernorm = nn.RMSNorm(HIDDEN)
        self.post_attention_layernorm = nn.RMSNorm(HIDDEN)

    def __call__(self, x, mask=None, cache=None):
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class FakeInnerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB, HIDDEN)
        self.layers = [FakeTransformerBlock() for _ in range(NUM_LAYERS)]
        self.norm = nn.RMSNorm(HIDDEN)

    def __call__(self, inputs, cache=None):
        from olmlx.engine.flash.prepare import _create_causal_mask

        h = self.embed_tokens(inputs)
        mask = _create_causal_mask(h)
        if cache is None:
            cache = [None] * len(self.layers)
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)
        return self.norm(h)


class FakeModel(nn.Module):
    """Mimics mlx-lm's outer Model wrapper."""

    def __init__(self):
        super().__init__()
        self.model = FakeInnerModel()

    def __call__(self, inputs, cache=None):
        return self.model(inputs, cache)

    @property
    def layers(self):
        return self.model.layers


class FakeTokenizer:
    def encode(self, text):
        # Return deterministic token ids based on text length
        return list(range(1, min(len(text) + 1, 20)))


class TestNullifyModuleParams:
    def test_replaces_all_params_with_placeholders(self):
        """After nullification, all parameter arrays should be size 1."""
        from olmlx.engine.flash.prepare import _nullify_module_params

        model = nn.Linear(32, 64)
        mx.eval(model.parameters())
        assert model.weight.size > 1

        _nullify_module_params(model)

        assert model.weight.size == 1

    def test_handles_nested_modules(self):
        """Nullification should work on modules with nested children."""
        from olmlx.engine.flash.prepare import _nullify_module_params

        class FakeMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = nn.Linear(32, 64, bias=False)
                self.up_proj = nn.Linear(32, 64, bias=False)
                self.down_proj = nn.Linear(64, 32, bias=False)

        mlp = FakeMLP()
        mx.eval(mlp.parameters())
        assert mlp.gate_proj.weight.size > 1

        _nullify_module_params(mlp)

        assert mlp.gate_proj.weight.size == 1
        assert mlp.up_proj.weight.size == 1
        assert mlp.down_proj.weight.size == 1


def _patch_mlx_lm(model, tokenizer):
    """Create a patch context that makes the deferred ``import mlx_lm`` return a mock."""
    mock = MagicMock()
    mock.load.return_value = (model, tokenizer)
    return patch.dict(sys.modules, {"mlx_lm": mock}), mock


class TestStreamRecordActivations:
    def _make_fake_model_and_tokenizer(self):
        model = FakeModel()
        mx.eval(model.parameters())
        tokenizer = FakeTokenizer()
        return model, tokenizer

    def test_produces_recordings_for_each_layer(self):
        from olmlx.engine.flash.prepare import _stream_record_activations

        model, tokenizer = self._make_fake_model_and_tokenizer()
        ctx, _ = _patch_mlx_lm(model, tokenizer)
        with ctx:
            texts = ["Hello world", "Test input"]
            recordings, hidden_size, intermediate_size, num_layers = (
                _stream_record_activations("fake_path", texts)
            )

        assert len(recordings) == NUM_LAYERS
        for i in range(NUM_LAYERS):
            inputs_list, targets_list = recordings[i]
            assert len(inputs_list) == len(texts)
            assert len(targets_list) == len(texts)
            # Each input should be of shape (hidden_size,)
            assert inputs_list[0].shape == (HIDDEN,)
            # Each target should be of shape (intermediate_size,)
            assert targets_list[0].shape == (INTER,)

        assert hidden_size == HIDDEN
        assert intermediate_size == INTER
        assert num_layers == NUM_LAYERS

    def test_frees_layer_params_after_processing(self):
        """After streaming, all layer params should be nullified."""
        from olmlx.engine.flash.prepare import _stream_record_activations

        model, tokenizer = self._make_fake_model_and_tokenizer()
        ctx, _ = _patch_mlx_lm(model, tokenizer)
        with ctx:
            _stream_record_activations("fake_path", ["Hello"])

        # All layer weights should be tiny placeholders now
        for layer in model.model.layers:
            assert layer.mlp.gate_proj.weight.size == 1
            assert layer.self_attn.q_proj.weight.size == 1

    def test_calls_progress_callback(self):
        from olmlx.engine.flash.prepare import _stream_record_activations

        model, tokenizer = self._make_fake_model_and_tokenizer()
        ctx, _ = _patch_mlx_lm(model, tokenizer)
        calls = []
        with ctx:
            _stream_record_activations(
                "fake_path",
                ["Hello"],
                progress_callback=lambda d, f: calls.append((d, f)),
            )

        assert len(calls) > 0
        # Last call should be at fraction 1.0
        assert calls[-1][1] == pytest.approx(1.0)

    def test_load_called_with_lazy(self):
        """Streaming path must request lazy loading."""
        from olmlx.engine.flash.prepare import _stream_record_activations

        model, tokenizer = self._make_fake_model_and_tokenizer()
        ctx, mock_mlx_lm = _patch_mlx_lm(model, tokenizer)
        with ctx:
            _stream_record_activations("fake_path", ["Hello"])

        mock_mlx_lm.load.assert_called_once_with("fake_path", lazy=True)


class TestStreamingEquivalence:
    """Verify streaming produces same recordings as full-forward pass."""

    def test_recordings_match_full_forward(self):
        """Streaming layer-by-layer should produce identical activation patterns."""
        from olmlx.engine.flash.prepare import (
            _record_activations,
            _stream_record_activations,
        )

        # Create two identical models with same weights
        mx.random.seed(42)
        model_a = FakeModel()
        mx.eval(model_a.parameters())

        # Deep-copy weights to model_b using flattened weights
        from mlx.utils import tree_flatten

        model_b = FakeModel()
        weights = tree_flatten(model_a.parameters())
        model_b.load_weights(weights)
        mx.eval(model_b.parameters())

        tokenizer = FakeTokenizer()
        texts = ["Explain the concept of machine learning in simple terms."]

        # Full-forward recordings
        full_recordings = _record_activations(
            model_a, tokenizer, texts, activation_threshold=0.01
        )

        # Streaming recordings (mock mlx_lm.load to return model_b)
        ctx, _ = _patch_mlx_lm(model_b, tokenizer)
        with ctx:
            stream_recordings, _, _, _ = _stream_record_activations(
                "fake_path", texts, activation_threshold=0.01
            )

        # Compare recordings layer by layer
        for layer_idx in range(NUM_LAYERS):
            full_inputs, full_targets = full_recordings[layer_idx]
            stream_inputs, stream_targets = stream_recordings[layer_idx]

            assert len(full_inputs) == len(stream_inputs)
            for fi, si in zip(full_inputs, stream_inputs):
                assert mx.allclose(fi, si, atol=1e-5).item(), (
                    f"Layer {layer_idx} inputs differ"
                )
            for ft, st in zip(full_targets, stream_targets):
                assert mx.array_equal(ft, st).item(), (
                    f"Layer {layer_idx} targets differ"
                )


class TestPrepareModelEndToEnd:
    """End-to-end test of prepare_model_for_flash with streaming."""

    @patch("olmlx.engine.flash.prepare.bundle_ffn_weights")
    def test_produces_output_files(self, mock_bundle, tmp_path):
        from olmlx.engine.flash.prepare import prepare_model_for_flash

        model = FakeModel()
        mx.eval(model.parameters())
        ctx, mock_mlx_lm = _patch_mlx_lm(model, FakeTokenizer())

        with ctx:
            output_dir = tmp_path / "flash"
            result = prepare_model_for_flash(
                model_path=str(tmp_path),
                output_dir=output_dir,
                rank=8,
                num_samples=4,
                epochs=1,
            )

        assert result == output_dir
        assert (output_dir / "flash_config.json").exists()
        assert (output_dir / "predictors").exists()

        # Verify config contents
        import json

        config = json.loads((output_dir / "flash_config.json").read_text())
        assert config["hidden_size"] == HIDDEN
        assert config["intermediate_size"] == INTER
        assert config["num_layers"] == NUM_LAYERS
        assert config["predictor_rank"] == 8

        # Verify mlx_lm.load was called with lazy=True
        mock_mlx_lm.load.assert_called_once_with(str(tmp_path), lazy=True)

        # Verify bundler was called
        mock_bundle.assert_called_once()


class TestStreamingEdgeCases:
    def test_layer_without_mlp_is_skipped(self):
        """Layers without gate_proj/up_proj should produce empty recordings."""
        from olmlx.engine.flash.prepare import _stream_record_activations

        model = FakeModel()
        mx.eval(model.parameters())

        # Replace layer 1's MLP with one that lacks gate_proj/up_proj
        class PassthroughMLP(nn.Module):
            def __call__(self, x):
                return x

        model.model.layers[1].mlp = PassthroughMLP()

        ctx, _ = _patch_mlx_lm(model, FakeTokenizer())
        with ctx:
            recordings, _, _, _ = _stream_record_activations("fake_path", ["Hello"])

        # Layer 0 should have recordings
        assert len(recordings[0][0]) == 1
        # Layer 1 should have empty recordings (no gate_proj/up_proj)
        assert len(recordings[1][0]) == 0
