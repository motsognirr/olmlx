"""Tests for HQQ weight quantization engine."""

import mlx.core as mx
import numpy as np
import pytest

from olmlx.engine.hqq.quantize import (
    HQQConfig,
    HQQLinear,
    hqq_quantize_weight,
    hqq_quantize_linear,
)


class TestHQQConfig:
    def test_defaults(self):
        c = HQQConfig(bits=4)
        assert c.bits == 4
        assert c.group_size == 64
        assert c.n_iters == 3

    def test_8bit_default_group_size(self):
        c = HQQConfig(bits=8)
        assert c.group_size == 128

    def test_explicit_group_size(self):
        c = HQQConfig(bits=4, group_size=128)
        assert c.group_size == 128

    def test_n_iters(self):
        c = HQQConfig(bits=4, n_iters=5)
        assert c.n_iters == 5

    def test_from_string_hqq4(self):
        c = HQQConfig.from_string("hqq:4")
        assert c.bits == 4
        assert c.group_size == 64

    def test_from_string_hqq8(self):
        c = HQQConfig.from_string("hqq:8")
        assert c.bits == 8
        assert c.group_size == 128

    def test_from_string_with_group_size(self):
        c = HQQConfig.from_string("hqq:4:128")
        assert c.bits == 4
        assert c.group_size == 128

    def test_from_string_none(self):
        assert HQQConfig.from_string(None) is None


class TestHQQQuantizeWeight:
    def test_quantize_dequantize_4bit(self):
        mx.random.seed(42)
        w = mx.random.normal(shape=(64, 64))
        cfg = HQQConfig(bits=4, group_size=64)
        packed, scales, biases = hqq_quantize_weight(w, cfg)
        restored = mx.dequantize(packed, scales, biases, cfg.group_size, cfg.bits)
        assert restored.shape == w.shape
        assert restored.dtype == w.dtype
        err = mx.mean(mx.abs(w - restored)).item()
        assert err < 0.3

    def test_quantize_dequantize_8bit(self):
        mx.random.seed(42)
        w = mx.random.normal(shape=(64, 64))
        cfg = HQQConfig(bits=8, group_size=64)
        packed, scales, biases = hqq_quantize_weight(w, cfg)
        restored = mx.dequantize(packed, scales, biases, cfg.group_size, cfg.bits)
        err = mx.mean(mx.abs(w - restored)).item()
        assert err < 0.05

    def test_quantize_dequantize_4bit_hqq_better_than_minmax(self):
        """HQQ should give lower reconstruction error than naive min/max quantization."""
        mx.random.seed(42)
        w = mx.random.normal(shape=(128, 128))
        cfg = HQQConfig(bits=4, group_size=64, n_iters=3)

        # HQQ
        hqq_packed, hqq_scales, hqq_biases = hqq_quantize_weight(w, cfg)
        hqq_restored = mx.dequantize(hqq_packed, hqq_scales, hqq_biases, cfg.group_size, cfg.bits)
        hqq_err = mx.mean(mx.abs(w - hqq_restored)).item()

        # MLX native min/max affine quantization
        mx_packed, mx_scales, mx_biases = mx.quantize(w, group_size=cfg.group_size, bits=cfg.bits)
        mx_restored = mx.dequantize(mx_packed, mx_scales, mx_biases, cfg.group_size, cfg.bits)
        mx_err = mx.mean(mx.abs(w - mx_restored)).item()

        assert hqq_err <= mx_err * 1.05  # HQQ should be at least as good or within 5%

    def test_output_shapes(self):
        w = mx.random.normal(shape=(256, 128))
        cfg = HQQConfig(bits=4, group_size=64)
        packed, scales, biases = hqq_quantize_weight(w, cfg)
        # packed: [out, in // (32 // bits)] = [256, 128 // 8] = [256, 16]
        assert packed.shape == (256, 128 // (32 // cfg.bits))
        expected_groups = 128 // 64
        assert scales.shape == (256, expected_groups)
        assert biases.shape == (256, expected_groups)

    def test_different_group_sizes(self):
        w = mx.random.normal(shape=(128, 256))
        cfg = HQQConfig(bits=4, group_size=128)
        packed, scales, biases = hqq_quantize_weight(w, cfg)
        assert scales.shape == (128, 256 // 128)
        restored = mx.dequantize(packed, scales, biases, cfg.group_size, cfg.bits)
        assert restored.shape == w.shape

    def test_trainable_weights_not_quantized(self):
        w = mx.random.normal(shape=(64, 64))
        cfg = HQQConfig(bits=4)
        w_before = np.array(w).tolist()
        packed, scales, biases = hqq_quantize_weight(w, cfg)
        w_after = np.array(w).tolist()
        assert w_before == w_after

    def test_bits_4_range(self):
        w = mx.random.normal(shape=(64, 64))
        cfg = HQQConfig(bits=4)
        packed, scales, biases = hqq_quantize_weight(w, cfg)
        # Check that all scales > 0
        assert mx.all(scales > 0).item()


class TestHQQQuantizeLinear:
    def test_quantize_simple_linear(self):
        import mlx.nn as nn

        layer = nn.Linear(64, 32)
        mx.eval(layer.parameters())
        x = mx.random.normal(shape=(4, 64))
        orig_output = layer(x)

        cfg = HQQConfig(bits=4, group_size=64)
        hqq_layer = hqq_quantize_linear(layer, cfg)

        new_output = hqq_layer(x)
        assert new_output.shape == orig_output.shape
        assert new_output.dtype == orig_output.dtype

    def test_forward_after_quantize_works(self):
        import mlx.nn as nn

        layer = nn.Linear(128, 64)
        mx.eval(layer.parameters())
        x = mx.random.normal(shape=(8, 128))
        orig = layer(x)

        cfg = HQQConfig(bits=4, group_size=64)
        hqq_layer = hqq_quantize_linear(layer, cfg)

        mx.eval(hqq_layer.parameters())
        quantized = hqq_layer(x)
        assert quantized.shape == orig.shape

    def test_8bit_preserves_output_better(self):
        import mlx.nn as nn

        layer = nn.Linear(128, 64)
        mx.eval(layer.parameters())
        x = mx.random.normal(shape=(8, 128))
        orig = layer(x)

        cfg = HQQConfig(bits=8, group_size=64)
        hqq_layer = hqq_quantize_linear(layer, cfg)
        mx.eval(hqq_layer.parameters())

        quantized = hqq_layer(x)
        err = mx.mean(mx.abs(orig - quantized)).item()
        assert err < 0.1  # 8-bit should be very close

    def test_hqq_linear_with_bias(self):
        import mlx.nn as nn

        layer = nn.Linear(64, 32, bias=True)
        mx.eval(layer.parameters())
        x = mx.random.normal(shape=(4, 64))

        cfg = HQQConfig(bits=4, group_size=64)
        hqq_layer = hqq_quantize_linear(layer, cfg)
        out = hqq_layer(x)
        assert out.shape == (4, 32)

    def test_hqq_linear_without_bias(self):
        import mlx.nn as nn

        layer = nn.Linear(64, 32, bias=False)
        mx.eval(layer.parameters())
        x = mx.random.normal(shape=(4, 64))

        cfg = HQQConfig(bits=4, group_size=64)
        hqq_layer = hqq_quantize_linear(layer, cfg)
        out = hqq_layer(x)
        assert out.shape == (4, 32)
