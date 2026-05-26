"""Tests for HQQ weight quantization engine."""

import mlx.core as mx
import numpy as np

from olmlx.engine.hqq.quantize import (
    HQQConfig,
    hqq_quantize_weight,
    hqq_quantize_linear,
    quantize_model,
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
        """HQQ should beat naive min/max quantization on reconstruction error."""
        mx.random.seed(42)
        w = mx.random.normal(shape=(128, 128))
        cfg = HQQConfig(bits=4, group_size=64, n_iters=3)

        # HQQ
        hqq_p, hqq_s, hqq_b = hqq_quantize_weight(w, cfg)
        hqq_r = mx.dequantize(hqq_p, hqq_s, hqq_b, cfg.group_size, cfg.bits)
        hqq_err = mx.mean(mx.abs(w - hqq_r)).item()

        # MLX native min/max affine quantization
        mx_p, mx_s, mx_b = mx.quantize(w, group_size=cfg.group_size, bits=cfg.bits)
        mx_r = mx.dequantize(mx_p, mx_s, mx_b, cfg.group_size, cfg.bits)
        mx_err = mx.mean(mx.abs(w - mx_r)).item()

        assert hqq_err <= mx_err * 1.05

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


class TestQuantizeModel:
    def test_nested_modules_are_replaced(self):
        import mlx.nn as nn

        class InnerBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(128, 64)

            def __call__(self, x):
                return self.linear(x)

        class OuterModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.block0 = InnerBlock()
                self.block1 = nn.Linear(64, 32)

            def __call__(self, x):
                return self.block1(self.block0(x))

        model = OuterModel()
        mx.eval(model.parameters())

        x = mx.random.normal(shape=(4, 128))
        out_before = model(x)

        cfg = HQQConfig(bits=4, group_size=64, skip_patterns=())
        quantize_model(model, cfg)

        from olmlx.engine.hqq.quantize import HQQLinear

        assert isinstance(model.block0.linear, HQQLinear)
        assert isinstance(model.block1, HQQLinear)

        out_after = model(x)
        assert out_after.shape == out_before.shape

    def test_skip_patterns_are_respected(self):
        import mlx.nn as nn

        class ModelWithLMHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.lm_head = nn.Linear(128, 256)
                self.ffn = nn.Linear(128, 128)

            def __call__(self, x):
                return self.lm_head(self.ffn(x))

        model = ModelWithLMHead()
        mx.eval(model.parameters())

        cfg = HQQConfig(bits=4, group_size=64)
        quantize_model(model, cfg)

        from olmlx.engine.hqq.quantize import HQQLinear

        assert not isinstance(model.lm_head, HQQLinear)
        assert isinstance(model.ffn, HQQLinear)

    def test_list_based_layers_are_replaced(self):
        import mlx.nn as nn

        class ListModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = [nn.Linear(128, 128) for _ in range(3)]

            def __call__(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        model = ListModel()
        mx.eval(model.parameters())
        x = mx.random.normal(shape=(4, 128))
        out_before = model(x)

        cfg = HQQConfig(bits=4, group_size=64, skip_patterns=())
        quantize_model(model, cfg)

        from olmlx.engine.hqq.quantize import HQQLinear

        for i, layer in enumerate(model.layers):
            assert isinstance(layer, HQQLinear), f"layers.{i} not replaced"

        out_after = model(x)
        assert out_after.shape == out_before.shape
