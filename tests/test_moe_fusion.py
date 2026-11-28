"""Load-time fusion of MoE expert gate/up projections (moe_fusion.py).

mlx-lm's ``SwitchGLU`` runs the routed experts' ``gate_proj`` and ``up_proj``
as two ``gather_qmm`` calls over the same input. Stacking the two weight
tensors along the output axis gives one call whose rows are the same
independent dot products, so the fused block must be bit-identical to the
original, not merely close.
"""

import threading

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models.switch_layers import (
    QuantizedSwitchLinear,
    SwitchGLU,
    SwitchLinear,
)

from olmlx.engine.moe_fusion import (
    FusedSwitchGLU,
    fuse_switch_glu_layers,
    unfuse_switch_glu_layers,
)

D, H, E, K = 128, 64, 8, 2


def _glu(*, quantized=True, bias=False, activation=None, bits=4, mode="affine"):
    mx.random.seed(0)
    kwargs = {"bias": bias}
    if activation is not None:
        kwargs["activation"] = activation
    glu = SwitchGLU(D, H, E, **kwargs)
    if bias:
        # Non-zero biases, so a bias that is dropped or misaligned shows up.
        for proj in (glu.gate_proj, glu.up_proj, glu.down_proj):
            proj.bias = mx.random.normal(proj.bias.shape) * 0.1
    if quantized:
        if mode == "affine":
            nn.quantize(glu, group_size=64, bits=bits)
        else:
            nn.quantize(glu, mode=mode)
    mx.eval(glu.parameters())
    return glu


class _Block(nn.Module):
    def __init__(self, glu):
        super().__init__()
        self.switch_mlp = glu


class _Model(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.layers = blocks


def _inputs(tokens):
    x = mx.random.normal((1, tokens, D))
    inds = mx.random.randint(0, E, (1, tokens, K)).astype(mx.uint32)
    mx.eval(x, inds)
    return x, inds


def _fuse_one(glu):
    model = _Model([_Block(glu)])
    n = fuse_switch_glu_layers(model)
    return n, model.layers[0].switch_mlp


def _forward_in_thread(module, x, inds):
    """Run and eval *module* on a fresh thread (as the generation worker does)."""

    def fn():
        y = module(x, inds)
        mx.eval(y)
        return y

    return _run_in_thread(fn)


def _run_in_thread(fn):
    out = {}

    def target():
        try:
            out["value"] = fn()
        except Exception as exc:  # noqa: BLE001 - surfaced to the assertion
            out["error"] = exc

    t = threading.Thread(target=target)
    t.start()
    t.join()
    return out


class TestFusedOutputIsBitIdentical:
    @pytest.mark.parametrize(
        "tokens",
        [1, 40],  # 1 token: decode; 40 tokens x K=2 = 80 routes >= 64: sorted path
        ids=["decode", "prefill-sorted"],
    )
    def test_quantized(self, tokens):
        glu = _glu()
        x, inds = _inputs(tokens)
        ref = glu(x, inds)
        n, fused = _fuse_one(glu)
        assert n == 1
        assert isinstance(fused, FusedSwitchGLU)
        assert mx.array_equal(fused(x, inds), ref).item()

    @pytest.mark.parametrize("tokens", [1, 64], ids=["decode", "prefill-sorted"])
    def test_real_model_shape(self, tokens):
        # Qwen3.6-35B-A3B expert shape (hidden 2048, expert width 512, top-8,
        # 4-bit/64). The fused projection is twice as wide; this pins that
        # MLX still computes each row identically at a production width.
        # Checked on Metal for more shapes (Qwen3-30B, GLM-4.5-Air,
        # DeepSeek-V3, gpt-oss, Mixtral, widths 32..14336): all bit-exact.
        mx.random.seed(0)
        glu = SwitchGLU(2048, 512, 8)
        nn.quantize(glu, group_size=64, bits=4)
        mx.eval(glu.parameters())
        x = mx.random.normal((1, tokens, 2048)).astype(mx.bfloat16)
        inds = mx.random.randint(0, 8, (1, tokens, 8)).astype(mx.uint32)
        ref = glu(x, inds)
        _, fused = _fuse_one(glu)
        assert mx.array_equal(fused(x, inds), ref).item()

    def test_mxfp4_with_bias(self):
        # gpt-oss: mxfp4 experts (no affine biases, uint8 scales) plus
        # per-expert bias vectors.
        glu = _glu(bias=True, mode="mxfp4")
        assert glu.gate_proj.mode == "mxfp4"
        x, inds = _inputs(3)
        ref = glu(x, inds)
        _, fused = _fuse_one(glu)
        assert isinstance(fused, FusedSwitchGLU)
        assert mx.array_equal(fused(x, inds), ref).item()

    def test_unquantized(self):
        glu = _glu(quantized=False)
        x, inds = _inputs(3)
        ref = glu(x, inds)
        _, fused = _fuse_one(glu)
        assert isinstance(fused, FusedSwitchGLU)
        assert mx.array_equal(fused(x, inds), ref).item()

    def test_bias_and_custom_activation_are_kept(self):
        # gpt-oss style: expert biases plus a non-standard activation.
        class ClampedSwiGLU(nn.Module):
            def __call__(self, x, gate):
                return nn.silu(mx.clip(gate, -1.0, 1.0)) * x

        glu = _glu(bias=True, activation=ClampedSwiGLU())
        x, inds = _inputs(3)
        ref = glu(x, inds)
        _, fused = _fuse_one(glu)
        assert isinstance(fused, FusedSwitchGLU)
        assert mx.array_equal(fused(x, inds), ref).item()


class TestFuseSwitchGluLayers:
    def test_replaces_every_block_and_returns_count(self):
        blocks = [_Block(_glu()) for _ in range(3)]
        model = _Model(blocks)
        assert fuse_switch_glu_layers(model) == 3
        assert all(isinstance(b.switch_mlp, FusedSwitchGLU) for b in blocks)

    def test_gate_and_up_are_removed(self):
        _, fused = _fuse_one(_glu())
        assert "gate_proj" not in fused and "up_proj" not in fused

    def test_skips_switch_glu_subclass(self):
        # A subclass may override __call__ with different semantics.
        class CustomGLU(SwitchGLU):
            pass

        glu = CustomGLU(D, H, E)
        model = _Model([_Block(glu)])
        assert fuse_switch_glu_layers(model) == 0
        assert model.layers[0].switch_mlp is glu

    def test_skips_mismatched_quantization(self):
        glu = _glu()
        up8 = _glu(bits=8).up_proj
        glu.up_proj = up8
        model = _Model([_Block(glu)])
        assert fuse_switch_glu_layers(model) == 0
        assert model.layers[0].switch_mlp is glu

    def test_non_module_is_a_noop(self):
        assert fuse_switch_glu_layers(object()) == 0

    def test_fused_weights_are_usable_from_another_thread(self):
        # The fusion runs on the model-load worker thread and generation runs
        # on another one. A lazy concatenate would stay bound to the load
        # thread's stream (mlx thread-local streams, #499) and crash the first
        # forward with "There is no Stream(...) in current thread".
        glu = _glu()
        x, inds = _inputs(1)
        ref = glu(x, inds)
        mx.eval(ref)
        _, fused = _fuse_one(glu)
        res = _forward_in_thread(fused, x, inds)
        assert res.get("error") is None, res.get("error")
        assert mx.array_equal(res["value"], ref).item()


class TestUnfuse:
    def test_restores_switch_glu_with_identical_output(self):
        glu = _glu()
        x, inds = _inputs(3)
        ref = glu(x, inds)
        model = _Model([_Block(glu)])
        assert fuse_switch_glu_layers(model) == 1
        assert unfuse_switch_glu_layers(model) == 1
        restored = model.layers[0].switch_mlp
        assert type(restored) is SwitchGLU
        assert type(restored.gate_proj) is QuantizedSwitchLinear
        assert type(restored.up_proj) is QuantizedSwitchLinear
        assert mx.array_equal(restored(x, inds), ref).item()

    def test_restored_projections_match_original_parameters(self):
        # LoRA adapters address switch_mlp.gate_proj / up_proj by name and
        # shape; full fine-tunes replace those weights outright.
        glu = _glu(quantized=False)
        gate_w = glu.gate_proj.weight
        up_w = glu.up_proj.weight
        model = _Model([_Block(glu)])
        assert fuse_switch_glu_layers(model) == 1
        assert unfuse_switch_glu_layers(model) == 1
        restored = model.layers[0].switch_mlp
        assert type(restored.gate_proj) is SwitchLinear
        assert mx.array_equal(restored.gate_proj.weight, gate_w).item()
        assert mx.array_equal(restored.up_proj.weight, up_w).item()

    @pytest.mark.parametrize("quantized", [False, True], ids=["fp32", "q4"])
    def test_restored_projections_share_the_fused_buffers(self, quantized):
        # An adapter on an MoE base would otherwise duplicate the whole
        # expert gate/up weight set (multi-GB). If MLX ever starts copying
        # basic slices, this must fail rather than silently use the memory.
        import numpy as np

        model = _Model([_Block(_glu(quantized=quantized))])
        assert fuse_switch_glu_layers(model) == 1
        fused_w = model.layers[0].switch_mlp.gate_up_proj.weight
        assert unfuse_switch_glu_layers(model) == 1
        restored = model.layers[0].switch_mlp
        for proj in (restored.gate_proj, restored.up_proj):
            assert np.shares_memory(np.asarray(fused_w), np.asarray(proj.weight))

    def test_restored_projections_are_usable_from_another_thread(self):
        glu = _glu()
        x, inds = _inputs(1)
        ref = glu(x, inds)
        mx.eval(ref)
        model = _Model([_Block(glu)])
        assert fuse_switch_glu_layers(model) == 1
        assert unfuse_switch_glu_layers(model) == 1
        restored = model.layers[0].switch_mlp
        res = _forward_in_thread(restored, x, inds)
        assert res.get("error") is None, res.get("error")
        assert mx.array_equal(res["value"], ref).item()


class TestAdapterOnFusedBase:
    """LoRA adapters address switch_mlp.gate_proj / up_proj by name.

    ``load_adapters`` loads with ``strict=False``, so on a fused base an
    adapter's expert weights would be dropped without any error.
    ``_build_adapter_model`` must hand the adapter the unfused layout.
    """

    def _write_switch_adapter(self, base, adapter_dir):
        import json

        from mlx.utils import tree_flatten
        from mlx_lm.tuner.utils import linear_to_lora_layers

        from olmlx.engine.loaded_model import structural_copy

        lora_params = {
            "rank": 4,
            "scale": 10.0,
            "dropout": 0.0,
            "keys": ["switch_mlp.gate_proj"],
        }
        donor = structural_copy(base)
        linear_to_lora_layers(donor, 1, lora_params)
        mx.random.seed(1)
        trained = {
            k: mx.random.normal(v.shape) * 0.05
            for k, v in tree_flatten(donor.trainable_parameters())
        }
        assert any("switch_mlp.gate_proj.lora_b" in k for k in trained)
        adapter_dir.mkdir()
        mx.save_safetensors(str(adapter_dir / "adapters.safetensors"), trained)
        (adapter_dir / "adapter_config.json").write_text(
            json.dumps(
                {
                    "fine_tune_type": "lora",
                    "num_layers": 1,
                    "lora_parameters": lora_params,
                }
            )
        )

    def test_expert_lora_applies_on_fused_base(self, tmp_path):
        from mlx_lm.tuner.utils import load_adapters

        from olmlx.engine.loaded_model import structural_copy
        from olmlx.engine.model_manager import ModelManager

        base = _Model([_Block(_glu())])
        adapter_dir = tmp_path / "adapter"
        self._write_switch_adapter(base, adapter_dir)
        x, inds = _inputs(3)
        base_out = base.layers[0].switch_mlp(x, inds)
        expected_model = load_adapters(structural_copy(base), str(adapter_dir))
        expected = expected_model.layers[0].switch_mlp(x, inds)
        assert not mx.array_equal(expected, base_out).item(), "adapter is a no-op"

        assert fuse_switch_glu_layers(base) == 1
        adapter_model = ModelManager._build_adapter_model(base, str(adapter_dir))

        got = adapter_model.layers[0].switch_mlp(x, inds)
        assert mx.array_equal(got, expected).item()
        # The base keeps serving fused and unchanged.
        assert isinstance(base.layers[0].switch_mlp, FusedSwitchGLU)
        assert mx.array_equal(base.layers[0].switch_mlp(x, inds), base_out).item()
