from __future__ import annotations

import json

import mlx.core as mx
import numpy as np
import pytest

from olmlx.engine.reap.apply import (
    ReapApplyError,
    apply_plan,
    rewrite_config_num_experts,
)
from olmlx.engine.reap.plan import ReapPlan
from tests.reap_factories import (
    make_tiny_deepseek_v3,
    make_tiny_gpt_oss,
    make_tiny_qwen3_moe,
    save_tiny_model,
    tiny_config_dict,
)


def _uniform_plan(moe_layers, keep, num_experts=8, top_k=2):
    return ReapPlan(
        "uniform",
        list(moe_layers),
        {layer: list(keep) for layer in moe_layers},
        None,
        None,
        None,
        num_experts,
        top_k,
    )


def _save(factory, model_type, tmp_path, name="src"):
    model, args = factory()
    cfg = tiny_config_dict(args, model_type)
    return save_tiny_model(model, cfg, tmp_path / name), model


class TestIdentityNoOp:
    """Invariant 1: identity prune is bit-for-bit a no-op."""

    @pytest.mark.parametrize(
        "factory,model_type,moe_layers",
        [
            (make_tiny_qwen3_moe, "qwen3_moe", [0, 1]),
            (make_tiny_gpt_oss, "gpt_oss", [0, 1]),
            (make_tiny_deepseek_v3, "deepseek_v3", [1]),
        ],
    )
    def test_identity(self, tmp_path, factory, model_type, moe_layers):
        src, _ = _save(factory, model_type, tmp_path)
        plan = _uniform_plan(moe_layers, list(range(8)))
        out = apply_plan(src, plan, tmp_path / "out")
        a = mx.load(str(src / "model.safetensors"))
        b = mx.load(str(out / "model.safetensors"))
        assert set(a) == set(b)
        for k in a:
            assert a[k].dtype == b[k].dtype, k
            assert mx.array_equal(a[k], b[k]), k
        cfg_a = json.loads((src / "config.json").read_text())
        cfg_b = json.loads((out / "config.json").read_text())
        assert cfg_a == cfg_b


class TestUniformPrune:
    def test_shapes_config_and_router_rows(self, tmp_path):
        src, model = _save(make_tiny_qwen3_moe, "qwen3_moe", tmp_path)
        keep = [0, 2, 5, 7]
        plan = _uniform_plan([0, 1], keep)
        out = apply_plan(src, plan, tmp_path / "out")
        w = mx.load(str(out / "model.safetensors"))
        orig = mx.load(str(src / "model.safetensors"))
        for layer in (0, 1):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                key = f"model.layers.{layer}.mlp.switch_mlp.{proj}.weight"
                assert w[key].shape[0] == 4
                assert mx.array_equal(w[key], orig[key][mx.array(keep)])
            gate = f"model.layers.{layer}.mlp.gate.weight"
            assert w[gate].shape[0] == 4
            assert mx.array_equal(w[gate], orig[gate][mx.array(keep)])
        cfg = json.loads((out / "config.json").read_text())
        assert cfg["num_experts"] == 4

    def test_gpt_oss_router_bias_and_expert_bias(self, tmp_path):
        src, _ = _save(make_tiny_gpt_oss, "gpt_oss", tmp_path)
        keep = [1, 3, 4, 6]
        out = apply_plan(src, _uniform_plan([0, 1], keep), tmp_path / "out")
        w = mx.load(str(out / "model.safetensors"))
        orig = mx.load(str(src / "model.safetensors"))
        rb = "model.layers.0.mlp.router.bias"
        assert mx.array_equal(w[rb], orig[rb][mx.array(keep)])
        eb = "model.layers.0.mlp.experts.gate_proj.bias"
        assert w[eb].shape[0] == 4
        cfg = json.loads((out / "config.json").read_text())
        assert cfg["num_local_experts"] == 4

    def test_deepseek_bias_and_shared_experts_untouched(self, tmp_path):
        src, _ = _save(make_tiny_deepseek_v3, "deepseek_v3", tmp_path)
        keep = [0, 1, 2, 3, 4, 5]
        out = apply_plan(src, _uniform_plan([1], keep), tmp_path / "out")
        w = mx.load(str(out / "model.safetensors"))
        orig = mx.load(str(src / "model.safetensors"))
        bias = "model.layers.1.mlp.gate.e_score_correction_bias"
        assert w[bias].shape[0] == 6
        shared = [k for k in orig if "shared_experts" in k]
        assert shared, "factory should produce shared experts"
        for k in shared:
            assert mx.array_equal(w[k], orig[k]), k
        cfg = json.loads((out / "config.json").read_text())
        assert cfg["n_routed_experts"] == 6

    def test_provenance_and_non_weight_copy(self, tmp_path):
        src, _ = _save(make_tiny_qwen3_moe, "qwen3_moe", tmp_path)
        (src / "tokenizer_config.json").write_text("{}")
        out = apply_plan(src, _uniform_plan([0, 1], [0, 1, 2, 3]), tmp_path / "out")
        prov = json.loads((out / "reap_provenance.json").read_text())
        assert prov["keep_count"] == 4 and prov["num_experts_orig"] == 8
        assert (out / "tokenizer_config.json").exists()

    def test_pruned_model_reloads_and_runs(self, tmp_path):
        from mlx_lm.models import qwen3_moe

        src, _ = _save(make_tiny_qwen3_moe, "qwen3_moe", tmp_path)
        out = apply_plan(src, _uniform_plan([0, 1], [0, 2, 5, 7]), tmp_path / "out")
        cfg = json.loads((out / "config.json").read_text())
        args = qwen3_moe.ModelArgs.from_dict(cfg)
        pruned = qwen3_moe.Model(args)
        pruned.load_weights(str(out / "model.safetensors"), strict=True)
        logits = pruned(mx.array([[1, 2, 3]]))
        assert logits.shape == (1, 3, cfg["vocab_size"])


class TestGuards:
    def test_rejects_non_uniform_plan(self, tmp_path):
        src, _ = _save(make_tiny_qwen3_moe, "qwen3_moe", tmp_path)
        plan = ReapPlan(
            "global", [0, 1], {0: [0, 1], 1: [0, 1]}, None, None, None, 8, 2
        )
        with pytest.raises(ReapApplyError, match="phase 3"):
            apply_plan(src, plan, tmp_path / "out")

    def test_rejects_top_k_above_keep(self, tmp_path):
        src, _ = _save(make_tiny_qwen3_moe, "qwen3_moe", tmp_path)
        plan = ReapPlan("uniform", [0, 1], {0: [0], 1: [0]}, None, None, None, 8, 2)
        with pytest.raises(ReapApplyError, match="num_experts_per_tok"):
            apply_plan(src, plan, tmp_path / "out")

    def test_unknown_expert_sized_tensor_is_error(self, tmp_path):
        """The misrouting trap: an unrecognized E-sized tensor under the MoE prefix
        must be a hard error, never a silent copy-through."""
        src, model = _save(make_tiny_qwen3_moe, "qwen3_moe", tmp_path)
        w = mx.load(str(src / "model.safetensors"))
        w["model.layers.0.mlp.mystery_bias"] = mx.zeros((8,))
        mx.save_safetensors(str(src / "model.safetensors"), w)
        with pytest.raises(ReapApplyError, match="mystery_bias"):
            apply_plan(src, _uniform_plan([0, 1], [0, 1, 2, 3]), tmp_path / "out")


class TestQuantizedSlicing:
    def test_quantized_triple_sliced_together(self, tmp_path):
        """Packed uint32 weight + scales + biases all slice on axis 0."""
        from safetensors.numpy import save_file

        E, out_dim, in_dim, gs = 8, 16, 32, 32
        d = tmp_path / "qsrc"
        d.mkdir()
        rng = np.random.RandomState(0)
        tensors = {}
        for proj in ("gate_proj", "up_proj", "down_proj"):
            base = f"model.layers.0.mlp.switch_mlp.{proj}"
            tensors[f"{base}.weight"] = rng.randint(
                0, 2**31, (E, out_dim, in_dim * 4 // 32), dtype=np.uint32
            )
            tensors[f"{base}.scales"] = rng.rand(E, out_dim, in_dim // gs).astype(
                np.float16
            )
            tensors[f"{base}.biases"] = rng.rand(E, out_dim, in_dim // gs).astype(
                np.float16
            )
        tensors["model.layers.0.mlp.gate.weight"] = rng.rand(E, 64).astype(np.float16)
        save_file(tensors, str(d / "model.safetensors"))
        (d / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "qwen3_moe",
                    "num_experts": E,
                    "num_experts_per_tok": 2,
                    "num_hidden_layers": 1,
                    "hidden_size": 64,
                    "quantization": {"bits": 4, "group_size": gs},
                }
            )
        )
        keep = [0, 3, 6]
        out = apply_plan(d, _uniform_plan([0], keep, num_experts=E), tmp_path / "qout")
        w = mx.load(str(out / "model.safetensors"))
        for proj in ("gate_proj", "up_proj", "down_proj"):
            for comp in ("weight", "scales", "biases"):
                key = f"model.layers.0.mlp.switch_mlp.{proj}.{comp}"
                assert w[key].shape[0] == 3, key
                assert w[key].dtype == mx.load(str(d / "model.safetensors"))[key].dtype


class TestShardedIndex:
    def test_multi_shard_with_index(self, tmp_path):
        src, _ = _save(make_tiny_qwen3_moe, "qwen3_moe", tmp_path)
        # split the single file into two shards + index
        w = mx.load(str(src / "model.safetensors"))
        keys = sorted(w)
        half = len(keys) // 2
        s1 = {k: w[k] for k in keys[:half]}
        s2 = {k: w[k] for k in keys[half:]}
        mx.save_safetensors(str(src / "model-00001-of-00002.safetensors"), s1)
        mx.save_safetensors(str(src / "model-00002-of-00002.safetensors"), s2)
        (src / "model.safetensors").unlink()
        index = {
            "metadata": {"total_size": 0},
            "weight_map": {k: "model-00001-of-00002.safetensors" for k in keys[:half]},
        }
        index["weight_map"].update(
            {k: "model-00002-of-00002.safetensors" for k in keys[half:]}
        )
        (src / "model.safetensors.index.json").write_text(json.dumps(index))

        out = apply_plan(src, _uniform_plan([0, 1], [0, 1, 2, 3]), tmp_path / "out")
        new_index = json.loads((out / "model.safetensors.index.json").read_text())
        assert set(new_index["weight_map"]) == set(keys)
        shard_files = set(new_index["weight_map"].values())
        assert len(shard_files) == 2
        assert new_index["metadata"]["total_size"] == sum(
            (out / f).stat().st_size for f in shard_files
        )
        merged = {}
        for shard in set(new_index["weight_map"].values()):
            merged.update(mx.load(str(out / shard)))
        assert merged["model.layers.0.mlp.switch_mlp.gate_proj.weight"].shape[0] == 4


class TestRewriteConfig:
    def test_alias_probing_order(self):
        cfg = {"num_experts": 8, "num_experts_per_tok": 2}
        assert rewrite_config_num_experts(cfg, 4) == "num_experts"
        assert cfg["num_experts"] == 4

    def test_text_config_unwrap(self):
        cfg = {"text_config": {"n_routed_experts": 8}}
        assert rewrite_config_num_experts(cfg, 4) == "n_routed_experts"
        assert cfg["text_config"]["n_routed_experts"] == 4
