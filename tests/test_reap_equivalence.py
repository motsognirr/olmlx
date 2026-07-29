from __future__ import annotations

import json

import mlx.core as mx
import pytest

from olmlx.engine.reap.apply import apply_plan
from olmlx.engine.reap.plan import ReapPlan
from olmlx.engine.reap.verify import mask_dropped_experts, max_logit_divergence
from tests.reap_factories import (
    make_tiny_deepseek_v3,
    make_tiny_gpt_oss,
    make_tiny_qwen3_moe,
    save_tiny_model,
    tiny_config_dict,
)

_CASES = [
    (make_tiny_qwen3_moe, "qwen3_moe", [0, 1]),
    (make_tiny_gpt_oss, "gpt_oss", [0, 1]),
    (make_tiny_deepseek_v3, "deepseek_v3", [1]),
]


def _reload_pruned(model_type, out_dir):
    import importlib

    module = importlib.import_module(f"mlx_lm.models.{model_type}")
    cfg = json.loads((out_dir / "config.json").read_text())
    args = module.ModelArgs.from_dict(cfg)
    model = module.Model(args)
    model.load_weights(str(out_dir / "model.safetensors"), strict=True)
    mx.eval(model.parameters())
    return model


def _batches():
    mx.random.seed(7)
    return [mx.random.randint(1, 100, (2, 16)) for _ in range(3)]


@pytest.mark.parametrize("factory,model_type,moe_layers", _CASES)
class TestPrunedEquivalence:
    """Invariant 2: pruned model == full model with dropped experts made unroutable."""

    def test_equivalence(self, tmp_path, factory, model_type, moe_layers):
        mx.random.seed(11)
        full, args = factory()
        keep = [0, 2, 4, 5, 6, 7]
        src = save_tiny_model(
            full, tiny_config_dict(args, model_type), tmp_path / "src"
        )
        plan = ReapPlan(
            "uniform",
            moe_layers,
            {layer_idx: keep for layer_idx in moe_layers},
            None,
            None,
            None,
            8,
            2,
        )
        pruned = _reload_pruned(model_type, apply_plan(src, plan, tmp_path / "out"))

        restore = mask_dropped_experts(
            full, {layer_idx: keep for layer_idx in moe_layers}
        )
        try:
            # remap: pruned token routing uses new indices; outputs must agree
            diff = max_logit_divergence(full, pruned, _batches())
        finally:
            restore()
        assert diff < 1e-3

    def test_restore_undoes_mask(self, tmp_path, factory, model_type, moe_layers):
        mx.random.seed(12)
        full, _ = factory()
        batch = _batches()[:1]
        ref = full(batch[0])
        restore = mask_dropped_experts(
            full, {layer_idx: [0, 1] for layer_idx in moe_layers}
        )
        restore()
        assert mx.allclose(full(batch[0]), ref, atol=1e-6)


@pytest.mark.parametrize("factory,model_type,moe_layers", _CASES)
class TestRotation:
    """Invariant 3: permuting router rows WITHOUT permuting experts must be detected —
    proves the equivalence check is not vacuous."""

    def test_rotated_router_diverges(self, tmp_path, factory, model_type, moe_layers):
        mx.random.seed(13)
        full, args = factory()
        keep = [0, 2, 4, 5, 6, 7]
        src = save_tiny_model(
            full, tiny_config_dict(args, model_type), tmp_path / "src"
        )
        plan = ReapPlan(
            "uniform",
            moe_layers,
            {layer_idx: keep for layer_idx in moe_layers},
            None,
            None,
            None,
            8,
            2,
        )
        out = apply_plan(src, plan, tmp_path / "out")

        # sabotage: rotate the pruned router rows by one position
        w = mx.load(str(out / "model.safetensors"))
        rotated = False
        for layer_idx in moe_layers:
            for cand in (
                f"model.layers.{layer_idx}.mlp.gate.weight",
                f"model.layers.{layer_idx}.mlp.router.weight",
            ):
                if cand in w:
                    perm = mx.array(list(range(1, w[cand].shape[0])) + [0])
                    w[cand] = w[cand][perm]
                    rotated = True
        assert rotated
        mx.eval(*w.values())
        mx.save_safetensors(str(out / "model.safetensors"), w)
        pruned = _reload_pruned(model_type, out)

        restore = mask_dropped_experts(
            full, {layer_idx: keep for layer_idx in moe_layers}
        )
        try:
            diff = max_logit_divergence(full, pruned, _batches())
        finally:
            restore()
        assert diff > 1e-2
