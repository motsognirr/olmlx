from __future__ import annotations

import json

import mlx.core as mx
import mlx.nn as nn
import pytest

from olmlx.engine.reap.apply import apply_plan
from olmlx.engine.reap.arch import STYLE_MINIMAX, find_moe_module
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


class _FakeExpertContainer(nn.Module):
    """Minimal SwitchGLU stand-in: only .gate_proj.weight.shape[0] matters to
    find_moe_module's expert-count probe -- it's never called."""

    def __init__(self, num_experts: int, hidden: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden, num_experts, bias=False)


class _FakeMiniMaxBlock(nn.Module):
    """Mirrors mlx_lm.models.minimax.MiniMaxSparseMoeBlock's attribute layout:
    linear .gate, block-level .e_score_correction_bias (NOT nested under the
    gate, unlike DeepSeek's MoEGate), and a switch_mlp expert container."""

    def __init__(self, hidden: int, num_experts: int, bias_init: list[float]) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden, num_experts, bias=False)
        self.e_score_correction_bias = mx.array(bias_init)
        self.switch_mlp = _FakeExpertContainer(num_experts, hidden)
        self.num_experts_per_tok = 2


class _FakeMiniMaxModel(nn.Module):
    def __init__(
        self,
        hidden: int = 8,
        num_experts: int = 4,
        bias_init: list[float] | None = None,
    ) -> None:
        super().__init__()
        bias_init = bias_init if bias_init is not None else [0.0] * num_experts

        class _Inner(nn.Module):
            def __init__(self) -> None:
                super().__init__()

                class _Layer(nn.Module):
                    def __init__(self) -> None:
                        super().__init__()
                        self.mlp = _FakeMiniMaxBlock(hidden, num_experts, bias_init)

                self.layers = [_Layer()]

        self.model = _Inner()


class TestMiniMaxMasking:
    """MiniMax selects off sigmoid(logits) + e_score_correction_bias (see
    arch.py's applied_scores docstring; mlx-lm minimax.py:174-188), while
    still *weighting* by the uncorrected sigmoid. A generic additive -inf on
    the raw gate logits (the mechanism used for plain linear-gate styles)
    washes out here: sigmoid(-inf) == 0 exactly, so a dropped expert with a
    large correction bias can still win selection. The fix masks the
    correction bias itself -- and MiniMax's bias lives directly on the MoE
    *block*, not nested under a separate gate module like DeepSeek's."""

    def test_dropped_experts_excluded_from_selection(self):
        mx.random.seed(21)
        hidden, num_experts = 8, 4
        # A deliberately large bias on a dropped index (1): under the old
        # gate-wrapping approach this would still win top-k selection over
        # legitimately kept experts despite sigmoid(-inf) == 0 for its own
        # logit -- exactly the wash-out bug this test guards against.
        bias_init = [2.0, 50.0, -3.0, 7.0]
        model = _FakeMiniMaxModel(hidden, num_experts, bias_init)
        block = model.model.layers[0].mlp

        info = find_moe_module(model.model.layers[0])
        assert info is not None
        assert info.style == STYLE_MINIMAX
        assert info.gate_attr == "gate"

        orig_gate = block.gate
        orig_bias = mx.array(bias_init)
        keep = [0, 2]  # drop 1 and 3

        restore = mask_dropped_experts(model, {0: keep})

        # the block-bias branch must have fired, not the generic gate-wrap:
        # the gate module itself is untouched.
        assert block.gate is orig_gate

        masked_bias = block.e_score_correction_bias
        assert float(masked_bias[1]) == float("-inf")
        assert float(masked_bias[3]) == float("-inf")
        assert mx.array_equal(masked_bias[mx.array(keep)], orig_bias[mx.array(keep)])

        # recompute MiniMax's actual selection math (minimax.py:178-188) and
        # confirm a dropped expert is never selected, for arbitrary input --
        # deterministic here since exactly `top_k` finite candidates remain.
        x = mx.random.normal((3, 5, hidden))
        logits = block.gate(x.astype(mx.float32))
        scores = mx.sigmoid(logits) + block.e_score_correction_bias
        inds = mx.argpartition(-scores, kth=block.num_experts_per_tok - 1, axis=-1)[
            ..., : block.num_experts_per_tok
        ]
        selected = set(inds.reshape(-1).tolist())
        assert selected.isdisjoint({1, 3})
        assert selected == {0, 2}

        restore()
        assert mx.array_equal(block.e_score_correction_bias, orig_bias)
