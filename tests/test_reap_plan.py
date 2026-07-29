from __future__ import annotations

import numpy as np
import pytest

from olmlx.engine.reap.plan import (
    load_plan,
    plan_global,
    plan_graded,
    plan_uniform,
    save_plan,
)


def _saliency(L=3, E=8, seed=0):
    rng = np.random.RandomState(seed)
    return (
        rng.rand(L, E) * np.array([1.0, 10.0, 0.1])[:, None]
    )  # wildly different layer scales


class TestUniform:
    def test_keeps_top_k_per_layer_ascending(self):
        sal = np.array([[0.1, 0.9, 0.5, 0.7]])
        p = plan_uniform(sal, [2], 2, top_k=1, num_experts=4)
        assert p.keep == {2: [1, 3]}  # top-2 by saliency, ascending order
        assert p.mode == "uniform" and p.bank_high is None

    def test_rejects_keep_below_top_k(self):
        with pytest.raises(ValueError, match="top_k"):
            plan_uniform(_saliency(), [0, 1, 2], 1, top_k=2, num_experts=8)

    def test_rejects_keep_above_num_experts(self):
        with pytest.raises(ValueError):
            plan_uniform(_saliency(), [0, 1, 2], 9, top_k=2, num_experts=8)

    def test_identity_keep(self):
        p = plan_uniform(_saliency(), [0, 1, 2], 8, top_k=2, num_experts=8)
        assert all(p.keep[i] == list(range(8)) for i in [0, 1, 2])


class TestGlobal:
    def test_floor_enforced_despite_layer_scale(self):
        sal = _saliency()  # layer 2 has 100x smaller scores
        p = plan_global(sal, [0, 1, 2], 0.5, top_k=1, num_experts=8, floor_multiple=4)
        assert all(len(p.keep[i]) >= 4 for i in [0, 1, 2])  # floor = 4*top_k

    def test_normalization_makes_layers_comparable(self):
        # without per-layer normalization, layer 1 (10x scale) would swallow the budget
        sal = _saliency()
        p = plan_global(sal, [0, 1, 2], 0.5, top_k=1, num_experts=8, floor_multiple=1)
        sizes = [len(p.keep[i]) for i in [0, 1, 2]]
        assert max(sizes) - min(sizes) <= 4  # roughly balanced, not 8/2/2

    def test_total_budget(self):
        p = plan_global(
            _saliency(), [0, 1, 2], 0.5, top_k=1, num_experts=8, floor_multiple=1
        )
        assert sum(len(v) for v in p.keep.values()) == round(0.5 * 3 * 8)

    def test_deterministic(self):
        a = plan_global(_saliency(), [0, 1, 2], 0.5, top_k=1, num_experts=8)
        b = plan_global(_saliency(), [0, 1, 2], 0.5, top_k=1, num_experts=8)
        assert a.keep == b.keep


class TestGraded:
    def test_bank_high_is_subset_and_sized(self):
        p = plan_graded(
            _saliency(),
            [0, 1, 2],
            0.75,
            high_fraction=0.25,
            high_bits=8,
            low_bits=4,
            top_k=1,
            num_experts=8,
        )
        for layer, kept in p.keep.items():
            high = p.bank_high[layer]
            assert set(high) <= set(kept)
            assert len(high) == -(-len(kept) * 25 // 100)  # ceil(0.25*kept)
        assert (p.high_bits, p.low_bits) == (8, 4)

    def test_high_bank_has_top_saliency(self):
        sal = np.array([[0.1, 0.9, 0.5, 0.7, 0.2, 0.8, 0.3, 0.6]])
        p = plan_graded(
            sal,
            [0],
            1.0,
            high_fraction=0.25,
            high_bits=8,
            low_bits=4,
            top_k=1,
            num_experts=8,
        )
        assert set(p.bank_high[0]) == {1, 5}  # ceil(8*0.25)=2 top experts


class TestPlanIO:
    def test_roundtrip(self, tmp_path):
        p = plan_graded(
            _saliency(),
            [0, 1, 2],
            0.75,
            high_fraction=0.25,
            high_bits=8,
            low_bits=4,
            top_k=1,
            num_experts=8,
        )
        save_plan(p, tmp_path / "reap_plan.json")
        q = load_plan(tmp_path / "reap_plan.json")
        assert q.keep == p.keep and q.bank_high == p.bank_high
        assert (q.mode, q.top_k, q.num_experts_orig) == (
            p.mode,
            p.top_k,
            p.num_experts_orig,
        )
