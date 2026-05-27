"""Tests for the model tier table."""

from __future__ import annotations

from olmlx.bench.tier_table import (
    ABLATION_ANCHORS,
    CORE_ONLY,
    EXTENDED,
    Tier,
    tier_for,
)


class TestTierAssignment:
    def test_total_count_is_23(self):
        assert len(EXTENDED) + len(CORE_ONLY) == 23

    def test_no_overlap(self):
        assert EXTENDED.isdisjoint(CORE_ONLY)

    def test_extended_count_is_13(self):
        assert len(EXTENDED) == 13

    def test_core_only_count_is_10(self):
        assert len(CORE_ONLY) == 10

    def test_known_extended_member(self):
        assert "mlx-community/Qwen3-Coder-Next-4bit" in EXTENDED

    def test_known_core_only_member(self):
        assert "mlx-community/gpt-oss-120b-MXFP4-Q4" in CORE_ONLY

    def test_pure_drafts_are_core_only(self):
        for hf in (
            "mlx-community/Qwen3-0.6B-4bit",
            "mlx-community/Qwen3.5-0.8B-MLX-4bit",
            "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit",
        ):
            assert hf in CORE_ONLY


class TestTierFor:
    def test_extended(self):
        assert tier_for("mlx-community/Qwen3-Coder-Next-4bit") == Tier.EXTENDED

    def test_core_only(self):
        assert tier_for("mlx-community/gpt-oss-120b-MXFP4-Q4") == Tier.CORE_ONLY

    def test_unknown(self):
        assert tier_for("some/unknown-model") is None


class TestAblationAnchors:
    def test_has_turboquant_anchor(self):
        assert "mlx-community/Qwen3-Coder-Next-4bit" in ABLATION_ANCHORS["turboquant"]

    def test_has_speculative_anchor(self):
        assert "mlx-community/Qwen3.6-35B-A3B-4bit" in ABLATION_ANCHORS["speculative"]
