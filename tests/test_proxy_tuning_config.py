"""Config + registry resolution tests for proxy-tuning."""

from __future__ import annotations

import pytest

from olmlx.config import Settings
from olmlx.engine.registry import (
    _FLASH_MOE_INCOMPATIBLE_STRATEGIES,
    _VALID_SPECULATIVE_STRATEGIES,
    ModelConfig,
)


def test_proxy_tuning_is_valid_strategy():
    assert "proxy_tuning" in _VALID_SPECULATIVE_STRATEGIES


def test_settings_accepts_proxy_strategy_and_models():
    s = Settings(
        speculative=True,
        speculative_strategy="proxy_tuning",
        speculative_proxy_expert_model="org/expert",
        speculative_proxy_antiexpert_model="org/anti",
        speculative_proxy_alpha=1.5,
    )
    assert s.speculative_strategy == "proxy_tuning"
    assert s.speculative_proxy_expert_model == "org/expert"
    assert s.speculative_proxy_antiexpert_model == "org/anti"
    assert s.speculative_proxy_alpha == 1.5


def test_proxy_strategy_requires_both_models():
    with pytest.raises(ValueError, match="proxy"):
        Settings(
            speculative=True,
            speculative_strategy="proxy_tuning",
            speculative_proxy_expert_model="org/expert",
            # antiexpert missing
        )


def test_proxy_alpha_default_is_one():
    s = Settings()
    assert s.speculative_proxy_alpha == 1.0


def test_proxy_alpha_rejects_non_finite():
    with pytest.raises(ValueError, match="finite"):
        Settings(
            speculative=True,
            speculative_strategy="proxy_tuning",
            speculative_proxy_expert_model="org/expert",
            speculative_proxy_antiexpert_model="org/anti",
            speculative_proxy_alpha=float("inf"),
        )


def test_proxy_strategy_requires_expert_when_only_antiexpert_set():
    with pytest.raises(ValueError, match="speculative_proxy_expert_model"):
        Settings(
            speculative=True,
            speculative_strategy="proxy_tuning",
            speculative_proxy_antiexpert_model="org/anti",
        )


def test_resolved_speculative_carries_proxy_fields(monkeypatch):
    from olmlx import config as config_mod

    # Set proxy fields before strategy to avoid triggering the model-validator
    # (validate_assignment=True fires on each setattr; strategy must come last
    # so the validator sees the expert/anti models already in place).
    monkeypatch.setattr(
        config_mod.settings,
        "speculative_proxy_expert_model",
        "org/expert",
        raising=False,
    )
    monkeypatch.setattr(
        config_mod.settings,
        "speculative_proxy_antiexpert_model",
        "org/anti",
        raising=False,
    )
    monkeypatch.setattr(
        config_mod.settings, "speculative_proxy_alpha", 1.25, raising=False
    )
    monkeypatch.setattr(config_mod.settings, "speculative", True, raising=False)
    monkeypatch.setattr(
        config_mod.settings, "speculative_strategy", "proxy_tuning", raising=False
    )

    mc = ModelConfig(hf_path="org/base")
    resolved = mc.resolved_speculative()
    assert resolved.strategy == "proxy_tuning"
    assert resolved.proxy_expert_model == "org/expert"
    assert resolved.proxy_antiexpert_model == "org/anti"
    assert resolved.proxy_alpha == 1.25


def test_proxy_tuning_incompatible_with_flash_moe():
    assert "proxy_tuning" in _FLASH_MOE_INCOMPATIBLE_STRATEGIES
