"""Config + registry resolution tests for proxy-tuning."""

from __future__ import annotations

import pytest

from olmlx.config import Settings
from olmlx.engine.registry import (
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
