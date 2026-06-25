"""Tests for the curated "known drafts" map (#513).

The map auto-fills the speculative draft for a known target when the user
enables speculative without naming a draft. Resolution precedence for the
draft model is:

    explicit per-model > global env > curated map > none

When the curated map supplies the draft, it also supplies the matching
strategy and block_size (those describe that specific draft), but an explicit
per-model strategy / token override still wins.

Precedence is verified by injecting entries into ``KNOWN_DRAFTS`` with
monkeypatch so the tests are independent of whatever the shipped map contains.
"""

import pytest

from olmlx.engine.registry import KnownDraft, ModelConfig, lookup_known_draft


@pytest.fixture
def curated(monkeypatch):
    """Inject a deterministic curated entry for ``target/model``."""
    from olmlx.engine import registry

    entry = KnownDraft(
        draft_repo="curated/draft",
        strategy="eagle",
        block_size=5,
        quant="4bit",
    )
    monkeypatch.setitem(registry.KNOWN_DRAFTS, "target/model", entry)
    return entry


def test_lookup_returns_none_for_unknown_target():
    assert lookup_known_draft("nobody/here") is None


def test_curated_fills_draft_when_none_configured(curated, monkeypatch):
    monkeypatch.setattr("olmlx.config.settings.speculative_draft_model", None)
    mc = ModelConfig(hf_path="target/model", speculative=True)
    resolved = mc.resolved_speculative()
    assert resolved.enabled is True
    assert resolved.draft_model == "curated/draft"
    # The map entry's strategy + block_size come along with the draft.
    assert resolved.strategy == "eagle"
    assert resolved.num_tokens == 5


def test_explicit_per_model_draft_beats_curated(curated):
    mc = ModelConfig(
        hf_path="target/model",
        speculative=True,
        speculative_draft_model="explicit/draft",
    )
    resolved = mc.resolved_speculative()
    assert resolved.draft_model == "explicit/draft"
    # The curated strategy must NOT leak in when the map was not the source.
    assert resolved.strategy == "classic"


def test_global_env_draft_beats_curated(curated, monkeypatch):
    monkeypatch.setattr("olmlx.config.settings.speculative", True)
    monkeypatch.setattr("olmlx.config.settings.speculative_draft_model", "global/draft")
    mc = ModelConfig(hf_path="target/model")
    resolved = mc.resolved_speculative()
    assert resolved.draft_model == "global/draft"
    assert resolved.strategy == "classic"


def test_per_model_strategy_beats_curated_strategy(curated, monkeypatch):
    monkeypatch.setattr("olmlx.config.settings.speculative_draft_model", None)
    mc = ModelConfig(
        hf_path="target/model",
        speculative=True,
        speculative_strategy="dflash",
    )
    resolved = mc.resolved_speculative()
    assert resolved.draft_model == "curated/draft"
    assert resolved.strategy == "dflash"


def test_per_model_tokens_beats_curated_block_size(curated, monkeypatch):
    monkeypatch.setattr("olmlx.config.settings.speculative_draft_model", None)
    monkeypatch.setattr("olmlx.config.settings.speculative_tokens", None)
    mc = ModelConfig(
        hf_path="target/model",
        speculative=True,
        speculative_tokens=12,
    )
    resolved = mc.resolved_speculative()
    assert resolved.draft_model == "curated/draft"
    assert resolved.num_tokens == 12


def test_global_tokens_beats_curated_block_size(curated, monkeypatch):
    monkeypatch.setattr("olmlx.config.settings.speculative_draft_model", None)
    monkeypatch.setattr("olmlx.config.settings.speculative_tokens", 9)
    mc = ModelConfig(hf_path="target/model", speculative=True)
    resolved = mc.resolved_speculative()
    assert resolved.draft_model == "curated/draft"
    assert resolved.num_tokens == 9


def test_curated_not_consulted_when_disabled(curated):
    mc = ModelConfig(hf_path="target/model", speculative=False)
    resolved = mc.resolved_speculative()
    assert resolved.enabled is False
    assert resolved.draft_model is None


def test_unknown_target_resolves_to_none(monkeypatch):
    monkeypatch.setattr("olmlx.config.settings.speculative_draft_model", None)
    mc = ModelConfig(hf_path="not/in-map", speculative=True)
    resolved = mc.resolved_speculative()
    assert resolved.draft_model is None


class TestShippedMapWellFormed:
    """Whatever entries ship in KNOWN_DRAFTS must be structurally valid."""

    def test_entries_well_formed(self):
        from olmlx.engine.registry import (
            _VALID_SPECULATIVE_STRATEGIES,
            KNOWN_DRAFTS,
        )

        for target, entry in KNOWN_DRAFTS.items():
            assert isinstance(entry, KnownDraft)
            # Keys and draft repos are "owner/repo" HF ids.
            assert target.count("/") == 1, target
            assert entry.draft_repo.count("/") == 1, entry.draft_repo
            assert entry.strategy in _VALID_SPECULATIVE_STRATEGIES
            # Draft-free strategies don't belong in a draft map.
            assert entry.strategy not in ("pld", "self_speculative")
            assert entry.block_size is None or (
                isinstance(entry.block_size, int) and entry.block_size > 0
            )
