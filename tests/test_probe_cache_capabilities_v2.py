"""Tests for the updated _probe_cache_capabilities (checkpoint path)."""

from olmlx.engine.model_manager import LoadedModel


def test_loaded_model_defaults_uses_checkpoint_persistence_off():
    lm = LoadedModel(name="x", hf_path="y", model=None, tokenizer=None)
    assert lm.uses_checkpoint_persistence is False
