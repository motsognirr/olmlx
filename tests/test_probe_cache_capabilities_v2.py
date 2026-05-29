"""Tests for the updated _probe_cache_capabilities (checkpoint path)."""

import asyncio

import pytest
from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache

from olmlx.engine.model_manager import LoadedModel, ModelManager


def test_loaded_model_defaults_uses_checkpoint_persistence_off():
    lm = LoadedModel(name="x", hf_path="y", model=None, tokenizer=None)
    assert lm.uses_checkpoint_persistence is False


def _fake_cache_with(*layer_types) -> list:
    """Stub: returns a cache list of the given classes (no model needed)."""
    return [t.__new__(t) for t in layer_types]


@pytest.mark.parametrize(
    "layers, expected_ckpt, expected_persist",
    [
        ([KVCache], False, True),  # pure trimmable: flat
        ([RotatingKVCache], True, True),  # sliding-window: ckpt
        ([ArraysCache], True, True),  # SSM: ckpt
        ([KVCache, ArraysCache], True, True),  # Qwen3.5 mix: ckpt
        ([KVCache, RotatingKVCache], True, True),  # Gemma 3 mix: ckpt
        ([RotatingKVCache, ArraysCache], False, False),  # Qwen3-Next: excluded
    ],
)
def test_probe_sets_uses_checkpoint_for_non_trimmable_only(
    layers, expected_ckpt, expected_persist, monkeypatch
):
    lm = LoadedModel(name="x", hf_path="y", model=None, tokenizer=None)
    cache = _fake_cache_with(*layers)
    # The probe does a local `from mlx_lm.models.cache import make_prompt_cache`
    # inside the try block, so we must patch the source module directly.
    monkeypatch.setattr(
        "mlx_lm.models.cache.make_prompt_cache",
        lambda model: cache,
    )
    mgr = ModelManager.__new__(ModelManager)
    # The probe's finally block checks self._pending_cleanups; set it so the
    # bare __new__ instance doesn't raise AttributeError.  A non-empty dict
    # skips the _flush_metal await — fine here since we have no Metal device.
    mgr._pending_cleanups = {"skip_metal_flush": True}
    asyncio.run(mgr._probe_cache_capabilities(lm))
    assert lm.uses_checkpoint_persistence is expected_ckpt, (
        f"layers={[t.__name__ for t in layers]}: "
        f"want uses_checkpoint={expected_ckpt}, got {lm.uses_checkpoint_persistence}"
    )
    assert lm.supports_cache_persistence is expected_persist, (
        f"layers={[t.__name__ for t in layers]}: "
        f"want supports_persist={expected_persist}, got {lm.supports_cache_persistence}"
    )
