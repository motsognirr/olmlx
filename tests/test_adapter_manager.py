"""LoRA-adapter loading, base-pinning, and teardown in ModelManager (issue #362).

These are non-Metal unit tests: the structural copy is exercised on toy
``nn.Module`` trees, and the manager's pinning/eviction logic is driven with
fake ``LoadedModel`` entries (the real model build is monkeypatched). The
end-to-end real-model path lives in ``test_adapter_loading.py`` (Metal-gated).
"""

import json
from unittest.mock import MagicMock

import mlx.core as mx
import mlx.nn as nn
import pytest

from olmlx.engine.model_manager import LoadedModel, ModelManager, structural_copy
from olmlx.engine.registry import ModelRegistry


# --------------------------------------------------------------------------- #
# structural_copy
# --------------------------------------------------------------------------- #
class _Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)


class _Net(nn.Module):
    def __init__(self, d, n):
        super().__init__()
        self.layers = [_Block(d) for _ in range(n)]


class TestStructuralCopy:
    def test_shares_weight_arrays_but_isolates_structure(self):
        base = _Net(8, 3)
        mx.eval(base.parameters())
        cp = structural_copy(base)

        # New container/module objects...
        assert cp is not base
        assert cp.layers is not base.layers
        assert cp.layers[0] is not base.layers[0]
        # ...but the actual weight array object is shared (no memory dup).
        assert dict.__getitem__(cp.layers[0].q_proj, "weight") is dict.__getitem__(
            base.layers[0].q_proj, "weight"
        )

    def test_isolates_submodule_replacement(self):
        base = _Net(8, 2)
        cp = structural_copy(base)
        # Simulate load_adapters replacing a submodule on the copy only.
        cp.layers[0].q_proj = nn.Linear(8, 8, bias=False)
        assert type(base.layers[0].q_proj).__name__ == "Linear"
        assert cp.layers[0].q_proj is not base.layers[0].q_proj

    def test_preserves_scalar_attrs(self):
        q = nn.QuantizedLinear(64, 64, bias=False, group_size=64, bits=4)
        cp = structural_copy(q)
        assert cp.bits == 4
        assert cp.group_size == 64

    def test_freeze_state_not_shared(self):
        base = _Net(8, 1)
        cp = structural_copy(base)
        assert (
            cp.layers[0].__dict__["_no_grad"]
            is not (base.layers[0].__dict__["_no_grad"])
        )

    def test_shared_submodule_copied_once(self):
        # A module instance referenced from two places (not a strict tree) must
        # be copied a single time and re-shared, not duplicated or recursed.
        shared = nn.Linear(8, 8, bias=False)

        class Twin(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = shared
                self.b = shared

        cp = structural_copy(Twin())
        assert cp.a is cp.b
        assert cp.a is not shared


# --------------------------------------------------------------------------- #
# Manager: base-pinning + child-ref accounting
# --------------------------------------------------------------------------- #
@pytest.fixture
def adapter_manager(tmp_path, monkeypatch):
    """Manager + registry with a base preloaded and one adapter registered."""
    config = {
        "qwen3-8b": "Qwen/Qwen3-8B-MLX",
        "adapters": {
            "qwen3-8b:my-lora": {"base": "qwen3-8b", "hf_path": "acme/my-lora"},
        },
    }
    config_path = tmp_path / "models.json"
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr("olmlx.engine.registry.settings.models_config", config_path)
    # Adapters pin their base, so a base + N adapters needs N+1 slots.
    monkeypatch.setattr("olmlx.engine.model_manager.settings.max_loaded_models", 4)
    reg = ModelRegistry()
    reg._aliases_path = tmp_path / "aliases.json"
    reg.load()

    store = MagicMock()
    store.ensure_adapter_downloaded = MagicMock(return_value=tmp_path / "adapter")
    manager = ModelManager(reg, store)

    # Preload a fake base so ensure_loaded(base) returns it without real I/O.
    base = LoadedModel(
        name="qwen3-8b:latest",
        hf_path="Qwen/Qwen3-8B-MLX",
        model=MagicMock(),
        tokenizer=MagicMock(),
    )
    manager._loaded["qwen3-8b:latest"] = base

    # Stub the heavy build so no Metal/model work happens.
    monkeypatch.setattr(
        manager, "_build_adapter_model", lambda base_model, adapter_dir: MagicMock()
    )
    return manager, base


class TestAdapterLoad:
    async def test_load_pins_base_and_shares(self, adapter_manager):
        manager, base = adapter_manager
        lm = await manager.ensure_loaded("qwen3-8b:my-lora")
        assert lm.name == "qwen3-8b:my-lora"
        assert lm.adapter_base == "qwen3-8b:latest"
        # Same tokenizer object as the base (grammar-cache identity).
        assert lm.tokenizer is base.tokenizer
        # Base is pinned by exactly one child, with no leaked active_refs.
        assert base._adapter_child_refs == 1
        assert base.active_refs == 0

    async def test_two_adapters_share_one_base(self, adapter_manager):
        manager, base = adapter_manager
        # Register and load a second adapter on the same base.
        manager.registry._adapters["qwen3-8b:other"] = manager.registry._adapters[
            "qwen3-8b:my-lora"
        ].__class__(name="qwen3-8b:other", base="qwen3-8b", hf_path="acme/other")
        await manager.ensure_loaded("qwen3-8b:my-lora")
        await manager.ensure_loaded("qwen3-8b:other")
        assert base._adapter_child_refs == 2
        assert "qwen3-8b:my-lora" in manager._loaded
        assert "qwen3-8b:other" in manager._loaded

    async def test_cached_adapter_returned(self, adapter_manager):
        manager, base = adapter_manager
        first = await manager.ensure_loaded("qwen3-8b:my-lora")
        second = await manager.ensure_loaded("qwen3-8b:my-lora")
        assert first is second
        assert base._adapter_child_refs == 1  # not double-counted


class TestAdapterEvictionAndUnload:
    async def test_lru_skips_pinned_base(self, adapter_manager, monkeypatch):
        manager, base = adapter_manager
        await manager.ensure_loaded("qwen3-8b:my-lora")
        # Force eviction pressure: only the adapter may be evicted, not the base.
        monkeypatch.setattr("olmlx.engine.model_manager.settings.max_loaded_models", 2)
        evictees = manager._pop_lru_evictees()
        evicted_names = {e.name for e in evictees}
        assert "qwen3-8b:latest" not in evicted_names
        assert "qwen3-8b:my-lora" in evicted_names
        # Evicting the adapter detached it from the base.
        assert base._adapter_child_refs == 0

    async def test_unload_base_with_adapter_refused(self, adapter_manager):
        from olmlx.engine.model_manager import ActiveRequestsError

        manager, base = adapter_manager
        await manager.ensure_loaded("qwen3-8b:my-lora")
        with pytest.raises(ActiveRequestsError, match="adapter"):
            await manager.unload("qwen3-8b")

    async def test_unload_adapter_then_base(self, adapter_manager):
        manager, base = adapter_manager
        await manager.ensure_loaded("qwen3-8b:my-lora")
        assert await manager.unload("qwen3-8b:my-lora") is True
        assert base._adapter_child_refs == 0
        # With no adapters left, the base unloads cleanly.
        assert await manager.unload("qwen3-8b") is True


class TestAdapterClose:
    def test_close_skips_grammar_drop_for_adapter(self, monkeypatch):
        called = []
        import olmlx.engine.grammar as grammar

        monkeypatch.setattr(
            grammar, "drop_for_tokenizer", lambda tok: called.append(tok)
        )
        adapter = LoadedModel(
            name="qwen3-8b:my-lora",
            hf_path="acme/my-lora",
            model=MagicMock(spec=[]),
            tokenizer=MagicMock(),
            adapter_base="qwen3-8b:latest",
        )
        ModelManager._close_loaded_model(adapter)
        assert called == []  # shared tokenizer must not be dropped by the adapter

    def test_close_drops_grammar_for_base(self, monkeypatch):
        called = []
        import olmlx.engine.grammar as grammar

        monkeypatch.setattr(
            grammar, "drop_for_tokenizer", lambda tok: called.append(tok)
        )
        base = LoadedModel(
            name="qwen3-8b:latest",
            hf_path="Qwen/Qwen3-8B-MLX",
            model=MagicMock(spec=[]),
            tokenizer=MagicMock(),
        )
        ModelManager._close_loaded_model(base)
        assert len(called) == 1


class TestRejectUnsupportedBase:
    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"is_vlm": True}, "vision"),
            ({"is_flash": True}, "flash"),
            ({"is_distributed": True}, "distributed"),
            ({"is_whisper": True}, "whisper"),
            ({"kv_cache_quant": "turbo:4"}, "KV-cache"),
        ],
    )
    def test_rejects(self, kwargs, match):
        base = LoadedModel(
            name="b:latest",
            hf_path="o/b",
            model=MagicMock(),
            tokenizer=MagicMock(),
            **kwargs,
        )
        with pytest.raises(ValueError, match=match):
            ModelManager._reject_adapter_base(base)

    def test_accepts_plain_base(self):
        base = LoadedModel(
            name="b:latest", hf_path="o/b", model=MagicMock(), tokenizer=MagicMock()
        )
        ModelManager._reject_adapter_base(base)  # no raise
