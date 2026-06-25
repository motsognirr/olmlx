"""End-to-end LoRA-adapter hot-swap on a real MLX model (issue #362).

Metal-gated: loads an actual ~300MB model. Skipped in CI via
``pytest -m "not real_model"``; runs on the inference runner.

Adapters are fabricated locally (zero-initialised LoRA → identity output) so the
test needs no network and asserts the load/share/serve/teardown lifecycle
directly. The weight-sharing mechanism itself was validated interactively on a
4-bit Qwen3-8B before this test was written.
"""

import json

import mlx.core as mx
import pytest

from olmlx.engine.model_manager import ModelManager
from olmlx.engine.registry import ModelRegistry
from olmlx.models.store import ModelStore

REAL_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

pytestmark = pytest.mark.real_model


def _inner(model):
    """The submodule holding ``.layers`` (mlx-lm wraps the decoder in .model)."""
    return getattr(model, "model", model)


def _write_adapter(store, hf_path, num_layers, rank=8):
    p = store.adapter_local_path(hf_path)
    p.mkdir(parents=True, exist_ok=True)
    (p / "adapter_config.json").write_text(
        json.dumps(
            {
                "fine_tune_type": "lora",
                "num_layers": num_layers,
                "lora_parameters": {"rank": rank, "scale": 20.0, "dropout": 0.0},
            }
        )
    )
    # strict=False load tolerates this placeholder; LoRALinear init zeroes
    # lora_b, so the adapter is an identity until real weights are trained.
    mx.save_safetensors(
        str(p / "adapters.safetensors"), {"_placeholder": mx.zeros((1,))}
    )


@pytest.fixture
async def manager(tmp_path, monkeypatch):
    models_config = tmp_path / "models.json"
    models_config.write_text(
        json.dumps(
            {
                "qwen": REAL_MODEL,
                "adapters": {
                    "qwen:lora-a": {"base": "qwen", "hf_path": "fake/lora-a"},
                    "qwen:lora-b": {"base": "qwen", "hf_path": "fake/lora-b"},
                },
            }
        )
    )
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.setattr("olmlx.config.settings.models_dir", models_dir)
    monkeypatch.setattr("olmlx.engine.registry.settings.models_config", models_config)
    monkeypatch.setattr("olmlx.models.store.settings.models_dir", models_dir)
    monkeypatch.setattr("olmlx.engine.model_manager.settings.max_loaded_models", 4)

    reg = ModelRegistry()
    reg._aliases_path = tmp_path / "aliases.json"
    reg.load()
    store = ModelStore(reg)
    mgr = ModelManager(reg, store)
    yield mgr, store
    await mgr.stop()


async def test_two_adapters_share_base_and_serve(manager):
    mgr, store = manager
    base = await mgr.ensure_loaded("qwen")
    n_layers = len(_inner(base.model).layers)
    _write_adapter(store, "fake/lora-a", n_layers)
    _write_adapter(store, "fake/lora-b", n_layers)

    x = mx.array([base.tokenizer.encode("The capital of France is")])

    def logits(lm):
        out = lm.model(x)
        mx.eval(out)
        return out[0, -1]

    base_logits = logits(base)

    mem_before = mx.get_active_memory()
    a = await mgr.ensure_loaded("qwen:lora-a")
    b = await mgr.ensure_loaded("qwen:lora-b")
    mem_after = mx.get_active_memory()

    # All three resident; both adapters pin the one base.
    assert set(mgr._loaded) == {"qwen:latest", "qwen:lora-a", "qwen:lora-b"}
    assert mgr._loaded["qwen:latest"]._adapter_child_refs == 2
    assert a.adapter_base == "qwen:latest"
    assert a.tokenizer is base.tokenizer
    # size_bytes reports the LoRA-delta footprint, not 0 and not the full base.
    assert 0 < a.size_bytes < base.size_bytes
    # Adapters serve via the per-request path (batching not validated for them).
    assert a.batching is False

    # Two full adapter models added far less than a second base (~250MB+);
    # the base weights are shared, not duplicated.
    assert (mem_after - mem_before) < 100 * 1024 * 1024

    # A specific base weight array is shared into the adapter's LoRA layer.
    base_w = dict.__getitem__(_inner(base.model).layers[-1].self_attn.q_proj, "weight")
    a_qproj = _inner(a.model).layers[-1].self_attn.q_proj
    assert type(a_qproj).__name__ == "LoRALinear"
    assert dict.__getitem__(a_qproj.linear, "weight") is base_w

    # Zero-init LoRA ⇒ identity, so each adapter reproduces the base logits and
    # the base itself is uninjured.
    assert mx.allclose(logits(a), base_logits, atol=1e-3)
    assert mx.allclose(logits(b), base_logits, atol=1e-3)
    assert mx.allclose(logits(base), base_logits)


async def test_unload_adapters_then_base(manager):
    mgr, store = manager
    base = await mgr.ensure_loaded("qwen")
    n_layers = len(_inner(base.model).layers)
    _write_adapter(store, "fake/lora-a", n_layers)

    await mgr.ensure_loaded("qwen:lora-a")
    assert mgr._loaded["qwen:latest"]._adapter_child_refs == 1

    from olmlx.engine.model_manager import ActiveRequestsError

    with pytest.raises(ActiveRequestsError):
        await mgr.unload("qwen")  # base pinned by the adapter

    assert await mgr.unload("qwen:lora-a") is True
    assert mgr._loaded["qwen:latest"]._adapter_child_refs == 0
    assert await mgr.unload("qwen") is True
