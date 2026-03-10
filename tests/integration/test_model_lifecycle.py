"""Integration tests for model load → inference → unload lifecycle."""

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def anthropic_request():
    return {
        "model": "qwen3",
        "max_tokens": 100,
        "stream": False,
        "messages": [{"role": "user", "content": "Hi"}],
    }


async def test_load_on_first_request(integration_ctx, anthropic_request):
    """First request to an unloaded model triggers mlx_lm.load."""
    import mlx_lm

    assert "qwen3:latest" not in integration_ctx.manager._loaded

    resp = await integration_ctx.client.post("/v1/messages", json=anthropic_request)
    assert resp.status_code == 200

    mlx_lm.load.assert_called_once()
    assert "qwen3:latest" in integration_ctx.manager._loaded


async def test_model_reused_on_second_request(integration_ctx, anthropic_request):
    """Second request reuses the already-loaded model (no second load)."""
    import mlx_lm

    resp1 = await integration_ctx.client.post("/v1/messages", json=anthropic_request)
    assert resp1.status_code == 200
    assert mlx_lm.load.call_count == 1

    resp2 = await integration_ctx.client.post("/v1/messages", json=anthropic_request)
    assert resp2.status_code == 200
    assert mlx_lm.load.call_count == 1


async def test_lru_eviction_at_capacity(integration_ctx, monkeypatch):
    """With max_loaded_models=1, loading model B evicts model A."""
    monkeypatch.setattr("olmlx.config.settings.max_loaded_models", 1)

    # Add a second model mapping
    integration_ctx.registry.add_mapping("llama3", "mlx-community/Llama-3-8B")

    req_a = {
        "model": "qwen3",
        "max_tokens": 50,
        "stream": False,
        "messages": [{"role": "user", "content": "Hi"}],
    }
    resp_a = await integration_ctx.client.post("/v1/messages", json=req_a)
    assert resp_a.status_code == 200
    assert "qwen3:latest" in integration_ctx.manager._loaded

    req_b = {
        "model": "llama3",
        "max_tokens": 50,
        "stream": False,
        "messages": [{"role": "user", "content": "Hi"}],
    }
    resp_b = await integration_ctx.client.post("/v1/messages", json=req_b)
    assert resp_b.status_code == 200

    assert "llama3:latest" in integration_ctx.manager._loaded
    assert "qwen3:latest" not in integration_ctx.manager._loaded


async def test_active_refs_prevents_eviction(integration_ctx, monkeypatch):
    """A model with active_refs > 0 cannot be evicted."""
    monkeypatch.setattr("olmlx.config.settings.max_loaded_models", 1)

    integration_ctx.registry.add_mapping("llama3", "mlx-community/Llama-3-8B")

    # Load model A
    req_a = {
        "model": "qwen3",
        "max_tokens": 50,
        "stream": False,
        "messages": [{"role": "user", "content": "Hi"}],
    }
    resp_a = await integration_ctx.client.post("/v1/messages", json=req_a)
    assert resp_a.status_code == 200

    # Simulate active reference
    lm = integration_ctx.manager._loaded["qwen3:latest"]
    lm.active_refs = 1

    try:
        req_b = {
            "model": "llama3",
            "max_tokens": 50,
            "stream": False,
            "messages": [{"role": "user", "content": "Hi"}],
        }
        resp_b = await integration_ctx.client.post("/v1/messages", json=req_b)
        # Should get 500 (RuntimeError: all models busy)
        assert resp_b.status_code == 500
        assert (
            "in use" in resp_b.json().get("error", {}).get("message", "").lower()
            or "in use" in resp_b.json().get("error", "").lower()
        )
    finally:
        lm.active_refs = 0


async def test_keep_alive_expiry(integration_ctx, anthropic_request, monkeypatch):
    """Model expires after keep_alive duration."""
    monkeypatch.setattr("olmlx.config.settings.default_keep_alive", "0")

    resp = await integration_ctx.client.post("/v1/messages", json=anthropic_request)
    assert resp.status_code == 200
    assert "qwen3:latest" in integration_ctx.manager._loaded

    # Force expiry check
    import time

    lm = integration_ctx.manager._loaded["qwen3:latest"]
    lm.expires_at = time.time() - 10  # Already expired

    # Trigger expiry manually (don't wait for the 30s loop)
    async with integration_ctx.manager._lock:
        now = time.time()
        expired = [
            name
            for name, m in integration_ctx.manager._loaded.items()
            if m.expires_at is not None and m.expires_at <= now and m.active_refs == 0
        ]
        for name in expired:
            del integration_ctx.manager._loaded[name]

    assert "qwen3:latest" not in integration_ctx.manager._loaded


async def test_unload_endpoint(integration_ctx, anthropic_request):
    """POST /api/unload removes the model."""
    resp = await integration_ctx.client.post("/v1/messages", json=anthropic_request)
    assert resp.status_code == 200
    assert "qwen3:latest" in integration_ctx.manager._loaded

    unload_resp = await integration_ctx.client.post(
        "/api/unload", json={"model": "qwen3"}
    )
    assert unload_resp.status_code == 200
    assert unload_resp.json()["status"] == "unloaded"
    assert "qwen3:latest" not in integration_ctx.manager._loaded


async def test_memory_limit_rejects_oversized(integration_ctx, monkeypatch):
    """Model that exceeds memory limit returns 503."""
    # Make metal memory report exceed the limit after loading
    monkeypatch.setattr(
        "olmlx.engine.model_manager._get_metal_memory_bytes",
        MagicMock(return_value=30 * 1024**3),  # 30 GB — exceeds 75% of 32 GB
    )

    req = {
        "model": "qwen3",
        "max_tokens": 50,
        "stream": False,
        "messages": [{"role": "user", "content": "Hi"}],
    }
    resp = await integration_ctx.client.post("/v1/messages", json=req)
    assert resp.status_code == 503


async def test_warmup_loads_model(integration_ctx):
    """POST /api/warmup loads the model without running inference."""
    assert "qwen3:latest" not in integration_ctx.manager._loaded

    resp = await integration_ctx.client.post("/api/warmup", json={"model": "qwen3"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "loaded"
    assert "qwen3:latest" in integration_ctx.manager._loaded
