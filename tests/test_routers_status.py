"""Tests for olmlx.routers.status."""

import pytest

from olmlx import __version__


class TestStatusRouter:
    @pytest.mark.asyncio
    async def test_root(self, app_client):
        resp = await app_client.get("/")
        assert resp.status_code == 200
        assert resp.text == "Ollama is running"

    @pytest.mark.asyncio
    async def test_root_head(self, app_client):
        resp = await app_client.head("/")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_version(self, app_client):
        resp = await app_client.get("/api/version")
        assert resp.status_code == 200
        data = resp.json()
        assert data["version"] == __version__

    @pytest.mark.asyncio
    async def test_ps_empty(self, app_client):
        # Clear loaded models first
        app_client._transport.app.state.model_manager._loaded.clear()
        resp = await app_client.get("/api/ps")
        assert resp.status_code == 200
        data = resp.json()
        assert data["models"] == []

    @pytest.mark.asyncio
    async def test_ps_with_model(self, app_client):
        resp = await app_client.get("/api/ps")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["models"]) == 1
        assert data["models"][0]["name"] == "qwen3:latest"

    @pytest.mark.asyncio
    async def test_ps_populates_details_without_manifest(self, app_client):
        """Issue #340: /api/ps fills details from config.json when manifest.json is missing."""
        import json

        store = app_client._transport.app.state.model_store
        # The mock loaded model is "qwen3:latest" → "Qwen/Qwen3-8B-MLX".
        local_dir = store.local_path("Qwen/Qwen3-8B-MLX")
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "qwen3",
                    "quantization": {"bits": 4, "group_size": 64},
                }
            )
        )
        (local_dir / "model.safetensors").write_bytes(b"\x00" * 512)

        resp = await app_client.get("/api/ps")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["models"]) == 1
        model = data["models"][0]
        assert model["details"]["family"] == "qwen3"
        assert model["details"]["quantization_level"] == "4-bit"
        assert model["size"] >= 512
        assert model["digest"] != ""

    @pytest.mark.asyncio
    async def test_ps_with_expiry(self, app_client):
        import time

        manager = app_client._transport.app.state.model_manager
        lm = manager._loaded["qwen3:latest"]
        lm.expires_at = time.time() + 300
        resp = await app_client.get("/api/ps")
        data = resp.json()
        assert data["models"][0]["expires_at"] != ""
