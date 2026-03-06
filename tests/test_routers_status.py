"""Tests for mlx_ollama.routers.status."""

import pytest

from mlx_ollama import __version__


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
    async def test_ps_with_expiry(self, app_client):
        import time

        manager = app_client._transport.app.state.model_manager
        lm = manager._loaded["qwen3:latest"]
        lm.expires_at = time.time() + 300
        resp = await app_client.get("/api/ps")
        data = resp.json()
        assert data["models"][0]["expires_at"] != ""
