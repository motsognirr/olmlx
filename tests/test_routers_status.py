"""Tests for olmlx.routers.status."""

import json
import time

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
    async def test_ps_with_expiry(self, app_client):
        manager = app_client._transport.app.state.model_manager
        lm = manager._loaded["qwen3:latest"]
        lm.expires_at = time.time() + 300
        resp = await app_client.get("/api/ps")
        data = resp.json()
        assert data["models"][0]["expires_at"] != ""

    @pytest.mark.asyncio
    async def test_ps_populates_details_from_config(self, app_client, tmp_path):
        """ps populates details.family etc. from config.json on disk."""
        lm = app_client._transport.app.state.model_manager._loaded["qwen3:latest"]
        # Create the model directory with a config.json
        store = app_client._transport.app.state.model_store
        local_dir = store.local_path(lm.hf_path)
        local_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "model_type": "llama",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "quantization": {"bits": 4},
        }
        (local_dir / "config.json").write_text(json.dumps(config))

        resp = await app_client.get("/api/ps")
        assert resp.status_code == 200
        data = resp.json()
        model = data["models"][0]
        assert model["details"]["family"] == "llama"
        assert model["details"]["parameter_size"] != ""
        assert model["details"]["quantization_level"] == "4-bit"
        assert model["details"]["format"] == "mlx"

    @pytest.mark.asyncio
    async def test_ps_populates_size_and_vram(self, app_client, tmp_path):
        """ps populates size and size_vram from disk when lm.size_bytes is 0."""
        lm = app_client._transport.app.state.model_manager._loaded["qwen3:latest"]
        store = app_client._transport.app.state.model_store
        local_dir = store.local_path(lm.hf_path)
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "weights.safetensors").write_bytes(b"\x00" * 1024)
        (local_dir / "config.json").write_text(json.dumps({"model_type": "test"}))

        resp = await app_client.get("/api/ps")
        data = resp.json()
        model = data["models"][0]
        assert model["size"] == 1024 + len(json.dumps({"model_type": "test"}))
        assert model["size_vram"] == model["size"]
