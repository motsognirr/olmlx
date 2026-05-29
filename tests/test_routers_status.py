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
        """ps populates details from manifest rather than config.json."""
        lm = app_client._transport.app.state.model_manager._loaded["qwen3:latest"]
        store = app_client._transport.app.state.model_store
        local_dir = store.local_path(lm.hf_path)
        local_dir.mkdir(parents=True, exist_ok=True)
        # Write a manifest with the metadata (mimics backfilling at load time)
        manifest = {
            "name": lm.name,
            "hf_path": lm.hf_path,
            "size": 0,
            "digest": "sha256:abc123",
            "format": "mlx",
            "family": "llama",
            "parameter_size": "8B",
            "quantization_level": "4-bit",
        }
        (local_dir / "manifest.json").write_text(json.dumps(manifest))

        resp = await app_client.get("/api/ps")
        assert resp.status_code == 200
        data = resp.json()
        model = data["models"][0]
        assert model["details"]["family"] == "llama"
        assert model["details"]["parameter_size"] == "8B"
        assert model["details"]["quantization_level"] == "4-bit"
        assert model["details"]["format"] == "mlx"
        assert model["digest"] == "sha256:abc123"

    @pytest.mark.asyncio
    async def test_ps_populates_size_and_vram(self, app_client, tmp_path):
        """ps populates size and size_vram from the manifest when lm.size_bytes is 0."""
        lm = app_client._transport.app.state.model_manager._loaded["qwen3:latest"]
        store = app_client._transport.app.state.model_store
        local_dir = store.local_path(lm.hf_path)
        local_dir.mkdir(parents=True, exist_ok=True)
        # Write a manifest with the size pre-computed (as _load_model does)
        manifest = {
            "name": lm.name,
            "hf_path": lm.hf_path,
            "size": 4096,
            "digest": "",
            "format": "mlx",
            "family": "test",
            "parameter_size": "",
            "quantization_level": "",
        }
        (local_dir / "manifest.json").write_text(json.dumps(manifest))
        (local_dir / "config.json").write_text(json.dumps({"model_type": "test"}))

        resp = await app_client.get("/api/ps")
        data = resp.json()
        model = data["models"][0]
        assert model["size"] == 4096
        assert model["size_vram"] == model["size"]
        assert model["details"]["family"] == "test"

    @pytest.mark.asyncio
    async def test_api_ps_includes_cache_metrics(self, app_client):
        """ps includes cache_metrics from the loaded model's prompt_cache_store."""
        from unittest.mock import MagicMock

        from olmlx.engine.prompt_cache import CachedPromptState, PromptCacheStore

        # Set up a model with a PromptCacheStore and some cache activity
        lm = app_client._transport.app.state.model_manager._loaded["qwen3:latest"]
        store = PromptCacheStore(max_slots=4)
        store.set("a", CachedPromptState(tokens=[1, 2, 3], cache=[MagicMock()]))
        store.get("a")  # bump cache_id_hits
        lm.prompt_cache_store = store

        resp = await app_client.get("/api/ps")
        assert resp.status_code == 200
        data = resp.json()
        model = data["models"][0]
        assert "cache_metrics" in model
        assert model["cache_metrics"]["cache_id_hits"] == 1
        assert "radix_hits" in model["cache_metrics"]

    @pytest.mark.asyncio
    async def test_api_ps_multi_model_preserves_model_store(self, app_client):
        """Regression: per-lm cache_store lookup must not shadow the outer
        model_store used for manifest reads on subsequent iterations."""
        from copy import copy
        from unittest.mock import MagicMock

        from olmlx.engine.prompt_cache import CachedPromptState, PromptCacheStore

        manager = app_client._transport.app.state.model_manager
        first = manager._loaded["qwen3:latest"]
        # Shallow-clone the first model under a new key so both rows hit the
        # manifest-reading branch (which depends on app.state.model_store).
        second = copy(first)
        second.name = "qwen3:second"
        second.hf_path = first.hf_path  # share manifest dir
        second.prompt_cache_store = PromptCacheStore(max_slots=4)
        second.prompt_cache_store.set(
            "x", CachedPromptState(tokens=[7, 8], cache=[MagicMock()])
        )
        manager._loaded["qwen3:second"] = second

        try:
            resp = await app_client.get("/api/ps")
            assert resp.status_code == 200
            data = resp.json()
            names = {m["name"] for m in data["models"]}
            assert names == {"qwen3:latest", "qwen3:second"}
            # Both rows must carry the cache_metrics field.
            for model in data["models"]:
                assert "cache_metrics" in model
                assert "cache_id_hits" in model["cache_metrics"]
        finally:
            manager._loaded.pop("qwen3:second", None)
