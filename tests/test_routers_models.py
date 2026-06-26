"""Tests for olmlx.routers.models."""

import pytest


class TestModelsRouter:
    @pytest.mark.asyncio
    async def test_list_tags(self, app_client):
        resp = await app_client.get("/api/tags")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        # Should include models from registry
        names = [m["name"] for m in data["models"]]
        assert any("qwen3" in n for n in names)

    @pytest.mark.asyncio
    async def test_show_not_found(self, app_client):
        resp = await app_client.post("/api/show", json={"name": "nonexistent"})
        assert resp.status_code == 404
        data = resp.json()
        assert "error" in data

    @pytest.mark.asyncio
    async def test_show_found(self, app_client, tmp_path):
        from olmlx.models.manifest import ModelManifest

        store = app_client._transport.app.state.model_store
        model_dir = store.models_dir / "qwen3_latest"
        model_dir.mkdir(parents=True, exist_ok=True)
        manifest = ModelManifest(
            name="qwen3:latest",
            hf_path="Qwen/Qwen3-8B-MLX",
            family="qwen",
            format="mlx",
        )
        manifest.save(model_dir / "manifest.json")

        resp = await app_client.post("/api/show", json={"name": "qwen3"})
        assert resp.status_code == 200
        data = resp.json()
        assert "details" in data
        assert data["details"]["family"] == "qwen"

    @pytest.mark.asyncio
    async def test_show_accepts_spec_name_field_not_found(self, app_client):
        # Ollama spec field 'name' must be accepted; not-found gives 404, not 422
        resp = await app_client.post("/api/show", json={"name": "nonexistent"})
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_show_accepts_legacy_model_field(self, app_client):
        # backward compat: old 'model' key still works via AliasChoices
        resp = await app_client.post("/api/show", json={"model": "nonexistent"})
        assert resp.status_code == 404  # not 422

    @pytest.mark.asyncio
    async def test_show_rejects_empty_body(self, app_client):
        # neither name nor model gives 422
        resp = await app_client.post("/api/show", json={})
        assert resp.status_code == 400
