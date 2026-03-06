"""Tests for mlx_ollama.routers.manage."""

import pytest


class TestManageRouter:
    @pytest.mark.asyncio
    async def test_copy_model(self, app_client):
        resp = await app_client.post("/api/copy", json={
            "source": "qwen3:latest",
            "destination": "my-qwen",
        })
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_copy_model_not_found(self, app_client):
        resp = await app_client.post("/api/copy", json={
            "source": "nonexistent",
            "destination": "alias",
        })
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_model_not_found(self, app_client):
        resp = await app_client.request("DELETE", "/api/delete", json={"model": "nonexistent"})
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_push_not_supported(self, app_client):
        resp = await app_client.post("/api/push")
        assert resp.status_code == 501

    @pytest.mark.asyncio
    async def test_create_model_no_modelfile(self, app_client):
        resp = await app_client.post("/api/create", json={"model": "test"})
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_create_model_no_from(self, app_client):
        resp = await app_client.post("/api/create", json={
            "model": "test",
            "modelfile": "SYSTEM You are helpful",
        })
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_create_model_success(self, app_client):
        resp = await app_client.post("/api/create", json={
            "model": "test",
            "modelfile": "FROM qwen3:latest\nSYSTEM You are helpful",
            "stream": False,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"

    @pytest.mark.asyncio
    async def test_create_model_base_not_found(self, app_client):
        resp = await app_client.post("/api/create", json={
            "model": "test",
            "modelfile": "FROM nonexistent\nSYSTEM test",
            "stream": False,
        })
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_create_model_streaming(self, app_client):
        resp = await app_client.post("/api/create", json={
            "model": "test",
            "modelfile": "FROM qwen3:latest\nSYSTEM You are helpful",
            "stream": True,
        })
        assert resp.status_code == 200
        assert "ndjson" in resp.headers["content-type"]
        assert "success" in resp.text

    @pytest.mark.asyncio
    async def test_create_model_with_parameter(self, app_client):
        resp = await app_client.post("/api/create", json={
            "model": "test",
            "modelfile": "FROM qwen3:latest\nPARAMETER temperature 0.5\nSYSTEM helpful",
            "stream": False,
        })
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_delete_model_success(self, app_client, tmp_path):
        from mlx_ollama.models.manifest import ModelManifest

        # Create a model directory with manifest
        models_dir = tmp_path / "models"
        model_dir = models_dir / "test_latest"
        model_dir.mkdir(parents=True)
        (model_dir / "file.bin").write_bytes(b"\x00")
        ModelManifest(name="test:latest", hf_path="test/model").save(model_dir / "manifest.json")

        resp = await app_client.request("DELETE", "/api/delete", json={"model": "test"})
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_pull_non_streaming(self, app_client):
        from unittest.mock import AsyncMock, patch

        async def mock_pull(name):
            yield {"status": "pulling manifest"}
            yield {"status": "success"}

        with patch.object(
            app_client._transport.app.state.model_store, "pull",
            side_effect=mock_pull,
        ):
            resp = await app_client.post("/api/pull", json={
                "model": "qwen3", "stream": False,
            })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"

    @pytest.mark.asyncio
    async def test_pull_streaming(self, app_client):
        from unittest.mock import AsyncMock, patch

        async def mock_pull(name):
            yield {"status": "pulling manifest"}
            yield {"status": "downloading"}
            yield {"status": "success"}

        with patch.object(
            app_client._transport.app.state.model_store, "pull",
            side_effect=mock_pull,
        ):
            resp = await app_client.post("/api/pull", json={
                "model": "qwen3", "stream": True,
            })
        assert resp.status_code == 200
        assert "success" in resp.text

    @pytest.mark.asyncio
    async def test_pull_error_streaming(self, app_client):
        from unittest.mock import patch

        async def mock_pull(name):
            yield {"status": "pulling"}
            raise RuntimeError("download failed")

        with patch.object(
            app_client._transport.app.state.model_store, "pull",
            side_effect=mock_pull,
        ):
            resp = await app_client.post("/api/pull", json={
                "model": "qwen3", "stream": True,
            })
        assert resp.status_code == 200
        assert "error" in resp.text

    @pytest.mark.asyncio
    async def test_pull_error_non_streaming(self, app_client):
        from unittest.mock import patch

        async def mock_pull(name):
            raise RuntimeError("download failed")
            yield  # make it a generator

        with patch.object(
            app_client._transport.app.state.model_store, "pull",
            side_effect=mock_pull,
        ):
            resp = await app_client.post("/api/pull", json={
                "model": "qwen3", "stream": False,
            })
        assert resp.status_code == 500
