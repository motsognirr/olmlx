"""Tests for olmlx.routers.manage."""

import pytest


class TestManageRouter:
    @pytest.mark.asyncio
    async def test_copy_model(self, app_client):
        resp = await app_client.post(
            "/api/copy",
            json={
                "source": "qwen3:latest",
                "destination": "my-qwen",
            },
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_copy_model_not_found(self, app_client):
        resp = await app_client.post(
            "/api/copy",
            json={
                "source": "nonexistent",
                "destination": "alias",
            },
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_model_not_found(self, app_client):
        resp = await app_client.request(
            "DELETE", "/api/delete", json={"model": "nonexistent"}
        )
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
        resp = await app_client.post(
            "/api/create",
            json={
                "model": "test",
                "modelfile": "SYSTEM You are helpful",
            },
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_create_model_success(self, app_client):
        resp = await app_client.post(
            "/api/create",
            json={
                "model": "test",
                "modelfile": "FROM qwen3:latest\nSYSTEM You are helpful",
                "stream": False,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"

    @pytest.mark.asyncio
    async def test_create_model_base_not_found(self, app_client):
        resp = await app_client.post(
            "/api/create",
            json={
                "model": "test",
                "modelfile": "FROM nonexistent\nSYSTEM test",
                "stream": False,
            },
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_create_model_streaming(self, app_client):
        resp = await app_client.post(
            "/api/create",
            json={
                "model": "test",
                "modelfile": "FROM qwen3:latest\nSYSTEM You are helpful",
                "stream": True,
            },
        )
        assert resp.status_code == 200
        assert "ndjson" in resp.headers["content-type"]
        assert "success" in resp.text

    @pytest.mark.asyncio
    async def test_create_model_with_parameter(self, app_client):
        resp = await app_client.post(
            "/api/create",
            json={
                "model": "test",
                "modelfile": "FROM qwen3:latest\nPARAMETER temperature 0.5\nSYSTEM helpful",
                "stream": False,
            },
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_delete_model_success(self, app_client, tmp_path):
        from olmlx.models.manifest import ModelManifest

        # Create a model directory with manifest
        models_dir = tmp_path / "models"
        model_dir = models_dir / "test_latest"
        model_dir.mkdir(parents=True)
        (model_dir / "file.bin").write_bytes(b"\x00")
        ModelManifest(name="test:latest", hf_path="test/model").save(
            model_dir / "manifest.json"
        )

        resp = await app_client.request("DELETE", "/api/delete", json={"model": "test"})
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_pull_non_streaming(self, app_client):
        from unittest.mock import patch

        async def mock_pull(name):
            yield {"status": "pulling manifest"}
            yield {"status": "success"}

        with patch.object(
            app_client._transport.app.state.model_store,
            "pull",
            side_effect=mock_pull,
        ):
            resp = await app_client.post(
                "/api/pull",
                json={
                    "model": "qwen3",
                    "stream": False,
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"

    @pytest.mark.asyncio
    async def test_pull_streaming(self, app_client):
        from unittest.mock import patch

        async def mock_pull(name):
            yield {"status": "pulling manifest"}
            yield {"status": "downloading"}
            yield {"status": "success"}

        with patch.object(
            app_client._transport.app.state.model_store,
            "pull",
            side_effect=mock_pull,
        ):
            resp = await app_client.post(
                "/api/pull",
                json={
                    "model": "qwen3",
                    "stream": True,
                },
            )
        assert resp.status_code == 200
        assert "success" in resp.text

    @pytest.mark.asyncio
    async def test_pull_error_streaming(self, app_client):
        from unittest.mock import patch

        async def mock_pull(name):
            yield {"status": "pulling"}
            raise RuntimeError("download failed")

        with patch.object(
            app_client._transport.app.state.model_store,
            "pull",
            side_effect=mock_pull,
        ):
            resp = await app_client.post(
                "/api/pull",
                json={
                    "model": "qwen3",
                    "stream": True,
                },
            )
        assert resp.status_code == 200
        assert "error" in resp.text

    @pytest.mark.asyncio
    async def test_unload_model_success(self, app_client):
        resp = await app_client.post("/api/unload", json={"model": "qwen3"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "unloaded"

    @pytest.mark.asyncio
    async def test_unload_model_not_loaded(self, app_client):
        resp = await app_client.post("/api/unload", json={"model": "nonexistent"})
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_unload_model_active_refs(self, app_client):
        manager = app_client._transport.app.state.model_manager
        lm = manager._loaded["qwen3:latest"]
        lm.active_refs = 2
        try:
            resp = await app_client.post("/api/unload", json={"model": "qwen3"})
            assert resp.status_code == 409
            assert "active" in resp.json()["error"].lower()
        finally:
            lm.active_refs = 0

    @pytest.mark.asyncio
    async def test_abort_returns_noop(self, app_client):
        resp = await app_client.post("/api/abort", json={"model": "qwen3"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "no-op"

    @pytest.mark.asyncio
    async def test_warmup_model_not_found(self, app_client):
        resp = await app_client.post("/api/warmup", json={"model": "nonexistent-model"})
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_pull_error_non_streaming(self, app_client):
        from unittest.mock import patch

        async def mock_pull(name):
            raise RuntimeError("download failed")
            yield  # make it a generator

        with patch.object(
            app_client._transport.app.state.model_store,
            "pull",
            side_effect=mock_pull,
        ):
            resp = await app_client.post(
                "/api/pull",
                json={
                    "model": "qwen3",
                    "stream": False,
                },
            )
        assert resp.status_code == 500

    @pytest.mark.asyncio
    async def test_pull_streaming_calls_aclose_on_iterator(self, app_client):
        """Verify the pull iterator's aclose() is explicitly called via finally block."""
        from unittest.mock import patch

        original_aclose_called = False

        async def mock_pull(name):
            nonlocal original_aclose_called
            try:
                yield {"status": "pulling manifest"}
                yield {"status": "success"}
            finally:
                original_aclose_called = True

        # We wrap mock_pull to track aclose calls on the actual iterator
        aclose_explicitly_called = False

        class TrackingAsyncGen:
            def __init__(self, gen):
                self._gen = gen

            def __aiter__(self):
                return self

            async def __anext__(self):
                return await self._gen.__anext__()

            async def aclose(self):
                nonlocal aclose_explicitly_called
                aclose_explicitly_called = True
                await self._gen.aclose()

        def tracked_pull(name):
            return TrackingAsyncGen(mock_pull(name))

        with patch.object(
            app_client._transport.app.state.model_store,
            "pull",
            side_effect=tracked_pull,
        ):
            resp = await app_client.post(
                "/api/pull",
                json={
                    "model": "qwen3",
                    "stream": True,
                },
            )
        assert resp.status_code == 200
        assert aclose_explicitly_called is True
