"""Tests for olmlx.app."""

import socket
from unittest.mock import MagicMock, patch

import pytest

from olmlx.app import _make_error_response, create_app
from olmlx.engine.model_manager import ModelLoadTimeoutError


class TestMakeErrorResponse:
    def test_anthropic_path(self):
        resp = _make_error_response(
            "/v1/messages",
            400,
            "bad request",
            "invalid_request_error",
            "invalid_request_error",
            "invalid_value",
        )
        assert resp.status_code == 400
        import json

        body = json.loads(resp.body)
        assert body["type"] == "error"
        assert body["error"]["type"] == "invalid_request_error"

    def test_openai_path(self):
        resp = _make_error_response(
            "/v1/chat/completions",
            500,
            "server error",
            "api_error",
            "server_error",
            "internal_error",
        )
        import json

        body = json.loads(resp.body)
        assert "error" in body
        assert body["error"]["type"] == "server_error"

    def test_ollama_path(self):
        resp = _make_error_response(
            "/api/generate",
            400,
            "bad",
            "invalid_request_error",
            "invalid_request_error",
            "invalid_value",
        )
        import json

        body = json.loads(resp.body)
        assert body["error"] == "bad"


class TestCreateApp:
    def test_app_created(self):
        app = create_app()
        assert app.title == "olmlx"

    def test_routers_registered(self):
        app = create_app()
        paths = [route.path for route in app.routes]
        assert "/" in paths
        assert "/api/version" in paths
        assert "/api/tags" in paths
        assert "/v1/messages" in paths
        assert "/v1/chat/completions" in paths

    def test_cors_uses_configured_origins(self, monkeypatch):
        monkeypatch.setattr(
            "olmlx.app.settings.cors_origins",
            ["http://localhost:3000", "http://localhost:8080"],
        )
        app = create_app()
        from starlette.middleware.cors import CORSMiddleware

        cors_mw = [m for m in app.user_middleware if m.cls is CORSMiddleware]
        assert len(cors_mw) == 1
        origins = cors_mw[0].kwargs["allow_origins"]
        assert "http://localhost:3000" in origins
        assert "*" not in origins


class TestLifespan:
    @pytest.mark.asyncio
    async def test_lifespan_startup_shutdown(self, tmp_path, monkeypatch):
        """Test that the lifespan creates and cleans up state properly."""
        from olmlx.app import lifespan

        monkeypatch.setattr("olmlx.app.settings.models_dir", tmp_path / "models")
        monkeypatch.setattr(
            "olmlx.app.settings.models_config", tmp_path / "models.json"
        )
        (tmp_path / "models.json").write_text("{}")

        app = MagicMock()
        app.state = MagicMock()

        async with lifespan(app):
            assert hasattr(app.state, "registry")
            assert hasattr(app.state, "model_manager")
            assert hasattr(app.state, "model_store")
            assert (tmp_path / "models").exists()


class TestForceJSONMiddleware:
    @pytest.mark.asyncio
    async def test_non_json_content_type_rewritten(self, app_client):
        # Send with form content-type — middleware should rewrite to JSON
        resp = await app_client.post(
            "/api/show",
            content='{"model": "nonexistent"}',
            headers={"content-type": "application/x-www-form-urlencoded"},
        )
        # Should still parse as JSON successfully (404 because model doesn't exist)
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_json_content_type_unchanged(self, app_client):
        resp = await app_client.post(
            "/api/show",
            json={"model": "nonexistent"},
        )
        assert resp.status_code == 404


class TestErrorHandlers:
    @pytest.mark.asyncio
    async def test_value_error_handler_ollama(self, app_client):
        from unittest.mock import AsyncMock

        with patch(
            "olmlx.routers.generate.generate_completion", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.side_effect = ValueError("bad input")
            resp = await app_client.post(
                "/api/generate",
                json={
                    "model": "qwen3",
                    "prompt": "hi",
                    "stream": False,
                },
            )
        assert resp.status_code == 400
        data = resp.json()
        assert data["error"] == "bad input"

    @pytest.mark.asyncio
    async def test_value_error_handler_anthropic(self, app_client):
        from unittest.mock import AsyncMock

        with patch(
            "olmlx.routers.anthropic.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.side_effect = ValueError("invalid model")
            resp = await app_client.post(
                "/v1/messages",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 100,
                },
            )
        assert resp.status_code == 400
        data = resp.json()
        assert data["type"] == "error"
        assert data["error"]["type"] == "invalid_request_error"

    @pytest.mark.asyncio
    async def test_value_error_handler_openai(self, app_client):
        from unittest.mock import AsyncMock

        with patch(
            "olmlx.routers.openai.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.side_effect = ValueError("bad request")
            resp = await app_client.post(
                "/v1/chat/completions",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        assert resp.status_code == 400
        data = resp.json()
        assert "error" in data
        assert data["error"]["type"] == "invalid_request_error"

    @pytest.mark.asyncio
    async def test_runtime_error_handler(self, app_client):
        from unittest.mock import AsyncMock

        with patch(
            "olmlx.routers.generate.generate_completion", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.side_effect = RuntimeError("engine crashed")
            resp = await app_client.post(
                "/api/generate",
                json={
                    "model": "qwen3",
                    "prompt": "hi",
                    "stream": False,
                },
            )
        assert resp.status_code == 500
        data = resp.json()
        assert data["error"] == "engine crashed"

    @pytest.mark.asyncio
    async def test_general_error_handler(self, app_client):
        from unittest.mock import AsyncMock

        with patch(
            "olmlx.routers.generate.generate_completion", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.side_effect = TypeError("unexpected error")
            resp = await app_client.post(
                "/api/generate",
                json={
                    "model": "qwen3",
                    "prompt": "hi",
                    "stream": False,
                },
            )
        assert resp.status_code == 500
        data = resp.json()
        assert "TypeError" in data["error"]

    @pytest.mark.asyncio
    async def test_timeout_error_handler_ollama(self, app_client):
        from unittest.mock import AsyncMock

        with patch(
            "olmlx.routers.generate.generate_completion", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.side_effect = ModelLoadTimeoutError("loading timed out after 60s")
            resp = await app_client.post(
                "/api/generate",
                json={
                    "model": "qwen3",
                    "prompt": "hi",
                    "stream": False,
                },
            )
        assert resp.status_code == 504
        data = resp.json()
        assert data["error"] == "loading timed out after 60s"

    @pytest.mark.asyncio
    async def test_timeout_error_handler_anthropic(self, app_client):
        from unittest.mock import AsyncMock

        with patch(
            "olmlx.routers.anthropic.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.side_effect = ModelLoadTimeoutError("loading timed out after 60s")
            resp = await app_client.post(
                "/v1/messages",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 100,
                },
            )
        assert resp.status_code == 504
        data = resp.json()
        assert data["type"] == "error"
        assert data["error"]["type"] == "api_error"

    @pytest.mark.asyncio
    async def test_timeout_error_handler_openai(self, app_client):
        from unittest.mock import AsyncMock

        with patch(
            "olmlx.routers.openai.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.side_effect = ModelLoadTimeoutError("loading timed out after 60s")
            resp = await app_client.post(
                "/v1/chat/completions",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        assert resp.status_code == 504
        data = resp.json()
        assert data["error"]["code"] == "timeout"

    @pytest.mark.asyncio
    async def test_socket_timeout_not_caught_as_504(self, app_client):
        """socket.timeout (a TimeoutError subclass) should NOT hit the 504 handler."""
        from unittest.mock import AsyncMock

        with patch(
            "olmlx.routers.generate.generate_completion", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.side_effect = socket.timeout("connection timed out")
            resp = await app_client.post(
                "/api/generate",
                json={
                    "model": "qwen3",
                    "prompt": "hi",
                    "stream": False,
                },
            )
        # Should be 500 (general handler), not 504 (model load timeout)
        assert resp.status_code == 500

    @pytest.mark.asyncio
    async def test_memory_error_handler_ollama(self, app_client):
        from unittest.mock import AsyncMock

        with patch(
            "olmlx.routers.generate.generate_completion", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.side_effect = MemoryError("model too large")
            resp = await app_client.post(
                "/api/generate",
                json={
                    "model": "qwen3",
                    "prompt": "hi",
                    "stream": False,
                },
            )
        assert resp.status_code == 503
        data = resp.json()
        assert data["error"] == "model too large"

    @pytest.mark.asyncio
    async def test_memory_error_handler_anthropic(self, app_client):
        from unittest.mock import AsyncMock

        with patch(
            "olmlx.routers.anthropic.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.side_effect = MemoryError("model too large")
            resp = await app_client.post(
                "/v1/messages",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 100,
                },
            )
        assert resp.status_code == 503
        data = resp.json()
        assert data["type"] == "error"
        assert data["error"]["type"] == "overloaded_error"

    @pytest.mark.asyncio
    async def test_memory_error_handler_openai(self, app_client):
        from unittest.mock import AsyncMock

        with patch(
            "olmlx.routers.openai.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.side_effect = MemoryError("model too large")
            resp = await app_client.post(
                "/v1/chat/completions",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        assert resp.status_code == 503
        data = resp.json()
        assert data["error"]["code"] == "model_too_large"

    @pytest.mark.asyncio
    async def test_server_busy_has_retry_after_header(self, app_client):
        """ServerBusyError 503 should include Retry-After header."""
        from unittest.mock import AsyncMock

        from olmlx.engine.inference import ServerBusyError

        with patch(
            "olmlx.routers.generate.generate_completion", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.side_effect = ServerBusyError("deferred cleanup")
            resp = await app_client.post(
                "/api/generate",
                json={
                    "model": "qwen3",
                    "prompt": "hi",
                    "stream": False,
                },
            )
        assert resp.status_code == 503
        assert resp.headers.get("retry-after") == "5"

    @pytest.mark.asyncio
    async def test_general_endpoints_exist(self, app_client):
        resp = await app_client.get("/")
        assert resp.status_code == 200
        assert "running" in resp.text.lower()
