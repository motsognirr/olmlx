"""#760 item 4: an unknown model is a 404 on every API surface, with each
provider's not-found error shape."""

import pytest


class TestUnknownModel404:
    @pytest.mark.asyncio
    async def test_ollama_chat(self, app_client):
        resp = await app_client.post(
            "/api/chat",
            json={
                "model": "nosuchmodel",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
        )
        assert resp.status_code == 404
        assert "not found" in resp.json()["error"]

    @pytest.mark.asyncio
    async def test_ollama_generate_streaming(self, app_client):
        resp = await app_client.post(
            "/api/generate", json={"model": "nosuchmodel", "prompt": "hi"}
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_ollama_preload(self, app_client):
        resp = await app_client.post("/api/generate", json={"model": "nosuchmodel"})
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_openai(self, app_client):
        resp = await app_client.post(
            "/v1/chat/completions",
            json={
                "model": "nosuchmodel",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 404
        err = resp.json()["error"]
        assert err["code"] == "model_not_found"
        assert err["type"] == "invalid_request_error"

    @pytest.mark.asyncio
    async def test_anthropic(self, app_client):
        resp = await app_client.post(
            "/v1/messages",
            json={
                "model": "nosuchmodel",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 404
        assert resp.json()["error"]["type"] == "not_found_error"

    def test_is_value_error_subclass(self):
        # Legacy ``except ValueError`` callers keep working.
        from olmlx.engine.model_manager import ModelNotFoundError

        assert issubclass(ModelNotFoundError, ValueError)

    @pytest.mark.asyncio
    async def test_warmup(self, app_client):
        resp = await app_client.post("/api/warmup", json={"model": "nosuchmodel"})
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_speech(self, app_client):
        resp = await app_client.post(
            "/v1/audio/speech",
            json={"model": "nosuchmodel", "input": "hi", "response_format": "wav"},
        )
        assert resp.status_code == 404
