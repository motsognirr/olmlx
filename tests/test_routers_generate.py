"""Tests for mlx_ollama.routers.generate."""

from unittest.mock import AsyncMock, patch

import pytest

from mlx_ollama.utils.timing import TimingStats


class TestGenerateRouter:
    @pytest.mark.asyncio
    async def test_generate_non_streaming(self, app_client):
        stats = TimingStats(eval_count=5)
        mock_result = {"text": "Generated text", "done": True, "stats": stats}

        with patch(
            "mlx_ollama.routers.generate.generate_completion", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            resp = await app_client.post(
                "/api/generate",
                json={
                    "model": "qwen3",
                    "prompt": "Hello",
                    "stream": False,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["response"] == "Generated text"
        assert data["done"] is True
        assert data["done_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_generate_streaming(self, app_client):
        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {"text": "Hello", "done": False}
                yield {"text": "", "done": True, "stats": TimingStats()}

            return gen()

        with patch(
            "mlx_ollama.routers.generate.generate_completion", side_effect=mock_stream
        ):
            resp = await app_client.post(
                "/api/generate",
                json={
                    "model": "qwen3",
                    "prompt": "Hello",
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        assert "ndjson" in resp.headers["content-type"]

    @pytest.mark.asyncio
    async def test_generate_with_system(self, app_client):
        mock_result = {"text": "result", "done": True, "stats": TimingStats()}

        with patch(
            "mlx_ollama.routers.generate.generate_completion", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            resp = await app_client.post(
                "/api/generate",
                json={
                    "model": "qwen3",
                    "prompt": "Hello",
                    "system": "You are helpful",
                    "stream": False,
                },
            )

        assert resp.status_code == 200
        # System should be prepended to prompt
        call_args = mock_gen.call_args
        prompt = call_args[0][2]  # 3rd positional arg
        assert "You are helpful" in prompt
        assert "Hello" in prompt

    @pytest.mark.asyncio
    async def test_generate_raw_ignores_system(self, app_client):
        mock_result = {"text": "result", "done": True, "stats": TimingStats()}

        with patch(
            "mlx_ollama.routers.generate.generate_completion", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            resp = await app_client.post(
                "/api/generate",
                json={
                    "model": "qwen3",
                    "prompt": "Hello",
                    "system": "You are helpful",
                    "raw": True,
                    "stream": False,
                },
            )

        assert resp.status_code == 200
        call_args = mock_gen.call_args
        prompt = call_args[0][2]
        assert prompt == "Hello"
