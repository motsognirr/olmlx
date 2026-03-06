"""Tests for mlx_ollama.routers.chat."""

from unittest.mock import AsyncMock, patch

import pytest

from mlx_ollama.utils.timing import TimingStats


class TestChatRouter:
    @pytest.mark.asyncio
    async def test_chat_non_streaming(self, app_client):
        stats = TimingStats(eval_count=10)
        mock_result = {"text": "Hello!", "done": True, "stats": stats}

        with patch(
            "mlx_ollama.routers.chat.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            resp = await app_client.post(
                "/api/chat",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["done"] is True
        assert data["message"]["content"] == "Hello!"
        assert data["message"]["role"] == "assistant"
        assert data["done_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_chat_streaming(self, app_client):
        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {"text": "Hi", "done": False}
                yield {"text": "", "done": True, "stats": TimingStats()}

            return gen()

        with patch("mlx_ollama.routers.chat.generate_chat", side_effect=mock_stream):
            resp = await app_client.post(
                "/api/chat",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        assert "ndjson" in resp.headers["content-type"]

    @pytest.mark.asyncio
    async def test_chat_with_options(self, app_client):
        mock_result = {"text": "hi", "done": True, "stats": TimingStats()}

        with patch(
            "mlx_ollama.routers.chat.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            resp = await app_client.post(
                "/api/chat",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                    "options": {"temperature": 0.5, "num_predict": 100},
                },
            )

        assert resp.status_code == 200
        # Verify options were passed
        call_args = mock_gen.call_args
        assert call_args[1]["max_tokens"] == 100
