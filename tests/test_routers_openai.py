"""Tests for olmlx.routers.openai."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from olmlx.utils.timing import TimingStats


class TestOpenAIRouter:
    @pytest.mark.asyncio
    async def test_list_models(self, app_client):
        resp = await app_client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) >= 2

    @pytest.mark.asyncio
    async def test_chat_completions_non_streaming(self, app_client):
        stats = TimingStats(prompt_eval_count=10, eval_count=20)
        mock_result = {"text": "Hello!", "done": True, "stats": stats}

        with patch(
            "olmlx.routers.openai.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            resp = await app_client.post(
                "/v1/chat/completions",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["content"] == "Hello!"
        assert data["choices"][0]["finish_reason"] == "stop"
        assert data["usage"]["prompt_tokens"] == 10
        assert data["usage"]["completion_tokens"] == 20

    @pytest.mark.asyncio
    async def test_chat_completions_streaming(self, app_client):
        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {"text": "Hello", "done": False}
                yield {"text": " world", "done": False}
                yield {"text": "", "done": True, "stats": TimingStats()}

            return gen()

        with patch("olmlx.routers.openai.generate_chat", side_effect=mock_stream):
            resp = await app_client.post(
                "/v1/chat/completions",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        lines = resp.text.strip().split("\n")
        # Should contain data lines and end with [DONE]
        assert any("[DONE]" in line for line in lines)

    @pytest.mark.asyncio
    async def test_completions_non_streaming(self, app_client):
        mock_result = {"text": "Completed text", "done": True, "stats": TimingStats()}

        with patch(
            "olmlx.routers.openai.generate_completion", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            resp = await app_client.post(
                "/v1/completions",
                json={
                    "model": "qwen3",
                    "prompt": "Once upon a time",
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "text_completion"
        assert data["choices"][0]["text"] == "Completed text"

    @pytest.mark.asyncio
    async def test_completions_list_prompt(self, app_client):
        mock_result = {"text": "result", "done": True, "stats": TimingStats()}

        with patch(
            "olmlx.routers.openai.generate_completion", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            resp = await app_client.post(
                "/v1/completions",
                json={
                    "model": "qwen3",
                    "prompt": ["first prompt", "second prompt"],
                },
            )

        assert resp.status_code == 200
        # Should use first prompt
        call_args = mock_gen.call_args
        assert call_args[1].get("prompt") is None or call_args[0][2] == "first prompt"

    @pytest.mark.asyncio
    async def test_embeddings(self, app_client):
        with patch(
            "olmlx.routers.openai.generate_embeddings", new_callable=AsyncMock
        ) as mock_emb:
            mock_emb.return_value = [[0.1, 0.2, 0.3]]
            resp = await app_client.post(
                "/v1/embeddings",
                json={
                    "model": "qwen3",
                    "input": "hello world",
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["embedding"] == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_embeddings_list_input(self, app_client):
        with patch(
            "olmlx.routers.openai.generate_embeddings", new_callable=AsyncMock
        ) as mock_emb:
            mock_emb.return_value = [[0.1], [0.2]]
            resp = await app_client.post(
                "/v1/embeddings",
                json={
                    "model": "qwen3",
                    "input": ["hello", "world"],
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["data"]) == 2

    @pytest.mark.asyncio
    async def test_completions_streaming(self, app_client):
        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {"text": "Once", "done": False}
                yield {"text": " upon", "done": False}
                yield {"text": "", "done": True, "stats": TimingStats()}

            return gen()

        with patch("olmlx.routers.openai.generate_completion", side_effect=mock_stream):
            resp = await app_client.post(
                "/v1/completions",
                json={
                    "model": "qwen3",
                    "prompt": "Once upon a time",
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        assert "[DONE]" in resp.text

    @pytest.mark.asyncio
    async def test_chat_options_mapping(self, app_client):
        mock_result = {"text": "hi", "done": True, "stats": TimingStats()}

        with patch(
            "olmlx.routers.openai.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            resp = await app_client.post(
                "/v1/chat/completions",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "hi"}],
                    "temperature": 0.5,
                    "top_p": 0.9,
                    "seed": 42,
                    "frequency_penalty": 0.3,
                    "presence_penalty": 0.2,
                    "stop": "END",
                },
            )

        assert resp.status_code == 200
        call_kwargs = mock_gen.call_args
        options = call_kwargs[0][3]  # 4th positional arg is options
        assert options["temperature"] == 0.5
        assert options["top_p"] == 0.9
        assert options["seed"] == 42
        assert options["frequency_penalty"] == 0.3
        assert options["stop"] == ["END"]

    @pytest.mark.asyncio
    async def test_chat_streaming_error_mid_stream(self, app_client):
        """Error during streaming emits an SSE error event instead of crashing."""

        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {"text": "partial", "done": False}
                raise RuntimeError("GPU exploded")

            return gen()

        with patch("olmlx.routers.openai.generate_chat", side_effect=mock_stream):
            resp = await app_client.post(
                "/v1/chat/completions",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        lines = resp.text.strip().split("\n")
        error_line = None
        for line in lines:
            if line.startswith("data:") and "server_error" in line:
                error_line = json.loads(line[5:].strip())
                break
        assert error_line is not None
        assert error_line["error"]["type"] == "server_error"
        assert "internal server error" in error_line["error"]["message"]
        assert any("[DONE]" in line for line in lines)


class TestXCacheIDHeader:
    @pytest.mark.asyncio
    async def test_header_passed_to_generate_chat(self, app_client):
        stats = TimingStats(prompt_eval_count=10, eval_count=5)
        mock_result = {"text": "response", "done": True, "stats": stats}

        with patch(
            "olmlx.routers.openai.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            resp = await app_client.post(
                "/v1/chat/completions",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "hi"}],
                },
                headers={"X-Cache-ID": "agent-beta"},
            )

        assert resp.status_code == 200
        mock_gen.assert_called_once()
        assert mock_gen.call_args.kwargs.get("cache_id") == "agent-beta"

    @pytest.mark.asyncio
    async def test_no_header_uses_default_cache_id(self, app_client):
        stats = TimingStats(prompt_eval_count=10, eval_count=5)
        mock_result = {"text": "response", "done": True, "stats": stats}

        with patch(
            "olmlx.routers.openai.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            resp = await app_client.post(
                "/v1/chat/completions",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )

        assert resp.status_code == 200
        mock_gen.assert_called_once()
        assert mock_gen.call_args.kwargs.get("cache_id") == ""
