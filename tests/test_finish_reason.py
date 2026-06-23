"""Tests that finish_reason='length' / stop_reason='max_tokens' is propagated
when max_tokens is exhausted (issue #543).

Router-level tests: mock generate_chat/generate_completion to return
done_reason='length' and verify each API surface maps it correctly.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest

from olmlx.utils.timing import TimingStats


def _done_chunk(done_reason=None):
    chunk = {"text": "hi", "done": True, "stats": TimingStats(eval_count=1)}
    if done_reason is not None:
        chunk["done_reason"] = done_reason
    return chunk


class TestOpenAIFinishReasonLength:
    @pytest.mark.asyncio
    async def test_chat_completions_non_streaming_length(self, app_client):
        with patch(
            "olmlx.routers.openai.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = _done_chunk("length")
            resp = await app_client.post(
                "/v1/chat/completions",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 1,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["finish_reason"] == "length"

    @pytest.mark.asyncio
    async def test_chat_completions_non_streaming_stop_unchanged(self, app_client):
        """EOS (no done_reason) must still produce finish_reason='stop'."""
        with patch(
            "olmlx.routers.openai.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = _done_chunk()
            resp = await app_client.post(
                "/v1/chat/completions",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_chat_completions_streaming_length(self, app_client):
        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {"text": "hi", "done": False}
                yield {
                    "text": "",
                    "done": True,
                    "stats": TimingStats(),
                    "done_reason": "length",
                }

            return gen()

        with patch("olmlx.routers.openai.generate_chat", side_effect=mock_stream):
            resp = await app_client.post(
                "/v1/chat/completions",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 1,
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        text = resp.text
        chunks = [
            json.loads(line[len("data: ") :])
            for line in text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        # The last non-DONE chunk carries finish_reason
        finish_chunks = [
            c
            for c in chunks
            if c.get("choices") and c["choices"][0].get("finish_reason") is not None
        ]
        assert finish_chunks, "No chunk with finish_reason found"
        assert finish_chunks[-1]["choices"][0]["finish_reason"] == "length"

    @pytest.mark.asyncio
    async def test_completions_non_streaming_length(self, app_client):
        with patch(
            "olmlx.routers.openai.generate_completion", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = _done_chunk("length")
            resp = await app_client.post(
                "/v1/completions",
                json={
                    "model": "qwen3",
                    "prompt": "Hello",
                    "max_tokens": 1,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["finish_reason"] == "length"


class TestAnthropicStopReasonMaxTokens:
    @pytest.mark.asyncio
    async def test_messages_non_streaming_max_tokens(self, app_client):
        with patch(
            "olmlx.routers.anthropic.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = _done_chunk("length")
            resp = await app_client.post(
                "/v1/messages",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 1,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["stop_reason"] == "max_tokens"

    @pytest.mark.asyncio
    async def test_messages_non_streaming_end_turn_unchanged(self, app_client):
        """EOS (no done_reason) must still produce stop_reason='end_turn'."""
        with patch(
            "olmlx.routers.anthropic.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = _done_chunk()
            resp = await app_client.post(
                "/v1/messages",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 100,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["stop_reason"] == "end_turn"

    @pytest.mark.asyncio
    async def test_messages_streaming_max_tokens(self, app_client):
        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {"text": "hi", "done": False}
                yield {
                    "text": "",
                    "done": True,
                    "stats": TimingStats(),
                    "done_reason": "length",
                }

            return gen()

        with patch("olmlx.routers.anthropic.generate_chat", side_effect=mock_stream):
            resp = await app_client.post(
                "/v1/messages",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 1,
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        text = resp.text
        # Find the message_delta event which carries stop_reason
        delta_data = [
            json.loads(line[len("data: ") :])
            for line in text.splitlines()
            if line.startswith("data: ") and "message_delta" in line
        ]
        assert delta_data, "No message_delta found in stream"
        assert delta_data[0]["delta"]["stop_reason"] == "max_tokens"


class TestResponsesFinishReasonLength:
    @pytest.mark.asyncio
    async def test_responses_non_streaming_incomplete(self, app_client):
        with patch(
            "olmlx.routers.responses.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = _done_chunk("length")
            resp = await app_client.post(
                "/v1/responses",
                json={
                    "model": "qwen3",
                    "input": "hi",
                    "max_output_tokens": 1,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "incomplete"
        assert data["incomplete_details"]["reason"] == "max_output_tokens"
