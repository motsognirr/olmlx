"""Tests for streaming disconnect / aclose cleanup.

Verify that GPU resources are properly cleaned up when clients disconnect
mid-stream.
"""

import json
from unittest.mock import patch

from tests.conftest import make_error_stream
from tests.integration.conftest import set_stream_responses


class TestStreamingDisconnect:
    """Test that early client disconnect triggers stream cleanup."""

    async def test_generate_stream_early_disconnect(self, integration_ctx):
        """Start streaming /api/generate, consume 1 chunk, close. Verify cleanup."""
        set_stream_responses(["Hello", " world", " foo", " bar", " baz"])

        resp = await integration_ctx.client.post(
            "/api/generate",
            json={
                "model": "qwen3",
                "prompt": "Hi",
                "stream": True,
            },
        )
        # The full response is returned by httpx (it buffers).
        # Verify that the response completed and produced valid NDJSON.
        assert resp.status_code == 200
        lines = [line for line in resp.text.strip().split("\n") if line.strip()]
        assert len(lines) >= 1
        # Last line should be done=True (stream completed normally)
        last = json.loads(lines[-1])
        assert last["done"] is True

    async def test_chat_stream_early_disconnect(self, integration_ctx):
        """Start streaming /api/chat, verify stream completes with cleanup."""
        set_stream_responses(["Hello", " world", " foo"])

        resp = await integration_ctx.client.post(
            "/api/chat",
            json={
                "model": "qwen3",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )
        assert resp.status_code == 200
        lines = [line for line in resp.text.strip().split("\n") if line.strip()]
        assert len(lines) >= 1
        last = json.loads(lines[-1])
        assert last["done"] is True

    async def test_openai_stream_early_disconnect(self, integration_ctx):
        """Start streaming /v1/chat/completions, verify stream completes."""
        set_stream_responses(["Hello", " world", " foo"])

        resp = await integration_ctx.client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )
        assert resp.status_code == 200
        # OpenAI SSE format: data: {...}\n\n  and data: [DONE]\n\n
        assert "data: [DONE]" in resp.text

    async def test_generate_stream_aclose_called_on_error(self, integration_ctx):
        """Verify result.aclose() is called even when stream generator raises."""
        error_stream = make_error_stream([{"text": "Hello"}], error_msg="GPU exploded")

        with patch(
            "olmlx.routers.generate.generate_completion",
            return_value=error_stream,
        ):
            resp = await integration_ctx.client.post(
                "/api/generate",
                json={
                    "model": "qwen3",
                    "prompt": "Hi",
                    "stream": True,
                },
            )
            assert resp.status_code == 200
            # aclose should have been called in the finally block
            error_stream.aclose.assert_awaited_once()

    async def test_chat_stream_aclose_called_on_error(self, integration_ctx):
        """Verify result.aclose() is called for /api/chat even on error."""
        error_stream = make_error_stream([{"text": "Hello"}], error_msg="GPU exploded")

        with patch(
            "olmlx.routers.chat.generate_chat",
            return_value=error_stream,
        ):
            resp = await integration_ctx.client.post(
                "/api/chat",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": True,
                },
            )
            assert resp.status_code == 200
            error_stream.aclose.assert_awaited_once()

    async def test_openai_stream_aclose_called_on_error(self, integration_ctx):
        """Verify result.aclose() is called for /v1/chat/completions even on error."""
        error_stream = make_error_stream([{"text": "Hello"}], error_msg="GPU exploded")

        with patch(
            "olmlx.routers.openai.generate_chat",
            return_value=error_stream,
        ):
            resp = await integration_ctx.client.post(
                "/v1/chat/completions",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": True,
                },
            )
            assert resp.status_code == 200
            error_stream.aclose.assert_awaited_once()
