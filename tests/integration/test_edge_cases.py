"""Tests for edge cases: zero-token generation and empty message lists."""

from tests.integration.conftest import set_stream_responses


# ---------------------------------------------------------------------------
# Zero-token generation edge cases
# ---------------------------------------------------------------------------


class TestZeroTokenGeneration:
    """Verify well-formed responses when generation produces zero tokens."""

    async def test_generate_zero_max_tokens(self, integration_ctx):
        """POST /api/generate with num_predict: 0 returns well-formed response."""
        set_stream_responses([""])
        resp = await integration_ctx.client.post(
            "/api/generate",
            json={
                "model": "qwen3",
                "prompt": "Hello",
                "stream": False,
                "options": {"num_predict": 0},
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["done"] is True
        assert "response" in data
        # No division-by-zero — TPS fields should be present or absent, not NaN
        for key in ("prompt_eval_duration", "eval_duration"):
            if key in data and data[key] is not None:
                assert isinstance(data[key], (int, float))

    async def test_chat_zero_max_tokens(self, integration_ctx):
        """POST /api/chat with num_predict: 0 returns well-formed response."""
        set_stream_responses([""])
        resp = await integration_ctx.client.post(
            "/api/chat",
            json={
                "model": "qwen3",
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
                "options": {"num_predict": 0},
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["done"] is True
        assert "message" in data
        assert data["message"]["role"] == "assistant"

    async def test_openai_zero_max_tokens_rejected(self, integration_ctx):
        """POST /v1/chat/completions with max_tokens: 0 is rejected (ge=1)."""
        # OpenAI schema requires max_tokens >= 1, so 0 should be a validation error
        resp = await integration_ctx.client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 0,
            },
        )
        assert resp.status_code == 400

    async def test_openai_zero_max_completion_tokens_rejected(self, integration_ctx):
        """POST /v1/chat/completions with max_completion_tokens: 0 is rejected."""
        resp = await integration_ctx.client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_completion_tokens": 0,
            },
        )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Empty message list edge cases
# ---------------------------------------------------------------------------


class TestEmptyMessages:
    """Verify behavior when message list is empty."""

    async def test_chat_empty_messages(self, integration_ctx):
        """POST /api/chat with messages: [] returns an error or valid response."""
        set_stream_responses([""])
        resp = await integration_ctx.client.post(
            "/api/chat",
            json={
                "model": "qwen3",
                "messages": [],
                "stream": False,
            },
        )
        # Empty messages may be accepted (model generates from empty prompt)
        # or rejected. Either way, no crash.
        assert resp.status_code in (200, 400, 422)
        # If 200, verify well-formed
        if resp.status_code == 200:
            data = resp.json()
            assert data["done"] is True
            assert "message" in data

    async def test_openai_empty_messages(self, integration_ctx):
        """POST /v1/chat/completions with messages: [] returns an error or valid response."""
        set_stream_responses([""])
        resp = await integration_ctx.client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3",
                "messages": [],
                "stream": False,
            },
        )
        # Pydantic may accept empty list or reject it
        assert resp.status_code in (200, 400, 422)
        if resp.status_code == 200:
            data = resp.json()
            assert "choices" in data

    async def test_anthropic_empty_messages(self, integration_ctx):
        """POST /v1/messages with messages: [] returns an error or valid response."""
        set_stream_responses([""])
        resp = await integration_ctx.client.post(
            "/v1/messages",
            json={
                "model": "qwen3",
                "max_tokens": 100,
                "messages": [],
                "stream": False,
            },
        )
        # Anthropic API requires at least one message, but our schema may not
        # enforce it. Either way, no crash.
        assert resp.status_code in (200, 400, 422)
        if resp.status_code == 200:
            data = resp.json()
            assert "content" in data
