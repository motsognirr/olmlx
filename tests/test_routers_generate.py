"""Tests for olmlx.routers.generate."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from olmlx.utils.timing import TimingStats


class TestGenerateRouter:
    @pytest.mark.asyncio
    async def test_generate_non_streaming(self, app_client):
        stats = TimingStats(eval_count=5)
        mock_result = {"text": "Generated text", "done": True, "stats": stats}

        with patch(
            "olmlx.routers.generate.generate_completion", new_callable=AsyncMock
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
            "olmlx.routers.generate.generate_completion", side_effect=mock_stream
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
        """System prompt is passed as a separate kwarg, not prepended to prompt."""
        mock_result = {"text": "result", "done": True, "stats": TimingStats()}

        with patch(
            "olmlx.routers.generate.generate_completion", new_callable=AsyncMock
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
        call_args = mock_gen.call_args
        prompt = call_args[0][2]  # 3rd positional arg
        assert prompt == "Hello"  # system NOT prepended
        assert call_args.kwargs["system"] == "You are helpful"

    @pytest.mark.asyncio
    async def test_generate_raw_ignores_system(self, app_client):
        """In raw mode, system is not passed (and not prepended)."""
        mock_result = {"text": "result", "done": True, "stats": TimingStats()}

        with patch(
            "olmlx.routers.generate.generate_completion", new_callable=AsyncMock
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
        assert call_args.kwargs["system"] is None

    @pytest.mark.asyncio
    async def test_apply_chat_template_default(self, app_client):
        """By default (raw=False), apply_chat_template=True is passed."""
        mock_result = {"text": "result", "done": True, "stats": TimingStats()}

        with patch(
            "olmlx.routers.generate.generate_completion", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            await app_client.post(
                "/api/generate",
                json={"model": "qwen3", "prompt": "Hello", "stream": False},
            )

        assert mock_gen.call_args.kwargs["apply_chat_template"] is True

    @pytest.mark.asyncio
    async def test_raw_skips_chat_template(self, app_client):
        """When raw=True, apply_chat_template=False is passed."""
        mock_result = {"text": "result", "done": True, "stats": TimingStats()}

        with patch(
            "olmlx.routers.generate.generate_completion", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            await app_client.post(
                "/api/generate",
                json={
                    "model": "qwen3",
                    "prompt": "Hello",
                    "raw": True,
                    "stream": False,
                },
            )

        assert mock_gen.call_args.kwargs["apply_chat_template"] is False

    @pytest.mark.asyncio
    async def test_generate_think_default_none(self, app_client):
        """No `think` field -> enable_thinking=None (off-by-default preserved)."""
        mock_result = {"text": "out", "done": True, "stats": TimingStats()}
        with patch(
            "olmlx.routers.generate.generate_completion", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            resp = await app_client.post(
                "/api/generate",
                json={"model": "qwen3", "prompt": "Hello", "stream": False},
            )
        assert resp.status_code == 200
        assert mock_gen.call_args.kwargs["enable_thinking"] is None

    @pytest.mark.asyncio
    async def test_generate_think_true(self, app_client):
        mock_result = {"text": "out", "done": True, "stats": TimingStats()}
        with patch(
            "olmlx.routers.generate.generate_completion", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            resp = await app_client.post(
                "/api/generate",
                json={
                    "model": "qwen3",
                    "prompt": "Hello",
                    "stream": False,
                    "think": True,
                },
            )
        assert resp.status_code == 200
        assert mock_gen.call_args.kwargs["enable_thinking"] is True

    @pytest.mark.asyncio
    async def test_generate_non_streaming_splits_thinking(self, app_client):
        """<think> blocks are routed to the `thinking` field, not `response`."""
        mock_result = {
            "text": "<think>reasoning here</think>the answer",
            "done": True,
            "thinking_expected": True,
            "stats": TimingStats(),
        }
        with patch(
            "olmlx.routers.generate.generate_completion", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            resp = await app_client.post(
                "/api/generate",
                json={
                    "model": "qwen3",
                    "prompt": "Hello",
                    "stream": False,
                    "think": True,
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["response"] == "the answer"
        assert data["thinking"] == "reasoning here"

    @pytest.mark.asyncio
    async def test_generate_streaming_splits_thinking(self, app_client):
        """Streaming routes thinking into a `thinking` field across chunks."""

        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {"thinking_expected": True}
                yield {"text": "<think>reason", "done": False}
                yield {"text": "ing</think>", "done": False}
                yield {"text": "visible answer", "done": False}
                yield {"text": "", "done": True, "stats": TimingStats()}

            return gen()

        with patch(
            "olmlx.routers.generate.generate_completion", side_effect=mock_stream
        ):
            resp = await app_client.post(
                "/api/generate",
                json={
                    "model": "qwen3",
                    "prompt": "Hello",
                    "stream": True,
                    "think": True,
                },
            )
        assert resp.status_code == 200
        lines = [json.loads(ln) for ln in resp.text.strip().split("\n") if ln.strip()]
        response = "".join(ln.get("response", "") for ln in lines)
        thinking = "".join(ln.get("thinking") or "" for ln in lines)
        assert response == "visible answer"
        assert "<think>" not in response
        assert thinking == "reasoning"
        # The done frame must be emitted last and well-formed: the done-chunk
        # handler returns list[str] (multi-line flush), so this also guards the
        # safe_ndjson_stream list-return contract.
        assert lines[-1]["done"] is True
        assert lines[-1]["done_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_generate_streaming_error_mid_stream(self, app_client):
        """Error during streaming emits an NDJSON error line instead of crashing."""

        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {"text": "partial", "done": False}
                raise RuntimeError("GPU exploded")

            return gen()

        with patch(
            "olmlx.routers.generate.generate_completion", side_effect=mock_stream
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
        lines = [line for line in resp.text.strip().split("\n") if line.strip()]
        last_line = json.loads(lines[-1])
        assert "error" in last_line
        assert "internal server error" in last_line["error"]
        assert last_line["done"] is True
        assert last_line["done_reason"] == "error"
        assert last_line["model"] == "qwen3"
        assert "created_at" in last_line


class TestFormatField:
    """Ollama ``format`` field maps to a GrammarSpec (issue #361)."""

    @pytest.mark.asyncio
    async def test_format_json_string(self, app_client):
        from olmlx.engine.grammar import GrammarSpec

        mock_result = {"text": '{"a": 1}', "done": True, "stats": TimingStats()}
        with patch(
            "olmlx.routers.generate.generate_completion", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            resp = await app_client.post(
                "/api/generate",
                json={
                    "model": "qwen3",
                    "prompt": "hi",
                    "stream": False,
                    "format": "json",
                },
            )
        assert resp.status_code == 200
        spec = mock_gen.call_args.kwargs.get("grammar_spec")
        assert isinstance(spec, GrammarSpec)
        assert spec.kind == "json_object"

    @pytest.mark.asyncio
    async def test_invalid_format_returns_422_not_500(self, app_client):
        """Regression: see chat-router twin (review #384, bug 2)."""
        resp = await app_client.post(
            "/api/generate",
            json={
                "model": "qwen3",
                "prompt": "hi",
                "stream": False,
                "format": "xml",
            },
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_format_schema_dict(self, app_client):
        from olmlx.engine.grammar import GrammarSpec

        mock_result = {"text": '{"x": 1}', "done": True, "stats": TimingStats()}
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        with patch(
            "olmlx.routers.generate.generate_completion", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            resp = await app_client.post(
                "/api/generate",
                json={
                    "model": "qwen3",
                    "prompt": "hi",
                    "stream": False,
                    "format": schema,
                },
            )
        assert resp.status_code == 200
        spec = mock_gen.call_args.kwargs.get("grammar_spec")
        assert isinstance(spec, GrammarSpec)
        assert spec.kind == "json_schema"
        assert spec.schema == schema


class TestEmptyPromptRejected:
    @pytest.mark.asyncio
    async def test_api_generate_rejects_empty_prompt(self, app_client):
        resp = await app_client.post(
            "/api/generate",
            json={"model": "qwen3", "prompt": ""},
        )
        assert resp.status_code == 422
        body = resp.text.lower()
        assert "prompt" in body
        assert "empty" in body

    @pytest.mark.asyncio
    async def test_api_generate_rejects_missing_prompt(self, app_client):
        resp = await app_client.post(
            "/api/generate",
            json={"model": "qwen3"},
        )
        assert resp.status_code == 422
        body = resp.text.lower()
        assert "prompt" in body
