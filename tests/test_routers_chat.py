"""Tests for olmlx.routers.chat."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from olmlx.routers.chat import _flush_split_thinking, _split_thinking_streaming
from olmlx.utils.timing import TimingStats


class TestSplitThinkingStreaming:
    """Unit tests for the Ollama-side thinking splitter (issue #307)."""

    def test_flush_in_think_phase_treats_buffer_as_thinking(self):
        """If the stream ends with an open `<think>` and no closer, the held
        buffer must be returned as thinking content (not silently lost)."""
        # Walk through chunks ending mid-think.
        state: dict = {}
        thinking, content = _split_thinking_streaming("<think>partial reasoning", state)
        # `<think>` consumed; "partial reasoning" is 17 chars in `in_think`
        # phase — the splitter emits all but the last 8 chars and holds the
        # tail in case `</think>` is straddling a chunk boundary.
        assert thinking == "partial r"
        assert content == ""
        assert state["phase"] == "in_think"
        assert state["buffer"] == "easoning"

        # Stream ends — flush must surface the buffered remainder as thinking.
        tail_thinking, tail_content = _flush_split_thinking(state)
        assert tail_thinking == "easoning"
        assert tail_content == ""


class TestChatRouter:
    @pytest.mark.asyncio
    async def test_chat_non_streaming(self, app_client):
        stats = TimingStats(eval_count=10)
        mock_result = {"text": "Hello!", "done": True, "stats": stats}

        with patch(
            "olmlx.routers.chat.generate_chat", new_callable=AsyncMock
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

        with patch("olmlx.routers.chat.generate_chat", side_effect=mock_stream):
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
            "olmlx.routers.chat.generate_chat", new_callable=AsyncMock
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

    @pytest.mark.asyncio
    async def test_chat_non_streaming_separates_thinking_from_content(self, app_client):
        """Issue #307: ``<think>...</think>`` and Qwen3.5-style orphan
        ``</think>`` thinking must be separated from ``message.content`` and
        surfaced under ``message.thinking`` to match Ollama's API."""
        # Qwen3.5 case: no opening <think>, closing tag present.
        raw = (
            "Thinking Process:\n\n"
            "1. Analyze the Request: The user wants 17 * 23.\n"
            "2. Compute: 17 * 23 = 391.\n"
            "3. Construct Final Response: 391\n"
            "</think>\n\n"
            "391"
        )
        # Engine signals thinking_expected via the non-streaming result dict,
        # mirroring the streaming meta chunk (issue #307 review).
        mock_result = {
            "text": raw,
            "done": True,
            "stats": TimingStats(eval_count=10),
            "thinking_expected": True,
        }

        with patch(
            "olmlx.routers.chat.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            resp = await app_client.post(
                "/api/chat",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "17*23"}],
                    "stream": False,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        msg = data["message"]
        assert msg["content"].strip() == "391"
        assert "</think>" not in msg["content"]
        assert "Thinking Process" not in msg["content"]
        assert "Thinking Process" in msg.get("thinking", "")
        assert "</think>" not in msg.get("thinking", "")

    @pytest.mark.asyncio
    async def test_chat_streaming_separates_thinking_from_content(self, app_client):
        """Issue #307: streaming path must put thinking into ``message.thinking``
        and not leak it into ``message.content`` — even when the orphan
        preamble is longer than the conservative non-thinking buffer."""
        preamble = (
            "Thinking Process:\n\n"
            "1. Analyze the Request: The user wants the product of 17 and 23.\n"
            "2. Compute: 17 * 23 = (17 * 20) + (17 * 3) = 340 + 51 = 391.\n"
            "3. Sanity check: 17 * 25 - 2 * 17 = 425 - 34 = 391.\n"
            "4. Format: just the number, no prose.\n"
            "5. Construct Final Response: 391"
        )
        # Must exceed the conservative 200-char limit used when thinking is
        # not expected so the test actually exercises the plumbed path.
        assert len(preamble) > 200

        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {"thinking_expected": True}
                for i in range(0, len(preamble), 16):
                    yield {"text": preamble[i : i + 16], "done": False}
                yield {"text": "\n</think>\n\n", "done": False}
                yield {"text": "391", "done": False}
                yield {"text": "", "done": True, "stats": TimingStats()}

            return gen()

        with patch("olmlx.routers.chat.generate_chat", side_effect=mock_stream):
            resp = await app_client.post(
                "/api/chat",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "17*23"}],
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        content = ""
        thinking = ""
        for line in resp.text.strip().split("\n"):
            if not line.strip():
                continue
            payload = json.loads(line)
            msg = payload.get("message", {})
            content += msg.get("content", "")
            thinking += msg.get("thinking", "") or ""

        assert "Thinking Process" in thinking
        assert "Thinking Process" not in content
        assert "</think>" not in content
        assert "</think>" not in thinking
        assert content.strip() == "391"

    @pytest.mark.asyncio
    async def test_chat_streaming_short_direct_answer_in_non_done_chunk(
        self, app_client
    ):
        """Issue #307 review round 9: when `thinking_expected=True` and the
        model gives a short direct answer (fits in the orphan-detect buffer
        without any `</think>`), the buffered content must be flushed to a
        non-done chunk before the done marker.  Standard Ollama clients
        only accumulate `message.content` from non-done chunks; putting
        accumulated content in the done frame's `message.content` would
        be silently dropped."""

        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {"thinking_expected": True}
                yield {"text": "The answer is 42.", "done": False}
                yield {"text": "", "done": True, "stats": TimingStats()}

            return gen()

        with patch("olmlx.routers.chat.generate_chat", side_effect=mock_stream):
            resp = await app_client.post(
                "/api/chat",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "?"}],
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        lines = [line for line in resp.text.strip().split("\n") if line.strip()]
        # Walk all lines; non-done chunks should carry the content.
        non_done_content = ""
        done_content = ""
        for line in lines:
            payload = json.loads(line)
            msg = payload.get("message", {})
            if payload.get("done"):
                done_content += msg.get("content", "")
            else:
                non_done_content += msg.get("content", "")
        # All content must be in non-done chunks; the done marker carries
        # an empty content field per the Ollama convention.
        assert non_done_content == "The answer is 42."
        assert done_content == ""

    @pytest.mark.asyncio
    async def test_chat_non_streaming_literal_close_think_preserved_when_not_thinking(
        self, app_client
    ):
        """Issue #307 review: the non-streaming path must agree with the
        streaming path — a literal `</think>` token from a non-thinking
        model must stay in `message.content`, not be silently routed to
        `message.thinking`."""
        raw = "Use </think> to close the thought block."
        mock_result = {
            "text": raw,
            "done": True,
            "stats": TimingStats(eval_count=10),
            "thinking_expected": False,
        }

        with patch(
            "olmlx.routers.chat.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            resp = await app_client.post(
                "/api/chat",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "syntax?"}],
                    "stream": False,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        msg = data["message"]
        assert msg["content"] == raw
        assert "thinking" not in msg or msg["thinking"] is None

    @pytest.mark.asyncio
    async def test_chat_streaming_literal_close_think_preserved_when_not_thinking(
        self, app_client
    ):
        """When `thinking_expected=False`, a literal `</think>` in the
        output must stay in `message.content` and not be silently routed
        to `message.thinking`."""

        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {"thinking_expected": False}
                yield {
                    "text": "Use </think> to close the thought block.",
                    "done": False,
                }
                yield {"text": "", "done": True, "stats": TimingStats()}

            return gen()

        with patch("olmlx.routers.chat.generate_chat", side_effect=mock_stream):
            resp = await app_client.post(
                "/api/chat",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "syntax?"}],
                    "stream": True,
                },
            )

        assert resp.status_code == 200
        content = ""
        thinking = ""
        for line in resp.text.strip().split("\n"):
            if not line.strip():
                continue
            payload = json.loads(line)
            msg = payload.get("message", {})
            content += msg.get("content", "")
            thinking += msg.get("thinking", "") or ""

        assert thinking == ""
        assert "</think>" in content
        assert content == "Use </think> to close the thought block."

    @pytest.mark.asyncio
    async def test_chat_streaming_error_mid_stream(self, app_client):
        """Error during streaming emits an NDJSON error line instead of crashing."""

        async def mock_stream(*args, **kwargs):
            async def gen():
                yield {"text": "partial", "done": False}
                raise RuntimeError("GPU exploded")

            return gen()

        with patch("olmlx.routers.chat.generate_chat", side_effect=mock_stream):
            resp = await app_client.post(
                "/api/chat",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "hi"}],
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


class TestXCacheIDHeader:
    @pytest.mark.asyncio
    async def test_header_passed_to_generate_chat(self, app_client):
        stats = TimingStats()
        mock_result = {"text": "response", "done": True, "stats": stats}

        with patch(
            "olmlx.routers.chat.generate_chat", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = mock_result
            resp = await app_client.post(
                "/api/chat",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                },
                headers={"X-Cache-ID": "agent-gamma"},
            )

        assert resp.status_code == 200
        mock_gen.assert_called_once()
        assert mock_gen.call_args.kwargs.get("cache_id") == "agent-gamma"

    @pytest.mark.asyncio
    async def test_no_header_uses_default_cache_id(self, app_client):
        stats = TimingStats()
        mock_result = {"text": "response", "done": True, "stats": stats}

        with patch(
            "olmlx.routers.chat.generate_chat", new_callable=AsyncMock
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
        mock_gen.assert_called_once()
        assert mock_gen.call_args.kwargs.get("cache_id") == ""
