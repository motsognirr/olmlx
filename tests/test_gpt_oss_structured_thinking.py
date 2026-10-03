"""Structured ``{"thinking": ...}`` stream chunks reach every surface (#713).

The engine streams gpt-oss analysis-channel text as structured thinking
chunks rather than in-band ``<think>`` markup. Every streaming consumer must
route them to its thinking field (or drop them where the surface has none) —
and a reasoning trace that mentions ``</think>`` must never leak into content.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest

from olmlx.utils.timing import TimingStats

THINKING = ["The tag </think> ", "ends reasoning."]
CONTENT = ["The answer", " is 56."]


def _gpt_oss_stream(*, done_extra=None):
    async def mock_stream(*args, **kwargs):
        async def gen():
            yield {"thinking_expected": False}
            for t in THINKING:
                yield {"thinking": t, "done": False}
            for c in CONTENT:
                yield {"text": c, "done": False}
            yield {
                "text": "",
                "done": True,
                "stats": TimingStats(),
                **(done_extra or {}),
            }

        return gen()

    return mock_stream


def _ndjson(text):
    return [json.loads(line) for line in text.strip().split("\n") if line]


def _sse_data(text):
    out = []
    for line in text.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            try:
                out.append(json.loads(line[6:]))
            except json.JSONDecodeError:
                pass
    return out


class TestRouters:
    @pytest.mark.asyncio
    async def test_ollama_chat(self, app_client):
        with patch("olmlx.routers.chat.generate_chat", side_effect=_gpt_oss_stream()):
            resp = await app_client.post(
                "/api/chat",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "7*8"}],
                    "stream": True,
                },
            )
        assert resp.status_code == 200
        lines = _ndjson(resp.text)
        thinking = "".join(ln["message"].get("thinking", "") for ln in lines)
        content = "".join(ln["message"].get("content", "") for ln in lines)
        assert thinking == "".join(THINKING)
        assert content == "".join(CONTENT)

    @pytest.mark.asyncio
    async def test_ollama_generate(self, app_client):
        with patch(
            "olmlx.routers.generate.generate_completion",
            side_effect=_gpt_oss_stream(),
        ):
            resp = await app_client.post(
                "/api/generate",
                json={"model": "qwen3", "prompt": "7*8", "stream": True},
            )
        assert resp.status_code == 200
        lines = _ndjson(resp.text)
        assert "".join(ln.get("thinking", "") for ln in lines) == "".join(THINKING)
        assert "".join(ln.get("response", "") for ln in lines) == "".join(CONTENT)

    @pytest.mark.asyncio
    async def test_ollama_generate_non_streaming_uses_engine_thinking(self, app_client):
        """gpt-oss non-streaming: the engine already parsed the channels into
        ``result["thinking"]`` — /api/generate must surface it."""
        with patch(
            "olmlx.routers.generate.generate_completion", new_callable=AsyncMock
        ) as gen:
            gen.return_value = {
                "text": "".join(CONTENT),
                "thinking": "".join(THINKING),
                "done": True,
                "stats": TimingStats(),
            }
            resp = await app_client.post(
                "/api/generate",
                json={"model": "qwen3", "prompt": "7*8", "stream": False},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["thinking"] == "".join(THINKING)
        assert body["response"] == "".join(CONTENT)

    @pytest.mark.asyncio
    async def test_anthropic_messages(self, app_client):
        with patch(
            "olmlx.routers.anthropic.generate_chat", side_effect=_gpt_oss_stream()
        ):
            resp = await app_client.post(
                "/v1/messages",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "7*8"}],
                    "max_tokens": 100,
                    "stream": True,
                },
            )
        assert resp.status_code == 200
        events = _sse_data(resp.text)
        starts = [
            e["content_block"]["type"]
            for e in events
            if e.get("type") == "content_block_start"
        ]
        assert starts == ["thinking", "text"]
        deltas = [e["delta"] for e in events if e.get("type") == "content_block_delta"]
        thinking = "".join(d.get("thinking", "") for d in deltas)
        text = "".join(d.get("text", "") for d in deltas)
        assert thinking == "".join(THINKING)
        assert text == "".join(CONTENT)

    @pytest.mark.asyncio
    async def test_responses(self, app_client):
        with patch(
            "olmlx.routers.responses.generate_chat", side_effect=_gpt_oss_stream()
        ):
            resp = await app_client.post(
                "/v1/responses",
                json={"model": "qwen3", "input": "7*8", "stream": True},
            )
        assert resp.status_code == 200
        text = resp.text
        events = _sse_data(text)
        reasoning = "".join(
            e.get("delta", "")
            for e in events
            if e.get("type") == "response.reasoning_text.delta"
        )
        output = "".join(
            e.get("delta", "")
            for e in events
            if e.get("type") == "response.output_text.delta"
        )
        assert reasoning == "".join(THINKING)
        assert output == "".join(CONTENT)

    @pytest.mark.asyncio
    async def test_openai_chat_drops_thinking(self, app_client):
        """Chat completions has no thinking field: reasoning is dropped, never
        leaked into content."""
        with patch("olmlx.routers.openai.generate_chat", side_effect=_gpt_oss_stream()):
            resp = await app_client.post(
                "/v1/chat/completions",
                json={
                    "model": "qwen3",
                    "messages": [{"role": "user", "content": "7*8"}],
                    "stream": True,
                },
            )
        assert resp.status_code == 200
        content = "".join(
            (c.get("delta") or {}).get("content") or ""
            for e in _sse_data(resp.text)
            for c in e.get("choices", [])
        )
        assert content == "".join(CONTENT)


class TestThinkingTracker:
    def test_structured_thinking_then_content(self):
        from olmlx.chat.session import ThinkingTracker

        t = ThinkingTracker()
        think, started = t.feed_thinking("The tag </think> ")
        assert (think, started) == ("The tag </think> ", True)
        assert t.in_thinking
        think, started = t.feed_thinking("ends reasoning.")
        assert (think, started) == ("ends reasoning.", False)
        think, visible, ended, started = t.feed("The answer")
        assert (think, visible, ended, started) == (None, "The answer", True, False)
        # Thinking is not part of the text parsed for tool calls at turn end.
        assert t.accumulated == "The answer"

    def test_structured_thinking_hidden_when_disabled(self):
        from olmlx.chat.session import ThinkingTracker

        t = ThinkingTracker(thinking_disabled=True)
        assert t.feed_thinking("reasoning") == (None, False)
        assert not t.in_thinking
