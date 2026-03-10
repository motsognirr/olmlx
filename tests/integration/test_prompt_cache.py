"""Integration tests for prompt cache create/hit/miss/invalidation/SSE stats."""

import json

import pytest

from tests.integration.conftest import parse_sse_events, set_multi_stream_responses


async def _stream_anthropic(client, messages, model="qwen3"):
    """POST streaming /v1/messages and return raw response text."""
    resp = await client.post(
        "/v1/messages",
        json={
            "model": model,
            "max_tokens": 100,
            "stream": True,
            "messages": messages,
        },
    )
    assert resp.status_code == 200
    return resp.text


def _get_cache_stats(sse_text: str) -> dict:
    """Extract cache stats from message_start SSE event."""
    events = parse_sse_events(sse_text)
    for ev in events:
        if ev.get("event") == "message_start":
            usage = ev["data"]["message"]["usage"]
            return {
                "cache_creation_input_tokens": usage.get(
                    "cache_creation_input_tokens", 0
                ),
                "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
            }
    pytest.fail("No message_start event found in SSE output")


def _get_text_content(sse_text: str) -> str:
    """Reassemble text content from SSE events."""
    events = parse_sse_events(sse_text)
    text = ""
    for ev in events:
        if ev.get("event") == "content_block_delta":
            delta = ev["data"].get("delta", {})
            if delta.get("type") == "text_delta":
                text += delta.get("text", "")
    return text


async def test_cache_created_on_first_request(integration_ctx):
    """First streaming request creates a cache — cache_creation_input_tokens > 0."""
    messages = [{"role": "user", "content": "Hello"}]
    sse_text = await _stream_anthropic(integration_ctx.client, messages)

    stats = _get_cache_stats(sse_text)
    assert stats["cache_creation_input_tokens"] > 0
    assert stats["cache_read_input_tokens"] == 0

    # Verify cache state stored on manager
    lm = integration_ctx.manager._loaded["qwen3:latest"]
    assert lm.prompt_cache_state is not None


async def test_cache_hit_on_second_request(integration_ctx):
    """Second request with overlapping prefix gets cache_read > 0."""
    messages = [{"role": "user", "content": "Hello, tell me about Python"}]

    set_multi_stream_responses([["First", " response"], ["Second", " response"]])

    sse1 = await _stream_anthropic(integration_ctx.client, messages)
    stats1 = _get_cache_stats(sse1)
    assert stats1["cache_creation_input_tokens"] > 0

    # Same messages — prefix should match
    sse2 = await _stream_anthropic(integration_ctx.client, messages)
    stats2 = _get_cache_stats(sse2)
    assert stats2["cache_read_input_tokens"] > 0


async def test_cache_miss_on_different_prompt(integration_ctx):
    """Completely different prompt gets cache_read = 0."""
    set_multi_stream_responses([["Response", " one"], ["Response", " two"]])

    msgs1 = [
        {"role": "user", "content": "Hello, tell me about Python programming language"}
    ]
    sse1 = await _stream_anthropic(integration_ctx.client, msgs1)
    stats1 = _get_cache_stats(sse1)
    assert stats1["cache_creation_input_tokens"] > 0

    # Entirely different prompt
    msgs2 = [
        {
            "role": "user",
            "content": "XYZZY completely different text with no overlap at all",
        }
    ]
    sse2 = await _stream_anthropic(integration_ctx.client, msgs2)
    stats2 = _get_cache_stats(sse2)
    # Should create fresh cache, not read from old one
    assert stats2["cache_creation_input_tokens"] > 0


async def test_cache_stats_in_sse(integration_ctx):
    """Parse full SSE and verify message_start has correct cache stats structure."""
    messages = [{"role": "user", "content": "Hello"}]
    sse_text = await _stream_anthropic(integration_ctx.client, messages)

    events = parse_sse_events(sse_text)
    msg_start = [e for e in events if e.get("event") == "message_start"]
    assert len(msg_start) == 1

    msg = msg_start[0]["data"]["message"]
    assert "usage" in msg
    usage = msg["usage"]
    assert "cache_creation_input_tokens" in usage
    assert "cache_read_input_tokens" in usage
    assert "input_tokens" in usage
    assert "output_tokens" in usage


async def test_cache_disabled_via_config(integration_ctx, monkeypatch):
    """With prompt_cache=False, cache_creation_input_tokens == 0."""
    monkeypatch.setattr("olmlx.config.settings.prompt_cache", False)

    messages = [{"role": "user", "content": "Hello"}]
    sse_text = await _stream_anthropic(integration_ctx.client, messages)

    stats = _get_cache_stats(sse_text)
    assert stats["cache_creation_input_tokens"] == 0
    assert stats["cache_read_input_tokens"] == 0


async def test_cache_survives_multi_turn(integration_ctx):
    """Multiple sequential requests building conversation — cache_read grows."""
    turns = [
        [{"role": "user", "content": "Hello, I want to discuss Python"}],
        [
            {"role": "user", "content": "Hello, I want to discuss Python"},
            {"role": "assistant", "content": "Sure!"},
            {"role": "user", "content": "What about decorators?"},
        ],
        [
            {"role": "user", "content": "Hello, I want to discuss Python"},
            {"role": "assistant", "content": "Sure!"},
            {"role": "user", "content": "What about decorators?"},
            {"role": "assistant", "content": "Decorators are great!"},
            {"role": "user", "content": "Show me an example"},
        ],
    ]

    responses = [
        ["Sure", "!"],
        ["Decorators", " are", " great!"],
        ["Here", " is", " one"],
    ]
    set_multi_stream_responses(responses)

    cache_reads = []
    for msgs in turns:
        sse = await _stream_anthropic(integration_ctx.client, msgs)
        stats = _get_cache_stats(sse)
        cache_reads.append(stats["cache_read_input_tokens"])

    # First turn: no cache read
    assert cache_reads[0] == 0
    # Subsequent turns should have increasing cache reads (the shared prefix grows)
    assert cache_reads[1] > 0
    assert cache_reads[2] > cache_reads[1]


async def test_ollama_chat_no_cache_info_leak(integration_ctx):
    """Ollama /api/chat streaming NDJSON should not contain cache_info."""
    # First load the model
    await integration_ctx.manager.ensure_loaded("qwen3")

    resp = await integration_ctx.client.post(
        "/api/chat",
        json={
            "model": "qwen3",
            "stream": True,
            "messages": [{"role": "user", "content": "Hi"}],
        },
    )
    assert resp.status_code == 200

    for line in resp.text.strip().split("\n"):
        if line.strip():
            data = json.loads(line)
            assert "cache_info" not in data
            assert "cache_read_tokens" not in data
            assert "cache_creation_tokens" not in data


async def test_openai_chat_no_cache_info_leak(integration_ctx):
    """OpenAI /v1/chat/completions SSE should not contain cache metadata."""
    # First load the model
    await integration_ctx.manager.ensure_loaded("qwen3")

    resp = await integration_ctx.client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3",
            "stream": True,
            "messages": [{"role": "user", "content": "Hi"}],
        },
    )
    assert resp.status_code == 200

    for line in resp.text.strip().split("\n"):
        line = line.strip()
        if line.startswith("data: ") and line != "data: [DONE]":
            data = json.loads(line[6:])
            assert "cache_info" not in data
            assert "cache_read_tokens" not in str(data)
            assert "cache_creation" not in str(data)
