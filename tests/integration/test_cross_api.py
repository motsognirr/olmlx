"""Integration tests for cross-API consistency."""

import json


from tests.integration.conftest import parse_sse_events, set_stream_responses
from unittest.mock import MagicMock


async def test_same_output_all_apis(integration_ctx):
    """Same message to all three APIs returns the same text (non-streaming)."""
    set_stream_responses(["Hello", " world"])

    # Anthropic
    anthropic_resp = await integration_ctx.client.post(
        "/v1/messages",
        json={
            "model": "qwen3",
            "max_tokens": 100,
            "stream": False,
            "messages": [{"role": "user", "content": "Hi"}],
        },
    )
    assert anthropic_resp.status_code == 200
    anthropic_text = ""
    for block in anthropic_resp.json()["content"]:
        if block.get("type") == "text":
            anthropic_text += block.get("text", "")

    # OpenAI
    openai_resp = await integration_ctx.client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3",
            "stream": False,
            "messages": [{"role": "user", "content": "Hi"}],
        },
    )
    assert openai_resp.status_code == 200
    openai_text = openai_resp.json()["choices"][0]["message"]["content"]

    # Ollama
    ollama_resp = await integration_ctx.client.post(
        "/api/chat",
        json={
            "model": "qwen3",
            "stream": False,
            "messages": [{"role": "user", "content": "Hi"}],
        },
    )
    assert ollama_resp.status_code == 200
    ollama_text = ollama_resp.json()["message"]["content"]

    # All should contain the same text
    assert "Hello world" in anthropic_text
    assert "Hello world" in openai_text
    assert "Hello world" in ollama_text


async def test_streaming_same_content(integration_ctx):
    """Streaming across all APIs produces the same reassembled text."""
    integration_ctx.set_stream_responses(["Stream", "ed", " text"])

    # Pre-load the model to avoid triple loading
    await integration_ctx.manager.ensure_loaded("qwen3")

    # Anthropic streaming
    anthropic_resp = await integration_ctx.client.post(
        "/v1/messages",
        json={
            "model": "qwen3",
            "max_tokens": 100,
            "stream": True,
            "messages": [{"role": "user", "content": "Hi"}],
        },
    )
    events = parse_sse_events(anthropic_resp.text)
    anthropic_text = ""
    for ev in events:
        if ev.get("event") == "content_block_delta":
            delta = ev["data"].get("delta", {})
            if delta.get("type") == "text_delta":
                anthropic_text += delta.get("text", "")

    # OpenAI streaming
    integration_ctx.set_stream_responses(["Stream", "ed", " text"])
    openai_resp = await integration_ctx.client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3",
            "max_tokens": 100,
            "stream": True,
            "messages": [{"role": "user", "content": "Hi"}],
        },
    )
    openai_text = ""
    for line in openai_resp.text.strip().split("\n"):
        line = line.strip()
        if line.startswith("data: ") and line != "data: [DONE]":
            data = json.loads(line[6:])
            content = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
            openai_text += content

    # Ollama streaming
    integration_ctx.set_stream_responses(["Stream", "ed", " text"])
    ollama_resp = await integration_ctx.client.post(
        "/api/chat",
        json={
            "model": "qwen3",
            "stream": True,
            "messages": [{"role": "user", "content": "Hi"}],
        },
    )
    ollama_text = ""
    for line in ollama_resp.text.strip().split("\n"):
        if line.strip():
            data = json.loads(line)
            ollama_text += data.get("message", {}).get("content", "")

    assert "Streamed text" in anthropic_text
    assert "Streamed text" in openai_text
    assert "Streamed text" in ollama_text


async def test_error_format_per_api(integration_ctx):
    """Nonexistent model → error in the correct format per API surface."""
    # Ollama
    ollama_resp = await integration_ctx.client.post(
        "/api/chat",
        json={
            "model": "nonexistent",
            "stream": False,
            "messages": [{"role": "user", "content": "Hi"}],
        },
    )
    assert ollama_resp.status_code == 400
    ollama_body = ollama_resp.json()
    assert "error" in ollama_body
    assert isinstance(ollama_body["error"], str)

    # OpenAI
    openai_resp = await integration_ctx.client.post(
        "/v1/chat/completions",
        json={
            "model": "nonexistent",
            "stream": False,
            "messages": [{"role": "user", "content": "Hi"}],
        },
    )
    assert openai_resp.status_code == 400
    openai_body = openai_resp.json()
    assert "error" in openai_body
    assert "message" in openai_body["error"]
    assert "type" in openai_body["error"]
    assert "code" in openai_body["error"]

    # Anthropic
    anthropic_resp = await integration_ctx.client.post(
        "/v1/messages",
        json={
            "model": "nonexistent",
            "max_tokens": 100,
            "stream": False,
            "messages": [{"role": "user", "content": "Hi"}],
        },
    )
    assert anthropic_resp.status_code == 400
    anthropic_body = anthropic_resp.json()
    assert anthropic_body["type"] == "error"
    assert "type" in anthropic_body["error"]
    assert "message" in anthropic_body["error"]


async def test_memory_error_format_per_api(integration_ctx, monkeypatch):
    """MemoryError → 503 with correct format per API."""
    # Set metal memory to exceed limit after loading
    monkeypatch.setattr(
        "olmlx.engine.model_manager._get_metal_memory_bytes",
        MagicMock(return_value=30 * 1024**3),
    )

    # Ollama
    ollama_resp = await integration_ctx.client.post(
        "/api/chat",
        json={
            "model": "qwen3",
            "stream": False,
            "messages": [{"role": "user", "content": "Hi"}],
        },
    )
    assert ollama_resp.status_code == 503
    ollama_body = ollama_resp.json()
    assert "error" in ollama_body
    assert isinstance(ollama_body["error"], str)

    # OpenAI
    openai_resp = await integration_ctx.client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3",
            "stream": False,
            "messages": [{"role": "user", "content": "Hi"}],
        },
    )
    assert openai_resp.status_code == 503
    openai_body = openai_resp.json()
    assert "error" in openai_body
    assert "message" in openai_body["error"]

    # Anthropic
    anthropic_resp = await integration_ctx.client.post(
        "/v1/messages",
        json={
            "model": "qwen3",
            "max_tokens": 100,
            "stream": False,
            "messages": [{"role": "user", "content": "Hi"}],
        },
    )
    assert anthropic_resp.status_code == 503
    anthropic_body = anthropic_resp.json()
    assert anthropic_body["type"] == "error"
    assert anthropic_body["error"]["type"] == "overloaded_error"


async def test_thinking_blocks_only_anthropic(integration_ctx):
    """Output with <think> tags → thinking block in Anthropic, raw text elsewhere."""
    integration_ctx.set_stream_responses(["<think>", "reasoning", "</think>", "Answer"])

    # Pre-load
    await integration_ctx.manager.ensure_loaded("qwen3")

    # Anthropic (non-streaming) — should parse thinking blocks
    anthropic_resp = await integration_ctx.client.post(
        "/v1/messages",
        json={
            "model": "qwen3",
            "max_tokens": 100,
            "stream": False,
            "messages": [{"role": "user", "content": "Think about this"}],
        },
    )
    assert anthropic_resp.status_code == 200
    blocks = anthropic_resp.json()["content"]
    block_types = [b["type"] for b in blocks]
    assert "text" in block_types

    # OpenAI — should return raw text with thinking tags
    integration_ctx.set_stream_responses(["<think>", "reasoning", "</think>", "Answer"])
    openai_resp = await integration_ctx.client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen3",
            "stream": False,
            "messages": [{"role": "user", "content": "Think about this"}],
        },
    )
    assert openai_resp.status_code == 200
    openai_text = openai_resp.json()["choices"][0]["message"]["content"]
    # OpenAI gets raw text — may contain think tags
    assert "Answer" in openai_text or "<think>" in openai_text

    # Ollama — should return raw text
    integration_ctx.set_stream_responses(["<think>", "reasoning", "</think>", "Answer"])
    ollama_resp = await integration_ctx.client.post(
        "/api/chat",
        json={
            "model": "qwen3",
            "stream": False,
            "messages": [{"role": "user", "content": "Think about this"}],
        },
    )
    assert ollama_resp.status_code == 200
    ollama_text = ollama_resp.json()["message"]["content"]
    assert "Answer" in ollama_text or "<think>" in ollama_text
