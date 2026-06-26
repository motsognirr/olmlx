"""Live acceptance test: the real OpenAI SDK against olmlx /v1/responses (#368).

Drives `AsyncOpenAI` over an in-process ASGI transport (no real server) so the
SDK's own response-schema validation exercises our /v1/responses shapes — the
acceptance criterion for #368 (SDK works for text, streaming text, tool use).

Outside tests/integration/ on purpose (that package's autouse mock_mlx_primitives
would replace the real model). real_model; skipped in CI and when the model is
not downloaded.
"""

import json

import pytest

from olmlx.config import settings

MODEL = "mlx-community/Qwen3-4B-4bit"


def _model_present() -> bool:
    from olmlx.models.store import _safe_dir_name

    return (settings.models_dir / _safe_dir_name(MODEL) / "config.json").exists()


pytestmark = [
    pytest.mark.real_model,
    pytest.mark.skipif(
        not _model_present(),
        reason=f"{MODEL} not downloaded in {settings.models_dir}",
    ),
]


@pytest.fixture
async def sdk_client(tmp_path, monkeypatch):
    """Real app + ModelManager, exposed via the real OpenAI AsyncOpenAI SDK
    bound to an in-process ASGI transport."""
    models_config = tmp_path / "models.json"
    models_config.write_text(json.dumps({}))
    aliases_path = tmp_path / "aliases.json"
    aliases_path.write_text("{}")
    monkeypatch.setattr("olmlx.config.settings.models_config", models_config)
    monkeypatch.setattr("olmlx.engine.registry.settings.models_config", models_config)

    from olmlx.engine.model_manager import ModelManager
    from olmlx.engine.registry import ModelRegistry
    from olmlx.models.store import ModelStore

    registry = ModelRegistry()
    registry._aliases_path = aliases_path
    registry.load()
    store = ModelStore(registry)
    manager = ModelManager(registry, store)
    manager.start_expiry_checker()

    from olmlx.app import create_app

    app = create_app()
    app.state.registry = registry
    app.state.model_manager = manager
    app.state.model_store = store

    import httpx
    from openai import AsyncOpenAI

    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    http_client = httpx.AsyncClient(transport=transport, base_url="http://test")
    client = AsyncOpenAI(
        base_url="http://test/v1", api_key="not-needed", http_client=http_client
    )
    try:
        yield client
    finally:
        await http_client.aclose()
        await manager.stop()


async def test_text(sdk_client):
    # Qwen3 is a thinking model; reasoning.effort="none" disables thinking so the
    # 64-token budget produces visible prose rather than being spent entirely on an
    # unclosed <think> block (which #555 correctly routes to reasoning, not content).
    resp = await sdk_client.responses.create(
        model=MODEL,
        input="Say the single word: pong.",
        max_output_tokens=64,
        reasoning={"effort": "none"},
    )
    assert resp.status in ("completed", "incomplete")
    assert resp.output_text  # SDK convenience accessor concatenates output_text


async def test_streaming_text(sdk_client):
    chunks = []
    async with sdk_client.responses.stream(
        model=MODEL,
        input="Count to three.",
        max_output_tokens=64,
        reasoning={"effort": "none"},
    ) as stream:
        async for event in stream:
            if event.type == "response.output_text.delta":
                chunks.append(event.delta)
        final = await stream.get_final_response()
    assert "".join(chunks)
    assert final.status in ("completed", "incomplete")


async def test_tool_use(sdk_client):
    tools = [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get the weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
    ]
    resp = await sdk_client.responses.create(
        model=MODEL,
        input="Use the get_weather tool to check the weather in Paris.",
        tools=tools,
        max_output_tokens=128,
    )
    # The acceptance criterion is that the SDK round-trips our response without
    # validation errors. A 4B model usually calls the tool here, but we don't
    # hard-fail on model competence: assert the response validated and, IF a
    # function_call was produced, its shape is correct.
    assert resp.status in ("completed", "incomplete")
    calls = [it for it in resp.output if it.type == "function_call"]
    for c in calls:
        assert c.name == "get_weather"
        json.loads(c.arguments)  # arguments must parse as JSON
