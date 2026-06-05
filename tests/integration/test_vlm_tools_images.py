"""Live VLM tools+images test (#428). Skipped in CI via -m "not real_model"."""

import base64
import io
import json

import pytest

pytestmark = pytest.mark.real_model

VLM_MODEL = "mlx-community/gemma-4-26b-a4b-it-4bit"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "record_number",
            "description": "Record the number shown in the image.",
            "parameters": {
                "type": "object",
                "properties": {"value": {"type": "integer"}},
                "required": ["value"],
            },
        },
    }
]


def _png_data_uri_with_number(text: str = "42") -> str:
    """Render a small white PNG with large black text and return a data URI."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (160, 96), "white")
    draw = ImageDraw.Draw(img)
    draw.text((40, 30), text, fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


@pytest.fixture
async def real_ctx(tmp_path, monkeypatch):
    models_config = tmp_path / "models.json"
    models_config.write_text(json.dumps({}))
    aliases_path = tmp_path / "aliases.json"
    aliases_path.write_text("{}")
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    monkeypatch.setattr("olmlx.config.settings.models_dir", models_dir)
    monkeypatch.setattr("olmlx.config.settings.models_config", models_config)
    monkeypatch.setattr("olmlx.engine.registry.settings.models_config", models_config)
    monkeypatch.setattr("olmlx.models.store.settings.models_dir", models_dir)

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

    from httpx import ASGITransport, AsyncClient

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, manager

    await manager.stop()


async def test_openai_vlm_tools_with_image_produces_tool_call(real_ctx):
    client, _manager = real_ctx
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": VLM_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Read the number and record it."},
                        {
                            "type": "image_url",
                            "image_url": {"url": _png_data_uri_with_number("42")},
                        },
                    ],
                }
            ],
            "tools": TOOLS,
            "max_tokens": 128,
        },
        timeout=600,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    tool_calls = body["choices"][0]["message"].get("tool_calls")
    assert tool_calls, f"expected a tool call, got: {body}"
    assert tool_calls[0]["function"]["name"] == "record_number"


async def test_ollama_vlm_tools_with_image_produces_tool_call(real_ctx):
    client, _manager = real_ctx
    resp = await client.post(
        "/api/chat",
        json={
            "model": VLM_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": "Read the number and record it.",
                    "images": [_png_data_uri_with_number("42")],
                }
            ],
            "tools": TOOLS,
            "stream": False,
        },
        timeout=600,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["message"].get("tool_calls"), f"expected a tool call, got: {body}"
