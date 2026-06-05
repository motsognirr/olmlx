"""Live VLM tools+images test (#428).

Loads a REAL Gemma 4 VLM and verifies that a request carrying both ``tools``
and an image produces a tool call grounded in the image — i.e. the image is
not dropped on the native-tools path.

This lives OUTSIDE ``tests/integration/`` on purpose: that package has an
``autouse`` ``mock_mlx_primitives`` fixture that patches ``mlx_lm.load`` /
``mlx_lm.generate`` / ``snapshot_download``, so any test placed there runs
against mocks (never a real model).  Here only the top-level ``tests/conftest``
applies, which does not mock MLX.

Skipped in CI via ``-m "not real_model"``.  Additionally skipped when the model
is not already present in the local olmlx store, so it never triggers a
multi-GB download — run it on a machine where the model is downloaded.
"""

import base64
import io
import json

import pytest

from olmlx.config import settings

# Capital "B" matches the on-disk store dir (mlx-community_gemma-4-26B-A4B-it-4bit).
VLM_MODEL = "mlx-community/gemma-4-26B-A4B-it-4bit"


def _model_present() -> bool:
    """True when the model is already downloaded in the local olmlx store."""
    from olmlx.models.store import _safe_dir_name

    return (settings.models_dir / _safe_dir_name(VLM_MODEL) / "config.json").exists()


pytestmark = [
    pytest.mark.real_model,
    pytest.mark.skipif(
        not _model_present(),
        reason=f"{VLM_MODEL} not downloaded in {settings.models_dir}",
    ),
]

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
    draw.text((50, 30), text, fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


@pytest.fixture
async def live_client(tmp_path, monkeypatch):
    """Real app + ModelManager backed by the real model store.

    Only the writable config (models.json / aliases.json) is redirected to a
    tmp dir so the auto-register-on-load does not mutate the developer's real
    ``~/.olmlx/models.json``.  ``models_dir`` is left at the real location so
    the already-downloaded model loads without a download.
    """
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

    from httpx import ASGITransport, AsyncClient

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    await manager.stop()


async def test_openai_vlm_tools_with_image_produces_tool_call(live_client):
    resp = await live_client.post(
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
    # Image must reach the model: its placeholder expands to many tokens, so a
    # text-only prompt (~tens of tokens) would never reach this floor.
    assert body["usage"]["prompt_tokens"] > 200, body["usage"]
    tool_calls = body["choices"][0]["message"].get("tool_calls")
    assert tool_calls, f"expected a tool call, got: {body}"
    assert tool_calls[0]["function"]["name"] == "record_number"


async def test_ollama_vlm_tools_with_image_produces_tool_call(live_client):
    resp = await live_client.post(
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
    assert body["prompt_eval_count"] > 200, body
    assert body["message"].get("tool_calls"), f"expected a tool call, got: {body}"
