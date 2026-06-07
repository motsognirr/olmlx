"""Live VLM prompt-cache + grammar + token-count tests (#429).

Loads a REAL Gemma 4 VLM and verifies the lifted gating:
  * grammar (response_format json_schema) applies with an image present,
  * a multi-turn vision chat reuses the image-prefix KV (cache_read > 0) with
    output identical to the uncached (slots=0) path,
  * non-streaming VLM reports non-zero token counts,
  * a fresh image request is not silently dropped (image reaches the model).

Lives OUTSIDE tests/integration/ to dodge that package's autouse MLX mock.
Run on a machine where the model is downloaded.
"""

import base64
import io
import json

import pytest

from olmlx.config import settings

VLM_MODEL = "mlx-community/gemma-4-26B-A4B-it-4bit"


def _model_present() -> bool:
    from olmlx.models.store import _safe_dir_name

    return (settings.models_dir / _safe_dir_name(VLM_MODEL) / "config.json").exists()


pytestmark = [
    pytest.mark.real_model,
    pytest.mark.skipif(
        not _model_present(),
        reason=f"{VLM_MODEL} not downloaded in {settings.models_dir}",
    ),
]


def _png_data_uri_with_number(text: str = "42") -> str:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (160, 96), "white")
    ImageDraw.Draw(img).text((50, 30), text, fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


async def _make_client(tmp_path, monkeypatch):
    """Build a real app + ModelManager over the real model store. Caller sets
    settings.vlm_prompt_cache_slots BEFORE calling this so the loaded model's
    VlmPromptCacheStore is created with the chosen capacity."""
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
    client = AsyncClient(transport=transport, base_url="http://test")
    return client, manager


@pytest.fixture
async def live_client(tmp_path, monkeypatch):
    """Default client with VLM caching enabled (slots=2)."""
    monkeypatch.setattr("olmlx.config.settings.vlm_prompt_cache_slots", 2)
    client, manager = await _make_client(tmp_path, monkeypatch)
    async with client:
        yield client
    await manager.stop()


def _image_message(text: str, number: str = "42"):
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {
                "type": "image_url",
                "image_url": {"url": _png_data_uri_with_number(number)},
            },
        ],
    }


async def test_grammar_json_schema_on_image_request(live_client):
    """Acceptance criterion 2: schema-valid JSON on an image request."""
    resp = await live_client.post(
        "/v1/chat/completions",
        json={
            "model": VLM_MODEL,
            "messages": [_image_message("Read the number in the image.")],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "number",
                    "schema": {
                        "type": "object",
                        "properties": {"value": {"type": "integer"}},
                        "required": ["value"],
                        "additionalProperties": False,
                    },
                },
            },
            "max_tokens": 64,
            "temperature": 0,
        },
        timeout=600,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    content = body["choices"][0]["message"]["content"]
    parsed = json.loads(content)  # must be valid JSON conforming to the schema
    assert isinstance(parsed["value"], int), parsed


async def test_non_streaming_vlm_reports_token_counts(live_client):
    """Non-streaming VLM must report non-zero prompt/eval counts (#429)."""
    resp = await live_client.post(
        "/v1/chat/completions",
        json={
            "model": VLM_MODEL,
            "messages": [_image_message("Describe the image briefly.")],
            "max_tokens": 32,
            "temperature": 0,
            "stream": False,
        },
        timeout=600,
    )
    assert resp.status_code == 200, resp.text
    usage = resp.json()["usage"]
    assert usage["prompt_tokens"] > 200, usage  # image placeholder expands large
    assert usage["completion_tokens"] > 0, usage


def _growing_vision_turns(n_assistant_turns):
    """A vision conversation that grows by appending assistant+user turns, so
    later turns share the (image-bearing) leading prefix with earlier ones."""
    img = _png_data_uri_with_number("42")
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What number is in the image?"},
                {"type": "image_url", "image_url": {"url": img}},
            ],
        }
    ]
    followups = ["Say it again.", "And once more."]
    for i in range(n_assistant_turns):
        msgs.append({"role": "assistant", "content": "42"})
        msgs.append({"role": "user", "content": followups[i]})
    return msgs


async def _ask(client, messages, cache_id):
    r = await client.post(
        "/v1/chat/completions",
        json={
            "model": VLM_MODEL,
            "messages": messages,
            "max_tokens": 16,
            "temperature": 0,
            "stream": False,
        },
        # cache_id is supplied via the x-cache-id header (olmlx/routers/openai.py).
        headers={"x-cache-id": cache_id},
        timeout=600,
    )
    assert r.status_code == 200, r.text
    return r.json()


async def test_multi_turn_vision_reuses_image_prefix_kv(live_client):
    """Acceptance criterion 1 (reuse half): a 3-turn vision chat under one
    cache_id reuses the image-prefix KV — observable via /api/ps."""
    out = []
    for turn in range(3):
        body = await _ask(live_client, _growing_vision_turns(turn), "vision-reuse")
        out.append(body["choices"][0]["message"]["content"])

    ps = await live_client.get("/api/ps")
    assert ps.status_code == 200, ps.text
    models = ps.json()["models"]
    me = next(m for m in models if m["model"] == VLM_MODEL or m["name"] == VLM_MODEL)
    metrics = me["cache_metrics"]
    # Turns 2 and 3 extend the cached prefix → real reuse.
    assert metrics.get("vlm_cache_hits", 0) >= 2, metrics
    assert metrics.get("vlm_cache_tokens_reused", 0) > 0, metrics


async def test_cached_and_uncached_outputs_match(tmp_path, monkeypatch):
    """Acceptance criterion 1 (correctness half): greedy output with caching ON
    equals output with caching OFF (slots=0). Two independently-built clients so
    each model's VlmPromptCacheStore is created with the intended capacity."""
    # Caching ON.
    monkeypatch.setattr("olmlx.config.settings.vlm_prompt_cache_slots", 2)
    client_on, manager_on = await _make_client(tmp_path, monkeypatch)
    async with client_on:
        cached = [
            (await _ask(client_on, _growing_vision_turns(t), "match-on"))["choices"][0][
                "message"
            ]["content"]
            for t in range(3)
        ]
    await manager_on.stop()

    # Caching OFF (slots=0): store disabled, fresh prefill every turn.
    monkeypatch.setattr("olmlx.config.settings.vlm_prompt_cache_slots", 0)
    off_dir = tmp_path / "off"
    off_dir.mkdir()
    client_off, manager_off = await _make_client(off_dir, monkeypatch)
    async with client_off:
        uncached = [
            (await _ask(client_off, _growing_vision_turns(t), "match-off"))["choices"][
                0
            ]["message"]["content"]
            for t in range(3)
        ]
    await manager_off.stop()

    assert cached == uncached, (cached, uncached)
