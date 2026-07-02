"""Real-model smoke tests — load an actual MLX model and run inference.

These tests require a GPU and download ~300MB once per machine (into the
HF cache, unless the model is already in the operator's olmlx store).
Skipped in CI via: pytest -m "not real_model"

The model is provisioned ONCE per pytest session (`real_model_files`) and
seeded into each test's isolated models_dir with hardlinks. Tests
themselves never touch the network: the nightly smoke runs used to flake
with `assert 500 == 200` whenever the per-test `snapshot_download` hit a
connection reset mid-test.
"""

import json
import os
import shutil
import time
from pathlib import Path

import pytest

from tests.integration.conftest import parse_sse_events

REAL_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

pytestmark = pytest.mark.real_model


@pytest.fixture(scope="session")
def real_model_files() -> Path:
    """Provision REAL_MODEL once per session, outside any test body.

    Prefers a copy already in the operator's store (the smoke runner
    pre-pulls it via scripts/real_model_smoke.sh); otherwise downloads
    into the shared HF cache with retries, so a transient network error
    surfaces here — clearly labeled as provisioning — instead of as a
    500 from an endpoint under test.
    """
    from olmlx.config import settings
    from olmlx.models.store import _safe_dir_name

    operator_copy = settings.models_dir / _safe_dir_name(REAL_MODEL)
    if (operator_copy / "config.json").exists() and not (
        operator_copy / ".downloading"
    ).exists():
        return operator_copy

    from huggingface_hub import snapshot_download

    last_err: Exception | None = None
    for attempt in range(3):
        if attempt:
            time.sleep(5 * attempt)
        try:
            return Path(snapshot_download(repo_id=REAL_MODEL))
        except Exception as exc:  # noqa: BLE001 — retry any transport error
            last_err = exc
    raise RuntimeError(
        f"could not provision {REAL_MODEL} after 3 attempts"
    ) from last_err


def _seed_model(models_dir: Path, src: Path) -> None:
    """Copy the provisioned model into a test's models_dir.

    Weights are hardlinked (cheap, read-only usage); everything else is a
    real copy so in-place writes (e.g. manifest.json) can't reach through
    a shared inode into the HF cache or the operator's store.
    """
    from olmlx.models.store import _safe_dir_name

    def link_weights(s: str, d: str) -> None:
        if s.endswith(".safetensors"):
            try:
                os.link(s, d)
                return
            except OSError:
                pass  # cross-filesystem tmp dir — fall through to a real copy
        shutil.copy2(s, d)

    shutil.copytree(
        src,
        models_dir / _safe_dir_name(REAL_MODEL),
        copy_function=link_weights,
        ignore=shutil.ignore_patterns(".cache", ".downloading*"),
    )


@pytest.fixture
async def real_ctx(tmp_path, monkeypatch, real_model_files):
    """Integration context with NO MLX mocks — uses a real model."""
    models_config = tmp_path / "models.json"
    models_config.write_text(json.dumps({}))
    aliases_path = tmp_path / "aliases.json"
    aliases_path.write_text("{}")

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    _seed_model(models_dir, real_model_files)

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


async def test_real_ollama_chat(real_ctx):
    client, manager = real_ctx
    resp = await client.post(
        "/api/chat",
        json={
            "model": REAL_MODEL,
            "stream": False,
            "messages": [{"role": "user", "content": "Say hello in one word."}],
        },
    )
    assert resp.status_code == 200
    text = resp.json()["message"]["content"]
    assert len(text) > 0


async def test_real_openai_chat(real_ctx):
    client, manager = real_ctx
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": REAL_MODEL,
            "stream": False,
            "messages": [{"role": "user", "content": "Say hello in one word."}],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "choices" in data
    assert len(data["choices"]) > 0
    assert len(data["choices"][0]["message"]["content"]) > 0


async def test_real_anthropic_messages(real_ctx):
    client, manager = real_ctx
    resp = await client.post(
        "/v1/messages",
        json={
            "model": REAL_MODEL,
            "max_tokens": 50,
            "stream": False,
            "messages": [{"role": "user", "content": "Say hello in one word."}],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["type"] == "message"
    assert data["role"] == "assistant"
    text = "".join(
        b.get("text", "") for b in data["content"] if b.get("type") == "text"
    )
    assert len(text) > 0


async def test_real_streaming(real_ctx):
    client, manager = real_ctx
    resp = await client.post(
        "/v1/messages",
        json={
            "model": REAL_MODEL,
            "max_tokens": 50,
            "stream": True,
            "messages": [{"role": "user", "content": "Count to 3."}],
        },
    )
    assert resp.status_code == 200

    events = parse_sse_events(resp.text)
    event_types = [e["event"] for e in events]

    assert "message_start" in event_types
    assert "content_block_start" in event_types
    assert "message_stop" in event_types


async def test_real_prompt_cache(real_ctx):
    client, manager = real_ctx
    messages = [
        {"role": "user", "content": "Tell me about Python programming language."}
    ]

    resp1 = await client.post(
        "/v1/messages",
        json={
            "model": REAL_MODEL,
            "max_tokens": 30,
            "stream": True,
            "messages": messages,
        },
    )
    assert resp1.status_code == 200
    events1 = parse_sse_events(resp1.text)
    msg_start1 = [e for e in events1 if e["event"] == "message_start"][0]
    _ = msg_start1["data"]["message"]["usage"]  # verify structure

    resp2 = await client.post(
        "/v1/messages",
        json={
            "model": REAL_MODEL,
            "max_tokens": 30,
            "stream": True,
            "messages": messages,
        },
    )
    assert resp2.status_code == 200
    events2 = parse_sse_events(resp2.text)
    msg_start2 = [e for e in events2 if e["event"] == "message_start"][0]
    usage2 = msg_start2["data"]["message"]["usage"]

    assert usage2["cache_read_input_tokens"] > 0


async def test_real_count_tokens(real_ctx):
    client, manager = real_ctx

    # First ensure the model is loaded
    await client.post(
        "/v1/messages",
        json={
            "model": REAL_MODEL,
            "max_tokens": 10,
            "stream": False,
            "messages": [{"role": "user", "content": "Hi"}],
        },
    )

    resp = await client.post(
        "/v1/messages/count_tokens",
        json={
            "model": REAL_MODEL,
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hello, how are you?"}],
        },
    )
    assert resp.status_code == 200
    assert resp.json()["input_tokens"] > 0
