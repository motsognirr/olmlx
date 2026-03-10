"""Integration tests for model pull → download → register flow."""

import asyncio
import json


def _setup_fake_download(tmp_path, hf_path="Qwen/Qwen3-8B-MLX"):
    """Create fake model files as if snapshot_download had run."""
    from olmlx.models.store import _safe_dir_name

    model_dir = tmp_path / "models" / _safe_dir_name(hf_path)
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(
        json.dumps(
            {"model_type": "qwen3", "hidden_size": 4096, "num_hidden_layers": 32}
        )
    )
    return model_dir


async def test_pull_downloads_and_registers(integration_ctx, tmp_path):
    """Pull triggers snapshot_download, creates manifest, and registers in registry."""
    import huggingface_hub

    hf_path = "Qwen/Qwen3-8B-MLX"

    # Make snapshot_download create fake files
    def fake_download(repo_id, local_dir):
        local = integration_ctx.store.local_path(hf_path)
        local.mkdir(parents=True, exist_ok=True)
        (local / "config.json").write_text(
            json.dumps(
                {"model_type": "qwen3", "hidden_size": 4096, "num_hidden_layers": 32}
            )
        )
        return str(local)

    huggingface_hub.snapshot_download.side_effect = fake_download

    resp = await integration_ctx.client.post(
        "/api/pull", json={"model": "qwen3", "stream": False}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("status") == "success"

    huggingface_hub.snapshot_download.assert_called_once()
    assert integration_ctx.registry.resolve("qwen3") == hf_path

    # Manifest should exist
    local = integration_ctx.store.local_path(hf_path)
    assert (local / "manifest.json").exists()


async def test_pull_skips_if_downloaded(integration_ctx, tmp_path):
    """Pull with model already present skips download."""
    import huggingface_hub

    _setup_fake_download(tmp_path)

    resp = await integration_ctx.client.post(
        "/api/pull", json={"model": "qwen3", "stream": False}
    )
    assert resp.status_code == 200

    # snapshot_download should NOT have been called
    huggingface_hub.snapshot_download.assert_not_called()


async def test_pull_concurrent_serialization(integration_ctx):
    """Two concurrent pulls for the same model — download happens once."""
    import huggingface_hub

    call_count = 0

    def fake_download(repo_id, local_dir):
        nonlocal call_count
        call_count += 1
        local = integration_ctx.store.local_path(repo_id)
        local.mkdir(parents=True, exist_ok=True)
        (local / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
        return str(local)

    huggingface_hub.snapshot_download.side_effect = fake_download

    # Use a new model to avoid the fast path
    integration_ctx.registry.add_mapping("newmodel", "org/new-model")

    results = await asyncio.gather(
        integration_ctx.client.post(
            "/api/pull", json={"model": "newmodel", "stream": False}
        ),
        integration_ctx.client.post(
            "/api/pull", json={"model": "newmodel", "stream": False}
        ),
    )

    assert all(r.status_code == 200 for r in results)
    # snapshot_download should be called at most once due to lock serialization
    assert call_count <= 1


async def test_pull_then_inference(integration_ctx):
    """Pull a model, then run inference through it."""
    import huggingface_hub

    def fake_download(repo_id, local_dir):
        local = integration_ctx.store.local_path(repo_id)
        local.mkdir(parents=True, exist_ok=True)
        (local / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
        return str(local)

    huggingface_hub.snapshot_download.side_effect = fake_download

    pull_resp = await integration_ctx.client.post(
        "/api/pull", json={"model": "qwen3", "stream": False}
    )
    assert pull_resp.status_code == 200

    integration_ctx.set_stream_responses(["Pulled", " and", " ready"])

    msg_resp = await integration_ctx.client.post(
        "/v1/messages",
        json={
            "model": "qwen3",
            "max_tokens": 100,
            "stream": False,
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )
    assert msg_resp.status_code == 200
    content = msg_resp.json()["content"]
    text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
    assert "Pulled" in text


async def test_pull_hf_path_auto_registers(integration_ctx):
    """Pulling a direct HF path auto-registers it."""
    import huggingface_hub

    hf_path = "org/some-model"

    def fake_download(repo_id, local_dir):
        local = integration_ctx.store.local_path(repo_id)
        local.mkdir(parents=True, exist_ok=True)
        (local / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
        return str(local)

    huggingface_hub.snapshot_download.side_effect = fake_download

    resp = await integration_ctx.client.post(
        "/api/pull", json={"model": hf_path, "stream": False}
    )
    assert resp.status_code == 200

    # Direct HF paths resolve to themselves
    assert integration_ctx.registry.resolve(hf_path) == hf_path


async def test_pull_failure_keeps_partial_dir(integration_ctx):
    """Failed download keeps partial dir (for resume) with .downloading marker."""
    import huggingface_hub

    hf_path = "Qwen/Qwen3-8B-MLX"

    def failing_download(repo_id, local_dir):
        # Create partial files but crash
        from pathlib import Path

        Path(local_dir).mkdir(parents=True, exist_ok=True)
        (Path(local_dir) / "partial.safetensors").write_text("partial")
        raise OSError("Network error")

    huggingface_hub.snapshot_download.side_effect = failing_download

    # Use a fresh model name so it isn't already downloaded
    integration_ctx.registry.add_mapping("failmodel", hf_path)

    resp = await integration_ctx.client.post(
        "/api/pull", json={"model": "failmodel", "stream": False}
    )
    # Pull catches the exception and returns error status
    assert resp.status_code == 500

    # .downloading marker should remain (or partial dir should exist)
    local = integration_ctx.store.local_path(hf_path)
    assert local.exists()
    assert (local / ".downloading").exists()
