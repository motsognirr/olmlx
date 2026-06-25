"""Tests for LoRA-adapter download/pull in olmlx.models.store (issue #362)."""

import json

import pytest

from olmlx.engine.registry import ModelRegistry
from olmlx.models.store import ModelStore, _safe_dir_name


@pytest.fixture
def adapter_store(tmp_path, monkeypatch):
    """A ModelStore + registry with one registered adapter, temp models_dir."""
    config = {
        "qwen3-8b": "Qwen/Qwen3-8B-MLX",
        "adapters": {
            "qwen3-8b:my-coder-lora": {
                "base": "qwen3-8b",
                "hf_path": "acme/my-coder-lora",
            }
        },
    }
    config_path = tmp_path / "models.json"
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr("olmlx.engine.registry.settings.models_config", config_path)
    monkeypatch.setattr("olmlx.models.store.settings.models_dir", tmp_path / "models")
    reg = ModelRegistry()
    reg._aliases_path = tmp_path / "aliases.json"
    reg.load()
    return ModelStore(reg), tmp_path


def _fake_snapshot(local_dir_marker_files):
    """Return a fake snapshot_download that writes adapter files into local_dir."""

    def _dl(*, repo_id, local_dir):
        from pathlib import Path

        p = Path(local_dir)
        p.mkdir(parents=True, exist_ok=True)
        for fname, content in local_dir_marker_files.items():
            (p / fname).write_text(content)

    return _dl


class TestAdapterLocalPath:
    def test_under_adapters_subdir(self, adapter_store):
        store, tmp_path = adapter_store
        p = store.adapter_local_path("acme/my-coder-lora")
        assert p == tmp_path / "models" / "adapters" / _safe_dir_name(
            "acme/my-coder-lora"
        )

    def test_distinct_from_model_path(self, adapter_store):
        store, _ = adapter_store
        # An adapter and a base model with the same repo id never collide.
        assert store.adapter_local_path("acme/x") != store.local_path("acme/x")


class TestEnsureAdapterDownloaded:
    def test_downloads_and_returns_path(self, adapter_store, monkeypatch):
        store, _ = adapter_store
        monkeypatch.setattr(
            "huggingface_hub.snapshot_download",
            _fake_snapshot({"adapter_config.json": "{}"}),
        )
        path = store.ensure_adapter_downloaded("acme/my-coder-lora")
        assert (path / "adapter_config.json").exists()
        assert store.is_adapter_downloaded("acme/my-coder-lora")

    def test_idempotent_skips_redownload(self, adapter_store, monkeypatch):
        store, _ = adapter_store
        calls = {"n": 0}

        def _dl(*, repo_id, local_dir):
            from pathlib import Path

            calls["n"] += 1
            Path(local_dir).mkdir(parents=True, exist_ok=True)
            (Path(local_dir) / "adapter_config.json").write_text("{}")

        monkeypatch.setattr("huggingface_hub.snapshot_download", _dl)
        store.ensure_adapter_downloaded("acme/my-coder-lora")
        store.ensure_adapter_downloaded("acme/my-coder-lora")
        assert calls["n"] == 1


class TestPullDispatch:
    async def test_pull_dispatches_to_adapter(self, adapter_store, monkeypatch):
        store, _ = adapter_store
        monkeypatch.setattr(
            "huggingface_hub.snapshot_download",
            _fake_snapshot({"adapter_config.json": "{}"}),
        )
        events = [e async for e in store.pull("qwen3-8b:my-coder-lora")]
        statuses = [e.get("status") for e in events]
        assert statuses[-1] == "success"
        assert any("adapter" in (s or "").lower() for s in statuses)
        # The adapter weights landed under the adapters/ subtree.
        assert store.is_adapter_downloaded("acme/my-coder-lora")

    async def test_pull_unknown_adapter_falls_through(self, adapter_store):
        store, _ = adapter_store
        # Not a registered adapter and not an HF path → model-not-found error.
        with pytest.raises(ValueError, match="not found"):
            [e async for e in store.pull("qwen3-8b:does-not-exist")]
