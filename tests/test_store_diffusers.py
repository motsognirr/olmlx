"""Diffusers-layout (image) models in the olmlx ModelStore (#723).

Image models are ordinary store models: downloaded by ``ensure_downloaded``
into ``OLMLX_MODELS_DIR`` and loaded from there — never via the Hugging Face
cache. Their repos have no top-level ``config.json``; the diffusers pipeline
marker is ``model_index.json``.
"""

import json
from unittest.mock import patch

import pytest

from olmlx.engine.registry import ModelRegistry
from olmlx.models.store import ModelStore, _extract_metadata

REPO = "Qwen/Qwen-Image-2.1"


@pytest.fixture
def store(tmp_path, monkeypatch):
    cfg = {"qwen-image:2.1": {"type": "image", "hf_path": REPO}}
    path = tmp_path / "models.json"
    path.write_text(json.dumps(cfg))
    monkeypatch.setattr("olmlx.engine.registry.settings.models_config", path)
    monkeypatch.setattr("olmlx.models.store.settings.models_dir", tmp_path / "models")
    reg = ModelRegistry()
    reg.load()
    return ModelStore(reg)


def _write_diffusers(d):
    d.mkdir(parents=True, exist_ok=True)
    (d / "model_index.json").write_text(
        json.dumps({"_class_name": "QwenImagePipeline"})
    )
    (d / "transformer").mkdir(exist_ok=True)
    (d / "transformer" / "config.json").write_text("{}")


def _fake_diffusers_download(store):
    def _download(**kwargs):
        assert kwargs["local_dir"] == str(store.local_path(REPO))
        _write_diffusers(store.local_path(REPO))

    return _download


def test_diffusers_dir_counts_as_downloaded(store):
    _write_diffusers(store.local_path(REPO))
    assert store.is_downloaded(REPO)


def test_ensure_downloaded_accepts_model_index(store):
    with patch(
        "huggingface_hub.snapshot_download", side_effect=_fake_diffusers_download(store)
    ) as dl:
        local = store.ensure_downloaded(REPO)
    dl.assert_called_once()
    # Downloaded into the store, not the HF cache.
    assert local == store.local_path(REPO)
    assert not (local / ".downloading").exists()


def test_metadata_family_image(store):
    d = store.local_path(REPO)
    _write_diffusers(d)
    assert _extract_metadata(d)["family"] == "image"


def test_list_local_includes_diffusers_model(store):
    _write_diffusers(store.local_path(REPO))
    names = {m.name: m for m in store.list_local()}
    assert names["qwen-image:2.1"].family == "image"


def test_show_and_delete_diffusers_model(store):
    _write_diffusers(store.local_path(REPO))
    assert store.show("qwen-image:2.1") is not None
    assert store.delete("qwen-image:2.1") is True
    assert not store.local_path(REPO).exists()


@pytest.mark.asyncio
async def test_pull_image_model_downloads_into_store(store):
    with patch(
        "huggingface_hub.snapshot_download", side_effect=_fake_diffusers_download(store)
    ):
        statuses = [e["status"] async for e in store.pull("qwen-image:2.1")]
    assert statuses[-1] == "success"
    manifest = json.loads((store.local_path(REPO) / "manifest.json").read_text())
    assert manifest["family"] == "image"
