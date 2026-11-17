"""Edge0 packs in the model store and manager: convert on download, force Flash-MoE."""

import json
import shutil

import huggingface_hub
import pytest

from olmlx.engine.edge0 import convert
from tests.edge0_factories import make_full_pack

HF_PATH = "Edge0/Edge0-tiny-q3k"


@pytest.fixture(scope="module")
def pack_template(tmp_path_factory):
    return make_full_pack(tmp_path_factory.mktemp("template"), with_vision=False).root


def _place(mock_store, pack_template):
    local = mock_store.local_path(HF_PATH)
    shutil.copytree(pack_template, local, symlinks=True)
    return local


class TestEnsureDownloaded:
    def test_downloaded_pack_is_converted(self, mock_store, pack_template, monkeypatch):
        calls = []

        def fake_download(repo_id, local_dir):
            calls.append(repo_id)
            shutil.copytree(pack_template, local_dir, dirs_exist_ok=True)

        monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_download)
        assert not mock_store.is_downloaded(HF_PATH)
        local = mock_store.ensure_downloaded(HF_PATH)
        assert calls == [HF_PATH]
        assert local == mock_store.local_path(HF_PATH)
        assert convert.is_converted(local)
        assert mock_store.is_downloaded(HF_PATH)
        assert mock_store.model_dir(HF_PATH) == local
        assert not (local / ".downloading").exists()

    def test_converted_pack_is_left_alone(self, mock_store, pack_template, monkeypatch):
        local = _place(mock_store, pack_template)
        convert.convert_pack(local, projection_bits=4)
        stamp = (local / "config.json").stat().st_mtime_ns
        monkeypatch.setattr(
            "olmlx.engine.edge0.convert.convert_pack",
            lambda *a, **k: pytest.fail("converted again"),
        )
        # The suite's guard fails the test if snapshot_download is reached.
        assert mock_store.ensure_downloaded(HF_PATH) == local
        assert (local / "config.json").stat().st_mtime_ns == stamp

    def test_stale_conversion_is_redone_without_downloading(
        self, mock_store, pack_template
    ):
        local = _place(mock_store, pack_template)
        convert.convert_pack(local)
        config = json.loads((local / "config.json").read_text())
        config["edge0"]["converter_version"] = convert.CONVERTER_VERSION - 1
        (local / "config.json").write_text(json.dumps(config))
        assert mock_store.ensure_downloaded(HF_PATH) == local
        assert convert.is_converted(local)

    def test_failed_conversion_leaves_the_model_not_downloaded(
        self, mock_store, pack_template, monkeypatch
    ):
        def fake_download(repo_id, local_dir):
            shutil.copytree(pack_template, local_dir, dirs_exist_ok=True)
            (mock_store.local_path(HF_PATH) / "model-skeleton.gguf").unlink()

        monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_download)
        with pytest.raises(ValueError, match="model-skeleton.gguf"):
            mock_store.ensure_downloaded(HF_PATH)
        assert not mock_store.is_downloaded(HF_PATH)

    def test_ordinary_models_are_untouched(self, mock_store, monkeypatch):
        def fake_download(repo_id, local_dir):
            from pathlib import Path

            (Path(local_dir) / "config.json").write_text("{}")

        monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_download)
        monkeypatch.setattr(
            "olmlx.engine.edge0.convert.convert_pack",
            lambda *a, **k: pytest.fail("converted a non-pack"),
        )
        local = mock_store.ensure_downloaded("org/plain-model")
        assert json.loads((local / "config.json").read_text()) == {}


class TestManagerRouting:
    def test_converted_pack_forces_flash_moe(
        self, mock_manager, mock_store, pack_template
    ):
        from olmlx.engine.registry import ModelConfig

        local = _place(mock_store, pack_template)
        convert.convert_pack(local)
        disabled = ModelConfig(hf_path=HF_PATH).resolved_flash_moe()
        assert disabled.enabled is False
        assert mock_manager._is_flash_moe_enabled(disabled, HF_PATH) is True
        assert mock_manager._flash_moe_dir(HF_PATH) == local / "flash_moe"

    def test_other_models_follow_the_flag(self, mock_manager, mock_store):
        from olmlx.engine.registry import ModelConfig

        local = mock_store.local_path("org/plain-model")
        local.mkdir(parents=True)
        (local / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
        disabled = ModelConfig(hf_path="org/plain-model").resolved_flash_moe()
        assert mock_manager._is_flash_moe_enabled(disabled, "org/plain-model") is False
        assert mock_manager._is_flash_moe_enabled(disabled) is False


class TestProjectionSetting:
    def test_changed_projection_setting_reconverts(
        self, mock_store, pack_template, monkeypatch
    ):
        local = _place(mock_store, pack_template)
        convert.convert_pack(local, projection_bits=8)
        monkeypatch.setattr("olmlx.models.store.settings.edge0_projection_bits", 8)
        mock_store.ensure_downloaded(HF_PATH)
        assert convert.read_edge0_config(local)["latent"]["projection_bits"] == 8
        monkeypatch.setattr("olmlx.models.store.settings.edge0_projection_bits", 4)
        mock_store.ensure_downloaded(HF_PATH)
        assert convert.read_edge0_config(local)["latent"]["projection_bits"] == 4

    def test_default_is_four_bit(self):
        from olmlx.config import Settings

        assert Settings().edge0_projection_bits == 4
