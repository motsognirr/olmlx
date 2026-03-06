"""Tests for mlx_ollama.models.store."""

import json

import pytest

from mlx_ollama.models.manifest import ModelManifest
from mlx_ollama.models.store import (
    ModelStore,
    _dir_size,
    _extract_metadata,
    _safe_dir_name,
)


class TestSafeDirName:
    def test_simple_name(self):
        assert _safe_dir_name("qwen3") == "qwen3"

    def test_colon_replaced(self):
        assert _safe_dir_name("qwen3:latest") == "qwen3_latest"

    def test_slash_replaced(self):
        assert _safe_dir_name("org/model") == "org_model"

    def test_special_chars(self):
        assert _safe_dir_name("model@v1!") == "model_v1_"


class TestDirSize:
    def test_empty_dir(self, tmp_path):
        assert _dir_size(tmp_path) == 0

    def test_with_files(self, tmp_path):
        (tmp_path / "a.txt").write_text("hello")
        (tmp_path / "b.txt").write_text("world!")
        assert _dir_size(tmp_path) == 5 + 6

    def test_nested(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "file.bin").write_bytes(b"\x00" * 100)
        assert _dir_size(tmp_path) == 100


class TestExtractMetadata:
    def test_no_config(self, tmp_path):
        meta = _extract_metadata(tmp_path)
        assert meta["family"] == ""
        assert meta["parameter_size"] == ""
        assert meta["quantization_level"] == ""

    def test_with_config(self, tmp_path):
        config = {
            "model_type": "qwen2",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        meta = _extract_metadata(tmp_path)
        assert meta["family"] == "qwen2"
        assert meta["parameter_size"] != ""

    def test_with_quantization_in_config(self, tmp_path):
        config = {
            "model_type": "llama",
            "quantization": {"bits": 4, "group_size": 64},
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        meta = _extract_metadata(tmp_path)
        assert meta["quantization_level"] == "4-bit"

    def test_with_quantize_config(self, tmp_path):
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "test"}))
        (tmp_path / "quantize_config.json").write_text(json.dumps({"bits": 8}))
        meta = _extract_metadata(tmp_path)
        # quantize_config.json is checked first, then config.json quantization overrides
        assert "bit" in meta["quantization_level"]


class TestLocalPath:
    def test_returns_safe_hf_path(self, mock_store):
        path = mock_store.local_path("mlx-community/Qwen3.5-27B-mxfp8")
        assert path.name == "mlx-community_Qwen3.5-27B-mxfp8"

    def test_different_names_same_hf_share_path(self, mock_store):
        """Multiple Ollama names mapping to the same HF path share one directory."""
        p1 = mock_store.local_path("org/model")
        p2 = mock_store.local_path("org/model")
        assert p1 == p2


class TestIsDownloaded:
    def test_false_when_missing(self, mock_store):
        assert mock_store.is_downloaded("org/model") is False

    def test_true_when_config_exists(self, mock_store):
        local_dir = mock_store.local_path("org/model")
        local_dir.mkdir(parents=True)
        (local_dir / "config.json").write_text("{}")
        assert mock_store.is_downloaded("org/model") is True


class TestModelStore:
    def test_list_local_empty(self, mock_store):
        assert mock_store.list_local() == []

    def test_list_local_with_model(self, mock_store, tmp_path):
        models_dir = tmp_path / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        model_dir = models_dir / "test_model"
        model_dir.mkdir()
        manifest = ModelManifest(name="test:latest", hf_path="test/model")
        manifest.save(model_dir / "manifest.json")

        results = mock_store.list_local()
        assert len(results) == 1
        assert results[0].name == "test:latest"

    def test_show_not_found(self, mock_store):
        assert mock_store.show("nonexistent") is None

    def test_show_found_hf_path(self, mock_store, tmp_path):
        """Show finds model stored under HF-path-based directory."""
        # Store under HF path (new convention)
        local_dir = mock_store.local_path("test/model")
        local_dir.mkdir(parents=True)
        # Register the mapping so resolve works
        mock_store.registry._mappings["test:latest"] = "test/model"
        manifest = ModelManifest(name="test:latest", hf_path="test/model")
        manifest.save(local_dir / "manifest.json")

        result = mock_store.show("test")
        assert result is not None
        assert result.name == "test:latest"

    def test_show_found_legacy_dir(self, mock_store, tmp_path):
        """Show still finds models stored under old name-based directories."""
        models_dir = tmp_path / "models"
        safe_name = "test_latest"
        model_dir = models_dir / safe_name
        model_dir.mkdir(parents=True)
        manifest = ModelManifest(name="test:latest", hf_path="test/model")
        manifest.save(model_dir / "manifest.json")

        result = mock_store.show("test")
        assert result is not None
        assert result.name == "test:latest"

    def test_delete_not_found(self, mock_store):
        assert mock_store.delete("nonexistent") is False

    def test_delete_found(self, mock_store, tmp_path):
        models_dir = tmp_path / "models"
        model_dir = models_dir / "test_latest"
        model_dir.mkdir(parents=True)
        (model_dir / "file.bin").write_bytes(b"\x00")
        manifest = ModelManifest(name="test:latest", hf_path="test/model")
        manifest.save(model_dir / "manifest.json")

        assert mock_store.delete("test") is True
        assert not model_dir.exists()

    def test_has_blob_false(self, mock_store):
        assert mock_store.has_blob("sha256:abc") is False

    def test_has_blob_true(self, mock_store, tmp_path):
        blobs_dir = tmp_path / "models" / "blobs"
        blobs_dir.mkdir(parents=True)
        (blobs_dir / "sha256:abc").write_bytes(b"data")
        assert mock_store.has_blob("sha256:abc") is True

    @pytest.mark.asyncio
    async def test_save_blob(self, mock_store, tmp_path):
        await mock_store.save_blob("sha256:test", b"blob_data")
        blob_path = tmp_path / "models" / "blobs" / "sha256:test"
        assert blob_path.exists()
        assert blob_path.read_bytes() == b"blob_data"

    @pytest.mark.asyncio
    async def test_pull(self, mock_store, tmp_path):
        from unittest.mock import patch

        with patch("huggingface_hub.snapshot_download"):
            events = []
            async for event in mock_store.pull("qwen3"):
                events.append(event)

        statuses = [e["status"] for e in events]
        assert "pulling manifest" in statuses
        assert "success" in statuses
        # Directory should be named after HF path, not Ollama name
        hf_dir = mock_store.local_path("Qwen/Qwen3-8B-MLX")
        assert hf_dir.exists()

    @pytest.mark.asyncio
    async def test_pull_auto_registers(self, mock_store, tmp_path, monkeypatch):
        """Pull should auto-register the mapping in models.json."""
        from unittest.mock import patch

        models_json = tmp_path / "models.json"
        monkeypatch.setattr(
            "mlx_ollama.engine.registry.settings.models_config", models_json
        )

        with patch("huggingface_hub.snapshot_download"):
            async for _ in mock_store.pull("qwen3"):
                pass

        # Check that registry was updated
        assert mock_store.registry.resolve("qwen3") == "Qwen/Qwen3-8B-MLX"

    @pytest.mark.asyncio
    async def test_pull_skips_if_downloaded(self, mock_store, tmp_path):
        """Pull should skip download if model is already present."""
        from unittest.mock import patch

        # Pre-create the model directory with config.json
        local_dir = mock_store.local_path("Qwen/Qwen3-8B-MLX")
        local_dir.mkdir(parents=True)
        (local_dir / "config.json").write_text("{}")

        with patch("huggingface_hub.snapshot_download") as mock_dl:
            events = []
            async for event in mock_store.pull("qwen3"):
                events.append(event)

        mock_dl.assert_not_called()
        statuses = [e["status"] for e in events]
        assert "already downloaded" in statuses
        assert "success" in statuses

    @pytest.mark.asyncio
    async def test_pull_hf_path_direct(self, mock_store, tmp_path):
        from unittest.mock import patch

        with patch("huggingface_hub.snapshot_download"):
            events = []
            async for event in mock_store.pull("org/some-model"):
                events.append(event)

        assert any(e["status"] == "success" for e in events)

    @pytest.mark.asyncio
    async def test_pull_unknown_model(self, mock_store, registry):
        with pytest.raises(ValueError, match="not found"):
            async for _ in mock_store.pull("totally_unknown"):
                pass
