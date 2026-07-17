"""Tests for olmlx.models.store."""

import json
from pathlib import Path

import pytest

from olmlx.engine.registry import KnownDraft, ModelRegistry
from olmlx.models.manifest import ModelManifest
from olmlx.models.store import (
    ModelStore,
    _dir_size,
    _extract_metadata,
    _safe_dir_name,
)


class TestRegisterSpeculativeDraft:
    """ModelStore.register_speculative_draft persists curated draft config (#514)."""

    def _store(self, tmp_path, monkeypatch, config):
        config_path = tmp_path / "models.json"
        config_path.write_text(json.dumps(config))
        monkeypatch.setattr("olmlx.engine.registry.settings.models_config", config_path)
        reg = ModelRegistry()
        reg.load()
        return ModelStore(reg), config_path

    def test_writes_speculative_fields(self, tmp_path, monkeypatch):
        store, config_path = self._store(
            tmp_path, monkeypatch, {"qwen3:latest": "Qwen/Qwen3-32B-MLX"}
        )
        draft = KnownDraft(
            draft_repo="org/Qwen3-32B-eagle", strategy="eagle", block_size=4
        )
        store.register_speculative_draft("qwen3:latest", "Qwen/Qwen3-32B-MLX", draft)

        resolved = store.registry.resolve("qwen3:latest")
        assert resolved.speculative is True
        assert resolved.speculative_strategy == "eagle"
        assert resolved.speculative_draft_model == "org/Qwen3-32B-eagle"
        assert resolved.speculative_tokens == 4
        # Persisted to disk.
        assert "org/Qwen3-32B-eagle" in config_path.read_text()

    def test_preserves_existing_per_model_config(self, tmp_path, monkeypatch):
        store, _ = self._store(
            tmp_path,
            monkeypatch,
            {"qwen3:latest": {"hf_path": "Qwen/Qwen3-32B-MLX", "keep_alive": "30m"}},
        )
        draft = KnownDraft(draft_repo="org/draft", strategy="dflash", block_size=8)
        store.register_speculative_draft("qwen3:latest", "Qwen/Qwen3-32B-MLX", draft)
        resolved = store.registry.resolve("qwen3:latest")
        assert resolved.keep_alive == "30m"
        assert resolved.speculative_draft_model == "org/draft"

    def test_works_for_full_hf_path_target(self, tmp_path, monkeypatch):
        store, _ = self._store(tmp_path, monkeypatch, {})
        draft = KnownDraft(draft_repo="org/draft", strategy="classic", block_size=None)
        store.register_speculative_draft(
            "Qwen/Qwen3-32B-MLX", "Qwen/Qwen3-32B-MLX", draft
        )
        resolved = store.registry.resolve("Qwen/Qwen3-32B-MLX")
        assert resolved.speculative is True
        assert resolved.speculative_draft_model == "org/draft"
        assert resolved.speculative_tokens is None


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
            "vocab_size": 151936,
            "intermediate_size": 11008,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
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

    @staticmethod
    def _parse_params(size: str) -> float:
        """Parse a "494M" / "4.0B" parameter_size string into a raw count."""
        assert size, "parameter_size was empty"
        unit = size[-1]
        value = float(size[:-1])
        return value * {"M": 1e6, "B": 1e9}[unit]

    def test_parameter_size_qwen2_dense_ballpark(self, tmp_path):
        """Issue #642: the estimate must include the embedding matrix and the
        real (gated) MLP width, not just ``hidden**2 * layers * 4``. Qwen2.5
        -0.5B-Instruct has ~494M real parameters; the old formula reported
        ~77M (6.4x too low)."""
        # Real Qwen2.5-0.5B-Instruct config values.
        config = {
            "model_type": "qwen2",
            "hidden_size": 896,
            "num_hidden_layers": 24,
            "vocab_size": 151936,
            "intermediate_size": 4864,
            "num_attention_heads": 14,
            "num_key_value_heads": 2,
            "tie_word_embeddings": True,
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        params = self._parse_params(_extract_metadata(tmp_path)["parameter_size"])
        assert 0.44e9 <= params <= 0.55e9, f"got {params:.3e}, want ~494M"

    def test_parameter_size_qwen3_dense_gqa_ballpark(self, tmp_path):
        """Qwen3-4B (GQA, head_dim != hidden/heads) has ~4.0B real params;
        the old formula reported ~944M (4.3x too low)."""
        config = {
            "model_type": "qwen3",
            "hidden_size": 2560,
            "num_hidden_layers": 36,
            "vocab_size": 151936,
            "intermediate_size": 9728,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "tie_word_embeddings": True,
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        params = self._parse_params(_extract_metadata(tmp_path)["parameter_size"])
        assert 3.5e9 <= params <= 4.5e9, f"got {params:.3e}, want ~4.0B"

    def test_parameter_size_moe_reads_text_config(self, tmp_path):
        """Newer MoE/hybrid configs nest the LM dims under ``text_config`` and
        the total is dominated by the routed experts. Qwen3.6-35B-A3B
        (256 experts) has ~35B total params; the estimator must read
        ``text_config`` and sum all experts, not report the top-level nulls
        as an empty size."""
        text_config = {
            "hidden_size": 2048,
            "num_hidden_layers": 40,
            "vocab_size": 248320,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "num_experts": 256,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512,
            "tie_word_embeddings": False,
        }
        config = {"model_type": "qwen3_5_moe", "text_config": text_config}
        (tmp_path / "config.json").write_text(json.dumps(config))
        params = self._parse_params(_extract_metadata(tmp_path)["parameter_size"])
        assert 25e9 <= params <= 45e9, f"got {params:.3e}, want ~35B"

    def test_parameter_size_missing_dims_stays_empty(self, tmp_path):
        """A config with no usable dimensions yields an empty size rather than
        a bogus number."""
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "mystery"}))
        assert _extract_metadata(tmp_path)["parameter_size"] == ""


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

    def test_false_when_downloading_marker_present(self, mock_store):
        """A directory with .downloading marker is not considered downloaded."""
        local_dir = mock_store.local_path("org/model")
        local_dir.mkdir(parents=True)
        (local_dir / "config.json").write_text("{}")
        (local_dir / ".downloading").touch()
        assert mock_store.is_downloaded("org/model") is False


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
        from olmlx.engine.registry import ModelConfig

        mock_store.registry._mappings["test:latest"] = ModelConfig(hf_path="test/model")
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
        monkeypatch.setattr("olmlx.engine.registry.settings.models_config", models_json)

        with patch("huggingface_hub.snapshot_download"):
            async for _ in mock_store.pull("qwen3"):
                pass

        # Check that registry was updated
        assert mock_store.registry.resolve("qwen3").hf_path == "Qwen/Qwen3-8B-MLX"

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
    async def test_pull_keeps_partial_dir_on_download_failure(
        self, mock_store, tmp_path
    ):
        """If snapshot_download fails, partial directory is kept for resume."""
        from unittest.mock import patch

        with patch(
            "huggingface_hub.snapshot_download",
            side_effect=Exception("network error"),
        ):
            with pytest.raises(Exception, match="network error"):
                async for _ in mock_store.pull("qwen3"):
                    pass

        # The partial directory should be kept (snapshot_download can resume),
        # and the .downloading marker should remain so is_downloaded() is False
        local_dir = mock_store.local_path("Qwen/Qwen3-8B-MLX")
        assert local_dir.exists()
        assert (local_dir / ".downloading").exists()
        assert not mock_store.is_downloaded("Qwen/Qwen3-8B-MLX")

    @pytest.mark.asyncio
    async def test_pull_removes_downloading_marker_on_success(
        self, mock_store, tmp_path
    ):
        """On successful download, .downloading marker is removed."""
        from unittest.mock import patch

        with patch("huggingface_hub.snapshot_download"):
            async for _ in mock_store.pull("qwen3"):
                pass

        local_dir = mock_store.local_path("Qwen/Qwen3-8B-MLX")
        assert local_dir.exists()
        assert not (local_dir / ".downloading").exists()

    @pytest.mark.asyncio
    async def test_pull_succeeds_when_marker_unlink_raises_oserror(
        self, mock_store, tmp_path
    ):
        """If marker.unlink() raises a non-ENOENT OSError, pull still succeeds."""
        from unittest.mock import patch

        original_unlink = Path.unlink

        def unlink_that_fails_on_downloading(self_path, **kwargs):
            if self_path.name == ".downloading":
                raise OSError("permission denied")
            return original_unlink(self_path, **kwargs)

        with (
            patch("huggingface_hub.snapshot_download"),
            patch.object(Path, "unlink", unlink_that_fails_on_downloading),
        ):
            events = []
            async for event in mock_store.pull("qwen3"):
                events.append(event)

        assert any(e["status"] == "success" for e in events)
        # Marker renamed so is_downloaded() isn't permanently poisoned
        local_dir = mock_store.local_path("Qwen/Qwen3-8B-MLX")
        assert not (local_dir / ".downloading").exists()
        assert (local_dir / ".downloading.failed").exists()

    @pytest.mark.asyncio
    async def test_pull_unknown_model(self, mock_store, registry):
        with pytest.raises(ValueError, match="not found"):
            async for _ in mock_store.pull("totally_unknown"):
                pass


class TestConcurrentPull:
    @pytest.mark.asyncio
    async def test_concurrent_pulls_same_model_download_once(self, mock_store):
        """Two concurrent pulls of the same model should only trigger one download."""
        import asyncio
        from unittest.mock import patch

        call_count = 0

        def counting_download(**kwargs):
            nonlocal call_count
            call_count += 1
            # Simulate download creating config.json
            local_dir = mock_store.local_path("Qwen/Qwen3-8B-MLX")
            local_dir.mkdir(parents=True, exist_ok=True)
            (local_dir / "config.json").write_text("{}")

        async def collect_pull(name: str) -> list[dict]:
            events = []
            async for event in mock_store.pull(name):
                events.append(event)
            return events

        with patch("huggingface_hub.snapshot_download", side_effect=counting_download):
            results = await asyncio.gather(
                collect_pull("qwen3"),
                collect_pull("qwen3"),
            )

        assert call_count == 1
        # Both should complete with success
        for events in results:
            statuses = [e["status"] for e in events]
            assert "success" in statuses

    @pytest.mark.asyncio
    async def test_concurrent_pull_second_sees_already_downloaded(self, mock_store):
        """Second concurrent pull should see 'already downloaded' after first completes."""
        import asyncio
        from unittest.mock import patch

        def downloading(**kwargs):
            local_dir = mock_store.local_path("Qwen/Qwen3-8B-MLX")
            local_dir.mkdir(parents=True, exist_ok=True)
            (local_dir / "config.json").write_text("{}")

        async def collect_pull(name: str) -> list[dict]:
            events = []
            async for event in mock_store.pull(name):
                events.append(event)
            return events

        with patch("huggingface_hub.snapshot_download", side_effect=downloading):
            results = await asyncio.gather(
                collect_pull("qwen3"),
                collect_pull("qwen3"),
            )

        # Exactly one should have "already downloaded"
        already_downloaded_count = sum(
            1
            for events in results
            if any(e["status"] == "already downloaded" for e in events)
        )
        assert already_downloaded_count == 1


class TestEnsureDownloaded:
    def test_skips_if_already_downloaded(self, mock_store):
        """Returns local path without downloading if model is already present."""
        local_dir = mock_store.local_path("org/model")
        local_dir.mkdir(parents=True)
        (local_dir / "config.json").write_text("{}")

        from unittest.mock import patch

        with patch("huggingface_hub.snapshot_download") as mock_dl:
            result = mock_store.ensure_downloaded("org/model")

        mock_dl.assert_not_called()
        assert result == local_dir

    def test_downloads_and_removes_marker(self, mock_store):
        """Downloads model and removes .downloading marker on success."""
        from unittest.mock import patch

        with patch("huggingface_hub.snapshot_download"):
            result = mock_store.ensure_downloaded("Qwen/Qwen3-8B-MLX")

        local_dir = mock_store.local_path("Qwen/Qwen3-8B-MLX")
        assert result == local_dir
        assert local_dir.exists()
        assert not (local_dir / ".downloading").exists()

    def test_keeps_partial_dir_on_failure(self, mock_store):
        """Partial directory is kept for resume on download failure."""
        from unittest.mock import patch

        with patch(
            "huggingface_hub.snapshot_download",
            side_effect=Exception("network error"),
        ):
            with pytest.raises(Exception, match="network error"):
                mock_store.ensure_downloaded("Qwen/Qwen3-8B-MLX")

        local_dir = mock_store.local_path("Qwen/Qwen3-8B-MLX")
        assert local_dir.exists()
        assert (local_dir / ".downloading").exists()
        assert not mock_store.is_downloaded("Qwen/Qwen3-8B-MLX")

    def test_succeeds_when_marker_unlink_raises_oserror(self, mock_store):
        """If marker.unlink() raises OSError, marker is renamed and download succeeds."""
        from unittest.mock import patch

        original_unlink = Path.unlink

        def unlink_that_fails_on_downloading(self_path, **kwargs):
            if self_path.name == ".downloading":
                raise OSError("permission denied")
            return original_unlink(self_path, **kwargs)

        with (
            patch("huggingface_hub.snapshot_download"),
            patch.object(Path, "unlink", unlink_that_fails_on_downloading),
        ):
            result = mock_store.ensure_downloaded("Qwen/Qwen3-8B-MLX")

        local_dir = mock_store.local_path("Qwen/Qwen3-8B-MLX")
        assert result == local_dir
        # Marker renamed so is_downloaded() isn't permanently poisoned
        assert not (local_dir / ".downloading").exists()
        assert (local_dir / ".downloading.failed").exists()

    def test_concurrent_calls_serialize(self, mock_store):
        """Concurrent ensure_downloaded for same model only downloads once."""
        import concurrent.futures
        from unittest.mock import patch

        call_count = 0

        def counting_download(**kwargs):
            nonlocal call_count
            call_count += 1
            # Simulate download creating config.json
            local_dir = mock_store.local_path("Qwen/Qwen3-8B-MLX")
            (local_dir / "config.json").write_text("{}")

        with patch("huggingface_hub.snapshot_download", side_effect=counting_download):
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                futures = [
                    pool.submit(mock_store.ensure_downloaded, "Qwen/Qwen3-8B-MLX")
                    for _ in range(4)
                ]
                results = [f.result() for f in futures]

        assert call_count == 1
        assert all(r == mock_store.local_path("Qwen/Qwen3-8B-MLX") for r in results)


class TestDeriveManifest:
    def test_derive_manifest_no_config(self, tmp_path):
        """_derive_manifest works even without config.json (returns empty metadata)."""
        from olmlx.models.store import _derive_manifest

        manifest = _derive_manifest(tmp_path, "test:latest", "test/model")
        assert manifest.name == "test:latest"
        assert manifest.hf_path == "test/model"
        assert manifest.family == ""
        assert manifest.parameter_size == ""
        assert manifest.quantization_level == ""
        assert manifest.format == "mlx"
        assert manifest.digest != ""

    def test_derive_manifest_with_config(self, tmp_path):
        """_derive_manifest reads metadata from config.json."""
        from olmlx.models.store import _derive_manifest

        config = {
            "model_type": "qwen2",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "quantization": {"bits": 4},
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        manifest = _derive_manifest(tmp_path, "test:latest", "test/model")
        assert manifest.family == "qwen2"
        assert manifest.quantization_level == "4-bit"
        assert manifest.size > 0  # at least the config.json bytes


class TestShowWithoutManifest:
    def test_show_derives_manifest_when_missing(self, mock_store, tmp_path):
        """show() returns derived manifest when manifest.json is absent."""
        # Create a model dir with config.json but no manifest.json
        local_dir = mock_store.local_path("test/model")
        local_dir.mkdir(parents=True)
        config = {
            "model_type": "llama",
            "hidden_size": 2048,
            "num_hidden_layers": 16,
        }
        (local_dir / "config.json").write_text(json.dumps(config))
        # Register the mapping so resolve works
        from olmlx.engine.registry import ModelConfig

        mock_store.registry._mappings["test:latest"] = ModelConfig(hf_path="test/model")

        result = mock_store.show("test")
        assert result is not None
        assert result.name == "test:latest"
        assert result.family == "llama"
        # After calling show(), manifest.json should be backfilled
        assert (local_dir / "manifest.json").exists()

    def test_show_backfilled_manifest_is_valid(self, mock_store, tmp_path):
        """After backfill, subsequent show() calls load from manifest."""
        local_dir = mock_store.local_path("test/model")
        local_dir.mkdir(parents=True)
        config = {"model_type": "llama"}
        (local_dir / "config.json").write_text(json.dumps(config))
        from olmlx.engine.registry import ModelConfig

        mock_store.registry._mappings["test:latest"] = ModelConfig(hf_path="test/model")

        # First call derives and backfills
        result1 = mock_store.show("test")
        # Second call loads from backfilled manifest
        result2 = mock_store.show("test")
        assert result1 is not None
        assert result2 is not None
        assert result1.name == result2.name
        assert result1.family == result2.family


class TestResolveModelDir:
    def test_resolve_returns_none_for_unknown(self, mock_store):
        assert mock_store._resolve_model_dir("nonexistent") is None

    def test_resolve_returns_path_with_manifest_flag(self, mock_store, tmp_path):
        """_resolve_model_dir returns (path, True) when manifest.json exists."""
        local_dir = mock_store.local_path("test/model")
        local_dir.mkdir(parents=True)
        (local_dir / "config.json").write_text("{}")
        (local_dir / "manifest.json").write_text(
            json.dumps({"name": "test:latest", "hf_path": "test/model"})
        )
        from olmlx.engine.registry import ModelConfig

        mock_store.registry._mappings["test:latest"] = ModelConfig(hf_path="test/model")

        result = mock_store._resolve_model_dir("test")
        assert result is not None
        path, has_manifest = result
        assert path == local_dir
        assert has_manifest is True

    def test_resolve_returns_path_without_manifest(self, mock_store, tmp_path):
        """_resolve_model_dir returns (path, False) when only config.json exists."""
        local_dir = mock_store.local_path("test/model")
        local_dir.mkdir(parents=True)
        (local_dir / "config.json").write_text("{}")
        from olmlx.engine.registry import ModelConfig

        mock_store.registry._mappings["test:latest"] = ModelConfig(hf_path="test/model")

        result = mock_store._resolve_model_dir("test")
        assert result is not None
        path, has_manifest = result
        assert path == local_dir
        assert has_manifest is False

    def test_resolve_hf_path_with_latest_tag_and_manifest(self, mock_store):
        """_resolve_model_dir finds the stripped path when name carries :latest tag."""
        # pull() strips the tag before writing to disk; dir is keyed on stripped path
        stripped_dir = mock_store.local_path("mlx-community/Qwen3-4B-4bit")
        stripped_dir.mkdir(parents=True)
        (stripped_dir / "config.json").write_text("{}")
        (stripped_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "name": "mlx-community/Qwen3-4B-4bit:latest",
                    "hf_path": "mlx-community/Qwen3-4B-4bit",
                }
            )
        )

        result = mock_store._resolve_model_dir("mlx-community/Qwen3-4B-4bit:latest")
        assert result is not None
        path, has_manifest = result
        assert path == stripped_dir
        assert has_manifest is True

    def test_resolve_hf_path_with_latest_tag_config_only(self, mock_store):
        """_resolve_model_dir returns (path, False) for :latest-tagged HF paths with only config.json."""
        stripped_dir = mock_store.local_path("mlx-community/Qwen3-4B-4bit")
        stripped_dir.mkdir(parents=True)
        (stripped_dir / "config.json").write_text("{}")

        result = mock_store._resolve_model_dir("mlx-community/Qwen3-4B-4bit:latest")
        assert result is not None
        path, has_manifest = result
        assert path == stripped_dir
        assert has_manifest is False


class TestShowHfPathWithTag:
    def test_show_hf_path_with_latest_tag(self, mock_store):
        """show() returns a ModelManifest for an HF-path name carrying :latest tag."""
        stripped_dir = mock_store.local_path("mlx-community/Qwen3-4B-4bit")
        stripped_dir.mkdir(parents=True)
        (stripped_dir / "config.json").write_text(json.dumps({"model_type": "qwen2"}))
        (stripped_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "name": "mlx-community/Qwen3-4B-4bit:latest",
                    "hf_path": "mlx-community/Qwen3-4B-4bit",
                }
            )
        )

        result = mock_store.show("mlx-community/Qwen3-4B-4bit:latest")
        assert result is not None
        assert isinstance(result, ModelManifest)


class TestListLocalWithoutManifest:
    def test_list_local_includes_dirs_without_manifest(self, mock_store, tmp_path):
        """list_local() includes model directories that lack manifest.json."""
        local_dir = mock_store.local_path("test/model")
        local_dir.mkdir(parents=True)
        config = {"model_type": "llama", "hidden_size": 2048, "num_hidden_layers": 16}
        (local_dir / "config.json").write_text(json.dumps(config))

        results = mock_store.list_local()
        assert len(results) == 1
        assert results[0].family == "llama"
        # Manifest should have been backfilled
        assert (local_dir / "manifest.json").exists()

    def test_list_local_skips_dirs_without_config(self, mock_store, tmp_path):
        """list_local() skips directories that have neither manifest.json nor config.json."""
        empty_dir = tmp_path / "models" / "empty_dir"
        empty_dir.mkdir(parents=True)
        results = mock_store.list_local()
        assert len(results) == 0
