"""Tests for olmlx.engine.model_manager."""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from olmlx.engine.model_manager import (
    LoadedModel,
    ModelLoadTimeoutError,
    ModelManager,
    _ensure_tokenizer_eos_in_stops,
    parse_keep_alive,
)
from olmlx.engine.registry import SpeculativeConfig
from olmlx.engine.template_caps import TemplateCaps


class TestParseKeepAlive:
    def test_seconds(self):
        assert parse_keep_alive("30s") == 30.0

    def test_minutes(self):
        assert parse_keep_alive("5m") == 300.0

    def test_hours(self):
        assert parse_keep_alive("2h") == 7200.0

    def test_zero(self):
        assert parse_keep_alive("0") == 0.0

    def test_negative_one(self):
        assert parse_keep_alive("-1") is None

    def test_integer(self):
        assert parse_keep_alive(60) == 60.0

    def test_negative_integer(self):
        assert parse_keep_alive(-1) is None

    def test_float(self):
        assert parse_keep_alive(30.5) == 30.5

    @pytest.mark.parametrize("value", ["invalid", "1d", "abc123", ""])
    def test_invalid_format_warns_and_defaults(self, value, caplog):
        with caplog.at_level(logging.WARNING, logger="olmlx.engine.model_manager"):
            assert parse_keep_alive(value) == 300.0  # default
        assert "Invalid keep_alive format" in caplog.text

    def test_zero_integer(self):
        assert parse_keep_alive(0) == 0.0

    def test_bare_integer_string(self):
        """Bare integer string '1800' should be treated as seconds."""
        assert parse_keep_alive("1800") == 1800.0

    def test_bare_integer_string_zero(self):
        assert parse_keep_alive("0") == 0.0


class TestLoadedModel:
    def test_defaults(self):
        lm = LoadedModel(
            name="test:latest",
            hf_path="test/model",
            model=MagicMock(),
            tokenizer=MagicMock(),
        )
        assert lm.is_vlm is False
        assert lm.size_bytes == 0
        assert lm.expires_at is None
        assert isinstance(lm.template_caps, TemplateCaps)
        assert lm.loaded_at > 0


class TestModelManager:
    def test_init(self, registry, mock_store):
        manager = ModelManager(registry, mock_store)
        assert manager._loaded == {}
        assert manager.store is mock_store

    def test_get_loaded_empty(self, registry, mock_store):
        manager = ModelManager(registry, mock_store)
        assert manager.get_loaded() == []

    def test_get_loaded(self, mock_manager):
        loaded = mock_manager.get_loaded()
        assert len(loaded) == 1
        assert loaded[0].name == "qwen3:latest"

    def test_unload(self, mock_manager):
        mock_manager.unload("qwen3")
        assert mock_manager.get_loaded() == []

    def test_unload_not_loaded(self, mock_manager):
        assert mock_manager.unload("nonexistent") is False

    def test_unload_active_refs_raises(self, mock_manager):
        from olmlx.engine.model_manager import ActiveRequestsError

        lm = mock_manager._loaded["qwen3:latest"]
        lm.active_refs = 1
        # ActiveRequestsError is the narrow type the unload HTTP handler
        # catches for 409. It also subclasses RuntimeError so legacy
        # callers using ``except RuntimeError:`` continue to work.
        with pytest.raises(ActiveRequestsError, match="active"):
            mock_manager.unload("qwen3")
        assert issubclass(ActiveRequestsError, RuntimeError)
        assert len(mock_manager.get_loaded()) == 1  # still loaded
        lm.active_refs = 0

    def test_unload_absorbs_close_failure(self, mock_manager):
        """unload() returns True even when _close_loaded_model raises.

        The model is already popped from ``_loaded`` before close is
        attempted, so the user-visible semantics are satisfied: the
        model is gone. Surfacing the ExceptionGroup as a 500 would
        leave the HTTP client unable to distinguish "close failed,
        model is gone" from an unrelated 500.
        """
        from unittest.mock import MagicMock

        # Replace _close_loaded_model with one that raises like the
        # real helper does when a resource close fails.
        mock_manager._close_loaded_model = MagicMock(
            side_effect=ExceptionGroup("simulated", [RuntimeError("prefetcher boom")])
        )
        result = mock_manager.unload("qwen3")
        assert result is True
        assert "qwen3:latest" not in mock_manager._loaded

    @pytest.mark.asyncio
    async def test_ensure_loaded_cached(self, mock_manager):
        lm = await mock_manager.ensure_loaded("qwen3")
        assert lm.name == "qwen3:latest"

    @pytest.mark.asyncio
    async def test_ensure_loaded_refreshes_expiry(self, mock_manager):
        lm = await mock_manager.ensure_loaded("qwen3", keep_alive="10m")
        assert lm.expires_at is not None
        assert lm.expires_at > time.time()

    @pytest.mark.asyncio
    async def test_ensure_loaded_never_expire(self, mock_manager):
        lm = await mock_manager.ensure_loaded("qwen3", keep_alive="-1")
        assert lm.expires_at is None

    @pytest.mark.asyncio
    async def test_ensure_loaded_unknown_model(self, registry, mock_store):
        manager = ModelManager(registry, mock_store)
        with pytest.raises(ValueError, match="not found"):
            await manager.ensure_loaded("unknown_model")

    @pytest.mark.asyncio
    async def test_ensure_loaded_evicts_lru(self, registry, mock_store, monkeypatch):
        monkeypatch.setattr("olmlx.engine.model_manager.settings.max_loaded_models", 1)
        manager = ModelManager(registry, mock_store)

        # Pre-load a model
        old_lm = LoadedModel(
            name="llama3:8b",
            hf_path="mlx-community/Llama-3-8B-Instruct",
            model=MagicMock(),
            tokenizer=MagicMock(),
            loaded_at=time.time() - 100,
        )
        manager._loaded["llama3:8b"] = old_lm

        # Mock _load_model
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None
        with patch.object(
            manager,
            "_load_model",
            return_value=(mock_model, mock_tokenizer, False, TemplateCaps(), None),
        ):
            await manager.ensure_loaded("qwen3")

        assert "llama3:8b" not in manager._loaded
        assert "qwen3:latest" in manager._loaded

    @pytest.mark.asyncio
    async def test_stop(self, mock_manager):
        # Should not raise even without expiry task
        await mock_manager.stop()
        assert mock_manager._loaded == {}

    @pytest.mark.asyncio
    async def test_stop_cancels_pending_cleanups(self, mock_manager):
        """stop() cancels and clears pending cleanup tasks."""

        async def dummy():
            await asyncio.sleep(100)

        task = asyncio.create_task(dummy())
        mock_manager._pending_cleanups["test:latest"] = task
        await mock_manager.stop()
        assert mock_manager._pending_cleanups == {}
        assert task.cancelled()

    @pytest.mark.asyncio
    async def test_stop_cancels_expiry_task(self, mock_manager):
        # Create a dummy task
        async def dummy():
            await asyncio.sleep(100)

        mock_manager._expiry_task = asyncio.create_task(dummy())
        await mock_manager.stop()
        assert mock_manager._expiry_task.cancelled()


class TestResolveDraftPath:
    """``_resolve_draft_path`` accepts both HF repo ids and local paths.

    Operators training drafts via ``olmlx dflash prepare`` end up with a
    directory under ``~/.olmlx/models/<target>/dflash/`` and configure
    ``--speculative-draft-model /abs/path/to/dflash``. Without
    short-circuiting local paths, the resolver passes the absolute path
    into ``store.ensure_downloaded`` → ``huggingface_hub.HfApi`` →
    ``HFValidationError`` ("Repo id must be in the form 'repo_name' or
    'namespace/repo_name'") and the request fails with a confusing 400.
    """

    def test_local_directory_returns_as_is(self, tmp_path, registry, mock_store):
        local = tmp_path / "my-draft"
        local.mkdir()
        manager = ModelManager(registry, mock_store)
        # Should not call ``store.ensure_downloaded`` — the path exists.
        with patch.object(
            mock_store, "ensure_downloaded", side_effect=AssertionError("called")
        ):
            resolved = manager._resolve_draft_path(str(local))
        assert resolved == str(local)

    def test_hf_repo_id_goes_through_store(self, tmp_path, registry, mock_store):
        manager = ModelManager(registry, mock_store)
        expected = tmp_path / "downloaded"
        expected.mkdir()
        with patch.object(
            mock_store, "ensure_downloaded", return_value=expected
        ) as mock_dl:
            resolved = manager._resolve_draft_path("namespace/repo_name")
        mock_dl.assert_called_once_with("namespace/repo_name")
        assert resolved == str(expected)

    def test_absolute_missing_path_raises_file_not_found(
        self, tmp_path, registry, mock_store
    ):
        """Absolute paths are unambiguous local references; they
        cannot also be valid HF repo ids. Falling through to
        ``ensure_downloaded`` for a missing absolute path would raise
        ``HFValidationError`` ("Repo id must be in the form
        'repo_name' or 'namespace/repo_name'") which is actively
        misleading. Raise ``FileNotFoundError`` with the actual path
        so a typo or a path pointing at a not-yet-trained draft
        produces a clear, actionable error.
        """
        manager = ModelManager(registry, mock_store)
        missing = tmp_path / "definitely-not-here"
        # Make sure ``ensure_downloaded`` is NOT consulted for a
        # missing absolute path.
        with patch.object(
            mock_store, "ensure_downloaded", side_effect=AssertionError("called")
        ):
            with pytest.raises(FileNotFoundError, match="definitely-not-here"):
                manager._resolve_draft_path(str(missing))

    def test_relative_path_collision_does_not_short_circuit(
        self, tmp_path, registry, mock_store, monkeypatch
    ):
        """Relative paths must NOT short-circuit even if a directory by
        that name happens to exist relative to CWD. Otherwise a valid HF
        repo id like ``"my-org/dflash-draft"`` is silently swapped for
        whatever the current working directory contains under that path.
        Only absolute paths are unambiguous local references.
        """
        # Set CWD to tmp_path and create a directory matching a plausible
        # HF repo id within it.
        monkeypatch.chdir(tmp_path)
        collision = tmp_path / "namespace" / "repo_name"
        collision.mkdir(parents=True)

        manager = ModelManager(registry, mock_store)
        downloaded = tmp_path / "downloaded"
        downloaded.mkdir()
        with patch.object(
            mock_store, "ensure_downloaded", return_value=downloaded
        ) as mock_dl:
            resolved = manager._resolve_draft_path("namespace/repo_name")
        # Must have gone through ``ensure_downloaded`` rather than
        # picking up the colliding local directory.
        mock_dl.assert_called_once_with("namespace/repo_name")
        assert resolved == str(downloaded)


class TestDetectModelKind:
    def _make_config(self, tmp_path, config_data):
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config_data))
        return str(config_path)

    def _make_manager(self, registry, mock_store):
        return ModelManager(registry, mock_store)

    def test_text_model(self, tmp_path, registry, mock_store):
        config_path = self._make_config(tmp_path, {"model_type": "llama"})
        manager = self._make_manager(registry, mock_store)

        with patch("huggingface_hub.hf_hub_download", return_value=config_path):
            kind = manager._detect_model_kind("test/model")
        assert kind == "text"

    def test_vlm_with_vision_keys(self, tmp_path, registry, mock_store):
        config_path = self._make_config(
            tmp_path,
            {
                "model_type": "qwen2_vl",
                "vision_config": {"hidden_size": 1024},
            },
        )
        manager = self._make_manager(registry, mock_store)

        with patch("huggingface_hub.hf_hub_download", return_value=config_path):
            kind = manager._detect_model_kind("test/vlm")
        assert kind == "vlm"

    def test_config_download_fails(self, registry, mock_store):
        manager = self._make_manager(registry, mock_store)
        with patch(
            "huggingface_hub.hf_hub_download", side_effect=Exception("not found")
        ):
            kind = manager._detect_model_kind("nonexistent/model")
        assert kind == "unknown"

    def test_no_model_type(self, tmp_path, registry, mock_store):
        config_path = self._make_config(tmp_path, {"hidden_size": 1024})
        manager = self._make_manager(registry, mock_store)

        with patch("huggingface_hub.hf_hub_download", return_value=config_path):
            kind = manager._detect_model_kind("test/model")
        assert kind == "unknown"

    def test_vision_keys_unsupported_by_vlm_but_supported_by_lm(
        self, tmp_path, registry, mock_store
    ):
        """Model has vision keys but mlx-vlm doesn't support it; mlx-lm does → text."""
        config_path = self._make_config(
            tmp_path,
            {
                "model_type": "llama",  # known to mlx-lm
                "vision_config": {},
                "image_token_id": 42,
            },
        )
        manager = self._make_manager(registry, mock_store)

        import importlib.util

        real_find_spec = importlib.util.find_spec

        def no_vlm_yes_lm(name, *args, **kwargs):
            if name.startswith("mlx_vlm.models."):
                return None
            return real_find_spec(name, *args, **kwargs)

        with patch("huggingface_hub.hf_hub_download", return_value=config_path):
            with patch("importlib.util.find_spec", side_effect=no_vlm_yes_lm):
                kind = manager._detect_model_kind("test/model")
        assert kind == "text"

    def test_vision_keys_unsupported_by_both(self, tmp_path, registry, mock_store):
        """Model has vision keys but neither library supports it → unknown."""
        config_path = self._make_config(
            tmp_path,
            {
                "model_type": "custom_vlm",
                "vision_config": {},
            },
        )
        manager = self._make_manager(registry, mock_store)

        import importlib.util

        real_find_spec = importlib.util.find_spec

        def none_for_models(name, *args, **kwargs):
            if name.startswith(("mlx_lm.models.", "mlx_vlm.models.")):
                return None
            return real_find_spec(name, *args, **kwargs)

        with patch("huggingface_hub.hf_hub_download", return_value=config_path):
            with patch("importlib.util.find_spec", side_effect=none_for_models):
                kind = manager._detect_model_kind("test/vlm")
        # Neither library supports it — return unknown to try both
        assert kind == "unknown"

    def test_text_model_with_real_imports(self, tmp_path, registry, mock_store):
        """Test _detect_model_kind with a model_type that exists in mlx-lm."""
        config_path = self._make_config(tmp_path, {"model_type": "llama"})
        manager = self._make_manager(registry, mock_store)

        with patch("huggingface_hub.hf_hub_download", return_value=config_path):
            kind = manager._detect_model_kind("test/model")
        # llama is a known text model type
        assert kind == "text"

    def test_uses_local_config_first(self, tmp_path, registry, mock_store):
        """When config.json exists locally, skip HF hub download."""
        # Write config.json in the store's local path
        local_dir = mock_store.local_path("test/model")
        local_dir.mkdir(parents=True)
        (local_dir / "config.json").write_text(json.dumps({"model_type": "llama"}))

        manager = self._make_manager(registry, mock_store)
        # Should NOT call hf_hub_download
        with patch("huggingface_hub.hf_hub_download") as mock_dl:
            kind = manager._detect_model_kind("test/model")
        assert kind == "text"
        mock_dl.assert_not_called()

    def test_hybrid_linear_attention_vlm_routes_to_text(
        self, tmp_path, registry, mock_store
    ):
        """Issue #284: VLMs with hybrid SSM+attention layers (Qwen3.5,
        Qwen3_5_moe) must route through mlx-lm's text path.  The mlx-vlm
        path crashes on stream synchronization for these models even on
        text-only requests; mlx-lm has dedicated text-only modules
        (qwen3_5.py, qwen3_5_moe.py) that work correctly.

        Discriminator: ``text_config.layer_types`` containing
        ``"linear_attention"`` signals the hybrid architecture.  Standard
        VLMs (Gemma 4, Qwen2-VL) lack this field and continue to route
        through mlx-vlm.
        """
        config_path = self._make_config(
            tmp_path,
            {
                "model_type": "qwen3_5",  # mlx-lm has qwen3_5.py
                "vision_config": {"hidden_size": 1024},
                "image_token_id": 248056,
                "text_config": {
                    "layer_types": [
                        "linear_attention",
                        "linear_attention",
                        "linear_attention",
                        "full_attention",
                    ],
                },
            },
        )
        manager = self._make_manager(registry, mock_store)

        # Mock find_spec so the test doesn't depend on the installed mlx-lm
        # version actually shipping mlx_lm.models.qwen3_5 — without the
        # mock, an older install would silently route through mlx-vlm and
        # the assertion would mask a regression.
        import importlib.util

        real_find_spec = importlib.util.find_spec

        def find_spec_with_qwen3_5(name, *args, **kwargs):
            if name == "mlx_lm.models.qwen3_5":
                return object()  # truthy sentinel
            return real_find_spec(name, *args, **kwargs)

        with patch("huggingface_hub.hf_hub_download", return_value=config_path):
            with patch("importlib.util.find_spec", side_effect=find_spec_with_qwen3_5):
                kind = manager._detect_model_kind("test/qwen3_5")
        assert kind == "text"

    def test_hybrid_linear_attention_vlm_raises_when_no_mlx_lm_module(
        self, tmp_path, registry, mock_store
    ):
        """Issue #284: when the discriminator fires (linear_attention layers
        present) but mlx-lm has no module for the model_type, raise a
        clear ValueError at detection time rather than falling through to
        the mlx-vlm path that we know crashes."""
        config_path = self._make_config(
            tmp_path,
            {
                "model_type": "future_hybrid_vlm",
                "vision_config": {"hidden_size": 1024},
                "text_config": {"layer_types": ["linear_attention", "full_attention"]},
            },
        )
        manager = self._make_manager(registry, mock_store)

        import importlib.util

        real_find_spec = importlib.util.find_spec

        def no_mlx_lm_module(name, *args, **kwargs):
            if name.startswith("mlx_lm.models."):
                return None
            return real_find_spec(name, *args, **kwargs)

        with (
            patch("huggingface_hub.hf_hub_download", return_value=config_path),
            patch("importlib.util.find_spec", side_effect=no_mlx_lm_module),
            pytest.raises(ValueError, match="hybrid linear-attention"),
        ):
            manager._detect_model_kind("test/future_hybrid_vlm")

    def test_hybrid_linear_attention_vlm_uses_text_config_model_type(
        self, tmp_path, registry, mock_store
    ):
        """Issue #284: when the top-level model_type is VLM-specific (e.g.
        a hypothetical ``qwen3_5_vl``) but ``text_config.model_type`` names
        the architecture mlx-lm actually has a module for (``qwen3_5``),
        the lookup should prefer the text_config key.  Otherwise the
        routing falls through to ``unknown`` and the model fails to load.
        """
        config_path = self._make_config(
            tmp_path,
            {
                # Top-level: VLM-specific name with no mlx-lm module.
                "model_type": "qwen3_5_vl",
                "vision_config": {"hidden_size": 1024},
                "text_config": {
                    # Inner: the architecture name mlx-lm has a module for.
                    "model_type": "qwen3_5",
                    "layer_types": ["linear_attention", "full_attention"],
                },
            },
        )
        manager = self._make_manager(registry, mock_store)

        import importlib.util

        real_find_spec = importlib.util.find_spec

        def find_spec_with_qwen3_5_only(name, *args, **kwargs):
            if name == "mlx_lm.models.qwen3_5":
                return object()
            if name == "mlx_lm.models.qwen3_5_vl":
                return None  # mlx-lm has no qwen3_5_vl module
            return real_find_spec(name, *args, **kwargs)

        with patch("huggingface_hub.hf_hub_download", return_value=config_path):
            with patch(
                "importlib.util.find_spec", side_effect=find_spec_with_qwen3_5_only
            ):
                kind = manager._detect_model_kind("test/qwen3_5_vl")
        assert kind == "text"

    def test_hybrid_linear_attention_vlm_falls_back_to_top_level_model_type(
        self, tmp_path, registry, mock_store
    ):
        """Inverse of the previous case: ``text_config.model_type`` names a
        module mlx-lm doesn't ship (e.g. Qwen3.6's ``qwen3_5_moe_text``),
        but the top-level ``model_type`` does (``qwen3_5_moe``). The
        discriminator should fall back to the top-level type instead of
        raising — otherwise olmlx serve refuses to load any
        ``_text``-suffixed hybrid VLM, even though ``mlx_lm.load()`` (which
        the rest of the engine actually uses) handles them via the
        top-level type without issue.
        """
        config_path = self._make_config(
            tmp_path,
            {
                # Top-level: the name mlx-lm has a module for.
                "model_type": "qwen3_5_moe",
                "vision_config": {"hidden_size": 1024},
                "text_config": {
                    # Inner: the new ``_text``-suffixed convention with no
                    # matching mlx-lm module.
                    "model_type": "qwen3_5_moe_text",
                    "layer_types": ["linear_attention", "full_attention"],
                },
            },
        )
        manager = self._make_manager(registry, mock_store)

        import importlib.util

        real_find_spec = importlib.util.find_spec

        def find_spec_with_top_level_only(name, *args, **kwargs):
            if name == "mlx_lm.models.qwen3_5_moe":
                return object()
            if name == "mlx_lm.models.qwen3_5_moe_text":
                return None  # mlx-lm has no _text-suffixed module
            return real_find_spec(name, *args, **kwargs)

        with patch("huggingface_hub.hf_hub_download", return_value=config_path):
            with patch(
                "importlib.util.find_spec", side_effect=find_spec_with_top_level_only
            ):
                kind = manager._detect_model_kind("test/qwen3_5_moe")
        assert kind == "text"

    def test_hybrid_linear_attention_vlm_raises_when_mlx_lm_import_fails(
        self, tmp_path, registry, mock_store
    ):
        """Issue #284: when ``from mlx_lm.utils import MODEL_REMAPPING``
        raises ImportError (older mlx-lm without that export, or mlx-lm
        absent), the discriminator has already fired — raise a clear
        ValueError at detection time rather than fall through to the
        mlx-vlm path that we know crashes."""
        config_path = self._make_config(
            tmp_path,
            {
                "model_type": "qwen3_5",
                "vision_config": {"hidden_size": 1024},
                "text_config": {"layer_types": ["linear_attention", "full_attention"]},
            },
        )
        manager = self._make_manager(registry, mock_store)

        # Setting sys.modules["mlx_lm.utils"] = None makes Python raise
        # ImportError on subsequent ``from mlx_lm.utils import ...``
        # statements, regardless of whether mlx_lm.utils was already
        # imported in the test environment.
        with (
            patch("huggingface_hub.hf_hub_download", return_value=config_path),
            patch.dict("sys.modules", {"mlx_lm.utils": None}),
            pytest.raises(ValueError, match="hybrid linear-attention"),
        ):
            manager._detect_model_kind("test/qwen3_5")

    def test_standard_vlm_without_linear_attention_stays_vlm(
        self, tmp_path, registry, mock_store
    ):
        """Regression fence: standard VLMs (no linear_attention) must
        continue to load through mlx-vlm.  Only the hybrid SSM bug warrants
        the mlx-lm detour."""
        config_path = self._make_config(
            tmp_path,
            {
                "model_type": "qwen2_vl",  # known mlx-vlm model
                "vision_config": {"hidden_size": 1024},
                "image_token_id": 151655,
                # No layer_types — standard transformer.
            },
        )
        manager = self._make_manager(registry, mock_store)

        # Mock find_spec for the mlx-vlm verification block so the test
        # doesn't depend on the installed mlx-vlm version actually shipping
        # mlx_vlm.models.qwen2_vl.
        import importlib.util

        real_find_spec = importlib.util.find_spec

        def find_spec_with_qwen2_vl(name, *args, **kwargs):
            if name == "mlx_vlm.models.qwen2_vl":
                return object()  # truthy sentinel
            return real_find_spec(name, *args, **kwargs)

        with patch("huggingface_hub.hf_hub_download", return_value=config_path):
            with patch("importlib.util.find_spec", side_effect=find_spec_with_qwen2_vl):
                kind = manager._detect_model_kind("test/qwen2_vl")
        assert kind == "vlm"


class TestProbeCacheCapabilities:
    """Exercise _probe_cache_capabilities, including the probe-failure path
    promoted to WARNING + non-persistable default in issue #284."""

    def _make_lm(self):
        from olmlx.engine.model_manager import LoadedModel

        lm = LoadedModel(
            name="probe-test:latest",
            hf_path="test/probe",
            model=MagicMock(),
            tokenizer=MagicMock(),
            template_caps=TemplateCaps(),
        )
        # Default for the dataclass is False (issue #284 safety default);
        # set True here so the test can verify the failure path explicitly
        # flips it back to False.
        lm.supports_cache_persistence = True
        lm.supports_cache_trim = False
        return lm

    def test_probe_empty_cache_list_disables_persistence(self, registry, mock_store):
        """If ``make_prompt_cache`` returns an empty list (a degenerate model
        with no cache layers), ``_cache_supports_persistence`` returns
        False — there's no evidence the cache layout is safe — and the
        probe leaves persistence disabled.  Trim's vacuous-True for the
        same input is fine because trim has a graceful fallback;
        persistence does not."""
        manager = ModelManager(registry, mock_store)
        lm = self._make_lm()

        with patch("mlx_lm.models.cache.make_prompt_cache", return_value=[]):
            manager._probe_cache_capabilities(lm)

        # Trim is vacuously True — trim of an empty cache is a no-op
        # that mlx-lm handles cleanly, so leaving the flag True is fine.
        assert lm.supports_cache_trim is True
        # Persistence: no evidence of safety → False.
        assert lm.supports_cache_persistence is False

    def test_probe_failure_warns_and_disables_persistence(
        self, registry, mock_store, caplog
    ):
        """When ``make_prompt_cache`` raises, the probe must log at WARNING,
        force ``supports_cache_persistence = False`` (no graceful fallback
        for cache reuse), and force ``supports_cache_trim = True`` (the
        request path's existing partial-trim fallback handles it)."""
        import logging

        manager = ModelManager(registry, mock_store)
        lm = self._make_lm()

        with (
            patch(
                "mlx_lm.models.cache.make_prompt_cache",
                side_effect=RuntimeError("simulated probe failure"),
            ),
            caplog.at_level(logging.WARNING, logger="olmlx.engine.model_manager"),
        ):
            manager._probe_cache_capabilities(lm)

        assert lm.supports_cache_trim is True
        assert lm.supports_cache_persistence is False
        assert "Cache probe raised an exception" in caplog.text


class TestLoadModel:
    def _make_manager(self, registry, mock_store):
        return ModelManager(registry, mock_store)

    def _pre_download(self, mock_store, hf_path):
        """Simulate a downloaded model by creating config.json in the store."""
        local_dir = mock_store.local_path(hf_path)
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "config.json").write_text("{}")

    def test_load_text_model(self, registry, mock_store):
        manager = self._make_manager(registry, mock_store)
        self._pre_download(mock_store, "test/path")
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None

        with patch.object(manager, "_detect_model_kind", return_value="text"):
            mock_mlx_lm = MagicMock()
            mock_mlx_lm.load.return_value = (mock_model, mock_tokenizer)
            with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
                model, tokenizer, is_vlm, caps, _ = manager._load_model("test/path")

        assert is_vlm is False
        assert model is mock_model

    def test_load_vlm_model(self, registry, mock_store):
        manager = self._make_manager(registry, mock_store)
        self._pre_download(mock_store, "test/vlm")
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_processor.tokenizer = MagicMock()
        mock_processor.tokenizer.chat_template = None

        with patch.object(manager, "_detect_model_kind", return_value="vlm"):
            mock_mlx_vlm = MagicMock()
            mock_mlx_vlm.load.return_value = (mock_model, mock_processor)
            with patch.dict("sys.modules", {"mlx_vlm": mock_mlx_vlm}):
                model, tokenizer, is_vlm, caps, _ = manager._load_model("test/vlm")

        assert is_vlm is True

    def test_load_vlm_loads_chat_template_from_jinja_file(self, registry, mock_store):
        """When VLM tokenizer has no chat_template, load from chat_template.jinja."""
        manager = self._make_manager(registry, mock_store)
        local_dir = mock_store.local_path("test/vlm")
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "config.json").write_text("{}")
        template = (
            "{% if tools %}tools{% endif %}{% if enable_thinking %}<think>{% endif %}"
        )
        (local_dir / "chat_template.jinja").write_text(template)

        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_tok = MagicMock()
        mock_tok.chat_template = None
        mock_processor.tokenizer = mock_tok

        with patch.object(manager, "_detect_model_kind", return_value="vlm"):
            mock_mlx_vlm = MagicMock()
            mock_mlx_vlm.load.return_value = (mock_model, mock_processor)
            with patch.dict("sys.modules", {"mlx_vlm": mock_mlx_vlm}):
                _, _, is_vlm, caps, _ = manager._load_model("test/vlm")

        assert is_vlm is True
        assert mock_tok.chat_template == template
        assert caps.supports_tools is True
        assert caps.supports_enable_thinking is True

    def test_load_vlm_loads_chat_template_from_json_file(self, registry, mock_store):
        """When VLM tokenizer has no chat_template, load from chat_template.json."""
        manager = self._make_manager(registry, mock_store)
        local_dir = mock_store.local_path("test/vlm")
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "config.json").write_text("{}")
        template = "{% if tools %}tools{% endif %}"
        (local_dir / "chat_template.json").write_text(
            json.dumps({"chat_template": template})
        )

        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_tok = MagicMock()
        mock_tok.chat_template = None
        mock_processor.tokenizer = mock_tok

        with patch.object(manager, "_detect_model_kind", return_value="vlm"):
            mock_mlx_vlm = MagicMock()
            mock_mlx_vlm.load.return_value = (mock_model, mock_processor)
            with patch.dict("sys.modules", {"mlx_vlm": mock_mlx_vlm}):
                _, _, is_vlm, caps, _ = manager._load_model("test/vlm")

        assert is_vlm is True
        assert mock_tok.chat_template == template
        assert caps.supports_tools is True

    def test_load_vlm_falls_back_on_oserror(self, registry, mock_store):
        """When mlx_vlm.load fails with OSError, fall back to _try_lm_then_vlm."""
        manager = self._make_manager(registry, mock_store)
        self._pre_download(mock_store, "test/vlm")
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None

        with patch.object(manager, "_detect_model_kind", return_value="vlm"):
            mock_mlx_vlm = MagicMock()
            mock_mlx_vlm.load.side_effect = OSError("preprocessor_config.json")
            mock_mlx_lm = MagicMock()
            mock_mlx_lm.load.return_value = (mock_model, mock_tokenizer)
            with patch.dict(
                "sys.modules", {"mlx_vlm": mock_mlx_vlm, "mlx_lm": mock_mlx_lm}
            ):
                model, tokenizer, is_vlm, caps, _ = manager._load_model("test/vlm")

        # Should have fallen back to mlx-lm (not VLM)
        assert is_vlm is False
        assert model is mock_model

    def test_load_vlm_skips_chat_template_when_already_set(self, registry, mock_store):
        """When VLM tokenizer already has chat_template, don't overwrite."""
        manager = self._make_manager(registry, mock_store)
        local_dir = mock_store.local_path("test/vlm")
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "config.json").write_text("{}")
        (local_dir / "chat_template.jinja").write_text("file template")

        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_tok = MagicMock()
        mock_tok.chat_template = "existing template"
        mock_processor.tokenizer = mock_tok

        with patch.object(manager, "_detect_model_kind", return_value="vlm"):
            mock_mlx_vlm = MagicMock()
            mock_mlx_vlm.load.return_value = (mock_model, mock_processor)
            with patch.dict("sys.modules", {"mlx_vlm": mock_mlx_vlm}):
                _, _, _, _, _ = manager._load_model("test/vlm")

        assert mock_tok.chat_template == "existing template"

    def test_load_fallback_loads_chat_template_from_jinja_file(
        self, registry, mock_store
    ):
        """When fallback to VLM and tokenizer has no chat_template, load from file."""
        manager = self._make_manager(registry, mock_store)
        local_dir = mock_store.local_path("test/path")
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "config.json").write_text("{}")
        template = "{% if tools %}tools{% endif %}"
        (local_dir / "chat_template.jinja").write_text(template)

        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_tok = MagicMock()
        mock_tok.chat_template = None
        mock_processor.tokenizer = mock_tok

        with patch.object(manager, "_detect_model_kind", return_value="unknown"):
            mock_mlx_lm = MagicMock()
            mock_mlx_lm.load.side_effect = ValueError("fail")
            mock_mlx_vlm = MagicMock()
            mock_mlx_vlm.load.return_value = (mock_model, mock_processor)
            with patch.dict(
                "sys.modules", {"mlx_lm": mock_mlx_lm, "mlx_vlm": mock_mlx_vlm}
            ):
                _, _, is_vlm, caps, _ = manager._load_model("test/path")

        assert is_vlm is True
        assert mock_tok.chat_template == template
        assert caps.supports_tools is True

    def test_load_vlm_downloads_chat_template_from_hub(self, registry, mock_store):
        """When chat_template.jinja not local, try downloading from HF hub."""
        manager = self._make_manager(registry, mock_store)
        local_dir = mock_store.local_path("test/vlm")
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "config.json").write_text("{}")
        # No chat_template files locally

        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_tok = MagicMock()
        mock_tok.chat_template = None
        mock_processor.tokenizer = mock_tok

        template = "{% if tools %}tools{% endif %}"
        # Write the downloaded file to a temp location
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jinja", delete=False) as f:
            f.write(template)
            downloaded_path = f.name

        mock_hf_mod = MagicMock()
        mock_hf_mod.hf_hub_download.return_value = downloaded_path

        with patch.object(manager, "_detect_model_kind", return_value="vlm"):
            mock_mlx_vlm = MagicMock()
            mock_mlx_vlm.load.return_value = (mock_model, mock_processor)
            with patch.dict(
                "sys.modules",
                {"mlx_vlm": mock_mlx_vlm, "huggingface_hub": mock_hf_mod},
            ):
                _, _, is_vlm, caps, _ = manager._load_model("test/vlm")

        assert is_vlm is True
        assert mock_tok.chat_template == template
        assert caps.supports_tools is True
        mock_hf_mod.hf_hub_download.assert_called_once_with(
            "test/vlm", "chat_template.jinja"
        )

        Path(downloaded_path).unlink(missing_ok=True)

    def test_load_vlm_falls_back_to_base_model_for_chat_template(
        self, registry, mock_store
    ):
        """When primary repo lacks chat_template, try base_model from HF card."""
        manager = self._make_manager(registry, mock_store)
        local_dir = mock_store.local_path("test/vlm")
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "config.json").write_text("{}")

        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_tok = MagicMock()
        mock_tok.chat_template = None
        mock_processor.tokenizer = mock_tok

        template = "{% if tools %}tools{% endif %}"
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jinja", delete=False) as f:
            f.write(template)
            downloaded_path = f.name

        # test/vlm fails, google/base-model fails, google/base-model-it succeeds
        mock_hf_mod = MagicMock()
        mock_hf_mod.hf_hub_download.side_effect = [
            Exception("404"),  # test/vlm
            Exception("404"),  # google/base-model
            downloaded_path,  # google/base-model-it
        ]
        mock_card_data = MagicMock()
        mock_card_data.base_model = "google/base-model"
        mock_info = MagicMock()
        mock_info.card_data = mock_card_data
        mock_hf_mod.model_info.return_value = mock_info

        with patch.object(manager, "_detect_model_kind", return_value="vlm"):
            mock_mlx_vlm = MagicMock()
            mock_mlx_vlm.load.return_value = (mock_model, mock_processor)
            with patch.dict(
                "sys.modules",
                {"mlx_vlm": mock_mlx_vlm, "huggingface_hub": mock_hf_mod},
            ):
                _, _, is_vlm, caps, _ = manager._load_model("test/vlm")

        assert is_vlm is True
        assert mock_tok.chat_template == template
        assert caps.supports_tools is True
        # Should have tried base_model-it
        mock_hf_mod.hf_hub_download.assert_any_call(
            "google/base-model-it", "chat_template.jinja"
        )

        Path(downloaded_path).unlink(missing_ok=True)

    def test_load_vlm_no_double_it_suffix_for_instruct_base(self, registry, mock_store):
        """When base_model already ends with -it, don't try base-model-it-it."""
        manager = self._make_manager(registry, mock_store)
        local_dir = mock_store.local_path("test/vlm")
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "config.json").write_text("{}")

        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_tok = MagicMock()
        mock_tok.chat_template = None
        mock_processor.tokenizer = mock_tok

        template = "{% if tools %}tools{% endif %}"
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jinja", delete=False) as f:
            f.write(template)
            downloaded_path = f.name

        # test/vlm fails, google/gemma-4-27b-it fails → must NOT try -it-it
        mock_hf_mod = MagicMock()
        mock_hf_mod.hf_hub_download.side_effect = [
            Exception("404"),  # test/vlm
            Exception("404"),  # google/gemma-4-27b-it (base)
        ]
        mock_card_data = MagicMock()
        mock_card_data.base_model = "google/gemma-4-27b-it"
        mock_info = MagicMock()
        mock_info.card_data = mock_card_data
        mock_hf_mod.model_info.return_value = mock_info

        with patch.object(manager, "_detect_model_kind", return_value="vlm"):
            mock_mlx_vlm = MagicMock()
            mock_mlx_vlm.load.return_value = (mock_model, mock_processor)
            with patch.dict(
                "sys.modules",
                {"mlx_vlm": mock_mlx_vlm, "huggingface_hub": mock_hf_mod},
            ):
                _, _, is_vlm, caps, _ = manager._load_model("test/vlm")

        assert is_vlm is True
        # Must NOT have tried google/gemma-4-27b-it-it
        all_repos = [c[0][0] for c in mock_hf_mod.hf_hub_download.call_args_list]
        assert "google/gemma-4-27b-it-it" not in all_repos, (
            f"Should not try double -it suffix, but tried: {all_repos}"
        )

        Path(downloaded_path).unlink(missing_ok=True)

    def test_load_vlm_hub_download_fails_gracefully(self, registry, mock_store):
        """When all HF hub attempts fail, caps remain empty but loading succeeds."""
        manager = self._make_manager(registry, mock_store)
        local_dir = mock_store.local_path("test/vlm")
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "config.json").write_text("{}")

        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_tok = MagicMock()
        mock_tok.chat_template = None
        mock_processor.tokenizer = mock_tok

        mock_hf_mod = MagicMock()
        mock_hf_mod.hf_hub_download.side_effect = Exception("network error")
        mock_hf_mod.model_info.side_effect = Exception("network error")

        with patch.object(manager, "_detect_model_kind", return_value="vlm"):
            mock_mlx_vlm = MagicMock()
            mock_mlx_vlm.load.return_value = (mock_model, mock_processor)
            with patch.dict(
                "sys.modules",
                {"mlx_vlm": mock_mlx_vlm, "huggingface_hub": mock_hf_mod},
            ):
                _, _, is_vlm, caps, _ = manager._load_model("test/vlm")

        assert is_vlm is True
        assert caps.supports_tools is False

    def test_load_text_fallback_to_vlm(self, registry, mock_store):
        manager = self._make_manager(registry, mock_store)
        self._pre_download(mock_store, "test/path")
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_processor.tokenizer = MagicMock()
        mock_processor.tokenizer.chat_template = None

        with patch.object(manager, "_detect_model_kind", return_value="text"):
            mock_mlx_lm = MagicMock()
            mock_mlx_lm.load.side_effect = ValueError("unsupported")
            mock_mlx_vlm = MagicMock()
            mock_mlx_vlm.load.return_value = (mock_model, mock_processor)
            with patch.dict(
                "sys.modules", {"mlx_lm": mock_mlx_lm, "mlx_vlm": mock_mlx_vlm}
            ):
                model, tokenizer, is_vlm, caps, _ = manager._load_model("test/path")

        assert is_vlm is True

    def test_load_unknown_tries_mlx_lm_first(self, registry, mock_store):
        manager = self._make_manager(registry, mock_store)
        self._pre_download(mock_store, "test/path")
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None

        with patch.object(manager, "_detect_model_kind", return_value="unknown"):
            mock_mlx_lm = MagicMock()
            mock_mlx_lm.load.return_value = (mock_model, mock_tokenizer)
            with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
                model, tokenizer, is_vlm, caps, _ = manager._load_model("test/path")

        assert is_vlm is False

    def test_load_unknown_fallback_to_vlm(self, registry, mock_store):
        manager = self._make_manager(registry, mock_store)
        self._pre_download(mock_store, "test/path")
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_processor.tokenizer = MagicMock()
        mock_processor.tokenizer.chat_template = None

        with patch.object(manager, "_detect_model_kind", return_value="unknown"):
            mock_mlx_lm = MagicMock()
            mock_mlx_lm.load.side_effect = ValueError("fail")
            mock_mlx_vlm = MagicMock()
            mock_mlx_vlm.load.return_value = (mock_model, mock_processor)
            with patch.dict(
                "sys.modules", {"mlx_lm": mock_mlx_lm, "mlx_vlm": mock_mlx_vlm}
            ):
                model, tokenizer, is_vlm, caps, _ = manager._load_model("test/path")

        assert is_vlm is True

    def test_load_uses_local_path(self, registry, mock_store):
        """When model is already downloaded, load from local path, not HF repo ID."""
        manager = self._make_manager(registry, mock_store)
        # Create a fake downloaded model
        local_dir = mock_store.local_path("test/path")
        local_dir.mkdir(parents=True)
        (local_dir / "config.json").write_text(json.dumps({"model_type": "llama"}))

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None
        mock_mlx_lm = MagicMock()
        mock_mlx_lm.load.return_value = (mock_model, mock_tokenizer)

        with patch.object(manager, "_detect_model_kind", return_value="text"):
            with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
                manager._load_model("test/path")

        # Should have been called with local path, not HF repo ID
        call_arg = mock_mlx_lm.load.call_args[0][0]
        assert call_arg == str(local_dir)

    def test_load_downloads_when_not_cached(self, registry, mock_store):
        """When model is not downloaded, download it first."""
        manager = self._make_manager(registry, mock_store)

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None
        mock_mlx_lm = MagicMock()
        mock_mlx_lm.load.return_value = (mock_model, mock_tokenizer)

        with patch.object(manager, "_detect_model_kind", return_value="text"):
            with patch("huggingface_hub.snapshot_download") as mock_dl:
                with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
                    manager._load_model("test/path")

        mock_dl.assert_called_once()
        assert mock_dl.call_args[1]["repo_id"] == "test/path"

    def test_load_keeps_partial_dir_on_download_failure(self, registry, mock_store):
        """If snapshot_download fails in _load_model, partial dir is kept for resume."""
        manager = self._make_manager(registry, mock_store)

        with patch(
            "huggingface_hub.snapshot_download",
            side_effect=Exception("download failed"),
        ):
            with pytest.raises(Exception, match="download failed"):
                manager._load_model("test/path")

        # Dir kept for resume, marker stays so is_downloaded() returns False
        local_dir = mock_store.local_path("test/path")
        assert local_dir.exists()
        assert (local_dir / ".downloading").exists()
        assert not mock_store.is_downloaded("test/path")

    def test_load_removes_downloading_marker_on_success(self, registry, mock_store):
        """After successful download in _load_model, .downloading marker is gone."""
        manager = self._make_manager(registry, mock_store)

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None
        mock_mlx_lm = MagicMock()
        mock_mlx_lm.load.return_value = (mock_model, mock_tokenizer)

        with patch.object(manager, "_detect_model_kind", return_value="text"):
            with patch("huggingface_hub.snapshot_download"):
                with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
                    manager._load_model("test/path")

        local_dir = mock_store.local_path("test/path")
        assert not (local_dir / ".downloading").exists()

    def test_load_succeeds_when_marker_unlink_raises_oserror(
        self, registry, mock_store
    ):
        """If marker.unlink() raises a non-ENOENT OSError, _load_model still succeeds."""
        manager = self._make_manager(registry, mock_store)

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None
        mock_mlx_lm = MagicMock()
        mock_mlx_lm.load.return_value = (mock_model, mock_tokenizer)

        original_unlink = Path.unlink

        def unlink_that_fails_on_downloading(self_path, **kwargs):
            if self_path.name == ".downloading":
                raise OSError("permission denied")
            return original_unlink(self_path, **kwargs)

        with patch.object(manager, "_detect_model_kind", return_value="text"):
            with patch("huggingface_hub.snapshot_download"):
                with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
                    with patch.object(Path, "unlink", unlink_that_fails_on_downloading):
                        model, tok, is_vlm, caps, _ = manager._load_model("test/path")

        assert model is mock_model


class TestFlashMoeVlmFallback:
    """Flash-MoE loading should fall back to mlx-vlm for unsupported model types."""

    def _make_manager(self, registry, mock_store):
        return ModelManager(registry, mock_store)

    def _pre_download(self, mock_store, hf_path):
        local_dir = mock_store.local_path(hf_path)
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "config.json").write_text("{}")

    def _make_flash_moe_dir(self, mock_store, hf_path):
        flash_moe_dir = mock_store.local_path(hf_path) / "flash_moe"
        flash_moe_dir.mkdir(parents=True, exist_ok=True)
        moe_config = {
            "moe_layer_indices": [0, 1],
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_experts": 4,
            "num_experts_per_tok": 2,
        }
        (flash_moe_dir / "flash_moe_config.json").write_text(json.dumps(moe_config))
        (flash_moe_dir / "flash_moe_layout.json").write_text("{}")
        return flash_moe_dir

    def _mock_model_exp(self):
        exp = MagicMock()
        exp.flash_moe = True
        exp.flash_moe_io_threads = 4
        exp.flash_moe_cache_budget_experts = 16
        exp.kv_cache_quant = None
        return exp

    def test_flash_moe_falls_back_to_vlm_on_unsupported_model_type(
        self, registry, mock_store
    ):
        """When mlx-lm can't load the model (e.g. gemma4), fall back to mlx-vlm."""
        manager = self._make_manager(registry, mock_store)
        self._pre_download(mock_store, "test/moe-vlm")
        flash_moe_dir = self._make_flash_moe_dir(mock_store, "test/moe-vlm")
        model_exp = self._mock_model_exp()

        mock_vlm_model = MagicMock()
        mock_vlm_model.language_model = MagicMock()
        mock_processor = MagicMock()
        mock_processor.tokenizer = MagicMock()
        mock_processor.tokenizer.chat_template = None

        mock_wrapped = MagicMock()

        mock_mlx_lm = MagicMock()
        mock_mlx_lm.load.side_effect = ValueError("Model type gemma4 not supported.")

        mock_mlx_vlm = MagicMock()
        mock_mlx_vlm.load.return_value = (mock_vlm_model, mock_processor)

        with patch.dict(
            "sys.modules", {"mlx_lm": mock_mlx_lm, "mlx_vlm": mock_mlx_vlm}
        ):
            with patch(
                "olmlx.engine.model_manager._load_with_model_type_fallback",
                side_effect=ValueError("Model type gemma4 not supported."),
            ):
                with patch(
                    "olmlx.engine.flash.flash_moe_model.FlashMoeModelWrapper",
                    return_value=mock_wrapped,
                ):
                    with patch(
                        "olmlx.engine.flash.moe_weight_store.FlashMoeWeightStore"
                    ):
                        model, tokenizer, is_vlm, caps = manager._load_flash_moe_model(
                            "test/moe-vlm",
                            str(mock_store.local_path("test/moe-vlm")),
                            flash_moe_dir,
                            model_exp=model_exp,
                        )

        assert is_vlm is True
        mock_mlx_vlm.load.assert_called_once()

    def test_flash_moe_uses_language_model_from_vlm(self, registry, mock_store):
        """VLM fallback should extract language_model for the MoE wrapper."""
        manager = self._make_manager(registry, mock_store)
        self._pre_download(mock_store, "test/moe-vlm2")
        flash_moe_dir = self._make_flash_moe_dir(mock_store, "test/moe-vlm2")
        model_exp = self._mock_model_exp()

        mock_language_model = MagicMock()
        mock_vlm_model = MagicMock()
        mock_vlm_model.language_model = mock_language_model
        mock_processor = MagicMock()
        mock_processor.tokenizer = MagicMock()
        mock_processor.tokenizer.chat_template = None

        mock_mlx_vlm = MagicMock()
        mock_mlx_vlm.load.return_value = (mock_vlm_model, mock_processor)

        captured_model = {}

        def capture_wrapper(model, store, **kwargs):
            captured_model["model"] = model
            return MagicMock()

        with patch.dict("sys.modules", {"mlx_vlm": mock_mlx_vlm}):
            with patch(
                "olmlx.engine.model_manager._load_with_model_type_fallback",
                side_effect=ValueError("Model type gemma4 not supported."),
            ):
                with patch(
                    "olmlx.engine.flash.flash_moe_model.FlashMoeModelWrapper",
                    side_effect=capture_wrapper,
                ):
                    with patch(
                        "olmlx.engine.flash.moe_weight_store.FlashMoeWeightStore"
                    ):
                        manager._load_flash_moe_model(
                            "test/moe-vlm2",
                            str(mock_store.local_path("test/moe-vlm2")),
                            flash_moe_dir,
                            model_exp=model_exp,
                        )

        assert captured_model["model"] is mock_language_model

    def test_flash_moe_still_works_with_mlx_lm(self, registry, mock_store):
        """When mlx-lm succeeds, it should NOT fall back to mlx-vlm."""
        manager = self._make_manager(registry, mock_store)
        self._pre_download(mock_store, "test/moe-text")
        flash_moe_dir = self._make_flash_moe_dir(mock_store, "test/moe-text")
        model_exp = self._mock_model_exp()

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None

        with patch(
            "olmlx.engine.model_manager._load_with_model_type_fallback",
            return_value=(mock_model, mock_tokenizer),
        ):
            with patch(
                "olmlx.engine.flash.flash_moe_model.FlashMoeModelWrapper",
                return_value=MagicMock(),
            ):
                with patch("olmlx.engine.flash.moe_weight_store.FlashMoeWeightStore"):
                    model, tokenizer, is_vlm, caps = manager._load_flash_moe_model(
                        "test/moe-text",
                        str(mock_store.local_path("test/moe-text")),
                        flash_moe_dir,
                        model_exp=model_exp,
                    )

        assert is_vlm is False


class TestModelLoadTimeout:
    """Test configurable timeout for model loading."""

    GB = 1024 * 1024 * 1024

    @pytest.mark.asyncio
    async def test_timeout_fires_on_slow_load(self, registry, mock_store, monkeypatch):
        """When _load_model takes longer than the timeout, raise TimeoutError."""
        monkeypatch.setattr(
            "olmlx.engine.model_manager.settings.model_load_timeout", 0.1
        )
        manager = ModelManager(registry, mock_store)

        def slow_load(hf_path, **kwargs):
            time.sleep(0.4)
            return (MagicMock(), MagicMock(), False, TemplateCaps(), None)

        with (
            patch.object(manager, "_load_model", side_effect=slow_load),
            patch(
                "olmlx.utils.memory.get_metal_memory",
                return_value=1 * self.GB,
            ),
            patch("olmlx.engine.model_manager.gc.collect"),
            patch("olmlx.engine.model_manager.mx.clear_cache"),
        ):
            with pytest.raises(ModelLoadTimeoutError, match="OLMLX_MODEL_LOAD_TIMEOUT"):
                await manager.ensure_loaded("qwen3")

    @pytest.mark.asyncio
    async def test_no_timeout_by_default(self, registry, mock_store, monkeypatch):
        """With default None timeout, fast loads succeed normally."""
        monkeypatch.setattr(
            "olmlx.engine.model_manager.settings.model_load_timeout", None
        )
        manager = ModelManager(registry, mock_store)

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None

        total_ram = 64 * self.GB

        with (
            patch.object(
                manager,
                "_load_model",
                return_value=(mock_model, mock_tokenizer, False, TemplateCaps(), None),
            ),
            patch(
                "olmlx.utils.memory.get_metal_memory",
                side_effect=[1 * self.GB, int(total_ram * 0.50)],
            ),
            patch(
                "olmlx.utils.memory.get_system_memory_bytes",
                return_value=total_ram,
            ),
            patch("olmlx.engine.model_manager.gc.collect"),
            patch("olmlx.engine.model_manager.mx.clear_cache"),
        ):
            lm = await manager.ensure_loaded("qwen3")

        assert lm.name == "qwen3:latest"

    @pytest.mark.asyncio
    async def test_timeout_allows_fast_loads(self, registry, mock_store, monkeypatch):
        """With a generous timeout, fast loads succeed."""
        monkeypatch.setattr(
            "olmlx.engine.model_manager.settings.model_load_timeout", 10.0
        )
        manager = ModelManager(registry, mock_store)

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None

        total_ram = 64 * self.GB

        with (
            patch.object(
                manager,
                "_load_model",
                return_value=(mock_model, mock_tokenizer, False, TemplateCaps(), None),
            ),
            patch(
                "olmlx.utils.memory.get_metal_memory",
                side_effect=[1 * self.GB, int(total_ram * 0.50)],
            ),
            patch(
                "olmlx.utils.memory.get_system_memory_bytes",
                return_value=total_ram,
            ),
            patch("olmlx.engine.model_manager.gc.collect"),
            patch("olmlx.engine.model_manager.mx.clear_cache"),
        ):
            lm = await manager.ensure_loaded("qwen3")

        assert lm.name == "qwen3:latest"

    @pytest.mark.asyncio
    async def test_cleanup_on_timeout(self, registry, mock_store, monkeypatch):
        """On timeout, gc.collect and mx.clear_cache are called, model not in _loaded."""
        monkeypatch.setattr(
            "olmlx.engine.model_manager.settings.model_load_timeout", 0.1
        )
        manager = ModelManager(registry, mock_store)

        def slow_load(hf_path, **kwargs):
            time.sleep(0.4)
            return (MagicMock(), MagicMock(), False, TemplateCaps(), None)

        with (
            patch.object(manager, "_load_model", side_effect=slow_load),
            patch(
                "olmlx.utils.memory.get_metal_memory",
                return_value=1 * self.GB,
            ),
            patch("olmlx.engine.model_manager.gc.collect") as mock_gc,
            patch("olmlx.engine.model_manager.mx.clear_cache") as mock_clear,
        ):
            with pytest.raises(TimeoutError):
                await manager.ensure_loaded("qwen3")

        # Only pre-load flush — the BaseException handler skips gc/clear
        # when a deferred cleanup is pending (background thread still running).
        assert mock_gc.call_count == 1
        assert mock_clear.call_count == 1
        assert "qwen3:latest" not in manager._loaded

    @pytest.mark.asyncio
    async def test_deferred_cleanup_after_timeout(
        self, registry, mock_store, monkeypatch
    ):
        """After timeout, a deferred task cleans up GPU memory when the thread finishes."""
        monkeypatch.setattr(
            "olmlx.engine.model_manager.settings.model_load_timeout", 0.1
        )
        manager = ModelManager(registry, mock_store)

        def slow_load(hf_path, **kwargs):
            time.sleep(0.3)  # Short enough to finish during the test
            return (MagicMock(), MagicMock(), False, TemplateCaps(), None)

        with (
            patch.object(manager, "_load_model", side_effect=slow_load),
            patch(
                "olmlx.utils.memory.get_metal_memory",
                return_value=1 * self.GB,
            ),
            patch("olmlx.engine.model_manager.gc.collect") as mock_gc,
            patch("olmlx.engine.model_manager.mx.clear_cache") as mock_clear,
        ):
            with pytest.raises(TimeoutError):
                await manager.ensure_loaded("qwen3")

            # Only pre-load flush (BaseException handler skips when deferred)
            assert mock_gc.call_count == 1
            assert mock_clear.call_count == 1

            # Await the cleanup task directly (deterministic, no sleep needed)
            cleanup_task = manager._pending_cleanups.get("qwen3:latest")
            assert cleanup_task is not None
            await cleanup_task

            # Deferred cleanup adds one more call each
            assert mock_gc.call_count == 2
            assert mock_clear.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_after_timeout_waits_for_cleanup(
        self, registry, mock_store, monkeypatch
    ):
        """Retrying after timeout waits for deferred cleanup before starting new load."""
        monkeypatch.setattr(
            "olmlx.engine.model_manager.settings.model_load_timeout", 0.1
        )
        manager = ModelManager(registry, mock_store)

        call_count = 0

        def slow_then_fast(hf_path, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                time.sleep(0.3)  # First call triggers timeout
            return (MagicMock(), MagicMock(), False, TemplateCaps(), None)

        total_ram = 64 * self.GB

        with (
            patch.object(manager, "_load_model", side_effect=slow_then_fast),
            patch(
                "olmlx.utils.memory.get_metal_memory",
                side_effect=[1 * self.GB, 1 * self.GB, int(total_ram * 0.50)],
            ),
            patch(
                "olmlx.utils.memory.get_system_memory_bytes",
                return_value=total_ram,
            ),
            patch("olmlx.engine.model_manager.gc.collect"),
            patch("olmlx.engine.model_manager.mx.clear_cache"),
        ):
            # First call times out
            with pytest.raises(TimeoutError):
                await manager.ensure_loaded("qwen3")

            # Pending cleanup should exist
            assert "qwen3:latest" in manager._pending_cleanups

            # Retry — should wait for cleanup to finish, then load fresh
            monkeypatch.setattr(
                "olmlx.engine.model_manager.settings.model_load_timeout", None
            )
            lm = await manager.ensure_loaded("qwen3")
            assert lm.name == "qwen3:latest"

            # Cleanup should be complete
            assert "qwen3:latest" not in manager._pending_cleanups

    @pytest.mark.asyncio
    async def test_cleanup_wait_does_not_block_other_models(
        self, registry, mock_store, monkeypatch
    ):
        """Retrying a timed-out model while another model loads concurrently.

        The cleanup wait for qwen3 must not hold the lock and block a
        concurrent ensure_loaded for llama3:8b.  Verified structurally
        by checking completion order (no timing dependency).
        """
        monkeypatch.setattr(
            "olmlx.engine.model_manager.settings.model_load_timeout", 0.1
        )
        monkeypatch.setattr("olmlx.engine.model_manager.settings.max_loaded_models", 2)
        manager = ModelManager(registry, mock_store)

        call_count = 0

        def slow_then_fast(hf_path, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                time.sleep(0.5)  # First call: triggers timeout, cleanup takes 0.5s
            return (MagicMock(), MagicMock(), False, TemplateCaps(), None)

        total_ram = 64 * self.GB
        mem_before = 1 * self.GB
        mem_after = int(total_ram * 0.50)

        results = []

        with (
            patch.object(manager, "_load_model", side_effect=slow_then_fast),
            patch(
                "olmlx.utils.memory.get_metal_memory",
                side_effect=[mem_before, mem_before, mem_after, mem_before, mem_after],
            ),
            patch(
                "olmlx.utils.memory.get_system_memory_bytes",
                return_value=total_ram,
            ),
            patch("olmlx.engine.model_manager.gc.collect"),
            patch("olmlx.engine.model_manager.mx.clear_cache"),
        ):
            # Timeout on qwen3 — orphaned thread runs for ~0.5s
            with pytest.raises(TimeoutError):
                await manager.ensure_loaded("qwen3")

            monkeypatch.setattr(
                "olmlx.engine.model_manager.settings.model_load_timeout", None
            )

            # Launch qwen3 retry (waits for cleanup) AND llama3 load
            # concurrently.  Track completion order.
            async def load_and_track(name):
                lm = await manager.ensure_loaded(name)
                results.append(lm.name)
                return lm

            qwen_task = asyncio.create_task(load_and_track("qwen3"))
            llama_task = asyncio.create_task(load_and_track("llama3:8b"))

            await asyncio.gather(qwen_task, llama_task)

            # llama3 should finish before qwen3 (which waits for cleanup).
            # If the lock were held during cleanup, qwen3 would block
            # llama3 and finish first.
            assert results.index("llama3:8b") < results.index("qwen3:latest"), (
                f"Expected llama3:8b to complete before qwen3:latest, got: {results}"
            )

    @pytest.mark.asyncio
    async def test_stale_cleanup_entry_on_gc_failure(
        self, registry, mock_store, monkeypatch
    ):
        """If gc.collect raises inside _cleanup, _pending_cleanups is still cleared."""
        monkeypatch.setattr(
            "olmlx.engine.model_manager.settings.model_load_timeout", 0.1
        )
        manager = ModelManager(registry, mock_store)

        def slow_load(hf_path, **kwargs):
            time.sleep(0.3)
            return (MagicMock(), MagicMock(), False, TemplateCaps(), None)

        gc_call_count = 0

        def gc_collect_that_fails_second_time():
            nonlocal gc_call_count
            gc_call_count += 1
            if gc_call_count == 2:
                # Second call is inside _cleanup — simulate failure
                raise RuntimeError("gc failure")

        with (
            patch.object(manager, "_load_model", side_effect=slow_load),
            patch(
                "olmlx.utils.memory.get_metal_memory",
                return_value=1 * self.GB,
            ),
            patch(
                "olmlx.engine.model_manager.gc.collect",
                side_effect=gc_collect_that_fails_second_time,
            ),
            patch("olmlx.engine.model_manager.mx.clear_cache"),
        ):
            with pytest.raises(ModelLoadTimeoutError):
                await manager.ensure_loaded("qwen3")

            # Await the cleanup task — it should fail but still clear the entry
            cleanup_task = manager._pending_cleanups.get("qwen3:latest")
            assert cleanup_task is not None
            # The task raises RuntimeError from gc.collect (which runs
            # first in the outer try), but the inner finally still pops
            # the entry regardless.
            with pytest.raises(RuntimeError, match="gc failure"):
                await cleanup_task

            # Key assertion: entry is cleared despite gc failure
            assert "qwen3:latest" not in manager._pending_cleanups

    @pytest.mark.asyncio
    async def test_stop_cancels_load_task(self, registry, mock_store, monkeypatch):
        """stop() cancels the underlying load_task when cancelling cleanup tasks."""
        monkeypatch.setattr(
            "olmlx.engine.model_manager.settings.model_load_timeout", 0.1
        )
        manager = ModelManager(registry, mock_store)

        def very_slow_load(hf_path, **kwargs):
            time.sleep(10)  # Would run forever without cancellation
            return (MagicMock(), MagicMock(), False, TemplateCaps(), None)

        with (
            patch.object(manager, "_load_model", side_effect=very_slow_load),
            patch(
                "olmlx.utils.memory.get_metal_memory",
                return_value=1 * self.GB,
            ),
            patch("olmlx.engine.model_manager.gc.collect"),
            patch("olmlx.engine.model_manager.mx.clear_cache"),
        ):
            with pytest.raises(ModelLoadTimeoutError):
                await manager.ensure_loaded("qwen3")

            assert "qwen3:latest" in manager._pending_cleanups
            cleanup_task = manager._pending_cleanups["qwen3:latest"]

            # stop() should cancel cleanup without "exception never retrieved"
            await manager.stop()
            assert cleanup_task.cancelled()
            assert manager._pending_cleanups == {}

    @pytest.mark.asyncio
    async def test_raises_model_load_timeout_error(
        self, registry, mock_store, monkeypatch
    ):
        """Timeout raises ModelLoadTimeoutError (not plain TimeoutError)."""
        monkeypatch.setattr(
            "olmlx.engine.model_manager.settings.model_load_timeout", 0.1
        )
        manager = ModelManager(registry, mock_store)

        def slow_load(hf_path, **kwargs):
            time.sleep(0.4)
            return (MagicMock(), MagicMock(), False, TemplateCaps(), None)

        with (
            patch.object(manager, "_load_model", side_effect=slow_load),
            patch(
                "olmlx.utils.memory.get_metal_memory",
                return_value=1 * self.GB,
            ),
            patch("olmlx.engine.model_manager.gc.collect"),
            patch("olmlx.engine.model_manager.mx.clear_cache"),
        ):
            with pytest.raises(ModelLoadTimeoutError):
                await manager.ensure_loaded("qwen3")

    @pytest.mark.asyncio
    async def test_memory_error_with_timeout_frees_load_task(
        self, registry, mock_store, monkeypatch
    ):
        """MemoryError after successful load with timeout frees model weights.

        When timeout is set, load_task holds the result tuple.  The except
        handler must del load_task before gc.collect so the Metal buffers
        are actually reclaimable.
        """
        monkeypatch.setattr(
            "olmlx.engine.model_manager.settings.model_load_timeout", 10.0
        )
        manager = ModelManager(registry, mock_store)

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None

        total_ram = 64 * self.GB
        # mem_after exceeds limit to trigger MemoryError
        mem_after = int(total_ram * 0.90)

        with (
            patch.object(
                manager,
                "_load_model",
                return_value=(mock_model, mock_tokenizer, False, TemplateCaps(), None),
            ),
            patch(
                "olmlx.utils.memory.get_metal_memory",
                side_effect=[1 * self.GB, mem_after],
            ),
            patch(
                "olmlx.utils.memory.get_system_memory_bytes",
                return_value=total_ram,
            ),
            patch("olmlx.engine.model_manager.gc.collect") as mock_gc,
            patch("olmlx.engine.model_manager.mx.clear_cache") as mock_clear,
        ):
            with pytest.raises(MemoryError):
                await manager.ensure_loaded("qwen3")

            # gc/clear should have been called for cleanup
            assert mock_gc.call_count >= 1
            assert mock_clear.call_count >= 1
            assert "qwen3:latest" not in manager._loaded


class TestTryLmThenVlmFallback:
    """Test that _try_lm_then_vlm only falls back on expected exceptions."""

    def _make_manager(self, registry, mock_store):
        return ModelManager(registry, mock_store)

    def _pre_download(self, mock_store, hf_path):
        local_dir = mock_store.local_path(hf_path)
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "config.json").write_text("{}")

    def test_fallback_on_value_error(self, registry, mock_store):
        manager = self._make_manager(registry, mock_store)
        self._pre_download(mock_store, "test/path")
        mock_processor = MagicMock()
        mock_processor.tokenizer = MagicMock()
        mock_processor.tokenizer.chat_template = None

        mock_mlx_lm = MagicMock()
        mock_mlx_lm.load.side_effect = ValueError("unsupported model type")
        mock_mlx_vlm = MagicMock()
        mock_mlx_vlm.load.return_value = (MagicMock(), mock_processor)

        with patch.dict(
            "sys.modules", {"mlx_lm": mock_mlx_lm, "mlx_vlm": mock_mlx_vlm}
        ):
            _, _, is_vlm, _ = manager._try_lm_then_vlm("test/path", "test")
        assert is_vlm is True

    def test_fallback_on_key_error(self, registry, mock_store):
        manager = self._make_manager(registry, mock_store)
        mock_processor = MagicMock()
        mock_processor.tokenizer = MagicMock()
        mock_processor.tokenizer.chat_template = None

        mock_mlx_lm = MagicMock()
        mock_mlx_lm.load.side_effect = KeyError("missing key")
        mock_mlx_vlm = MagicMock()
        mock_mlx_vlm.load.return_value = (MagicMock(), mock_processor)

        with patch.dict(
            "sys.modules", {"mlx_lm": mock_mlx_lm, "mlx_vlm": mock_mlx_vlm}
        ):
            _, _, is_vlm, _ = manager._try_lm_then_vlm("test/path", "test")
        assert is_vlm is True

    def test_no_fallback_on_import_error(self, registry, mock_store):
        manager = self._make_manager(registry, mock_store)

        mock_mlx_lm = MagicMock()
        mock_mlx_lm.load.side_effect = ImportError("no module")

        with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
            with pytest.raises(ImportError):
                manager._try_lm_then_vlm("test/path", "test")

    def test_no_fallback_on_runtime_error(self, registry, mock_store):
        manager = self._make_manager(registry, mock_store)

        mock_mlx_lm = MagicMock()
        mock_mlx_lm.load.side_effect = RuntimeError("GPU error")

        with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
            with pytest.raises(RuntimeError):
                manager._try_lm_then_vlm("test/path", "test")

    def test_no_fallback_on_memory_error(self, registry, mock_store):
        manager = self._make_manager(registry, mock_store)

        mock_mlx_lm = MagicMock()
        mock_mlx_lm.load.side_effect = MemoryError("out of memory")

        with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
            with pytest.raises(MemoryError):
                manager._try_lm_then_vlm("test/path", "test")


class _FakeTokenizerWrapper:
    """Minimal stand-in for mlx-lm's TokenizerWrapper for stop-token tests."""

    def __init__(
        self,
        inner_eos: int | list[int] | None,
        stops: set[int] | None,
    ):
        inner = MagicMock()
        inner.eos_token_id = inner_eos
        self._tokenizer = inner
        self.eos_token_ids: set[int] | None = stops

    def add_eos_token(self, token: str) -> None:
        assert self.eos_token_ids is not None
        self.eos_token_ids.add(int(token))


class TestEnsureTokenizerEosInStops:
    """Issue #308: <|im_end|> leaks when config.json eos_token_id != template EOT."""

    def test_adds_inner_eos_when_missing(self):
        # Repro: Qwen2.5-Coder-1.5B has config.eos_token_id=151643 (<|endoftext|>)
        # but tokenizer_config eos_token=<|im_end|> (151645). The chat template
        # ends turns with 151645, so it must be in the stop set.
        tok = _FakeTokenizerWrapper(inner_eos=151645, stops={151643})
        _ensure_tokenizer_eos_in_stops(tok)
        assert tok.eos_token_ids == {151643, 151645}

    def test_noop_when_already_present(self):
        tok = _FakeTokenizerWrapper(inner_eos=151645, stops={151645})
        _ensure_tokenizer_eos_in_stops(tok)
        assert tok.eos_token_ids == {151645}

    def test_noop_when_inner_eos_missing(self, caplog):
        # None is legitimate (HF tokenizers without an EOS); must not warn.
        tok = _FakeTokenizerWrapper(inner_eos=None, stops={151643})
        with caplog.at_level(logging.WARNING):
            _ensure_tokenizer_eos_in_stops(tok)
        assert tok.eos_token_ids == {151643}
        assert not caplog.records, f"unexpected warnings: {caplog.records}"

    def test_adds_list_inner_eos(self):
        # Defensive: HF stock tokenizers expose eos_token_id as a single int,
        # but custom trust_remote_code=True tokenizers may surface list[int].
        tok = _FakeTokenizerWrapper(inner_eos=[151645, 151643], stops={151643})
        _ensure_tokenizer_eos_in_stops(tok)
        assert tok.eos_token_ids == {151643, 151645}

    def test_noop_on_non_wrapper(self):
        # mlx-vlm processors / plain HF tokenizers don't expose add_eos_token.
        processor = MagicMock(spec=["tokenizer", "eos_token_id"])
        # Calling on a non-wrapper must not raise.
        _ensure_tokenizer_eos_in_stops(processor)


class TestLoadWithModelTypeFallbackEosFix:
    """Issue #308: _load_with_model_type_fallback must augment stop tokens."""

    def test_main_path_augments_stops(self, tmp_path):
        from olmlx.engine.model_manager import _load_with_model_type_fallback

        (tmp_path / "config.json").write_text("{}")

        tok = _FakeTokenizerWrapper(inner_eos=151645, stops={151643})
        mock_mlx_lm = MagicMock()
        mock_mlx_lm.load.return_value = (MagicMock(), tok)

        _, returned = _load_with_model_type_fallback(mock_mlx_lm, str(tmp_path))
        assert returned is tok
        assert 151645 in tok.eos_token_ids
        assert 151643 in tok.eos_token_ids

    def test_fallback_path_augments_stops(self, tmp_path):
        # Exercises the model_type-remapping branch: mlx_lm.load raises, then
        # load_model + load_tokenizer are called with the stripped model_type.
        # Same EOS mismatch scenario as the main path must still be repaired.
        # We patch transformers' CONFIG_MAPPING so the test owns its
        # precondition — without the patch a future transformers release
        # dropping the chosen model_type would silently re-raise instead of
        # exercising the fallback.
        import transformers.models.auto.configuration_auto as auto_cfg

        from olmlx.engine.model_manager import _load_with_model_type_fallback

        original_cfg = {"model_type": "fakemodel2", "eos_token_id": 151643}
        (tmp_path / "config.json").write_text(json.dumps(original_cfg))

        tok = _FakeTokenizerWrapper(inner_eos=151645, stops={151643})
        mock_mlx_lm = MagicMock()
        mock_mlx_lm.load.side_effect = ValueError("unsupported model_type")
        mock_mlx_lm.utils.load_model.return_value = (
            MagicMock(),
            {"eos_token_id": 151643},
        )
        mock_mlx_lm.utils.load_tokenizer.return_value = tok

        with patch.object(auto_cfg, "CONFIG_MAPPING", {"fakemodel": object()}):
            _, returned = _load_with_model_type_fallback(mock_mlx_lm, str(tmp_path))
        assert returned is tok
        assert 151645 in tok.eos_token_ids
        assert 151643 in tok.eos_token_ids
        # config.json must be restored after the temporary remap.
        assert json.loads((tmp_path / "config.json").read_text()) == original_cfg


class TestExpiryChecker:
    @pytest.mark.asyncio
    async def test_expired_models_removed(self, registry, mock_store):
        manager = ModelManager(registry, mock_store)
        lm = LoadedModel(
            name="expired:latest",
            hf_path="test/model",
            model=MagicMock(),
            tokenizer=MagicMock(),
            expires_at=time.time() - 10,  # already expired
        )
        manager._loaded["expired:latest"] = lm

        await manager._expire_stale()

        assert "expired:latest" not in manager._loaded

    @pytest.mark.asyncio
    async def test_non_expired_models_kept(self, registry, mock_store):
        manager = ModelManager(registry, mock_store)
        lm = LoadedModel(
            name="active:latest",
            hf_path="test/model",
            model=MagicMock(),
            tokenizer=MagicMock(),
            expires_at=time.time() + 1000,
        )
        manager._loaded["active:latest"] = lm

        await manager._expire_stale()

        assert "active:latest" in manager._loaded

    @pytest.mark.asyncio
    async def test_expiry_skips_model_with_active_refs(self, registry, mock_store):
        """Models with active_refs > 0 must not be expired."""
        manager = ModelManager(registry, mock_store)
        lm = LoadedModel(
            name="busy:latest",
            hf_path="test/model",
            model=MagicMock(),
            tokenizer=MagicMock(),
            expires_at=time.time() - 10,  # expired
            active_refs=1,  # but actively in use
        )
        manager._loaded["busy:latest"] = lm

        await manager._expire_stale()

        assert "busy:latest" in manager._loaded

    @pytest.mark.asyncio
    async def test_expire_stale_isolates_per_model_failures(
        self, registry, mock_store, caplog
    ):
        """A failing close() on model A must not skip models B and C.

        Without per-model isolation, a single broken prefetcher would block
        every other expired model in the same cycle from being cleaned up.
        """
        manager = ModelManager(registry, mock_store)

        def _flash_lm(name: str, *, raises: bool = False):
            prefetcher = MagicMock()
            if raises:
                prefetcher.close.side_effect = RuntimeError(f"{name} boom")
            weight_store = MagicMock()
            model = MagicMock()
            model.prefetcher = prefetcher
            lm = LoadedModel(
                name=name,
                hf_path=f"x/{name}",
                model=model,
                tokenizer=MagicMock(),
                weight_store=weight_store,
                is_flash=True,
                expires_at=time.time() - 10,
            )
            return lm, prefetcher, weight_store

        a, _, ws_a = _flash_lm("a", raises=True)
        b, _, ws_b = _flash_lm("b")
        c, _, ws_c = _flash_lm("c")
        manager._loaded["a"] = a
        manager._loaded["b"] = b
        manager._loaded["c"] = c

        with caplog.at_level(logging.ERROR, logger="olmlx.engine.model_manager"):
            await manager._expire_stale()  # must not raise

        assert "a" not in manager._loaded
        assert "b" not in manager._loaded
        assert "c" not in manager._loaded
        # Sibling weight stores must have been closed despite A's failure.
        ws_a.close.assert_called_once()
        ws_b.close.assert_called_once()
        ws_c.close.assert_called_once()
        # _close_loaded_model logs per-resource; A's prefetcher failure
        # surfaces as "Error closing prefetcher for a".
        assert any(
            "Error closing prefetcher for a" in r.message for r in caplog.records
        )

    @pytest.mark.asyncio
    async def test_expire_stale_drops_refs_before_gc(
        self, registry, mock_store, monkeypatch
    ):
        """The expired-models list must be dropped before gc.collect().

        Otherwise gc.collect() can't reclaim the Metal buffers referenced
        by the LoadedModel objects, and the mx.clear_cache() that was
        specifically added to flush expired-model memory is effectively
        a no-op. Mirrors the ``del evicted`` pattern in
        _evict_lru_if_needed.

        Uses a weakref to assert the LoadedModel is unreachable at the
        moment gc.collect() runs — proving expired_lms was dropped.

        Assumes CPython refcount semantics: an object with refcount 0 is
        deallocated immediately, so the weakref resolves to None as soon
        as the last strong reference goes away. On a non-refcounting
        runtime (PyPy, Jython) a back-reference cycle introduced by
        MagicMock could keep the LM alive — but we also monkeypatch
        gc.collect here, so the cycle collector would not run to clean
        it up. The codebase is CPython-only (uv-managed cpython-3.11),
        so this is fine.
        """
        import weakref

        manager = ModelManager(registry, mock_store)
        weakref_alive_at_gc: list[bool] = []
        ref_holder: dict[str, Any] = {}

        def _fake_gc():
            weakref_alive_at_gc.append(ref_holder["wr"]() is not None)

        monkeypatch.setattr("olmlx.engine.model_manager.gc.collect", _fake_gc)
        monkeypatch.setattr("olmlx.engine.model_manager.mx.clear_cache", lambda: None)

        lm = LoadedModel(
            name="expired:latest",
            hf_path="test/model",
            model=MagicMock(),
            tokenizer=MagicMock(),
            expires_at=time.time() - 10,
        )
        manager._loaded["expired:latest"] = lm
        ref_holder["wr"] = weakref.ref(lm)
        del lm  # only manager._loaded holds it now

        await manager._expire_stale()

        # If expired_lms was still alive at gc time, the weakref would
        # resolve to a live object. The fix asserts it's dead.
        assert weakref_alive_at_gc == [False]

    @pytest.mark.asyncio
    async def test_expire_stale_offloads_close_to_thread(
        self, registry, mock_store, monkeypatch
    ):
        """Close runs off the event loop.

        ``executor.shutdown(wait=True)`` is synchronous. Running it on
        the event loop thread would stall every concurrent coroutine
        until the pools drained, even with the lock released. The fix
        is ``await asyncio.to_thread(self._close_loaded_model, lm)``.
        This test asserts the call went through ``asyncio.to_thread``.
        """
        manager = ModelManager(registry, mock_store)
        original_to_thread = asyncio.to_thread
        to_thread_calls: list[Any] = []

        async def _tracking_to_thread(fn, *args, **kwargs):
            to_thread_calls.append(fn)
            return await original_to_thread(fn, *args, **kwargs)

        monkeypatch.setattr(
            "olmlx.engine.model_manager.asyncio.to_thread", _tracking_to_thread
        )

        lm = LoadedModel(
            name="expired:latest",
            hf_path="test/model",
            model=MagicMock(),
            tokenizer=MagicMock(),
            expires_at=time.time() - 10,
        )
        manager._loaded["expired:latest"] = lm

        await manager._expire_stale()

        # The close was routed through to_thread (off-event-loop).
        assert manager._close_loaded_model in to_thread_calls

    @pytest.mark.asyncio
    async def test_expire_stale_releases_lock_before_closing(
        self, registry, mock_store
    ):
        """_close_loaded_model must run outside self._lock.

        ``executor.shutdown(wait=True)`` is synchronous and can take long
        enough to be noticeable. Holding ``self._lock`` during that would
        stall every concurrent ``ensure_loaded()`` caller until the pool
        drained — a latency spike on a server doing real inference when
        a keep-alive happens to expire.
        """
        manager = ModelManager(registry, mock_store)
        lock_held_during_close: list[bool] = []

        def _record_lock_state(_lm):
            lock_held_during_close.append(manager._lock.locked())

        manager._close_loaded_model = _record_lock_state  # type: ignore[assignment]
        lm = LoadedModel(
            name="expired:latest",
            hf_path="test/model",
            model=MagicMock(),
            tokenizer=MagicMock(),
            expires_at=time.time() - 10,
        )
        manager._loaded["expired:latest"] = lm

        await manager._expire_stale()

        assert lock_held_during_close == [False]

    @pytest.mark.asyncio
    async def test_check_expiry_loop_survives_unhandled_error(
        self, registry, mock_store, caplog, monkeypatch
    ):
        """The background expiry task must survive a raising _expire_stale.

        If _expire_stale ever propagates, the unguarded `while True` in
        _check_expiry_loop exits permanently — no log, no restart, models
        accumulate forever. Defense in depth on top of per-model isolation.
        """
        manager = ModelManager(registry, mock_store)
        sleep_calls = {"n": 0}

        async def _fake_sleep(_seconds):
            sleep_calls["n"] += 1
            if sleep_calls["n"] >= 2:
                raise asyncio.CancelledError()

        monkeypatch.setattr("olmlx.engine.model_manager.asyncio.sleep", _fake_sleep)
        call_count = {"n": 0}

        async def _raising_expire():
            call_count["n"] += 1
            raise RuntimeError("simulated failure")

        manager._expire_stale = _raising_expire  # type: ignore[method-assign]

        with caplog.at_level(logging.ERROR, logger="olmlx.engine.model_manager"):
            with pytest.raises(asyncio.CancelledError):
                await manager._check_expiry_loop()

        # First iteration raised → loop continued → second iteration cancelled.
        assert call_count["n"] == 1
        assert any("Expiry check failed" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_expire_stale_closes_flash_resources(self, registry, mock_store):
        """_expire_stale must close prefetcher + weight_store on a Flash model.

        Otherwise the keep-alive timer leaks ThreadPoolExecutor workers and
        per-layer file descriptors for every expired Flash model (issue #178).
        """
        manager = ModelManager(registry, mock_store)
        # Wire both prefetcher and weight_store through the same ``parent``
        # MagicMock so their .close() calls are recorded in a single ordered
        # mock_calls list. _close_loaded_model accesses prefetcher via
        # ``lm.model.prefetcher`` and weight_store via ``lm.weight_store``;
        # both end up resolving to attributes on ``parent`` here, which is
        # what makes the cross-resource ordering assertion work.
        parent = MagicMock()
        prefetcher = parent.prefetcher
        weight_store = parent.weight_store
        flash_model = MagicMock()
        flash_model.prefetcher = prefetcher
        lm = LoadedModel(
            name="expired:latest",
            hf_path="test/model",
            model=flash_model,
            tokenizer=MagicMock(),
            weight_store=weight_store,
            is_flash=True,
            expires_at=time.time() - 10,
        )
        manager._loaded["expired:latest"] = lm

        await manager._expire_stale()

        assert "expired:latest" not in manager._loaded
        prefetcher.close.assert_called_once()
        weight_store.close.assert_called_once()
        call_names = [c[0] for c in parent.mock_calls]
        assert call_names.index("prefetcher.close") < call_names.index(
            "weight_store.close"
        )

    @pytest.mark.asyncio
    async def test_expire_stale_skips_active(self, registry, mock_store):
        """Models with active_refs > 0 must not be expired or closed."""
        manager = ModelManager(registry, mock_store)
        prefetcher = MagicMock()
        weight_store = MagicMock()
        flash_model = MagicMock()
        flash_model.prefetcher = prefetcher
        lm = LoadedModel(
            name="busy:latest",
            hf_path="test/model",
            model=flash_model,
            tokenizer=MagicMock(),
            weight_store=weight_store,
            is_flash=True,
            expires_at=time.time() - 10,
            active_refs=1,
        )
        manager._loaded["busy:latest"] = lm

        await manager._expire_stale()

        assert "busy:latest" in manager._loaded
        prefetcher.close.assert_not_called()
        weight_store.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_expired_model_object_still_usable(self, registry, mock_store):
        """Even if a model is removed from _loaded, the object remains usable."""
        manager = ModelManager(registry, mock_store)
        model_mock = MagicMock()
        tokenizer_mock = MagicMock()
        lm = LoadedModel(
            name="removed:latest",
            hf_path="test/model",
            model=model_mock,
            tokenizer=tokenizer_mock,
            expires_at=time.time() - 10,
        )
        manager._loaded["removed:latest"] = lm

        # Hold a reference, then expire it
        held_ref = lm
        now = time.time()
        expired = [
            name
            for name, m in manager._loaded.items()
            if m.expires_at is not None and m.expires_at <= now and m.active_refs == 0
        ]
        for name in expired:
            del manager._loaded[name]

        # Model object still accessible via held reference
        assert "removed:latest" not in manager._loaded
        assert held_ref.model is model_mock
        assert held_ref.tokenizer is tokenizer_mock


class TestMemoryCheck:
    """Test that models exceeding the memory limit are rejected on load."""

    GB = 1024 * 1024 * 1024

    @pytest.mark.asyncio
    async def test_model_exceeding_memory_limit_raises(
        self, registry, mock_store, monkeypatch
    ):
        """When Metal memory after loading exceeds the limit, raise MemoryError."""
        manager = ModelManager(registry, mock_store)

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None

        total_ram = 64 * self.GB
        # Before load: 1 GB baseline, after load: 80% of RAM (exceeds 75% limit)
        mem_before = 1 * self.GB
        mem_after = int(total_ram * 0.80)

        with (
            patch.object(
                manager,
                "_load_model",
                return_value=(mock_model, mock_tokenizer, False, TemplateCaps(), None),
            ),
            patch(
                "olmlx.utils.memory.get_metal_memory",
                side_effect=[mem_before, mem_after],
            ),
            patch(
                "olmlx.utils.memory.get_system_memory_bytes",
                return_value=total_ram,
            ),
            patch("olmlx.engine.model_manager.gc.collect"),
            patch("olmlx.engine.model_manager.mx.clear_cache"),
        ):
            with pytest.raises(MemoryError, match="memory limit"):
                await manager.ensure_loaded("qwen3")

        # Model should NOT be in _loaded
        assert "qwen3:latest" not in manager._loaded

    @pytest.mark.asyncio
    async def test_model_within_memory_limit_loads(
        self, registry, mock_store, monkeypatch
    ):
        """When Metal memory is within the limit, the model loads normally."""
        manager = ModelManager(registry, mock_store)

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None

        total_ram = 64 * self.GB
        mem_before = 1 * self.GB
        mem_after = int(total_ram * 0.50)

        with (
            patch.object(
                manager,
                "_load_model",
                return_value=(mock_model, mock_tokenizer, False, TemplateCaps(), None),
            ),
            patch(
                "olmlx.utils.memory.get_metal_memory",
                side_effect=[mem_before, mem_after],
            ),
            patch(
                "olmlx.utils.memory.get_system_memory_bytes",
                return_value=total_ram,
            ),
            patch("olmlx.engine.model_manager.gc.collect"),
            patch("olmlx.engine.model_manager.mx.clear_cache"),
        ):
            lm = await manager.ensure_loaded("qwen3")

        assert lm.name == "qwen3:latest"
        assert "qwen3:latest" in manager._loaded

    @pytest.mark.asyncio
    async def test_custom_memory_limit_fraction(
        self, registry, mock_store, monkeypatch
    ):
        """Configurable memory_limit_fraction is respected."""
        monkeypatch.setattr(
            "olmlx.engine.model_manager.settings.memory_limit_fraction", 0.90
        )
        manager = ModelManager(registry, mock_store)

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None

        total_ram = 64 * self.GB
        # 80% usage after load — below the 90% custom limit, should load fine
        mem_before = 1 * self.GB
        mem_after = int(total_ram * 0.80)

        with (
            patch.object(
                manager,
                "_load_model",
                return_value=(mock_model, mock_tokenizer, False, TemplateCaps(), None),
            ),
            patch(
                "olmlx.utils.memory.get_metal_memory",
                side_effect=[mem_before, mem_after],
            ),
            patch(
                "olmlx.utils.memory.get_system_memory_bytes",
                return_value=total_ram,
            ),
            patch("olmlx.engine.model_manager.gc.collect"),
            patch("olmlx.engine.model_manager.mx.clear_cache"),
        ):
            lm = await manager.ensure_loaded("qwen3")

        assert lm.name == "qwen3:latest"

    @pytest.mark.asyncio
    async def test_memory_error_message_includes_guidance(self, registry, mock_store):
        """The error message should include actionable guidance."""
        manager = ModelManager(registry, mock_store)

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None

        total_ram = 64 * self.GB
        mem_before = 1 * self.GB
        mem_after = int(total_ram * 0.80)

        with (
            patch.object(
                manager,
                "_load_model",
                return_value=(mock_model, mock_tokenizer, False, TemplateCaps(), None),
            ),
            patch(
                "olmlx.utils.memory.get_metal_memory",
                side_effect=[mem_before, mem_after],
            ),
            patch(
                "olmlx.utils.memory.get_system_memory_bytes",
                return_value=total_ram,
            ),
            patch("olmlx.engine.model_manager.gc.collect"),
            patch("olmlx.engine.model_manager.mx.clear_cache"),
        ):
            with pytest.raises(MemoryError) as exc_info:
                await manager.ensure_loaded("qwen3")

        msg = str(exc_info.value)
        assert "OLMLX_MEMORY_LIMIT_FRACTION" in msg
        assert "smaller" in msg or "quantized" in msg

    @pytest.mark.asyncio
    async def test_cleanup_called_on_rejection(self, registry, mock_store):
        """gc.collect() and mx.clear_cache() are called when a model is rejected."""
        manager = ModelManager(registry, mock_store)

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None

        total_ram = 64 * self.GB
        mem_before = 1 * self.GB
        mem_after = int(total_ram * 0.80)

        with (
            patch.object(
                manager,
                "_load_model",
                return_value=(mock_model, mock_tokenizer, False, TemplateCaps(), None),
            ),
            patch(
                "olmlx.utils.memory.get_metal_memory",
                side_effect=[mem_before, mem_after],
            ),
            patch(
                "olmlx.utils.memory.get_system_memory_bytes",
                return_value=total_ram,
            ),
            patch("olmlx.engine.model_manager.gc.collect") as mock_gc,
            patch("olmlx.engine.model_manager.mx.clear_cache") as mock_clear,
        ):
            with pytest.raises(MemoryError):
                await manager.ensure_loaded("qwen3")

        # Called twice: once for pre-load cache flush, once for post-rejection cleanup
        assert mock_gc.call_count == 2
        assert mock_clear.call_count == 2

    @pytest.mark.asyncio
    async def test_cache_flushed_after_eviction(
        self, registry, mock_store, monkeypatch
    ):
        """After LRU eviction, Metal cache is flushed before measuring mem_before."""
        monkeypatch.setattr("olmlx.engine.model_manager.settings.max_loaded_models", 1)
        manager = ModelManager(registry, mock_store)

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None

        # Pre-load a model
        existing = LoadedModel(
            name="old:latest",
            hf_path="org/old",
            model=MagicMock(),
            tokenizer=MagicMock(),
        )
        manager._loaded["old:latest"] = existing

        total_ram = 64 * self.GB
        # After cache flush + load, 50% usage — well within 75% limit
        mem_before = 1 * self.GB
        mem_after = int(total_ram * 0.50)

        call_order = []

        def track_gc():
            call_order.append("gc.collect")

        def track_clear():
            call_order.append("mx.clear_cache")

        def track_get_metal(*args):
            call_order.append("get_metal")
            return (
                mem_before
                if len([c for c in call_order if c == "get_metal"]) == 1
                else mem_after
            )

        with (
            patch.object(
                manager,
                "_load_model",
                return_value=(mock_model, mock_tokenizer, False, TemplateCaps(), None),
            ),
            patch(
                "olmlx.utils.memory.get_metal_memory",
                side_effect=track_get_metal,
            ),
            patch(
                "olmlx.utils.memory.get_system_memory_bytes",
                return_value=total_ram,
            ),
            patch("olmlx.engine.model_manager.gc.collect", side_effect=track_gc),
            patch("olmlx.engine.model_manager.mx.clear_cache", side_effect=track_clear),
        ):
            lm = await manager.ensure_loaded("qwen3")

        assert lm.name == "qwen3:latest"
        # Cache flush must happen before the first memory measurement
        gc_idx = call_order.index("gc.collect")
        clear_idx = call_order.index("mx.clear_cache")
        first_metal_idx = call_order.index("get_metal")
        assert gc_idx < first_metal_idx
        assert clear_idx < first_metal_idx

    @pytest.mark.asyncio
    async def test_model_mb_not_negative_in_error(self, registry, mock_store):
        """When MLX reuses cached buffers, model_mb should not be negative."""
        manager = ModelManager(registry, mock_store)

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None

        total_ram = 64 * self.GB
        # Simulate cache reuse: mem_after < mem_before but total still over limit
        mem_before = int(total_ram * 0.70)
        mem_after = int(total_ram * 0.80)

        with (
            patch.object(
                manager,
                "_load_model",
                return_value=(mock_model, mock_tokenizer, False, TemplateCaps(), None),
            ),
            patch(
                "olmlx.utils.memory.get_metal_memory",
                side_effect=[mem_before, mem_after],
            ),
            patch(
                "olmlx.utils.memory.get_system_memory_bytes",
                return_value=total_ram,
            ),
            patch("olmlx.engine.model_manager.gc.collect"),
            patch("olmlx.engine.model_manager.mx.clear_cache"),
        ):
            with pytest.raises(MemoryError) as exc_info:
                await manager.ensure_loaded("qwen3")

        msg = str(exc_info.value)
        # Extract the MB number from "requires ~X MB"
        import re

        match = re.search(r"requires ~(\d+) MB", msg)
        assert match is not None
        assert int(match.group(1)) >= 0

    @pytest.mark.asyncio
    async def test_cleanup_on_unexpected_exception_after_load(
        self, registry, mock_store
    ):
        """If get_metal_memory raises after load, GPU cleanup must still run."""
        manager = ModelManager(registry, mock_store)

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None

        with (
            patch.object(
                manager,
                "_load_model",
                return_value=(mock_model, mock_tokenizer, False, TemplateCaps(), None),
            ),
            patch(
                "olmlx.utils.memory.get_metal_memory",
                side_effect=[1 * self.GB, OSError("Metal query failed")],
            ),
            patch("olmlx.engine.model_manager.gc.collect") as mock_gc,
            patch("olmlx.engine.model_manager.mx.clear_cache") as mock_clear,
        ):
            with pytest.raises(OSError, match="Metal query failed"):
                await manager.ensure_loaded("qwen3")

        # Cleanup must have been called: once pre-load flush + once post-failure
        assert mock_gc.call_count == 2
        assert mock_clear.call_count == 2
        # Model must NOT be in _loaded
        assert "qwen3:latest" not in manager._loaded

    @pytest.mark.asyncio
    async def test_load_succeeds_when_system_memory_unknown(self, registry, mock_store):
        """If get_system_memory_bytes returns 0, memory check is skipped and load succeeds."""
        manager = ModelManager(registry, mock_store)

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None

        with (
            patch.object(
                manager,
                "_load_model",
                return_value=(mock_model, mock_tokenizer, False, TemplateCaps(), None),
            ),
            patch(
                "olmlx.utils.memory.get_metal_memory",
                side_effect=[1 * self.GB, 2 * self.GB],
            ),
            patch(
                "olmlx.utils.memory.get_system_memory_bytes",
                return_value=0,
            ),
        ):
            await manager.ensure_loaded("qwen3")

        assert "qwen3:latest" in manager._loaded

    @pytest.mark.asyncio
    async def test_cleanup_when_load_model_itself_fails(self, registry, mock_store):
        """If _load_model raises (e.g. partial GPU alloc then OOM), GPU cache must be flushed."""
        manager = ModelManager(registry, mock_store)

        with (
            patch.object(
                manager,
                "_load_model",
                side_effect=RuntimeError("Metal OOM during mlx_lm.load"),
            ),
            patch(
                "olmlx.utils.memory.get_metal_memory",
                return_value=1 * self.GB,
            ),
            patch("olmlx.engine.model_manager.gc.collect") as mock_gc,
            patch("olmlx.engine.model_manager.mx.clear_cache") as mock_clear,
        ):
            with pytest.raises(RuntimeError, match="Metal OOM"):
                await manager.ensure_loaded("qwen3")

        # Pre-load flush + post-failure flush = 2 calls each
        assert mock_gc.call_count == 2
        assert mock_clear.call_count == 2
        assert "qwen3:latest" not in manager._loaded


class TestEnsureLoadedNotFoundSuggestions:
    @pytest.mark.asyncio
    async def test_ensure_loaded_not_found_suggests_similar(self, registry, mock_store):
        """When model not found, error should include 'Did you mean' with suggestions."""
        manager = ModelManager(registry, mock_store)
        with pytest.raises(ValueError, match="Did you mean") as exc_info:
            await manager.ensure_loaded("qwem3")  # typo for qwen3
        # Should mention the similar model
        assert "qwen3:latest" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_not_found_does_not_evict_loaded_model(
        self, registry, mock_store, monkeypatch
    ):
        """Requesting a non-existent model must not evict already-loaded models."""
        monkeypatch.setattr("olmlx.engine.model_manager.settings.max_loaded_models", 1)
        manager = ModelManager(registry, mock_store)

        existing = LoadedModel(
            name="qwen3:latest",
            hf_path="Qwen/Qwen3-8B",
            model=MagicMock(),
            tokenizer=MagicMock(),
        )
        manager._loaded["qwen3:latest"] = existing

        with pytest.raises(ValueError, match="not found"):
            await manager.ensure_loaded("claude-haiku-4-5-20251001")

        # The existing model must still be loaded
        assert "qwen3:latest" in manager._loaded


class TestPerModelConfig:
    @pytest.mark.asyncio
    async def test_kv_cache_quant_stored_on_loaded_model(self, mock_manager):
        """LoadedModel should have kv_cache_quant from per-model config."""
        lm = LoadedModel(
            name="test:latest",
            hf_path="test/model",
            model=MagicMock(),
            tokenizer=MagicMock(),
            kv_cache_quant="turboquant:4",
        )
        assert lm.kv_cache_quant == "turboquant:4"

    @pytest.mark.asyncio
    async def test_default_options_stored_on_loaded_model(self):
        """LoadedModel should have default_options from per-model config."""
        lm = LoadedModel(
            name="test:latest",
            hf_path="test/model",
            model=MagicMock(),
            tokenizer=MagicMock(),
            default_options={"temperature": 0.5, "num_predict": 1024},
        )
        assert lm.default_options == {"temperature": 0.5, "num_predict": 1024}

    @pytest.mark.asyncio
    async def test_default_options_empty_by_default(self):
        """LoadedModel default_options should be empty dict by default."""
        lm = LoadedModel(
            name="test:latest",
            hf_path="test/model",
            model=MagicMock(),
            tokenizer=MagicMock(),
        )
        assert lm.default_options == {}
        assert lm.kv_cache_quant is None

    @pytest.mark.asyncio
    async def test_ensure_loaded_uses_model_config_keep_alive(
        self, tmp_path, monkeypatch
    ):
        """Per-model keep_alive is used when request doesn't specify one."""
        from olmlx.engine.registry import ModelRegistry
        from olmlx.models.store import ModelStore

        config = {
            "qwen3:latest": {
                "hf_path": "Qwen/Qwen3-8B-MLX",
                "keep_alive": "30m",
            }
        }
        config_path = tmp_path / "models.json"
        config_path.write_text(json.dumps(config))
        monkeypatch.setattr("olmlx.engine.registry.settings.models_config", config_path)
        monkeypatch.setattr(
            "olmlx.models.store.settings.models_dir", tmp_path / "models"
        )

        reg = ModelRegistry()
        reg.load()
        store = ModelStore(reg)
        manager = ModelManager(reg, store)

        # Pre-load a mock model to avoid actual loading
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None

        with patch.object(
            manager,
            "_load_model",
            return_value=(mock_model, mock_tokenizer, False, TemplateCaps(), None),
        ):
            lm = await manager.ensure_loaded("qwen3")  # no keep_alive specified

        # Should use per-model keep_alive of 30m = 1800s
        assert lm.expires_at is not None
        assert lm.expires_at >= time.time() + 1790  # ~30 minutes

    @pytest.mark.asyncio
    async def test_ensure_loaded_request_keep_alive_wins(self, tmp_path, monkeypatch):
        """Request keep_alive takes priority over per-model keep_alive."""
        from olmlx.engine.registry import ModelRegistry
        from olmlx.models.store import ModelStore

        config = {
            "qwen3:latest": {
                "hf_path": "Qwen/Qwen3-8B-MLX",
                "keep_alive": "30m",
            }
        }
        config_path = tmp_path / "models.json"
        config_path.write_text(json.dumps(config))
        monkeypatch.setattr("olmlx.engine.registry.settings.models_config", config_path)
        monkeypatch.setattr(
            "olmlx.models.store.settings.models_dir", tmp_path / "models"
        )

        reg = ModelRegistry()
        reg.load()
        store = ModelStore(reg)
        manager = ModelManager(reg, store)

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None

        with patch.object(
            manager,
            "_load_model",
            return_value=(mock_model, mock_tokenizer, False, TemplateCaps(), None),
        ):
            lm = await manager.ensure_loaded("qwen3", keep_alive="1m")

        # Should use request keep_alive of 1m = 60s, not model's 30m
        assert lm.expires_at is not None
        assert lm.expires_at < time.time() + 70  # ~1 minute


class TestEvictLruIfNeeded:
    """Tests for ModelManager._evict_lru_if_needed."""

    def test_no_eviction_below_capacity(self, registry, mock_store, monkeypatch):
        monkeypatch.setattr("olmlx.engine.model_manager.settings.max_loaded_models", 3)
        manager = ModelManager(registry, mock_store)
        lm = LoadedModel(
            name="a", hf_path="a/a", model=MagicMock(), tokenizer=MagicMock()
        )
        manager._loaded["a"] = lm
        manager._evict_lru_if_needed()
        assert "a" in manager._loaded

    def test_evicts_oldest_inactive(self, registry, mock_store, monkeypatch):
        monkeypatch.setattr("olmlx.engine.model_manager.settings.max_loaded_models", 1)
        manager = ModelManager(registry, mock_store)
        old = LoadedModel(
            name="old",
            hf_path="o/o",
            model=MagicMock(),
            tokenizer=MagicMock(),
            loaded_at=time.time() - 100,
        )
        manager._loaded["old"] = old
        manager._evict_lru_if_needed()
        assert "old" not in manager._loaded

    def test_raises_when_all_active(self, registry, mock_store, monkeypatch):
        monkeypatch.setattr("olmlx.engine.model_manager.settings.max_loaded_models", 1)
        manager = ModelManager(registry, mock_store)
        active = LoadedModel(
            name="active",
            hf_path="a/a",
            model=MagicMock(),
            tokenizer=MagicMock(),
        )
        active.active_refs = 1
        manager._loaded["active"] = active
        with pytest.raises(RuntimeError, match="All loaded models are in use"):
            manager._evict_lru_if_needed()

    def test_skips_gc_when_pending_cleanup(self, registry, mock_store, monkeypatch):
        monkeypatch.setattr("olmlx.engine.model_manager.settings.max_loaded_models", 1)
        manager = ModelManager(registry, mock_store)
        old = LoadedModel(
            name="old",
            hf_path="o/o",
            model=MagicMock(),
            tokenizer=MagicMock(),
            loaded_at=time.time() - 100,
        )
        manager._loaded["old"] = old
        # Simulate pending cleanup — use a truthy sentinel (dict only checks key presence)
        manager._pending_cleanups["other"] = True
        with patch("olmlx.engine.model_manager.gc.collect") as mock_gc:
            manager._evict_lru_if_needed()
            mock_gc.assert_not_called()
        assert "old" not in manager._loaded
        del manager._pending_cleanups["other"]

    def test_nulls_speculative_decoder(self, registry, mock_store, monkeypatch):
        monkeypatch.setattr("olmlx.engine.model_manager.settings.max_loaded_models", 1)
        manager = ModelManager(registry, mock_store)
        decoder = MagicMock()
        old = LoadedModel(
            name="old",
            hf_path="o/o",
            model=MagicMock(),
            tokenizer=MagicMock(),
            speculative_decoder=decoder,
            loaded_at=time.time() - 100,
        )
        manager._loaded["old"] = old
        manager._evict_lru_if_needed()
        assert "old" not in manager._loaded

    def test_close_loaded_model_continues_on_failure(self, registry, mock_store):
        """A raising prefetcher.close() must not skip weight_store/decoder cleanup.

        Without try/finally chaining, a single resource failure during eviction
        or expiry would leak the weight store's file descriptors and leave the
        speculative decoder's GDN monkey-patch installed indefinitely.

        _close_loaded_model always raises ExceptionGroup on any error
        (single or multiple) so callers see a stable exception type.
        """
        manager = ModelManager(registry, mock_store)
        prefetcher = MagicMock()
        prefetcher.close.side_effect = RuntimeError("boom")
        weight_store = MagicMock()
        decoder = MagicMock()
        flash_model = MagicMock()
        flash_model.prefetcher = prefetcher
        lm = LoadedModel(
            name="x",
            hf_path="x/x",
            model=flash_model,
            tokenizer=MagicMock(),
            weight_store=weight_store,
            speculative_decoder=decoder,
            is_flash=True,
        )

        with pytest.raises(ExceptionGroup) as exc_info:
            manager._close_loaded_model(lm)

        # Single-failure case is still wrapped in ExceptionGroup for a
        # stable contract — see docstring on _close_loaded_model.
        assert len(exc_info.value.exceptions) == 1
        assert isinstance(exc_info.value.exceptions[0], RuntimeError)
        assert "boom" in str(exc_info.value.exceptions[0])
        # Both subsequent resources must still be released.
        weight_store.close.assert_called_once()
        decoder.close.assert_called_once()
        # LoadedModel-owned references are nulled so later code can't
        # accidentally observe a closed resource. ``lm.model.prefetcher``
        # is intentionally left alone — see helper docstring.
        assert lm.weight_store is None
        assert lm.speculative_decoder is None

    def test_close_loaded_model_surfaces_multiple_failures(self, registry, mock_store):
        """When two close() calls raise, both errors must surface.

        Python's nested-try/finally silently replaces an earlier exception
        with a later one. The per-resource try/except pattern collects all
        failures and raises an ExceptionGroup so neither failure is hidden.
        """
        manager = ModelManager(registry, mock_store)
        prefetcher = MagicMock()
        prefetcher.close.side_effect = RuntimeError("prefetcher-boom")
        weight_store = MagicMock()
        weight_store.close.side_effect = RuntimeError("weight-store-boom")
        decoder = MagicMock()
        flash_model = MagicMock()
        flash_model.prefetcher = prefetcher
        lm = LoadedModel(
            name="x",
            hf_path="x/x",
            model=flash_model,
            tokenizer=MagicMock(),
            weight_store=weight_store,
            speculative_decoder=decoder,
            is_flash=True,
        )

        with pytest.raises(ExceptionGroup) as exc_info:
            manager._close_loaded_model(lm)

        messages = [str(e) for e in exc_info.value.exceptions]
        assert any("prefetcher-boom" in m for m in messages)
        assert any("weight-store-boom" in m for m in messages)
        # Decoder still closed despite both prior failures.
        decoder.close.assert_called_once()
        # Failed close() leaves the reference alive — preserves the
        # partially-closed object for inspection / retry. Successful
        # decoder close still nulls its field.
        assert lm.weight_store is weight_store
        assert lm.speculative_decoder is None

    def test_evict_absorbs_close_failure(
        self, registry, mock_store, monkeypatch, caplog
    ):
        """LRU eviction must not propagate close() failures.

        A stuck prefetcher would otherwise permanently block all future
        model loads. The eviction site logs the error and continues.
        """
        monkeypatch.setattr("olmlx.engine.model_manager.settings.max_loaded_models", 1)
        manager = ModelManager(registry, mock_store)
        prefetcher = MagicMock()
        prefetcher.close.side_effect = RuntimeError("stuck-prefetcher")
        flash_model = MagicMock()
        flash_model.prefetcher = prefetcher
        old = LoadedModel(
            name="old",
            hf_path="o/o",
            model=flash_model,
            tokenizer=MagicMock(),
            is_flash=True,
            loaded_at=time.time() - 100,
        )
        manager._loaded["old"] = old

        with caplog.at_level(logging.ERROR, logger="olmlx.engine.model_manager"):
            manager._evict_lru_if_needed()  # must not raise

        assert "old" not in manager._loaded
        # _close_loaded_model logs per-resource; eviction site absorbs silently.
        assert any(
            "Error closing prefetcher for old" in r.message for r in caplog.records
        )

    def test_closes_flash_resources_on_evict(self, registry, mock_store, monkeypatch):
        """LRU eviction of a Flash model must close prefetcher + weight_store.

        Otherwise ThreadPoolExecutor workers and per-layer file descriptors
        leak for every evicted Flash model (issue #178).
        """
        monkeypatch.setattr("olmlx.engine.model_manager.settings.max_loaded_models", 1)
        manager = ModelManager(registry, mock_store)
        # Wire both resources through the same ``parent`` MagicMock so the
        # cross-resource ordering assertion below has a single ordered call
        # log to inspect. See the matching test in TestExpiryChecker.
        parent = MagicMock()
        prefetcher = parent.prefetcher
        weight_store = parent.weight_store
        flash_model = MagicMock()
        flash_model.prefetcher = prefetcher
        old = LoadedModel(
            name="old",
            hf_path="o/o",
            model=flash_model,
            tokenizer=MagicMock(),
            weight_store=weight_store,
            is_flash=True,
            loaded_at=time.time() - 100,
        )
        manager._loaded["old"] = old
        manager._evict_lru_if_needed()
        prefetcher.close.assert_called_once()
        weight_store.close.assert_called_once()
        # Order matters: prefetcher tasks submit into the weight store's pool,
        # so the prefetcher must shut down before the weight store.
        call_names = [c[0] for c in parent.mock_calls]
        assert call_names.index("prefetcher.close") < call_names.index(
            "weight_store.close"
        )


class TestSpeculativeLoading:
    """Tests for standalone speculative decoder loading in _load_model."""

    def test_load_model_creates_speculative_decoder(self, monkeypatch):
        """When speculative is enabled, _load_model should return a SpeculativeDecoder."""
        from olmlx.config import ExperimentalSettings
        from olmlx.engine.speculative import SpeculativeDecoder

        target_model = MagicMock()
        target_model.args.vocab_size = 32000
        target_tokenizer = MagicMock()
        caps = TemplateCaps()

        draft_model = MagicMock()
        draft_model.args.vocab_size = 32000

        registry = MagicMock()
        store = MagicMock()
        store.ensure_downloaded.return_value = Path("/tmp/test-draft")

        manager = ModelManager(registry, store)
        monkeypatch.setattr(
            manager,
            "_try_lm_then_vlm",
            lambda *a, **kw: (target_model, target_tokenizer, False, caps),
        )
        monkeypatch.setattr(manager, "_detect_model_kind", lambda *a: "text")
        monkeypatch.setattr(manager, "_is_flash_enabled", lambda *a: False)
        monkeypatch.setattr(manager, "_is_flash_moe_enabled", lambda *a: False)

        mock_mlx_lm = MagicMock()
        mock_mlx_lm.load.return_value = (draft_model, MagicMock())
        monkeypatch.setitem(__import__("sys").modules, "mlx_lm", mock_mlx_lm)

        model_exp = ExperimentalSettings(_env_file=None)
        spec_config = SpeculativeConfig(True, "test/draft-model", 5)

        model, tok, is_vlm, caps_out, decoder = manager._load_model(
            "test/target-model", model_exp=model_exp, spec_config=spec_config
        )

        assert isinstance(decoder, SpeculativeDecoder)
        assert decoder._lambda == 5
        assert model is target_model

    def test_load_model_rejects_vocab_mismatch(self, monkeypatch):
        """Should raise ValueError when draft/target vocab sizes differ."""
        from olmlx.config import ExperimentalSettings

        target_model = MagicMock()
        target_model.args.vocab_size = 32000

        draft_model = MagicMock()
        draft_model.args.vocab_size = 64000

        registry = MagicMock()
        store = MagicMock()
        store.ensure_downloaded.return_value = Path("/tmp/test-draft")

        manager = ModelManager(registry, store)
        monkeypatch.setattr(
            manager,
            "_try_lm_then_vlm",
            lambda *a, **kw: (target_model, MagicMock(), False, TemplateCaps()),
        )
        monkeypatch.setattr(manager, "_detect_model_kind", lambda *a: "text")
        monkeypatch.setattr(manager, "_is_flash_enabled", lambda *a: False)
        monkeypatch.setattr(manager, "_is_flash_moe_enabled", lambda *a: False)

        mock_mlx_lm = MagicMock()
        mock_mlx_lm.load.return_value = (draft_model, MagicMock())
        monkeypatch.setitem(__import__("sys").modules, "mlx_lm", mock_mlx_lm)

        model_exp = ExperimentalSettings(_env_file=None)
        spec_config = SpeculativeConfig(True, "test/draft-model", 4)

        with pytest.raises(ValueError, match="vocab_size"):
            manager._load_model(
                "test/target-model", model_exp=model_exp, spec_config=spec_config
            )

    def test_load_speculative_decoder_rejects_disabled_config(self, monkeypatch):
        """``_load_speculative_decoder`` keeps the invariant that callers gate
        on ``spec_config.enabled`` before invoking it. Direct invocation with
        ``enabled=False`` must raise — assert is elided under ``python -O``,
        so the guard is a real ``RuntimeError``."""
        registry = MagicMock()
        store = MagicMock()
        manager = ModelManager(registry, store)

        spec_config = SpeculativeConfig(False, "test/draft", 4)
        with pytest.raises(RuntimeError, match="spec_config.enabled=False"):
            manager._load_speculative_decoder(MagicMock(), "test/target", spec_config)

    def test_load_model_requires_draft_model_path(self, monkeypatch):
        """Should raise ValueError when speculative is enabled but no draft model."""
        from olmlx.config import ExperimentalSettings

        registry = MagicMock()
        store = MagicMock()

        manager = ModelManager(registry, store)
        monkeypatch.setattr(
            manager,
            "_try_lm_then_vlm",
            lambda *a, **kw: (MagicMock(), MagicMock(), False, TemplateCaps()),
        )
        monkeypatch.setattr(manager, "_detect_model_kind", lambda *a: "text")
        monkeypatch.setattr(manager, "_is_flash_enabled", lambda *a: False)
        monkeypatch.setattr(manager, "_is_flash_moe_enabled", lambda *a: False)

        model_exp = ExperimentalSettings(_env_file=None)
        spec_config = SpeculativeConfig(True, None, 4)

        with pytest.raises(ValueError, match="speculative_draft_model"):
            manager._load_model(
                "test/target-model", model_exp=model_exp, spec_config=spec_config
            )

    def test_flash_path_warns_when_standalone_speculative_set(
        self, monkeypatch, caplog
    ):
        """A Flash model combined with the standalone speculative flag must
        log a warning so the user notices the redirect to flash_speculative."""
        import logging

        from olmlx.config import ExperimentalSettings

        registry = MagicMock()
        store = MagicMock()
        store.ensure_downloaded.return_value = Path("/tmp/test-flash")

        manager = ModelManager(registry, store)
        monkeypatch.setattr(manager, "_is_flash_moe_enabled", lambda *a: False)
        monkeypatch.setattr(manager, "_is_flash_enabled", lambda *a: True)
        monkeypatch.setattr(
            manager, "_flash_dir", lambda hf_path: Path("/tmp/test-flash/flash")
        )
        sentinel = (object(), object(), False, TemplateCaps(), object())
        monkeypatch.setattr(
            manager,
            "_load_flash_model",
            lambda hf_path, load_path, flash_dir, *, model_exp: sentinel,
        )

        model_exp = ExperimentalSettings(_env_file=None)
        spec_config = SpeculativeConfig(True, "test/draft", 4)

        with caplog.at_level(logging.WARNING, logger="olmlx.engine.model_manager"):
            result = manager._load_model(
                "test/flash-model", model_exp=model_exp, spec_config=spec_config
            )
        # Flash path returns its own tuple unchanged.
        assert result is sentinel
        assert "OLMLX_SPECULATIVE" in caplog.text
        assert "Flash" in caplog.text

    def test_flash_moe_path_supports_classic_speculative(self, monkeypatch, caplog):
        """Flash-MoE + classic speculative should load the decoder (not drop it)."""
        import logging

        from olmlx.config import ExperimentalSettings

        registry = MagicMock()
        store = MagicMock()
        store.ensure_downloaded.return_value = Path("/tmp/test-moe")

        manager = ModelManager(registry, store)
        monkeypatch.setattr(manager, "_is_flash_moe_enabled", lambda *a: True)
        monkeypatch.setattr(
            manager, "_flash_moe_dir", lambda hf_path: Path("/tmp/test-moe/flash_moe")
        )
        sentinel_load = (object(), object(), False, TemplateCaps())
        monkeypatch.setattr(
            manager,
            "_load_flash_moe_model",
            lambda hf_path, load_path, flash_moe_dir, *, model_exp: sentinel_load,
        )
        sentinel_decoder = object()
        monkeypatch.setattr(
            manager,
            "_load_speculative_decoder",
            lambda model, hf_path, spec_config, *, is_vlm=False: sentinel_decoder,
        )

        model_exp = ExperimentalSettings(_env_file=None)
        spec_config = SpeculativeConfig(True, "test/draft", 4)

        with caplog.at_level(logging.WARNING, logger="olmlx.engine.model_manager"):
            model, tokenizer, is_vlm, caps, decoder = manager._load_model(
                "test/moe-model", model_exp=model_exp, spec_config=spec_config
            )
        # Flash-MoE now supports classic speculative; decoder is loaded.
        assert (model, tokenizer, is_vlm, caps) == sentinel_load
        assert decoder is sentinel_decoder
        assert "OLMLX_SPECULATIVE" not in caplog.text

    def test_flash_moe_path_rejects_dflash(self, monkeypatch):
        """Flash-MoE + dflash should raise ValueError."""
        from olmlx.config import ExperimentalSettings

        registry = MagicMock()
        store = MagicMock()
        store.ensure_downloaded.return_value = Path("/tmp/test-moe")

        manager = ModelManager(registry, store)
        monkeypatch.setattr(manager, "_is_flash_moe_enabled", lambda *a: True)
        monkeypatch.setattr(
            manager, "_flash_moe_dir", lambda hf_path: Path("/tmp/test-moe/flash_moe")
        )

        model_exp = ExperimentalSettings(_env_file=None)
        spec_config = SpeculativeConfig(True, "test/draft", 4, strategy="dflash")

        with pytest.raises(ValueError, match="dflash.*not supported on Flash-MoE"):
            manager._load_model(
                "test/moe-model", model_exp=model_exp, spec_config=spec_config
            )

    def test_flash_moe_path_rejects_flash_speculative(self, monkeypatch):
        """Flash-MoE + flash_speculative should raise ValueError."""
        from olmlx.config import ExperimentalSettings

        registry = MagicMock()
        store = MagicMock()
        store.ensure_downloaded.return_value = Path("/tmp/test-moe")

        manager = ModelManager(registry, store)
        monkeypatch.setattr(manager, "_is_flash_moe_enabled", lambda *a: True)
        monkeypatch.setattr(
            manager, "_flash_moe_dir", lambda hf_path: Path("/tmp/test-moe/flash_moe")
        )

        model_exp = ExperimentalSettings(_env_file=None, flash_speculative=True)
        spec_config = SpeculativeConfig(False, None, 4)

        with pytest.raises(
            ValueError, match="flash_speculative.*not supported on Flash-MoE"
        ):
            manager._load_model(
                "test/moe-model", model_exp=model_exp, spec_config=spec_config
            )


class TestDFlashLoading:
    """Tests for dflash decoder loading in _load_model."""

    def test_load_model_requires_dflash_draft_model(self, monkeypatch):
        """speculative_strategy='dflash' without a draft model should raise."""
        from olmlx.config import ExperimentalSettings
        from olmlx.engine.registry import SpeculativeConfig

        registry = MagicMock()
        store = MagicMock()

        manager = ModelManager(registry, store)
        monkeypatch.setattr(
            manager,
            "_try_lm_then_vlm",
            lambda *a, **kw: (MagicMock(), MagicMock(), False, TemplateCaps()),
        )
        monkeypatch.setattr(manager, "_detect_model_kind", lambda *a: "text")
        monkeypatch.setattr(manager, "_is_flash_enabled", lambda *a: False)
        monkeypatch.setattr(manager, "_is_flash_moe_enabled", lambda *a: False)

        model_exp = ExperimentalSettings(_env_file=None)
        spec_config = SpeculativeConfig(
            enabled=True,
            draft_model=None,
            num_tokens=4,
            strategy="dflash",
        )

        with pytest.raises(ValueError, match="speculative_draft_model"):
            manager._load_model(
                "test/target-model",
                model_exp=model_exp,
                spec_config=spec_config,
            )

    def test_load_model_creates_dflash_decoder(self, monkeypatch):
        """speculative_strategy='dflash' should route through _load_dflash_decoder."""
        from olmlx.config import ExperimentalSettings
        from olmlx.engine.dflash.decoder import DFlashDecoder
        from olmlx.engine.registry import SpeculativeConfig

        target_model = MagicMock()
        target_model.args.vocab_size = 32000

        registry = MagicMock()
        store = MagicMock()
        store.ensure_downloaded.return_value = Path("/tmp/test-dflash-draft")
        store.local_path.return_value = Path("/tmp/test-target")

        manager = ModelManager(registry, store)
        monkeypatch.setattr(
            manager,
            "_try_lm_then_vlm",
            lambda *a, **kw: (target_model, MagicMock(), False, TemplateCaps()),
        )
        monkeypatch.setattr(manager, "_detect_model_kind", lambda *a: "text")
        monkeypatch.setattr(manager, "_is_flash_enabled", lambda *a: False)
        monkeypatch.setattr(manager, "_is_flash_moe_enabled", lambda *a: False)

        # Mock _load_dflash_decoder to verify it's called
        mock_decoder = MagicMock(spec=DFlashDecoder)
        monkeypatch.setattr(
            manager,
            "_load_dflash_decoder",
            lambda *a, **kw: mock_decoder,
        )

        model_exp = ExperimentalSettings(_env_file=None)
        spec_config = SpeculativeConfig(
            enabled=True,
            draft_model="test/dflash-draft",
            num_tokens=4,
            strategy="dflash",
        )

        model, tok, is_vlm, caps_out, decoder = manager._load_model(
            "test/target-model",
            model_exp=model_exp,
            spec_config=spec_config,
        )

        assert decoder is mock_decoder


class TestEagleLoading:
    """Tests for ``_load_eagle_decoder`` schema validation and the
    decoder-construction path.

    Strategy mirrors ``TestDFlashLoading``: mock the heavy lifting
    (``_try_lm_then_vlm``, kind detection, flash gates) and stub
    ``_load_eagle_decoder`` itself when we want to verify routing.
    For schema-validation paths we drive ``_load_eagle_decoder``
    directly with a synthetic draft directory.
    """

    def _make_target_with(self, vocab_size: int, hidden_size: int) -> Any:
        """Build a fake target whose ``.args`` exposes both fields the
        loader walks for cross-checks. Uses a plain object rather than
        MagicMock so ``getattr(args, ...)`` returns None for absent
        fields instead of an auto-generated child mock."""
        target = MagicMock()

        class _Args:
            pass

        target.args = _Args()
        target.args.vocab_size = vocab_size
        target.args.hidden_size = hidden_size
        # Strip the chained-attr search paths the loader walks
        # (``.model``, ``.language_model``) so the first match wins.
        target.model = None
        target.language_model = None
        return target

    def _write_eagle_draft_dir(
        self,
        tmp_path: Path,
        *,
        vocab_size: int = 64,
        hidden_size: int = 16,
        block_size: int = 4,
        target_layer_id: int | None = 2,
        omit_eagle_config: bool = False,
    ) -> Path:
        """Write a minimal EAGLE draft directory (config + 1 weight
        shard) that ``_load_eagle_decoder`` can parse. Weight tensors
        are zero-filled — we only test the loader's metadata path,
        not real inference."""
        import mlx.core as mx

        draft_dir = tmp_path / "eagle_draft"
        draft_dir.mkdir()
        cfg: dict[str, Any] = {
            "hidden_size": hidden_size,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": hidden_size // 2,
            "intermediate_size": hidden_size * 2,
            "vocab_size": vocab_size,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "max_position_embeddings": 512,
        }
        if not omit_eagle_config:
            eagle_block: dict[str, Any] = {"block_size": block_size}
            if target_layer_id is not None:
                eagle_block["target_layer_id"] = target_layer_id
            cfg["eagle_config"] = eagle_block
        (draft_dir / "config.json").write_text(json.dumps(cfg))

        from olmlx.engine.eagle.draft_model import EagleConfig, EagleDraftModel

        m = EagleDraftModel(
            EagleConfig(
                hidden_size=hidden_size,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=hidden_size // 2,
                intermediate_size=hidden_size * 2,
                vocab_size=vocab_size,
                rms_norm_eps=1e-6,
                rope_theta=10000.0,
                max_position_embeddings=512,
                block_size=block_size,
            )
        )
        import mlx.utils as mx_utils

        weights = dict(mx_utils.tree_flatten(m.parameters()))
        weights = {
            k: v
            for k, v in weights.items()
            if not k.startswith("embed_tokens.") and not k.startswith("lm_head.")
        }
        mx.save_safetensors(
            str(draft_dir / "model-00001-of-00001.safetensors"), weights
        )
        return draft_dir

    def test_rejects_missing_eagle_config_block(self, tmp_path, registry, mock_store):
        from olmlx.engine.registry import SpeculativeConfig

        draft_dir = self._write_eagle_draft_dir(tmp_path, omit_eagle_config=True)
        manager = ModelManager(registry, mock_store)
        target = self._make_target_with(vocab_size=64, hidden_size=16)
        spec = SpeculativeConfig(
            enabled=True, draft_model=str(draft_dir), num_tokens=4, strategy="eagle"
        )
        with pytest.raises(ValueError, match="eagle_config"):
            manager._load_eagle_decoder(target, spec)

    def test_rejects_vocab_mismatch(self, tmp_path, registry, mock_store):
        from olmlx.engine.registry import SpeculativeConfig

        draft_dir = self._write_eagle_draft_dir(tmp_path, vocab_size=64)
        manager = ModelManager(registry, mock_store)
        # Target's vocab is 128, draft's is 64 — mismatch.
        target = self._make_target_with(vocab_size=128, hidden_size=16)
        spec = SpeculativeConfig(
            enabled=True, draft_model=str(draft_dir), num_tokens=4, strategy="eagle"
        )
        with pytest.raises(ValueError, match="vocab_size"):
            manager._load_eagle_decoder(target, spec)

    def test_rejects_hidden_size_mismatch(self, tmp_path, registry, mock_store):
        """Cross-target ``hidden_size`` mismatch must be caught at load
        time, not at the first prefill — the latter surfaces as a
        cryptic shape error inside ``input_proj``."""
        from olmlx.engine.registry import SpeculativeConfig

        draft_dir = self._write_eagle_draft_dir(tmp_path, hidden_size=16)
        manager = ModelManager(registry, mock_store)
        target = self._make_target_with(vocab_size=64, hidden_size=32)
        spec = SpeculativeConfig(
            enabled=True, draft_model=str(draft_dir), num_tokens=4, strategy="eagle"
        )
        with pytest.raises(ValueError, match="hidden_size"):
            manager._load_eagle_decoder(target, spec)

    def test_constructs_decoder_with_block_size_override(
        self, tmp_path, registry, mock_store
    ):
        """``spec_config.num_tokens`` overrides the saved ``block_size``.
        This is how operators tune block_size at the CLI without
        retraining."""
        from olmlx.engine.eagle.decoder import EagleDecoder
        from olmlx.engine.registry import SpeculativeConfig

        draft_dir = self._write_eagle_draft_dir(
            tmp_path, vocab_size=64, hidden_size=16, block_size=4
        )
        manager = ModelManager(registry, mock_store)
        # Build a real synthetic target so EagleDecoder's
        # ``_get_layers(target_model)`` works.
        from tests.test_dflash import _Target

        target = _Target(vocab_size=64, hidden_size=16, num_layers=4)
        spec = SpeculativeConfig(
            enabled=True,
            draft_model=str(draft_dir),
            num_tokens=2,  # override the saved 4
            strategy="eagle",
        )
        decoder = manager._load_eagle_decoder(target, spec)
        assert isinstance(decoder, EagleDecoder)
        assert decoder._block_size == 2
        # target_layer_id from saved config (2) should have been threaded.
        assert decoder._target_layer_id == 2

    def test_warns_on_missing_target_layer_id(
        self, tmp_path, registry, mock_store, caplog
    ):
        """Pre-fix checkpoints have no ``target_layer_id``. The loader
        must emit a ``logger.warning`` so the operator gets nudged to
        retrain rather than silently shipping a degraded draft."""
        import logging

        from olmlx.engine.registry import SpeculativeConfig

        draft_dir = self._write_eagle_draft_dir(
            tmp_path, vocab_size=64, hidden_size=16, target_layer_id=None
        )
        manager = ModelManager(registry, mock_store)
        from tests.test_dflash import _Target

        target = _Target(vocab_size=64, hidden_size=16, num_layers=4)
        spec = SpeculativeConfig(
            enabled=True, draft_model=str(draft_dir), num_tokens=2, strategy="eagle"
        )
        with caplog.at_level(logging.WARNING, logger="olmlx.engine.model_manager"):
            decoder = manager._load_eagle_decoder(target, spec)
        assert any("target_layer_id" in r.message for r in caplog.records)
        # Falls back to last layer.
        assert decoder._target_layer_id == 3  # last index of 4 layers
