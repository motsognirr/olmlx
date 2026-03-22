"""Tests for flash inference integration with olmlx config and ModelManager."""

from unittest.mock import MagicMock, patch

import pytest

from olmlx.config import ExperimentalSettings


class TestFlashSettings:
    def test_defaults(self):
        with patch.dict("os.environ", {}, clear=False):
            s = ExperimentalSettings()
            assert s.flash is False
            assert s.flash_sparsity_threshold == 0.5
            assert s.flash_min_active_neurons == 128
            assert s.flash_max_active_neurons is None
            assert s.flash_window_size == 5
            assert s.flash_io_threads == 32
            assert s.flash_cache_budget_neurons == 1024
            assert s.flash_predictor_rank == 128

    def test_env_override(self):
        env = {
            "OLMLX_EXPERIMENTAL_FLASH": "true",
            "OLMLX_EXPERIMENTAL_FLASH_SPARSITY_THRESHOLD": "0.3",
            "OLMLX_EXPERIMENTAL_FLASH_MIN_ACTIVE_NEURONS": "64",
            "OLMLX_EXPERIMENTAL_FLASH_WINDOW_SIZE": "10",
        }
        with patch.dict("os.environ", env, clear=False):
            s = ExperimentalSettings()
            assert s.flash is True
            assert s.flash_sparsity_threshold == 0.3
            assert s.flash_min_active_neurons == 64
            assert s.flash_window_size == 10


class TestModelManagerFlashDetection:
    """These tests import ModelManager which pulls in mlx_lm → transformers.

    They may be slow on first import. Mark with slow marker if needed.
    """

    @pytest.fixture(autouse=True)
    def _import_mm(self):
        """Lazy import to avoid import-time hang when running other tests."""
        from olmlx.engine.model_manager import LoadedModel, ModelManager

        self.ModelManager = ModelManager
        self.LoadedModel = LoadedModel

    def test_flash_dir_returns_none_without_store(self):
        registry = MagicMock()
        manager = self.ModelManager(registry, store=None)
        assert manager._flash_dir("some/model") is None

    def test_flash_dir_returns_none_when_not_prepared(self, tmp_path):
        registry = MagicMock()
        store = MagicMock()
        store.local_path.return_value = tmp_path / "model"
        (tmp_path / "model").mkdir()

        manager = self.ModelManager(registry, store)
        assert manager._flash_dir("some/model") is None

    def test_flash_dir_returns_path_when_prepared(self, tmp_path):
        registry = MagicMock()
        store = MagicMock()
        model_path = tmp_path / "model"
        model_path.mkdir()
        flash_path = model_path / "flash"
        flash_path.mkdir()
        (flash_path / "flash_layout.json").write_text("{}")

        store.local_path.return_value = model_path
        manager = self.ModelManager(registry, store)
        assert manager._flash_dir("some/model") == flash_path

    def test_is_flash_enabled_reads_experimental(self):
        registry = MagicMock()
        manager = self.ModelManager(registry, store=None)

        with patch("olmlx.config.experimental") as mock_cfg_exp:
            mock_cfg_exp.flash = True
            assert manager._is_flash_enabled() is True

    def test_loaded_model_is_flash_field(self):
        lm = self.LoadedModel(
            name="test",
            hf_path="test/model",
            model=MagicMock(),
            tokenizer=MagicMock(),
            is_flash=True,
        )
        assert lm.is_flash is True

    def test_loaded_model_is_flash_defaults_false(self):
        lm = self.LoadedModel(
            name="test",
            hf_path="test/model",
            model=MagicMock(),
            tokenizer=MagicMock(),
        )
        assert lm.is_flash is False
