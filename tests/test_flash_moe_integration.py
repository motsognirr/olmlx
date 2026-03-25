"""Integration tests for Flash-MoE: config, model_manager, CLI, prepare."""

import json
from unittest.mock import patch


from tests.test_flash_moe_bundler import _make_synthetic_moe_weights


class TestFlashMoeConfig:
    def test_flash_moe_defaults(self):
        from olmlx.config import ExperimentalSettings

        s = ExperimentalSettings()
        assert s.flash_moe is False
        assert s.flash_moe_cache_budget_experts == 48
        assert s.flash_moe_io_threads == 32

    def test_flash_moe_env_override(self):
        from olmlx.config import ExperimentalSettings

        with patch.dict(
            "os.environ",
            {
                "OLMLX_EXPERIMENTAL_FLASH_MOE": "true",
                "OLMLX_EXPERIMENTAL_FLASH_MOE_CACHE_BUDGET_EXPERTS": "64",
                "OLMLX_EXPERIMENTAL_FLASH_MOE_IO_THREADS": "16",
            },
        ):
            s = ExperimentalSettings()
            assert s.flash_moe is True
            assert s.flash_moe_cache_budget_experts == 64
            assert s.flash_moe_io_threads == 16


class TestIsMoeModel:
    def test_moe_model_detected(self, tmp_path):
        from olmlx.engine.flash.moe_prepare import is_moe_model

        config = {"n_routed_experts": 8, "hidden_size": 64}
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert is_moe_model(tmp_path) is True

    def test_dense_model_not_detected(self, tmp_path):
        from olmlx.engine.flash.moe_prepare import is_moe_model

        config = {"hidden_size": 64, "intermediate_size": 256}
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert is_moe_model(tmp_path) is False

    def test_wrapper_model_detected(self, tmp_path):
        """Models like Kimi-K2.5 wrap the text config inside text_config."""
        from olmlx.engine.flash.moe_prepare import is_moe_model

        config = {
            "model_type": "kimi_k25",
            "text_config": {"n_routed_experts": 384, "hidden_size": 7168},
        }
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert is_moe_model(tmp_path) is True

    def test_qwen3_moe_detected(self, tmp_path):
        """Qwen3-MoE uses 'num_experts' instead of 'n_routed_experts'."""
        from olmlx.engine.flash.moe_prepare import is_moe_model

        config = {"model_type": "qwen3_moe", "num_experts": 160, "hidden_size": 6144}
        (tmp_path / "config.json").write_text(json.dumps(config))
        assert is_moe_model(tmp_path) is True

    def test_missing_config(self, tmp_path):
        from olmlx.engine.flash.moe_prepare import is_moe_model

        assert is_moe_model(tmp_path) is False


class TestPrepareMoeForFlash:
    def test_full_preparation(self, tmp_path):
        """Full preparation pipeline should produce correct output files."""
        hidden, inter, experts = 64, 32, 4
        model_dir = _make_synthetic_moe_weights(hidden, inter, experts, 2, 1, tmp_path)
        output_dir = tmp_path / "flash_moe"

        from olmlx.engine.flash.moe_prepare import prepare_moe_for_flash

        result = prepare_moe_for_flash(str(model_dir), output_dir)

        assert result == output_dir
        assert (output_dir / "flash_moe_config.json").exists()
        assert (output_dir / "flash_moe_layout.json").exists()

        config = json.loads((output_dir / "flash_moe_config.json").read_text())
        assert config["num_experts"] == experts
        assert config["hidden_size"] == hidden
        assert config["num_moe_layers"] == 2
        assert config["moe_layer_indices"] == [1, 2]
        assert "prepared_at" in config


class TestModelManagerFlashMoe:
    def test_flash_moe_dir_detection(self, tmp_path):
        """ModelManager should detect flash_moe directory."""
        from olmlx.engine.model_manager import ModelManager

        # Create a fake model store structure
        model_dir = tmp_path / "models" / "test-model"
        flash_moe_dir = model_dir / "flash_moe"
        flash_moe_dir.mkdir(parents=True)
        (flash_moe_dir / "flash_moe_layout.json").write_text("{}")

        from unittest.mock import MagicMock

        store = MagicMock()
        store.local_path.return_value = model_dir

        mgr = ModelManager.__new__(ModelManager)
        mgr.store = store

        result = mgr._flash_moe_dir("test-model")
        assert result == flash_moe_dir

    def test_flash_moe_dir_not_found(self, tmp_path):
        """Should return None when no flash_moe directory exists."""
        from olmlx.engine.model_manager import ModelManager
        from unittest.mock import MagicMock

        store = MagicMock()
        store.local_path.return_value = tmp_path / "nonexistent"

        mgr = ModelManager.__new__(ModelManager)
        mgr.store = store

        assert mgr._flash_moe_dir("test-model") is None


class TestLoadedModelField:
    def test_is_flash_moe_field_exists(self):
        """LoadedModel should have is_flash_moe field."""
        from olmlx.engine.model_manager import LoadedModel
        import dataclasses

        fields = {f.name for f in dataclasses.fields(LoadedModel)}
        assert "is_flash_moe" in fields
