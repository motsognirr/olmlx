"""Tests for flash CLI subcommands."""

import json
from unittest.mock import MagicMock, patch


from olmlx.cli import build_parser


class TestFlashCLIParsing:
    def test_flash_prepare_parses(self):
        parser = build_parser()
        args = parser.parse_args(["flash", "prepare", "Qwen/Qwen3-8B"])
        assert args.command == "flash"
        assert args.flash_command == "prepare"
        assert args.model == "Qwen/Qwen3-8B"
        assert args.rank == 128
        assert args.samples == 256
        assert args.threshold == 0.01
        assert args.epochs == 5

    def test_flash_prepare_custom_args(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "flash",
                "prepare",
                "Qwen/Qwen3-8B",
                "--rank",
                "64",
                "--samples",
                "128",
                "--threshold",
                "0.05",
                "--epochs",
                "10",
            ]
        )
        assert args.rank == 64
        assert args.samples == 128
        assert args.threshold == 0.05
        assert args.epochs == 10

    def test_flash_info_parses(self):
        parser = build_parser()
        args = parser.parse_args(["flash", "info", "Qwen/Qwen3-8B"])
        assert args.command == "flash"
        assert args.flash_command == "info"
        assert args.model == "Qwen/Qwen3-8B"


class TestFlashInfo:
    def test_info_not_prepared(self, tmp_path, capsys):
        from olmlx.cli import cmd_flash_info

        args = MagicMock()
        args.model = "test/model"

        (tmp_path / "model").mkdir()

        # Mock the local imports inside cmd_flash_info
        mock_registry_cls = MagicMock()
        mock_registry = mock_registry_cls.return_value
        mock_registry.resolve.return_value = "test/model"

        mock_store_cls = MagicMock()
        mock_store = mock_store_cls.return_value
        mock_store.local_path.return_value = tmp_path / "model"

        with (
            patch("olmlx.cli.ensure_config"),
            patch.dict(
                "sys.modules",
                {},
            ),
        ):
            # Patch at the import point
            import olmlx.engine.registry as reg_mod
            import olmlx.models.store as store_mod

            orig_reg = reg_mod.ModelRegistry
            orig_store = store_mod.ModelStore
            reg_mod.ModelRegistry = mock_registry_cls
            store_mod.ModelStore = mock_store_cls
            try:
                cmd_flash_info(args)
            finally:
                reg_mod.ModelRegistry = orig_reg
                store_mod.ModelStore = orig_store

        captured = capsys.readouterr()
        assert "not been prepared" in captured.out

    def test_info_prepared(self, tmp_path, capsys):
        from olmlx.cli import cmd_flash_info

        args = MagicMock()
        args.model = "test/model"

        # Create flash dir
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        flash_dir = model_dir / "flash"
        flash_dir.mkdir()
        (flash_dir / "flash_config.json").write_text(
            json.dumps(
                {
                    "hidden_size": 4096,
                    "intermediate_size": 14336,
                    "num_layers": 32,
                    "predictor_rank": 128,
                    "num_calibration_samples": 256,
                    "prepared_at": "2026-03-22T00:00:00Z",
                }
            )
        )
        (flash_dir / "layer_00.flashweights").write_bytes(b"\x00" * 100)

        mock_registry_cls = MagicMock()
        mock_registry = mock_registry_cls.return_value
        mock_registry.resolve.return_value = "test/model"

        mock_store_cls = MagicMock()
        mock_store = mock_store_cls.return_value
        mock_store.local_path.return_value = model_dir

        with patch("olmlx.cli.ensure_config"):
            import olmlx.engine.registry as reg_mod
            import olmlx.models.store as store_mod

            orig_reg = reg_mod.ModelRegistry
            orig_store = store_mod.ModelStore
            reg_mod.ModelRegistry = mock_registry_cls
            store_mod.ModelStore = mock_store_cls
            try:
                cmd_flash_info(args)
            finally:
                reg_mod.ModelRegistry = orig_reg
                store_mod.ModelStore = orig_store

        captured = capsys.readouterr()
        assert "prepared" in captured.out
        assert "4096" in captured.out
        assert "14336" in captured.out
        assert "32" in captured.out
