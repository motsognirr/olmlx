"""Wiring tests: dispatch, calibration discovery, serialization guard (#377)."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


class TestPromptCacheDispatch:
    def test_shard_method_dispatches_to_make_shard_cache(self):
        from olmlx.engine import inference

        lm = MagicMock()
        lm.kv_cache_quant = "shard:4"
        lm.is_vlm = False
        lm.shard_calibration_dir = Path("/tmp/fake-shard")
        with patch(
            "olmlx.engine.shardquant_cache.make_shard_cache",
            return_value=["sentinel"],
        ) as mk:
            result = inference._make_prompt_cache_for_lm(lm)
        assert result == ["sentinel"]
        mk.assert_called_once_with(lm.model, Path("/tmp/fake-shard"), bits=4)

    def test_spectral_dispatch_unchanged(self):
        from olmlx.engine import inference

        lm = MagicMock()
        lm.kv_cache_quant = "spectral:4"
        lm.is_vlm = False
        with patch(
            "olmlx.engine.spectralquant_cache.make_spectral_cache",
            return_value=["spectral-sentinel"],
        ):
            assert inference._make_prompt_cache_for_lm(lm) == ["spectral-sentinel"]


class TestSerializableGuard:
    def test_shard_cache_blocks_disk_save(self):
        from olmlx.engine.model_manager import _is_serializable_cache
        from olmlx.engine.shardquant_cache import ShardKVCache

        cache = ShardKVCache.__new__(ShardKVCache)  # isinstance check only
        assert _is_serializable_cache([cache]) is False

    def test_plain_cache_still_serializable(self):
        from mlx_lm.models.cache import KVCache

        from olmlx.engine.model_manager import _is_serializable_cache

        assert _is_serializable_cache([KVCache()]) is True


def _manager_with_store(tmp_path):
    from olmlx.engine.model_manager import ModelManager

    mgr = ModelManager.__new__(ModelManager)
    store = MagicMock()
    store.local_path.return_value = tmp_path
    mgr.store = store
    return mgr


class TestFindShardDir:
    def test_none_when_not_shard(self, tmp_path):
        mgr = _manager_with_store(tmp_path)
        assert mgr._find_shard_dir("m", None) is None
        assert mgr._find_shard_dir("m", "spectral:4") is None

    def test_missing_calibration_raises_with_command(self, tmp_path):
        from olmlx.engine.model_manager import ShardCalibrationMissingError

        mgr = _manager_with_store(tmp_path)
        with pytest.raises(ShardCalibrationMissingError) as exc:
            mgr._find_shard_dir("org/model", "shard:4")
        assert "olmlx shard prepare org/model" in str(exc.value)

    def test_found_when_calibration_exists(self, tmp_path):
        shard = tmp_path / "shard"
        shard.mkdir()
        (shard / "shard_config.json").write_text(json.dumps({"meta": {"bits": 4}}))
        mgr = _manager_with_store(tmp_path)
        assert mgr._find_shard_dir("org/model", "shard:4") == shard

    def test_bits_mismatch_raises(self, tmp_path):
        from olmlx.engine.model_manager import ShardCalibrationMissingError

        shard = tmp_path / "shard"
        shard.mkdir()
        (shard / "shard_config.json").write_text(json.dumps({"meta": {"bits": 2}}))
        mgr = _manager_with_store(tmp_path)
        with pytest.raises(ShardCalibrationMissingError) as exc:
            mgr._find_shard_dir("org/model", "shard:4")
        assert "--bits 4" in str(exc.value)

    def test_shard_error_is_spectral_subclass(self):
        """Starlette resolves handlers by walking type(exc).__mro__, so the
        existing 400 handler for SpectralCalibrationMissingError catches the
        shard error too — no app.py change needed."""
        from olmlx.engine.model_manager import (
            ShardCalibrationMissingError,
            SpectralCalibrationMissingError,
        )

        assert issubclass(
            ShardCalibrationMissingError, SpectralCalibrationMissingError
        )


class TestCliShardPrepare:
    def test_shard_prepare_invokes_calibration(self, tmp_path):
        from olmlx import cli

        args = SimpleNamespace(
            model="org/model",
            samples=8,
            bits=4,
            calibration_dataset=None,
            max_tokens=1024,
        )
        store = MagicMock()
        store.registry.resolve.return_value = None
        store.ensure_downloaded.return_value = tmp_path
        with (
            patch.object(cli, "_create_store", return_value=store),
            patch.object(cli, "_configure_logging"),
            patch(
                "olmlx.engine.shardquant_calibrate.calibrate_model_shard",
                return_value=tmp_path / "shard",
            ) as cal,
        ):
            cli.cmd_shard_prepare(args)
        cal.assert_called_once_with(
            model_path=str(tmp_path),
            num_samples=8,
            calibration_dataset=None,
            bits=4,
            max_tokens_per_head=1024,
            progress_callback=cli._flash_progress,
        )

    def test_parser_accepts_shard_prepare(self):
        from olmlx.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["shard", "prepare", "org/model", "--bits", "8"])
        assert args.bits == 8
        assert args.model == "org/model"
        assert args.shard_command == "prepare"
