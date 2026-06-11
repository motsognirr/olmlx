"""Regression coverage for olmlx.cli pure helpers and validation branches.

Focus: argparse type validators, config/info printers, legacy-env mirroring,
and the distributed/dflash/eagle validation guards that exit before any model
load. Everything here is hermetic — no network, no model loads, no server.
"""

import json
import logging

import pytest

from olmlx.cli import (
    _flash_progress,
    _legacy_kv_cache_quant_in_dotenv,
    _non_empty_str,
    _positive_int,
    _show_flash_dense_info,
    _show_flash_moe_info,
    _surface_legacy_dflash_env,
    _surface_legacy_distributed_env,
    build_parser,
    cmd_bench_compare,
    cmd_bench_list,
    cmd_config_show,
    cmd_dflash_prepare,
    cmd_eagle_prepare,
    cmd_flash_info,
    validate_remote_python,
)


class TestPositiveInt:
    def test_accepts_positive(self):
        assert _positive_int("5") == 5

    def test_rejects_zero(self):
        import argparse

        with pytest.raises(argparse.ArgumentTypeError, match="must be >= 1"):
            _positive_int("0")

    def test_rejects_non_integer(self):
        import argparse

        with pytest.raises(argparse.ArgumentTypeError, match="invalid integer"):
            _positive_int("abc")


class TestNonEmptyStr:
    def test_strips_whitespace(self):
        assert _non_empty_str("  hf/path  ") == "hf/path"

    def test_rejects_empty(self):
        import argparse

        with pytest.raises(argparse.ArgumentTypeError, match="non-empty"):
            _non_empty_str("   ")


class TestValidateRemotePython:
    def test_accepts_simple_path(self):
        # No exception for a plausible interpreter path.
        validate_remote_python("/usr/bin/python3")

    def test_accepts_multi_word(self):
        validate_remote_python("uv run python")

    def test_rejects_shell_metacharacters(self):
        with pytest.raises(ValueError, match="Invalid remote_python"):
            validate_remote_python("python; rm -rf /")

    def test_rejects_command_substitution(self):
        with pytest.raises(ValueError):
            validate_remote_python("python $(whoami)")


class TestFlashProgress:
    def test_renders_bar_and_percentage(self, capsys):
        _flash_progress("loading", 0.5)
        out = capsys.readouterr().out
        assert "loading" in out
        assert "50.0%" in out
        # Half-filled bar at 50%.
        assert "█" in out and "░" in out

    def test_newline_emitted_at_completion(self, capsys):
        _flash_progress("done", 1.0)
        out = capsys.readouterr().out
        assert "100.0%" in out
        assert out.endswith("\n")


class TestConfigShow:
    def test_shows_kv_and_flash_and_distributed_sections(self, monkeypatch, capsys):
        from olmlx.config import settings as _settings

        # Drive the conditional branches: KV quant + flash + distributed.
        monkeypatch.setattr(_settings, "kv_cache_quant", "turboquant:4")
        monkeypatch.setattr(_settings, "flash", True)
        monkeypatch.setattr(_settings, "flash_max_active_neurons", 256)
        monkeypatch.setattr(_settings, "flash_memory_budget_fraction", 0.25)
        monkeypatch.setattr(_settings, "distributed", True)

        cmd_config_show(None)
        out = capsys.readouterr().out
        assert "KV cache quant:" in out
        assert "turboquant:4" in out
        assert "Flash inference:        enabled" in out
        assert "Max active neurons:" in out
        assert "256" in out
        assert "Memory budget frac:" in out
        assert "Distributed inference:" in out
        assert "Sideband port:" in out

    def test_omits_optional_sections_when_disabled(self, monkeypatch, capsys):
        from olmlx.config import settings as _settings

        monkeypatch.setattr(_settings, "kv_cache_quant", None)
        monkeypatch.setattr(_settings, "flash", False)
        monkeypatch.setattr(_settings, "distributed", False)

        cmd_config_show(None)
        out = capsys.readouterr().out
        assert "Host:" in out  # base section always prints
        assert "KV cache quant:" not in out
        assert "Flash inference:" not in out
        assert "Distributed inference:" not in out


class TestShowFlashInfo:
    def test_show_flash_moe_info(self, tmp_path, capsys):
        moe_dir = tmp_path / "flash_moe"
        moe_dir.mkdir()
        (moe_dir / "flash_moe_config.json").write_text(
            json.dumps(
                {
                    "hidden_size": 4096,
                    "intermediate_size": 11008,
                    "num_experts": 128,
                    "num_experts_per_tok": 8,
                    "num_moe_layers": 60,
                    "prepared_at": "2026-01-01",
                }
            )
        )
        (moe_dir / "layer0.flashexperts").write_bytes(b"x" * 16)
        _show_flash_moe_info("deepseek:latest", moe_dir)
        out = capsys.readouterr().out
        assert "Flash-MoE info for 'deepseek:latest'" in out
        assert "Num experts:        128" in out
        assert "Expert files:       1" in out
        # Size is well under 1GB → printed in MB.
        assert "MB" in out

    def test_show_flash_dense_info(self, tmp_path, capsys):
        flash_dir = tmp_path / "flash"
        flash_dir.mkdir()
        (flash_dir / "flash_config.json").write_text(
            json.dumps(
                {
                    "hidden_size": 2048,
                    "intermediate_size": 8192,
                    "num_layers": 28,
                    "predictor_rank": 64,
                    "num_calibration_samples": 512,
                    "prepared_at": "2026-01-01",
                }
            )
        )
        (flash_dir / "layer0.flashweights").write_bytes(b"x" * 32)
        pred_dir = flash_dir / "predictors"
        pred_dir.mkdir()
        (pred_dir / "layer0.npz").write_bytes(b"y" * 8)
        _show_flash_dense_info("qwen:latest", flash_dir)
        out = capsys.readouterr().out
        assert "Flash info for 'qwen:latest'" in out
        assert "Predictor rank:     64" in out
        assert "Weight files:       1" in out
        assert "Predictor files:    1" in out

    def test_show_flash_dense_info_missing_config(self, tmp_path, capsys):
        flash_dir = tmp_path / "flash"
        flash_dir.mkdir()
        _show_flash_dense_info("qwen:latest", flash_dir)
        out = capsys.readouterr().out
        assert "no config found" in out


class TestCmdFlashInfo:
    def test_reports_not_prepared(self, tmp_path, monkeypatch, capsys):
        """When neither flash nor flash_moe dir exists, prints the prep hint."""
        from olmlx.cli import _create_store

        store = _create_store()
        # local_path returns a dir with no flash artifacts.
        monkeypatch.setattr(store.registry, "resolve", lambda name: None)
        monkeypatch.setattr(store, "local_path", lambda hf: tmp_path / "model")
        monkeypatch.setattr("olmlx.cli._create_store", lambda: store)

        ns = type("NS", (), {"model": "missing/model"})()
        cmd_flash_info(ns)
        out = capsys.readouterr().out
        assert "has not been prepared for flash inference" in out
        assert "olmlx flash prepare missing/model" in out


class TestDflashPrepareValidation:
    def _args(self, **overrides):
        parser = build_parser()
        argv = ["dflash", "prepare", "some/model"]
        for k, v in overrides.items():
            argv += [f"--{k.replace('_', '-')}", str(v)]
        return parser.parse_args(argv)

    def test_block_size_must_be_positive(self):
        args = self._args(block_size=0)
        with pytest.raises(SystemExit, match="block-size must be >= 1"):
            cmd_dflash_prepare(args)

    def test_seq_len_too_small_for_block_size(self):
        # 2*block_size + 1 = 9 > seq_len 4 → reject before any download.
        args = self._args(block_size=4, seq_len=4)
        with pytest.raises(SystemExit, match="too small for --block-size"):
            cmd_dflash_prepare(args)

    def test_train_windows_per_step_must_be_positive(self):
        # Keep seq_len valid so the windows check is the one that fires.
        args = self._args(block_size=2, seq_len=64, train_windows_per_step=0)
        with pytest.raises(SystemExit, match="train-windows-per-step must be >= 1"):
            cmd_dflash_prepare(args)

    def test_self_generate_rejects_use_precomputed_before_download(self):
        # The function-level guard in prepare_dflash_draft fires only
        # after ensure_downloaded(); the CLI must reject the combination
        # before paying a multi-GB target download.
        parser = build_parser()
        args = parser.parse_args(
            [
                "dflash",
                "prepare",
                "some/model",
                "--self-generate",
                "--use-precomputed",
                "/tmp/shards",
            ]
        )
        with pytest.raises(SystemExit, match="mutually exclusive"):
            cmd_dflash_prepare(args)


class TestEaglePrepareValidation:
    def _args(self, **overrides):
        parser = build_parser()
        # --use-precomputed is a required argparse arg; supply it so the
        # in-function guards (block_size, etc.) are the ones exercised.
        argv = ["eagle", "prepare", "some/model", "--use-precomputed", "/tmp/shards"]
        for k, v in overrides.items():
            argv += [f"--{k.replace('_', '-')}", str(v)]
        return parser.parse_args(argv)

    def test_block_size_must_be_positive(self):
        args = self._args(block_size=0)
        with pytest.raises(SystemExit, match="block-size must be >= 1"):
            cmd_eagle_prepare(args)

    def test_use_precomputed_required_at_function_level(self):
        """The function re-checks ``use_precomputed`` even though argparse
        marks it required — exercise that guard directly so a programmatic
        caller (or a parser change) can't slip an empty value past it."""
        import argparse

        ns = argparse.Namespace(model="some/model", use_precomputed="", block_size=4)
        with pytest.raises(SystemExit, match="use-precomputed is required"):
            cmd_eagle_prepare(ns)


class TestSurfaceLegacyDflashEnv:
    def test_forwards_enable_and_strategy(self, monkeypatch, caplog):
        from olmlx.config import settings as _settings

        monkeypatch.setattr(_settings, "speculative", False)
        monkeypatch.setattr(_settings, "speculative_strategy", "classic")
        monkeypatch.setattr(_settings, "speculative_draft_model", None)
        monkeypatch.delenv("OLMLX_SPECULATIVE", raising=False)
        monkeypatch.delenv("OLMLX_SPECULATIVE_DRAFT_MODEL", raising=False)
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_DFLASH", "true")
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_DFLASH_DRAFT_MODEL", "draft/dflash")

        with caplog.at_level(logging.WARNING, logger="olmlx.cli"):
            _surface_legacy_dflash_env()

        assert _settings.speculative is True
        assert _settings.speculative_strategy == "dflash"
        assert _settings.speculative_draft_model == "draft/dflash"
        assert "Deprecated env vars detected" in caplog.text

    def test_noop_when_unset(self, monkeypatch, caplog):
        from olmlx.config import settings as _settings

        monkeypatch.setattr(_settings, "speculative", False)
        monkeypatch.setattr(_settings, "speculative_strategy", "classic")
        for v in (
            "OLMLX_EXPERIMENTAL_DFLASH",
            "OLMLX_EXPERIMENTAL_DFLASH_DRAFT_MODEL",
            "OLMLX_EXPERIMENTAL_DFLASH_BLOCK_SIZE",
        ):
            monkeypatch.delenv(v, raising=False)

        with caplog.at_level(logging.WARNING, logger="olmlx.cli"):
            _surface_legacy_dflash_env()

        assert _settings.speculative is False
        assert _settings.speculative_strategy == "classic"
        assert "Deprecated env vars detected" not in caplog.text

    def test_block_size_forwarded_as_int(self, monkeypatch):
        from olmlx.config import settings as _settings

        monkeypatch.setattr(_settings, "speculative", False)
        monkeypatch.setattr(_settings, "speculative_tokens", None)
        monkeypatch.delenv("OLMLX_SPECULATIVE_TOKENS", raising=False)
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_DFLASH_BLOCK_SIZE", "7")

        _surface_legacy_dflash_env()
        assert _settings.speculative_tokens == 7


class TestSurfaceLegacyDistributedEnv:
    def test_forwards_bool_int_and_path(self, monkeypatch, caplog):
        from olmlx.config import settings as _settings

        monkeypatch.setattr(_settings, "distributed", False)
        monkeypatch.setattr(_settings, "distributed_port", 5555)
        # New names absent so legacy wins.
        monkeypatch.delenv("OLMLX_DISTRIBUTED", raising=False)
        monkeypatch.delenv("OLMLX_DISTRIBUTED_PORT", raising=False)
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_DISTRIBUTED", "true")
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_DISTRIBUTED_PORT", "9001")

        with caplog.at_level(logging.WARNING, logger="olmlx.cli"):
            _surface_legacy_distributed_env()

        assert _settings.distributed is True
        assert _settings.distributed_port == 9001
        assert "Forwarding legacy" in caplog.text

    def test_new_var_wins_over_legacy(self, monkeypatch):
        from olmlx.config import settings as _settings

        monkeypatch.setattr(_settings, "distributed", False)
        monkeypatch.setenv("OLMLX_DISTRIBUTED", "false")
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_DISTRIBUTED", "true")

        _surface_legacy_distributed_env()
        # New (explicit) var present → legacy must not flip it on.
        assert _settings.distributed is False

    def test_invalid_int_logged_and_skipped(self, monkeypatch, caplog):
        from olmlx.config import settings as _settings

        monkeypatch.setattr(_settings, "distributed_port", 5555)
        monkeypatch.delenv("OLMLX_DISTRIBUTED_PORT", raising=False)
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_DISTRIBUTED_PORT", "not-a-number")

        with caplog.at_level(logging.WARNING, logger="olmlx.cli"):
            _surface_legacy_distributed_env()

        assert _settings.distributed_port == 5555
        assert "not a valid int" in caplog.text


class TestLegacyKvCacheQuantDotenvComment:
    def test_strips_inline_comment_on_unquoted_value(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".env").write_text(
            "OLMLX_EXPERIMENTAL_KV_CACHE_QUANT=turboquant:4  # comment\n"
        )
        assert _legacy_kv_cache_quant_in_dotenv() == "turboquant:4"

    def test_export_prefix_stripped(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".env").write_text(
            'export OLMLX_EXPERIMENTAL_KV_CACHE_QUANT="spectral:2"\n'
        )
        assert _legacy_kv_cache_quant_in_dotenv() == "spectral:2"


class TestCmdBenchCompare:
    def test_run_not_found_raises(self, tmp_path):
        ns = type("NS", (), {"run1": "no-such-run", "run2": "also-missing"})()
        with pytest.raises(FileNotFoundError, match="Run not found"):
            cmd_bench_compare(ns)


class TestCmdBenchList:
    def test_no_runs_message(self, tmp_path, capsys):
        ns = type("NS", (), {"bench_dir": str(tmp_path)})()
        cmd_bench_list(ns)
        out = capsys.readouterr().out
        assert "No benchmark runs found." in out
