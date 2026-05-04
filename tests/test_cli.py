"""Tests for olmlx.cli."""

import json
import plistlib
import subprocess
from unittest.mock import MagicMock

import pytest

from olmlx.cli import (
    DEFAULT_MODELS,
    PLIST_LABEL,
    _build_plist,
    _create_store,
    build_parser,
    cli_main,
    cmd_config_show,
    cmd_models_delete,
    cmd_models_list,
    cmd_models_pull,
    cmd_models_search,
    cmd_models_show,
    cmd_service_install,
    cmd_service_status,
    cmd_service_uninstall,
    ensure_config,
)
from olmlx.models.manifest import ModelManifest


class TestEnsureConfig:
    def test_creates_dir_and_models_json(self, tmp_path, monkeypatch):
        config_path = tmp_path / "subdir" / "models.json"
        monkeypatch.setattr("olmlx.cli.settings.models_config", config_path)
        ensure_config()
        assert config_path.exists()
        data = json.loads(config_path.read_text())
        assert data == DEFAULT_MODELS

    def test_does_not_overwrite_existing(self, tmp_path, monkeypatch):
        config_path = tmp_path / "models.json"
        existing = {"custom:latest": "some/model"}
        config_path.write_text(json.dumps(existing))
        monkeypatch.setattr("olmlx.cli.settings.models_config", config_path)
        ensure_config()
        data = json.loads(config_path.read_text())
        assert data == existing


class TestBuildPlist:
    def test_plist_structure(self, monkeypatch):
        monkeypatch.setattr("olmlx.cli.shutil.which", lambda _: "/usr/local/bin/olmlx")
        plist = _build_plist()
        assert plist["Label"] == PLIST_LABEL
        assert plist["ProgramArguments"] == ["/usr/local/bin/olmlx"]
        assert plist["RunAtLoad"] is True
        assert plist["KeepAlive"] is True
        assert "StandardOutPath" in plist
        assert "StandardErrorPath" in plist

    def test_plist_fallback_to_python(self, monkeypatch):
        monkeypatch.setattr("olmlx.cli.shutil.which", lambda _: None)
        monkeypatch.setattr("olmlx.cli.sys.executable", "/usr/bin/python3")
        plist = _build_plist()
        assert plist["ProgramArguments"] == ["/usr/bin/python3", "-m", "olmlx"]

    def test_plist_forwards_env_vars(self, monkeypatch):
        monkeypatch.setattr("olmlx.cli.shutil.which", lambda _: "/usr/local/bin/olmlx")
        monkeypatch.setenv("OLMLX_PORT", "9999")
        plist = _build_plist()
        assert plist["EnvironmentVariables"]["OLMLX_PORT"] == "9999"


class TestServiceInstall:
    def test_install_creates_plist_and_loads(self, tmp_path, monkeypatch):
        plist_path = tmp_path / "com.dpalmqvist.olmlx.plist"
        monkeypatch.setattr("olmlx.cli.PLIST_PATH", plist_path)
        monkeypatch.setattr(
            "olmlx.cli.settings.models_config", tmp_path / "models.json"
        )
        monkeypatch.setattr("olmlx.cli.shutil.which", lambda _: "/usr/local/bin/olmlx")
        mock_run = MagicMock()
        monkeypatch.setattr("olmlx.cli.subprocess.run", mock_run)
        cmd_service_install(None)
        assert plist_path.exists()
        with open(plist_path, "rb") as f:
            plist = plistlib.load(f)
        assert plist["Label"] == PLIST_LABEL
        mock_run.assert_called_once_with(
            ["launchctl", "load", str(plist_path)],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_install_handles_launchctl_failure(self, tmp_path, monkeypatch, capsys):
        plist_path = tmp_path / "com.dpalmqvist.olmlx.plist"
        monkeypatch.setattr("olmlx.cli.PLIST_PATH", plist_path)
        monkeypatch.setattr(
            "olmlx.cli.settings.models_config", tmp_path / "models.json"
        )
        monkeypatch.setattr("olmlx.cli.shutil.which", lambda _: "/usr/local/bin/olmlx")

        def failing_run(*args, **kwargs):
            raise subprocess.CalledProcessError(1, "launchctl", stderr="Load failed")

        monkeypatch.setattr("olmlx.cli.subprocess.run", failing_run)
        with pytest.raises(SystemExit) as exc_info:
            cmd_service_install(None)
        assert exc_info.value.code == 1
        assert plist_path.exists()  # plist was written before the failure
        captured = capsys.readouterr()
        assert "could not be loaded" in captured.err.lower()
        assert "Load failed" in captured.err


class TestServiceUninstall:
    def test_uninstall_removes_plist(self, tmp_path, monkeypatch):
        plist_path = tmp_path / "com.dpalmqvist.olmlx.plist"
        plist_path.write_text("dummy")
        monkeypatch.setattr("olmlx.cli.PLIST_PATH", plist_path)
        mock_run = MagicMock()
        monkeypatch.setattr("olmlx.cli.subprocess.run", mock_run)
        cmd_service_uninstall(None)
        assert not plist_path.exists()
        mock_run.assert_called_once_with(
            ["launchctl", "unload", str(plist_path)],
            check=False,
        )

    def test_uninstall_no_plist(self, tmp_path, monkeypatch, capsys):
        plist_path = tmp_path / "com.dpalmqvist.olmlx.plist"
        monkeypatch.setattr("olmlx.cli.PLIST_PATH", plist_path)
        mock_run = MagicMock()
        monkeypatch.setattr("olmlx.cli.subprocess.run", mock_run)
        cmd_service_uninstall(None)
        mock_run.assert_not_called()
        assert "No plist found" in capsys.readouterr().out


class TestServiceStatus:
    def test_status_loaded(self, monkeypatch, capsys):
        mock_result = MagicMock(
            returncode=0, stdout="PID\tStatus\tLabel\n123\t0\tcom.dpalmqvist.olmlx"
        )
        monkeypatch.setattr("olmlx.cli.subprocess.run", lambda *a, **kw: mock_result)
        cmd_service_status(None)
        out = capsys.readouterr().out
        assert "is loaded" in out

    def test_status_not_loaded(self, monkeypatch, capsys):
        mock_result = MagicMock(returncode=1, stdout="")
        monkeypatch.setattr("olmlx.cli.subprocess.run", lambda *a, **kw: mock_result)
        cmd_service_status(None)
        out = capsys.readouterr().out
        assert "is not loaded" in out


class TestBuildParser:
    def test_default_command(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.command is None

    def test_serve_command(self):
        parser = build_parser()
        args = parser.parse_args(["serve"])
        assert args.command == "serve"

    def test_serve_speculative_flags(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "serve",
                "--speculative",
                "--speculative-draft-model",
                "Qwen/Qwen3-0.6B",
                "--speculative-tokens",
                "6",
            ]
        )
        assert args.speculative is True
        assert args.speculative_draft_model == "Qwen/Qwen3-0.6B"
        assert args.speculative_tokens == 6

    def test_serve_no_speculative_flag(self):
        parser = build_parser()
        args = parser.parse_args(["serve", "--no-speculative"])
        assert args.speculative is False

    def test_apply_serve_overrides_accepts_global_no_draft_when_per_model_supplies(
        self, monkeypatch, tmp_path
    ):
        """Global speculative=True with no global draft must NOT exit when
        every registered model supplies its own ``speculative_draft_model``."""
        from olmlx.cli import _apply_serve_overrides
        from olmlx.config import settings as _settings

        models_json = tmp_path / "models.json"
        models_json.write_text(
            json.dumps(
                {
                    "good/model:latest": {
                        "hf_path": "good/model",
                        "speculative": True,
                        "speculative_draft_model": "good/draft",
                    },
                }
            )
        )
        monkeypatch.setattr(_settings, "models_config", models_json)
        monkeypatch.setattr(_settings, "speculative", True)
        monkeypatch.setattr(_settings, "speculative_draft_model", None)
        parser = build_parser()
        args = parser.parse_args(["serve"])
        # Should not raise SystemExit.
        _apply_serve_overrides(args)

    def test_apply_serve_overrides_forwards_legacy_env_vars(self, monkeypatch, caplog):
        """Legacy OLMLX_EXPERIMENTAL_SPECULATIVE* values are forwarded to
        the new Settings during the deprecation window so users don't
        silently lose speculative decoding on upgrade. Each forwarded
        field also produces a per-field warning so the override is
        visible alongside the bulk deprecation banner."""
        import logging

        from olmlx.cli import _apply_serve_overrides
        from olmlx.config import settings as _settings

        # Snapshot settings so other tests don't see leakage.
        monkeypatch.setattr(_settings, "speculative", False)
        monkeypatch.setattr(_settings, "speculative_draft_model", None)
        monkeypatch.setattr(_settings, "speculative_tokens", 4)
        # Stub out registry-walking helpers so the test is hermetic.
        monkeypatch.setattr(
            "olmlx.cli._audit_speculative_config",
            lambda: ([], [], [], False),
        )
        monkeypatch.setattr(
            "olmlx.cli._models_with_promoted_keys_in_experimental", lambda: []
        )
        monkeypatch.delenv("OLMLX_SPECULATIVE", raising=False)
        monkeypatch.delenv("OLMLX_SPECULATIVE_DRAFT_MODEL", raising=False)
        monkeypatch.delenv("OLMLX_SPECULATIVE_TOKENS", raising=False)
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_SPECULATIVE", "true")
        monkeypatch.setenv(
            "OLMLX_EXPERIMENTAL_SPECULATIVE_DRAFT_MODEL", "Qwen/Qwen3-0.6B"
        )
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_SPECULATIVE_TOKENS", "8")

        parser = build_parser()
        args = parser.parse_args(["serve"])
        with caplog.at_level(logging.WARNING, logger="olmlx.cli"):
            _apply_serve_overrides(args)

        assert _settings.speculative is True
        assert _settings.speculative_draft_model == "Qwen/Qwen3-0.6B"
        assert _settings.speculative_tokens == 8
        # Per-field forward warning fired for each value.
        assert "Forwarding legacy OLMLX_EXPERIMENTAL_SPECULATIVE_TOKENS" in caplog.text
        assert (
            "Forwarding legacy OLMLX_EXPERIMENTAL_SPECULATIVE_DRAFT_MODEL"
            in caplog.text
        )

    def test_legacy_env_var_validation_errors_are_swallowed(self, monkeypatch, caplog):
        """A legacy value that fails Settings validation must not block
        startup; ``validate_assignment=True`` raises pydantic ValidationError
        which is not a ValueError subclass."""
        import logging

        from olmlx.cli import _apply_serve_overrides
        from olmlx.config import settings as _settings

        monkeypatch.setattr(_settings, "speculative", False)
        monkeypatch.setattr(_settings, "speculative_draft_model", None)
        monkeypatch.setattr(_settings, "speculative_tokens", 4)
        monkeypatch.setattr(
            "olmlx.cli._audit_speculative_config",
            lambda: ([], [], [], False),
        )
        monkeypatch.setattr(
            "olmlx.cli._models_with_promoted_keys_in_experimental", lambda: []
        )
        monkeypatch.delenv("OLMLX_SPECULATIVE_TOKENS", raising=False)
        # 0 fails Field(gt=0) on assignment.
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_SPECULATIVE_TOKENS", "0")

        parser = build_parser()
        args = parser.parse_args(["serve"])
        with caplog.at_level(logging.WARNING, logger="olmlx.cli"):
            _apply_serve_overrides(args)
        assert "Could not forward legacy env var" in caplog.text
        # Settings keeps its prior default.
        assert _settings.speculative_tokens == 4

    def test_apply_serve_overrides_new_env_var_wins_over_legacy(self, monkeypatch):
        """When both legacy and new env vars are set, the new one wins."""
        from olmlx.cli import _apply_serve_overrides
        from olmlx.config import settings as _settings

        monkeypatch.setattr(_settings, "speculative", False)
        monkeypatch.setattr(_settings, "speculative_draft_model", "new/draft")
        monkeypatch.setattr(_settings, "speculative_tokens", 4)
        monkeypatch.setattr(
            "olmlx.cli._audit_speculative_config",
            lambda: ([], [], [], False),
        )
        monkeypatch.setattr(
            "olmlx.cli._models_with_promoted_keys_in_experimental", lambda: []
        )
        # The new env var is set in os.environ; legacy is also set.
        monkeypatch.setenv("OLMLX_SPECULATIVE_DRAFT_MODEL", "new/draft")
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_SPECULATIVE_DRAFT_MODEL", "legacy/draft")

        parser = build_parser()
        args = parser.parse_args(["serve"])
        _apply_serve_overrides(args)
        # The legacy value must NOT overwrite the (already-applied) new one.
        assert _settings.speculative_draft_model == "new/draft"

    def test_legacy_does_not_clobber_new_shell_var_equal_to_default(self, monkeypatch):
        """If the user explicitly sets the new shell var to a value that
        happens to equal the schema default, the legacy var must not
        win. Regression: the prior implementation only checked the
        resolved Settings value against the default, missing this case."""
        from olmlx.cli import _apply_serve_overrides
        from olmlx.config import settings as _settings

        monkeypatch.setattr(_settings, "speculative", False)
        monkeypatch.setattr(_settings, "speculative_draft_model", None)
        monkeypatch.setattr(_settings, "speculative_tokens", 4)
        monkeypatch.setattr(
            "olmlx.cli._audit_speculative_config",
            lambda: ([], [], [], False),
        )
        monkeypatch.setattr(
            "olmlx.cli._models_with_promoted_keys_in_experimental", lambda: []
        )
        # Mimic "user explicitly set OLMLX_SPECULATIVE_TOKENS=4 (the
        # default) in their shell". Settings already has the default,
        # but the env var's presence must short-circuit forwarding.
        monkeypatch.setenv("OLMLX_SPECULATIVE_TOKENS", "4")
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_SPECULATIVE_TOKENS", "8")

        parser = build_parser()
        args = parser.parse_args(["serve"])
        _apply_serve_overrides(args)
        assert _settings.speculative_tokens == 4

    def test_legacy_clobbers_explicit_dotenv_default_documented_blind_spot(
        self, monkeypatch
    ):
        """Pin the documented blind spot: a ``.env`` value that equals the
        schema default is indistinguishable from "field unset", so the
        legacy shell var still wins. Catching this would require parsing
        the ``.env`` file directly (or tracking provenance through
        pydantic-settings), neither of which is worth the complexity for
        a one-release deprecation window. If the behaviour ever changes,
        update the README migration note and this test together."""
        from olmlx.cli import _apply_serve_overrides
        from olmlx.config import settings as _settings

        # Mimic ".env contains OLMLX_SPECULATIVE=false (explicit
        # opt-out)". Settings was constructed with the default, so the
        # comparison `getattr(_settings, attr) == field_default` is True.
        monkeypatch.setattr(_settings, "speculative", False)
        monkeypatch.setattr(_settings, "speculative_draft_model", "x/draft")
        monkeypatch.setattr(_settings, "speculative_tokens", 4)
        monkeypatch.setattr(
            "olmlx.cli._audit_speculative_config",
            lambda: ([], [], [], False),
        )
        monkeypatch.setattr(
            "olmlx.cli._models_with_promoted_keys_in_experimental", lambda: []
        )
        monkeypatch.delenv("OLMLX_SPECULATIVE", raising=False)
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_SPECULATIVE", "true")

        parser = build_parser()
        args = parser.parse_args(["serve"])
        _apply_serve_overrides(args)
        # Documented behaviour: legacy wins over the matching-default
        # ``.env`` opt-out. The per-field forwarding warning makes the
        # override visible.
        assert _settings.speculative is True

    def test_legacy_does_not_clobber_dotenv_value(self, monkeypatch):
        """If pydantic-settings already loaded the new value from a .env
        file (so it never appears in ``os.environ``), the legacy shell var
        must not overwrite it. The forwarder gates on the resolved
        Settings value, not the raw env dict."""
        from olmlx.cli import _apply_serve_overrides
        from olmlx.config import settings as _settings

        # Simulate "pydantic-settings already populated the field from .env".
        monkeypatch.setattr(_settings, "speculative", False)
        monkeypatch.setattr(_settings, "speculative_draft_model", "dotenv/draft")
        monkeypatch.setattr(_settings, "speculative_tokens", 4)
        monkeypatch.setattr(
            "olmlx.cli._audit_speculative_config",
            lambda: ([], [], [], False),
        )
        monkeypatch.setattr(
            "olmlx.cli._models_with_promoted_keys_in_experimental", lambda: []
        )
        monkeypatch.delenv("OLMLX_SPECULATIVE_DRAFT_MODEL", raising=False)
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_SPECULATIVE_DRAFT_MODEL", "legacy/draft")

        parser = build_parser()
        args = parser.parse_args(["serve"])
        _apply_serve_overrides(args)
        assert _settings.speculative_draft_model == "dotenv/draft"

    def test_apply_serve_overrides_warns_on_deprecated_env_vars(
        self, monkeypatch, caplog
    ):
        import logging

        from olmlx.cli import _apply_serve_overrides
        from olmlx.config import settings as _settings

        monkeypatch.setattr(_settings, "speculative", False)
        monkeypatch.setattr(_settings, "speculative_draft_model", None)
        monkeypatch.setattr(
            "olmlx.cli._audit_speculative_config", lambda: ([], [], [], False)
        )
        monkeypatch.setattr(
            "olmlx.cli._models_with_promoted_keys_in_experimental", lambda: []
        )
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_SPECULATIVE", "true")
        monkeypatch.setenv(
            "OLMLX_EXPERIMENTAL_SPECULATIVE_DRAFT_MODEL", "Qwen/Qwen3-0.6B"
        )
        parser = build_parser()
        args = parser.parse_args(["serve"])
        with caplog.at_level(logging.WARNING, logger="olmlx.cli"):
            _apply_serve_overrides(args)
        msg = caplog.text
        assert "Deprecated env vars" in msg
        assert "OLMLX_EXPERIMENTAL_SPECULATIVE" in msg
        assert "OLMLX_EXPERIMENTAL_SPECULATIVE_DRAFT_MODEL" in msg

    def test_apply_serve_overrides_rejects_per_model_misconfig(self, monkeypatch):
        """Serve fails fast when a models.json entry enables speculative but
        has no draft model anywhere."""
        from olmlx.cli import _apply_serve_overrides
        from olmlx.config import settings as _settings

        monkeypatch.setattr(_settings, "speculative", False)
        monkeypatch.setattr(_settings, "speculative_draft_model", None)
        monkeypatch.setattr(
            "olmlx.cli._audit_speculative_config",
            lambda: (["bad/model:latest"], [], [], False),
        )
        monkeypatch.setattr(
            "olmlx.cli._models_with_promoted_keys_in_experimental", lambda: []
        )
        parser = build_parser()
        args = parser.parse_args(["serve"])
        with pytest.raises(SystemExit) as excinfo:
            _apply_serve_overrides(args)
        assert excinfo.value.code == 2

    def test_apply_serve_overrides_rejects_promoted_keys_in_experimental(
        self, monkeypatch, tmp_path
    ):
        """Loading a models.json that still places speculative keys under
        ``experimental`` exits with a clear migration error rather than
        burying it in registry warnings."""
        from olmlx.cli import _apply_serve_overrides
        from olmlx.config import settings as _settings

        models_json = tmp_path / "models.json"
        models_json.write_text(
            json.dumps(
                {
                    "stale/model:latest": {
                        "hf_path": "stale/model",
                        "experimental": {
                            "speculative": True,
                            "speculative_draft_model": "stale/draft",
                        },
                    },
                }
            )
        )
        monkeypatch.setattr(_settings, "models_config", models_json)
        monkeypatch.setattr(_settings, "speculative", False)
        monkeypatch.setattr(_settings, "speculative_draft_model", None)
        parser = build_parser()
        args = parser.parse_args(["serve"])
        with pytest.raises(SystemExit) as excinfo:
            _apply_serve_overrides(args)
        assert excinfo.value.code == 2

    def test_apply_serve_overrides_warns_on_global_dormant_draft(
        self, monkeypatch, caplog
    ):
        """Global ``speculative_draft_model`` set without ``speculative=True``
        emits a warning so the user notices the dormant config."""
        import logging

        from olmlx.cli import _apply_serve_overrides
        from olmlx.config import settings as _settings

        monkeypatch.setattr(_settings, "speculative", False)
        monkeypatch.setattr(_settings, "speculative_draft_model", "global/draft")
        monkeypatch.setattr(
            "olmlx.cli._audit_speculative_config", lambda: ([], [], [], False)
        )
        monkeypatch.setattr(
            "olmlx.cli._models_with_promoted_keys_in_experimental", lambda: []
        )
        parser = build_parser()
        args = parser.parse_args(["serve"])
        with caplog.at_level(logging.WARNING, logger="olmlx.cli"):
            _apply_serve_overrides(args)
        assert "OLMLX_SPECULATIVE_DRAFT_MODEL" in caplog.text
        assert "global/draft" in caplog.text

    def test_models_with_promoted_keys_warns_on_corrupt_json(
        self, monkeypatch, tmp_path, caplog
    ):
        """A corrupt models.json should produce a visible warning rather
        than silently passing the migration check."""
        import logging

        from olmlx.cli import _models_with_promoted_keys_in_experimental
        from olmlx.config import settings as _settings

        models_json = tmp_path / "models.json"
        models_json.write_text("{not valid json")
        monkeypatch.setattr(_settings, "models_config", models_json)
        with caplog.at_level(logging.WARNING, logger="olmlx.cli"):
            result = _models_with_promoted_keys_in_experimental()
        assert result == []
        assert "models.json is invalid JSON" in caplog.text

    def test_apply_serve_overrides_warns_on_flash_conflict(self, monkeypatch, caplog):
        """A model that combines speculative with Flash gets a warning so the
        user knows the standalone speculative knob is dropped on the Flash
        load path."""
        import logging

        from olmlx.cli import _apply_serve_overrides
        from olmlx.config import settings as _settings

        monkeypatch.setattr(_settings, "speculative", False)
        monkeypatch.setattr(_settings, "speculative_draft_model", None)
        monkeypatch.setattr(
            "olmlx.cli._audit_speculative_config",
            lambda: ([], [], ["flash/model:latest"], False),
        )
        monkeypatch.setattr(
            "olmlx.cli._models_with_promoted_keys_in_experimental", lambda: []
        )
        parser = build_parser()
        args = parser.parse_args(["serve"])
        with caplog.at_level(logging.WARNING, logger="olmlx.cli"):
            _apply_serve_overrides(args)
        assert "flash/model:latest" in caplog.text
        assert "flash_speculative" in caplog.text

    def test_apply_serve_overrides_warns_on_dormant_draft(self, monkeypatch, caplog):
        """A draft configured for a model with speculative=False emits a
        warning so the user notices the dormant config."""
        import logging

        from olmlx.cli import _apply_serve_overrides
        from olmlx.config import settings as _settings

        monkeypatch.setattr(_settings, "speculative", False)
        monkeypatch.setattr(_settings, "speculative_draft_model", None)
        monkeypatch.setattr(
            "olmlx.cli._audit_speculative_config",
            lambda: ([], ["dormant/model:latest"], [], False),
        )
        monkeypatch.setattr(
            "olmlx.cli._models_with_promoted_keys_in_experimental", lambda: []
        )
        parser = build_parser()
        args = parser.parse_args(["serve"])
        with caplog.at_level(logging.WARNING, logger="olmlx.cli"):
            _apply_serve_overrides(args)
        assert "dormant/model:latest" in caplog.text
        assert "speculative_draft_model" in caplog.text

    def test_global_dormant_warning_when_per_model_uses_own_draft(
        self, monkeypatch, tmp_path, caplog
    ):
        """If global draft is set but each speculative-enabled model has its
        own draft (so nobody consumes the global), the warning must fire —
        and the wording must not falsely claim speculative is off everywhere."""
        import logging

        from olmlx.cli import _apply_serve_overrides
        from olmlx.config import settings as _settings

        models_json = tmp_path / "models.json"
        models_json.write_text(
            json.dumps(
                {
                    "with-own/m:latest": {
                        "hf_path": "with-own/m",
                        "speculative": True,
                        "speculative_draft_model": "with-own/draft",
                    },
                }
            )
        )
        monkeypatch.setattr(_settings, "models_config", models_json)
        monkeypatch.setattr(_settings, "speculative", False)
        monkeypatch.setattr(_settings, "speculative_draft_model", "global/draft")

        parser = build_parser()
        args = parser.parse_args(["serve"])
        with caplog.at_level(logging.WARNING, logger="olmlx.cli"):
            _apply_serve_overrides(args)
        assert "no model" in caplog.text
        assert "global/draft" in caplog.text
        # Must not falsely claim speculative is disabled globally.
        assert "no per-model entry enables speculative decoding" not in caplog.text

    def test_global_dormant_warning_suppressed_when_per_model_consumes(
        self, monkeypatch, tmp_path, caplog
    ):
        """A global draft + global speculative=False used to warn even when a
        per-model entry enables speculative and consumes that global draft."""
        import logging

        from olmlx.cli import _apply_serve_overrides
        from olmlx.config import settings as _settings

        models_json = tmp_path / "models.json"
        models_json.write_text(
            json.dumps(
                {
                    "consumer/m:latest": {
                        "hf_path": "consumer/m",
                        "speculative": True,
                    },
                }
            )
        )
        monkeypatch.setattr(_settings, "models_config", models_json)
        monkeypatch.setattr(_settings, "speculative", False)
        monkeypatch.setattr(_settings, "speculative_draft_model", "global/draft")

        parser = build_parser()
        args = parser.parse_args(["serve"])
        with caplog.at_level(logging.WARNING, logger="olmlx.cli"):
            _apply_serve_overrides(args)
        assert "draft model will not be loaded" not in caplog.text

    def test_global_dormant_warning_when_global_on_but_per_model_has_own_draft(
        self, monkeypatch, tmp_path, caplog
    ):
        """Global ``speculative=True`` and global draft set, but every
        per-model entry has its own draft override. The global draft is
        unused — warning must fire."""
        import logging

        from olmlx.cli import _apply_serve_overrides
        from olmlx.config import settings as _settings

        models_json = tmp_path / "models.json"
        models_json.write_text(
            json.dumps(
                {
                    "with-own/m:latest": {
                        "hf_path": "with-own/m",
                        "speculative": True,
                        "speculative_draft_model": "with-own/draft",
                    },
                }
            )
        )
        monkeypatch.setattr(_settings, "models_config", models_json)
        monkeypatch.setattr(_settings, "speculative", True)
        monkeypatch.setattr(_settings, "speculative_draft_model", "global/draft")

        parser = build_parser()
        args = parser.parse_args(["serve"])
        with caplog.at_level(logging.WARNING, logger="olmlx.cli"):
            _apply_serve_overrides(args)
        assert "global/draft" in caplog.text
        assert "no model" in caplog.text

    def test_global_dormant_warning_when_all_per_model_override_disabled(
        self, monkeypatch, tmp_path, caplog
    ):
        """Global ``speculative=True`` + global draft set, but every per-model
        entry overrides ``speculative=false``. The global draft is dormant in
        practice and the warning must fire."""
        import logging

        from olmlx.cli import _apply_serve_overrides
        from olmlx.config import settings as _settings

        models_json = tmp_path / "models.json"
        models_json.write_text(
            json.dumps(
                {
                    "off/m:latest": {
                        "hf_path": "off/m",
                        "speculative": False,
                    },
                }
            )
        )
        monkeypatch.setattr(_settings, "models_config", models_json)
        monkeypatch.setattr(_settings, "speculative", True)
        monkeypatch.setattr(_settings, "speculative_draft_model", "global/draft")

        parser = build_parser()
        args = parser.parse_args(["serve"])
        with caplog.at_level(logging.WARNING, logger="olmlx.cli"):
            _apply_serve_overrides(args)
        assert "no model" in caplog.text
        assert "global/draft" in caplog.text

    def test_audit_flags_global_flash_with_speculative(self, monkeypatch, tmp_path):
        """Globally enabled Flash + speculative=True (per model, no per-model
        flash override) must be caught as a flash-conflict — the previous
        per-model-only check missed this case."""
        from olmlx.cli import _audit_speculative_config
        from olmlx.config import experimental as _exp
        from olmlx.config import settings as _settings

        models_json = tmp_path / "models.json"
        models_json.write_text(
            json.dumps(
                {
                    "global-flash/m:latest": {
                        "hf_path": "global-flash/m",
                        "speculative": True,
                        "speculative_draft_model": "global-flash/draft",
                    },
                }
            )
        )
        monkeypatch.setattr(_settings, "models_config", models_json)
        monkeypatch.setattr(_settings, "speculative", False)
        monkeypatch.setattr(_settings, "speculative_draft_model", None)
        monkeypatch.setattr(_exp, "flash", True)

        bad, dormant, flash_conflicts, _global_used = _audit_speculative_config()
        assert bad == []
        assert dormant == []
        assert flash_conflicts == ["global-flash/m:latest"]

    def test_audit_speculative_config_classifies_models(self, monkeypatch, tmp_path):
        """End-to-end: registry walk classifies models into bad / dormant /
        flash-conflict buckets."""
        from olmlx.cli import _audit_speculative_config
        from olmlx.config import settings as _settings

        models_json = tmp_path / "models.json"
        models_json.write_text(
            json.dumps(
                {
                    "good/a:latest": {
                        "hf_path": "good/a",
                        "speculative": True,
                        "speculative_draft_model": "good/draft",
                    },
                    "bad/no-draft:latest": {
                        "hf_path": "bad/no-draft",
                        "speculative": True,
                    },
                    "dormant/has-draft:latest": {
                        "hf_path": "dormant/has-draft",
                        "speculative_draft_model": "dormant/draft",
                    },
                    "conflict/flash-and-spec:latest": {
                        "hf_path": "conflict/flash-and-spec",
                        "speculative": True,
                        "speculative_draft_model": "conflict/draft",
                        "experimental": {"flash": True},
                    },
                }
            )
        )
        monkeypatch.setattr(_settings, "models_config", models_json)
        monkeypatch.setattr(_settings, "speculative", False)
        monkeypatch.setattr(_settings, "speculative_draft_model", None)

        bad, dormant, flash_conflicts, global_used = _audit_speculative_config()
        # Compare as sets so the test isn't coupled to the registry's
        # internal iteration order.
        assert set(bad) == {"bad/no-draft:latest"}
        assert set(dormant) == {"dormant/has-draft:latest"}
        assert set(flash_conflicts) == {"conflict/flash-and-spec:latest"}
        # No model in this fixture uses the global draft (good/a has its own).
        assert global_used is False

    def test_service_install(self):
        parser = build_parser()
        args = parser.parse_args(["service", "install"])
        assert args.command == "service"
        assert args.service_command == "install"

    def test_service_uninstall(self):
        parser = build_parser()
        args = parser.parse_args(["service", "uninstall"])
        assert args.command == "service"
        assert args.service_command == "uninstall"

    def test_service_status(self):
        parser = build_parser()
        args = parser.parse_args(["service", "status"])
        assert args.command == "service"
        assert args.service_command == "status"


class TestCliMain:
    def test_default_calls_serve(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["olmlx"])
        mock_serve = MagicMock()
        monkeypatch.setattr("olmlx.cli.cmd_serve", mock_serve)
        cli_main()
        mock_serve.assert_called_once()

    def test_bare_invocation_synthesizes_serve_defaults(self, monkeypatch):
        """Regression: bare ``olmlx`` must populate the serve-subparser
        defaults on ``args`` so cmd_serve can read them uniformly. If the
        list goes out of sync with the parser, _apply_serve_overrides
        would AttributeError instead of seeing a None default."""
        monkeypatch.setattr("sys.argv", ["olmlx"])
        captured: dict[str, object] = {}

        def fake_serve(args):
            captured["speculative"] = args.speculative
            captured["speculative_draft_model"] = args.speculative_draft_model
            captured["speculative_tokens"] = args.speculative_tokens

        monkeypatch.setattr("olmlx.cli.cmd_serve", fake_serve)
        cli_main()
        assert captured == {
            "speculative": None,
            "speculative_draft_model": None,
            "speculative_tokens": None,
        }

    def test_serve_calls_serve(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["olmlx", "serve"])
        mock_serve = MagicMock()
        monkeypatch.setattr("olmlx.cli.cmd_serve", mock_serve)
        cli_main()
        mock_serve.assert_called_once()

    def test_service_install_calls_install(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["olmlx", "service", "install"])
        mock_install = MagicMock()
        monkeypatch.setattr("olmlx.cli.cmd_service_install", mock_install)
        cli_main()
        mock_install.assert_called_once()

    def test_models_list_calls_handler(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["olmlx", "models", "list"])
        mock_fn = MagicMock()
        monkeypatch.setattr("olmlx.cli.cmd_models_list", mock_fn)
        cli_main()
        mock_fn.assert_called_once()

    def test_models_pull_calls_handler(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["olmlx", "models", "pull", "qwen2.5:3b"])
        mock_fn = MagicMock()
        monkeypatch.setattr("olmlx.cli.cmd_models_pull", mock_fn)
        cli_main()
        mock_fn.assert_called_once()

    def test_models_show_calls_handler(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["olmlx", "models", "show", "qwen2.5:3b"])
        mock_fn = MagicMock()
        monkeypatch.setattr("olmlx.cli.cmd_models_show", mock_fn)
        cli_main()
        mock_fn.assert_called_once()

    def test_models_delete_calls_handler(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["olmlx", "models", "delete", "qwen2.5:3b"])
        mock_fn = MagicMock()
        monkeypatch.setattr("olmlx.cli.cmd_models_delete", mock_fn)
        cli_main()
        mock_fn.assert_called_once()

    def test_config_show_calls_handler(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["olmlx", "config", "show"])
        mock_fn = MagicMock()
        monkeypatch.setattr("olmlx.cli.cmd_config_show", mock_fn)
        cli_main()
        mock_fn.assert_called_once()

    def test_models_no_subcommand_shows_help(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["olmlx", "models"])
        with pytest.raises(SystemExit) as exc_info:
            cli_main()
        assert exc_info.value.code == 0

    def test_config_no_subcommand_shows_help(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["olmlx", "config"])
        with pytest.raises(SystemExit) as exc_info:
            cli_main()
        assert exc_info.value.code == 0


class TestCreateStore:
    def test_calls_ensure_config(self, tmp_path, monkeypatch):
        """_create_store must call ensure_config so fresh installs get models.json."""
        config_path = tmp_path / "models.json"
        monkeypatch.setattr("olmlx.cli.settings.models_config", config_path)
        monkeypatch.setattr("olmlx.cli.settings.models_dir", tmp_path / "models")
        _create_store()
        # ensure_config should have created models.json
        assert config_path.exists()
        data = json.loads(config_path.read_text())
        assert data == DEFAULT_MODELS

    def test_malformed_config_logs_warning(self, tmp_path, monkeypatch, caplog):
        """Malformed models.json should log a warning and start with empty config."""
        config_path = tmp_path / "models.json"
        config_path.write_text("{bad json")
        monkeypatch.setattr("olmlx.cli.settings.models_config", config_path)
        import logging

        with caplog.at_level(logging.WARNING):
            store = _create_store()
        assert store is not None
        assert "Corrupted" in caplog.text

    def test_ensure_config_permission_error_raises(self, tmp_path, monkeypatch):
        """Permission error in ensure_config should propagate."""
        monkeypatch.setattr(
            "olmlx.cli.ensure_config",
            lambda: (_ for _ in ()).throw(PermissionError("Permission denied")),
        )
        with pytest.raises(PermissionError):
            _create_store()


@pytest.fixture
def mock_store():
    """Create a mock ModelStore for CLI tests."""
    store = MagicMock()
    return store


@pytest.fixture
def _patch_store(monkeypatch, mock_store):
    """Patch _create_store to return mock_store."""
    monkeypatch.setattr("olmlx.cli._create_store", lambda: mock_store)


class TestCreateStoreErrorHandlingInCommands:
    """Commands should catch _create_store errors and exit cleanly."""

    def test_list_handles_store_error(self, monkeypatch, capsys):
        monkeypatch.setattr(
            "olmlx.cli._create_store",
            lambda: (_ for _ in ()).throw(Exception("bad config")),
        )
        with pytest.raises(SystemExit) as exc_info:
            cmd_models_list(None)
        assert exc_info.value.code == 1
        assert "bad config" in capsys.readouterr().err

    def test_show_handles_store_error(self, monkeypatch, capsys):
        monkeypatch.setattr(
            "olmlx.cli._create_store",
            lambda: (_ for _ in ()).throw(Exception("bad config")),
        )
        args = MagicMock(model_name="test")
        with pytest.raises(SystemExit) as exc_info:
            cmd_models_show(args)
        assert exc_info.value.code == 1
        assert "bad config" in capsys.readouterr().err

    def test_pull_handles_store_error(self, monkeypatch, capsys):
        monkeypatch.setattr(
            "olmlx.cli._create_store",
            lambda: (_ for _ in ()).throw(Exception("bad config")),
        )
        args = MagicMock(model_name="test")
        with pytest.raises(SystemExit) as exc_info:
            cmd_models_pull(args)
        assert exc_info.value.code == 1
        assert "bad config" in capsys.readouterr().err

    def test_delete_handles_store_error(self, monkeypatch, capsys):
        monkeypatch.setattr(
            "olmlx.cli._create_store",
            lambda: (_ for _ in ()).throw(Exception("bad config")),
        )
        args = MagicMock(model_name="test", yes=True)
        with pytest.raises(SystemExit) as exc_info:
            cmd_models_delete(args)
        assert exc_info.value.code == 1
        assert "bad config" in capsys.readouterr().err


class TestModelsListCmd:
    def test_lists_models(self, capsys, mock_store, _patch_store):
        mock_store.list_local.return_value = [
            ModelManifest(
                name="qwen2.5:3b",
                hf_path="mlx-community/Qwen2.5-3B-Instruct-4bit",
                size=2_000_000_000,
                parameter_size="3B",
                quantization_level="4-bit",
            ),
            ModelManifest(
                name="llama3.2:latest",
                hf_path="mlx-community/Llama-3.2-3B-Instruct-4bit",
                size=1_500_000_000,
            ),
        ]
        cmd_models_list(None)
        out = capsys.readouterr().out
        assert "qwen2.5:3b" in out
        assert "llama3.2:latest" in out
        assert "2.0 GB" in out
        assert "1.5 GB" in out

    def test_truncates_long_names(self, capsys, mock_store, _patch_store):
        long_name = "a" * 40
        mock_store.list_local.return_value = [
            ModelManifest(
                name=long_name,
                hf_path="some/model",
                size=1_000_000,
                parameter_size="x" * 15,
                quantization_level="y" * 15,
            ),
        ]
        cmd_models_list(None)
        out = capsys.readouterr().out
        lines = out.strip().split("\n")
        # The data line (after header + separator) should not contain the full 40-char name
        data_line = lines[2]
        assert long_name not in data_line
        assert "x" * 15 not in data_line
        assert "y" * 15 not in data_line

    def test_exact_column_width_not_truncated(self, capsys, mock_store, _patch_store):
        """A name exactly at column width (30) should not be truncated."""
        name_30 = "a" * 30
        mock_store.list_local.return_value = [
            ModelManifest(
                name=name_30,
                hf_path="some/model",
                size=1_000_000,
                parameter_size="1234567890",
                quantization_level="1234567890",
            ),
        ]
        cmd_models_list(None)
        out = capsys.readouterr().out
        data_line = out.strip().split("\n")[2]
        assert name_30 in data_line
        assert "1234567890" in data_line

    def test_handles_none_string_fields(self, capsys, mock_store, _patch_store):
        """Manifests with None for string fields should not crash."""
        m = ModelManifest(
            name="test:latest",
            hf_path="some/model",
            size=1_000_000,
        )
        m.parameter_size = None
        m.quantization_level = None
        mock_store.list_local.return_value = [m]
        cmd_models_list(None)
        out = capsys.readouterr().out
        assert "test:latest" in out

    def test_handles_none_name(self, capsys, mock_store, _patch_store):
        """Manifests with None name should not crash the sort."""
        m = ModelManifest(name="valid:latest", hf_path="some/model", size=100)
        m2 = ModelManifest(name="other:latest", hf_path="some/model2", size=200)
        m2.name = None
        mock_store.list_local.return_value = [m, m2]
        cmd_models_list(None)
        out = capsys.readouterr().out
        assert "valid:latest" in out

    def test_lists_no_models(self, capsys, mock_store, _patch_store):
        mock_store.list_local.return_value = []
        cmd_models_list(None)
        out = capsys.readouterr().out
        assert "No models" in out

    def test_handles_corrupt_manifest(self, capsys, mock_store, _patch_store):
        """Corrupt manifest in list_local should produce clean error, not traceback."""
        mock_store.list_local.side_effect = ValueError("Field 'size' should be int")
        with pytest.raises(SystemExit) as exc_info:
            cmd_models_list(None)
        assert exc_info.value.code == 1
        assert "size" in capsys.readouterr().err


class TestModelsShowCmd:
    def test_show_model(self, capsys, mock_store, _patch_store):
        mock_store.show.return_value = ModelManifest(
            name="qwen2.5:3b",
            hf_path="mlx-community/Qwen2.5-3B-Instruct-4bit",
            size=2_000_000_000,
            family="qwen2",
            parameter_size="3B",
            quantization_level="4-bit",
            format="mlx",
        )
        args = MagicMock(model_name="qwen2.5:3b")
        cmd_models_show(args)
        out = capsys.readouterr().out
        assert "qwen2.5:3b" in out
        assert "mlx-community/Qwen2.5-3B-Instruct-4bit" in out
        assert "qwen2" in out

    def test_show_handles_corrupt_manifest(self, capsys, mock_store, _patch_store):
        """Corrupt manifest should produce clean error, not traceback."""
        mock_store.show.side_effect = ValueError("Field 'size' should be int")
        args = MagicMock(model_name="qwen2.5:3b")
        with pytest.raises(SystemExit) as exc_info:
            cmd_models_show(args)
        assert exc_info.value.code == 1
        assert "size" in capsys.readouterr().err

    def test_show_not_found_exits_nonzero(self, capsys, mock_store, _patch_store):
        mock_store.show.return_value = None
        args = MagicMock(model_name="nonexistent")
        with pytest.raises(SystemExit) as exc_info:
            cmd_models_show(args)
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "not found" in captured.err.lower()
        assert "not found" not in captured.out.lower()


class TestModelsPullCmd:
    def test_pull_model(self, capsys, mock_store, _patch_store):
        async def fake_pull(name):
            yield {"status": "pulling manifest"}
            yield {"status": "downloading mlx-community/Qwen2.5-3B-Instruct-4bit"}
            yield {"status": "success"}

        mock_store.pull = fake_pull
        args = MagicMock(model_name="qwen2.5:3b")
        cmd_models_pull(args)
        out = capsys.readouterr().out
        assert "pulling manifest" in out
        assert "success" in out

    def test_pull_model_not_found_exits_nonzero(self, capsys, mock_store, _patch_store):
        async def fake_pull(name):
            if False:
                yield {}
            raise ValueError(f"Model '{name}' not found in config")

        mock_store.pull = fake_pull
        args = MagicMock(model_name="nonexistent")
        with pytest.raises(SystemExit) as exc_info:
            cmd_models_pull(args)
        assert exc_info.value.code == 1
        assert "not found" in capsys.readouterr().err.lower()

    def test_pull_handles_os_error_exits_nonzero(
        self, capsys, mock_store, _patch_store
    ):
        async def fake_pull(name):
            yield {"status": "pulling manifest"}
            raise OSError("Disk full")

        mock_store.pull = fake_pull
        args = MagicMock(model_name="qwen2.5:3b")
        with pytest.raises(SystemExit) as exc_info:
            cmd_models_pull(args)
        assert exc_info.value.code == 1
        assert "Disk full" in capsys.readouterr().err

    def test_pull_handles_generic_exception_exits_nonzero(
        self, capsys, mock_store, _patch_store
    ):
        async def fake_pull(name):
            if False:
                yield {}
            raise RuntimeError("Something unexpected")

        mock_store.pull = fake_pull
        args = MagicMock(model_name="qwen2.5:3b")
        with pytest.raises(SystemExit) as exc_info:
            cmd_models_pull(args)
        assert exc_info.value.code == 1
        assert "Something unexpected" in capsys.readouterr().err

    def test_pull_error_goes_to_stderr(self, mock_store, _patch_store, capsys):
        """Error output should go to stderr, not stdout."""

        async def fake_pull(name):
            if False:
                yield {}
            raise ValueError("Model not found")

        mock_store.pull = fake_pull
        args = MagicMock(model_name="nonexistent")
        with pytest.raises(SystemExit):
            cmd_models_pull(args)
        captured = capsys.readouterr()
        assert "Model not found" in captured.err
        assert "Model not found" not in captured.out

    def test_pull_keyboard_interrupt_exits_130(self, capsys, mock_store, _patch_store):
        """Ctrl+C during pull should exit with code 130, not crash."""

        async def fake_pull(name):
            yield {"status": "pulling manifest"}
            raise KeyboardInterrupt

        mock_store.pull = fake_pull
        args = MagicMock(model_name="qwen2.5:3b")
        with pytest.raises(SystemExit) as exc_info:
            cmd_models_pull(args)
        assert exc_info.value.code == 130

    def test_pull_no_blank_lines(self, capsys, mock_store, _patch_store):
        """Status dicts without 'status' key should not produce blank lines."""

        async def fake_pull(name):
            yield {"status": "pulling manifest"}
            yield {"digest": "sha256:abc123"}  # no "status" key
            yield {}  # empty dict
            yield {"status": "success"}

        mock_store.pull = fake_pull
        args = MagicMock(model_name="qwen2.5:3b")
        cmd_models_pull(args)
        out = capsys.readouterr().out
        assert "pulling manifest" in out
        assert "success" in out
        # No blank lines in output
        lines = out.rstrip("\n").split("\n")
        for line in lines:
            assert line.strip() != "", f"Unexpected blank line in output: {lines!r}"

    def test_pull_flushes_output(self, mock_store, _patch_store, monkeypatch):
        """Status lines should be flushed immediately for piped output."""
        flush_calls = []
        original_print = print

        def tracking_print(*args, **kwargs):
            if kwargs.get("flush"):
                flush_calls.append(True)
            original_print(*args, **kwargs)

        monkeypatch.setattr("builtins.print", tracking_print)

        async def fake_pull(name):
            yield {"status": "pulling manifest"}
            yield {"status": "success"}

        mock_store.pull = fake_pull
        args = MagicMock(model_name="qwen2.5:3b")
        cmd_models_pull(args)
        assert len(flush_calls) == 2


class TestModelsDeleteCmd:
    def test_delete_model_with_yes_flag(self, capsys, mock_store, _patch_store):
        mock_store.delete.return_value = True
        args = MagicMock(model_name="qwen2.5:3b", yes=True)
        cmd_models_delete(args)
        out = capsys.readouterr().out
        assert "deleted" in out.lower()

    def test_delete_not_found_exits_nonzero(self, capsys, mock_store, _patch_store):
        mock_store.delete.return_value = False
        args = MagicMock(model_name="nonexistent", yes=True)
        with pytest.raises(SystemExit) as exc_info:
            cmd_models_delete(args)
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "not found" in captured.err.lower()
        assert "not found" not in captured.out.lower()

    def test_delete_prompts_for_confirmation(
        self, capsys, monkeypatch, mock_store, _patch_store
    ):
        mock_store.delete.return_value = True
        monkeypatch.setattr("builtins.input", lambda _: "y")
        args = MagicMock(model_name="qwen2.5:3b", yes=False)
        cmd_models_delete(args)
        out = capsys.readouterr().out
        assert "deleted" in out.lower()

    def test_delete_aborts_on_no(self, capsys, monkeypatch, mock_store, _patch_store):
        monkeypatch.setattr("builtins.input", lambda _: "n")
        args = MagicMock(model_name="qwen2.5:3b", yes=False)
        cmd_models_delete(args)
        out = capsys.readouterr().out
        assert "aborted" in out.lower()
        mock_store.delete.assert_not_called()

    def test_delete_aborts_on_eof(self, capsys, monkeypatch, mock_store, _patch_store):
        """Closed stdin (piped/CI) should abort cleanly, not crash."""

        def raise_eof(_):
            raise EOFError

        monkeypatch.setattr("builtins.input", raise_eof)
        args = MagicMock(model_name="qwen2.5:3b", yes=False)
        cmd_models_delete(args)
        out = capsys.readouterr().out
        assert "aborted" in out.lower()
        mock_store.delete.assert_not_called()

    def test_delete_handles_os_error(self, capsys, mock_store, _patch_store):
        """OSError from shutil.rmtree should produce clean error, not traceback."""
        mock_store.delete.side_effect = OSError("Permission denied")
        args = MagicMock(model_name="qwen2.5:3b", yes=True)
        with pytest.raises(SystemExit) as exc_info:
            cmd_models_delete(args)
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "permission denied" in captured.err.lower()

    def test_delete_handles_non_os_error(self, capsys, mock_store, _patch_store):
        """Non-OSError from store.delete should also produce clean error."""
        mock_store.delete.side_effect = ValueError("Invalid model dir")
        args = MagicMock(model_name="qwen2.5:3b", yes=True)
        with pytest.raises(SystemExit) as exc_info:
            cmd_models_delete(args)
        assert exc_info.value.code == 1
        assert "Invalid model dir" in capsys.readouterr().err


class TestConfigShowCmd:
    def test_shows_config(self, capsys):
        cmd_config_show(None)
        out = capsys.readouterr().out
        assert "host" in out.lower()
        assert "port" in out.lower()
        assert "models dir" in out.lower()


class TestBuildParserModels:
    def test_models_list(self):
        parser = build_parser()
        args = parser.parse_args(["models", "list"])
        assert args.command == "models"
        assert args.models_command == "list"

    def test_models_pull(self):
        parser = build_parser()
        args = parser.parse_args(["models", "pull", "qwen2.5:3b"])
        assert args.command == "models"
        assert args.models_command == "pull"
        assert args.model_name == "qwen2.5:3b"

    def test_models_show(self):
        parser = build_parser()
        args = parser.parse_args(["models", "show", "qwen2.5:3b"])
        assert args.command == "models"
        assert args.models_command == "show"
        assert args.model_name == "qwen2.5:3b"

    def test_models_delete(self):
        parser = build_parser()
        args = parser.parse_args(["models", "delete", "qwen2.5:3b"])
        assert args.command == "models"
        assert args.models_command == "delete"
        assert args.model_name == "qwen2.5:3b"
        assert args.yes is False

    def test_models_delete_yes_flag(self):
        parser = build_parser()
        args = parser.parse_args(["models", "delete", "--yes", "qwen2.5:3b"])
        assert args.yes is True

    def test_models_delete_y_flag(self):
        parser = build_parser()
        args = parser.parse_args(["models", "delete", "-y", "qwen2.5:3b"])
        assert args.yes is True

    def test_config_show(self):
        parser = build_parser()
        args = parser.parse_args(["config", "show"])
        assert args.command == "config"
        assert args.config_command == "show"

    def test_models_search(self):
        parser = build_parser()
        args = parser.parse_args(["models", "search", "qwen"])
        assert args.command == "models"
        assert args.models_command == "search"
        assert args.query == "qwen"


class TestModelsSearchCmd:
    def test_cmd_models_search_found(self, capsys, mock_store, _patch_store):
        """Search with matches should print model names and HF paths."""
        # mock_store.registry must have a search method
        mock_store.registry = MagicMock()
        mock_store.registry.search.return_value = [
            ("qwen3:latest", "Qwen/Qwen3-8B-MLX"),
        ]
        args = MagicMock(query="qwen3")
        cmd_models_search(args)
        out = capsys.readouterr().out
        assert "qwen3:latest" in out
        assert "Qwen/Qwen3-8B-MLX" in out

    def test_cmd_models_search_not_found(self, capsys, mock_store, _patch_store):
        """Search with no matches should show 'No models matching' message."""
        mock_store.registry = MagicMock()
        mock_store.registry.search.return_value = []
        args = MagicMock(query="zzzzzzz")
        cmd_models_search(args)
        out = capsys.readouterr().out
        assert "No models matching" in out


class TestCliMainModelsSearch:
    def test_models_search_calls_handler(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["olmlx", "models", "search", "qwen"])
        mock_fn = MagicMock()
        monkeypatch.setattr("olmlx.cli.cmd_models_search", mock_fn)
        cli_main()
        mock_fn.assert_called_once()


class TestModelsShowSuggestion:
    def test_show_not_found_suggests_search(self, capsys, mock_store, _patch_store):
        """When model not found, stderr should suggest running search."""
        mock_store.show.return_value = None
        args = MagicMock(model_name="qwem3")
        with pytest.raises(SystemExit) as exc_info:
            cmd_models_show(args)
        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert "olmlx models search" in err


class TestModelsPullSuggestion:
    def test_pull_not_found_suggests_search(self, capsys, mock_store, _patch_store):
        """When pull fails with 'not found', stderr should suggest running search."""

        async def fake_pull(name):
            if False:
                yield {}
            raise ValueError(f"Model '{name}' not found in config")

        mock_store.pull = fake_pull
        args = MagicMock(model_name="nonexistent")
        with pytest.raises(SystemExit) as exc_info:
            cmd_models_pull(args)
        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert "olmlx models search" in err


class TestBenchLeaderboardArgs:
    def test_limit_rejects_zero(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["bench", "leaderboard", "--limit", "0"])

    def test_limit_rejects_negative(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["bench", "leaderboard", "--limit", "-3"])

    def test_limit_accepts_positive(self):
        parser = build_parser()
        args = parser.parse_args(["bench", "leaderboard", "--limit", "5"])
        assert args.limit == 5

    def test_bench_dir_parsed(self):
        parser = build_parser()
        args = parser.parse_args(["bench", "leaderboard", "--bench-dir", "/tmp/foo"])
        assert args.bench_dir == "/tmp/foo"

    def test_bench_dir_defaults_to_none(self):
        parser = build_parser()
        args = parser.parse_args(["bench", "leaderboard"])
        assert args.bench_dir is None

    def test_list_bench_dir_parsed(self):
        parser = build_parser()
        args = parser.parse_args(["bench", "list", "--bench-dir", "/tmp/foo"])
        assert args.bench_dir == "/tmp/foo"

    def test_run_bench_dir_stores_on_bench_dir(self):
        parser = build_parser()
        args = parser.parse_args(
            ["bench", "run", "--model", "m", "--bench-dir", "/tmp/foo"]
        )
        assert args.bench_dir == "/tmp/foo"
        # dest must be consistent with list/leaderboard, not output_dir
        assert not hasattr(args, "output_dir")

    def test_run_output_dir_alias_still_works(self):
        parser = build_parser()
        args = parser.parse_args(
            ["bench", "run", "--model", "m", "--output-dir", "/tmp/foo"]
        )
        assert args.bench_dir == "/tmp/foo"


class TestBenchLeaderboardCmd:
    def test_reads_from_custom_bench_dir(self, tmp_path, capsys):
        from olmlx.bench.results import (
            PromptResult,
            RunResult,
            ScenarioResult,
            save_run,
        )
        from olmlx.cli import cmd_bench_leaderboard

        save_run(
            RunResult(
                model="custom-model",
                timestamp="20260101T000000Z",
                git_sha="abc1234",
                scenarios=[
                    ScenarioResult(
                        scenario_name="baseline",
                        scenario_description="d",
                        env_overrides={},
                        prompt_results=[
                            PromptResult(
                                prompt_name="p",
                                category="c",
                                output_text="",
                                status_code=200,
                                eval_count=100,
                                eval_duration_ns=1_000_000_000,
                            )
                        ],
                    )
                ],
            ),
            tmp_path,
        )
        args = MagicMock(bench_dir=str(tmp_path), all_runs=False, limit=None)
        cmd_bench_leaderboard(args)
        out = capsys.readouterr().out
        assert "custom-model" in out

    def test_default_bench_dir_does_not_see_custom(self, tmp_path, capsys):
        from olmlx.cli import cmd_bench_leaderboard

        # No runs in the custom dir, command reports no runs.
        args = MagicMock(bench_dir=str(tmp_path), all_runs=False, limit=None)
        cmd_bench_leaderboard(args)
        out = capsys.readouterr().out
        assert "No bench runs" in out
