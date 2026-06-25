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


def test_default_models_include_whisper():
    from olmlx.cli import DEFAULT_MODELS

    assert DEFAULT_MODELS["whisper-turbo:latest"] == (
        "mlx-community/whisper-large-v3-turbo"
    )
    assert "whisper-large:latest" in DEFAULT_MODELS


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

    def test_plist_filters_secret_env_vars(self, monkeypatch):
        # Secrets must never be persisted into the cleartext launchd plist
        # (~/Library/LaunchAgents is readable by any local process). #454/#2
        monkeypatch.setattr("olmlx.cli.shutil.which", lambda _: "/usr/local/bin/olmlx")
        monkeypatch.setenv("OLMLX_PORT", "9999")
        monkeypatch.setenv("OLMLX_DISTRIBUTED_SECRET", "topsecret")
        monkeypatch.setenv("OLMLX_HF_TOKEN", "hf_xxx")
        monkeypatch.setenv("OLMLX_API_KEY", "key123")
        monkeypatch.setenv("OLMLX_REDIS_PASSWORD", "pw")
        # Token-count config keys must NOT be mistaken for credentials.
        monkeypatch.setenv("OLMLX_SPECULATIVE_TOKENS", "4")
        monkeypatch.setenv("OLMLX_PROMPT_CACHE_MAX_TOKENS", "32768")
        env = _build_plist()["EnvironmentVariables"]
        # Non-sensitive keys still forwarded
        assert env["OLMLX_PORT"] == "9999"
        assert env["OLMLX_SPECULATIVE_TOKENS"] == "4"
        assert env["OLMLX_PROMPT_CACHE_MAX_TOKENS"] == "32768"
        # Sensitive keys filtered out entirely
        assert "OLMLX_DISTRIBUTED_SECRET" not in env
        assert "OLMLX_HF_TOKEN" not in env
        assert "OLMLX_API_KEY" not in env
        assert "OLMLX_REDIS_PASSWORD" not in env


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

    def test_root_parser_has_no_serve_flag_collisions(self):
        """The bare-invocation default synthesis in ``cli_main`` copies
        serve-subparser defaults onto ``args`` only when the attribute
        is missing. That is correct only as long as the root parser's
        attribute names don't overlap with the serve subparser's
        names — otherwise the serve default would be silently skipped.
        Pin the invariant so adding a colliding flag fails CI."""
        parser = build_parser()
        root_attrs = set(vars(parser.parse_args([])))
        # ``command`` is owned by the root parser (subparsers dest).
        root_attrs.discard("command")
        serve_attrs = set(vars(parser.parse_args(["serve"]))) - {"command"}
        assert root_attrs.isdisjoint(serve_attrs), (
            "Root parser and serve subparser share attribute names: "
            f"{root_attrs & serve_attrs}. The bare-invocation default "
            "synthesis in cli_main would silently skip serve defaults "
            "for these names. Rename one side or update cli_main."
        )

    def test_serve_empty_draft_model_rejected(self, capsys):
        """``--speculative-draft-model ""`` should be rejected by argparse,
        not propagate as a Pydantic ``ValidationError`` at startup."""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["serve", "--speculative-draft-model", ""])
        captured = capsys.readouterr()
        assert "non-empty" in captured.err

    def test_serve_draft_model_strips_whitespace(self):
        """A value like ``" hf/path "`` should be accepted but stripped, so
        Settings doesn't end up with a path that has leading/trailing
        spaces (which would later fail with a confusing not-found
        error at model-load time)."""
        parser = build_parser()
        args = parser.parse_args(
            ["serve", "--speculative-draft-model", "  Qwen/Qwen3-0.6B  "]
        )
        assert args.speculative_draft_model == "Qwen/Qwen3-0.6B"

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

    def test_cmd_chat_forwards_legacy_env_vars(self, monkeypatch, capsys):
        """``olmlx chat`` must run the deprecation forwarder too —
        otherwise users who only run chat after upgrading silently lose
        speculative even though ``serve`` honours the legacy names."""
        from olmlx.cli import cmd_chat
        from olmlx.config import settings as _settings

        monkeypatch.setattr(_settings, "speculative", False)
        monkeypatch.setattr(_settings, "speculative_draft_model", None)
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_SPECULATIVE", "true")
        monkeypatch.setenv(
            "OLMLX_EXPERIMENTAL_SPECULATIVE_DRAFT_MODEL", "Qwen/Qwen3-0.6B"
        )
        monkeypatch.delenv("OLMLX_SPECULATIVE", raising=False)
        monkeypatch.delenv("OLMLX_SPECULATIVE_DRAFT_MODEL", raising=False)
        # ``cmd_chat`` calls ``_configure_logging`` which clears caplog's
        # handler — read warnings off stderr instead.

        # Drive cmd_chat to the point right after the forwarder runs by
        # giving it a missing model name — it exits before doing any
        # real model loading.
        ns = MagicMock()
        ns.model_name = None
        with pytest.raises(SystemExit):
            cmd_chat(ns)
        captured = capsys.readouterr()
        assert "Deprecated env vars detected" in captured.err
        assert _settings.speculative is True
        assert _settings.speculative_draft_model == "Qwen/Qwen3-0.6B"

    def test_legacy_dotenv_strips_inline_comments(self, monkeypatch, tmp_path):
        """An unquoted ``.env`` value with a trailing ``# …`` comment
        must parse to just the value. Without comment stripping, the
        boolean forwarder would coerce ``true  # enable`` to False
        and the user's intent would silently invert."""
        from olmlx.cli import _legacy_speculative_values_in_dotenv

        monkeypatch.chdir(tmp_path)
        (tmp_path / ".env").write_text(
            "OLMLX_EXPERIMENTAL_SPECULATIVE=true  # enable speculative\n"
            "OLMLX_EXPERIMENTAL_SPECULATIVE_DRAFT_MODEL=Qwen/Qwen3-0.6B # draft\n"
            'OLMLX_EXPERIMENTAL_SPECULATIVE_TOKENS="6"\n'
        )
        values = _legacy_speculative_values_in_dotenv()
        assert values["OLMLX_EXPERIMENTAL_SPECULATIVE"] == "true"
        assert values["OLMLX_EXPERIMENTAL_SPECULATIVE_DRAFT_MODEL"] == "Qwen/Qwen3-0.6B"
        # Properly-quoted values keep their content verbatim.
        assert values["OLMLX_EXPERIMENTAL_SPECULATIVE_TOKENS"] == "6"

    def test_legacy_dotenv_values_are_forwarded(self, monkeypatch, tmp_path, caplog):
        """A user with the legacy ``OLMLX_EXPERIMENTAL_SPECULATIVE*`` in
        their ``.env`` (not in shell) should still see their config
        forwarded during the deprecation window — pydantic-settings
        reads ``.env`` without touching ``os.environ``."""
        import logging
        import os

        from olmlx.cli import _apply_serve_overrides
        from olmlx.config import settings as _settings

        # Move into tmp_path so the cwd-relative .env scan picks up our
        # fixture and doesn't see the developer's real .env.
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".env").write_text(
            "OLMLX_EXPERIMENTAL_SPECULATIVE=true\n"
            'OLMLX_EXPERIMENTAL_SPECULATIVE_DRAFT_MODEL="Qwen/Qwen3-0.6B"\n'
            "OLMLX_EXPERIMENTAL_SPECULATIVE_TOKENS=7\n"
        )
        # Ensure none of the legacy or new vars are in the shell.
        for v in (
            "OLMLX_EXPERIMENTAL_SPECULATIVE",
            "OLMLX_EXPERIMENTAL_SPECULATIVE_DRAFT_MODEL",
            "OLMLX_EXPERIMENTAL_SPECULATIVE_TOKENS",
            "OLMLX_SPECULATIVE",
            "OLMLX_SPECULATIVE_DRAFT_MODEL",
            "OLMLX_SPECULATIVE_TOKENS",
        ):
            monkeypatch.delenv(v, raising=False)
        # Settings is at default; the .env scan supplies the legacy values.
        monkeypatch.setattr(_settings, "speculative", False)
        monkeypatch.setattr(_settings, "speculative_draft_model", None)
        monkeypatch.setattr(_settings, "speculative_tokens", None)
        monkeypatch.setattr(
            "olmlx.cli._audit_speculative_config",
            lambda registry=None: ([], [], [], [], False),
        )
        monkeypatch.setattr(
            "olmlx.cli._models_with_promoted_keys_in_experimental", lambda: []
        )

        parser = build_parser()
        args = parser.parse_args(["serve"])
        with caplog.at_level(logging.WARNING, logger="olmlx.cli"):
            _apply_serve_overrides(args)

        assert _settings.speculative is True
        assert _settings.speculative_draft_model == "Qwen/Qwen3-0.6B"
        assert _settings.speculative_tokens == 7
        assert "Deprecated env vars detected" in caplog.text
        # Belt-and-braces: confirm we didn't pollute os.environ.
        assert os.environ.get("OLMLX_EXPERIMENTAL_SPECULATIVE") is None

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
        monkeypatch.setattr(_settings, "speculative_tokens", None)
        # Stub out registry-walking helpers so the test is hermetic.
        monkeypatch.setattr(
            "olmlx.cli._audit_speculative_config",
            lambda registry=None: ([], [], [], [], False),
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
        monkeypatch.setattr(_settings, "speculative_tokens", None)
        monkeypatch.setattr(
            "olmlx.cli._audit_speculative_config",
            lambda registry=None: ([], [], [], [], False),
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
        # Settings keeps its prior default (None means "use strategy default").
        assert _settings.speculative_tokens is None

    def test_apply_serve_overrides_new_env_var_wins_over_legacy(self, monkeypatch):
        """When both legacy and new env vars are set, the new one wins."""
        from olmlx.cli import _apply_serve_overrides
        from olmlx.config import settings as _settings

        monkeypatch.setattr(_settings, "speculative", False)
        monkeypatch.setattr(_settings, "speculative_draft_model", "new/draft")
        monkeypatch.setattr(_settings, "speculative_tokens", 4)
        monkeypatch.setattr(
            "olmlx.cli._audit_speculative_config",
            lambda registry=None: ([], [], [], [], False),
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
            lambda registry=None: ([], [], [], [], False),
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
            lambda registry=None: ([], [], [], [], False),
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
            lambda registry=None: ([], [], [], [], False),
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
            "olmlx.cli._audit_speculative_config",
            lambda registry=None: ([], [], [], [], False),
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
            lambda registry=None: (["bad/model:latest"], [], [], [], False),
        )
        monkeypatch.setattr(
            "olmlx.cli._models_with_promoted_keys_in_experimental", lambda: []
        )
        parser = build_parser()
        args = parser.parse_args(["serve"])
        with pytest.raises(SystemExit) as excinfo:
            _apply_serve_overrides(args)
        assert excinfo.value.code == 1

    def test_audit_does_not_flag_proxy_tuning_without_draft(self, monkeypatch):
        """proxy_tuning is draftless (steers via expert/anti-expert), so an
        enabled proxy_tuning model with no speculative_draft_model must NOT be
        flagged as misconfigured."""
        from olmlx.cli import _audit_speculative_config
        from olmlx.engine.registry import ModelConfig, ModelRegistry

        reg = ModelRegistry()
        reg._mappings = {
            "org/base:latest": ModelConfig(
                hf_path="org/base",
                speculative=True,
                speculative_strategy="proxy_tuning",
            ),
            # Contrast: classic with no draft IS still flagged.
            "org/classic:latest": ModelConfig(
                hf_path="org/classic",
                speculative=True,
                speculative_strategy="classic",
            ),
        }
        bad, _dormant, _flash, _dflash_moe, _global_used = _audit_speculative_config(
            reg
        )
        assert "org/base:latest" not in bad
        assert "org/classic:latest" in bad

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
        assert excinfo.value.code == 1

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
            "olmlx.cli._audit_speculative_config",
            lambda registry=None: ([], [], [], [], False),
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

    def test_flash_conflict_warning_suppressed_for_bad_models(
        self, monkeypatch, caplog
    ):
        """A model that's both ``bad`` (missing draft) and a flash-conflict
        should only surface the missing-draft error — the
        ``use flash_speculative`` warning would be misleading."""
        import logging

        from olmlx.cli import _apply_serve_overrides
        from olmlx.config import settings as _settings

        monkeypatch.setattr(_settings, "speculative", False)
        monkeypatch.setattr(_settings, "speculative_draft_model", None)
        monkeypatch.setattr(
            "olmlx.cli._audit_speculative_config",
            lambda registry=None: (["m:latest"], [], ["m:latest"], [], False),
        )
        monkeypatch.setattr(
            "olmlx.cli._models_with_promoted_keys_in_experimental", lambda: []
        )
        parser = build_parser()
        args = parser.parse_args(["serve"])
        with caplog.at_level(logging.WARNING, logger="olmlx.cli"):
            with pytest.raises(SystemExit):
                _apply_serve_overrides(args)
        # No flash-conflict warning when the model is also missing its
        # draft. ``flash_speculative`` is the unique substring of the
        # flash-conflict warning, so its absence rules the warning out.
        assert "flash_speculative" not in caplog.text

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
            lambda registry=None: ([], [], ["flash/model:latest"], [], False),
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
            lambda registry=None: ([], ["dormant/model:latest"], [], [], False),
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
        # Flash primary knob was promoted out of ``experimental`` —
        # set it on ``settings`` so ``mc.resolved_flash().enabled`` is True.
        monkeypatch.setattr(_settings, "flash", True)

        bad, dormant, flash_conflicts, dflash_moe_conflicts, _global_used = (
            _audit_speculative_config()
        )
        assert bad == []
        assert dormant == []
        assert flash_conflicts == ["global-flash/m:latest"]
        assert dflash_moe_conflicts == []

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
                        "flash": True,
                    },
                }
            )
        )
        monkeypatch.setattr(_settings, "models_config", models_json)
        monkeypatch.setattr(_settings, "speculative", False)
        monkeypatch.setattr(_settings, "speculative_draft_model", None)

        bad, dormant, flash_conflicts, dflash_moe_conflicts, global_used = (
            _audit_speculative_config()
        )
        # Compare as sets so the test isn't coupled to the registry's
        # internal iteration order.
        assert set(bad) == {"bad/no-draft:latest"}
        assert set(dormant) == {"dormant/has-draft:latest"}
        assert set(flash_conflicts) == {"conflict/flash-and-spec:latest"}
        assert dflash_moe_conflicts == []
        # No model in this fixture uses the global draft (good/a has its own).
        assert global_used is False

    def test_audit_dflags_flash_moe_conflicts(self, monkeypatch, tmp_path):
        """dflash + Flash-MoE populates the dflash_moe_conflicts bucket."""
        from olmlx.cli import _audit_speculative_config
        from olmlx.config import settings as _settings

        models_json = tmp_path / "models.json"
        models_json.write_text(
            json.dumps(
                {
                    "moe-dflash/m:latest": {
                        "hf_path": "moe-dflash/m",
                        "speculative": True,
                        "speculative_strategy": "dflash",
                        "speculative_draft_model": "moe-dflash/draft",
                        "flash_moe": True,
                    },
                }
            )
        )
        monkeypatch.setattr(_settings, "models_config", models_json)
        monkeypatch.setattr(_settings, "speculative", False)
        monkeypatch.setattr(_settings, "speculative_draft_model", None)

        bad, dormant, flash, dflash_moe, global_used = _audit_speculative_config()
        assert bad == []
        assert dormant == []
        assert flash == []
        assert dflash_moe == ["moe-dflash/m:latest"]
        assert global_used is False

    def test_audit_eagle_flash_moe_conflicts(self, monkeypatch, tmp_path):
        """eagle + Flash-MoE also populates the dflash_moe_conflicts bucket."""
        from olmlx.cli import _audit_speculative_config
        from olmlx.config import settings as _settings

        models_json = tmp_path / "models.json"
        models_json.write_text(
            json.dumps(
                {
                    "moe-eagle/m:latest": {
                        "hf_path": "moe-eagle/m",
                        "speculative": True,
                        "speculative_strategy": "eagle",
                        "speculative_draft_model": "moe-eagle/draft",
                        "flash_moe": True,
                    },
                }
            )
        )
        monkeypatch.setattr(_settings, "models_config", models_json)
        monkeypatch.setattr(_settings, "speculative", False)
        monkeypatch.setattr(_settings, "speculative_draft_model", None)

        bad, dormant, flash, dflash_moe, global_used = _audit_speculative_config()
        assert bad == []
        assert dormant == []
        assert flash == []
        assert dflash_moe == ["moe-eagle/m:latest"]
        assert global_used is False

    def test_audit_pld_not_flagged_as_bad_without_draft(self, monkeypatch, tmp_path):
        """PLD strategy needs no draft model — must not appear in ``bad``."""
        from olmlx.cli import _audit_speculative_config
        from olmlx.config import settings as _settings

        models_json = tmp_path / "models.json"
        models_json.write_text(
            json.dumps(
                {
                    "pld/m:latest": {
                        "hf_path": "pld/m",
                        "speculative": True,
                        "speculative_strategy": "pld",
                    },
                }
            )
        )
        monkeypatch.setattr(_settings, "models_config", models_json)
        monkeypatch.setattr(_settings, "speculative", False)
        monkeypatch.setattr(_settings, "speculative_draft_model", None)

        bad, dormant, flash, dflash_moe, global_used = _audit_speculative_config()
        assert bad == [], "pld model should not be flagged as bad (no draft needed)"
        assert dormant == []

    def test_per_model_flash_mismatch_in_distributed_exits(
        self, monkeypatch, tmp_path, capsys
    ):
        """Per-model ``flash: true`` while ``settings.flash=False`` (or
        vice versa) under distributed mode must hard-exit at startup —
        the coordinator and workers would load structurally different
        models and crash the ring all_sum at first inference."""
        from olmlx.cli import _audit_per_model_flash_in_distributed
        from olmlx.config import settings as _settings

        models_json = tmp_path / "models.json"
        models_json.write_text(
            json.dumps(
                {
                    "qwen/m:latest": {
                        "hf_path": "qwen/m",
                        "flash": True,
                    },
                }
            )
        )
        monkeypatch.setattr(_settings, "models_config", models_json)
        monkeypatch.setattr(_settings, "distributed", True)
        monkeypatch.setattr(_settings, "flash", False)

        with pytest.raises(SystemExit) as exc_info:
            _audit_per_model_flash_in_distributed()
        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert "qwen/m:latest" in err
        assert "structurally different models" in err

    def test_per_model_flash_numeric_only_in_distributed_warns(
        self, monkeypatch, tmp_path, caplog
    ):
        """A per-model numeric override (e.g. ``flash_sparsity_threshold``)
        on a model whose resolved ``flash`` is True logs a warning —
        the registry isn't consulted on workers, so the override is
        silently dropped on every rank but the coordinator."""
        import logging

        from olmlx.cli import _audit_per_model_flash_in_distributed
        from olmlx.config import settings as _settings

        models_json = tmp_path / "models.json"
        models_json.write_text(
            json.dumps(
                {
                    "qwen/m:latest": {
                        "hf_path": "qwen/m",
                        "flash_sparsity_threshold": 0.3,
                    },
                }
            )
        )
        monkeypatch.setattr(_settings, "models_config", models_json)
        monkeypatch.setattr(_settings, "distributed", True)
        # Flash globally on so the model resolves to ``enabled=True``;
        # the per-model numeric override is the silent-drop case.
        monkeypatch.setattr(_settings, "flash", True)

        with caplog.at_level(logging.WARNING, logger="olmlx.cli"):
            _audit_per_model_flash_in_distributed()

        assert "per-model Flash numeric overrides" in caplog.text
        assert "qwen/m:latest" in caplog.text
        # The warning names the specific fields the operator needs to
        # promote globally, not just the model entries.
        assert "flash_sparsity_threshold" in caplog.text

    def test_per_model_flash_numeric_inert_when_flash_disabled(
        self, monkeypatch, tmp_path, caplog
    ):
        """A per-model numeric override on a model that resolves to
        ``flash=False`` is inert on both coordinator and worker — no
        warning, no false positive."""
        import logging

        from olmlx.cli import _audit_per_model_flash_in_distributed
        from olmlx.config import settings as _settings

        models_json = tmp_path / "models.json"
        models_json.write_text(
            json.dumps(
                {
                    "qwen/m:latest": {
                        "hf_path": "qwen/m",
                        "flash_sparsity_threshold": 0.3,
                    },
                }
            )
        )
        monkeypatch.setattr(_settings, "models_config", models_json)
        monkeypatch.setattr(_settings, "distributed", True)
        monkeypatch.setattr(_settings, "flash", False)

        with caplog.at_level(logging.WARNING, logger="olmlx.cli"):
            _audit_per_model_flash_in_distributed()

        assert "per-model Flash numeric overrides" not in caplog.text

    def test_per_model_flash_silent_without_distributed(
        self, monkeypatch, tmp_path, caplog, capsys
    ):
        """No diagnostics when distributed is off, even with per-model flash set."""
        import logging

        from olmlx.cli import _audit_per_model_flash_in_distributed
        from olmlx.config import settings as _settings

        models_json = tmp_path / "models.json"
        models_json.write_text(
            json.dumps({"qwen/m:latest": {"hf_path": "qwen/m", "flash": True}})
        )
        monkeypatch.setattr(_settings, "models_config", models_json)
        monkeypatch.setattr(_settings, "distributed", False)

        with caplog.at_level(logging.WARNING, logger="olmlx.cli"):
            _audit_per_model_flash_in_distributed()
        # No exit, no warning.
        assert "per-model Flash" not in caplog.text
        assert "structurally different models" not in capsys.readouterr().err

    def test_audit_moe_check_survives_flash_resolution_failure(
        self, monkeypatch, tmp_path
    ):
        """A model with both flash-MoE/dflash *and* an inverted flash range
        must still be caught by the MoE conflict check, even though
        ``resolved_flash()`` raises on the inverted range.

        Regression test for the ``continue`` that previously short-
        circuited both checks together when either resolver failed.
        """
        from olmlx.cli import _audit_speculative_config
        from olmlx.config import settings as _settings

        models_json = tmp_path / "models.json"
        models_json.write_text(
            json.dumps(
                {
                    "moe-dflash-bad/m:latest": {
                        "hf_path": "moe-dflash-bad/m",
                        "speculative": True,
                        "speculative_strategy": "dflash",
                        "speculative_draft_model": "moe-dflash-bad/draft",
                        # Per-model min crosses the (default) global max
                        # → ``resolved_flash()`` raises "inverted range".
                        "flash_min_active_neurons": 1_000_000,
                        "flash_moe": True,
                    },
                }
            )
        )
        monkeypatch.setattr(_settings, "models_config", models_json)
        monkeypatch.setattr(_settings, "speculative", False)
        monkeypatch.setattr(_settings, "speculative_draft_model", None)
        monkeypatch.setattr(_settings, "flash_min_active_neurons", 32)
        monkeypatch.setattr(_settings, "flash_max_active_neurons", 100)

        bad, dormant, flash, dflash_moe, _global_used = _audit_speculative_config()
        # Flash list is empty (resolution failed) but MoE check still
        # populated the dflash_moe bucket.
        assert flash == []
        assert dflash_moe == ["moe-dflash-bad/m:latest"]

    def test_audit_flash_conflict_skipped_when_resolved_flash_raises(
        self, monkeypatch, tmp_path, caplog
    ):
        """A model with ``speculative=true`` plus a flash configuration
        that makes ``resolved_flash()`` raise (e.g. per-model min above
        the global max) is intentionally excluded from
        ``flash_conflicts``. The startup audit logs a "Skipping flash
        conflict check" warning so the operator sees the resolution
        failure; the conflict itself surfaces at model load time when
        ``resolved_flash`` is called again and raises a clear
        ``ValueError``.

        Regression test pinning the intended behaviour — without it a
        future reader could "fix" the silent skip by speculatively
        adding the model to ``flash_conflicts`` even when its flash
        state is unknown.
        """
        import logging

        from olmlx.cli import _audit_speculative_config
        from olmlx.config import settings as _settings

        models_json = tmp_path / "models.json"
        models_json.write_text(
            json.dumps(
                {
                    "flash-bad/m:latest": {
                        "hf_path": "flash-bad/m",
                        "speculative": True,
                        "speculative_draft_model": "flash-bad/draft",
                        # Per-model min crosses the global max →
                        # ``resolved_flash()`` raises "inverted range".
                        "flash_min_active_neurons": 1_000_000,
                    },
                }
            )
        )
        monkeypatch.setattr(_settings, "models_config", models_json)
        monkeypatch.setattr(_settings, "speculative", False)
        monkeypatch.setattr(_settings, "speculative_draft_model", None)
        monkeypatch.setattr(_settings, "flash_min_active_neurons", 32)
        monkeypatch.setattr(_settings, "flash_max_active_neurons", 100)

        with caplog.at_level(logging.WARNING, logger="olmlx.cli"):
            bad, dormant, flash, dflash_moe, _global_used = _audit_speculative_config()

        # Flash conflict NOT reported (resolution failed, intent is to
        # skip rather than guess) but the warning is emitted.
        assert flash == []
        assert "Skipping flash conflict check" in caplog.text

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

    # ── Legacy kv_cache_quant forwarding tests ────────────────────────

    def test_legacy_kv_cache_quant_dotenv_parsing(self, monkeypatch, tmp_path):
        """_legacy_kv_cache_quant_in_dotenv reads the value from .env."""
        from olmlx.cli import _legacy_kv_cache_quant_in_dotenv

        monkeypatch.chdir(tmp_path)
        (tmp_path / ".env").write_text(
            "OLMLX_EXPERIMENTAL_KV_CACHE_QUANT=turboquant:4\n"
        )
        assert _legacy_kv_cache_quant_in_dotenv() == "turboquant:4"

    def test_legacy_kv_cache_quant_dotenv_missing(self, monkeypatch, tmp_path):
        """_legacy_kv_cache_quant_in_dotenv returns None when .env is absent."""
        from olmlx.cli import _legacy_kv_cache_quant_in_dotenv

        monkeypatch.chdir(tmp_path)
        assert _legacy_kv_cache_quant_in_dotenv() is None

    def test_legacy_kv_cache_quant_dotenv_not_this_key(self, monkeypatch, tmp_path):
        """_legacy_kv_cache_quant_in_dotenv ignores unrelated keys."""
        from olmlx.cli import _legacy_kv_cache_quant_in_dotenv

        monkeypatch.chdir(tmp_path)
        (tmp_path / ".env").write_text("OTHER_VAR=hello\n")
        assert _legacy_kv_cache_quant_in_dotenv() is None

    def test_legacy_kv_cache_quant_shell_env_forwarding(self, monkeypatch, caplog):
        """Legacy OLMLX_EXPERIMENTAL_KV_CACHE_QUANT in the shell is forwarded."""
        import logging

        from olmlx.cli import _surface_legacy_kv_cache_quant_env
        from olmlx.config import settings as _settings

        monkeypatch.setattr(_settings, "kv_cache_quant", None)
        monkeypatch.delenv("OLMLX_KV_CACHE_QUANT", raising=False)
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_KV_CACHE_QUANT", "turboquant:4")

        with caplog.at_level(logging.WARNING, logger="olmlx.cli"):
            _surface_legacy_kv_cache_quant_env()

        assert _settings.kv_cache_quant == "turboquant:4"
        assert "Forwarding legacy OLMLX_EXPERIMENTAL_KV_CACHE_QUANT" in caplog.text

    def test_legacy_kv_cache_quant_dotenv_forwarding(
        self, monkeypatch, tmp_path, caplog
    ):
        """Legacy OLMLX_EXPERIMENTAL_KV_CACHE_QUANT in .env (not shell) is forwarded."""
        import logging

        from olmlx.cli import _surface_legacy_kv_cache_quant_env
        from olmlx.config import settings as _settings

        monkeypatch.chdir(tmp_path)
        (tmp_path / ".env").write_text(
            "OLMLX_EXPERIMENTAL_KV_CACHE_QUANT=turboquant:2\n"
        )
        monkeypatch.setattr(_settings, "kv_cache_quant", None)
        monkeypatch.delenv("OLMLX_KV_CACHE_QUANT", raising=False)
        monkeypatch.delenv("OLMLX_EXPERIMENTAL_KV_CACHE_QUANT", raising=False)

        with caplog.at_level(logging.WARNING, logger="olmlx.cli"):
            _surface_legacy_kv_cache_quant_env()

        assert _settings.kv_cache_quant == "turboquant:2"
        assert "Forwarding legacy OLMLX_EXPERIMENTAL_KV_CACHE_QUANT" in caplog.text

    def test_legacy_kv_cache_quant_new_env_var_wins(self, monkeypatch, caplog):
        """When both legacy and new env vars are set, the new one wins."""
        import logging

        from olmlx.cli import _surface_legacy_kv_cache_quant_env
        from olmlx.config import settings as _settings

        monkeypatch.setattr(_settings, "kv_cache_quant", "turboquant:4")
        monkeypatch.setenv("OLMLX_KV_CACHE_QUANT", "turboquant:4")
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_KV_CACHE_QUANT", "turboquant:2")

        with caplog.at_level(logging.WARNING, logger="olmlx.cli"):
            _surface_legacy_kv_cache_quant_env()

        # New env var value retained; legacy not forwarded.
        assert _settings.kv_cache_quant == "turboquant:4"
        assert "Forwarding legacy" not in caplog.text

    def test_legacy_kv_cache_quant_bad_value_swallowed(self, monkeypatch, caplog):
        """A legacy value that fails validation must not block startup."""
        import logging

        from olmlx.cli import _surface_legacy_kv_cache_quant_env
        from olmlx.config import settings as _settings

        monkeypatch.setattr(_settings, "kv_cache_quant", None)
        monkeypatch.delenv("OLMLX_KV_CACHE_QUANT", raising=False)
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_KV_CACHE_QUANT", "bogus:99")

        with caplog.at_level(logging.WARNING, logger="olmlx.cli"):
            _surface_legacy_kv_cache_quant_env()

        assert "Could not forward legacy env var" in caplog.text
        assert _settings.kv_cache_quant is None

    def test_serve_kv_cache_quant_flag(self):
        """CLI --kv-cache-quant flag is parsed correctly."""
        parser = build_parser()
        args = parser.parse_args(["serve", "--kv-cache-quant", "turboquant:4"])
        assert args.kv_cache_quant == "turboquant:4"

    def test_serve_flash_flag_default_none(self):
        """Bare ``olmlx serve`` leaves ``args.flash`` as None so the env-var
        value wins (no implicit CLI override)."""
        parser = build_parser()
        args = parser.parse_args(["serve"])
        assert args.flash is None

    def test_serve_flash_flag_enable(self):
        parser = build_parser()
        args = parser.parse_args(["serve", "--flash"])
        assert args.flash is True

    def test_serve_no_flash_flag(self):
        parser = build_parser()
        args = parser.parse_args(["serve", "--no-flash"])
        assert args.flash is False

    def test_apply_serve_overrides_no_flash_overrides_env(self, monkeypatch):
        """``--no-flash`` on the CLI overrides ``OLMLX_FLASH=true``."""
        from olmlx.cli import _apply_serve_overrides
        from olmlx.config import settings as _settings

        # Stub the audit helpers so the test exercises just the flash
        # branch in ``_apply_serve_overrides`` without hitting the
        # registry / file system. Mirrors the pattern used by the other
        # ``_apply_serve_overrides`` tests in this class.
        monkeypatch.setattr(
            "olmlx.cli._audit_speculative_config",
            lambda registry=None: ([], [], [], [], False),
        )
        monkeypatch.setattr(
            "olmlx.cli._models_with_promoted_keys_in_experimental", lambda: []
        )
        monkeypatch.setattr(
            "olmlx.cli._audit_per_model_flash_in_distributed",
            lambda registry=None: None,
        )

        parser = build_parser()
        monkeypatch.setattr(_settings, "flash", True)
        args = parser.parse_args(["serve", "--no-flash"])
        _apply_serve_overrides(args)
        assert _settings.flash is False

    def test_apply_serve_overrides_flash_overrides_env(self, monkeypatch):
        """``--flash`` on the CLI overrides ``OLMLX_FLASH=false`` (the default)."""
        from olmlx.cli import _apply_serve_overrides
        from olmlx.config import settings as _settings

        monkeypatch.setattr(
            "olmlx.cli._audit_speculative_config",
            lambda registry=None: ([], [], [], [], False),
        )
        monkeypatch.setattr(
            "olmlx.cli._models_with_promoted_keys_in_experimental", lambda: []
        )
        monkeypatch.setattr(
            "olmlx.cli._audit_per_model_flash_in_distributed",
            lambda registry=None: None,
        )

        parser = build_parser()
        monkeypatch.setattr(_settings, "flash", False)
        args = parser.parse_args(["serve", "--flash"])
        _apply_serve_overrides(args)
        assert _settings.flash is True

    def test_legacy_flash_enable_forwarded(self, monkeypatch, caplog):
        """OLMLX_EXPERIMENTAL_FLASH=true → settings.flash=True with a warning."""
        import logging

        from olmlx.cli import _surface_legacy_flash_env
        from olmlx.config import settings as _settings

        monkeypatch.setattr(_settings, "flash", False)
        monkeypatch.delenv("OLMLX_FLASH", raising=False)
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_FLASH", "true")

        with caplog.at_level(logging.WARNING, logger="olmlx.config"):
            _surface_legacy_flash_env()

        assert _settings.flash is True
        assert "OLMLX_EXPERIMENTAL_FLASH" in caplog.text

    def test_legacy_flash_numeric_knobs_forwarded(self, monkeypatch, caplog):
        """Each promoted numeric knob is forwarded from its legacy name."""
        import logging

        from olmlx.cli import _surface_legacy_flash_env
        from olmlx.config import Settings, settings as _settings

        # Reset all five primary knobs to schema defaults so the legacy
        # forwarder considers the live Settings "unset" for each one.
        defaults = Settings.model_fields
        monkeypatch.setattr(_settings, "flash", defaults["flash"].default)
        monkeypatch.setattr(
            _settings,
            "flash_sparsity_threshold",
            defaults["flash_sparsity_threshold"].default,
        )
        monkeypatch.setattr(
            _settings,
            "flash_min_active_neurons",
            defaults["flash_min_active_neurons"].default,
        )
        monkeypatch.setattr(
            _settings,
            "flash_max_active_neurons",
            defaults["flash_max_active_neurons"].default,
        )
        monkeypatch.setattr(
            _settings,
            "flash_memory_budget_fraction",
            defaults["flash_memory_budget_fraction"].default,
        )
        for new_name in (
            "OLMLX_FLASH",
            "OLMLX_FLASH_SPARSITY_THRESHOLD",
            "OLMLX_FLASH_MIN_ACTIVE_NEURONS",
            "OLMLX_FLASH_MAX_ACTIVE_NEURONS",
            "OLMLX_FLASH_MEMORY_BUDGET_FRACTION",
        ):
            monkeypatch.delenv(new_name, raising=False)
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_FLASH_SPARSITY_THRESHOLD", "0.3")
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_FLASH_MIN_ACTIVE_NEURONS", "64")
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_FLASH_MAX_ACTIVE_NEURONS", "256")
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_FLASH_MEMORY_BUDGET_FRACTION", "0.4")

        with caplog.at_level(logging.WARNING, logger="olmlx.config"):
            _surface_legacy_flash_env()

        assert _settings.flash_sparsity_threshold == 0.3
        assert _settings.flash_min_active_neurons == 64
        assert _settings.flash_max_active_neurons == 256
        assert _settings.flash_memory_budget_fraction == 0.4

    def test_legacy_flash_new_env_var_wins(self, monkeypatch, caplog):
        """When the new OLMLX_FLASH* env var is set, the legacy value is
        fully shadowed: Settings keeps its current value and the banner
        is suppressed (nothing to migrate)."""
        import logging

        from olmlx.cli import _surface_legacy_flash_env
        from olmlx.config import settings as _settings

        monkeypatch.setattr(_settings, "flash", True)
        monkeypatch.setenv("OLMLX_FLASH", "true")
        # Legacy says false; new says true — without precedence, the
        # legacy value would clobber back to False.
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_FLASH", "false")
        for legacy in (
            "OLMLX_EXPERIMENTAL_FLASH_SPARSITY_THRESHOLD",
            "OLMLX_EXPERIMENTAL_FLASH_MIN_ACTIVE_NEURONS",
            "OLMLX_EXPERIMENTAL_FLASH_MAX_ACTIVE_NEURONS",
            "OLMLX_EXPERIMENTAL_FLASH_MEMORY_BUDGET_FRACTION",
        ):
            monkeypatch.delenv(legacy, raising=False)

        with caplog.at_level(logging.WARNING, logger="olmlx.config"):
            _surface_legacy_flash_env()

        # New env var explicitly set → live Settings retains its
        # current value, not the legacy "false".
        assert _settings.flash is True
        # The new env var fully shadows the legacy one; no banner.
        assert "Deprecated env vars detected" not in caplog.text

    def test_legacy_flash_no_op_when_unset(self, monkeypatch):
        """No legacy env vars set → no Settings mutation, no warning."""
        from olmlx.cli import _surface_legacy_flash_env
        from olmlx.config import settings as _settings

        for name in (
            "OLMLX_EXPERIMENTAL_FLASH",
            "OLMLX_EXPERIMENTAL_FLASH_SPARSITY_THRESHOLD",
            "OLMLX_EXPERIMENTAL_FLASH_MIN_ACTIVE_NEURONS",
            "OLMLX_EXPERIMENTAL_FLASH_MAX_ACTIVE_NEURONS",
            "OLMLX_EXPERIMENTAL_FLASH_MEMORY_BUDGET_FRACTION",
        ):
            monkeypatch.delenv(name, raising=False)
        before = _settings.flash
        _surface_legacy_flash_env()
        assert _settings.flash == before

    def test_legacy_flash_inverted_neuron_range_drops_both(self, monkeypatch, caplog):
        """An inverted legacy min/max pair is detected before any setattr,
        so *both* legacy values are dropped — applying only one would
        leave the other at its default and silently remove the user's
        intended ceiling/floor. Verified end-state: Settings retains
        defaults for both fields."""
        import logging

        from olmlx.cli import _surface_legacy_flash_env
        from olmlx.config import Settings, settings as _settings

        defaults = Settings.model_fields
        monkeypatch.setattr(_settings, "flash", defaults["flash"].default)
        default_min = defaults["flash_min_active_neurons"].default
        default_max = defaults["flash_max_active_neurons"].default
        monkeypatch.setattr(_settings, "flash_min_active_neurons", default_min)
        monkeypatch.setattr(_settings, "flash_max_active_neurons", default_max)
        for new_name in (
            "OLMLX_FLASH",
            "OLMLX_FLASH_MIN_ACTIVE_NEURONS",
            "OLMLX_FLASH_MAX_ACTIVE_NEURONS",
        ):
            monkeypatch.delenv(new_name, raising=False)
        # Inverted pair: min=200 > max=100.
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_FLASH_MIN_ACTIVE_NEURONS", "200")
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_FLASH_MAX_ACTIVE_NEURONS", "100")

        with caplog.at_level(logging.WARNING, logger="olmlx.config"):
            _surface_legacy_flash_env()

        # Neither value was applied — Settings retains its defaults.
        assert _settings.flash_min_active_neurons == default_min
        assert _settings.flash_max_active_neurons == default_max
        # A specific warning explains why both were dropped.
        assert "min > max" in caplog.text
        # Neither legacy var appears in the bulk migration banner.
        banner_lines = [
            line
            for line in caplog.text.splitlines()
            if "Deprecated env vars detected" in line
        ]
        for line in banner_lines:
            assert "OLMLX_EXPERIMENTAL_FLASH_MIN_ACTIVE_NEURONS" not in line
            assert "OLMLX_EXPERIMENTAL_FLASH_MAX_ACTIVE_NEURONS" not in line

    def test_legacy_flash_pending_min_below_live_max_drops_both(
        self, monkeypatch, caplog
    ):
        """Case (b): only legacy *min* is pending; the live ``settings``
        max (e.g. set in ``.env`` or shell env under the new name) is
        below the pending min. The pre-check must catch this via the
        ``effective_min/max`` combination and drop both legacy entries —
        without it, the legacy min would land first and the live max
        would silently disappear from the user's mental model."""
        import logging

        from olmlx.cli import _surface_legacy_flash_env
        from olmlx.config import Settings, settings as _settings

        defaults = Settings.model_fields
        default_min = defaults["flash_min_active_neurons"].default
        monkeypatch.setattr(_settings, "flash", defaults["flash"].default)
        # Live state: min at schema default (so the legacy min would
        # otherwise be forwarded), max=150 — a consistent live pair
        # set via ``.env``/shell under the new name. Pending legacy
        # min is 200, which conflicts with the live max=150.
        monkeypatch.setattr(_settings, "flash_min_active_neurons", default_min)
        monkeypatch.setattr(_settings, "flash_max_active_neurons", 150)
        # OLMLX_FLASH_MAX_* is "set" from the shim's perspective so the
        # max forward branch is short-circuited before it gets here.
        monkeypatch.setenv("OLMLX_FLASH_MAX_ACTIVE_NEURONS", "150")
        for new_name in ("OLMLX_FLASH", "OLMLX_FLASH_MIN_ACTIVE_NEURONS"):
            monkeypatch.delenv(new_name, raising=False)
        monkeypatch.delenv("OLMLX_EXPERIMENTAL_FLASH_MAX_ACTIVE_NEURONS", raising=False)
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_FLASH_MIN_ACTIVE_NEURONS", "200")

        with caplog.at_level(logging.WARNING, logger="olmlx.config"):
            _surface_legacy_flash_env()

        # Legacy min was rejected; live state is unchanged.
        assert _settings.flash_min_active_neurons == default_min
        assert _settings.flash_max_active_neurons == 150
        assert "min > max" in caplog.text

    def test_legacy_flash_pending_max_below_live_min_drops_both(
        self, monkeypatch, caplog
    ):
        """Case (c): only legacy *max* is pending; the live ``settings``
        min is above the pending max. Symmetric to case (b)."""
        import logging

        from olmlx.cli import _surface_legacy_flash_env
        from olmlx.config import Settings, settings as _settings

        defaults = Settings.model_fields
        default_max = defaults["flash_max_active_neurons"].default
        monkeypatch.setattr(_settings, "flash", defaults["flash"].default)
        # Live min set to 200 (e.g. via OLMLX_FLASH_MIN_ACTIVE_NEURONS
        # in shell or .env). Pending legacy max is 100.
        monkeypatch.setattr(_settings, "flash_min_active_neurons", 200)
        monkeypatch.setattr(_settings, "flash_max_active_neurons", default_max)
        monkeypatch.setenv("OLMLX_FLASH_MIN_ACTIVE_NEURONS", "200")
        for new_name in ("OLMLX_FLASH", "OLMLX_FLASH_MAX_ACTIVE_NEURONS"):
            monkeypatch.delenv(new_name, raising=False)
        monkeypatch.delenv("OLMLX_EXPERIMENTAL_FLASH_MIN_ACTIVE_NEURONS", raising=False)
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_FLASH_MAX_ACTIVE_NEURONS", "100")

        with caplog.at_level(logging.WARNING, logger="olmlx.config"):
            _surface_legacy_flash_env()

        assert _settings.flash_min_active_neurons == 200
        assert _settings.flash_max_active_neurons == default_max
        assert "min > max" in caplog.text

    def test_legacy_flash_out_of_range_value_logs_failure_only(
        self, monkeypatch, caplog
    ):
        """A legacy value that parses but fails a single-field Pydantic
        validator (e.g. ``Field(le=1.0)``) is dropped with the per-field
        ``Could not forward`` log. The migration banner is suppressed
        because nothing actually landed — listing the var as "rename
        this" would just point the user at the same validator.

        Documents the current behaviour: the per-field log is the only
        signal the user sees in this case.
        """
        import logging

        from olmlx.cli import _surface_legacy_flash_env
        from olmlx.config import Settings, settings as _settings

        defaults = Settings.model_fields
        monkeypatch.setattr(_settings, "flash", defaults["flash"].default)
        monkeypatch.setattr(
            _settings,
            "flash_sparsity_threshold",
            defaults["flash_sparsity_threshold"].default,
        )
        for new_name in (
            "OLMLX_FLASH",
            "OLMLX_FLASH_SPARSITY_THRESHOLD",
            "OLMLX_FLASH_MIN_ACTIVE_NEURONS",
            "OLMLX_FLASH_MAX_ACTIVE_NEURONS",
            "OLMLX_FLASH_MEMORY_BUDGET_FRACTION",
        ):
            monkeypatch.delenv(new_name, raising=False)
        for legacy in (
            "OLMLX_EXPERIMENTAL_FLASH",
            "OLMLX_EXPERIMENTAL_FLASH_MIN_ACTIVE_NEURONS",
            "OLMLX_EXPERIMENTAL_FLASH_MAX_ACTIVE_NEURONS",
            "OLMLX_EXPERIMENTAL_FLASH_MEMORY_BUDGET_FRACTION",
        ):
            monkeypatch.delenv(legacy, raising=False)
        # 1.5 parses as float but fails ``Field(gt=0, le=1.0)``.
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_FLASH_SPARSITY_THRESHOLD", "1.5")

        with caplog.at_level(logging.WARNING, logger="olmlx.config"):
            _surface_legacy_flash_env()

        # Per-field failure logged, Settings stays at default, banner
        # NOT emitted.
        assert "Could not forward legacy env var" in caplog.text
        assert (
            _settings.flash_sparsity_threshold
            == defaults["flash_sparsity_threshold"].default
        )
        assert "Deprecated env vars detected" not in caplog.text

    def test_legacy_flash_false_value_does_not_warn(self, monkeypatch, caplog):
        """``OLMLX_EXPERIMENTAL_FLASH=false`` parses to the schema default
        (``flash=False``) so the deprecation banner should NOT fire — there
        is nothing for the user to migrate."""
        import logging

        from olmlx.cli import _surface_legacy_flash_env
        from olmlx.config import settings as _settings

        monkeypatch.setattr(_settings, "flash", False)
        monkeypatch.delenv("OLMLX_FLASH", raising=False)
        for legacy in (
            "OLMLX_EXPERIMENTAL_FLASH_SPARSITY_THRESHOLD",
            "OLMLX_EXPERIMENTAL_FLASH_MIN_ACTIVE_NEURONS",
            "OLMLX_EXPERIMENTAL_FLASH_MAX_ACTIVE_NEURONS",
            "OLMLX_EXPERIMENTAL_FLASH_MEMORY_BUDGET_FRACTION",
        ):
            monkeypatch.delenv(legacy, raising=False)
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_FLASH", "false")

        with caplog.at_level(logging.WARNING, logger="olmlx.config"):
            _surface_legacy_flash_env()

        # Nothing migrated, no warning.
        assert _settings.flash is False
        assert "Deprecated env vars detected" not in caplog.text

    def test_kv_cache_quant_disk_incompat_warning(self, monkeypatch, caplog):
        """prompt_cache_disk + kv_cache_quant together produce a warning."""
        import logging

        from olmlx.cli import _warn_kv_cache_quant_incompatibilities
        from olmlx.config import settings as _settings

        monkeypatch.setattr(_settings, "prompt_cache_disk", True)
        monkeypatch.setattr(_settings, "kv_cache_quant", "turboquant:4")

        with caplog.at_level(logging.WARNING, logger="olmlx.cli"):
            _warn_kv_cache_quant_incompatibilities()

        assert "Prompt cache disk offload" in caplog.text
        assert "turboquant:4" in caplog.text

    def test_kv_cache_quant_disk_incompat_no_warning_when_disabled(
        self, monkeypatch, caplog
    ):
        """No warning when either setting is off."""
        import logging

        from olmlx.cli import _warn_kv_cache_quant_incompatibilities
        from olmlx.config import settings as _settings

        monkeypatch.setattr(_settings, "prompt_cache_disk", False)
        monkeypatch.setattr(_settings, "kv_cache_quant", "turboquant:4")

        with caplog.at_level(logging.WARNING, logger="olmlx.cli"):
            _warn_kv_cache_quant_incompatibilities()

        assert "Prompt cache disk offload" not in caplog.text

    def test_legacy_kv_cache_quant_dotenv_quoted_with_comment(
        self, monkeypatch, tmp_path
    ):
        """Quoted value with trailing inline comment is parsed correctly."""
        from olmlx.cli import _legacy_kv_cache_quant_in_dotenv

        monkeypatch.chdir(tmp_path)
        (tmp_path / ".env").write_text(
            'OLMLX_EXPERIMENTAL_KV_CACHE_QUANT="turboquant:4" # use 4-bit\n'
        )
        assert _legacy_kv_cache_quant_in_dotenv() == "turboquant:4"

    def test_legacy_kv_cache_quant_dotenv_quoted_no_comment(
        self, monkeypatch, tmp_path
    ):
        """Quoted value without trailing comment works."""
        from olmlx.cli import _legacy_kv_cache_quant_in_dotenv

        monkeypatch.chdir(tmp_path)
        (tmp_path / ".env").write_text(
            'OLMLX_EXPERIMENTAL_KV_CACHE_QUANT="turboquant:2"\n'
        )
        assert _legacy_kv_cache_quant_in_dotenv() == "turboquant:2"

    def test_legacy_kv_cache_quant_dotenv_unquoted_with_comment(
        self, monkeypatch, tmp_path
    ):
        """Unquoted value with trailing inline comment is parsed correctly."""
        from olmlx.cli import _legacy_kv_cache_quant_in_dotenv

        monkeypatch.chdir(tmp_path)
        (tmp_path / ".env").write_text(
            "OLMLX_EXPERIMENTAL_KV_CACHE_QUANT=turboquant:4 # comment\n"
        )
        assert _legacy_kv_cache_quant_in_dotenv() == "turboquant:4"


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

    # --- Dispatch gaps: unregistered subcommand fallback paths ---

    def test_service_no_subcommand_shows_help(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["olmlx", "service"])
        with pytest.raises(SystemExit) as exc_info:
            cli_main()
        assert exc_info.value.code == 0

    def test_flash_no_subcommand_shows_help(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["olmlx", "flash"])
        with pytest.raises(SystemExit) as exc_info:
            cli_main()
        assert exc_info.value.code == 0

    def test_flash_prepare_calls_handler(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["olmlx", "flash", "prepare", "some/model"])
        mock_fn = MagicMock()
        monkeypatch.setattr("olmlx.cli.cmd_flash_prepare", mock_fn)
        cli_main()
        mock_fn.assert_called_once()

    def test_flash_info_calls_handler(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["olmlx", "flash", "info", "some/model"])
        mock_fn = MagicMock()
        monkeypatch.setattr("olmlx.cli.cmd_flash_info", mock_fn)
        cli_main()
        mock_fn.assert_called_once()

    def test_spectral_no_subcommand_shows_help(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["olmlx", "spectral"])
        with pytest.raises(SystemExit) as exc_info:
            cli_main()
        assert exc_info.value.code == 0

    def test_spectral_prepare_calls_handler(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["olmlx", "spectral", "prepare", "some/model"])
        mock_fn = MagicMock()
        monkeypatch.setattr("olmlx.cli.cmd_spectral_prepare", mock_fn)
        cli_main()
        mock_fn.assert_called_once()

    def test_bench_no_subcommand_shows_help(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["olmlx", "bench"])
        with pytest.raises(SystemExit) as exc_info:
            cli_main()
        assert exc_info.value.code == 0

    def test_bench_run_calls_handler(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["olmlx", "bench", "run"])
        mock_fn = MagicMock()
        monkeypatch.setattr("olmlx.cli.cmd_bench_run", mock_fn)
        cli_main()
        mock_fn.assert_called_once()

    def test_bench_compare_calls_handler(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["olmlx", "bench", "compare", "a", "b"])
        mock_fn = MagicMock()
        monkeypatch.setattr("olmlx.cli.cmd_bench_compare", mock_fn)
        cli_main()
        mock_fn.assert_called_once()

    def test_bench_list_calls_handler(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["olmlx", "bench", "list"])
        mock_fn = MagicMock()
        monkeypatch.setattr("olmlx.cli.cmd_bench_list", mock_fn)
        cli_main()
        mock_fn.assert_called_once()

    def test_bench_leaderboard_calls_handler(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["olmlx", "bench", "leaderboard"])
        mock_fn = MagicMock()
        monkeypatch.setattr("olmlx.cli.cmd_bench_leaderboard", mock_fn)
        cli_main()
        mock_fn.assert_called_once()

    def test_completely_unknown_command_prints_help(self, monkeypatch):
        """Unregistered top-level command should fall through to help."""
        monkeypatch.setattr("sys.argv", ["olmlx", "nope"])
        with pytest.raises(SystemExit) as exc_info:
            cli_main()
        # Argparse exits 2 for unknown args when they don't match any subparser
        assert exc_info.value.code in (0, 2)

    def test_unknown_models_subcommand_prints_help(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["olmlx", "models", "nonexistent"])
        with pytest.raises(SystemExit) as exc_info:
            cli_main()
        # argparse validates choices before dispatch; it may exit 0 (our
        # fallback) or 2 (argparse's built-in choice validation).
        assert exc_info.value.code in (0, 2)

    def test_chat_calls_handler(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["olmlx", "chat"])
        mock_fn = MagicMock()
        monkeypatch.setattr("olmlx.cli.cmd_chat", mock_fn)
        cli_main()
        mock_fn.assert_called_once()


class TestCommandHandlerRegistry:
    """Verify every handler in _COMMAND_HANDLERS resolves at import time."""

    def test_all_handlers_resolve(self):
        from olmlx.cli import _validate_command_handlers

        # Raises NameError/TypeError on any broken entry
        _validate_command_handlers()


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

    def test_malformed_config_raises_without_clobbering(self, tmp_path, monkeypatch):
        """Malformed models.json must raise rather than silently clearing the file."""
        from olmlx.engine.registry import ModelsConfigError

        config_path = tmp_path / "models.json"
        original = "{bad json"
        config_path.write_text(original)
        monkeypatch.setattr("olmlx.cli.settings.models_config", config_path)
        with pytest.raises(ModelsConfigError):
            _create_store()
        # File contents must be untouched — losing the user's config to a
        # parse error is exactly what we're guarding against.
        assert config_path.read_text() == original

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


class TestLegacyFlashPrefetchSpeculativeForwarding:
    def test_legacy_prefetch_speculative_forwarded(self, monkeypatch, caplog):
        import logging
        from olmlx.config import settings, surface_legacy_flash_prefetch_speculative_env

        for k in (
            "flash_prefetch",
            "flash_speculative",
            "flash_speculative_draft_model",
            "flash_speculative_tokens",
        ):
            monkeypatch.delenv("OLMLX_" + k.upper(), raising=False)
        monkeypatch.setattr(settings, "flash_prefetch", False, raising=False)
        monkeypatch.setattr(settings, "flash_speculative", False, raising=False)
        monkeypatch.setattr(
            settings, "flash_speculative_draft_model", None, raising=False
        )
        monkeypatch.setattr(settings, "flash_speculative_tokens", 4, raising=False)

        monkeypatch.setenv("OLMLX_EXPERIMENTAL_FLASH_PREFETCH", "true")
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_FLASH_SPECULATIVE", "true")
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_FLASH_SPECULATIVE_DRAFT_MODEL", "d/m")
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_FLASH_SPECULATIVE_TOKENS", "6")
        with caplog.at_level(logging.WARNING):
            surface_legacy_flash_prefetch_speculative_env()
        assert settings.flash_prefetch is True
        assert settings.flash_speculative is True
        assert settings.flash_speculative_draft_model == "d/m"
        assert settings.flash_speculative_tokens == 6
        assert "OLMLX_FLASH_SPECULATIVE" in caplog.text

    def test_new_env_var_wins_over_legacy(self, monkeypatch):
        from olmlx.config import settings, surface_legacy_flash_prefetch_speculative_env

        monkeypatch.setenv("OLMLX_FLASH_SPECULATIVE_TOKENS", "9")
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_FLASH_SPECULATIVE_TOKENS", "6")
        # Simulate that the new env var already drove Settings to 9.
        monkeypatch.setattr(settings, "flash_speculative_tokens", 9, raising=False)
        surface_legacy_flash_prefetch_speculative_env()
        assert settings.flash_speculative_tokens == 9

    def test_legacy_prefetch_speculative_no_op_when_unset(
        self, monkeypatch, tmp_path, caplog
    ):
        """No legacy or new env vars set → no Settings mutation, no warning.

        Uses monkeypatch.chdir(tmp_path) to avoid picking up a stray .env
        from the developer's working directory (the shim reads Path('.env')
        relative to cwd).
        """
        import logging

        from olmlx.config import settings, surface_legacy_flash_prefetch_speculative_env

        monkeypatch.chdir(tmp_path)
        for name in (
            "OLMLX_EXPERIMENTAL_FLASH_PREFETCH",
            "OLMLX_EXPERIMENTAL_FLASH_SPECULATIVE",
            "OLMLX_EXPERIMENTAL_FLASH_SPECULATIVE_DRAFT_MODEL",
            "OLMLX_EXPERIMENTAL_FLASH_SPECULATIVE_TOKENS",
            "OLMLX_FLASH_PREFETCH",
            "OLMLX_FLASH_SPECULATIVE",
            "OLMLX_FLASH_SPECULATIVE_DRAFT_MODEL",
            "OLMLX_FLASH_SPECULATIVE_TOKENS",
        ):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr(settings, "flash_prefetch", False, raising=False)
        monkeypatch.setattr(settings, "flash_speculative", False, raising=False)
        monkeypatch.setattr(
            settings, "flash_speculative_draft_model", None, raising=False
        )
        monkeypatch.setattr(settings, "flash_speculative_tokens", 4, raising=False)

        with caplog.at_level(logging.WARNING):
            surface_legacy_flash_prefetch_speculative_env()

        assert settings.flash_prefetch is False
        assert settings.flash_speculative is False
        assert settings.flash_speculative_draft_model is None
        assert settings.flash_speculative_tokens == 4
        assert not any(rec.levelno >= logging.WARNING for rec in caplog.records), (
            "Expected no WARNING-level logs but got: " + caplog.text
        )

    def test_legacy_prefetch_dotenv_forwarding(self, monkeypatch, tmp_path, caplog):
        """Legacy OLMLX_EXPERIMENTAL_FLASH_PREFETCH in .env (not shell) is forwarded."""
        import logging

        from olmlx.config import settings, surface_legacy_flash_prefetch_speculative_env

        monkeypatch.chdir(tmp_path)
        (tmp_path / ".env").write_text("OLMLX_EXPERIMENTAL_FLASH_PREFETCH=true\n")
        for name in (
            "OLMLX_EXPERIMENTAL_FLASH_PREFETCH",
            "OLMLX_EXPERIMENTAL_FLASH_SPECULATIVE",
            "OLMLX_EXPERIMENTAL_FLASH_SPECULATIVE_DRAFT_MODEL",
            "OLMLX_EXPERIMENTAL_FLASH_SPECULATIVE_TOKENS",
            "OLMLX_FLASH_PREFETCH",
            "OLMLX_FLASH_SPECULATIVE",
            "OLMLX_FLASH_SPECULATIVE_DRAFT_MODEL",
            "OLMLX_FLASH_SPECULATIVE_TOKENS",
        ):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr(settings, "flash_prefetch", False, raising=False)
        monkeypatch.setattr(settings, "flash_speculative", False, raising=False)
        monkeypatch.setattr(
            settings, "flash_speculative_draft_model", None, raising=False
        )
        monkeypatch.setattr(settings, "flash_speculative_tokens", 4, raising=False)

        with caplog.at_level(logging.WARNING):
            surface_legacy_flash_prefetch_speculative_env()

        assert settings.flash_prefetch is True
        assert "OLMLX_FLASH_PREFETCH" in caplog.text

    def test_whitespace_draft_model_becomes_none(self, monkeypatch, tmp_path):
        """Whitespace-only OLMLX_EXPERIMENTAL_FLASH_SPECULATIVE_DRAFT_MODEL → None.

        Regression for the strip()-only parser that returned "" (empty string)
        instead of None, which would later fail the min_length=1 validator.
        """
        from olmlx.config import settings, surface_legacy_flash_prefetch_speculative_env

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_FLASH_SPECULATIVE_DRAFT_MODEL", "   ")
        monkeypatch.delenv("OLMLX_FLASH_SPECULATIVE_DRAFT_MODEL", raising=False)
        monkeypatch.setattr(
            settings, "flash_speculative_draft_model", None, raising=False
        )

        surface_legacy_flash_prefetch_speculative_env()

        assert settings.flash_speculative_draft_model is None

    def test_dotenv_new_var_opt_out_not_clobbered_by_legacy_shell(
        self, monkeypatch, tmp_path
    ):
        """OLMLX_FLASH_PREFETCH=false in .env must not be clobbered by legacy shell var.

        Regression for the .env blind spot: when the new var is written only to
        .env (not the shell env), os.environ.get(new) returns None and the old
        guard failed to detect the explicit opt-out, allowing the legacy shell
        var to overwrite the user's intended False value.
        """
        from olmlx.config import settings, surface_legacy_flash_prefetch_speculative_env

        monkeypatch.chdir(tmp_path)
        # New var set only in .env — pydantic-settings picks it up but does
        # NOT write it to os.environ, so os.environ.get("OLMLX_FLASH_PREFETCH")
        # returns None even though the user explicitly opted out.
        (tmp_path / ".env").write_text("OLMLX_FLASH_PREFETCH=false\n")
        # Legacy var is in the shell env, simulating a user who hasn't cleaned
        # up their shell after the rename.
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_FLASH_PREFETCH", "true")
        # Ensure the new shell var is absent (only in .env).
        monkeypatch.delenv("OLMLX_FLASH_PREFETCH", raising=False)
        # Settings reflect the .env opt-out (False == default, but the user
        # explicitly wrote it — the bug scenario).
        monkeypatch.setattr(settings, "flash_prefetch", False, raising=False)

        surface_legacy_flash_prefetch_speculative_env()

        # The legacy shell var must NOT have clobbered the explicit .env opt-out.
        assert settings.flash_prefetch is False

    def test_dotenv_new_var_opt_out_not_clobbered_by_legacy_shell_flash_moe(
        self, monkeypatch, tmp_path
    ):
        """OLMLX_FLASH_MOE=false in .env must not be clobbered by legacy shell var.

        Regression for the .env blind spot in surface_legacy_flash_moe_env: when
        the new var is written only to .env (not the shell env),
        os.environ.get(new) returns None and the old guard failed to detect the
        explicit opt-out, allowing the legacy shell var to overwrite the user's
        intended False value.
        """
        from olmlx.config import settings, surface_legacy_flash_moe_env

        monkeypatch.chdir(tmp_path)
        # New var set only in .env — pydantic-settings picks it up but does
        # NOT write it to os.environ, so os.environ.get("OLMLX_FLASH_MOE")
        # returns None even though the user explicitly opted out.
        (tmp_path / ".env").write_text("OLMLX_FLASH_MOE=false\n")
        # Legacy var is in the shell env, simulating a user who hasn't cleaned
        # up their shell after the rename.
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_FLASH_MOE", "true")
        # Ensure the new shell var is absent (only in .env).
        monkeypatch.delenv("OLMLX_FLASH_MOE", raising=False)
        # Settings reflect the .env opt-out (False == default, but the user
        # explicitly wrote it — the bug scenario).
        monkeypatch.setattr(settings, "flash_moe", False, raising=False)

        surface_legacy_flash_moe_env()

        # The legacy shell var must NOT have clobbered the explicit .env opt-out.
        assert settings.flash_moe is False

    def test_cmd_flash_prepare_calls_legacy_shim(self, monkeypatch):
        """cmd_flash_prepare must surface legacy env vars before reading settings.

        Regression test for the fix that adds _surface_legacy_flash_env() and
        _surface_legacy_flash_prefetch_speculative_env() at the top of
        cmd_flash_prepare — without this, OLMLX_EXPERIMENTAL_FLASH_PREFETCH=true
        was silently ignored and no LookaheadBank was trained.
        """
        import inspect
        import olmlx.cli as cli_mod

        src = inspect.getsource(cli_mod.cmd_flash_prepare)
        assert "_surface_legacy_flash_prefetch_speculative_env" in src, (
            "cmd_flash_prepare must call _surface_legacy_flash_prefetch_speculative_env() "
            "so that OLMLX_EXPERIMENTAL_FLASH_PREFETCH=true reaches settings.flash_prefetch "
            "before _cmd_flash_dense_prepare reads train_lookahead=settings.flash_prefetch."
        )
        assert "_surface_legacy_flash_env" in src, (
            "cmd_flash_prepare must also call _surface_legacy_flash_env() "
            "for consistency with other commands that surface both shims."
        )


class TestServeFlashPrefetchSpeculativeFlags:
    def test_flags_apply_to_settings(self, monkeypatch):
        import argparse

        from olmlx.cli import _apply_serve_overrides
        from olmlx.config import settings

        monkeypatch.setattr(settings, "flash_speculative", False, raising=False)
        monkeypatch.setattr(settings, "flash_prefetch", False, raising=False)
        args = argparse.Namespace(
            flash_speculative=True,
            flash_speculative_draft_model="d/m",
            flash_speculative_tokens=5,
            flash_prefetch=True,
        )
        _apply_serve_overrides(args)
        assert settings.flash_speculative is True
        assert settings.flash_speculative_draft_model == "d/m"
        assert settings.flash_speculative_tokens == 5
        assert settings.flash_prefetch is True


class TestFlashWithoutFlashGuards:
    """Warn when flash_speculative / flash_prefetch are set without flash."""

    def test_flash_speculative_without_flash_warns(self, monkeypatch, caplog):
        import argparse
        import logging

        from olmlx.cli import _apply_serve_overrides
        from olmlx.config import settings

        monkeypatch.setattr(settings, "flash", False, raising=True)
        monkeypatch.setattr(settings, "flash_speculative", False, raising=True)
        monkeypatch.setattr(settings, "flash_prefetch", False, raising=True)
        args = argparse.Namespace(
            flash_speculative=True,
            flash_speculative_draft_model=None,
            flash_speculative_tokens=None,
            flash_prefetch=None,
        )
        with caplog.at_level(logging.WARNING, logger="olmlx.cli"):
            _apply_serve_overrides(args)
        assert (
            "flash_speculative is set but flash is not enabled globally" in caplog.text
        )

    def test_flash_prefetch_without_flash_warns(self, monkeypatch, caplog):
        import argparse
        import logging

        from olmlx.cli import _apply_serve_overrides
        from olmlx.config import settings

        monkeypatch.setattr(settings, "flash", False, raising=True)
        monkeypatch.setattr(settings, "flash_speculative", False, raising=True)
        monkeypatch.setattr(settings, "flash_prefetch", False, raising=True)
        args = argparse.Namespace(
            flash_speculative=None,
            flash_speculative_draft_model=None,
            flash_speculative_tokens=None,
            flash_prefetch=True,
        )
        with caplog.at_level(logging.WARNING, logger="olmlx.cli"):
            _apply_serve_overrides(args)
        assert "flash_prefetch is set but flash is not enabled globally" in caplog.text


class TestPullWithDraft:
    """`olmlx models pull --with-draft` co-downloads the curated draft (#514)."""

    @staticmethod
    def _fake_pull():
        async def fake_pull(name):
            yield {"status": f"downloading {name}"}
            yield {"status": "success"}

        return fake_pull

    def test_with_draft_pulls_and_registers(
        self, capsys, mock_store, _patch_store, monkeypatch
    ):
        from olmlx.engine import registry
        from olmlx.engine.registry import KnownDraft

        entry = KnownDraft(draft_repo="org/draft", strategy="eagle", block_size=4)
        monkeypatch.setitem(registry.KNOWN_DRAFTS, "Qwen/Qwen3-32B-MLX", entry)

        mock_store.pull = self._fake_pull()
        mock_store.registry.resolve.return_value = MagicMock(
            hf_path="Qwen/Qwen3-32B-MLX"
        )
        mock_store.register_speculative_draft = MagicMock()

        args = MagicMock(model_name="qwen3:32b", with_draft=True)
        cmd_models_pull(args)

        out = capsys.readouterr().out
        assert "downloading qwen3:32b" in out
        assert "pulling speculative draft org/draft" in out
        assert "downloading org/draft" in out
        assert "wired speculative draft org/draft" in out
        mock_store.register_speculative_draft.assert_called_once_with(
            "qwen3:32b", "Qwen/Qwen3-32B-MLX", entry
        )

    def test_with_draft_no_known_draft_skips(self, capsys, mock_store, _patch_store):
        mock_store.pull = self._fake_pull()
        mock_store.registry.resolve.return_value = MagicMock(hf_path="unknown/model")
        mock_store.register_speculative_draft = MagicMock()

        args = MagicMock(model_name="unknown", with_draft=True)
        cmd_models_pull(args)

        out = capsys.readouterr().out
        assert "no known speculative draft" in out
        assert "pulling speculative draft" not in out
        mock_store.register_speculative_draft.assert_not_called()

    def test_without_flag_does_not_touch_draft(
        self, capsys, mock_store, _patch_store, monkeypatch
    ):
        from olmlx.engine import registry
        from olmlx.engine.registry import KnownDraft

        monkeypatch.setitem(
            registry.KNOWN_DRAFTS,
            "Qwen/Qwen3-32B-MLX",
            KnownDraft(draft_repo="org/draft", strategy="eagle", block_size=4),
        )
        mock_store.pull = self._fake_pull()
        mock_store.registry.resolve.return_value = MagicMock(
            hf_path="Qwen/Qwen3-32B-MLX"
        )
        mock_store.register_speculative_draft = MagicMock()

        args = MagicMock(model_name="qwen3:32b", with_draft=False)
        cmd_models_pull(args)

        out = capsys.readouterr().out
        assert "downloading qwen3:32b" in out
        assert "pulling speculative draft" not in out
        mock_store.register_speculative_draft.assert_not_called()

    def test_with_draft_strips_tag_on_full_path_target(
        self, capsys, mock_store, _patch_store, monkeypatch
    ):
        """A full-path pull with a tag (org/model:q4) resolves to the canonical
        untagged key in the curated map."""
        from olmlx.engine import registry
        from olmlx.engine.registry import KnownDraft

        entry = KnownDraft(draft_repo="org/draft", strategy="eagle", block_size=4)
        monkeypatch.setitem(registry.KNOWN_DRAFTS, "org/Big-Model", entry)

        mock_store.pull = self._fake_pull()
        # resolve() passes a full path through verbatim, tag included.
        mock_store.registry.resolve.return_value = MagicMock(hf_path="org/Big-Model:q4")
        mock_store.register_speculative_draft = MagicMock()

        args = MagicMock(model_name="org/Big-Model:q4", with_draft=True)
        cmd_models_pull(args)

        out = capsys.readouterr().out
        assert "pulling speculative draft org/draft" in out
        mock_store.register_speculative_draft.assert_called_once_with(
            "org/Big-Model:q4", "org/Big-Model", entry
        )
