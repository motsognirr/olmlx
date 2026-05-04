"""Tests for olmlx.config."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from olmlx.config import ExperimentalSettings, Settings, resolve_experimental


class TestSettings:
    def test_defaults(self, monkeypatch):
        # Clear any env vars
        for key in (
            "OLMLX_HOST",
            "OLMLX_PORT",
            "OLMLX_MODELS_DIR",
            "OLMLX_MODELS_CONFIG",
            "OLMLX_DEFAULT_KEEP_ALIVE",
            "OLMLX_MAX_LOADED_MODELS",
            "OLMLX_MODEL_LOAD_TIMEOUT",
        ):
            monkeypatch.delenv(key, raising=False)
        s = Settings()
        assert s.host == "0.0.0.0"
        assert s.port == 11434
        assert s.default_keep_alive == "5m"
        assert s.max_loaded_models == 1
        assert s.model_load_timeout is None
        assert isinstance(s.models_dir, Path)
        assert s.models_config == Path.home() / ".olmlx" / "models.json"

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("OLMLX_HOST", "127.0.0.1")
        monkeypatch.setenv("OLMLX_PORT", "8080")
        monkeypatch.setenv("OLMLX_MAX_LOADED_MODELS", "3")
        monkeypatch.setenv("OLMLX_MODEL_LOAD_TIMEOUT", "120")
        s = Settings()
        assert s.host == "127.0.0.1"
        assert s.port == 8080
        assert s.max_loaded_models == 3
        assert s.model_load_timeout == 120.0

    def test_model_load_timeout_rejects_zero(self, monkeypatch):
        monkeypatch.setenv("OLMLX_MODEL_LOAD_TIMEOUT", "0")
        with pytest.raises(ValidationError):
            Settings()

    def test_model_load_timeout_rejects_negative(self, monkeypatch):
        monkeypatch.setenv("OLMLX_MODEL_LOAD_TIMEOUT", "-1")
        with pytest.raises(ValidationError):
            Settings()

    def test_anthropic_models_default(self, monkeypatch):
        monkeypatch.delenv("OLMLX_ANTHROPIC_MODELS", raising=False)
        s = Settings()
        assert s.anthropic_models == {}

    def test_anthropic_models_from_env(self, monkeypatch):
        monkeypatch.setenv(
            "OLMLX_ANTHROPIC_MODELS",
            '{"haiku": "qwen3:latest", "sonnet": "qwen3-8b:latest"}',
        )
        s = Settings()
        assert s.anthropic_models == {
            "haiku": "qwen3:latest",
            "sonnet": "qwen3-8b:latest",
        }

    def test_anthropic_models_rejects_dash_in_key(self, monkeypatch):
        monkeypatch.setenv(
            "OLMLX_ANTHROPIC_MODELS",
            '{"claude-sonnet": "qwen3:latest"}',
        )
        with pytest.raises(ValidationError, match="single segment"):
            Settings()

    def test_port_rejects_zero(self, monkeypatch):
        monkeypatch.setenv("OLMLX_PORT", "0")
        with pytest.raises(ValidationError):
            Settings()

    def test_port_rejects_negative(self, monkeypatch):
        monkeypatch.setenv("OLMLX_PORT", "-1")
        with pytest.raises(ValidationError):
            Settings()

    def test_port_rejects_above_65535(self, monkeypatch):
        monkeypatch.setenv("OLMLX_PORT", "65536")
        with pytest.raises(ValidationError):
            Settings()

    def test_port_accepts_boundary_values(self, monkeypatch):
        monkeypatch.setenv("OLMLX_PORT", "1")
        s = Settings()
        assert s.port == 1

        monkeypatch.setenv("OLMLX_PORT", "65535")
        s = Settings()
        assert s.port == 65535

    def test_anthropic_models_rejects_colon_in_key(self, monkeypatch):
        monkeypatch.setenv(
            "OLMLX_ANTHROPIC_MODELS",
            '{"sonnet:latest": "qwen3:latest"}',
        )
        with pytest.raises(ValidationError, match="single segment"):
            Settings()


class TestResolveExperimental:
    def test_empty_overrides_returns_global(self, monkeypatch):
        monkeypatch.delenv("OLMLX_EXPERIMENTAL_FLASH", raising=False)
        monkeypatch.delenv("OLMLX_EXPERIMENTAL_KV_CACHE_QUANT", raising=False)
        base = ExperimentalSettings()
        result = resolve_experimental(base, {})
        assert result.flash == base.flash
        assert result.kv_cache_quant == base.kv_cache_quant

    def test_flash_override(self, monkeypatch):
        monkeypatch.delenv("OLMLX_EXPERIMENTAL_FLASH", raising=False)
        base = ExperimentalSettings()
        assert base.flash is False
        result = resolve_experimental(base, {"flash": True})
        assert result.flash is True
        # Original should be unchanged
        assert base.flash is False

    def test_kv_cache_quant_override(self, monkeypatch):
        monkeypatch.delenv("OLMLX_EXPERIMENTAL_KV_CACHE_QUANT", raising=False)
        base = ExperimentalSettings()
        result = resolve_experimental(base, {"kv_cache_quant": "turboquant:4"})
        assert result.kv_cache_quant == "turboquant:4"

    def test_partial_override_preserves_other_fields(self, monkeypatch):
        monkeypatch.delenv("OLMLX_EXPERIMENTAL_FLASH", raising=False)
        monkeypatch.delenv("OLMLX_EXPERIMENTAL_FLASH_SPARSITY_THRESHOLD", raising=False)
        base = ExperimentalSettings()
        result = resolve_experimental(base, {"flash": True})
        # flash changed, but sparsity_threshold stays at default
        assert result.flash is True
        assert result.flash_sparsity_threshold == base.flash_sparsity_threshold

    def test_multiple_overrides(self, monkeypatch):
        monkeypatch.delenv("OLMLX_EXPERIMENTAL_FLASH", raising=False)
        monkeypatch.delenv("OLMLX_EXPERIMENTAL_FLASH_SPARSITY_THRESHOLD", raising=False)
        base = ExperimentalSettings()
        result = resolve_experimental(
            base,
            {
                "flash": True,
                "flash_sparsity_threshold": 0.3,
                "flash_moe": True,
            },
        )
        assert result.flash is True
        assert result.flash_sparsity_threshold == 0.3
        assert result.flash_moe is True

    def test_env_var_does_not_leak_into_overrides(self, monkeypatch):
        """resolve_experimental should not re-read env vars for unrelated fields."""
        monkeypatch.delenv("OLMLX_EXPERIMENTAL_FLASH", raising=False)
        monkeypatch.delenv("OLMLX_EXPERIMENTAL_KV_CACHE_QUANT", raising=False)
        base = ExperimentalSettings()
        # Set a bad env var *after* base is created
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_KV_CACHE_QUANT", "bad_value")
        # Overriding only 'flash' should not trigger kv_cache_quant env var parsing
        result = resolve_experimental(base, {"flash": True})
        assert result.flash is True
        assert result.kv_cache_quant is None


class TestSpeculativeConfig:
    """Tests for standalone speculative decoding config fields (now on Settings)."""

    def test_speculative_defaults(self, monkeypatch):
        monkeypatch.delenv("OLMLX_SPECULATIVE", raising=False)
        monkeypatch.delenv("OLMLX_SPECULATIVE_DRAFT_MODEL", raising=False)
        monkeypatch.delenv("OLMLX_SPECULATIVE_TOKENS", raising=False)
        s = Settings()
        assert s.speculative is False
        assert s.speculative_draft_model is None
        assert s.speculative_tokens == 4

    def test_speculative_env_override(self, monkeypatch):
        monkeypatch.setenv("OLMLX_SPECULATIVE", "true")
        monkeypatch.setenv("OLMLX_SPECULATIVE_DRAFT_MODEL", "Qwen/Qwen3-0.6B")
        monkeypatch.setenv("OLMLX_SPECULATIVE_TOKENS", "6")
        s = Settings()
        assert s.speculative is True
        assert s.speculative_draft_model == "Qwen/Qwen3-0.6B"
        assert s.speculative_tokens == 6

    def test_speculative_tokens_rejects_zero(self):
        with pytest.raises(ValidationError):
            Settings(speculative_tokens=0, _env_file=None)

    def test_speculative_draft_model_rejects_empty_string(self, monkeypatch):
        """``OLMLX_SPECULATIVE_DRAFT_MODEL=""`` slips past a ``str | None``
        check; ``Field(min_length=1)`` blocks it at parse time so the
        load path doesn't surface a misleading "draft not set" error."""
        monkeypatch.setenv("OLMLX_SPECULATIVE_DRAFT_MODEL", "")
        with pytest.raises(ValidationError):
            Settings()

    def test_speculative_draft_model_rejects_whitespace_only(self, monkeypatch):
        """``Field(min_length=1)`` accepts ``"   "`` (length > 0). The
        custom validator strips and re-rejects so a whitespace-only env
        var doesn't surface as a misleading load-time error."""
        monkeypatch.setenv("OLMLX_SPECULATIVE_DRAFT_MODEL", "   ")
        with pytest.raises(ValidationError, match="non-empty"):
            Settings()

    def test_speculative_draft_model_strips_whitespace(self, monkeypatch):
        """Surrounding whitespace is stripped on parse, so a
        ``OLMLX_SPECULATIVE_DRAFT_MODEL=" hf/path "`` doesn't reach the
        loader with a path containing spaces."""
        monkeypatch.setenv("OLMLX_SPECULATIVE_DRAFT_MODEL", "  Qwen/Qwen3-0.6B  ")
        s = Settings()
        assert s.speculative_draft_model == "Qwen/Qwen3-0.6B"

    def test_model_config_speculative_tokens_validated_on_construct(self):
        """Direct ModelConfig construction must enforce the >=1 invariant."""
        from olmlx.engine.registry import ModelConfig

        with pytest.raises(ValueError, match="speculative_tokens"):
            ModelConfig(hf_path="x/y", speculative_tokens=0)
        with pytest.raises(ValueError, match="speculative_tokens"):
            ModelConfig(hf_path="x/y", speculative_tokens=-1)

    def test_model_config_empty_draft_validated_on_construct(self):
        """Direct construction must reject empty / whitespace draft paths,
        not just from_entry — same invariant either way."""
        from olmlx.engine.registry import ModelConfig

        for value in ("", "   ", "\t"):
            with pytest.raises(ValueError, match="non-empty"):
                ModelConfig(hf_path="x/y", speculative_draft_model=value)

    def test_speculative_tokens_assignment_validated(self, monkeypatch):
        """validate_assignment=True keeps Field(gt=0) honoured for programmatic writes."""
        monkeypatch.delenv("OLMLX_SPECULATIVE_TOKENS", raising=False)
        s = Settings(_env_file=None)
        with pytest.raises(ValidationError):
            s.speculative_tokens = 0
        with pytest.raises(ValidationError):
            s.speculative_tokens = -3

    def test_speculative_no_longer_in_experimental(self):
        """Promoted fields must not exist on ExperimentalSettings anymore."""
        e = ExperimentalSettings()
        assert not hasattr(e, "speculative")
        assert not hasattr(e, "speculative_draft_model")
        assert not hasattr(e, "speculative_tokens")

    def test_speculative_rejected_in_experimental_overrides(self):
        """Per-model overrides under 'experimental' should raise a clear migration error."""
        from olmlx.engine.registry import _validate_experimental_overrides

        with pytest.raises(ValueError, match="promoted out of 'experimental'"):
            _validate_experimental_overrides({"speculative": True})

    def test_empty_speculative_draft_model_rejected(self):
        """Empty / whitespace strings used to slip past parse and surface
        as the misleading 'draft not set' error at load time."""
        from olmlx.engine.registry import ModelConfig

        for value in ("", "   ", "\t"):
            with pytest.raises(ValueError, match="non-empty"):
                ModelConfig.from_entry(
                    {"hf_path": "x/y", "speculative_draft_model": value}
                )

    def test_promoted_keys_renamed_branch_message(self, monkeypatch):
        """Cover the 'rename' branch of the migration error so it doesn't
        rot before the next promotion exercises it for real."""
        from olmlx.engine import registry

        monkeypatch.setitem(registry.PROMOTED_EXPERIMENTAL_KEYS, "old_name", "new_name")
        with pytest.raises(
            ValueError, match="rename 'old_name' → top-level 'new_name'"
        ):
            registry._validate_experimental_overrides({"old_name": True})

    def test_speculative_per_model_top_level(self):
        """Per-model overrides go at the top level of the models.json entry."""
        from olmlx.engine.registry import ModelConfig

        mc = ModelConfig.from_entry(
            {
                "hf_path": "Qwen/Qwen3-32B",
                "speculative": True,
                "speculative_draft_model": "Qwen/Qwen3-0.6B",
                "speculative_tokens": 6,
            }
        )
        assert mc.speculative is True
        assert mc.speculative_draft_model == "Qwen/Qwen3-0.6B"
        assert mc.speculative_tokens == 6
        # Promoted keys must NOT leak into ``_extra`` (which is reserved
        # for unknown forward-compat keys). Reviewers have flagged this
        # as a suspected bug multiple times — _KNOWN_CONFIG_KEYS is
        # auto-derived from dataclass fields, but lock it down with an
        # explicit assertion so the invariant is visible in the suite.
        assert mc._extra == {}
        # Round-trip preserves the new top-level fields
        entry = mc.to_entry()
        assert isinstance(entry, dict)
        assert entry["speculative"] is True
        assert entry["speculative_draft_model"] == "Qwen/Qwen3-0.6B"
        assert entry["speculative_tokens"] == 6

    def test_known_config_keys_includes_promoted_speculative_fields(self):
        """Regression: the auto-derived ``_KNOWN_CONFIG_KEYS`` must include
        the three promoted speculative fields so they don't leak into
        ``_extra``. This has been a recurring false-positive review finding;
        locking the invariant down explicitly."""
        from olmlx.engine.registry import _KNOWN_CONFIG_KEYS

        for key in ("speculative", "speculative_draft_model", "speculative_tokens"):
            assert key in _KNOWN_CONFIG_KEYS

    def test_per_model_speculative_without_draft_resolves_to_none(self):
        """A per-model entry that enables speculative but provides no
        draft model resolves to (True, None, default). The error surfaces
        either at startup (registry walk) or at first model load."""
        from olmlx.engine.registry import ModelConfig

        mc = ModelConfig.from_entry({"hf_path": "Qwen/Qwen3-32B", "speculative": True})
        enabled, draft, _tokens = mc.resolved_speculative()
        assert enabled is True
        assert draft is None

    def test_resolved_speculative_falls_back_to_settings(self, monkeypatch):
        from olmlx.engine.registry import ModelConfig

        monkeypatch.setattr("olmlx.config.settings.speculative", True)
        monkeypatch.setattr(
            "olmlx.config.settings.speculative_draft_model", "Qwen/Qwen3-0.6B"
        )
        monkeypatch.setattr("olmlx.config.settings.speculative_tokens", 8)

        mc = ModelConfig(hf_path="Qwen/Qwen3-32B")
        assert mc.resolved_speculative() == (True, "Qwen/Qwen3-0.6B", 8)

        # Per-model overrides win, and a disabled per-model setting
        # zeros out the draft slot even if a global draft is configured.
        mc_override = ModelConfig(
            hf_path="Qwen/Qwen3-32B",
            speculative=False,
            speculative_tokens=2,
        )
        assert mc_override.resolved_speculative() == (False, None, 2)


class TestDFlashConfig:
    """Tests for dflash config fields."""

    def test_dflash_defaults(self, monkeypatch):
        monkeypatch.delenv("OLMLX_EXPERIMENTAL_DFLASH", raising=False)
        monkeypatch.delenv("OLMLX_EXPERIMENTAL_DFLASH_DRAFT_MODEL", raising=False)
        monkeypatch.delenv("OLMLX_EXPERIMENTAL_DFLASH_BLOCK_SIZE", raising=False)
        s = ExperimentalSettings()
        assert s.dflash is False
        assert s.dflash_draft_model is None
        assert s.dflash_block_size == 4

    def test_dflash_env_override(self, monkeypatch):
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_DFLASH", "true")
        monkeypatch.setenv(
            "OLMLX_EXPERIMENTAL_DFLASH_DRAFT_MODEL", "aryagm/dflash-qwen3"
        )
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_DFLASH_BLOCK_SIZE", "8")
        s = ExperimentalSettings()
        assert s.dflash is True
        assert s.dflash_draft_model == "aryagm/dflash-qwen3"
        assert s.dflash_block_size == 8

    def test_dflash_block_size_rejects_zero(self):
        with pytest.raises(ValidationError):
            ExperimentalSettings(dflash_block_size=0, _env_file=None)
