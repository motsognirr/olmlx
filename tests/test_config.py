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

    def test_inference_headroom_rejects_when_ge_limit(self, monkeypatch):
        # headroom >= limit collapses the effective load budget to <= 0,
        # which would silently fail every model load — reject at startup.
        monkeypatch.setenv("OLMLX_MEMORY_LIMIT_FRACTION", "0.5")
        monkeypatch.setenv("OLMLX_INFERENCE_HEADROOM_FRACTION", "0.75")
        with pytest.raises(ValidationError, match="inference_headroom_fraction"):
            Settings()

    def test_inference_headroom_rejects_when_equal_limit(self, monkeypatch):
        monkeypatch.setenv("OLMLX_MEMORY_LIMIT_FRACTION", "0.5")
        monkeypatch.setenv("OLMLX_INFERENCE_HEADROOM_FRACTION", "0.5")
        with pytest.raises(ValidationError):
            Settings()

    def test_inference_headroom_below_limit_ok(self, monkeypatch):
        monkeypatch.setenv("OLMLX_MEMORY_LIMIT_FRACTION", "0.75")
        monkeypatch.setenv("OLMLX_INFERENCE_HEADROOM_FRACTION", "0.1")
        s = Settings()
        assert s.inference_headroom_fraction == 0.1

    def test_effective_load_budget_fraction(self, monkeypatch):
        monkeypatch.setenv("OLMLX_MEMORY_LIMIT_FRACTION", "0.75")
        monkeypatch.setenv("OLMLX_INFERENCE_HEADROOM_FRACTION", "0.30")
        s = Settings()
        assert s.effective_load_budget_fraction == pytest.approx(0.45)

    def test_effective_load_budget_fraction_default(self, monkeypatch):
        monkeypatch.setenv("OLMLX_MEMORY_LIMIT_FRACTION", "0.75")
        monkeypatch.delenv("OLMLX_INFERENCE_HEADROOM_FRACTION", raising=False)
        s = Settings()
        # Default headroom 0.0 → effective equals the raw limit.
        assert s.effective_load_budget_fraction == pytest.approx(0.75)

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
        monkeypatch.delenv("OLMLX_EXPERIMENTAL_FLASH_WINDOW_SIZE", raising=False)
        base = ExperimentalSettings()
        result = resolve_experimental(base, {})
        assert result.flash_window_size == base.flash_window_size

    def test_flash_override(self, monkeypatch):
        """Advanced flash tuning fields still resolve through experimental.

        ``flash_window_size`` is one of the keys deliberately left under
        ``ExperimentalSettings`` after the PR #274 promotion — the
        primary user-facing knobs (``flash``,
        ``flash_sparsity_threshold``, etc.) moved to ``Settings`` and
        ``ModelConfig.resolved_flash()``.
        """
        monkeypatch.delenv("OLMLX_EXPERIMENTAL_FLASH_WINDOW_SIZE", raising=False)
        base = ExperimentalSettings()
        assert base.flash_window_size == 5
        result = resolve_experimental(base, {"flash_window_size": 10})
        assert result.flash_window_size == 10
        # Original should be unchanged
        assert base.flash_window_size == 5

    def test_kv_cache_quant_top_level(self, monkeypatch):
        """kv_cache_quant is a top-level ModelConfig field, not an experimental override."""
        monkeypatch.delenv("OLMLX_KV_CACHE_QUANT", raising=False)
        from olmlx.engine.registry import ModelConfig

        mc = ModelConfig(hf_path="test/model", kv_cache_quant="turboquant:4")
        assert mc.kv_cache_quant == "turboquant:4"

    def test_partial_override_preserves_other_fields(self, monkeypatch):
        monkeypatch.delenv("OLMLX_EXPERIMENTAL_FLASH_WINDOW_SIZE", raising=False)
        monkeypatch.delenv("OLMLX_EXPERIMENTAL_FLASH_IO_THREADS", raising=False)
        base = ExperimentalSettings()
        result = resolve_experimental(base, {"flash_window_size": 10})
        # flash_window_size changed, but flash_io_threads stays at default
        assert result.flash_window_size == 10
        assert result.flash_io_threads == base.flash_io_threads

    def test_multiple_overrides(self, monkeypatch):
        monkeypatch.delenv("OLMLX_EXPERIMENTAL_FLASH_WINDOW_SIZE", raising=False)
        monkeypatch.delenv("OLMLX_EXPERIMENTAL_FLASH_IO_THREADS", raising=False)
        monkeypatch.delenv(
            "OLMLX_EXPERIMENTAL_FLASH_CACHE_BUDGET_NEURONS", raising=False
        )
        base = ExperimentalSettings()
        result = resolve_experimental(
            base,
            {
                "flash_window_size": 10,
                "flash_io_threads": 64,
                "flash_cache_budget_neurons": 512,
            },
        )
        assert result.flash_window_size == 10
        assert result.flash_io_threads == 64
        assert result.flash_cache_budget_neurons == 512

    def test_env_var_does_not_leak_into_overrides(self, monkeypatch):
        """resolve_experimental should not re-read env vars for unrelated fields."""
        monkeypatch.delenv("OLMLX_EXPERIMENTAL_FLASH_WINDOW_SIZE", raising=False)
        monkeypatch.delenv("OLMLX_KV_CACHE_QUANT", raising=False)
        base = ExperimentalSettings()
        # Set a bad env var *after* base is created — kv_cache_quant is now on
        # Settings (OLMLX_ prefix), not ExperimentalSettings, so it won't leak.
        monkeypatch.setenv("OLMLX_KV_CACHE_QUANT", "bad_value")
        # Overriding only 'flash_window_size' should not trigger any env var parsing
        result = resolve_experimental(base, {"flash_window_size": 10})
        assert result.flash_window_size == 10


class TestSpeculativeConfig:
    """Tests for standalone speculative decoding config fields (now on Settings)."""

    def test_speculative_defaults(self, monkeypatch):
        monkeypatch.delenv("OLMLX_SPECULATIVE", raising=False)
        monkeypatch.delenv("OLMLX_SPECULATIVE_DRAFT_MODEL", raising=False)
        monkeypatch.delenv("OLMLX_SPECULATIVE_TOKENS", raising=False)
        s = Settings()
        assert s.speculative is False
        assert s.speculative_draft_model is None
        # ``None`` means "use the strategy default" (4 for classic, the
        # draft's pre-trained block_size for DFlash) — see Settings docs.
        assert s.speculative_tokens is None

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

    def test_speculative_cache_slots_default(self, monkeypatch):
        monkeypatch.delenv("OLMLX_SPECULATIVE_CACHE_SLOTS", raising=False)
        assert Settings().speculative_cache_slots == 2

    def test_speculative_cache_slots_env_override(self, monkeypatch):
        monkeypatch.setenv("OLMLX_SPECULATIVE_CACHE_SLOTS", "0")
        assert Settings().speculative_cache_slots == 0

    def test_speculative_cache_slots_rejects_negative(self):
        with pytest.raises(ValidationError):
            Settings(speculative_cache_slots=-1, _env_file=None)

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

    def test_known_config_keys_includes_promoted_fields(self):
        """Regression: the auto-derived ``_KNOWN_CONFIG_KEYS`` must include
        all promoted fields so they don't leak into ``_extra``."""
        from olmlx.engine.registry import _KNOWN_CONFIG_KEYS

        for key in (
            "speculative",
            "speculative_draft_model",
            "speculative_tokens",
            "kv_cache_quant",
        ):
            assert key in _KNOWN_CONFIG_KEYS

    def test_per_model_speculative_without_draft_resolves_to_none(self):
        """A per-model entry that enables speculative but provides no
        draft model resolves to (True, None, default). The error surfaces
        either at startup (registry walk) or at first model load."""
        from olmlx.engine.registry import ModelConfig

        mc = ModelConfig.from_entry({"hf_path": "Qwen/Qwen3-32B", "speculative": True})
        resolved = mc.resolved_speculative()
        assert resolved.enabled is True
        assert resolved.draft_model is None

    def test_resolved_speculative_falls_back_to_settings(self, monkeypatch):
        from olmlx.engine.registry import ModelConfig

        monkeypatch.setattr("olmlx.config.settings.speculative", True)
        monkeypatch.setattr(
            "olmlx.config.settings.speculative_draft_model", "Qwen/Qwen3-0.6B"
        )
        monkeypatch.setattr("olmlx.config.settings.speculative_tokens", 8)

        mc = ModelConfig(hf_path="Qwen/Qwen3-32B")
        resolved = mc.resolved_speculative()
        assert resolved.enabled is True
        assert resolved.draft_model == "Qwen/Qwen3-0.6B"
        assert resolved.num_tokens == 8
        assert resolved.strategy == "classic"

        # Per-model overrides win, and a disabled per-model setting
        # zeros out the draft slot even if a global draft is configured.
        mc_override = ModelConfig(
            hf_path="Qwen/Qwen3-32B",
            speculative=False,
            speculative_tokens=2,
        )
        resolved_override = mc_override.resolved_speculative()
        assert resolved_override.enabled is False
        assert resolved_override.draft_model is None
        assert resolved_override.num_tokens == 2
        assert resolved_override.strategy == "classic"


class TestSpeculativeStrategySettings:
    """Tests for the unified speculative_strategy field on Settings."""

    def test_default_is_classic(self, monkeypatch):
        monkeypatch.delenv("OLMLX_SPECULATIVE_STRATEGY", raising=False)
        s = Settings(_env_file=None)
        assert s.speculative_strategy == "classic"

    def test_env_override_dflash(self, monkeypatch):
        monkeypatch.setenv("OLMLX_SPECULATIVE_STRATEGY", "dflash")
        s = Settings()
        assert s.speculative_strategy == "dflash"

    def test_invalid_strategy_rejected(self):
        with pytest.raises(ValidationError):
            Settings(speculative_strategy="bogus", _env_file=None)  # type: ignore[arg-type]


class TestFlashPrefetchSpeculativePromotion:
    def test_promoted_fields_on_settings_via_env(self, monkeypatch):
        from olmlx.config import Settings

        monkeypatch.setenv("OLMLX_FLASH_PREFETCH", "true")
        monkeypatch.setenv("OLMLX_FLASH_SPECULATIVE", "true")
        monkeypatch.setenv(
            "OLMLX_FLASH_SPECULATIVE_DRAFT_MODEL",
            "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        )
        monkeypatch.setenv("OLMLX_FLASH_SPECULATIVE_TOKENS", "6")
        s = Settings()
        assert s.flash_prefetch is True
        assert s.flash_speculative is True
        assert (
            s.flash_speculative_draft_model
            == "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
        )
        assert s.flash_speculative_tokens == 6

    def test_flash_speculative_draft_model_rejects_blank(self):
        import pytest
        from pydantic import ValidationError
        from olmlx.config import Settings

        with pytest.raises(ValidationError):
            Settings(flash_speculative_draft_model="")
        with pytest.raises(ValidationError):
            Settings(flash_speculative_draft_model="   ")

    def test_prefetch_tuning_knobs_stay_experimental(self, monkeypatch):
        from olmlx.config import ExperimentalSettings

        monkeypatch.setenv("OLMLX_EXPERIMENTAL_FLASH_PREFETCH_IO_THREADS", "8")
        e = ExperimentalSettings()
        assert e.flash_prefetch_io_threads == 8
        # The promoted toggle is no longer an ExperimentalSettings field.
        assert "flash_prefetch" not in ExperimentalSettings.model_fields


class TestWeightQuant:
    """Tests for OLMLX_WEIGHT_QUANT config validation."""

    def test_default_is_none(self, monkeypatch):
        monkeypatch.delenv("OLMLX_WEIGHT_QUANT", raising=False)
        s = Settings(_env_file=None)
        assert s.weight_quant is None

    def test_valid_hqq_4(self, monkeypatch):
        monkeypatch.setenv("OLMLX_WEIGHT_QUANT", "hqq:4")
        s = Settings()
        assert s.weight_quant == "hqq:4"

    def test_valid_hqq_8(self, monkeypatch):
        monkeypatch.setenv("OLMLX_WEIGHT_QUANT", "hqq:8")
        s = Settings()
        assert s.weight_quant == "hqq:8"

    def test_valid_hqq_4_with_group_size(self, monkeypatch):
        monkeypatch.setenv("OLMLX_WEIGHT_QUANT", "hqq:4:64")
        s = Settings()
        assert s.weight_quant == "hqq:4:64"

    def test_valid_hqq_4_with_group_size_128(self, monkeypatch):
        monkeypatch.setenv("OLMLX_WEIGHT_QUANT", "hqq:4:128")
        s = Settings()
        assert s.weight_quant == "hqq:4:128"

    def test_invalid_method(self, monkeypatch):
        monkeypatch.setenv("OLMLX_WEIGHT_QUANT", "bogus:4")
        with pytest.raises(ValidationError, match="weight_quant"):
            Settings()

    def test_invalid_bits(self, monkeypatch):
        monkeypatch.setenv("OLMLX_WEIGHT_QUANT", "hqq:16")
        with pytest.raises(ValidationError, match="weight_quant"):
            Settings()

    def test_invalid_format_no_colon(self, monkeypatch):
        monkeypatch.setenv("OLMLX_WEIGHT_QUANT", "hqq4")
        with pytest.raises(ValidationError, match="weight_quant"):
            Settings()

    def test_invalid_group_size(self, monkeypatch):
        monkeypatch.setenv("OLMLX_WEIGHT_QUANT", "hqq:4:bad")
        with pytest.raises(ValidationError, match="weight_quant"):
            Settings()

    def test_invalid_group_size_not_multiple_of_32(self, monkeypatch):
        monkeypatch.setenv("OLMLX_WEIGHT_QUANT", "hqq:4:7")
        with pytest.raises(ValidationError, match="weight_quant"):
            Settings()


class TestDistributedTensorOnly:
    def test_pipeline_strategy_rejected(self):
        with pytest.raises(ValidationError):
            Settings(distributed_strategy="pipeline")

    def test_tensor_strategy_accepted(self):
        assert Settings(distributed_strategy="tensor").distributed_strategy == "tensor"


def test_audio_max_bytes_default():
    from olmlx.config import Settings

    s = Settings()
    assert s.audio_max_bytes == 100 * 1024 * 1024


def test_audio_max_bytes_env_override(monkeypatch):
    monkeypatch.setenv("OLMLX_AUDIO_MAX_BYTES", "1048576")
    from olmlx.config import Settings

    s = Settings()
    assert s.audio_max_bytes == 1048576


def test_tracing_defaults_off(monkeypatch):
    monkeypatch.delenv("OLMLX_TRACING", raising=False)
    from olmlx.config import Settings

    assert Settings().tracing is False


def test_tracing_env_toggle(monkeypatch):
    monkeypatch.setenv("OLMLX_TRACING", "true")
    from olmlx.config import Settings

    assert Settings().tracing is True


def test_awq_gptq_convert_bits_rejects_unsupported(monkeypatch):
    """mlx supports bits {2,3,4,5,6,8}; 1 and 7 must be rejected, not just range-checked."""
    monkeypatch.setenv("OLMLX_AWQ_GPTQ_CONVERT_BITS", "7")
    from olmlx.config import Settings

    with pytest.raises(ValidationError):
        Settings()


def test_awq_gptq_convert_group_size_rejects_unsupported(monkeypatch):
    monkeypatch.setenv("OLMLX_AWQ_GPTQ_CONVERT_GROUP_SIZE", "16")
    from olmlx.config import Settings

    with pytest.raises(ValidationError):
        Settings()


def test_awq_gptq_convert_defaults(monkeypatch):
    for key in ("OLMLX_AWQ_GPTQ_CONVERT_BITS", "OLMLX_AWQ_GPTQ_CONVERT_GROUP_SIZE"):
        monkeypatch.delenv(key, raising=False)
    from olmlx.config import Settings

    s = Settings()
    assert s.awq_gptq_convert_bits == 4
    assert s.awq_gptq_convert_group_size == 64


class TestWarnLegacyFlashEnv:
    """warn_legacy_flash_env() detects promoted OLMLX_EXPERIMENTAL_FLASH*
    names (shell or .env) and warns WITHOUT forwarding the value."""

    def _clear_all(self, monkeypatch):
        from olmlx.config import PROMOTED_FLASH_ENV_RENAMES

        for old, new in PROMOTED_FLASH_ENV_RENAMES.items():
            monkeypatch.delenv(old, raising=False)
            monkeypatch.delenv(new, raising=False)

    def test_warns_on_shell_var_and_does_not_apply(self, monkeypatch, tmp_path, caplog):
        import logging

        from olmlx.config import settings, warn_legacy_flash_env

        monkeypatch.chdir(tmp_path)  # avoid the dev's real .env
        self._clear_all(monkeypatch)
        monkeypatch.setattr(settings, "flash", False, raising=False)
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_FLASH", "true")

        with caplog.at_level(logging.WARNING, logger="olmlx.config"):
            warn_legacy_flash_env()

        # Value is NOT forwarded.
        assert settings.flash is False
        # Warning names the legacy var and its replacement.
        assert "OLMLX_EXPERIMENTAL_FLASH" in caplog.text
        assert "OLMLX_FLASH" in caplog.text

    def test_warns_on_dotenv_var(self, monkeypatch, tmp_path, caplog):
        import logging

        from olmlx.config import settings, warn_legacy_flash_env

        monkeypatch.chdir(tmp_path)
        self._clear_all(monkeypatch)
        (tmp_path / ".env").write_text("OLMLX_EXPERIMENTAL_FLASH_PREFETCH=true\n")
        monkeypatch.setattr(settings, "flash_prefetch", False, raising=False)

        with caplog.at_level(logging.WARNING, logger="olmlx.config"):
            warn_legacy_flash_env()

        assert settings.flash_prefetch is False
        assert "OLMLX_EXPERIMENTAL_FLASH_PREFETCH" in caplog.text

    def test_no_warn_when_unset(self, monkeypatch, tmp_path, caplog):
        import logging

        from olmlx.config import warn_legacy_flash_env

        monkeypatch.chdir(tmp_path)
        self._clear_all(monkeypatch)

        with caplog.at_level(logging.WARNING, logger="olmlx.config"):
            warn_legacy_flash_env()

        assert not any(r.levelno >= logging.WARNING for r in caplog.records), (
            "Expected no warnings but got: " + caplog.text
        )

    def test_warns_for_each_family(self, monkeypatch, tmp_path, caplog):
        import logging

        from olmlx.config import warn_legacy_flash_env

        monkeypatch.chdir(tmp_path)
        self._clear_all(monkeypatch)
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_FLASH", "true")
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_FLASH_MOE", "true")
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_FLASH_SPECULATIVE", "true")

        with caplog.at_level(logging.WARNING, logger="olmlx.config"):
            warn_legacy_flash_env()

        assert "OLMLX_EXPERIMENTAL_FLASH_MOE" in caplog.text
        assert "OLMLX_EXPERIMENTAL_FLASH_SPECULATIVE" in caplog.text

    def test_presence_warns_even_for_false_value(self, monkeypatch, tmp_path, caplog):
        import logging

        from olmlx.config import warn_legacy_flash_env

        monkeypatch.chdir(tmp_path)
        self._clear_all(monkeypatch)
        # A dead name is dead regardless of value — warn on presence.
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_FLASH", "false")

        with caplog.at_level(logging.WARNING, logger="olmlx.config"):
            warn_legacy_flash_env()

        assert "OLMLX_EXPERIMENTAL_FLASH" in caplog.text

    def test_valid_experimental_tuning_knob_not_warned(
        self, monkeypatch, tmp_path, caplog
    ):
        import logging

        from olmlx.config import warn_legacy_flash_env

        monkeypatch.chdir(tmp_path)
        self._clear_all(monkeypatch)
        # Still-valid experimental tuning knob — must NOT be in the table.
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_FLASH_WINDOW_SIZE", "8")

        with caplog.at_level(logging.WARNING, logger="olmlx.config"):
            warn_legacy_flash_env()

        assert not any(r.levelno >= logging.WARNING for r in caplog.records)


class TestLegacyNamesInDotenv:
    """_legacy_names_in_dotenv returns the set of legacy names present in
    .env — key membership only. warn_legacy_flash_env never reads values
    (it warns on presence regardless of value), so the helper does not
    parse them."""

    def test_returns_set_of_present_names(self, monkeypatch, tmp_path):
        from olmlx.config import _legacy_names_in_dotenv

        monkeypatch.chdir(tmp_path)
        (tmp_path / ".env").write_text(
            "# comment\n"
            "\n"
            "OLMLX_EXPERIMENTAL_FLASH=true\n"
            "export OLMLX_EXPERIMENTAL_FLASH_MOE=false  # inline comment\n"
            "not-an-assignment\n"
            "OLMLX_FLASH_PREFETCH=true\n"
        )

        names = _legacy_names_in_dotenv(
            ("OLMLX_EXPERIMENTAL_FLASH", "OLMLX_EXPERIMENTAL_FLASH_MOE")
        )
        assert names == {"OLMLX_EXPERIMENTAL_FLASH", "OLMLX_EXPERIMENTAL_FLASH_MOE"}

    def test_empty_when_no_dotenv(self, monkeypatch, tmp_path):
        from olmlx.config import _legacy_names_in_dotenv

        monkeypatch.chdir(tmp_path)
        assert _legacy_names_in_dotenv(("OLMLX_EXPERIMENTAL_FLASH",)) == set()
