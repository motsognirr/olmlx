"""Tests for DFlash/EAGLE draft quantization compatibility tracking (#516).

Covers:
- ``_quant_descriptor_from_path``: parse target config.json for quant desc
- ``_detect_live_quant``: walk a loaded model for QuantizedLinear layers
- ``_check_quant_compat``: warn / raise on mismatch
- DraftConfig / EagleConfig roundtrip with ``target_quant``
- stats_summary includes ``target_quant``
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# _quant_descriptor_from_path
# ---------------------------------------------------------------------------


def test_quant_descriptor_from_bf16_config(tmp_path: Path) -> None:
    """No quantization block → 'bf16'."""
    cfg = {"hidden_size": 512, "vocab_size": 1000}
    (tmp_path / "config.json").write_text(json.dumps(cfg))

    from olmlx.engine.speculative_loaders import _quant_descriptor_from_path

    assert _quant_descriptor_from_path(tmp_path) == "bf16"


def test_quant_descriptor_from_q4_config(tmp_path: Path) -> None:
    """config.json with quantization block → 'q4_g64'."""
    cfg = {
        "hidden_size": 512,
        "vocab_size": 1000,
        "quantization": {"bits": 4, "group_size": 64},
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg))

    from olmlx.engine.speculative_loaders import _quant_descriptor_from_path

    assert _quant_descriptor_from_path(tmp_path) == "q4_g64"


def test_quant_descriptor_from_quantization_config(tmp_path: Path) -> None:
    """config.json with quantization_config block → 'q4_g32'."""
    cfg = {
        "hidden_size": 512,
        "vocab_size": 1000,
        "quantization_config": {"bits": 4, "group_size": 32},
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg))

    from olmlx.engine.speculative_loaders import _quant_descriptor_from_path

    assert _quant_descriptor_from_path(tmp_path) == "q4_g32"


def test_quant_descriptor_null_group_size(tmp_path: Path) -> None:
    """config.json with group_size: null → falls back to 64, not 'q4_gNone'."""
    cfg = {
        "quantization": {"bits": 4, "group_size": None},
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg))

    from olmlx.engine.speculative_loaders import _quant_descriptor_from_path

    assert _quant_descriptor_from_path(tmp_path) == "q4_g64"


def test_quant_descriptor_from_gptq_config(tmp_path: Path) -> None:
    """quantize_config.json (GPTQ format) with only bits → 'q4_g64' (default group_size)."""
    cfg = {"hidden_size": 512, "vocab_size": 1000}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    # GPTQ uses quantize_config.json with bits; no group_size → fallback to 64
    (tmp_path / "quantize_config.json").write_text(json.dumps({"bits": 4}))

    from olmlx.engine.speculative_loaders import _quant_descriptor_from_path

    assert _quant_descriptor_from_path(tmp_path) == "q4_g64"


# ---------------------------------------------------------------------------
# DraftConfig / _draft_config_to_disk roundtrip
# ---------------------------------------------------------------------------


def test_dflash_config_roundtrip_with_target_quant() -> None:
    """DraftConfig(target_quant=...) roundtrips through _draft_config_to_disk."""
    from olmlx.engine.dflash.draft_model import DraftConfig
    from olmlx.engine.dflash.prepare import _draft_config_to_disk

    cfg = DraftConfig(
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=64,
        intermediate_size=512,
        vocab_size=1000,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        max_position_embeddings=512,
        block_size=4,
        num_target_layers=2,
        target_layer_ids=[1, 3],
        mask_token_id=0,
        target_quant="q4_g64",
    )
    on_disk = _draft_config_to_disk(cfg)
    assert on_disk["dflash_config"]["target_quant"] == "q4_g64"


def test_dflash_config_roundtrip_no_target_quant() -> None:
    """DraftConfig without target_quant → dflash_config has no target_quant key."""
    from olmlx.engine.dflash.draft_model import DraftConfig
    from olmlx.engine.dflash.prepare import _draft_config_to_disk

    cfg = DraftConfig(
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=64,
        intermediate_size=512,
        vocab_size=1000,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        max_position_embeddings=512,
        block_size=4,
        num_target_layers=2,
        target_layer_ids=[1, 3],
        mask_token_id=0,
    )
    on_disk = _draft_config_to_disk(cfg)
    assert "target_quant" not in on_disk["dflash_config"]


# ---------------------------------------------------------------------------
# EagleConfig / _config_to_disk roundtrip
# ---------------------------------------------------------------------------


def test_eagle_config_roundtrip_with_target_quant() -> None:
    """EagleConfig(target_quant=...) roundtrips through _config_to_disk."""
    from olmlx.engine.eagle.draft_model import EagleConfig
    from olmlx.engine.eagle.prepare import _config_to_disk

    cfg = EagleConfig(
        hidden_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=64,
        intermediate_size=512,
        vocab_size=1000,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        max_position_embeddings=512,
        block_size=4,
        target_quant="q4_g64",
    )
    on_disk = _config_to_disk(cfg)
    assert on_disk["eagle_config"]["target_quant"] == "q4_g64"


def test_eagle_config_roundtrip_no_target_quant() -> None:
    """EagleConfig without target_quant → eagle_config has no target_quant key."""
    from olmlx.engine.eagle.draft_model import EagleConfig
    from olmlx.engine.eagle.prepare import _config_to_disk

    cfg = EagleConfig(
        hidden_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=64,
        intermediate_size=512,
        vocab_size=1000,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        max_position_embeddings=512,
        block_size=4,
    )
    on_disk = _config_to_disk(cfg)
    assert "target_quant" not in on_disk["eagle_config"]


# ---------------------------------------------------------------------------
# _check_quant_compat
# ---------------------------------------------------------------------------


def test_compat_check_matching_no_warning(caplog: pytest.LogCaptureFixture) -> None:
    """Matching quant descriptors → no warning."""
    from olmlx.engine.speculative_loaders import _check_quant_compat

    with caplog.at_level(logging.WARNING, logger="olmlx.engine.speculative_loaders"):
        _check_quant_compat("q4_g64", "q4_g64", draft_path=Path("/fake/draft"))
    assert not caplog.records


def test_compat_check_mismatch_warns(caplog: pytest.LogCaptureFixture) -> None:
    """Mismatched quant descriptors → warning in log."""
    from olmlx.engine.speculative_loaders import _check_quant_compat

    with caplog.at_level(logging.WARNING, logger="olmlx.engine.speculative_loaders"):
        _check_quant_compat("q4_g64", "bf16", draft_path=Path("/fake/draft"))
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_compat_check_none_skips(caplog: pytest.LogCaptureFixture) -> None:
    """draft_quant=None (old checkpoint) → no warning, no error."""
    from olmlx.engine.speculative_loaders import _check_quant_compat

    with caplog.at_level(logging.WARNING, logger="olmlx.engine.speculative_loaders"):
        _check_quant_compat(None, "bf16", draft_path=Path("/fake/draft"))
    assert not caplog.records


def test_compat_check_strict_raises() -> None:
    """Mismatch with spec_strict_compat=True → RuntimeError."""
    from olmlx.engine.speculative_loaders import _check_quant_compat

    with patch("olmlx.engine.speculative_loaders.settings") as mock_settings:
        mock_settings.spec_strict_compat = True
        with pytest.raises(RuntimeError, match="q4_g64"):
            _check_quant_compat("q4_g64", "bf16", draft_path=Path("/fake/draft"))


# ---------------------------------------------------------------------------
# _detect_live_quant
# ---------------------------------------------------------------------------


def test_detect_live_quant_bf16_model() -> None:
    """Model with no QuantizedLinear → 'bf16'."""
    import mlx.nn as nn

    from olmlx.engine.speculative_loaders import _detect_live_quant

    class _PlainModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(16, 16, bias=False)

    model = _PlainModel()
    assert _detect_live_quant(model) == "bf16"


def test_detect_live_quant_quantized_model() -> None:
    """Model with QuantizedLinear child → 'q4_g64'."""
    import mlx.nn as nn

    from olmlx.engine.speculative_loaders import _detect_live_quant

    # Build a real QuantizedLinear to avoid relying on mock internals.
    # The last input dimension must be divisible by group_size (64).
    ql = nn.QuantizedLinear(64, 64, bits=4, group_size=64)

    class _QuantModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = ql

    model = _QuantModel()
    result = _detect_live_quant(model)
    assert result == "q4_g64"


# ---------------------------------------------------------------------------
# Decoder constructors propagate target_quant to _target_quant
# ---------------------------------------------------------------------------


def test_dflash_decoder_sets_target_quant() -> None:
    """DFlashDecoder.__init__ propagates draft_config.target_quant to _target_quant."""
    import mlx.nn as nn

    from olmlx.engine.dflash.decoder import DFlashDecoder
    from olmlx.engine.dflash.draft_model import DraftConfig, DFlashDraftModel

    cfg = DraftConfig(
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=64,
        intermediate_size=512,
        vocab_size=1000,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        max_position_embeddings=512,
        block_size=4,
        num_target_layers=2,
        target_layer_ids=[1, 3],
        mask_token_id=0,
        target_quant="q4_g64",
    )
    target = MagicMock(spec=nn.Module)
    draft = DFlashDraftModel(cfg)
    decoder = DFlashDecoder(
        target_model=target, draft_model=draft, draft_config=cfg, block_size=4
    )
    assert decoder._target_quant == "q4_g64"


def test_dflash_decoder_target_quant_none_defaults_empty() -> None:
    """DFlashDecoder._target_quant is '' when draft_config.target_quant is None."""
    import mlx.nn as nn

    from olmlx.engine.dflash.decoder import DFlashDecoder
    from olmlx.engine.dflash.draft_model import DraftConfig, DFlashDraftModel

    cfg = DraftConfig(
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=64,
        intermediate_size=512,
        vocab_size=1000,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        max_position_embeddings=512,
        block_size=4,
        num_target_layers=2,
        target_layer_ids=[1, 3],
        mask_token_id=0,
    )
    target = MagicMock(spec=nn.Module)
    draft = DFlashDraftModel(cfg)
    decoder = DFlashDecoder(
        target_model=target, draft_model=draft, draft_config=cfg, block_size=4
    )
    assert decoder._target_quant == ""


def test_eagle_decoder_sets_target_quant() -> None:
    """EagleDecoder.__init__ propagates target_quant to _target_quant."""
    import mlx.nn as nn

    from olmlx.engine.eagle.decoder import EagleDecoder
    from olmlx.engine.eagle.draft_model import EagleConfig, EagleDraftModel

    cfg = EagleConfig(
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=32,
        intermediate_size=128,
        vocab_size=1000,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        max_position_embeddings=512,
        block_size=4,
    )
    target = MagicMock(spec=nn.Module)
    target.layers = [MagicMock()]
    draft = EagleDraftModel(cfg)
    decoder = EagleDecoder(
        target_model=target, draft_model=draft, block_size=4, target_quant="q4_g64"
    )
    assert decoder._target_quant == "q4_g64"


def test_eagle_decoder_target_quant_none_defaults_empty() -> None:
    """EagleDecoder._target_quant is '' when target_quant is not passed."""
    import mlx.nn as nn

    from olmlx.engine.eagle.decoder import EagleDecoder
    from olmlx.engine.eagle.draft_model import EagleConfig, EagleDraftModel

    cfg = EagleConfig(
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=32,
        intermediate_size=128,
        vocab_size=1000,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        max_position_embeddings=512,
        block_size=4,
    )
    target = MagicMock(spec=nn.Module)
    target.layers = [MagicMock()]
    draft = EagleDraftModel(cfg)
    decoder = EagleDecoder(target_model=target, draft_model=draft, block_size=4)
    assert decoder._target_quant == ""


# ---------------------------------------------------------------------------
# stats_summary includes target_quant
# ---------------------------------------------------------------------------


def test_stats_summary_includes_target_quant() -> None:
    """SpecDecoderBase.stats_summary includes target_quant when set."""
    from olmlx.engine.spec_decoder_base import SpecDecoderBase

    class _ConcreteDecoder(SpecDecoderBase):
        def __init__(self) -> None:
            super().__init__()
            self._target = MagicMock()
            self._draft = MagicMock()

        def _prefill_impl(self, prompt: Any, **kwargs: Any) -> int:
            return 0

        def _step_impl(self) -> tuple[list[int], int]:
            return [], 0

        def _reset_state(self) -> None:
            pass

    dec = _ConcreteDecoder()
    dec._target_quant = "q4_g64"
    summary = dec.stats_summary()
    assert summary["target_quant"] == "q4_g64"


def test_stats_summary_target_quant_default_empty() -> None:
    """stats_summary target_quant is empty string by default."""
    from olmlx.engine.spec_decoder_base import SpecDecoderBase

    class _ConcreteDecoder(SpecDecoderBase):
        def __init__(self) -> None:
            super().__init__()
            self._target = MagicMock()
            self._draft = MagicMock()

        def _prefill_impl(self, prompt: Any, **kwargs: Any) -> int:
            return 0

        def _step_impl(self) -> tuple[list[int], int]:
            return [], 0

        def _reset_state(self) -> None:
            pass

    dec = _ConcreteDecoder()
    summary = dec.stats_summary()
    assert summary["target_quant"] == ""
