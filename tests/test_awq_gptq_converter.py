"""Unit tests for convert_to_mlx() in olmlx.engine.awq_gptq_converter."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from olmlx.engine.awq_gptq_converter import (
    convert_to_mlx,
    converting_marker,
)


def _make_awq_src(tmp_path: Path, fmt: str = "awq") -> Path:
    src = tmp_path / "src"
    src.mkdir()
    (src / "config.json").write_text(
        json.dumps({"model_type": "qwen2", "quantization_config": {"quant_type": fmt}})
    )
    return src


def _fake_mlx_convert(*, hf_path, mlx_path, **kwargs):
    """Faithful stand-in for ``mlx_lm.convert``.

    Mirrors the real contract: it REFUSES a pre-existing ``mlx_path`` (the real
    one raises ``ValueError`` in that case) and otherwise creates the output
    directory.  Using this as the mock is what guards against the regression of
    pre-creating ``dst`` before calling convert.
    """
    p = Path(mlx_path)
    if p.exists():
        raise ValueError(f"Cannot save to the path {p} as it already exists.")
    p.mkdir(parents=True)
    (p / "config.json").write_text(json.dumps({"quantization": {"bits": 4}}))


def test_convert_calls_mlx_lm_convert_with_correct_args(tmp_path):
    src = _make_awq_src(tmp_path)
    dst = tmp_path / "dst"

    with patch("mlx_lm.convert", side_effect=_fake_mlx_convert) as mock_convert:
        convert_to_mlx(src, dst, bits=4, group_size=64)

    mock_convert.assert_called_once_with(
        hf_path=str(src),
        mlx_path=str(dst),
        quantize=True,
        q_bits=4,
        q_group_size=64,
    )


def test_convert_does_not_precreate_mlx_path(tmp_path):
    """Regression: dst must NOT exist when mlx_lm.convert is called.

    The real mlx_lm.convert raises ValueError if mlx_path already exists, so
    convert_to_mlx must let convert own the directory creation.
    """
    src = _make_awq_src(tmp_path)
    dst = tmp_path / "dst"

    seen_existing: list[bool] = []

    def check_not_preexisting(*, hf_path, mlx_path, **kwargs):
        seen_existing.append(Path(mlx_path).exists())
        _fake_mlx_convert(hf_path=hf_path, mlx_path=mlx_path, **kwargs)

    with patch("mlx_lm.convert", side_effect=check_not_preexisting):
        convert_to_mlx(src, dst, bits=4, group_size=64)

    assert seen_existing == [False], "dst must not exist when convert is called"
    assert (dst / "config.json").exists()


def test_convert_clears_partial_dst_from_prior_failure(tmp_path):
    """A leftover partial dst from a failed attempt is cleared before retry."""
    src = _make_awq_src(tmp_path)
    dst = tmp_path / "dst"
    # Simulate a partial dir left by a previous failed conversion.
    dst.mkdir()
    (dst / "garbage.bin").write_text("partial")

    with patch("mlx_lm.convert", side_effect=_fake_mlx_convert):
        convert_to_mlx(src, dst, bits=4, group_size=64)

    assert (dst / "config.json").exists()
    assert not (dst / "garbage.bin").exists(), "partial output must be cleared"


def test_convert_writes_conversion_source_json(tmp_path):
    src = _make_awq_src(tmp_path, fmt="awq")
    dst = tmp_path / "dst"

    with patch("mlx_lm.convert", side_effect=_fake_mlx_convert):
        convert_to_mlx(src, dst, bits=4, group_size=64)

    provenance = dst / "conversion_source.json"
    assert provenance.exists(), "conversion_source.json should be written on success"
    data = json.loads(provenance.read_text())
    assert data["original_hf_path"] == str(src)
    assert data["format"] == "awq"


def test_convert_writes_gptq_format_in_provenance(tmp_path):
    src = _make_awq_src(tmp_path, fmt="gptq")
    dst = tmp_path / "dst"

    with patch("mlx_lm.convert", side_effect=_fake_mlx_convert):
        convert_to_mlx(src, dst, bits=4, group_size=64)

    data = json.loads((dst / "conversion_source.json").read_text())
    assert data["format"] == "gptq"


def test_convert_sets_and_clears_converting_marker(tmp_path):
    src = _make_awq_src(tmp_path)
    dst = tmp_path / "dst"
    marker = converting_marker(dst)

    marker_states: list[bool] = []

    def capture_marker_state(*, hf_path, mlx_path, **kwargs):
        marker_states.append(marker.exists())
        _fake_mlx_convert(hf_path=hf_path, mlx_path=mlx_path, **kwargs)

    with patch("mlx_lm.convert", side_effect=capture_marker_state):
        convert_to_mlx(src, dst, bits=4, group_size=64)

    assert marker_states == [True], "marker must be present during mlx_lm.convert"
    assert not marker.exists(), "marker must be removed on success"
    # The marker lives OUTSIDE dst so it can't make mlx_path "already exist".
    assert marker.parent == dst.parent
    assert not (dst / ".converting").exists()


def test_convert_leaves_converting_marker_on_failure(tmp_path):
    src = _make_awq_src(tmp_path)
    dst = tmp_path / "dst"
    marker = converting_marker(dst)

    with patch("mlx_lm.convert", side_effect=RuntimeError("OOM")):
        with pytest.raises(RuntimeError, match="OOM"):
            convert_to_mlx(src, dst, bits=4, group_size=64)

    assert marker.exists(), (
        "marker must remain after failure so _is_valid_mlx_dir stays False"
    )
    assert not (dst / "conversion_source.json").exists()


def test_convert_creates_dst_directory(tmp_path):
    src = _make_awq_src(tmp_path)
    dst = tmp_path / "nested" / "dst"  # parent doesn't exist yet

    with patch("mlx_lm.convert", side_effect=_fake_mlx_convert):
        convert_to_mlx(src, dst, bits=4, group_size=64)

    assert dst.exists()


def test_convert_uses_caller_supplied_bits_and_group_size(tmp_path):
    src = _make_awq_src(tmp_path)
    dst = tmp_path / "dst"

    with patch("mlx_lm.convert", side_effect=_fake_mlx_convert) as mock_convert:
        convert_to_mlx(src, dst, bits=8, group_size=128)

    _, kwargs = mock_convert.call_args
    assert kwargs["q_bits"] == 8
    assert kwargs["q_group_size"] == 128
