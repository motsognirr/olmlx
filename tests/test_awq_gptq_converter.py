"""Unit tests for convert_to_mlx() in olmlx.engine.awq_gptq_converter."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from olmlx.engine.awq_gptq_converter import convert_to_mlx


def _make_awq_src(tmp_path: Path, fmt: str = "awq") -> Path:
    src = tmp_path / "src"
    src.mkdir()
    (src / "config.json").write_text(
        json.dumps({"model_type": "qwen2", "quantization_config": {"quant_type": fmt}})
    )
    return src


def test_convert_calls_mlx_lm_convert_with_correct_args(tmp_path):
    src = _make_awq_src(tmp_path)
    dst = tmp_path / "dst"

    with patch("mlx_lm.convert") as mock_convert:
        convert_to_mlx(src, dst, bits=4, group_size=64)

    mock_convert.assert_called_once_with(
        model=str(src),
        mlx_path=str(dst),
        quantize=True,
        q_bits=4,
        q_group_size=64,
    )


def test_convert_writes_conversion_source_json(tmp_path):
    src = _make_awq_src(tmp_path, fmt="awq")
    dst = tmp_path / "dst"

    with patch("mlx_lm.convert"):
        convert_to_mlx(src, dst, bits=4, group_size=64)

    provenance = dst / "conversion_source.json"
    assert provenance.exists(), "conversion_source.json should be written on success"
    data = json.loads(provenance.read_text())
    assert data["original_hf_path"] == str(src)
    assert data["format"] == "awq"


def test_convert_writes_gptq_format_in_provenance(tmp_path):
    src = _make_awq_src(tmp_path, fmt="gptq")
    dst = tmp_path / "dst"

    with patch("mlx_lm.convert"):
        convert_to_mlx(src, dst, bits=4, group_size=64)

    data = json.loads((dst / "conversion_source.json").read_text())
    assert data["format"] == "gptq"


def test_convert_sets_and_clears_converting_marker(tmp_path):
    src = _make_awq_src(tmp_path)
    dst = tmp_path / "dst"

    marker_states: list[bool] = []

    def capture_marker_state(model, mlx_path, **kwargs):
        marker_states.append((Path(mlx_path) / ".converting").exists())

    with patch("mlx_lm.convert", side_effect=capture_marker_state):
        convert_to_mlx(src, dst, bits=4, group_size=64)

    assert marker_states == [True], ".converting must be present during mlx_lm.convert"
    assert not (dst / ".converting").exists(), ".converting must be removed on success"


def test_convert_leaves_converting_marker_on_failure(tmp_path):
    src = _make_awq_src(tmp_path)
    dst = tmp_path / "dst"

    with patch("mlx_lm.convert", side_effect=RuntimeError("OOM")):
        with pytest.raises(RuntimeError, match="OOM"):
            convert_to_mlx(src, dst, bits=4, group_size=64)

    assert (dst / ".converting").exists(), (
        ".converting must remain after failure so _is_valid_mlx_dir stays False"
    )
    assert not (dst / "conversion_source.json").exists()


def test_convert_creates_dst_directory(tmp_path):
    src = _make_awq_src(tmp_path)
    dst = tmp_path / "nested" / "dst"  # parent doesn't exist yet

    with patch("mlx_lm.convert"):
        convert_to_mlx(src, dst, bits=4, group_size=64)

    assert dst.exists()


def test_convert_uses_caller_supplied_bits_and_group_size(tmp_path):
    src = _make_awq_src(tmp_path)
    dst = tmp_path / "dst"

    with patch("mlx_lm.convert") as mock_convert:
        convert_to_mlx(src, dst, bits=8, group_size=128)

    _, kwargs = mock_convert.call_args
    assert kwargs["q_bits"] == 8
    assert kwargs["q_group_size"] == 128
