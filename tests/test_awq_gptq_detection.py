"""Unit tests for detect_format() in olmlx.engine.awq_gptq_converter."""

import json

from olmlx.engine.awq_gptq_converter import detect_format


def test_detect_awq_via_quantization_config_quant_type(tmp_path):
    config = {
        "model_type": "qwen2",
        "quantization_config": {"quant_type": "awq", "version": "gemm"},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    assert detect_format(tmp_path) == "awq"


def test_detect_awq_via_quantization_config_quant_method(tmp_path):
    """Canonical HF AutoAWQ checkpoints carry quant_method, not quant_type."""
    config = {
        "model_type": "qwen2",
        "quantization_config": {"quant_method": "awq", "bits": 4, "version": "gemm"},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    assert detect_format(tmp_path) == "awq"


def test_detect_gptq_via_quantization_config_quant_method(tmp_path):
    """Canonical HF GPTQ checkpoints carry quant_method, not quant_type."""
    config = {
        "model_type": "qwen2",
        "quantization_config": {"quant_method": "gptq", "bits": 4, "group_size": 128},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    assert detect_format(tmp_path) == "gptq"


def test_detect_case_insensitive_quant_method(tmp_path):
    """quant_method is normalised to lowercase before comparison."""
    config = {"quantization_config": {"quant_method": "GPTQ"}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    assert detect_format(tmp_path) == "gptq"


def test_detect_gptq_via_quantize_config_json(tmp_path):
    """quantize_config.json present (auto_gptq convention) → gptq."""
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen2"}))
    (tmp_path / "quantize_config.json").write_text(
        json.dumps({"bits": 4, "group_size": 128})
    )
    assert detect_format(tmp_path) == "gptq"


def test_detect_gptq_via_quantization_config_quant_type(tmp_path):
    config = {
        "model_type": "qwen2",
        "quantization_config": {"quant_type": "gptq"},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    assert detect_format(tmp_path) == "gptq"


def test_detect_none_for_mlx_native(tmp_path):
    """MLX-native quantization (top-level 'quantization' key, no quant_type) → None."""
    config = {
        "model_type": "qwen2",
        "quantization": {"bits": 4, "group_size": 64},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    assert detect_format(tmp_path) is None


def test_detect_none_for_plain_fp16(tmp_path):
    """No quantization keys at all → None."""
    config = {"model_type": "qwen2", "hidden_size": 4096}
    (tmp_path / "config.json").write_text(json.dumps(config))
    assert detect_format(tmp_path) is None


def test_detect_none_when_no_config_json(tmp_path):
    """No config.json and no quantize_config.json → None."""
    assert detect_format(tmp_path) is None


def test_detect_gptq_fallback_without_config_json(tmp_path):
    """quantize_config.json alone (no config.json) → gptq."""
    (tmp_path / "quantize_config.json").write_text(json.dumps({"bits": 4}))
    assert detect_format(tmp_path) == "gptq"


def test_detect_case_insensitive_quant_type(tmp_path):
    """quant_type is normalised to lowercase before comparison."""
    config = {"quantization_config": {"quant_type": "AWQ"}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    assert detect_format(tmp_path) == "awq"


def test_detect_none_for_malformed_config(tmp_path):
    """Invalid JSON in config.json → None (graceful fallback)."""
    (tmp_path / "config.json").write_text("{not valid json")
    assert detect_format(tmp_path) is None


def test_detect_awq_ignores_top_level_quantization(tmp_path):
    """quantization_config.quant_type=awq wins even when top-level quantization exists."""
    config = {
        "quantization": {"bits": 4},
        "quantization_config": {"quant_type": "awq"},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    assert detect_format(tmp_path) == "awq"
