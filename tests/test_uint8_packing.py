"""uint8-packed MLX checkpoints (e.g. RecursiveSI/flint-int4) → stock uint32 layout."""

import json

import mlx.core as mx
import pytest

from olmlx.engine.uint8_packing import (
    PackedWeightsError,
    normalize_uint8_packing_in_place,
)

GROUP = 64


def _quantized(shape, bits, key):
    w = mx.random.normal(shape, key=mx.random.key(key))
    q, scales, biases = mx.quantize(w, group_size=GROUP, bits=bits)
    return w, q, scales, biases


def _write_checkpoint(model_dir, *, as_uint8):
    """Two shards, no index: a 4-bit projection and an 8-bit embedding (config says 4)."""
    model_dir.mkdir()
    pack = (lambda q: q.view(mx.uint8)) if as_uint8 else (lambda q: q)
    originals = {}
    _, q, s, b = _quantized((32, 256), 4, 0)
    originals["model.layers.0.proj"] = (q, s, b, 4)
    mx.save_safetensors(
        str(model_dir / "model-00000.safetensors"),
        {
            "model.layers.0.proj.weight": pack(q),
            "model.layers.0.proj.scales": s,
            "model.layers.0.proj.biases": b,
            "model.layers.0.norm.weight": mx.ones((256,)),
        },
        metadata={"format": "mlx"},
    )
    _, q, s, b = _quantized((100, 128), 8, 1)
    originals["model.embed_tokens"] = (q, s, b, 8)
    mx.save_safetensors(
        str(model_dir / "model-00001.safetensors"),
        {
            "model.embed_tokens.weight": pack(q),
            "model.embed_tokens.scales": s,
            "model.embed_tokens.biases": b,
        },
        metadata={"format": "mlx"},
    )
    (model_dir / "config.json").write_text(
        json.dumps(
            {"model_type": "x", "quantization": {"group_size": GROUP, "bits": 4}}
        )
    )
    return originals


def _all_weights(model_dir):
    out = {}
    for p in sorted(model_dir.glob("*.safetensors")):
        out.update(mx.load(str(p)))
    return out


def test_converts_uint8_and_adds_bit_overrides(tmp_path):
    model_dir = tmp_path / "m"
    originals = _write_checkpoint(model_dir, as_uint8=True)

    assert normalize_uint8_packing_in_place(model_dir) is True

    weights = _all_weights(model_dir)
    for module, (q, s, b, bits) in originals.items():
        w = weights[f"{module}.weight"]
        assert w.dtype == mx.uint32
        assert mx.array_equal(w, q).item()
        restored = mx.dequantize(w, s, b, group_size=GROUP, bits=bits)
        assert mx.array_equal(
            restored, mx.dequantize(q, s, b, group_size=GROUP, bits=bits)
        ).item()
    assert weights["model.layers.0.norm.weight"].dtype == mx.float32

    cfg = json.loads((model_dir / "config.json").read_text())
    expected = {
        "group_size": GROUP,
        "bits": 4,
        "model.embed_tokens": {"group_size": GROUP, "bits": 8},
    }
    assert cfg["quantization"] == expected
    assert cfg["quantization_config"] == expected


def test_second_call_is_noop(tmp_path):
    model_dir = tmp_path / "m"
    _write_checkpoint(model_dir, as_uint8=True)
    normalize_uint8_packing_in_place(model_dir)
    mtimes = {p: p.stat().st_mtime_ns for p in model_dir.iterdir()}

    assert normalize_uint8_packing_in_place(model_dir) is False
    assert {p: p.stat().st_mtime_ns for p in model_dir.iterdir()} == mtimes


def test_stock_uint32_checkpoint_untouched(tmp_path):
    model_dir = tmp_path / "m"
    _write_checkpoint(model_dir, as_uint8=False)
    config_before = (model_dir / "config.json").read_text()

    assert normalize_uint8_packing_in_place(model_dir) is False
    assert (model_dir / "config.json").read_text() == config_before


def test_resumes_after_interrupted_rewrite(tmp_path):
    """Config written, one shard converted, a stale temp left behind: a rerun finishes."""
    model_dir = tmp_path / "m"
    originals = _write_checkpoint(model_dir, as_uint8=True)
    q, s, b, _ = originals["model.layers.0.proj"]
    mx.save_safetensors(
        str(model_dir / "model-00000.safetensors"),
        {
            "model.layers.0.proj.weight": q,
            "model.layers.0.proj.scales": s,
            "model.layers.0.proj.biases": b,
            "model.layers.0.norm.weight": mx.ones((256,)),
        },
    )
    (model_dir / "model-00001.u32tmp.safetensors").write_bytes(b"partial")

    assert normalize_uint8_packing_in_place(model_dir) is True

    assert not list(model_dir.glob("*.u32tmp.safetensors"))
    weights = _all_weights(model_dir)
    assert all(weights[f"{m}.weight"].dtype == mx.uint32 for m in originals)
    cfg = json.loads((model_dir / "config.json").read_text())
    assert cfg["quantization"]["model.embed_tokens"]["bits"] == 8


def test_rejects_uninferable_bit_width(tmp_path):
    model_dir = tmp_path / "m"
    model_dir.mkdir()
    mx.save_safetensors(
        str(model_dir / "model.safetensors"),
        {
            "a.weight": mx.zeros((4, 7), dtype=mx.uint8),
            "a.scales": mx.zeros((4, 1)),
            "a.biases": mx.zeros((4, 1)),
        },
    )
    (model_dir / "config.json").write_text(
        json.dumps({"quantization": {"group_size": GROUP, "bits": 4}})
    )

    with pytest.raises(PackedWeightsError, match="Cannot infer bit width"):
        normalize_uint8_packing_in_place(model_dir)


def test_existing_per_module_group_size_is_respected(tmp_path):
    """A module override with its own group_size must not be recomputed with the global one."""
    model_dir = tmp_path / "m"
    model_dir.mkdir()
    w = mx.random.normal((16, 256), key=mx.random.key(3))
    q, s, b = mx.quantize(w, group_size=32, bits=4)
    mx.save_safetensors(
        str(model_dir / "model.safetensors"),
        {"a.weight": q.view(mx.uint8), "a.scales": s, "a.biases": b},
    )
    quant = {"group_size": 64, "bits": 4, "a": {"group_size": 32, "bits": 4}}
    (model_dir / "config.json").write_text(json.dumps({"quantization": quant}))

    assert normalize_uint8_packing_in_place(model_dir) is True

    assert json.loads((model_dir / "config.json").read_text())["quantization"] == quant


def test_interrupted_config_write_leaves_original(tmp_path):
    """A stale config.json.tmp from a crash is discarded; config.json itself stays valid."""
    model_dir = tmp_path / "m"
    _write_checkpoint(model_dir, as_uint8=True)
    (model_dir / "config.json.tmp").write_text("{trunc")

    assert normalize_uint8_packing_in_place(model_dir) is True

    assert not (model_dir / "config.json.tmp").exists()
    cfg = json.loads((model_dir / "config.json").read_text())
    assert cfg["quantization"]["model.embed_tokens"]["bits"] == 8


def test_concurrent_callers_convert_once(tmp_path):
    """Two first loads racing: the flock serializes them, the second finds nothing to do."""
    import threading

    model_dir = tmp_path / "m"
    originals = _write_checkpoint(model_dir, as_uint8=True)
    results, errors = [], []

    def run():
        try:
            results.append(normalize_uint8_packing_in_place(model_dir))
        except Exception as exc:  # surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert sorted(results) == [False, True]
    weights = _all_weights(model_dir)
    assert all(weights[f"{m}.weight"].dtype == mx.uint32 for m in originals)


def test_gemma4_unified_path_is_normalized_first(tmp_path, monkeypatch):
    """The gemma4_unified early return must not bypass normalization."""
    from unittest.mock import MagicMock

    import olmlx.engine.model_load_utils as mlu

    calls = []
    monkeypatch.setattr(
        mlu, "normalize_uint8_packing_in_place", lambda p: calls.append("norm")
    )
    monkeypatch.setattr(
        mlu,
        "_maybe_load_gemma4_unified_text",
        lambda p: calls.append("gemma4") or (MagicMock(), MagicMock()),
    )
    monkeypatch.setattr(mlu, "_materialize_module_buffers", lambda m: None)

    mlu._load_with_model_type_fallback(MagicMock(), tmp_path)

    assert calls == ["norm", "gemma4"]
