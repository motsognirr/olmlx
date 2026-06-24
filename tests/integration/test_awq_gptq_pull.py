"""Integration tests for AWQ/GPTQ auto-conversion on model pull."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from olmlx.models.manifest import ModelManifest
from olmlx.models.store import _converted_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_awq_dir(path: Path, fmt: str = "awq") -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2",
                "hidden_size": 512,
                "num_hidden_layers": 4,
                "quantization_config": {"quant_type": fmt},
            }
        )
    )


def _fake_convert_to_mlx(src: Path, dst: Path, bits: int, group_size: int) -> None:
    """Simulates mlx_lm.convert: creates a minimal MLX model dir."""
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2",
                "quantization": {"bits": bits, "group_size": group_size},
            }
        )
    )
    (dst / "conversion_source.json").write_text(
        json.dumps({"original_hf_path": str(src), "format": "awq"})
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pull_awq_triggers_conversion_and_updates_local_dir(
    mock_store, monkeypatch
):
    hf_path = "org/awq-model"
    raw_dir = mock_store.local_path(hf_path)

    def fake_ensure_downloaded(path: str) -> Path:
        _make_awq_dir(raw_dir, fmt="awq")
        return raw_dir

    monkeypatch.setattr(mock_store, "ensure_downloaded", fake_ensure_downloaded)

    with patch("olmlx.models.store.convert_to_mlx", side_effect=_fake_convert_to_mlx):
        events = []
        async for event in mock_store.pull(hf_path):
            events.append(event)

    assert any(e["status"] == "success" for e in events)

    converted_dir = _converted_path(mock_store.models_dir, hf_path)
    assert converted_dir.exists()
    manifest = ModelManifest.load(converted_dir / "manifest.json")
    assert manifest.format == "mlx"
    assert manifest.quantization_level == "4-bit"


@pytest.mark.asyncio
async def test_pull_gptq_triggers_conversion(mock_store, monkeypatch):
    hf_path = "org/gptq-model"
    raw_dir = mock_store.local_path(hf_path)

    def fake_ensure_downloaded(path: str) -> Path:
        _make_awq_dir(raw_dir, fmt="gptq")
        return raw_dir

    monkeypatch.setattr(mock_store, "ensure_downloaded", fake_ensure_downloaded)

    convert_calls: list[tuple] = []

    def tracking_convert(src, dst, bits, group_size):
        convert_calls.append((src, dst, bits, group_size))
        _fake_convert_to_mlx(src, dst, bits, group_size)

    with patch("olmlx.models.store.convert_to_mlx", side_effect=tracking_convert):
        async for _ in mock_store.pull(hf_path):
            pass

    assert len(convert_calls) == 1
    _, dst, _, _ = convert_calls[0]
    assert dst == _converted_path(mock_store.models_dir, hf_path)


@pytest.mark.asyncio
async def test_pull_non_awq_gptq_skips_conversion(mock_store, monkeypatch):
    hf_path = "org/plain-fp16"
    raw_dir = mock_store.local_path(hf_path)

    def fake_ensure_downloaded(path: str) -> Path:
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / "config.json").write_text(
            json.dumps({"model_type": "qwen2", "hidden_size": 512})
        )
        return raw_dir

    monkeypatch.setattr(mock_store, "ensure_downloaded", fake_ensure_downloaded)

    convert_called = []

    with patch(
        "olmlx.models.store.convert_to_mlx",
        side_effect=lambda *a, **kw: convert_called.append(a),
    ):
        async for _ in mock_store.pull(hf_path):
            pass

    assert convert_called == [], "convert_to_mlx must NOT be called for plain FP16"


@pytest.mark.asyncio
async def test_pull_already_converted_skips_download_and_conversion(
    mock_store, monkeypatch
):
    hf_path = "org/awq-already-converted"
    converted_dir = _converted_path(mock_store.models_dir, hf_path)
    # Pre-create a valid converted dir (simulates a prior pull)
    _fake_convert_to_mlx(Path("/fake/src"), converted_dir, bits=4, group_size=64)

    ensure_called = []
    monkeypatch.setattr(
        mock_store,
        "ensure_downloaded",
        lambda p: ensure_called.append(p) or Path(p),
    )
    convert_called = []

    with patch(
        "olmlx.models.store.convert_to_mlx",
        side_effect=lambda *a, **kw: convert_called.append(a),
    ):
        events = []
        async for event in mock_store.pull(hf_path):
            events.append(event)

    assert any(e["status"] == "already downloaded" for e in events)
    assert ensure_called == [], "ensure_downloaded must NOT be called"
    assert convert_called == [], "convert_to_mlx must NOT be called"


@pytest.mark.asyncio
async def test_pull_progress_events_include_converting_status(mock_store, monkeypatch):
    hf_path = "org/awq-progress-test"
    raw_dir = mock_store.local_path(hf_path)

    def fake_ensure_downloaded(path: str) -> Path:
        _make_awq_dir(raw_dir, fmt="awq")
        return raw_dir

    monkeypatch.setattr(mock_store, "ensure_downloaded", fake_ensure_downloaded)

    with patch("olmlx.models.store.convert_to_mlx", side_effect=_fake_convert_to_mlx):
        events = []
        async for event in mock_store.pull(hf_path):
            events.append(event)

    statuses = [e.get("status", "") for e in events]
    assert any("converting" in s for s in statuses), (
        f"Expected a 'converting' status event; got: {statuses}"
    )


@pytest.mark.asyncio
async def test_pull_removes_source_when_flag_set(mock_store, monkeypatch):
    hf_path = "org/awq-remove-src"
    raw_dir = mock_store.local_path(hf_path)

    def fake_ensure_downloaded(path: str) -> Path:
        _make_awq_dir(raw_dir, fmt="awq")
        return raw_dir

    monkeypatch.setattr(mock_store, "ensure_downloaded", fake_ensure_downloaded)
    monkeypatch.setattr("olmlx.models.store.settings.awq_gptq_remove_source", True)

    with patch("olmlx.models.store.convert_to_mlx", side_effect=_fake_convert_to_mlx):
        async for _ in mock_store.pull(hf_path):
            pass

    assert not raw_dir.exists(), (
        "source dir must be deleted when awq_gptq_remove_source=True"
    )


@pytest.mark.asyncio
async def test_pull_keeps_source_when_flag_cleared(mock_store, monkeypatch):
    hf_path = "org/awq-keep-src"
    raw_dir = mock_store.local_path(hf_path)

    def fake_ensure_downloaded(path: str) -> Path:
        _make_awq_dir(raw_dir, fmt="awq")
        return raw_dir

    monkeypatch.setattr(mock_store, "ensure_downloaded", fake_ensure_downloaded)
    monkeypatch.setattr("olmlx.models.store.settings.awq_gptq_remove_source", False)

    with patch("olmlx.models.store.convert_to_mlx", side_effect=_fake_convert_to_mlx):
        async for _ in mock_store.pull(hf_path):
            pass

    assert raw_dir.exists(), "source dir must be kept when awq_gptq_remove_source=False"


@pytest.mark.asyncio
async def test_pull_mlx_native_dir_not_re_converted(mock_store, monkeypatch):
    """An already-MLX model (top-level 'quantization') must not trigger conversion."""
    hf_path = "mlx-community/already-mlx-4bit"
    raw_dir = mock_store.local_path(hf_path)

    def fake_ensure_downloaded(path: str) -> Path:
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / "config.json").write_text(
            json.dumps({"model_type": "qwen2", "quantization": {"bits": 4}})
        )
        return raw_dir

    monkeypatch.setattr(mock_store, "ensure_downloaded", fake_ensure_downloaded)

    convert_called = []
    with patch(
        "olmlx.models.store.convert_to_mlx",
        side_effect=lambda *a, **kw: convert_called.append(a),
    ):
        async for _ in mock_store.pull(hf_path):
            pass

    assert convert_called == []
