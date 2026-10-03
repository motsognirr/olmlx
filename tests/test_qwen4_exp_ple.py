"""Qwen4-Exp (Qwen3.8-Flash-Next) external PLE view.

The checkpoint carries ~51B n-gram (PLE) embedding parameters as 128 Q4
shards (~30 GiB at 4-bit). Loaded as ordinary MLX parameters they are
materialized in RAM — on the flash-MoE path ``wrap_flash_moe`` evals every
non-expert parameter, so they would all be wired. mlx-vlm can instead serve
them by mmap row lookup when ``text_config.ple_storage`` names a manifest;
``ensure_external_ple_view`` builds the hard-linked model view that enables it
and ``load_vlm`` routes qwen4_exp loads through it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest

from olmlx.engine.qwen4_exp_ple import VIEW_DIRNAME, ensure_external_ple_view

PLE_PREFIX = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards"
ROWS = 8
ROW_WIDTH = 160  # matches the real checkpoint's 160-wide rows
GROUP = 32


def _write_checkpoint(model_dir: Path, *, model_type: str = "qwen4_exp") -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    ple = {}
    for shard in range(2):
        prefix = f"{PLE_PREFIX}.{shard}"
        ple[f"{prefix}.weight"] = mx.zeros((ROWS, ROW_WIDTH // 8), dtype=mx.uint32)
        ple[f"{prefix}.scales"] = mx.ones((ROWS, ROW_WIDTH // GROUP), dtype=mx.bfloat16)
        ple[f"{prefix}.biases"] = mx.zeros(
            (ROWS, ROW_WIDTH // GROUP), dtype=mx.bfloat16
        )
    other = {"language_model.model.embed_tokens.weight": mx.zeros((4, 8))}
    mx.save_safetensors(str(model_dir / "model-00001-of-00002.safetensors"), ple)
    mx.save_safetensors(str(model_dir / "model-00002-of-00002.safetensors"), other)
    weight_map = {k: "model-00001-of-00002.safetensors" for k in ple}
    weight_map.update({k: "model-00002-of-00002.safetensors" for k in other})
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map})
    )
    quant = {"group_size": GROUP, "bits": 4, "mode": "affine"}
    for shard in range(2):
        quant[f"{PLE_PREFIX}.{shard}"] = dict(quant)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": model_type,
                "quantization": quant,
                "text_config": {"model_type": f"{model_type}_text"},
            }
        )
    )
    (model_dir / "tokenizer.json").write_text("{}")


def test_builds_hard_linked_view_with_ple_storage(tmp_path):
    model_dir = tmp_path / "model"
    _write_checkpoint(model_dir)

    view = ensure_external_ple_view(model_dir)

    assert view == model_dir / VIEW_DIRNAME
    config = json.loads((view / "config.json").read_text())
    assert config["text_config"]["ple_storage"]["manifest"] == "ple-store.json"
    # PLE quantization overrides are dropped (their modules no longer exist).
    assert not any(".ngram_embedding.shards." in k for k in config["quantization"])
    index = json.loads((view / "model.safetensors.index.json").read_text())
    assert not any(".ngram_embedding.shards." in k for k in index["weight_map"])
    # Weight files are hard links, not copies: no extra disk.
    for name in (
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ):
        if (view / name).exists():
            assert os.stat(view / name).st_ino == os.stat(model_dir / name).st_ino
    # Non-weight sidecars (tokenizer) come along so the processor loads.
    assert (view / "tokenizer.json").exists()
    manifest = json.loads((view / "ple-store.json").read_text())
    assert manifest["row_count"] == 2 * ROWS
    assert manifest["row_width"] == ROW_WIDTH


def test_view_mmap_table_reads_rows(tmp_path):
    """The manifest resolves back to the source checkpoint from the view."""
    from mlx_vlm.models.qwen4_exp.ple_storage import QuantizedMMapNGramEmbedding

    model_dir = tmp_path / "model"
    _write_checkpoint(model_dir)
    view = ensure_external_ple_view(model_dir)

    table = QuantizedMMapNGramEmbedding(view / "ple-store.json")
    out = table(mx.array([0, ROWS + 1]))
    assert out.shape == (2, ROW_WIDTH)


def test_idempotent(tmp_path):
    model_dir = tmp_path / "model"
    _write_checkpoint(model_dir)
    view = ensure_external_ple_view(model_dir)
    marker = (view / "config.json").stat().st_mtime_ns

    assert ensure_external_ple_view(model_dir) == view
    assert (view / "config.json").stat().st_mtime_ns == marker


def test_rebuilds_when_source_weights_replaced(tmp_path):
    """A re-download replaces the files; stale hard links must not be served."""
    model_dir = tmp_path / "model"
    _write_checkpoint(model_dir)
    view = ensure_external_ple_view(model_dir)

    src = model_dir / "model-00002-of-00002.safetensors"
    data = src.read_bytes()
    src.unlink()
    src.write_bytes(data)  # new inode

    view = ensure_external_ple_view(model_dir)
    assert (view / src.name).stat().st_ino == src.stat().st_ino


def test_non_qwen4_exp_returns_none(tmp_path):
    model_dir = tmp_path / "model"
    _write_checkpoint(model_dir, model_type="qwen3_5_moe")
    assert ensure_external_ple_view(model_dir) is None
    assert not (model_dir / VIEW_DIRNAME).exists()


def test_already_external_returns_none(tmp_path):
    model_dir = tmp_path / "model"
    _write_checkpoint(model_dir)
    cfg = json.loads((model_dir / "config.json").read_text())
    cfg["text_config"]["ple_storage"] = {"manifest": "ple-store.json"}
    (model_dir / "config.json").write_text(json.dumps(cfg))
    assert ensure_external_ple_view(model_dir) is None


def test_missing_dir_or_repo_id_returns_none(tmp_path):
    assert ensure_external_ple_view(tmp_path / "nope") is None
    assert ensure_external_ple_view("mlx-community/whatever") is None


def test_load_vlm_routes_qwen4_exp_through_view(tmp_path, monkeypatch):
    import mlx_vlm

    from olmlx.engine.vlm_load import load_vlm

    model_dir = tmp_path / "model"
    _write_checkpoint(model_dir)
    seen = []

    def fake_load(path, **kwargs):
        seen.append(path)
        return nn.Module(), object()

    monkeypatch.setattr(mlx_vlm, "load", fake_load)
    load_vlm(str(model_dir), lazy=True)
    assert seen == [str(model_dir / VIEW_DIRNAME)]


def test_load_vlm_leaves_other_models_alone(tmp_path, monkeypatch):
    import mlx_vlm

    from olmlx.engine.vlm_load import load_vlm

    model_dir = tmp_path / "model"
    _write_checkpoint(model_dir, model_type="gemma4")
    seen = []
    monkeypatch.setattr(
        mlx_vlm, "load", lambda path, **kw: (seen.append(path), (nn.Module(), None))[1]
    )
    load_vlm(str(model_dir))
    assert seen == [str(model_dir)]


@pytest.mark.parametrize("linked", [True, False])
def test_store_dir_size_counts_hard_links_once(tmp_path, linked):
    from olmlx.models.store import _dir_size

    a = tmp_path / "a.bin"
    a.write_bytes(b"x" * 1000)
    sub = tmp_path / "view"
    sub.mkdir()
    if linked:
        os.link(a, sub / "a.bin")
        assert _dir_size(tmp_path) == 1000
    else:
        (sub / "a.bin").write_bytes(b"x" * 1000)
        assert _dir_size(tmp_path) == 2000
