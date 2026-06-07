"""Regression coverage for olmlx.engine.flash.moe_weight_store.

Targets the per-layer LRU expert cache, pread loading, stats bookkeeping,
the legacy (manifest-free) raw-byte parsers, and failure handling. All tests
are hermetic: small synthetic .flashexperts bundles under tmp_path, no GPU
model loads, no network.
"""

import json

import mlx.core as mx
import numpy as np
import pytest

from olmlx.engine.flash.moe_bundler import (
    MoeExpertLayout,
    bundle_moe_experts,
)
from olmlx.engine.flash.moe_weight_store import (
    ExpertCacheStats,
    FlashMoeWeightStore,
)
from tests.test_flash_moe_bundler import _make_synthetic_moe_weights


# ---------------------------------------------------------------------------
# Fixtures: real bundles built via the bundler, loaded by the store.
# ---------------------------------------------------------------------------


@pytest.fixture()
def fp16_bundle(tmp_path):
    """Build a non-quantized MoE bundle and return (output_dir, params)."""
    hidden, inter, experts = 32, 16, 8
    num_dense, num_moe = 1, 2
    model_dir = _make_synthetic_moe_weights(
        hidden, inter, experts, num_moe, num_dense, tmp_path
    )
    output_dir = tmp_path / "flash_moe"
    bundle_moe_experts(model_dir, output_dir)
    # MoE layers are 1 and 2 (dense layer 0 skipped).
    return output_dir, hidden, inter, experts


@pytest.fixture()
def quant_bundle(tmp_path):
    """Build a 4-bit quantized MoE bundle."""
    hidden, inter, experts = 64, 32, 8
    num_dense, num_moe = 0, 1
    model_dir = _make_synthetic_moe_weights(
        hidden, inter, experts, num_moe, num_dense, tmp_path, quantized=True
    )
    output_dir = tmp_path / "flash_moe_q"
    bundle_moe_experts(model_dir, output_dir)
    return output_dir, hidden, inter, experts


# ---------------------------------------------------------------------------
# load_experts: hits, misses, stacking, remap.
# ---------------------------------------------------------------------------


def test_load_experts_miss_then_hit_updates_stats(fp16_bundle):
    output_dir, hidden, inter, experts = fp16_bundle
    with FlashMoeWeightStore(
        output_dir, num_io_threads=2, cache_budget_experts=16
    ) as store:
        # First call: all 3 are cold misses.
        loaded = store.load_experts(1, [0, 1, 2])
        snap1 = store.stats.snapshot()
        assert snap1["load_calls"] == 1
        assert snap1["cache_misses"] == 3
        assert snap1["cache_hits"] == 0
        assert snap1["load_failures"] == 0

        # Shapes: stacked over the requested experts.
        assert loaded.gate_weight.shape == (3, inter, hidden)
        assert loaded.up_weight.shape == (3, inter, hidden)
        assert loaded.down_weight.shape == (3, hidden, inter)
        assert loaded.is_quantized is False
        # fp16 experts carry no scales/biases.
        assert loaded.gate_scales is None

        # Second call: same experts are warm hits.
        store.load_experts(1, [0, 1, 2])
        snap2 = store.stats.snapshot()
        assert snap2["load_calls"] == 2
        assert snap2["cache_hits"] == 3
        assert snap2["cache_misses"] == 3  # unchanged from first call
        assert store.stats.hit_rate() == pytest.approx(3 / 6)


def test_load_experts_partial_hit(fp16_bundle):
    output_dir, _hidden, _inter, _experts = fp16_bundle
    with FlashMoeWeightStore(
        output_dir, num_io_threads=2, cache_budget_experts=16
    ) as store:
        store.load_experts(1, [0, 1])  # warm 0, 1
        store.stats.snapshot()
        store.load_experts(1, [1, 2, 3])  # 1 hit, 2 misses
        snap = store.stats.snapshot()
        assert snap["cache_hits"] == 1
        assert snap["cache_misses"] == 2 + 2  # first call's 2 + this call's 2


def test_remap_lut_and_index_map(fp16_bundle):
    output_dir, _hidden, _inter, experts = fp16_bundle
    with FlashMoeWeightStore(
        output_dir, num_io_threads=2, cache_budget_experts=16
    ) as store:
        req = [5, 2, 7]
        loaded = store.load_experts(1, req)
        # Index map maps global expert idx -> position in requested order.
        assert loaded.expert_index_map == {5: 0, 2: 1, 7: 2}
        lut = np.array(loaded.remap_lut)
        assert lut.shape == (experts,)
        assert lut[5] == 0
        assert lut[2] == 1
        assert lut[7] == 2
        # Unused entries carry the sentinel.
        assert lut[0] == 0xFFFFFFFF


def test_load_experts_empty_raises(fp16_bundle):
    output_dir, *_ = fp16_bundle
    with FlashMoeWeightStore(output_dir) as store:
        with pytest.raises(ValueError, match="must not be empty"):
            store.load_experts(1, [])


def test_load_experts_preserves_request_order(fp16_bundle):
    output_dir, _hidden, _inter, _experts = fp16_bundle
    with FlashMoeWeightStore(
        output_dir, num_io_threads=4, cache_budget_experts=16
    ) as store:
        # Load each expert individually so we can compare against a reordered
        # batch load — stacking must follow the *requested* order, not the
        # completion order of the parallel reads.
        singles = {
            i: np.array(store.load_experts(1, [i]).gate_weight[0]) for i in (3, 6, 1)
        }
        batch = store.load_experts(1, [6, 1, 3])
        gw = np.array(batch.gate_weight)
        assert np.array_equal(gw[0], singles[6])
        assert np.array_equal(gw[1], singles[1])
        assert np.array_equal(gw[2], singles[3])


# ---------------------------------------------------------------------------
# Cache eviction with a tiny budget.
# ---------------------------------------------------------------------------


def test_cache_eviction_with_small_budget(fp16_bundle):
    output_dir, _hidden, _inter, _experts = fp16_bundle
    # Budget of 2 experts per layer.
    with FlashMoeWeightStore(
        output_dir, num_io_threads=2, cache_budget_experts=2
    ) as store:
        store.load_experts(1, [0, 1])  # cache now {0, 1}
        store.load_experts(1, [2])  # inserting 2 evicts LRU (0)
        # Re-requesting 0 must miss again (it was evicted).
        store.stats.snapshot()
        store.load_experts(1, [0])
        snap = store.stats.snapshot()
        # 0 evicted -> cold miss again.
        assert snap["cache_misses"] >= 3
        # 2 is still resident -> warm hit.
        store.load_experts(1, [2])
        assert store.stats.snapshot()["cache_hits"] >= 1


def test_cache_is_per_layer(fp16_bundle):
    output_dir, _hidden, _inter, _experts = fp16_bundle
    with FlashMoeWeightStore(
        output_dir, num_io_threads=2, cache_budget_experts=16
    ) as store:
        store.load_experts(1, [0])
        # Same expert index, different layer -> still a miss.
        store.load_experts(2, [0])
        snap = store.stats.snapshot()
        assert snap["cache_misses"] == 2
        assert snap["cache_hits"] == 0


# ---------------------------------------------------------------------------
# Quantized loading.
# ---------------------------------------------------------------------------


def test_load_quantized_experts(quant_bundle):
    output_dir, hidden, inter, _experts = quant_bundle
    with FlashMoeWeightStore(
        output_dir, num_io_threads=2, cache_budget_experts=16
    ) as store:
        loaded = store.load_experts(0, [0, 1])
        assert loaded.is_quantized is True
        assert loaded.bits == 4
        assert loaded.group_size == 32
        # Packed uint32 weights and float16 scales/biases must be present.
        assert loaded.gate_weight.dtype == mx.uint32
        assert loaded.gate_scales is not None
        assert loaded.gate_biases is not None
        # Scales group along the input dim: hidden // group_size for gate/up.
        assert loaded.gate_scales.shape == (2, inter, hidden // 32)


# ---------------------------------------------------------------------------
# Failure path: a read that raises must count failures and re-raise first exc.
# ---------------------------------------------------------------------------


def test_load_failure_counts_and_raises(fp16_bundle, monkeypatch):
    output_dir, _hidden, _inter, _experts = fp16_bundle
    with FlashMoeWeightStore(
        output_dir, num_io_threads=2, cache_budget_experts=16
    ) as store:

        def boom(layer_idx, expert_idx):
            raise OSError("simulated SSD read failure")

        monkeypatch.setattr(store, "_read_expert", boom)
        with pytest.raises(OSError, match="simulated SSD read failure"):
            store.load_experts(1, [0, 1, 2])
        snap = store.stats.snapshot()
        # Every requested-but-missing expert is recorded as a failure.
        assert snap["load_failures"] == 3
        assert snap["cache_misses"] == 3
        assert snap["load_calls"] == 1


# ---------------------------------------------------------------------------
# __init__ failure cleanup when open_fds raises.
# ---------------------------------------------------------------------------


def test_init_open_fds_failure_calls_close(fp16_bundle, monkeypatch):
    output_dir, *_ = fp16_bundle
    import olmlx.engine.flash.moe_weight_store as mod

    def fail_open(*_a, **_k):
        raise OSError("cannot open fds")

    monkeypatch.setattr(mod, "open_fds", fail_open)
    with pytest.raises(OSError, match="cannot open fds"):
        FlashMoeWeightStore(output_dir, num_io_threads=2)


# ---------------------------------------------------------------------------
# ExpertCacheStats unit behavior.
# ---------------------------------------------------------------------------


def test_stats_hit_rate_zero_when_empty():
    stats = ExpertCacheStats()
    assert stats.hit_rate() == 0.0
    assert stats.snapshot() == {
        "load_calls": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "load_failures": 0,
    }


def test_stats_record_accumulates_and_hit_rate():
    stats = ExpertCacheStats()
    stats.record(hits=3, misses=1)
    stats.record(hits=0, misses=0, failures=2)
    snap = stats.snapshot()
    assert snap == {
        "load_calls": 2,
        "cache_hits": 3,
        "cache_misses": 1,
        "load_failures": 2,
    }
    assert stats.hit_rate() == pytest.approx(3 / 4)


# ---------------------------------------------------------------------------
# Legacy (manifest-free) raw-byte parsers, exercised directly.
# ---------------------------------------------------------------------------


def test_parse_float16_expert_roundtrip():
    hidden, inter = 8, 4
    rng = np.random.RandomState(0)
    gate = rng.randn(inter, hidden).astype(np.float16)
    up = rng.randn(inter, hidden).astype(np.float16)
    down = rng.randn(hidden, inter).astype(np.float16)
    raw = gate.tobytes() + up.tobytes() + down.tobytes()

    out = FlashMoeWeightStore._parse_float16_expert(raw, hidden, inter)
    assert np.array_equal(np.array(out["gate_weight"]), gate)
    assert np.array_equal(np.array(out["up_weight"]), up)
    assert np.array_equal(np.array(out["down_weight"]), down)
    # All scale/bias slots are None for non-quantized.
    for key in ("gate_scales", "up_biases", "down_bias"):
        assert out[key] is None


def test_parse_quantized_expert_roundtrip():
    hidden, inter, bits, group_size = 64, 32, 4, 32
    layout = MoeExpertLayout(
        layer_idx=0,
        num_experts=4,
        hidden_size=hidden,
        intermediate_size=inter,
        expert_byte_size=0,
        file_path=None,
        offsets=np.zeros(4, dtype=np.uint64),
        is_quantized=True,
        bits=bits,
        group_size=group_size,
    )
    gate_packed = hidden * bits // 32
    down_packed = inter * bits // 32
    rng = np.random.RandomState(1)

    parts = []
    expected = {}
    for proj, out_dim, packed_dim, in_dim in (
        ("gate", inter, gate_packed, hidden),
        ("up", inter, gate_packed, hidden),
        ("down", hidden, down_packed, inter),
    ):
        w = rng.randint(0, 2**31, (out_dim, packed_dim)).astype(np.uint32)
        s_dim = in_dim // group_size
        s = rng.randn(out_dim, s_dim).astype(np.float16)
        b = rng.randn(out_dim, s_dim).astype(np.float16)
        parts += [w.tobytes(), s.tobytes(), b.tobytes()]
        expected[proj] = (w, s, b)

    raw = b"".join(parts)
    out = FlashMoeWeightStore._parse_quantized_expert(raw, layout)
    for proj, (w, s, b) in expected.items():
        assert np.array_equal(np.array(out[f"{proj}_weight"]), w)
        assert np.array_equal(np.array(out[f"{proj}_scales"]), s)
        assert np.array_equal(np.array(out[f"{proj}_biases"]), b)
        assert out[f"{proj}_bias"] is None


def test_parse_expert_with_manifest_fc_alias_and_fill():
    # fc1/fc2 style (non-gated): fc1 -> up, fc2 -> down, gate missing -> None.
    hidden, inter = 4, 3
    rng = np.random.RandomState(2)
    up = rng.randn(inter, hidden).astype(np.float16)
    down = rng.randn(hidden, inter).astype(np.float16)
    raw = up.tobytes() + down.tobytes()
    manifest = [
        {
            "name": "fc1.weight",
            "shape": [inter, hidden],
            "dtype": "float16",
            "nbytes": up.nbytes,
        },
        {
            "name": "fc2.weight",
            "shape": [hidden, inter],
            "dtype": "float16",
            "nbytes": down.nbytes,
        },
    ]
    out = FlashMoeWeightStore._parse_expert_with_manifest(raw, manifest)
    assert np.array_equal(np.array(out["up_weight"]), up)
    assert np.array_equal(np.array(out["down_weight"]), down)
    # gate_* never present in fc1/fc2 -> filled with None.
    assert out["gate_weight"] is None
    # Missing scales/biases filled for present projections too.
    assert out["up_scales"] is None
    assert out["down_bias"] is None


def test_parse_expert_with_manifest_proj_suffix_stripping():
    # gate_proj.scales -> "gate_scales", verifying _proj suffix stripping.
    hidden, inter, group_size = 8, 4, 8
    rng = np.random.RandomState(3)
    scales = rng.randn(inter, hidden // group_size).astype(np.float16)
    raw = scales.tobytes()
    manifest = [
        {
            "name": "gate_proj.scales",
            "shape": [inter, hidden // group_size],
            "dtype": "float16",
            "nbytes": scales.nbytes,
        }
    ]
    out = FlashMoeWeightStore._parse_expert_with_manifest(raw, manifest)
    assert np.array_equal(np.array(out["gate_scales"]), scales)
    assert out["gate_weight"] is None


def test_parse_expert_with_manifest_unsupported_dtype():
    manifest = [
        {"name": "gate_proj.weight", "shape": [1], "dtype": "bogus64", "nbytes": 8}
    ]
    with pytest.raises(ValueError, match="Unsupported dtype"):
        FlashMoeWeightStore._parse_expert_with_manifest(b"\x00" * 8, manifest)


# ---------------------------------------------------------------------------
# Legacy _read_expert dispatch (no manifest) — float16 path.
# ---------------------------------------------------------------------------


def test_read_expert_legacy_float16_path(fp16_bundle, monkeypatch):
    """When a layout has no manifest and the store has no global manifest,
    _read_expert falls back to _parse_float16_expert."""
    output_dir, hidden, inter, _experts = fp16_bundle
    with FlashMoeWeightStore(
        output_dir, num_io_threads=2, cache_budget_experts=16
    ) as store:
        # Strip manifests so the legacy float16 branch is taken.
        store._manifest = None
        for layout in store._layouts.values():
            layout.manifest = None
            layout.is_quantized = False
        out = store._read_expert(1, 0)
        assert np.array(out["gate_weight"]).shape == (inter, hidden)
        assert out["gate_scales"] is None


def test_read_expert_legacy_quantized_path(quant_bundle):
    """No manifest + is_quantized -> _parse_quantized_expert branch."""
    output_dir, hidden, inter, _experts = quant_bundle
    with FlashMoeWeightStore(
        output_dir, num_io_threads=2, cache_budget_experts=16
    ) as store:
        store._manifest = None
        for layout in store._layouts.values():
            layout.manifest = None
        out = store._read_expert(0, 0)
        assert np.array(out["gate_weight"]).dtype == np.uint32
        assert np.array(out["gate_scales"]).shape == (inter, hidden // 32)


# ---------------------------------------------------------------------------
# Lifecycle: close / context manager / __del__.
# ---------------------------------------------------------------------------


def test_close_releases_fds(fp16_bundle):
    output_dir, *_ = fp16_bundle
    store = FlashMoeWeightStore(output_dir, num_io_threads=2)
    fds = dict(store._fds)
    assert fds  # opened some
    store.close()
    # close_fds clears the mapping.
    assert store._fds == {}
    # File descriptors are actually closed.
    import os

    for fd in fds.values():
        with pytest.raises(OSError):
            os.fstat(fd)


def test_context_manager_enter_returns_self(fp16_bundle):
    output_dir, *_ = fp16_bundle
    store = FlashMoeWeightStore(output_dir, num_io_threads=2)
    with store as ctx:
        assert ctx is store
    assert store._fds == {}


def test_del_does_not_raise(fp16_bundle):
    output_dir, *_ = fp16_bundle
    store = FlashMoeWeightStore(output_dir, num_io_threads=2)
    # Explicitly invoke the finalizer; must be idempotent / non-raising.
    store.__del__()
    assert store._fds == {}


# ---------------------------------------------------------------------------
# quant_mode propagation through the layout config.
# ---------------------------------------------------------------------------


def test_quant_mode_default_affine(fp16_bundle):
    output_dir, *_ = fp16_bundle
    # Default quant_mode is "affine" when not specified.
    layout_cfg = json.loads((output_dir / "flash_moe_layout.json").read_text())
    assert layout_cfg.get("quant_mode", "affine") == "affine"
    with FlashMoeWeightStore(output_dir, num_io_threads=2) as store:
        assert store._quant_mode == "affine"
        loaded = store.load_experts(1, [0])
        assert loaded.quant_mode == "affine"
