"""Attention-aware spectral calibration (``spectral-qa``, #749).

The ``spectral-qa`` variant weights the key-side basis choice and bit split by
query statistics, so quantization error lands where queries don't look. The
runtime cache is the unchanged ``SpectralQuantKVCache``; only calibration
differs.
"""

from __future__ import annotations

import json
import threading
from unittest.mock import MagicMock, patch

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.cache import KVCache

import olmlx.engine.spectralquant_calibrate as sc
from olmlx.engine.spectralquant import (
    SpectralRotation,
    spectral_dequantize,
    spectral_quantize,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _runtime_roundtrip(keys: np.ndarray, cal: dict) -> np.ndarray:
    """Quantize + dequantize through the real runtime spectral codec."""
    x = mx.array(keys[None, None].astype(np.float32))  # (1, 1, N, D)
    rot = SpectralRotation(cal["eigenvectors"])
    ps, pt, norms = spectral_quantize(
        x,
        rot,
        cal["codebook_sem"],
        cal["codebook_tail"],
        cal["d_eff"],
        cal["bits_high"],
        cal["bits_low"],
    )
    out = spectral_dequantize(
        ps,
        pt,
        norms,
        rot,
        cal["codebook_sem"],
        cal["codebook_tail"],
        cal["d_eff"],
        cal["bits_high"],
        cal["bits_low"],
    )
    return np.array(out[0, 0])


def _qk_mse(keys: np.ndarray, recon: np.ndarray, queries: np.ndarray) -> float:
    err = keys - recon  # (N, D)
    return float(np.mean((queries @ err.T) ** 2))


def _synthetic(D=16, n=6000, seed=0):
    """Keys with high variance where queries DON'T look, and vice versa."""
    rng = np.random.default_rng(seed)
    std = np.ones(D)
    std[:4] = 4.0  # dominant key directions — queries ignore them
    std[-4:] = 0.6  # low-variance directions — queries focus here
    keys = rng.normal(size=(n, D)) * std
    q_std = np.full(D, 0.05)
    q_std[-4:] = 1.0
    queries = rng.normal(size=(2000, D)) * q_std
    c_q = (queries.T @ queries) / len(queries)
    return keys, queries, c_q


# ---------------------------------------------------------------------------
# calibrate_head_qa
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("avg_bits", [2, 4])
def test_qa_beats_key_pca_when_queries_look_at_low_variance_dims(avg_bits):
    keys, queries, c_q = _synthetic()
    train, test = keys[:4000], keys[4000:]

    base = sc.calibrate_head(mx.array(train.astype(np.float32)), avg_bits=avg_bits)
    qa = sc.calibrate_head_qa(
        mx.array(train.astype(np.float32)), c_q, avg_bits=avg_bits
    )

    base_err = _qk_mse(test, _runtime_roundtrip(test, base), queries)
    qa_err = _qk_mse(test, _runtime_roundtrip(test, qa), queries)
    assert qa_err < 0.6 * base_err, (qa_err, base_err, qa["objective_basis"])


def test_qa_not_worse_than_key_pca_with_isotropic_queries():
    rng = np.random.default_rng(1)
    D = 16
    std = np.linspace(3.0, 0.2, D)
    keys = rng.normal(size=(6000, D)) * std
    queries = rng.normal(size=(2000, D))
    c_q = (queries.T @ queries) / len(queries)
    train, test = keys[:4000], keys[4000:]

    base = sc.calibrate_head(mx.array(train.astype(np.float32)), avg_bits=2)
    qa = sc.calibrate_head_qa(mx.array(train.astype(np.float32)), c_q, avg_bits=2)

    base_err = _qk_mse(test, _runtime_roundtrip(test, base), queries)
    qa_err = _qk_mse(test, _runtime_roundtrip(test, qa), queries)
    assert qa_err <= 1.1 * base_err


def test_qa_result_is_runtime_compatible():
    keys, _queries, c_q = _synthetic(D=16)
    qa = sc.calibrate_head_qa(mx.array(keys.astype(np.float32)), c_q, avg_bits=2)
    D = keys.shape[1]

    V = np.array(qa["eigenvectors"])
    assert V.shape == (D, D)
    # Runtime unrotates with V (not V^-1), so the basis must be orthonormal.
    np.testing.assert_allclose(V @ V.T, np.eye(D), atol=1e-4)
    assert 1 <= qa["d_eff"] <= D
    assert qa["bits_high"] in (1, 2, 4, 8)
    assert qa["bits_low"] in (1, 2, 4, 8)
    assert qa["bits_high"] >= qa["bits_low"]
    assert qa["codebook_sem"].shape == (1 << qa["bits_high"],)
    assert isinstance(qa["objective_basis"], str)


@pytest.mark.parametrize("avg_bits", [2, 4])
def test_allocate_bits_weighted_respects_budget(avg_bits):
    rng = np.random.default_rng(2)
    D = 24
    imp = np.sort(rng.exponential(size=D))[::-1]
    d, bh, bl = sc.allocate_bits_weighted(imp, avg_bits)
    assert 1 <= d <= D
    assert bh >= bl
    assert d * bh + (D - d) * bl <= D * avg_bits


def test_allocate_bits_weighted_concentrates_bits_on_important_dims():
    imp = np.array([100.0] * 4 + [1e-4] * 12)
    d, bh, bl = sc.allocate_bits_weighted(imp, 2)
    # Spending more bits on the 4 important coordinates beats uniform 2-bit.
    assert bh > 2
    assert d <= 6


def test_allocate_bits_weighted_uniform_importance_is_uniform():
    d, bh, bl = sc.allocate_bits_weighted(np.ones(16), 4)
    assert bh == bl == 4


# ---------------------------------------------------------------------------
# SDPA query capture
# ---------------------------------------------------------------------------


def test_capture_records_queries_and_restores_sdpa():
    from mlx_lm.models.base import scaled_dot_product_attention

    orig = mx.fast.scaled_dot_product_attention
    q = mx.random.normal((1, 2, 3, 8))
    k = mx.random.normal((1, 1, 3, 8))
    with sc._capture_sdpa_queries() as records:
        scaled_dot_product_attention(q, k, k, cache=None, scale=1.0, mask=None)
    assert mx.fast.scaled_dot_product_attention is orig
    assert len(records) == 1
    rq, rk = records[0]
    assert rq.shape == q.shape and rk.shape == k.shape


def test_capture_restores_sdpa_on_exception():
    orig = mx.fast.scaled_dot_product_attention
    with pytest.raises(RuntimeError):
        with sc._capture_sdpa_queries():
            raise RuntimeError("boom")
    assert mx.fast.scaled_dot_product_attention is orig


def test_capture_ignores_other_threads():
    errors: list[BaseException] = []

    def _other():
        try:
            # Built on this thread: mlx streams are thread-local (#499).
            q = mx.ones((1, 1, 2, 8))
            mx.eval(mx.fast.scaled_dot_product_attention(q, q, q, scale=1.0))
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    with sc._capture_sdpa_queries() as records:
        t = threading.Thread(target=_other)
        t.start()
        t.join()
    assert errors == []
    assert records == []


def test_capture_nested_is_safe():
    orig = mx.fast.scaled_dot_product_attention
    q = mx.random.normal((1, 1, 2, 8))
    with sc._capture_sdpa_queries() as outer:
        with sc._capture_sdpa_queries() as inner:
            mx.fast.scaled_dot_product_attention(q, q, q, scale=1.0)
        mx.fast.scaled_dot_product_attention(q, q, q, scale=1.0)
    assert mx.fast.scaled_dot_product_attention is orig
    assert len(inner) == 1
    assert len(outer) == 1


# ---------------------------------------------------------------------------
# collect_kv_vectors(query_stats=...)
# ---------------------------------------------------------------------------


class _Backbone:
    def __init__(self, num_layers, n_kv):
        self.layers = [object() for _ in range(num_layers)]
        args = MagicMock()
        args.num_key_value_heads = n_kv
        args.num_attention_heads = n_kv
        self.args = args


class _SdpaModel:
    """Fake model whose layers run SDPA over their own cache's keys."""

    def __init__(self, backbone, head_dim, n_kv=2, n_q=4, distinct=True, sdpa=True):
        self._backbone = backbone
        self.head_dim = head_dim
        self.n_kv = n_kv
        self.n_q = n_q
        self.distinct = distinct
        self.sdpa = sdpa
        self.queries: dict[int, list[np.ndarray]] = {}
        self.calls = 0

    def __call__(self, input_ids, cache=None):
        self.calls += 1
        seq = input_ids.shape[1]
        for i, entry in enumerate(cache):
            if self.distinct:
                keys = mx.random.normal((1, self.n_kv, seq, self.head_dim))
            else:
                keys = mx.ones((1, self.n_kv, seq, self.head_dim))
            values = mx.random.normal((1, self.n_kv, seq, self.head_dim))
            k, v = entry.update_and_fetch(keys, values)
            if self.sdpa:
                q = mx.random.normal((1, self.n_q, seq, self.head_dim))
                mx.eval(q)
                self.queries.setdefault(i, []).append(
                    np.array(q).reshape(-1, self.head_dim)
                )
                mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0)
        return mx.zeros((1, seq, 4))


def _collect(model, num_layers, texts, query_stats):
    with (
        patch(
            "olmlx.engine.flash.prepare._encode_tokens",
            side_effect=lambda tok, text: list(range(len(text.split()))),
        ),
        patch(
            "mlx_lm.models.cache.make_prompt_cache",
            side_effect=lambda _o: [KVCache() for _ in range(num_layers)],
        ),
    ):
        return sc.collect_kv_vectors(
            model,
            MagicMock(),
            model._backbone,
            num_layers=num_layers,
            n_kv_heads=model.n_kv,
            head_dim=model.head_dim,
            texts=texts,
            max_tokens_per_head=10_000,
            query_stats=query_stats,
        )


@pytest.mark.parametrize("distinct", [True, False])
def test_collect_kv_vectors_accumulates_query_second_moment(distinct):
    num_layers, D = 3, 8
    model = _SdpaModel(_Backbone(num_layers, 2), D, distinct=distinct)
    stats: dict = {}
    _collect(model, num_layers, ["a b c d e f"] * 3, stats)

    assert sorted(stats) == list(range(num_layers))
    for layer in range(num_layers):
        q = np.concatenate(model.queries[layer])
        np.testing.assert_allclose(stats[layer]["sum"], q.T @ q, rtol=1e-4, atol=1e-3)
        assert stats[layer]["count"] == q.shape[0]


def test_collect_kv_vectors_without_query_stats_does_not_patch():
    num_layers, D = 2, 8
    model = _SdpaModel(_Backbone(num_layers, 2), D)
    with patch.object(sc, "_capture_sdpa_queries") as cap:
        _collect(model, num_layers, ["a b c"], None)
    cap.assert_not_called()


# ---------------------------------------------------------------------------
# calibrate_model(objective="attention")
# ---------------------------------------------------------------------------


def _patch_calibrate(model, head_dim, num_layers, texts):
    return [
        patch(
            "olmlx.engine.flash.prepare.load_model_with_strict_fallback",
            return_value=(model, MagicMock()),
        ),
        patch(
            "olmlx.engine.flash.prepare._get_backbone",
            return_value=model._backbone,
        ),
        patch(
            "olmlx.engine.flash.prepare._get_c4_calibration_data",
            return_value=texts,
        ),
        patch(
            "olmlx.engine.flash.prepare._encode_tokens",
            side_effect=lambda tok, text: list(range(len(text.split()))),
        ),
        patch(
            "olmlx.engine.turboquant_cache._detect_head_dim",
            return_value=head_dim,
        ),
        patch(
            "mlx_lm.models.cache.make_prompt_cache",
            side_effect=lambda _o: [KVCache() for _ in range(num_layers)],
        ),
    ]


def test_calibrate_model_attention_objective_end_to_end(tmp_path):
    num_layers, D = 2, 8
    model = _SdpaModel(_Backbone(num_layers, 2), D)
    texts = [" ".join(["w"] * 40)] * 4
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    patches = _patch_calibrate(model, D, num_layers, texts)
    for p in patches:
        p.start()
    try:
        out = sc.calibrate_model(
            str(model_dir), num_samples=4, avg_bits=2, objective="attention"
        )
    finally:
        for p in patches:
            p.stop()

    assert out == model_dir / "spectral_qa"
    cfg = json.loads((out / "spectral_config.json").read_text())
    assert cfg["meta"]["objective"] == "attention"
    assert cfg["meta"]["avg_bits"] == 2
    loaded = sc.load_calibration(out)
    assert len(loaded) == num_layers * 2


def test_calibrate_model_default_objective_is_reconstruction(tmp_path):
    num_layers, D = 2, 8
    model = _SdpaModel(_Backbone(num_layers, 2), D)
    texts = [" ".join(["w"] * 20)] * 2
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    patches = _patch_calibrate(model, D, num_layers, texts)
    for p in patches:
        p.start()
    try:
        out = sc.calibrate_model(str(model_dir), num_samples=2)
    finally:
        for p in patches:
            p.stop()
    assert out == model_dir / "spectral"
    cfg = json.loads((out / "spectral_config.json").read_text())
    assert cfg["meta"]["objective"] == "reconstruction"


def test_calibrate_model_attention_raises_when_no_queries_captured(tmp_path):
    num_layers, D = 2, 8
    model = _SdpaModel(_Backbone(num_layers, 2), D, sdpa=False)
    texts = [" ".join(["w"] * 20)] * 2
    patches = _patch_calibrate(model, D, num_layers, texts)
    for p in patches:
        p.start()
    try:
        with pytest.raises(RuntimeError, match="scaled_dot_product_attention"):
            sc.calibrate_model(str(tmp_path), num_samples=2, objective="attention")
    finally:
        for p in patches:
            p.stop()


def test_calibrate_model_rejects_unknown_objective(tmp_path):
    with pytest.raises(ValueError, match="objective"):
        sc.calibrate_model(str(tmp_path), objective="bogus")


# ---------------------------------------------------------------------------
# Config / routing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec", ["spectral-qa:2", "spectral-qa:4"])
def test_config_accepts_spectral_qa(spec):
    from olmlx.config import validate_kv_cache_quant_format

    assert validate_kv_cache_quant_format(spec) == spec


@pytest.mark.parametrize("spec", ["spectral-qa:8", "spectral-qa", "spectral-qa:"])
def test_config_rejects_bad_spectral_qa(spec):
    from olmlx.config import validate_kv_cache_quant_format

    with pytest.raises(ValueError):
        validate_kv_cache_quant_format(spec)


def test_auto_calibrate_allows_spectral_qa(monkeypatch):
    from olmlx.config import Settings

    s = Settings(kv_cache_quant="spectral-qa:2", kv_cache_auto_calibrate=True)
    assert s.kv_cache_quant == "spectral-qa:2"


def test_parse_kv_cache_quant_maps_spectral_qa_to_spectral_codec():
    from olmlx.engine.kv_budget import _parse_kv_cache_quant_kv

    assert _parse_kv_cache_quant_kv("spectral-qa:2") == ("spectral", 2, 2)


def _manager_with_store(tmp_path):
    from olmlx.engine.model_manager import ModelManager

    manager = ModelManager(MagicMock(), MagicMock())
    store = MagicMock()
    store.model_dir.return_value = store.local_path.return_value = tmp_path
    manager.store = store
    return manager


def test_find_spectral_dir_spectral_qa_uses_its_own_dir(tmp_path):
    for name, objective in (
        ("spectral", "reconstruction"),
        ("spectral_qa", "attention"),
    ):
        d = tmp_path / name
        d.mkdir()
        (d / "spectral_config.json").write_text(
            json.dumps({"meta": {"avg_bits": 2, "objective": objective}})
        )
    manager = _manager_with_store(tmp_path)
    assert manager._find_spectral_dir("m", "spectral-qa:2") == tmp_path / "spectral_qa"
    assert manager._find_spectral_dir("m", "spectral:2") == tmp_path / "spectral"


def test_find_spectral_dir_spectral_qa_missing_suggests_attention_objective(
    tmp_path, monkeypatch
):
    from olmlx.engine.model_manager import SpectralCalibrationMissingError

    monkeypatch.setattr(
        "olmlx.engine.model_manager.settings.kv_cache_auto_calibrate", False
    )
    # A plain spectral calibration must not satisfy spectral-qa.
    d = tmp_path / "spectral"
    d.mkdir()
    (d / "spectral_config.json").write_text(json.dumps({"meta": {"avg_bits": 2}}))
    manager = _manager_with_store(tmp_path)
    with pytest.raises(SpectralCalibrationMissingError) as exc_info:
        manager._find_spectral_dir("m", "spectral-qa:2")
    msg = str(exc_info.value)
    assert "--objective attention" in msg
    assert "--avg-bits 2" in msg
    assert str(tmp_path / "spectral_qa") in msg


def test_find_spectral_dir_spectral_qa_rejects_wrong_objective(tmp_path):
    from olmlx.engine.model_manager import SpectralCalibrationMissingError

    d = tmp_path / "spectral_qa"
    d.mkdir()
    (d / "spectral_config.json").write_text(
        json.dumps({"meta": {"avg_bits": 4, "objective": "reconstruction"}})
    )
    manager = _manager_with_store(tmp_path)
    with pytest.raises(SpectralCalibrationMissingError, match="--objective attention"):
        manager._find_spectral_dir("m", "spectral-qa:4")


def test_auto_calibrate_spectral_qa_passes_attention_objective(tmp_path, monkeypatch):
    from olmlx.config import settings as _settings

    monkeypatch.setattr(_settings, "kv_cache_quant", "spectral-qa:2")
    monkeypatch.setattr(_settings, "kv_cache_auto_calibrate", True)
    manager = _manager_with_store(tmp_path)
    out = tmp_path / "spectral_qa"

    def fake_calibrate(**kwargs):
        assert kwargs["objective"] == "attention"
        assert kwargs["avg_bits"] == 2
        out.mkdir()
        (out / "spectral_config.json").write_text("{}")
        return out

    with patch(
        "olmlx.engine.spectralquant_calibrate.calibrate_model",
        side_effect=fake_calibrate,
    ):
        assert manager._find_spectral_dir("m", "spectral-qa:2") == out


def test_cli_spectral_prepare_objective_flag():
    from olmlx.cli.parser import build_parser

    parser = build_parser()
    args = parser.parse_args(["spectral", "prepare", "m", "--objective", "attention"])
    assert args.objective == "attention"
    args = parser.parse_args(["spectral", "prepare", "m"])
    assert args.objective == "reconstruction"


# ---------------------------------------------------------------------------
# Regressions found on Qwen3-0.6B (4-bit spectral-qa was worse than spectral)
# ---------------------------------------------------------------------------


def _heavy_tailed(seed=0, D=64, N=12000):
    rng = np.random.default_rng(seed)
    keys = rng.standard_t(3, size=(N, D)) * np.linspace(1.5, 0.3, D)
    keys[:, 0] += 8.0  # outlier channel, like real post-RoPE keys
    q = rng.standard_t(4, size=(4000, D)) * np.linspace(0.5, 1.5, D)
    return keys, (q.T @ q) / len(q)


def _in_sample_qk(keys, c_q, cal):
    err = keys - _runtime_roundtrip(keys, cal)
    return float(np.mean(np.einsum("nd,de,ne->n", err, c_q, err)))


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_qa_never_worse_than_spectral_on_calibration_data(seed):
    """Selection must compare candidates with converged codebooks, and the
    baseline must be exactly ``calibrate_head`` — a handicapped baseline let
    a worse basis win on 19/28 Qwen3-0.6B layers at 4 bits."""
    keys, c_q = _heavy_tailed(seed)
    kv = mx.array(keys.astype(np.float32))
    base = sc.calibrate_head(kv, avg_bits=4)
    qa = sc.calibrate_head_qa(kv, c_q, avg_bits=4)
    assert _in_sample_qk(keys, c_q, qa) <= 1.02 * _in_sample_qk(keys, c_q, base)


def test_qa_returns_calibrate_head_verbatim_when_baseline_wins():
    rng = np.random.default_rng(5)
    D = 16
    keys = rng.normal(size=(4000, D)) * np.linspace(3.0, 0.2, D)
    kv = mx.array(keys.astype(np.float32))
    base = sc.calibrate_head(kv, avg_bits=4)
    # Isotropic queries: q·k error == key reconstruction error, so nothing
    # can clear the switching margin over the baseline.
    qa = sc.calibrate_head_qa(kv, np.eye(D), avg_bits=4)
    assert qa["objective_basis"] == "key_pca"
    np.testing.assert_array_equal(
        np.array(qa["eigenvectors"]), np.array(base["eigenvectors"])
    )
    np.testing.assert_array_equal(
        np.array(qa["codebook_sem"]), np.array(base["codebook_sem"])
    )
    assert (qa["d_eff"], qa["bits_high"], qa["bits_low"]) == (
        base["d_eff"],
        base["bits_high"],
        base["bits_low"],
    )


# ---------------------------------------------------------------------------
# fit_codebook: vectorized Lloyd-Max must match the reference algorithm
# ---------------------------------------------------------------------------


def _reference_lloyd(data_np, bits, max_iter=100, tol=1e-6):
    """The original O(N*K) per-iteration implementation."""
    n_levels = 1 << bits
    lo, hi = float(data_np.min()), float(data_np.max())
    centroids = np.linspace(lo, hi, n_levels).astype(np.float32)
    for _ in range(max_iter):
        assignments = np.abs(data_np[:, None] - centroids[None, :]).argmin(axis=1)
        new = centroids.copy()
        for k in range(n_levels):
            mask = assignments == k
            if mask.any():
                new[k] = data_np[mask].mean()
        change = np.max(np.abs(new - centroids))
        centroids = new
        if change < tol:
            break
    centroids.sort()
    return centroids


@pytest.mark.parametrize("bits", [1, 2, 4, 8])
@pytest.mark.parametrize("dist", ["normal", "t", "skewed"])
def test_fit_codebook_matches_reference_lloyd(bits, dist):
    from olmlx.engine.spectralquant import fit_codebook

    rng = np.random.default_rng(bits)
    if dist == "normal":
        data = rng.normal(size=20000) * 0.3
    elif dist == "t":
        data = rng.standard_t(3, size=20000) * 0.2
    else:
        data = rng.exponential(size=20000) - 0.4
    data = data.astype(np.float32)
    got = np.array(fit_codebook(mx.array(data), bits=bits))
    want = _reference_lloyd(data, bits)
    np.testing.assert_allclose(got, want, atol=2e-4)


def test_fit_codebook_constant_data():
    from olmlx.engine.spectralquant import fit_codebook

    cb = np.array(fit_codebook(mx.full((100,), 0.25), bits=2))
    np.testing.assert_allclose(cb, np.full(4, 0.25))


def test_fit_codebook_is_fast_at_8_bits():
    import time

    from olmlx.engine.spectralquant import fit_codebook

    data = mx.array(np.random.default_rng(0).standard_t(3, size=500_000) * 0.2)
    t = time.perf_counter()
    fit_codebook(data, bits=8)
    # The O(N*K) reference takes minutes here.
    assert time.perf_counter() - t < 10.0
