"""Regression coverage for olmlx.engine.spectralquant_calibrate.

Exercises the calibration math (covariance, eigendecomposition, effective
dimensionality, per-head calibration), the save/load round-trip of calibration
artifacts, and the full `calibrate_model` pipeline with a fully mocked model,
tokenizer, and helper-function set.  All hermetic: no network, no real model,
no GPU-only paths, no filesystem outside tmp_path.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models.cache import KVCache

import olmlx.engine.spectralquant_calibrate as sc


# ---------------------------------------------------------------------------
# Pure math: compute_covariance
# ---------------------------------------------------------------------------


def test_compute_covariance_matches_numpy_population():
    rng = np.random.default_rng(0)
    data_np = rng.standard_normal((200, 5)).astype(np.float32)
    cov = sc.compute_covariance(mx.array(data_np))
    # Module uses /n (population covariance, not /(n-1)).
    expected = np.cov(data_np, rowvar=False, bias=True)
    assert cov.shape == (5, 5)
    np.testing.assert_allclose(np.array(cov), expected, rtol=1e-3, atol=1e-4)


def test_compute_covariance_is_symmetric_and_float32():
    rng = np.random.default_rng(1)
    data = mx.array(rng.standard_normal((50, 4)))
    cov = sc.compute_covariance(data)
    assert cov.dtype == mx.float32
    np.testing.assert_allclose(np.array(cov), np.array(cov).T, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Pure math: eigendecompose
# ---------------------------------------------------------------------------


def test_eigendecompose_descending_and_clamped():
    # Diagonal covariance -> eigenvalues are the diagonal, sorted descending.
    cov = mx.array(np.diag([2.0, 9.0, 0.5, 4.0]).astype(np.float32))
    evals, evecs = sc.eigendecompose(cov)
    ev = np.array(evals)
    assert evecs.shape == (4, 4)
    # Descending order.
    assert np.all(np.diff(ev) <= 1e-5)
    np.testing.assert_allclose(
        sorted(ev, reverse=True), [9.0, 4.0, 2.0, 0.5], atol=1e-4
    )


def test_eigendecompose_clamps_negative_eigenvalues():
    # A matrix with a genuinely negative eigenvalue; clamp to >= 0.
    cov = mx.array(np.array([[1.0, 0.0], [0.0, -3.0]], dtype=np.float32))
    evals, _ = sc.eigendecompose(cov)
    ev = np.array(evals)
    assert np.all(ev >= 0.0)
    # The negative one is clamped to 0, positive preserved.
    np.testing.assert_allclose(sorted(ev), [0.0, 1.0], atol=1e-5)


# ---------------------------------------------------------------------------
# Pure math: compute_d_eff
# ---------------------------------------------------------------------------


def test_compute_d_eff_single_dominant_eigenvalue():
    # One huge eigenvalue -> participation ratio ~1.
    evals = mx.array([100.0, 1e-6, 1e-6, 1e-6])
    assert sc.compute_d_eff(evals) == 1


def test_compute_d_eff_uniform_spectrum_equals_dim():
    # All equal -> d_eff == dimension.
    evals = mx.array([3.0, 3.0, 3.0, 3.0, 3.0])
    assert sc.compute_d_eff(evals) == 5


def test_compute_d_eff_all_zero_returns_dim():
    # total_sq < 1e-12 branch returns len(eigenvalues).
    evals = mx.array([0.0, 0.0, 0.0])
    assert sc.compute_d_eff(evals) == 3


def test_compute_d_eff_clamped_to_at_least_one():
    evals = mx.array([1e6, 0.0])
    result = sc.compute_d_eff(evals)
    assert result >= 1


# ---------------------------------------------------------------------------
# calibrate_head
# ---------------------------------------------------------------------------


def test_calibrate_head_structure_and_shapes():
    rng = np.random.default_rng(42)
    head_dim = 8
    # Anisotropic data so d_eff < head_dim is plausible.
    base = rng.standard_normal((300, head_dim)).astype(np.float32)
    base[:, 4:] *= 0.01  # squash tail dims
    result = sc.calibrate_head(mx.array(base), avg_bits=4)

    assert set(result.keys()) == {
        "eigenvectors",
        "d_eff",
        "codebook_sem",
        "codebook_tail",
        "bits_high",
        "bits_low",
    }
    assert result["eigenvectors"].shape == (head_dim, head_dim)
    assert 1 <= result["d_eff"] <= head_dim
    assert result["bits_high"] >= result["bits_low"] >= 1
    # Semantic codebook has 2**bits_high centroids.
    assert result["codebook_sem"].shape[0] == (1 << result["bits_high"])


def test_calibrate_head_full_rank_tail_codebook_is_sentinel():
    # When d_eff == head_dim, codebook_tail is the [0.0] sentinel.
    rng = np.random.default_rng(7)
    head_dim = 4
    # Isotropic data -> d_eff == head_dim.
    data = mx.array(rng.standard_normal((500, head_dim)).astype(np.float32))
    result = sc.calibrate_head(data, avg_bits=4)
    if result["d_eff"] == head_dim:
        np.testing.assert_array_equal(np.array(result["codebook_tail"]), [0.0])
    else:
        # Anisotropy crept in; tail codebook must then be a real codebook.
        assert result["codebook_tail"].shape[0] == (1 << result["bits_low"])


def test_calibrate_head_respects_avg_bits_budget():
    rng = np.random.default_rng(3)
    head_dim = 8
    data = mx.array(rng.standard_normal((200, head_dim)).astype(np.float32))
    result = sc.calibrate_head(data, avg_bits=2)
    d_eff = result["d_eff"]
    total = d_eff * result["bits_high"] + (head_dim - d_eff) * result["bits_low"]
    # allocate_bits minimizes |total - head_dim*avg_bits|.
    assert abs(total - head_dim * 2) <= head_dim  # within a reasonable slack


# ---------------------------------------------------------------------------
# save_calibration / load_calibration round-trip
# ---------------------------------------------------------------------------


def _make_calibration_entry(head_dim=4):
    eigvecs = mx.array(np.eye(head_dim, dtype=np.float32))
    return {
        "eigenvectors": eigvecs,
        "d_eff": 2,
        "codebook_sem": mx.array(np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)),
        "codebook_tail": mx.array(np.array([-0.5, 0.5], dtype=np.float32)),
        "bits_high": 5,
        "bits_low": 3,
    }


def test_save_calibration_writes_expected_files(tmp_path):
    calib = {(0, 0, "key"): _make_calibration_entry()}
    sc.save_calibration(calib, tmp_path)
    assert (tmp_path / "spectral_config.json").exists()
    assert (tmp_path / "calibration.safetensors").exists()


def test_save_load_round_trip_preserves_values(tmp_path):
    calib = {
        (0, 0, "key"): _make_calibration_entry(),
        (0, 0, "value"): _make_calibration_entry(),
        (3, 0, "key"): _make_calibration_entry(),
    }
    sc.save_calibration(calib, tmp_path)
    loaded = sc.load_calibration(tmp_path)

    assert set(loaded.keys()) == set(calib.keys())
    for key, orig in calib.items():
        got = loaded[key]
        assert got["d_eff"] == orig["d_eff"]
        assert got["bits_high"] == orig["bits_high"]
        assert got["bits_low"] == orig["bits_low"]
        np.testing.assert_allclose(
            np.array(got["eigenvectors"]), np.array(orig["eigenvectors"])
        )
        np.testing.assert_allclose(
            np.array(got["codebook_sem"]), np.array(orig["codebook_sem"])
        )
        np.testing.assert_allclose(
            np.array(got["codebook_tail"]), np.array(orig["codebook_tail"])
        )


def test_save_calibration_creates_nested_output_dir(tmp_path):
    nested = tmp_path / "a" / "b" / "spectral"
    sc.save_calibration({(0, 0, "key"): _make_calibration_entry()}, nested)
    assert (nested / "spectral_config.json").exists()


def test_load_calibration_parses_layer_head_kind_from_prefix(tmp_path):
    calib = {(7, 0, "value"): _make_calibration_entry()}
    sc.save_calibration(calib, tmp_path)
    loaded = sc.load_calibration(tmp_path)
    assert (7, 0, "value") in loaded


# ---------------------------------------------------------------------------
# Helper: _is_attention_cache
# ---------------------------------------------------------------------------


def _kv_cache_with(keys_shape):
    cache = KVCache()
    keys = mx.zeros(keys_shape)
    values = mx.zeros(keys_shape)
    cache.update_and_fetch(keys, values)
    return cache


def test_is_attention_cache_accepts_valid_kvcache():
    cache = _kv_cache_with((1, 2, 4, 8))
    assert sc._is_attention_cache(cache, expected_head_dim=8) is True


def test_is_attention_cache_rejects_head_dim_mismatch():
    cache = _kv_cache_with((1, 2, 4, 8))
    assert sc._is_attention_cache(cache, expected_head_dim=16) is False


def test_is_attention_cache_rejects_too_few_tokens():
    cache = _kv_cache_with((1, 2, 1, 8))
    assert sc._is_attention_cache(cache, expected_head_dim=8) is False


def test_is_attention_cache_rejects_non_4d_keys():
    # keys present but ndim != 4 (e.g. a 2D state) -> rejected.
    class _Bad2DCache(KVCache):
        @property
        def state(self):
            return [mx.zeros((4, 8)), mx.zeros((4, 8))]

    assert sc._is_attention_cache(_Bad2DCache(), expected_head_dim=8) is False


def test_is_attention_cache_rejects_non_kvcache():
    not_a_cache = MagicMock()
    assert sc._is_attention_cache(not_a_cache, expected_head_dim=8) is False


def test_is_attention_cache_rejects_empty_state():
    # A KVCache subclass whose .state is empty exercises the `not state` guard
    # without depending on mlx's unpopulated-cache `.state` behavior.
    class _EmptyStateCache(KVCache):
        @property
        def state(self):
            return []

    assert sc._is_attention_cache(_EmptyStateCache(), expected_head_dim=8) is False


def test_is_attention_cache_rejects_one_element_state():
    # len(state) < 2 guard (keys present but no values).
    class _OneElemCache(KVCache):
        @property
        def state(self):
            return [mx.zeros((1, 2, 4, 8))]

    assert sc._is_attention_cache(_OneElemCache(), expected_head_dim=8) is False


# ---------------------------------------------------------------------------
# Helpers: config holder resolution / cache owner
# ---------------------------------------------------------------------------


def test_resolve_config_holder_prefers_args_on_inner():
    inner = MagicMock()
    inner.args = MagicMock()
    inner.config = MagicMock()
    model = MagicMock()
    assert sc._resolve_config_holder(inner, model) is inner


def test_resolve_config_holder_falls_back_to_model_args():
    inner = MagicMock()
    inner.args = None
    model = MagicMock()
    model.args = MagicMock()
    assert sc._resolve_config_holder(inner, model) is model


def test_resolve_config_holder_uses_config_when_no_args():
    inner = MagicMock()
    inner.args = None
    inner.config = MagicMock()
    model = MagicMock()
    model.args = None
    assert sc._resolve_config_holder(inner, model) is inner


def test_resolve_config_holder_raises_when_nothing():
    inner = MagicMock()
    inner.args = None
    inner.config = None
    model = MagicMock()
    model.args = None
    model.config = None
    with pytest.raises(RuntimeError, match="Cannot detect model configuration"):
        sc._resolve_config_holder(inner, model)


def test_config_namespace_prefers_args():
    holder = MagicMock()
    holder.args = "ARGS"
    assert sc._config_namespace(holder) == "ARGS"


def test_config_namespace_falls_back_to_config():
    holder = MagicMock()
    holder.args = None
    holder.config = "CFG"
    assert sc._config_namespace(holder) == "CFG"


def test_config_namespace_raises_when_neither():
    class Empty:
        args = None
        config = None

    with pytest.raises(RuntimeError, match="neither"):
        sc._config_namespace(Empty())


def test_resolve_cache_owner_prefers_model_make_cache():
    inner = MagicMock(spec=[])  # no make_cache
    model = MagicMock()
    model.make_cache = MagicMock()
    assert sc._resolve_cache_owner(inner, model) is model


def test_resolve_cache_owner_falls_back_to_inner():
    inner = MagicMock()
    model = MagicMock(spec=[])  # no make_cache attribute
    assert sc._resolve_cache_owner(inner, model) is inner


# ---------------------------------------------------------------------------
# Error builder
# ---------------------------------------------------------------------------


def test_build_empty_collection_error_with_cause():
    cause = ValueError("forward boom")
    err = sc._build_empty_collection_error(cause)
    assert isinstance(err, RuntimeError)
    assert err.__cause__ is cause
    assert err.__suppress_context__ is True


def test_build_empty_collection_error_without_cause():
    err = sc._build_empty_collection_error(None)
    assert isinstance(err, RuntimeError)
    assert err.__cause__ is None
    assert "No attention-layer cache entries" in str(err)


# ---------------------------------------------------------------------------
# Full pipeline: calibrate_model (mocked model/tokenizer/helpers)
# ---------------------------------------------------------------------------


class _FakeBackbone:
    """Backbone exposing .layers and .args with config fields."""

    def __init__(self, num_layers, n_kv_heads, head_dim):
        self.layers = [object() for _ in range(num_layers)]
        args = MagicMock()
        args.num_key_value_heads = n_kv_heads
        args.num_attention_heads = n_kv_heads
        self.args = args


class _FakeModel:
    """Callable model that populates the passed prompt_cache with KV state."""

    def __init__(self, backbone, head_dim, n_kv=None):
        self._backbone = backbone
        self._head_dim = head_dim
        self._n_kv = n_kv if n_kv is not None else backbone.args.num_key_value_heads
        self.calls = 0

    def __call__(self, input_ids, cache=None):
        self.calls += 1
        seq = input_ids.shape[1]
        n_kv = self._n_kv
        for entry in cache:
            keys = mx.ones((1, n_kv, seq, self._head_dim))
            values = mx.ones((1, n_kv, seq, self._head_dim)) * 2.0
            entry.update_and_fetch(keys, values)


def _patch_helpers(model, tokenizer, head_dim, texts, cache_factory):
    """Patch the lazily-imported helpers used by calibrate_model."""
    return [
        patch(
            "olmlx.engine.flash.prepare.load_model_with_strict_fallback",
            return_value=(model, tokenizer),
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
            "olmlx.engine.flash.prepare._get_calibration_data",
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
            side_effect=cache_factory,
        ),
    ]


def test_calibrate_model_end_to_end(tmp_path):
    head_dim = 8
    num_layers = 2
    n_kv_heads = 2
    backbone = _FakeBackbone(num_layers, n_kv_heads, head_dim)
    model = _FakeModel(backbone, head_dim)
    tokenizer = MagicMock()
    # Each text has >= 2 "tokens" after split.
    texts = ["one two three four five six seven eight nine ten"] * 4

    def cache_factory(_owner):
        return [KVCache() for _ in range(num_layers)]

    progress = []

    patches = _patch_helpers(model, tokenizer, head_dim, texts, cache_factory)
    for p in patches:
        p.start()
    try:
        out = sc.calibrate_model(
            "fake/model",
            output_dir=tmp_path / "spectral",
            num_samples=4,
            progress_callback=lambda desc, frac: progress.append((desc, frac)),
        )
    finally:
        for p in patches:
            p.stop()

    assert out == tmp_path / "spectral"
    # Forward run once per sample.
    assert model.calls == len(texts)

    # Artifacts written and loadable.
    loaded = sc.load_calibration(out)
    # One key + one value calibration per layer (heads aggregated under head 0).
    assert (0, 0, "key") in loaded
    assert (0, 0, "value") in loaded
    assert (1, 0, "key") in loaded
    assert len(loaded) == num_layers * 2

    # Metadata merged into spectral_config.json.
    import json

    cfg = json.loads((out / "spectral_config.json").read_text())
    assert cfg["meta"]["num_layers"] == num_layers
    assert cfg["meta"]["n_kv_heads"] == n_kv_heads
    assert cfg["meta"]["head_dim"] == head_dim
    assert cfg["meta"]["calibration_dataset"] == "c4"
    # Progress callback reached completion.
    assert progress[-1] == ("Done", 1.0)


def test_calibrate_model_raises_when_nothing_collected(tmp_path):
    head_dim = 8
    num_layers = 1
    n_kv_heads = 1
    backbone = _FakeBackbone(num_layers, n_kv_heads, head_dim)
    model = _FakeModel(backbone, head_dim)
    tokenizer = MagicMock()
    # All samples too short (< 2 tokens) -> skipped, nothing collected.
    texts = ["single", "x", ""]

    def cache_factory(_owner):
        return [KVCache() for _ in range(num_layers)]

    patches = _patch_helpers(model, tokenizer, head_dim, texts, cache_factory)
    for p in patches:
        p.start()
    try:
        with pytest.raises(RuntimeError, match="No KV vectors were collected"):
            sc.calibrate_model(
                "fake/model",
                output_dir=tmp_path / "spectral",
                num_samples=3,
            )
    finally:
        for p in patches:
            p.stop()
    # Model never successfully ran a forward with >= 2 tokens.
    assert model.calls == 0


def test_calibrate_model_forward_error_chained_as_cause(tmp_path):
    head_dim = 8
    num_layers = 1
    n_kv_heads = 1
    backbone = _FakeBackbone(num_layers, n_kv_heads, head_dim)

    class _ExplodingModel(_FakeModel):
        def __call__(self, input_ids, cache=None):
            self.calls += 1
            raise ValueError("forward exploded")

    model = _ExplodingModel(backbone, head_dim)
    tokenizer = MagicMock()
    texts = ["one two three four"] * 2

    def cache_factory(_owner):
        return [KVCache() for _ in range(num_layers)]

    patches = _patch_helpers(model, tokenizer, head_dim, texts, cache_factory)
    for p in patches:
        p.start()
    try:
        with pytest.raises(RuntimeError) as excinfo:
            sc.calibrate_model("fake/model", output_dir=tmp_path / "spectral")
    finally:
        for p in patches:
            p.stop()
    # The forward-pass ValueError is chained as the cause.
    assert isinstance(excinfo.value.__cause__, ValueError)
    assert "forward exploded" in str(excinfo.value.__cause__)


def test_calibrate_model_truncates_long_prompts_and_default_output_dir(tmp_path):
    # Exercise the >512-token truncation path and default output_dir=model/spectral.
    head_dim = 4
    num_layers = 1
    n_kv_heads = 1
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    backbone = _FakeBackbone(num_layers, n_kv_heads, head_dim)

    captured_seq = {}

    class _SeqCapModel(_FakeModel):
        def __call__(self, input_ids, cache=None):
            captured_seq["seq"] = input_ids.shape[1]
            super().__call__(input_ids, cache=cache)

    model = _SeqCapModel(backbone, head_dim)
    tokenizer = MagicMock()
    texts = ["w " * 600]  # 600 whitespace tokens after split

    def cache_factory(_owner):
        return [KVCache() for _ in range(num_layers)]

    patches = _patch_helpers(model, tokenizer, head_dim, texts, cache_factory)
    for p in patches:
        p.start()
    try:
        out = sc.calibrate_model(str(model_dir), num_samples=1)
    finally:
        for p in patches:
            p.stop()
    assert out == model_dir / "spectral"
    # Token list was truncated to 512 before the forward pass.
    assert captured_seq["seq"] == 512


def test_calibrate_model_n_kv_heads_falls_back_to_attention_heads(tmp_path):
    # num_key_value_heads is None -> falls back to num_attention_heads.
    head_dim = 4
    num_layers = 1
    backbone = _FakeBackbone(num_layers, 3, head_dim)
    backbone.args.num_key_value_heads = None
    backbone.args.num_attention_heads = 3
    model = _FakeModel(backbone, head_dim, n_kv=3)
    tokenizer = MagicMock()
    texts = ["one two three four"] * 2

    def cache_factory(_owner):
        return [KVCache() for _ in range(num_layers)]

    patches = _patch_helpers(model, tokenizer, head_dim, texts, cache_factory)
    for p in patches:
        p.start()
    try:
        out = sc.calibrate_model("fake/model", output_dir=tmp_path / "s", num_samples=2)
    finally:
        for p in patches:
            p.stop()
    cfg = __import__("json").loads((out / "spectral_config.json").read_text())
    assert cfg["meta"]["n_kv_heads"] == 3


def test_calibrate_model_skips_non_attention_cache_entries(tmp_path):
    # A cache entry that is not a valid attention cache is skipped, leaving its
    # layer with no KV data; the layer is then skipped in calibration.
    head_dim = 8
    num_layers = 1
    n_kv_heads = 1
    backbone = _FakeBackbone(num_layers, n_kv_heads, head_dim)

    class _NoOpModel(_FakeModel):
        def __call__(self, input_ids, cache=None):
            # Populate with a WRONG head_dim so _is_attention_cache rejects it.
            self.calls += 1
            seq = input_ids.shape[1]
            for entry in cache:
                entry.update_and_fetch(
                    mx.ones((1, n_kv_heads, seq, head_dim + 4)),
                    mx.ones((1, n_kv_heads, seq, head_dim + 4)),
                )

    model = _NoOpModel(backbone, head_dim)
    tokenizer = MagicMock()
    texts = ["one two three four"] * 2

    def cache_factory(_owner):
        return [KVCache() for _ in range(num_layers)]

    patches = _patch_helpers(model, tokenizer, head_dim, texts, cache_factory)
    for p in patches:
        p.start()
    try:
        with pytest.raises(RuntimeError, match="No KV vectors were collected"):
            sc.calibrate_model("fake/model", output_dir=tmp_path / "s", num_samples=2)
    finally:
        for p in patches:
            p.stop()


def test_calibrate_model_skips_empty_layer_but_calibrates_others(tmp_path):
    # Two layers: layer 0 gets valid KV (right head_dim), layer 1 gets a
    # rejected cache (wrong head_dim) -> layer 1 has no chunks and is skipped
    # in the per-layer calibration loop, but the run still succeeds.
    head_dim = 8
    num_layers = 2
    n_kv_heads = 1
    backbone = _FakeBackbone(num_layers, n_kv_heads, head_dim)

    class _MixedModel(_FakeModel):
        def __call__(self, input_ids, cache=None):
            self.calls += 1
            seq = input_ids.shape[1]
            # Layer 0: correct head_dim (accepted).
            cache[0].update_and_fetch(
                mx.ones((1, n_kv_heads, seq, head_dim)),
                mx.ones((1, n_kv_heads, seq, head_dim)),
            )
            # Layer 1: wrong head_dim (rejected by _is_attention_cache).
            cache[1].update_and_fetch(
                mx.ones((1, n_kv_heads, seq, head_dim + 2)),
                mx.ones((1, n_kv_heads, seq, head_dim + 2)),
            )

    model = _MixedModel(backbone, head_dim)
    tokenizer = MagicMock()
    texts = ["one two three four"] * 2

    def cache_factory(_owner):
        return [KVCache() for _ in range(num_layers)]

    patches = _patch_helpers(model, tokenizer, head_dim, texts, cache_factory)
    for p in patches:
        p.start()
    try:
        out = sc.calibrate_model("fake/model", output_dir=tmp_path / "s", num_samples=2)
    finally:
        for p in patches:
            p.stop()
    loaded = sc.load_calibration(out)
    # Only layer 0 was calibrated; layer 1 produced no data and was skipped.
    assert (0, 0, "key") in loaded
    assert (1, 0, "key") not in loaded


def test_calibrate_model_vlm_fallback_on_value_error(tmp_path):
    # load_model_with_strict_fallback raises ValueError -> mlx_vlm.load path.
    head_dim = 4
    num_layers = 1
    n_kv_heads = 1
    backbone = _FakeBackbone(num_layers, n_kv_heads, head_dim)
    model = _FakeModel(backbone, head_dim)
    processor = MagicMock()
    processor.tokenizer = MagicMock()
    texts = ["one two three four"] * 2

    def cache_factory(_owner):
        return [KVCache() for _ in range(num_layers)]

    fake_vlm = MagicMock()
    fake_vlm.load = MagicMock(return_value=(model, processor))

    patches = [
        patch(
            "olmlx.engine.flash.prepare.load_model_with_strict_fallback",
            side_effect=ValueError("not an mlx-lm model"),
        ),
        patch("olmlx.engine.flash.prepare._get_backbone", return_value=backbone),
        patch(
            "olmlx.engine.flash.prepare._get_c4_calibration_data", return_value=texts
        ),
        patch("olmlx.engine.flash.prepare._get_calibration_data", return_value=texts),
        patch(
            "olmlx.engine.flash.prepare._encode_tokens",
            side_effect=lambda tok, text: list(range(len(text.split()))),
        ),
        patch("olmlx.engine.turboquant_cache._detect_head_dim", return_value=head_dim),
        patch("mlx_lm.models.cache.make_prompt_cache", side_effect=cache_factory),
    ]
    for p in patches:
        p.start()
    try:
        with patch.dict("sys.modules", {"mlx_vlm": fake_vlm}):
            out = sc.calibrate_model(
                "fake/vlm", output_dir=tmp_path / "s", num_samples=2
            )
    finally:
        for p in patches:
            p.stop()
    fake_vlm.load.assert_called_once()
    assert (out / "calibration.safetensors").exists()


def test_calibrate_model_synthetic_dataset_branch(tmp_path):
    head_dim = 4
    num_layers = 1
    n_kv_heads = 1
    backbone = _FakeBackbone(num_layers, n_kv_heads, head_dim)
    model = _FakeModel(backbone, head_dim)
    tokenizer = MagicMock()
    texts = ["alpha beta gamma delta epsilon"] * 3

    def cache_factory(_owner):
        return [KVCache() for _ in range(num_layers)]

    synthetic_called = {"hit": False}

    def fake_synthetic(n):
        synthetic_called["hit"] = True
        return texts

    patches = _patch_helpers(model, tokenizer, head_dim, texts, cache_factory)
    # Replace synthetic patch with one that records the call.
    for p in patches:
        p.start()
    try:
        with patch(
            "olmlx.engine.flash.prepare._get_calibration_data",
            side_effect=fake_synthetic,
        ):
            out = sc.calibrate_model(
                "fake/model",
                output_dir=tmp_path / "spectral",
                num_samples=3,
                calibration_dataset="synthetic",
            )
    finally:
        for p in patches:
            p.stop()
    assert synthetic_called["hit"] is True
    cfg = __import__("json").loads((out / "spectral_config.json").read_text())
    assert cfg["meta"]["calibration_dataset"] == "synthetic"
