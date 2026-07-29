from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest

from olmlx.engine.reap.calibrate import SaliencyAccumulator
from olmlx.engine.reap.report import build_report, source_overlap, streaming_perplexity
from tests.reap_factories import make_tiny_qwen3_moe
from tests.test_reap_calibrate import FakeTokenizer


class TestSourceOverlap:
    def _acc(self):
        acc = SaliencyAccumulator(1, 8, ["a", "b"])
        # source a favors experts 0-3, source b favors 2-5 -> top4 overlap = {2,3} = 0.5
        for e, v in enumerate([9, 8, 7, 6, 1, 1, 1, 1]):
            acc.add(0, 0, np.array([e]), np.array([float(v)]))
        for e, v in enumerate([1, 1, 9, 8, 7, 6, 1, 1]):
            acc.add(0, 1, np.array([e]), np.array([float(v)]))
        return acc

    def test_pairwise_overlap(self):
        rep = source_overlap(self._acc(), keep_count=4)
        assert rep["pairs"]["a|b"]["mean_overlap"] == pytest.approx(0.5)
        assert rep["chance"] == pytest.approx(0.5)

    def test_identical_sources_full_overlap(self):
        acc = SaliencyAccumulator(2, 8, ["a", "b"])
        for src in (0, 1):
            for layer in (0, 1):
                acc.add(layer, src, np.arange(8), np.arange(8, dtype=np.float64))
        rep = source_overlap(acc, keep_count=3)
        assert rep["pairs"]["a|b"]["mean_overlap"] == pytest.approx(1.0)


class TestStreamingPerplexity:
    def test_matches_direct_forward(self, monkeypatch):
        from olmlx.engine.reap import calibrate as cal_mod

        mx.random.seed(21)
        model, _ = make_tiny_qwen3_moe()
        tok = FakeTokenizer()
        texts = [
            ("english", "hello world example " * 4),
            ("code", "def f(): pass " * 4),
        ]

        # direct reference on the same (fresh) model
        expected: dict[str, list[float]] = {}
        for source, text in texts:
            ids = tok.encode(text)[:16]
            logits = model(mx.array([ids])).astype(mx.float32)
            logprobs = logits[:, :-1] - mx.logsumexp(
                logits[:, :-1], axis=-1, keepdims=True
            )
            tgt = mx.array([ids[1:]])
            nll = -mx.take_along_axis(logprobs, tgt[..., None], axis=-1).sum()
            expected.setdefault(source, []).append((float(nll), len(ids) - 1))

        monkeypatch.setattr(
            cal_mod, "load_model_with_strict_fallback", lambda path, lazy: (model, tok)
        )
        ppl = streaming_perplexity("/fake", texts, max_tokens=16)
        for source in ("english", "code"):
            nll, n = expected[source][0]
            assert ppl[source] == pytest.approx(math.exp(nll / n), rel=1e-3)


class TestBuildReport:
    def test_overlap_only_when_no_models(self, tmp_path):
        from olmlx.engine.reap.calibrate import save_saliency

        acc = SaliencyAccumulator(1, 8, ["a", "b"])
        acc.add(0, 0, np.arange(8), np.linspace(1, 8, 8))
        acc.add(0, 1, np.arange(8), np.linspace(8, 1, 8))
        meta = {
            "model_type": "qwen3_moe",
            "num_experts": 8,
            "top_k": 2,
            "moe_layer_indices": [0],
            "sources": ["a", "b"],
        }
        save_saliency(tmp_path / "s.npz", acc, meta)
        rep = build_report(tmp_path / "s.npz", keep_count=4)
        assert rep["perplexity"] is None
        assert "a|b" in rep["overlap"]["pairs"]
        assert rep["meta"]["model_type"] == "qwen3_moe"
