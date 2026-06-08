import mlx.core as mx

from olmlx.engine.inference import _build_rerank_results, _score_pairs


class _FakeTokenizer:
    model_max_length = 512

    def __call__(
        self, query, documents, truncation, max_length, padding, return_tensors=None
    ):
        import numpy as np

        n = len(documents)
        ids = np.ones((n, 4), dtype=np.int64) * 5
        mask = np.ones((n, 4), dtype=np.int64)
        return {"input_ids": ids, "attention_mask": mask}


class _FakeModel:
    def __init__(self, scores):
        self._scores = scores
        self.calls = 0

    def __call__(self, input_ids, attention_mask):
        b = input_ids.shape[0]
        out = self._scores[self.calls : self.calls + b]
        self.calls += b
        return mx.array(out).reshape(b, 1)


def test_score_pairs_sigmoid_and_order():
    model = _FakeModel([2.0, -2.0, 0.0])
    scores = _score_pairs(
        model,
        _FakeTokenizer(),
        "q",
        ["a", "b", "c"],
        max_tokens_per_doc=256,
        batch_size=8,
    )
    assert len(scores) == 3
    assert scores[0] > scores[2] > scores[1]
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_build_rerank_results_top_n_and_sort():
    results = _build_rerank_results(
        scores=[0.1, 0.9, 0.5],
        documents=["a", "b", "c"],
        top_n=2,
        return_documents=False,
    )
    assert [r["index"] for r in results] == [1, 2]
    assert results[0]["relevance_score"] == 0.9
    assert "document" not in results[0] or results[0]["document"] is None


def test_build_rerank_results_return_documents():
    results = _build_rerank_results(
        scores=[0.1, 0.9],
        documents=["a", "b"],
        top_n=None,
        return_documents=True,
    )
    assert results[0]["document"] == "b"


def test_build_rerank_results_top_n_clamped():
    results = _build_rerank_results(
        scores=[0.1, 0.9],
        documents=["a", "b"],
        top_n=10,
        return_documents=False,
    )
    assert len(results) == 2
