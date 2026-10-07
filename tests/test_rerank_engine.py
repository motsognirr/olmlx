import mlx.core as mx

from olmlx.engine.inference import _build_rerank_results, _score_pairs


class _FakeTokenizer:
    def __init__(self, model_max_length=512):
        self.model_max_length = model_max_length
        self.seen_max_length = None

    def __call__(
        self, query, documents, truncation, max_length, padding, return_tensors=None
    ):
        import numpy as np

        self.seen_max_length = max_length
        self.seen_truncation = truncation
        self.seen_query = query[0]
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


def test_score_pairs_multi_batch_preserves_order():
    # 5 docs with batch_size=2 spans 3 batches; scores must stay in doc order.
    model = _FakeModel([3.0, -3.0, 1.0, -1.0, 0.0])
    scores = _score_pairs(
        model,
        _FakeTokenizer(),
        "q",
        ["a", "b", "c", "d", "e"],
        max_tokens_per_doc=256,
        batch_size=2,
    )
    assert len(scores) == 5
    assert model.calls == 5
    # Monotonic in the input logits: 3 > 1 > 0 > -1 > -3.
    assert scores[0] > scores[2] > scores[4] > scores[3] > scores[1]


def test_score_pairs_clamps_sentinel_model_max_length():
    # transformers reports a huge sentinel; max_length must clamp to 512, then
    # be further bounded by max_tokens_per_doc.
    tok = _FakeTokenizer(model_max_length=1_000_000_019)
    _score_pairs(_FakeModel([0.0]), tok, "q", ["a"], max_tokens_per_doc=4096)
    assert tok.seen_max_length == 512

    tok2 = _FakeTokenizer(model_max_length=512)
    _score_pairs(_FakeModel([0.0]), tok2, "q", ["a"], max_tokens_per_doc=128)
    assert tok2.seen_max_length == 128


def test_build_rerank_results_top_n_and_sort():
    results = _build_rerank_results(
        scores=[0.1, 0.9, 0.5],
        documents=["a", "b", "c"],
        top_n=2,
        return_documents=False,
    )
    assert [r["index"] for r in results] == [1, 2]
    assert results[0]["relevance_score"] == 0.9
    assert "document" not in results[0]


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


def test_score_pairs_keeps_fitting_query_intact():
    # PR #768 review: docs absorb truncation; a query that fits is untouched
    # (longest_first would have trimmed it too, changing scores).
    tok = _FakeTokenizer()
    tok.encode = lambda text, add_special_tokens=False: list(range(len(text)))
    tok.decode = lambda ids: "x" * len(ids)
    tok.num_special_tokens_to_add = lambda pair=True: 4
    _score_pairs(_FakeModel([0.0]), tok, "q" * 50, ["a"], max_tokens_per_doc=128)
    assert tok.seen_truncation == "only_second"
    assert tok.seen_query == "q" * 50


def test_score_pairs_shortens_over_window_query():
    # #760: a query that alone exceeds the window is trimmed up front, leaving
    # room for the doc, instead of crashing / overrunning the position table.
    tok = _FakeTokenizer()
    tok.encode = lambda text, add_special_tokens=False: list(range(len(text)))
    tok.decode = lambda ids: "x" * len(ids)
    tok.num_special_tokens_to_add = lambda pair=True: 4
    _score_pairs(_FakeModel([0.0]), tok, "q" * 500, ["a"], max_tokens_per_doc=128)
    assert len(tok.seen_query) == 128 - 4 - 8


def test_score_pairs_bounded_by_position_table():
    # A tokenizer limit past the model's position table must not win.
    model = _FakeModel([0.0])
    model.max_input_tokens = 100
    tok = _FakeTokenizer(model_max_length=512)
    _score_pairs(model, tok, "q", ["a"], max_tokens_per_doc=4096)
    assert tok.seen_max_length == 100


def _len_tokenizer():
    tok = _FakeTokenizer()
    tok.encode = lambda text, add_special_tokens=False: list(range(len(text)))
    tok.decode = lambda ids: "x" * len(ids)
    tok.num_special_tokens_to_add = lambda pair=True: 4
    return tok


def test_score_pairs_small_window_still_bounds_query():
    # #768 review: a tiny max_tokens_per_doc must not skip query trimming.
    tok = _len_tokenizer()
    _score_pairs(_FakeModel([0.0]), tok, "q" * 500, ["a"], max_tokens_per_doc=10)
    assert len(tok.seen_query) == (10 - 4) - (10 - 4) // 2


def test_score_pairs_window_too_small_rejected():
    import pytest

    with pytest.raises(ValueError, match="max_tokens_per_doc"):
        _score_pairs(
            _FakeModel([0.0]), _len_tokenizer(), "q", ["a"], max_tokens_per_doc=5
        )
