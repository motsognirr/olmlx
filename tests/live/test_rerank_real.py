"""Live cross-encoder rerank tests against real models (#369).

Covers BOTH target checkpoint layouts:
  * BAAI/bge-reranker-v2-m3            -> standard XLM-RoBERTa key layout
  * jinaai/jina-reranker-v2-base-multilingual -> flash (fused-Wqkv) layout

This is the validation gate for remap_flash: it asserts jina's real checkpoint
keys match the layout our weight loader infers. Skips cleanly when a model is
not already downloaded (no forced multi-GB pull). Lives outside
tests/integration/ to avoid that package's autouse MLX mock.
"""

import pytest

pytestmark = [pytest.mark.real_model]

BGE = "BAAI/bge-reranker-v2-m3"
JINA = "jinaai/jina-reranker-v2-base-multilingual"


def _local_dir_or_skip(repo: str) -> str:
    """Return the local snapshot dir for an already-cached repo, else skip."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        return snapshot_download(repo, local_files_only=True)
    except LocalEntryNotFoundError:
        pytest.skip(f"{repo} not downloaded; skipping live rerank test")


def _safetensors_keys(path: str) -> list[str]:
    import glob
    import os

    from safetensors import safe_open

    # Header-only read — avoids materializing ~2.3 GB of weights just to
    # inspect the key names for layout detection.
    keys: list[str] = []
    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        with safe_open(f, framework="numpy") as st:
            keys.extend(st.keys())
    return keys


@pytest.mark.parametrize("repo,expected", [(BGE, "standard"), (JINA, "flash")])
def test_checkpoint_layout_matches_expectation(repo, expected):
    from olmlx.engine.rerank.weights import detect_layout

    path = _local_dir_or_skip(repo)
    keys = _safetensors_keys(path)
    got = detect_layout(keys)
    assert got == expected, f"{repo}: detected {got}; sample keys: {keys[:8]}"


@pytest.mark.parametrize("repo", [BGE, JINA])
def test_load_and_rank(repo):
    from transformers import AutoTokenizer

    from olmlx.engine.inference import _build_rerank_results, _score_pairs
    from olmlx.engine.rerank.weights import load_cross_encoder

    path = _local_dir_or_skip(repo)
    model = load_cross_encoder(path)
    tok = AutoTokenizer.from_pretrained(path)

    query = "What is the capital of France?"
    docs = [
        "The capital of France is Paris.",
        "Bananas are a good source of potassium.",
    ]
    scores = _score_pairs(model, tok, query, docs, max_tokens_per_doc=512)
    results = _build_rerank_results(
        scores=scores, documents=docs, top_n=None, return_documents=False
    )
    assert results[0]["index"] == 0  # the relevant doc ranks first
    assert scores[0] > scores[1]


def test_batch_of_100_documents():
    # Throughput / no-crash check: scores a 100-doc batch (multiple internal
    # micro-batches) and confirms the count and that scores are valid
    # probabilities. Not a ranking-quality assertion.
    from transformers import AutoTokenizer

    from olmlx.engine.inference import _score_pairs
    from olmlx.engine.rerank.weights import load_cross_encoder

    path = _local_dir_or_skip(BGE)
    model = load_cross_encoder(path)
    tok = AutoTokenizer.from_pretrained(path)
    docs = [f"document number {i} about various topics" for i in range(100)]
    scores = _score_pairs(
        model, tok, "find the relevant document", docs, max_tokens_per_doc=256
    )
    assert len(scores) == 100
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_parity_against_transformers_reference():
    pytest.importorskip("torch")
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from olmlx.engine.inference import _score_pairs
    from olmlx.engine.rerank.weights import load_cross_encoder

    path = _local_dir_or_skip(BGE)
    tok = AutoTokenizer.from_pretrained(path)
    query = "What is MLX?"
    docs = ["MLX is an array framework for Apple silicon.", "I like sandwiches."]

    ours = _score_pairs(
        load_cross_encoder(path), tok, query, docs, max_tokens_per_doc=512
    )

    ref_model = AutoModelForSequenceClassification.from_pretrained(path).eval()
    with torch.no_grad():
        enc = tok(
            [query, query],
            docs,
            padding=True,
            truncation="only_second",
            max_length=512,
            return_tensors="pt",
        )
        ref = torch.sigmoid(ref_model(**enc).logits.squeeze(-1)).tolist()

    assert np.allclose(np.array(ours), np.array(ref), atol=1e-2), (ours, ref)
