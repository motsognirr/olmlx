import pytest
from pydantic import ValidationError

from olmlx.schemas.rerank import RerankRequest, RerankResponse, RerankResult


def test_rerank_request_defaults():
    req = RerankRequest(model="bge-reranker", query="q", documents=["a", "b"])
    assert req.top_n is None
    assert req.max_tokens_per_doc == 4096
    assert req.return_documents is False


def test_rerank_request_rejects_empty_documents():
    with pytest.raises(ValidationError):
        RerankRequest(model="m", query="q", documents=[])


def test_rerank_request_rejects_empty_query():
    with pytest.raises(ValidationError):
        RerankRequest(model="m", query="  ", documents=["a"])


def test_rerank_request_rejects_nonpositive_top_n():
    with pytest.raises(ValidationError):
        RerankRequest(model="m", query="q", documents=["a"], top_n=0)


def test_rerank_response_shape():
    resp = RerankResponse(
        id="rerank-xyz",
        results=[RerankResult(index=1, relevance_score=0.9)],
        meta={"api_version": {"version": "2"}},
    )
    dumped = resp.model_dump()
    assert dumped["results"][0]["index"] == 1
    assert dumped["results"][0]["relevance_score"] == 0.9
    assert (
        "document" not in dumped["results"][0]
        or dumped["results"][0]["document"] is None
    )
