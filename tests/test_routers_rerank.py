from unittest.mock import AsyncMock, patch

import pytest


class TestRerankRouter:
    @pytest.mark.asyncio
    async def test_rerank_v1(self, app_client):
        with patch(
            "olmlx.routers.rerank.generate_rerank", new_callable=AsyncMock
        ) as mock:
            mock.return_value = {
                "results": [
                    {"index": 1, "relevance_score": 0.92},
                    {"index": 0, "relevance_score": 0.10},
                ]
            }
            resp = await app_client.post(
                "/v1/rerank",
                json={
                    "model": "bge-reranker",
                    "query": "what is mlx",
                    "documents": ["unrelated", "mlx is an array framework"],
                    "top_n": 2,
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["results"][0]["index"] == 1
        assert data["results"][0]["relevance_score"] == 0.92
        assert data["id"].startswith("rerank-")
        assert data["meta"]["api_version"]["version"] == "2"

    @pytest.mark.asyncio
    async def test_rerank_alias(self, app_client):
        with patch(
            "olmlx.routers.rerank.generate_rerank", new_callable=AsyncMock
        ) as mock:
            mock.return_value = {"results": []}
            resp = await app_client.post(
                "/rerank",
                json={"model": "m", "query": "q", "documents": ["a"]},
            )
        assert resp.status_code == 200
        assert resp.json()["id"].startswith("rerank-")

    @pytest.mark.asyncio
    async def test_rerank_returns_documents_when_requested(self, app_client):
        with patch(
            "olmlx.routers.rerank.generate_rerank", new_callable=AsyncMock
        ) as mock:
            mock.return_value = {
                "results": [{"index": 0, "relevance_score": 0.7, "document": "a"}]
            }
            resp = await app_client.post(
                "/v1/rerank",
                json={
                    "model": "m",
                    "query": "q",
                    "documents": ["a"],
                    "return_documents": True,
                },
            )
        assert resp.status_code == 200
        assert resp.json()["results"][0]["document"] == "a"

    @pytest.mark.asyncio
    async def test_rerank_rejects_empty_documents(self, app_client):
        resp = await app_client.post(
            "/v1/rerank",
            json={"model": "m", "query": "q", "documents": []},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_rerank_non_reranker_model_returns_400(self, app_client):
        # generate_rerank raises ValueError for a non-reranker model; the
        # app-level ValueError handler must turn it into a clean HTTP 400.
        with patch(
            "olmlx.routers.rerank.generate_rerank", new_callable=AsyncMock
        ) as mock:
            mock.side_effect = ValueError("Model 'qwen3' is not a reranker.")
            resp = await app_client.post(
                "/v1/rerank",
                json={"model": "qwen3", "query": "q", "documents": ["a"]},
            )
        assert resp.status_code == 400
        assert "not a reranker" in resp.text
