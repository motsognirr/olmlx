"""Tests for olmlx.routers.embed."""

from unittest.mock import AsyncMock, patch

import pytest


class TestEmbedRouter:
    @pytest.mark.asyncio
    async def test_embed_string(self, app_client):
        with patch(
            "olmlx.routers.embed.generate_embeddings", new_callable=AsyncMock
        ) as mock_emb:
            mock_emb.return_value = ([[0.1, 0.2, 0.3]], 1)
            resp = await app_client.post(
                "/api/embed",
                json={
                    "model": "qwen3",
                    "input": "hello",
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == "qwen3"
        assert data["embeddings"] == [[0.1, 0.2, 0.3]]
        assert "total_duration" in data
        # ``generate_embeddings`` returns the real summed token count; the route
        # must surface it as ``prompt_eval_count`` rather than discarding it.
        assert data["prompt_eval_count"] == 1

    @pytest.mark.asyncio
    async def test_embed_list(self, app_client):
        with patch(
            "olmlx.routers.embed.generate_embeddings", new_callable=AsyncMock
        ) as mock_emb:
            mock_emb.return_value = ([[0.1], [0.2]], 2)
            resp = await app_client.post(
                "/api/embed",
                json={
                    "model": "qwen3",
                    "input": ["hello", "world"],
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["embeddings"]) == 2
        assert data["prompt_eval_count"] == 2

    @pytest.mark.asyncio
    async def test_embeddings_legacy(self, app_client):
        with patch(
            "olmlx.routers.embed.generate_embeddings", new_callable=AsyncMock
        ) as mock_emb:
            mock_emb.return_value = ([[0.5, 0.6]], 1)
            resp = await app_client.post(
                "/api/embeddings",
                json={
                    "model": "qwen3",
                    "prompt": "hello",
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["embedding"] == [0.5, 0.6]

    @pytest.mark.asyncio
    async def test_embed_rejects_empty_string_input(self, app_client):
        resp = await app_client.post(
            "/api/embed",
            json={"model": "qwen3", "input": ""},
        )
        assert resp.status_code == 400
        assert "input" in resp.text.lower()
        assert "empty" in resp.text.lower()

    @pytest.mark.asyncio
    async def test_embed_rejects_empty_list_input(self, app_client):
        resp = await app_client.post(
            "/api/embed",
            json={"model": "qwen3", "input": []},
        )
        assert resp.status_code == 400
        assert "input" in resp.text.lower()
        assert "empty" in resp.text.lower()

    @pytest.mark.asyncio
    async def test_embeddings_legacy_rejects_empty_prompt(self, app_client):
        resp = await app_client.post(
            "/api/embeddings",
            json={"model": "qwen3", "prompt": ""},
        )
        assert resp.status_code == 400
        assert "prompt" in resp.text.lower()
        assert "empty" in resp.text.lower()
