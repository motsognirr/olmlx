"""Tests for mlx_ollama.routers.embed."""

from unittest.mock import AsyncMock, patch

import pytest


class TestEmbedRouter:
    @pytest.mark.asyncio
    async def test_embed_string(self, app_client):
        with patch(
            "mlx_ollama.routers.embed.generate_embeddings", new_callable=AsyncMock
        ) as mock_emb:
            mock_emb.return_value = [[0.1, 0.2, 0.3]]
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

    @pytest.mark.asyncio
    async def test_embed_list(self, app_client):
        with patch(
            "mlx_ollama.routers.embed.generate_embeddings", new_callable=AsyncMock
        ) as mock_emb:
            mock_emb.return_value = [[0.1], [0.2]]
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

    @pytest.mark.asyncio
    async def test_embeddings_legacy(self, app_client):
        with patch(
            "mlx_ollama.routers.embed.generate_embeddings", new_callable=AsyncMock
        ) as mock_emb:
            mock_emb.return_value = [[0.5, 0.6]]
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
