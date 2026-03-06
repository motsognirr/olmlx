"""Tests for mlx_ollama.routers.blobs."""

import hashlib

import pytest


class TestBlobsRouter:
    @pytest.mark.asyncio
    async def test_check_blob_not_found(self, app_client):
        resp = await app_client.head("/api/blobs/sha256:abc")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_upload_and_check_blob(self, app_client):
        data = b"test blob data"
        digest = "sha256:" + hashlib.sha256(data).hexdigest()

        # Upload
        resp = await app_client.post(
            f"/api/blobs/{digest}",
            content=data,
        )
        assert resp.status_code == 201

        # Check
        resp = await app_client.head(f"/api/blobs/{digest}")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_upload_blob_digest_mismatch(self, app_client):
        resp = await app_client.post(
            "/api/blobs/sha256:wrong",
            content=b"test data",
        )
        assert resp.status_code == 400
