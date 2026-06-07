"""Regression coverage for olmlx.routers.blobs.

Targets the Ollama /api/blobs HEAD/POST endpoints: digest validation, the
Content-Length and streamed-size 413 guards, digest-mismatch, the success
path (201 + on-disk blob), and HEAD presence checks.
"""

import hashlib
import types
from pathlib import Path

import pytest

from olmlx.routers import blobs as blobs_mod


def _digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


class _FakeStore:
    """Minimal stand-in for ModelStore exposing models_dir + has_blob."""

    def __init__(self, models_dir: Path):
        self.models_dir = models_dir

    def has_blob(self, digest: str) -> bool:
        return (self.models_dir / "blobs" / digest).exists()


def _make_request(store, headers, chunks):
    """Build a fake starlette Request that drives upload_blob's loop directly.

    Calling the endpoint coroutine in-process (rather than through ASGI
    streaming) lets coverage trace the ``async for chunk in request.stream()``
    body that the TestClient path leaves untraced.
    """

    async def stream():
        for chunk in chunks:
            yield chunk

    app = types.SimpleNamespace(state=types.SimpleNamespace(model_store=store))
    return types.SimpleNamespace(app=app, headers=headers, stream=stream)


class TestCheckBlob:
    async def test_head_missing_blob_returns_404(self, app_client):
        # 64 hex chars but no such blob on disk.
        digest = "sha256:" + "0" * 64
        resp = await app_client.head(f"/api/blobs/{digest}")
        assert resp.status_code == 404

    async def test_head_present_blob_returns_200(self, app_client, mock_store):
        data = b"present-blob-payload"
        digest = _digest(data)
        blobs_dir = mock_store.models_dir / "blobs"
        blobs_dir.mkdir(parents=True, exist_ok=True)
        (blobs_dir / digest).write_bytes(data)

        resp = await app_client.head(f"/api/blobs/{digest}")
        assert resp.status_code == 200


class TestUploadDigestValidation:
    async def test_invalid_digest_format_returns_400(self, app_client):
        # Missing sha256: prefix / not 64 hex chars.
        resp = await app_client.post("/api/blobs/not-a-digest", content=b"x")
        assert resp.status_code == 400
        assert resp.json() == {"error": "invalid digest format"}

    async def test_uppercase_hex_digest_rejected(self, app_client):
        # Regex requires lowercase hex; uppercase must be rejected.
        digest = "sha256:" + "A" * 64
        resp = await app_client.post(f"/api/blobs/{digest}", content=b"x")
        assert resp.status_code == 400
        assert resp.json()["error"] == "invalid digest format"

    async def test_short_hex_digest_rejected(self, app_client):
        digest = "sha256:" + "0" * 63
        resp = await app_client.post(f"/api/blobs/{digest}", content=b"x")
        assert resp.status_code == 400


class TestUploadSuccess:
    async def test_upload_writes_blob_and_returns_201(self, app_client, mock_store):
        data = b"hello world blob"
        digest = _digest(data)

        resp = await app_client.post(f"/api/blobs/{digest}", content=data)
        assert resp.status_code == 201

        # Blob landed under models_dir/blobs/<digest> with exact bytes.
        blob_path = mock_store.models_dir / "blobs" / digest
        assert blob_path.exists()
        assert blob_path.read_bytes() == data

        # And HEAD now reports it present.
        head = await app_client.head(f"/api/blobs/{digest}")
        assert head.status_code == 200

    async def test_upload_empty_blob_success(self, app_client, mock_store):
        data = b""
        digest = _digest(data)

        resp = await app_client.post(f"/api/blobs/{digest}", content=data)
        assert resp.status_code == 201
        assert (mock_store.models_dir / "blobs" / digest).read_bytes() == b""

    async def test_upload_multichunk_blob_success(self, app_client, mock_store):
        # Large-ish payload so httpx streams it as several chunks, exercising
        # the async-for accumulation loop more than once.
        data = b"abc123" * 100_000  # ~600 KB
        digest = _digest(data)

        resp = await app_client.post(f"/api/blobs/{digest}", content=data)
        assert resp.status_code == 201
        assert (mock_store.models_dir / "blobs" / digest).read_bytes() == data

    async def test_blobs_dir_created_when_absent(self, app_client, mock_store):
        # Upload path must mkdir(parents=True) the blobs dir itself.
        blobs_dir = mock_store.models_dir / "blobs"
        assert not blobs_dir.exists()

        data = b"creates the dir"
        digest = _digest(data)
        resp = await app_client.post(f"/api/blobs/{digest}", content=data)
        assert resp.status_code == 201
        assert blobs_dir.is_dir()


class TestUploadDigestMismatch:
    async def test_digest_mismatch_returns_400_and_no_blob(
        self, app_client, mock_store
    ):
        # Well-formed digest, but the body hashes to something else.
        claimed = "sha256:" + "1" * 64
        body = b"this body does not match the claimed digest"
        resp = await app_client.post(f"/api/blobs/{claimed}", content=body)

        assert resp.status_code == 400
        assert resp.json() == {"error": "digest mismatch"}

        # Mismatched temp file must be unlinked; nothing left under blobs/.
        blobs_dir = mock_store.models_dir / "blobs"
        leftovers = list(blobs_dir.iterdir()) if blobs_dir.exists() else []
        assert leftovers == []

    async def test_digest_mismatch_leaves_no_partial_named_blob(
        self, app_client, mock_store
    ):
        claimed = "sha256:" + "2" * 64
        resp = await app_client.post(f"/api/blobs/{claimed}", content=b"abc")
        assert resp.status_code == 400
        assert not (mock_store.models_dir / "blobs" / claimed).exists()


class TestContentLengthGuard:
    async def test_content_length_over_limit_returns_413_fast_path(
        self, app_client, mock_store
    ):
        digest = "sha256:" + "3" * 64
        oversize = blobs_mod.MAX_BLOB_SIZE + 1
        # Send a small body but lie about Content-Length so the fast-path
        # rejects before reading the stream.
        resp = await app_client.post(
            f"/api/blobs/{digest}",
            content=b"tiny",
            headers={"content-length": str(oversize)},
        )
        assert resp.status_code == 413
        assert "blob too large" in resp.json()["error"]
        # No blobs dir work should have produced a stored blob.
        assert not (mock_store.models_dir / "blobs" / digest).exists()

    async def test_malformed_content_length_treated_as_zero(
        self, app_client, mock_store
    ):
        # A non-integer Content-Length triggers the ValueError branch (cl=0),
        # so the upload proceeds normally based on the actual body.
        data = b"normal payload despite bad header"
        digest = _digest(data)
        resp = await app_client.post(
            f"/api/blobs/{digest}",
            content=data,
            headers={"content-length": "not-a-number"},
        )
        # httpx may recompute content-length; the key behavior is that a bad
        # value does not crash and the upload completes.
        assert resp.status_code == 201
        assert (mock_store.models_dir / "blobs" / digest).read_bytes() == data


class TestStreamedSizeGuard:
    async def test_streamed_body_over_limit_returns_413(
        self, app_client, mock_store, monkeypatch
    ):
        # Shrink the cap so a modest body trips the streamed-size guard
        # (received > MAX_BLOB_SIZE) without lying about Content-Length.
        monkeypatch.setattr(blobs_mod, "MAX_BLOB_SIZE", 8)
        digest = "sha256:" + "4" * 64
        resp = await app_client.post(
            f"/api/blobs/{digest}",
            content=b"0123456789abcdef",  # 16 bytes > 8
        )
        assert resp.status_code == 413
        assert "blob too large" in resp.json()["error"]
        # Temp file from the aborted stream must be cleaned up.
        blobs_dir = mock_store.models_dir / "blobs"
        leftovers = list(blobs_dir.iterdir()) if blobs_dir.exists() else []
        assert leftovers == []


class TestFdopenFailure:
    async def test_fdopen_failure_closes_fd_and_propagates(
        self, app_client, monkeypatch
    ):
        # Force os.fdopen to raise so the except branch closes the raw fd and
        # re-raises (surfaced as a 500 by TestClient with raise_app_exceptions
        # off). Track that os.close was invoked on the leaked descriptor.
        closed = []
        real_close = blobs_mod.os.close

        def fake_close(fd):
            closed.append(fd)
            return real_close(fd)

        def boom_fdopen(fd, mode):
            raise OSError("fdopen failed")

        monkeypatch.setattr(blobs_mod.os, "fdopen", boom_fdopen)
        monkeypatch.setattr(blobs_mod.os, "close", fake_close)

        digest = "sha256:" + "5" * 64
        resp = await app_client.post(f"/api/blobs/{digest}", content=b"data")

        assert resp.status_code == 500
        # The fd opened by mkstemp must have been explicitly closed.
        assert closed, "expected os.close to run on the orphaned fd"


class TestUploadBlobDirect:
    """Drive the endpoint coroutine in-process to exercise the stream loop."""

    async def test_direct_success_writes_blob_returns_201(self, tmp_path):
        store = _FakeStore(tmp_path / "models")
        data = b"direct-path-payload-spanning-chunks"
        digest = _digest(data)
        # Split into several chunks to loop multiple times.
        chunks = [data[i : i + 7] for i in range(0, len(data), 7)]

        resp = await blobs_mod.upload_blob(digest, _make_request(store, {}, chunks))
        assert resp.status_code == 201

        blob_path = store.models_dir / "blobs" / digest
        assert blob_path.read_bytes() == data

    async def test_direct_digest_mismatch_returns_400(self, tmp_path):
        store = _FakeStore(tmp_path / "models")
        claimed = "sha256:" + "9" * 64
        resp = await blobs_mod.upload_blob(
            claimed, _make_request(store, {}, [b"abc", b"def"])
        )
        assert resp.status_code == 400
        assert resp.body == b'{"error":"digest mismatch"}'
        # Temp file removed; no stored blob.
        blobs_dir = store.models_dir / "blobs"
        assert list(blobs_dir.iterdir()) == []

    async def test_direct_streamed_size_limit_returns_413(self, tmp_path, monkeypatch):
        monkeypatch.setattr(blobs_mod, "MAX_BLOB_SIZE", 4)
        store = _FakeStore(tmp_path / "models")
        digest = "sha256:" + "8" * 64
        resp = await blobs_mod.upload_blob(
            digest, _make_request(store, {}, [b"12", b"345"])
        )
        assert resp.status_code == 413
        # Aborted temp file cleaned up.
        blobs_dir = store.models_dir / "blobs"
        assert list(blobs_dir.iterdir()) == []

    async def test_direct_streamed_413_swallows_unlink_oserror(
        self, tmp_path, monkeypatch
    ):
        # When the over-limit cleanup unlink itself fails, the OSError is
        # swallowed and a 413 is still returned (lines 62-63).
        monkeypatch.setattr(blobs_mod, "MAX_BLOB_SIZE", 4)
        store = _FakeStore(tmp_path / "models")

        def boom_unlink(path):
            raise OSError("cannot unlink")

        monkeypatch.setattr(blobs_mod.os, "unlink", boom_unlink)
        digest = "sha256:" + "6" * 64
        resp = await blobs_mod.upload_blob(
            digest, _make_request(store, {}, [b"12345678"])
        )
        assert resp.status_code == 413

    async def test_direct_content_length_fast_path_413(self, tmp_path):
        store = _FakeStore(tmp_path / "models")
        digest = "sha256:" + "7" * 64
        headers = {"content-length": str(blobs_mod.MAX_BLOB_SIZE + 1)}
        resp = await blobs_mod.upload_blob(
            digest, _make_request(store, headers, [b"x"])
        )
        assert resp.status_code == 413
        # Fast-path rejects before creating the blobs dir.
        assert not (store.models_dir / "blobs").exists()


class TestDigestRegex:
    def test_digest_regex_accepts_canonical_form(self):
        assert blobs_mod._DIGEST_RE.match("sha256:" + "a" * 64)

    @pytest.mark.parametrize(
        "bad",
        [
            "sha256:" + "a" * 63,
            "sha256:" + "a" * 65,
            "sha512:" + "a" * 64,
            "sha256:" + "g" * 64,
            "a" * 64,
        ],
    )
    def test_digest_regex_rejects_malformed(self, bad):
        assert blobs_mod._DIGEST_RE.match(bad) is None
