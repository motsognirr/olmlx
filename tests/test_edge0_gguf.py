"""Tests for olmlx.engine.edge0.gguf: the minimal GGUF container reader."""

import struct

import numpy as np
import pytest

from olmlx.engine.edge0.gguf import GGUFFile


def _string(text: str) -> bytes:
    data = text.encode()
    return struct.pack("<Q", len(data)) + data


def _write_gguf(path, metadata, tensors, *, alignment=32, version=3):
    """Write a GGUF v3 file. *tensors*: list of (name, dims, ggml_type, bytes)."""
    out = bytearray(b"GGUF")
    out += struct.pack("<IQQ", version, len(tensors), len(metadata))
    for key, (value_type, payload) in metadata.items():
        out += _string(key) + struct.pack("<I", value_type) + payload
    offset = 0
    offsets = []
    for name, dims, ggml_type, data in tensors:
        out += _string(name) + struct.pack("<I", len(dims))
        out += struct.pack(f"<{len(dims)}Q", *dims)
        out += struct.pack("<IQ", ggml_type, offset)
        offsets.append(offset)
        offset += len(data)
        offset += (-offset) % alignment
    out += b"\0" * ((-len(out)) % alignment)
    data_start = len(out)
    for (_, _, _, data), tensor_offset in zip(tensors, offsets):
        out += b"\0" * (data_start + tensor_offset - len(out))
        out += data
    path.write_bytes(bytes(out))
    return data_start


@pytest.fixture
def sample(tmp_path):
    f32 = np.arange(6, dtype="<f4")
    q8 = np.arange(68, dtype=np.uint8)
    metadata = {
        "general.architecture": (8, _string("qwen4exp")),
        "general.alignment": (4, struct.pack("<I", 64)),
        "qwen4exp.block_count": (4, struct.pack("<I", 48)),
        "some.float": (6, struct.pack("<f", 1.5)),
        "some.bool": (7, b"\x01"),
        "some.u64": (10, struct.pack("<Q", 2**40)),
        "some.ints": (9, struct.pack("<IQ", 5, 3) + struct.pack("<3i", 1, -2, 3)),
        "some.strings": (9, struct.pack("<IQ", 8, 2) + _string("a") + _string("bc")),
    }
    tensors = [
        # GGUF dims are innermost-first: [3, 2] is a 2x3 row-major matrix.
        ("blk.0.norm.weight", [3, 2], 0, f32.tobytes()),
        ("blk.0.attn_q.weight", [32, 2], 8, q8.tobytes()),
    ]
    path = tmp_path / "sample.gguf"
    data_start = _write_gguf(path, metadata, tensors, alignment=64)
    return path, data_start, f32, q8


class TestGGUFFile:
    def test_reads_metadata(self, sample):
        path, _, _, _ = sample
        with GGUFFile(path) as gguf:
            assert gguf.metadata["general.architecture"] == "qwen4exp"
            assert gguf.metadata["qwen4exp.block_count"] == 48
            assert gguf.metadata["some.float"] == 1.5
            assert gguf.metadata["some.bool"] is True
            assert gguf.metadata["some.u64"] == 2**40
            assert gguf.metadata["some.ints"] == [1, -2, 3]
            assert gguf.metadata["some.strings"] == ["a", "bc"]

    def test_tensor_directory(self, sample):
        path, data_start, f32, q8 = sample
        with GGUFFile(path) as gguf:
            assert list(gguf.tensors) == ["blk.0.norm.weight", "blk.0.attn_q.weight"]
            norm = gguf.tensors["blk.0.norm.weight"]
            # Shape is reported outermost-first (numpy order).
            assert norm.shape == (2, 3)
            assert norm.ggml_type == 0
            assert norm.offset == data_start
            assert norm.nbytes == f32.nbytes
            attn = gguf.tensors["blk.0.attn_q.weight"]
            assert attn.shape == (2, 32)
            assert attn.nbytes == 68
            # Second tensor starts on the file's declared 64-byte alignment.
            assert attn.offset == data_start + 64

    def test_raw_is_a_view_of_the_file(self, sample):
        path, _, f32, q8 = sample
        with GGUFFile(path) as gguf:
            np.testing.assert_array_equal(
                np.frombuffer(gguf.raw("blk.0.norm.weight"), dtype="<f4"), f32
            )
            np.testing.assert_array_equal(gguf.raw("blk.0.attn_q.weight"), q8)

    def test_load_decodes_and_shapes(self, sample):
        path, _, f32, _ = sample
        with GGUFFile(path) as gguf:
            np.testing.assert_array_equal(
                gguf.load("blk.0.norm.weight"), f32.reshape(2, 3)
            )
            decoded = gguf.load("blk.0.attn_q.weight")
            assert decoded.shape == (2, 32) and decoded.dtype == np.float32

    def test_missing_tensor(self, sample):
        path, _, _, _ = sample
        with GGUFFile(path) as gguf:
            with pytest.raises(KeyError, match="nope"):
                gguf.raw("nope")

    def test_rejects_non_gguf(self, tmp_path):
        path = tmp_path / "bad.gguf"
        path.write_bytes(b"NOPE" + b"\0" * 64)
        with pytest.raises(ValueError, match="not a GGUF file"):
            GGUFFile(path)

    def test_rejects_unsupported_version(self, tmp_path):
        path = tmp_path / "old.gguf"
        path.write_bytes(b"GGUF" + struct.pack("<IQQ", 1, 0, 0))
        with pytest.raises(ValueError, match="version 1"):
            GGUFFile(path)

    def test_rejects_tensor_past_end_of_file(self, tmp_path):
        path = tmp_path / "trunc.gguf"
        _write_gguf(path, {}, [("t", [4], 0, np.zeros(4, "<f4").tobytes())])
        path.write_bytes(path.read_bytes()[:-8])
        with pytest.raises(ValueError, match="past the end"):
            GGUFFile(path)
