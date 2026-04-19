"""Tests for shared SSD weight-store infrastructure.

Covers the generic LayerLruCache and HeaderSpec codec helpers that back
FlashWeightStore and FlashMoeWeightStore.
"""

from __future__ import annotations

import struct
import threading

import pytest

from olmlx.engine.flash._ssd_base import (
    HeaderSpec,
    LayerLruCache,
    encode_header,
    full_pread,
    parse_header,
)


# ---------------------------------------------------------------------------
# LayerLruCache
# ---------------------------------------------------------------------------


class TestLayerLruCache:
    def test_put_then_get_returns_value(self):
        cache: LayerLruCache[int, str] = LayerLruCache(max_per_layer=4)
        cache.put(0, 7, "seven")
        assert cache.get(0, 7) == "seven"

    def test_get_missing_returns_none(self):
        cache: LayerLruCache[int, str] = LayerLruCache(max_per_layer=4)
        assert cache.get(0, 7) is None

    def test_get_missing_layer_returns_none(self):
        cache: LayerLruCache[int, str] = LayerLruCache(max_per_layer=4)
        cache.put(1, 0, "x")
        assert cache.get(0, 0) is None

    def test_eviction_drops_oldest_when_full(self):
        cache: LayerLruCache[int, str] = LayerLruCache(max_per_layer=3)
        for i in range(4):
            cache.put(0, i, f"v{i}")
        # Key 0 is oldest; expect it evicted
        assert cache.get(0, 0) is None
        for i in range(1, 4):
            assert cache.get(0, i) == f"v{i}"

    def test_get_refreshes_lru_order(self):
        cache: LayerLruCache[int, str] = LayerLruCache(max_per_layer=3)
        for i in range(3):
            cache.put(0, i, f"v{i}")
        # Touch key 0 so key 1 becomes the oldest
        assert cache.get(0, 0) == "v0"
        cache.put(0, 3, "v3")
        assert cache.get(0, 1) is None
        assert cache.get(0, 0) == "v0"
        assert cache.get(0, 2) == "v2"
        assert cache.get(0, 3) == "v3"

    def test_put_updates_existing_entry_without_eviction(self):
        cache: LayerLruCache[int, str] = LayerLruCache(max_per_layer=2)
        cache.put(0, 0, "a")
        cache.put(0, 1, "b")
        cache.put(0, 0, "a2")  # update
        cache.put(0, 2, "c")  # evicts LRU, which is now 1
        assert cache.get(0, 0) == "a2"
        assert cache.get(0, 1) is None
        assert cache.get(0, 2) == "c"

    def test_put_with_zero_budget_is_noop(self):
        cache: LayerLruCache[int, str] = LayerLruCache(max_per_layer=0)
        cache.put(0, 0, "a")
        assert cache.get(0, 0) is None

    def test_layers_are_independent(self):
        cache: LayerLruCache[int, str] = LayerLruCache(max_per_layer=2)
        cache.put(0, 0, "a")
        cache.put(1, 0, "b")
        assert cache.get(0, 0) == "a"
        assert cache.get(1, 0) == "b"

    def test_get_batch_returns_only_cached(self):
        cache: LayerLruCache[int, str] = LayerLruCache(max_per_layer=4)
        cache.put(0, 0, "a")
        cache.put(0, 2, "c")
        got = cache.get_batch(0, [0, 1, 2, 3])
        assert got == {0: "a", 2: "c"}

    def test_get_batch_missing_layer_returns_empty(self):
        cache: LayerLruCache[int, str] = LayerLruCache(max_per_layer=4)
        assert cache.get_batch(99, [0, 1]) == {}

    def test_get_cached_indices_splits_hits_and_misses(self):
        cache: LayerLruCache[int, str] = LayerLruCache(max_per_layer=4)
        cache.put(0, 0, "a")
        cache.put(0, 2, "c")
        cached, missing = cache.get_cached_indices(0, [0, 1, 2, 3])
        assert cached == [0, 2]
        assert missing == [1, 3]

    def test_get_cached_indices_missing_layer_all_miss(self):
        cache: LayerLruCache[int, str] = LayerLruCache(max_per_layer=4)
        cached, missing = cache.get_cached_indices(0, [0, 1])
        assert cached == []
        assert missing == [0, 1]

    def test_thread_safety_under_concurrent_put(self):
        """Concurrent puts from many threads must not corrupt internal state."""
        cache: LayerLruCache[int, int] = LayerLruCache(max_per_layer=512)
        num_threads = 8
        per_thread = 64

        def worker(tid: int) -> None:
            for i in range(per_thread):
                cache.put(0, tid * per_thread + i, tid * per_thread + i)

        threads = [
            threading.Thread(target=worker, args=(t,)) for t in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All entries fit (max 512 > 8 * 64 = 512); every value must be readable
        for tid in range(num_threads):
            for i in range(per_thread):
                key = tid * per_thread + i
                assert cache.get(0, key) == key


# ---------------------------------------------------------------------------
# HeaderSpec codec
# ---------------------------------------------------------------------------


DENSE_SPEC = HeaderSpec(
    magic=0x464C5348,
    version=1,
    size=64,
    body_format="<III16s",
)

MOE_SPEC = HeaderSpec(
    magic=0x464C4D45,
    version=1,
    size=128,
    body_format="<IIIIIIQ",
)


class TestHeaderCodec:
    def test_dense_round_trip(self):
        num_neurons = 11008
        hidden = 4096
        dtype = b"float16"

        raw = encode_header(
            DENSE_SPEC,
            num_neurons,
            hidden,
            len(dtype),
            dtype.ljust(16, b"\x00"),
        )
        assert len(raw) == DENSE_SPEC.size

        nn, h, dlen, draw = parse_header(DENSE_SPEC, raw)
        assert nn == num_neurons
        assert h == hidden
        assert dlen == len(dtype)
        assert draw.rstrip(b"\x00") == dtype

    def test_moe_round_trip(self):
        raw = encode_header(
            MOE_SPEC,
            64,  # num_experts
            2048,  # hidden
            5632,  # intermediate
            1,  # is_quantized
            4,  # bits
            64,  # group_size
            123456,  # expert_byte_size
        )
        assert len(raw) == MOE_SPEC.size

        num_experts, hidden, inter, is_q, bits, gs, byte_size = parse_header(
            MOE_SPEC, raw
        )
        assert num_experts == 64
        assert hidden == 2048
        assert inter == 5632
        assert is_q == 1
        assert bits == 4
        assert gs == 64
        assert byte_size == 123456

    def test_pads_to_full_size(self):
        raw = encode_header(
            DENSE_SPEC,
            1,
            2,
            0,
            b"x".ljust(16, b"\x00"),
        )
        # Trailing bytes past body must be zeros (padding region)
        body_size = struct.calcsize(DENSE_SPEC.body_format) + 8  # + magic + version
        assert raw[body_size:] == b"\x00" * (DENSE_SPEC.size - body_size)

    def test_rejects_bad_magic(self):
        # Build a byte buffer that has the right size but wrong magic
        bogus = b"\xff\xff\xff\xff" + b"\x00" * (DENSE_SPEC.size - 4)
        with pytest.raises(ValueError, match="magic"):
            parse_header(DENSE_SPEC, bogus)

    def test_rejects_bad_version(self):
        bogus = struct.pack("<II", DENSE_SPEC.magic, 99) + b"\x00" * (
            DENSE_SPEC.size - 8
        )
        with pytest.raises(ValueError, match="version"):
            parse_header(DENSE_SPEC, bogus)


# ---------------------------------------------------------------------------
# full_pread
# ---------------------------------------------------------------------------


class TestFullPread:
    def test_reads_exact_bytes_at_offset(self, tmp_path):
        import os

        path = tmp_path / "blob.bin"
        payload = bytes(range(256)) * 4  # 1024 bytes
        path.write_bytes(payload)

        fd = os.open(str(path), os.O_RDONLY)
        try:
            assert full_pread(fd, 10, 0) == payload[:10]
            assert full_pread(fd, 10, 100) == payload[100:110]
            assert full_pread(fd, 256, 0) == payload[:256]
        finally:
            os.close(fd)

    def test_raises_on_eof(self, tmp_path):
        import os

        path = tmp_path / "blob.bin"
        path.write_bytes(b"abc")

        fd = os.open(str(path), os.O_RDONLY)
        try:
            with pytest.raises(OSError, match="EOF"):
                full_pread(fd, 100, 0)
        finally:
            os.close(fd)

    def test_reassembles_short_reads(self, tmp_path):
        """When os.pread returns partial chunks, full_pread must retry and stitch."""
        import os
        from unittest.mock import patch

        data = b"abcdefghijklmnopqrstuvwxyz"
        path = tmp_path / "blob.bin"
        path.write_bytes(data)

        fd = os.open(str(path), os.O_RDONLY)
        try:
            real_pread = os.pread
            call_count = [0]

            def short_pread(fd_, size_, offset_):
                call_count[0] += 1
                return real_pread(fd_, min(size_, 5), offset_)

            with patch("os.pread", side_effect=short_pread):
                result = full_pread(fd, len(data), 0)

            assert result == data
            assert call_count[0] > 1
        finally:
            os.close(fd)
