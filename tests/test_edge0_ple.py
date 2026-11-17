"""Tests for olmlx.engine.edge0.ple: the mmap FP8 n-gram table."""

import json
import math

import mlx.core as mx
import numpy as np
import pytest

from olmlx.engine.edge0 import ple
from tests.edge0_factories import (
    PLE_ROW_WIDTH,
    PLE_ROWS,
    PLE_SCALE,
    write_tiny_ple,
)


def _manifest(tmp_path, rows=()):
    rng = np.random.default_rng(0)
    known = write_tiny_ple(tmp_path / "ple-fp8.gguf", rng, rows)
    from olmlx.engine.edge0.gguf import GGUFFile

    with GGUFFile(tmp_path / "ple-fp8.gguf") as gguf:
        table = gguf.tensors["per_layer_token_embd.weight"]
        ple.write_manifest(
            tmp_path / ple.MANIFEST_NAME,
            data_file="ple-fp8.gguf",
            data_offset=table.offset,
            row_count=table.shape[0],
            row_width=table.shape[1],
            scale=PLE_SCALE,
        )
    return tmp_path / ple.MANIFEST_NAME, known


class TestE4M3:
    def test_table_matches_the_format(self):
        table = ple.E4M3_TABLE
        assert table[0] == 0.0
        assert table[0b0_0111_000] == 1.0  # exponent 7 -> 2**0
        assert table[0b1_0111_000] == -1.0
        assert table[0b0_1000_100] == 3.0  # 2 * 1.5
        assert table[0b0_0000_001] == math.ldexp(1 / 8, -6)  # subnormal
        assert table[0b0_1111_110] == 448.0  # largest finite
        assert table[0b0_1111_111] == 0.0 and table[0xFF] == 0.0  # NaN -> 0

    def test_agrees_with_mlx_from_fp8_on_finite_values(self):
        codes = np.array([b for b in range(256) if b & 0x7F != 0x7F], dtype=np.uint8)
        np.testing.assert_array_equal(
            np.array(mx.from_fp8(mx.array(codes), dtype=mx.float32)),
            ple.E4M3_TABLE[codes],
        )


class TestFp8NGramEmbedding:
    def test_lookup_decodes_and_restores_order(self, tmp_path):
        manifest, known = _manifest(tmp_path, rows=(5, 20_000_003, PLE_ROWS - 1))
        table = ple.Fp8NGramEmbedding(manifest)
        assert (table.row_count, table.row_width) == (PLE_ROWS, PLE_ROW_WIDTH)
        ids = mx.array([[PLE_ROWS - 1, 5], [5, 20_000_003]])
        out = table(ids)
        assert out.shape == (2, 2, PLE_ROW_WIDTH) and out.dtype == mx.bfloat16
        got = np.array(out.astype(mx.float32))
        for position, row in (
            ((0, 0), PLE_ROWS - 1),
            ((0, 1), 5),
            ((1, 0), 5),
            ((1, 1), 20_000_003),
        ):
            expected = mx.array(ple.E4M3_TABLE[known[row]] * PLE_SCALE).astype(
                mx.bfloat16
            )
            np.testing.assert_array_equal(
                got[position], np.array(expected.astype(mx.float32))
            )

    def test_out_of_range_row(self, tmp_path):
        manifest, _ = _manifest(tmp_path)
        with pytest.raises(IndexError):
            ple.Fp8NGramEmbedding(manifest)(mx.array([PLE_ROWS]))

    def test_empty_lookup(self, tmp_path):
        manifest, _ = _manifest(tmp_path)
        out = ple.Fp8NGramEmbedding(manifest)(mx.zeros((1, 0, 16), dtype=mx.int64))
        assert out.shape == (1, 0, 16, PLE_ROW_WIDTH)

    def test_truncated_table_is_rejected(self, tmp_path):
        manifest, _ = _manifest(tmp_path)
        data = tmp_path / "ple-fp8.gguf"
        with open(data, "r+b") as f:
            f.truncate(data.stat().st_size - 1)
        with pytest.raises(ValueError, match="incomplete"):
            ple.Fp8NGramEmbedding(manifest)


class TestMlxVlmHook:
    def test_dispatches_on_manifest_format(self, tmp_path):
        from mlx_vlm.models.qwen4_exp import ple_storage

        manifest, _ = _manifest(tmp_path)
        ple.install_mlx_vlm_hook()
        ple.install_mlx_vlm_hook()  # idempotent
        built = ple_storage.QuantizedMMapNGramEmbedding(manifest, cache_rows=0)
        assert isinstance(built, ple.Fp8NGramEmbedding)

    def test_other_manifests_reach_mlx_vlm(self, tmp_path):
        from mlx_vlm.models.qwen4_exp import ple_storage

        ple.install_mlx_vlm_hook()
        other = tmp_path / "other.json"
        other.write_text(json.dumps({"version": 2}))
        # mlx-vlm's own class rejects this incomplete manifest itself.
        with pytest.raises(ValueError, match="manifest missing fields"):
            ple_storage.QuantizedMMapNGramEmbedding(other)

    def test_hooked_name_is_still_the_mlx_vlm_class(self):
        from mlx_vlm.models.qwen4_exp import ple_storage

        ple.install_mlx_vlm_hook()
        hooked = ple_storage.QuantizedMMapNGramEmbedding
        assert isinstance(hooked, type)
        assert hooked.MANIFEST_VERSION == 2
