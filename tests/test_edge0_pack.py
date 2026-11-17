"""Tests for olmlx.engine.edge0.pack: the Edge0 latent pack reader."""

import json

import numpy as np
import pytest

from olmlx.engine.edge0 import ggml_quants as gq
from olmlx.engine.edge0.pack import Edge0Pack, Edge0PackError, is_edge0_pack
from tests.edge0_factories import make_pack


class TestDetection:
    def test_recognizes_a_pack(self, tmp_path):
        assert is_edge0_pack(make_pack(tmp_path).root)

    def test_plain_directory_is_not_a_pack(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        assert not is_edge0_pack(tmp_path)
        with pytest.raises(Edge0PackError, match="not an Edge0 pack"):
            Edge0Pack(tmp_path)


class TestGeometry:
    def test_geometry_from_headers(self, tmp_path):
        truth = make_pack(tmp_path, layers=3, experts=8, groups=2)
        g = Edge0Pack(truth.root).geometry
        assert (g.n_layers, g.n_experts, g.n_groups) == (3, 8, 2)
        assert (g.hidden, g.latent, g.ff) == (64, 256, 32)
        assert (g.gu_type, g.down_type) == (gq.GGML_Q3_K, gq.GGML_IQ4_NL)
        assert g.record_bytes == truth.record_bytes
        assert g.gate_bytes == 32 * 110
        assert g.down_offset == 2 * 32 * 110

    def test_real_145b_record_size(self):
        from olmlx.engine.edge0.pack import LatentGeometry

        g = LatentGeometry(48, 512, 8, 2560, 1792, 640, gq.GGML_Q3_K, gq.GGML_IQ4_NL)
        assert g.record_bytes == 1_630_720
        assert g.gu_row_bytes == 770 and g.down_row_bytes == 360

    def test_q4_projection_file(self, tmp_path):
        truth = make_pack(tmp_path, dense_fmt=4)
        pack = Edge0Pack(truth.root)
        assert pack.latent_dense_type == gq.GGML_Q4_0
        assert pack.geometry.n_groups == 2

    def test_expert_offsets(self, tmp_path):
        truth = make_pack(tmp_path)
        pack = Edge0Pack(truth.root)
        assert pack.expert_offset(0, 0) == 0
        assert pack.expert_offset(1, 3) == (8 + 3) * truth.record_bytes
        with pytest.raises(IndexError):
            pack.expert_offset(2, 0)
        with pytest.raises(IndexError):
            pack.expert_offset(0, 8)


class TestContents:
    def test_labels(self, tmp_path):
        truth = make_pack(tmp_path)
        np.testing.assert_array_equal(Edge0Pack(truth.root).labels, truth.labels)

    def test_inverse_column_scale(self, tmp_path):
        truth = make_pack(tmp_path)
        pack = Edge0Pack(truth.root)
        np.testing.assert_allclose(
            pack.inverse_column_scale(1), 1.0 / truth.scales[1], rtol=1e-6
        )

    def test_non_positive_scale_is_identity(self, tmp_path):
        truth = make_pack(tmp_path)
        scale = truth.scales[0].copy()
        scale[:3] = [0.0, -1.0, 2.0]
        scale.tofile(truth.root / "scales" / "l0.f32")
        inverse = Edge0Pack(truth.root).inverse_column_scale(0)
        np.testing.assert_allclose(inverse[:3], [1.0, 1.0, 0.5])

    def test_missing_scales_mean_no_scaling(self, tmp_path):
        truth = make_pack(tmp_path, with_scales=False)
        np.testing.assert_array_equal(
            Edge0Pack(truth.root).inverse_column_scale(0), np.ones(256, np.float32)
        )

    @pytest.mark.parametrize("dense_fmt", [8, 4])
    def test_projection_blocks(self, tmp_path, dense_fmt):
        truth = make_pack(tmp_path, dense_fmt=dense_fmt)
        pack = Edge0Pack(truth.root)
        for layer in range(truth.layers):
            down = gq.dequantize(
                pack.down_projection_raw(layer), pack.latent_dense_type
            )
            up = gq.dequantize(pack.up_projection_raw(layer), pack.latent_dense_type)
            np.testing.assert_array_equal(
                down.reshape(2, 256, 64), truth.down_proj[layer]
            )
            np.testing.assert_array_equal(up.reshape(2, 64, 256), truth.up_proj[layer])

    def test_index_shapes(self, tmp_path):
        pack = Edge0Pack(make_pack(tmp_path).root)
        assert pack.index["blk.0.ffn_gate_inp.weight"] == (8, 64)
        assert pack.index["blk.0.hc_ffn_norm.weight"] == (64,)
        assert pack.index["blk.0.ffn_up_shexp.weight"] == (32, 64)

    def test_manifest_paths(self, tmp_path):
        truth = make_pack(tmp_path)
        pack = Edge0Pack(truth.root)
        root = truth.root.resolve()
        assert pack.skeleton_path == root / "model-skeleton.gguf"
        assert pack.ple_path == root / "ple-fp8.gguf"

    def test_unread_manifest_fields_cannot_reject_a_pack(self, tmp_path):
        truth = make_pack(tmp_path)
        manifest = json.loads((truth.root / "model.json").read_text())
        manifest["tokenizer"] = "../../somewhere/else"
        (truth.root / "model.json").write_text(json.dumps(manifest))
        Edge0Pack(truth.root)


class TestValidation:
    def test_truncated_experts_file(self, tmp_path):
        truth = make_pack(tmp_path)
        path = truth.root / "experts.bin"
        path.write_bytes(path.read_bytes()[:-1])
        with pytest.raises(Edge0PackError, match="experts.bin.*incomplete"):
            Edge0Pack(truth.root)

    def test_missing_labels(self, tmp_path):
        truth = make_pack(tmp_path)
        (truth.root / "labels.bin").unlink()
        with pytest.raises(Edge0PackError, match="labels.bin"):
            Edge0Pack(truth.root)

    def test_label_out_of_range(self, tmp_path):
        truth = make_pack(tmp_path)
        bad = truth.labels.copy()
        bad[0, 0] = 2
        bad.tofile(truth.root / "labels.bin")
        with pytest.raises(Edge0PackError, match="group id"):
            Edge0Pack(truth.root).labels

    def test_non_contiguous_layer_rows(self, tmp_path):
        truth = make_pack(tmp_path)
        path = truth.root / "native_experts.txt"
        lines = path.read_text().splitlines()
        lines[2] = f"1 11 20 7 {truth.record_bytes}"
        path.write_text("\n".join(lines) + "\n")
        with pytest.raises(Edge0PackError, match="contiguous"):
            Edge0Pack(truth.root)

    def test_manifest_cannot_point_outside_the_pack(self, tmp_path):
        truth = make_pack(tmp_path)
        manifest = json.loads((truth.root / "model.json").read_text())
        manifest["env"]["STRATA_LATENT_A"] = "../../etc/passwd"
        (truth.root / "model.json").write_text(json.dumps(manifest))
        with pytest.raises(Edge0PackError, match="escapes"):
            Edge0Pack(truth.root)

    def test_non_latent_pack_is_rejected(self, tmp_path):
        truth = make_pack(tmp_path)
        path = truth.root / "native_experts.txt"
        path.write_text(path.read_text().replace("expert_embd 256", "expert_embd 64"))
        with pytest.raises(Edge0PackError, match="not latent"):
            Edge0Pack(truth.root)


class TestProjectionChoice:
    def test_selects_the_sibling_projection_file(self, tmp_path):
        truth = make_pack(tmp_path, dense_fmt=8, extra_dense_fmts=(4,))
        pack = Edge0Pack(truth.root, projection_bits=4)
        assert pack.latent_dense_type == gq.GGML_Q4_0
        assert pack.projection_bits == 4
        assert pack.latent_dense_path.name == "latent_dense_q4.bin"
        assert pack.geometry.n_groups == 2
        down = gq.dequantize(pack.down_projection_raw(1), gq.GGML_Q4_0)
        np.testing.assert_array_equal(
            down.reshape(2, 256, 64), truth.projections[4][0][1]
        )

    def test_default_follows_the_manifest(self, tmp_path):
        truth = make_pack(tmp_path, dense_fmt=8, extra_dense_fmts=(4,))
        assert Edge0Pack(truth.root).latent_dense_type == gq.GGML_Q8_0
        assert (
            Edge0Pack(truth.root, projection_bits=8).latent_dense_type == gq.GGML_Q8_0
        )

    def test_missing_sibling_falls_back_to_the_manifest_file(self, tmp_path, caplog):
        truth = make_pack(tmp_path, dense_fmt=8)
        with caplog.at_level("WARNING"):
            pack = Edge0Pack(truth.root, projection_bits=4)
        assert pack.projection_bits == 8
        assert pack.latent_dense_type == gq.GGML_Q8_0
        assert "latent_dense_q4.bin" in caplog.text


class TestBlockAlignment:
    def test_expert_width_must_be_whole_blocks(self, tmp_path):
        truth = make_pack(tmp_path)
        path = truth.root / "native_experts.txt"
        path.write_text(path.read_text().replace("expert_embd 256", "expert_embd 300"))
        with pytest.raises(Edge0PackError, match="expert_embd 300 is not a multiple"):
            Edge0Pack(truth.root)

    def test_unsupported_expert_format_is_rejected_up_front(self, tmp_path):
        # e.g. an NVFP4 pack (ggml type 40), or any type the latent forward
        # has no kernels for.
        truth = make_pack(tmp_path)
        path = truth.root / "native_experts.txt"
        for types in ("40 40", "8 8"):
            path.write_text(
                "\n".join(
                    line
                    if line.startswith("#")
                    else line.replace(" 11 20 ", f" {types} ")
                    for line in (
                        "# strata native experts v6 (expert_embd 256, expert_ff 32)",
                        f"0 11 20 0 {truth.record_bytes}",
                        f"1 11 20 {8 * truth.record_bytes} {truth.record_bytes}",
                    )
                )
                + "\n"
            )
            with pytest.raises(Edge0PackError, match="unsupported expert format"):
                Edge0Pack(truth.root)
