"""Tests for olmlx.engine.edge0.convert + loader: pack -> served qwen4_exp."""

import json

import mlx.core as mx
import numpy as np
import pytest

from olmlx.engine.edge0 import convert, ple
from olmlx.engine.edge0.pack import Edge0PackError
from tests.edge0_factories import (
    PLE_ROW_WIDTH,
    PLE_ROWS,
    TINY_HIDDEN,
    TINY_LAYERS,
    make_full_pack,
    tiny_tokens,
)

_LM = "language_model.model."
# Tiny model: 2 key heads x 2 repeats. GGUF groups value heads by repeat
# (r * 2 + k); HF head h = k * 2 + r reads GGUF head (h % 2) * 2 + h // 2.
_HEADS = [0, 2, 1, 3]


def _bf16(values):
    return np.array(mx.array(values).astype(mx.bfloat16).astype(mx.float32))


def _load_weights(root):
    weights = {}
    for shard in sorted(root.glob("model-*.safetensors")):
        weights.update(mx.load(str(shard)))
    return weights


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    full = make_full_pack(tmp_path_factory.mktemp("edge0"))
    messages = []
    convert.convert_pack(full.root, progress=messages.append)
    return full, messages


class TestMapping:
    def test_layer_tensor(self):
        assert convert.mlx_key_for("blk.7.attn_qkv.weight") == (
            _LM + "layers.7.linear_attn.in_proj_qkv.weight",
            "linear",
        )

    def test_global_tensor(self):
        assert convert.mlx_key_for("output.weight") == (
            "language_model.lm_head.weight",
            "linear",
        )

    def test_indexer_halves_and_expert_leftovers_are_skipped(self):
        assert convert.mlx_key_for("blk.3.indexer.q_proj.weight") is None
        assert convert.mlx_key_for("blk.0.ffn_gate_exps.scale") is None

    def test_unknown_tensor_is_an_error(self):
        with pytest.raises(Edge0PackError, match="no known mapping"):
            convert.mlx_key_for("blk.0.mystery.weight")


class TestConvertedFiles:
    def test_marks_the_directory_converted(self, converted):
        full, messages = converted
        assert convert.is_converted(full.root)
        assert messages[-1] == "done"

    def test_config(self, converted):
        full, _ = converted
        config = json.loads((full.root / "config.json").read_text())
        tokens, _, _ = tiny_tokens()
        assert config["model_type"] == "qwen4_exp"
        text = config["text_config"]
        assert text["hidden_size"] == TINY_HIDDEN
        assert text["num_hidden_layers"] == TINY_LAYERS
        assert text["layer_types"] == ["linear_attention"] * 3 + ["full_attention"]
        assert text["ple_layer_ids"] == [2]
        assert text["ple_embed_dim"] == TINY_HIDDEN
        assert text["linear_value_head_dim"] == 32
        assert text["rope_parameters"]["mrope_section"] == [2, 1, 1]
        assert text["rope_parameters"]["partial_rotary_factor"] == 0.25
        assert text["ple_storage"]["manifest"] == ple.MANIFEST_NAME
        assert text["eos_token_id"] == tokens.index("<|endoftext|>")
        assert config["eos_token_id"] == [
            tokens.index("<|im_end|>"),
            tokens.index("<|endoftext|>"),
        ]
        assert config["image_token_id"] == tokens.index("<|image_pad|>")
        assert config["vision_config"]["depth"] == 1
        assert config["vision_config"]["hidden_size"] == 16
        assert config["vision_config"]["num_position_embeddings"] == 16
        latent = config["edge0"]["latent"]
        assert (latent["latent"], latent["n_groups"], latent["projection_bits"]) == (
            256,
            2,
            8,
        )
        assert config["edge0"]["has_vision"] is True

    def test_ple_manifest_points_at_the_table_in_place(self, converted):
        full, _ = converted
        manifest = json.loads((full.root / ple.MANIFEST_NAME).read_text())
        assert manifest["data_file"] == "ple-fp8.gguf"
        assert (manifest["row_count"], manifest["row_width"]) == (
            PLE_ROWS,
            PLE_ROW_WIDTH,
        )

    def test_q8_matrix_is_repacked_losslessly(self, converted):
        full, _ = converted
        weights = _load_weights(full.root)
        prefix = _LM + "layers.0.linear_attn.in_proj_qkv"
        assert weights[prefix + ".weight"].dtype == mx.uint32
        assert weights[prefix + ".scales"].dtype == mx.bfloat16
        scales = weights[prefix + ".scales"].astype(mx.float32)
        # Bias is derived from the rounded scale, so there is no offset:
        # dequantized weights are the originals times one per-block factor.
        np.testing.assert_array_equal(
            np.array(weights[prefix + ".biases"].astype(mx.float32)),
            np.array(scales) * -128.0,
        )
        dense = mx.dequantize(
            weights[prefix + ".weight"],
            scales,
            scales * -128.0,
            group_size=32,
            bits=8,
        )
        # Rows 0..127 are the query/key projections, which keep their order.
        expected = full.skeleton["blk.0.attn_qkv.weight"][:128]
        np.testing.assert_allclose(
            np.array(dense)[:128], expected, rtol=5e-3, atol=1e-7
        )

    def test_token_embedding_and_head_are_quantized(self, converted):
        full, _ = converted
        weights = _load_weights(full.root)
        assert _LM + "embed_tokens.scales" in weights
        assert "language_model.lm_head.scales" in weights

    def test_norm_weights_are_recentred(self, converted):
        full, _ = converted
        weights = _load_weights(full.root)
        got = weights[_LM + "layers.0.attn_hyper_connection.hc_norm.weight"]
        assert got.dtype == mx.bfloat16
        np.testing.assert_array_equal(
            np.array(got.astype(mx.float32)),
            _bf16(full.skeleton["blk.0.hc_attn_norm.weight"] - 1.0),
        )

    def test_gated_norm_weight_is_copied(self, converted):
        full, _ = converted
        weights = _load_weights(full.root)
        np.testing.assert_array_equal(
            np.array(
                weights[_LM + "layers.0.linear_attn.norm.weight"].astype(mx.float32)
            ),
            _bf16(full.skeleton["blk.0.ssm_norm.weight"]),
        )

    def test_conv_gets_a_trailing_axis(self, converted):
        full, _ = converted
        weights = _load_weights(full.root)
        got = weights[_LM + "layers.0.linear_attn.conv1d.weight"]
        assert got.shape == (256, 4, 1)
        source = full.skeleton["blk.0.ssm_conv1d.weight"]
        expected = np.concatenate(
            [source[:128], source[128:].reshape(4, 32, 4)[_HEADS].reshape(128, 4)]
        )
        np.testing.assert_array_equal(
            np.array(got.astype(mx.float32))[..., 0], _bf16(expected)
        )

    def test_gdn_per_head_vectors_follow_hf_head_order(self, converted):
        full, _ = converted
        weights = _load_weights(full.root)
        prefix = _LM + "layers.0.linear_attn."

        def got(name):
            return np.array(weights[prefix + name].astype(mx.float32))

        np.testing.assert_array_equal(
            got("dt_bias"), _bf16(full.skeleton["blk.0.ssm_dt.bias"][_HEADS])
        )
        np.testing.assert_array_equal(
            got("in_proj_a.weight"),
            _bf16(full.skeleton["blk.0.ssm_alpha.weight"][_HEADS]),
        )
        np.testing.assert_array_equal(
            got("in_proj_b.weight"),
            _bf16(full.skeleton["blk.0.ssm_beta.weight"][_HEADS]),
        )

    def test_gdn_decay_is_stored_as_log(self, converted):
        full, _ = converted
        weights = _load_weights(full.root)
        got = np.array(weights[_LM + "layers.0.linear_attn.A_log"].astype(mx.float32))
        np.testing.assert_array_equal(
            got, _bf16(np.log(-full.skeleton["blk.0.ssm_a"][_HEADS]))
        )

    def test_gdn_quantized_matrices_follow_hf_head_order(self, converted):
        full, _ = converted
        weights = _load_weights(full.root)

        def dense(name):
            prefix = _LM + "layers.0.linear_attn." + name
            scales = weights[prefix + ".scales"].astype(mx.float32)
            return np.array(
                mx.dequantize(
                    weights[prefix + ".weight"],
                    scales,
                    scales * -128.0,
                    group_size=32,
                    bits=8,
                )
            )

        qkv = full.skeleton["blk.0.attn_qkv.weight"]
        expected = np.concatenate(
            [qkv[:128], qkv[128:].reshape(4, 32, 64)[_HEADS].reshape(128, 64)]
        )
        np.testing.assert_allclose(dense("in_proj_qkv"), expected, rtol=5e-3, atol=1e-7)
        gate = full.skeleton["blk.0.attn_gate.weight"]
        np.testing.assert_allclose(
            dense("in_proj_z"),
            gate.reshape(4, 32, 64)[_HEADS].reshape(128, 64),
            rtol=5e-3,
            atol=1e-7,
        )
        out = full.skeleton["blk.0.ssm_out.weight"]
        np.testing.assert_allclose(
            dense("out_proj"),
            out.reshape(64, 4, 32)[:, _HEADS].reshape(64, 128),
            rtol=5e-3,
            atol=1e-7,
        )

    def test_indexer_projection_is_query_then_key(self, converted):
        full, _ = converted
        weights = _load_weights(full.root)
        got = np.array(
            weights[_LM + "layers.3.self_attn.indexer.index_qk_proj.weight"].astype(
                mx.float32
            )
        )
        np.testing.assert_array_equal(
            got[:16], _bf16(full.skeleton["blk.3.indexer.q_proj.weight"])
        )
        np.testing.assert_array_equal(
            got[16:], _bf16(full.skeleton["blk.3.indexer.k_proj.weight"])
        )

    def test_vision_patch_embedding_layout(self, converted):
        full, _ = converted
        weights = _load_weights(full.root)
        got = np.array(
            weights["vision_tower.patch_embed.proj.weight"].astype(mx.float32)
        )
        assert got.shape == (16, 2, 4, 4, 3)
        for slice_index, name in enumerate(
            ("v.patch_embd.weight", "v.patch_embd.weight.1")
        ):
            np.testing.assert_array_equal(
                got[:, slice_index], _bf16(full.vision[name].transpose(0, 2, 3, 1))
            )
        np.testing.assert_array_equal(
            np.array(
                weights["vision_tower.merger.linear_fc2.weight"].astype(mx.float32)
            ),
            _bf16(full.vision["mm.2.weight"]),
        )

    def test_latent_projection_files(self, converted):
        full, _ = converted
        tensors = mx.load(
            str(full.root / "flash_moe" / "latent" / "layer_002.safetensors")
        )
        assert tensors["down.scales"].dtype == mx.float16
        np.testing.assert_array_equal(
            np.array(tensors["labels"]), full.latent.labels[2]
        )
        np.testing.assert_allclose(
            np.array(tensors["inverse_scale"]), 1.0 / full.latent.scales[2], rtol=1e-6
        )
        layout = json.loads(
            (full.root / "flash_moe" / "flash_moe_layout.json").read_text()
        )
        assert layout["format"] == "edge0_latent"

    def test_tokenizer_round_trip(self, converted):
        from transformers import AutoTokenizer

        full, _ = converted
        tokens, _, _ = tiny_tokens()
        tokenizer = AutoTokenizer.from_pretrained(str(full.root))
        ids = tokenizer.encode("the cat<|im_end|>", add_special_tokens=False)
        # Merges apply: "the" -> t + he, and " the" is the single token Ġthe.
        assert ids[:2] == [tokens.index("t"), tokens.index("he")]
        assert tokenizer.encode(" the", add_special_tokens=False) == [
            tokens.index("Ġthe")
        ]
        assert ids[-1] == tokens.index("<|im_end|>")
        assert tokenizer.decode(ids) == "the cat<|im_end|>"
        assert tokenizer.decode(ids, skip_special_tokens=True) == "the cat"
        # llama.cpp calls it "control", HF does not mark it special.
        fim = tokenizer.encode("<|fim_prefix|>", add_special_tokens=False)
        assert fim == [tokens.index("<|fim_prefix|>")]
        assert tokenizer.decode(fim, skip_special_tokens=True) == "<|fim_prefix|>"
        assert tokenizer.eos_token == "<|im_end|>"


class TestReconversion:
    def test_stale_version_is_not_converted(self, tmp_path):
        full = make_full_pack(tmp_path, with_vision=False)
        convert.convert_pack(full.root)
        config = json.loads((full.root / "config.json").read_text())
        assert config["edge0"]["has_vision"] is False
        config["edge0"]["converter_version"] = convert.CONVERTER_VERSION - 1
        (full.root / "config.json").write_text(json.dumps(config))
        assert not convert.is_converted(full.root)
        assert convert.read_edge0_config(full.root) is not None

    def test_plain_model_dir_is_not_edge0(self, tmp_path):
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
        assert convert.read_edge0_config(tmp_path) is None
        assert not convert.is_converted(tmp_path)

    def test_missing_skeleton(self, tmp_path):
        full = make_full_pack(tmp_path, with_vision=False)
        (full.root / "model-skeleton.gguf").unlink()
        with pytest.raises(Edge0PackError, match="model-skeleton.gguf"):
            convert.convert_pack(full.root)
        assert not (full.root / "config.json").exists()

    def test_hash_mismatch_is_refused(self, tmp_path, monkeypatch):
        from mlx_vlm.models.qwen4_exp import language

        full = make_full_pack(tmp_path, with_vision=False)
        monkeypatch.setattr(
            language, "_build_layer_multipliers", lambda *a, **k: mx.array([1, 2, 3])
        )
        with pytest.raises(Edge0PackError, match="n-gram hashing"):
            convert.convert_pack(full.root)
        assert not (full.root / "config.json").exists()


class TestProjectionBits:
    def test_conversion_records_and_checks_the_projection_bits(self, tmp_path):
        full = make_full_pack(tmp_path, with_vision=False)
        convert.convert_pack(full.root, projection_bits=4)
        assert convert.is_converted(full.root)
        assert convert.is_converted(full.root, projection_bits=4)
        assert not convert.is_converted(full.root, projection_bits=8)
        tensors = mx.load(
            str(full.root / "flash_moe" / "latent" / "layer_000.safetensors")
        )
        down = mx.dequantize(
            tensors["down.weight"],
            tensors["down.scales"].astype(mx.float32),
            tensors["down.biases"].astype(mx.float32),
            group_size=32,
            bits=4,
        )
        np.testing.assert_allclose(
            np.array(down).reshape(2, 256, 64),
            full.latent.projections[4][0][0],
            rtol=1e-5,
            atol=1e-7,
        )

    def test_pack_with_only_one_projection_file_does_not_look_stale(self, tmp_path):
        # Asking for 4-bit on a pack that ships only 8-bit converts with 8-bit
        # and must then count as converted *for that request*, or every load
        # would convert again.
        full = make_full_pack(tmp_path, with_vision=False, with_draft_head=False)
        (full.root / "latent_dense_q4.bin").unlink()
        convert.convert_pack(full.root, projection_bits=4)
        section = convert.read_edge0_config(full.root)
        assert section["latent"]["projection_bits"] == 8
        assert convert.is_converted(full.root, projection_bits=4)

    def test_failed_validation_keeps_the_previous_conversion(
        self, tmp_path, monkeypatch
    ):
        from mlx_vlm.models.qwen4_exp import language

        full = make_full_pack(tmp_path, with_vision=False, with_draft_head=False)
        convert.convert_pack(full.root)
        shards = sorted(p.name for p in full.root.glob("model-*.safetensors"))
        monkeypatch.setattr(
            language, "_build_layer_multipliers", lambda *a, **k: mx.array([1, 2, 3])
        )
        with pytest.raises(Edge0PackError, match="n-gram hashing"):
            convert.convert_pack(full.root)
        assert convert.is_converted(full.root)
        assert sorted(p.name for p in full.root.glob("model-*.safetensors")) == shards
