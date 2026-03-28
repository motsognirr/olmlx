"""Tests for olmlx.engine.flash.moe_bundler — MoE expert weight bundling."""

import json
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MOE_HEADER_MAGIC = 0x464C4D45  # "FLME"


def _make_synthetic_moe_weights(
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    num_moe_layers: int,
    num_dense_layers: int,
    tmp_path: Path,
    quantized: bool = False,
) -> Path:
    """Create synthetic safetensors with stacked MoE expert weights.

    Mimics the sanitized mlx-lm format where experts are stacked:
      model.layers.{l}.mlp.switch_mlp.gate_proj.weight  shape (num_experts, inter, hidden)
      model.layers.{l}.mlp.switch_mlp.up_proj.weight    shape (num_experts, inter, hidden)
      model.layers.{l}.mlp.switch_mlp.down_proj.weight  shape (num_experts, hidden, inter)

    Also includes gate (router) and shared_experts weights that should NOT be bundled.
    Dense layers (first num_dense_layers) have standard MLP weights.
    """
    from safetensors.numpy import save_file

    rng = np.random.RandomState(42)
    tensors = {}

    # Dense layers (should be skipped by MoE bundler)
    for layer in range(num_dense_layers):
        prefix = f"model.layers.{layer}.mlp"
        tensors[f"{prefix}.gate_proj.weight"] = rng.randn(
            intermediate_size, hidden_size
        ).astype(np.float16)
        tensors[f"{prefix}.up_proj.weight"] = rng.randn(
            intermediate_size, hidden_size
        ).astype(np.float16)
        tensors[f"{prefix}.down_proj.weight"] = rng.randn(
            hidden_size, intermediate_size
        ).astype(np.float16)

    # MoE layers
    for layer in range(num_dense_layers, num_dense_layers + num_moe_layers):
        prefix = f"model.layers.{layer}.mlp"

        # Router (gate) — should NOT be bundled
        tensors[f"{prefix}.gate.weight"] = rng.randn(num_experts, hidden_size).astype(
            np.float16
        )
        tensors[f"{prefix}.gate.e_score_correction_bias"] = rng.randn(
            num_experts
        ).astype(np.float16)

        # Shared expert — should NOT be bundled
        tensors[f"{prefix}.shared_experts.gate_proj.weight"] = rng.randn(
            intermediate_size, hidden_size
        ).astype(np.float16)
        tensors[f"{prefix}.shared_experts.up_proj.weight"] = rng.randn(
            intermediate_size, hidden_size
        ).astype(np.float16)
        tensors[f"{prefix}.shared_experts.down_proj.weight"] = rng.randn(
            hidden_size, intermediate_size
        ).astype(np.float16)

        # Stacked expert weights (the target for bundling)
        if not quantized:
            tensors[f"{prefix}.switch_mlp.gate_proj.weight"] = rng.randn(
                num_experts, intermediate_size, hidden_size
            ).astype(np.float16)
            tensors[f"{prefix}.switch_mlp.up_proj.weight"] = rng.randn(
                num_experts, intermediate_size, hidden_size
            ).astype(np.float16)
            tensors[f"{prefix}.switch_mlp.down_proj.weight"] = rng.randn(
                num_experts, hidden_size, intermediate_size
            ).astype(np.float16)
        else:
            # Quantized: packed uint32 weights + float16 scales + float16 biases
            group_size = 32
            bits = 4
            # Packed weight shape: (num_experts, out_dim, in_dim * bits / 32)
            gate_packed_dim = hidden_size * bits // 32
            down_packed_dim = intermediate_size * bits // 32

            tensors[f"{prefix}.switch_mlp.gate_proj.weight"] = rng.randint(
                0, 2**31, (num_experts, intermediate_size, gate_packed_dim)
            ).astype(np.uint32)
            tensors[f"{prefix}.switch_mlp.gate_proj.scales"] = rng.randn(
                num_experts, intermediate_size, hidden_size // group_size
            ).astype(np.float16)
            tensors[f"{prefix}.switch_mlp.gate_proj.biases"] = rng.randn(
                num_experts, intermediate_size, hidden_size // group_size
            ).astype(np.float16)

            tensors[f"{prefix}.switch_mlp.up_proj.weight"] = rng.randint(
                0, 2**31, (num_experts, intermediate_size, gate_packed_dim)
            ).astype(np.uint32)
            tensors[f"{prefix}.switch_mlp.up_proj.scales"] = rng.randn(
                num_experts, intermediate_size, hidden_size // group_size
            ).astype(np.float16)
            tensors[f"{prefix}.switch_mlp.up_proj.biases"] = rng.randn(
                num_experts, intermediate_size, hidden_size // group_size
            ).astype(np.float16)

            tensors[f"{prefix}.switch_mlp.down_proj.weight"] = rng.randint(
                0, 2**31, (num_experts, hidden_size, down_packed_dim)
            ).astype(np.uint32)
            tensors[f"{prefix}.switch_mlp.down_proj.scales"] = rng.randn(
                num_experts, hidden_size, intermediate_size // group_size
            ).astype(np.float16)
            tensors[f"{prefix}.switch_mlp.down_proj.biases"] = rng.randn(
                num_experts, hidden_size, intermediate_size // group_size
            ).astype(np.float16)

    # Non-MLP weights (should be ignored)
    tensors["model.embed_tokens.weight"] = rng.randn(100, hidden_size).astype(
        np.float16
    )

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    save_file(tensors, str(model_dir / "model.safetensors"))

    # Write config.json with MoE architecture info
    config = {
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,  # dense MLP intermediate
        "moe_intermediate_size": intermediate_size,  # per-expert intermediate
        "num_hidden_layers": num_dense_layers + num_moe_layers,
        "n_routed_experts": num_experts,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "first_k_dense_replace": num_dense_layers,
        "moe_layer_freq": 1,
    }
    if quantized:
        config["quantization"] = {"bits": 4, "group_size": 32}
    (model_dir / "config.json").write_text(json.dumps(config))

    return model_dir


# ---------------------------------------------------------------------------
# Bundler tests
# ---------------------------------------------------------------------------


class TestBundleMoeExperts:
    def test_bundle_creates_layer_files(self, tmp_path):
        """Each MoE layer should produce a .flashexperts file."""
        hidden, inter, experts = 64, 32, 4
        num_moe, num_dense = 2, 1
        model_dir = _make_synthetic_moe_weights(
            hidden, inter, experts, num_moe, num_dense, tmp_path
        )
        output_dir = tmp_path / "flash_moe"

        from olmlx.engine.flash.moe_bundler import bundle_moe_experts

        layouts = bundle_moe_experts(model_dir, output_dir)

        assert len(layouts) == num_moe
        # MoE layers are 1 and 2 (dense layer 0 is skipped)
        for layer_idx in [1, 2]:
            assert (output_dir / f"layer_{layer_idx:02d}.flashexperts").exists()
            assert layer_idx in layouts

    def test_bundle_skips_dense_layers(self, tmp_path):
        """Dense layers should not produce .flashexperts files."""
        hidden, inter, experts = 64, 32, 4
        num_moe, num_dense = 2, 1
        model_dir = _make_synthetic_moe_weights(
            hidden, inter, experts, num_moe, num_dense, tmp_path
        )
        output_dir = tmp_path / "flash_moe"

        from olmlx.engine.flash.moe_bundler import bundle_moe_experts

        layouts = bundle_moe_experts(model_dir, output_dir)

        # Layer 0 is dense — should not be bundled
        assert 0 not in layouts
        assert not (output_dir / "layer_00.flashexperts").exists()

    def test_bundle_header_is_correct(self, tmp_path):
        """Header should contain correct magic, version, and dimensions."""
        hidden, inter, experts = 64, 32, 4
        model_dir = _make_synthetic_moe_weights(hidden, inter, experts, 1, 0, tmp_path)
        output_dir = tmp_path / "flash_moe"

        from olmlx.engine.flash.moe_bundler import (
            MOE_HEADER_SIZE,
            bundle_moe_experts,
            parse_moe_header,
        )

        bundle_moe_experts(model_dir, output_dir)

        fp = output_dir / "layer_00.flashexperts"
        with open(fp, "rb") as f:
            header = parse_moe_header(f.read(MOE_HEADER_SIZE))

        assert header["magic"] == MOE_HEADER_MAGIC
        assert header["version"] == 1
        assert header["num_experts"] == experts
        assert header["hidden_size"] == hidden
        assert header["intermediate_size"] == inter
        assert header["is_quantized"] is False

    def test_bundle_preserves_expert_data(self, tmp_path):
        """Bundled expert data must match the original safetensors weights."""
        hidden, inter, experts = 64, 32, 8
        model_dir = _make_synthetic_moe_weights(hidden, inter, experts, 1, 0, tmp_path)
        output_dir = tmp_path / "flash_moe"

        from safetensors.numpy import load_file

        from olmlx.engine.flash.moe_bundler import bundle_moe_experts

        original = load_file(str(model_dir / "model.safetensors"))
        gate_w = original["model.layers.0.mlp.switch_mlp.gate_proj.weight"]
        up_w = original["model.layers.0.mlp.switch_mlp.up_proj.weight"]
        down_w = original["model.layers.0.mlp.switch_mlp.down_proj.weight"]

        layouts = bundle_moe_experts(model_dir, output_dir)
        layout = layouts[0]

        # Read expert 3's data manually
        expert_idx = 3
        expert_offset = int(layout.offsets[expert_idx])

        with open(layout.file_path, "rb") as f:
            f.seek(expert_offset)
            raw = f.read(layout.expert_byte_size)

        # For non-quantized: gate_proj(inter, hidden) + up_proj(inter, hidden) + down_proj(hidden, inter)
        gate_size = inter * hidden * 2  # float16
        up_size = inter * hidden * 2
        down_size = hidden * inter * 2

        gate_read = np.frombuffer(raw[:gate_size], dtype=np.float16).reshape(
            inter, hidden
        )
        up_read = np.frombuffer(
            raw[gate_size : gate_size + up_size], dtype=np.float16
        ).reshape(inter, hidden)
        down_read = np.frombuffer(
            raw[gate_size + up_size : gate_size + up_size + down_size], dtype=np.float16
        ).reshape(hidden, inter)

        np.testing.assert_array_equal(gate_read, gate_w[expert_idx])
        np.testing.assert_array_equal(up_read, up_w[expert_idx])
        np.testing.assert_array_equal(down_read, down_w[expert_idx])

    def test_bundle_preserves_quantized_data(self, tmp_path):
        """Quantized model: packed weights + scales + biases should round-trip."""
        hidden, inter, experts = 64, 32, 4
        model_dir = _make_synthetic_moe_weights(
            hidden, inter, experts, 1, 0, tmp_path, quantized=True
        )
        output_dir = tmp_path / "flash_moe"

        from safetensors.numpy import load_file

        from olmlx.engine.flash.moe_bundler import (
            bundle_moe_experts,
            parse_moe_header,
            MOE_HEADER_SIZE,
        )

        original = load_file(str(model_dir / "model.safetensors"))

        layouts = bundle_moe_experts(model_dir, output_dir)
        layout = layouts[0]

        # Verify header reports quantized
        with open(layout.file_path, "rb") as f:
            header = parse_moe_header(f.read(MOE_HEADER_SIZE))
        assert header["is_quantized"] is True
        assert header["bits"] == 4
        assert header["group_size"] == 32

        # Verify expert 1's gate_proj packed weight matches original
        expert_idx = 1
        gate_packed_orig = original["model.layers.0.mlp.switch_mlp.gate_proj.weight"][
            expert_idx
        ]
        gate_scales_orig = original["model.layers.0.mlp.switch_mlp.gate_proj.scales"][
            expert_idx
        ]
        gate_biases_orig = original["model.layers.0.mlp.switch_mlp.gate_proj.biases"][
            expert_idx
        ]

        # Read expert from bundle
        expert_offset = int(layout.offsets[expert_idx])
        with open(layout.file_path, "rb") as f:
            f.seek(expert_offset)
            raw = f.read(layout.expert_byte_size)

        # First component: gate_proj.weight (packed uint32)
        gate_packed_size = gate_packed_orig.nbytes
        gate_packed_read = np.frombuffer(
            raw[:gate_packed_size], dtype=np.uint32
        ).reshape(gate_packed_orig.shape)
        np.testing.assert_array_equal(gate_packed_read, gate_packed_orig)

        # Then gate_proj.scales
        gate_scales_size = gate_scales_orig.nbytes
        gate_scales_read = np.frombuffer(
            raw[gate_packed_size : gate_packed_size + gate_scales_size],
            dtype=np.float16,
        ).reshape(gate_scales_orig.shape)
        np.testing.assert_array_equal(gate_scales_read, gate_scales_orig)

        # Then gate_proj.biases
        gate_biases_size = gate_biases_orig.nbytes
        offset = gate_packed_size + gate_scales_size
        gate_biases_read = np.frombuffer(
            raw[offset : offset + gate_biases_size],
            dtype=np.float16,
        ).reshape(gate_biases_orig.shape)
        np.testing.assert_array_equal(gate_biases_read, gate_biases_orig)

    def test_bundle_offset_table_sequential(self, tmp_path):
        """Offsets should be sequential, each expert_byte_size apart."""
        hidden, inter, experts = 64, 32, 8
        model_dir = _make_synthetic_moe_weights(hidden, inter, experts, 1, 0, tmp_path)
        output_dir = tmp_path / "flash_moe"

        from olmlx.engine.flash.moe_bundler import bundle_moe_experts

        layouts = bundle_moe_experts(model_dir, output_dir)
        layout = layouts[0]

        for i in range(1, experts):
            assert layout.offsets[i] == layout.offsets[i - 1] + layout.expert_byte_size

    def test_bundle_writes_layout_json(self, tmp_path):
        """flash_moe_layout.json should contain correct metadata."""
        hidden, inter, experts = 64, 32, 4
        num_moe, num_dense = 2, 1
        model_dir = _make_synthetic_moe_weights(
            hidden, inter, experts, num_moe, num_dense, tmp_path
        )
        output_dir = tmp_path / "flash_moe"

        from olmlx.engine.flash.moe_bundler import bundle_moe_experts

        bundle_moe_experts(model_dir, output_dir)

        config_path = output_dir / "flash_moe_layout.json"
        assert config_path.exists()
        config = json.loads(config_path.read_text())
        assert config["num_moe_layers"] == num_moe
        assert config["num_experts"] == experts
        assert config["hidden_size"] == hidden
        assert config["intermediate_size"] == inter
        # Should list only MoE layer indices
        assert sorted(config["layers"].keys()) == ["1", "2"]

    def test_bundle_handles_sharded_safetensors(self, tmp_path):
        """Should handle models split across multiple safetensors shards."""
        from safetensors.numpy import save_file

        hidden, inter, experts = 64, 32, 4
        rng = np.random.RandomState(99)

        model_dir = tmp_path / "sharded_model"
        model_dir.mkdir()

        # Shard 1: gate_proj weights
        shard1 = {
            "model.layers.0.mlp.switch_mlp.gate_proj.weight": rng.randn(
                experts, inter, hidden
            ).astype(np.float16),
            "model.layers.0.mlp.gate.weight": rng.randn(experts, hidden).astype(
                np.float16
            ),
        }
        save_file(shard1, str(model_dir / "model-00001-of-00002.safetensors"))

        # Shard 2: up_proj and down_proj weights
        shard2 = {
            "model.layers.0.mlp.switch_mlp.up_proj.weight": rng.randn(
                experts, inter, hidden
            ).astype(np.float16),
            "model.layers.0.mlp.switch_mlp.down_proj.weight": rng.randn(
                experts, hidden, inter
            ).astype(np.float16),
        }
        save_file(shard2, str(model_dir / "model-00002-of-00002.safetensors"))

        # Write index
        weight_map = {
            "model.layers.0.mlp.switch_mlp.gate_proj.weight": "model-00001-of-00002.safetensors",
            "model.layers.0.mlp.gate.weight": "model-00001-of-00002.safetensors",
            "model.layers.0.mlp.switch_mlp.up_proj.weight": "model-00002-of-00002.safetensors",
            "model.layers.0.mlp.switch_mlp.down_proj.weight": "model-00002-of-00002.safetensors",
        }
        index = {"weight_map": weight_map}
        (model_dir / "model.safetensors.index.json").write_text(json.dumps(index))

        # Write config
        config = {
            "hidden_size": hidden,
            "moe_intermediate_size": inter,
            "num_hidden_layers": 1,
            "n_routed_experts": experts,
            "first_k_dense_replace": 0,
            "moe_layer_freq": 1,
        }
        (model_dir / "config.json").write_text(json.dumps(config))

        output_dir = tmp_path / "flash_moe"

        from olmlx.engine.flash.moe_bundler import bundle_moe_experts

        layouts = bundle_moe_experts(model_dir, output_dir)

        assert 0 in layouts
        assert layouts[0].num_experts == experts


def _make_synthetic_minimax_moe_weights(
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    num_moe_layers: int,
    tmp_path: Path,
    quantized: bool = False,
) -> Path:
    """Create synthetic safetensors with MiniMax-style block_sparse_moe naming.

    MiniMax uses block_sparse_moe instead of mlp as the MoE module name:
      model.layers.{l}.block_sparse_moe.switch_mlp.gate_proj.weight
    """
    from safetensors.numpy import save_file

    rng = np.random.RandomState(42)
    tensors = {}

    for layer in range(num_moe_layers):
        prefix = f"model.layers.{layer}.block_sparse_moe"

        # Router weights
        tensors[f"{prefix}.gate.weight"] = rng.randn(num_experts, hidden_size).astype(
            np.float16
        )

        if not quantized:
            tensors[f"{prefix}.switch_mlp.gate_proj.weight"] = rng.randn(
                num_experts, intermediate_size, hidden_size
            ).astype(np.float16)
            tensors[f"{prefix}.switch_mlp.up_proj.weight"] = rng.randn(
                num_experts, intermediate_size, hidden_size
            ).astype(np.float16)
            tensors[f"{prefix}.switch_mlp.down_proj.weight"] = rng.randn(
                num_experts, hidden_size, intermediate_size
            ).astype(np.float16)
        else:
            group_size = 32
            bits = 4
            gate_packed_dim = hidden_size * bits // 32
            down_packed_dim = intermediate_size * bits // 32

            for proj, out_dim, packed_dim, in_dim in [
                ("gate_proj", intermediate_size, gate_packed_dim, hidden_size),
                ("up_proj", intermediate_size, gate_packed_dim, hidden_size),
                ("down_proj", hidden_size, down_packed_dim, intermediate_size),
            ]:
                tensors[f"{prefix}.switch_mlp.{proj}.weight"] = rng.randint(
                    0, 2**31, (num_experts, out_dim, packed_dim)
                ).astype(np.uint32)
                tensors[f"{prefix}.switch_mlp.{proj}.scales"] = rng.randn(
                    num_experts, out_dim, in_dim // group_size
                ).astype(np.float16)
                tensors[f"{prefix}.switch_mlp.{proj}.biases"] = rng.randn(
                    num_experts, out_dim, in_dim // group_size
                ).astype(np.float16)

    # Non-MLP weights
    tensors["model.embed_tokens.weight"] = rng.randn(100, hidden_size).astype(
        np.float16
    )

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    save_file(tensors, str(model_dir / "model.safetensors"))

    # MiniMax config: all layers are MoE, no moe_layer_freq
    config = {
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_hidden_layers": num_moe_layers,
        "num_local_experts": num_experts,
        "num_experts_per_tok": 2,
        "model_type": "minimax_m2",
    }
    if quantized:
        config["quantization"] = {"bits": 4, "group_size": 32}
    (model_dir / "config.json").write_text(json.dumps(config))

    return model_dir


class TestBundleMiniMaxMoeExperts:
    """Test bundler with MiniMax-style block_sparse_moe weight naming."""

    def test_detect_block_sparse_moe_prefix(self, tmp_path):
        """Should detect block_sparse_moe.switch_mlp as the expert prefix."""
        hidden, inter, experts = 64, 32, 4
        model_dir = _make_synthetic_minimax_moe_weights(
            hidden, inter, experts, 1, tmp_path
        )

        from olmlx.engine.flash.moe_bundler import bundle_moe_experts

        layouts = bundle_moe_experts(model_dir, tmp_path / "flash_moe")

        assert 0 in layouts
        assert layouts[0].num_experts == experts

    def test_bundle_preserves_minimax_expert_data(self, tmp_path):
        """Bundled data should match original block_sparse_moe weights."""
        hidden, inter, experts = 64, 32, 8
        model_dir = _make_synthetic_minimax_moe_weights(
            hidden, inter, experts, 1, tmp_path
        )
        output_dir = tmp_path / "flash_moe"

        from safetensors.numpy import load_file

        from olmlx.engine.flash.moe_bundler import bundle_moe_experts

        original = load_file(str(model_dir / "model.safetensors"))
        gate_w = original["model.layers.0.block_sparse_moe.switch_mlp.gate_proj.weight"]

        layouts = bundle_moe_experts(model_dir, output_dir)
        layout = layouts[0]

        expert_idx = 3
        expert_offset = int(layout.offsets[expert_idx])
        with open(layout.file_path, "rb") as f:
            f.seek(expert_offset)
            raw = f.read(inter * hidden * 2)  # gate_proj float16

        gate_read = np.frombuffer(raw, dtype=np.float16).reshape(inter, hidden)
        np.testing.assert_array_equal(gate_read, gate_w[expert_idx])

    def test_bundle_quantized_minimax(self, tmp_path):
        """Quantized MiniMax model should bundle correctly."""
        hidden, inter, experts = 64, 32, 4
        model_dir = _make_synthetic_minimax_moe_weights(
            hidden, inter, experts, 1, tmp_path, quantized=True
        )
        output_dir = tmp_path / "flash_moe"

        from olmlx.engine.flash.moe_bundler import (
            MOE_HEADER_SIZE,
            bundle_moe_experts,
            parse_moe_header,
        )

        layouts = bundle_moe_experts(model_dir, output_dir)
        layout = layouts[0]

        with open(layout.file_path, "rb") as f:
            header = parse_moe_header(f.read(MOE_HEADER_SIZE))
        assert header["is_quantized"] is True
        assert header["bits"] == 4

    def test_bundle_sharded_minimax(self, tmp_path):
        """Should handle sharded MiniMax models via safetensors index."""
        from safetensors.numpy import save_file

        hidden, inter, experts = 64, 32, 4
        rng = np.random.RandomState(99)

        model_dir = tmp_path / "sharded_model"
        model_dir.mkdir()

        # Shard 1: gate_proj
        shard1 = {
            "model.layers.0.block_sparse_moe.switch_mlp.gate_proj.weight": rng.randn(
                experts, inter, hidden
            ).astype(np.float16),
            "model.layers.0.block_sparse_moe.gate.weight": rng.randn(
                experts, hidden
            ).astype(np.float16),
        }
        save_file(shard1, str(model_dir / "model-00001-of-00002.safetensors"))

        # Shard 2: up_proj and down_proj
        shard2 = {
            "model.layers.0.block_sparse_moe.switch_mlp.up_proj.weight": rng.randn(
                experts, inter, hidden
            ).astype(np.float16),
            "model.layers.0.block_sparse_moe.switch_mlp.down_proj.weight": rng.randn(
                experts, hidden, inter
            ).astype(np.float16),
        }
        save_file(shard2, str(model_dir / "model-00002-of-00002.safetensors"))

        # Index
        weight_map = {k: "model-00001-of-00002.safetensors" for k in shard1}
        weight_map.update({k: "model-00002-of-00002.safetensors" for k in shard2})
        (model_dir / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map})
        )

        # Config
        config = {
            "hidden_size": hidden,
            "intermediate_size": inter,
            "num_hidden_layers": 1,
            "num_local_experts": experts,
            "num_experts_per_tok": 2,
            "model_type": "minimax_m2",
        }
        (model_dir / "config.json").write_text(json.dumps(config))

        from olmlx.engine.flash.moe_bundler import bundle_moe_experts

        layouts = bundle_moe_experts(model_dir, tmp_path / "flash_moe")
        assert 0 in layouts
        assert layouts[0].num_experts == experts


# ---------------------------------------------------------------------------
# Nemotron-H helpers and tests
# ---------------------------------------------------------------------------


def _make_synthetic_nemotron_moe_weights(
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    num_layers: int,
    hybrid_override_pattern: str,
    tmp_path: Path,
    quantized: bool = False,
) -> Path:
    """Create synthetic safetensors with Nemotron-H-style naming.

    Nemotron-H uses:
      - backbone.layers.{l} (not model.layers.{l})
      - mixer.switch_mlp (not mlp.switch_mlp)
      - fc1/fc2 projections (not gate_proj/up_proj/down_proj)
      - hybrid_override_pattern for MoE layer detection ('E' = MoE)
    """
    from safetensors.numpy import save_file

    rng = np.random.RandomState(42)
    tensors = {}

    for layer_idx, block_type in enumerate(hybrid_override_pattern):
        if block_type == "E":
            # MoE layer
            prefix = f"backbone.layers.{layer_idx}.mixer"

            # Router weights
            tensors[f"{prefix}.gate.weight"] = rng.randn(
                num_experts, hidden_size
            ).astype(np.float16)
            tensors[f"{prefix}.gate.e_score_correction_bias"] = rng.randn(
                num_experts
            ).astype(np.float16)

            # Shared expert
            tensors[f"{prefix}.shared_experts.up_proj.weight"] = rng.randn(
                intermediate_size * 2, hidden_size
            ).astype(np.float16)
            tensors[f"{prefix}.shared_experts.down_proj.weight"] = rng.randn(
                hidden_size, intermediate_size * 2
            ).astype(np.float16)

            # Stacked expert weights (fc1/fc2)
            if not quantized:
                tensors[f"{prefix}.switch_mlp.fc1.weight"] = rng.randn(
                    num_experts, intermediate_size, hidden_size
                ).astype(np.float16)
                tensors[f"{prefix}.switch_mlp.fc2.weight"] = rng.randn(
                    num_experts, hidden_size, intermediate_size
                ).astype(np.float16)
            else:
                group_size = 64
                bits = 8
                fc1_packed_dim = hidden_size * bits // 32
                fc2_packed_dim = intermediate_size * bits // 32

                tensors[f"{prefix}.switch_mlp.fc1.weight"] = rng.randint(
                    0, 2**31, (num_experts, intermediate_size, fc1_packed_dim)
                ).astype(np.uint32)
                tensors[f"{prefix}.switch_mlp.fc1.scales"] = rng.randn(
                    num_experts, intermediate_size, hidden_size // group_size
                ).astype(np.float16)
                tensors[f"{prefix}.switch_mlp.fc1.biases"] = rng.randn(
                    num_experts, intermediate_size, hidden_size // group_size
                ).astype(np.float16)

                tensors[f"{prefix}.switch_mlp.fc2.weight"] = rng.randint(
                    0, 2**31, (num_experts, hidden_size, fc2_packed_dim)
                ).astype(np.uint32)
                tensors[f"{prefix}.switch_mlp.fc2.scales"] = rng.randn(
                    num_experts, hidden_size, intermediate_size // group_size
                ).astype(np.float16)
                tensors[f"{prefix}.switch_mlp.fc2.biases"] = rng.randn(
                    num_experts, hidden_size, intermediate_size // group_size
                ).astype(np.float16)
        elif block_type == "M":
            # Mamba layer — just a placeholder weight
            tensors[f"backbone.layers.{layer_idx}.norm.weight"] = rng.randn(
                hidden_size
            ).astype(np.float16)
        elif block_type == "*":
            # Attention layer — placeholder
            tensors[f"backbone.layers.{layer_idx}.mixer.q_proj.weight"] = rng.randn(
                hidden_size, hidden_size
            ).astype(np.float16)

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    save_file(tensors, str(model_dir / "model.safetensors"))

    config = {
        "model_type": "nemotron_h",
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "moe_intermediate_size": intermediate_size,
        "num_hidden_layers": num_layers,
        "n_routed_experts": num_experts,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "hybrid_override_pattern": hybrid_override_pattern,
    }
    if quantized:
        config["quantization"] = {"bits": 8, "group_size": 64}
    (model_dir / "config.json").write_text(json.dumps(config))

    return model_dir


class TestBundleNemotronMoeExperts:
    """Test bundler with Nemotron-H-style backbone.layers + mixer + fc1/fc2."""

    # Pattern: M=Mamba, E=MoE, *=Attention — 6 layers, 3 MoE
    PATTERN = "ME*EME"

    def test_detect_nemotron_moe_layers(self, tmp_path):
        """Should detect MoE layers from hybrid_override_pattern ('E' chars)."""
        hidden, inter, experts = 64, 32, 4
        model_dir = _make_synthetic_nemotron_moe_weights(
            hidden, inter, experts, 6, self.PATTERN, tmp_path
        )

        from olmlx.engine.flash.moe_bundler import bundle_moe_experts

        layouts = bundle_moe_experts(model_dir, tmp_path / "flash_moe")

        # E is at indices 1, 3, 5 in "ME*EME"
        assert sorted(layouts.keys()) == [1, 3, 5]

    def test_bundle_nemotron_creates_correct_files(self, tmp_path):
        """Each Nemotron MoE layer should produce a .flashexperts file."""
        hidden, inter, experts = 64, 32, 4
        model_dir = _make_synthetic_nemotron_moe_weights(
            hidden, inter, experts, 6, self.PATTERN, tmp_path
        )
        output_dir = tmp_path / "flash_moe"

        from olmlx.engine.flash.moe_bundler import bundle_moe_experts

        layouts = bundle_moe_experts(model_dir, output_dir)

        assert len(layouts) == 3
        for layer_idx in [1, 3, 5]:
            assert (output_dir / f"layer_{layer_idx:02d}.flashexperts").exists()

    def test_bundle_nemotron_preserves_fc1_fc2_data(self, tmp_path):
        """Bundled data should match original fc1/fc2 weights."""
        hidden, inter, experts = 64, 32, 8
        model_dir = _make_synthetic_nemotron_moe_weights(
            hidden, inter, experts, 6, self.PATTERN, tmp_path
        )
        output_dir = tmp_path / "flash_moe"

        from safetensors.numpy import load_file

        from olmlx.engine.flash.moe_bundler import bundle_moe_experts

        original = load_file(str(model_dir / "model.safetensors"))
        fc1_w = original["backbone.layers.1.mixer.switch_mlp.fc1.weight"]
        fc2_w = original["backbone.layers.1.mixer.switch_mlp.fc2.weight"]

        layouts = bundle_moe_experts(model_dir, output_dir)
        layout = layouts[1]

        expert_idx = 3
        expert_offset = int(layout.offsets[expert_idx])
        with open(layout.file_path, "rb") as f:
            f.seek(expert_offset)
            raw = f.read(layout.expert_byte_size)

        # fc1: (inter, hidden) float16, fc2: (hidden, inter) float16
        fc1_size = inter * hidden * 2
        fc2_size = hidden * inter * 2

        fc1_read = np.frombuffer(raw[:fc1_size], dtype=np.float16).reshape(
            inter, hidden
        )
        fc2_read = np.frombuffer(
            raw[fc1_size : fc1_size + fc2_size], dtype=np.float16
        ).reshape(hidden, inter)

        np.testing.assert_array_equal(fc1_read, fc1_w[expert_idx])
        np.testing.assert_array_equal(fc2_read, fc2_w[expert_idx])

    def test_bundle_nemotron_quantized(self, tmp_path):
        """Quantized Nemotron model with fc1/fc2 should bundle correctly."""
        hidden, inter, experts = 64, 32, 4
        model_dir = _make_synthetic_nemotron_moe_weights(
            hidden, inter, experts, 6, self.PATTERN, tmp_path, quantized=True
        )
        output_dir = tmp_path / "flash_moe"

        from olmlx.engine.flash.moe_bundler import (
            MOE_HEADER_SIZE,
            bundle_moe_experts,
            parse_moe_header,
        )

        layouts = bundle_moe_experts(model_dir, output_dir)
        layout = layouts[1]

        with open(layout.file_path, "rb") as f:
            header = parse_moe_header(f.read(MOE_HEADER_SIZE))
        assert header["is_quantized"] is True
        assert header["bits"] == 8

    def test_bundle_nemotron_layout_json_has_projections(self, tmp_path):
        """Layout JSON should record fc1/fc2 projection names in manifest."""
        hidden, inter, experts = 64, 32, 4
        model_dir = _make_synthetic_nemotron_moe_weights(
            hidden, inter, experts, 6, self.PATTERN, tmp_path
        )
        output_dir = tmp_path / "flash_moe"

        from olmlx.engine.flash.moe_bundler import bundle_moe_experts

        bundle_moe_experts(model_dir, output_dir)

        config = json.loads((output_dir / "flash_moe_layout.json").read_text())
        manifest = config["component_manifest"]
        proj_names = [e["name"] for e in manifest]
        assert "fc1.weight" in proj_names
        assert "fc2.weight" in proj_names
        # Should NOT have gate_proj/up_proj/down_proj
        assert all("gate_proj" not in n for n in proj_names)

    def test_bundle_nemotron_sharded(self, tmp_path):
        """Should handle sharded Nemotron models via safetensors index."""
        from safetensors.numpy import save_file

        hidden, inter, experts = 64, 32, 4
        rng = np.random.RandomState(99)

        model_dir = tmp_path / "sharded_model"
        model_dir.mkdir()

        # Shard 1: fc1
        shard1 = {
            "backbone.layers.1.mixer.switch_mlp.fc1.weight": rng.randn(
                experts, inter, hidden
            ).astype(np.float16),
            "backbone.layers.1.mixer.gate.weight": rng.randn(experts, hidden).astype(
                np.float16
            ),
        }
        save_file(shard1, str(model_dir / "model-00001-of-00002.safetensors"))

        # Shard 2: fc2
        shard2 = {
            "backbone.layers.1.mixer.switch_mlp.fc2.weight": rng.randn(
                experts, hidden, inter
            ).astype(np.float16),
        }
        save_file(shard2, str(model_dir / "model-00002-of-00002.safetensors"))

        weight_map = {k: "model-00001-of-00002.safetensors" for k in shard1}
        weight_map.update({k: "model-00002-of-00002.safetensors" for k in shard2})
        (model_dir / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map})
        )

        config = {
            "model_type": "nemotron_h",
            "hidden_size": hidden,
            "moe_intermediate_size": inter,
            "num_hidden_layers": 2,
            "n_routed_experts": experts,
            "num_experts_per_tok": 2,
            "hybrid_override_pattern": "ME",  # only layer 1 is MoE
        }
        (model_dir / "config.json").write_text(json.dumps(config))

        from olmlx.engine.flash.moe_bundler import bundle_moe_experts

        layouts = bundle_moe_experts(model_dir, output_dir=tmp_path / "flash_moe")
        assert 1 in layouts
        assert layouts[1].num_experts == experts
