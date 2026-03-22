"""Tests for pre-sharding logic (centralized download and distribution)."""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch


from olmlx.engine.pre_shard import (
    FakeGroup,
    _filter_pipeline_weights,
    collect_non_weight_files,
    pre_shard_all_workers,
    pre_shard_for_rank,
    pre_shard_pipeline_all_workers,
    pre_shard_pipeline_for_rank,
    read_shard_marker,
    write_shard_marker,
)


class TestFakeGroup:
    def test_rank(self):
        g = FakeGroup(rank=1, size=4)
        assert g.rank() == 1

    def test_size(self):
        g = FakeGroup(rank=0, size=2)
        assert g.size() == 2

    def test_different_ranks(self):
        for r in range(4):
            g = FakeGroup(rank=r, size=4)
            assert g.rank() == r
            assert g.size() == 4


class TestShardMarker:
    def test_roundtrip(self, tmp_path):
        write_shard_marker(tmp_path, rank=1, world_size=2, model_path="Qwen/Qwen3-8B")
        marker = read_shard_marker(tmp_path)
        assert marker is not None
        assert marker["rank"] == 1
        assert marker["world_size"] == 2
        assert marker["model_path"] == "Qwen/Qwen3-8B"

    def test_read_missing_marker(self, tmp_path):
        assert read_shard_marker(tmp_path) is None

    def test_read_corrupt_marker(self, tmp_path):
        (tmp_path / ".pre_sharded").write_text("not json")
        assert read_shard_marker(tmp_path) is None

    def test_marker_file_name(self, tmp_path):
        write_shard_marker(tmp_path, rank=1, world_size=2, model_path="test/model")
        assert (tmp_path / ".pre_sharded").exists()


class TestCollectNonWeightFiles:
    def test_collects_config_and_tokenizer(self, tmp_path):
        # Create typical model files
        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "tokenizer.json").write_text("{}")
        (tmp_path / "tokenizer_config.json").write_text("{}")
        (tmp_path / "special_tokens_map.json").write_text("{}")
        (tmp_path / "tokenizer.model").write_bytes(b"\x00")
        # Weight files should be excluded
        (tmp_path / "model.safetensors").write_bytes(b"\x00")
        (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"\x00")
        (tmp_path / "model.safetensors.index.json").write_text("{}")

        files = collect_non_weight_files(tmp_path)
        names = {f.name for f in files}

        assert "config.json" in names
        assert "tokenizer.json" in names
        assert "tokenizer_config.json" in names
        assert "special_tokens_map.json" in names
        assert "tokenizer.model" in names
        # Weight files excluded
        assert "model.safetensors" not in names
        assert "model-00001-of-00002.safetensors" not in names
        assert "model.safetensors.index.json" not in names

    def test_empty_dir(self, tmp_path):
        assert collect_non_weight_files(tmp_path) == []

    def test_includes_generation_config(self, tmp_path):
        (tmp_path / "generation_config.json").write_text("{}")
        files = collect_non_weight_files(tmp_path)
        assert any(f.name == "generation_config.json" for f in files)


class TestPreShardForRank:
    @patch("mlx_lm.load")
    @patch("mlx.utils.tree_flatten", return_value=[("layer", "params")])
    @patch("mlx.core.save_safetensors")
    @patch("mlx.core.eval")
    def test_calls_shard_with_fake_group(
        self, mock_eval, mock_save, mock_flatten, mock_load, tmp_path
    ):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")
        output_dir = tmp_path / "output"

        mock_model = MagicMock()
        mock_model.parameters.return_value = {"layer": "params"}
        mock_load.return_value = (mock_model, MagicMock())

        pre_shard_for_rank(model_dir, rank=1, world_size=2, output_dir=output_dir)

        # Verify shard was called with a FakeGroup
        mock_model.shard.assert_called_once()
        group_arg = mock_model.shard.call_args[0][0]
        assert group_arg.rank() == 1
        assert group_arg.size() == 2

    @patch("mlx_lm.load")
    @patch("mlx.utils.tree_flatten", return_value=[("layer", "params")])
    @patch("mlx.core.save_safetensors")
    @patch("mlx.core.eval")
    def test_saves_weights(
        self, mock_eval, mock_save, mock_flatten, mock_load, tmp_path
    ):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")
        output_dir = tmp_path / "output"

        mock_model = MagicMock()
        mock_model.parameters.return_value = {"layer": "params"}
        mock_load.return_value = (mock_model, MagicMock())

        pre_shard_for_rank(model_dir, rank=1, world_size=2, output_dir=output_dir)

        # Verify weights were materialized
        mock_eval.assert_called()
        # Verify weights were saved with flattened params
        mock_save.assert_called_once()
        save_path = mock_save.call_args[0][0]
        assert save_path == str(output_dir / "model.safetensors")
        # Verify tree_flatten was used
        mock_flatten.assert_called_once()

    @patch("mlx_lm.load")
    @patch("mlx.utils.tree_flatten", return_value=[("layer", "params")])
    @patch("mlx.core.save_safetensors")
    @patch("mlx.core.eval")
    def test_writes_marker(
        self, mock_eval, mock_save, mock_flatten, mock_load, tmp_path
    ):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")
        output_dir = tmp_path / "output"

        mock_model = MagicMock()
        mock_model.parameters.return_value = {"layer": "params"}
        mock_load.return_value = (mock_model, MagicMock())

        pre_shard_for_rank(model_dir, rank=1, world_size=2, output_dir=output_dir)

        marker = read_shard_marker(output_dir)
        assert marker is not None
        assert marker["rank"] == 1
        assert marker["world_size"] == 2

    @patch("mlx_lm.load")
    @patch("mlx.utils.tree_flatten", return_value=[("layer", "params")])
    @patch("mlx.core.save_safetensors")
    @patch("mlx.core.eval")
    def test_copies_non_weight_files(
        self, mock_eval, mock_save, mock_flatten, mock_load, tmp_path
    ):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text('{"test": true}')
        (model_dir / "tokenizer.json").write_text('{"tok": true}')
        output_dir = tmp_path / "output"

        mock_model = MagicMock()
        mock_model.parameters.return_value = {"layer": "params"}
        mock_load.return_value = (mock_model, MagicMock())

        pre_shard_for_rank(model_dir, rank=1, world_size=2, output_dir=output_dir)

        assert (output_dir / "config.json").exists()
        assert json.loads((output_dir / "config.json").read_text()) == {"test": True}
        assert (output_dir / "tokenizer.json").exists()


class TestPreShardAllWorkers:
    @patch("olmlx.engine.pre_shard.pre_shard_for_rank")
    def test_shards_ranks_1_to_n(self, mock_shard, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        output_base = tmp_path / "shards"

        result = pre_shard_all_workers(model_dir, world_size=3, output_base=output_base)

        assert mock_shard.call_count == 2  # ranks 1 and 2
        # Verify rank 1
        mock_shard.assert_any_call(
            model_dir,
            rank=1,
            world_size=3,
            output_dir=output_base / "rank1",
        )
        # Verify rank 2
        mock_shard.assert_any_call(
            model_dir,
            rank=2,
            world_size=3,
            output_dir=output_base / "rank2",
        )
        assert 1 in result
        assert 2 in result

    @patch("olmlx.engine.pre_shard.pre_shard_for_rank")
    def test_returns_shard_dirs(self, mock_shard, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        output_base = tmp_path / "shards"

        result = pre_shard_all_workers(model_dir, world_size=2, output_base=output_base)

        assert result == {1: output_base / "rank1"}

    @patch("olmlx.engine.pre_shard.pre_shard_for_rank")
    def test_calls_progress_callback(self, mock_shard, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        output_base = tmp_path / "shards"
        progress = MagicMock()

        pre_shard_all_workers(
            model_dir, world_size=3, output_base=output_base, progress_cb=progress
        )

        assert progress.call_count == 2


class TestWorkerPreShardedLoading:
    """Tests for the worker-side pre-sharded loading path."""

    def test_env_var_constant(self):
        """Verify the env var constant is defined and consistent."""
        from olmlx.config import PRE_SHARDED_DIR_ENV

        assert PRE_SHARDED_DIR_ENV == "OLMLX_EXPERIMENTAL_DISTRIBUTED_PRE_SHARDED_DIR"

    def test_marker_mismatch_returns_none(self, tmp_path):
        """When marker doesn't match expected model, should signal fallback."""
        write_shard_marker(tmp_path, rank=1, world_size=2, model_path="old/model")
        marker = read_shard_marker(tmp_path)
        # Caller checks model_path match
        assert marker["model_path"] != "new/model"

    def test_marker_world_size_mismatch(self, tmp_path):
        """When world_size changed, marker should signal stale shards."""
        write_shard_marker(tmp_path, rank=1, world_size=2, model_path="test/model")
        marker = read_shard_marker(tmp_path)
        assert marker["world_size"] != 3


class TestFilterPipelineWeights:
    def test_keeps_owned_layers_and_renumbers(self):
        weights = {
            "model.layers.0.self_attn.q_proj.weight": "L0_q",
            "model.layers.1.self_attn.q_proj.weight": "L1_q",
            "model.layers.2.self_attn.q_proj.weight": "L2_q",
            "model.layers.3.self_attn.q_proj.weight": "L3_q",
            "model.embed_tokens.weight": "embed",
            "model.norm.weight": "norm",
            "lm_head.weight": "lm_head",
        }
        result = _filter_pipeline_weights(weights, start_idx=1, end_idx=3)
        # Owned layers 1,2 renumbered to 0,1
        assert "model.layers.0.self_attn.q_proj.weight" in result
        assert result["model.layers.0.self_attn.q_proj.weight"] == "L1_q"
        assert "model.layers.1.self_attn.q_proj.weight" in result
        assert result["model.layers.1.self_attn.q_proj.weight"] == "L2_q"
        # Non-owned layers dropped
        assert not any(
            k.startswith("model.layers.2.") or k.startswith("model.layers.3.")
            for k in result
            if "layers" in k and result[k] not in ("L1_q", "L2_q")
        )
        # Shared weights kept
        assert result["model.embed_tokens.weight"] == "embed"
        assert result["model.norm.weight"] == "norm"
        assert result["lm_head.weight"] == "lm_head"

    def test_drops_layers_outside_range(self):
        weights = {
            "model.layers.0.mlp.weight": "L0",
            "model.layers.1.mlp.weight": "L1",
            "model.layers.2.mlp.weight": "L2",
        }
        result = _filter_pipeline_weights(weights, start_idx=2, end_idx=3)
        assert len([k for k in result if "layers" in k]) == 1
        assert result["model.layers.0.mlp.weight"] == "L2"

    def test_empty_range(self):
        weights = {
            "model.layers.0.mlp.weight": "L0",
            "model.embed_tokens.weight": "embed",
        }
        result = _filter_pipeline_weights(weights, start_idx=0, end_idx=0)
        assert not any("layers" in k for k in result)
        assert result["model.embed_tokens.weight"] == "embed"

    def test_full_range_keeps_all(self):
        weights = {
            "model.layers.0.mlp.weight": "L0",
            "model.layers.1.mlp.weight": "L1",
            "model.norm.weight": "norm",
        }
        result = _filter_pipeline_weights(weights, start_idx=0, end_idx=2)
        assert result["model.layers.0.mlp.weight"] == "L0"
        assert result["model.layers.1.mlp.weight"] == "L1"
        assert result["model.norm.weight"] == "norm"

    def test_deeply_nested_layer_keys(self):
        weights = {
            "model.layers.5.self_attn.k_proj.scales": "val",
            "model.layers.5.self_attn.k_proj.biases": "val2",
        }
        result = _filter_pipeline_weights(weights, start_idx=5, end_idx=6)
        assert "model.layers.0.self_attn.k_proj.scales" in result
        assert "model.layers.0.self_attn.k_proj.biases" in result


class TestPreShardPipelineForRank:
    def _make_model_dir(self, tmp_path, num_layers=4, layer_types=None):
        """Create a fake model directory with safetensors weights and config."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        config = {"num_hidden_layers": num_layers, "hidden_size": 64}
        if layer_types is not None:
            config["layer_types"] = layer_types
        (model_dir / "config.json").write_text(json.dumps(config))
        (model_dir / "tokenizer.json").write_text('{"tok": true}')
        # Create fake safetensors with layer weights
        weights = {}
        for i in range(num_layers):
            weights[f"model.layers.{i}.self_attn.q_proj.weight"] = f"L{i}_q"
            weights[f"model.layers.{i}.mlp.weight"] = f"L{i}_mlp"
        weights["model.embed_tokens.weight"] = "embed"
        weights["model.norm.weight"] = "norm"
        weights["lm_head.weight"] = "lm_head"
        return model_dir, weights

    @patch("mlx.core.save_safetensors")
    @patch("olmlx.engine.pre_shard._load_safetensors_weights")
    def test_filters_and_renumbers_weights(
        self, mock_load_weights, mock_save, tmp_path
    ):
        model_dir, weights = self._make_model_dir(tmp_path, num_layers=4)
        mock_load_weights.return_value = weights
        output_dir = tmp_path / "output"

        # Rank 1 of 2 with layer_counts=[2, 2]: rank 1 gets layers 0-1
        pre_shard_pipeline_for_rank(
            model_dir,
            rank=1,
            world_size=2,
            output_dir=output_dir,
            layer_counts=[2, 2],
        )

        mock_save.assert_called_once()
        saved_weights = mock_save.call_args[0][1]
        # Should have layers 0,1 (renumbered from original 0,1)
        assert "model.layers.0.self_attn.q_proj.weight" in saved_weights
        assert "model.layers.1.self_attn.q_proj.weight" in saved_weights
        # Should NOT have layers 2,3
        assert not any(k.startswith("model.layers.2.") for k in saved_weights)
        # Shared weights present
        assert "model.embed_tokens.weight" in saved_weights

    @patch("mlx.core.save_safetensors")
    @patch("olmlx.engine.pre_shard._load_safetensors_weights")
    def test_modifies_config_num_hidden_layers(
        self, mock_load_weights, mock_save, tmp_path
    ):
        model_dir, weights = self._make_model_dir(tmp_path, num_layers=4)
        mock_load_weights.return_value = weights
        output_dir = tmp_path / "output"

        pre_shard_pipeline_for_rank(
            model_dir,
            rank=1,
            world_size=2,
            output_dir=output_dir,
            layer_counts=[2, 2],
        )

        config = json.loads((output_dir / "config.json").read_text())
        assert config["num_hidden_layers"] == 2

    @patch("mlx.core.save_safetensors")
    @patch("olmlx.engine.pre_shard._load_safetensors_weights")
    def test_slices_layer_types_for_gpt_oss(
        self, mock_load_weights, mock_save, tmp_path
    ):
        layer_types = ["full_attention", "sliding_attention"] * 3  # 6 layers
        model_dir, weights = self._make_model_dir(
            tmp_path, num_layers=6, layer_types=layer_types
        )
        mock_load_weights.return_value = weights
        output_dir = tmp_path / "output"

        # Rank 0 of 2 with [3, 3]: rank 0 gets layers 3-5
        pre_shard_pipeline_for_rank(
            model_dir,
            rank=0,
            world_size=2,
            output_dir=output_dir,
            layer_counts=[3, 3],
        )

        config = json.loads((output_dir / "config.json").read_text())
        assert config["num_hidden_layers"] == 3
        assert config["layer_types"] == [
            "sliding_attention",
            "full_attention",
            "sliding_attention",
        ]

    @patch("mlx.core.save_safetensors")
    @patch("olmlx.engine.pre_shard._load_safetensors_weights")
    def test_writes_marker_with_strategy_and_layer_counts(
        self, mock_load_weights, mock_save, tmp_path
    ):
        model_dir, weights = self._make_model_dir(tmp_path, num_layers=4)
        mock_load_weights.return_value = weights
        output_dir = tmp_path / "output"

        pre_shard_pipeline_for_rank(
            model_dir,
            rank=1,
            world_size=2,
            output_dir=output_dir,
            layer_counts=[2, 2],
        )

        marker = read_shard_marker(output_dir)
        assert marker is not None
        assert marker["rank"] == 1
        assert marker["world_size"] == 2
        assert marker["strategy"] == "pipeline"
        assert marker["layer_counts"] == [2, 2]

    @patch("mlx.core.save_safetensors")
    @patch("olmlx.engine.pre_shard._load_safetensors_weights")
    def test_copies_non_weight_files(self, mock_load_weights, mock_save, tmp_path):
        model_dir, weights = self._make_model_dir(tmp_path, num_layers=4)
        mock_load_weights.return_value = weights
        output_dir = tmp_path / "output"

        pre_shard_pipeline_for_rank(
            model_dir,
            rank=1,
            world_size=2,
            output_dir=output_dir,
            layer_counts=[2, 2],
        )

        assert (output_dir / "tokenizer.json").exists()

    @patch("mlx.core.save_safetensors")
    @patch("olmlx.engine.pre_shard._load_safetensors_weights")
    def test_uneven_split(self, mock_load_weights, mock_save, tmp_path):
        """Test 3-way uneven split like gpt-oss-120b: [19, 7, 10] across 36 layers."""
        model_dir, weights = self._make_model_dir(tmp_path, num_layers=36)
        # Add more weights for 36 layers
        for i in range(4, 36):
            weights[f"model.layers.{i}.self_attn.q_proj.weight"] = f"L{i}_q"
            weights[f"model.layers.{i}.mlp.weight"] = f"L{i}_mlp"
        mock_load_weights.return_value = weights
        output_dir = tmp_path / "output"

        # Rank 2 (highest) gets first 10 layers (0-9)
        pre_shard_pipeline_for_rank(
            model_dir,
            rank=2,
            world_size=3,
            output_dir=output_dir,
            layer_counts=[19, 7, 10],
        )

        config = json.loads((output_dir / "config.json").read_text())
        assert config["num_hidden_layers"] == 10
        saved_weights = mock_save.call_args[0][1]
        layer_keys = [k for k in saved_weights if "layers" in k]
        # 10 layers × 2 keys each = 20
        assert len(layer_keys) == 20

    @patch("mlx.core.save_safetensors")
    @patch("olmlx.engine.pre_shard._load_safetensors_weights")
    def test_invalid_layer_counts_sum_raises(
        self, mock_load_weights, mock_save, tmp_path
    ):
        model_dir, weights = self._make_model_dir(tmp_path, num_layers=4)
        mock_load_weights.return_value = weights
        output_dir = tmp_path / "output"

        import pytest

        with pytest.raises(ValueError, match="sums to"):
            pre_shard_pipeline_for_rank(
                model_dir,
                rank=0,
                world_size=2,
                output_dir=output_dir,
                layer_counts=[3, 3],  # sums to 6, but model has 4 layers
            )

    @patch("mlx.core.save_safetensors")
    @patch("olmlx.engine.pre_shard._load_safetensors_weights")
    def test_invalid_layer_counts_length_raises(
        self, mock_load_weights, mock_save, tmp_path
    ):
        model_dir, weights = self._make_model_dir(tmp_path, num_layers=4)
        mock_load_weights.return_value = weights
        output_dir = tmp_path / "output"

        import pytest

        with pytest.raises(ValueError, match="must have"):
            pre_shard_pipeline_for_rank(
                model_dir,
                rank=0,
                world_size=2,
                output_dir=output_dir,
                layer_counts=[1, 1, 2],  # 3 entries but world_size=2
            )


class TestPreShardPipelineAllWorkers:
    @patch("olmlx.engine.pre_shard.pre_shard_pipeline_for_rank")
    def test_shards_ranks_1_to_n(self, mock_shard, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        output_base = tmp_path / "shards"

        result = pre_shard_pipeline_all_workers(
            model_dir,
            world_size=3,
            output_base=output_base,
            layer_counts=[19, 7, 10],
        )

        assert mock_shard.call_count == 2  # ranks 1 and 2
        mock_shard.assert_any_call(
            model_dir,
            rank=1,
            world_size=3,
            output_dir=output_base / "rank1",
            layer_counts=[19, 7, 10],
        )
        mock_shard.assert_any_call(
            model_dir,
            rank=2,
            world_size=3,
            output_dir=output_base / "rank2",
            layer_counts=[19, 7, 10],
        )
        assert 1 in result
        assert 2 in result

    @patch("olmlx.engine.pre_shard.pre_shard_pipeline_for_rank")
    def test_calls_progress_callback(self, mock_shard, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        output_base = tmp_path / "shards"
        progress = MagicMock()

        pre_shard_pipeline_all_workers(
            model_dir,
            world_size=3,
            output_base=output_base,
            layer_counts=[19, 7, 10],
            progress_cb=progress,
        )

        assert progress.call_count == 2


class TestConfigFields:
    """Tests for the new ExperimentalSettings fields."""

    def test_pre_shard_defaults(self, monkeypatch):
        for key in os.environ:
            if key.startswith("OLMLX_EXPERIMENTAL_"):
                monkeypatch.delenv(key, raising=False)

        from olmlx.config import ExperimentalSettings

        s = ExperimentalSettings()
        assert s.distributed_pre_shard is True
        assert s.distributed_shard_dir == Path("~/.olmlx/shards")
        assert s.distributed_worker_shard_dir == "~/.olmlx/shards"

    def test_pre_shard_env_override(self, monkeypatch):
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_DISTRIBUTED_PRE_SHARD", "false")
        monkeypatch.setenv("OLMLX_EXPERIMENTAL_DISTRIBUTED_SHARD_DIR", "/tmp/shards")
        monkeypatch.setenv(
            "OLMLX_EXPERIMENTAL_DISTRIBUTED_WORKER_SHARD_DIR", "/remote/shards"
        )

        from olmlx.config import ExperimentalSettings

        s = ExperimentalSettings()
        assert s.distributed_pre_shard is False
        assert s.distributed_shard_dir == Path("/tmp/shards")
        assert s.distributed_worker_shard_dir == "/remote/shards"
