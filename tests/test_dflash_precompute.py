"""Tests for sharded target hidden-state precompute + reuse."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import pytest

from olmlx.engine.dflash.decoder import _patch_model, _unpatch_model
from olmlx.engine.dflash.precompute import (
    INDEX_FILENAME,
    iter_precomputed_shards,
    precompute_target_hiddens,
)

# Re-use the synthetic target/tokenizer/batch helpers from the prepare tests.
from tests.test_dflash_prepare import (
    _Target,
    _mock_target_loader,
    _synthetic_batches,
    _write_target_config,
)


# ---------------------------------------------------------------------------
# Precompute writer
# ---------------------------------------------------------------------------


class TestPrecomputeWriter:
    def test_writes_shards_and_index(self, tmp_path):
        target = _Target(vocab_size=64, hidden_size=16, num_layers=4)
        layer_ids = [1, 3]
        storage: list = [None] * len(layer_ids)
        _patch_model(target, layer_ids, storage)
        try:
            batches = _synthetic_batches(vocab=64, batch_size=2, seq_len=32, n=4)
            out = precompute_target_hiddens(
                target, batches, tmp_path / "cache", storage, num_shards=4
            )
        finally:
            _unpatch_model(target)

        assert out == tmp_path / "cache"
        shards = sorted(out.glob("shard-*.safetensors"))
        assert len(shards) == 4
        index = json.loads((out / INDEX_FILENAME).read_text())
        assert index["num_shards"] == 4
        assert index["batch_size"] == 2
        assert index["seq_len"] == 32
        # 2 target layers * model hidden_size 16 = concat 32 (this is
        # the on-disk concatenated dim, not the per-layer model dim).
        assert index["concat_hidden_size"] == 32

    def test_caps_at_num_shards(self, tmp_path):
        target = _Target(vocab_size=64, hidden_size=16, num_layers=2)
        layer_ids = [0, 1]
        storage: list = [None] * len(layer_ids)
        _patch_model(target, layer_ids, storage)
        try:
            # Iterator yields 10 but cap at 3.
            batches = _synthetic_batches(vocab=64, batch_size=2, seq_len=32, n=10)
            precompute_target_hiddens(
                target, batches, tmp_path / "cache", storage, num_shards=3
            )
        finally:
            _unpatch_model(target)
        assert len(list((tmp_path / "cache").glob("shard-*.safetensors"))) == 3


# ---------------------------------------------------------------------------
# Precompute reader
# ---------------------------------------------------------------------------


class TestPrecomputeReader:
    def _write_two_shards(self, tmp_path) -> Path:
        target = _Target(vocab_size=64, hidden_size=16, num_layers=2)
        layer_ids = [0, 1]
        storage: list = [None] * len(layer_ids)
        _patch_model(target, layer_ids, storage)
        try:
            batches = _synthetic_batches(vocab=64, batch_size=2, seq_len=32, n=2)
            precompute_target_hiddens(
                target, batches, tmp_path / "cache", storage, num_shards=2
            )
        finally:
            _unpatch_model(target)
        return tmp_path / "cache"

    def test_round_trip_single_pass(self, tmp_path):
        cache = self._write_two_shards(tmp_path)
        out = list(iter_precomputed_shards(cache))
        assert len(out) == 2
        for input_ids, hidden in out:
            assert input_ids.shape == (2, 32)
            assert hidden.shape == (2, 32, 32)

    def test_cycles_when_max_examples_exceeds_shards(self, tmp_path):
        cache = self._write_two_shards(tmp_path)
        # 5 examples from 2 shards → cycles 2.5 times.
        out = list(iter_precomputed_shards(cache, max_examples=5))
        assert len(out) == 5

    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            list(iter_precomputed_shards(tmp_path / "no_such_dir"))

    def test_empty_dir_raises(self, tmp_path):
        (tmp_path / "empty_cache").mkdir()
        with pytest.raises(FileNotFoundError, match="No shard"):
            list(iter_precomputed_shards(tmp_path / "empty_cache"))


# ---------------------------------------------------------------------------
# Training with --use-precomputed
# ---------------------------------------------------------------------------


class TestPrepareWithPrecomputed:
    def test_training_consumes_precomputed_shards(self, tmp_path):
        from olmlx.engine.dflash.prepare import prepare_dflash_draft

        mx.random.seed(0)
        vocab, hidden, num_layers = 64, 16, 4
        _write_target_config(tmp_path, vocab, hidden)

        # 1) Precompute shards using the test target.
        target = _Target(vocab_size=vocab, hidden_size=hidden, num_layers=num_layers)
        layer_ids = [1, 3]  # matches num_target_layers=2 below
        storage: list = [None] * len(layer_ids)
        _patch_model(target, layer_ids, storage)
        try:
            batches = _synthetic_batches(vocab=vocab, batch_size=2, seq_len=32, n=8)
            shard_dir = precompute_target_hiddens(
                target, batches, tmp_path / "cache", storage, num_shards=8
            )
        finally:
            _unpatch_model(target)

        # 2) Track whether the target's forward gets called during training.
        calls = {"n": 0}
        original_call = type(target).__call__

        def counting_call(self, input_ids, cache=None):
            calls["n"] += 1
            return original_call(self, input_ids, cache=cache)

        # 3) Train with --use-precomputed; the loader doesn't run the target.
        loader_target = _Target(
            vocab_size=vocab, hidden_size=hidden, num_layers=num_layers
        )
        type(loader_target).__call__ = counting_call  # type: ignore[method-assign]
        try:
            prepare_dflash_draft(
                tmp_path,
                steps=4,
                batch_size=2,
                seq_len=32,
                block_size=2,
                num_hidden_layers=1,
                num_target_layers=2,
                target_layer_ids=layer_ids,
                lr=1e-2,
                output_dir=tmp_path / "dflash_out",
                use_precomputed=str(shard_dir),
                _target_loader=lambda _p: (
                    loader_target,
                    _mock_target_loader(vocab, hidden, num_layers)(_p)[1],
                ),
                # No _batch_iterator: forces the training loop to read
                # precomputed shards from disk.
            )
        finally:
            type(loader_target).__call__ = original_call  # type: ignore[method-assign]

        # The target was loaded (once for binding) but its forward was
        # never called inside the training loop.
        assert calls["n"] == 0, (
            f"Expected zero target forward passes during precomputed "
            f"training, got {calls['n']}"
        )
        assert (tmp_path / "dflash_out" / "config.json").exists()
