"""Tests for ``olmlx dflash prepare`` — DFlash draft training.

These tests exercise the full pipeline against a synthetic target so
they run without network access and without downloading any real
models. Coverage:

- Target-config → DraftConfig derivation (vocab, head dims, GQA shape)
- Target layer-id selection (explicit + evenly-spaced default)
- One end-to-end training run that asserts loss decreases over a few
  steps
- Round-trip: a freshly-saved draft loads back through the same
  ``DFlashDraftModel(DraftConfig(...))`` constructor used by
  ``_load_dflash_decoder``
- Saved ``config.json`` matches the upstream-compatible schema (nested
  ``dflash_config`` block)
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest


# ---------------------------------------------------------------------------
# Synthetic target + tokenizer
# ---------------------------------------------------------------------------


class _MockSelfAttn(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.n_heads = 1
        self.n_kv_heads = 1
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def __call__(self, x, mask=None, cache=None):
        if cache is not None:
            k = v = x.reshape(x.shape[0], 1, -1, x.shape[-1])
            cache.update_and_fetch(k, v)
        return self.proj(x)


class _MockLayer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.self_attn = _MockSelfAttn(hidden_size)

    def __call__(self, x, mask=None, cache=None):
        return x + self.self_attn(x, mask=mask, cache=cache)


class _Inner(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = [_MockLayer(hidden_size) for _ in range(num_layers)]
        self.norm = nn.RMSNorm(hidden_size)


class _Target(nn.Module):
    def __init__(self, vocab_size=64, hidden_size=16, num_layers=4):
        super().__init__()
        self.model = _Inner(vocab_size, hidden_size, num_layers)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    @property
    def layers(self):
        return self.model.layers

    def __call__(self, input_ids, cache=None):
        h = self.model.embed_tokens(input_ids)
        for i, layer in enumerate(self.model.layers):
            h = layer(h, cache=cache[i] if cache is not None else None)
        h = self.model.norm(h)
        return self.lm_head(h)


class _MockTokenizer:
    """Tokenizer just rich enough for ``stream_training_batches``."""

    pad_token_id = 0
    eos_token_id = 1

    def __init__(self, vocab_size: int = 64, seq_len: int = 64):
        self.vocab_size = vocab_size
        self._seq_len = seq_len

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        # Deterministic hash-based fake tokenization.
        tokens = []
        for i, ch in enumerate(text):
            tokens.append((ord(ch) + i) % self.vocab_size)
            if len(tokens) >= self._seq_len * 2:
                break
        # Ensure we exceed min_seq_len in the loader so the example isn't dropped.
        while len(tokens) < self._seq_len:
            tokens.append((len(tokens) + 7) % self.vocab_size)
        return tokens


def _mock_target_loader(vocab_size: int, hidden_size: int, num_layers: int):
    target = _Target(vocab_size, hidden_size, num_layers)
    tokenizer = _MockTokenizer(vocab_size=vocab_size, seq_len=64)
    return lambda _path: (target, tokenizer)


def _synthetic_batches(vocab: int, batch_size: int, seq_len: int, n: int):
    """Deterministic batch iterator so tests don't hit the network."""
    rng = mx.random.key(0)
    for _ in range(n):
        rng, sub = mx.random.split(rng)
        yield mx.random.randint(0, vocab, (batch_size, seq_len), key=sub)


def _write_target_config(tmp_path: Path, vocab_size: int, hidden_size: int) -> Path:
    cfg = {
        "model_type": "qwen3",
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": hidden_size // 2,
        "intermediate_size": hidden_size * 2,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "max_position_embeddings": 2048,
    }
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    return p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestEvenlySpaced:
    def test_picks_inner_indices(self):
        from olmlx.engine.dflash.prepare import _evenly_spaced

        # 32-layer target, 4 hooks
        ids = _evenly_spaced(32, 4)
        assert len(ids) == 4
        assert all(0 < i < 31 for i in ids)
        # Strictly increasing
        assert ids == sorted(ids)
        assert len(set(ids)) == 4

    def test_handles_small_targets(self):
        from olmlx.engine.dflash.prepare import _evenly_spaced

        ids = _evenly_spaced(2, 4)
        # When k >= num_layers, returns the full layer range.
        assert ids == [0, 1]


class TestResolveTargetLayerIds:
    def test_explicit_list(self):
        from olmlx.engine.dflash.prepare import _resolve_target_layer_ids

        out = _resolve_target_layer_ids([1, 5, 11], None, 16)
        assert out == [1, 5, 11]

    def test_out_of_range_raises(self):
        from olmlx.engine.dflash.prepare import _resolve_target_layer_ids

        with pytest.raises(ValueError, match="out of range"):
            _resolve_target_layer_ids([3, 99], None, 16)

    def test_default_when_none(self):
        from olmlx.engine.dflash.prepare import _resolve_target_layer_ids

        out = _resolve_target_layer_ids(None, 4, 16)
        assert len(out) == 4


class TestDraftConfigDerivation:
    def test_inherits_target_dims(self):
        from olmlx.engine.dflash.prepare import _build_draft_config

        target_cfg = {
            "vocab_size": 32000,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "intermediate_size": 11008,
            "rms_norm_eps": 1e-5,
            "rope_theta": 1000000.0,
            "max_position_embeddings": 32768,
        }
        cfg = _build_draft_config(
            target_cfg,
            target_layer_ids=[5, 11, 17, 23],
            num_hidden_layers=4,
            block_size=4,
            mask_token_id=0,
        )
        assert cfg.hidden_size == 4096
        assert cfg.head_dim == 128
        assert cfg.num_key_value_heads == 8
        assert cfg.vocab_size == 32000
        assert cfg.target_layer_ids == [5, 11, 17, 23]
        # ``block_size`` on disk is the *total* upstream length:
        # ``draft tokens + 1`` for the visible pending token.
        assert cfg.block_size == 5
        assert cfg.mask_token_id == 0


# ---------------------------------------------------------------------------
# End-to-end training
# ---------------------------------------------------------------------------


class TestPrepareDflashDraft:
    def test_loss_decreases(self, tmp_path):
        from olmlx.engine.dflash.prepare import prepare_dflash_draft

        # Seed for determinism; the synthetic-data + tiny-target setup
        # is noisy enough that an unseeded run occasionally fails to
        # show monotone descent in 20 steps.
        mx.random.seed(0)
        vocab, hidden, num_layers = 64, 16, 4
        _write_target_config(tmp_path, vocab, hidden)

        # Capture losses by patching the optimizer step indirectly: we
        # wrap stream_training_batches to deterministic batches AND read
        # the training log via a custom callback.
        losses: list[float] = []

        def cb(msg: str, _frac: float) -> None:
            # The progress callback message is "Training step N/M loss=X.XXXX"
            if "loss=" in msg:
                losses.append(float(msg.split("loss=")[-1].strip()))

        out_dir = prepare_dflash_draft(
            tmp_path,
            steps=20,
            batch_size=2,
            seq_len=32,
            block_size=2,
            num_hidden_layers=1,
            num_target_layers=2,
            lr=1e-2,
            output_dir=tmp_path / "dflash_out",
            progress_callback=cb,
            log_every=1,
            _target_loader=_mock_target_loader(vocab, hidden, num_layers),
            _batch_iterator=_synthetic_batches(vocab, batch_size=2, seq_len=32, n=20),
        )
        assert (out_dir / "config.json").exists()
        assert (out_dir / "model-00001-of-00001.safetensors").exists()
        assert len(losses) >= 5
        # Losses are noisy on synthetic data; require the *trailing*
        # window to be lower than the *leading* window rather than a
        # strict per-step monotone descent.
        head = sum(losses[:3]) / 3
        tail = sum(losses[-3:]) / 3
        assert tail < head, f"loss did not decrease: head={head}, tail={tail}"

    def test_saved_config_round_trips_through_loader(self, tmp_path):
        from olmlx.engine.dflash.draft_model import DFlashDraftModel, DraftConfig
        from olmlx.engine.dflash.prepare import prepare_dflash_draft

        vocab, hidden, num_layers = 64, 16, 4
        _write_target_config(tmp_path, vocab, hidden)

        out_dir = prepare_dflash_draft(
            tmp_path,
            steps=2,
            batch_size=2,
            seq_len=32,
            block_size=2,
            num_hidden_layers=1,
            num_target_layers=2,
            lr=1e-2,
            output_dir=tmp_path / "dflash_out",
            _target_loader=_mock_target_loader(vocab, hidden, num_layers),
            _batch_iterator=_synthetic_batches(vocab, batch_size=2, seq_len=32, n=20),
        )

        # Re-parse via the same logic _load_dflash_decoder uses on disk.
        cfg_dict = json.loads((out_dir / "config.json").read_text())
        assert "dflash_config" in cfg_dict
        assert "target_layer_ids" in cfg_dict["dflash_config"]
        assert "mask_token_id" in cfg_dict["dflash_config"]

        cfg = DraftConfig(
            hidden_size=cfg_dict["hidden_size"],
            num_hidden_layers=cfg_dict["num_hidden_layers"],
            num_attention_heads=cfg_dict["num_attention_heads"],
            num_key_value_heads=cfg_dict["num_key_value_heads"],
            head_dim=cfg_dict["head_dim"],
            intermediate_size=cfg_dict["intermediate_size"],
            vocab_size=cfg_dict["vocab_size"],
            rms_norm_eps=cfg_dict["rms_norm_eps"],
            rope_theta=cfg_dict["rope_theta"],
            max_position_embeddings=cfg_dict["max_position_embeddings"],
            block_size=cfg_dict["block_size"],
            num_target_layers=cfg_dict["num_target_layers"],
            target_layer_ids=cfg_dict["dflash_config"]["target_layer_ids"],
            mask_token_id=cfg_dict["dflash_config"]["mask_token_id"],
            layer_types=tuple(cfg_dict.get("layer_types", ())),
        )
        draft = DFlashDraftModel(cfg)

        # Load the saved weights — should match shapes exactly.
        weights = mx.load(str(out_dir / "model-00001-of-00001.safetensors"))
        draft.load_weights(list(weights.items()), strict=True)

    def test_unpatches_on_failure(self, tmp_path, monkeypatch):
        """If training raises mid-loop, the target hooks must be uninstalled."""
        from olmlx.engine.dflash.prepare import prepare_dflash_draft

        vocab, hidden, num_layers = 64, 16, 4
        _write_target_config(tmp_path, vocab, hidden)

        target_holder = {}
        original_loader = _mock_target_loader(vocab, hidden, num_layers)

        def trapping_loader(path):
            target, tok = original_loader(path)
            target_holder["target"] = target
            return target, tok

        # Force the draft step to raise.
        from olmlx.engine.dflash import prepare as prepare_mod

        def boom(*_a, **_kw):
            raise RuntimeError("simulated failure")

        monkeypatch.setattr(prepare_mod, "_draft_loss", boom)

        with pytest.raises(RuntimeError, match="simulated failure"):
            prepare_dflash_draft(
                tmp_path,
                steps=5,
                batch_size=2,
                seq_len=32,
                block_size=2,
                num_hidden_layers=1,
                num_target_layers=2,
                _target_loader=trapping_loader,
                _batch_iterator=_synthetic_batches(vocab, 2, 32, n=5),
            )

        target = target_holder["target"]
        assert not hasattr(target, "_hidden_states"), (
            "Target hooks must be uninstalled even when training raises"
        )
