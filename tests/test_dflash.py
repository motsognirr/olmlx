"""Tests for dflash block-diffusion speculative decoding."""

import mlx.core as mx
import mlx.nn as nn
import pytest

from tests.test_flash_speculative import MockModel


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


class TestTargetAdapterInterface:
    """Test the TargetAdapter abstract interface and registry."""

    def test_adapter_registry_exists(self):
        from olmlx.engine.dflash.adapters import ADAPTERS

        assert isinstance(ADAPTERS, dict)

    def test_get_adapter_returns_registered(self):
        from olmlx.engine.dflash.adapters import ADAPTERS, TargetAdapter, get_adapter

        class DummyAdapter(TargetAdapter):
            def forward_with_hidden(self, model, tokens, cache, target_layer_ids):
                return None, {}, None

            def trim_cache(self, cache, num_tokens):
                pass

        ADAPTERS["dummy"] = DummyAdapter
        try:
            adapter = get_adapter("dummy")
            assert isinstance(adapter, DummyAdapter)
        finally:
            del ADAPTERS["dummy"]

    def test_get_adapter_raises_for_unknown(self):
        from olmlx.engine.dflash.adapters import get_adapter

        with pytest.raises(KeyError, match="no_such_model"):
            get_adapter("no_such_model")


# ---------------------------------------------------------------------------
# Qwen3 Adapter
# ---------------------------------------------------------------------------


class MockAttention(nn.Module):
    """Attention with KV cache support for testing."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.n_heads = 1
        self.n_kv_heads = 1

    def __call__(self, x, cache=None):
        if cache is not None:
            k = v = x.reshape(x.shape[0], 1, -1, x.shape[-1])
            cache.update_and_fetch(k, v)
        return x


class Qwen3LikeLayer(nn.Module):
    """Minimal Qwen3-like layer with KV cache support."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.self_attn = MockAttention(hidden_size)

    def __call__(self, x, cache=None):
        return self.self_attn(x, cache=cache)


class Qwen3LikeModel(nn.Module):
    """Minimal Qwen3-like model for testing adapter."""

    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int = 4):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = [Qwen3LikeLayer(hidden_size) for _ in range(num_layers)]
        self.norm = nn.RMSNorm(hidden_size)

    def __call__(self, input_ids, cache=None):
        h = self.embed_tokens(input_ids)
        for i, layer in enumerate(self.layers):
            h = layer(h, cache=cache[i] if cache is not None else None)
        return self.norm(h)


class TestQwen3Adapter:
    def test_registered_in_adapters(self):
        from olmlx.engine.dflash.adapters import ADAPTERS

        assert "qwen3" in ADAPTERS

    def test_forward_with_hidden_extracts_layers(self):
        from olmlx.engine.dflash.adapters import get_adapter

        adapter = get_adapter("qwen3")
        vocab_size, hidden_size, num_layers = 32, 16, 4
        inner = Qwen3LikeModel(vocab_size, hidden_size, num_layers)

        # Wrap with lm_head (adapter requires it)
        class ModelWithHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = inner
                self.layers = inner.layers
                self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        model = ModelWithHead()
        tokens = mx.array([[1, 2, 3]])
        target_layer_ids = [1, 3]

        logits, hidden_states, cache = adapter.forward_with_hidden(
            model, tokens, cache=None, target_layer_ids=target_layer_ids
        )

        assert set(hidden_states.keys()) == {1, 3}
        for layer_id in target_layer_ids:
            assert hidden_states[layer_id].shape == (1, 3, hidden_size)
        assert logits.shape == (1, 3, vocab_size)

    def test_forward_with_hidden_raises_without_lm_head(self):
        from olmlx.engine.dflash.adapters import get_adapter

        adapter = get_adapter("qwen3")
        model = Qwen3LikeModel(32, 16, 4)  # no lm_head
        tokens = mx.array([[1, 2, 3]])

        with pytest.raises(RuntimeError, match="no lm_head"):
            adapter.forward_with_hidden(model, tokens, cache=None, target_layer_ids=[1])

    def test_trim_cache(self):
        from mlx_lm.models.cache import make_prompt_cache

        from olmlx.engine.dflash.adapters import get_adapter

        adapter = get_adapter("qwen3")
        model = MockModel(vocab_size=32, hidden_size=16, num_layers=2)
        cache = make_prompt_cache(model)

        tokens = mx.array([[1, 2, 3, 4, 5]])
        model(tokens, cache=cache)
        assert cache[0].offset == 5

        adapter.trim_cache(cache, 2)
        assert cache[0].offset == 3


# ---------------------------------------------------------------------------
# Draft Model
# ---------------------------------------------------------------------------


class TestDFlashDraftModel:
    def test_draft_config_from_dict(self):
        from olmlx.engine.dflash.draft_model import DraftConfig

        cfg = DraftConfig(
            hidden_size=256,
            num_attention_heads=4,
            num_layers=2,
            target_layer_ids=[1, 3],
            vocab_size=32000,
        )
        assert cfg.hidden_size == 256
        assert cfg.target_layer_ids == [1, 3]

    def test_draft_model_forward(self):
        from olmlx.engine.dflash.draft_model import DFlashDraftModel, DraftConfig

        cfg = DraftConfig(
            hidden_size=32,
            num_attention_heads=2,
            num_layers=1,
            target_layer_ids=[0],
            vocab_size=64,
            target_hidden_size=16,
        )
        draft = DFlashDraftModel(cfg)

        # Simulate target hidden states from 1 layer
        target_hidden = {0: mx.random.normal((1, 3, 16))}
        current_tokens = mx.array([[1, 2, 3]])

        logits = draft(current_tokens, target_hidden)
        mx.eval(logits)

        # Output should be (1, 3, vocab_size)
        assert logits.shape == (1, 3, 64)

    def test_draft_model_projects_target_hidden(self):
        """Draft model should project target hidden_size to draft hidden_size."""
        from olmlx.engine.dflash.draft_model import DFlashDraftModel, DraftConfig

        cfg = DraftConfig(
            hidden_size=32,
            num_attention_heads=2,
            num_layers=1,
            target_layer_ids=[0, 2],
            vocab_size=64,
            target_hidden_size=48,  # different from draft hidden_size
        )
        draft = DFlashDraftModel(cfg)

        target_hidden = {
            0: mx.random.normal((1, 3, 48)),
            2: mx.random.normal((1, 3, 48)),
        }
        logits = draft(mx.array([[1, 2, 3]]), target_hidden)
        mx.eval(logits)
        assert logits.shape == (1, 3, 64)


# ---------------------------------------------------------------------------
# DFlash Decoder
# ---------------------------------------------------------------------------


class TestDFlashDecoder:
    @pytest.fixture()
    def decoder_components(self):
        """Create a minimal dflash decoder for testing."""
        from olmlx.engine.dflash.adapters import get_adapter
        from olmlx.engine.dflash.draft_model import DFlashDraftModel, DraftConfig

        vocab_size, hidden_size = 32, 16
        num_layers = 4

        # Target: Qwen3-like model with lm_head
        target_inner = Qwen3LikeModel(vocab_size, hidden_size, num_layers)

        class TargetWithHead(nn.Module):
            def __init__(self, inner, vocab_size, hidden_size):
                super().__init__()
                self.model = inner
                self.layers = inner.layers  # expose for make_prompt_cache
                self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

            def __call__(self, input_ids, cache=None):
                h = self.model(input_ids, cache=cache)
                return self.lm_head(h)

        target = TargetWithHead(target_inner, vocab_size, hidden_size)

        # Draft config
        cfg = DraftConfig(
            hidden_size=16,
            num_attention_heads=2,
            num_layers=1,
            target_layer_ids=[1, 3],
            vocab_size=vocab_size,
            target_hidden_size=hidden_size,
        )
        draft = DFlashDraftModel(cfg)
        adapter = get_adapter("qwen3")

        return target, draft, adapter, cfg

    def test_decoder_creation(self, decoder_components):
        from olmlx.engine.dflash.decoder import DFlashDecoder

        target, draft, adapter, cfg = decoder_components
        decoder = DFlashDecoder(
            target_model=target,
            draft_model=draft,
            adapter=adapter,
            draft_config=cfg,
            block_size=3,
        )
        assert decoder._block_size == 3

    def test_prefill_returns_token(self, decoder_components):
        from olmlx.engine.dflash.decoder import DFlashDecoder

        target, draft, adapter, cfg = decoder_components
        decoder = DFlashDecoder(
            target_model=target,
            draft_model=draft,
            adapter=adapter,
            draft_config=cfg,
            block_size=3,
        )
        prompt = mx.array([[1, 2, 3]])
        first_token = decoder.prefill(prompt)

        assert isinstance(first_token, int)
        assert 0 <= first_token < 32

    def test_step_returns_accepted_tokens(self, decoder_components):
        from olmlx.engine.dflash.decoder import DFlashDecoder

        target, draft, adapter, cfg = decoder_components
        decoder = DFlashDecoder(
            target_model=target,
            draft_model=draft,
            adapter=adapter,
            draft_config=cfg,
            block_size=3,
        )
        prompt = mx.array([[1, 2, 3]])
        decoder.prefill(prompt)

        accepted, num_draft = decoder.step()

        assert len(accepted) >= 1
        assert num_draft == 3
        assert all(isinstance(t, int) for t in accepted)

    def test_reset_clears_state(self, decoder_components):
        from olmlx.engine.dflash.decoder import DFlashDecoder

        target, draft, adapter, cfg = decoder_components
        decoder = DFlashDecoder(
            target_model=target,
            draft_model=draft,
            adapter=adapter,
            draft_config=cfg,
        )
        prompt = mx.array([[1, 2, 3]])
        decoder.prefill(prompt)
        decoder.step()

        decoder.reset()
        assert decoder._cache is None
        assert decoder._cache_seq_len == 0

    def test_multi_step_consistency(self, decoder_components):
        from olmlx.engine.dflash.decoder import DFlashDecoder

        target, draft, adapter, cfg = decoder_components
        decoder = DFlashDecoder(
            target_model=target,
            draft_model=draft,
            adapter=adapter,
            draft_config=cfg,
            block_size=2,
        )
        prompt = mx.array([[1, 2, 3]])
        decoder.prefill(prompt)

        for _ in range(3):
            accepted, _ = decoder.step()
            assert len(accepted) >= 1

    def test_compatible_with_speculative_stream(self, decoder_components):
        """DFlashDecoder should work with speculative_stream_generate."""
        import threading

        from olmlx.engine.dflash.decoder import DFlashDecoder
        from olmlx.engine.speculative_stream import speculative_stream_generate

        target, draft, adapter, cfg = decoder_components
        decoder = DFlashDecoder(
            target_model=target,
            draft_model=draft,
            adapter=adapter,
            draft_config=cfg,
            block_size=2,
        )

        cancel = threading.Event()
        responses = list(
            speculative_stream_generate(
                decoder, [1, 2, 3], max_tokens=5, cancel_event=cancel
            )
        )

        assert len(responses) >= 1
        for resp in responses:
            assert hasattr(resp, "token")
            assert 0 <= resp.token < 32
