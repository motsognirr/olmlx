"""Prompts longer than the model's context window are rejected up front (#715).

An over-window prompt used to be handed to generation, where it hung forever
while holding the global inference lock — wedging every later request. It must
now fail fast with a 400 *before* the lock is taken.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import olmlx.engine.inference as inf
from olmlx.engine.inference import (
    ContextLengthExceededError,
    generate_chat,
    generate_completion,
)
from olmlx.engine.kv_budget import resolve_context_length
from olmlx.engine.model_manager import LoadedModel, ModelManager
from olmlx.engine.registry import ModelConfig
from olmlx.engine.template_caps import TemplateCaps


class TestResolveContextLength:
    def test_max_position_embeddings(self):
        assert resolve_context_length({"max_position_embeddings": 32768}) == 32768

    def test_nested_text_config(self):
        cfg = {"text_config": {"max_position_embeddings": 131072}}
        assert resolve_context_length(cfg) == 131072

    def test_alternate_keys(self):
        assert resolve_context_length({"n_positions": 2048}) == 2048
        assert resolve_context_length({"max_seq_len": 4096}) == 4096

    def test_yarn_rope_scaling_extends_window(self):
        cfg = {
            "max_position_embeddings": 32768,
            "rope_scaling": {
                "type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 32768,
            },
        }
        assert resolve_context_length(cfg) == 131072

    def test_llama3_rope_scaling_does_not_inflate(self):
        # Llama 3.1: max_position_embeddings is already the extended window.
        cfg = {
            "max_position_embeddings": 131072,
            "rope_scaling": {
                "rope_type": "llama3",
                "factor": 8.0,
                "original_max_position_embeddings": 8192,
            },
        }
        assert resolve_context_length(cfg) == 131072

    def test_llama32_rope_scaling_does_not_inflate(self):
        # Llama 3.2: factor 32 x original 8192 = 262144, but llama3 scaling
        # doesn't extend the window — max_position_embeddings is the window.
        cfg = {
            "max_position_embeddings": 131072,
            "rope_scaling": {
                "rope_type": "llama3",
                "factor": 32.0,
                "original_max_position_embeddings": 8192,
            },
        }
        assert resolve_context_length(cfg) == 131072

    def test_non_extending_rope_type_without_mpe_is_unknown(self):
        cfg = {
            "rope_scaling": {
                "rope_type": "longrope",
                "factor": 32.0,
                "original_max_position_embeddings": 4096,
            }
        }
        assert resolve_context_length(cfg) is None

    def test_tokenizer_model_max_length_wins_when_larger(self):
        # Qwen2.5: config says 32768, tokenizer_config says 131072 (the value
        # in the #715 log). Take the more permissive declared limit.
        tok = SimpleNamespace(model_max_length=131072)
        assert resolve_context_length({"max_position_embeddings": 32768}, tok) == (
            131072
        )

    def test_tokenizer_sentinel_ignored(self):
        tok = SimpleNamespace(model_max_length=int(1e30))
        assert resolve_context_length({"max_position_embeddings": 4096}, tok) == 4096
        assert resolve_context_length(None, tok) is None

    def test_unknown_is_none(self):
        assert resolve_context_length(None) is None
        assert resolve_context_length({}) is None
        assert resolve_context_length({"max_position_embeddings": "big"}) is None
        assert resolve_context_length({"max_position_embeddings": True}) is None
        assert resolve_context_length({"max_position_embeddings": 0}) is None

    def test_magicmock_tokenizer_ignored(self):
        assert resolve_context_length({"n_ctx": 1024}, MagicMock()) == 1024


def _set_prompt_tokens(lm: LoadedModel, n: int) -> None:
    lm.tokenizer.encode = MagicMock(return_value=list(range(n)))
    lm.tokenizer.bos_token = None


class TestGenerateChatRejectsOverWindow:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.parametrize("prompt_cache", [True, False])
    async def test_over_window_rejected_before_generation(
        self, mock_manager, monkeypatch, stream, prompt_cache
    ):
        lm = mock_manager._loaded["qwen3:latest"]
        lm.context_length = 100
        lm.prompt_cache = prompt_cache
        _set_prompt_tokens(lm, 150)
        # Force the tokenize fallback even for the short test prompt string.
        monkeypatch.setattr(inf, "_CONTEXT_CHECK_BYTE_MARGIN", 10**9)
        full = AsyncMock()
        streamed = MagicMock()
        with (
            patch.object(inf, "_full_completion", full),
            patch.object(inf, "_stream_completion", streamed),
            patch.object(inf, "_acquire_inference_lock") as lock,
        ):
            with pytest.raises(ContextLengthExceededError) as ei:
                await generate_chat(
                    mock_manager,
                    "qwen3",
                    [{"role": "user", "content": "hi"}],
                    stream=stream,
                )
        assert isinstance(ei.value, ValueError)  # -> 400 via the app handler
        assert "150" in str(ei.value) and "100" in str(ei.value)
        full.assert_not_called()
        streamed.assert_not_called()
        lock.assert_not_called()
        assert lm.active_refs == 0  # pin released

    @pytest.mark.asyncio
    async def test_prompt_exactly_at_window_rejected(self, mock_manager, monkeypatch):
        # No room left for even one generated token.
        lm = mock_manager._loaded["qwen3:latest"]
        lm.context_length = 100
        _set_prompt_tokens(lm, 100)
        monkeypatch.setattr(inf, "_CONTEXT_CHECK_BYTE_MARGIN", 10**9)
        with pytest.raises(ContextLengthExceededError):
            await generate_chat(
                mock_manager, "qwen3", [{"role": "user", "content": "hi"}], stream=False
            )

    @pytest.mark.asyncio
    async def test_under_window_proceeds(self, mock_manager, monkeypatch):
        lm = mock_manager._loaded["qwen3:latest"]
        lm.context_length = 100
        _set_prompt_tokens(lm, 99)
        monkeypatch.setattr(inf, "_CONTEXT_CHECK_BYTE_MARGIN", 10**9)
        with patch.object(
            inf, "_full_completion", AsyncMock(return_value={"text": "ok"})
        ) as full:
            result = await generate_chat(
                mock_manager, "qwen3", [{"role": "user", "content": "hi"}], stream=False
            )
        assert result["text"] == "ok"
        full.assert_called_once()

    @pytest.mark.asyncio
    async def test_unknown_window_skips_check(self, mock_manager, monkeypatch):
        lm = mock_manager._loaded["qwen3:latest"]
        lm.context_length = None
        _set_prompt_tokens(lm, 10**6)
        monkeypatch.setattr(inf, "_CONTEXT_CHECK_BYTE_MARGIN", 10**9)
        with patch.object(
            inf, "_full_completion", AsyncMock(return_value={"text": "ok"})
        ):
            result = await generate_chat(
                mock_manager, "qwen3", [{"role": "user", "content": "hi"}], stream=False
            )
        assert result["text"] == "ok"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("stream", [False, True])
    async def test_active_kv_eviction_skips_check(
        self, mock_manager, monkeypatch, stream
    ):
        # StreamingLLM sink+window eviction (#505) exists to serve prompts
        # longer than the nominal window; since #755 both directions reach
        # the bounded cache (chat prompt-cache path, or the attach).
        lm = mock_manager._loaded["qwen3:latest"]
        lm.context_length = 100
        lm.kv_eviction = "4:64"
        lm.kv_eviction_effective = True
        _set_prompt_tokens(lm, 150)
        monkeypatch.setattr(inf, "_CONTEXT_CHECK_BYTE_MARGIN", 10**9)

        async def _mock_stream(*args, **kwargs):
            yield {"text": "ok", "done": True}

        full = AsyncMock(return_value={"text": "ok"})
        stream_fn = MagicMock(side_effect=_mock_stream)
        with (
            patch.object(inf, "_full_completion", full),
            patch.object(inf, "_stream_completion", stream_fn),
        ):
            result = await generate_chat(
                mock_manager, "qwen3", [{"role": "user", "content": "hi"}], stream=stream
            )
        if stream:
            async for _ in result:
                pass
            stream_fn.assert_called_once()
        else:
            assert result["text"] == "ok"
            full.assert_called_once()

    @pytest.mark.asyncio
    async def test_kv_eviction_without_prompt_cache_now_served(
        self, mock_manager, monkeypatch
    ):
        # #755: with the prompt cache off, `_attach_kv_cache` now installs
        # the bounded eviction cache post-preflight (the pre-lock check and
        # the attach share one gate, `_eviction_will_bound`), so the
        # over-window prompt is served instead of rejected.
        lm = mock_manager._loaded["qwen3:latest"]
        lm.context_length = 100
        lm.kv_eviction = "4:64"
        lm.kv_eviction_effective = True
        lm.prompt_cache = False
        _set_prompt_tokens(lm, 150)
        monkeypatch.setattr(inf, "_CONTEXT_CHECK_BYTE_MARGIN", 10**9)
        with patch.object(
            inf, "_full_completion", AsyncMock(return_value={"text": "ok"})
        ) as full:
            result = await generate_chat(
                mock_manager,
                "qwen3",
                [{"role": "user", "content": "hi"}],
                stream=False,
            )
        assert result["text"] == "ok"
        full.assert_called_once()

    @pytest.mark.asyncio
    async def test_speculative_request_not_exempt(self, mock_manager, monkeypatch):
        # Speculative decoders never read prompt_cache, so the window still
        # applies (#755): the attach skips them.
        lm = mock_manager._loaded["qwen3:latest"]
        lm.context_length = 100
        lm.kv_eviction = "4:64"
        lm.kv_eviction_effective = True
        lm.prompt_cache = False
        lm.speculative_decoder = object()  # property is_speculative -> True
        _set_prompt_tokens(lm, 150)
        monkeypatch.setattr(inf, "_CONTEXT_CHECK_BYTE_MARGIN", 10**9)
        with patch.object(inf, "_full_completion", AsyncMock()) as full:
            with pytest.raises(ContextLengthExceededError):
                await generate_chat(
                    mock_manager,
                    "qwen3",
                    [{"role": "user", "content": "hi"}],
                    stream=False,
                )
        full.assert_not_called()

    @pytest.mark.asyncio
    async def test_ineffective_kv_eviction_still_rejected(
        self, mock_manager, monkeypatch
    ):
        # Hybrid/SWA models: _make_eviction_prompt_cache skips eviction and
        # returns the default cache, so the window still applies.
        lm = mock_manager._loaded["qwen3:latest"]
        lm.context_length = 100
        lm.kv_eviction = "4:64"
        lm.kv_eviction_effective = False
        _set_prompt_tokens(lm, 150)
        monkeypatch.setattr(inf, "_CONTEXT_CHECK_BYTE_MARGIN", 10**9)
        with pytest.raises(ContextLengthExceededError):
            await generate_chat(
                mock_manager, "qwen3", [{"role": "user", "content": "hi"}], stream=False
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("stream", [False, True])
    async def test_completion_kv_eviction_now_served(
        self, mock_manager, monkeypatch, stream
    ):
        # #755: the completion surface gets the bounded cache through the
        # attach too, so the over-window prompt is served.
        lm = mock_manager._loaded["qwen3:latest"]
        lm.context_length = 100
        lm.kv_eviction = "4:64"
        lm.kv_eviction_effective = True
        _set_prompt_tokens(lm, 150)
        monkeypatch.setattr(inf, "_CONTEXT_CHECK_BYTE_MARGIN", 10**9)

        async def _mock_stream(*args, **kwargs):
            yield {"text": "ok", "done": True}

        full = AsyncMock(return_value={"text": "ok"})
        stream_fn = MagicMock(side_effect=_mock_stream)
        with (
            patch.object(inf, "_full_completion", full),
            patch.object(inf, "_stream_completion", stream_fn),
        ):
            result = await generate_completion(
                mock_manager, "qwen3", "Hello", stream=stream
            )
        if stream:
            async for _ in result:
                pass
            stream_fn.assert_called_once()
        else:
            assert result["text"] == "ok"
            full.assert_called_once()

    @pytest.mark.asyncio
    async def test_completion_speculative_request_not_exempt(
        self, mock_manager, monkeypatch
    ):
        lm = mock_manager._loaded["qwen3:latest"]
        lm.context_length = 100
        lm.kv_eviction = "4:64"
        lm.kv_eviction_effective = True
        lm.speculative_decoder = object()  # property is_speculative -> True
        _set_prompt_tokens(lm, 150)
        monkeypatch.setattr(inf, "_CONTEXT_CHECK_BYTE_MARGIN", 10**9)
        with patch.object(inf, "_full_completion", AsyncMock()) as full:
            with pytest.raises(ContextLengthExceededError):
                await generate_completion(mock_manager, "qwen3", "Hello", stream=False)
        full.assert_not_called()

    @pytest.mark.asyncio
    async def test_short_prompt_skips_tokenization(self, mock_manager):
        # A prompt whose UTF-8 byte length is well under the window can't
        # exceed it, so no extra tokenization pass is paid for it.
        lm = mock_manager._loaded["qwen3:latest"]
        lm.context_length = 100_000
        lm.prompt_cache = False
        _set_prompt_tokens(lm, 10**6)  # would be rejected if tokenized
        with patch.object(
            inf, "_full_completion", AsyncMock(return_value={"text": "ok"})
        ):
            result = await generate_chat(
                mock_manager, "qwen3", [{"role": "user", "content": "hi"}], stream=False
            )
        assert result["text"] == "ok"
        lm.tokenizer.encode.assert_not_called()


class TestGenerateCompletionRejectsOverWindow:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("stream", [False, True])
    async def test_over_window_rejected(self, mock_manager, monkeypatch, stream):
        lm = mock_manager._loaded["qwen3:latest"]
        lm.context_length = 100
        _set_prompt_tokens(lm, 150)
        monkeypatch.setattr(inf, "_CONTEXT_CHECK_BYTE_MARGIN", 10**9)
        full = AsyncMock()
        streamed = MagicMock()
        with (
            patch.object(inf, "_full_completion", full),
            patch.object(inf, "_stream_completion", streamed),
        ):
            with pytest.raises(ContextLengthExceededError):
                await generate_completion(mock_manager, "qwen3", "Hello", stream=stream)
        full.assert_not_called()
        streamed.assert_not_called()
        assert lm.active_refs == 0

    @pytest.mark.asyncio
    async def test_prior_context_counts_toward_window(self, mock_manager):
        # /api/generate `context` continuation: the prepended prior tokens are
        # prefilled too, so they count against the window.
        lm = mock_manager._loaded["qwen3:latest"]
        lm.context_length = 100
        _set_prompt_tokens(lm, 10)
        with patch.object(inf, "_full_completion", AsyncMock()) as full:
            with pytest.raises(ContextLengthExceededError):
                await generate_completion(
                    mock_manager,
                    "qwen3",
                    "Hello",
                    stream=False,
                    return_context=True,
                    context=list(range(95)),
                )
        full.assert_not_called()


class TestHttpSurface:
    @pytest.mark.asyncio
    async def test_openai_chat_returns_400(self, app_client, mock_manager, monkeypatch):
        lm = mock_manager._loaded["qwen3:latest"]
        lm.context_length = 100
        _set_prompt_tokens(lm, 150)
        monkeypatch.setattr(inf, "_CONTEXT_CHECK_BYTE_MARGIN", 10**9)
        resp = await app_client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen3",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 10,
            },
        )
        assert resp.status_code == 400
        err = resp.json()["error"]
        assert err["code"] == "context_length_exceeded"
        assert "context window" in err["message"]

    @pytest.mark.asyncio
    async def test_ollama_chat_returns_400(self, app_client, mock_manager, monkeypatch):
        lm = mock_manager._loaded["qwen3:latest"]
        lm.context_length = 100
        _set_prompt_tokens(lm, 150)
        monkeypatch.setattr(inf, "_CONTEXT_CHECK_BYTE_MARGIN", 10**9)
        resp = await app_client.post(
            "/api/chat",
            json={
                "model": "qwen3",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_anthropic_messages_returns_400(
        self, app_client, mock_manager, monkeypatch
    ):
        lm = mock_manager._loaded["qwen3:latest"]
        lm.context_length = 100
        _set_prompt_tokens(lm, 150)
        monkeypatch.setattr(inf, "_CONTEXT_CHECK_BYTE_MARGIN", 10**9)
        resp = await app_client.post(
            "/v1/messages",
            json={
                "model": "qwen3",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 400
        assert resp.json()["error"]["type"] == "invalid_request_error"


class TestLoaderRecordsContextLength:
    @pytest.mark.asyncio
    async def test_ensure_loaded_reads_store_config(
        self, registry, mock_store, monkeypatch
    ):
        monkeypatch.setattr(
            "olmlx.engine.model_manager.settings.model_load_timeout", None
        )
        manager = ModelManager(registry, mock_store)
        manager.registry.resolve = MagicMock(  # type: ignore[method-assign]
            return_value=ModelConfig(hf_path="new/repo")
        )
        manager.registry.normalize_name = MagicMock(  # type: ignore[method-assign]
            side_effect=lambda n: f"{n}:latest"
        )
        local = mock_store.local_path("new/repo")
        local.mkdir(parents=True)
        (local / "config.json").write_text(
            json.dumps({"model_type": "qwen2", "max_position_embeddings": 32768})
        )

        tokenizer = MagicMock()
        tokenizer.chat_template = None
        tokenizer.model_max_length = 131072

        def _shard(*args, **kwargs):
            return (MagicMock(), tokenizer, False, TemplateCaps(), False, None)

        monkeypatch.setattr(manager, "_load_model_and_shard", _shard)
        monkeypatch.setattr(manager, "_probe_cache_capabilities", AsyncMock())

        lm = await manager.ensure_loaded("new")
        assert lm.context_length == 131072

    def test_logs_when_tokenizer_widens_config_window(
        self, registry, mock_store, caplog
    ):
        manager = ModelManager(registry, mock_store)
        local = mock_store.local_path("qwen/repo")
        local.mkdir(parents=True)
        (local / "config.json").write_text(
            json.dumps({"max_position_embeddings": 32768})
        )
        tok = SimpleNamespace(model_max_length=131072)
        with caplog.at_level("INFO", logger="olmlx.engine.model_manager"):
            assert manager._read_context_length("qwen/repo", tok, False) == 131072
        assert any(
            "32768" in r.getMessage() and "131072" in r.getMessage()
            for r in caplog.records
        )

    def test_no_log_when_limits_agree(self, registry, mock_store, caplog):
        manager = ModelManager(registry, mock_store)
        local = mock_store.local_path("llama/repo")
        local.mkdir(parents=True)
        (local / "config.json").write_text(
            json.dumps({"max_position_embeddings": 8192})
        )
        tok = SimpleNamespace(model_max_length=8192)
        with caplog.at_level("INFO", logger="olmlx.engine.model_manager"):
            assert manager._read_context_length("llama/repo", tok, False) == 8192
        assert not any("declares" in r.getMessage() for r in caplog.records)

    def test_absolute_local_path_reads_its_own_config(
        self, registry, mock_store, tmp_path
    ):
        # models.json entries may point at a local directory; its config.json
        # lives there, not under the store (mirrors _build_speculative_decoder).
        model_dir = tmp_path / "my-local-model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(
            json.dumps({"max_position_embeddings": 4096})
        )
        manager = ModelManager(registry, mock_store)
        tok = SimpleNamespace(model_max_length=int(1e30))
        assert manager._read_context_length(str(model_dir), tok, False) == 4096

    def test_missing_config_is_none(self, registry, mock_store):
        manager = ModelManager(registry, mock_store)
        assert manager._read_context_length("absent/repo", MagicMock(), False) is None

    def test_vlm_uses_inner_tokenizer(self, registry, mock_store):
        manager = ModelManager(registry, mock_store)
        processor = SimpleNamespace(tokenizer=SimpleNamespace(model_max_length=8192))
        assert manager._read_context_length("absent/repo", processor, True) == 8192

    def test_direct_construction_default_is_unknown(self):
        # Unknown window -> the check is disabled rather than guessed.
        lm = LoadedModel(
            name="x", hf_path="x", model=MagicMock(), tokenizer=MagicMock()
        )
        assert lm.context_length is None


class TestKvEvictionBoundsCache:
    """``_kv_eviction_bounds_cache`` mirrors ``_make_eviction_prompt_cache``."""

    def _lm(self, layers):
        from mlx_lm.models.cache import KVCache  # noqa: F401

        model = SimpleNamespace(make_cache=lambda: layers)
        return LoadedModel(
            name="m",
            hf_path="m",
            model=model,
            tokenizer=MagicMock(),
            kv_eviction="4:64",
        )

    def test_pure_full_attention_is_bounded(self):
        from mlx_lm.models.cache import KVCache

        lm = self._lm([KVCache(), KVCache()])
        assert inf._kv_eviction_bounds_cache(lm) is True
        assert lm.kv_eviction_effective is True  # memoized

    def test_hybrid_is_not_bounded(self):
        from mlx_lm.models.cache import KVCache, RotatingKVCache

        lm = self._lm([KVCache(), RotatingKVCache(max_size=8)])
        assert inf._kv_eviction_bounds_cache(lm) is False

    def test_no_eviction_or_quant(self):
        from mlx_lm.models.cache import KVCache

        lm = self._lm([KVCache()])
        lm.kv_eviction = None
        assert inf._kv_eviction_bounds_cache(lm) is False
        lm.kv_eviction = "4:64"
        lm.kv_cache_quant = "turboquant:4"
        assert inf._kv_eviction_bounds_cache(lm) is False

    def test_vlm_is_not_bounded(self):
        from mlx_lm.models.cache import KVCache

        lm = self._lm([KVCache()])
        lm.is_vlm = True
        assert inf._kv_eviction_bounds_cache(lm) is False

    def test_probe_failure_is_not_bounded(self):
        lm = self._lm([])
        lm.model = SimpleNamespace(make_cache=MagicMock(side_effect=RuntimeError))
        assert inf._kv_eviction_bounds_cache(lm) is False

    def test_probe_failure_is_not_memoized(self):
        # A transient failure must not pin "not bounded" for the model's
        # whole residency — the next request probes again.
        from mlx_lm.models.cache import KVCache

        lm = self._lm([])
        make_cache = MagicMock(side_effect=[RuntimeError, [KVCache()]])
        lm.model = SimpleNamespace(make_cache=make_cache)
        assert inf._kv_eviction_bounds_cache(lm) is False
        assert lm.kv_eviction_effective is None
        assert inf._kv_eviction_bounds_cache(lm) is True


class TestEvictionCacheLossRemediatedByAttach:
    """Before #755, an exempted kv_eviction request whose bounded cache was
    dropped before generation (setup skipped it, or the memory preflight
    popped it) fell back to mlx-lm's unbounded default cache, so the window
    check re-ran under the lock. Since #755 ``_attach_kv_cache`` rebuilds the
    bounded eviction cache in exactly that situation — the request is still
    served by the bounded cache instead of rejected, and the exemption
    decision (``_eviction_will_bound``) can't disagree with the attach.
    """

    def _lm(self, mock_manager):
        lm = mock_manager._loaded["qwen3:latest"]
        lm.context_length = 100
        lm.kv_eviction = "4:64"
        lm.kv_eviction_effective = True
        return lm

    def test_speculation_and_distribution_block_the_bound(self, mock_manager):
        # _eviction_will_bound duplicates the attach's skip conditions.
        lm = self._lm(mock_manager)
        assert inf._eviction_will_bound(lm, will_speculate=False) is True
        assert inf._eviction_will_bound(lm, will_speculate=True) is False
        lm.is_distributed = True
        assert inf._eviction_will_bound(lm, will_speculate=False) is False

    @pytest.mark.asyncio
    async def test_full_completion_rebuilds_bounded_cache_after_preflight_drop(
        self, mock_manager
    ):
        lm = self._lm(mock_manager)
        tokens = list(range(150))
        evict_cache = ["evict"]
        setup = inf._CacheSetupResult(
            prompt=tokens, full_prompt_tokens=tokens, cache_setup_done=True
        )

        async def _preflight(lm_, prompt, max_tokens, gen_kwargs, **kw):
            gen_kwargs.pop("prompt_cache", None)  # memory-pressure drop
            return inf._PreflightResult(prompt=prompt)

        async def _setup(lm_, prompt, gen_kwargs, **kw):
            gen_kwargs["prompt_cache"] = object()
            return setup

        async def _store(*a, **kw):
            return None

        captured: dict = {}

        async def inner_capture(lm_, prompt, mt, gen_kwargs_, *a, **kw):
            # Snapshot the gen_kwargs _full_completion_inner (the generation
            # worker) actually receives — the outer finally pops prompt_cache.
            captured.update(gen_kwargs_)
            return {}

        with (
            patch.object(inf, "_setup_prompt_cache", _setup),
            patch.object(inf, "_kv_cache_preflight_check", _preflight),
            patch.object(inf, "_full_completion_inner", inner_capture),
            patch.object(inf, "_store_prompt_cache_after_generation", _store),
            patch.object(inf, "_batch_eligible", return_value=False),
            patch.object(
                inf,
                "_make_prompt_cache_for_lm",
                MagicMock(return_value=evict_cache),
            ),
        ):
            await inf._full_completion(
                lm,
                tokens,
                10,
                {},
                inf.TimingStats(),
                use_prompt_cache=True,
                prompt_tokens=tokens,
            )
        # The dropped cache was rebuilt as the bounded eviction cache and the
        # request was served — not rejected with a 400.
        assert captured["prompt_cache"] is evict_cache

    @pytest.mark.asyncio
    async def test_stream_completion_rebuilds_bounded_cache_when_setup_skips(
        self, mock_manager
    ):
        """Same remediation on the streaming path: the freed setup cache is
        rebuilt so the exempted request is served, ending with an error-free
        terminal chunk."""
        lm = self._lm(mock_manager)
        tokens = list(range(150))
        evict_cache = ["evict"]

        async def _setup(lm_, prompt, gen_kwargs, **kw):
            # memory-pressure skip: no prompt_cache installed
            return inf._CacheSetupResult(prompt=tokens, full_prompt_tokens=tokens)

        async def _preflight(lm_, prompt, max_tokens, gen_kwargs, **kw):
            return inf._PreflightResult(prompt=prompt)

        from tests.test_prompt_cache import _make_mock_stream, _make_stream_tokens

        mock_stream_fn = MagicMock(
            return_value=_make_mock_stream(
                _make_stream_tokens("Hi", prompt_tokens=150)
            )
        )
        with (
            patch.object(inf, "_setup_prompt_cache", _setup),
            patch.object(inf, "_kv_cache_preflight_check", _preflight),
            patch.object(inf, "async_mlx_stream", mock_stream_fn),
            patch.object(inf, "_batch_eligible", return_value=False),
            patch.object(
                inf,
                "_make_prompt_cache_for_lm",
                MagicMock(return_value=evict_cache),
            ),
        ):
            chunks = []
            async for chunk in inf._stream_completion(
                lm,
                tokens,
                10,
                {},
                inf.TimingStats(),
                use_prompt_cache=True,
                prompt_tokens=tokens,
            ):
                chunks.append(chunk)
        # The attached bounded cache reached mlx-lm's stream_generate.
        assert mock_stream_fn.call_args.kwargs["prompt_cache"] is evict_cache
        assert any(c.get("done") for c in chunks)

    @pytest.mark.asyncio
    async def test_full_completion_under_window_unaffected(self, mock_manager):
        # Sanity: without eviction configured the over-window reject still
        # applies pre-lock; the attach path changes nothing for plain models.
        lm = mock_manager._loaded["qwen3:latest"]
        lm.context_length = 100
        _set_prompt_tokens(lm, 150)
        with patch.object(inf, "_full_completion", AsyncMock()) as full:
            with pytest.raises(ContextLengthExceededError):
                await generate_chat(
                    mock_manager,
                    "qwen3",
                    [{"role": "user", "content": "hi"}],
                    stream=False,
                )
        full.assert_not_called()

