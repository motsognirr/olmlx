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
