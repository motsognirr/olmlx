"""Tests for prompt caching (KV cache reuse across requests)."""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from olmlx.utils.streaming import CancellableStream, StreamToken


class TestFindCommonPrefix:
    def test_identical(self):
        from olmlx.engine.inference import _find_common_prefix

        assert _find_common_prefix([1, 2, 3], [1, 2, 3]) == 3

    def test_partial_match(self):
        from olmlx.engine.inference import _find_common_prefix

        assert _find_common_prefix([1, 2, 3, 4], [1, 2, 5, 6]) == 2

    def test_no_match(self):
        from olmlx.engine.inference import _find_common_prefix

        assert _find_common_prefix([1, 2, 3], [4, 5, 6]) == 0

    def test_empty_first(self):
        from olmlx.engine.inference import _find_common_prefix

        assert _find_common_prefix([], [1, 2, 3]) == 0

    def test_empty_second(self):
        from olmlx.engine.inference import _find_common_prefix

        assert _find_common_prefix([1, 2, 3], []) == 0

    def test_both_empty(self):
        from olmlx.engine.inference import _find_common_prefix

        assert _find_common_prefix([], []) == 0

    def test_different_lengths_prefix_match(self):
        from olmlx.engine.inference import _find_common_prefix

        assert _find_common_prefix([1, 2], [1, 2, 3, 4, 5]) == 2

    def test_single_element_match(self):
        from olmlx.engine.inference import _find_common_prefix

        assert _find_common_prefix([1], [1]) == 1


class TestTokenizeForCache:
    def test_with_bos_in_prompt(self):
        from olmlx.engine.inference import _tokenize_for_cache

        tokenizer = MagicMock()
        tokenizer.bos_token = "<s>"
        tokenizer.encode = MagicMock(return_value=[1, 2, 3])
        result = _tokenize_for_cache(tokenizer, "<s>hello")
        tokenizer.encode.assert_called_once_with("<s>hello", add_special_tokens=False)
        assert result == [1, 2, 3]

    def test_without_bos_in_prompt(self):
        from olmlx.engine.inference import _tokenize_for_cache

        tokenizer = MagicMock()
        tokenizer.bos_token = "<s>"
        tokenizer.encode = MagicMock(return_value=[1, 2, 3])
        result = _tokenize_for_cache(tokenizer, "hello")
        tokenizer.encode.assert_called_once_with("hello", add_special_tokens=True)
        assert result == [1, 2, 3]

    def test_no_bos_token(self):
        from olmlx.engine.inference import _tokenize_for_cache

        tokenizer = MagicMock(spec=[])  # no bos_token attribute
        tokenizer.encode = MagicMock(return_value=[1, 2, 3])
        result = _tokenize_for_cache(tokenizer, "hello")
        tokenizer.encode.assert_called_once_with("hello", add_special_tokens=True)
        assert result == [1, 2, 3]

    def test_bos_token_is_none(self):
        from olmlx.engine.inference import _tokenize_for_cache

        tokenizer = MagicMock()
        tokenizer.bos_token = None
        tokenizer.encode = MagicMock(return_value=[1, 2, 3])
        result = _tokenize_for_cache(tokenizer, "hello")
        tokenizer.encode.assert_called_once_with("hello", add_special_tokens=True)
        assert result == [1, 2, 3]


def _make_mock_stream(tokens):
    """Create a mock CancellableStream that yields the given StreamTokens."""
    mock_stream = MagicMock(spec=CancellableStream)
    mock_stream.drain_and_join = AsyncMock()
    token_iter = iter(tokens)

    async def anext_impl():
        try:
            return next(token_iter)
        except StopIteration:
            raise StopAsyncIteration

    mock_stream.__aiter__ = lambda self: self
    mock_stream.__anext__ = lambda self: anext_impl()
    return mock_stream


def _make_stream_tokens(*texts, prompt_tokens=10):
    """Create StreamToken objects for testing."""
    return [
        StreamToken(
            text=text,
            token=100 + i,
            prompt_tokens=prompt_tokens,
            generation_tokens=i + 1,
            prompt_tps=100.0,
            generation_tps=50.0,
        )
        for i, text in enumerate(texts)
    ]


class TestCacheCreatedOnFirstRequest:
    @pytest.mark.asyncio
    async def test_fresh_cache_created_and_stored(self, mock_manager):
        """On first streaming chat request, a fresh cache is created and stored after generation."""
        from olmlx.engine.inference import generate_chat
        from olmlx.engine.model_manager import CachedPromptState

        lm = mock_manager._loaded["qwen3:latest"]
        lm.tokenizer.apply_chat_template = MagicMock(return_value="formatted prompt")
        lm.tokenizer.bos_token = None
        lm.tokenizer.encode = MagicMock(return_value=[10, 20, 30, 40, 50])

        tokens = _make_stream_tokens("Hello", " world", prompt_tokens=5)
        mock_stream = _make_mock_stream(tokens)

        mock_make_cache = MagicMock(return_value=[MagicMock()])

        mock_mx = MagicMock()
        with (
            patch("olmlx.engine.inference.mx", mock_mx),
            patch(
                "olmlx.engine.inference.async_mlx_stream",
                return_value=mock_stream,
            ),
            patch(
                "olmlx.engine.inference.make_prompt_cache",
                mock_make_cache,
            ),
            patch("olmlx.engine.inference.settings") as mock_settings,
        ):
            mock_settings.prompt_cache = True
            mock_settings.default_keep_alive = "5m"
            gen = await generate_chat(
                mock_manager,
                "qwen3",
                [{"role": "user", "content": "hi"}],
                stream=True,
            )
            chunks = []
            async for chunk in gen:
                chunks.append(chunk)

        # Cache should have been created
        mock_make_cache.assert_called_once_with(lm.model)
        # After successful generation, cache state should be stored
        assert lm.prompt_cache_state is not None
        assert isinstance(lm.prompt_cache_state, CachedPromptState)
        # Stored tokens should include prompt tokens + generated tokens
        assert lm.prompt_cache_state.tokens[:5] == [10, 20, 30, 40, 50]
        assert len(lm.prompt_cache_state.tokens) == 7  # 5 prompt + 2 generated


class TestCacheReusedOnPrefixMatch:
    @pytest.mark.asyncio
    async def test_cache_trimmed_and_reused(self, mock_manager):
        """When cached tokens share a prefix with new tokens, cache is trimmed and reused."""
        from olmlx.engine.inference import generate_chat
        from olmlx.engine.model_manager import CachedPromptState

        lm = mock_manager._loaded["qwen3:latest"]
        lm.tokenizer.apply_chat_template = MagicMock(return_value="formatted prompt v2")
        lm.tokenizer.bos_token = None

        # Previously cached: 5 prompt tokens + 2 generated tokens
        existing_cache = [MagicMock()]
        lm.prompt_cache_state = CachedPromptState(
            tokens=[10, 20, 30, 40, 50, 100, 101],  # prompt + generated
            cache=existing_cache,
        )

        # New prompt: shares first 5 tokens, adds 3 more
        lm.tokenizer.encode = MagicMock(return_value=[10, 20, 30, 40, 50, 60, 70, 80])

        tokens = _make_stream_tokens("New", " output", prompt_tokens=3)
        mock_stream = _make_mock_stream(tokens)

        mock_trim = MagicMock(return_value=2)

        mock_mx = MagicMock()
        with (
            patch("olmlx.engine.inference.mx", mock_mx),
            patch(
                "olmlx.engine.inference.async_mlx_stream",
                return_value=mock_stream,
            ) as mock_async_stream,
            patch(
                "olmlx.engine.inference.trim_prompt_cache",
                mock_trim,
            ),
            patch("olmlx.engine.inference.settings") as mock_settings,
        ):
            mock_settings.prompt_cache = True
            mock_settings.default_keep_alive = "5m"
            gen = await generate_chat(
                mock_manager,
                "qwen3",
                [{"role": "user", "content": "hi again"}],
                stream=True,
            )
            chunks = []
            async for chunk in gen:
                chunks.append(chunk)

        # Cache should have been trimmed by 2 (generated tokens beyond prefix)
        mock_trim.assert_called_once_with(existing_cache, 2)

        # async_mlx_stream should receive only suffix tokens (3 new ones)
        call_args = mock_async_stream.call_args
        prompt_arg = call_args[1].get("prompt") or call_args[0][2]
        assert prompt_arg == [60, 70, 80]

        # prompt_cache should be passed in kwargs
        assert call_args[1].get("prompt_cache") is existing_cache


class TestCacheMissCreatesFresh:
    @pytest.mark.asyncio
    async def test_no_common_prefix_creates_fresh_cache(self, mock_manager):
        """Completely different prompt creates a fresh cache."""
        from olmlx.engine.inference import generate_chat
        from olmlx.engine.model_manager import CachedPromptState

        lm = mock_manager._loaded["qwen3:latest"]
        lm.tokenizer.apply_chat_template = MagicMock(return_value="different prompt")
        lm.tokenizer.bos_token = None

        # Previously cached: completely different tokens
        old_cache = [MagicMock()]
        lm.prompt_cache_state = CachedPromptState(
            tokens=[99, 98, 97],
            cache=old_cache,
        )

        # New prompt: no common prefix
        lm.tokenizer.encode = MagicMock(return_value=[1, 2, 3])

        tokens = _make_stream_tokens("output", prompt_tokens=3)
        mock_stream = _make_mock_stream(tokens)

        new_cache = [MagicMock()]
        mock_make_cache = MagicMock(return_value=new_cache)

        mock_mx = MagicMock()
        with (
            patch("olmlx.engine.inference.mx", mock_mx),
            patch(
                "olmlx.engine.inference.async_mlx_stream",
                return_value=mock_stream,
            ),
            patch(
                "olmlx.engine.inference.make_prompt_cache",
                mock_make_cache,
            ),
            patch("olmlx.engine.inference.settings") as mock_settings,
        ):
            mock_settings.prompt_cache = True
            mock_settings.default_keep_alive = "5m"
            gen = await generate_chat(
                mock_manager,
                "qwen3",
                [{"role": "user", "content": "hi"}],
                stream=True,
            )
            async for chunk in gen:
                pass

        # Should create fresh cache, not reuse old one
        mock_make_cache.assert_called_once_with(lm.model)
        assert lm.prompt_cache_state is not None
        assert lm.prompt_cache_state.cache is new_cache


class TestCacheInvalidatedOnCancel:
    @pytest.mark.asyncio
    async def test_client_disconnect_clears_cache(self, mock_manager):
        """When streaming is interrupted (client disconnect), cache is invalidated."""
        from olmlx.engine.inference import generate_chat

        lm = mock_manager._loaded["qwen3:latest"]
        lm.tokenizer.apply_chat_template = MagicMock(return_value="prompt")
        lm.tokenizer.bos_token = None
        lm.tokenizer.encode = MagicMock(return_value=[10, 20, 30])

        # Create a stream that will be interrupted
        mock_stream = MagicMock(spec=CancellableStream)
        mock_stream.drain_and_join = AsyncMock()

        call_count = 0

        async def anext_impl():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return StreamToken(
                    text="partial",
                    token=100,
                    prompt_tokens=3,
                    generation_tokens=1,
                    prompt_tps=100.0,
                    generation_tps=50.0,
                )
            # Simulate client disconnect
            raise asyncio.CancelledError()

        mock_stream.__aiter__ = lambda self: self
        mock_stream.__anext__ = lambda self: anext_impl()

        mock_make_cache = MagicMock(return_value=[MagicMock()])

        mock_mx = MagicMock()
        with (
            patch("olmlx.engine.inference.mx", mock_mx),
            patch(
                "olmlx.engine.inference.async_mlx_stream",
                return_value=mock_stream,
            ),
            patch(
                "olmlx.engine.inference.make_prompt_cache",
                mock_make_cache,
            ),
            patch("olmlx.engine.inference.settings") as mock_settings,
        ):
            mock_settings.prompt_cache = True
            mock_settings.default_keep_alive = "5m"
            gen = await generate_chat(
                mock_manager,
                "qwen3",
                [{"role": "user", "content": "hi"}],
                stream=True,
            )
            chunks = []
            try:
                async for chunk in gen:
                    chunks.append(chunk)
            except (asyncio.CancelledError, RuntimeError):
                pass

        # Cache should be invalidated after cancellation
        assert lm.prompt_cache_state is None


class TestCacheClearedOnModelUnload:
    def test_unloaded_model_has_no_cache(self, mock_manager):
        """When a model is unloaded, its cache state is released."""
        from olmlx.engine.model_manager import CachedPromptState

        lm = mock_manager._loaded["qwen3:latest"]
        lm.prompt_cache_state = CachedPromptState(
            tokens=[1, 2, 3],
            cache=[MagicMock()],
        )

        mock_manager.unload("qwen3")
        # LoadedModel is removed from _loaded, so its cache is freed with it
        assert "qwen3:latest" not in mock_manager._loaded


class TestCacheDisabledViaConfig:
    @pytest.mark.asyncio
    async def test_no_cache_when_disabled(self, mock_manager):
        """With OLMLX_PROMPT_CACHE=false, no caching occurs."""
        from olmlx.engine.inference import generate_chat

        lm = mock_manager._loaded["qwen3:latest"]
        lm.tokenizer.apply_chat_template = MagicMock(return_value="prompt")

        tokens = _make_stream_tokens("Hello", prompt_tokens=3)
        mock_stream = _make_mock_stream(tokens)

        mock_mx = MagicMock()
        with (
            patch("olmlx.engine.inference.mx", mock_mx),
            patch(
                "olmlx.engine.inference.async_mlx_stream",
                return_value=mock_stream,
            ) as mock_async_stream,
            patch("olmlx.engine.inference.settings") as mock_settings,
        ):
            mock_settings.prompt_cache = False
            mock_settings.default_keep_alive = "5m"
            gen = await generate_chat(
                mock_manager,
                "qwen3",
                [{"role": "user", "content": "hi"}],
                stream=True,
            )
            async for chunk in gen:
                pass

        # prompt should be passed as string (no tokenization for cache)
        call_args = mock_async_stream.call_args
        prompt_arg = call_args[1].get("prompt") or call_args[0][2]
        assert isinstance(prompt_arg, str)
        # No cache state should be set
        assert lm.prompt_cache_state is None
        # No prompt_cache kwarg
        assert "prompt_cache" not in call_args[1]


class TestVlmSkipsCache:
    @pytest.mark.asyncio
    async def test_vlm_does_not_use_cache(self, mock_manager):
        """VLM models skip prompt caching entirely."""
        from olmlx.engine.inference import generate_chat

        lm = mock_manager._loaded["qwen3:latest"]
        lm.is_vlm = True

        mock_mlx_vlm = MagicMock()
        mock_mlx_vlm.apply_chat_template.return_value = "vlm prompt"

        tokens = _make_stream_tokens("Hello", prompt_tokens=3)
        mock_stream = _make_mock_stream(tokens)

        mock_mx = MagicMock()
        with (
            patch("olmlx.engine.inference.mx", mock_mx),
            patch.dict("sys.modules", {"mlx_vlm": mock_mlx_vlm}),
            patch(
                "olmlx.engine.inference.async_mlx_stream",
                return_value=mock_stream,
            ) as mock_async_stream,
            patch("olmlx.engine.inference.settings") as mock_settings,
        ):
            mock_settings.prompt_cache = True
            mock_settings.default_keep_alive = "5m"
            gen = await generate_chat(
                mock_manager,
                "qwen3",
                [{"role": "user", "content": "describe"}],
                stream=True,
            )
            async for chunk in gen:
                pass

        # prompt should be passed as string (VLM path)
        call_args = mock_async_stream.call_args
        prompt_arg = call_args[1].get("prompt") or call_args[0][2]
        assert isinstance(prompt_arg, str)
        # No prompt_cache kwarg
        assert "prompt_cache" not in call_args[1]
        assert lm.prompt_cache_state is None


class TestCacheTokenCountLogging:
    @pytest.mark.asyncio
    async def test_cache_hit_logged(self, mock_manager, caplog):
        """Cache hit logs the number of reused and new tokens."""
        from olmlx.engine.inference import generate_chat
        from olmlx.engine.model_manager import CachedPromptState

        lm = mock_manager._loaded["qwen3:latest"]
        lm.tokenizer.apply_chat_template = MagicMock(return_value="prompt v2")
        lm.tokenizer.bos_token = None

        # Cached: 5 prompt + 2 generated = 7 tokens
        lm.prompt_cache_state = CachedPromptState(
            tokens=[10, 20, 30, 40, 50, 100, 101],
            cache=[MagicMock()],
        )

        # New prompt: shares 5 token prefix, adds 3 new
        lm.tokenizer.encode = MagicMock(return_value=[10, 20, 30, 40, 50, 60, 70, 80])

        tokens = _make_stream_tokens("out", prompt_tokens=3)
        mock_stream = _make_mock_stream(tokens)

        mock_trim = MagicMock(return_value=2)

        mock_mx = MagicMock()
        with (
            patch("olmlx.engine.inference.mx", mock_mx),
            patch(
                "olmlx.engine.inference.async_mlx_stream",
                return_value=mock_stream,
            ),
            patch(
                "olmlx.engine.inference.trim_prompt_cache",
                mock_trim,
            ),
            patch("olmlx.engine.inference.settings") as mock_settings,
            caplog.at_level(logging.INFO, logger="olmlx.engine.inference"),
        ):
            mock_settings.prompt_cache = True
            mock_settings.default_keep_alive = "5m"
            gen = await generate_chat(
                mock_manager,
                "qwen3",
                [{"role": "user", "content": "hi again"}],
                stream=True,
            )
            async for chunk in gen:
                pass

        assert any("cache hit" in r.message.lower() for r in caplog.records)
        # Should mention reused token count
        assert any(
            "5" in r.message and "reuse" in r.message.lower() for r in caplog.records
        )


class TestCacheStatsInCacheInfoChunk:
    @pytest.mark.asyncio
    async def test_cache_info_chunk_emitted_first(self, mock_manager):
        """A cache_info chunk with cache stats is yielded before streaming tokens."""
        from olmlx.engine.inference import generate_chat
        from olmlx.engine.model_manager import CachedPromptState

        lm = mock_manager._loaded["qwen3:latest"]
        lm.tokenizer.apply_chat_template = MagicMock(return_value="prompt v2")
        lm.tokenizer.bos_token = None

        # Cached: 5 prompt + 2 generated
        lm.prompt_cache_state = CachedPromptState(
            tokens=[10, 20, 30, 40, 50, 100, 101],
            cache=[MagicMock()],
        )

        # New prompt: shares 5 prefix, adds 3 new
        lm.tokenizer.encode = MagicMock(return_value=[10, 20, 30, 40, 50, 60, 70, 80])

        tokens = _make_stream_tokens("out", prompt_tokens=3)
        mock_stream = _make_mock_stream(tokens)

        mock_trim = MagicMock(return_value=2)

        mock_mx = MagicMock()
        with (
            patch("olmlx.engine.inference.mx", mock_mx),
            patch(
                "olmlx.engine.inference.async_mlx_stream",
                return_value=mock_stream,
            ),
            patch(
                "olmlx.engine.inference.trim_prompt_cache",
                mock_trim,
            ),
            patch("olmlx.engine.inference.settings") as mock_settings,
        ):
            mock_settings.prompt_cache = True
            mock_settings.default_keep_alive = "5m"
            gen = await generate_chat(
                mock_manager,
                "qwen3",
                [{"role": "user", "content": "hi again"}],
                stream=True,
            )
            chunks = []
            async for chunk in gen:
                chunks.append(chunk)

        # First chunk should be cache_info
        cache_info = chunks[0]
        assert cache_info.get("cache_info") is True
        assert cache_info["cache_read_tokens"] == 5
        assert cache_info["cache_creation_tokens"] == 3

        # Done chunk should not have cache stats
        done_chunk = chunks[-1]
        assert done_chunk["done"] is True
        assert "cache_read_tokens" not in done_chunk


class TestTokenizeForCacheEmptyBos:
    def test_empty_bos_token_still_adds_special(self):
        """When bos_token is empty string, add_special_tokens should be True."""
        from olmlx.engine.inference import _tokenize_for_cache

        tokenizer = MagicMock()
        tokenizer.bos_token = ""
        tokenizer.encode = MagicMock(return_value=[1, 2, 3])
        result = _tokenize_for_cache(tokenizer, "hello")
        tokenizer.encode.assert_called_once_with("hello", add_special_tokens=True)
        assert result == [1, 2, 3]


class TestCacheExactMatchTrimAlignment:
    @pytest.mark.asyncio
    async def test_exact_match_trims_to_suffix_start(self, mock_manager):
        """When prompt is exact prefix of cached tokens, trim aligns with suffix_start."""
        from olmlx.engine.inference import generate_chat
        from olmlx.engine.model_manager import CachedPromptState

        lm = mock_manager._loaded["qwen3:latest"]
        lm.tokenizer.apply_chat_template = MagicMock(return_value="same prompt")
        lm.tokenizer.bos_token = None

        # Previously cached: 3 prompt tokens + 2 generated
        existing_cache = [MagicMock()]
        lm.prompt_cache_state = CachedPromptState(
            tokens=[10, 20, 30, 100, 101],
            cache=existing_cache,
        )

        # New prompt: exact same 3 tokens (prefix_len == len(prompt_tokens))
        lm.tokenizer.encode = MagicMock(return_value=[10, 20, 30])

        tokens = _make_stream_tokens("out", prompt_tokens=1)
        mock_stream = _make_mock_stream(tokens)

        mock_trim = MagicMock()

        mock_mx = MagicMock()
        with (
            patch("olmlx.engine.inference.mx", mock_mx),
            patch(
                "olmlx.engine.inference.async_mlx_stream",
                return_value=mock_stream,
            ) as mock_async_stream,
            patch(
                "olmlx.engine.inference.trim_prompt_cache",
                mock_trim,
            ),
            patch("olmlx.engine.inference.settings") as mock_settings,
        ):
            mock_settings.prompt_cache = True
            mock_settings.default_keep_alive = "5m"
            gen = await generate_chat(
                mock_manager,
                "qwen3",
                [{"role": "user", "content": "same"}],
                stream=True,
            )
            chunks = []
            async for chunk in gen:
                chunks.append(chunk)

        # suffix_start = min(3, 3-1) = 2
        # Trim should leave 2 entries (not 3), so trim_amount = 5 - 2 = 3
        mock_trim.assert_called_once_with(existing_cache, 3)

        # Only the last token should be sent (suffix_tokens = [30])
        call_args = mock_async_stream.call_args
        prompt_arg = call_args[1].get("prompt") or call_args[0][2]
        assert prompt_arg == [30]

        # Cache stats: cache_read should be suffix_start (2), not prefix_len (3)
        cache_info = chunks[0]
        assert cache_info.get("cache_info") is True
        assert cache_info["cache_read_tokens"] == 2
        assert cache_info["cache_creation_tokens"] == 1


class TestLockReleasedOnCacheInfoDisconnect:
    @pytest.mark.asyncio
    async def test_lock_released_when_generator_closed_at_cache_info(
        self, mock_manager
    ):
        """If client disconnects right after cache_info yield, the lock must still be released."""
        from olmlx.engine.inference import _inference_lock, generate_chat

        lm = mock_manager._loaded["qwen3:latest"]
        lm.tokenizer.apply_chat_template = MagicMock(return_value="prompt")
        lm.tokenizer.bos_token = None
        lm.tokenizer.encode = MagicMock(return_value=[10, 20, 30])

        mock_make_cache = MagicMock(return_value=[MagicMock()])

        mock_mx = MagicMock()
        with (
            patch("olmlx.engine.inference.mx", mock_mx),
            patch(
                "olmlx.engine.inference.async_mlx_stream",
                return_value=_make_mock_stream(
                    _make_stream_tokens("hi", prompt_tokens=3)
                ),
            ),
            patch(
                "olmlx.engine.inference.make_prompt_cache",
                mock_make_cache,
            ),
            patch("olmlx.engine.inference.settings") as mock_settings,
        ):
            mock_settings.prompt_cache = True
            mock_settings.default_keep_alive = "5m"
            gen = await generate_chat(
                mock_manager,
                "qwen3",
                [{"role": "user", "content": "hi"}],
                stream=True,
            )
            # Read only the cache_info chunk, then close (simulates client disconnect)
            first = await gen.__anext__()
            assert first.get("cache_info") is True
            await gen.aclose()

        # Lock must be released — if not, this acquire would deadlock
        acquired = await asyncio.wait_for(_inference_lock.acquire(), timeout=1.0)
        assert acquired
        _inference_lock.release()


class TestConfigPromptCacheSetting:
    def test_default_enabled(self, monkeypatch):
        monkeypatch.delenv("OLMLX_PROMPT_CACHE", raising=False)
        from olmlx.config import Settings

        s = Settings()
        assert s.prompt_cache is True

    def test_disabled_via_env(self, monkeypatch):
        monkeypatch.setenv("OLMLX_PROMPT_CACHE", "false")
        from olmlx.config import Settings

        s = Settings()
        assert s.prompt_cache is False
