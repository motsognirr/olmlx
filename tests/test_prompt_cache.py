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
    mock_stream._thread = None  # No thread — normal completion path
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
            mock_settings.prompt_cache_max_tokens = 32768
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
        assert lm.prompt_cache_store.get("") is not None
        assert isinstance(lm.prompt_cache_store.get(""), CachedPromptState)
        # Stored tokens should include prompt tokens + generated tokens
        assert lm.prompt_cache_store.get("").tokens[:5] == [10, 20, 30, 40, 50]
        assert len(lm.prompt_cache_store.get("").tokens) == 7  # 5 prompt + 2 generated


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
        lm.prompt_cache_store.set(
            "",
            CachedPromptState(
                tokens=[10, 20, 30, 40, 50, 100, 101],  # prompt + generated
                cache=existing_cache,
            ),
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
            mock_settings.prompt_cache_max_tokens = 32768
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
        lm.prompt_cache_store.set(
            "",
            CachedPromptState(
                tokens=[99, 98, 97],
                cache=old_cache,
            ),
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
            mock_settings.prompt_cache_max_tokens = 32768
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
        assert lm.prompt_cache_store.get("") is not None
        assert lm.prompt_cache_store.get("").cache is new_cache


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
        mock_stream._thread = None  # No thread — normal completion path

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
            mock_settings.prompt_cache_max_tokens = 32768
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
        assert lm.prompt_cache_store.get("") is None


class TestCacheClearedOnModelUnload:
    def test_unloaded_model_has_no_cache(self, mock_manager):
        """When a model is unloaded, its cache state is released."""
        from olmlx.engine.model_manager import CachedPromptState

        lm = mock_manager._loaded["qwen3:latest"]
        lm.prompt_cache_store.set(
            "",
            CachedPromptState(
                tokens=[1, 2, 3],
                cache=[MagicMock()],
            ),
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
        assert lm.prompt_cache_store.get("") is None
        # No prompt_cache kwarg
        assert "prompt_cache" not in call_args[1]


class TestVlmUsesCache:
    def _setup_vlm(self, mock_manager):
        """Set up a VLM model for cache testing."""
        lm = mock_manager._loaded["qwen3:latest"]
        lm.is_vlm = True
        # _get_model_for_cache accesses lm.model.language_model, not lm.language_model
        lm.model.language_model = MagicMock()
        lm.model.language_model.layers = [None] * 32
        lm.tokenizer.tokenizer = MagicMock()
        lm.tokenizer.tokenizer.chat_template = "{{ messages }}{{ tools }}"
        lm.tokenizer.tokenizer.bos_token = None
        lm.tokenizer.tokenizer.encode = MagicMock(return_value=[10, 20, 30, 40, 50])
        lm.tokenizer.apply_chat_template = MagicMock(return_value="formatted prompt")
        lm.model.config = {}
        return lm

    @pytest.mark.asyncio
    async def test_vlm_uses_cache(self, mock_manager):
        """VLM models use prompt caching with language_model."""
        from olmlx.engine.inference import generate_chat

        lm = self._setup_vlm(mock_manager)

        mock_mlx_vlm = MagicMock()
        mock_mlx_vlm.apply_chat_template.return_value = "vlm prompt"

        tokens = _make_stream_tokens("Hello", " world", prompt_tokens=5)
        mock_stream = _make_mock_stream(tokens)

        mock_make_cache = MagicMock(return_value=[MagicMock()])

        mock_mx = MagicMock()
        with (
            patch("olmlx.engine.inference.mx", mock_mx),
            patch.dict("sys.modules", {"mlx_vlm": mock_mlx_vlm}),
            patch(
                "olmlx.engine.inference.async_mlx_stream",
                return_value=mock_stream,
            ),
            patch("olmlx.engine.inference.make_prompt_cache", mock_make_cache),
            patch("olmlx.engine.inference.settings") as mock_settings,
        ):
            mock_settings.prompt_cache = True
            mock_settings.prompt_cache_max_tokens = 32768
            mock_settings.default_keep_alive = "5m"
            gen = await generate_chat(
                mock_manager,
                "qwen3",
                [{"role": "user", "content": "describe"}],
                stream=True,
            )
            async for chunk in gen:
                pass

        # Cache should have been created with language_model (not the full VLM model)
        call_args = mock_make_cache.call_args
        assert call_args is not None, "make_prompt_cache was not called"
        assert call_args[0][0] is lm.model.language_model
        assert lm.prompt_cache_store.get("") is not None

    @pytest.mark.asyncio
    async def test_vlm_passes_input_ids_not_token_list(self, mock_manager):
        """VLM cache passes input_ids kwarg instead of overwriting prompt with tokens.

        mlx_vlm.stream_generate expects prompt as str; passing a list[int]
        causes ValueError in prepare_inputs. The fix is to pass pre-tokenized
        tokens via the input_ids kwarg which bypasses prepare_inputs.
        """
        from olmlx.engine.inference import generate_chat

        self._setup_vlm(mock_manager)

        mock_mlx_vlm = MagicMock()
        mock_mlx_vlm.apply_chat_template.return_value = "vlm prompt"

        tokens = _make_stream_tokens("Hello", prompt_tokens=5)
        mock_stream = _make_mock_stream(tokens)

        mock_make_cache = MagicMock(return_value=[MagicMock()])

        mock_mx = MagicMock()
        with (
            patch("olmlx.engine.inference.mx", mock_mx),
            patch.dict("sys.modules", {"mlx_vlm": mock_mlx_vlm}),
            patch(
                "olmlx.engine.inference.async_mlx_stream",
                return_value=mock_stream,
            ) as mock_async_stream,
            patch("olmlx.engine.inference.make_prompt_cache", mock_make_cache),
            patch("olmlx.engine.inference.settings") as mock_settings,
        ):
            mock_settings.prompt_cache = True
            mock_settings.prompt_cache_max_tokens = 32768
            mock_settings.default_keep_alive = "5m"
            gen = await generate_chat(
                mock_manager,
                "qwen3",
                [{"role": "user", "content": "describe"}],
                stream=True,
            )
            async for chunk in gen:
                pass

        # Verify prompt is still a string (not overwritten with token list)
        call_args = mock_async_stream.call_args
        prompt_arg = (
            call_args[0][2] if len(call_args[0]) > 2 else call_args[1]["prompt"]
        )
        assert isinstance(prompt_arg, str), (
            f"VLM prompt should be a string, got {type(prompt_arg)}"
        )
        # input_ids should be passed as a kwarg
        assert "input_ids" in call_args[1], (
            "VLM cache should pass input_ids kwarg to bypass prepare_inputs"
        )


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
        lm.prompt_cache_store.set(
            "",
            CachedPromptState(
                tokens=[10, 20, 30, 40, 50, 100, 101],
                cache=[MagicMock()],
            ),
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
            mock_settings.prompt_cache_max_tokens = 32768
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
        lm.prompt_cache_store.set(
            "",
            CachedPromptState(
                tokens=[10, 20, 30, 40, 50, 100, 101],
                cache=[MagicMock()],
            ),
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
            mock_settings.prompt_cache_max_tokens = 32768
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
    def test_empty_bos_matches_stream_generate(self):
        """When bos_token is empty string, match stream_generate: add_special_tokens=False.

        stream_generate uses `bos_token is None`, not `not bos_token`, so empty
        string falls through to `not prompt.startswith("")` which is always False.
        We must match exactly to avoid token sequence divergence and cache misses.
        """
        from olmlx.engine.inference import _tokenize_for_cache

        tokenizer = MagicMock()
        tokenizer.bos_token = ""
        tokenizer.encode = MagicMock(return_value=[1, 2, 3])
        result = _tokenize_for_cache(tokenizer, "hello")
        tokenizer.encode.assert_called_once_with("hello", add_special_tokens=False)
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
        lm.prompt_cache_store.set(
            "",
            CachedPromptState(
                tokens=[10, 20, 30, 100, 101],
                cache=existing_cache,
            ),
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
            mock_settings.prompt_cache_max_tokens = 32768
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

        # Cache stats: cache_read should be suffix_start (2) — the number of
        # tokens whose KV entries are actually reused from cache.  The token at
        # suffix_start is re-processed by stream_generate, so its KV is not
        # served from cache.
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
            mock_settings.prompt_cache_max_tokens = 32768
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


class TestNoneTokenWarning:
    @pytest.mark.asyncio
    async def test_none_token_logged_not_silently_dropped(self, mock_manager, caplog):
        """When token.token is None, a debug warning is logged."""
        from olmlx.engine.inference import generate_chat

        lm = mock_manager._loaded["qwen3:latest"]
        lm.tokenizer.apply_chat_template = MagicMock(return_value="prompt")
        lm.tokenizer.bos_token = None
        lm.tokenizer.encode = MagicMock(return_value=[10, 20, 30])

        # Create tokens where one has token=None
        tokens_with_none = [
            StreamToken(
                text="Hello",
                token=100,
                prompt_tokens=3,
                generation_tokens=1,
                prompt_tps=100.0,
                generation_tps=50.0,
            ),
            StreamToken(
                text=" world",
                token=None,  # EOS or missing token
                prompt_tokens=3,
                generation_tokens=2,
                prompt_tps=100.0,
                generation_tps=50.0,
            ),
        ]
        mock_stream = _make_mock_stream(tokens_with_none)

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
            caplog.at_level(logging.DEBUG, logger="olmlx.engine.inference"),
        ):
            mock_settings.prompt_cache = True
            mock_settings.prompt_cache_max_tokens = 32768
            mock_settings.default_keep_alive = "5m"
            gen = await generate_chat(
                mock_manager,
                "qwen3",
                [{"role": "user", "content": "hi"}],
                stream=True,
            )
            async for chunk in gen:
                pass

        # Should log a warning about the missing/None token ID
        assert any(
            "skipping token" in r.message.lower() or "token id" in r.message.lower()
            for r in caplog.records
        ), (
            f"Expected warning about None token ID, got: {[r.message for r in caplog.records]}"
        )


class TestSingleTokenPromptCacheEdgeCase:
    @pytest.mark.asyncio
    async def test_single_token_prompt_treated_as_miss(self, mock_manager):
        """Single-token prompt with cache hit should not wastefully trim entire cache."""
        from olmlx.engine.inference import generate_chat
        from olmlx.engine.model_manager import CachedPromptState

        lm = mock_manager._loaded["qwen3:latest"]
        lm.tokenizer.apply_chat_template = MagicMock(return_value="a")
        lm.tokenizer.bos_token = None

        # Previously cached: starts with same token
        existing_cache = [MagicMock()]
        lm.prompt_cache_store.set(
            "",
            CachedPromptState(
                tokens=[10, 100, 101],  # [prompt_token, gen1, gen2]
                cache=existing_cache,
            ),
        )

        # New prompt: single token that matches
        lm.tokenizer.encode = MagicMock(return_value=[10])

        tokens = _make_stream_tokens("out", prompt_tokens=1)
        mock_stream = _make_mock_stream(tokens)

        mock_make_cache = MagicMock(return_value=[MagicMock()])
        mock_trim = MagicMock()

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
            patch(
                "olmlx.engine.inference.trim_prompt_cache",
                mock_trim,
            ),
            patch("olmlx.engine.inference.settings") as mock_settings,
        ):
            mock_settings.prompt_cache = True
            mock_settings.prompt_cache_max_tokens = 32768
            mock_settings.default_keep_alive = "5m"
            gen = await generate_chat(
                mock_manager,
                "qwen3",
                [{"role": "user", "content": "a"}],
                stream=True,
            )
            chunks = []
            async for chunk in gen:
                chunks.append(chunk)

        # Should create fresh cache rather than wastefully trimming entire existing cache
        mock_make_cache.assert_called_once()
        # Should NOT have trimmed (fresh cache instead)
        mock_trim.assert_not_called()


class TestCacheTrimmedWhenExceedsTokenLimit:
    @pytest.mark.asyncio
    async def test_cache_trimmed_when_over_limit(self, mock_manager):
        """When stored tokens exceed prompt_cache_max_tokens, cache is trimmed to the limit."""
        from olmlx.engine.inference import generate_chat
        from olmlx.engine.model_manager import CachedPromptState

        lm = mock_manager._loaded["qwen3:latest"]
        lm.tokenizer.apply_chat_template = MagicMock(return_value="prompt")
        lm.tokenizer.bos_token = None
        lm.tokenizer.encode = MagicMock(return_value=[10, 20, 30, 40, 50])

        # 5 prompt tokens + 2 generated = 7, but limit is 6
        tokens = _make_stream_tokens("Hello", " world", prompt_tokens=5)
        mock_stream = _make_mock_stream(tokens)

        mock_cache_obj = [MagicMock()]
        mock_make_cache = MagicMock(return_value=mock_cache_obj)
        mock_trim = MagicMock()

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
            patch(
                "olmlx.engine.inference.trim_prompt_cache",
                mock_trim,
            ),
            patch("olmlx.engine.inference.settings") as mock_settings,
        ):
            mock_settings.prompt_cache = True
            mock_settings.prompt_cache_max_tokens = 6
            mock_settings.default_keep_alive = "5m"
            gen = await generate_chat(
                mock_manager,
                "qwen3",
                [{"role": "user", "content": "hi"}],
                stream=True,
            )
            async for chunk in gen:
                pass

        # 5 prompt + 2 generated = 7 > limit of 6 → cache should be trimmed, not invalidated
        assert lm.prompt_cache_store.get("") is not None
        assert isinstance(lm.prompt_cache_store.get(""), CachedPromptState)
        assert len(lm.prompt_cache_store.get("").tokens) == 6
        # trim_prompt_cache called with trim_amount = 7 - 6 = 1
        mock_trim.assert_called_once_with(mock_cache_obj, 1)

    @pytest.mark.asyncio
    async def test_trimmed_tokens_are_prefix(self, mock_manager):
        """After trimming, stored tokens contain exactly the first max_cache_tokens tokens."""
        from olmlx.engine.inference import generate_chat

        lm = mock_manager._loaded["qwen3:latest"]
        lm.tokenizer.apply_chat_template = MagicMock(return_value="prompt")
        lm.tokenizer.bos_token = None
        lm.tokenizer.encode = MagicMock(return_value=[10, 20, 30, 40, 50])

        # 5 prompt tokens + 2 generated (token ids 100, 101) = 7 total
        # stored_tokens = [10, 20, 30, 40, 50, 100, 101]
        # After trim to limit=4: [10, 20, 30, 40]
        tokens = _make_stream_tokens("Hello", " world", prompt_tokens=5)
        mock_stream = _make_mock_stream(tokens)

        mock_cache_obj = [MagicMock()]
        mock_make_cache = MagicMock(return_value=mock_cache_obj)
        mock_trim = MagicMock()

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
            patch(
                "olmlx.engine.inference.trim_prompt_cache",
                mock_trim,
            ),
            patch("olmlx.engine.inference.settings") as mock_settings,
        ):
            mock_settings.prompt_cache = True
            mock_settings.prompt_cache_max_tokens = 4
            mock_settings.default_keep_alive = "5m"
            gen = await generate_chat(
                mock_manager,
                "qwen3",
                [{"role": "user", "content": "hi"}],
                stream=True,
            )
            async for chunk in gen:
                pass

        # Stored tokens should be first 4 of [10, 20, 30, 40, 50, 100, 101]
        assert lm.prompt_cache_store.get("").tokens == [10, 20, 30, 40]
        # trim_amount = 7 - 4 = 3
        mock_trim.assert_called_once_with(mock_cache_obj, 3)

    @pytest.mark.asyncio
    async def test_trimmed_cache_reusable_as_prefix(self, mock_manager):
        """After trimming, a longer prompt sharing the trimmed prefix reuses the cache."""
        from olmlx.engine.inference import generate_chat

        lm = mock_manager._loaded["qwen3:latest"]
        lm.tokenizer.apply_chat_template = MagicMock(return_value="prompt")
        lm.tokenizer.bos_token = None
        # First request: 5 prompt tokens
        lm.tokenizer.encode = MagicMock(return_value=[10, 20, 30, 40, 50])

        mock_cache_obj = [MagicMock()]
        mock_make_cache = MagicMock(return_value=mock_cache_obj)
        mock_trim = MagicMock()

        # First request: 5 prompt + 2 generated = 7 > limit of 5 → trimmed to 5
        tokens1 = _make_stream_tokens("Hello", " world", prompt_tokens=5)
        mock_stream1 = _make_mock_stream(tokens1)

        mock_mx = MagicMock()
        with (
            patch("olmlx.engine.inference.mx", mock_mx),
            patch(
                "olmlx.engine.inference.async_mlx_stream",
                return_value=mock_stream1,
            ),
            patch(
                "olmlx.engine.inference.make_prompt_cache",
                mock_make_cache,
            ),
            patch(
                "olmlx.engine.inference.trim_prompt_cache",
                mock_trim,
            ),
            patch("olmlx.engine.inference.settings") as mock_settings,
        ):
            mock_settings.prompt_cache = True
            mock_settings.prompt_cache_max_tokens = 5
            mock_settings.default_keep_alive = "5m"
            gen = await generate_chat(
                mock_manager,
                "qwen3",
                [{"role": "user", "content": "hi"}],
                stream=True,
            )
            async for chunk in gen:
                pass

        # After first request: cache trimmed to 5 tokens [10, 20, 30, 40, 50]
        assert lm.prompt_cache_store.get("") is not None
        assert lm.prompt_cache_store.get("").tokens == [10, 20, 30, 40, 50]

        # Second request: longer prompt that shares the trimmed prefix
        # [10, 20, 30, 40, 50, 60, 70] — first 5 tokens match the trimmed cache
        lm.tokenizer.encode = MagicMock(return_value=[10, 20, 30, 40, 50, 60, 70])
        tokens2 = _make_stream_tokens("Again", prompt_tokens=7)
        mock_stream2 = _make_mock_stream(tokens2)
        mock_make_cache.reset_mock()
        mock_trim.reset_mock()

        with (
            patch("olmlx.engine.inference.mx", mock_mx),
            patch(
                "olmlx.engine.inference.async_mlx_stream",
                return_value=mock_stream2,
            ),
            patch(
                "olmlx.engine.inference.make_prompt_cache",
                mock_make_cache,
            ),
            patch(
                "olmlx.engine.inference.trim_prompt_cache",
                mock_trim,
            ),
            patch("olmlx.engine.inference.settings") as mock_settings,
        ):
            mock_settings.prompt_cache = True
            mock_settings.prompt_cache_max_tokens = 20
            mock_settings.default_keep_alive = "5m"
            gen = await generate_chat(
                mock_manager,
                "qwen3",
                [{"role": "user", "content": "hi there"}],
                stream=True,
            )
            async for chunk in gen:
                pass

        # Should NOT have created a fresh cache — reused trimmed prefix
        mock_make_cache.assert_not_called()

    @pytest.mark.asyncio
    async def test_trim_uses_actual_generation_steps(self, mock_manager):
        """Trim amount accounts for None-ID tokens in KV cache depth."""
        from olmlx.engine.inference import generate_chat
        from olmlx.engine.model_manager import CachedPromptState

        lm = mock_manager._loaded["qwen3:latest"]
        lm.tokenizer.apply_chat_template = MagicMock(return_value="prompt")
        lm.tokenizer.bos_token = None
        lm.tokenizer.encode = MagicMock(return_value=[10, 20, 30])

        # 3 generation steps, but token at step 2 is None
        # KV cache depth = 3 prompt + 3 steps = 6
        # stored_tokens = [10, 20, 30] + [100, 102] = 5 (skips None)
        stream_tokens = [
            StreamToken(
                text="Hello",
                token=100,
                prompt_tokens=3,
                generation_tokens=1,
                prompt_tps=100.0,
                generation_tps=50.0,
            ),
            StreamToken(
                text="",
                token=None,
                prompt_tokens=3,
                generation_tokens=2,
                prompt_tps=100.0,
                generation_tps=50.0,
            ),
            StreamToken(
                text=" world",
                token=102,
                prompt_tokens=3,
                generation_tokens=3,
                prompt_tps=100.0,
                generation_tps=50.0,
            ),
        ]
        mock_stream = _make_mock_stream(stream_tokens)

        mock_cache_obj = [MagicMock()]
        mock_make_cache = MagicMock(return_value=mock_cache_obj)
        mock_trim = MagicMock()

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
            patch(
                "olmlx.engine.inference.trim_prompt_cache",
                mock_trim,
            ),
            patch("olmlx.engine.inference.settings") as mock_settings,
        ):
            mock_settings.prompt_cache = True
            mock_settings.prompt_cache_max_tokens = 4
            mock_settings.default_keep_alive = "5m"
            gen = await generate_chat(
                mock_manager,
                "qwen3",
                [{"role": "user", "content": "hi"}],
                stream=True,
            )
            async for chunk in gen:
                pass

        # actual_total = 3 prompt + 3 generation steps = 6 > limit 4
        # First trim: trim_amount = 6 - 4 = 2, KV depth → 4
        # Second trim: None-ID detected, trim extra 1 gen step KV entry
        # to align KV depth with stored prompt tokens (3)
        from unittest.mock import call

        assert mock_trim.call_count == 2
        assert mock_trim.call_args_list[0] == call(mock_cache_obj, 2)
        # extra = max_cache_tokens(4) - len(prompt)(3) = 1
        assert mock_trim.call_args_list[1] == call(mock_cache_obj, 1)
        assert lm.prompt_cache_store.get("") is not None
        assert isinstance(lm.prompt_cache_store.get(""), CachedPromptState)
        # Only prompt tokens stored — KV depth matches len(stored_tokens)
        assert lm.prompt_cache_store.get("").tokens == [10, 20, 30]

    @pytest.mark.asyncio
    async def test_trim_none_id_limit_smaller_than_prompt(self, mock_manager):
        """When max_cache_tokens < len(prompt), only a prefix of prompt is stored."""
        from olmlx.engine.inference import generate_chat
        from olmlx.engine.model_manager import CachedPromptState

        lm = mock_manager._loaded["qwen3:latest"]
        lm.tokenizer.apply_chat_template = MagicMock(return_value="prompt")
        lm.tokenizer.bos_token = None
        lm.tokenizer.encode = MagicMock(return_value=[10, 20, 30, 40, 50])

        # 5 prompt + 3 gen steps (all None-ID) = 8, limit = 3
        stream_tokens = [
            StreamToken(
                text="",
                token=None,
                prompt_tokens=5,
                generation_tokens=i + 1,
                prompt_tps=100.0,
                generation_tps=50.0,
            )
            for i in range(3)
        ]
        # Need at least one non-None token for text output
        stream_tokens[-1] = StreamToken(
            text="x",
            token=200,
            prompt_tokens=5,
            generation_tokens=3,
            prompt_tps=100.0,
            generation_tps=50.0,
        )
        mock_stream = _make_mock_stream(stream_tokens)

        mock_cache_obj = [MagicMock()]
        mock_make_cache = MagicMock(return_value=mock_cache_obj)
        mock_trim = MagicMock()

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
            patch(
                "olmlx.engine.inference.trim_prompt_cache",
                mock_trim,
            ),
            patch("olmlx.engine.inference.settings") as mock_settings,
        ):
            mock_settings.prompt_cache = True
            mock_settings.prompt_cache_max_tokens = 3
            mock_settings.default_keep_alive = "5m"
            gen = await generate_chat(
                mock_manager,
                "qwen3",
                [{"role": "user", "content": "hi"}],
                stream=True,
            )
            async for chunk in gen:
                pass

        # actual_total = 5 + 3 = 8 > limit 3
        # trim_amount = 8 - 3 = 5 → KV depth = 3 (pure prompt prefix)
        # extra = 3 - 5 = -2 → no second trim needed
        # stored_tokens = [10, 20, 30, 40, 50][:3] = [10, 20, 30]
        mock_trim.assert_called_once_with(mock_cache_obj, 5)
        assert lm.prompt_cache_store.get("") is not None
        assert isinstance(lm.prompt_cache_store.get(""), CachedPromptState)
        assert lm.prompt_cache_store.get("").tokens == [10, 20, 30]

    @pytest.mark.asyncio
    async def test_cache_invalidated_on_trim_exception(self, mock_manager):
        """If trim_prompt_cache raises, cache is invalidated but response completes."""
        from olmlx.engine.inference import generate_chat

        lm = mock_manager._loaded["qwen3:latest"]
        lm.tokenizer.apply_chat_template = MagicMock(return_value="prompt")
        lm.tokenizer.bos_token = None
        lm.tokenizer.encode = MagicMock(return_value=[10, 20, 30, 40, 50])

        tokens = _make_stream_tokens("Hello", " world", prompt_tokens=5)
        mock_stream = _make_mock_stream(tokens)

        mock_cache_obj = [MagicMock()]
        mock_make_cache = MagicMock(return_value=mock_cache_obj)
        mock_trim = MagicMock(side_effect=RuntimeError("trim failed"))

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
            patch(
                "olmlx.engine.inference.trim_prompt_cache",
                mock_trim,
            ),
            patch("olmlx.engine.inference.settings") as mock_settings,
        ):
            mock_settings.prompt_cache = True
            mock_settings.prompt_cache_max_tokens = 6
            mock_settings.default_keep_alive = "5m"
            # Trim failure is post-generation bookkeeping — should not kill
            # the response. Generation completes, final done chunk emitted.
            chunks = []
            gen = await generate_chat(
                mock_manager,
                "qwen3",
                [{"role": "user", "content": "hi"}],
                stream=True,
            )
            async for chunk in gen:
                chunks.append(chunk)

        # Response should complete with a final done chunk
        assert chunks[-1]["done"] is True
        # Cache should be invalidated, not left in corrupted state
        assert lm.prompt_cache_store.get("") is None


class TestCacheStoredWhenWithinTokenLimit:
    @pytest.mark.asyncio
    async def test_cache_stored_when_under_limit(self, mock_manager):
        """When stored tokens are within prompt_cache_max_tokens, cache is stored normally."""
        from olmlx.engine.inference import generate_chat
        from olmlx.engine.model_manager import CachedPromptState

        lm = mock_manager._loaded["qwen3:latest"]
        lm.tokenizer.apply_chat_template = MagicMock(return_value="prompt")
        lm.tokenizer.bos_token = None
        lm.tokenizer.encode = MagicMock(return_value=[10, 20, 30, 40, 50])

        # 5 prompt + 2 generated = 7, limit is 20
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
            mock_settings.prompt_cache_max_tokens = 20
            mock_settings.default_keep_alive = "5m"
            gen = await generate_chat(
                mock_manager,
                "qwen3",
                [{"role": "user", "content": "hi"}],
                stream=True,
            )
            async for chunk in gen:
                pass

        # 7 tokens <= 20 limit → cache should be stored
        assert lm.prompt_cache_store.get("") is not None
        assert isinstance(lm.prompt_cache_store.get(""), CachedPromptState)
        assert len(lm.prompt_cache_store.get("").tokens) == 7


class TestCacheStoredWhenLimitDisabled:
    @pytest.mark.asyncio
    async def test_cache_stored_when_limit_is_none(self, mock_manager):
        """When prompt_cache_max_tokens is None, cache is stored regardless of size."""
        from olmlx.engine.inference import generate_chat
        from olmlx.engine.model_manager import CachedPromptState

        lm = mock_manager._loaded["qwen3:latest"]
        lm.tokenizer.apply_chat_template = MagicMock(return_value="prompt")
        lm.tokenizer.bos_token = None
        lm.tokenizer.encode = MagicMock(return_value=list(range(100)))

        # 100 prompt + 2 generated = 102 tokens, but no limit
        tokens = _make_stream_tokens("Hello", " world", prompt_tokens=100)
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
            mock_settings.prompt_cache_max_tokens = None
            mock_settings.default_keep_alive = "5m"
            gen = await generate_chat(
                mock_manager,
                "qwen3",
                [{"role": "user", "content": "hi"}],
                stream=True,
            )
            async for chunk in gen:
                pass

        # No limit → cache stored regardless
        assert lm.prompt_cache_store.get("") is not None
        assert isinstance(lm.prompt_cache_store.get(""), CachedPromptState)
        assert len(lm.prompt_cache_store.get("").tokens) == 102


class TestCacheSkippedOnMemoryPressure:
    @pytest.mark.asyncio
    async def test_cache_invalidated_on_high_memory(self, mock_manager):
        """When memory pressure is high, cache is invalidated and no prompt_cache kwarg passed."""
        from olmlx.engine.inference import generate_chat

        lm = mock_manager._loaded["qwen3:latest"]
        lm.tokenizer.apply_chat_template = MagicMock(return_value="prompt")
        lm.tokenizer.bos_token = None
        lm.tokenizer.encode = MagicMock(return_value=[10, 20, 30])

        # Set up existing cache that should be invalidated
        from olmlx.engine.model_manager import CachedPromptState

        lm.prompt_cache_store.set(
            "", CachedPromptState(tokens=[10, 20, 30], cache=[MagicMock()])
        )

        tokens = _make_stream_tokens("Hello", prompt_tokens=3)
        mock_stream = _make_mock_stream(tokens)

        mock_mx = MagicMock()
        with (
            patch("olmlx.engine.inference.mx", mock_mx),
            patch(
                "olmlx.engine.inference.async_mlx_stream",
                return_value=mock_stream,
            ) as mock_async_stream,
            patch(
                "olmlx.engine.inference.make_prompt_cache",
                MagicMock(return_value=[MagicMock()]),
            ),
            patch("olmlx.engine.inference.settings") as mock_settings,
            patch(
                "olmlx.engine.inference._is_memory_pressure_high",
                return_value=True,
            ),
        ):
            mock_settings.prompt_cache = True
            mock_settings.prompt_cache_max_tokens = 32768
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

        # No prompt_cache should be passed to async_mlx_stream
        call_args = mock_async_stream.call_args
        assert "prompt_cache" not in call_args[1]
        # No cache_info chunk should be emitted
        assert not any(c.get("cache_info") for c in chunks)
        # Cache should be invalidated
        assert lm.prompt_cache_store.get("") is None


class TestCacheRebuiltAfterPressureResolves:
    @pytest.mark.asyncio
    async def test_fresh_cache_built_when_pressure_clears(self, mock_manager):
        """When memory pressure resolves after clearing, a fresh cache is built."""
        from olmlx.engine.inference import generate_chat

        lm = mock_manager._loaded["qwen3:latest"]
        lm.tokenizer.apply_chat_template = MagicMock(return_value="prompt")
        lm.tokenizer.bos_token = None
        lm.tokenizer.encode = MagicMock(return_value=[10, 20, 30])

        from olmlx.engine.model_manager import CachedPromptState

        lm.prompt_cache_store.set(
            "", CachedPromptState(tokens=[10, 20, 30], cache=[MagicMock()])
        )

        tokens = _make_stream_tokens("Hello", prompt_tokens=3)
        mock_stream = _make_mock_stream(tokens)

        mock_make_cache = MagicMock(return_value=[MagicMock()])

        # First call returns True (pressure high), second returns False (resolved)
        pressure_returns = iter([True, False])

        mock_mx = MagicMock()
        with (
            patch("olmlx.engine.inference.mx", mock_mx),
            patch(
                "olmlx.engine.inference.async_mlx_stream",
                return_value=mock_stream,
            ) as mock_async_stream,
            patch(
                "olmlx.engine.inference.make_prompt_cache",
                mock_make_cache,
            ),
            patch("olmlx.engine.inference.settings") as mock_settings,
            patch(
                "olmlx.engine.inference._is_memory_pressure_high",
                side_effect=pressure_returns,
            ),
        ):
            mock_settings.prompt_cache = True
            mock_settings.prompt_cache_max_tokens = 32768
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

        # After clearing resolved pressure, a fresh cache should be built
        mock_make_cache.assert_called_once()
        call_args = mock_async_stream.call_args
        assert "prompt_cache" in call_args[1]
        # Cache info chunk should be emitted
        assert any(c.get("cache_info") for c in chunks)
        # Cache should be stored after generation
        assert lm.prompt_cache_store.get("") is not None


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


class TestConfigPromptCacheMaxTokensSetting:
    def test_default_is_32768(self, monkeypatch):
        monkeypatch.delenv("OLMLX_PROMPT_CACHE_MAX_TOKENS", raising=False)
        from olmlx.config import Settings

        s = Settings()
        assert s.prompt_cache_max_tokens == 32768

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("OLMLX_PROMPT_CACHE_MAX_TOKENS", "8192")
        from olmlx.config import Settings

        s = Settings()
        assert s.prompt_cache_max_tokens == 8192

    def test_none_disables_limit(self):
        """Setting prompt_cache_max_tokens=None in code disables the limit."""
        from olmlx.config import Settings

        s = Settings(prompt_cache_max_tokens=None)
        assert s.prompt_cache_max_tokens is None


class TestConfigPromptCacheMaxSlotsSetting:
    def test_default_is_4(self, monkeypatch):
        monkeypatch.delenv("OLMLX_PROMPT_CACHE_MAX_SLOTS", raising=False)
        from olmlx.config import Settings

        s = Settings()
        assert s.prompt_cache_max_slots == 4

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("OLMLX_PROMPT_CACHE_MAX_SLOTS", "8")
        from olmlx.config import Settings

        s = Settings()
        assert s.prompt_cache_max_slots == 8


class TestMultiCacheBehavior:
    @pytest.mark.asyncio
    async def test_different_cache_ids_get_separate_caches(self, mock_manager):
        """Two requests with different cache_id values get independent caches."""
        from olmlx.engine.inference import generate_chat

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
            mock_settings.prompt_cache_max_tokens = 32768
            mock_settings.default_keep_alive = "5m"
            gen = await generate_chat(
                mock_manager,
                "qwen3",
                [{"role": "user", "content": "hi"}],
                stream=True,
                cache_id="agent-a",
            )
            async for _ in gen:
                pass

        # Agent-a should have a cache
        assert lm.prompt_cache_store.get("agent-a") is not None
        # Default and agent-b should not
        assert lm.prompt_cache_store.get("") is None
        assert lm.prompt_cache_store.get("agent-b") is None

    @pytest.mark.asyncio
    async def test_cache_id_reuse_hits_correct_cache(self, mock_manager):
        """Second request with same cache_id reuses the stored cache."""
        from olmlx.engine.inference import generate_chat
        from olmlx.engine.model_manager import CachedPromptState

        lm = mock_manager._loaded["qwen3:latest"]
        lm.tokenizer.apply_chat_template = MagicMock(return_value="formatted prompt")
        lm.tokenizer.bos_token = None
        lm.tokenizer.encode = MagicMock(return_value=[10, 20, 30, 40, 50])

        # Pre-populate cache for agent-a
        existing_cache = [MagicMock()]
        lm.prompt_cache_store.set(
            "agent-a",
            CachedPromptState(
                tokens=[10, 20, 30, 40, 50, 100, 101],
                cache=existing_cache,
            ),
        )

        tokens = _make_stream_tokens("Hello", prompt_tokens=5)
        mock_stream = _make_mock_stream(tokens)

        mock_trim = MagicMock()
        mock_mx = MagicMock()
        with (
            patch("olmlx.engine.inference.mx", mock_mx),
            patch(
                "olmlx.engine.inference.async_mlx_stream",
                return_value=mock_stream,
            ),
            patch(
                "olmlx.engine.inference.make_prompt_cache",
                MagicMock(return_value=[MagicMock()]),
            ),
            patch(
                "olmlx.engine.inference.trim_prompt_cache",
                mock_trim,
            ),
            patch("olmlx.engine.inference.settings") as mock_settings,
        ):
            mock_settings.prompt_cache = True
            mock_settings.prompt_cache_max_tokens = 32768
            mock_settings.default_keep_alive = "5m"
            gen = await generate_chat(
                mock_manager,
                "qwen3",
                [{"role": "user", "content": "hi"}],
                stream=True,
                cache_id="agent-a",
            )
            chunks = []
            async for chunk in gen:
                chunks.append(chunk)

        # Should have found cache hit — trim_prompt_cache called to align
        mock_trim.assert_called()
        # Cache info should show cache read
        cache_info = [c for c in chunks if c.get("cache_info")]
        assert cache_info
        assert cache_info[0]["cache_read_tokens"] > 0

    @pytest.mark.asyncio
    async def test_cache_id_miss_does_not_affect_other_caches(self, mock_manager):
        """A miss on one cache_id doesn't invalidate caches for other IDs."""
        from olmlx.engine.inference import generate_chat
        from olmlx.engine.model_manager import CachedPromptState

        lm = mock_manager._loaded["qwen3:latest"]
        lm.tokenizer.apply_chat_template = MagicMock(return_value="formatted prompt")
        lm.tokenizer.bos_token = None
        lm.tokenizer.encode = MagicMock(return_value=[10, 20, 30, 40, 50])

        # Pre-populate cache for agent-a
        agent_a_state = CachedPromptState(
            tokens=[10, 20, 30, 40, 50], cache=[MagicMock()]
        )
        lm.prompt_cache_store.set("agent-a", agent_a_state)

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
            mock_settings.prompt_cache_max_tokens = 32768
            mock_settings.default_keep_alive = "5m"
            gen = await generate_chat(
                mock_manager,
                "qwen3",
                [{"role": "user", "content": "hi"}],
                stream=True,
                cache_id="agent-b",
            )
            async for _ in gen:
                pass

        # agent-b should now have a cache
        assert lm.prompt_cache_store.get("agent-b") is not None
        # agent-a's cache should be untouched
        assert lm.prompt_cache_store.get("agent-a") is agent_a_state

    @pytest.mark.asyncio
    async def test_memory_pressure_clears_all_caches(self, mock_manager):
        """Memory pressure clears all cache slots, not just the active one."""
        from olmlx.engine.inference import generate_chat
        from olmlx.engine.model_manager import CachedPromptState

        lm = mock_manager._loaded["qwen3:latest"]
        lm.tokenizer.apply_chat_template = MagicMock(return_value="formatted prompt")
        lm.tokenizer.bos_token = None
        lm.tokenizer.encode = MagicMock(return_value=[10, 20, 30])

        # Pre-populate caches for multiple agents
        lm.prompt_cache_store.set(
            "agent-a", CachedPromptState(tokens=[10, 20], cache=[MagicMock()])
        )
        lm.prompt_cache_store.set(
            "agent-b", CachedPromptState(tokens=[10, 20], cache=[MagicMock()])
        )

        tokens = _make_stream_tokens("Hello", prompt_tokens=3)
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
            patch(
                "olmlx.engine.inference._is_memory_pressure_high",
                side_effect=[True, False],
            ),
            patch("olmlx.engine.inference.settings") as mock_settings,
        ):
            mock_settings.prompt_cache = True
            mock_settings.prompt_cache_max_tokens = 32768
            mock_settings.default_keep_alive = "5m"
            gen = await generate_chat(
                mock_manager,
                "qwen3",
                [{"role": "user", "content": "hi"}],
                stream=True,
                cache_id="agent-a",
            )
            async for _ in gen:
                pass

        # Both caches cleared, but agent-a gets a new one from this request
        assert lm.prompt_cache_store.get("agent-a") is not None
        assert lm.prompt_cache_store.get("agent-b") is None
