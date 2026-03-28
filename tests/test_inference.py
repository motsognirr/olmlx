"""Tests for olmlx.engine.inference (non-GPU parts)."""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import olmlx.engine.inference as _inf_mod
from olmlx.engine.inference import (
    _apply_seed,
    _build_generate_kwargs,
    _estimate_kv_cache_bytes,
    _extract_images,
    _inference_lock,
    _inference_locked,
    _inject_tools_into_system,
    _apply_chat_template_text,
    _safe_sync,
    _schedule_deferred_inference_cleanup,
    generate_chat,
    generate_completion,
    generate_embeddings,
)
from olmlx.engine.template_caps import TemplateCaps
from olmlx.utils.streaming import CancellableStream, StreamToken
from olmlx.utils.timing import TimingStats


class TestBuildGenerateKwargs:
    def test_empty_options(self):
        assert _build_generate_kwargs(None) == {}
        assert _build_generate_kwargs({}) == {}

    def test_temperature_returns_sampler(self):
        """Text model: temperature should produce a sampler callable, not a temp key."""
        result = _build_generate_kwargs({"temperature": 0.7})
        assert "temp" not in result
        assert "temperature" not in result
        assert callable(result["sampler"])

    def test_sampling_params_folded_into_sampler(self):
        """Text model: top_p, top_k, min_p all folded into sampler callable."""
        result = _build_generate_kwargs(
            {"temperature": 0.5, "top_p": 0.9, "top_k": 40, "min_p": 0.05}
        )
        assert callable(result["sampler"])
        for key in ("temp", "top_p", "top_k", "min_p", "temperature"):
            assert key not in result

    def test_repetition_penalty_returns_logits_processors(self):
        """Text model: repetition_penalty produces logits_processors list."""
        result = _build_generate_kwargs({"repeat_penalty": 1.1, "repeat_last_n": 64})
        assert "repetition_penalty" not in result
        assert "repetition_context_size" not in result
        assert isinstance(result["logits_processors"], list)

    def test_temperature_vlm_unchanged(self):
        """VLM: sampling params passed directly (not via sampler)."""
        result = _build_generate_kwargs({"temperature": 0.7}, is_vlm=True)
        assert result == {"temperature": 0.7}

    def test_vlm_all_params_direct(self):
        """VLM: all sampling params passed as direct kwargs."""
        result = _build_generate_kwargs({"temperature": 0.5, "top_p": 0.9}, is_vlm=True)
        assert result["temperature"] == 0.5
        assert result["top_p"] == 0.9
        assert "sampler" not in result

    def test_all_mappings(self):
        opts = {
            "temperature": 0.5,
            "top_p": 0.9,
            "top_k": 40,
            "seed": 42,
            "num_predict": 100,
            "repeat_penalty": 1.1,
            "repeat_last_n": 64,
            "min_p": 0.05,
        }
        result = _build_generate_kwargs(opts)
        # Sampling params folded into sampler
        assert callable(result["sampler"])
        for key in ("temp", "top_p", "top_k", "min_p"):
            assert key not in result
        # Penalty params folded into logits_processors
        assert isinstance(result["logits_processors"], list)
        for key in ("repetition_penalty", "repetition_context_size"):
            assert key not in result
        # max_tokens still direct
        assert result["max_tokens"] == 100
        # seed in kwargs for _apply_seed to consume before generation
        assert result["seed"] == 42

    def test_stop_warns_and_ignored_for_text_model(self, caplog):
        """stop is silently ignored with a warning (not rejected) for text models."""
        with caplog.at_level(logging.WARNING, logger="olmlx.engine.inference"):
            result = _build_generate_kwargs({"stop": [".", "\n"]})
        assert "stop" not in result
        assert any("stop" in r.message for r in caplog.records)

    def test_stop_vlm_ignored(self):
        result = _build_generate_kwargs({"stop": [".", "\n"]}, is_vlm=True)
        assert "stop" not in result

    def test_frequency_presence_penalty_dropped_with_warning(self, caplog):
        """frequency_penalty/presence_penalty dropped with warning."""
        with caplog.at_level(logging.WARNING, logger="olmlx.engine.inference"):
            result = _build_generate_kwargs(
                {"frequency_penalty": 0.5, "presence_penalty": 0.3}
            )
        assert "frequency_penalty" not in result
        assert "presence_penalty" not in result
        assert any("frequency_penalty" in r.message for r in caplog.records)
        assert any("presence_penalty" in r.message for r in caplog.records)

    def test_zero_penalty_warns(self, caplog):
        """Even 0.0 values should warn when explicitly set."""
        with caplog.at_level(logging.WARNING, logger="olmlx.engine.inference"):
            result = _build_generate_kwargs(
                {"frequency_penalty": 0.0, "presence_penalty": 0.0}
            )
        assert "frequency_penalty" not in result
        assert "presence_penalty" not in result
        assert any("frequency_penalty" in r.message for r in caplog.records)
        assert any("presence_penalty" in r.message for r in caplog.records)

    def test_unknown_options_ignored(self):
        result = _build_generate_kwargs({"unknown_key": 99})
        assert result == {}

    def test_no_sampler_without_sampling_params(self):
        """No sampler created when only non-sampling params present."""
        result = _build_generate_kwargs({"num_predict": 100})
        assert "sampler" not in result
        assert result["max_tokens"] == 100

    def test_make_sampler_none_raises(self, monkeypatch):
        """When make_sampler is None (mlx-lm not installed), should raise RuntimeError."""
        monkeypatch.setattr(_inf_mod, "make_sampler", None)
        with pytest.raises(RuntimeError, match="mlx-lm is not installed"):
            _build_generate_kwargs({"temperature": 0.7})

    def test_make_logits_processors_none_raises(self, monkeypatch):
        """When make_logits_processors is None (mlx-lm not installed), should raise RuntimeError."""
        monkeypatch.setattr(_inf_mod, "make_logits_processors", None)
        with pytest.raises(RuntimeError, match="mlx-lm is not installed"):
            _build_generate_kwargs({"repeat_penalty": 1.1})

    def test_seed_in_text_kwargs_for_apply_seed(self):
        """Text model: seed must be in kwargs so _apply_seed can read it."""
        result = _build_generate_kwargs({"seed": 42})
        assert result["seed"] == 42

    def test_vlm_seed_in_kwargs(self):
        """VLM: seed should be in kwargs (mlx-vlm accepts it directly)."""
        result = _build_generate_kwargs({"seed": 42}, is_vlm=True)
        assert result["seed"] == 42

    def test_repeat_last_n_without_penalty_warns(self, caplog):
        """repeat_last_n alone is a no-op — should warn and not build processors."""
        with caplog.at_level(logging.WARNING, logger="olmlx.engine.inference"):
            result = _build_generate_kwargs({"repeat_last_n": 64})
        assert "logits_processors" not in result
        assert any("repeat_last_n" in r.message for r in caplog.records)

    def test_sampling_without_temperature_no_sampler(self):
        """top_k/top_p/min_p without temperature should not build a sampler.

        make_sampler defaults temp=0.0 (greedy), which makes other sampling
        params irrelevant — a silent regression from old behavior.
        """
        result = _build_generate_kwargs({"top_k": 40})
        assert "sampler" not in result


class TestApplySeed:
    def test_apply_seed_consume_pops(self):
        """_apply_seed(consume=True) should pop seed — text models must not forward it."""
        kwargs = {"seed": 42, "max_tokens": 100}
        with patch("olmlx.engine.inference.mx") as mock_mx:
            _apply_seed(kwargs, consume=True)
        assert "seed" not in kwargs
        mock_mx.random.seed.assert_called_once_with(42)

    def test_apply_seed_no_consume_keeps(self):
        """_apply_seed(consume=False) should keep seed — VLMs forward it to mlx-vlm."""
        kwargs = {"seed": 42, "max_tokens": 100}
        with patch("olmlx.engine.inference.mx") as mock_mx:
            _apply_seed(kwargs, consume=False)
        assert "seed" in kwargs
        mock_mx.random.seed.assert_called_once_with(42)

    def test_apply_seed_no_seed(self):
        """_apply_seed with no seed key should be a no-op."""
        kwargs = {"max_tokens": 100}
        with patch("olmlx.engine.inference.mx") as mock_mx:
            _apply_seed(kwargs)
        mock_mx.random.seed.assert_not_called()


class TestExtractImages:
    def test_no_images(self):
        messages = [{"role": "user", "content": "hi"}]
        assert _extract_images(messages) is None

    def test_with_images(self):
        messages = [
            {"role": "user", "content": "describe", "images": ["img1.jpg", "img2.png"]},
        ]
        result = _extract_images(messages)
        assert result == ["img1.jpg", "img2.png"]

    def test_multiple_messages(self):
        messages = [
            {"role": "user", "content": "first", "images": ["a.jpg"]},
            {"role": "user", "content": "second", "images": ["b.jpg"]},
        ]
        result = _extract_images(messages)
        assert result == ["a.jpg", "b.jpg"]

    def test_empty_images_list(self):
        messages = [{"role": "user", "content": "hi", "images": []}]
        assert _extract_images(messages) is None


class TestInjectToolsIntoSystem:
    def test_with_existing_system(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
        ]
        tools = [
            {
                "function": {
                    "name": "search",
                    "description": "Search the web",
                    "parameters": {},
                }
            }
        ]
        result = _inject_tools_into_system(messages, tools)
        assert result[0]["role"] == "system"
        assert "search" in result[0]["content"]
        assert "You are helpful." in result[0]["content"]
        # Original should not be modified
        assert "search" not in messages[0]["content"]

    def test_without_system(self):
        messages = [{"role": "user", "content": "hi"}]
        tools = [
            {
                "function": {
                    "name": "func",
                    "description": "A function",
                    "parameters": {},
                }
            }
        ]
        result = _inject_tools_into_system(messages, tools)
        assert result[0]["role"] == "system"
        assert "func" in result[0]["content"]
        assert result[1] == messages[0]

    def test_tool_without_function_key(self):
        messages = [{"role": "user", "content": "hi"}]
        tools = [{"name": "direct_tool", "description": "Direct", "parameters": {}}]
        result = _inject_tools_into_system(messages, tools)
        assert "direct_tool" in result[0]["content"]


class TestApplyChatTemplateText:
    def test_basic(self):
        tokenizer = MagicMock()
        tokenizer.apply_chat_template = MagicMock(return_value="formatted prompt")
        messages = [{"role": "user", "content": "hi"}]
        result = _apply_chat_template_text(tokenizer, messages)
        assert result == "formatted prompt"
        tokenizer.apply_chat_template.assert_called_once_with(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def test_with_tools_supported(self):
        """Default (enable_thinking=None) + tools → enable_thinking=False (backward compat)."""
        tokenizer = MagicMock()
        tokenizer.apply_chat_template = MagicMock(return_value="prompt with tools")
        messages = [{"role": "user", "content": "hi"}]
        tools = [{"type": "function", "function": {"name": "f"}}]
        caps = TemplateCaps(supports_tools=True, supports_enable_thinking=True)
        _apply_chat_template_text(tokenizer, messages, tools, caps)
        call_kwargs = tokenizer.apply_chat_template.call_args[1]
        assert call_kwargs["tools"] == tools
        assert call_kwargs["enable_thinking"] is False

    def test_with_tools_not_supported(self):
        tokenizer = MagicMock()
        tokenizer.apply_chat_template = MagicMock(return_value="prompt")
        messages = [{"role": "user", "content": "hi"}]
        tools = [
            {
                "type": "function",
                "function": {"name": "f", "description": "desc", "parameters": {}},
            }
        ]
        caps = TemplateCaps(supports_tools=False)
        _apply_chat_template_text(tokenizer, messages, tools, caps)
        # Should inject tools into system message instead
        call_args = tokenizer.apply_chat_template.call_args[0][0]
        assert call_args[0]["role"] == "system"
        assert "f" in call_args[0]["content"]

    def test_fallback_on_template_error(self):
        tokenizer = MagicMock()
        call_count = [0]

        def side_effect(messages, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1 and "tools" in kwargs:
                raise TypeError("tools not supported")
            return "fallback prompt"

        tokenizer.apply_chat_template = side_effect
        messages = [{"role": "user", "content": "hi"}]
        tools = [
            {
                "type": "function",
                "function": {"name": "f", "description": "d", "parameters": {}},
            }
        ]
        caps = TemplateCaps(supports_tools=True)
        result = _apply_chat_template_text(tokenizer, messages, tools, caps)
        assert result == "fallback prompt"
        assert call_count[0] == 2

    def test_fallback_preserves_enable_thinking(self):
        """When tools kwarg fails and falls back to injection, enable_thinking is preserved."""
        tokenizer = MagicMock()
        call_count = [0]
        retry_kwargs = {}

        def side_effect(messages, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1 and "tools" in kwargs:
                raise TypeError("tools not supported")
            retry_kwargs.update(kwargs)
            return "fallback prompt"

        tokenizer.apply_chat_template = side_effect
        messages = [{"role": "user", "content": "hi"}]
        tools = [
            {
                "type": "function",
                "function": {"name": "f", "description": "d", "parameters": {}},
            }
        ]
        caps = TemplateCaps(supports_tools=True, supports_enable_thinking=True)
        result = _apply_chat_template_text(
            tokenizer, messages, tools, caps, enable_thinking=False
        )
        assert result == "fallback prompt"
        assert retry_kwargs.get("enable_thinking") is False
        assert "tools" not in retry_kwargs

    def test_enable_thinking_true(self):
        """Explicit enable_thinking=True → passed through to template."""
        tokenizer = MagicMock()
        tokenizer.apply_chat_template = MagicMock(return_value="prompt")
        messages = [{"role": "user", "content": "hi"}]
        caps = TemplateCaps(supports_tools=False, supports_enable_thinking=True)
        _apply_chat_template_text(tokenizer, messages, caps=caps, enable_thinking=True)
        call_kwargs = tokenizer.apply_chat_template.call_args[1]
        assert call_kwargs["enable_thinking"] is True

    def test_enable_thinking_false(self):
        """Explicit enable_thinking=False → passed through to template."""
        tokenizer = MagicMock()
        tokenizer.apply_chat_template = MagicMock(return_value="prompt")
        messages = [{"role": "user", "content": "hi"}]
        caps = TemplateCaps(supports_tools=False, supports_enable_thinking=True)
        _apply_chat_template_text(tokenizer, messages, caps=caps, enable_thinking=False)
        call_kwargs = tokenizer.apply_chat_template.call_args[1]
        assert call_kwargs["enable_thinking"] is False

    def test_enable_thinking_none_no_tools_defaults_true(self):
        """enable_thinking=None + no tools → defaults to True."""
        tokenizer = MagicMock()
        tokenizer.apply_chat_template = MagicMock(return_value="prompt")
        messages = [{"role": "user", "content": "hi"}]
        caps = TemplateCaps(supports_tools=False, supports_enable_thinking=True)
        _apply_chat_template_text(tokenizer, messages, caps=caps)
        call_kwargs = tokenizer.apply_chat_template.call_args[1]
        assert call_kwargs["enable_thinking"] is True

    def test_enable_thinking_true_with_tools(self):
        """Explicit enable_thinking=True + tools → both tools and enable_thinking=True passed."""
        tokenizer = MagicMock()
        tokenizer.apply_chat_template = MagicMock(return_value="prompt")
        messages = [{"role": "user", "content": "hi"}]
        tools = [{"type": "function", "function": {"name": "f"}}]
        caps = TemplateCaps(supports_tools=True, supports_enable_thinking=True)
        _apply_chat_template_text(
            tokenizer, messages, tools, caps, enable_thinking=True
        )
        call_kwargs = tokenizer.apply_chat_template.call_args[1]
        assert call_kwargs["tools"] == tools
        assert call_kwargs["enable_thinking"] is True

    def test_enable_thinking_not_supported(self):
        """caps.supports_enable_thinking=False → enable_thinking kwarg not passed."""
        tokenizer = MagicMock()
        tokenizer.apply_chat_template = MagicMock(return_value="prompt")
        messages = [{"role": "user", "content": "hi"}]
        caps = TemplateCaps(supports_tools=False, supports_enable_thinking=False)
        _apply_chat_template_text(tokenizer, messages, caps=caps, enable_thinking=True)
        call_kwargs = tokenizer.apply_chat_template.call_args[1]
        assert "enable_thinking" not in call_kwargs

    def test_no_caps_defaults(self):
        tokenizer = MagicMock()
        tokenizer.apply_chat_template = MagicMock(return_value="result")
        _apply_chat_template_text(tokenizer, [], None, caps=None)
        tokenizer.apply_chat_template.assert_called_once()

    def test_template_fails_completely(self):
        tokenizer = MagicMock()
        tokenizer.apply_chat_template = MagicMock(side_effect=RuntimeError("broken"))
        with pytest.raises(RuntimeError, match="Chat template failed"):
            _apply_chat_template_text(tokenizer, [], None)

    def test_tools_kwarg_fails_then_injection_fails(self):
        tokenizer = MagicMock()
        tokenizer.apply_chat_template = MagicMock(
            side_effect=TypeError("all calls fail")
        )
        tools = [
            {
                "type": "function",
                "function": {"name": "f", "description": "d", "parameters": {}},
            }
        ]
        caps = TemplateCaps(supports_tools=True)
        with pytest.raises(
            RuntimeError, match="Chat template failed even without tools"
        ):
            _apply_chat_template_text(tokenizer, [], tools, caps)


class TestApplyChatTemplateVlm:
    def test_vlm_template(self):
        from olmlx.engine.inference import _apply_chat_template_vlm

        mock_mlx_vlm = MagicMock()
        mock_mlx_vlm.apply_chat_template.return_value = "vlm prompt"
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_model.config = {"model_type": "qwen2_vl"}

        with patch.dict("sys.modules", {"mlx_vlm": mock_mlx_vlm}):
            result = _apply_chat_template_vlm(
                mock_processor,
                mock_model,
                [{"role": "user", "content": "describe"}],
                images=["img.jpg"],
            )

        assert result == "vlm prompt"
        mock_mlx_vlm.apply_chat_template.assert_called_once()

    def test_vlm_template_no_config(self):
        from olmlx.engine.inference import _apply_chat_template_vlm

        mock_mlx_vlm = MagicMock()
        mock_mlx_vlm.apply_chat_template.return_value = "result"
        mock_processor = MagicMock()
        mock_model = MagicMock(spec=[])  # no config attr

        with patch.dict("sys.modules", {"mlx_vlm": mock_mlx_vlm}):
            result = _apply_chat_template_vlm(
                mock_processor,
                mock_model,
                [{"role": "user", "content": "hi"}],
            )
        assert result == "result"


class TestGenerateCompletion:
    @pytest.mark.asyncio
    async def test_non_streaming(self, mock_manager):
        mock_mx = MagicMock()

        mock_mx.core = mock_mx

        mock_mlx_lm = MagicMock()
        mock_mlx_lm.generate.return_value = "Generated output"

        with patch("olmlx.engine.inference.mx", mock_mx):
            with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
                with patch(
                    "olmlx.engine.inference.asyncio.to_thread",
                    new_callable=AsyncMock,
                ) as mock_thread:
                    mock_thread.return_value = "Generated output"
                    result = await generate_completion(
                        mock_manager,
                        "qwen3",
                        "Hello",
                        stream=False,
                    )

        assert result["text"] == "Generated output"
        assert result["done"] is True

    @pytest.mark.asyncio
    async def test_streaming(self, mock_manager):
        mock_mx = MagicMock()

        tokens = [
            StreamToken(
                text="Hello",
                token=1,
                prompt_tokens=5,
                generation_tokens=1,
                prompt_tps=100.0,
                generation_tps=50.0,
            ),
            StreamToken(
                text=" world",
                token=2,
                prompt_tokens=5,
                generation_tokens=2,
                prompt_tps=100.0,
                generation_tps=50.0,
            ),
        ]

        mock_stream = MagicMock(spec=CancellableStream)
        mock_stream.drain_and_join = AsyncMock()
        mock_stream._thread = None  # No thread — normal completion path

        # Make it a proper async iterator
        token_iter = iter(tokens)

        async def anext_impl():
            try:
                return next(token_iter)
            except StopIteration:
                raise StopAsyncIteration

        mock_stream.__aiter__ = lambda self: self
        mock_stream.__anext__ = lambda self: anext_impl()

        with patch("olmlx.engine.inference.mx", mock_mx):
            with patch(
                "olmlx.engine.inference.async_mlx_stream", return_value=mock_stream
            ):
                gen = await generate_completion(
                    mock_manager,
                    "qwen3",
                    "Hello",
                    stream=True,
                )
                chunks = []
                async for chunk in gen:
                    chunks.append(chunk)

        # last chunk is the done signal
        assert chunks[-1]["done"] is True
        assert any(c.get("text") == "Hello" for c in chunks if not c.get("done"))


class TestGenerateChat:
    @pytest.mark.asyncio
    async def test_non_streaming(self, mock_manager):
        mock_mx = MagicMock()

        mock_manager._loaded["qwen3:latest"].tokenizer.apply_chat_template = MagicMock(
            return_value="formatted prompt"
        )

        with patch("olmlx.engine.inference.mx", mock_mx):
            with patch(
                "olmlx.engine.inference.asyncio.to_thread", new_callable=AsyncMock
            ) as mock_thread:
                mock_thread.return_value = "Chat response"
                result = await generate_chat(
                    mock_manager,
                    "qwen3",
                    [{"role": "user", "content": "hi"}],
                    stream=False,
                )

        assert result["text"] == "Chat response"
        assert result["done"] is True

    @pytest.mark.asyncio
    async def test_with_tools(self, mock_manager):
        mock_mx = MagicMock()

        mock_manager._loaded["qwen3:latest"].tokenizer.apply_chat_template = MagicMock(
            return_value="formatted prompt with tools"
        )

        with patch("olmlx.engine.inference.mx", mock_mx):
            with patch(
                "olmlx.engine.inference.asyncio.to_thread", new_callable=AsyncMock
            ) as mock_thread:
                mock_thread.return_value = "tool response"
                result = await generate_chat(
                    mock_manager,
                    "qwen3",
                    [{"role": "user", "content": "search"}],
                    tools=[{"type": "function", "function": {"name": "search"}}],
                    stream=False,
                )

        assert result["text"] == "tool response"


class TestGenerateChatEnableThinking:
    @pytest.mark.asyncio
    async def test_enable_thinking_passed_to_template(self, mock_manager):
        """generate_chat(enable_thinking=True) passes it to template application."""
        mock_mx = MagicMock()

        mock_manager._loaded["qwen3:latest"].tokenizer.apply_chat_template = MagicMock(
            return_value="formatted prompt"
        )

        with patch("olmlx.engine.inference.mx", mock_mx):
            with patch(
                "olmlx.engine.inference.asyncio.to_thread", new_callable=AsyncMock
            ) as mock_thread:
                mock_thread.return_value = "response"
                await generate_chat(
                    mock_manager,
                    "qwen3",
                    [{"role": "user", "content": "hi"}],
                    stream=False,
                    enable_thinking=True,
                )

        call_kwargs = mock_manager._loaded[
            "qwen3:latest"
        ].tokenizer.apply_chat_template.call_args[1]
        assert call_kwargs["enable_thinking"] is True

    @pytest.mark.asyncio
    async def test_enable_thinking_false_passed_to_template(self, mock_manager):
        """generate_chat(enable_thinking=False) passes it to template application."""
        mock_mx = MagicMock()

        mock_manager._loaded["qwen3:latest"].tokenizer.apply_chat_template = MagicMock(
            return_value="formatted prompt"
        )

        with patch("olmlx.engine.inference.mx", mock_mx):
            with patch(
                "olmlx.engine.inference.asyncio.to_thread", new_callable=AsyncMock
            ) as mock_thread:
                mock_thread.return_value = "response"
                await generate_chat(
                    mock_manager,
                    "qwen3",
                    [{"role": "user", "content": "hi"}],
                    stream=False,
                    enable_thinking=False,
                )

        call_kwargs = mock_manager._loaded[
            "qwen3:latest"
        ].tokenizer.apply_chat_template.call_args[1]
        assert call_kwargs["enable_thinking"] is False


class TestFullCompletionInner:
    @pytest.mark.asyncio
    async def test_result_with_text_attr(self, mock_manager):
        """Test when result has .text attribute (GenerationResult dataclass)."""
        from olmlx.engine.inference import _full_completion_inner

        lm = mock_manager._loaded["qwen3:latest"]

        mock_result = MagicMock()
        mock_result.text = "result text"

        with patch(
            "olmlx.engine.inference.asyncio.to_thread", new_callable=AsyncMock
        ) as mock_thread:
            mock_thread.return_value = mock_result
            result = await _full_completion_inner(
                lm,
                "prompt",
                100,
                {},
                TimingStats(),
            )

        assert result["text"] == "result text"

    @pytest.mark.asyncio
    async def test_result_as_string(self, mock_manager):
        from olmlx.engine.inference import _full_completion_inner

        lm = mock_manager._loaded["qwen3:latest"]

        with patch(
            "olmlx.engine.inference.asyncio.to_thread", new_callable=AsyncMock
        ) as mock_thread:
            mock_thread.return_value = "plain string"
            result = await _full_completion_inner(
                lm,
                "prompt",
                100,
                {},
                TimingStats(),
            )

        assert result["text"] == "plain string"

    @pytest.mark.asyncio
    async def test_result_other_type(self, mock_manager):
        from olmlx.engine.inference import _full_completion_inner

        lm = mock_manager._loaded["qwen3:latest"]

        with patch(
            "olmlx.engine.inference.asyncio.to_thread", new_callable=AsyncMock
        ) as mock_thread:
            mock_thread.return_value = 42  # unusual type
            result = await _full_completion_inner(
                lm,
                "prompt",
                100,
                {},
                TimingStats(),
            )

        assert result["text"] == "42"

    @pytest.mark.asyncio
    async def test_vlm_completion(self, mock_manager):
        from olmlx.engine.inference import _full_completion_inner

        lm = mock_manager._loaded["qwen3:latest"]
        lm.is_vlm = True

        mock_mlx_vlm = MagicMock()
        with patch(
            "olmlx.engine.inference.asyncio.to_thread", new_callable=AsyncMock
        ) as mock_thread:
            mock_thread.return_value = "vlm output"
            with patch.dict("sys.modules", {"mlx_vlm": mock_mlx_vlm}):
                result = await _full_completion_inner(
                    lm,
                    "prompt",
                    100,
                    {},
                    TimingStats(),
                    images=["img.jpg"],
                )

        assert result["text"] == "vlm output"


class TestGenerateChatVlm:
    @pytest.mark.asyncio
    async def test_vlm_enable_thinking_uses_text_template(self, mock_manager):
        """VLM path uses text template when enable_thinking is set and supported."""
        lm = mock_manager._loaded["qwen3:latest"]
        lm.is_vlm = True
        lm.template_caps = TemplateCaps(
            supports_tools=True, supports_enable_thinking=True
        )

        mock_mx = MagicMock()

        with patch("olmlx.engine.inference.mx", mock_mx):
            with patch(
                "olmlx.engine.inference._apply_chat_template_text",
                return_value="text prompt",
            ) as mock_text_tpl:
                with patch(
                    "olmlx.engine.inference.asyncio.to_thread",
                    new_callable=AsyncMock,
                ) as mock_thread:
                    mock_thread.return_value = "response"
                    await generate_chat(
                        mock_manager,
                        "qwen3",
                        [{"role": "user", "content": "describe"}],
                        stream=False,
                        enable_thinking=False,
                    )

        # Should have used text template path (not VLM template)
        mock_text_tpl.assert_called_once()
        call_kwargs = mock_text_tpl.call_args
        assert call_kwargs.kwargs.get("enable_thinking") is False

    @pytest.mark.asyncio
    async def test_vlm_enable_thinking_warns_when_unsupported(
        self, mock_manager, caplog
    ):
        """VLM path logs warning when enable_thinking is set but template doesn't support it."""
        lm = mock_manager._loaded["qwen3:latest"]
        lm.is_vlm = True
        lm.template_caps = TemplateCaps(
            supports_tools=False, supports_enable_thinking=False
        )

        mock_mx = MagicMock()

        mock_mlx_vlm = MagicMock()
        mock_mlx_vlm.apply_chat_template.return_value = "vlm prompt"

        with patch("olmlx.engine.inference.mx", mock_mx):
            with patch.dict("sys.modules", {"mlx_vlm": mock_mlx_vlm}):
                with patch(
                    "olmlx.engine.inference.asyncio.to_thread",
                    new_callable=AsyncMock,
                ) as mock_thread:
                    mock_thread.return_value = "vlm response"
                    with caplog.at_level(
                        logging.WARNING, logger="olmlx.engine.inference"
                    ):
                        await generate_chat(
                            mock_manager,
                            "qwen3",
                            [{"role": "user", "content": "describe"}],
                            stream=False,
                            enable_thinking=True,
                        )

        assert any("ignored for VLM" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_vlm_without_tools(self, mock_manager):
        """Test chat with VLM model (no tools)."""
        lm = mock_manager._loaded["qwen3:latest"]
        lm.is_vlm = True

        mock_mx = MagicMock()

        mock_mlx_vlm = MagicMock()
        mock_mlx_vlm.apply_chat_template.return_value = "vlm prompt"

        with patch("olmlx.engine.inference.mx", mock_mx):
            with patch.dict("sys.modules", {"mlx_vlm": mock_mlx_vlm}):
                with patch(
                    "olmlx.engine.inference.asyncio.to_thread",
                    new_callable=AsyncMock,
                ) as mock_thread:
                    mock_thread.return_value = "vlm response"
                    result = await generate_chat(
                        mock_manager,
                        "qwen3",
                        [
                            {
                                "role": "user",
                                "content": "describe",
                                "images": ["img.jpg"],
                            }
                        ],
                        stream=False,
                    )

        assert result["text"] == "vlm response"


class TestGenerateEmbeddings:
    def _setup_tokenizer(self, mock_manager):
        """Set tokenizer.encode to return real ints so mx.array works."""
        tok = mock_manager._loaded["qwen3:latest"].tokenizer
        # Remove .tokenizer sub-attr to avoid double-unwrap
        if hasattr(tok, "tokenizer"):
            del tok.tokenizer
        tok.encode = MagicMock(return_value=[1, 2, 3])
        return tok

    @pytest.mark.asyncio
    async def test_with_embed_layer_3d(self, mock_manager):
        import mlx.core as mx

        self._setup_tokenizer(mock_manager)

        model = mock_manager._loaded["qwen3:latest"].model
        model.model = MagicMock()
        model.model.embed_tokens = MagicMock(return_value=mx.zeros((1, 3, 4)))

        result = await generate_embeddings(mock_manager, "qwen3", ["hello"])
        assert len(result) == 1
        assert len(result[0]) == 4

    @pytest.mark.asyncio
    async def test_without_embed_layer_2d(self, mock_manager):
        import mlx.core as mx

        self._setup_tokenizer(mock_manager)

        model = mock_manager._loaded["qwen3:latest"].model
        model.model = MagicMock(spec=[])  # no embed_tokens attribute
        model.return_value = mx.zeros((3, 4))

        result = await generate_embeddings(mock_manager, "qwen3", ["hello"])
        assert len(result) == 1
        assert len(result[0]) == 4

    @pytest.mark.asyncio
    async def test_1d_embedding(self, mock_manager):
        import mlx.core as mx

        self._setup_tokenizer(mock_manager)

        model = mock_manager._loaded["qwen3:latest"].model
        model.model = MagicMock()
        model.model.embed_tokens = MagicMock(return_value=mx.zeros((4,)))

        result = await generate_embeddings(mock_manager, "qwen3", ["hello"])
        assert len(result) == 1
        assert len(result[0]) == 4

    @pytest.mark.asyncio
    async def test_multiple_texts(self, mock_manager):
        import mlx.core as mx

        self._setup_tokenizer(mock_manager)

        model = mock_manager._loaded["qwen3:latest"].model
        model.model = MagicMock()
        model.model.embed_tokens = MagicMock(return_value=mx.zeros((1, 3, 4)))

        result = await generate_embeddings(mock_manager, "qwen3", ["hello", "world"])
        assert len(result) == 2


class TestStreamCancellationHoldsLock:
    @pytest.mark.asyncio
    async def test_lock_held_during_drain(self, mock_manager):
        """Lock should stay held until drain_and_join completes."""
        mock_mx = MagicMock()

        tokens = [
            StreamToken(
                text="a",
                token=1,
                prompt_tokens=5,
                generation_tokens=1,
                prompt_tps=100.0,
                generation_tps=50.0,
            ),
        ]

        lock_held_during_drain = None

        mock_stream = MagicMock(spec=CancellableStream)
        mock_stream._thread = None  # No thread — normal completion path
        token_iter = iter(tokens)

        async def anext_impl():
            try:
                return next(token_iter)
            except StopIteration:
                raise StopAsyncIteration

        mock_stream.__aiter__ = lambda self: self
        mock_stream.__anext__ = lambda self: anext_impl()

        async def spy_drain():
            nonlocal lock_held_during_drain
            lock_held_during_drain = _inference_lock.locked()

        mock_stream.drain_and_join = spy_drain

        with patch("olmlx.engine.inference.mx", mock_mx):
            with patch(
                "olmlx.engine.inference.async_mlx_stream", return_value=mock_stream
            ):
                gen = await generate_completion(
                    mock_manager,
                    "qwen3",
                    "Hello",
                    stream=True,
                )
                chunks = []
                async for chunk in gen:
                    chunks.append(chunk)

        assert lock_held_during_drain is True


class TestGenerateEmbeddingsAcquiresLock:
    @pytest.mark.asyncio
    async def test_embeddings_blocks_when_lock_held(self, mock_manager):
        """generate_embeddings should wait for the inference lock."""
        import mlx.core as mx

        tok = mock_manager._loaded["qwen3:latest"].tokenizer
        if hasattr(tok, "tokenizer"):
            del tok.tokenizer
        tok.encode = MagicMock(return_value=[1, 2, 3])

        model = mock_manager._loaded["qwen3:latest"].model
        model.model = MagicMock()
        model.model.embed_tokens = MagicMock(return_value=mx.zeros((1, 3, 4)))

        order = []

        async with _inference_lock:
            # Start embeddings in background — it should block
            task = asyncio.create_task(
                generate_embeddings(mock_manager, "qwen3", ["hello"])
            )
            await asyncio.sleep(0.05)
            assert not task.done(), "embeddings should be blocked by lock"
            order.append("lock_released")

        # Now it should complete
        result = await task
        order.append("embeddings_done")
        assert order == ["lock_released", "embeddings_done"]
        assert len(result) == 1


class TestSafeSync:
    def test_success(self):
        """_safe_sync() should sync default stream and any resolved generation streams."""
        mock_stream = MagicMock()
        with (
            patch("olmlx.engine.inference.mx") as mock_mx,
            patch("olmlx.engine.inference._generation_streams", [mock_stream]),
        ):
            _safe_sync()
            # Default stream + 1 generation stream
            assert mock_mx.synchronize.call_count == 2
            mock_mx.synchronize.assert_any_call(mock_stream)

    def test_suppresses_exception(self):
        """_safe_sync() should suppress exceptions from mx.synchronize()."""
        with patch("olmlx.engine.inference.mx") as mock_mx:
            mock_mx.synchronize.side_effect = RuntimeError("Metal error")
            _safe_sync()  # should not raise


class TestInferenceLocked:
    @pytest.mark.asyncio
    async def test_acquires_and_releases_lock(self):
        """_inference_locked() should acquire lock on entry and release on exit."""
        with patch("olmlx.engine.inference.mx"):
            assert not _inference_lock.locked()
            async with _inference_locked():
                assert _inference_lock.locked()
            assert not _inference_lock.locked()

    @pytest.mark.asyncio
    async def test_releases_lock_on_exception(self):
        """_inference_locked() should release lock even if body raises."""
        with patch("olmlx.engine.inference.mx"):
            with pytest.raises(ValueError):
                async with _inference_locked():
                    raise ValueError("test error")
            assert not _inference_lock.locked()

    @pytest.mark.asyncio
    async def test_syncs_on_entry_and_exit(self):
        """_inference_locked() should call mx.synchronize on entry and exit."""
        with patch("olmlx.engine.inference.mx") as mock_mx:
            async with _inference_locked():
                pass
            # Called at least twice: entry sync + exit sync
            assert mock_mx.synchronize.call_count >= 2


class TestInferenceLockedWaitsDeferredCleanup:
    @pytest.mark.asyncio
    async def test_waits_for_deferred_cleanup_pre_lock(self):
        """_inference_locked() should wait for deferred cleanup instead of raising."""
        completed = False

        async def cleanup():
            nonlocal completed
            await asyncio.sleep(0.05)
            completed = True

        _inf_mod._deferred_cleanup_task = asyncio.create_task(cleanup())
        with patch("olmlx.engine.inference.mx"):
            async with _inference_locked():
                pass
        assert completed
        _inf_mod._deferred_cleanup_task = None

    @pytest.mark.asyncio
    async def test_waits_for_deferred_cleanup_post_lock(self):
        """_inference_locked() should wait for deferred cleanup created during lock acquisition (TOCTOU)."""
        cleanup_done = asyncio.Event()

        async def cleanup():
            await asyncio.sleep(0.05)
            cleanup_done.set()

        # Wrap _acquire_inference_lock to inject a deferred task after lock
        # is acquired but before the post-lock _await_deferred_cleanup runs.
        original_acquire = _inf_mod._acquire_inference_lock

        async def acquire_then_inject():
            await original_acquire()
            _inf_mod._deferred_cleanup_task = asyncio.create_task(cleanup())

        with patch("olmlx.engine.inference.mx"):
            _inf_mod._deferred_cleanup_task = None
            with patch(
                "olmlx.engine.inference._acquire_inference_lock",
                side_effect=acquire_then_inject,
            ):
                async with _inference_locked():
                    pass
            assert cleanup_done.is_set()
        _inf_mod._deferred_cleanup_task = None


class TestQueueDepth:
    @pytest.mark.asyncio
    async def test_queue_depth_incremented_during_lock_wait(self):
        """Queue depth should be logged when requests wait for the lock."""
        # Use a fresh lock to avoid event loop binding issues
        fresh_lock = asyncio.Lock()
        _inf_mod._queue_depth = 0
        _inf_mod._deferred_cleanup_task = None
        with (
            patch("olmlx.engine.inference.mx"),
            patch("olmlx.engine.inference._inference_lock", fresh_lock),
        ):
            # Hold the lock
            await fresh_lock.acquire()

            async def second_request():
                async with _inference_locked():
                    pass

            task = asyncio.create_task(second_request())
            await asyncio.sleep(0.05)
            # Release lock so second request can proceed
            fresh_lock.release()
            await task

        assert _inf_mod._queue_depth == 0  # should be back to 0


class TestInferenceQueueTimeout:
    @pytest.mark.asyncio
    async def test_lock_acquisition_times_out(self):
        """_inference_locked() should raise ServerBusyError when queue timeout expires."""
        from olmlx.engine.inference import ServerBusyError

        fresh_lock = asyncio.Lock()
        _inf_mod._queue_depth = 0
        _inf_mod._deferred_cleanup_task = None
        with (
            patch("olmlx.engine.inference.mx"),
            patch("olmlx.engine.inference._inference_lock", fresh_lock),
            patch("olmlx.engine.inference.settings") as mock_settings,
        ):
            mock_settings.inference_queue_timeout = 0.05
            # Hold the lock
            await fresh_lock.acquire()
            with pytest.raises(ServerBusyError, match="queue timeout"):
                async with _inference_locked():
                    pass
            fresh_lock.release()


class TestStreamCompletionQueueTimeout:
    @pytest.mark.asyncio
    async def test_streaming_lock_acquisition_times_out(self, mock_manager):
        """_stream_completion() should raise ServerBusyError when queue timeout expires."""
        from olmlx.engine.inference import ServerBusyError, _stream_completion

        fresh_lock = asyncio.Lock()
        _inf_mod._queue_depth = 0
        _inf_mod._deferred_cleanup_task = None
        with (
            patch("olmlx.engine.inference.mx"),
            patch("olmlx.engine.inference._inference_lock", fresh_lock),
            patch("olmlx.engine.inference.settings") as mock_settings,
        ):
            mock_settings.inference_queue_timeout = 0.05
            await fresh_lock.acquire()
            lm = mock_manager._loaded["qwen3:latest"]
            with pytest.raises(ServerBusyError, match="queue timeout"):
                gen = _stream_completion(lm, "prompt", 100, {}, TimingStats())
                async for _ in gen:
                    pass
            fresh_lock.release()


class TestStreamCompletionFallbackJoinLogging:
    @pytest.mark.asyncio
    async def test_fallback_join_defers_cleanup_when_thread_alive(
        self, mock_manager, caplog
    ):
        """When shield is interrupted and fallback join times out, should defer cleanup."""
        mock_mx = MagicMock()

        # Create a mock stream that simulates a stuck thread
        mock_stream = MagicMock(spec=CancellableStream)
        mock_thread = MagicMock()
        # Thread is alive during cleanup, then exits when deferred task polls
        alive_calls = [True, True, True, False]
        mock_thread.is_alive.side_effect = lambda: (
            alive_calls.pop(0) if alive_calls else False
        )
        mock_thread.join = MagicMock()  # join returns but thread still "alive"
        mock_stream._thread = mock_thread

        token_iter = iter(
            [
                StreamToken(
                    text="a",
                    token=1,
                    prompt_tokens=5,
                    generation_tokens=1,
                    prompt_tps=100.0,
                    generation_tps=50.0,
                ),
            ]
        )

        async def anext_impl():
            try:
                return next(token_iter)
            except StopIteration:
                raise StopAsyncIteration

        mock_stream.__aiter__ = lambda self: self
        mock_stream.__anext__ = lambda self: anext_impl()

        # Make drain_and_join raise CancelledError to trigger fallback path
        async def drain_raises():
            raise asyncio.CancelledError()

        mock_stream.drain_and_join = drain_raises

        with patch("olmlx.engine.inference.mx", mock_mx):
            with patch(
                "olmlx.engine.inference.async_mlx_stream",
                return_value=mock_stream,
            ):
                with caplog.at_level(logging.WARNING, logger="olmlx.engine.inference"):
                    gen = await generate_completion(
                        mock_manager,
                        "qwen3",
                        "Hello",
                        stream=True,
                    )
                    chunks = []
                    async for chunk in gen:
                        chunks.append(chunk)

        assert any(
            "deferring Metal sync" in record.message for record in caplog.records
        )
        # Wait for deferred cleanup task to complete and release the lock
        await _inf_mod._deferred_cleanup_task
        assert not _inference_lock.locked(), (
            "_inference_lock must be released by deferred cleanup task"
        )


class TestDeferredInferenceCleanup:
    @pytest.mark.asyncio
    async def test_safe_sync_called_after_thread_exits(self):
        """_safe_sync should be called once the thread exits."""
        mock_stream = MagicMock()
        mock_thread = MagicMock()
        # Thread alive on first check (while loop), dead on re-check and finally guard
        mock_thread.is_alive.side_effect = [True, False, False]
        mock_thread.join = MagicMock()
        mock_stream._thread = mock_thread

        # Reset lazy lock for this test's event loop
        _inf_mod._deferred_cleanup_lock = None

        # Manually acquire the lock to simulate _stream_completion holding it
        await _inference_lock.acquire()

        with patch("olmlx.engine.inference._safe_sync") as mock_safe_sync:
            await _schedule_deferred_inference_cleanup(mock_stream)
            await _inf_mod._deferred_cleanup_task

        # _safe_sync should have been called (after thread exited)
        mock_safe_sync.assert_called_once()
        assert not _inference_lock.locked()

    @pytest.mark.asyncio
    async def test_lock_released_after_thread_exits(self):
        """The deferred task must release the lock once the thread exits."""
        mock_stream = MagicMock()
        mock_thread = MagicMock()
        # Thread alive on first check, dead on re-check and finally guard
        mock_thread.is_alive.side_effect = [True, False, False]
        mock_thread.join = MagicMock()
        mock_stream._thread = mock_thread

        # Reset lazy lock for this test's event loop
        _inf_mod._deferred_cleanup_lock = None

        await _inference_lock.acquire()
        assert _inference_lock.locked()

        with patch("olmlx.engine.inference._safe_sync"):
            await _schedule_deferred_inference_cleanup(mock_stream)
            # Task is running — lock should still be held initially
            assert _inference_lock.locked()
            # Wait for deferred task to complete
            await _inf_mod._deferred_cleanup_task

        assert not _inference_lock.locked()


class TestAwaitDeferredCleanup:
    @pytest.mark.asyncio
    async def test_returns_immediately_when_no_task(self):
        """Should return immediately when no deferred cleanup task exists."""
        from olmlx.engine.inference import _await_deferred_cleanup

        _inf_mod._deferred_cleanup_task = None
        await _await_deferred_cleanup()  # should not raise

    @pytest.mark.asyncio
    async def test_returns_immediately_when_task_done(self):
        """Should return immediately when deferred cleanup task is already done."""
        from olmlx.engine.inference import _await_deferred_cleanup

        done_task = asyncio.create_task(asyncio.sleep(0))
        await done_task  # let it finish
        _inf_mod._deferred_cleanup_task = done_task
        await _await_deferred_cleanup()  # should not raise
        _inf_mod._deferred_cleanup_task = None

    @pytest.mark.asyncio
    async def test_waits_for_running_task(self):
        """Should wait for a running deferred cleanup task to complete."""
        from olmlx.engine.inference import _await_deferred_cleanup

        completed = False

        async def slow_cleanup():
            nonlocal completed
            await asyncio.sleep(0.05)
            completed = True

        _inf_mod._deferred_cleanup_task = asyncio.create_task(slow_cleanup())
        await _await_deferred_cleanup()
        assert completed
        _inf_mod._deferred_cleanup_task = None

    @pytest.mark.asyncio
    async def test_raises_on_timeout(self):
        """Should raise ServerBusyError if cleanup doesn't complete within timeout."""
        import contextlib

        from olmlx.engine.inference import (
            ServerBusyError,
            _await_deferred_cleanup,
        )

        async def stuck_cleanup():
            await asyncio.sleep(999)

        _inf_mod._deferred_cleanup_task = asyncio.create_task(stuck_cleanup())
        try:
            with patch("olmlx.engine.inference._DEFERRED_WAIT_TIMEOUT", 0.05):
                with pytest.raises(ServerBusyError, match="did not complete"):
                    await _await_deferred_cleanup()
        finally:
            _inf_mod._deferred_cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await _inf_mod._deferred_cleanup_task
            _inf_mod._deferred_cleanup_task = None

    @pytest.mark.asyncio
    async def test_does_not_cancel_cleanup_task(self):
        """asyncio.shield should prevent wait_for from cancelling the cleanup task."""
        import contextlib

        from olmlx.engine.inference import _await_deferred_cleanup

        task_was_cancelled = False

        async def slow_cleanup():
            nonlocal task_was_cancelled
            try:
                await asyncio.sleep(999)
            except asyncio.CancelledError:
                task_was_cancelled = True
                raise

        _inf_mod._deferred_cleanup_task = asyncio.create_task(slow_cleanup())
        try:
            with patch("olmlx.engine.inference._DEFERRED_WAIT_TIMEOUT", 0.05):
                with pytest.raises(Exception):
                    await _await_deferred_cleanup()
            # The task should NOT have been cancelled by shield
            assert not task_was_cancelled
        finally:
            _inf_mod._deferred_cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await _inf_mod._deferred_cleanup_task
            _inf_mod._deferred_cleanup_task = None


class TestEstimateKvCacheBytes:
    """Tests for _estimate_kv_cache_bytes()."""

    def _make_model_args(
        self,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_size=4096,
        head_dim=None,
    ):
        args = MagicMock(spec=[])
        args.num_hidden_layers = num_hidden_layers
        args.num_attention_heads = num_attention_heads
        args.num_key_value_heads = num_key_value_heads
        args.hidden_size = hidden_size
        if head_dim is not None:
            args.head_dim = head_dim
        return args

    def _make_model(self, **kwargs):
        model = MagicMock(spec=[])
        model.args = self._make_model_args(**kwargs)
        return model

    def test_basic_estimate(self):
        """KV cache bytes = layers * 2 * kv_heads * head_dim * tokens * 2 * MEMORY_SAFETY_FACTOR."""
        model = self._make_model(
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            hidden_size=4096,
        )
        # head_dim = 4096 / 32 = 128
        # raw = 32 * 2 * 8 * 128 * 1000 * 2 = 131_072_000
        # expected = int(131_072_000 * 1.3) = 170_393_600
        result = _estimate_kv_cache_bytes(model, 1000)
        assert result == int(131_072_000 * _inf_mod.MEMORY_SAFETY_FACTOR)

    def test_zero_tokens(self):
        model = self._make_model()
        assert _estimate_kv_cache_bytes(model, 0) == 0

    def test_gqa_fallback_to_mha(self):
        """When num_key_value_heads is missing, fall back to num_attention_heads."""
        model = MagicMock()
        model.args = MagicMock(spec=[])
        model.args.num_hidden_layers = 32
        model.args.num_attention_heads = 32
        model.args.hidden_size = 4096
        # Delete num_key_value_heads so hasattr returns False
        del model.args.num_key_value_heads
        # head_dim = 128, kv_heads = 32 (fallback)
        # raw = 32 * 2 * 32 * 128 * 100 * 2 = 52_428_800
        result = _estimate_kv_cache_bytes(model, 100)
        assert result == int(52_428_800 * _inf_mod.MEMORY_SAFETY_FACTOR)

    def test_large_prompt(self):
        """Test with a 22k token prompt (the crash scenario)."""
        model = self._make_model(
            num_hidden_layers=28,
            num_attention_heads=28,
            num_key_value_heads=4,
            hidden_size=3584,
        )
        # head_dim = 3584 / 28 = 128
        # raw = 28 * 2 * 4 * 128 * 22000 * 2 = 1_261_568_000 (~1.2 GB)
        result = _estimate_kv_cache_bytes(model, 22000)
        assert result == int(1_261_568_000 * _inf_mod.MEMORY_SAFETY_FACTOR)

    def test_explicit_head_dim(self):
        """When model.args.head_dim exists, use it instead of hidden_size // num_heads."""
        model = self._make_model(
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            hidden_size=4096,
            head_dim=256,
        )
        # raw = 32 * 2 * 8 * 256 * 100 * 2 = 26_214_400
        result = _estimate_kv_cache_bytes(model, 100)
        assert result == int(26_214_400 * _inf_mod.MEMORY_SAFETY_FACTOR)

    def test_vlm_fallback_to_language_model_args(self):
        """Falls back to model.language_model.args for VLM models."""
        model = MagicMock(spec=[])
        model.language_model = MagicMock(spec=[])
        model.language_model.args = self._make_model_args(
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            hidden_size=4096,
        )
        result = _estimate_kv_cache_bytes(model, 1000)
        assert result == int(131_072_000 * _inf_mod.MEMORY_SAFETY_FACTOR)

    def test_raises_when_no_args_found(self):
        """Raises AttributeError when model has no discoverable args."""
        model = MagicMock(spec=[])
        with pytest.raises(AttributeError, match="no 'args' attribute"):
            _estimate_kv_cache_bytes(model, 1000)


class TestKvCachePreflightCheck:
    """Tests for the pre-flight KV cache memory check in _stream_completion."""

    @pytest.fixture
    def mock_lm(self):
        lm = MagicMock()
        lm.is_vlm = False
        lm.is_distributed = False
        lm.speculative_decoder = None
        type(lm).is_speculative = property(
            lambda self: self.speculative_decoder is not None
        )
        lm.model = MagicMock()
        lm.model.args = MagicMock()
        lm.model.args.num_hidden_layers = 32
        lm.model.args.num_attention_heads = 32
        lm.model.args.num_key_value_heads = 8
        lm.model.args.hidden_size = 4096
        lm.tokenizer = MagicMock()
        lm.text_tokenizer = MagicMock()
        lm.prompt_cache_store = MagicMock()
        lm.prompt_cache_store.get.return_value = None
        lm.prompt_cache_store.evict_all_to_disk = MagicMock()
        lm.active_refs = 0
        return lm

    @pytest.mark.asyncio
    async def test_rejects_when_kv_cache_exceeds_memory(self, mock_lm):
        """_stream_completion should raise MemoryError when estimated KV cache
        would push Metal memory over the limit."""
        stats = TimingStats()

        # Simulate: current memory is 10GB, limit is 12GB, KV cache needs 3GB
        # 10GB + 3GB > 12GB → should reject
        total_mem = 24 * 1024**3  # 24GB system
        current_metal = 10 * 1024**3  # 10GB used
        kv_estimate = 3 * 1024**3  # 3GB KV cache

        # Pre-acquire the real lock so the finally block can release it
        await _inference_lock.acquire()
        try:
            with (
                patch(
                    "olmlx.utils.memory.get_system_memory_bytes", return_value=total_mem
                ),
                patch(
                    "olmlx.utils.memory.get_metal_memory", return_value=current_metal
                ),
                patch("olmlx.engine.inference.settings") as mock_settings,
                patch("olmlx.engine.inference.mx"),
                patch(
                    "olmlx.engine.inference._estimate_kv_cache_bytes",
                    return_value=kv_estimate,
                ),
                patch("olmlx.engine.inference._safe_sync"),
                patch(
                    "olmlx.engine.inference._await_deferred_cleanup",
                    new_callable=AsyncMock,
                ),
                patch(
                    "olmlx.engine.inference._acquire_inference_lock",
                    new_callable=AsyncMock,
                ),
            ):
                mock_settings.memory_limit_fraction = 0.5  # 12GB limit
                mock_settings.prompt_cache = False

                gen = _inf_mod._stream_completion(
                    mock_lm, list(range(22000)), 512, {}, stats
                )
                with pytest.raises(MemoryError, match="prompt too long"):
                    async for _ in gen:
                        pass
        finally:
            # Ensure lock is released even if test fails
            if _inference_lock.locked():
                _inference_lock.release()

    @pytest.mark.asyncio
    async def test_allows_when_kv_cache_fits(self, mock_lm):
        """Should proceed normally when KV cache fits in memory."""
        stats = TimingStats()

        total_mem = 24 * 1024**3
        current_metal = 5 * 1024**3  # 5GB used
        kv_estimate = 1 * 1024**3  # 1GB KV cache — fits within 12GB limit

        # Create a proper async iterable mock with one token
        final_token = StreamToken(
            text="hi",
            token=1,
            prompt_tokens=100,
            generation_tokens=1,
            prompt_tps=100.0,
            generation_tps=50.0,
        )
        mock_stream = AsyncMock()
        mock_stream.__aiter__ = MagicMock(return_value=mock_stream)
        mock_stream.__anext__ = AsyncMock(side_effect=[final_token, StopAsyncIteration])
        mock_stream._thread = None
        mock_stream.drain_and_join = AsyncMock()
        mock_stream.cancel = MagicMock()

        # Pre-acquire the real lock so the finally block can release it
        await _inference_lock.acquire()
        try:
            with (
                patch(
                    "olmlx.utils.memory.get_system_memory_bytes", return_value=total_mem
                ),
                patch(
                    "olmlx.utils.memory.get_metal_memory", return_value=current_metal
                ),
                patch("olmlx.engine.inference.settings") as mock_settings,
                patch("olmlx.engine.inference.mx"),
                patch(
                    "olmlx.engine.inference._estimate_kv_cache_bytes",
                    return_value=kv_estimate,
                ),
                patch("olmlx.engine.inference._safe_sync"),
                patch(
                    "olmlx.engine.inference._await_deferred_cleanup",
                    new_callable=AsyncMock,
                ),
                patch(
                    "olmlx.engine.inference._acquire_inference_lock",
                    new_callable=AsyncMock,
                ),
                patch(
                    "olmlx.engine.inference.async_mlx_stream", return_value=mock_stream
                ),
                patch("olmlx.engine.inference.parse_keep_alive", return_value=300),
            ):
                mock_settings.memory_limit_fraction = 0.5  # 12GB limit
                mock_settings.prompt_cache = False
                mock_settings.default_keep_alive = "5m"

                gen = _inf_mod._stream_completion(
                    mock_lm, list(range(100)), 512, {}, stats
                )
                # Should not raise — just exhaust the generator
                chunks = []
                async for chunk in gen:
                    chunks.append(chunk)
                # Generator completed without MemoryError
        finally:
            if _inference_lock.locked():
                _inference_lock.release()


class TestPrefillMemoryCallback:
    """Tests for memory checking during prefill."""

    def test_prefill_callback_returns_false_on_memory_exceeded(self):
        """The prefill progress callback should return False when memory exceeds limit."""
        from olmlx.utils.streaming import _make_prefill_progress

        cancel_event = MagicMock()
        cancel_event.is_set.return_value = False

        # memory_limit = 12GB, current memory at 13GB → over limit
        mock_mx = MagicMock()
        mock_mx.get_active_memory.return_value = 10 * 1024**3
        mock_mx.get_cache_memory.return_value = 3 * 1024**3

        callback = _make_prefill_progress(
            cancel_event, memory_limit=12 * 1024**3, mx_module=mock_mx
        )
        # At 50% progress, should check memory and return False
        result = callback(0.5)
        assert result is False

    def test_prefill_callback_returns_true_when_memory_ok(self):
        """The prefill progress callback should return True when memory is fine."""
        from olmlx.utils.streaming import _make_prefill_progress

        cancel_event = MagicMock()
        cancel_event.is_set.return_value = False

        mock_mx = MagicMock()
        mock_mx.get_active_memory.return_value = 5 * 1024**3
        mock_mx.get_cache_memory.return_value = 1 * 1024**3

        callback = _make_prefill_progress(
            cancel_event, memory_limit=12 * 1024**3, mx_module=mock_mx
        )
        result = callback(0.5)
        assert result is True

    def test_prefill_callback_respects_cancel_event(self):
        """Cancel event should still take priority."""
        from olmlx.utils.streaming import _make_prefill_progress

        cancel_event = MagicMock()
        cancel_event.is_set.return_value = True

        callback = _make_prefill_progress(cancel_event, memory_limit=12 * 1024**3)
        result = callback(0.5)
        assert result is False

    def test_prefill_callback_no_limit(self):
        """With no memory limit, should only check cancel event."""
        from olmlx.utils.streaming import _make_prefill_progress

        cancel_event = MagicMock()
        cancel_event.is_set.return_value = False

        callback = _make_prefill_progress(cancel_event, memory_limit=0)
        result = callback(0.5)
        assert result is True
