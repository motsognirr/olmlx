"""Tests for mlx_ollama.engine.inference (non-GPU parts)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mlx_ollama.engine.inference import (
    _build_generate_kwargs,
    _extract_images,
    _inject_tools_into_system,
    _apply_chat_template_text,
    generate_chat,
    generate_completion,
    generate_embeddings,
)
from mlx_ollama.engine.model_manager import LoadedModel, ModelManager
from mlx_ollama.engine.template_caps import TemplateCaps
from mlx_ollama.utils.streaming import StreamToken
from mlx_ollama.utils.timing import TimingStats


class TestBuildGenerateKwargs:
    def test_empty_options(self):
        assert _build_generate_kwargs(None) == {}
        assert _build_generate_kwargs({}) == {}

    def test_temperature(self):
        result = _build_generate_kwargs({"temperature": 0.7})
        assert result == {"temp": 0.7}

    def test_temperature_vlm(self):
        result = _build_generate_kwargs({"temperature": 0.7}, is_vlm=True)
        assert result == {"temperature": 0.7}

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
        assert result["temp"] == 0.5
        assert result["top_p"] == 0.9
        assert result["top_k"] == 40
        assert result["seed"] == 42
        assert result["max_tokens"] == 100
        assert result["repetition_penalty"] == 1.1
        assert result["repetition_context_size"] == 64
        assert result["min_p"] == 0.05

    def test_stop_text_model(self):
        result = _build_generate_kwargs({"stop": [".", "\n"]})
        assert result["stop"] == [".", "\n"]

    def test_stop_vlm_ignored(self):
        result = _build_generate_kwargs({"stop": [".", "\n"]}, is_vlm=True)
        assert "stop" not in result

    def test_frequency_penalty(self):
        result = _build_generate_kwargs({"frequency_penalty": 0.5})
        assert result["frequency_penalty"] == 0.5

    def test_presence_penalty(self):
        result = _build_generate_kwargs({"presence_penalty": 0.3})
        assert result["presence_penalty"] == 0.3

    def test_zero_penalty_not_passed(self):
        result = _build_generate_kwargs({"frequency_penalty": 0.0, "presence_penalty": 0.0})
        assert "frequency_penalty" not in result
        assert "presence_penalty" not in result

    def test_unknown_options_ignored(self):
        result = _build_generate_kwargs({"unknown_key": 99})
        assert result == {}


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
        tools = [{"function": {"name": "search", "description": "Search the web", "parameters": {}}}]
        result = _inject_tools_into_system(messages, tools)
        assert result[0]["role"] == "system"
        assert "search" in result[0]["content"]
        assert "You are helpful." in result[0]["content"]
        # Original should not be modified
        assert "search" not in messages[0]["content"]

    def test_without_system(self):
        messages = [{"role": "user", "content": "hi"}]
        tools = [{"function": {"name": "func", "description": "A function", "parameters": {}}}]
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
            messages, tokenize=False, add_generation_prompt=True,
        )

    def test_with_tools_supported(self):
        tokenizer = MagicMock()
        tokenizer.apply_chat_template = MagicMock(return_value="prompt with tools")
        messages = [{"role": "user", "content": "hi"}]
        tools = [{"type": "function", "function": {"name": "f"}}]
        caps = TemplateCaps(supports_tools=True, supports_enable_thinking=True)
        result = _apply_chat_template_text(tokenizer, messages, tools, caps)
        call_kwargs = tokenizer.apply_chat_template.call_args[1]
        assert call_kwargs["tools"] == tools
        assert call_kwargs["enable_thinking"] is False

    def test_with_tools_not_supported(self):
        tokenizer = MagicMock()
        tokenizer.apply_chat_template = MagicMock(return_value="prompt")
        messages = [{"role": "user", "content": "hi"}]
        tools = [{"type": "function", "function": {"name": "f", "description": "desc", "parameters": {}}}]
        caps = TemplateCaps(supports_tools=False)
        result = _apply_chat_template_text(tokenizer, messages, tools, caps)
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
        tools = [{"type": "function", "function": {"name": "f", "description": "d", "parameters": {}}}]
        caps = TemplateCaps(supports_tools=True)
        result = _apply_chat_template_text(tokenizer, messages, tools, caps)
        assert result == "fallback prompt"
        assert call_count[0] == 2

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
        tokenizer.apply_chat_template = MagicMock(side_effect=TypeError("all calls fail"))
        tools = [{"type": "function", "function": {"name": "f", "description": "d", "parameters": {}}}]
        caps = TemplateCaps(supports_tools=True)
        with pytest.raises(RuntimeError, match="Chat template failed even without tools"):
            _apply_chat_template_text(tokenizer, [], tools, caps)


class TestApplyChatTemplateVlm:
    def test_vlm_template(self):
        from mlx_ollama.engine.inference import _apply_chat_template_vlm

        mock_mlx_vlm = MagicMock()
        mock_mlx_vlm.apply_chat_template.return_value = "vlm prompt"
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_model.config = {"model_type": "qwen2_vl"}

        with patch.dict("sys.modules", {"mlx_vlm": mock_mlx_vlm}):
            result = _apply_chat_template_vlm(
                mock_processor, mock_model,
                [{"role": "user", "content": "describe"}],
                images=["img.jpg"],
            )

        assert result == "vlm prompt"
        mock_mlx_vlm.apply_chat_template.assert_called_once()

    def test_vlm_template_no_config(self):
        from mlx_ollama.engine.inference import _apply_chat_template_vlm

        mock_mlx_vlm = MagicMock()
        mock_mlx_vlm.apply_chat_template.return_value = "result"
        mock_processor = MagicMock()
        mock_model = MagicMock(spec=[])  # no config attr

        with patch.dict("sys.modules", {"mlx_vlm": mock_mlx_vlm}):
            result = _apply_chat_template_vlm(
                mock_processor, mock_model,
                [{"role": "user", "content": "hi"}],
            )
        assert result == "result"


class TestGenerateCompletion:
    @pytest.mark.asyncio
    async def test_non_streaming(self, mock_manager):
        mock_mx = MagicMock()
        mock_mx.metal.clear_cache = MagicMock()
        mock_mx.core = mock_mx

        mock_mlx_lm = MagicMock()
        mock_mlx_lm.generate.return_value = "Generated output"

        with patch("mlx_ollama.engine.inference.mx", mock_mx):
            with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
                with patch("mlx_ollama.engine.inference.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
                    mock_thread.return_value = "Generated output"
                    result = await generate_completion(
                        mock_manager, "qwen3", "Hello", stream=False,
                    )

        assert result["text"] == "Generated output"
        assert result["done"] is True

    @pytest.mark.asyncio
    async def test_streaming(self, mock_manager):
        mock_mx = MagicMock()
        mock_mx.metal.clear_cache = MagicMock()

        async def mock_stream(*args, **kwargs):
            yield StreamToken(text="Hello", token=1, prompt_tokens=5,
                              generation_tokens=1, prompt_tps=100.0, generation_tps=50.0)
            yield StreamToken(text=" world", token=2, prompt_tokens=5,
                              generation_tokens=2, prompt_tps=100.0, generation_tps=50.0)

        with patch("mlx_ollama.engine.inference.mx", mock_mx):
            with patch("mlx_ollama.engine.inference.async_mlx_stream", return_value=mock_stream()):
                gen = await generate_completion(
                    mock_manager, "qwen3", "Hello", stream=True,
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
        mock_mx.metal.clear_cache = MagicMock()

        mock_manager._loaded["qwen3:latest"].tokenizer.apply_chat_template = MagicMock(
            return_value="formatted prompt"
        )

        with patch("mlx_ollama.engine.inference.mx", mock_mx):
            with patch("mlx_ollama.engine.inference.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
                mock_thread.return_value = "Chat response"
                result = await generate_chat(
                    mock_manager, "qwen3",
                    [{"role": "user", "content": "hi"}],
                    stream=False,
                )

        assert result["text"] == "Chat response"
        assert result["done"] is True

    @pytest.mark.asyncio
    async def test_with_tools(self, mock_manager):
        mock_mx = MagicMock()
        mock_mx.metal.clear_cache = MagicMock()

        mock_manager._loaded["qwen3:latest"].tokenizer.apply_chat_template = MagicMock(
            return_value="formatted prompt with tools"
        )

        with patch("mlx_ollama.engine.inference.mx", mock_mx):
            with patch("mlx_ollama.engine.inference.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
                mock_thread.return_value = "tool response"
                result = await generate_chat(
                    mock_manager, "qwen3",
                    [{"role": "user", "content": "search"}],
                    tools=[{"type": "function", "function": {"name": "search"}}],
                    stream=False,
                )

        assert result["text"] == "tool response"


class TestFullCompletionInner:
    @pytest.mark.asyncio
    async def test_result_with_text_attr(self, mock_manager):
        """Test when result has .text attribute (GenerationResult dataclass)."""
        from mlx_ollama.engine.inference import _full_completion_inner
        lm = mock_manager._loaded["qwen3:latest"]

        mock_result = MagicMock()
        mock_result.text = "result text"

        with patch("mlx_ollama.engine.inference.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.return_value = mock_result
            result = await _full_completion_inner(
                lm, "prompt", 100, {}, TimingStats(),
            )

        assert result["text"] == "result text"

    @pytest.mark.asyncio
    async def test_result_as_string(self, mock_manager):
        from mlx_ollama.engine.inference import _full_completion_inner
        lm = mock_manager._loaded["qwen3:latest"]

        with patch("mlx_ollama.engine.inference.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.return_value = "plain string"
            result = await _full_completion_inner(
                lm, "prompt", 100, {}, TimingStats(),
            )

        assert result["text"] == "plain string"

    @pytest.mark.asyncio
    async def test_result_other_type(self, mock_manager):
        from mlx_ollama.engine.inference import _full_completion_inner
        lm = mock_manager._loaded["qwen3:latest"]

        with patch("mlx_ollama.engine.inference.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.return_value = 42  # unusual type
            result = await _full_completion_inner(
                lm, "prompt", 100, {}, TimingStats(),
            )

        assert result["text"] == "42"

    @pytest.mark.asyncio
    async def test_vlm_completion(self, mock_manager):
        from mlx_ollama.engine.inference import _full_completion_inner
        lm = mock_manager._loaded["qwen3:latest"]
        lm.is_vlm = True

        mock_mlx_vlm = MagicMock()
        with patch("mlx_ollama.engine.inference.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.return_value = "vlm output"
            with patch.dict("sys.modules", {"mlx_vlm": mock_mlx_vlm}):
                result = await _full_completion_inner(
                    lm, "prompt", 100, {}, TimingStats(), images=["img.jpg"],
                )

        assert result["text"] == "vlm output"


class TestGenerateChatVlm:
    @pytest.mark.asyncio
    async def test_vlm_without_tools(self, mock_manager):
        """Test chat with VLM model (no tools)."""
        lm = mock_manager._loaded["qwen3:latest"]
        lm.is_vlm = True

        mock_mx = MagicMock()
        mock_mx.metal.clear_cache = MagicMock()

        mock_mlx_vlm = MagicMock()
        mock_mlx_vlm.apply_chat_template.return_value = "vlm prompt"

        with patch("mlx_ollama.engine.inference.mx", mock_mx):
            with patch.dict("sys.modules", {"mlx_vlm": mock_mlx_vlm}):
                with patch("mlx_ollama.engine.inference.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
                    mock_thread.return_value = "vlm response"
                    result = await generate_chat(
                        mock_manager, "qwen3",
                        [{"role": "user", "content": "describe", "images": ["img.jpg"]}],
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
