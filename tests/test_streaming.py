"""Tests for mlx_ollama.utils.streaming."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from mlx_ollama.utils.streaming import StreamToken, async_mlx_stream


class TestStreamToken:
    def test_defaults(self):
        tok = StreamToken(
            text="hi", token=1, prompt_tokens=5,
            generation_tokens=1, prompt_tps=100.0, generation_tps=50.0,
        )
        assert tok.text == "hi"
        assert tok.finish_reason is None

    def test_with_finish_reason(self):
        tok = StreamToken(
            text="", token=None, prompt_tokens=5,
            generation_tokens=10, prompt_tps=100.0, generation_tps=50.0,
            finish_reason="stop",
        )
        assert tok.finish_reason == "stop"


class TestAsyncMlxStream:
    @pytest.mark.asyncio
    async def test_text_model_stream(self):
        mock_responses = [
            MagicMock(text="Hello", token=1, prompt_tokens=5,
                      generation_tokens=1, prompt_tps=100.0, generation_tps=50.0,
                      finish_reason=None),
            MagicMock(text=" world", token=2, prompt_tokens=5,
                      generation_tokens=2, prompt_tps=100.0, generation_tps=50.0,
                      finish_reason="stop"),
        ]

        mock_mlx_lm = MagicMock()
        mock_mlx_lm.stream_generate = MagicMock(return_value=iter(mock_responses))

        with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
            tokens = []
            async for tok in async_mlx_stream(
                MagicMock(), MagicMock(), "prompt", max_tokens=10, is_vlm=False,
            ):
                tokens.append(tok)

        assert len(tokens) == 2
        assert tokens[0].text == "Hello"
        assert tokens[1].text == " world"

    @pytest.mark.asyncio
    async def test_vlm_stream(self):
        mock_responses = [
            MagicMock(text="Description", token=None, prompt_tokens=10,
                      generation_tokens=1, prompt_tps=50.0, generation_tps=30.0),
        ]

        mock_mlx_vlm = MagicMock()
        mock_mlx_vlm.stream_generate = MagicMock(return_value=iter(mock_responses))

        with patch.dict("sys.modules", {"mlx_vlm": mock_mlx_vlm}):
            tokens = []
            async for tok in async_mlx_stream(
                MagicMock(), MagicMock(), "prompt",
                max_tokens=10, is_vlm=True, images=["img.jpg"],
            ):
                tokens.append(tok)

        assert len(tokens) == 1
        assert tokens[0].text == "Description"

    @pytest.mark.asyncio
    async def test_error_propagation(self):
        mock_mlx_lm = MagicMock()
        mock_mlx_lm.stream_generate = MagicMock(
            side_effect=RuntimeError("GPU error"),
        )

        with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
            with pytest.raises(RuntimeError, match="GPU error"):
                async for _ in async_mlx_stream(
                    MagicMock(), MagicMock(), "prompt", max_tokens=10, is_vlm=False,
                ):
                    pass
