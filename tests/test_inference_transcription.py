"""Tests for generate_transcription."""

from unittest.mock import AsyncMock, patch

import pytest

from olmlx.engine.inference import generate_transcription


@pytest.mark.asyncio
async def test_generate_transcription_injects_and_calls(mock_manager):
    lm = mock_manager._loaded["qwen3:latest"]
    lm.is_whisper = True
    mock_manager.ensure_loaded = AsyncMock(return_value=lm)

    result = {"text": "hello", "segments": [], "language": "en"}

    import importlib

    # ``import mlx_whisper.transcribe as wt`` would bind the *function*
    # ``transcribe`` (the package __init__ shadows the submodule name); use
    # importlib to get the genuine submodule that holds ModelHolder.
    wt = importlib.import_module("mlx_whisper.transcribe")

    # Reset holder so we can assert injection happened.
    wt.ModelHolder.model = None
    wt.ModelHolder.model_path = None

    with patch("mlx_whisper.transcribe.transcribe", return_value=result) as mock_tx:
        out = await generate_transcription(
            mock_manager,
            "whisper-turbo",
            "/tmp/clip.wav",
            language="en",
            prompt="hi",
            temperature=0.0,
            word_timestamps=False,
        )

    assert out == result
    # Our managed model was injected into the holder.
    assert wt.ModelHolder.model is lm.model
    mock_tx.assert_called_once()
    _, kwargs = mock_tx.call_args
    assert kwargs["initial_prompt"] == "hi"
    assert kwargs["language"] == "en"


@pytest.mark.asyncio
async def test_generate_transcription_ffmpeg_missing(mock_manager):
    lm = mock_manager._loaded["qwen3:latest"]
    lm.is_whisper = True
    mock_manager.ensure_loaded = AsyncMock(return_value=lm)

    with patch(
        "mlx_whisper.transcribe.transcribe",
        side_effect=FileNotFoundError("ffmpeg"),
    ):
        with pytest.raises(ValueError, match="ffmpeg"):
            await generate_transcription(mock_manager, "whisper-turbo", "/tmp/clip.wav")


@pytest.mark.asyncio
async def test_generate_transcription_rejects_non_whisper_model(mock_manager):
    lm = mock_manager._loaded["qwen3:latest"]
    lm.is_whisper = False  # a text LLM, not a whisper model
    mock_manager.ensure_loaded = AsyncMock(return_value=lm)

    with patch("mlx_whisper.transcribe.transcribe") as mock_tx:
        with pytest.raises(ValueError, match="not a Whisper model"):
            await generate_transcription(mock_manager, "qwen3", "/tmp/clip.wav")

    # Must reject before touching mlx_whisper at all.
    mock_tx.assert_not_called()
