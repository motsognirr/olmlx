"""OpenAI-compatible audio transcription route (/v1/audio/transcriptions).

Backed by mlx-whisper. Whisper models are managed through ModelManager;
see engine/inference.py::generate_transcription and CLAUDE.md.
"""

from __future__ import annotations

import os
import tempfile

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import PlainTextResponse, Response, StreamingResponse

from olmlx.config import settings
from olmlx.engine.inference import generate_speech, generate_transcription
from olmlx.engine.tts import UnknownVoiceError, resolve_voice
from olmlx.schemas.audio import (
    SpeechRequest,
    TranscriptionResponse,
    TranscriptionSegment,
    VerboseTranscriptionResponse,
)
from olmlx.utils import audio as audio_utils

router = APIRouter()

_VALID_FORMATS = {"json", "verbose_json", "text", "srt", "vtt"}


def _format_timestamp(seconds: float, *, decimal: str) -> str:
    """Format seconds as HH:MM:SS<decimal>mmm (decimal "," for SRT, "." for VTT)."""
    if seconds < 0:
        seconds = 0.0
    millis = round(seconds * 1000.0)
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{decimal}{millis:03d}"


def srt_from_segments(segments: list[dict]) -> str:
    """Render whisper segments as SubRip (.srt) text."""
    blocks = []
    for i, seg in enumerate(segments, start=1):
        start = _format_timestamp(float(seg["start"]), decimal=",")
        end = _format_timestamp(float(seg["end"]), decimal=",")
        text = str(seg.get("text", "")).strip()
        blocks.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(blocks)


def vtt_from_segments(segments: list[dict]) -> str:
    """Render whisper segments as WebVTT (.vtt) text."""
    lines = ["WEBVTT", ""]
    for seg in segments:
        start = _format_timestamp(float(seg["start"]), decimal=".")
        end = _format_timestamp(float(seg["end"]), decimal=".")
        text = str(seg.get("text", "")).strip()
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


@router.post("/v1/audio/transcriptions")
async def create_transcription(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form(...),
    language: str | None = Form(None),
    prompt: str | None = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    timestamp_granularities: list[str] | None = Form(
        None, alias="timestamp_granularities[]"
    ),
):
    if response_format not in _VALID_FORMATS:
        raise ValueError(
            f"Unsupported response_format '{response_format}'. "
            f"Must be one of: {', '.join(sorted(_VALID_FORMATS))}."
        )

    word_timestamps = bool(
        timestamp_granularities and "word" in timestamp_granularities
    )

    manager = request.app.state.model_manager
    max_bytes = settings.audio_max_bytes

    # Stream the upload to a temp file under the size cap.
    suffix = os.path.splitext(file.filename or "")[1] or ".wav"
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    received = 0
    try:
        with os.fdopen(tmp_fd, "wb") as tmp:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                received += len(chunk)
                if received > max_bytes:
                    return PlainTextResponse(
                        f"Audio file exceeds the limit of {max_bytes} bytes "
                        f"(OLMLX_AUDIO_MAX_BYTES).",
                        status_code=413,
                    )
                tmp.write(chunk)

        result = await generate_transcription(
            manager,
            model,
            tmp_path,
            language=language,
            prompt=prompt,
            temperature=temperature,
            word_timestamps=word_timestamps,
        )
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    text = result.get("text", "")
    segments = result.get("segments", []) or []

    if response_format == "text":
        return PlainTextResponse(text)
    if response_format == "srt":
        return PlainTextResponse(srt_from_segments(segments))
    if response_format == "vtt":
        return PlainTextResponse(vtt_from_segments(segments))
    if response_format == "verbose_json":
        duration = float(segments[-1]["end"]) if segments else 0.0
        return VerboseTranscriptionResponse(
            language=result.get("language", "") or "",
            duration=duration,
            text=text,
            segments=[TranscriptionSegment(**s) for s in segments],
        )
    # default: json
    return TranscriptionResponse(text=text)


_TTS_SAMPLE_RATE = 24000


@router.post("/v1/audio/speech")
async def create_speech(request: Request, body: SpeechRequest):
    # --- cheap validation before touching the model -----------------------
    if not body.input.strip():
        raise HTTPException(status_code=422, detail="input must not be empty.")
    if len(body.input) > settings.tts_max_input_chars:
        raise HTTPException(
            status_code=413,
            detail=(
                f"input exceeds {settings.tts_max_input_chars} characters "
                "(OLMLX_TTS_MAX_INPUT_CHARS)."
            ),
        )
    try:
        audio_utils.validate_format(body.response_format)
    except audio_utils.UnsupportedFormatError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    try:
        kokoro_voice = resolve_voice(body.voice)
    except UnknownVoiceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    manager = request.app.state.model_manager
    fmt = body.response_format

    # Float segments -> PCM byte chunks.
    async def _pcm_chunks():
        async for seg in generate_speech(
            manager,
            body.model,
            body.input,
            voice=kokoro_voice,
            speed=body.speed,
        ):
            yield audio_utils.float_to_pcm16(seg)

    media_type = audio_utils.FORMAT_MEDIA_TYPES[fmt]

    # WAV is buffered so the header sizes are exact (small utterances).
    if fmt == "wav":
        # Collect into a list and join once — `pcm += chunk` in the loop is
        # O(n^2) (each += copies the whole accumulated buffer).
        parts: list[bytes] = []
        try:
            async for chunk in _pcm_chunks():
                parts.append(chunk)
        except ValueError as exc:  # non-TTS model etc.
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return Response(
            audio_utils.wav_bytes(b"".join(parts), sample_rate=_TTS_SAMPLE_RATE),
            media_type=media_type,
        )

    # Prime the first PCM chunk so model-load / non-TTS errors (ValueError)
    # become a clean HTTP 400 *before* the 200 streaming response starts. We
    # prime the PCM source itself rather than the encoded stream because ffmpeg
    # emits a container header before consuming any input, which would let the
    # upstream error slip past after the response had already begun.
    pcm_agen = _pcm_chunks().__aiter__()
    try:
        first = await pcm_agen.__anext__()
    except StopAsyncIteration:
        first = b""
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    async def _primed_pcm():
        if first:
            yield first
        async for chunk in pcm_agen:
            yield chunk

    # PCM streams raw; compressed formats stream through ffmpeg.
    if fmt == "pcm":
        byte_stream = _primed_pcm()
    else:
        byte_stream = audio_utils.ffmpeg_encode(
            _primed_pcm(), fmt=fmt, sample_rate=_TTS_SAMPLE_RATE
        )

    return StreamingResponse(byte_stream, media_type=media_type)
