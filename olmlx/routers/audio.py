"""OpenAI-compatible audio transcription route (/v1/audio/transcriptions).

Backed by mlx-whisper. Whisper models are managed through ModelManager;
see engine/inference.py::generate_transcription and CLAUDE.md.
"""

from __future__ import annotations

import os
import tempfile

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import PlainTextResponse

from olmlx.config import settings
from olmlx.engine.inference import generate_transcription
from olmlx.schemas.audio import (
    TranscriptionResponse,
    TranscriptionSegment,
    VerboseTranscriptionResponse,
)

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
