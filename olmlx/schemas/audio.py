"""Response schemas for /v1/audio/transcriptions."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class TranscriptionResponse(BaseModel):
    """Default `response_format=json` body."""

    text: str


class TranscriptionSegment(BaseModel):
    """One whisper segment. Tolerant of extra keys mlx-whisper may add."""

    model_config = ConfigDict(extra="ignore")

    id: int = 0
    seek: int = 0
    start: float = 0.0
    end: float = 0.0
    text: str = ""
    tokens: list[int] = []
    temperature: float = 0.0
    avg_logprob: float = 0.0
    compression_ratio: float = 0.0
    no_speech_prob: float = 0.0


class VerboseTranscriptionResponse(BaseModel):
    """`response_format=verbose_json` body (OpenAI-compatible shape)."""

    task: str = "transcribe"
    language: str = ""
    duration: float = 0.0
    text: str = ""
    segments: list[TranscriptionSegment] = []
