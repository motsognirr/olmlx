"""OpenAI-compatible audio transcription route (/v1/audio/transcriptions).

Backed by mlx-whisper. Whisper models are managed through ModelManager;
see engine/inference.py::generate_transcription and CLAUDE.md.
"""

from __future__ import annotations


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
