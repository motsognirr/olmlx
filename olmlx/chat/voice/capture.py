"""Push-to-talk mic capture via sounddevice (issue #444).

Records 16 kHz mono int16 audio to a temp WAV file (a format Whisper's ffmpeg
decode reads directly). Recording starts immediately and runs until the user
presses Enter — deterministic, no VAD.
"""

from __future__ import annotations

import tempfile
import wave
from pathlib import Path

import numpy as np

_INSTALL_HINT = (
    "Voice mode needs the 'sounddevice' package (PortAudio). "
    "Install it with: uv sync --extra voice"
)


def _require_sounddevice():
    import sys

    sd = sys.modules.get("sounddevice", "__unset__")
    if sd is None:  # explicitly stubbed-out in tests / unavailable
        raise RuntimeError(_INSTALL_HINT)
    try:
        import sounddevice as sd  # noqa: F811
    except Exception as exc:  # ImportError or PortAudio load failure
        raise RuntimeError(_INSTALL_HINT) from exc
    return sd


def _wait_for_stop() -> None:
    """Block until the user presses Enter."""
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        pass


def record_to_wav(*, samplerate: int = 16000, out_dir: Path | None = None) -> str:
    """Record from the default mic until Enter; return the temp WAV path."""
    sd = _require_sounddevice()

    chunks: list[np.ndarray] = []

    def _callback(indata, frames, time_info, status):  # noqa: ANN001
        chunks.append(np.asarray(indata, dtype=np.int16).copy())

    with sd.InputStream(
        samplerate=samplerate, channels=1, dtype="int16", callback=_callback
    ):
        _wait_for_stop()

    if chunks:
        audio = np.concatenate(chunks).reshape(-1)
    else:
        audio = np.zeros(0, dtype=np.int16)

    directory = Path(out_dir) if out_dir is not None else None
    fd, path = tempfile.mkstemp(
        suffix=".wav", dir=str(directory) if directory else None
    )
    import os

    os.close(fd)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(samplerate)
        w.writeframes(audio.tobytes())
    return path
