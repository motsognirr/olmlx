"""VoiceIO: glue between mic/speaker and the STT/TTS engines (issue #444).

Half-duplex by nature: STT, the chat LLM, and TTS all serialize on the single
inference lock, so listen() and speak() never overlap generation.
"""

from __future__ import annotations

import asyncio
import os
import re

from olmlx.chat.voice import capture, playback
from olmlx.engine.inference import generate_speech, generate_transcription
from olmlx.engine.model_manager import ModelManager
from olmlx.engine.tts import resolve_voice

_SENTENCE_RE = re.compile(r".+?(?:[.!?](?:\s+|$)|$)", re.DOTALL)

# Kokoro emits 24 kHz mono float32 (matches engine/inference.generate_speech).
_TTS_SAMPLE_RATE = 24000


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentence-ish chunks for incremental synthesis."""
    if not text or not text.strip():
        return []
    out = [m.group().strip() for m in _SENTENCE_RE.finditer(text)]
    return [s for s in out if s]


class VoiceIO:
    def __init__(
        self,
        *,
        manager: ModelManager,
        stt_model: str,
        tts_model: str,
        voice: str,
    ) -> None:
        self._manager = manager
        self._stt_model = stt_model
        self._tts_model = tts_model
        # Resolve the OpenAI alias / native Kokoro id once, up front, so a bad
        # voice fails at session start (raises UnknownVoiceError) rather than on
        # the first reply.
        self._voice = resolve_voice(voice)

    async def listen(self) -> str:
        """Record push-to-talk audio and return the transcribed text."""
        # record_to_wav blocks on input() until the user presses Enter; run it
        # off the event loop so other async work isn't frozen during recording.
        path = await asyncio.to_thread(capture.record_to_wav)
        try:
            result = await generate_transcription(self._manager, self._stt_model, path)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
        return (result.get("text") or "").strip()

    async def speak(self, text: str) -> None:
        """Synthesize and play ``text`` one sentence at a time.

        ``generate_speech`` is a streaming async generator (issue #367) yielding
        24 kHz float32 segments; each is played as it arrives.
        """
        for sentence in split_into_sentences(text):
            async for segment in generate_speech(
                self._manager, self._tts_model, sentence, voice=self._voice
            ):
                # play() blocks on sd.wait() until the segment finishes; keep it
                # off the event loop.
                await asyncio.to_thread(playback.play, segment, _TTS_SAMPLE_RATE)
