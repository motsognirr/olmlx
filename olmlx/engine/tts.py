"""Voice resolution for /v1/audio/speech (Kokoro). Issue #367.

Maps the six OpenAI voice names to Kokoro voice ids and validates native
Kokoro ids passed through directly. Pure / dependency-free so routers and
tests import it without pulling FastAPI or mlx.
"""

from __future__ import annotations


class UnknownVoiceError(ValueError):
    """Raised when a requested voice is neither an OpenAI alias nor a known
    Kokoro voice id. Surfaced as HTTP 422 by the router."""


# Kokoro-82M ships 54 voices (prince-canuma/Kokoro-82M, voices/*.pt).
KOKORO_VOICES: frozenset[str] = frozenset(
    {
        "af_alloy",
        "af_aoede",
        "af_bella",
        "af_heart",
        "af_jessica",
        "af_kore",
        "af_nicole",
        "af_nova",
        "af_river",
        "af_sarah",
        "af_sky",
        "am_adam",
        "am_echo",
        "am_eric",
        "am_fenrir",
        "am_liam",
        "am_michael",
        "am_onyx",
        "am_puck",
        "am_santa",
        "bf_alice",
        "bf_emma",
        "bf_isabella",
        "bf_lily",
        "bm_daniel",
        "bm_fable",
        "bm_george",
        "bm_lewis",
        "ef_dora",
        "em_alex",
        "em_santa",
        "ff_siwis",
        "hf_alpha",
        "hf_beta",
        "hm_omega",
        "hm_psi",
        "if_sara",
        "im_nicola",
        "jf_alpha",
        "jf_gongitsune",
        "jf_nezumi",
        "jf_tebukuro",
        "jm_kumo",
        "pf_dora",
        "pm_alex",
        "pm_santa",
        "zf_xiaobei",
        "zf_xiaoni",
        "zf_xiaoxiao",
        "zf_xiaoyi",
        "zm_yunjian",
        "zm_yunxi",
        "zm_yunxia",
        "zm_yunyang",
    }
)

# OpenAI voice name -> Kokoro voice id. Names chosen to match where Kokoro
# ships an identically-named voice; shimmer has no Kokoro twin (-> af_sky).
OPENAI_VOICE_MAP: dict[str, str] = {
    "alloy": "af_alloy",
    "echo": "am_echo",
    "fable": "bm_fable",
    "onyx": "am_onyx",
    "nova": "af_nova",
    "shimmer": "af_sky",
}


def resolve_voice(voice: str) -> str:
    """Map an OpenAI voice name to a Kokoro voice id, or pass a native id
    through. Raise :class:`UnknownVoiceError` for anything unrecognized."""
    if voice in OPENAI_VOICE_MAP:
        return OPENAI_VOICE_MAP[voice]
    if voice in KOKORO_VOICES:
        return voice
    raise UnknownVoiceError(
        f"Unknown voice '{voice}'. Use an OpenAI voice "
        f"({', '.join(sorted(OPENAI_VOICE_MAP))}) or a Kokoro voice id."
    )
