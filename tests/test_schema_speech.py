"""SpeechRequest schema (#367)."""

import pytest
from pydantic import ValidationError

from olmlx.schemas.audio import SpeechRequest


def test_defaults():
    r = SpeechRequest(model="kokoro", input="hello", voice="alloy")
    assert r.response_format == "mp3"
    assert r.speed == 1.0


def test_speed_bounds():
    with pytest.raises(ValidationError):
        SpeechRequest(model="m", input="x", voice="alloy", speed=5.0)
    with pytest.raises(ValidationError):
        SpeechRequest(model="m", input="x", voice="alloy", speed=0.1)


def test_required_fields():
    with pytest.raises(ValidationError):
        SpeechRequest(model="m", voice="alloy")  # missing input
