"""Tests for srt/vtt formatting helpers in olmlx.routers.audio."""

from olmlx.routers.audio import _format_timestamp, srt_from_segments, vtt_from_segments

SEGMENTS = [
    {"start": 0.0, "end": 2.5, "text": " Hello world"},
    {"start": 2.5, "end": 3661.0, "text": " Second line"},
]


def test_format_timestamp_srt():
    assert _format_timestamp(0.0, decimal=",") == "00:00:00,000"
    assert _format_timestamp(3661.5, decimal=",") == "01:01:01,500"


def test_format_timestamp_vtt():
    assert _format_timestamp(3661.5, decimal=".") == "01:01:01.500"


def test_srt_from_segments():
    out = srt_from_segments(SEGMENTS)
    lines = out.splitlines()
    assert lines[0] == "1"
    assert lines[1] == "00:00:00,000 --> 00:00:02,500"
    assert lines[2] == "Hello world"
    assert lines[3] == ""
    assert lines[4] == "2"
    assert lines[5] == "00:00:02,500 --> 01:01:01,000"
    assert lines[6] == "Second line"


def test_vtt_from_segments():
    out = vtt_from_segments(SEGMENTS)
    lines = out.splitlines()
    assert lines[0] == "WEBVTT"
    assert lines[1] == ""
    assert lines[2] == "00:00:00.000 --> 00:00:02.500"
    assert lines[3] == "Hello world"


def test_empty_segments():
    assert srt_from_segments([]) == ""
    assert vtt_from_segments([]) == "WEBVTT\n"
