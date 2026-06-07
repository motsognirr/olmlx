"""Tests for olmlx.routers.responses and its schemas."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from olmlx.schemas.responses import ResponsesRequest
from olmlx.utils.timing import TimingStats


class TestResponsesRequestSchema:
    def test_string_input_accepted(self):
        req = ResponsesRequest(model="qwen3", input="hello")
        assert req.input == "hello"
        assert req.store is True
        assert req.stream is False

    def test_list_input_accepted(self):
        req = ResponsesRequest(
            model="qwen3",
            input=[{"role": "user", "content": "hi"}],
        )
        assert isinstance(req.input, list)

    def test_defaults(self):
        req = ResponsesRequest(model="qwen3", input="x")
        assert req.previous_response_id is None
        assert req.tools is None
        assert req.max_output_tokens is None
