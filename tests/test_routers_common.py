"""Tests for routers/common.py shared utilities."""

import json

from olmlx.routers.common import format_error


class TestFormatError:
    def test_json_structure(self):
        result = json.loads(format_error("test-model"))
        assert result["model"] == "test-model"
        assert "created_at" in result
        assert result["done"] is True
        assert result["done_reason"] == "error"
        assert "error" in result

    def test_model_propagation(self):
        result = json.loads(format_error("my-fancy-model"))
        assert result["model"] == "my-fancy-model"

    def test_ends_with_newline(self):
        assert format_error("m").endswith("\n")
