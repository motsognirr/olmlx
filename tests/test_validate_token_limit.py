"""Tests for schemas/common.py validate_token_limit helper."""

import pytest

from olmlx.schemas.common import validate_token_limit


class TestValidateTokenLimit:
    def test_under_limit(self, monkeypatch):
        monkeypatch.setattr("olmlx.config.settings.max_tokens_limit", 1000)
        assert validate_token_limit(500, "max_tokens") == 500

    def test_at_limit(self, monkeypatch):
        monkeypatch.setattr("olmlx.config.settings.max_tokens_limit", 1000)
        assert validate_token_limit(1000, "max_tokens") == 1000

    def test_over_limit(self, monkeypatch):
        monkeypatch.setattr("olmlx.config.settings.max_tokens_limit", 1000)
        with pytest.raises(ValueError, match="max_tokens 1001 exceeds configured limit 1000"):
            validate_token_limit(1001, "max_tokens")

    def test_error_message_includes_field_name(self, monkeypatch):
        monkeypatch.setattr("olmlx.config.settings.max_tokens_limit", 100)
        with pytest.raises(ValueError, match="num_predict 200"):
            validate_token_limit(200, "num_predict")
