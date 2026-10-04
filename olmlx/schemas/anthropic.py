from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from olmlx.schemas.common import ModelName


class AnthropicToolInputSchema(BaseModel):
    type: str = "object"
    properties: dict[str, Any] | None = None
    required: list[str] | None = None
    # Allow arbitrary extra keys for JSON Schema passthrough
    model_config = {"extra": "allow"}


class AnthropicTool(BaseModel):
    name: str
    description: str | None = None
    input_schema: AnthropicToolInputSchema


class AnthropicContentBlock(BaseModel):
    type: str = "text"
    text: str | None = None
    # thinking fields
    thinking: str | None = None
    signature: str | None = None
    # tool_use fields
    id: str | None = None
    name: str | None = None
    input: dict | None = None
    # tool_result fields
    tool_use_id: str | None = None
    content: str | list[Any] | None = None
    is_error: bool | None = None
    # image fields
    source: dict | None = None

    model_config = {"extra": "allow"}


_MAX_CONTENT_LENGTH = 1_000_000
_MAX_CONTENT_BLOCKS = 1_000


# Roles the Anthropic router renders (#710, mirrors chat.py #696): anything else
# renders to nothing in the chat template and silently drops the turn. No
# "tool" — tool results are tool_result blocks inside a user message. "system"
# is deliberately kept although not in the Anthropic wire spec: the router folds
# inline system messages into the leading system block (Claude Code sends them).
_VALID_ROLES = frozenset({"system", "user", "assistant"})


class AnthropicMessage(BaseModel):
    role: str
    content: str | list[AnthropicContentBlock]

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v not in _VALID_ROLES:
            raise ValueError(f"role must be one of {sorted(_VALID_ROLES)}, got {v!r}")
        return v

    @field_validator("content")
    @classmethod
    def validate_content_length(cls, v: str | list) -> str | list:
        if isinstance(v, str) and len(v) > _MAX_CONTENT_LENGTH:
            raise ValueError(
                f"content length {len(v)} exceeds limit {_MAX_CONTENT_LENGTH}"
            )
        if isinstance(v, list) and len(v) > _MAX_CONTENT_BLOCKS:
            raise ValueError(
                f"content block count {len(v)} exceeds limit {_MAX_CONTENT_BLOCKS}"
            )
        return v


class AnthropicThinkingParam(BaseModel):
    type: str
    # Range-checked (>= 0, < max_tokens) by AnthropicMessagesRequest, and only
    # for type == "enabled" — see validate_budget_below_max_tokens.
    budget_tokens: int | None = None

    model_config = {"extra": "allow"}


class AnthropicMessagesRequest(BaseModel):
    model: ModelName
    messages: list[AnthropicMessage]
    # Required per the Anthropic API spec — omitting it must yield a 400, not a
    # silent default. count_tokens uses AnthropicCountTokensRequest, which makes
    # this optional (the real count_tokens endpoint takes no max_tokens).
    max_tokens: int = Field(..., ge=1)
    stream: bool = False

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, v: int) -> int:
        from olmlx.schemas.common import validate_token_limit

        return validate_token_limit(v, "max_tokens")

    @field_validator("messages")
    @classmethod
    def validate_messages_non_empty(
        cls, v: list[AnthropicMessage]
    ) -> list[AnthropicMessage]:
        if not v:
            raise ValueError("messages cannot be empty")
        return v

    temperature: float | None = Field(None, ge=0, le=1)
    top_p: float | None = Field(None, ge=0, le=1)
    top_k: int | None = Field(None, ge=1)  # Anthropic spec: top_k >= 1
    stop_sequences: list[str] | None = None
    system: str | list[AnthropicContentBlock] | None = None
    tools: list[AnthropicTool] | None = None
    tool_choice: dict | None = None
    thinking: AnthropicThinkingParam | None = None

    @field_validator("system")
    @classmethod
    def validate_system_length(cls, v: str | list | None) -> str | list | None:
        if isinstance(v, str) and len(v) > _MAX_CONTENT_LENGTH:
            raise ValueError(
                f"system length {len(v)} exceeds limit {_MAX_CONTENT_LENGTH}"
            )
        if isinstance(v, list) and len(v) > _MAX_CONTENT_BLOCKS:
            raise ValueError(
                f"system block count {len(v)} exceeds limit {_MAX_CONTENT_BLOCKS}"
            )
        return v

    metadata: dict | None = None

    model_config = {"extra": "allow"}

    @model_validator(mode="after")
    def validate_budget_below_max_tokens(self) -> "AnthropicMessagesRequest":
        # Match Anthropic (#743): budget_tokens must be < max_tokens, else 400
        # invalid_request_error. Anthropic's 1024 minimum is deliberately NOT
        # enforced — small budgets are legitimate for local models. Gated on
        # type == "enabled" (the only type Anthropic defines the rule for):
        # adaptive / forward-compat types must not be 400'd, and a budget
        # >= max_tokens can never trigger before max_tokens anyway.
        # A negative budget is likewise a 400 (rather than being silently
        # dropped downstream) under the same gate; 0 is meaningful (close
        # immediately).
        thinking = self.thinking
        if (
            thinking is None
            or thinking.type != "enabled"
            or thinking.budget_tokens is None
        ):
            return self
        if thinking.budget_tokens < 0:
            raise ValueError(
                f"thinking.budget_tokens ({thinking.budget_tokens}) must be "
                "non-negative"
            )
        if thinking.budget_tokens >= self.max_tokens:
            raise ValueError(
                f"thinking.budget_tokens ({thinking.budget_tokens}) must be "
                f"less than max_tokens ({self.max_tokens})"
            )
        return self


class AnthropicCountTokensRequest(AnthropicMessagesRequest):
    """Request schema for /v1/messages/count_tokens.

    The real Anthropic count_tokens endpoint takes no max_tokens field, so —
    unlike /v1/messages — omitting it must not be an error. Re-declaring the
    field with a default makes it optional while inheriting every other field
    and validator from AnthropicMessagesRequest, except the thinking-budget
    range check, which is overridden to a no-op below (comparing against a
    placeholder max_tokens would be meaningless, and the budget is never
    honored here).
    """

    max_tokens: int = Field(1, ge=1)

    @model_validator(mode="after")
    def validate_budget_below_max_tokens(self) -> "AnthropicCountTokensRequest":
        # Overrides (disables) the parent's thinking-budget range check:
        # max_tokens here is a placeholder, never client-supplied.
        return self


class AnthropicUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class AnthropicTokenCountResponse(BaseModel):
    input_tokens: int


class AnthropicMessagesResponse(BaseModel):
    id: str
    type: str = "message"
    role: str = "assistant"
    content: list[AnthropicContentBlock]
    model: str
    stop_reason: str | None = None
    stop_sequence: str | None = None
    usage: AnthropicUsage
