from typing import Any

from pydantic import BaseModel, Field

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
    # tool_use fields
    id: str | None = None
    name: str | None = None
    input: dict | None = None
    # tool_result fields
    tool_use_id: str | None = None
    content: str | list[Any] | None = None
    is_error: bool | None = None

    model_config = {"extra": "allow"}


class AnthropicMessage(BaseModel):
    role: str
    content: str | list[AnthropicContentBlock]


class AnthropicThinkingParam(BaseModel):
    type: str
    budget_tokens: int | None = None

    model_config = {"extra": "allow"}


class AnthropicMessagesRequest(BaseModel):
    model: ModelName
    messages: list[AnthropicMessage]
    max_tokens: int = Field(4096, ge=1)
    stream: bool = False
    temperature: float | None = Field(None, ge=0, le=1)
    top_p: float | None = Field(None, ge=0, le=1)
    top_k: int | None = Field(None, ge=1)  # Anthropic spec: top_k >= 1
    stop_sequences: list[str] | None = None
    system: str | list[AnthropicContentBlock] | None = None
    tools: list[AnthropicTool] | None = None
    tool_choice: dict | None = None
    thinking: AnthropicThinkingParam | None = None
    metadata: dict | None = None

    model_config = {"extra": "allow"}


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
    stop_reason: str | None = "end_turn"
    stop_sequence: str | None = None
    usage: AnthropicUsage
