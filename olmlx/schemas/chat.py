from typing import Any

from pydantic import BaseModel, Field, field_validator

from olmlx.schemas.common import ModelName, ModelOptions


class ToolCallFunction(BaseModel):
    name: str
    arguments: dict[str, Any]


class ToolCall(BaseModel):
    function: ToolCallFunction


class Message(BaseModel):
    role: str
    content: str = Field("", max_length=1_000_000)
    thinking: str | None = Field(None, max_length=1_000_000)
    images: list[str] | None = None
    tool_calls: list[ToolCall] | None = None


class Tool(BaseModel):
    type: str = "function"
    function: dict[str, Any]


class ChatRequest(BaseModel):
    model: ModelName
    messages: list[Message]
    tools: list[Tool] | None = None
    # Ollama accepts either ``"json"`` (any JSON value) or a JSON Schema
    # dict (strict adherence). Both are passed through to xgrammar.
    format: str | dict[str, Any] | None = None
    stream: bool = True
    think: bool | str | None = None
    options: ModelOptions | None = None
    keep_alive: int | str | None = None

    @field_validator("messages")
    @classmethod
    def validate_messages_non_empty(cls, v: list[Message]) -> list[Message]:
        if not v:
            raise ValueError("messages cannot be empty")
        return v


class ChatResponse(BaseModel):
    model: str
    created_at: str
    message: Message
    done: bool
    done_reason: str | None = None
    total_duration: int | None = None
    load_duration: int | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration: int | None = None
    eval_count: int | None = None
    eval_duration: int | None = None
