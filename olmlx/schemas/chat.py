from typing import Any

from pydantic import BaseModel, Field

from olmlx.schemas.common import ModelName, ModelOptions


class ToolCallFunction(BaseModel):
    name: str
    arguments: dict[str, Any]


class ToolCall(BaseModel):
    function: ToolCallFunction


class Message(BaseModel):
    role: str
    content: str = Field("", max_length=1_000_000)
    images: list[str] | None = None
    tool_calls: list[ToolCall] | None = None


class Tool(BaseModel):
    type: str = "function"
    function: dict[str, Any]


class ChatRequest(BaseModel):
    model: ModelName
    messages: list[Message]
    tools: list[Tool] | None = None
    format: str | None = None
    stream: bool = True
    options: ModelOptions | None = None
    keep_alive: str | None = None


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
