from typing import Any
from typing_extensions import TypedDict

from pydantic import BaseModel, Field

from olmlx.schemas.common import ModelName, ModelOptions


class ToolCallFunctionDict(TypedDict):
    """TypedDict for tool call function in OpenAI format."""

    name: str
    arguments: dict[str, Any]


class ToolCallDict(TypedDict):
    """TypedDict for tool call in OpenAI format."""

    id: str
    type: str
    function: ToolCallFunctionDict


class ToolFunctionDict(TypedDict, total=False):
    """TypedDict for tool function definition in OpenAI format."""

    name: str
    description: str
    parameters: dict[str, Any]


class ToolDict(TypedDict):
    """TypedDict for tool in OpenAI format."""

    type: str
    function: ToolFunctionDict


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
