from pydantic import BaseModel, Field

from olmlx.schemas.common import ModelOptions


class ToolCallFunction(BaseModel):
    name: str
    arguments: dict


class ToolCall(BaseModel):
    function: ToolCallFunction


class Message(BaseModel):
    role: str
    content: str = ""
    images: list[str] | None = None
    tool_calls: list[ToolCall] | None = None


class Tool(BaseModel):
    type: str = "function"
    function: dict


class ChatRequest(BaseModel):
    model: str = Field(..., min_length=1, max_length=256)
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
