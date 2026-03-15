from pydantic import BaseModel, Field

from olmlx.schemas.common import ModelName


# --- Chat Completions ---


class OpenAIChatMessage(BaseModel):
    role: str
    content: str | None = None
    name: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None


class OpenAIChatRequest(BaseModel):
    model: ModelName
    messages: list[OpenAIChatMessage]
    temperature: float | None = Field(None, ge=0, le=2)
    top_p: float | None = Field(None, ge=0, le=1)
    n: int = Field(1, ge=1, le=1, description="Only n=1 is supported.")
    stream: bool = False
    stop: str | list[str] | None = None
    max_tokens: int | None = Field(None, ge=1)
    max_completion_tokens: int | None = Field(None, ge=1)
    presence_penalty: float = Field(0.0, ge=-2, le=2)
    frequency_penalty: float = Field(0.0, ge=-2, le=2)
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None
    seed: int | None = None


class OpenAIUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class OpenAIChoice(BaseModel):
    index: int = 0
    message: OpenAIChatMessage | None = None
    delta: OpenAIChatMessage | None = None
    finish_reason: str | None = None


class OpenAIChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[OpenAIChoice]
    usage: OpenAIUsage | None = None


# --- Completions ---


class OpenAICompletionRequest(BaseModel):
    model: ModelName
    prompt: str | list[str]
    temperature: float | None = Field(None, ge=0, le=2)
    top_p: float | None = Field(None, ge=0, le=1)
    n: int = Field(1, ge=1, le=1, description="Only n=1 is supported.")
    stream: bool = False
    stop: str | list[str] | None = None
    max_tokens: int | None = Field(None, ge=1)
    presence_penalty: float = Field(0.0, ge=-2, le=2)
    frequency_penalty: float = Field(0.0, ge=-2, le=2)
    seed: int | None = None


class OpenAICompletionChoice(BaseModel):
    index: int = 0
    text: str = ""
    finish_reason: str | None = None


class OpenAICompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[OpenAICompletionChoice]
    usage: OpenAIUsage | None = None


# --- Models ---


class OpenAIModel(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "olmlx"


class OpenAIModelList(BaseModel):
    object: str = "list"
    data: list[OpenAIModel]


# --- Embeddings ---


class OpenAIEmbeddingRequest(BaseModel):
    model: ModelName
    input: str | list[str]
    encoding_format: str = "float"


class OpenAIEmbeddingData(BaseModel):
    object: str = "embedding"
    index: int = 0
    embedding: list[float]


class OpenAIEmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[OpenAIEmbeddingData]
    model: str
    usage: OpenAIUsage = OpenAIUsage()
