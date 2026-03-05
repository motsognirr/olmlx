from pydantic import BaseModel, Field


# --- Chat Completions ---

class OpenAIChatMessage(BaseModel):
    role: str
    content: str | None = None
    name: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None


class OpenAIChatRequest(BaseModel):
    model: str
    messages: list[OpenAIChatMessage]
    temperature: float | None = None
    top_p: float | None = None
    n: int = 1
    stream: bool = False
    stop: str | list[str] | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
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
    model: str
    prompt: str | list[str]
    temperature: float | None = None
    top_p: float | None = None
    n: int = 1
    stream: bool = False
    stop: str | list[str] | None = None
    max_tokens: int | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
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
    owned_by: str = "mlx-ollama"


class OpenAIModelList(BaseModel):
    object: str = "list"
    data: list[OpenAIModel]


# --- Embeddings ---

class OpenAIEmbeddingRequest(BaseModel):
    model: str
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
