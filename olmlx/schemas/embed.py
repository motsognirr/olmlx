from pydantic import BaseModel, Field


class EmbedRequest(BaseModel):
    model: str = Field(..., min_length=1, max_length=256)
    input: str | list[str]
    truncate: bool = True
    options: dict | None = None
    keep_alive: str | None = None


class EmbedResponse(BaseModel):
    model: str
    embeddings: list[list[float]]
    total_duration: int | None = None
    load_duration: int | None = None
    prompt_eval_count: int | None = None


class EmbeddingsRequest(BaseModel):
    model: str = Field(..., min_length=1, max_length=256)
    prompt: str
    options: dict | None = None
    keep_alive: str | None = None


class EmbeddingsResponse(BaseModel):
    embedding: list[float]
