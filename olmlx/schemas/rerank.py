from __future__ import annotations

from pydantic import BaseModel, field_validator

from olmlx.schemas.common import ModelName, validate_non_empty_text_input


class RerankRequest(BaseModel):
    model: ModelName
    query: str
    documents: list[str]
    top_n: int | None = None
    max_tokens_per_doc: int = 4096
    return_documents: bool = False
    keep_alive: int | str | None = None

    @field_validator("query")
    @classmethod
    def _query_non_empty(cls, v: str) -> str:
        validate_non_empty_text_input(v, "query")
        if not v.strip():
            raise ValueError("query cannot be blank")
        return v

    @field_validator("documents")
    @classmethod
    def _documents_non_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("documents must be a non-empty list")
        return v

    @field_validator("top_n")
    @classmethod
    def _top_n_positive(cls, v: int | None) -> int | None:
        if v is not None and v <= 0:
            raise ValueError("top_n must be a positive integer")
        return v


class RerankResult(BaseModel):
    index: int
    relevance_score: float
    document: str | None = None


class RerankResponse(BaseModel):
    id: str
    results: list[RerankResult]
    meta: dict = {}
