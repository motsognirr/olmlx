from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from olmlx.schemas.common import ModelName, validate_non_empty_text_input


class RerankRequest(BaseModel):
    model: ModelName
    query: str
    documents: list[str]
    top_n: int | None = None
    max_tokens_per_doc: int = Field(default=4096, gt=0)
    return_documents: bool = False
    keep_alive: int | str | None = None

    @field_validator("query")
    @classmethod
    def validate_query_non_empty(cls, v: str) -> str:
        # Reject empty and whitespace-only queries (a blank query can't rank).
        if not v.strip():
            raise ValueError("query cannot be empty or blank")
        return v

    @field_validator("documents")
    @classmethod
    def validate_documents_non_empty(cls, v: list[str]) -> list[str]:
        # Rejects an empty list or any empty-string document (matches embed).
        return validate_non_empty_text_input(v, "documents")

    @field_validator("top_n")
    @classmethod
    def validate_top_n_positive(cls, v: int | None) -> int | None:
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
    meta: dict[str, Any] = {}
