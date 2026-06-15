"""Schemas for the autonomous agent HTTP surface (issue #446)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class AgentRunCreateRequest(BaseModel):
    """Create + start an autonomous agent run."""

    goal: str = Field(min_length=1)
    #: Falls back to ``settings.agent_model`` when omitted.
    model: str | None = None
    #: Per-run budget overrides (fall back to ``OLMLX_AGENT_*`` defaults).
    max_iterations: int | None = Field(default=None, gt=0)
    token_budget: int | None = Field(default=None, gt=0)
    wallclock_timeout: float | None = Field(default=None, gt=0)

    @field_validator("goal")
    @classmethod
    def _goal_non_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("goal cannot be empty or blank")
        return v


class AgentRunResponse(BaseModel):
    id: str
    goal: str
    status: str
    model: str
    parent_id: str | None = None
    depth: int = 0
    iterations: int = 0
    tokens: int = 0
    result: str | None = None
    error: str | None = None
    created_at: float
    updated_at: float

    @classmethod
    def from_run(cls, run: dict[str, Any]) -> "AgentRunResponse":
        return cls(
            id=run["id"],
            goal=run["goal"],
            status=run["status"],
            model=run["model"],
            parent_id=run["parent_id"],
            depth=run["depth"],
            iterations=run["iterations"],
            tokens=run["tokens"],
            result=run["result"],
            error=run["error"],
            created_at=run["created_at"],
            updated_at=run["updated_at"],
        )


class AgentRunListResponse(BaseModel):
    runs: list[AgentRunResponse]
