"""HTTP surface for autonomous agent runs (issue #446).

Registered in ``app.py`` only when ``OLMLX_AGENT_ENABLED`` is true, so the routes
simply do not exist when the agent is off. ``AgentService`` lives on
``app.state.agent_service``.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from olmlx.schemas.agent import (
    AgentRunCreateRequest,
    AgentRunListResponse,
    AgentRunResponse,
)

router = APIRouter()


def _service(request: Request):
    service = getattr(request.app.state, "agent_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="agent service unavailable")
    return service


@router.post("/v1/agent/runs", response_model=AgentRunResponse)
async def create_run(req: AgentRunCreateRequest, request: Request) -> AgentRunResponse:
    service = _service(request)
    run = await service.create_and_start(
        goal=req.goal,
        model=req.model,
        max_iterations=req.max_iterations,
        token_budget=req.token_budget,
        wallclock_timeout=req.wallclock_timeout,
    )
    return AgentRunResponse.from_run(run)


@router.get("/v1/agent/runs", response_model=AgentRunListResponse)
async def list_runs(request: Request) -> AgentRunListResponse:
    service = _service(request)
    runs = await service.list_runs()
    return AgentRunListResponse(runs=[AgentRunResponse.from_run(r) for r in runs])


@router.get("/v1/agent/runs/{run_id}", response_model=AgentRunResponse)
async def get_run(run_id: str, request: Request) -> AgentRunResponse:
    service = _service(request)
    run = await service.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"run {run_id!r} not found")
    return AgentRunResponse.from_run(run)


@router.post("/v1/agent/runs/{run_id}/cancel", response_model=AgentRunResponse)
async def cancel_run(run_id: str, request: Request) -> AgentRunResponse:
    service = _service(request)
    run = await service.cancel(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"run {run_id!r} not found")
    return AgentRunResponse.from_run(run)


@router.post("/v1/agent/runs/{run_id}/resume", response_model=AgentRunResponse)
async def resume_run(run_id: str, request: Request) -> AgentRunResponse:
    service = _service(request)
    run = await service.resume(run_id)  # raises ValueError (→400) if not resumable
    if run is None:
        raise HTTPException(status_code=404, detail=f"run {run_id!r} not found")
    return AgentRunResponse.from_run(run)


@router.get("/v1/agent/runs/{run_id}/events")
async def stream_events(run_id: str, request: Request) -> StreamingResponse:
    service = _service(request)
    run = await service.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"run {run_id!r} not found")

    async def _gen():
        async for event in service.stream_events(run_id):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(_gen(), media_type="text/event-stream")
