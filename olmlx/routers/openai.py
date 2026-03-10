import json
import time
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from olmlx.engine.inference import (
    generate_chat,
    generate_completion,
    generate_embeddings,
)
from olmlx.schemas.openai import (
    OpenAIChatMessage,
    OpenAIChatRequest,
    OpenAIChatResponse,
    OpenAIChoice,
    OpenAICompletionChoice,
    OpenAICompletionRequest,
    OpenAICompletionResponse,
    OpenAIEmbeddingData,
    OpenAIEmbeddingRequest,
    OpenAIEmbeddingResponse,
    OpenAIModel,
    OpenAIModelList,
    OpenAIUsage,
)

router = APIRouter()


def _make_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:8]}"


async def _stream_openai_sse(
    result, response_id, model, created, object_type, format_content, format_done
):
    """Shared SSE streaming for OpenAI-compatible endpoints.

    format_content(text) -> choices[0] dict for content chunks
    format_done() -> choices[0] dict for the final chunk
    """
    try:
        async for chunk in result:
            if chunk.get("cache_info"):
                continue
            if chunk.get("done"):
                data = {
                    "id": response_id,
                    "object": object_type,
                    "created": created,
                    "model": model,
                    "choices": [format_done()],
                }
                yield f"data: {json.dumps(data)}\n\n"
                yield "data: [DONE]\n\n"
            else:
                data = {
                    "id": response_id,
                    "object": object_type,
                    "created": created,
                    "model": model,
                    "choices": [format_content(chunk.get("text", ""))],
                }
                yield f"data: {json.dumps(data)}\n\n"
    finally:
        await result.aclose()


def _build_options(req) -> dict:
    opts = {}
    if req.temperature is not None:
        opts["temperature"] = req.temperature
    if req.top_p is not None:
        opts["top_p"] = req.top_p
    if req.seed is not None:
        opts["seed"] = req.seed
    if req.stop is not None:
        opts["stop"] = req.stop if isinstance(req.stop, list) else [req.stop]
    if req.frequency_penalty:
        opts["frequency_penalty"] = req.frequency_penalty
    if req.presence_penalty:
        opts["presence_penalty"] = req.presence_penalty
    return opts


@router.post("/v1/chat/completions")
async def openai_chat(req: OpenAIChatRequest, request: Request):
    manager = request.app.state.model_manager
    messages = [m.model_dump(exclude_none=True) for m in req.messages]
    options = _build_options(req)
    max_tokens = req.max_completion_tokens or req.max_tokens or 512
    chat_id = _make_id()
    created = int(time.time())

    if req.stream:
        result = await generate_chat(
            manager,
            req.model,
            messages,
            options,
            tools=req.tools,
            stream=True,
            max_tokens=max_tokens,
        )

        return StreamingResponse(
            _stream_openai_sse(
                result,
                chat_id,
                req.model,
                created,
                "chat.completion.chunk",
                lambda text: {
                    "index": 0,
                    "delta": {"role": "assistant", "content": text},
                    "finish_reason": None,
                },
                lambda: {"index": 0, "delta": {}, "finish_reason": "stop"},
            ),
            media_type="text/event-stream",
        )
    else:
        result = await generate_chat(
            manager,
            req.model,
            messages,
            options,
            tools=req.tools,
            stream=False,
            max_tokens=max_tokens,
        )
        text = result.get("text", "")
        stats = result.get("stats")
        usage = OpenAIUsage(
            prompt_tokens=stats.prompt_eval_count if stats else 0,
            completion_tokens=stats.eval_count if stats else 0,
            total_tokens=(stats.prompt_eval_count + stats.eval_count) if stats else 0,
        )
        return OpenAIChatResponse(
            id=chat_id,
            created=created,
            model=req.model,
            choices=[
                OpenAIChoice(
                    index=0,
                    message=OpenAIChatMessage(role="assistant", content=text),
                    finish_reason="stop",
                )
            ],
            usage=usage,
        )


@router.post("/v1/completions")
async def openai_completions(req: OpenAICompletionRequest, request: Request):
    manager = request.app.state.model_manager
    options = _build_options(req)
    prompt = req.prompt if isinstance(req.prompt, str) else req.prompt[0]
    max_tokens = req.max_tokens or 512
    comp_id = f"cmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    if req.stream:
        result = await generate_completion(
            manager,
            req.model,
            prompt,
            options,
            stream=True,
            max_tokens=max_tokens,
        )

        return StreamingResponse(
            _stream_openai_sse(
                result,
                comp_id,
                req.model,
                created,
                "text_completion",
                lambda text: {"index": 0, "text": text, "finish_reason": None},
                lambda: {"index": 0, "text": "", "finish_reason": "stop"},
            ),
            media_type="text/event-stream",
        )
    else:
        result = await generate_completion(
            manager,
            req.model,
            prompt,
            options,
            stream=False,
            max_tokens=max_tokens,
        )
        return OpenAICompletionResponse(
            id=comp_id,
            created=created,
            model=req.model,
            choices=[
                OpenAICompletionChoice(
                    index=0,
                    text=result.get("text", ""),
                    finish_reason="stop",
                )
            ],
        )


@router.get("/v1/models")
async def openai_list_models(request: Request):
    registry = request.app.state.registry
    models = registry.list_models()
    data = [OpenAIModel(id=name, created=int(time.time())) for name in models]
    return OpenAIModelList(data=data)


@router.post("/v1/embeddings")
async def openai_embeddings(req: OpenAIEmbeddingRequest, request: Request):
    manager = request.app.state.model_manager
    texts = req.input if isinstance(req.input, list) else [req.input]
    embeddings = await generate_embeddings(manager, req.model, texts)
    data = [
        OpenAIEmbeddingData(index=i, embedding=emb) for i, emb in enumerate(embeddings)
    ]
    return OpenAIEmbeddingResponse(
        data=data,
        model=req.model,
    )
