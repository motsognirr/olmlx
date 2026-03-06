import logging
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from mlx_ollama.config import settings
from mlx_ollama.engine.model_manager import ModelManager
from mlx_ollama.engine.registry import ModelRegistry
from mlx_ollama.models.store import ModelStore
from mlx_ollama.routers import (
    anthropic,
    blobs,
    chat,
    embed,
    generate,
    manage,
    models,
    openai,
    status,
)

logger = logging.getLogger("mlx_ollama")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    registry = ModelRegistry()
    registry.load()
    store = ModelStore(registry)
    manager = ModelManager(registry, store)
    manager.start_expiry_checker()

    settings.models_dir.mkdir(parents=True, exist_ok=True)

    app.state.registry = registry
    app.state.model_manager = manager
    app.state.model_store = store

    logger.info("MLX Ollama server started on %s:%d", settings.host, settings.port)
    yield

    # Shutdown
    await manager.stop()
    logger.info("MLX Ollama server stopped")


class ForceJSONMiddleware(BaseHTTPMiddleware):
    """Treat all POST/PUT/PATCH bodies as JSON, matching Ollama's behavior.

    curl -d sends application/x-www-form-urlencoded by default, but the real
    Ollama server accepts it as JSON regardless of Content-Type.
    """

    async def dispatch(self, request: Request, call_next):
        if request.method in ("POST", "PUT", "PATCH"):
            content_type = request.headers.get("content-type", "")
            if "json" not in content_type:
                scope = request.scope
                headers = dict(scope["headers"])
                headers[b"content-type"] = b"application/json"
                scope["headers"] = list(headers.items())
        return await call_next(request)


def _make_error_response(
    path: str,
    status_code: int,
    msg: str,
    anthropic_type: str,
    openai_type: str,
    openai_code: str,
) -> JSONResponse:
    """Build a JSON error response in the appropriate format for the API surface."""
    if path.startswith("/v1/messages"):
        content = {"type": "error", "error": {"type": anthropic_type, "message": msg}}
    elif path.startswith("/v1/"):
        content = {"error": {"message": msg, "type": openai_type, "code": openai_code}}
    else:
        content = {"error": msg}
    return JSONResponse(status_code=status_code, content=content)


def create_app() -> FastAPI:
    app = FastAPI(title="MLX Ollama", lifespan=lifespan)
    app.add_middleware(ForceJSONMiddleware)

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        msg = str(exc)
        logger.warning("ValueError on %s: %s", request.url.path, msg)
        return _make_error_response(
            request.url.path,
            400,
            msg,
            "invalid_request_error",
            "invalid_request_error",
            "invalid_value",
        )

    @app.exception_handler(RuntimeError)
    async def runtime_error_handler(request: Request, exc: RuntimeError):
        msg = str(exc)
        logger.error("RuntimeError on %s: %s", request.url.path, msg)
        return _make_error_response(
            request.url.path,
            500,
            msg,
            "api_error",
            "server_error",
            "internal_error",
        )

    @app.exception_handler(Exception)
    async def general_error_handler(request: Request, exc: Exception):
        msg = f"{type(exc).__name__}: {exc}"
        logger.error(
            "Unhandled exception on %s: %s\n%s",
            request.url.path,
            msg,
            traceback.format_exc(),
        )
        return _make_error_response(
            request.url.path,
            500,
            msg,
            "api_error",
            "server_error",
            "internal_error",
        )

    app.include_router(status.router)
    app.include_router(generate.router)
    app.include_router(chat.router)
    app.include_router(models.router)
    app.include_router(manage.router)
    app.include_router(embed.router)
    app.include_router(blobs.router)
    app.include_router(openai.router)
    app.include_router(anthropic.router)

    return app
