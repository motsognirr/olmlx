import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware

from mlx_ollama.config import settings
from mlx_ollama.engine.model_manager import ModelManager
from mlx_ollama.engine.registry import ModelRegistry
from mlx_ollama.models.store import ModelStore
from mlx_ollama.routers import anthropic, blobs, chat, embed, generate, manage, models, openai, status

logger = logging.getLogger("mlx_ollama")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    registry = ModelRegistry()
    registry.load()
    manager = ModelManager(registry)
    manager.start_expiry_checker()
    store = ModelStore(registry)

    settings.models_dir.mkdir(parents=True, exist_ok=True)

    app.state.registry = registry
    app.state.model_manager = manager
    app.state.model_store = store

    logger.info(
        "MLX Ollama server started on %s:%d", settings.host, settings.port
    )
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


def create_app() -> FastAPI:
    app = FastAPI(title="MLX Ollama", lifespan=lifespan)
    app.add_middleware(ForceJSONMiddleware)

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
