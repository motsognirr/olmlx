import logging
import traceback
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from olmlx.config import settings
from olmlx.context import request_id_var
from olmlx.engine.inference import ServerBusyError
from olmlx.engine.model_manager import ModelLoadTimeoutError, ModelManager
from olmlx.engine.registry import ModelRegistry
from olmlx.models.store import ModelStore
from olmlx.routers import (
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

logger = logging.getLogger("olmlx")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    registry = ModelRegistry()
    registry.load()
    store = ModelStore(registry)

    # -- Experimental: Distributed inference --
    from olmlx.config import experimental

    distributed_group = None
    coordinator = None
    if experimental.distributed:
        from olmlx.engine.inference import set_distributed_coordinator

        # The CLI starts the ring backend and sideband server before uvicorn
        # (to avoid the slow transformers import blocking the sideband).
        # Retrieve the pre-created state from cmd_serve().
        from olmlx.cli import (
            _cli_distributed_coordinator,
            _cli_distributed_group,
            _cli_distributed_layer_counts,
            _cli_distributed_strategy,
        )

        distributed_group = _cli_distributed_group
        coordinator = _cli_distributed_coordinator

        if distributed_group is None or coordinator is None:
            raise RuntimeError(
                "Distributed mode requires starting via 'olmlx serve'. "
                "The ring backend and sideband server must be initialized "
                "before the app starts."
            )

        world_size = distributed_group.size()
        logger.info(
            "Distributed mode: rank %d, world_size %d, backend %s",
            distributed_group.rank(),
            world_size,
            experimental.distributed_backend,
        )
        import asyncio

        try:
            await asyncio.to_thread(coordinator.wait_for_workers, 60.0)
        except Exception:
            coordinator.close()
            raise
        set_distributed_coordinator(coordinator)

    distributed_strategy = "tensor"
    distributed_layer_counts = None
    if experimental.distributed:
        distributed_strategy = _cli_distributed_strategy
        distributed_layer_counts = _cli_distributed_layer_counts

    manager = ModelManager(
        registry,
        store,
        distributed_group=distributed_group,
        distributed_strategy=distributed_strategy,
        distributed_layer_counts=distributed_layer_counts,
    )
    manager.start_expiry_checker()

    settings.models_dir.mkdir(parents=True, exist_ok=True)

    app.state.registry = registry
    app.state.model_manager = manager
    app.state.model_store = store

    logger.info("olmlx server started on %s:%d", settings.host, settings.port)
    yield

    # Shutdown — drain in-flight requests before tearing down coordinator
    await manager.stop()
    if coordinator is not None:
        coordinator.broadcast_shutdown()
        coordinator.close()
        from olmlx.engine.inference import set_distributed_coordinator

        set_distributed_coordinator(None)
    logger.info("olmlx server stopped")


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


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Generate and attach request IDs for log tracing.

    The ContextVar is reset in the finally block before the response body is fully
    consumed. This relies on Starlette's BaseHTTPMiddleware running the inner app
    as a sub-task that copies the current context, so request_id_var is available
    for log messages during streaming inference.
    """

    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        token = request_id_var.set(request_id)
        response = None
        try:
            response = await call_next(request)
            return response
        finally:
            if response is not None:
                response.headers["X-Request-ID"] = request_id
            request_id_var.reset(token)


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
    app = FastAPI(title="olmlx", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    app.add_middleware(ForceJSONMiddleware)
    app.add_middleware(RequestIDMiddleware)

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

    @app.exception_handler(MemoryError)
    async def memory_error_handler(request: Request, exc: MemoryError):
        msg = str(exc)
        logger.error("MemoryError on %s: %s", request.url.path, msg)
        return _make_error_response(
            request.url.path,
            503,
            msg,
            "overloaded_error",
            "server_error",
            "model_too_large",
        )

    @app.exception_handler(ModelLoadTimeoutError)
    async def timeout_error_handler(request: Request, exc: ModelLoadTimeoutError):
        msg = str(exc)
        logger.error("TimeoutError on %s: %s", request.url.path, msg)
        return _make_error_response(
            request.url.path,
            504,
            msg,
            "api_error",
            "server_error",
            "timeout",
        )

    @app.exception_handler(ServerBusyError)
    async def server_busy_error_handler(request: Request, exc: ServerBusyError):
        msg = str(exc)
        logger.warning("ServerBusyError on %s: %s", request.url.path, msg)
        response = _make_error_response(
            request.url.path,
            503,
            msg,
            "overloaded_error",
            "server_error",
            "server_busy",
        )
        response.headers["Retry-After"] = "5"
        return response

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
