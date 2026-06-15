import logging
import time
import traceback
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from olmlx.config import settings
from olmlx.context import request_id_var, surface_var
from olmlx.engine.inference import ServerBusyError
from olmlx.engine.model_manager import (
    ModelLoadTimeoutError,
    ModelManager,
    SpectralCalibrationMissingError,
)
from olmlx.engine.registry import ModelRegistry
from olmlx.models.store import ModelStore
from olmlx.routers import (
    agent,
    anthropic,
    audio,
    blobs,
    chat,
    embed,
    generate,
    manage,
    models,
    openai,
    rerank,
    responses,
    status,
)
from olmlx.routers import metrics as metrics_router
from olmlx.utils import loop_affinity
from olmlx.utils import metrics as metrics_mod
from olmlx.utils import tracing as tracing_mod

logger = logging.getLogger("olmlx")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Bind the event-loop thread so the lock-free mutators (registry RMW,
    # unload's check-then-pop, prompt-cache stores) can enforce their
    # loop-affinity contract (issue #463).  Unbound in the finally so a
    # failed startup or repeated create_app() in tests leaves no stale
    # binding behind.
    loop_affinity.bind_loop_thread()
    try:
        async with _lifespan_inner(app):
            yield
    finally:
        loop_affinity.unbind_loop_thread()


@asynccontextmanager
async def _lifespan_inner(app: FastAPI):
    # SIGUSR1 dumps Python stacks of all threads to stderr — use
    # `kill -USR1 <pid>` to diagnose hangs without needing sudo/py-spy.
    import faulthandler
    import signal
    import sys

    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)

    # Startup
    registry = ModelRegistry()
    registry.load()
    store = ModelStore(registry)

    if settings.tracing:
        tracing_mod.init_tracing(settings)

    # -- Distributed inference --
    distributed_group = None
    coordinator = None
    distributed_strategy = "tensor"
    distributed_layer_counts = None
    if settings.distributed:
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
            settings.distributed_backend,
        )
        import asyncio

        try:
            await asyncio.to_thread(coordinator.wait_for_workers, 60.0)
        except Exception:
            coordinator.close()
            raise
        set_distributed_coordinator(coordinator)

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

    # Lazy gauge/counter collector reads the manager at scrape time.
    stats_collector = metrics_mod.OlmlxStatsCollector(manager)
    metrics_mod.REGISTRY.register(stats_collector)
    app.state.metrics_collector = stats_collector

    # Autonomous agent (#445): recover crash-orphaned runs at startup. The
    # service itself is built in create_app() (gated on agent_enabled) so the
    # routes exist without a lifespan; this only runs the async startup scan.
    agent_service = (
        getattr(app.state, "agent_service", None) if settings.agent_enabled else None
    )
    if agent_service is not None:
        await agent_service.startup()

    logger.info("olmlx server started on %s:%d", settings.host, settings.port)
    yield

    # Shutdown — drain in-flight requests before tearing down coordinator
    if agent_service is not None:
        await agent_service.aclose()
    await manager.stop()
    tracing_mod.shutdown_tracing()
    # Unregister so a subsequent create_app() (e.g. in tests) does not raise a
    # duplicate-collector error against the module-level REGISTRY.
    try:
        metrics_mod.REGISTRY.unregister(app.state.metrics_collector)
    except Exception:
        logger.debug(
            "Failed to unregister metrics collector during shutdown", exc_info=True
        )
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
            if "json" not in content_type and "multipart/form-data" not in content_type:
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


class MetricsMiddleware(BaseHTTPMiddleware):
    """Record per-request HTTP metrics and set the API-surface ContextVar.

    Mirrors RequestIDMiddleware's reliance on BaseHTTPMiddleware copying the
    current context into the inner-app sub-task, so surface_var is visible to
    engine inference instrumentation during streaming responses.
    """

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        surface_token = surface_var.set(metrics_mod.surface_for_path(path))
        metrics_mod.HTTP_IN_FLIGHT.inc()
        start = time.perf_counter()
        status_code = 500
        response = None
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            metrics_mod.HTTP_IN_FLIGHT.dec()
            # Prefer the matched route template to bound cardinality; fall back
            # to a literal for unmatched paths (404 with no route).
            route = request.scope.get("route")
            label_path = getattr(route, "path", None) or "<unmatched>"
            metrics_mod.record_http_request(
                label_path, request.method, status_code, time.perf_counter() - start
            )
            surface_var.reset(surface_token)


class RootSpanMiddleware(BaseHTTPMiddleware):
    """Open the per-request root span.

    Added last in ``create_app`` so it runs *outermost* of the tracing-relevant
    middlewares (Starlette executes middleware in reverse of add order), giving
    every downstream engine span a single root → one coherent trace per request.
    Because it runs before RequestIDMiddleware, ``request.state.request_id`` is
    not yet set when the span opens, so request_id / route / status are all
    stamped *after* ``call_next`` returns (request_id survives via
    ``request.state``, an object attribute, not a ContextVar). Surface is derived
    directly from the path the same way MetricsMiddleware does, rather than read
    from ``surface_var``: BaseHTTPMiddleware copies the context into the inner
    sub-task, so a ContextVar set by an inner middleware is not visible to this
    outer one after ``call_next``. When tracing is disabled, span() returns the
    no-op context manager and this path costs one function call.
    """

    async def dispatch(self, request: Request, call_next):
        with tracing_mod.span("http.request", **{"http.method": request.method}) as sp:
            response = await call_next(request)
            # Resolve the route template (bounded cardinality) after routing,
            # mirroring MetricsMiddleware's label_path.
            route = request.scope.get("route")
            sp.set_attributes(
                {
                    "http.route": getattr(route, "path", None) or "<unmatched>",
                    "surface": metrics_mod.surface_for_path(request.url.path),
                    "request_id": getattr(request.state, "request_id", ""),
                    "http.status_code": response.status_code,
                }
            )
            return response


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
    app.add_middleware(MetricsMiddleware)
    # Added last → runs outermost, so the per-request root span wraps every
    # other middleware and all downstream engine spans nest under it.
    app.add_middleware(RootSpanMiddleware)

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

    @app.exception_handler(SpectralCalibrationMissingError)
    async def spectral_calibration_missing_handler(
        request: Request, exc: SpectralCalibrationMissingError
    ):
        msg = str(exc)
        logger.warning(
            "SpectralCalibrationMissingError on %s: %s", request.url.path, msg
        )
        return _make_error_response(
            request.url.path,
            400,
            msg,
            "invalid_request_error",
            "invalid_request_error",
            "calibration_missing",
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
    app.include_router(rerank.router)
    app.include_router(blobs.router)
    app.include_router(openai.router)
    app.include_router(responses.router)
    app.include_router(audio.router)
    app.include_router(anthropic.router)
    app.include_router(metrics_router.router)

    # Autonomous agent (#445): the HTTP surface and run registry only exist
    # when explicitly enabled. The service resolves the model manager lazily
    # (set later by the lifespan / test fixture), so it can be built here.
    if settings.agent_enabled:
        from olmlx.engine.agent.service import AgentService
        from olmlx.engine.agent.store import AgentStore

        store = AgentStore(settings.agent_db_path)
        app.state.agent_service = AgentService(
            store=store,
            manager_getter=lambda: app.state.model_manager,
        )
        app.include_router(agent.router)

    return app
