from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from olmlx.context import surface_var
from olmlx.routers import metrics as metrics_router
from olmlx.utils import metrics


def _app():
    app = FastAPI()
    app.include_router(metrics_router.router)
    return app


def test_metrics_endpoint_returns_prometheus_text():
    client = TestClient(_app())
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    assert "olmlx_inference_requests_total" in resp.text


def test_metrics_middleware_counts_requests_and_sets_surface():
    from olmlx.app import MetricsMiddleware

    app = FastAPI()
    app.add_middleware(MetricsMiddleware)

    captured = {}

    @app.get("/api/tags")
    async def tags():
        captured["surface"] = surface_var.get()
        return {"ok": True}

    app.include_router(metrics_router.router)
    client = TestClient(app)

    before = metrics.HTTP_REQUESTS.labels(
        path="/api/tags", method="GET", status="200"
    )._value.get()
    resp = client.get("/api/tags")
    assert resp.status_code == 200
    assert captured["surface"] == "ollama"
    after = metrics.HTTP_REQUESTS.labels(
        path="/api/tags", method="GET", status="200"
    )._value.get()
    assert after == before + 1
    # In-flight returns to baseline after the request completes.
    assert metrics.HTTP_IN_FLIGHT._value.get() == 0.0


def test_surface_var_survives_into_streaming_body():
    """Regression: the engine reads surface_var.get() while the streaming body
    is being consumed, which happens AFTER MetricsMiddleware.dispatch's finally
    block resets the ContextVar. BaseHTTPMiddleware runs the inner app in a
    sub-task that copies the context, so the reset in the middleware's own
    context does not clobber the value the streaming body sees (same mechanism
    RequestIDMiddleware relies on). Without this property, every streaming
    generation would be mislabelled "unknown".
    """
    from olmlx.app import MetricsMiddleware

    app = FastAPI()
    app.add_middleware(MetricsMiddleware)
    seen: list[str] = []

    @app.get("/v1/chat/completions")
    async def stream():
        async def body():
            for i in range(3):
                seen.append(surface_var.get())
                yield f"chunk{i}\n"

        return StreamingResponse(body())

    client = TestClient(app)
    resp = client.get("/v1/chat/completions")
    assert resp.status_code == 200
    assert seen == ["openai", "openai", "openai"]
