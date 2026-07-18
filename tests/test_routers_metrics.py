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


def test_in_flight_and_duration_span_the_whole_streaming_body():
    """MetricsMiddleware must record duration / decrement in-flight only after
    the *entire* response body has streamed — not at TTFT, which is when
    BaseHTTPMiddleware's ``call_next`` returns for a streaming response.

    Old behavior: the ``finally`` block decremented HTTP_IN_FLIGHT and recorded
    the duration histogram as soon as headers were sent, so in-flight read 0
    and the duration measured ~TTFT during the entire generation (#635).
    """
    from olmlx.app import MetricsMiddleware

    app = FastAPI()
    app.add_middleware(MetricsMiddleware)
    inflight_during: list[float] = []

    @app.get("/v1/chat/completions")
    async def stream():
        async def body():
            for i in range(3):
                inflight_during.append(metrics.HTTP_IN_FLIGHT._value.get())
                yield f"chunk{i}\n"

        return StreamingResponse(body())

    client = TestClient(app)
    baseline = metrics.HTTP_IN_FLIGHT._value.get()
    before = metrics.HTTP_REQUESTS.labels(
        path="/v1/chat/completions", method="GET", status="200"
    )._value.get()

    resp = client.get("/v1/chat/completions")
    assert resp.status_code == 200
    # While the body streams, the request is still in flight (baseline + 1).
    assert inflight_during == [baseline + 1.0] * 3
    # After the full body is consumed, in-flight returns to baseline and the
    # request is counted exactly once.
    assert metrics.HTTP_IN_FLIGHT._value.get() == baseline
    after = metrics.HTTP_REQUESTS.labels(
        path="/v1/chat/completions", method="GET", status="200"
    )._value.get()
    assert after == before + 1


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
