import pytest

pytest.importorskip("opentelemetry")


@pytest.fixture
def memory_exporter():
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    import olmlx.utils.tracing as tracing

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracing.install_test_provider(provider)
    yield exporter
    tracing.shutdown_tracing()


def test_root_span_emitted_for_request(memory_exporter):
    """A request through the middleware stack emits a root span with
    http.route / surface / request_id / status attributes."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from olmlx.app import MetricsMiddleware, RequestIDMiddleware, RootSpanMiddleware

    app = FastAPI()

    @app.get("/api/version")
    def version():
        return {"version": "test"}

    app.add_middleware(MetricsMiddleware)
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(RootSpanMiddleware)

    client = TestClient(app)
    resp = client.get("/api/version")
    assert resp.status_code == 200

    roots = [s for s in memory_exporter.get_finished_spans() if s.parent is None]
    assert len(roots) == 1
    attrs = dict(roots[0].attributes)
    assert attrs["http.method"] == "GET"
    assert attrs["http.route"] == "/api/version"
    assert attrs["surface"] == "ollama"
    assert attrs["http.status_code"] == 200
    assert "request_id" in attrs
