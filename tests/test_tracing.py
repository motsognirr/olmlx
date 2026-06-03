import sys

import pytest


def test_span_disabled_returns_noop_singleton():
    """With tracing off, span() returns the same no-op object every call and
    does not import opentelemetry on the off path.

    Note: we assert the off path imports *no new* opentelemetry modules rather
    than that ``sys.modules`` is globally pristine. Installing the ``[otel]``
    dev dependency makes ``opentelemetry-api`` importable, which silently
    activates an unrelated third-party library's optional OTel integration at
    pytest-collection time — so the shared interpreter may already hold otel-api
    modules through no fault of olmlx. The authoritative zero-import guarantee
    (AC #2) is asserted in a clean subprocess by
    ``test_off_path_never_imports_opentelemetry``.
    """
    import olmlx.utils.tracing as tracing

    tracing.shutdown_tracing()  # ensure clean disabled state

    otel_before = {m for m in sys.modules if m.startswith("opentelemetry")}

    with tracing.span("prefill", model="m") as sp:
        sp.set_attribute("eval_count", 5)
        sp.set_attributes({"a": 1})
        sp.add_event("x")
        sp.record_exception(ValueError("nope"))

    a = tracing.span("a")
    b = tracing.span("b")
    assert a is b  # shared singleton, no per-call allocation

    otel_after = {m for m in sys.modules if m.startswith("opentelemetry")}
    assert otel_after == otel_before  # off path imported nothing new


def test_noop_span_is_a_context_manager_returning_self():
    import olmlx.utils.tracing as tracing

    tracing.shutdown_tracing()
    cm = tracing.span("x", foo="bar")
    entered = cm.__enter__()
    assert entered is cm
    assert cm.__exit__(None, None, None) is None


@pytest.fixture
def otel_memory_exporter():
    """Initialize tracing with an in-memory exporter; tear down after."""
    pytest.importorskip("opentelemetry")
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
    yield tracing, exporter
    tracing.shutdown_tracing()


def test_enabled_span_records_attributes(otel_memory_exporter):
    tracing, exporter = otel_memory_exporter
    with tracing.span("prefill", model="m", prompt_tokens=10) as sp:
        sp.set_attribute("ttft_ns", 123)
    spans = exporter.get_finished_spans()
    assert [s.name for s in spans] == ["prefill"]
    attrs = dict(spans[0].attributes)
    assert attrs["model"] == "m"
    assert attrs["prompt_tokens"] == 10
    assert attrs["ttft_ns"] == 123


def test_enabled_span_records_exception_and_error_status(otel_memory_exporter):
    tracing, exporter = otel_memory_exporter
    from opentelemetry.trace import StatusCode

    with pytest.raises(ValueError):
        with tracing.span("boom"):
            raise ValueError("kaboom")
    span = exporter.get_finished_spans()[0]
    assert span.status.status_code == StatusCode.ERROR
    assert any(e.name == "exception" for e in span.events)
