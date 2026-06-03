"""OpenTelemetry tracing for olmlx (issue #372).

This is the ONLY module that imports ``opentelemetry``, and it does so lazily —
nothing is imported unless ``init_tracing`` runs (gated on ``OLMLX_TRACING``).
Every call site in the codebase uses ``span(name, **attrs)``; when tracing is
disabled that returns a shared no-op singleton, so the off path costs one
attribute-less function call and allocates nothing.

Mirrors the metrics layer (``olmlx.utils.metrics``): one module owns the
third-party dependency, state is module-level, init happens in ``lifespan``,
and shutdown resets state so repeated ``create_app()`` in tests is clean.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from olmlx.config import Settings

logger = logging.getLogger(__name__)

# Module state. ``_TRACER`` / ``_PROVIDER`` are opentelemetry objects once
# initialized; typed loosely to avoid importing opentelemetry at module load.
_ENABLED: bool = False
_TRACER: Any = None
_PROVIDER: Any = None
_LOG_FILTER: Any = None


class _NoopSpan:
    """A span and context-manager that does nothing.

    Returned by ``span()`` when tracing is disabled so call sites stay
    branch-free. Implements the subset of the OTel Span API the codebase uses.
    A single shared instance is reused for every call — no per-call allocation.
    """

    __slots__ = ()

    def set_attribute(self, key: str, value: Any) -> None:
        return None

    def set_attributes(self, attrs: dict[str, Any]) -> None:
        return None

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        return None

    def record_exception(self, exc: BaseException) -> None:
        return None

    def set_status(self, status: Any) -> None:
        return None

    def __enter__(self) -> "_NoopSpan":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None


_NOOP_SPAN = _NoopSpan()


def span(name: str, **attrs: Any) -> Any:
    """Open a span as a context manager. THE single tracing call site.

    Disabled → returns the shared ``_NOOP_SPAN`` singleton (no import, no
    allocation). Enabled → returns a live span context manager that applies
    ``attrs`` as span attributes and records exceptions / sets ERROR status on
    exit (see Task 6).
    """
    if not _ENABLED:
        return _NOOP_SPAN
    return _LiveSpan(name, attrs)


def enabled() -> bool:
    """Whether tracing is active. Lets hot paths skip building tracing-only
    wrappers (e.g. an extra streaming generator layer) when disabled, keeping
    the off path allocation-free."""
    return _ENABLED


def shutdown_tracing() -> None:
    """Tear down the tracer and reset module state.

    Idempotent and safe to call when tracing was never initialized — that is
    the path the no-op tests exercise.
    """
    global _ENABLED, _TRACER, _PROVIDER, _LOG_FILTER
    provider = _PROVIDER
    _ENABLED = False
    _TRACER = None
    _PROVIDER = None
    if _LOG_FILTER is not None:
        _uninstall_log_filter(_LOG_FILTER)
        _LOG_FILTER = None
    if provider is not None:
        try:
            provider.shutdown()  # flushes the BatchSpanProcessor
        except Exception:
            logger.debug("tracing: provider shutdown failed", exc_info=True)


def _uninstall_log_filter(f: Any) -> None:
    for handler in logging.getLogger().handlers:
        try:
            handler.removeFilter(f)
        except Exception:
            pass


class _LiveSpan:
    """Context manager wrapping ``tracer.start_as_current_span``.

    Applies attributes on entry, records exceptions and sets ERROR status on
    an exception exit. Only constructed when tracing is enabled.
    """

    def __init__(self, name: str, attrs: dict[str, Any]) -> None:
        self._name = name
        self._attrs = attrs
        self._cm: Any = None
        self._span: Any = None

    def __enter__(self) -> Any:
        self._cm = _TRACER.start_as_current_span(self._name)
        self._span = self._cm.__enter__()
        if self._attrs:
            self._span.set_attributes(self._attrs)
        return self._span

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        if exc is not None and self._span is not None:
            from opentelemetry.trace import Status, StatusCode

            self._span.record_exception(exc)
            self._span.set_status(Status(StatusCode.ERROR, str(exc)))
        return self._cm.__exit__(exc_type, exc, tb)


def init_tracing(settings: "Settings") -> None:
    """Initialize the global tracer. Called from lifespan startup only when
    ``settings.tracing`` is true. Lazily imports opentelemetry. Idempotent.

    Endpoint/protocol/headers/sampling come from the native OTEL_* env vars,
    which the OTLPSpanExporter and SDK read on construction.
    """
    global _ENABLED, _TRACER, _PROVIDER, _LOG_FILTER
    if _ENABLED:
        return
    import os

    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter,
    )
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    service_name = os.environ.get("OTEL_SERVICE_NAME", "olmlx")
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    # BatchSpanProcessor exports asynchronously off the inference thread; the
    # SDK swallows export failures so they never surface to a request.
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))

    _PROVIDER = provider
    _TRACER = provider.get_tracer("olmlx")
    _ENABLED = True
    _LOG_FILTER = _install_log_filter()  # Task 5
    logger.info("OpenTelemetry tracing enabled (service.name=%s)", service_name)


def install_test_provider(provider: Any) -> None:
    """Test hook: enable tracing against a caller-supplied TracerProvider
    (e.g. one wired to InMemorySpanExporter) without OTLP or env config."""
    global _ENABLED, _TRACER, _PROVIDER, _LOG_FILTER
    _PROVIDER = provider
    _TRACER = provider.get_tracer("olmlx")
    _ENABLED = True
    _LOG_FILTER = _install_log_filter()


def current_context() -> Any:
    """Capture the current OTel context for re-attachment in another thread.

    Returns ``None`` when tracing is disabled; ``attach_context(None)`` is a
    no-op, so call sites stay branch-free.
    """
    if not _ENABLED:
        return None
    from opentelemetry import context as otel_context

    return otel_context.get_current()


def attach_context(ctx: Any) -> Any:
    """Attach a context captured by ``current_context`` inside a worker thread.

    Returns a token to pass to ``detach_context``. No-op (returns ``None``)
    when disabled or when ``ctx`` is ``None``.
    """
    if not _ENABLED or ctx is None:
        return None
    from opentelemetry import context as otel_context

    return otel_context.attach(ctx)


def detach_context(token: Any) -> None:
    """Detach a context attached by ``attach_context``. No-op on ``None``."""
    if token is None:
        return None
    from opentelemetry import context as otel_context

    otel_context.detach(token)


class _TraceCorrelationFilter(logging.Filter):
    """Stamp hex trace_id/span_id onto each record when a span is recording.

    Installed on the root logger's handlers by ``init_tracing``; removed by
    ``shutdown_tracing``. Reads the active span; when none is recording the
    record is left untouched (request_id from RequestIDMiddleware still
    applies independently).
    """

    def filter(self, record: logging.LogRecord) -> bool:
        from opentelemetry.trace import get_current_span

        span = get_current_span()
        ctx = span.get_span_context()
        if ctx.is_valid:
            record.trace_id = format(ctx.trace_id, "032x")
            record.span_id = format(ctx.span_id, "016x")
        return True


def _install_log_filter() -> Any:
    """Add the correlation filter to every root-logger handler. Returns the
    filter so ``shutdown_tracing`` can remove it."""
    f = _TraceCorrelationFilter()
    for handler in logging.getLogger().handlers:
        handler.addFilter(f)
    return f
