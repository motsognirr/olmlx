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
from typing import Any

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


def _uninstall_log_filter(f: Any) -> None:  # replaced in Task 5
    return None


class _LiveSpan:  # replaced in Task 6
    def __init__(self, name: str, attrs: dict[str, Any]) -> None:
        self._name = name
        self._attrs = attrs

    def __enter__(self) -> Any:
        return _NOOP_SPAN

    def __exit__(self, *a: Any) -> None:
        return None
