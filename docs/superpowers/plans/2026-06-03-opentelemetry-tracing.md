# OpenTelemetry Tracing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Emit per-phase OpenTelemetry spans (prefill, decode, speculative step/verify, Flash prefetch/weight-load, disk-cache I/O, MCP tool calls) for each request, exportable to a standard OTLP collector, with a single `OLMLX_TRACING` toggle and **zero cost when disabled** (no `opentelemetry` import on the off path).

**Architecture:** One module (`olmlx/utils/tracing.py`) owns *all* contact with `opentelemetry`. Everything else calls a single `span(name, **attrs)` context manager that returns a shared no-op object when tracing is off (no import, no allocation). A manually-opened root span lives in the middleware stack; child spans nest via OTel's `contextvars` context, which is explicitly captured and re-attached at every worker-thread hop (decode worker, Flash pool, disk `to_thread`) since OTel context does not auto-propagate across threads.

**Tech Stack:** OpenTelemetry SDK + OTLP HTTP exporter (optional `[otel]` extra, lazy-imported), FastAPI/Starlette middleware, pydantic-settings, pytest with OTel's `InMemorySpanExporter`.

---

## Background for the implementer

Read these before starting — they define the patterns this plan mirrors:

- **Spec:** `docs/superpowers/specs/2026-06-03-opentelemetry-tracing-design.md` — the authoritative design. This plan implements it section by section.
- **Metrics layer (the pattern to mirror):** `olmlx/utils/metrics.py` + `olmlx/routers/metrics.py`. Tracing follows the same shape: one module owns the third-party dep, a private registry/provider, lazy init in `lifespan`, unregister/shutdown on teardown so repeated `create_app()` in tests is clean.
- **Middleware stack:** `olmlx/app.py:185-247` (`MetricsMiddleware`, `RequestIDMiddleware`, and the `create_app` add order). Starlette's `add_middleware` order is **reverse** of execution: the **last** `add_middleware` runs **outermost** (verified empirically). The root span must be outermost of the tracing-relevant trio, so `RootSpanMiddleware` is added *last*. Because it then runs *before* `RequestIDMiddleware`, `request.state.request_id` isn't set when the span opens — so the span sets `request_id` *after* `call_next` returns (Task 7, Option A). See Task 7 for the exact add order.
- **Context vars:** `olmlx/context.py` — `request_id_var`, `surface_var`. The root span reads these for attributes.
- **Worker-thread hop:** `olmlx/utils/streaming.py:413` (`CancellableStream._run`, runs the decode loop in a background thread) and `:643` (`async_mlx_stream`). The decode span context must be captured on the calling coroutine and re-attached inside `_run`.
- **Inference seams:** `olmlx/engine/inference.py` — `generate_chat` (line 4178), `generate_completion` (2271), `generate_embeddings` (4490), `generate_transcription` (4570), `_stream_completion` (3330), `_full_completion_inner` (3905). The metrics layer already hooks the same two seams via `_metrics.observe_inference` at 3666 / 4122 — put the inference span around the same scope.
- **Disk cache:** `olmlx/engine/prompt_cache/store.py` — `_save_to_disk` (147), `_load_from_disk` (184), called under `asyncio.to_thread` at 493/517/522/538.
- **MCP tool exec:** `olmlx/chat/session.py:498` (`_exec_tool`), dispatches to `self.mcp.call_tool` (515) and `self.builtin.call_tool`.

**TDD is mandatory** (CLAUDE.md): write the failing test first for every behavior, watch it fail, implement, watch it pass, commit. **Run `ruff check` + `ruff format` before every commit** (user preference). Tests that need OTel use `pytest.importorskip("opentelemetry")`.

---

## File Structure

**Created:**
- `olmlx/utils/tracing.py` — the entire OTel integration surface: `init_tracing`, `shutdown_tracing`, `span`, `current_context`, `attach_context`, `_NoopSpan`, the logging filter. Only file that imports `opentelemetry`.
- `tests/test_tracing.py` — unit tests for the core module (no-op, init/shutdown, cross-thread, logging filter, off-path no-import).
- `tests/test_tracing_integration.py` — end-to-end span-tree assertions through the inference + middleware path using `InMemorySpanExporter`.

**Modified:**
- `pyproject.toml` — add `[project.optional-dependencies] otel`, add the two OTel packages to the `dev` dependency group.
- `olmlx/config.py` — add `tracing: bool = False`.
- `olmlx/app.py` — `init_tracing`/`shutdown_tracing` in `lifespan`; root-span middleware.
- `olmlx/engine/inference.py` — inference + prefill + decode spans at the existing seams.
- `olmlx/engine/speculative.py` — `spec.prefill` / `spec.step` / `spec.verify` spans.
- `olmlx/engine/flash/` — `flash.prefetch` / `flash.weight_load` spans.
- `olmlx/engine/prompt_cache/store.py` — `cache.disk_read` / `cache.disk_write` spans.
- `olmlx/chat/session.py` — `mcp.tool_call` span in `_exec_tool`.
- `CLAUDE.md` — design-decisions entry for tracing.
- `docs/` — user-facing tracing doc (mirror of the `/metrics` doc added in #419).

---

## Task 1: Dependencies and config toggle

**Files:**
- Modify: `pyproject.toml:28-29` (optional-dependencies), `pyproject.toml:44-53` (dev group)
- Modify: `olmlx/config.py:106` (add field near the other bool toggles)
- Test: `tests/test_config.py` (append)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config.py`:

```python
def test_tracing_defaults_off(monkeypatch):
    monkeypatch.delenv("OLMLX_TRACING", raising=False)
    from olmlx.config import Settings

    assert Settings().tracing is False


def test_tracing_env_toggle(monkeypatch):
    monkeypatch.setenv("OLMLX_TRACING", "true")
    from olmlx.config import Settings

    assert Settings().tracing is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py::test_tracing_defaults_off tests/test_config.py::test_tracing_env_toggle -v`
Expected: FAIL — `AttributeError: 'Settings' object has no attribute 'tracing'`

- [ ] **Step 3: Add the config field**

In `olmlx/config.py`, add directly after the `log_level` line (currently line 105):

```python
    # OpenTelemetry tracing master switch (OLMLX_TRACING). Default off.
    # When off, nothing under olmlx.utils.tracing imports opentelemetry, so
    # there is no import-time or per-request cost. All endpoint/protocol/
    # headers/sampling/service-name configuration comes from the native
    # OTEL_* env vars the OTLP exporter and SDK already honor
    # (OTEL_EXPORTER_OTLP_ENDPOINT, OTEL_EXPORTER_OTLP_PROTOCOL,
    # OTEL_TRACES_SAMPLER, OTEL_SERVICE_NAME, OTEL_TRACES_EXPORTER=console, …).
    tracing: bool = False
```

- [ ] **Step 4: Add the optional + dev dependencies**

In `pyproject.toml`, change the optional-dependencies block (line 28-29) to:

```toml
[project.optional-dependencies]
search = ["duckduckgo-search>=6.0"]
otel = [
    "opentelemetry-sdk>=1.27",
    "opentelemetry-exporter-otlp-proto-http>=1.27",
]
```

(`opentelemetry-api` comes transitively via `-sdk`.)

In the `dev` dependency group (line 45), add the same two packages so CI installs them and the tracing tests run instead of skipping:

```toml
[dependency-groups]
dev = [
    "datasets>=3.0",
    "httpx>=0.28.1",
    "matplotlib>=3.9",
    "opentelemetry-sdk>=1.27",
    "opentelemetry-exporter-otlp-proto-http>=1.27",
    "pytest>=9.0.2",
    "pytest-asyncio>=0.24",
    "pytest-cov>=7.0.0",
    "pyright>=1.1.380",
    "ruff>=0.9",
]
```

- [ ] **Step 5: Sync and verify the tests pass**

Run: `uv sync --no-editable && uv run pytest tests/test_config.py::test_tracing_defaults_off tests/test_config.py::test_tracing_env_toggle -v`
Expected: PASS. `uv sync` installs the OTel packages into the dev environment.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check olmlx/config.py && uv run ruff format olmlx/config.py
git add pyproject.toml uv.lock olmlx/config.py tests/test_config.py
git commit -m "feat(tracing): add OLMLX_TRACING toggle and [otel] extra"
```

---

## Task 2: Core no-op span (the always-on, zero-cost path)

The no-op span is what every call site gets when tracing is disabled. It must implement the full span surface as no-ops and be a shared singleton (no allocation per call).

**Files:**
- Create: `olmlx/utils/tracing.py`
- Test: `tests/test_tracing.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_tracing.py`:

```python
import sys


def test_span_disabled_returns_noop_singleton():
    """With tracing off, span() returns the same no-op object every call and
    does not import opentelemetry."""
    import olmlx.utils.tracing as tracing

    tracing.shutdown_tracing()  # ensure clean disabled state

    with tracing.span("prefill", model="m") as sp:
        sp.set_attribute("eval_count", 5)
        sp.set_attributes({"a": 1})
        sp.add_event("x")
        sp.record_exception(ValueError("nope"))

    a = tracing.span("a")
    b = tracing.span("b")
    assert a is b  # shared singleton, no per-call allocation
    assert "opentelemetry" not in sys.modules


def test_noop_span_is_a_context_manager_returning_self():
    import olmlx.utils.tracing as tracing

    tracing.shutdown_tracing()
    cm = tracing.span("x", foo="bar")
    entered = cm.__enter__()
    assert entered is cm
    assert cm.__exit__(None, None, None) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tracing.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'olmlx.utils.tracing'`

- [ ] **Step 3: Create the module with the no-op surface**

Create `olmlx/utils/tracing.py`:

```python
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
```

`_LiveSpan`, `init_tracing`, `current_context`, `attach_context`, `_install_log_filter`/`_uninstall_log_filter` are added in later tasks. For now, add minimal stubs so the module imports cleanly:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_tracing.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check olmlx/utils/tracing.py tests/test_tracing.py && uv run ruff format olmlx/utils/tracing.py tests/test_tracing.py
git add olmlx/utils/tracing.py tests/test_tracing.py
git commit -m "feat(tracing): no-op span singleton (zero-cost off path)"
```

---

## Task 3: `init_tracing` and the live tracer

Wire up the real OTel provider/exporter, lazily imported. After this task `span()` produces real spans when enabled.

**Files:**
- Modify: `olmlx/utils/tracing.py`
- Test: `tests/test_tracing.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_tracing.py`:

```python
import pytest


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tracing.py::test_enabled_span_records_attributes -v`
Expected: FAIL — `AttributeError: module 'olmlx.utils.tracing' has no attribute 'install_test_provider'`

- [ ] **Step 3: Implement `_LiveSpan`, `init_tracing`, `install_test_provider`**

In `olmlx/utils/tracing.py`, replace the `_LiveSpan` stub from Task 2 with:

```python
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
```

Add `init_tracing` and the test helper:

```python
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
```

Add a temporary stub for `_install_log_filter` (replaced in Task 5):

```python
def _install_log_filter() -> Any:  # replaced in Task 5
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_tracing.py -v`
Expected: PASS (all five tests).

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check olmlx/utils/tracing.py tests/test_tracing.py && uv run ruff format olmlx/utils/tracing.py tests/test_tracing.py
git add olmlx/utils/tracing.py tests/test_tracing.py
git commit -m "feat(tracing): live tracer via init_tracing + OTLP exporter"
```

---

## Task 4: Cross-thread context propagation

OTel context is `contextvars`-based and does **not** flow into `ThreadPoolExecutor`/bare threads. The decode loop, Flash prefetch, and disk `to_thread` all run off the request coroutine. Provide explicit capture/attach helpers (no-ops when disabled).

**Files:**
- Modify: `olmlx/utils/tracing.py`
- Test: `tests/test_tracing.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_tracing.py`:

```python
def test_cross_thread_child_nests_under_parent(otel_memory_exporter):
    """A child span opened in a worker thread that re-attaches the captured
    context is a child of the event-loop parent, not a new root."""
    import concurrent.futures

    tracing, exporter = otel_memory_exporter

    with tracing.span("parent"):
        ctx = tracing.current_context()

        def worker():
            token = tracing.attach_context(ctx)
            try:
                with tracing.span("child"):
                    pass
            finally:
                tracing.detach_context(token)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(worker).result()

    by_name = {s.name: s for s in exporter.get_finished_spans()}
    assert by_name["child"].parent is not None
    assert by_name["child"].parent.span_id == by_name["parent"].context.span_id


def test_context_helpers_are_noops_when_disabled():
    import olmlx.utils.tracing as tracing

    tracing.shutdown_tracing()
    ctx = tracing.current_context()
    token = tracing.attach_context(ctx)
    tracing.detach_context(token)  # must not raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tracing.py::test_cross_thread_child_nests_under_parent -v`
Expected: FAIL — `AttributeError: module 'olmlx.utils.tracing' has no attribute 'current_context'`

- [ ] **Step 3: Implement the helpers**

Add to `olmlx/utils/tracing.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_tracing.py -v`
Expected: PASS (all tests).

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check olmlx/utils/tracing.py tests/test_tracing.py && uv run ruff format olmlx/utils/tracing.py tests/test_tracing.py
git add olmlx/utils/tracing.py tests/test_tracing.py
git commit -m "feat(tracing): cross-thread context capture/attach helpers"
```

---

## Task 5: Log correlation filter

When a span is active, stamp `trace_id`/`span_id` (hex) onto each log record so a log line can be pivoted to its trace. No `opentelemetry-instrumentation-logging` dependency — a plain `logging.Filter`.

**Files:**
- Modify: `olmlx/utils/tracing.py`
- Test: `tests/test_tracing.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_tracing.py`:

```python
import logging


def test_log_record_gets_trace_ids_when_span_active(otel_memory_exporter):
    tracing, _ = otel_memory_exporter
    captured = {}

    class _Capture(logging.Handler):
        def emit(self, record):
            captured["trace_id"] = getattr(record, "trace_id", None)
            captured["span_id"] = getattr(record, "span_id", None)

    handler = _Capture()
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        with tracing.span("work"):
            logging.getLogger("olmlx").warning("inside span")
    finally:
        root.removeHandler(handler)

    assert captured["trace_id"] and captured["trace_id"] != "0" * 32
    assert captured["span_id"] and captured["span_id"] != "0" * 16


def test_log_filter_not_installed_when_disabled():
    import olmlx.utils.tracing as tracing

    tracing.shutdown_tracing()
    record = logging.LogRecord(
        "olmlx", logging.INFO, __file__, 1, "msg", None, None
    )
    # No filter installed → attribute absent.
    assert not hasattr(record, "trace_id")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tracing.py::test_log_record_gets_trace_ids_when_span_active -v`
Expected: FAIL — `record.trace_id` is `None` (stub filter does nothing).

- [ ] **Step 3: Implement the filter and install/uninstall**

In `olmlx/utils/tracing.py`, replace the `_install_log_filter` stub and the `_uninstall_log_filter` stub with:

```python
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


def _uninstall_log_filter(f: Any) -> None:
    for handler in logging.getLogger().handlers:
        try:
            handler.removeFilter(f)
        except Exception:
            pass
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_tracing.py -v`
Expected: PASS (all tests).

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check olmlx/utils/tracing.py tests/test_tracing.py && uv run ruff format olmlx/utils/tracing.py tests/test_tracing.py
git add olmlx/utils/tracing.py tests/test_tracing.py
git commit -m "feat(tracing): log-record trace/span id correlation filter"
```

---

## Task 6: Off-path no-import guard (Acceptance Criterion #2)

A dedicated regression test that proves `opentelemetry` is never imported when tracing is off — the AC the whole no-op design exists to satisfy.

**Files:**
- Test: `tests/test_tracing.py` (append)

- [ ] **Step 1: Write the failing-then-passing guard test**

Append to `tests/test_tracing.py`:

```python
def test_off_path_never_imports_opentelemetry():
    """AC #2: with tracing disabled, exercising the span call site must not
    pull opentelemetry into sys.modules."""
    import subprocess
    import sys

    code = (
        "import sys; import olmlx.utils.tracing as t; "
        "t.shutdown_tracing(); "
        "[t.span('s', a=1).__enter__() or t.span('s').__exit__(None,None,None) "
        "for _ in range(3)]; "
        "assert 'opentelemetry' not in sys.modules, sorted(sys.modules)"
    )
    # Subprocess so a prior test that imported opentelemetry can't pollute the
    # assertion via the shared interpreter's module cache.
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
```

- [ ] **Step 2: Run test to verify it passes (guards existing behavior)**

Run: `uv run pytest tests/test_tracing.py::test_off_path_never_imports_opentelemetry -v`
Expected: PASS. If it FAILS, something in `olmlx.utils.tracing` import chain pulls opentelemetry at module scope — move that import inside a function.

- [ ] **Step 3: Lint and commit**

```bash
uv run ruff check tests/test_tracing.py && uv run ruff format tests/test_tracing.py
git add tests/test_tracing.py
git commit -m "test(tracing): guard zero-import off path (AC #2)"
```

---

## Task 7: App wiring — lifespan init/shutdown + root-span middleware

Open one root span per request inside the middleware stack so every downstream span has a single parent. Initialize/teardown the tracer in `lifespan`.

**Files:**
- Modify: `olmlx/app.py` (imports, `lifespan`, new middleware class, `create_app`)
- Test: `tests/test_tracing_integration.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_tracing_integration.py`:

```python
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

    from olmlx.app import RootSpanMiddleware, RequestIDMiddleware, MetricsMiddleware

    app = FastAPI()

    @app.get("/api/version")
    def version():
        return {"version": "test"}

    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(MetricsMiddleware)
    app.add_middleware(RootSpanMiddleware)

    client = TestClient(app)
    resp = client.get("/api/version")
    assert resp.status_code == 200

    roots = [s for s in memory_exporter.get_finished_spans()
             if s.parent is None]
    assert len(roots) == 1
    attrs = dict(roots[0].attributes)
    assert attrs["http.method"] == "GET"
    assert attrs["http.route"] == "/api/version"
    assert attrs["surface"] == "ollama"
    assert attrs["http.status_code"] == 200
    assert "request_id" in attrs
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tracing_integration.py::test_root_span_emitted_for_request -v`
Expected: FAIL — `ImportError: cannot import name 'RootSpanMiddleware' from 'olmlx.app'`

- [ ] **Step 3: Add the middleware and wiring**

In `olmlx/app.py`, add the import near the other olmlx imports (after line 35):

```python
from olmlx.utils import tracing as tracing_mod
```

Add the middleware class after `MetricsMiddleware` (after line 213):

```python
class RootSpanMiddleware(BaseHTTPMiddleware):
    """Open the per-request root span.

    Placed just inside RequestIDMiddleware and around MetricsMiddleware so the
    request id and surface are available as attributes and every downstream
    engine span nests under one root → one coherent trace per request. When
    tracing is disabled, span() returns the no-op context manager and this
    path is unchanged in cost (one function call returning a singleton).
    """

    async def dispatch(self, request: Request, call_next):
        method = request.method
        request_id = getattr(request.state, "request_id", "")
        with tracing_mod.span(
            "http.request",
            **{"http.method": method, "request_id": request_id},
        ) as sp:
            response = await call_next(request)
            # Resolve the route template (bounded cardinality) after routing,
            # mirroring MetricsMiddleware's label_path.
            route = request.scope.get("route")
            sp.set_attributes(
                {
                    "http.route": getattr(route, "path", None) or "<unmatched>",
                    "surface": surface_var.get(),
                    "http.status_code": response.status_code,
                }
            )
            return response
```

`surface_var` is already imported in `app.py` (line 13). Confirm `surface_var` is set by `MetricsMiddleware` *inside* this span — that requires `MetricsMiddleware` to run *inside* `RootSpanMiddleware`. Add the middleware in `create_app` so it runs **outermost of the three** (root span wraps everything). Starlette executes middleware in the **reverse** of `add_middleware` order, so the last-added runs first (outermost). Change the block at `olmlx/app.py:245-247` to:

```python
    app.add_middleware(ForceJSONMiddleware)
    app.add_middleware(MetricsMiddleware)
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(RootSpanMiddleware)
```

Execution order (outermost→innermost) becomes `RootSpan, RequestID, Metrics, ForceJSON`. RequestID runs inside RootSpan so `request.state.request_id` is set before — wait: RootSpan needs `request.state.request_id`, which RequestIDMiddleware sets. Since RootSpan is outermost it runs *before* RequestID, so `request.state.request_id` is not yet set when the span opens. Two options — pick option A:

- **Option A (use):** open the span with `request_id=""` and set the `request_id` attribute *after* `call_next` returns (RequestID has run by then and set `request.state.request_id`). Update `dispatch` to set `request_id` in the post-`call_next` `set_attributes` block instead of at span open:

```python
    async def dispatch(self, request: Request, call_next):
        with tracing_mod.span(
            "http.request", **{"http.method": request.method}
        ) as sp:
            response = await call_next(request)
            route = request.scope.get("route")
            sp.set_attributes(
                {
                    "http.route": getattr(route, "path", None) or "<unmatched>",
                    "surface": surface_var.get(),
                    "request_id": getattr(request.state, "request_id", ""),
                    "http.status_code": response.status_code,
                }
            )
            return response
```

In `lifespan` startup, after settings are loaded (after line 56's `from olmlx.config import settings`, before/around model manager setup — put it right after the `settings` import block at line 56), add:

```python
    if settings.tracing:
        tracing_mod.init_tracing(settings)
```

In `lifespan` shutdown, after `await manager.stop()` (after line 128), add:

```python
    tracing_mod.shutdown_tracing()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_tracing_integration.py::test_root_span_emitted_for_request -v`
Expected: PASS.

- [ ] **Step 5: Verify existing app tests still pass**

Run: `uv run pytest tests/test_app.py tests/test_routers_status.py -q`
Expected: PASS (no regression from the middleware reorder). If `tests/test_app.py` doesn't exist, run `uv run pytest tests/ -k "app or middleware or status" -q`.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check olmlx/app.py tests/test_tracing_integration.py && uv run ruff format olmlx/app.py tests/test_tracing_integration.py
git add olmlx/app.py tests/test_tracing_integration.py
git commit -m "feat(tracing): root-span middleware + lifespan init/shutdown"
```

---

## Task 8: Inference spans (top + prefill + decode)

Wrap the public inference entry points and the prefill/decode phases. The decode loop runs in the `CancellableStream` worker thread, so the decode span's context is captured on the coroutine and re-attached in the thread.

**Files:**
- Modify: `olmlx/engine/inference.py` (entry points + the two seams at 3330 / 3905)
- Modify: `olmlx/utils/streaming.py` (re-attach context in `CancellableStream._run`)
- Test: `tests/test_tracing_integration.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_tracing_integration.py`. This uses the existing inference test fixture/fakes — locate the fake model fixture used by `tests/test_inference*.py` (search: `grep -rln "generate_chat" tests/`) and reuse it. Pattern:

```python
@pytest.mark.asyncio
async def test_chat_trace_has_inference_prefill_decode(memory_exporter, fake_loaded_model):
    """AC #1: a chat request renders inference → {prefill, decode} sub-spans."""
    from olmlx.engine import inference

    # Drive a minimal non-streaming chat generation through the fake model.
    # (Reuse the helper the existing inference tests use to build messages +
    # call generate_chat; assert on span tree, not output text.)
    await _run_minimal_chat(inference, fake_loaded_model)

    names = {s.name for s in memory_exporter.get_finished_spans()}
    assert "inference" in names
    assert "prefill" in names
    assert "decode" in names

    by_name = {s.name: s for s in memory_exporter.get_finished_spans()}
    assert by_name["prefill"].parent.span_id == by_name["inference"].context.span_id
    assert by_name["decode"].parent.span_id == by_name["inference"].context.span_id
    assert dict(by_name["inference"].attributes)["strategy"] == "none"
```

> Implementer note: `_run_minimal_chat` and `fake_loaded_model` should be thin wrappers over whatever `tests/test_inference.py` (or `conftest.py`) already provides for driving `generate_chat` against a fake mlx model. Do not build a new model harness — reuse the existing one. If non-streaming chat is hard to drive in isolation, assert the same tree on `generate_completion` instead; the spans are emitted at the shared seams either way.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tracing_integration.py::test_chat_trace_has_inference_prefill_decode -v`
Expected: FAIL — span names `inference`/`prefill`/`decode` absent.

- [ ] **Step 3: Add the top inference span at each entry point**

In `olmlx/engine/inference.py`, add the import at the top of the module (near the metrics import — search for `from olmlx.utils import metrics as _metrics`):

```python
from olmlx.utils import tracing as _tracing
```

Wrap the body of each public entry point in an `inference` span. For `generate_chat` (the real impl at line 4178 — the earlier signatures are `@overload`s), open the span immediately after resolving the model/strategy and keep it open across the call. Concretely, find where `lm` and the speculative strategy are known and wrap the remaining body:

```python
    strategy = getattr(lm, "speculative_strategy", None) or "none"
    with _tracing.span(
        "inference",
        model=lm.name,
        surface=surface_var.get(),
        strategy=strategy,
        **{"gen.stream": bool(stream)},
    ):
        # ... existing generate_chat body ...
```

Apply the same wrapper (with the entry-point-appropriate `model`/`surface`/`strategy`, and `gen.stream` where a stream flag exists) to `generate_completion` (2271), `generate_embeddings` (4490), and `generate_transcription` (4570). For embeddings/transcription use `strategy="none"` and omit `gen.stream` (or set False).

> Implementer note: read each entry point's body first; the span must enclose the work but sit *inside* any early validation that raises a 4xx ValueError so a rejected request doesn't emit a misleading inference span. Place the `with` after argument validation, before the generation call.

- [ ] **Step 4: Add the prefill + decode spans at the shared seams**

The two seams are `_stream_completion` (3330) and `_full_completion_inner` (3905), where `observe_inference` is already called (3666 / 4122).

In `_full_completion_inner`: open a `prefill` span around stream creation up to the first token, and a `decode` span around the token-consumption loop. Read the function body (3905–4124) and place:

```python
    with _tracing.span(
        "prefill", prompt_tokens=len(prompt_tokens or []), cache_hit=bool(adopt_pin)
    ):
        # ... existing stream creation / first-token work ...
    with _tracing.span("decode") as _decode_span:
        # ... existing token loop ...
        _decode_span.set_attributes(
            {"eval_count": stats.eval_count,
             "decode_tok_s": _metrics._decode_tps(stats)}
        )
```

In `_stream_completion` (3330): the decode loop runs through the `CancellableStream` worker thread. Open the `prefill` span around stream startup, then capture context and open the `decode` span such that its context is available to the worker thread:

```python
    with _tracing.span("prefill", prompt_tokens=len(prompt_tokens or [])):
        stream = async_mlx_stream(...)  # existing call
    _decode_ctx = _tracing.current_context()
    with _tracing.span("decode") as _decode_span:
        # existing `async for token in stream:` loop
        _decode_span.set_attributes(
            {"eval_count": stats.eval_count,
             "decode_tok_s": _metrics._decode_tps(stats)}
        )
```

Pass `_decode_ctx` into `async_mlx_stream` so the worker thread can re-attach it (next step).

> Implementer note: `ttft_ns` and `cache_hit` attributes — set `ttft_ns` on the `prefill` span from `stats.prompt_eval_duration` once known (set it at span close via `set_attribute`), and `cache_hit` from the prompt-cache adopt/reuse flag visible at the seam. Read the seam to find the exact local holding cache-hit state; `adopt_pin` is the reuse signal in `_stream_completion`'s signature.

- [ ] **Step 5: Re-attach decode context in the worker thread**

In `olmlx/utils/streaming.py`, thread an optional captured context into `async_mlx_stream` → `CancellableStream` → `_run`. Add a parameter to `async_mlx_stream` (line 643):

```python
def async_mlx_stream(
    ...,
    trace_context: Any = None,
    **kwargs: Any,
) -> CancellableStream:
```

Pass it into `CancellableStream.__init__` and store it. In `_run` (line 413), at the very top of the method body, re-attach:

```python
    def _run(self):
        from olmlx.utils import tracing as _tracing

        _trace_token = _tracing.attach_context(self._trace_context)
        try:
            # ... existing _run body ...
        finally:
            _tracing.detach_context(_trace_token)
```

(Wrap the existing `try/except/finally` content; the new `finally` detaches.) When tracing is off, `_trace_context` is `None` and `attach_context`/`detach_context` are no-ops.

In `_stream_completion`, pass `trace_context=_decode_ctx` to the `async_mlx_stream(...)` call.

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_tracing_integration.py -v && uv run pytest tests/test_streaming.py -q`
Expected: PASS. Confirm no regression in streaming tests.

- [ ] **Step 7: Lint and commit**

```bash
uv run ruff check olmlx/engine/inference.py olmlx/utils/streaming.py tests/test_tracing_integration.py && uv run ruff format olmlx/engine/inference.py olmlx/utils/streaming.py tests/test_tracing_integration.py
git add olmlx/engine/inference.py olmlx/utils/streaming.py tests/test_tracing_integration.py
git commit -m "feat(tracing): inference, prefill, and decode spans"
```

---

## Task 9: Speculative spans (`spec.prefill` / `spec.step` / `spec.verify`)

One span per decode step (literal, per the locked decision; volume bounded by `OTEL_TRACES_SAMPLER`), plus a verify sub-span so the AC-named "verify" phase is visible.

**Files:**
- Modify: `olmlx/engine/speculative.py` (shared `prefill` / `step` / `verify_draft_greedy` surface)
- Test: `tests/test_tracing_integration.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_tracing_integration.py`:

```python
def test_speculative_step_and_verify_spans(memory_exporter, fake_speculative_decoder):
    """A speculative decode emits spec.prefill, per-step spec.step, and a
    spec.verify sub-span with proposed/accepted attributes."""
    # Reuse the existing speculative decoder test harness to run a few steps.
    _run_speculative_steps(fake_speculative_decoder, n_steps=3)

    spans = memory_exporter.get_finished_spans()
    names = [s.name for s in spans]
    assert "spec.prefill" in names
    assert names.count("spec.step") == 3
    assert "spec.verify" in names

    step = next(s for s in spans if s.name == "spec.step")
    attrs = dict(step.attributes)
    assert "proposed" in attrs and "accepted" in attrs
    assert attrs["strategy"] in {"classic", "pld", "dflash", "eagle", "self"}
```

> Implementer note: locate the existing speculative test harness (`grep -rln "SpeculativeDecoder\|prefill\|verify_draft_greedy" tests/`) and reuse its fakes for `_run_speculative_steps`/`fake_speculative_decoder`.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tracing_integration.py::test_speculative_step_and_verify_spans -v`
Expected: FAIL — `spec.*` spans absent.

- [ ] **Step 3: Add the spans**

In `olmlx/engine/speculative.py`, add `from olmlx.utils import tracing as _tracing` at the top. Read the shared decoder surface and wrap:

- The `prefill` method body:

```python
    with _tracing.span("spec.prefill", strategy=self._strategy_label, prompt_tokens=len(prompt)):
        # ... existing prefill body ...
```

- The `step` method body, recording proposed/accepted as attributes (the values `step()` already computes/returns):

```python
    with _tracing.span("spec.step", strategy=self._strategy_label) as _sp:
        # ... existing step body producing `num_drafted`/`num_accepted` ...
        _sp.set_attributes({"proposed": num_drafted, "accepted": num_accepted})
```

- The `verify_draft_greedy` call inside `step` (or the method itself):

```python
    with _tracing.span("spec.verify", strategy=self._strategy_label):
        # ... existing verify_draft_greedy work ...
```

For `self._strategy_label`, reuse the class→label mapping the metrics layer already defines (`olmlx/utils/metrics.py:172` `_STRATEGY_BY_CLASS`). Add a small helper or a property; simplest is:

```python
from olmlx.utils.metrics import _STRATEGY_BY_CLASS

    @property
    def _strategy_label(self) -> str:
        return _STRATEGY_BY_CLASS.get(type(self).__name__, "unknown")
```

Apply the same property/import to the sibling decoders (`olmlx/engine/dflash/decoder.py`, `olmlx/engine/eagle/decoder.py`) if their `prefill`/`step` are separate classes rather than inherited — if they share a base class, define it once on the base.

> Implementer note: read `speculative.py` first to find the actual local variable names for drafted/accepted counts (the metrics path at `record_speculative` and `speculative_stream._log_stats` shows them as `proposed`/`accepted`). The per-step span is intentionally high-volume; do not add a code-side cap (the spec mandates the operator use `OTEL_TRACES_SAMPLER`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tracing_integration.py::test_speculative_step_and_verify_spans tests/test_speculative.py -q`
Expected: PASS.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check olmlx/engine/speculative.py olmlx/engine/dflash/decoder.py olmlx/engine/eagle/decoder.py tests/test_tracing_integration.py && uv run ruff format olmlx/engine/speculative.py olmlx/engine/dflash/decoder.py olmlx/engine/eagle/decoder.py tests/test_tracing_integration.py
git add olmlx/engine/speculative.py olmlx/engine/dflash/decoder.py olmlx/engine/eagle/decoder.py tests/test_tracing_integration.py
git commit -m "feat(tracing): speculative prefill/step/verify spans"
```

---

## Task 10: Flash spans (`flash.prefetch` / `flash.weight_load`)

These run inside the Flash prefetch thread and the weight-load path. The prefetch thread re-attaches the decode context.

**Files:**
- Modify: `olmlx/engine/flash/` (prefetcher + weight store)
- Test: `tests/test_tracing.py` or `tests/test_tracing_integration.py` (append a unit test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_tracing_integration.py`:

```python
def test_flash_weight_load_span(memory_exporter, fake_flash_weight_store):
    """A flash weight load emits a flash.weight_load span with layer_idx."""
    _trigger_weight_load(fake_flash_weight_store, layer_idx=7)
    span = next(
        s for s in memory_exporter.get_finished_spans()
        if s.name == "flash.weight_load"
    )
    assert dict(span.attributes)["layer_idx"] == 7
```

> Implementer note: reuse the flash weight-store test fakes (`grep -rln "FlashWeightStore\|wait_for_layer\|load_experts" tests/`). If a true unit fake is heavy, assert the span on a direct call to the wrapped method with a stubbed loader.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tracing_integration.py::test_flash_weight_load_span -v`
Expected: FAIL — `flash.weight_load` absent.

- [ ] **Step 3: Add the spans**

Add `from olmlx.utils import tracing as _tracing` to the relevant flash modules. Wrap:

- The per-layer weight/neuron/expert load (FlashWeightStore load method):

```python
    with _tracing.span("flash.weight_load", layer_idx=layer_idx,
                       active_neurons=n_active):
        # ... existing per-layer load ...
```

(use `experts=` instead of `active_neurons=` on the Flash-MoE expert-load path).

- The prefetch predict+submit / `wait_for_layer`:

```python
    with _tracing.span("flash.prefetch", layer_idx=layer_idx):
        # ... existing predict + submit / wait_for_layer ...
```

Because prefetch I/O runs in a dedicated `ThreadPoolExecutor`, the prefetch span is opened inside the thread. If the prefetch worker needs to nest under the request's decode span, capture `current_context()` when the prefetch is submitted and `attach_context` it inside the worker (same pattern as Task 8 Step 5). If that plumbing is non-trivial, it is acceptable for the first cut to let `flash.prefetch` be a root-less span — note this explicitly in the commit message and the CLAUDE.md entry; the spec calls out thread-context propagation as the main correctness risk, so prefer wiring it through if the submit site has the context.

> Implementer note: the prefetcher predict path is synchronous (MLX `mx.eval` deadlocks under concurrent multi-thread use); only the I/O is async. Open the `flash.prefetch` span around predict+submit on the calling side where the context is naturally present; that avoids the cross-thread hop for the span itself while leaving I/O async.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tracing_integration.py::test_flash_weight_load_span tests/ -k flash -q`
Expected: PASS.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check olmlx/engine/flash/ tests/test_tracing_integration.py && uv run ruff format olmlx/engine/flash/ tests/test_tracing_integration.py
git add olmlx/engine/flash/ tests/test_tracing_integration.py
git commit -m "feat(tracing): flash prefetch and weight-load spans"
```

---

## Task 11: Disk-cache spans (`cache.disk_read` / `cache.disk_write`)

These methods run under `asyncio.to_thread`; capture context before offloading and re-attach inside the worker.

**Files:**
- Modify: `olmlx/engine/prompt_cache/store.py` (`_save_to_disk` 147, `_load_from_disk` 184, the `to_thread` call sites 493/517/522/538)
- Test: `tests/test_tracing.py` (append) or `tests/test_prompt_cache*.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_tracing_integration.py`:

```python
def test_disk_cache_spans(memory_exporter, tmp_path, fake_cache_store):
    """_save_to_disk and _load_from_disk emit cache.disk_write / cache.disk_read
    spans with cache_id and bytes attributes."""
    cid = "abc123"
    state = _make_fake_cache_state()
    fake_cache_store._save_to_disk(cid, state)
    loaded = fake_cache_store._load_from_disk(cid)
    assert loaded is not None

    by_name = {s.name: s for s in memory_exporter.get_finished_spans()}
    assert "cache.disk_write" in by_name
    assert "cache.disk_read" in by_name
    assert dict(by_name["cache.disk_write"].attributes)["cache_id"] == cid
    assert "bytes" in dict(by_name["cache.disk_write"].attributes)
    assert dict(by_name["cache.disk_read"].attributes)["hit"] is True
```

> Implementer note: reuse the prompt-cache store test fakes (`grep -rln "_save_to_disk\|PromptCacheStore\|CachedPromptState" tests/`).

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tracing_integration.py::test_disk_cache_spans -v`
Expected: FAIL — `cache.disk_*` spans absent.

- [ ] **Step 3: Add the spans**

In `olmlx/engine/prompt_cache/store.py`, add `from olmlx.utils import tracing as _tracing`. Wrap `_save_to_disk` (147):

```python
    def _save_to_disk(self, cache_id: str, state: CachedPromptState) -> None:
        with _tracing.span("cache.disk_write", cache_id=cache_id) as _sp:
            # ... existing body ...
            _sp.set_attribute("bytes", written_bytes)  # size after serialize
```

Wrap `_load_from_disk` (184):

```python
    def _load_from_disk(self, cache_id: str) -> CachedPromptState | None:
        with _tracing.span("cache.disk_read", cache_id=cache_id) as _sp:
            # ... existing body, result in `loaded` ...
            _sp.set_attributes({"hit": loaded is not None, "bytes": read_bytes})
            return loaded
```

For cross-thread parenting: these methods are invoked via `asyncio.to_thread(self._save_to_disk, ...)` (493/517/522/538). To nest the disk span under the request span, the async wrapper that calls `to_thread` should capture `current_context()` and re-attach inside a tiny wrapper. Add a private helper used at each `to_thread` site:

```python
    async def _to_thread_traced(self, fn, *fn_args):
        ctx = _tracing.current_context()

        def _runner():
            token = _tracing.attach_context(ctx)
            try:
                return fn(*fn_args)
            finally:
                _tracing.detach_context(token)

        return await asyncio.to_thread(_runner)
```

Replace the `await asyncio.to_thread(self._save_to_disk, ...)` / `self._load_from_disk` / `self._read_from_disk` calls that should be traced with `await self._to_thread_traced(self._save_to_disk, ...)`. Leave unrelated `to_thread` calls (`_unlink_and_refresh`, `_save_entries_to_disk` bulk snapshot) unwrapped unless you want spans for them too (out of scope here).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tracing_integration.py::test_disk_cache_spans tests/ -k prompt_cache -q`
Expected: PASS.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check olmlx/engine/prompt_cache/store.py tests/test_tracing_integration.py && uv run ruff format olmlx/engine/prompt_cache/store.py tests/test_tracing_integration.py
git add olmlx/engine/prompt_cache/store.py tests/test_tracing_integration.py
git commit -m "feat(tracing): prompt-cache disk read/write spans"
```

---

## Task 12: MCP tool-call span (`mcp.tool_call`)

Wrap each tool invocation in the terminal-chat agent loop.

**Files:**
- Modify: `olmlx/chat/session.py` (`_exec_tool` 498)
- Test: `tests/test_chat_session.py` (or the existing chat-session test module) (append)

- [ ] **Step 1: Write the failing test**

Append to the chat-session test module (find it: `grep -rln "_exec_tool\|ChatSession" tests/`):

```python
@pytest.mark.asyncio
async def test_mcp_tool_call_span(memory_exporter, fake_chat_session):
    """_exec_tool emits an mcp.tool_call span with tool.name."""
    await fake_chat_session._exec_tool(
        {"name": "echo", "input": {"x": 1}, "id": "t1"}
    )
    span = next(
        s for s in memory_exporter.get_finished_spans()
        if s.name == "mcp.tool_call"
    )
    assert dict(span.attributes)["tool.name"] == "echo"
```

> Implementer note: reuse the existing chat-session test fixtures for `fake_chat_session`. Copy the `memory_exporter` fixture into this test module or move it to `conftest.py` if more than one test module needs it (preferred — see Task 13).

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/<chat-session-test>.py::test_mcp_tool_call_span -v`
Expected: FAIL — `mcp.tool_call` absent.

- [ ] **Step 3: Add the span**

In `olmlx/chat/session.py`, add `from olmlx.utils import tracing as _tracing`. Wrap the dispatch in `_exec_tool` (498). Include `mcp.server` when the call goes through MCP:

```python
    async def _exec_tool(self, tu: dict) -> dict:
        tool_name = tu["name"]
        tool_input = tu["input"]
        tool_id = tu["id"]
        with _tracing.span("mcp.tool_call", **{"tool.name": tool_name}) as _sp:
            try:
                if tool_name == "use_skill" and self.skills:
                    result = self.skills.handle_use_skill(tool_input)
                elif self.builtin and tool_name in self.builtin.tool_names:
                    result = await self.builtin.call_tool(tool_name, tool_input)
                elif self.mcp is not None:
                    _sp.set_attribute("mcp.server", getattr(self.mcp, "name", "mcp"))
                    result = await self.mcp.call_tool(
                        tool_name, tool_input, timeout=self.config.tool_timeout
                    )
                else:
                    result = ToolError(...)  # unchanged
                # ... rest of existing body unchanged, still inside `with` ...
```

Indent the existing `_exec_tool` body under the new `with`. Keep all existing return paths intact.

> Implementer note: the spec says the tool-call span is "parented under a per-turn span so a chat turn's tool calls group together." A per-turn span is optional for this task — if the agent loop has an obvious per-turn boundary (the method that iterates assistant turns and calls `_exec_tool`), wrap that in a `with _tracing.span("chat.turn"):` too. If not obvious, ship `mcp.tool_call` alone and leave per-turn grouping as a follow-up noted in the commit body.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/<chat-session-test>.py -q`
Expected: PASS.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check olmlx/chat/session.py && uv run ruff format olmlx/chat/session.py
git add olmlx/chat/session.py tests/<chat-session-test>.py
git commit -m "feat(tracing): MCP tool-call span in chat agent loop"
```

---

## Task 13: Shared test fixture + full-suite verification

Consolidate the `memory_exporter` fixture and run the full tracing test set plus a representative slice of the existing suite.

**Files:**
- Modify: `conftest.py` (add shared `memory_exporter` fixture) or `tests/conftest.py` if tests have their own
- Test: all tracing tests

- [ ] **Step 1: Move the `memory_exporter` fixture to a shared conftest**

If more than one test module defines `memory_exporter`, move it to `conftest.py` (project root — there is already one) so it's shared. Use the exact fixture body from Task 7. Remove the per-module duplicates.

- [ ] **Step 2: Run the full tracing test set**

Run: `uv run pytest tests/test_tracing.py tests/test_tracing_integration.py -v`
Expected: PASS, no skips (CI/dev installs the `[otel]` packages so `importorskip` doesn't trigger).

- [ ] **Step 3: Run a representative slice of the existing suite for regressions**

Run: `uv run pytest tests/ -k "app or middleware or streaming or inference or speculative or prompt_cache or chat or config or metrics" -q`
Expected: PASS. (Per the `project_full_pytest_sigabrt_flake` memory, the *full* local suite intermittently SIGABRTs — trust CI + these targeted suites rather than a single full run.)

- [ ] **Step 4: Verify the off-path guard one more time end-to-end**

Run: `uv run pytest tests/test_tracing.py::test_off_path_never_imports_opentelemetry -v`
Expected: PASS — proves AC #2 after all instrumentation is in place.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check . && uv run ruff format --check .
git add conftest.py tests/
git commit -m "test(tracing): shared memory_exporter fixture + suite verification"
```

---

## Task 14: Documentation

Update CLAUDE.md (user preference: keep design decisions current) and add a user-facing doc mirroring the `/metrics` manual entry from #419.

**Files:**
- Modify: `CLAUDE.md`
- Create/Modify: the user manual doc (find the `/metrics` doc added in #419: `grep -rln "metrics" docs/`)

- [ ] **Step 1: Add the CLAUDE.md design-decisions entry**

In `CLAUDE.md`, under "Key Design Decisions", add a bullet near the Prometheus metrics entry:

```markdown
- **OpenTelemetry tracing** (`utils/tracing.py`, #372): `OLMLX_TRACING=true`
  (default off) emits per-phase spans (root `http.request` → `inference` →
  `prefill`/`decode`, `spec.{prefill,step,verify}`, `flash.{prefetch,weight_load}`,
  `cache.disk_{read,write}`, `mcp.tool_call`) to an OTLP collector. **Zero cost
  when off**: `utils/tracing.py` is the only module importing `opentelemetry`
  and does so lazily; `span(name, **attrs)` returns a shared no-op singleton
  when disabled (no import, no allocation) — guarded by a subprocess test
  asserting `opentelemetry` stays out of `sys.modules`. All endpoint/protocol/
  sampling/service-name config comes from native `OTEL_*` env vars (no extra
  olmlx settings). Optional `[otel]` extra. OTel context is `contextvars`-based
  and does not cross threads, so the decode worker, Flash prefetch pool, and
  disk `to_thread` capture `current_context()` and re-attach it
  (`attach_context`) — a missed attach shows up as an orphaned single-span
  trace, caught by the cross-thread test. Per-step speculative spans are
  intentionally high-volume; the operator bounds them via `OTEL_TRACES_SAMPLER`.
  Out of scope: distributed worker trace propagation; FastAPI auto-instrumentation
  (root span opened manually in `RootSpanMiddleware`).
```

- [ ] **Step 2: Add the user-facing doc**

Mirror the `/metrics` user-manual section. Add a "Tracing" section documenting: the `OLMLX_TRACING` toggle, the `uv sync --extra otel` install, the relevant `OTEL_*` env vars (`OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_EXPORTER_OTLP_PROTOCOL`, `OTEL_TRACES_SAMPLER`, `OTEL_SERVICE_NAME`, `OTEL_TRACES_EXPORTER=console` for collector-free local debugging), the span inventory, and a one-paragraph Jaeger/Tempo quick-start (`docker run jaegertracing/all-in-one`, point `OTEL_EXPORTER_OTLP_ENDPOINT` at it, open the UI).

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md docs/
git commit -m "docs: document OpenTelemetry tracing (OLMLX_TRACING)"
```

---

## Self-Review (completed during planning)

**Spec coverage:**

| Spec section | Task |
|---|---|
| §1 Core wrapper `utils/tracing.py` (`init_tracing`, `shutdown_tracing`, `span`, `current_context`/`attach_context`, no-op) | Tasks 2–5 |
| §2 Root span in middleware | Task 7 |
| §3 Span inventory — inference (top/prefill/decode) | Task 8 |
| §3 — speculative (`spec.prefill/step/verify`) | Task 9 |
| §3 — flash (`flash.prefetch/weight_load`) | Task 10 |
| §3 — prompt-cache disk (`cache.disk_read/write`) | Task 11 |
| §3 — chat (`mcp.tool_call`) | Task 12 |
| §4 Log correlation filter | Task 5 |
| §5 Config & dependencies | Task 1 |
| §6 App wiring (lifespan + middleware) | Task 7 |
| Testing — off-path no-import (AC #2) | Task 6 |
| Testing — coherent chat trace (AC #1) | Task 8 (+ Task 9 speculative variant) |
| Testing — cross-thread propagation | Task 4 |
| Testing — disk-cache & flash spans | Tasks 10, 11 |
| Testing — logging correlation | Task 5 |

All spec sections map to a task. AC #1 (coherent chat trace with prefill/decode/verify) → Tasks 8+9; AC #2 (no latency/import regression when off) → Tasks 2 + 6.

**Type consistency:** `span()` returns the same context-manager protocol (`__enter__`/`__exit__` + `set_attribute`/`set_attributes`/`record_exception`/`add_event`/`set_status`) whether `_NoopSpan` or `_LiveSpan`. `current_context`/`attach_context`/`detach_context` form a consistent capture→attach→detach triple used identically in Tasks 8 (streaming), 10 (flash), and 11 (disk). `_strategy_label` (Task 9) reuses `_STRATEGY_BY_CLASS` from the metrics module rather than inventing a parallel map.

**Known soft spots flagged inline for the implementer (not placeholders — explicit instructions to read-then-wrap against real code):** the exact local-variable names inside `_stream_completion`/`_full_completion_inner` (Task 8), the speculative `step` drafted/accepted locals (Task 9), and the flash/cache/chat test fakes are to be reused from existing test harnesses rather than rebuilt. Each such spot names the precise method, line number, and the literal span code to insert.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-03-opentelemetry-tracing.md`.
