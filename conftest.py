"""Root conftest — mock mlx C-extension modules on non-Apple platforms."""

import sys
from unittest.mock import MagicMock

# Only mock if mlx is not actually importable (i.e. not on Apple Silicon)
try:
    import mlx.core  # noqa: F401
except (ImportError, OSError):
    _mock_mx = MagicMock()
    _mock_mx.synchronize = MagicMock()
    _mock_mx.clear_cache = MagicMock()
    _mock_mx.get_active_memory = MagicMock(return_value=1024 * 1024 * 1024)
    _mock_mx.get_cache_memory = MagicMock(return_value=0)

    for _mod in [
        "mlx",
        "mlx.core",
        "mlx.nn",
        "mlx_lm",
        "mlx_lm.load",
        "mlx_lm.generate",
        "mlx_lm.models",
        "mlx_lm.models.cache",
        "mlx_lm.stream_generate",
        "mlx_vlm",
        "huggingface_hub",
    ]:
        sys.modules.setdefault(_mod, MagicMock())
    sys.modules["mlx.core"] = _mock_mx


import pytest


@pytest.fixture
def memory_exporter():
    """Enable olmlx tracing against an in-memory OTel exporter; tear down after.

    Shared by the tracing integration tests. Requires the [otel] dev deps
    (skips cleanly otherwise).
    """
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
    yield exporter
    tracing.shutdown_tracing()
