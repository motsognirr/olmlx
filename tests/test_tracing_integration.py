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


@pytest.mark.asyncio
async def test_completion_trace_has_inference_prefill_decode(
    memory_exporter, mock_manager
):
    """AC #1: a (non-streaming) completion renders inference → {prefill, decode}.

    Drives generate_completion through the mocked non-streaming harness used by
    tests/test_inference.py (mx / mlx_lm / asyncio.to_thread patched). The
    inference, prefill and decode spans all live on the coroutine side, so they
    are exercised even though the actual generation thread is mocked away.
    """
    from unittest.mock import AsyncMock, MagicMock, patch

    from olmlx.engine.inference import generate_completion

    mock_mx = MagicMock()
    mock_mx.core = mock_mx
    mock_mlx_lm = MagicMock()

    with patch("olmlx.engine.inference.mx", mock_mx):
        with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
            with patch(
                "olmlx.engine.inference.asyncio.to_thread",
                new_callable=AsyncMock,
                return_value="Generated output",
            ):
                result = await generate_completion(
                    mock_manager, "qwen3", "Hello", stream=False
                )

    assert result["text"] == "Generated output"

    by_name = {s.name: s for s in memory_exporter.get_finished_spans()}
    assert "inference" in by_name
    assert "prefill" in by_name
    assert "decode" in by_name
    assert by_name["prefill"].parent.span_id == by_name["inference"].context.span_id
    assert by_name["decode"].parent.span_id == by_name["inference"].context.span_id
    assert dict(by_name["inference"].attributes)["strategy"] == "none"


@pytest.mark.asyncio
async def test_streaming_trace_has_inference_prefill_decode(
    memory_exporter, mock_manager
):
    """Streaming completion renders inference → {prefill, decode}.

    The inference span is held open across the streamed generator's lifetime by
    _trace_inference_gen (only wired in when tracing is enabled); prefill wraps
    stream creation and decode wraps the consumption loop inside the seam.
    """
    from unittest.mock import AsyncMock, MagicMock, patch

    from olmlx.engine.inference import generate_completion
    from olmlx.utils.streaming import CancellableStream, StreamToken

    tokens = [
        StreamToken(
            text="Hello",
            token=1,
            prompt_tokens=5,
            generation_tokens=1,
            prompt_tps=100.0,
            generation_tps=50.0,
        ),
        StreamToken(
            text=" world",
            token=2,
            prompt_tokens=5,
            generation_tokens=2,
            prompt_tps=100.0,
            generation_tps=50.0,
        ),
    ]
    mock_stream = MagicMock(spec=CancellableStream)
    mock_stream.drain_and_join = AsyncMock()
    mock_stream._thread = None
    token_iter = iter(tokens)

    async def anext_impl():
        try:
            return next(token_iter)
        except StopIteration:
            raise StopAsyncIteration

    mock_stream.__aiter__ = lambda self: self
    mock_stream.__anext__ = lambda self: anext_impl()

    mock_mx = MagicMock()
    with patch("olmlx.engine.inference.mx", mock_mx):
        with patch("olmlx.engine.inference.async_mlx_stream", return_value=mock_stream):
            gen = await generate_completion(mock_manager, "qwen3", "Hello", stream=True)
            async for _chunk in gen:
                pass

    by_name = {s.name: s for s in memory_exporter.get_finished_spans()}
    assert "inference" in by_name
    assert "prefill" in by_name
    assert "decode" in by_name
    assert dict(by_name["inference"].attributes)["gen.stream"] is True
    assert by_name["prefill"].parent.span_id == by_name["inference"].context.span_id
    assert by_name["decode"].parent.span_id == by_name["inference"].context.span_id


def test_speculative_step_and_verify_spans(memory_exporter):
    """A classic speculative decode emits spec.prefill, per-step spec.step, and
    a spec.verify sub-span with proposed/accepted attributes."""
    import mlx.core as mx

    from olmlx.engine.speculative import SpeculativeDecoder
    from tests.test_flash_speculative import MockModel

    vocab_size, hidden_size = 32, 16
    draft = MockModel(vocab_size, hidden_size)
    target = MockModel(vocab_size, hidden_size)
    decoder = SpeculativeDecoder(
        draft_model=draft,
        target_model=target,
        num_speculative_tokens=3,
    )

    decoder.prefill(mx.array([[1, 2, 3, 4, 5]]))
    for _ in range(3):
        decoder.step()

    spans = memory_exporter.get_finished_spans()
    names = [s.name for s in spans]
    assert "spec.prefill" in names
    assert names.count("spec.step") == 3
    assert "spec.verify" in names

    step = next(s for s in spans if s.name == "spec.step")
    attrs = dict(step.attributes)
    assert "proposed" in attrs and "accepted" in attrs
    assert attrs["strategy"] in {"classic", "pld", "dflash", "eagle", "self"}
