import pytest

pytest.importorskip("opentelemetry")

# memory_exporter fixture is provided by the root conftest.py.


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
    # ttft_ns / cache_hit are surfaced on the decode span (the prefill forward
    # happens lazily in the worker thread, so the prefill span can't time it).
    decode_attrs = dict(by_name["decode"].attributes)
    assert "ttft_ns" in decode_attrs
    assert decode_attrs["cache_hit"] is False  # completions path, no prompt cache


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


def test_flash_weight_load_span(memory_exporter, tmp_path):
    """A flash weight load emits a flash.weight_load span with layer_idx."""
    import numpy as np
    from safetensors.numpy import save_file

    from olmlx.engine.flash.bundler import bundle_ffn_weights
    from olmlx.engine.flash.weight_store import FlashWeightStore

    hidden, inter, num_layers = 16, 8, 2
    tensors = {}
    for layer in range(num_layers):
        prefix = f"model.layers.{layer}.mlp"
        tensors[f"{prefix}.gate_proj.weight"] = np.random.randn(inter, hidden).astype(
            np.float16
        )
        tensors[f"{prefix}.up_proj.weight"] = np.random.randn(inter, hidden).astype(
            np.float16
        )
        tensors[f"{prefix}.down_proj.weight"] = np.random.randn(hidden, inter).astype(
            np.float16
        )
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    save_file(tensors, str(model_dir / "model.safetensors"))
    output_dir = tmp_path / "flash"
    bundle_ffn_weights(model_dir, output_dir)

    store = FlashWeightStore(output_dir, num_io_threads=4, cache_budget_neurons=32)
    store.load_neurons(1, [0, 2, 5])

    span = next(
        s for s in memory_exporter.get_finished_spans() if s.name == "flash.weight_load"
    )
    attrs = dict(span.attributes)
    assert attrs["layer_idx"] == 1
    assert attrs["active_neurons"] == 3


def test_disk_cache_spans(memory_exporter, tmp_path):
    """_save_to_disk and _load_from_disk emit cache.disk_write / cache.disk_read
    spans with cache_id, bytes, and hit attributes."""
    from pathlib import Path
    from unittest.mock import patch

    from olmlx.engine.model_manager import CachedPromptState, PromptCacheStore

    store = PromptCacheStore(max_slots=4, disk_path=tmp_path, model_name="test-model")
    state = CachedPromptState(tokens=[1, 2, 3], cache=["kv"])
    cid = "abc123"

    def fake_save(path, cache, metadata):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x" * 128)

    def fake_load(path, return_metadata=False):
        return ["kv"], {
            "tokens": "[1, 2, 3]",
            "cache_type": "assistant",
            "is_checkpoint": "0",
        }

    with patch(
        "olmlx.engine.prompt_cache.store.save_prompt_cache", side_effect=fake_save
    ):
        store._save_to_disk(cid, state)
    with patch(
        "olmlx.engine.prompt_cache.store.load_prompt_cache", side_effect=fake_load
    ):
        loaded = store._load_from_disk(cid)
    assert loaded is not None

    by_name = {s.name: s for s in memory_exporter.get_finished_spans()}
    assert "cache.disk_write" in by_name
    assert "cache.disk_read" in by_name
    assert dict(by_name["cache.disk_write"].attributes)["cache_id"] == cid
    assert "bytes" in dict(by_name["cache.disk_write"].attributes)
    assert dict(by_name["cache.disk_read"].attributes)["hit"] is True


@pytest.mark.asyncio
async def test_mcp_tool_call_span(memory_exporter):
    """_exec_tool emits an mcp.tool_call span with tool.name (and mcp.server
    when the call is dispatched through MCP)."""
    from unittest.mock import AsyncMock, MagicMock

    from olmlx.chat.config import ChatConfig
    from olmlx.chat.session import ChatSession

    config = ChatConfig(model_name="test:latest")
    manager = MagicMock()
    mcp = MagicMock()
    mcp.name = "myserver"
    mcp.call_tool = AsyncMock(return_value="ok")
    session = ChatSession(config=config, manager=manager, mcp=mcp)

    await session._exec_tool({"name": "echo", "input": {"x": 1}, "id": "t1"})

    span = next(
        s for s in memory_exporter.get_finished_spans() if s.name == "mcp.tool_call"
    )
    attrs = dict(span.attributes)
    assert attrs["tool.name"] == "echo"
    assert attrs["mcp.server"] == "myserver"
