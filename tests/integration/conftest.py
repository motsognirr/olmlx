"""Integration test fixtures — mock only at the MLX primitive boundary."""

import json
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from olmlx.engine.model_manager import ModelManager
from olmlx.engine.registry import ModelRegistry
from olmlx.models.store import ModelStore


# ---------------------------------------------------------------------------
# FakeTokenizer — real-ish tokenizer that supports the full inference.py API
# ---------------------------------------------------------------------------


class FakeTokenizer:
    """Deterministic tokenizer for integration tests.

    Supports apply_chat_template, encode, and the attributes that
    inference.py / template_caps.py inspect.
    """

    def __init__(self):
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self.eos_token_id = 2
        # Template string that references tools and enable_thinking so
        # detect_caps() returns both capabilities.
        self.chat_template = (
            "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}"
            "{% if tools %}[TOOLS]{% endif %}"
            "{% if enable_thinking %}[THINK]{% endif %}"
        )

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=False,
        add_generation_prompt=True,
        tools=None,
        enable_thinking=None,
        **kwargs,
    ):
        parts = []
        for m in messages:
            role = m.get("role", "unknown") if isinstance(m, dict) else "unknown"
            content = m.get("content", "") if isinstance(m, dict) else str(m)
            parts.append(f"{role}: {content}")
        prompt = "\n".join(parts) + "\nassistant:"
        if tools:
            prompt += " [TOOLS]"
        if enable_thinking:
            prompt += " [THINK]"
        if tokenize:
            return self.encode(prompt)
        return prompt

    def encode(self, text, add_special_tokens=True):
        """Deterministic char-based encoding."""
        tokens = [ord(c) % 1000 for c in text]
        if add_special_tokens and self.bos_token is not None:
            tokens = [1] + tokens  # BOS token = 1
        return tokens

    def decode(self, tokens, skip_special_tokens=True):
        return "".join(chr(t % 128) for t in tokens if t > 1)


# ---------------------------------------------------------------------------
# Fake stream_generate response objects
# ---------------------------------------------------------------------------


@dataclass
class FakeStreamResponse:
    """Mimics the response objects from mlx_lm.stream_generate."""

    text: str
    token: int = 42
    prompt_tokens: int = 10
    generation_tokens: int = 1
    prompt_tps: float = 100.0
    generation_tps: float = 50.0
    finish_reason: str | None = None


# ---------------------------------------------------------------------------
# Stream response control
# ---------------------------------------------------------------------------

# Module-level list that fake stream_generate reads from.
_stream_responses: list[list[str]] = []
_stream_call_count = 0
_generate_call_count = 0


def _reset_stream_responses():
    global _stream_responses, _stream_call_count, _generate_call_count
    _stream_responses = [["Hello", " world"]]
    _stream_call_count = 0
    _generate_call_count = 0


def set_stream_responses(responses: list[str]):
    """Configure the text chunks that fake stream_generate will yield."""
    global _stream_responses
    _stream_responses = [responses]


def set_multi_stream_responses(responses_list: list[list[str]]):
    """Configure multiple sequential stream responses (one per request)."""
    global _stream_responses
    _stream_responses = list(responses_list)


def _fake_stream_generate(model, tokenizer, *, prompt, max_tokens=512, **kwargs):
    """Fake mlx_lm.stream_generate that yields FakeStreamResponse objects."""
    global _stream_call_count
    idx = min(_stream_call_count, len(_stream_responses) - 1)
    _stream_call_count += 1
    texts = _stream_responses[idx] if _stream_responses else ["Hello", " world"]
    for i, text in enumerate(texts):
        yield FakeStreamResponse(
            text=text,
            token=42 + i,
            prompt_tokens=len(prompt) if isinstance(prompt, list) else 10,
            generation_tokens=i + 1,
            finish_reason="stop" if i == len(texts) - 1 else None,
        )


def _fake_generate(model, tokenizer, *, prompt, max_tokens=512, **kwargs):
    """Fake mlx_lm.generate for non-streaming path."""
    global _generate_call_count
    idx = min(_generate_call_count, len(_stream_responses) - 1)
    _generate_call_count += 1
    texts = _stream_responses[idx] if _stream_responses else ["Hello", " world"]
    return "".join(texts)


# ---------------------------------------------------------------------------
# mock_mlx_primitives — patches all MLX boundaries (autouse)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def mock_mlx_primitives(monkeypatch):
    """Patch all MLX primitives so tests don't need a GPU."""
    patches = []

    def _start(target, replacement=None):
        if replacement is None:
            replacement = MagicMock()
        p = patch(target, replacement)
        patches.append(p)
        return p.start()

    # mx.synchronize / mx.clear_cache everywhere
    _start("olmlx.engine.inference.mx.synchronize")
    _start("olmlx.engine.inference.mx.clear_cache")

    _start("olmlx.engine.model_manager.mx.synchronize")
    _start("olmlx.engine.model_manager.mx.clear_cache")

    # mlx.core.synchronize (used by CancellableStream._run thread)
    _start("mlx.core.synchronize")

    # mlx_lm.load → returns (MagicMock, FakeTokenizer)
    fake_tokenizer = FakeTokenizer()
    _start(
        "mlx_lm.load",
        MagicMock(return_value=(MagicMock(), fake_tokenizer)),
    )

    # mlx_lm.stream_generate
    _start("mlx_lm.stream_generate", _fake_stream_generate)

    # ORDER MATTERS: these two patches are order-dependent and must not be
    # separated.  The first targets an attribute on the mlx_lm.generate MODULE.
    # The second replaces mlx_lm.generate itself with a plain function.  If
    # swapped, the first patch would resolve to the replacement function (not
    # the module) and silently patch nothing useful.
    _start("mlx_lm.generate.generation_stream", MagicMock())
    _start("mlx_lm.generate", _fake_generate)

    # Prompt cache
    mock_make_cache = MagicMock(return_value=[MagicMock()])
    _start("olmlx.engine.inference.make_prompt_cache", mock_make_cache)
    # The load-time probe checks cache-layer class names against an allowlist;
    # MagicMock fails that check and would mark every test model non-trimmable,
    # silently disabling cache reuse in the integration tests.  Short-circuit
    # the probe so models behave like standard KVCache-only architectures.
    _start(
        "olmlx.engine.model_manager._cache_supports_trim",
        MagicMock(return_value=True),
    )
    # Successful trim returns the requested amount; without this, the
    # `trimmed != trim_amount` guard in _stream_completion would fire the
    # non-trimmable cache fallback and skip the suffix path.
    _start(
        "olmlx.engine.inference.trim_prompt_cache",
        MagicMock(side_effect=lambda c, n: n),
    )
    # _find_common_prefix is pure Python — let it run unpatched

    # HuggingFace download
    _start("huggingface_hub.snapshot_download", MagicMock(return_value="/tmp/fake"))

    # Memory functions — patch mx on the utils.memory module for Metal memory,
    # but patch get_system_memory_bytes directly to avoid @functools.cache poisoning
    # (patching os.sysconf has no effect after the first cached call).
    _start(
        "olmlx.utils.memory.mx.get_active_memory", MagicMock(return_value=1 * 1024**3)
    )
    _start("olmlx.utils.memory.mx.get_cache_memory", MagicMock(return_value=0))
    _start(
        "olmlx.utils.memory.get_system_memory_bytes",
        MagicMock(return_value=32 * 1024**3),
    )

    _reset_stream_responses()

    yield {
        "make_prompt_cache": mock_make_cache,
        "fake_tokenizer": fake_tokenizer,
    }

    for p in patches:
        p.stop()


# ---------------------------------------------------------------------------
# IntegrationContext
# ---------------------------------------------------------------------------


@dataclass
class IntegrationContext:
    client: AsyncClient
    manager: ModelManager
    store: ModelStore
    registry: ModelRegistry
    mocks: dict = field(default_factory=dict)

    def set_stream_responses(self, responses: list[str]):
        set_stream_responses(responses)

    def set_multi_stream_responses(self, responses_list: list[list[str]]):
        set_multi_stream_responses(responses_list)


@pytest.fixture
async def integration_ctx(tmp_path, mock_mlx_primitives, monkeypatch):
    """Create a real FastAPI app with real components but temp dirs and mocked MLX."""
    models_config = tmp_path / "models.json"
    models_config.write_text(json.dumps({"qwen3:latest": "Qwen/Qwen3-8B-MLX"}))
    aliases_path = tmp_path / "aliases.json"
    aliases_path.write_text("{}")

    models_dir = tmp_path / "models"
    models_dir.mkdir()

    monkeypatch.setattr("olmlx.config.settings.models_dir", models_dir)
    monkeypatch.setattr("olmlx.config.settings.models_config", models_config)
    monkeypatch.setattr("olmlx.engine.registry.settings.models_config", models_config)
    monkeypatch.setattr("olmlx.models.store.settings.models_dir", models_dir)

    registry = ModelRegistry()
    registry._aliases_path = aliases_path
    registry.load()

    store = ModelStore(registry)
    manager = ModelManager(registry, store)
    manager.start_expiry_checker()

    from olmlx.app import create_app

    app = create_app()
    app.state.registry = registry
    app.state.model_manager = manager
    app.state.model_store = store

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield IntegrationContext(
            client=client,
            manager=manager,
            store=store,
            registry=registry,
            mocks=mock_mlx_primitives,
        )

    await manager.stop()


# ---------------------------------------------------------------------------
# Lock leak safety (autouse)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
async def check_inference_lock():
    """Assert _inference_lock is not held after each test."""
    yield
    from olmlx.engine.inference import _inference_lock

    if _inference_lock.locked():
        # Force-release to not poison other tests, but fail this one
        _inference_lock.release()
        pytest.fail("_inference_lock was still held after test completed — lock leak!")


# ---------------------------------------------------------------------------
# SSE parsing helper
# ---------------------------------------------------------------------------


def parse_sse_events(text: str) -> list[dict]:
    """Parse SSE text into a list of {event, data} dicts."""
    events = []
    current_event = None
    current_data = []

    for line in text.split("\n"):
        if line.startswith("event: "):
            current_event = line[7:]
        elif line.startswith("data: "):
            current_data.append(line[6:])
        elif line == "" and (current_event or current_data):
            data_str = "\n".join(current_data)
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                data = data_str
            events.append({"event": current_event, "data": data})
            current_event = None
            current_data = []

    return events
