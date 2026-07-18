"""Tests for Ollama /api/generate legacy `context` continuation (issue #656)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from olmlx.engine.inference import (
    _augment_stream_with_context,
    _full_completion_inner,
    _stream_completion,
    build_context_input_tokens,
    generate_completion,
)
from olmlx.utils.timing import TimingStats


class _FakeTokenizer:
    """Minimal tokenizer honoring the `add_special_tokens` BOS heuristic."""

    def __init__(self, mapping, bos_token=None, bos_token_id=None):
        self._mapping = mapping  # (text, add_special) -> token ids
        self.bos_token = bos_token
        self.bos_token_id = bos_token_id

    def encode(self, text, add_special_tokens=True):
        return list(self._mapping[(text, add_special_tokens)])


class TestBuildContextInputTokens:
    def test_no_context_returns_prompt_tokens(self):
        tok = _FakeTokenizer({("Hello", True): [10, 11, 12]})
        assert build_context_input_tokens(tok, "Hello", None) == [10, 11, 12]

    def test_empty_context_returns_prompt_tokens(self):
        tok = _FakeTokenizer({("Hello", True): [10, 11, 12]})
        assert build_context_input_tokens(tok, "Hello", []) == [10, 11, 12]

    def test_context_prepended_no_bos(self):
        # bos_token None -> add_special True; no bos_token_id -> nothing stripped.
        tok = _FakeTokenizer({("Hello", True): [10, 11]})
        out = build_context_input_tokens(tok, "Hello", [1, 2, 3])
        assert out == [1, 2, 3, 10, 11]

    def test_context_strips_leading_bos_from_fresh_prompt(self):
        # Fresh prompt tokenizes with a leading BOS (id 1); prepending prior
        # context must not repeat the sequence-initial BOS.
        tok = _FakeTokenizer(
            {("Hello", True): [1, 10, 11]}, bos_token="<s>", bos_token_id=1
        )
        out = build_context_input_tokens(tok, "Hello", [1, 5, 6])
        assert out == [1, 5, 6, 10, 11]

    def test_context_keeps_prompt_when_no_leading_bos(self):
        tok = _FakeTokenizer(
            {("Hello", True): [10, 11]}, bos_token="<s>", bos_token_id=1
        )
        out = build_context_input_tokens(tok, "Hello", [1, 5, 6])
        assert out == [1, 5, 6, 10, 11]


class TestAugmentStreamWithContext:
    @pytest.mark.asyncio
    async def test_done_chunk_gets_context_and_drops_generated_tokens(self):
        async def src():
            yield {"text": "hi", "done": False}
            yield {"text": "", "done": True, "generated_tokens": [10, 11]}

        chunks = [c async for c in _augment_stream_with_context(src(), [1, 2, 3])]
        assert chunks[0] == {"text": "hi", "done": False}
        done = chunks[1]
        assert done["context"] == [1, 2, 3, 10, 11]
        assert "generated_tokens" not in done

    @pytest.mark.asyncio
    async def test_done_chunk_without_generated_tokens_uses_input_only(self):
        async def src():
            yield {"text": "", "done": True}

        chunks = [c async for c in _augment_stream_with_context(src(), [1, 2])]
        assert chunks[0]["context"] == [1, 2]


class TestGenerateCompletionContext:
    """generate_completion assembles `context` on the text (non-VLM) path."""

    @pytest.mark.asyncio
    async def test_non_stream_returns_context_with_prior_context(self, mock_manager):
        lm = mock_manager._loaded["qwen3:latest"]
        lm.text_tokenizer.encode = MagicMock(return_value=[5, 6, 7])
        lm.text_tokenizer.bos_token = None
        with patch(
            "olmlx.engine.inference._full_completion",
            new_callable=AsyncMock,
            return_value={"text": "hi", "done": True, "generated_tokens": [10, 11]},
        ) as mock_full:
            result = await generate_completion(
                mock_manager,
                "qwen3",
                "Hello",
                stream=False,
                apply_chat_template=False,
                return_context=True,
                context=[1, 2, 3],
            )
        assert result["context"] == [1, 2, 3, 5, 6, 7, 10, 11]
        assert "generated_tokens" not in result
        # Generation is fed the exact token ids and asked to collect them.
        _, kwargs = mock_full.call_args
        assert mock_full.call_args[0][1] == [1, 2, 3, 5, 6, 7]
        assert kwargs["collect_generated_tokens"] is True

    @pytest.mark.asyncio
    async def test_non_stream_returns_context_without_prior_context(self, mock_manager):
        lm = mock_manager._loaded["qwen3:latest"]
        lm.text_tokenizer.encode = MagicMock(return_value=[5, 6, 7])
        lm.text_tokenizer.bos_token = None
        with patch(
            "olmlx.engine.inference._full_completion",
            new_callable=AsyncMock,
            return_value={"text": "hi", "done": True, "generated_tokens": [10]},
        ):
            result = await generate_completion(
                mock_manager,
                "qwen3",
                "Hello",
                stream=False,
                apply_chat_template=False,
                return_context=True,
            )
        assert result["context"] == [5, 6, 7, 10]

    @pytest.mark.asyncio
    async def test_no_context_when_return_context_false(self, mock_manager):
        lm = mock_manager._loaded["qwen3:latest"]
        lm.text_tokenizer.encode = MagicMock(return_value=[5, 6, 7])
        lm.text_tokenizer.bos_token = None
        with patch(
            "olmlx.engine.inference._full_completion",
            new_callable=AsyncMock,
            return_value={"text": "hi", "done": True},
        ) as mock_full:
            result = await generate_completion(
                mock_manager,
                "qwen3",
                "Hello",
                stream=False,
                apply_chat_template=False,
                return_context=False,
            )
        assert "context" not in result
        _, kwargs = mock_full.call_args
        assert kwargs["collect_generated_tokens"] is False

    @pytest.mark.asyncio
    async def test_vlm_skips_context(self, mock_manager):
        lm = mock_manager._loaded["qwen3:latest"]
        lm.is_vlm = True
        with patch(
            "olmlx.engine.inference._full_completion",
            new_callable=AsyncMock,
            return_value={"text": "hi", "done": True},
        ) as mock_full:
            result = await generate_completion(
                mock_manager,
                "qwen3",
                "Hello",
                stream=False,
                apply_chat_template=False,
                return_context=True,
                context=[1, 2, 3],
            )
        assert "context" not in result
        _, kwargs = mock_full.call_args
        assert kwargs["collect_generated_tokens"] is False


class TestContextCollectionGaps:
    """Regressions for the speculative + batched context-collection gaps
    (aider review on PR #673)."""

    @pytest.mark.asyncio
    async def test_speculative_branch_collects_generated_tokens(self, mock_manager):
        """The non-streaming speculative path must populate
        ``generated_tokens_out`` — otherwise ``/api/generate`` returns a
        prompt-only context for speculative models."""
        lm = mock_manager._loaded["qwen3:latest"]
        # is_speculative derives from speculative_decoder; is_vlm/is_distributed
        # default False on the fixture.
        lm.speculative_decoder = MagicMock()
        lm.text_tokenizer.eos_token_id = 999

        fake_tokens = [
            SimpleNamespace(
                text="a",
                token=101,
                prompt_tokens=3,
                generation_tokens=1,
                prompt_tps=0.0,
                generation_tps=0.0,
            ),
            SimpleNamespace(
                text="b",
                token=102,
                prompt_tokens=3,
                generation_tokens=2,
                prompt_tps=0.0,
                generation_tps=0.0,
            ),
        ]

        def fake_spec_gen(*args, **kwargs):
            yield from fake_tokens

        async def fake_to_thread(fn, *a, **k):
            return fn(*a, **k)

        collector: list[int] = []
        with (
            patch(
                "olmlx.engine.speculative_stream.speculative_stream_generate",
                fake_spec_gen,
            ),
            patch("olmlx.engine.inference.mx", MagicMock()),
            patch("olmlx.engine.inference.materialize_audio", return_value=([], [])),
            patch(
                "olmlx.engine.inference.asyncio.to_thread",
                side_effect=fake_to_thread,
            ),
        ):
            await _full_completion_inner(
                lm,
                [1, 2, 3],
                max_tokens=8,
                gen_kwargs={},
                stats=TimingStats(),
                generated_tokens_out=collector,
            )
        assert collector == [101, 102]

    @pytest.mark.asyncio
    async def test_stream_context_request_skips_batched_path(self, mock_manager):
        """A streaming context request (collect_generated_tokens=True) must not
        take the batched path, which never emits per-token ids — mirroring the
        guard in _full_completion."""
        lm = mock_manager._loaded["qwen3:latest"]

        class _Sentinel(Exception):
            pass

        batched = MagicMock()

        async def fake_batched(*args, **kwargs):
            batched()
            yield {"text": "batched", "done": False}

        async def boom(*args, **kwargs):
            raise _Sentinel

        with (
            patch("olmlx.engine.inference._batch_eligible", return_value=True),
            patch("olmlx.engine.inference._stream_completion_batched", fake_batched),
            patch("olmlx.engine.inference._enter_inference_lock", boom),
        ):
            # collect=False -> batched path taken (yields its sentinel chunk).
            chunks = []
            async for c in _stream_completion(
                lm, "Hi", 5, {}, TimingStats(), collect_generated_tokens=False
            ):
                chunks.append(c)
            assert chunks == [{"text": "batched", "done": False}]
            assert batched.call_count == 1

            # collect=True -> batched path skipped; proceeds to the real path
            # (here stubbed to raise once past the guard).
            batched.reset_mock()
            with pytest.raises(_Sentinel):
                async for _ in _stream_completion(
                    lm, "Hi", 5, {}, TimingStats(), collect_generated_tokens=True
                ):
                    pass
            assert batched.call_count == 0


@pytest.mark.slow
@pytest.mark.real_model
class TestGenerateContextRealModel:
    """End-to-end round-trip against a real small model."""

    MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

    async def _manager(self):
        from olmlx.engine.model_manager import ModelManager
        from olmlx.engine.registry import ModelRegistry
        from olmlx.models.store import ModelStore

        registry = ModelRegistry()
        registry.load()
        store = ModelStore(registry)
        return ModelManager(registry, store)

    @pytest.mark.asyncio
    async def test_context_round_trip(self):
        try:
            manager = await self._manager()
        except Exception as e:  # pragma: no cover - env dependent
            pytest.skip(f"manager unavailable: {e}")
        try:
            first = await generate_completion(
                manager,
                self.MODEL,
                "Count: one, two,",
                {"num_predict": 8, "temperature": 0.0},
                stream=False,
                apply_chat_template=True,
                return_context=True,
            )
        except Exception as e:  # pragma: no cover - model may be absent
            pytest.skip(f"model not available: {e}")

        ctx = first["context"]
        assert isinstance(ctx, list) and ctx
        assert all(isinstance(t, int) for t in ctx)
        # Context ends with the generated tokens.
        eval_count = first["stats"].eval_count
        assert eval_count > 0
        assert len(ctx) > eval_count  # prompt tokens precede the generated ones

        # Passing the context back continues without error and grows it.
        second = await generate_completion(
            manager,
            self.MODEL,
            "three,",
            {"num_predict": 8, "temperature": 0.0},
            stream=False,
            apply_chat_template=True,
            return_context=True,
            context=ctx,
        )
        assert second["context"][: len(ctx)] == ctx
        assert len(second["context"]) > len(ctx)

    @pytest.mark.asyncio
    async def test_streaming_final_chunk_carries_context(self):
        try:
            manager = await self._manager()
        except Exception as e:  # pragma: no cover - env dependent
            pytest.skip(f"manager unavailable: {e}")
        try:
            gen = await generate_completion(
                manager,
                self.MODEL,
                "Count: one, two,",
                {"num_predict": 8, "temperature": 0.0},
                stream=True,
                apply_chat_template=True,
                return_context=True,
            )
        except Exception as e:  # pragma: no cover - model may be absent
            pytest.skip(f"model not available: {e}")

        done_chunk = None
        async for chunk in gen:
            if chunk.get("done"):
                done_chunk = chunk
        assert done_chunk is not None
        ctx = done_chunk["context"]
        assert isinstance(ctx, list) and all(isinstance(t, int) for t in ctx)
        assert "generated_tokens" not in done_chunk
