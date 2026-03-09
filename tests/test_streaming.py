"""Tests for olmlx.utils.streaming."""

import logging
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from olmlx.utils.streaming import CancellableStream, StreamToken, async_mlx_stream


class TestStreamToken:
    def test_defaults(self):
        tok = StreamToken(
            text="hi",
            token=1,
            prompt_tokens=5,
            generation_tokens=1,
            prompt_tps=100.0,
            generation_tps=50.0,
        )
        assert tok.text == "hi"
        assert tok.finish_reason is None

    def test_with_finish_reason(self):
        tok = StreamToken(
            text="",
            token=None,
            prompt_tokens=5,
            generation_tokens=10,
            prompt_tps=100.0,
            generation_tps=50.0,
            finish_reason="stop",
        )
        assert tok.finish_reason == "stop"


class TestAsyncMlxStream:
    @pytest.mark.asyncio
    async def test_text_model_stream(self):
        mock_responses = [
            MagicMock(
                text="Hello",
                token=1,
                prompt_tokens=5,
                generation_tokens=1,
                prompt_tps=100.0,
                generation_tps=50.0,
                finish_reason=None,
            ),
            MagicMock(
                text=" world",
                token=2,
                prompt_tokens=5,
                generation_tokens=2,
                prompt_tps=100.0,
                generation_tps=50.0,
                finish_reason="stop",
            ),
        ]

        mock_mlx_lm = MagicMock()
        mock_mlx_lm.stream_generate = MagicMock(return_value=iter(mock_responses))

        with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
            stream = async_mlx_stream(
                MagicMock(),
                MagicMock(),
                "prompt",
                max_tokens=10,
                is_vlm=False,
            )
            tokens = []
            async for tok in stream:
                tokens.append(tok)

        assert len(tokens) == 2
        assert tokens[0].text == "Hello"
        assert tokens[1].text == " world"

    @pytest.mark.asyncio
    async def test_vlm_stream(self):
        mock_responses = [
            MagicMock(
                text="Description",
                token=None,
                prompt_tokens=10,
                generation_tokens=1,
                prompt_tps=50.0,
                generation_tps=30.0,
            ),
        ]

        mock_mlx_vlm = MagicMock()
        mock_mlx_vlm.stream_generate = MagicMock(return_value=iter(mock_responses))

        with patch.dict("sys.modules", {"mlx_vlm": mock_mlx_vlm}):
            stream = async_mlx_stream(
                MagicMock(),
                MagicMock(),
                "prompt",
                max_tokens=10,
                is_vlm=True,
                images=["img.jpg"],
            )
            tokens = []
            async for tok in stream:
                tokens.append(tok)

        assert len(tokens) == 1
        assert tokens[0].text == "Description"

    @pytest.mark.asyncio
    async def test_error_propagation(self):
        mock_mlx_lm = MagicMock()
        mock_mlx_lm.stream_generate = MagicMock(
            side_effect=RuntimeError("GPU error"),
        )

        with patch.dict("sys.modules", {"mlx_lm": mock_mlx_lm}):
            stream = async_mlx_stream(
                MagicMock(),
                MagicMock(),
                "prompt",
                max_tokens=10,
                is_vlm=False,
            )
            with pytest.raises(RuntimeError, match="GPU error"):
                async for _ in stream:
                    pass


def _fake_generator(count=10, delay=0.01):
    """A generator that yields count items with a small delay."""
    for i in range(count):
        time.sleep(delay)
        yield type(
            "Resp",
            (),
            {
                "text": f"tok{i}",
                "token": i,
                "prompt_tokens": 5,
                "generation_tokens": i + 1,
                "prompt_tps": 100.0,
                "generation_tps": 50.0,
                "finish_reason": "stop" if i == count - 1 else None,
            },
        )()


class TestCancellableStream:
    @pytest.mark.asyncio
    async def test_cancel_event_stops_generation(self):
        """Cancelling the stream should stop the background thread early."""
        tokens = []
        stream = CancellableStream(
            lambda cancel_event: (
                tok
                for tok in _fake_generator(100, delay=0.01)
                if not cancel_event.is_set()
            ),
        )
        stream.start()

        count = 0
        async for token in stream:
            tokens.append(token)
            count += 1
            if count >= 3:
                stream.cancel()
                break

        await stream.drain_and_join()
        assert len(tokens) <= 10

    @pytest.mark.asyncio
    async def test_drain_and_join_completes(self):
        """After drain_and_join, the background thread should be dead."""
        stream = CancellableStream(
            lambda cancel_event: _fake_generator(5, delay=0.001),
        )
        stream.start()

        async for token in stream:
            stream.cancel()
            break

        await stream.drain_and_join()
        assert not stream._thread.is_alive()

    @pytest.mark.asyncio
    async def test_normal_completion_still_works(self):
        """Happy path: all tokens are yielded without cancellation."""
        stream = CancellableStream(
            lambda cancel_event: _fake_generator(5, delay=0.001),
        )
        stream.start()

        tokens = []
        async for token in stream:
            tokens.append(token)

        assert len(tokens) == 5
        assert tokens[0].text == "tok0"
        assert tokens[-1].text == "tok4"
        assert tokens[-1].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_drain_after_normal_completion(self):
        """drain_and_join is safe to call after normal completion."""
        stream = CancellableStream(
            lambda cancel_event: _fake_generator(3, delay=0.001),
        )
        stream.start()

        async for _ in stream:
            pass

        await stream.drain_and_join()
        assert not stream._thread.is_alive()

    @pytest.mark.asyncio
    async def test_error_propagated(self):
        """Errors from the generator should propagate as RuntimeError."""

        def failing_gen(cancel_event):
            yield type(
                "Resp",
                (),
                {
                    "text": "ok",
                    "token": 0,
                    "prompt_tokens": 1,
                    "generation_tokens": 1,
                    "prompt_tps": 0,
                    "generation_tps": 0,
                    "finish_reason": None,
                },
            )()
            raise ValueError("boom")

        stream = CancellableStream(failing_gen)
        stream.start()

        with pytest.raises(RuntimeError, match="boom"):
            async for _ in stream:
                pass


def _blocking_generator(block_event: threading.Event):
    """A generator factory that blocks until block_event is set."""

    def gen(cancel_event):
        # Yield one token then block
        yield type(
            "Resp",
            (),
            {
                "text": "tok0",
                "token": 0,
                "prompt_tokens": 5,
                "generation_tokens": 1,
                "prompt_tps": 100.0,
                "generation_tps": 50.0,
                "finish_reason": None,
            },
        )()
        # Simulate GPU deadlock — block until explicitly released
        block_event.wait()

    return gen


class TestDrainAndJoinTimeout:
    @pytest.mark.asyncio
    async def test_drain_and_join_returns_within_timeout_on_stuck_thread(self):
        """drain_and_join with a stuck thread should return within the timeout, not hang."""
        block_event = threading.Event()
        stream = CancellableStream(_blocking_generator(block_event))
        stream.start()

        # Consume the one token
        async for _ in stream:
            break

        start = time.monotonic()
        await stream.drain_and_join(timeout=0.5)
        elapsed = time.monotonic() - start

        # Should return within timeout + some slack, not hang forever
        assert elapsed < 2.0

        # Clean up: unblock the thread so it can exit
        block_event.set()
        stream._thread.join(timeout=5)

    @pytest.mark.asyncio
    async def test_drain_and_join_logs_error_on_timeout(self, caplog):
        """drain_and_join should log an error when the thread won't exit."""
        block_event = threading.Event()
        stream = CancellableStream(_blocking_generator(block_event))
        stream.start()

        # Consume the one token
        async for _ in stream:
            break

        with caplog.at_level(logging.ERROR, logger="olmlx.utils.streaming"):
            await stream.drain_and_join(timeout=0.5)

        assert any("GPU resource leak" in record.message for record in caplog.records)

        # Clean up
        block_event.set()
        stream._thread.join(timeout=5)

    @pytest.mark.asyncio
    async def test_drain_and_join_respects_timeout_value(self):
        """drain_and_join should respect different timeout values."""
        block_event = threading.Event()
        stream = CancellableStream(_blocking_generator(block_event))
        stream.start()

        async for _ in stream:
            break

        start = time.monotonic()
        await stream.drain_and_join(timeout=0.3)
        elapsed = time.monotonic() - start

        # Should be close to the timeout, not 60s default or infinite
        assert elapsed < 1.5

        block_event.set()
        stream._thread.join(timeout=5)

    @pytest.mark.asyncio
    async def test_drain_and_join_normal_completion_unaffected_by_timeout(self):
        """Normal completion should still work quickly with timeout parameter."""
        stream = CancellableStream(
            lambda cancel_event: _fake_generator(3, delay=0.001),
        )
        stream.start()

        async for _ in stream:
            pass

        start = time.monotonic()
        await stream.drain_and_join(timeout=30.0)
        elapsed = time.monotonic() - start

        # Normal completion should be fast, well under the timeout
        assert elapsed < 5.0
        assert not stream._thread.is_alive()
