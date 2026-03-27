"""Streaming adapter for speculative decoding.

Bridges SpeculativeFlashDecoder into the CancellableStream / StreamToken
contract used by the inference pipeline.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Generator
from typing import Any

import mlx.core as mx

from olmlx.engine.flash.speculative import SpeculativeFlashDecoder
from olmlx.utils.streaming import StreamToken

logger = logging.getLogger(__name__)


def speculative_stream_generate(
    decoder: SpeculativeFlashDecoder,
    prompt_tokens: list[int],
    max_tokens: int,
    cancel_event: threading.Event,
    eos_token_id: int | None = None,
    tokenizer: Any = None,
) -> Generator[StreamToken, None, None]:
    """Sync generator that yields StreamToken objects for speculative decoding.

    Args:
        decoder: SpeculativeFlashDecoder with prefill/step API.
        prompt_tokens: Token IDs for the prompt.
        max_tokens: Maximum number of tokens to generate.
        cancel_event: Set to stop generation.
        eos_token_id: Stop generation when this token is produced.
        tokenizer: Tokenizer for incremental text decoding. If None, text is empty.
    """
    prompt_arr = mx.array([prompt_tokens])
    prompt_len = len(prompt_tokens)

    t0 = time.perf_counter()
    first_token = decoder.prefill(prompt_arr)
    prefill_elapsed = time.perf_counter() - t0
    prompt_tps_val = prompt_len / max(prefill_elapsed, 1e-9)

    generated: list[int] = [first_token]
    gen_count = 1

    # Incremental text decoding: decode full sequence and diff against previous length.
    prev_text_len = 0
    if tokenizer is not None:
        full_text = tokenizer.decode(generated)
        new_text = full_text
        prev_text_len = len(full_text)
    else:
        new_text = ""

    yield StreamToken(
        text=new_text,
        token=first_token,
        prompt_tokens=prompt_len,
        generation_tokens=gen_count,
        prompt_tps=prompt_tps_val,
        generation_tps=gen_count / max(prefill_elapsed, 1e-9),
    )

    if (
        eos_token_id is not None and first_token == eos_token_id
    ) or cancel_event.is_set():
        return

    while gen_count < max_tokens:
        if cancel_event.is_set():
            break

        accepted, _ = decoder.step()

        for token in accepted:
            if cancel_event.is_set() or gen_count >= max_tokens:
                break

            gen_count += 1
            generated.append(token)
            gen_elapsed = time.perf_counter() - t0

            # Decode full sequence and diff to get new text.
            # O(n) per token; a suffix-based approach could improve this for
            # very long sequences, but BPE boundary handling is subtle.
            if tokenizer is not None:
                full_text = tokenizer.decode(generated)
                new_text = full_text[prev_text_len:]
                prev_text_len = len(full_text)
            else:
                new_text = ""

            finish = None
            if eos_token_id is not None and token == eos_token_id:
                finish = "stop"
            elif gen_count >= max_tokens:
                finish = "length"

            yield StreamToken(
                text=new_text,
                token=token,
                prompt_tokens=prompt_len,
                generation_tokens=gen_count,
                prompt_tps=prompt_tps_val,
                generation_tps=gen_count / max(gen_elapsed, 1e-9),
                finish_reason=finish,
            )

            if finish is not None:
                return


def async_speculative_stream(
    decoder: SpeculativeFlashDecoder,
    tokenizer: Any,
    prompt: str | list[int],
    max_tokens: int,
) -> Any:
    """Create a CancellableStream for speculative decoding.

    Matches the interface of async_mlx_stream from utils/streaming.py.
    """
    from olmlx.utils.streaming import CancellableStream

    if isinstance(prompt, str):
        prompt_tokens = tokenizer.encode(prompt)
    else:
        prompt_tokens = prompt

    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    def gen_factory(cancel_event: threading.Event):
        return speculative_stream_generate(
            decoder,
            prompt_tokens,
            max_tokens=max_tokens,
            cancel_event=cancel_event,
            eos_token_id=eos_token_id,
            tokenizer=tokenizer,
        )

    stream = CancellableStream(gen_factory)
    stream.start()
    return stream
