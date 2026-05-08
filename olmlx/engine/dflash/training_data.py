"""Streaming HuggingFace dataset loader for DFlash draft training.

Provides a small, dependency-light wrapper around ``datasets`` for
streaming chat-style training data. Falls back to synthetic prompts
when ``datasets`` is unavailable so tests and air-gapped environments
still get a working signal.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)

# Default corpus: instruction-tuning data is the right shape for drafts
# attached to instruction-tuned targets, which is the dominant use case.
DEFAULT_DATASET = "HuggingFaceH4/ultrachat_200k"
DEFAULT_DATASET_SPLIT = "train_sft"

# Synthetic-fallback prompts used when ``datasets`` is unavailable.
# Diverse enough to give the draft non-trivial gradients but small —
# this is a smoke-test fallback, not a real training corpus.
_FALLBACK_PROMPTS = [
    "Explain how a transformer language model works in simple terms.",
    "Write a short paragraph about the history of programming languages.",
    "Describe the differences between supervised and unsupervised learning.",
    "What are the main applications of speculative decoding?",
    "Summarize the key ideas behind block-diffusion drafting.",
    "Compare and contrast Apple Silicon and CUDA for ML inference.",
    "Outline the steps to fine-tune a small language model.",
    "Discuss the trade-offs between draft model size and acceptance rate.",
]


def _format_chat(example: dict[str, Any], tokenizer: Any) -> str | None:
    """Render a chat-style example through the tokenizer's chat template.

    Returns ``None`` when the example shape is not recognized so the
    caller can skip and continue streaming.
    """
    messages = example.get("messages")
    if isinstance(messages, list) and messages:
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        except Exception:  # noqa: BLE001 — render path is best-effort
            return None
    text = example.get("text")
    if isinstance(text, str):
        return text
    return None


def _stream_hf_dataset(
    dataset: str,
    split: str,
) -> Iterator[dict[str, Any]] | None:
    """Open a streaming HuggingFace dataset; return None if datasets is missing."""
    try:
        import datasets as _ds
    except ImportError:
        logger.warning(
            "HuggingFace datasets not installed; using fallback prompts. "
            "Install with: pip install datasets"
        )
        return None
    try:
        ds = _ds.load_dataset(dataset, split=split, streaming=True)
    except Exception as exc:  # noqa: BLE001
        # Only fall back when the user accepted the default dataset.
        # If they passed ``--data foo/bar`` and that fails, re-raise:
        # silently swapping in 8 synthetic prompts would otherwise let
        # a multi-thousand-step training run produce a real-looking
        # checkpoint that trained on essentially nothing.
        if dataset != DEFAULT_DATASET:
            raise
        logger.warning(
            "Failed to open default dataset %s split=%s: %s. Falling "
            "back to synthetic prompts.",
            dataset,
            split,
            exc,
        )
        return None
    return iter(ds)


def stream_training_batches(
    tokenizer: Any,
    *,
    dataset: str = DEFAULT_DATASET,
    split: str = DEFAULT_DATASET_SPLIT,
    batch_size: int = 4,
    seq_len: int = 2048,
    min_seq_len: int = 64,
    max_examples: int | None = None,
) -> Iterator[mx.array]:
    """Yield ``(batch_size, seq_len)`` token-id batches.

    Pads short sequences with the tokenizer's pad/eos token so every
    batch has a uniform length. Truncates long sequences. Skips
    sequences shorter than ``min_seq_len`` after tokenization since
    they don't span a meaningful training window.

    Falls back to a small synthetic-prompts loop when ``datasets`` is
    unavailable or the dataset can't be opened — this keeps tests
    deterministic without network access.
    """
    # ``or`` would short-circuit on token ID 0, which is the standard pad
    # token for Llama 2 / Mistral / Qwen 1.x — fall through only on None.
    _pad = getattr(tokenizer, "pad_token_id", None)
    _eos = getattr(tokenizer, "eos_token_id", None)
    pad_token_id = _pad if _pad is not None else (_eos if _eos is not None else 0)

    def _tokenize(text: str) -> list[int] | None:
        try:
            ids = tokenizer.encode(text, add_special_tokens=True)
        except Exception:  # noqa: BLE001
            return None
        if len(ids) < min_seq_len:
            return None
        if len(ids) > seq_len:
            ids = ids[:seq_len]
        else:
            ids = ids + [pad_token_id] * (seq_len - len(ids))
        return ids

    def _emit(batch: list[list[int]]) -> mx.array:
        return mx.array(batch)

    yielded = 0
    batch: list[list[int]] = []

    stream = _stream_hf_dataset(dataset, split)
    if stream is not None:
        try:
            for example in stream:
                rendered = _format_chat(example, tokenizer)
                if rendered is None:
                    continue
                ids = _tokenize(rendered)
                if ids is None:
                    continue
                batch.append(ids)
                if len(batch) == batch_size:
                    yield _emit(batch)
                    batch = []
                    yielded += 1
                    if max_examples is not None and yielded >= max_examples:
                        return
        finally:
            if hasattr(stream, "close"):
                stream.close()
        if batch:
            # Top up the final partial batch by cycling through the
            # *current partial batch's* examples (not earlier examples
            # in the stream, which we no longer have references to)
            # rather than emitting an undersized batch — that would
            # break the static-shape assumptions in the training loop.
            # If ``orig == 1`` the final batch is one example repeated;
            # acceptable for one step out of thousands but worth knowing
            # for short test runs.
            orig = len(batch)
            while len(batch) < batch_size:
                batch.append(batch[len(batch) % orig])
            yield _emit(batch)
        return

    # Fallback path — synthetic prompts cycled until ``max_examples``
    # batches have been yielded. ``max_examples is None`` means
    # "uncapped" — let the consumer (e.g. ``precompute_target_hiddens``
    # capping via ``num_shards``) decide when to stop iterating. A
    # previous version hardcoded a 64-batch cap when ``max_examples``
    # was None, silently truncating ``--shards 500`` precompute runs
    # whose HF dataset was unavailable.
    while max_examples is None or yielded < max_examples:
        # Detect a full pass with no usable prompts so a caller that
        # raises ``min_seq_len`` above every fallback's tokenized length
        # gets a clear error instead of an infinite loop.
        made_progress = False
        for prompt in _FALLBACK_PROMPTS:
            ids = _tokenize(prompt)
            if ids is None:
                continue
            made_progress = True
            batch.append(ids)
            if len(batch) == batch_size:
                yield _emit(batch)
                batch = []
                yielded += 1
                if max_examples is not None and yielded >= max_examples:
                    return
        if not made_progress:
            raise RuntimeError(
                f"All fallback prompts tokenize to fewer than "
                f"min_seq_len={min_seq_len} tokens; lower min_seq_len or "
                "provide a real dataset via --data."
            )
