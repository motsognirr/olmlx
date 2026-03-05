import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Any

from mlx_ollama.engine.model_manager import LoadedModel, ModelManager
from mlx_ollama.utils.streaming import async_mlx_stream
from mlx_ollama.utils.timing import Timer, TimingStats

logger = logging.getLogger(__name__)


def _build_generate_kwargs(options: dict | None, is_vlm: bool = False) -> dict:
    """Convert Ollama options dict to mlx_lm/mlx_vlm generate kwargs."""
    if not options:
        return {}
    kwargs = {}
    # mlx-lm uses "temp", mlx-vlm uses "temperature"
    temp_key = "temperature" if is_vlm else "temp"
    mappings = {
        "temperature": temp_key,
        "top_p": "top_p",
        "top_k": "top_k",
        "seed": "seed",
        "num_predict": "max_tokens",
        "repeat_penalty": "repetition_penalty",
    }
    for ollama_key, mlx_key in mappings.items():
        if ollama_key in options:
            kwargs[mlx_key] = options[ollama_key]
    # stop is only supported by mlx-lm
    if not is_vlm and "stop" in options:
        kwargs["stop"] = options["stop"]
    return kwargs


def _apply_chat_template_text(
    tokenizer: Any,
    messages: list[dict],
    tools: list[dict] | None = None,
) -> str:
    """Apply chat template for text-only models (mlx-lm)."""
    try:
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if tools:
            kwargs["tools"] = tools
        return tokenizer.apply_chat_template(messages, **kwargs)
    except Exception:
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        parts.append("assistant: ")
        return "\n".join(parts)


def _apply_chat_template_vlm(
    processor: Any,
    model: Any,
    messages: list[dict],
    images: list[str] | None = None,
) -> str:
    """Apply chat template for vision-language models (mlx-vlm)."""
    import mlx_vlm

    config = model.config if hasattr(model, "config") else {}
    num_images = len(images) if images else 0
    # Pass the full message list so the model gets proper conversation context
    return mlx_vlm.apply_chat_template(
        processor, config, messages, num_images=num_images
    )


def _extract_images(messages: list[dict]) -> list[str] | None:
    """Extract image URLs/paths from message content."""
    images = []
    for msg in messages:
        if msg.get("images"):
            images.extend(msg["images"])
    return images if images else None


async def generate_completion(
    manager: ModelManager,
    model_name: str,
    prompt: str,
    options: dict | None = None,
    stream: bool = True,
    keep_alive: str | None = None,
    max_tokens: int = 512,
    images: list[str] | None = None,
) -> AsyncGenerator[dict, None] | dict:
    """Generate a text completion, streaming or not."""
    stats = TimingStats()

    with Timer() as load_timer:
        lm = await manager.ensure_loaded(model_name, keep_alive)
    stats.load_duration = load_timer.duration_ns

    gen_kwargs = _build_generate_kwargs(options, is_vlm=lm.is_vlm)
    mt = gen_kwargs.pop("max_tokens", max_tokens)

    if stream:
        return _stream_completion(lm, prompt, mt, gen_kwargs, stats, images)
    else:
        return await _full_completion(lm, prompt, mt, gen_kwargs, stats, images)


async def _stream_completion(
    lm: LoadedModel,
    prompt: str,
    max_tokens: int,
    gen_kwargs: dict,
    stats: TimingStats,
    images: list[str] | None = None,
) -> AsyncGenerator[dict, None]:
    with Timer() as total_timer:
        with Timer() as eval_timer:
            async for token in async_mlx_stream(
                lm.model, lm.tokenizer, prompt,
                max_tokens=max_tokens,
                is_vlm=lm.is_vlm,
                images=images,
                **gen_kwargs,
            ):
                yield {"text": token.text, "done": False}
                stats.eval_count = token.generation_tokens
                stats.prompt_eval_count = token.prompt_tokens

        stats.eval_duration = eval_timer.duration_ns

    stats.total_duration = total_timer.duration_ns
    yield {"text": "", "done": True, "stats": stats}


async def _full_completion(
    lm: LoadedModel,
    prompt: str,
    max_tokens: int,
    gen_kwargs: dict,
    stats: TimingStats,
    images: list[str] | None = None,
) -> dict:
    with Timer() as total_timer:
        with Timer() as eval_timer:
            if lm.is_vlm:
                import mlx_vlm
                result = await asyncio.to_thread(
                    mlx_vlm.generate,
                    lm.model,
                    lm.tokenizer,
                    prompt=prompt,
                    image=images,
                    max_tokens=max_tokens,
                    **gen_kwargs,
                )
            else:
                import mlx_lm
                result = await asyncio.to_thread(
                    mlx_lm.generate,
                    lm.model,
                    lm.tokenizer,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    **gen_kwargs,
                )

    stats.eval_duration = eval_timer.duration_ns
    stats.total_duration = total_timer.duration_ns

    # mlx_vlm.generate returns GenerationResult dataclass
    if hasattr(result, "text"):
        text = result.text
    elif isinstance(result, str):
        text = result
    else:
        text = str(result)
    return {"text": text, "done": True, "stats": stats}


async def generate_chat(
    manager: ModelManager,
    model_name: str,
    messages: list[dict],
    options: dict | None = None,
    tools: list[dict] | None = None,
    stream: bool = True,
    keep_alive: str | None = None,
    max_tokens: int = 512,
) -> AsyncGenerator[dict, None] | dict:
    """Generate a chat completion."""
    stats = TimingStats()

    with Timer() as load_timer:
        lm = await manager.ensure_loaded(model_name, keep_alive)
    stats.load_duration = load_timer.duration_ns

    images = _extract_images(messages)

    if lm.is_vlm:
        prompt = _apply_chat_template_vlm(lm.tokenizer, lm.model, messages, images)
    else:
        prompt = _apply_chat_template_text(lm.tokenizer, messages, tools)
        tokenizer = lm.tokenizer
        stats.prompt_eval_count = len(tokenizer.encode(prompt))

    gen_kwargs = _build_generate_kwargs(options, is_vlm=lm.is_vlm)
    mt = gen_kwargs.pop("max_tokens", max_tokens)

    if stream:
        return _stream_completion(lm, prompt, mt, gen_kwargs, stats, images)
    else:
        return await _full_completion(lm, prompt, mt, gen_kwargs, stats, images)


async def generate_embeddings(
    manager: ModelManager,
    model_name: str,
    texts: list[str],
    keep_alive: str | None = None,
) -> list[list[float]]:
    """Generate embeddings using the model's hidden states."""
    import mlx.core as mx

    lm = await manager.ensure_loaded(model_name, keep_alive)
    embeddings = []

    tokenizer = lm.tokenizer
    if hasattr(tokenizer, "tokenizer"):
        tokenizer = tokenizer.tokenizer

    for text in texts:
        tokens = tokenizer.encode(text)
        input_ids = mx.array([tokens])
        outputs = lm.model(input_ids)
        if hasattr(outputs, "hidden_states") and outputs.hidden_states:
            hidden = outputs.hidden_states[-1]
        elif hasattr(outputs, "last_hidden_state"):
            hidden = outputs.last_hidden_state
        else:
            hidden = outputs
        embedding = mx.mean(hidden[0], axis=0)
        embeddings.append(embedding.tolist())

    return embeddings
