"""Sampling-defaults layering and generate-kwargs assembly (extracted from inference.py)."""

import logging
from typing import TYPE_CHECKING

import mlx.core as mx

from olmlx.config import settings

if TYPE_CHECKING:
    pass

try:
    from mlx_lm.models.cache import (
        KVCache,
        RotatingKVCache,
        make_prompt_cache,
        trim_prompt_cache,
    )
    from mlx_lm.utils import common_prefix_len as _find_common_prefix
except ImportError:  # pragma: no cover
    make_prompt_cache = None  # type: ignore[assignment]
    trim_prompt_cache = None  # type: ignore[assignment]
    KVCache = None  # type: ignore[assignment]
    RotatingKVCache = None  # type: ignore[assignment]
    _find_common_prefix = None  # type: ignore[assignment]
    logging.getLogger(__name__).warning(
        "mlx-lm prompt cache imports unavailable — prompt caching disabled"
    )

try:
    from mlx_lm.sample_utils import make_logits_processors, make_sampler
except ImportError:  # pragma: no cover
    make_sampler = None  # type: ignore[assignment]
    make_logits_processors = None  # type: ignore[assignment]
    logging.getLogger(__name__).warning(
        "mlx-lm sample_utils unavailable (mlx-lm < 0.30.7?) — sampler/logits_processors disabled"
    )

# Logits processors / decoding-output filters were extracted to a focused
# module (#454); re-exported here so existing call sites and the tests that
# import them from ``olmlx.engine.inference`` keep working.
from olmlx.engine.logits_processors import (
    _make_frequency_penalty_processor,
    _make_presence_penalty_processor,
    # Re-exported only for back-compat with tests that import them from here.
    _gpt_oss_filter as _gpt_oss_filter,
    _GPT_OSS_STRUCTURAL_TOKENS as _GPT_OSS_STRUCTURAL_TOKENS,
    _resolve_model_vocab_size as _resolve_model_vocab_size,
)

# Chat-template application + message normalization were extracted to a focused
# module (#454); re-exported here so existing call sites and the tests that
# import them from ``olmlx.engine.inference`` keep working.
from olmlx.engine.chat_templating import (
    _message_boundary_token_ids as _message_boundary_token_ids,
    _NATIVE_TOOL_HINT as _NATIVE_TOOL_HINT,
)


logger = logging.getLogger(__name__)


def _merge_default_options(defaults: dict | None, request: dict | None) -> dict:
    """Merge per-model default options with per-request options.

    Request values win per-key; keys absent from the request fall back to
    model defaults.  ``request=None`` and ``request={}`` both mean "use
    defaults"; any non-None ``request`` is layered on top of ``defaults``.
    ``defaults=None`` is accepted symmetrically with ``request`` and treated
    as an empty dict so callers that haven't normalised the field can still
    use this helper without a guard.

    History: prior versions dropped *all* defaults whenever the request
    supplied *any* options — so a request that sent ``top_k`` without
    ``temperature`` silently lost the model's default temperature and ran
    greedy (no sampler built).  Surfaced via opencode + Qwen3-Coder-Next-4bit
    where opencode sent ``{top_k, top_p, min_p}`` and ``models.json``'s
    ``"temperature": 0.7`` was discarded.  The current always-merge form
    matches Ollama's per-model options semantics.
    """
    return {**(defaults or {}), **(request or {})}


def _apply_sampling_defaults(options: dict, *, is_distributed: bool = False) -> dict:
    """Layer Ollama-parity sampling defaults *under* already-merged options.

    olmlx otherwise decodes greedily with no repetition penalty when a request
    (and the model's own defaults) leave the sampling params unset.  On weaker
    models that combination — greedy + a JSON grammar, which removes the natural
    end-of-sequence escape — walks deterministically into unbounded repetition
    and runs to ``max_tokens`` (#646).  Real Ollama applies these defaults
    server-side even when the client omits them, which prevents the
    degeneration; matching that here closes the gap for every surface that
    routes through ``generate_chat`` / ``generate_completion`` (Ollama and
    OpenAI alike).

    The defaults are layered *under* ``options`` so an explicit per-request or
    per-model value always wins — including a deliberate ``temperature=0``
    (greedy), which is a real value, not "unset".  Gated by
    ``settings.sampling_defaults_enabled`` so the historical greedy-by-default
    behaviour is one setting away.  Does not mutate *options*.

    Skipped for distributed models: ``_build_generate_kwargs`` folds these into
    a ``sampler`` callable + ``logits_processors``, which ``broadcast_inference``
    cannot ``json.dumps`` to the workers — injecting them would crash *every*
    distributed request (grammar is already rejected on that path, so the
    defaults have no benefit there anyway).
    """
    # Identity check, not truthiness (mirrors ``_batch_eligible``): tests patch
    # ``inference.settings`` with a MagicMock whose every attribute is truthy,
    # which must never inject MagicMock sampling values into ``make_sampler``.
    if is_distributed or settings.sampling_defaults_enabled is not True:
        return options
    defaults = {
        "temperature": settings.default_temperature,
        "top_p": settings.default_top_p,
        "top_k": settings.default_top_k,
        "repeat_penalty": settings.default_repeat_penalty,
        "repeat_last_n": settings.default_repeat_last_n,
    }
    return {**defaults, **options}


def _build_generate_kwargs(options: dict | None, is_vlm: bool = False) -> dict:
    """Convert Ollama options dict to mlx_lm/mlx_vlm generate kwargs.

    For text models (mlx-lm ≥ 0.30.7), sampling params are folded into a
    ``sampler`` callable via ``make_sampler``, and penalty params into a
    ``logits_processors`` list via ``make_logits_processors``.

    For VLMs (mlx-vlm), params are passed directly as before.
    """
    if not options:
        return {}
    kwargs = {}

    if is_vlm:
        # mlx-vlm still accepts direct keyword arguments
        vlm_mappings = {
            "temperature": "temperature",
            "top_p": "top_p",
            "top_k": "top_k",
            "seed": "seed",
            "num_predict": "max_tokens",
            "repeat_penalty": "repetition_penalty",
            "repeat_last_n": "repetition_context_size",
            "min_p": "min_p",
        }
        for ollama_key, mlx_key in vlm_mappings.items():
            if ollama_key in options:
                kwargs[mlx_key] = options[ollama_key]
        # Forward stop sequences for downstream (popped before passing to mlx-vlm)
        if "stop" in options and options["stop"]:
            raw = options["stop"]
            kwargs["stop"] = [raw] if isinstance(raw, str) else raw
    else:
        # mlx-lm ≥ 0.30.7: sampling via make_sampler / make_logits_processors
        sampler_args = {}
        sampling_map = {
            "temperature": "temp",
            "top_p": "top_p",
            "top_k": "top_k",
            "min_p": "min_p",
        }
        for ollama_key, sampler_key in sampling_map.items():
            if ollama_key in options:
                sampler_args[sampler_key] = options[ollama_key]
        # Only build sampler when temperature is explicitly set — make_sampler
        # defaults temp=0.0 (greedy), which makes top_k/top_p/min_p irrelevant.
        if sampler_args and "temp" in sampler_args:
            if make_sampler is None:
                raise RuntimeError("mlx-lm is not installed; cannot build sampler")
            kwargs["sampler"] = make_sampler(**sampler_args)
        elif sampler_args:
            logger.warning(
                "top_k/top_p/min_p provided without temperature; no sampler "
                "will be built and these params will have no effect"
            )

        # Collect penalty params — only build processors when repeat_penalty
        # is present; repeat_last_n alone is a no-op (no penalty to apply).
        if "repeat_penalty" in options:
            penalty_args = {"repetition_penalty": options["repeat_penalty"]}
            if "repeat_last_n" in options:
                penalty_args["repetition_context_size"] = options["repeat_last_n"]
            if make_logits_processors is None:
                raise RuntimeError(
                    "mlx-lm is not installed; cannot build logits processors"
                )
            kwargs["logits_processors"] = make_logits_processors(**penalty_args)
        elif "repeat_last_n" in options:
            logger.warning(
                "repeat_last_n without repeat_penalty has no effect; ignored"
            )

        if "num_predict" in options:
            kwargs["max_tokens"] = options["num_predict"]

        # Forward seed so _apply_seed can consume it before generation
        if "seed" in options:
            kwargs["seed"] = options["seed"]

        # Forward stop sequences for downstream (popped before passing to mlx-lm)
        if "stop" in options and options["stop"]:
            raw = options["stop"]
            kwargs["stop"] = [raw] if isinstance(raw, str) else raw

        # Build custom logits processors for frequency/presence penalty
        # and merge with any existing repeat_penalty processors.
        _existing = kwargs.pop("logits_processors", [])
        fp = options.get("frequency_penalty")
        if fp is not None and fp != 0:
            _existing.append(_make_frequency_penalty_processor(fp))
        pp = options.get("presence_penalty")
        if pp is not None and pp != 0:
            _existing.append(_make_presence_penalty_processor(pp))
        if _existing:
            kwargs["logits_processors"] = _existing

    return kwargs


def _apply_seed(kwargs: dict, *, consume: bool = True) -> None:
    """Read ``seed`` from *kwargs* and set the MLX RNG state.

    Must be called from the inference thread, not the event loop.

    Args:
        kwargs: Generate kwargs dict (may contain ``seed``).
        consume: If True, pop the key so it is not forwarded to the
                 underlying generate call (required for mlx-lm which
                 does not accept a ``seed`` kwarg).  If False, the key
                 is left in place (VLMs forward it to mlx-vlm).
    """
    seed = kwargs.pop("seed", None) if consume else kwargs.get("seed", None)
    if seed is not None:
        mx.random.seed(seed)
