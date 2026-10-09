"""KV-cache memory budget estimation and prompt tokenization helpers (extracted from inference.py)."""

import logging
from typing import TYPE_CHECKING, Any


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
from olmlx.engine.turboquant_cache import _is_plain_kv_cache


MEMORY_SAFETY_FACTOR = 1.3
"""Safety multiplier for KV cache memory estimates (Bug #125).

Metal alignment, intermediate buffers, and allocator overhead can cause actual
memory usage to exceed the raw 2-bytes-per-element calculation by 20-30%.
"""


logger = logging.getLogger(__name__)


def _parse_kv_cache_quant_kv(spec: str) -> tuple[str, int, int]:
    """Split an ``OLMLX_KV_CACHE_QUANT`` value like ``"spectral:4"`` or
    ``"kvarn:k4v2"`` (#748) into ``(method, key_bits, value_bits)``.
    Symmetric methods report the same width for K and V. Format is validated
    at config load time. ``spectral-qa`` (#749) differs from ``spectral``
    only in calibration, so it reports the ``spectral`` codec."""
    method, bits_str = spec.split(":")
    if method == "spectral-qa":
        method = "spectral"
    if method == "kvarn":
        from olmlx.config import parse_kvarn_bits

        key_bits, value_bits = parse_kvarn_bits(bits_str)
        return method, key_bits, value_bits
    return method, int(bits_str), int(bits_str)


def estimate_kv_cache_bytes(
    model: Any, num_tokens: int, *, kv_cache_quant: str | None = None
) -> int:
    """Estimate KV cache memory for a given number of tokens.

    Formula: sum_over_attn_layers(2 * kv_heads_i * head_dim) * num_tokens * bytes_per_element * MEMORY_SAFETY_FACTOR

    When *kv_cache_quant* is set (e.g. ``"turboquant:4"``), each layer's
    per-head bytes are scaled by its *live-generation* footprint relative to
    fp16 (``_quant_ratio``): packed indices + side data, plus — for the
    TurboQuant family (``turboquant``/``kvarn``) — the full-precision dequant
    side buffer they keep resident while generating. Layers whose head dim
    the codec can't take stay fp16.

    For NAS models (e.g. nemotron-nas) that have per-layer variable attention
    (some layers are no-op with self_attn=None, and KV head counts vary per
    layer), we introspect model.model.layers to count only actual attention
    layers and read their n_kv_heads.  Falls back to args-based estimation
    when layer introspection isn't possible.
    """
    if num_tokens <= 0:
        return 0

    # MLA models use a different cache layout; KV quantization does not
    # apply there (see the early return below).

    # mlx-lm text models: model.args
    # mlx-vlm vision-language models: model.language_model.args or .config
    # Wrapper args (e.g. Qwen3_5_MoE ModelArgs) carry only a ``text_config``
    # dict; the real attention fields live on ``model.language_model.args``.
    args = getattr(model, "args", None)
    args_owner: Any = model
    is_wrapper = (
        args is not None
        and hasattr(args, "text_config")
        and not hasattr(args, "num_attention_heads")
        and not hasattr(args, "kv_lora_rank")
    )
    if args is None or is_wrapper:
        lang_model = getattr(model, "language_model", None)
        inner_args = None
        if lang_model is not None:
            inner_args = getattr(lang_model, "args", None) or getattr(
                lang_model, "config", None
            )
        if inner_args is not None:
            args = inner_args
            args_owner = lang_model
        elif is_wrapper:
            # Fail loudly — otherwise we'd fall through to args.num_attention_heads
            # on the wrapper itself and crash with an opaque AttributeError.
            raise AttributeError(
                "model.args is a text_config wrapper but could not resolve "
                "inner attention config (model.language_model missing or has "
                "no 'args'/'config')"
            )
    if args is None:
        args = getattr(model, "config", None)
    if args is None:
        raise AttributeError(
            "Model has no 'args' attribute (checked model.args, "
            "model.language_model.args/config, model.config)"
        )

    # MLA (Multi-head Latent Attention) models like DeepSeek V3 compress the
    # KV cache to (kv_lora_rank + qk_rope_head_dim) per layer instead of
    # (2 * num_kv_heads * head_dim).  Detect via kv_lora_rank in model args.
    kv_lora_rank = getattr(args, "kv_lora_rank", None)
    if isinstance(kv_lora_rank, int) and kv_lora_rank > 0:
        qk_rope_head_dim = getattr(args, "qk_rope_head_dim", 0)
        num_layers = args.num_hidden_layers
        bytes_per_element = 2  # float16/bfloat16
        # MLA stores compressed_kv (kv_lora_rank dims) as keys and
        # k_pe (qk_rope_head_dim dims) as values, each with 1 effective head.
        raw = (
            num_layers
            * 2
            * (kv_lora_rank + qk_rope_head_dim)
            * num_tokens
            * bytes_per_element
        )
        return int(raw * MEMORY_SAFETY_FACTOR)

    num_heads = args.num_attention_heads
    head_dim = (
        args.head_dim if hasattr(args, "head_dim") else args.hidden_size // num_heads
    )
    bytes_per_element = 2  # float16/bfloat16

    def _quant_ratio(layer_head_dim: int) -> float:
        """Live-generation bytes for one K+V entry at ``layer_head_dim``,
        relative to fp16. Computed per layer because the cache factories
        decide per layer: a head dim the codec can't take keeps a plain fp16
        ``KVCache`` (ratio 1.0)."""
        if kv_cache_quant is None:
            return 1.0
        method, key_bits, value_bits = _parse_kv_cache_quant_kv(kv_cache_quant)
        fp16_per_entry = layer_head_dim * bytes_per_element
        if method in ("turboquant", "kvarn"):
            # TurboQuant-family caches (KVarNKVCache subclasses
            # TurboQuantKVCache) hold the packed state AND a full-precision
            # dequant side buffer for the whole history while generating —
            # shed only when the cache is stored. This estimate gates live
            # generation, so it must charge both: the packed size alone
            # admitted prompts ~5x too large, OOMing Metal mid-prefill instead
            # of a clean MemoryError/400 (#748 review). Live TurboQuant is
            # therefore slightly *above* fp16; its compression applies to the
            # stored (between-turn) cache.
            if method == "kvarn":
                from olmlx.engine.kvarn import choose_tile

                tile = choose_tile(layer_head_dim)
                if tile is None:
                    return 1.0
                side = 8 * (layer_head_dim // tile)  # f32 (mean, scale)/tile
            else:
                if layer_head_dim % (8 // key_bits) != 0:
                    return 1.0
                side = 4  # f32 norm
            k_entry = layer_head_dim // (8 // key_bits) + side + fp16_per_entry
            v_entry = layer_head_dim // (8 // value_bits) + side + fp16_per_entry
            return (k_entry + v_entry) / (2 * fp16_per_entry)
        if method == "spectral":
            # SpectralQuant: two packed regimes (semantic + tail) + float32 norm
            # Conservative estimate using avg_bits (actual varies per head).
            # Dequantizes on read — no resident side buffer.
            return (layer_head_dim // (8 // key_bits) + 4) / fp16_per_entry
        if method == "shard":
            # ShardQuant: PCA-basis-projected packed indices + float32 norm.
            # Rank truncation makes the real footprint smaller than this, so
            # the turboquant-style estimate is a safe upper bound — but still
            # far below fp16. Without it, shard-quant models were estimated at
            # full fp16 KV size, 503-ing long prompts that would actually fit
            # (#634).
            return (layer_head_dim // (8 // key_bits) + 4) / fp16_per_entry
        return 1.0

    # Try layer introspection for NAS/variable-attention/hybrid models.
    # ``args_owner`` was set above to the component whose args we resolved
    # (model.language_model for VLMs/wrappers, else model) so we introspect
    # the correct layer tree and avoid hitting a vision encoder.
    inner = getattr(args_owner, "model", None)
    layers = getattr(inner, "layers", None) if inner is not None else None
    if isinstance(layers, (list, tuple)) and len(layers) > 0:
        # Per-layer accounting for hybrid attention (e.g. Gemma 4): some
        # layers may use sliding-window attention with a different
        # n_kv_heads/head_dim and a hard cap on cache depth, while others
        # use full attention with their own dimensions.
        sliding_window = getattr(args, "sliding_window", None)
        # The cache layout the factories actually build (#762): entry ``i``
        # belongs to layer ``i``, a layer past the end owns no cache (Gemma
        # 4's KV-shared tail), and only plain ``KVCache`` entries get
        # quantized — rotating (sliding) entries and model-specific
        # subclasses (Qwen3.8's ``QSAKVCache``) stay fp16.
        layout = _default_cache_layout(args_owner)
        if layout is not None and len(layout) != len(layers):
            # The positional mapping is only verifiable for a layout that is
            # exactly one entry per layer, or a prefix whose missing tail is
            # explicitly flagged KV-shared. Anything else (a ``make_cache``
            # that skips non-attention layers) could misalign and undercount,
            # letting the preflight admit an OOM — so ignore it.
            if len(layout) > len(layers) or not all(
                _is_kv_shared_attn(getattr(layer, "self_attn", None))
                for layer in layers[len(layout) :]
            ):
                layout = None
        raw_total = 0
        found_attn_layer = False
        introspection_complete = True
        for i, layer in enumerate(layers):
            self_attn = getattr(layer, "self_attn", None)
            if self_attn is None:
                continue  # no-op attention layer — no KV cache
            # KV-shared layers (Gemma 4) read an earlier layer's cache and own
            # none.
            if _is_kv_shared_attn(self_attn):
                continue
            entry = None
            if layout is not None:
                if i >= len(layout):
                    continue
                entry = layout[i]
            layer_kv_heads = getattr(self_attn, "n_kv_heads", None)
            if not isinstance(layer_kv_heads, int):
                # Try alternate attribute name (e.g. Qwen3-Next uses
                # "num_key_value_heads" instead of "n_kv_heads")
                layer_kv_heads = getattr(self_attn, "num_key_value_heads", None)
            if not isinstance(layer_kv_heads, int):
                # Standard model — fall back to args
                introspection_complete = False
                break
            found_attn_layer = True
            # Per-layer head_dim falls back to the global head_dim if the
            # attention module doesn't expose its own as an int (most
            # uniform models).  isinstance check guards against test
            # MagicMocks auto-creating non-numeric attributes.
            attn_head_dim = getattr(self_attn, "head_dim", None)
            layer_head_dim = (
                attn_head_dim if isinstance(attn_head_dim, int) else head_dim
            )
            # Sliding-window attention: cap effective tokens at the window
            # size.  Use `is True` to avoid being fooled by truthy MagicMocks
            # in tests; production code sets a literal bool.  Prefer a
            # per-layer window if exposed (defensive — Gemma 4 today shares
            # a single window across all sliding layers via args, but a
            # future model could expose heterogeneous windows).
            is_sliding = getattr(self_attn, "is_sliding", None) is True
            layer_sw: int | None = None
            for attr in ("sliding_window_size", "sliding_window"):
                v = getattr(self_attn, attr, None)
                if isinstance(v, int) and v > 0:
                    layer_sw = v
                    break
            if layer_sw is None and isinstance(sliding_window, int):
                layer_sw = sliding_window
            rotating_size = _rotating_max_size(entry)
            if rotating_size is not None:
                # The layout is authoritative: a rotating entry is capped at
                # its own size whatever ``self_attn`` reports.
                is_sliding = True
                layer_sw = rotating_size
            if is_sliding and layer_sw is None:
                # A sliding-window layer with no resolvable window size
                # falls through to a full-prompt estimate (safe overestimate
                # — won't cause OOM, just a spurious 503 on long prompts).
                # Log so the condition is diagnosable without a debugger.
                logger.debug(
                    "Layer %d reports is_sliding=True but no window size "
                    "found on self_attn or args; using full token count for "
                    "KV estimation (safe overestimate)",
                    getattr(self_attn, "layer_idx", -1),
                )
            effective_tokens = (
                min(num_tokens, layer_sw)
                if is_sliding and layer_sw is not None
                else num_tokens
            )
            raw_total += (
                2
                * layer_kv_heads
                * layer_head_dim
                * effective_tokens
                * bytes_per_element
                * (
                    _quant_ratio(layer_head_dim)
                    if entry is None or _is_plain_kv_cache(entry)
                    else 1.0
                )
            )
        # Only trust introspection when every encountered layer reported its
        # KV heads.  found_attn_layer == False likely means the attention
        # module uses a different attribute name (e.g. "attention" instead of
        # "self_attn"); fall through to the args-based estimate in that case.
        if introspection_complete and found_attn_layer:
            return int(raw_total * MEMORY_SAFETY_FACTOR)

    # Fallback: uniform estimate from args
    num_layers = args.num_hidden_layers
    num_kv_heads = getattr(args, "num_key_value_heads", num_heads)
    raw = num_layers * 2 * num_kv_heads * head_dim * num_tokens * bytes_per_element
    return int(raw * _quant_ratio(head_dim) * MEMORY_SAFETY_FACTOR)


def _default_cache_layout(model: Any) -> list | None:
    """The model's default per-layer cache list (``model.make_cache()``),
    the layout every KV-quant factory starts from — or ``None`` when the
    model has none (mlx-lm then builds one plain ``KVCache`` per layer).

    Building it is cheap: the caches are empty until the first forward.
    A failing or non-list ``make_cache`` (e.g. a MagicMock) yields ``None``
    so the estimate falls back to per-layer introspection.
    """
    make_cache = getattr(model, "make_cache", None)
    if not callable(make_cache):
        return None
    try:
        layout = make_cache()
    except Exception:
        logger.debug("make_cache() failed; estimating from layers", exc_info=True)
        return None
    return layout if isinstance(layout, list) else None


def _is_kv_shared_attn(self_attn: Any) -> bool:
    """A KV-shared attention module (Gemma 4) reads an earlier layer's cache
    and owns none. mlx-lm marks it ``has_kv=False``, mlx-vlm
    ``is_kv_shared_layer=True``. Identity checks so MagicMock layers aren't
    mistaken for shared ones."""
    return (
        getattr(self_attn, "has_kv", None) is False
        or getattr(self_attn, "is_kv_shared_layer", None) is True
    )


def _rotating_max_size(entry: Any) -> int | None:
    """``max_size`` of a rotating (sliding-window) cache entry, else ``None``.

    Matched by class name so mlx-vlm's ``RotatingKVCache`` (which doesn't
    subclass mlx-lm's) counts too."""
    if entry is None or not any(
        cls.__name__ == "RotatingKVCache" for cls in type(entry).__mro__
    ):
        return None
    size = getattr(entry, "max_size", None)
    return size if isinstance(size, int) and size > 0 else None


def tokenize_for_cache(tokenizer: Any, prompt_text: str) -> list[int]:
    """Tokenize prompt text matching stream_generate's tokenization logic.

    Must exactly replicate the BOS heuristic in mlx_lm.generate.stream_generate
    to avoid token sequence divergence (which would cause every request to be a
    cache miss).  stream_generate uses ``bos_token is None``, NOT ``not bos_token``.
    """
    bos = getattr(tokenizer, "bos_token", None)
    add_special = bos is None or not prompt_text.startswith(bos)
    return tokenizer.encode(prompt_text, add_special_tokens=add_special)


def build_context_input_tokens(
    tokenizer: Any, prompt_text: str, context: list[int] | None
) -> list[int]:
    """Build the full input token sequence for Ollama ``/api/generate`` context.

    Tokenizes *prompt_text* (via :func:`tokenize_for_cache`, so the ids match
    what generation would produce for the string) and, when *context* is a
    non-empty prior token sequence, prepends it — the legacy Ollama
    stateless-continuation mechanism (issue #656).  A leading BOS on the fresh
    prompt is dropped when *context* is supplied so the concatenated sequence
    doesn't repeat the sequence-initial BOS (whether the BOS came from
    ``add_special_tokens`` or from a chat template that emits it as literal
    text — both surface as ``bos_token_id`` at position 0).
    """
    prompt_tokens = tokenize_for_cache(tokenizer, prompt_text)
    if not context:
        return prompt_tokens
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if bos_id is not None and prompt_tokens and prompt_tokens[0] == bos_id:
        prompt_tokens = prompt_tokens[1:]
    return list(context) + prompt_tokens


# config.json keys that declare a model's positional window, across the HF
# architectures mlx-lm/mlx-vlm serve (GPT-2 ``n_positions``, MPT
# ``max_seq_len``, ChatGLM ``seq_length``, ...).
_CONTEXT_LENGTH_KEYS = (
    "max_position_embeddings",
    "n_positions",
    "max_seq_len",
    "max_sequence_length",
    "seq_length",
    "n_ctx",
)
# RoPE-scaling types whose ``factor`` multiplies the positional window.
_WINDOW_EXTENDING_ROPE_TYPES = ("yarn", "linear", "dynamic")
# transformers reports ``int(1e30)`` for "no limit"; anything this large is
# not a real window.
_TOKENIZER_MAX_LENGTH_SENTINEL = 10_000_000


def _positive_int(value: Any) -> int | None:
    # bool is an int subclass; a ``true`` in config.json is not a length.
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, float) and value > 0 and value.is_integer():
        return int(value)
    return None


def _config_context_candidates(cfg: dict) -> list[int]:
    out: list[int] = []
    mpe = None
    for key in _CONTEXT_LENGTH_KEYS:
        v = _positive_int(cfg.get(key))
        if v is not None:
            out.append(v)
            if key == "max_position_embeddings":
                mpe = v
    # A window-extending RoPE-scaling block (``rope_parameters`` in newer
    # transformers) can push the window past ``max_position_embeddings`` —
    # Qwen2.5's recommended YaRN block keeps 32768 there and adds factor=4.
    # Other types (``llama3``, ``longrope``, ...) already report the extended
    # window in ``max_position_embeddings``; their ``factor`` isn't a window
    # multiplier (Llama 3.2: 32 x 8192 = 262144 vs a real 131072), so only the
    # extending types contribute a candidate.
    for scaling_key in ("rope_scaling", "rope_parameters"):
        scaling = cfg.get(scaling_key)
        if not isinstance(scaling, dict):
            continue
        factor = scaling.get("factor")
        if isinstance(factor, bool) or not isinstance(factor, (int, float)):
            continue
        if factor <= 1:
            continue
        rope_type = scaling.get("rope_type") or scaling.get("type")
        if rope_type not in _WINDOW_EXTENDING_ROPE_TYPES:
            continue
        original = _positive_int(scaling.get("original_max_position_embeddings"))
        if original is not None:
            out.append(int(original * factor))
        elif mpe is not None:
            out.append(int(mpe * factor))
    return out


def resolve_context_length(config: dict | None, tokenizer: Any = None) -> int | None:
    """Return the model's context window in tokens, or None when unknown.

    Considers the top-level config.json, a nested ``text_config`` (VLMs and
    multimodal wrappers), RoPE-scaling extensions, and the tokenizer's
    ``model_max_length``, returning the **largest** declared limit. The value
    backs a hard request rejection (#715), so it errs permissive: a prompt is
    only refused when it is beyond every limit the model declares.
    """
    candidates: list[int] = []
    if isinstance(config, dict):
        candidates.extend(_config_context_candidates(config))
        text_cfg = config.get("text_config")
        if isinstance(text_cfg, dict):
            candidates.extend(_config_context_candidates(text_cfg))
    tok_max = _positive_int(getattr(tokenizer, "model_max_length", None))
    if tok_max is not None and tok_max < _TOKENIZER_MAX_LENGTH_SENTINEL:
        candidates.append(tok_max)
    return max(candidates) if candidates else None
