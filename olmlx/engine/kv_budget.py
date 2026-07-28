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


MEMORY_SAFETY_FACTOR = 1.3
"""Safety multiplier for KV cache memory estimates (Bug #125).

Metal alignment, intermediate buffers, and allocator overhead can cause actual
memory usage to exceed the raw 2-bytes-per-element calculation by 20-30%.
"""


logger = logging.getLogger(__name__)


def _parse_kv_cache_quant(spec: str) -> tuple[str, int]:
    """Split an `OLMLX_KV_CACHE_QUANT` value like `"spectral:4"`
    into `(method, bits)`.  Format is validated at config load time."""
    method, bits_str = spec.split(":")
    return method, int(bits_str)


def estimate_kv_cache_bytes(
    model: Any, num_tokens: int, *, kv_cache_quant: str | None = None
) -> int:
    """Estimate KV cache memory for a given number of tokens.

    Formula: sum_over_attn_layers(2 * kv_heads_i * head_dim) * num_tokens * bytes_per_element * MEMORY_SAFETY_FACTOR

    When *kv_cache_quant* is set (e.g. ``"turboquant:4"``), the per-head
    storage is reduced from ``head_dim * 2`` bytes (fp16) to the compressed
    size: ``head_dim / (8 / bits)`` packed-index bytes + 4 norm bytes.

    For NAS models (e.g. nemotron-nas) that have per-layer variable attention
    (some layers are no-op with self_attn=None, and KV head counts vary per
    layer), we introspect model.model.layers to count only actual attention
    layers and read their n_kv_heads.  Falls back to args-based estimation
    when layer introspection isn't possible.
    """
    if num_tokens <= 0:
        return 0

    # TurboQuant compression ratio.  Normal fp16 stores head_dim * 2 bytes
    # per K or V entry.  TurboQuant stores head_dim/(8/bits) packed-index
    # bytes + 4 float32 norm bytes.  We compute a multiplier <1 to scale
    # the fp16 estimate.  MLA models use a different cache layout so
    # TurboQuant does not apply there (ratio stays 1.0).
    _tq_ratio = 1.0  # applied to raw estimate before safety factor

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

    if kv_cache_quant is not None:
        method, quant_bits = _parse_kv_cache_quant(kv_cache_quant)
        fp16_per_entry = head_dim * bytes_per_element
        if method == "turboquant":
            # TurboQuant: packed indices + float32 norm
            tq_per_entry = head_dim // (8 // quant_bits) + 4  # 4 bytes for f32 norm
            _tq_ratio = tq_per_entry / fp16_per_entry
        elif method == "spectral":
            # SpectralQuant: two packed regimes (semantic + tail) + float32 norm
            # Conservative estimate using avg_bits (actual varies per head)
            sq_per_entry = head_dim // (8 // quant_bits) + 4
            _tq_ratio = sq_per_entry / fp16_per_entry
        elif method == "shard":
            # ShardQuant: PCA-basis-projected packed indices + float32 norm.
            # Rank truncation makes the real footprint smaller than this, so
            # the turboquant-style estimate is a safe upper bound — but still
            # far below fp16. Without it, shard-quant models were estimated at
            # full fp16 KV size, 503-ing long prompts that would actually fit
            # (#634).
            shard_per_entry = head_dim // (8 // quant_bits) + 4
            _tq_ratio = shard_per_entry / fp16_per_entry

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
        raw_total = 0
        found_attn_layer = False
        introspection_complete = True
        for layer in layers:
            self_attn = getattr(layer, "self_attn", None)
            if self_attn is None:
                continue  # no-op attention layer — no KV cache
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
            )
        # Only trust introspection when every encountered layer reported its
        # KV heads.  found_attn_layer == False likely means the attention
        # module uses a different attribute name (e.g. "attention" instead of
        # "self_attn"); fall through to the args-based estimate in that case.
        if introspection_complete and found_attn_layer:
            return int(raw_total * _tq_ratio * MEMORY_SAFETY_FACTOR)

    # Fallback: uniform estimate from args
    num_layers = args.num_hidden_layers
    num_kv_heads = getattr(args, "num_key_value_heads", num_heads)
    raw = num_layers * 2 * num_kv_heads * head_dim * num_tokens * bytes_per_element
    return int(raw * _tq_ratio * MEMORY_SAFETY_FACTOR)


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
