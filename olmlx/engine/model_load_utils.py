"""Model loading utilities shared by ModelManager (extracted from model_manager.py)."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlx.core as mx
import mlx.nn as nn

from olmlx.engine.prompt_cache import CachedPromptState, PromptCacheStore  # noqa: F401

# Speculative-decoder draft loaders live in a focused mixin module (#454).
# ``_resolve_attention_causal`` is re-exported for back-compat with tests that
# import it from here.
from olmlx.engine.speculative_loaders import (
    _resolve_attention_causal as _resolve_attention_causal,
)

if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


def _sanitize_model_config_in_place(load_path) -> None:
    """Fix known on-disk config.json issues that block transformers loading.

    Currently handles: ``layer_types`` longer than ``num_hidden_layers``.
    Step-3.5 ships with ``num_hidden_layers=45`` but ``len(layer_types)=48``
    (the trailing 3 entries describe MTP layers whose weights mlx-lm's
    sanitize() drops). Newer transformers' ``validate_layer_type`` rejects
    the mismatch. Truncating ``layer_types`` to match is consistent with
    what mlx-lm does for the weights and is idempotent on subsequent loads.
    """
    config_file = Path(load_path) / "config.json"
    if not config_file.exists():
        return
    try:
        cfg = json.loads(config_file.read_text())
    except json.JSONDecodeError:
        return

    nhl = cfg.get("num_hidden_layers")
    layer_types = cfg.get("layer_types")
    if (
        isinstance(layer_types, list)
        and isinstance(nhl, int)
        and nhl > 0
        and len(layer_types) > nhl
    ):
        logger.info(
            "Truncating layer_types from %d to %d entries in %s "
            "(num_hidden_layers); excess entries describe layers mlx-lm drops "
            "(e.g. MTP).",
            len(layer_types),
            nhl,
            config_file,
        )
        cfg["layer_types"] = layer_types[:nhl]
        try:
            config_file.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
        except OSError:
            logger.warning(
                "Failed to write sanitized layer_types to %s "
                "(read-only?); model may fail to load if transformers validates "
                "the mismatch.",
                config_file,
            )


def _ensure_tokenizer_eos_in_stops(tokenizer: Any) -> None:
    """Add the tokenizer's own ``eos_token_id`` to its stop-token set.

    Workaround for repos (e.g. ``mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit``,
    issue #308) whose ``config.json`` declares ``eos_token_id`` as a token
    different from the chat template's real end-of-turn token (the
    ``eos_token`` field in ``tokenizer_config.json``). mlx-lm's ``load()``
    feeds only the config.json value into ``TokenizerWrapper.eos_token_ids``,
    so generation does not stop at the template's actual EOT and the EOT
    string leaks into the decoded response.
    """
    # ``add_eos_token`` is the TokenizerWrapper marker — keep the gate on it
    # so plain HF tokenizers and mlx-vlm processors are skipped — but mutate
    # the ``eos_token_ids`` set directly to bypass the wrapper's stringly-typed
    # API (``add_eos_token(str)`` does ``int(token)`` first then a vocab
    # lookup; we already have the integer so set-mutation is unambiguous).
    if not callable(getattr(tokenizer, "add_eos_token", None)):
        return
    stops = getattr(tokenizer, "eos_token_ids", None)
    if not isinstance(stops, set):
        # Symmetric with the missing-_tokenizer DEBUG log below: an mlx-lm
        # change from set to list/frozenset/dict for eos_token_ids would
        # otherwise silently disable the workaround.
        logger.debug(
            "TokenizerWrapper.eos_token_ids is %s (not set) on %s; eos "
            "stop-set augmentation skipped (issue #308).",
            type(stops).__name__,
            type(tokenizer).__name__,
        )
        return
    inner_tok = getattr(tokenizer, "_tokenizer", None)
    if inner_tok is None:
        # Past the ``add_eos_token`` gate but no ``_tokenizer`` attribute —
        # mlx-lm likely renamed the field. Log at debug so the regression is
        # discoverable in DEBUG-enabled environments without spamming the
        # warning channel for a possibly-deliberate variant.
        logger.debug(
            "TokenizerWrapper._tokenizer not accessible on %s; eos stop-set "
            "augmentation skipped (issue #308).",
            type(tokenizer).__name__,
        )
        return
    inner_eos = getattr(inner_tok, "eos_token_id", None)
    if isinstance(inner_eos, list):
        # Stock HF tokenizers always surface a single int here; defensive
        # against custom trust_remote_code=True tokenizers that override
        # ``eos_token_id`` to return list[int].
        stops.update(t for t in inner_eos if isinstance(t, int))
        return
    if inner_eos is None:
        # Tokenizer has no EOS configured — legitimate for some HF tokenizers
        # (e.g. base/non-instruction-tuned variants). Silent no-op.
        return
    if not isinstance(inner_eos, int):
        # mlx-lm renamed ``_tokenizer`` or the inner HF tokenizer surfaces an
        # unexpected type. ``warning``, not ``debug``: this branch indicates
        # the #308 workaround has silently regressed because mlx-lm changed
        # its internals, and we recover by no-op'ing — operators need to see
        # the signal in default logging configs, not only under DEBUG.
        logger.warning(
            "Inner eos_token_id has unexpected type %s on %s; skipping "
            "eos stop-set augmentation (issue #308 workaround may have "
            "regressed against mlx-lm internals).",
            type(inner_eos).__name__,
            type(tokenizer).__name__,
        )
        return
    if inner_eos in stops:
        return
    stops.add(inner_eos)


def _materialize_module_buffers(model: Any) -> None:
    """Eager-eval a model's non-parameter array buffers on the current thread.

    mlx-lm's ``load`` materializes ``mx.eval(model.parameters())``, but
    ``nn.Module.parameters()`` skips **underscore-prefixed** dict keys (the
    same traversal gap behind the flash-MoE #657 crash). Scaled-RoPE variants
    — Yarn, longrope, llama3, proportional — precompute ``self._freqs`` (and
    ``self._scale``) as *lazy* arrays in ``__init__``, so they are never
    reached by the load-time parameter eval and stay bound to the load
    thread's Metal stream.

    Under mlx >= 0.31.2 thread-local streams (#499) the model is loaded on one
    ``asyncio``/generation worker thread and generated on another, so the first
    forward that evaluates a graph referencing the lazy ``_freqs`` raises
    "There is no Stream(gpu, N) in current thread" — deterministically, on
    every request. This surfaces on long-context models whose RoPE is scaled
    (e.g. the 1M-context Qwen3.5 hybrid ``empero-ai/Qwythos-9B``); models on
    the default/mrope RoPE carry no ``_freqs`` buffer and are unaffected.

    Materializing the buffers here, on the load thread, turns them into leaves
    (which carry no stream binding) so they can be read from any worker thread.
    ``nn.Module`` is a ``dict`` subclass whose attributes are stored as items,
    so we walk ``named_modules()`` and collect the underscore-keyed array
    leaves (recursing into plain list/tuple/dict containers, but not into
    child ``nn.Module``s — those are visited by ``named_modules()`` in turn).
    """
    buffers: list[mx.array] = []

    def _collect(value: Any) -> None:
        if isinstance(value, nn.Module):
            return  # visited independently by named_modules()
        if isinstance(value, mx.array):
            buffers.append(value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                _collect(item)
        elif isinstance(value, dict):
            for item in value.values():
                _collect(item)

    try:
        modules = model.named_modules()
    except AttributeError:
        return  # not an nn.Module tree (e.g. a test double) — nothing to do
    for _name, module in modules:
        for key, value in module.items():
            if key.startswith("_"):
                _collect(value)
    if buffers:
        mx.eval(buffers)


def _load_with_model_type_fallback(mlx_lm, load_path, **kwargs):
    """Load model + tokenizer, remapping unrecognised model_type if needed.

    Some very new models (e.g. DeepSeek V3.2 with model_type "deepseek_v32")
    aren't in the installed transformers' CONFIG_MAPPING yet, causing
    ``PreTrainedConfig.__post_init__`` to crash during tokenizer loading.

    The model architecture (loaded by mlx-lm's own registry) is fine — only
    the tokenizer loading via ``AutoTokenizer`` / ``PreTrainedConfig`` fails.
    So we load the model first with the real config, then patch config.json
    temporarily to load only the tokenizer.
    """
    # mlx-community 'gemma4_unified' text checkpoints (e.g. gemma-4-12B-it-4bit)
    # need a dedicated loader: their model_type has no mlx-lm module and their
    # multimodal weights must be dropped to load the language tower. The loader
    # materializes weights eagerly and ignores kwargs like ``lazy`` — these
    # checkpoints are dense text towers, so the lazy flash-MoE caller never
    # routes here.
    gemma4_unified = _maybe_load_gemma4_unified_text(str(load_path))
    if gemma4_unified is not None:
        return gemma4_unified

    _sanitize_model_config_in_place(load_path)
    kwargs.setdefault("tokenizer_config", {"trust_remote_code": True})
    try:
        model, tokenizer = mlx_lm.load(str(load_path), **kwargs)
    except (AttributeError, ValueError, KeyError) as exc:
        config_file = Path(load_path) / "config.json"
        if not config_file.exists():
            raise
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        original_text = config_file.read_text()
        cfg = json.loads(original_text)
        mt = cfg.get("model_type", "")
        # Strip last digit: deepseek_v32 → deepseek_v3
        fallback = re.sub(r"\d+$", lambda m: m.group()[:-1], mt) if mt else ""
        if not fallback or fallback == mt or fallback not in CONFIG_MAPPING:
            raise

        logger.info(
            "Loading model with real config, tokenizer with %r -> %r (%s)",
            mt,
            fallback,
            exc,
        )
        # Load model with the real config (mlx-lm knows the architecture).
        model, model_cfg = mlx_lm.utils.load_model(
            Path(load_path),
            **{k: v for k, v in kwargs.items() if k in ("lazy", "model_config")},
        )
        # Temporarily patch config.json for tokenizer loading only.
        try:
            cfg["model_type"] = fallback
            config_file.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
            tokenizer = mlx_lm.utils.load_tokenizer(
                Path(load_path),
                kwargs.get("tokenizer_config"),
                eos_token_ids=model_cfg.get("eos_token_id", None),
            )
        finally:
            config_file.write_text(original_text)
    # Outside try/except: an AttributeError from the EOS helper must not
    # trigger the model-type remapping fallback with a misleading error.
    _ensure_tokenizer_eos_in_stops(tokenizer)
    # Materialize non-parameter buffers (scaled-RoPE ``_freqs``, ...) on THIS
    # (load) thread — mlx-lm's ``mx.eval(model.parameters())`` skips them, and
    # left lazy they crash when first evaluated on the generation worker thread
    # (#499 thread-local streams). Covers both the mlx-lm happy path and the
    # config-remap fallback above.
    _materialize_module_buffers(model)
    return model, tokenizer


def _quantize_language_tower(
    model: Any, quant: dict, lang_weights: dict, prefix: str = "language_model."
) -> None:
    """Quantize a rebuilt ``gemma4_text`` language tower in place, honoring the
    checkpoint's *per-layer* quantization overrides.

    mlx-community ``gemma4_unified`` checkpoints carry mixed precision: a global
    ``bits``/``group_size`` plus per-module overrides (e.g. MLP projections at
    8-bit) keyed by the full ``language_model.*`` path.  A blanket
    ``nn.quantize`` at the global bits builds QuantizedLinear params whose packed
    shapes mismatch the stored 8-bit weights, so the strict load fails.  This
    mirrors mlx-lm's own ``load_model`` predicate, mapping our stripped tower
    paths (``model.layers.0.mlp.gate_proj``) back to the prefixed config keys.
    """
    import mlx.nn as nn

    def class_predicate(path: str, module: Any) -> bool | dict:
        # Per-layer override wins: config keys retain the language_model prefix.
        override = quant.get(f"{prefix}{path}")
        if isinstance(override, dict):
            return override
        if not hasattr(module, "to_quantized"):
            return False
        # Otherwise quantize at the global bits only if the checkpoint stored
        # this module as quantized (a non-quantized module has no .scales).
        return f"{path}.scales" in lang_weights

    nn.quantize(
        model,
        group_size=quant.get("group_size", 64),
        bits=quant.get("bits", 4),
        mode=quant.get("mode", "affine"),
        class_predicate=class_predicate,
    )


def _load_gemma4_unified_text(load_path: str, config: dict) -> tuple[Any, Any]:
    """Load only the *language tower* of a mlx-community ``gemma4_unified``
    checkpoint (e.g. ``gemma-4-12B-it-4bit``) via mlx-lm's ``gemma4_text``.

    These repos ship the full multimodal checkpoint under a non-standard
    ``gemma4_unified`` model_type whose vision tower is stored under a
    ``vision_embedder.*`` prefix.  That layout matches neither mlx-lm's
    ``gemma4_text`` module (it wants the language tower under ``model.*``) nor
    mlx-vlm 0.4.4's ``gemma4`` module (it models the vision tower as
    ``vision_tower`` + ``embed_vision``, so the 11 ``vision_embedder.*`` params
    have no home).  For text inference we build ``gemma4_text`` from
    ``text_config``, load the ``language_model.*`` weights (stripping the
    prefix), and drop the multimodal weights.  Standard ``gemma4`` checkpoints
    (e4b/31b) are unaffected and keep their normal mlx-vlm routing.
    """
    import glob

    import mlx.core as mx
    import mlx_lm
    from mlx_lm.models import gemma4_text

    text_cfg = dict(config.get("text_config", {}))
    # gemma4_text's own module_type; the checkpoint carries
    # "gemma4_unified_text" which the mlx-lm ModelArgs doesn't recognise.
    text_cfg["model_type"] = "gemma4_text"
    args = gemma4_text.ModelArgs.from_dict(text_cfg)
    model = gemma4_text.Model(args)

    weights: dict[str, Any] = {}
    for shard in sorted(glob.glob(str(Path(load_path) / "*.safetensors"))):
        # mx.load is typed Union[array, dict, tuple]; the dict case is what
        # safetensors returns. Same suppression as pre_shard.py / flash loaders.
        weights.update(mx.load(shard))  # pyright: ignore[reportCallIssue]
    # Keep only the language tower; the multimodal weights (vision_embedder,
    # embed_vision, embed_audio) are intentionally dropped for the text path.
    prefix = "language_model."
    lang_weights = {
        k[len(prefix) :]: v for k, v in weights.items() if k.startswith(prefix)
    }
    if not lang_weights:
        raise ValueError(
            f"gemma4_unified checkpoint at {load_path} has no "
            f"'{prefix}*' weights; cannot load the language tower."
        )
    lang_weights = model.sanitize(lang_weights)

    # Quantize *before* loading so the packed param shapes match the stored
    # weights.  These checkpoints carry mixed precision (a global bits plus
    # per-module 8-bit overrides), so quantize from the per-layer config rather
    # than a single blanket bits — sanitize first to mirror mlx-lm's order.
    quant = config.get("quantization") or config.get("quantization_config")
    if quant:
        _quantize_language_tower(model, quant, lang_weights, prefix)

    model.load_weights(list(lang_weights.items()), strict=True)
    mx.eval(model.parameters())

    # Thread the full eos_token_id list into the tokenizer's stop set.  Gemma 4
    # terminates turns with ``<turn|>`` (id 106) and pauses for tool results at
    # ``<|tool_response>`` (id 50) — both are stop tokens alongside ``<eos>``
    # (id 1).  Without them generation runs past every turn/tool boundary and
    # degenerates into a repetition loop.  generation_config.json wins over
    # config.json (mirrors mlx-lm's load_config precedence).
    eos_token_ids = config.get("eos_token_id")
    gen_config_file = Path(load_path) / "generation_config.json"
    if gen_config_file.exists():
        try:
            gen_config = json.loads(gen_config_file.read_text())
        except (OSError, ValueError):
            gen_config = {}
        if gen_config.get("eos_token_id") is not None:
            eos_token_ids = gen_config["eos_token_id"]

    tokenizer = mlx_lm.utils.load_tokenizer(
        Path(load_path),
        {"trust_remote_code": True},
        eos_token_ids=eos_token_ids,
    )
    _ensure_tokenizer_eos_in_stops(tokenizer)
    return model, tokenizer


def _maybe_load_gemma4_unified_text(load_path: str) -> tuple[Any, Any] | None:
    """Return ``(model, tokenizer)`` for a text-routable ``gemma4_unified``
    checkpoint, or ``None`` if this isn't one (so callers fall through to the
    normal mlx-lm load).  Reads config.json only — never mutates it.
    """
    config_file = Path(load_path) / "config.json"
    if not config_file.exists():
        return None
    try:
        config = json.loads(config_file.read_text())
    except (OSError, ValueError):
        return None
    if config.get("model_type", "").lower() != "gemma4_unified":
        return None
    text_model_type = config.get("text_config", {}).get("model_type", "").lower()
    if text_model_type != "gemma4_unified_text":
        return None
    return _load_gemma4_unified_text(load_path, config)
