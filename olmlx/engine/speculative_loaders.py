"""Speculative-decoder draft loading for the model manager.

Extracted from ``engine/model_manager.py`` (#454) as a mixin to shrink that
module. ``SpeculativeLoaderMixin`` holds the draft-model loaders for every
speculative strategy (classic, dflash, eagle, mtp, pld, self-speculative)
plus the draft-path resolver and vocab check. ``ModelManager`` inherits it,
so the methods are unchanged in behaviour and still bind ``self`` exactly as
before. This module must NOT import ``model_manager`` at top level — it is
imported *by* model_manager to build the class; the one shared free helper
(``_load_with_model_type_fallback``) is pulled in lazily at call time.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import mlx.core as mx

from olmlx.config import settings
from olmlx.engine.registry import SpeculativeConfig

logger = logging.getLogger(__name__)


def _quant_descriptor_from_path(model_path: Path) -> str:
    """Return a quant descriptor string for a model at *model_path*.

    Reads ``config.json`` and looks for a ``"quantization"`` or
    ``"quantization_config"`` block (the same fields ``model_manager.py``
    already uses for ``nn.quantize``). Falls back to ``quantize_config.json``
    (GPTQ format). Returns ``f"q{bits}_g{group_size}"`` or ``"bf16"`` if no
    quantization block is present.
    """
    cfg_path = model_path / "config.json"
    try:
        cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    except Exception:
        cfg = {}

    quant_block = cfg.get("quantization") or cfg.get("quantization_config")
    if not quant_block:
        gptq_path = model_path / "quantize_config.json"
        if gptq_path.exists():
            try:
                quant_block = json.loads(gptq_path.read_text())
            except Exception:
                quant_block = None

    if quant_block and isinstance(quant_block, dict):
        bits = quant_block.get("bits")
        group_size_raw = quant_block.get("group_size")
        group_size = group_size_raw if group_size_raw is not None else 64
        if bits is not None:
            return f"q{bits}_g{group_size}"

    return "bf16"


def _detect_live_quant(model: Any) -> str:
    """Return a quant descriptor by inspecting a loaded MLX model.

    Walks the model looking for the first ``nn.QuantizedLinear`` via
    ``module.modules()``. Returns ``f"q{bits}_g{group_size}"`` or ``"bf16"``
    if none are found.
    """
    import mlx.nn as nn

    try:
        for _name, module in model.named_modules():
            if isinstance(module, nn.QuantizedLinear):
                bits = module.bits
                group_size = module.group_size
                return f"q{bits}_g{group_size}"
    except Exception:
        pass
    return "bf16"


def _check_quant_compat(
    draft_quant: str | None,
    live_quant: str,
    *,
    draft_path: Path,
) -> None:
    """Warn (or raise, in strict mode) when the draft's recorded target quant
    does not match the live target's effective quantization.

    ``draft_quant=None`` means the draft checkpoint predates this field; skip
    silently (backwards compatibility).
    """
    if draft_quant is None:
        return
    if draft_quant == live_quant:
        return
    msg = (
        f"DFlash/EAGLE draft at {draft_path} was trained on a '{draft_quant}' "
        f"target but the loaded target is '{live_quant}' — acceptance rate may "
        "degrade to ~0.4%%. Retrain the draft against the current target or "
        "set OLMLX_SPEC_STRICT_COMPAT=1 to treat this as an error."
    )
    if settings.spec_strict_compat:
        raise RuntimeError(msg)
    logger.warning(msg)


def _resolve_attention_causal(dflash_cfg: dict) -> bool:
    """Detect legacy draft checkpoints that were trained with causal attention.

    DFlash draft attention switched from causal to bidirectional (mask=None)
    in v2. Checkpoints carrying ``dflash_attention_version`` >= 2 use
    bidirectional; version 1 or missing defaults to causal with a warning
    so operators know to re-train.
    """
    version = dflash_cfg.get("dflash_attention_version", 2)
    # Accept int, float, and string: JSON doesn't distinguish ``2``
    # from ``2.0`` at the wire level, and a hand-edited config might
    # store ``\"2\"``.  Convert to int so fractional values like
    # ``1.5`` are treated as v1 rather than silently misclassified —
    # version bumps are integers; fractional values are misconfigs.
    # Default to 2 (bidirectional) when the key is absent — matches
    # both z-lab pre-trained drafts and current training output.
    try:
        version_int = int(float(version))
    except (TypeError, ValueError):
        version_int = 2
    if version_int >= 2:
        return False
    logger.warning(
        "DFlash draft checkpoint was trained with causal attention "
        "(dflash_attention_version=%r → %d < 2). Re-training with the "
        "current code is recommended — running an old checkpoint "
        "produces a distribution mismatch that degrades acceptance rate.",
        version,
        version_int,
    )
    return True


class SpeculativeLoaderMixin:
    """Draft-model loading for speculative decoding (mixed into ModelManager)."""

    def _resolve_draft_path(self, hf_path: str) -> str:
        """Download a draft model if needed and return the local path.

        Accepts either a HuggingFace repo id (``"namespace/repo_name"``)
        or an *absolute* filesystem path to a local draft directory.
        Local-path short-circuiting is gated on ``is_absolute()`` to
        avoid a false positive where a valid HF repo id (e.g.
        ``"my-org/dflash-draft"``) happens to match a directory under
        the server's CWD; that would silently swap the operator's
        intended remote artifact for whatever the working directory
        contains. Without short-circuiting, feeding an absolute path
        through ``ensure_downloaded`` raises ``HFValidationError``
        ("Repo id must be in the form 'repo_name' or
        'namespace/repo_name'").
        """
        candidate = Path(hf_path).expanduser()
        if candidate.is_absolute():
            # Absolute paths are unambiguous local references — they
            # cannot be HF repo ids. If the directory is missing, fall
            # through to ``ensure_downloaded`` would surface as an
            # ``HFValidationError`` ("Repo id must be in the form
            # 'repo_name' or 'namespace/repo_name'") which is actively
            # misleading for someone who passed e.g.
            # ``/Users/.../dflash`` and made a typo or pointed at a
            # path before training finished. Raise a clear
            # ``FileNotFoundError`` with the actual path instead.
            if not candidate.is_dir():
                raise FileNotFoundError(f"Draft model directory not found: {candidate}")
            return str(candidate)
        if self.store is not None:
            local_dir = self.store.ensure_downloaded(hf_path)
            return str(local_dir)
        return hf_path

    @staticmethod
    def _check_vocab_match(
        target: Any,
        draft: Any,
        *,
        secondary_label: str = "Draft model",
        feature: str = "Speculative decoding",
    ) -> None:
        """Raise ValueError if target and secondary-model vocab sizes differ.

        ``secondary_label`` / ``feature`` tailor the error text for non-draft
        callers (e.g. proxy-tuning's expert / anti-expert) so the message names
        the right model and decode mode.
        """
        target_vocab = getattr(getattr(target, "args", None), "vocab_size", None)
        draft_vocab = getattr(getattr(draft, "args", None), "vocab_size", None)
        if target_vocab is None or draft_vocab is None:
            logger.warning(
                "Could not verify vocab compatibility: target_vocab=%s %s_vocab=%s",
                target_vocab,
                secondary_label.lower(),
                draft_vocab,
            )
            return
        if target_vocab != draft_vocab:
            raise ValueError(
                f"{secondary_label} vocab_size ({draft_vocab}) does not match "
                f"target model vocab_size ({target_vocab}). "
                f"{feature} requires matching vocabularies."
            )

    def _load_dflash_decoder(
        self,
        target_model: Any,
        spec_config: SpeculativeConfig,
    ) -> Any:
        """Load a dflash draft model and create a DFlashDecoder.

        Universal target support: no per-architecture adapter is required —
        the decoder hooks the target's selected layers in place via
        ``_patch_model``. The draft borrows ``embed_tokens`` and
        ``lm_head`` from the target via ``draft.bind(target_model)``.
        """
        from olmlx.engine.dflash.decoder import DFlashDecoder
        from olmlx.engine.dflash.draft_model import DFlashDraftModel, DraftConfig

        if not spec_config.enabled:
            raise RuntimeError(
                "_load_dflash_decoder called with spec_config.enabled=False"
            )
        if not spec_config.draft_model:
            raise ValueError(
                "speculative_strategy='dflash' requires speculative_draft_model "
                "to be set (OLMLX_SPECULATIVE_DRAFT_MODEL or per-model "
                "'speculative_draft_model' in models.json)"
            )

        logger.info("Loading dflash draft model %s", spec_config.draft_model)
        load_path = self._resolve_draft_path(spec_config.draft_model)

        config_file = Path(load_path) / "config.json"
        if not config_file.exists():
            raise FileNotFoundError(
                f"DFlash draft model config not found at {config_file}"
            )

        draft_cfg_dict = json.loads(config_file.read_text())
        dflash_cfg = draft_cfg_dict.get("dflash_config")
        if not isinstance(dflash_cfg, dict):
            raise ValueError(
                f"DFlash draft config at {config_file} is missing the "
                "'dflash_config' object (must contain 'target_layer_ids' "
                "and 'mask_token_id'). This loader expects the upstream "
                "z-lab DFlash schema."
            )
        _required_top = [
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "intermediate_size",
            "vocab_size",
            "rms_norm_eps",
            "rope_theta",
            "max_position_embeddings",
            "block_size",
        ]
        missing = [k for k in _required_top if k not in draft_cfg_dict]
        _required_dflash = ["target_layer_ids", "mask_token_id"]
        missing += [
            f"dflash_config.{k}" for k in _required_dflash if k not in dflash_cfg
        ]
        if missing:
            raise ValueError(
                f"DFlash draft config at {config_file} is missing "
                f"required keys: {missing}"
            )

        layer_types_raw = (
            draft_cfg_dict.get("layer_types")
            or ["full_attention"] * draft_cfg_dict["num_hidden_layers"]
        )

        draft_quant_raw = dflash_cfg.get("target_quant")
        draft_config = DraftConfig(
            hidden_size=draft_cfg_dict["hidden_size"],
            num_hidden_layers=draft_cfg_dict["num_hidden_layers"],
            num_attention_heads=draft_cfg_dict["num_attention_heads"],
            num_key_value_heads=draft_cfg_dict["num_key_value_heads"],
            head_dim=draft_cfg_dict["head_dim"],
            intermediate_size=draft_cfg_dict["intermediate_size"],
            vocab_size=draft_cfg_dict["vocab_size"],
            rms_norm_eps=draft_cfg_dict["rms_norm_eps"],
            rope_theta=draft_cfg_dict["rope_theta"],
            max_position_embeddings=draft_cfg_dict["max_position_embeddings"],
            block_size=draft_cfg_dict["block_size"],
            num_target_layers=len(dflash_cfg["target_layer_ids"]),
            target_layer_ids=list(dflash_cfg["target_layer_ids"]),
            mask_token_id=int(dflash_cfg["mask_token_id"]),
            rope_scaling=draft_cfg_dict.get("rope_scaling"),
            layer_types=tuple(layer_types_raw),
            sliding_window=draft_cfg_dict.get("sliding_window"),
            final_logit_softcapping=draft_cfg_dict.get("final_logit_softcapping"),
            attention_causal=_resolve_attention_causal(dflash_cfg),
            target_quant=draft_quant_raw,
        )

        draft_model = DFlashDraftModel(draft_config)
        draft_dir = Path(load_path)
        # Prefer the conventional ``model*.safetensors`` (HF/mlx-lm
        # convention, also covers sharded ``model-00001-of-N``). Only
        # fall back to ``*.safetensors`` if no conventional file is
        # present, so a co-located ``adapter_model.safetensors`` (LoRA)
        # or tokenizer projection file can't silently overwrite draft
        # weights via shared key names under ``strict=False``.
        weight_files = sorted(draft_dir.glob("model*.safetensors"))
        if not weight_files:
            weight_files = sorted(draft_dir.glob("*.safetensors"))
        if not weight_files:
            raise FileNotFoundError(
                f"DFlash draft model weights not found in {draft_dir}. "
                "A pre-trained dflash draft model is required."
            )
        weights: list[tuple[str, Any]] = []
        for wf in weight_files:
            weights.extend(mx.load(str(wf)).items())
        # ``strict=False`` permits missing keys for ``embed_tokens`` and
        # ``lm_head`` — those are bound from the target via
        # ``DFlashDraftModel.bind()`` and are intentionally absent from
        # the draft safetensors.
        draft_model.load_weights(weights, strict=False)
        logger.info(
            "Loaded dflash draft weights from %s (%d file(s))",
            draft_dir,
            len(weight_files),
        )

        # Vocab-size check: ``DFlashDraftModel.bind`` borrows the
        # target's ``embed_tokens`` / ``lm_head``, so a mismatch between
        # the draft's pre-trained vocab and the target produces an
        # ``mx.array`` shape error at the first draft forward pass —
        # surface it here at load time with a clear message instead.
        # ``DFlashDraftModel`` doesn't expose ``args.vocab_size``
        # (config lives on ``draft_config``), so we read the two
        # values directly rather than via ``_check_vocab_match``. Probe
        # the same locations ``_get_layers`` walks (top-level,
        # ``.model``, ``.language_model``) so VLM/wrapped targets that
        # don't expose ``args`` at the outer level still get the check.
        target_vocab: int | None = None
        for chain in ((), ("model",), ("language_model",), ("language_model", "model")):
            obj: Any = target_model
            for attr in chain:
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is None:
                continue
            args = getattr(obj, "args", None) or getattr(obj, "config", None)
            v = getattr(args, "vocab_size", None) if args is not None else None
            if v is not None:
                target_vocab = int(v)
                break
        if target_vocab is None:
            logger.warning(
                "Could not determine target vocab_size for DFlash draft "
                "compatibility check (target has no .args/.config at the "
                "probed locations). A mismatch will surface as an mx.array "
                "shape error at the first draft forward pass."
            )
        elif target_vocab != draft_config.vocab_size:
            raise ValueError(
                f"DFlash draft vocab_size ({draft_config.vocab_size}) does "
                f"not match target vocab_size ({target_vocab}). The draft "
                "must be trained against a target with the same vocabulary."
            )

        # Quant compatibility check: warn (or raise in strict mode) when the
        # draft was trained on a target with a different quantization.
        _check_quant_compat(
            draft_config.target_quant,
            _detect_live_quant(target_model),
            draft_path=Path(load_path),
        )

        # ``draft_config.block_size`` is treated as the *draft token
        # count* directly (matching the convention #287 ships with —
        # the value used verbatim by ``SpeculativeDecoder``). Local
        # ``olmlx dflash prepare`` writes the same convention to disk
        # so a checkpoint trained here loads back without translation.
        # ``None`` (no user override) inherits the draft's pre-trained
        # block.
        block_size = (
            spec_config.num_tokens
            if spec_config.num_tokens is not None
            else draft_config.block_size
        )
        # Going *above* the trained draft count runs the draft on block
        # lengths it has never seen; the positional encoding and
        # block-diffusion training are bound to the trained length.
        # Warn (don't fail) — users may experiment.
        if (
            spec_config.num_tokens is not None
            and spec_config.num_tokens > draft_config.block_size
        ):
            logger.warning(
                "speculative_tokens=%d exceeds the draft's pre-trained "
                "block_size=%d; output quality may degrade. Omit "
                "speculative_tokens (or pass <= %d) to stay within the "
                "trained block length.",
                spec_config.num_tokens,
                draft_config.block_size,
                draft_config.block_size,
            )
        return DFlashDecoder(
            target_model=target_model,
            draft_model=draft_model,
            draft_config=draft_config,
            block_size=block_size,
        )

    def _load_eagle_decoder(
        self,
        target_model: Any,
        spec_config: SpeculativeConfig,
    ) -> Any:
        """Load an EAGLE draft model and create an EagleDecoder.

        Mirrors ``_load_dflash_decoder`` but consumes the EAGLE saved
        schema (flat target dims plus a top-level ``eagle_config``
        block carrying ``block_size`` and ``target_layer_id``). The
        EAGLE draft has no ``mask_token_id`` or ``target_layer_ids``
        — EAGLE conditions on a single target hidden state per step,
        and ``olmlx eagle prepare`` records the chosen layer in
        ``eagle_config.target_layer_id`` (the deepest layer of the
        precomputed shard ladder). When that field is absent
        (pre-fix checkpoints), the decoder falls back to
        ``num_layers - 1`` and a ``logger.warning`` surfaces the
        misconfiguration with a nudge to retrain.
        """
        from olmlx.engine.eagle.decoder import EagleDecoder
        from olmlx.engine.eagle.draft_model import EagleConfig, EagleDraftModel

        if not spec_config.enabled:
            raise RuntimeError(
                "_load_eagle_decoder called with spec_config.enabled=False"
            )
        if not spec_config.draft_model:
            raise ValueError(
                "speculative_strategy='eagle' requires speculative_draft_model "
                "to be set (OLMLX_SPECULATIVE_DRAFT_MODEL or per-model "
                "'speculative_draft_model' in models.json)"
            )

        logger.info("Loading EAGLE draft model %s", spec_config.draft_model)
        load_path = self._resolve_draft_path(spec_config.draft_model)

        config_file = Path(load_path) / "config.json"
        if not config_file.exists():
            raise FileNotFoundError(
                f"EAGLE draft model config not found at {config_file}"
            )

        draft_cfg_dict = json.loads(config_file.read_text())
        eagle_cfg = draft_cfg_dict.get("eagle_config")
        if not isinstance(eagle_cfg, dict):
            raise ValueError(
                f"EAGLE draft config at {config_file} is missing the "
                "'eagle_config' object (must contain 'block_size'). The "
                "saved checkpoint may be a DFlash draft — pass "
                "speculative_strategy='dflash' instead, or retrain via "
                "`olmlx eagle prepare`."
            )
        _required_top = [
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "intermediate_size",
            "vocab_size",
            "rms_norm_eps",
            "rope_theta",
            "max_position_embeddings",
        ]
        missing = [k for k in _required_top if k not in draft_cfg_dict]
        if "block_size" not in eagle_cfg:
            missing.append("eagle_config.block_size")
        if missing:
            raise ValueError(
                f"EAGLE draft config at {config_file} is missing required "
                f"keys: {missing}"
            )

        eagle_draft_quant_raw = eagle_cfg.get("target_quant")
        draft_config = EagleConfig(
            hidden_size=draft_cfg_dict["hidden_size"],
            num_hidden_layers=draft_cfg_dict["num_hidden_layers"],
            num_attention_heads=draft_cfg_dict["num_attention_heads"],
            num_key_value_heads=draft_cfg_dict["num_key_value_heads"],
            head_dim=draft_cfg_dict["head_dim"],
            intermediate_size=draft_cfg_dict["intermediate_size"],
            vocab_size=draft_cfg_dict["vocab_size"],
            rms_norm_eps=draft_cfg_dict["rms_norm_eps"],
            rope_theta=draft_cfg_dict["rope_theta"],
            max_position_embeddings=draft_cfg_dict["max_position_embeddings"],
            block_size=int(eagle_cfg["block_size"]),
            rope_scaling=draft_cfg_dict.get("rope_scaling"),
            target_quant=eagle_draft_quant_raw,
        )

        draft_model = EagleDraftModel(draft_config)
        draft_dir = Path(load_path)
        # Same conventional-then-fallback search ``_load_dflash_decoder``
        # uses; comment there explains the precedence rationale.
        weight_files = sorted(draft_dir.glob("model*.safetensors"))
        if not weight_files:
            weight_files = sorted(draft_dir.glob("*.safetensors"))
        if not weight_files:
            raise FileNotFoundError(
                f"EAGLE draft model weights not found in {draft_dir}. "
                "Train one via `olmlx eagle prepare <target>`."
            )
        weights: list[tuple[str, Any]] = []
        for wf in weight_files:
            weights.extend(mx.load(str(wf)).items())
        # ``strict=False`` permits the absent ``embed_tokens`` /
        # ``lm_head`` (re-bound from target on every prefill).
        draft_model.load_weights(weights, strict=False)
        logger.info(
            "Loaded EAGLE draft weights from %s (%d file(s))",
            draft_dir,
            len(weight_files),
        )

        # Vocab-size + hidden-size cross-checks, mirroring the DFlash
        # loader so a cross-target draft surfaces here rather than at
        # the first forward pass.
        #
        # ``hidden_size`` matters because EAGLE's input projection is
        # shape ``(2 * hidden_size, hidden_size)`` — it concatenates
        # the target's hidden (shape ``hidden_size``) with the
        # embedding (shape ``hidden_size``). A draft trained against
        # Qwen3.5-7B (hidden=3584) loaded against Qwen3.5-27B
        # (hidden=5120) would pass vocab and crash with a cryptic
        # shape error inside ``input_proj`` on the first prefill.
        target_vocab: int | None = None
        target_hidden: int | None = None
        for chain in ((), ("model",), ("language_model",), ("language_model", "model")):
            obj: Any = target_model
            for attr in chain:
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is None:
                continue
            args = getattr(obj, "args", None) or getattr(obj, "config", None)
            if args is not None:
                if target_vocab is None:
                    v = getattr(args, "vocab_size", None)
                    if v is not None:
                        target_vocab = int(v)
                if target_hidden is None:
                    h = getattr(args, "hidden_size", None)
                    if h is not None:
                        target_hidden = int(h)
            if target_vocab is not None and target_hidden is not None:
                break
        if target_vocab is None:
            logger.warning(
                "Could not determine target vocab_size for EAGLE draft "
                "compatibility check. A mismatch will surface as an mx.array "
                "shape error at the first draft forward pass."
            )
        elif target_vocab != draft_config.vocab_size:
            raise ValueError(
                f"EAGLE draft vocab_size ({draft_config.vocab_size}) does "
                f"not match target vocab_size ({target_vocab}). The draft "
                "must be trained against a target with the same vocabulary."
            )
        if target_hidden is None:
            logger.warning(
                "Could not determine target hidden_size for EAGLE draft "
                "compatibility check. A mismatch will surface as an mx.array "
                "shape error inside the draft's input_proj on the first "
                "prefill."
            )
        elif target_hidden != draft_config.hidden_size:
            raise ValueError(
                f"EAGLE draft hidden_size ({draft_config.hidden_size}) does "
                f"not match target hidden_size ({target_hidden}). The draft's "
                "input_proj is shaped (2 * hidden_size, hidden_size); a "
                "mismatch would crash inside input_proj at the first "
                "prefill. Retrain the draft against the current target."
            )

        # Quant compatibility check: warn (or raise in strict mode) when the
        # draft was trained on a target with a different quantization.
        _check_quant_compat(
            draft_config.target_quant,
            _detect_live_quant(target_model),
            draft_path=Path(load_path),
        )

        block_size = (
            spec_config.num_tokens
            if spec_config.num_tokens is not None
            else draft_config.block_size
        )
        # ``target_layer_id`` (optional, recorded by ``olmlx eagle prepare``
        # at training time) tells the decoder which target layer to hook.
        # MUST match the layer the draft was trained against — feeding
        # the draft hiddens from a different layer at inference produces
        # ~5% acceptance even for an otherwise well-converged draft, since
        # mid-network and post-final-norm hiddens have very different
        # distributions. ``None`` falls back to the decoder's default
        # (last layer) — appropriate for older checkpoints from before
        # this field was recorded.
        target_layer_id_raw = eagle_cfg.get("target_layer_id")
        target_layer_id = (
            int(target_layer_id_raw) if target_layer_id_raw is not None else None
        )
        if target_layer_id is None:
            # Pre-fix checkpoints (trained before
            # ``olmlx eagle prepare`` persisted the layer ID) silently
            # fall back to ``len(layers) - 1``. If the precompute
            # captured a mid-network layer (e.g. 50 of 64), this is the
            # exact configuration that collapsed bench acceptance to
            # ~5% in the original Phase F bench — the operator gets a
            # working-looking but mis-routed checkpoint. Surface it.
            logger.warning(
                "EAGLE draft at %s has no 'target_layer_id' in its "
                "config (likely a pre-fix checkpoint). The decoder will "
                "fall back to the target's last layer; if the draft was "
                "actually trained against a mid-network layer, bench "
                "acceptance will be significantly degraded. Retrain "
                "with `olmlx eagle prepare` against the current target "
                "to get the field persisted into the saved config.",
                config_file,
            )
        return EagleDecoder(
            target_model=target_model,
            draft_model=draft_model,
            block_size=block_size,
            target_layer_id=target_layer_id,
            target_quant=draft_config.target_quant,
        )

    def _load_mtp_decoder(
        self,
        target_model: Any,
        spec_config: SpeculativeConfig,
    ) -> Any:
        """Load Qwen3.6's native MTP head (``qwen3_5_mtp``) as the draft.

        Pretrained/shipped — no training step. flash_moe exclusivity is
        enforced upstream via ``_FLASH_MOE_INCOMPATIBLE_STRATEGIES``.
        """
        from olmlx.engine.mtp.decoder import MTPDecoder
        from olmlx.engine.mtp.draft_model import MTPConfig, load_mtp_draft

        if not spec_config.enabled:
            raise RuntimeError(
                "_load_mtp_decoder called with spec_config.enabled=False"
            )
        if not spec_config.draft_model:
            raise ValueError(
                "speculative_strategy='mtp' requires speculative_draft_model "
                "to point at the MTP head repo (e.g. "
                "mlx-community/Qwen3.6-27B-MTP-4bit)."
            )
        load_path = self._resolve_draft_path(spec_config.draft_model)
        cfg_dict = json.loads((Path(load_path) / "config.json").read_text())
        if cfg_dict.get("model_type") != "qwen3_5_mtp":
            raise ValueError(
                f"Expected an MTP head (model_type 'qwen3_5_mtp'); got "
                f"'{cfg_dict.get('model_type')}' at {spec_config.draft_model}."
            )
        cfg = MTPConfig.from_dict(cfg_dict)
        draft = load_mtp_draft(load_path, cfg)
        self._check_vocab_match(target_model, draft)
        block_size = (
            spec_config.num_tokens
            if spec_config.num_tokens is not None
            else cfg.block_size
        )
        return MTPDecoder(target_model, draft, block_size=block_size)

    def _load_speculative_decoder(
        self,
        target_model: Any,
        hf_path: str,
        spec_config: SpeculativeConfig,
        *,
        is_vlm: bool = False,
    ) -> Any:
        """Load a draft model and create a SpeculativeDecoder.

        For VLM targets (``is_vlm=True``), the decoder runs on the unwrapped
        language model (``target_model.language_model``) so the draft only
        needs to match the text decoder's vocabulary and the speculative loop
        can call the language model directly with token inputs and a KV cache.
        """
        from olmlx.engine.speculative import SpeculativeDecoder

        # Hard guard rather than assert — assert is elided under
        # ``python -O``, and this invariant must hold in production too.
        if not spec_config.enabled:
            raise RuntimeError(
                "_load_speculative_decoder called with spec_config.enabled=False"
            )
        draft_model_path = spec_config.draft_model
        # ``None`` means "no user override"; classic speculative decoding
        # uses 4 as its strategy default.
        num_tokens = spec_config.num_tokens if spec_config.num_tokens is not None else 4
        if not draft_model_path:
            raise ValueError(
                "speculative requires speculative_draft_model to be set "
                "(OLMLX_SPECULATIVE_DRAFT_MODEL or per-model "
                "'speculative_draft_model' in models.json)"
            )

        logger.info(
            "Loading draft model %s for speculative decoding",
            draft_model_path,
        )
        load_path = self._resolve_draft_path(draft_model_path)

        import mlx_lm

        # Imported at call time to avoid a circular import — this module is
        # imported by model_manager to build ModelManager (#454).
        from olmlx.engine.model_manager import _load_with_model_type_fallback

        draft_model, _draft_tokenizer = _load_with_model_type_fallback(
            mlx_lm, load_path, lazy=False
        )

        if is_vlm:
            spec_target = getattr(target_model, "language_model", None)
            if spec_target is None:
                raise ValueError(
                    "VLM model does not expose .language_model; speculative "
                    "decoding requires direct access to the text decoder"
                )
        else:
            spec_target = target_model
        self._check_vocab_match(spec_target, draft_model)

        return SpeculativeDecoder(
            draft_model=draft_model,
            target_model=spec_target,
            num_speculative_tokens=num_tokens,
            tree_width=settings.tree_width if settings.tree_speculative else 1,
            tree_max_nodes=settings.tree_max_nodes,
            cache_slots=settings.speculative_cache_slots,
        )

    def _load_pld_decoder(
        self,
        target_model: Any,
        spec_config: SpeculativeConfig,
        *,
        is_vlm: bool = False,
    ) -> Any:
        """Construct a PromptLookupDecoder (no draft model required).

        For VLM targets, decoder runs on the unwrapped language model so
        the prompt-cache state is the same one mlx-vlm's generate would
        touch. All PLD knobs (max-draft, ngram range, lookup window) are
        read from ``spec_config`` so per-model ``models.json`` overrides
        compose with the global ``OLMLX_SPECULATIVE_PLD_*`` env vars
        (``ModelConfig.resolved_speculative`` handles the fallback chain).
        """
        from olmlx.engine.speculative import PromptLookupDecoder

        if not spec_config.enabled:
            raise RuntimeError(
                "_load_pld_decoder called with spec_config.enabled=False"
            )
        # PLD default max-draft is 10 (Saxena's reference value); classic
        # speculative defaults to 4 but the regime is different — PLD's
        # per-step compute is dominated by the target forward whose cost
        # scales sub-linearly with draft length up to a point.
        num_tokens = (
            spec_config.num_tokens if spec_config.num_tokens is not None else 10
        )
        if spec_config.draft_model:
            logger.warning(
                "speculative_strategy='pld' ignores speculative_draft_model "
                "(%s) — PLD has no draft model.",
                spec_config.draft_model,
            )

        if is_vlm:
            pld_target = getattr(target_model, "language_model", None)
            if pld_target is None:
                raise ValueError(
                    "VLM model does not expose .language_model; PLD "
                    "requires direct access to the text decoder"
                )
        else:
            pld_target = target_model

        # ``resolved_speculative`` populates these from the global
        # Settings defaults (3, 1, 8192) when no per-model override is
        # present, so they are never None at this point in normal use.
        # Use explicit ``raise`` rather than ``assert`` so the misuse
        # also surfaces under ``python -O`` (which would otherwise
        # strip the check and let ``None`` fall through to
        # ``PromptLookupDecoder.__init__`` with a confusing ``TypeError``
        # on the ``<`` comparison there).
        if spec_config.pld_max_ngram is None:
            raise ValueError(
                "_load_pld_decoder: spec_config.pld_max_ngram is None; "
                "caller must go through ModelConfig.resolved_speculative()"
            )
        if spec_config.pld_min_ngram is None:
            raise ValueError(
                "_load_pld_decoder: spec_config.pld_min_ngram is None; "
                "caller must go through ModelConfig.resolved_speculative()"
            )
        if spec_config.pld_lookup_window is None:
            raise ValueError(
                "_load_pld_decoder: spec_config.pld_lookup_window is None; "
                "caller must go through ModelConfig.resolved_speculative()"
            )
        logger.info(
            "Constructing PLD decoder (max_draft=%d, ngram=%d..%d, lookup_window=%d)",
            num_tokens,
            spec_config.pld_min_ngram,
            spec_config.pld_max_ngram,
            spec_config.pld_lookup_window,
        )
        return PromptLookupDecoder(
            target_model=pld_target,
            num_speculative_tokens=num_tokens,
            max_ngram_size=spec_config.pld_max_ngram,
            min_ngram_size=spec_config.pld_min_ngram,
            lookup_window=spec_config.pld_lookup_window,
            cache_slots=settings.speculative_cache_slots,
        )

    def _load_self_speculative_decoder(
        self,
        target_model: Any,
        spec_config: SpeculativeConfig,
    ) -> Any:
        """Create a SelfSpeculativeDecoder using the target's own early layers.

        No external draft model is loaded. ``spec_config.layers_skip``
        determines how many layers the draft skips (defaulting to
        ``L // 4`` when ``None``).
        """
        from olmlx.engine.gdn_rollback import get_model_layers
        from olmlx.engine.self_speculative import SelfSpeculativeDecoder

        num_tokens = spec_config.num_tokens if spec_config.num_tokens is not None else 4

        total_layers = len(get_model_layers(target_model))
        if spec_config.layers_skip is not None:
            layers_skip = spec_config.layers_skip
        else:
            layers_skip = max(total_layers // 4, 1)
        num_early_layers = total_layers - layers_skip
        if num_early_layers < 1:
            num_early_layers = 1
            layers_skip = total_layers - 1

        logger.info(
            "Self-speculative: draft uses %d/%d layers (skip=%d, λ=%d)",
            num_early_layers,
            total_layers,
            layers_skip,
            num_tokens,
        )

        return SelfSpeculativeDecoder(
            target_model=target_model,
            num_early_layers=num_early_layers,
            num_speculative_tokens=num_tokens,
        )

    def _load_proxy_tuning_decoder(
        self,
        target_model: Any,
        target_tokenizer: Any,
        spec_config: SpeculativeConfig,
    ) -> Any:
        """Load expert + anti-expert and build a ProxyTuningDecoder.

        The base model is the already-loaded ``target_model``; the small expert
        (``M+``) and anti-expert (``M-``) are loaded inline here via mlx-lm —
        the same pattern ``_load_speculative_decoder`` uses for the draft model.
        They are held by the returned decoder (not registered in the model
        manager), so they coexist with the base for the decoder's lifetime
        without a ``max_loaded_models`` bump.

        All three models must share one exact tokenizer/vocabulary: we hard-fail
        on a ``vocab_size`` mismatch and additionally verify token-mapping
        identity via the tokenizers when available.
        """
        from olmlx.engine.proxy_tuning import ProxyTuningDecoder, check_vocab_identity

        if not spec_config.enabled:
            raise RuntimeError(
                "_load_proxy_tuning_decoder called with spec_config.enabled=False"
            )
        expert_path = spec_config.proxy_expert_model
        antiexpert_path = spec_config.proxy_antiexpert_model
        if not expert_path or not antiexpert_path:
            raise ValueError(
                "speculative_strategy='proxy_tuning' requires both "
                "speculative_proxy_expert_model and "
                "speculative_proxy_antiexpert_model to be set "
                "(OLMLX_SPECULATIVE_PROXY_EXPERT_MODEL / "
                "OLMLX_SPECULATIVE_PROXY_ANTIEXPERT_MODEL)."
            )

        import mlx_lm

        # Imported at call time to avoid the circular import (this module is
        # imported by model_manager to build ModelManager).
        from olmlx.engine.model_manager import _load_with_model_type_fallback

        logger.info(
            "Loading proxy-tuning expert %s and anti-expert %s",
            expert_path,
            antiexpert_path,
        )
        expert_load_path = self._resolve_draft_path(expert_path)
        antiexpert_load_path = self._resolve_draft_path(antiexpert_path)
        expert_model, expert_tokenizer = _load_with_model_type_fallback(
            mlx_lm, expert_load_path, lazy=False
        )
        antiexpert_model, antiexpert_tokenizer = _load_with_model_type_fallback(
            mlx_lm, antiexpert_load_path, lazy=False
        )

        # Hard floor: integer vocab_size must match across all three models.
        self._check_vocab_match(
            target_model,
            expert_model,
            secondary_label="Expert model",
            feature="Proxy-tuning",
        )
        self._check_vocab_match(
            target_model,
            antiexpert_model,
            secondary_label="Anti-expert model",
            feature="Proxy-tuning",
        )
        # Stronger check: token->id mapping identity (catches same-size,
        # different-mapping vocabularies that the size check misses).
        check_vocab_identity(
            target_tokenizer,
            expert_tokenizer,
            reference_label="base",
            other_label="expert",
        )
        check_vocab_identity(
            target_tokenizer,
            antiexpert_tokenizer,
            reference_label="base",
            other_label="anti-expert",
        )

        return ProxyTuningDecoder(
            base_model=target_model,
            expert_model=expert_model,
            antiexpert_model=antiexpert_model,
            alpha=spec_config.proxy_alpha,
        )
