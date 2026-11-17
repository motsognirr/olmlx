"""Load-time wiring for converted Edge0 packs.

A converted pack (see :mod:`olmlx.engine.edge0.convert`) loads through the
ordinary Flash-MoE path: mlx-vlm builds the ``qwen4_exp`` model from the
trunk safetensors, then :func:`wrap_edge0_latent` (dispatched from
``wrap_flash_moe``) swaps every layer's routed experts for a
:class:`~olmlx.engine.edge0.latent_moe.LatentMoE` reading the pack's own
``experts.bin``.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx.utils import tree_flatten

from olmlx.engine.edge0.convert import LATENT_DIRNAME, read_edge0_config
from olmlx.engine.edge0.latent_moe import LatentExpertStore, LatentMoE
from olmlx.engine.edge0.pack import Edge0Pack, Edge0PackError

logger = logging.getLogger(__name__)

# Parameters a converted pack legitimately does not store: mlx-vlm derives the
# n-gram hash tables from the config (the converter checks they agree).
_DERIVED_PARAMETER = re.compile(
    r"\.ple\.ple_embedding\.(layer_multipliers|ngram_heads_offsets|ngram_heads_vocab_sizes)$"
)


def prepare_mlx_vlm_load(model_dir: Path | str) -> dict[str, Any]:
    """Set up an mlx-vlm load of *model_dir* if it is a converted Edge0 pack.

    Returns extra keyword arguments for ``mlx_vlm.load`` (empty for any other
    model). The routed experts are absent from the safetensors by design, so
    the load is non-strict; :func:`verify_parameters_loaded` then checks that
    nothing *else* was left at its random initialization.
    """
    if read_edge0_config(model_dir) is None:
        return {}
    from olmlx.engine.edge0.ple import install_mlx_vlm_hook

    install_mlx_vlm_hook()
    return {"strict": False}


def _stored_keys(model_dir: Path) -> set[str]:
    keys: set[str] = set()
    for shard in sorted(model_dir.glob("model-*.safetensors")):
        # Lazy load: only the header is read.
        loaded = mx.load(str(shard))
        assert isinstance(loaded, dict)
        keys.update(loaded.keys())
    return keys


def verify_parameters_loaded(model: Any, model_dir: Path | str) -> None:
    """Fail if any served parameter is missing from the converted weights.

    Call after the routed experts have been replaced. A non-strict load would
    otherwise serve a randomly-initialized matrix silently.
    """
    model_dir = Path(model_dir)
    section = read_edge0_config(model_dir) or {}
    stored = _stored_keys(model_dir)
    missing = []
    for key, _ in tree_flatten(model.parameters()):
        if key in stored or _DERIVED_PARAMETER.search(key):
            continue
        if key.startswith("vision_tower.") and not section.get("has_vision"):
            continue
        missing.append(key)
    if missing:
        raise Edge0PackError(
            f"converted Edge0 pack at {model_dir} lacks {len(missing)} model "
            "parameters (re-run the conversion): " + ", ".join(sorted(missing)[:8])
        )
    if not section.get("has_vision"):
        logger.warning(
            "Edge0 pack %s ships no vision projector; image input is unavailable",
            model_dir,
        )


def _load_layer_tensors(latent_dir: Path, layer: int, bits: int) -> dict[str, Any]:
    loaded = mx.load(str(latent_dir / f"layer_{layer:03d}.safetensors"))
    assert isinstance(loaded, dict)
    tensors: dict[str, Any] = dict(loaded)
    for key in ("down.scales", "down.biases", "up.scales", "up.biases"):
        tensors[key] = tensors[key].astype(mx.float32)
    tensors["bits"] = bits
    return tensors


def load_draft_head(model: Any, model_dir: Path) -> Any | None:
    """Load the converted MTP draft head for *model*, or ``None``.

    Evaluated here, on the load thread: the drafter's weights and rotary
    buffers would otherwise stay lazy and bound to this thread, and crash the
    generation worker on the first drafted token. A head that fails to load
    only costs the speed-up, so it is logged rather than raised.
    """
    from olmlx.config import settings

    section = read_edge0_config(model_dir) or {}
    dirname = section.get("draft_head")
    if settings.edge0_draft_head is not True or not isinstance(dirname, str):
        return None
    draft_dir = model_dir / dirname
    if not (draft_dir / "config.json").is_file():
        return None
    try:
        from mlx_vlm.speculative import load_drafter
        from mlx_vlm.speculative.drafters import validate_drafter_compatibility

        from olmlx.engine.model_manager import _materialize_module_buffers

        drafter, kind = load_drafter(str(draft_dir), kind="mtp")
        validate_drafter_compatibility(model, drafter, kind)
        mx.eval(drafter.parameters())  # pyright: ignore[reportAttributeAccessIssue]
        _materialize_module_buffers(drafter)
    except Exception:  # noqa: BLE001 — optional accelerator
        logger.warning(
            "Edge0 draft head in %s could not be loaded; decoding without it",
            draft_dir,
            exc_info=True,
        )
        return None
    return drafter


def wrap_edge0_latent(
    model: Any,
    flash_moe_dir: Path | str,
    moe_config: dict[str, Any],
    *,
    io_threads: int,
    cache_budget_experts: int,
) -> tuple[Any, Any]:
    """Replace a lazily-loaded ``qwen4_exp`` model's experts with latent ones.

    Mirrors ``wrap_flash_moe``: returns ``(wrapped_model, store)``, evaluates
    the non-expert parameters on the calling (load) thread, and closes the
    store if anything fails.
    """
    from olmlx.engine.flash.flash_moe_model import FlashMoeModelWrapper

    flash_moe_dir = Path(flash_moe_dir)
    model_dir = flash_moe_dir.parent
    pack = Edge0Pack(model_dir)
    geometry = pack.geometry
    latent = moe_config["latent"]
    if (latent["n_layers"], latent["n_experts"], latent["latent"], latent["ff"]) != (
        geometry.n_layers,
        geometry.n_experts,
        geometry.latent,
        geometry.ff,
    ):
        raise Edge0PackError(
            f"{flash_moe_dir} was converted from a different pack; re-run the conversion"
        )
    bits = int(latent["projection_bits"])
    latent_dir = flash_moe_dir / LATENT_DIRNAME
    labels = pack.labels

    store = LatentExpertStore(
        pack, num_io_threads=io_threads, cache_budget_experts=cache_budget_experts
    )

    def dispatch_factory(layer_idx: int) -> LatentMoE:
        # LatentMoE evaluates its tables here, on the load thread, so they
        # are materialized leaves readable from the generation thread.
        return LatentMoE(
            layer_idx=layer_idx,
            geometry=geometry,
            labels=labels[layer_idx],
            tensors=_load_layer_tensors(latent_dir, layer_idx, bits),
            store=store,
        )

    try:
        wrapped = FlashMoeModelWrapper(
            model,
            store,  # type: ignore[arg-type] — duck-typed: close() + stats
            moe_layer_indices=moe_config["moe_layer_indices"],
            hidden_size=moe_config["hidden_size"],
            intermediate_size=moe_config["intermediate_size"],
            num_experts=moe_config["num_experts"],
            num_experts_per_tok=moe_config["num_experts_per_tok"],
            dispatch_factory=dispatch_factory,
        )
        verify_parameters_loaded(model, model_dir)
        # Same reason as wrap_flash_moe: eval the INNER model's parameters
        # (the wrapper keeps it under an underscore attribute nn.Module does
        # not traverse) so no weight stays lazy and bound to this thread.
        mx.eval(model.parameters())
        # Underscore attribute: found by ``inference._vlm_draft_head``, not a
        # parameter of the served model.
        wrapped._vlm_draft_head = load_draft_head(model, model_dir)
    except Exception:
        store.close()
        raise
    return wrapped, store
