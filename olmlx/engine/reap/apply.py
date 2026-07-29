"""REAP pruning surgery: streaming shard-by-shard restack of kept experts.

The trap (#701): pruning renumbers experts, so router rows
(gate.weight / e_score_correction_bias / router.bias / ...) MUST be sliced with
the same keep list as the expert stacks — and any expert-count-sized tensor we
do not recognize is a hard error, because a missed one produces a model that
loads, runs, and routes every token to the wrong expert.
"""

from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path
from typing import Callable

import mlx.core as mx

from olmlx.engine.reap.plan import ReapPlan

_EXPERT_COUNT_KEYS = (
    "n_routed_experts",
    "num_local_experts",
    "num_experts",
    "moe_num_experts",
)

# Router-tensor suffixes relative to the MoE module (e.g. "mlp"). Axis 0 is the
# expert axis in every one; QuantizedLinear packs along the last axis so row
# slicing is always safe.
_ROUTER_SUFFIXES = (
    r"gate\.(weight|scales|biases|bias)",
    r"gate\.gate\.(weight|scales|biases)",  # Step-3.5 inner linear
    r"gate\.e_score_correction_bias",
    r"gate\.router_bias",  # Step-3.5
    r"e_score_correction_bias",  # MiniMax block-level
    r"router\.(weight|scales|biases|bias)",  # gpt-oss
)
_EXPERT_COMPONENTS = ("weight", "scales", "biases", "bias")


class ReapApplyError(RuntimeError):
    pass


def rewrite_config_num_experts(config: dict, new_count: int) -> str:
    """Mutate `config` in place, setting the expert-count key to `new_count`.

    Unwraps `text_config` (VLM-style nested text config) if present. Returns
    the alias key that was rewritten.
    """
    target = config.get("text_config", config)
    for key in _EXPERT_COUNT_KEYS:
        if key in target:
            target[key] = new_count
            return key
    raise ReapApplyError(f"no expert-count key in config (tried {_EXPERT_COUNT_KEYS})")


def _classify(
    name: str,
    keep_map: dict[int, list[int]],
    num_experts_orig: int,
    arr_shape: tuple,
    layer_re: re.Pattern,
    router_res: list[re.Pattern],
    expert_re: re.Pattern,
) -> tuple[str, int | None]:
    """Returns ("expert"|"router"|"copy"|"error", layer_idx)."""
    m = layer_re.match(name)
    if not m:
        return "copy", None
    layer_idx = int(m.group(1))
    if layer_idx not in keep_map:
        return "copy", None
    rest = m.group(2)
    if expert_re.match(rest):
        return "expert", layer_idx
    for rre in router_res:
        if rre.match(rest):
            return "router", layer_idx
    if "shared_expert" not in rest and arr_shape and arr_shape[0] == num_experts_orig:
        return "error", layer_idx
    return "copy", None


def apply_plan(
    model_dir: Path,
    plan: ReapPlan,
    output_dir: Path,
    *,
    progress_callback: Callable[[str, float], None] | None = None,
) -> Path:
    from olmlx.engine.flash.moe_bundler import _detect_expert_format
    from olmlx.engine.pre_shard import collect_non_weight_files

    model_dir, output_dir = Path(model_dir), Path(output_dir)
    if plan.mode != "uniform":
        raise ReapApplyError(
            f"apply supports uniform plans only for now; {plan.mode!r} plans need "
            f"the phase 3 serving-side support (per-layer expert_counts / bank_split)"
        )
    keep_counts = {len(v) for v in plan.keep.values()}
    if len(keep_counts) != 1:
        raise ReapApplyError("uniform plan has non-uniform keep counts")
    keep_count = keep_counts.pop()

    config = json.loads((model_dir / "config.json").read_text())
    cfg_view = config.get("text_config", config)

    actual_num_experts = None
    for key in _EXPERT_COUNT_KEYS:
        if key in cfg_view:
            actual_num_experts = cfg_view[key]
            break
    if actual_num_experts is not None and actual_num_experts != plan.num_experts_orig:
        raise ReapApplyError(
            f"plan.num_experts_orig={plan.num_experts_orig} does not match the "
            f"checkpoint config's expert count ({actual_num_experts}); refusing "
            f"to apply a stale/mismatched plan (the misrouting guard keys off "
            f"num_experts_orig, so a mismatch would silently weaken it)"
        )

    n_group = cfg_view.get("n_group")
    if n_group is not None and n_group > 1:
        raise ReapApplyError(
            f"group-routed config (n_group={n_group}) is not supported by "
            f"apply_plan yet; a uniform prune leaves n_group/topk_group "
            f"untouched, which either breaks MoEGate's group reshape or "
            f"silently shifts group partition boundaries (phase 3 territory)"
        )

    num_hidden_layers = cfg_view.get("num_hidden_layers")
    if num_hidden_layers is not None:
        bad_layers = sorted(
            idx
            for idx in set(plan.keep) | set(plan.moe_layer_indices)
            if not (0 <= idx < num_hidden_layers)
        )
        if bad_layers:
            raise ReapApplyError(
                f"plan references layer index/indices {bad_layers} outside "
                f"num_hidden_layers={num_hidden_layers}"
            )

    top_k = cfg_view.get("num_experts_per_tok", plan.top_k)
    if top_k > keep_count:
        raise ReapApplyError(f"num_experts_per_tok={top_k} > keep_count={keep_count}")

    index_path = model_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else None
    if index:
        shards = sorted(set(index["weight_map"].values()))
    else:
        # Without an index, format detection and the shard loop below both
        # assume the single-file layout — refuse anything else explicitly
        # rather than failing later with a confusing FileNotFoundError.
        shards = sorted(p.name for p in model_dir.glob("*.safetensors"))
        if not shards:
            raise ReapApplyError(f"no .safetensors files found in {model_dir}")
        if shards != ["model.safetensors"]:
            raise ReapApplyError(
                f"multi-file checkpoint without model.safetensors.index.json "
                f"is unsupported (found {shards}); regenerate the index"
            )
    fmt = _detect_expert_format(model_dir, plan.moe_layer_indices[0], index)
    moe_prefix = fmt.expert_prefix.split(".")[0]  # e.g. "mlp"
    container_rel = fmt.expert_prefix[len(moe_prefix) + 1 :]  # e.g. "switch_mlp"

    layer_re = re.compile(
        rf"^{re.escape(fmt.layer_prefix)}\.(\d+)\.({re.escape(moe_prefix)}\..+)$"
    )
    proj_alt = "|".join(re.escape(p) for p in fmt.projections)
    comp_alt = "|".join(_EXPERT_COMPONENTS)
    expert_re = re.compile(
        rf"^{re.escape(moe_prefix)}\.{re.escape(container_rel)}\.({proj_alt})\.({comp_alt})$"
    )
    router_res = [
        re.compile(rf"^{re.escape(moe_prefix)}\.{suffix}$")
        for suffix in _ROUTER_SUFFIXES
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    weight_map: dict[str, str] = {}
    total_size = 0

    for si, shard in enumerate(shards):
        tensors = mx.load(str(model_dir / shard))
        out: dict[str, mx.array] = {}
        for name, arr in tensors.items():
            kind, layer_idx = _classify(
                name,
                plan.keep,
                plan.num_experts_orig,
                tuple(arr.shape),
                layer_re,
                router_res,
                expert_re,
            )
            if kind == "error":
                raise ReapApplyError(
                    f"unrecognized expert-count-sized tensor {name!r}: refusing to "
                    f"copy it through unsliced (would silently misroute)"
                )
            if kind in ("expert", "router"):
                out[name] = arr[mx.array(plan.keep[layer_idx])]
            else:
                out[name] = arr
        mx.eval(*out.values())
        mx.save_safetensors(str(output_dir / shard), out)
        for name in out:
            weight_map[name] = shard
        total_size += (output_dir / shard).stat().st_size
        del tensors, out
        mx.clear_cache()
        if progress_callback:
            progress_callback(
                f"Pruned shard {si + 1}/{len(shards)}", (si + 1) / len(shards)
            )

    rewrite_config_num_experts(config, keep_count)
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False)
    )
    if index is not None or len(shards) > 1:
        (output_dir / "model.safetensors.index.json").write_text(
            json.dumps(
                {"metadata": {"total_size": total_size}, "weight_map": weight_map},
                indent=2,
            )
        )
    for f in collect_non_weight_files(model_dir):
        if f.name != "config.json":
            shutil.copy2(f, output_dir / f.name)
    (output_dir / "reap_provenance.json").write_text(
        json.dumps(
            {
                "pruned_from": str(model_dir),
                "plan_mode": plan.mode,
                "keep_count": keep_count,
                "num_experts_orig": plan.num_experts_orig,
                "moe_layer_indices": plan.moe_layer_indices,
                "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            indent=2,
        )
    )
    return output_dir
