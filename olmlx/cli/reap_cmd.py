"""olmlx reap: REAP expert pruning CLI (calibrate -> plan -> apply -> report)."""

from __future__ import annotations

import json
from pathlib import Path

from olmlx.cli.config_cmd import _configure_logging
from olmlx.cli.models_cmd import _resolve_and_download
from olmlx.cli.prepare_cmd import _flash_progress


def _reap_dir(local_dir: Path) -> Path:
    return Path(local_dir) / "reap"


def cmd_reap_calibrate(args):
    _configure_logging()
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    if not sources:
        raise SystemExit("--sources must name at least one source")
    _store, local_dir = _resolve_and_download(args.model)
    from olmlx.engine.reap.calibrate import calibrate_saliency_streaming, save_saliency
    from olmlx.engine.reap.corpus import build_calibration_corpus

    print(f"Calibrating REAP saliency for {args.model}")
    print(
        f"  Sources: {', '.join(sources)}  samples/source: {args.samples_per_source}"
        f"  max tokens: {args.max_tokens}"
    )
    texts = build_calibration_corpus(
        sources, args.samples_per_source, progress_callback=_flash_progress
    )
    acc, meta = calibrate_saliency_streaming(
        str(local_dir),
        texts,
        max_tokens=args.max_tokens,
        progress_callback=_flash_progress,
    )
    output = Path(args.output) if args.output else _reap_dir(local_dir) / "saliency.npz"
    save_saliency(output, acc, meta)
    print("\nCalibration complete!")
    print(f"  Saliency: {output}")
    print(
        f"\nNext: olmlx reap plan {args.model} --keep <n>   "
        f"(num_experts={meta['num_experts']})"
    )


def cmd_reap_plan(args):
    _configure_logging()
    if args.mode == "uniform" and args.keep is None:
        raise SystemExit("--mode uniform requires --keep")
    if args.mode in ("global", "graded") and args.keep_fraction is None:
        raise SystemExit(f"--mode {args.mode} requires --keep-fraction")
    _store, local_dir = _resolve_and_download(args.model, download=False)
    from olmlx.engine.reap.calibrate import load_saliency
    from olmlx.engine.reap.plan import plan_global, plan_graded, plan_uniform, save_plan

    saliency_path = (
        Path(args.saliency) if args.saliency else _reap_dir(local_dir) / "saliency.npz"
    )
    acc, meta = load_saliency(saliency_path)
    sources = (
        [s.strip() for s in args.sources.split(",") if s.strip()]
        if args.sources
        else None
    )
    sal = acc.saliency(sources)
    common = dict(top_k=meta["top_k"], num_experts=meta["num_experts"])
    if args.mode == "uniform":
        plan = plan_uniform(sal, meta["moe_layer_indices"], args.keep, **common)
    elif args.mode == "global":
        plan = plan_global(
            sal,
            meta["moe_layer_indices"],
            args.keep_fraction,
            floor_multiple=args.floor_multiple,
            **common,
        )
    else:
        plan = plan_graded(
            sal,
            meta["moe_layer_indices"],
            args.keep_fraction,
            high_fraction=args.high_fraction,
            high_bits=args.high_bits,
            low_bits=args.low_bits,
            floor_multiple=args.floor_multiple,
            **common,
        )
    plan.meta = {"sources": sources or acc.sources, "saliency": str(saliency_path)}
    output = (
        Path(args.output) if args.output else _reap_dir(local_dir) / "reap_plan.json"
    )
    save_plan(plan, output)
    kept = sum(len(v) for v in plan.keep.values())
    total = len(plan.keep) * meta["num_experts"]
    print(f"\nPlan written: {output}")
    print(f"  Mode: {plan.mode}  kept experts: {kept}/{total}")
    print(f"\nNext: olmlx reap apply {args.model} --plan {output}")


def cmd_reap_apply(args):
    _configure_logging()
    _store, local_dir = _resolve_and_download(args.model, download=True)
    from olmlx.engine.reap.apply import apply_plan
    from olmlx.engine.reap.plan import load_plan

    plan = load_plan(Path(args.plan))
    keep_count = len(next(iter(plan.keep.values())))
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else _reap_dir(local_dir) / f"pruned-{plan.mode}{keep_count}"
    )
    out = apply_plan(
        Path(local_dir), plan, output_dir, progress_callback=_flash_progress
    )
    print("\nPruning complete!")
    print(f"  Output: {out}")
    print(
        f"\nRegister it in models.json with a path entry, then: olmlx reap report "
        f"{args.model} --pruned-dir {out}"
    )


def cmd_reap_report(args):
    _configure_logging()
    _store, local_dir = _resolve_and_download(args.model, download=not args.skip_ppl)
    from olmlx.engine.reap.calibrate import load_saliency
    from olmlx.engine.reap.corpus import build_calibration_corpus
    from olmlx.engine.reap.report import build_report

    saliency_path = (
        Path(args.saliency) if args.saliency else _reap_dir(local_dir) / "saliency.npz"
    )
    keep = args.keep
    if keep is None:
        plan_path = _reap_dir(local_dir) / "reap_plan.json"
        if plan_path.exists():
            from olmlx.engine.reap.plan import load_plan

            keep = len(next(iter(load_plan(plan_path).keep.values())))
        else:
            raise SystemExit("--keep required (no reap_plan.json to infer it from)")
    held_out = None
    if not args.skip_ppl and args.pruned_dir:
        _acc, meta = load_saliency(saliency_path)
        held_out = build_calibration_corpus(
            meta["sources"], args.samples_per_source, skip=meta.get("num_samples", 256)
        )
    rep = build_report(
        saliency_path,
        keep_count=keep,
        full_model_dir=str(local_dir) if held_out else None,
        pruned_model_dir=args.pruned_dir if held_out else None,
        held_out_texts=held_out,
        progress_callback=_flash_progress,
    )
    print(
        f"\nREAP report (keep={keep}, chance overlap={rep['overlap']['chance']:.2f}):"
    )
    for pair, stats in rep["overlap"]["pairs"].items():
        print(f"  {pair}: mean keep-set overlap {stats['mean_overlap']:.2f}")
    if rep["perplexity"]:
        print("\n  Held-out perplexity (pruned/full):")
        for source, ratio in rep["perplexity"]["ratio"].items():
            full = rep["perplexity"]["full"][source]
            pruned = rep["perplexity"]["pruned"][source]
            print(f"    {source}: {pruned:.2f} / {full:.2f}  (x{ratio:.3f})")
    out_json = _reap_dir(local_dir) / "reap_report.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rep, indent=2))
    print(f"\n  Report JSON: {out_json}")
