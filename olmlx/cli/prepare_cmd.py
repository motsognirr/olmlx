"""Prepare subcommands: spectral, shard, flash, dflash, eagle; flash info."""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

from olmlx.config import (
    settings,
    warn_legacy_flash_env as _warn_legacy_flash_env,
)

from olmlx.cli.config_cmd import _configure_logging
from olmlx.cli.models_cmd import _resolve_and_download

logger = logging.getLogger(__name__)


def _flash_progress(desc, frac):
    bar_len = 30
    filled = int(bar_len * frac)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"\r  {desc:<40s} [{bar}] {frac:5.1%}", end="", flush=True)
    if frac >= 1.0:
        print()


def cmd_spectral_prepare(args):
    """Prepare a model for spectral quant (eigenspectral calibration)."""
    _configure_logging()

    _store, local_dir = _resolve_and_download(args.model)
    model_path = str(local_dir)

    print(f"Running spectral calibration for {args.model}...")
    print(f"  Model path: {model_path}")
    print(f"  Average bits: {args.avg_bits}")
    dataset_label = args.calibration_dataset or "c4"
    print(f"  Calibration dataset: {dataset_label}")
    print(f"  Calibration samples: {args.samples}")
    print(f"  Max tokens per head: {args.max_tokens}")
    print()

    from olmlx.engine.spectralquant_calibrate import calibrate_model

    output_dir = calibrate_model(
        model_path=model_path,
        num_samples=args.samples,
        calibration_dataset=args.calibration_dataset,
        avg_bits=args.avg_bits,
        max_tokens_per_head=args.max_tokens,
        progress_callback=_flash_progress,
    )

    print("\nSpectral calibration complete!")
    print(f"  Output: {output_dir}")
    print("\nTo use spectral quant:")
    print(f"  OLMLX_KV_CACHE_QUANT=spectral:{args.avg_bits} olmlx serve")


def cmd_shard_prepare(args):
    """Prepare a model for shard quant (per-head PCA-K + VQ-V calibration)."""
    _configure_logging()

    _store, local_dir = _resolve_and_download(args.model)
    model_path = str(local_dir)

    print(f"Running shard calibration for {args.model}...")
    print(f"  Model path: {model_path}")
    print(f"  Bits: {args.bits}")
    dataset_label = args.calibration_dataset or "c4"
    print(f"  Calibration dataset: {dataset_label}")
    print(f"  Calibration samples: {args.samples}")
    print(f"  Max tokens per head: {args.max_tokens}")
    print(f"  K rank energy: {args.k_energy}")
    print()

    from olmlx.engine.shardquant_calibrate import calibrate_model_shard

    output_dir = calibrate_model_shard(
        model_path=model_path,
        num_samples=args.samples,
        calibration_dataset=args.calibration_dataset,
        bits=args.bits,
        max_tokens_per_head=args.max_tokens,
        progress_callback=_flash_progress,
        k_energy=args.k_energy,
    )

    print("\nShard calibration complete!")
    print(f"  Output: {output_dir}")
    print("\nTo use shard quant:")
    print(f"  OLMLX_KV_CACHE_QUANT=shard:{args.bits} olmlx serve")


def cmd_flash_prepare(args):
    """Prepare a model for flash inference (auto-detects MoE vs dense)."""
    _configure_logging()
    # Warn about stale OLMLX_EXPERIMENTAL_FLASH* env vars (warn-only;
    # does not forward values).
    _warn_legacy_flash_env()

    _store, local_dir = _resolve_and_download(args.model)
    model_path = str(local_dir)

    # Auto-detect MoE model
    from olmlx.engine.flash.moe_prepare import is_moe_model

    if is_moe_model(model_path):
        _cmd_flash_moe_prepare(args, model_path)
    else:
        _cmd_flash_dense_prepare(args, model_path)


def _cmd_flash_moe_prepare(args, model_path):
    """Prepare an MoE model for Flash-MoE inference."""
    from olmlx.engine.flash.moe_prepare import prepare_moe_for_flash

    print("Detected MoE model — using Flash-MoE preparation (no model loading needed)")
    print(f"  Model path: {model_path}")
    print()

    output_dir = prepare_moe_for_flash(
        model_path=model_path,
        progress_callback=_flash_progress,
    )

    print("\nFlash-MoE preparation complete!")
    print(f"  Output: {output_dir}")
    print("\nTo use Flash-MoE inference:")
    print("  OLMLX_FLASH_MOE=true olmlx serve")


def _cmd_flash_dense_prepare(args, model_path):
    """Prepare a dense model for flash inference."""
    from olmlx.engine.flash.prepare import prepare_model_for_flash

    print(f"Preparing {args.model} for flash inference...")
    print(f"  Model path: {model_path}")
    print(f"  Predictor rank: {args.rank}")
    if args.sensitive_layers > 0:
        print(
            f"  Sensitive layers: last {args.sensitive_layers} (rank x{args.sensitive_rank_multiplier})"
        )
    dataset_label = args.calibration_dataset or "c4"
    print(f"  Calibration dataset: {dataset_label}")
    print(f"  Calibration samples: {args.samples}")
    print(f"  Activation threshold: {args.threshold}")
    print(f"  Training epochs: {args.epochs}")
    print()

    output_dir = prepare_model_for_flash(
        model_path=model_path,
        rank=args.rank,
        sensitive_layers=args.sensitive_layers,
        sensitive_rank_multiplier=args.sensitive_rank_multiplier,
        num_samples=args.samples,
        calibration_dataset=args.calibration_dataset,
        activation_threshold=args.threshold,
        epochs=args.epochs,
        train_lookahead=settings.flash_prefetch,
        progress_callback=_flash_progress,
    )

    print("\nFlash preparation complete!")
    print(f"  Output: {output_dir}")
    print("\nTo use flash inference:")
    print("  olmlx serve --flash")
    print("  # or set OLMLX_FLASH=true")


def cmd_dflash_precompute(args):
    """Precompute target hidden states for DFlash draft training."""
    _configure_logging()

    _store, local_dir = _resolve_and_download(args.model)
    model_path = str(local_dir)

    target_layer_ids: list[int] | None = None
    if args.target_layer_ids:
        try:
            target_layer_ids = [int(x) for x in args.target_layer_ids.split(",")]
        except ValueError as exc:
            raise SystemExit(
                f"--target-layer-ids must be a comma-separated list of "
                f"integers, got {args.target_layer_ids!r}: {exc}"
            ) from exc

    output_dir = Path(args.output) if args.output else Path(model_path) / "dflash_cache"

    print(f"Precomputing target hidden states for {args.model}...")
    print(f"  Target path: {model_path}")
    print(f"  Output: {output_dir}")
    print(f"  Shards: {args.shards}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Seq len: {args.seq_len}")
    if target_layer_ids:
        print(f"  Target layer ids: {target_layer_ids}")
    else:
        print(f"  Target layers: {args.num_target_layers} (evenly spaced)")
    print()

    from mlx_lm import load as _mlx_lm_load

    from olmlx.engine.dflash.decoder import _patch_model, _unpatch_model
    from olmlx.engine.dflash.precompute import precompute_target_hiddens
    from olmlx.engine.dflash.prepare import _resolve_target_layer_ids
    from olmlx.engine.dflash.training_data import stream_training_batches

    # ``mlx_lm.load`` returns a 2-tuple in current versions; older
    # variants returned 3-tuples. Slice to the first two so either
    # works.
    loaded = _mlx_lm_load(model_path)
    target, tokenizer = loaded[0], loaded[1]
    target.eval()
    if hasattr(target, "freeze"):
        target.freeze()

    target_layers_attr = (
        target.layers if hasattr(target, "layers") else target.model.layers
    )
    layer_ids = _resolve_target_layer_ids(
        target_layer_ids, args.num_target_layers, len(target_layers_attr)
    )
    print(f"  Resolved target_layer_ids: {layer_ids}\n")

    # Caller-owned hidden-state storage — kept off the ``nn.Module`` so
    # mlx's parameter tracker doesn't pick the captures up.
    hidden_capture: list[Any] = [None] * len(layer_ids)
    _patch_model(target, layer_ids, hidden_capture)
    try:
        # ``max_examples`` is intentionally omitted — ``num_shards``
        # below is the load-bearing cap (it's also what writes the
        # correct count into ``index.json``); a duplicate cap on the
        # iterator side would just mask off-by-one issues.
        batches = stream_training_batches(
            tokenizer,
            dataset=args.data or "HuggingFaceH4/ultrachat_200k",
            split=args.split or "train_sft",
            batch_size=args.batch_size,
            seq_len=args.seq_len,
        )
        precompute_target_hiddens(
            target,
            batches,
            output_dir,
            hidden_capture,
            target_layer_ids=layer_ids,
            num_shards=args.shards,
            progress_callback=_flash_progress,
        )
    finally:
        _unpatch_model(target)

    print("\nPrecompute complete!")
    print(f"  Output: {output_dir}")
    print(
        f"  Re-use with: olmlx dflash prepare {args.model} "
        f"--use-precomputed {output_dir}"
    )


def cmd_dflash_prepare(args):
    """Train a DFlash draft model for a target."""
    _configure_logging()

    # Validate up-front so the user gets a clear message before we
    # download the target and run hook-installation. The same check
    # exists deep inside the training loop, but surfacing it after a
    # multi-GB download is a poor UX.
    if args.block_size < 1:
        raise SystemExit(f"--block-size must be >= 1, got {args.block_size}")
    min_seq_len = 2 * args.block_size + 1
    if args.seq_len < min_seq_len:
        raise SystemExit(
            f"--seq-len ({args.seq_len}) too small for --block-size "
            f"({args.block_size}); need at least 2*block_size + 1 = "
            f"{min_seq_len} tokens per sequence."
        )
    if args.train_windows_per_step < 1:
        raise SystemExit(
            f"--train-windows-per-step must be >= 1, got {args.train_windows_per_step}"
        )
    if args.self_generate and args.use_precomputed:
        # ``prepare_dflash_draft`` re-checks this, but only after
        # ``ensure_downloaded`` — reject here so the user doesn't pay a
        # multi-GB target download before seeing the error.
        raise SystemExit(
            "--self-generate and --use-precomputed are mutually exclusive: "
            "self-generation runs the target online, while precomputed "
            "shards were captured over a different token stream. Pass one "
            "or the other."
        )

    _store, local_dir = _resolve_and_download(args.model)
    model_path = str(local_dir)

    target_layer_ids: list[int] | None = None
    if args.target_layer_ids:
        try:
            target_layer_ids = [int(x) for x in args.target_layer_ids.split(",")]
        except ValueError as exc:
            raise SystemExit(
                f"--target-layer-ids must be a comma-separated list of "
                f"integers, got {args.target_layer_ids!r}: {exc}"
            ) from exc

    print(f"Training DFlash draft for {args.model}...")
    print(f"  Target path: {model_path}")
    print(f"  Steps: {args.steps}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Seq len: {args.seq_len}")
    print(f"  Block size (draft tokens): {args.block_size}")
    print(f"  Draft layers: {args.num_hidden_layers}")
    if target_layer_ids:
        print(f"  Target layer ids: {target_layer_ids}")
    else:
        print(f"  Target layers: {args.num_target_layers} (evenly spaced)")
    print(f"  Dataset: {args.data}")
    print(f"  LR: {args.lr}")
    if args.distill:
        print(f"  Distillation: alpha={args.distill_alpha} temp={args.distill_temp}")
    if args.train_windows_per_step != 1:
        print(f"  Train windows per step: {args.train_windows_per_step}")
    if args.use_precomputed:
        print(f"  Precomputed shards: {args.use_precomputed}")
    if args.self_generate:
        print(
            f"  Self-generate: {args.selfgen_seqs} sequences x "
            f"{args.selfgen_max_new} tokens (target greedy)"
        )
    print()

    from olmlx.engine.dflash.prepare import prepare_dflash_draft

    output_dir = prepare_dflash_draft(
        model_path=model_path,
        dataset=args.data,
        dataset_split=args.split,
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        block_size=args.block_size,
        num_hidden_layers=args.num_hidden_layers,
        target_layer_ids=target_layer_ids,
        num_target_layers=args.num_target_layers,
        mask_token_id=args.mask_token_id,
        lr=args.lr,
        output_dir=args.output,
        distill=args.distill,
        distill_alpha=args.distill_alpha,
        distill_temp=args.distill_temp,
        position_decay_gamma=args.position_decay_gamma,
        train_windows_per_step=args.train_windows_per_step,
        use_precomputed=args.use_precomputed,
        self_generate=args.self_generate,
        selfgen_num_seqs=args.selfgen_seqs,
        selfgen_max_new=args.selfgen_max_new,
        progress_callback=_flash_progress,
    )

    print("\nDFlash draft training complete!")
    print(f"  Output: {output_dir}")
    print("\nTo use the trained draft:")
    print("  OLMLX_SPECULATIVE=true \\")
    print("  OLMLX_SPECULATIVE_STRATEGY=dflash \\")
    print(f"  OLMLX_SPECULATIVE_DRAFT_MODEL={output_dir} \\")
    print("  olmlx serve")


def cmd_eagle_prepare(args):
    """Train an EAGLE draft model for a target.

    Phase D supports only ``--use-precomputed`` mode: pass the
    directory of (input_ids, target_hidden) shards produced by
    ``olmlx dflash precompute`` (the shard format is shared between
    DFlash and EAGLE — EAGLE consumes the deepest captured layer).
    """
    _configure_logging()

    if args.block_size < 1:
        raise SystemExit(f"--block-size must be >= 1, got {args.block_size}")
    if not args.use_precomputed:
        raise SystemExit(
            "--use-precomputed is required for EAGLE training. Run "
            "`olmlx dflash precompute <target>` first to dump target hidden "
            "states; the same shards work for EAGLE (it just slices the "
            "deepest layer from the concatenated ladder)."
        )

    _store, local_dir = _resolve_and_download(args.model)
    model_path = str(local_dir)

    print(f"Training EAGLE draft for {args.model}...")
    print(f"  Target path: {model_path}")
    print(f"  Steps: {args.steps}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Seq len: {args.seq_len}")
    print(f"  Block size (draft tokens): {args.block_size}")
    print(f"  Draft layers: {args.num_hidden_layers}")
    print(f"  LR: {args.lr}")
    sample_positions = args.sample_positions if args.sample_positions > 0 else None
    print(f"  Sample positions/step: {sample_positions or 'all'}")
    print(f"  Precomputed shards: {args.use_precomputed}")
    print()

    from olmlx.engine.eagle.prepare import prepare_eagle_draft

    output_dir = prepare_eagle_draft(
        model_path=model_path,
        use_precomputed=args.use_precomputed,
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        block_size=args.block_size,
        num_hidden_layers=args.num_hidden_layers,
        lr=args.lr,
        sample_positions=sample_positions,
        seed=args.seed,
        output_dir=args.output,
        progress_callback=_flash_progress,
    )

    print("\nEAGLE draft training complete!")
    print(f"  Output: {output_dir}")
    print("\nTo use the trained draft:")
    print("  OLMLX_SPECULATIVE=true \\")
    print("  OLMLX_SPECULATIVE_STRATEGY=eagle \\")
    print(f"  OLMLX_SPECULATIVE_DRAFT_MODEL={output_dir} \\")
    print("  olmlx serve")


def cmd_flash_info(args):
    """Show flash preparation info for a model."""
    # Warn about stale ``OLMLX_EXPERIMENTAL_FLASH*`` env vars (warn-only;
    # does not forward values).
    _warn_legacy_flash_env()
    _store, local_dir = _resolve_and_download(args.model, download=False)

    # Check for Flash-MoE first
    flash_moe_dir = local_dir / "flash_moe"
    flash_dir = local_dir / "flash"

    if flash_moe_dir.exists() and (flash_moe_dir / "flash_moe_config.json").exists():
        _show_flash_moe_info(args.model, flash_moe_dir)
    elif flash_dir.exists():
        _show_flash_dense_info(args.model, flash_dir)
    else:
        print(f"Model '{args.model}' has not been prepared for flash inference.")
        print(f"\nRun: olmlx flash prepare {args.model}")


def _show_flash_moe_info(model_name, flash_moe_dir):
    config_path = flash_moe_dir / "flash_moe_config.json"
    config = json.loads(config_path.read_text())
    print(f"Flash-MoE info for '{model_name}':")
    print("  Status:             prepared")
    print("  Type:               MoE (expert offloading)")
    print(f"  Flash directory:    {flash_moe_dir}")
    print(f"  Hidden size:        {config.get('hidden_size')}")
    print(f"  Intermediate size:  {config.get('intermediate_size')}")
    print(f"  Num experts:        {config.get('num_experts')}")
    print(f"  Experts per token:  {config.get('num_experts_per_tok')}")
    print(f"  MoE layers:         {config.get('num_moe_layers')}")
    print(f"  Prepared at:        {config.get('prepared_at')}")

    fe_files = list(flash_moe_dir.glob("*.flashexperts"))
    print(f"  Expert files:       {len(fe_files)}")

    total_bytes = sum(f.stat().st_size for f in flash_moe_dir.rglob("*") if f.is_file())
    if total_bytes > 1024**3:
        print(f"  Total size:         {total_bytes / (1024**3):.1f} GB")
    else:
        print(f"  Total size:         {total_bytes / (1024**2):.1f} MB")

    print("\nTo use Flash-MoE inference:")
    print("  OLMLX_FLASH_MOE=true olmlx serve")


def _show_flash_dense_info(model_name, flash_dir):
    config_path = flash_dir / "flash_config.json"
    if not config_path.exists():
        print(f"Flash directory exists but no config found: {flash_dir}")
        return

    config = json.loads(config_path.read_text())
    print(f"Flash info for '{model_name}':")
    print("  Status:             prepared")
    print("  Type:               Dense (neuron offloading)")
    print(f"  Flash directory:    {flash_dir}")
    print(f"  Hidden size:        {config.get('hidden_size')}")
    print(f"  Intermediate size:  {config.get('intermediate_size')}")
    print(f"  Num layers:         {config.get('num_layers')}")
    print(f"  Predictor rank:     {config.get('predictor_rank')}")
    print(f"  Calibration samples:{config.get('num_calibration_samples')}")
    print(f"  Prepared at:        {config.get('prepared_at')}")

    fw_files = list(flash_dir.glob("*.flashweights"))
    print(f"  Weight files:       {len(fw_files)}")

    pred_dir = flash_dir / "predictors"
    if pred_dir.exists():
        pred_files = list(pred_dir.glob("*.npz"))
        print(f"  Predictor files:    {len(pred_files)}")

    total_bytes = sum(f.stat().st_size for f in flash_dir.rglob("*") if f.is_file())
    print(f"  Total size:         {total_bytes / (1024**2):.1f} MB")

    print("\nTo use flash inference:")
    print("  olmlx serve --flash")
    print("  # or set OLMLX_FLASH=true")
