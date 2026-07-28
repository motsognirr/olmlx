"""argparse parser assembly for the olmlx CLI."""

import argparse
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


from olmlx.cli.bench_cmd import _non_empty_str, _positive_int

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    # Each subparser group MUST use ``{cmd}_command`` as its ``dest``
    # so that ``cli_main()`` can look up the subcommand via
    # ``getattr(args, f"{cmd}_command", None)``.
    parser = argparse.ArgumentParser(
        prog="olmlx",
        description="Ollama-compatible API server using Apple MLX",
    )
    sub = parser.add_subparsers(dest="command")

    serve_p = sub.add_parser("serve", help="Start the server (default)")
    serve_p.add_argument(
        "--speculative",
        dest="speculative",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable speculative decoding (overrides OLMLX_SPECULATIVE)",
    )
    serve_p.add_argument(
        "--speculative-strategy",
        dest="speculative_strategy",
        choices=(
            "classic",
            "dflash",
            "eagle",
            "pld",
            "lookahead",
            "self_speculative",
            "proxy_tuning",
        ),
        default=None,
        help=(
            "Speculative decoding strategy: 'classic' (standalone draft LM), "
            "'dflash' (block-diffusion draft conditioned on target hidden "
            "states), 'eagle' (autoregressive draft head conditioned on "
            "target last-layer hidden, arxiv 2401.15077), 'pld' "
            "(prompt-lookup decoding — n-gram lookup in the prompt+generated "
            "history, no draft model required), 'self_speculative' "
            "(LayerSkip-style — uses target's own early layers as draft; "
            "no external draft model required), or 'proxy_tuning' "
            "(decode-time logit arithmetic base + α·(expert−antiexpert); "
            "set OLMLX_SPECULATIVE_PROXY_EXPERT_MODEL / "
            "_ANTIEXPERT_MODEL). Default: classic."
        ),
    )
    serve_p.add_argument(
        "--speculative-draft-model",
        dest="speculative_draft_model",
        type=_non_empty_str,
        default=None,
        help="HuggingFace path of the draft model used for speculative decoding",
    )
    serve_p.add_argument(
        "--speculative-tokens",
        dest="speculative_tokens",
        type=_positive_int,
        default=None,
        help=(
            "Number of tokens drafted per verification step (default: 4 "
            "for classic, 10 for PLD). For DFlash this is the block size "
            "(excluding the pending token); for PLD it's the max draft "
            "length (actual draft is bounded by the longest n-gram match)."
        ),
    )
    serve_p.add_argument(
        "--speculative-layers-skip",
        dest="speculative_layers_skip",
        type=_positive_int,
        default=None,
        help=(
            "Number of layers skipped during self_speculative draft "
            "(default: L//4, where L is the total number of layers). "
            "Only applies to strategy='self_speculative'."
        ),
    )
    serve_p.add_argument(
        "--kv-cache-quant",
        dest="kv_cache_quant",
        type=str,
        default=None,
        help=(
            "KV cache quantization method and bits "
            "(e.g. turboquant:4, spectral:2, shard:4)"
        ),
    )
    serve_p.add_argument(
        "--flash",
        dest="flash",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable Flash inference (LLM in a Flash; sparse FFN with SSD-"
            "backed neuron loading). Overrides OLMLX_FLASH. Requires the "
            "model to be prepared first via 'olmlx flash prepare'."
        ),
    )
    serve_p.add_argument(
        "--flash-prefetch",
        dest="flash_prefetch",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable Flash speculative neuron prefetch (overrides OLMLX_FLASH_PREFETCH).",
    )
    serve_p.add_argument(
        "--flash-speculative",
        dest="flash_speculative",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable Flash + speculative decoding (overrides OLMLX_FLASH_SPECULATIVE).",
    )
    serve_p.add_argument(
        "--flash-speculative-draft-model",
        dest="flash_speculative_draft_model",
        type=_non_empty_str,
        default=None,
        help="HuggingFace path of the draft model used for Flash speculative decoding.",
    )
    serve_p.add_argument(
        "--flash-speculative-tokens",
        dest="flash_speculative_tokens",
        type=_positive_int,
        default=None,
        help="Tokens drafted per verification step for Flash speculative (default: 4).",
    )

    svc = sub.add_parser("service", help="Manage the launchd service")
    svc_sub = svc.add_subparsers(dest="service_command")
    svc_sub.add_parser("install", help="Install and start the launchd service")
    svc_sub.add_parser("uninstall", help="Stop and remove the launchd service")
    svc_sub.add_parser("status", help="Show service status")

    mdl = sub.add_parser("models", help="Manage local models")
    mdl_sub = mdl.add_subparsers(dest="models_command")
    mdl_sub.add_parser("list", help="List locally downloaded models")
    pull_p = mdl_sub.add_parser("pull", help="Pull/download a model")
    pull_p.add_argument("model_name", help="Model name or HF path")
    pull_p.add_argument(
        "--with-draft",
        action="store_true",
        help="Also pull the matching speculative draft (if one is known for "
        "this target) and wire it into the model's config",
    )
    show_p = mdl_sub.add_parser("show", help="Show model details")
    show_p.add_argument("model_name", help="Model name or HF path")
    del_p = mdl_sub.add_parser("delete", help="Delete a local model")
    del_p.add_argument("model_name", help="Model name or HF path")
    del_p.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")
    search_p = mdl_sub.add_parser("search", help="Search for models by name")
    search_p.add_argument("query", help="Search query")

    chat_p = sub.add_parser("chat", help="Interactive chat")
    chat_p.add_argument("model_name", nargs="?", help="Model name or HF path")
    chat_p.add_argument("--system", "-s", help="System prompt")
    chat_p.add_argument(
        "--mcp-config", help="MCP config path (default: ~/.olmlx/mcp.json)"
    )
    chat_p.add_argument(
        "--no-mcp", action="store_true", default=False, help="Disable MCP"
    )
    chat_p.add_argument(
        "--no-thinking", action="store_true", default=False, help="Disable thinking"
    )
    chat_p.add_argument("--max-tokens", type=int, default=4096)
    chat_p.add_argument("--max-turns", type=int, default=25)
    chat_p.add_argument(
        "--repeat-penalty",
        type=float,
        default=1.1,
        help="Repetition penalty (1.0 = disabled, default: 1.1)",
    )
    chat_p.add_argument(
        "--repeat-last-n",
        type=int,
        default=64,
        help="Context window for repetition penalty (default: 64)",
    )
    chat_p.add_argument(
        "--no-skills", action="store_true", default=False, help="Disable skills"
    )
    chat_p.add_argument(
        "--no-builtin-tools",
        action="store_true",
        default=False,
        help="Disable built-in tools",
    )
    chat_p.add_argument(
        "--skills-dir", help="Skills directory (default: ~/.olmlx/skills)"
    )
    chat_p.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (default: model default)",
    )
    chat_p.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p sampling (default: model default)",
    )
    chat_p.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling (default: model default)",
    )
    chat_p.add_argument(
        "--tool-timeout",
        type=float,
        default=None,
        help=(
            "Timeout in seconds for tool calls, MCP and builtin "
            "(default: per-tool defaults — MCP 30, bash 120)"
        ),
    )
    chat_p.add_argument(
        "--mcp-connect-retries",
        type=int,
        default=None,
        help="MCP server connection retry attempts (default: 3)",
    )
    chat_p.add_argument(
        "--local-tool-safety",
        action="store_true",
        default=False,
        help="Apply tool safety policy to local tools (builtins, skills)",
    )
    chat_p.add_argument(
        "--tool-result-truncation",
        type=int,
        default=None,
        help="Max chars for tool result display (default: 2000)",
    )
    chat_p.add_argument(
        "--max-consecutive-tool-failures",
        type=int,
        default=None,
        help="Max consecutive tool failure turns before stopping (default: 3, 0=unlimited)",
    )
    chat_p.add_argument(
        "--voice",
        action="store_true",
        help="Enable push-to-talk STT input and Kokoro TTS output (issue #444).",
    )
    chat_p.add_argument(
        "--stt-model", default=None, help="Override STT (Whisper) model."
    )
    chat_p.add_argument(
        "--tts-model", default=None, help="Override TTS (Kokoro) model."
    )
    chat_p.add_argument(
        "--voice-name", default=None, help="Kokoro voice (e.g. af_heart)."
    )

    # Flash inference
    flash = sub.add_parser("flash", help="Flash inference (LLM in a Flash)")
    flash_sub = flash.add_subparsers(dest="flash_command")

    prepare_p = flash_sub.add_parser(
        "prepare", help="Prepare a model for flash inference"
    )
    prepare_p.add_argument("model", help="Model name or HF path")
    prepare_p.add_argument(
        "--rank", type=int, default=128, help="Predictor rank (default: 128)"
    )
    prepare_p.add_argument(
        "--samples",
        type=int,
        default=256,
        help="Number of calibration samples (default: 256)",
    )
    prepare_p.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Activation threshold (default: 0.01)",
    )
    prepare_p.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Predictor training epochs (default: 5)",
    )
    prepare_p.add_argument(
        "--calibration-dataset",
        type=str,
        default=None,
        help="Calibration dataset: 'c4' (default) or 'synthetic'",
    )
    prepare_p.add_argument(
        "--sensitive-layers",
        type=int,
        default=0,
        help="Number of last layers to use higher predictor rank (default: 0, disabled)",
    )
    prepare_p.add_argument(
        "--sensitive-rank-multiplier",
        type=int,
        default=4,
        help="Rank multiplier for sensitive layers (default: 4)",
    )
    info_p = flash_sub.add_parser(
        "info", help="Show flash preparation info for a model"
    )
    info_p.add_argument("model", help="Model name or HF path")

    # DFlash draft training
    dflash = sub.add_parser("dflash", help="DFlash block-diffusion draft training")
    dflash_sub = dflash.add_subparsers(dest="dflash_command")
    dflash_prepare_p = dflash_sub.add_parser(
        "prepare", help="Train a DFlash draft model for a target"
    )
    dflash_prepare_p.add_argument("model", help="Target model name or HF path")
    dflash_prepare_p.add_argument(
        "--data",
        type=str,
        default=None,
        help="HuggingFace dataset path (default: HuggingFaceH4/ultrachat_200k)",
    )
    dflash_prepare_p.add_argument(
        "--split",
        type=str,
        default=None,
        help="Dataset split (default: train_sft)",
    )
    dflash_prepare_p.add_argument(
        "--steps", type=int, default=2000, help="Training steps (default: 2000)"
    )
    dflash_prepare_p.add_argument(
        "--batch-size", type=int, default=4, help="Batch size (default: 4)"
    )
    dflash_prepare_p.add_argument(
        "--seq-len", type=int, default=2048, help="Sequence length (default: 2048)"
    )
    dflash_prepare_p.add_argument(
        "--block-size",
        type=int,
        default=16,
        help="Number of draft tokens per step (default: 16, per paper)",
    )
    dflash_prepare_p.add_argument(
        "--num-hidden-layers",
        type=int,
        default=5,
        help="Draft model hidden layer count (default: 5, per paper)",
    )
    dflash_prepare_p.add_argument(
        "--num-target-layers",
        type=int,
        default=5,
        help=(
            "Number of target hidden states to extract (default: 5, "
            "per paper). Ignored when --target-layer-ids is set."
        ),
    )
    dflash_prepare_p.add_argument(
        "--target-layer-ids",
        type=str,
        default=None,
        help=(
            "Comma-separated target layer indices to extract hidden states "
            "from (e.g. '5,11,17,23'). Defaults to evenly spaced layers."
        ),
    )
    dflash_prepare_p.add_argument(
        "--mask-token-id",
        type=int,
        default=None,
        help=(
            "Token id used as MASK in the block-diffusion draft input. "
            "Defaults to the tokenizer's pad_token_id (or eos_token_id if "
            "no pad). For tokenizers with neither, this flag is required — "
            "token 0 is not a safe fallback (often <bos>/<unk>)."
        ),
    )
    dflash_prepare_p.add_argument(
        "--lr", type=float, default=5e-4, help="Peak learning rate (default: 5e-4)"
    )
    dflash_prepare_p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: <target-model-dir>/dflash)",
    )
    dflash_prepare_p.add_argument(
        "--distill",
        action="store_true",
        default=False,
        help=(
            "Enable Hinton-style KL distillation against the target's "
            "logits at the masked positions. Incompatible with "
            "--use-precomputed (precomputed shards do not store logits). "
            "Memory note: peak usage ~2x the CE-only path because both "
            "target and draft probability tensors of shape "
            "(batch_size, block_size, vocab_size) are live during the "
            "KL reduction. For high-vocab targets (e.g. Gemma, Command R "
            "at ~256k vocab) lower --batch-size accordingly."
        ),
    )
    dflash_prepare_p.add_argument(
        "--distill-alpha",
        type=float,
        default=0.5,
        help=(
            "Distillation mixing weight: loss = (1-alpha)*CE + "
            "alpha*T^2*KL. Default 0.5; ignored when --distill is unset."
        ),
    )
    dflash_prepare_p.add_argument(
        "--distill-temp",
        type=float,
        default=2.0,
        help="Distillation temperature T (default: 2.0)",
    )
    dflash_prepare_p.add_argument(
        "--self-generate",
        action="store_true",
        default=False,
        help=(
            "Build the training set from responses GENERATED BY THE "
            "TARGET (greedy) instead of ground-truth dataset text — the "
            "upstream DFlash recipe, and empirically decisive for "
            "acceptance rate (ground-truth training produced ~0%% "
            "acceptance drafts). Dataset prompts are chat-templated and "
            "the target greedy-decodes the responses before training "
            "starts (a one-time generation pass; budget extra wall-clock "
            "for large targets). Training windows are restricted to the "
            "response region. Incompatible with --use-precomputed."
        ),
    )
    dflash_prepare_p.add_argument(
        "--selfgen-seqs",
        type=int,
        default=1500,
        help=(
            "Number of sequences to self-generate when --self-generate "
            "is set (default: 1500). The set is cycled (reshuffled per "
            "epoch) until --steps is exhausted."
        ),
    )
    dflash_prepare_p.add_argument(
        "--selfgen-max-new",
        type=int,
        default=640,
        help=(
            "Max tokens to generate per sequence when --self-generate "
            "is set (default: 640). Generation stops early at EOS."
        ),
    )
    dflash_prepare_p.add_argument(
        "--position-decay-gamma",
        type=float,
        default=None,
        help=(
            "Per-position loss weight decay: w_k = exp(-(k-1)/gamma) for "
            "k=1..block_size. Emphasises early positions because "
            "acceptance length compounds. Pass 0 (or negative) to "
            "disable and use the uniform-mean reduction (the default "
            "when this flag is omitted). Suggested starting value: "
            "block_size/2."
        ),
    )
    dflash_prepare_p.add_argument(
        "--train-windows-per-step",
        type=int,
        default=1,
        help=(
            "Number of non-overlapping masked windows to train on per "
            "batch (per optimizer step). Default 1 reproduces the "
            "legacy single-window behaviour bit-for-bit. K > 1 "
            "amortises the target forward across K draft-loss windows "
            "in a single optimizer step; the optimizer-step budget "
            "(--steps) is unchanged but each step sees K times more "
            "training signal. When the batch's shared unpadded prefix "
            "is too short for K non-overlapping windows, fewer are "
            "used (K is a target, not a guarantee). See gh#382."
        ),
    )
    dflash_prepare_p.add_argument(
        "--use-precomputed",
        type=str,
        default=None,
        help=(
            "Read (input_ids, target_hidden) shards from this directory "
            "instead of running the target each step. Produced by "
            "`olmlx dflash precompute`."
        ),
    )

    dflash_precompute_p = dflash_sub.add_parser(
        "precompute",
        help="Precompute target hidden states for DFlash draft training",
    )
    dflash_precompute_p.add_argument("model", help="Target model name or HF path")
    dflash_precompute_p.add_argument(
        "--data", type=str, default=None, help="HuggingFace dataset path"
    )
    dflash_precompute_p.add_argument(
        "--split", type=str, default=None, help="Dataset split"
    )
    dflash_precompute_p.add_argument(
        "--shards",
        type=int,
        default=500,
        help="Number of shards to write (default: 500)",
    )
    dflash_precompute_p.add_argument(
        "--batch-size", type=int, default=4, help="Batch size (default: 4)"
    )
    dflash_precompute_p.add_argument(
        "--seq-len", type=int, default=2048, help="Sequence length (default: 2048)"
    )
    dflash_precompute_p.add_argument(
        "--num-target-layers",
        type=int,
        default=5,
        help="Number of target hidden states to extract (default: 5, per paper)",
    )
    dflash_precompute_p.add_argument(
        "--target-layer-ids",
        type=str,
        default=None,
        help="Comma-separated target layer indices (overrides --num-target-layers)",
    )
    dflash_precompute_p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: <target-model-dir>/dflash_cache)",
    )

    # EAGLE draft training (arxiv 2401.15077)
    eagle = sub.add_parser(
        "eagle", help="EAGLE autoregressive speculative draft training"
    )
    eagle_sub = eagle.add_subparsers(dest="eagle_command")
    eagle_prepare_p = eagle_sub.add_parser(
        "prepare", help="Train an EAGLE draft model for a target"
    )
    eagle_prepare_p.add_argument("model", help="Target model name or HF path")
    eagle_prepare_p.add_argument(
        "--use-precomputed",
        type=str,
        required=True,
        help=(
            "Directory of (input_ids, target_hidden) shards produced by "
            "`olmlx dflash precompute`. EAGLE consumes the deepest captured "
            "layer; the same shards work for both DFlash and EAGLE."
        ),
    )
    eagle_prepare_p.add_argument(
        "--steps", type=int, default=2000, help="Training steps (default: 2000)"
    )
    eagle_prepare_p.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help=(
            "Batch size (default: 4). Under --use-precomputed this is "
            "validated against the shard layout (shards are written at a "
            "fixed batch shape; pass the value that matches or rerun "
            "`olmlx dflash precompute` at the desired batch size)."
        ),
    )
    eagle_prepare_p.add_argument(
        "--seq-len",
        type=int,
        default=2048,
        help=(
            "Sequence length (default: 2048). Under --use-precomputed "
            "this is validated against the shard layout (shards are "
            "written at a fixed sequence length; pass the value that "
            "matches or rerun precompute at the desired length)."
        ),
    )
    eagle_prepare_p.add_argument(
        "--block-size",
        type=int,
        default=4,
        help=(
            "Number of draft tokens per verify (default: 4). Note: each "
            "drafted token forces one Metal command-buffer flush via "
            "``.item()`` (autoregressive feature-space drafting needs the "
            "integer token id for the next iteration). On Apple Silicon "
            "that's ~0.5–1 ms per flush, so block_size=4 adds ~2–4 ms of "
            "sync overhead per verify before the target's parallel forward. "
            "If real-bench acceptance is low for your draft, smaller "
            "block_size (1 or 2) may be Pareto-optimal — it halves/quarters "
            "the sync overhead and avoids deeper compounding-error positions."
        ),
    )
    eagle_prepare_p.add_argument(
        "--num-hidden-layers",
        type=int,
        default=1,
        help="Draft model decoder layer count (default: 1, EAGLE-1 default)",
    )
    eagle_prepare_p.add_argument(
        "--lr", type=float, default=5e-4, help="Peak learning rate (default: 5e-4)"
    )
    eagle_prepare_p.add_argument(
        "--sample-positions",
        type=int,
        default=256,
        help=(
            "Per-step subsample of positions where lm_head is applied "
            "during loss computation. The full sequence still runs through "
            "draft self-attention; only the final vocab projection is "
            "subsampled. Set to 0 to disable subsampling and score every "
            "position (~10x slower on large vocabs). Default: 256."
        ),
    )
    eagle_prepare_p.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "PRNG seed for reproducible training. Seeds both mlx (weight "
            "init etc.) and stdlib random (the per-step position subsample "
            "used by sample-positions). Default: 0."
        ),
    )
    eagle_prepare_p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: <target-model-dir>/eagle)",
    )

    # Spectral quant calibration
    spectral = sub.add_parser("spectral", help="SpectralQuant KV cache compression")
    spectral_sub = spectral.add_subparsers(dest="spectral_command")

    spectral_prepare_p = spectral_sub.add_parser(
        "prepare", help="Run spectral calibration for a model"
    )
    spectral_prepare_p.add_argument("model", help="Model name or HF path")
    spectral_prepare_p.add_argument(
        "--samples",
        type=int,
        default=256,
        help="Number of calibration samples (default: 256)",
    )
    spectral_prepare_p.add_argument(
        "--avg-bits",
        type=int,
        default=4,
        choices=[2, 4],
        help="Target average bits per dimension (default: 4)",
    )
    spectral_prepare_p.add_argument(
        "--calibration-dataset",
        type=str,
        default=None,
        help="Calibration dataset: 'c4' (default) or 'synthetic'",
    )
    spectral_prepare_p.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Max tokens to collect per head (default: 8192)",
    )

    # Shard quant calibration (#377)
    shard = sub.add_parser("shard", help="Shard KV cache compression")
    shard_sub = shard.add_subparsers(dest="shard_command")

    shard_prepare_p = shard_sub.add_parser(
        "prepare", help="Run shard calibration for a model"
    )
    shard_prepare_p.add_argument("model", help="Model name or HF path")
    shard_prepare_p.add_argument(
        "--samples",
        type=int,
        default=256,
        help="Number of calibration samples (default: 256)",
    )
    shard_prepare_p.add_argument(
        "--bits",
        type=int,
        default=4,
        choices=[2, 4, 8],
        help="Bits per dimension for K and V (default: 4)",
    )
    shard_prepare_p.add_argument(
        "--calibration-dataset",
        type=str,
        default=None,
        help="Calibration dataset: 'c4' (default) or 'synthetic'",
    )
    shard_prepare_p.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Max tokens to collect per head (default: 8192)",
    )
    shard_prepare_p.add_argument(
        "--k-energy",
        type=float,
        default=0.999,
        help=(
            "Fraction of eigenvalue energy the kept K rank must capture "
            "(default: 0.999). Lower trades K quality for bytes."
        ),
    )

    # Bench (benchmarking)
    bench = sub.add_parser("bench", help="Benchmarking and functional tests")
    bench_sub = bench.add_subparsers(dest="bench_command")

    bench_run = bench_sub.add_parser("run", help="Run benchmark scenarios")
    bench_run.add_argument(
        "--model",
        default="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        help="Model name or HF path (default: mlx-community/Qwen2.5-0.5B-Instruct-4bit)",
    )
    bench_run.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Override max tokens for all prompts",
    )
    bench_run.add_argument(
        "--scenarios",
        type=str,
        default=None,
        help="Comma-separated scenario names (default: all)",
    )
    bench_run.add_argument(
        "--prompt-set",
        choices=["throughput", "quality", "all"],
        default="throughput",
        help=(
            "Which prompts to run: 'throughput' (7 tok/s probes, default), "
            "'quality' (GSM8K+MMLU+HumanEval graded sets), or 'all'"
        ),
    )
    bench_run.add_argument(
        "--enable-code-exec",
        action="store_true",
        help=(
            "Run model-generated code for the HumanEval code_exec grader in a "
            "resource-limited subprocess (off by default; only the local user's "
            "chosen model produces this code)"
        ),
    )
    bench_run.add_argument(
        "--bench-dir",
        "--output-dir",
        dest="bench_dir",
        type=str,
        default=None,
        help="Directory to save the run in (default: ~/.olmlx/bench/runs)",
    )

    bench_compare = bench_sub.add_parser("compare", help="Compare two benchmark runs")
    bench_compare.add_argument("run1", help="First run (timestamp or path)")
    bench_compare.add_argument("run2", help="Second run (timestamp or path)")

    bench_list = bench_sub.add_parser("list", help="List past benchmark runs")
    bench_list.add_argument(
        "--bench-dir",
        type=str,
        default=None,
        help="Directory to read runs from (default: ~/.olmlx/bench/runs)",
    )

    bench_lb = bench_sub.add_parser(
        "leaderboard", help="Show model leaderboard derived from past runs"
    )
    bench_lb.add_argument(
        "--all-runs",
        action="store_true",
        help=(
            "Show every run instead of latest per model. Use this if a "
            "recent regression run has displaced an earlier faster result."
        ),
    )
    bench_lb.add_argument(
        "--limit",
        type=_positive_int,
        default=None,
        help="Limit rows (default: all); must be >= 1",
    )
    bench_lb.add_argument(
        "--bench-dir",
        type=str,
        default=None,
        help="Directory to read runs from (default: ~/.olmlx/bench/runs)",
    )

    cfg = sub.add_parser("config", help="Show configuration")
    cfg_sub = cfg.add_subparsers(dest="config_command")
    cfg_sub.add_parser("show", help="Show current configuration")

    return parser
