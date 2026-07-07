import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

#: Metal sync behavior at inference-lock boundaries. Single source of truth
#: shared by ``Settings.sync_mode``, ``ModelConfig.sync_mode``, and
#: ``_lock_boundary_sync`` — keep them in lockstep.
SyncMode = Literal["full", "minimal", "none"]


def validate_weight_quant_format(v: str | None) -> str | None:
    """Validate that *v* is a valid ``weight_quant`` value.

    Called by both the Pydantic field validator and the per-model config
    path (``ModelConfig.from_entry``).
    """
    if v is None:
        return v
    parts = v.split(":")
    if len(parts) < 2 or len(parts) > 3:
        raise ValueError(
            f"Invalid weight_quant={v!r}. "
            f"Expected '<method>:<bits>' or '<method>:<bits>:<group_size>'."
        )
    method, bits = parts[0], parts[1]
    if method != "hqq":
        raise ValueError(f"Invalid weight_quant method={method!r}. Expected 'hqq'.")
    if bits not in ("4", "8"):
        raise ValueError(f"Invalid weight_quant bits={bits!r}. Expected '4' or '8'.")
    if len(parts) == 3:
        try:
            gs = int(parts[2])
        except ValueError:
            raise ValueError(
                f"Invalid weight_quant group_size={parts[2]!r}. "
                f"Expected a positive integer."
            )
        if gs < 32 or gs % 32 != 0:
            raise ValueError(
                f"Invalid weight_quant group_size={gs}. "
                f"Expected a multiple of 32 (e.g. 32, 64, 128)."
            )
    return v


def validate_kv_cache_quant_format(v: str | None) -> str | None:
    """Validate that *v* is a valid ``kv_cache_quant`` value.

    Called by both the Pydantic field validator and the per-model config
    path (``ModelConfig.from_entry``) so bad values are caught at config
    load time regardless of whether they come from an env var or a
    ``models.json`` entry.
    """
    if v is None:
        return v
    _VALID_BITS_BY_METHOD = {
        "turboquant": {"2", "4"},
        "spectral": {"2", "4"},
        "shard": {"2", "4", "8"},
    }
    parts = v.split(":", 1)
    if (
        len(parts) != 2
        or parts[0] not in _VALID_BITS_BY_METHOD
        or parts[1] not in _VALID_BITS_BY_METHOD[parts[0]]
    ):
        raise ValueError(
            f"Invalid kv_cache_quant={v!r}. "
            f"Expected '<method>:<bits>' where method:bits is one of "
            f"turboquant:{{2,4}}, spectral:{{2,4}}, shard:{{2,4,8}}."
        )
    return v


def validate_kv_eviction_format(v: str | None) -> str | None:
    """Validate a ``kv_eviction`` value of the form ``'<sink>:<window>'`` (#505).

    StreamingLLM-style eviction keeps the first ``sink`` tokens (attention
    sinks) plus the last ``window`` tokens, dropping the middle. ``sink >= 0``,
    ``window >= 1``. Validated for both the env var and ``models.json`` paths.
    """
    if v is None:
        return v
    parts = v.split(":", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Invalid kv_eviction={v!r}. Expected '<sink>:<window>' (e.g. '4:512')."
        )
    try:
        sink, window = int(parts[0]), int(parts[1])
    except ValueError:
        raise ValueError(
            f"Invalid kv_eviction={v!r}. Both sink and window must be integers "
            f"(e.g. '4:512')."
        ) from None
    if sink < 0 or window < 1:
        raise ValueError(
            f"Invalid kv_eviction={v!r}. Require sink >= 0 and window >= 1."
        )
    return v


class Settings(BaseSettings):
    # Note: ``validate_assignment=True`` applies to *all* fields, not just
    # the new speculative ones. Programmatic writes that previously
    # silently set out-of-range values (e.g. ``settings.port = 0``) now
    # raise ``ValidationError``. This is intentional — tests and CLI
    # overrides should not be able to construct invalid Settings — but
    # it does broaden the surface beyond the immediate use case.
    model_config = {
        "env_prefix": "OLMLX_",
        "env_file": ".env",
        "extra": "ignore",
        "validate_assignment": True,
    }

    host: str = "127.0.0.1"
    port: Annotated[int, Field(ge=1, le=65535)] = 11434
    models_dir: Path = Path.home() / ".olmlx" / "models"
    models_config: Path = Path.home() / ".olmlx" / "models.json"
    default_keep_alive: str = "5m"
    # Max models resident at once. A LoRA adapter (``base:adapter``, issue #362)
    # pins its base, so serving a base plus N adapters concurrently needs this
    # set to at least N+1 (the base and each adapter each occupy one slot).
    max_loaded_models: int = 1
    memory_limit_fraction: Annotated[float, Field(gt=0, le=1.0)] = 0.75
    # Fraction of system RAM reserved below ``memory_limit_fraction`` for the
    # KV cache and activations that allocate on top of model weights during
    # decode.  The model-load admission check uses an effective weight budget
    # of ``memory_limit_fraction - inference_headroom_fraction``.  Default 0.0
    # preserves the legacy weights-only check; raise it (e.g. 0.1) on machines
    # where a model that loads near the limit then swaps mid-generation
    # (issue #223).
    inference_headroom_fraction: Annotated[float, Field(ge=0, lt=1.0)] = 0.0
    model_load_timeout: Annotated[float, Field(gt=0)] | None = None
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    # OpenTelemetry tracing master switch (OLMLX_TRACING). Default off.
    # When off, nothing under olmlx.utils.tracing imports opentelemetry, so
    # there is no import-time or per-request cost. All endpoint/protocol/
    # headers/sampling/service-name configuration comes from the native
    # OTEL_* env vars the OTLP exporter and SDK already honor
    # (OTEL_EXPORTER_OTLP_ENDPOINT, OTEL_EXPORTER_OTLP_PROTOCOL,
    # OTEL_TRACES_SAMPLER, OTEL_SERVICE_NAME, OTEL_TRACES_EXPORTER=console, …).
    tracing: bool = False
    prompt_cache: bool = True
    # Number of per-cache_id VLM KV-cache slots retained for cross-turn image-prefix
    # reuse (mlx_vlm PromptCacheState). 0 disables VLM prompt caching entirely.
    # 2 slots bound the in-memory tier; cold entries spill to disk when
    # vlm_prompt_cache_disk is on (#491).
    vlm_prompt_cache_slots: int = 2
    # Disk spill for the VLM prompt cache (#491): mirrors the text-path
    # prompt_cache_disk_* knobs. Off by default — the common single-conversation
    # case is fully covered by the in-memory slots.
    vlm_prompt_cache_disk: bool = False
    vlm_prompt_cache_disk_path: Path = Path.home() / ".olmlx" / "cache" / "vlm"
    vlm_prompt_cache_disk_max_gb: Annotated[float, Field(gt=0)] = 10.0
    prompt_cache_max_tokens: Annotated[int, Field(gt=0)] | None = 32768
    prompt_cache_max_slots: Annotated[int, Field(gt=0)] = 4
    prompt_cache_disk: bool = False
    prompt_cache_disk_path: Path = Path.home() / ".olmlx" / "cache" / "kv"
    prompt_cache_disk_max_gb: Annotated[float, Field(gt=0)] = 10.0
    audio_max_bytes: Annotated[
        int,
        Field(
            gt=0,
            description="Max upload size for /v1/audio/transcriptions (OLMLX_AUDIO_MAX_BYTES).",
        ),
    ] = 100 * 1024 * 1024
    tts_max_input_chars: Annotated[
        int,
        Field(
            gt=0,
            description="Max input length for /v1/audio/speech (OLMLX_TTS_MAX_INPUT_CHARS).",
        ),
    ] = 8192
    # Voice mode for `olmlx chat --voice` (issue #444). STT reuses the Whisper
    # ModelManager kind; TTS reuses the Kokoro `tts` kind + streaming
    # generate_speech from the /v1/audio/speech work (#367). Models resolve
    # through the normal registry/store, so these accept Ollama names or HF repos.
    chat_stt_model: str = "mlx-community/whisper-large-v3-turbo"
    chat_tts_model: str = "prince-canuma/Kokoro-82M"
    chat_tts_voice: str = "af_heart"
    # Cross-request radix prefix cache (issue #365). When enabled, a
    # cache_id miss falls back to a token-prefix lookup over the in-memory
    # store; on hit, the matched entry is re-keyed to the new cache_id
    # (takeover semantics — no KV copy). The old cache_id loses its entry.
    prompt_cache_radix: bool = True
    # Soft RAM byte budget for the in-memory tier. Best-effort estimate;
    # slot count (prompt_cache_max_slots) is the hard cap.
    prompt_cache_ram_budget_gb: Annotated[float, Field(gt=0)] = 8.0
    # Below this token count, a radix-prefix hit falls back to fresh
    # prefill rather than taking over a near-empty match.
    prompt_cache_radix_min_prefix_tokens: Annotated[int, Field(ge=0)] = 256
    # Max number of stored Responses-API responses for previous_response_id
    # continuation (in-memory LRU; lost on restart).
    responses_store_max: Annotated[int, Field(gt=0)] = 256
    inference_queue_timeout: Annotated[float, Field(gt=0)] | None = 300.0
    inference_timeout: Annotated[float, Field(gt=0)] | None = None
    sync_mode: SyncMode = "full"
    max_tokens_limit: Annotated[int, Field(gt=0)] = 131072
    # Default generation length when a request omits max_tokens /
    # num_predict / max_completion_tokens / max_output_tokens. Shared by
    # every router (Ollama, OpenAI chat/completions, Responses) so the
    # fallback is configured in one place rather than hardcoded per route.
    default_max_tokens: Annotated[int, Field(gt=0)] = 512
    cors_origins: list[str] = ["http://localhost:*", "http://127.0.0.1:*"]
    anthropic_models: dict[str, str] = {}

    # KV cache quantization (TurboQuant, SpectralQuant, or Shard).
    # Format: "<method>:<bits>" where method ∈ {turboquant, spectral, shard};
    # bits ∈ {2, 4} for turboquant/spectral, {2, 4, 8} for shard. Per-model
    # overrides live on ``ModelConfig`` in ``olmlx.engine.registry``.
    kv_cache_quant: str | None = None

    # StreamingLLM-style sink+window KV eviction (#505): '<sink>:<window>'
    # keeps the first <sink> tokens + last <window> tokens and drops the middle,
    # bounding KV *count* (per-step attention compute + memory) for long
    # contexts. Lossy and opt-in; applied only to pure full-attention models
    # (RotatingKVCache under the hood). Takes effect when kv_cache_quant is
    # unset. Per-model overrides live on ``ModelConfig``.
    kv_eviction: str | None = None

    # Fused Metal decode path for shard KV quant (#377 Tier 2): attention
    # over the compressed middle is computed from the packed form (no FP16
    # middle materialization). Kill switch; unsupported configurations fall
    # back to the Tier-1 dequantize-on-read path automatically.
    shard_fused: bool = True

    # Auto-run spectral/shard calibration when the configured KV quant
    # needs calibration data that is missing. Uses the default calibration
    # dataset (c4) and 64 samples. Set to ``true`` to avoid manual
    # ``olmlx {spectral,shard} prepare <model>`` on first load.
    kv_cache_auto_calibrate: bool = False

    # Weight quantization (HQQ).
    # Format: "hqq:<bits>" or "hqq:<bits>:<group_size>" where bits ∈ {4, 8}
    # and group_size is a positive integer (default 64 for 4-bit, 128 for
    # 8-bit). Per-model overrides live on ``ModelConfig`` in
    # ``olmlx.engine.registry``.
    weight_quant: str | None = None

    # Distributed inference — split models across Apple Silicon machines.
    # Configured via hostfile (``distributed_hostfile``), launched by
    # ``olmlx serve``.
    distributed: bool = False
    # Distributed inference is tensor-only. The dormant pipeline.py /
    # pre_shard_pipeline / worker pipeline branches are unreachable and
    # kept for a future PR (#273).
    distributed_strategy: Literal["tensor"] = "tensor"
    distributed_hostfile: Path = Path("~/.olmlx/hostfile.json")
    distributed_backend: str = "ring"
    distributed_port: int = 32323
    distributed_sideband_port: int = 32400
    distributed_secret: str = ""
    distributed_remote_working_dir: str = ""
    distributed_remote_python: str = "python"
    distributed_pre_shard: bool = True
    distributed_shard_dir: Path = Path("~/.olmlx/shards")
    distributed_worker_shard_dir: str = "~/.olmlx/shards"

    # Speculative decoding (works with any model, not just Flash).
    # Per-model overrides live on ``ModelConfig`` in ``olmlx.engine.registry``.
    # ``min_length=1`` rejects ``OLMLX_SPECULATIVE_DRAFT_MODEL=""`` at parse
    # time so the load process doesn't surface a misleading "draft not set"
    # error for an empty string.
    #
    # ``speculative_strategy`` selects the algorithm:
    # - ``classic``: standalone draft LM (autoregressive lambda candidates).
    # - ``dflash``: block-diffusion draft conditioned on target hidden states
    #   (mask-token parallel block prediction). Requires a draft model whose
    #   ``config.json`` carries the upstream z-lab ``dflash_config`` schema.
    # - ``eagle``: autoregressive draft head conditioned on the target's
    #   last-layer hidden state (arxiv 2401.15077). Predicts the next
    #   *hidden* in feature space; the target's lm_head (shared via
    #   ``bind()``) maps that hidden to the next-token distribution. Higher
    #   acceptance rate than DFlash for the same draft-parameter budget on
    #   most targets, at the cost of being autoregressive (one draft
    #   forward per drafted token). Requires a draft whose ``config.json``
    #   carries an ``eagle_config`` block.
    # - ``pld``: prompt-lookup decoding. No draft model — the "draft" comes
    #   from n-gram lookup in the prompt+generated history. Free acceptance
    #   on code edits, repeated context, and JSON/structured replies.
    # - ``self_speculative``: LayerSkip-style decoding — uses the target's
    #   own early layers (0..L-skip-1) as an autoregressive draft, then
    #   verifies with all L layers in one forward pass. No external draft
    #   model required. Works under Flash and Flash-MoE.
    #   Configured with ``speculative_layers_skip`` (default: L//4).
    # Both ``pld`` and ``self_speculative`` compose with Flash-MoE.
    # ``speculative_tokens`` is reused as DFlash's ``block_size``, EAGLE's
    # per-verify draft-token count, PLD's max draft length, and
    # self_speculative's draft token count. ``None`` means "use the
    # strategy default": 4 for classic and self_speculative, 10 for PLD,
    # the draft model's pre-trained ``block_size`` for DFlash and EAGLE.
    speculative: bool = False
    speculative_strategy: Literal[
        "classic",
        "dflash",
        "eagle",
        "pld",
        "lookahead",
        "self_speculative",
        "proxy_tuning",
    ] = "classic"
    speculative_draft_model: Annotated[str, Field(min_length=1)] | None = None
    speculative_tokens: Annotated[int, Field(gt=0)] | None = None
    #: PLD-only knobs (ignored by other strategies). N-gram sizes for the
    #: prompt-lookup search: try sizes from ``max`` down to ``min``,
    #: returning the first match found. Smaller n-grams match more often
    #: but yield noisier drafts; larger n-grams are more precise.
    speculative_pld_max_ngram: Annotated[int, Field(gt=0)] = 3
    speculative_pld_min_ngram: Annotated[int, Field(gt=0)] = 1
    #: Cap the lookup history (most recent N tokens) so per-step search
    #: cost doesn't grow unbounded with context length. Default 8192 is
    #: roughly the largest history where pure-Python n-gram scan stays
    #: well under one forward pass on Apple Silicon (~30 ms ceiling on
    #: max_ngram=3); raise it if you need matches against an older
    #: prefix, lower it if you're seeing scan-time regressions.
    speculative_pld_lookup_window: Annotated[int, Field(gt=0)] = 8192
    #: Self-speculative knob — number of layers skipped during the draft
    #: pass. ``None`` defaults to L//4 at load time.
    speculative_layers_skip: Annotated[int, Field(ge=1)] | None = None
    #: Proxy-tuning (engine/proxy_tuning.py) model paths + steering strength.
    #: Only read when ``speculative_strategy == "proxy_tuning"``. The expert
    #: (``M+``, tuned) and anti-expert (``M-``, untuned) are small models that
    #: must share the base model's exact tokenizer/vocabulary. ``alpha`` scales
    #: the tuning delta ``(expert - antiexpert)``; 1.0 is the paper's default.
    speculative_proxy_expert_model: Annotated[str, Field(min_length=1)] | None = None
    speculative_proxy_antiexpert_model: Annotated[str, Field(min_length=1)] | None = (
        None
    )
    speculative_proxy_alpha: float = 1.0

    #: Strict quantization compatibility for speculative decoding (issue #516).
    #: When True, a mismatch between the draft's recorded target_quant and the
    #: live target's detected quantization raises RuntimeError instead of just
    #: emitting a warning. Default False (warn only).
    spec_strict_compat: bool = False

    #: Cross-request KV-cache reuse for speculative decoding (issue #421).
    #: Max persisted speculative cache lineages held on the per-model
    #: decoder. Each slot is a full *live* target (+draft) KV snapshot — a
    #: materialized deepcopy, not a page-mapped reference, multi-GB for a 27B
    #: target — so the default is deliberately small. A reuse hit briefly
    #: holds an extra working copy (deepcopy of the stored snapshot) on top of
    #: the resident slots; this peak is not counted against
    #: ``OLMLX_MEMORY_LIMIT_FRACTION``, so raise this only with headroom to
    #: spare. ``0`` disables reuse entirely (fresh prefill every turn — the
    #: pre-#421 behavior, useful as a kill switch). Only the ``classic`` and
    #: ``pld`` strategies honor this; ``dflash``/``eagle`` always fresh-prefill.
    speculative_cache_slots: Annotated[int, Field(ge=0)] = 2

    # Tree-structured speculative verification (#358).  When enabled,
    # the classic speculative strategy produces a tree of draft alternatives
    # (top-K candidates per step) and verifies them against the target in
    # one forward pass using a sparse attention mask.
    #
    # ``tree_width`` controls how many candidates are kept per draft step
    # (1 = linear, ≥2 = tree).  ``tree_max_nodes`` is a hard cap on the
    # total number of tree nodes (including the root).  The tree automatically
    # stops growing when it hits either the draft length (``speculative_tokens``)
    # or ``tree_max_nodes``.
    #
    # Setting ``tree_width`` to 1 or ``tree_speculative`` to False falls
    # back to the existing linear verification path.
    tree_speculative: bool = False
    tree_width: Annotated[int, Field(ge=1)] = 2
    tree_max_nodes: Annotated[int, Field(ge=3)] = 8

    # Continuous batching of concurrent text-chat requests via mlx-lm's
    # BatchGenerator (docs/batching-plan.md). Opt-in; eligible requests
    # (plain-KVCache text LLMs — no VLM/speculative/KV-quant/grammar/seed)
    # join a per-model batch instead of queueing on the inference lock.
    # ``batch_completion_size`` caps concurrent decoding sequences;
    # ``batch_prefill_size`` caps sequences in chunked prefill;
    # ``batch_prefill_step`` is the prefill chunk size (matches
    # inference.py's _PREFILL_CHUNK).
    batching: bool = False
    batch_completion_size: Annotated[int, Field(ge=1)] = 8
    batch_prefill_size: Annotated[int, Field(ge=1)] = 4
    batch_prefill_step: Annotated[int, Field(ge=1)] = 2048
    # Backpressure (plan §11): a batched sequence whose SSE consumer falls
    # more than this many events behind the worker is dropped from the
    # batch — a stalled-but-connected client otherwise pins a slot and
    # decodes to max_tokens unread (the unbounded per-sequence event queue
    # has no flow control of its own). 0 disables the cut-off.
    batch_consumer_lag_limit: Annotated[int, Field(ge=0)] = 2048
    # Aggregate (cross-sequence) KV admission (plan §10). When on, the batch
    # worker only admits a queued sequence while the batch's projected KV
    # stays under ``memory_limit_fraction`` of system RAM (per-sequence KV
    # estimate vs current Metal headroom); an over-budget request waits in
    # the inbox until a co-tenant frees its slot, instead of all admitting at
    # once and risking an uncatchable Metal OOM. Per-request admission (one
    # request's own KV) is always enforced; this gates the cross-sequence
    # sum. Disable to fall back to admit-all (per-request preflight only).
    batch_kv_admission: bool = True
    # Fairness quantum (seconds). When an exclusive (non-batched) request
    # starts waiting on the inference lock, the batch worker normally stops
    # admitting new sequences at once and drains so the waiter isn't starved.
    # Under interleaved mixed traffic that collapses the batch to one
    # sequence. This quantum guarantees the batch at least this many seconds
    # of admission-open service per busy period before it latches the pause —
    # a throughput floor that bounds the *extra* admission window imposed on
    # the waiting exclusive request to `batch_fairness_quantum`. 0.0 (default)
    # = latch immediately (today's behavior). Per-model overridable.
    # ``allow_inf_nan=False``: a non-finite quantum would make ``held >=
    # quantum`` never true, silently disabling the fairness latch and
    # starving exclusive requests (Field(ge=0) alone admits +inf).
    batch_fairness_quantum: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0.0

    # Flash inference (LLM in a Flash). Primary, user-facing knobs.
    # Advanced tuning (window size, IO threads, cache budget, predictor
    # rank, buffer modes) lives on ``ExperimentalSettings`` and the
    # ``olmlx flash prepare`` CLI — these five fields are the ones a
    # typical user needs to touch. Per-model overrides live on
    # ``ModelConfig`` in ``olmlx.engine.registry``.
    flash: bool = False
    flash_sparsity_threshold: Annotated[float, Field(gt=0, le=1.0)] = 0.5
    flash_min_active_neurons: Annotated[int, Field(gt=0)] = 128
    flash_max_active_neurons: Annotated[int, Field(gt=0)] | None = None
    flash_memory_budget_fraction: Annotated[float, Field(gt=0, le=1.0)] | None = None

    # Flash MoE (SSD-based expert offloading for MoE models)
    flash_moe: bool = False
    flash_moe_cache_budget_experts: Annotated[int, Field(ge=0)] = 48
    flash_moe_io_threads: Annotated[int, Field(gt=0)] = 32

    # Flash prefetch — promoted toggle. Advanced prefetch tuning
    # (confidence_threshold, min/max_neurons, io_threads) stays on
    # ``ExperimentalSettings``. Controls both runtime prefetch and whether
    # ``olmlx flash prepare`` trains the LookaheadBank.
    flash_prefetch: bool = False

    # Flash + speculative decoding (SpeculativeFlashDecoder). Per-model
    # overrides live on ``ModelConfig`` in ``olmlx.engine.registry``.
    flash_speculative: bool = False
    flash_speculative_draft_model: Annotated[str, Field(min_length=1)] | None = None
    flash_speculative_tokens: Annotated[int, Field(gt=0)] = 4

    # AWQ / GPTQ auto-conversion on pull (issue #363).
    # When detect_format() identifies a downloaded model as AWQ or GPTQ,
    # olmlx re-quantizes it to MLX int4/int8 via mlx_lm.convert.  The
    # converted artifact lands in a sibling directory; the original is deleted
    # when awq_gptq_remove_source=True.
    # Peak disk: ~2× model size during conversion (src + dst coexist).
    # Peak RAM: BF16 weights in flight (~2 bytes/param) — large models OOM
    # on 16-24 GB devices; see issue #363 risks.
    # Restricted to the values mlx's quantizer actually accepts — bits 1 and 7
    # and group sizes like 16/256 pass a naive range check but crash conversion.
    awq_gptq_convert_bits: Literal[2, 3, 4, 5, 6, 8] = 4
    awq_gptq_convert_group_size: Literal[32, 64, 128] = 64
    awq_gptq_remove_source: bool = True

    # Autonomous agent (engine/agent/, routers/agent.py — issue #445). The
    # orchestrator drives the existing ``ChatSession`` ReAct loop across many
    # turns toward a goal, with hard budgets, stall detection, SQLite-persisted
    # resumable runs, cross-session memory, self-improving skills, and bounded
    # subagent delegation. Gated off by default; the HTTP surface and run
    # registry only exist when ``agent_enabled`` is true.
    agent_enabled: bool = False
    agent_db_path: Path = Path.home() / ".olmlx" / "agent.db"
    #: Where learned skills (Phase 3) are written; shared with the chat skill
    #: library by default so the agent's self-authored skills are reusable.
    agent_skills_dir: Path = Path.home() / ".olmlx" / "skills"
    #: Default model for agent runs when a create request omits ``model``.
    #: Empty string means the request must supply one (else HTTP 422).
    agent_model: str = ""
    #: Hard budgets enforced by the orchestrator regardless of model output.
    #: ``token_budget`` / ``wallclock_timeout`` of ``None`` mean unlimited.
    agent_max_iterations: Annotated[int, Field(gt=0)] = 50
    agent_token_budget: Annotated[int, Field(gt=0)] | None = None
    agent_wallclock_timeout: Annotated[float, Field(gt=0)] | None = None
    #: Abort a run after this many consecutive iterations make no progress
    #: (identical assistant output / no new memory). Run-level extension of
    #: ``ChatSession``'s repetition + consecutive-tool-failure guards.
    agent_stall_max_no_progress: Annotated[int, Field(gt=0)] = 3
    #: Per-iteration max ReAct turns inside the wrapped ``ChatSession`` — kept
    #: small so ``finish`` / budgets are checked often at the outer loop.
    agent_inner_max_turns: Annotated[int, Field(gt=0)] = 8
    #: Cross-session memory (Phase 2) bounds.
    agent_memory_max_entries: Annotated[int, Field(gt=0)] = 1000
    agent_memory_recall_k: Annotated[int, Field(gt=0)] = 5
    #: Subagent delegation (Phase 4) bounds. ``depth`` 0 disables delegation
    #: (root run is depth 0; a child would be depth 1 > depth cap of 0).
    agent_max_subagent_depth: Annotated[int, Field(ge=0)] = 2
    agent_max_subagent_fanout: Annotated[int, Field(gt=0)] = 4

    @model_validator(mode="after")
    def validate_auto_calibrate(self) -> "Settings":
        if self.kv_cache_auto_calibrate and (
            self.kv_cache_quant is None
            or not self.kv_cache_quant.startswith(("spectral:", "shard:"))
        ):
            raise ValueError(
                "OLMLX_KV_CACHE_AUTO_CALIBRATE=true requires "
                "OLMLX_KV_CACHE_QUANT=spectral:<bits> or shard:<bits>"
            )
        return self

    @property
    def effective_load_budget_fraction(self) -> float:
        """Fraction of system RAM a model's weights may occupy at load time.

        Single source of truth shared by the model-load admission check and
        the pre-load memory-pressure / idle-eviction trigger so the two never
        drift: ``memory_limit_fraction`` minus the inference headroom reserve.
        ``max(0.0, ...)`` is defensive — ``validate_inference_headroom``
        already guarantees headroom < limit, so this is strictly positive.
        """
        return max(0.0, self.memory_limit_fraction - self.inference_headroom_fraction)

    @model_validator(mode="after")
    def validate_inference_headroom(self) -> "Settings":
        # The load budget is ``memory_limit_fraction - inference_headroom_fraction``.
        # If headroom >= limit the effective budget is <= 0, which would
        # silently reject every model load at request time. Fail fast at
        # startup with an actionable message instead.
        if self.inference_headroom_fraction >= self.memory_limit_fraction:
            raise ValueError(
                f"inference_headroom_fraction ({self.inference_headroom_fraction}) "
                f"must be < memory_limit_fraction ({self.memory_limit_fraction}); "
                f"otherwise the effective load budget is zero and every model "
                f"load fails. Lower OLMLX_INFERENCE_HEADROOM_FRACTION or raise "
                f"OLMLX_MEMORY_LIMIT_FRACTION."
            )
        return self

    @model_validator(mode="after")
    def validate_pld_ngram_range(self) -> "Settings":
        # Scope to ``strategy=pld`` so a stray inverted env var
        # (e.g. ``PLD_MIN=5, PLD_MAX=3``) doesn't reject an otherwise
        # valid config that never activates PLD. The per-model
        # resolution in ``ModelConfig.resolved_speculative`` does the
        # equivalent cross-field check when PLD actually loads.
        if self.speculative_strategy != "pld":
            return self
        if self.speculative_pld_min_ngram > self.speculative_pld_max_ngram:
            raise ValueError(
                f"speculative_pld_min_ngram ({self.speculative_pld_min_ngram}) "
                f"must be <= speculative_pld_max_ngram "
                f"({self.speculative_pld_max_ngram})"
            )
        # Parallel to the per-model ``ModelConfig.__post_init__`` check
        # — without this, a bad env pair like ``MAX_NGRAM=3
        # LOOKUP_WINDOW=1`` would pass startup and only blow up at
        # first model load with a confusing stack trace from
        # ``resolved_speculative``.
        if self.speculative_pld_lookup_window < self.speculative_pld_max_ngram:
            raise ValueError(
                f"speculative_pld_lookup_window "
                f"({self.speculative_pld_lookup_window}) must be >= "
                f"speculative_pld_max_ngram "
                f"({self.speculative_pld_max_ngram})"
            )
        return self

    @model_validator(mode="after")
    def validate_tree_speculative(self) -> "Settings":
        if self.tree_speculative and self.flash_speculative:
            raise ValueError(
                "OLMLX_TREE_SPECULATIVE=true and OLMLX_FLASH_SPECULATIVE=true "
                "are incompatible. Tree verification is only supported with "
                "OLMLX_SPECULATIVE=true (classic strategy). Flash speculative "
                "decoding uses a separate code path that does not support tree "
                "drafts (see #358)."
            )
        if self.tree_speculative and not self.speculative:
            raise ValueError(
                "OLMLX_TREE_SPECULATIVE=true requires OLMLX_SPECULATIVE=true"
            )
        if self.tree_speculative and self.speculative_strategy != "classic":
            raise ValueError(
                "OLMLX_TREE_SPECULATIVE=true is only supported with "
                "OLMLX_SPECULATIVE_STRATEGY=classic "
                f"(got {self.speculative_strategy!r})"
            )
        return self

    @model_validator(mode="after")
    def validate_proxy_tuning(self) -> "Settings":
        if self.speculative_strategy != "proxy_tuning":
            return self
        if not self.speculative:
            # Scope the requirement to an actually-enabled proxy-tuning config,
            # mirroring how the PLD validator avoids rejecting inert settings.
            return self
        missing = [
            name
            for name, val in (
                ("speculative_proxy_expert_model", self.speculative_proxy_expert_model),
                (
                    "speculative_proxy_antiexpert_model",
                    self.speculative_proxy_antiexpert_model,
                ),
            )
            if not val
        ]
        if missing:
            env_hints = " / ".join(f"OLMLX_{name.upper()}" for name in missing)
            raise ValueError(
                "speculative_strategy='proxy_tuning' requires "
                + " and ".join(missing)
                + f" to be set ({env_hints})."
            )
        if not math.isfinite(self.speculative_proxy_alpha):
            raise ValueError(
                "speculative_proxy_alpha must be a finite number, got "
                f"{self.speculative_proxy_alpha!r}"
            )
        return self

    @model_validator(mode="after")
    def validate_flash_neuron_range(self) -> "Settings":
        # Cross-field check: an inverted min/max would produce a silently
        # broken ``FlashConfig`` at runtime (FlashMLP clamps each token's
        # active neurons into ``[min, max]``, and a min > max collapses
        # the interval). Fail at config load instead.
        if (
            self.flash_max_active_neurons is not None
            and self.flash_min_active_neurons > self.flash_max_active_neurons
        ):
            raise ValueError(
                f"flash_min_active_neurons ({self.flash_min_active_neurons}) "
                f"must be <= flash_max_active_neurons "
                f"({self.flash_max_active_neurons})"
            )
        return self

    @field_validator("kv_cache_quant")
    @classmethod
    def validate_kv_cache_quant(cls, v: str | None) -> str | None:
        return validate_kv_cache_quant_format(v)

    @field_validator("kv_eviction")
    @classmethod
    def validate_kv_eviction(cls, v: str | None) -> str | None:
        return validate_kv_eviction_format(v)

    @field_validator("weight_quant")
    @classmethod
    def validate_weight_quant(cls, v: str | None) -> str | None:
        return validate_weight_quant_format(v)

    @field_validator("speculative_draft_model")
    @classmethod
    def validate_speculative_draft_model(cls, v: str | None) -> str | None:
        # ``Field(min_length=1)`` already rejects ``""``, but a
        # whitespace-only value (``"   "``) has length > 0 and would
        # otherwise reach the load path and surface as a misleading
        # "draft not set" error. Strip and reject empty here.
        if v is None:
            return v
        stripped = v.strip()
        if not stripped:
            raise ValueError(
                "speculative_draft_model must be a non-empty HuggingFace path"
            )
        return stripped

    @field_validator("flash_speculative_draft_model")
    @classmethod
    def validate_flash_speculative_draft_model(cls, v: str | None) -> str | None:
        # ``Field(min_length=1)`` already rejects ``""``, but a
        # whitespace-only value (``"   "``) has length > 0 and would
        # otherwise reach the load path and surface as a misleading
        # "flash draft not set" error. Strip and reject empty here.
        if v is None:
            return v
        stripped = v.strip()
        if not stripped:
            raise ValueError(
                "flash_speculative_draft_model must be a non-empty HuggingFace path"
            )
        return stripped

    @field_validator("anthropic_models")
    @classmethod
    def validate_anthropic_model_keys(cls, v: dict[str, str]) -> dict[str, str]:
        for key in v:
            if "-" in key or ":" in key:
                raise ValueError(
                    f"anthropic_models key {key!r} must be a single segment "
                    "(no dashes or colons)"
                )
        return v


settings = Settings()


class ExperimentalSettings(BaseSettings):
    model_config = {
        "env_prefix": "OLMLX_EXPERIMENTAL_",
        "env_file": ".env",
        "extra": "ignore",
    }

    # Flash inference — advanced/tuning knobs. The primary user-facing
    # fields (``flash``, ``flash_sparsity_threshold``,
    # ``flash_min_active_neurons``, ``flash_max_active_neurons``,
    # ``flash_memory_budget_fraction``) were promoted to ``Settings``;
    # what remains here is rarely-touched tuning that most users should
    # not need to set. Per-model overrides for advanced knobs still go
    # under the ``experimental`` block in models.json. (``distributed_*``
    # was promoted to ``Settings`` in #326 and removed from here too.)
    # ``flash_prefetch`` (toggle) and ``flash_speculative*`` were promoted
    # to ``Settings`` in #275/#276; only the prefetch *tuning* knobs
    # (confidence_threshold, min/max_neurons, io_threads) remain here.
    flash_window_size: Annotated[int, Field(gt=0)] = 5
    flash_io_threads: Annotated[int, Field(gt=0)] = 32
    flash_cache_budget_neurons: Annotated[int, Field(ge=0)] = 1024
    flash_predictor_rank: Annotated[int, Field(gt=0)] = 128  # prepare-time only
    flash_predictor_sensitive_layers: Annotated[int, Field(ge=0)] = 0
    flash_predictor_sensitive_rank_multiplier: Annotated[int, Field(gt=0)] = 4
    flash_bypass_os_cache: bool = False
    flash_preallocated_buffer: bool = False
    flash_prefetch_confidence_threshold: Annotated[float, Field(gt=0, le=1.0)] = 0.3
    flash_prefetch_min_neurons: Annotated[int, Field(gt=0)] = 64
    flash_prefetch_max_neurons: Annotated[int, Field(gt=0)] | None = None
    flash_prefetch_io_threads: Annotated[int, Field(gt=0)] = 16


experimental = ExperimentalSettings()

PRE_SHARDED_DIR_ENV = "OLMLX_DISTRIBUTED_PRE_SHARDED_DIR"

#: Promoted flash env vars: legacy ``OLMLX_EXPERIMENTAL_FLASH*`` name -> the
#: bare ``OLMLX_FLASH*`` name that replaced it (PRs #327/#329/#336). These are
#: no longer honored; ``warn_legacy_flash_env`` detects and warns but does NOT
#: forward them. Only PROMOTED names belong here — the still-valid experimental
#: *tuning* knobs (``OLMLX_EXPERIMENTAL_FLASH_WINDOW_SIZE``, ``..._PREDICTOR_*``,
#: ``..._PREFETCH_CONFIDENCE_THRESHOLD``, ``..._IO_THREADS``, ...) stay valid
#: and are intentionally absent.
PROMOTED_FLASH_ENV_RENAMES: dict[str, str] = {
    "OLMLX_EXPERIMENTAL_FLASH": "OLMLX_FLASH",
    "OLMLX_EXPERIMENTAL_FLASH_SPARSITY_THRESHOLD": "OLMLX_FLASH_SPARSITY_THRESHOLD",
    "OLMLX_EXPERIMENTAL_FLASH_MIN_ACTIVE_NEURONS": "OLMLX_FLASH_MIN_ACTIVE_NEURONS",
    "OLMLX_EXPERIMENTAL_FLASH_MAX_ACTIVE_NEURONS": "OLMLX_FLASH_MAX_ACTIVE_NEURONS",
    "OLMLX_EXPERIMENTAL_FLASH_MEMORY_BUDGET_FRACTION": "OLMLX_FLASH_MEMORY_BUDGET_FRACTION",
    "OLMLX_EXPERIMENTAL_FLASH_MOE": "OLMLX_FLASH_MOE",
    "OLMLX_EXPERIMENTAL_FLASH_MOE_CACHE_BUDGET_EXPERTS": "OLMLX_FLASH_MOE_CACHE_BUDGET_EXPERTS",
    "OLMLX_EXPERIMENTAL_FLASH_MOE_IO_THREADS": "OLMLX_FLASH_MOE_IO_THREADS",
    "OLMLX_EXPERIMENTAL_FLASH_PREFETCH": "OLMLX_FLASH_PREFETCH",
    "OLMLX_EXPERIMENTAL_FLASH_SPECULATIVE": "OLMLX_FLASH_SPECULATIVE",
    "OLMLX_EXPERIMENTAL_FLASH_SPECULATIVE_DRAFT_MODEL": "OLMLX_FLASH_SPECULATIVE_DRAFT_MODEL",
    "OLMLX_EXPERIMENTAL_FLASH_SPECULATIVE_TOKENS": "OLMLX_FLASH_SPECULATIVE_TOKENS",
}


def warn_legacy_flash_env() -> None:
    """Warn that promoted ``OLMLX_EXPERIMENTAL_FLASH*`` env vars are no longer
    honored, naming the new ``OLMLX_FLASH*`` replacement for each.

    Detects names set in the shell environment OR the project ``.env``. Does
    NOT forward the value: these knobs were promoted out of the experimental
    prefix several releases ago (PRs #327/#329/#336) and the one-release
    deprecation window has long closed. Warns on *presence* of the name
    regardless of its value, since the name itself is dead.

    Lives in ``olmlx.config`` so the distributed-worker entry point can reuse
    it without importing ``olmlx.cli`` (and its argparse/uvicorn baggage).
    """
    dotenv_keys = _legacy_names_in_dotenv(tuple(PROMOTED_FLASH_ENV_RENAMES))
    detected = sorted(
        old
        for old in PROMOTED_FLASH_ENV_RENAMES
        if old in os.environ or old in dotenv_keys
    )
    if not detected:
        return
    renames = "; ".join(
        f"{old} -> {PROMOTED_FLASH_ENV_RENAMES[old]}" for old in detected
    )
    logger.warning(
        "Unsupported env vars detected and IGNORED: %s. These were renamed out "
        "of the experimental prefix and are no longer honored: %s. Update your "
        "shell environment / .env to the new names.",
        ", ".join(detected),
        renames,
    )


def _legacy_names_in_dotenv(names: tuple[str, ...]) -> set[str]:
    """Return the subset of *names* present as keys in the project ``.env``.

    Key membership only — ``warn_legacy_flash_env`` warns on presence
    regardless of value, so values are never parsed.
    """
    dotenv_path = Path(".env")
    try:
        text = dotenv_path.read_text()
    except (FileNotFoundError, OSError):
        return set()
    found: set[str] = set()
    legacy = set(names)
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key = line.partition("=")[0].strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        if key in legacy:
            found.add(key)
    return found


def resolve_experimental(
    base: ExperimentalSettings,
    overrides: dict,
) -> ExperimentalSettings:
    """Create a new ExperimentalSettings with per-model overrides applied.

    Fields not present in *overrides* retain their value from *base*.
    Returns *base* unchanged if overrides is empty.

    Uses the pydantic core validator directly to avoid pydantic-settings
    re-reading ``OLMLX_EXPERIMENTAL_*`` env vars, which would produce
    confusing errors if an unrelated env var is malformed.
    """
    if not overrides:
        return base
    merged = base.model_dump()
    merged.update(overrides)
    # Validate field constraints and custom validators without triggering
    # BaseSettings env var resolution.  __pydantic_validator__ is the core
    # schema validator shared by __init__ and model_validate; calling it
    # directly bypasses _settings_build_values().
    return ExperimentalSettings.__pydantic_validator__.validate_python(merged)


@dataclass
class FlashMoeConfig:
    """Resolved Flash-MoE configuration: per-model overrides global Settings."""

    enabled: bool
    cache_budget_experts: int
    io_threads: int
