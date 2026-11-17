"""Make an Edge0 pack loadable by mlx-vlm's ``qwen4_exp`` model.

An Edge0 pack is not an MLX checkpoint (see :mod:`olmlx.engine.edge0.pack`).
:func:`convert_pack` writes, *into the pack directory itself*, the files the
ordinary olmlx Flash-MoE loader needs:

* ``model-*.safetensors`` — the trunk (everything but the routed experts and
  the n-gram table) and the vision tower, under mlx-vlm's parameter names.
  ``Q8_0`` matrices are repacked losslessly as MLX 8-bit affine weights.
* ``config.json`` — a ``qwen4_exp`` config derived from the skeleton GGUF's
  metadata, plus an ``edge0`` section describing the latent pool.
* tokenizer / processor files, built from the GGUF vocabulary.
* ``edge0_ple.json`` — manifest pointing at the FP8 n-gram table in place.
* ``flash_moe/`` — the per-layer ``D``/``U`` projections and the marker files
  that route the model through Flash-MoE with latent experts.
* ``mtp_draft/`` — the pack's MTP draft head as an mlx-vlm drafter, for
  speculative decoding (see :mod:`olmlx.engine.edge0.draft_head`).

The 37 GiB of routed experts and the 48 GiB n-gram table are never copied or
re-quantized: they are read from the pack's own files at inference time.
Converting in place keeps a model one directory, so the store's
list/show/delete need no special case. ``config.json`` is written last and
is what marks the directory as a model.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import numpy as np

from olmlx.engine.edge0 import draft_head, ggml_quants, ple, templates
from olmlx.engine.edge0.gguf import GGUFFile
from olmlx.engine.edge0.latent_moe import _blocks_to_affine, pack_projections
from olmlx.engine.edge0.pack import Edge0Pack, Edge0PackError

logger = logging.getLogger(__name__)

#: Bump when the written layout changes; a directory converted by an older
#: version is converted again.
CONVERTER_VERSION = 2
CONFIG_KEY = "edge0"
FLASH_MOE_FORMAT = "edge0_latent"
FLASH_MOE_DIRNAME = "flash_moe"
LATENT_DIRNAME = "latent"

_SHARD_BYTES = 1 << 30
_LANGUAGE = "language_model.model."

# GGUF tensor suffix -> (mlx-vlm parameter suffix, kind). Kinds:
#   linear  matrix; Q8_0 stays quantized, floats become bfloat16
#   norm    RMSNorm weight; GGUF stores 1 + w, mlx-vlm's Qwen4ExpRMSNorm wants w
#   plain   copied as bfloat16
#   neg_exp GGUF stores -exp(w); mlx-vlm wants w
#   conv    [channels, kernel] -> [channels, kernel, 1]
_HYPER = {
    "inject.weight": ("block_inject_weight.weight", "linear"),
    "norm.weight": ("hc_norm.weight", "norm"),
    "down.weight": ("input_mix_weight_down.weight", "linear"),
    "up.weight": ("input_mix_weight_up.weight", "linear"),
}
_LAYER_TENSORS: dict[str, tuple[str, str]] = {
    **{
        f"hc_attn_{k}": (f"attn_hyper_connection.{v}", kind)
        for k, (v, kind) in _HYPER.items()
    },
    **{
        f"hc_ffn_{k}": (f"mlp_hyper_connection.{v}", kind)
        for k, (v, kind) in _HYPER.items()
    },
    # Gated DeltaNet layers
    "ssm_a": ("linear_attn.A_log", "neg_exp"),
    "ssm_dt.bias": ("linear_attn.dt_bias", "plain"),
    "ssm_conv1d.weight": ("linear_attn.conv1d.weight", "conv"),
    "ssm_alpha.weight": ("linear_attn.in_proj_a.weight", "linear"),
    "ssm_beta.weight": ("linear_attn.in_proj_b.weight", "linear"),
    "attn_qkv.weight": ("linear_attn.in_proj_qkv.weight", "linear"),
    "attn_gate.weight": ("linear_attn.in_proj_z.weight", "linear"),
    "ssm_norm.weight": ("linear_attn.norm.weight", "plain"),
    "ssm_out.weight": ("linear_attn.out_proj.weight", "linear"),
    # Sparse-attention layers
    "attn_q.weight": ("self_attn.q_proj.weight", "linear"),
    "attn_k.weight": ("self_attn.k_proj.weight", "linear"),
    "attn_v.weight": ("self_attn.v_proj.weight", "linear"),
    "attn_output.weight": ("self_attn.o_proj.weight", "linear"),
    "attn_q_norm.weight": ("self_attn.q_norm.weight", "norm"),
    "attn_k_norm.weight": ("self_attn.k_norm.weight", "norm"),
    "indexer.q_norm.weight": ("self_attn.indexer.q_layernorm.weight", "norm"),
    "indexer.k_norm.weight": ("self_attn.indexer.k_layernorm.weight", "norm"),
    # Router + shared expert (the routed experts stay in experts.bin)
    "ffn_gate_inp.weight": ("mlp.gate.weight", "linear"),
    "ffn_gate_inp_shexp.weight": ("mlp.shared_expert_gate.weight", "linear"),
    "ffn_gate_shexp.weight": ("mlp.shared_expert.gate_proj.weight", "linear"),
    "ffn_up_shexp.weight": ("mlp.shared_expert.up_proj.weight", "linear"),
    "ffn_down_shexp.weight": ("mlp.shared_expert.down_proj.weight", "linear"),
    # Per-layer embedding
    "ple_conv1d.weight": ("ple.conv1d.weight", "conv"),
    "ple_key.weight": ("ple.key_proj.weight", "linear"),
    "ple_value.weight": ("ple.value_proj.weight", "linear"),
    "ple_norm_conv.weight": ("ple.norm_conv.weight", "norm"),
    "ple_norm_key.weight": ("ple.norm_key.weight", "norm"),
    "ple_norm_query.weight": ("ple.norm_query.weight", "norm"),
}
# Gated DeltaNet tensors indexed by value head. llama.cpp stores the value
# heads grouped by repeat (``repeat * key_heads + key_head``) where the HF
# layout mlx-vlm expects interleaves them (``key_head * repeats + repeat``).
# suffix -> (axis, leading entries that are not value heads, in units of
# key-head width; False = none).
_GDN_HEAD_ORDERED: dict[str, tuple[int, bool]] = {
    "ssm_a": (0, False),
    "ssm_dt.bias": (0, False),
    "ssm_alpha.weight": (0, False),
    "ssm_beta.weight": (0, False),
    "attn_gate.weight": (0, False),
    "ssm_out.weight": (1, False),
    # [query | key | value] stacked along the rows: only the value part moves.
    "attn_qkv.weight": (0, True),
    "ssm_conv1d.weight": (0, True),
}

# The sparse-attention indexer's query and key projections are one HF matrix.
_INDEXER_PARTS = ("indexer.q_proj.weight", "indexer.k_proj.weight")
_INDEXER_TARGET = "self_attn.indexer.index_qk_proj.weight"
# Routed-expert leftovers a latent pack never reads.
_IGNORED_SUFFIXES = ("_exps.scale", "_exps.input_scale", "_exps.weight")

_GLOBAL_TENSORS: dict[str, tuple[str, str]] = {
    "token_embd.weight": (_LANGUAGE + "embed_tokens.weight", "linear"),
    "output.weight": ("language_model.lm_head.weight", "linear"),
    "output_hc_norm.weight": (
        _LANGUAGE + "hyper_connection_mixer.hc_norm.weight",
        "norm",
    ),
    "output_hc_down.weight": (
        _LANGUAGE + "hyper_connection_mixer.input_mix_weight_down.weight",
        "linear",
    ),
    "output_hc_up.weight": (
        _LANGUAGE + "hyper_connection_mixer.input_mix_weight_up.weight",
        "linear",
    ),
}

_VISION_BLOCK = {
    "attn_qkv": "attn.qkv",
    "attn_out": "attn.proj",
    "ffn_up": "mlp.linear_fc1",
    "ffn_down": "mlp.linear_fc2",
    "ln1": "norm1",
    "ln2": "norm2",
}
_VISION_GLOBAL = {
    "v.patch_embd.bias": "patch_embed.proj.bias",
    "v.position_embd.weight": "pos_embed.weight",
    "v.post_ln.weight": "merger.norm.weight",
    "v.post_ln.bias": "merger.norm.bias",
    "mm.0.weight": "merger.linear_fc1.weight",
    "mm.0.bias": "merger.linear_fc1.bias",
    "mm.2.weight": "merger.linear_fc2.weight",
    "mm.2.bias": "merger.linear_fc2.bias",
}


class Edge0ConversionError(Edge0PackError):
    """The pack cannot be turned into a loadable model."""


def read_edge0_config(model_dir: Path | str) -> dict[str, Any] | None:
    """Return the ``edge0`` section of a converted pack's config, else ``None``."""
    try:
        config = json.loads((Path(model_dir) / "config.json").read_text())
    except (OSError, ValueError):
        return None
    section = config.get(CONFIG_KEY) if isinstance(config, dict) else None
    return section if isinstance(section, dict) else None


def is_converted(model_dir: Path | str, *, projection_bits: int | None = None) -> bool:
    """True when *model_dir* holds an up-to-date conversion.

    With *projection_bits*, the conversion must also have been made with that
    D/U quantization (it is baked into ``flash_moe/latent``).
    """
    section = read_edge0_config(model_dir)
    if section is None or section.get("converter_version") != CONVERTER_VERSION:
        return False
    if projection_bits is None:
        return True
    return (section.get("latent") or {}).get("projection_bits") == projection_bits


def mlx_key_for(gguf_name: str) -> tuple[str, str] | None:
    """Map a trunk GGUF tensor name to ``(mlx-vlm key, kind)``.

    ``None`` for tensors that are deliberately not converted on their own
    (indexer halves, routed-expert leftovers). Unknown names raise.
    """
    if gguf_name in _GLOBAL_TENSORS:
        return _GLOBAL_TENSORS[gguf_name]
    if gguf_name.startswith("blk."):
        _, layer, suffix = gguf_name.split(".", 2)
        if suffix in _LAYER_TENSORS:
            target, kind = _LAYER_TENSORS[suffix]
            return f"{_LANGUAGE}layers.{int(layer)}.{target}", kind
        if suffix in _INDEXER_PARTS or suffix.endswith(_IGNORED_SUFFIXES):
            return None
    raise Edge0ConversionError(f"skeleton tensor {gguf_name!r} has no known mapping")


def _bf16(values: np.ndarray) -> mx.array:
    return mx.array(np.ascontiguousarray(values, dtype=np.float32)).astype(mx.bfloat16)


def _gdn_head_index(name: str, shape: tuple[int, ...], gdn: dict[str, int] | None):
    """``(axis, index)`` reordering a GDN tensor's value heads to HF order.

    ``None`` when *name* is not value-head indexed.
    """
    suffix = name.split(".", 2)[-1] if name.startswith("blk.") else name
    if gdn is None or suffix not in _GDN_HEAD_ORDERED:
        return None
    axis, has_key_prefix = _GDN_HEAD_ORDERED[suffix]
    key_heads, value_heads = gdn["key_heads"], gdn["value_heads"]
    repeats, remainder = divmod(value_heads, key_heads)
    if remainder:
        raise Edge0ConversionError(
            f"{value_heads} value heads are not a multiple of {key_heads} key heads"
        )
    prefix = 2 * key_heads * gdn["key_head_dim"] if has_key_prefix else 0
    width, remainder = divmod(shape[axis] - prefix, value_heads)
    if remainder or width < 1:
        raise Edge0ConversionError(f"{name}: cannot split {shape} into value heads")
    hf_heads = np.arange(value_heads)
    source_heads = (hf_heads % repeats) * key_heads + hf_heads // repeats
    index = prefix + (source_heads[:, None] * width + np.arange(width)).reshape(-1)
    return axis, np.concatenate([np.arange(prefix), index])


def _convert_tensor(
    gguf: GGUFFile,
    name: str,
    key: str,
    kind: str,
    gdn: dict[str, int] | None = None,
) -> dict[str, mx.array]:
    tensor = gguf.tensors[name]
    reorder = _gdn_head_index(name, tensor.shape, gdn)
    if kind == "linear" and tensor.ggml_type == ggml_quants.GGML_Q8_0:
        rows = tensor.shape[0]
        codes, scales, _ = _blocks_to_affine(gguf.raw(name), tensor.ggml_type, rows)
        if reorder is not None:
            # Rows are independent, and a head's columns are whole 32-value
            # blocks, so the packed form reorders without touching a weight.
            axis, index = reorder
            if axis == 1:
                if index.size % 32 or np.any(
                    np.diff(index.reshape(-1, 32), axis=1) != 1
                ):
                    raise Edge0ConversionError(
                        f"{name}: value heads do not align with Q8_0 blocks"
                    )
                index = index.reshape(-1, 32)[:, 0] // 32
            codes = np.take(codes, index, axis=axis)
            scales = np.take(scales, index, axis=axis)
        # The model runs in bfloat16, so the block scales are rounded to it.
        # Derive the bias from the *rounded* scale (x128 is exact) so the
        # weight stays ``scale * (u - 128)``: a relative scale error only,
        # never a constant offset on every weight of the block.
        scales_bf16 = _bf16(scales)
        prefix = key[: -len(".weight")]
        return {
            f"{prefix}.weight": mx.array(codes.reshape(rows, -1).view("<u4")),
            f"{prefix}.scales": scales_bf16,
            f"{prefix}.biases": scales_bf16 * -128.0,
        }
    values = gguf.load(name)
    if reorder is not None:
        values = np.take(values, reorder[1], axis=reorder[0])
    if kind == "norm":
        values = values - 1.0
    elif kind == "conv":
        values = values[..., None]
    elif kind == "neg_exp":
        # llama.cpp stores the decay as -exp(A_log).
        if np.any(values >= 0):
            raise Edge0ConversionError(f"{name}: expected strictly negative decay")
        values = np.log(-values)
    return {key: _bf16(values)}


class _ShardWriter:
    """Accumulates tensors and flushes them as numbered safetensors shards."""

    def __init__(self, directory: Path, stem: str):
        self._directory = directory
        self._stem = stem
        self._pending: dict[str, mx.array] = {}
        self._pending_bytes = 0
        self.count = 0
        self.keys: set[str] = set()

    def add(self, tensors: dict[str, mx.array]) -> None:
        for key, value in tensors.items():
            if key in self.keys:
                raise Edge0ConversionError(f"duplicate converted tensor {key!r}")
            self.keys.add(key)
            self._pending[key] = value
            self._pending_bytes += value.nbytes
        if self._pending_bytes >= _SHARD_BYTES:
            self.flush()

    def flush(self) -> None:
        if not self._pending:
            return
        self.count += 1
        path = self._directory / f"{self._stem}-{self.count:05d}.safetensors"
        mx.save_safetensors(str(path), self._pending, metadata={"format": "mlx"})
        self._pending = {}
        self._pending_bytes = 0


def _convert_trunk(
    gguf: GGUFFile, writer: _ShardWriter, n_layers: int, gdn: dict[str, int]
) -> None:
    for name in gguf.tensors:
        mapped = mlx_key_for(name)
        if mapped is not None:
            writer.add(_convert_tensor(gguf, name, *mapped, gdn=gdn))
    for layer in range(n_layers):
        parts = [f"blk.{layer}.{suffix}" for suffix in _INDEXER_PARTS]
        present = [part in gguf.tensors for part in parts]
        if not any(present):
            continue
        if not all(present):
            raise Edge0ConversionError(f"layer {layer} has half an indexer projection")
        joined = np.concatenate([gguf.load(part) for part in parts], axis=0)
        writer.add({f"{_LANGUAGE}layers.{layer}.{_INDEXER_TARGET}": _bf16(joined)})


def _convert_vision(gguf: GGUFFile, writer: _ShardWriter) -> None:
    names = set(gguf.tensors)
    patch = [gguf.load(n) for n in ("v.patch_embd.weight", "v.patch_embd.weight.1")]
    names -= {"v.patch_embd.weight", "v.patch_embd.weight.1"}
    # [out, in, h, w] per temporal slice -> MLX conv3d [out, t, h, w, in].
    stacked = np.stack(patch, axis=1).transpose(0, 1, 3, 4, 2)
    writer.add({"vision_tower.patch_embed.proj.weight": _bf16(stacked)})
    for name in sorted(names):
        if name in _VISION_GLOBAL:
            key = _VISION_GLOBAL[name]
        elif name.startswith("v.blk."):
            _, _, block, part, leaf = name.split(".")
            if part not in _VISION_BLOCK:
                raise Edge0ConversionError(f"unknown vision tensor {name!r}")
            key = f"blocks.{int(block)}.{_VISION_BLOCK[part]}.{leaf}"
        else:
            raise Edge0ConversionError(f"unknown vision tensor {name!r}")
        writer.add({f"vision_tower.{key}": _bf16(gguf.load(name))})


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _text_config(metadata: dict[str, Any], row_width: int) -> dict[str, Any]:
    def meta(key: str, default: Any = None) -> Any:
        value = metadata.get(f"qwen4exp.{key}", default)
        if value is None:
            raise Edge0ConversionError(f"skeleton GGUF lacks qwen4exp.{key}")
        return value

    recurrent = meta("attention.recurrent_layers")
    heads_per_ngram = int(meta("ple.heads_per_ngram"))
    ngram_size = int(meta("ple.ngram_size"))
    value_heads = int(meta("ssm.time_step_rank"))
    head_dim = int(meta("attention.key_length"))
    compress = [int(v) for v in meta("attention.compress_ratios") if int(v)]
    sections = [int(v) for v in meta("rope.dimension_sections") if int(v)]
    rotary_factor = int(meta("rope.dimension_count")) / head_dim
    return {
        "model_type": "qwen4_exp_text",
        "dtype": "bfloat16",
        "hidden_size": int(meta("embedding_length")),
        "num_hidden_layers": int(meta("block_count")),
        "num_attention_heads": int(meta("attention.head_count")),
        "num_key_value_heads": int(meta("attention.head_count_kv")),
        "head_dim": head_dim,
        "max_position_embeddings": int(meta("context_length")),
        "vocab_size": len(metadata["tokenizer.ggml.tokens"]),
        "rms_norm_eps": 1e-6,
        "layer_types": [
            "linear_attention" if is_recurrent else "full_attention"
            for is_recurrent in recurrent
        ],
        "full_attention_interval": int(meta("full_attention_interval")),
        "linear_conv_kernel_dim": int(meta("ssm.conv_kernel")),
        "linear_key_head_dim": int(meta("ssm.state_size")),
        "linear_num_key_heads": int(meta("ssm.group_count")),
        "linear_num_value_heads": value_heads,
        "linear_value_head_dim": int(meta("ssm.inner_size")) // value_heads,
        "num_experts": int(meta("expert_count")),
        "num_experts_per_tok": int(meta("expert_used_count")),
        "moe_intermediate_size": int(meta("expert_feed_forward_length")),
        "shared_expert_intermediate_size": int(
            meta("expert_shared_feed_forward_length")
        ),
        "hc_count": int(meta("hyper_connection.count")),
        "hc_lowrank": int(meta("hyper_connection.low_rank")),
        "indexer_n_heads": int(meta("attention.indexer.head_count")),
        "indexer_head_dim": int(meta("attention.indexer.key_length")),
        "indexer_budget": int(meta("attention.indexer.top_k")),
        "indexer_compress_ratio": compress[0] if compress else 4,
        "indexer_kv_heads": 1,
        # GGUF numbers the layer the PLE block belongs to; mlx-vlm's
        # ``ple_layer_ids`` are one-based.
        "ple_layer_ids": [int(layer) + 1 for layer in meta("ple.layers")],
        "ple_embed_dim": row_width * heads_per_ngram * (ngram_size - 1),
        "ple_conv_kernel_size": int(meta("ple.conv_kernel")),
        "ngram_size": ngram_size,
        "heads_per_ngram": heads_per_ngram,
        "eos_token_id": int(meta("ple.eos_token_id")),
        "bos_token_id": int(metadata.get("tokenizer.ggml.bos_token_id", 0)),
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": sections,
            "partial_rotary_factor": rotary_factor,
            "rope_theta": float(meta("rope.freq_base")),
            "rope_type": "default",
        },
        "partial_rotary_factor": rotary_factor,
        "hidden_act": "silu",
        "output_gate_type": "sigmoid",
        "attention_bias": False,
        "tie_word_embeddings": False,
        "ple_storage": {"manifest": ple.MANIFEST_NAME, "cache_rows": 0},
    }


def _check_ple_hashing(metadata: dict[str, Any], text_config: dict[str, Any]) -> None:
    """Refuse a pack whose n-gram hashing mlx-vlm would not reproduce.

    mlx-vlm derives the hash multipliers and per-head table sizes from the
    config (seed + primes) instead of reading them; the GGUF states them. A
    mismatch would silently read the wrong embedding rows.
    """
    try:
        from mlx_vlm.models.qwen4_exp import config as vlm_config
        from mlx_vlm.models.qwen4_exp import language
    except ImportError:  # pragma: no cover — mlx-vlm is a hard dependency
        logger.warning("mlx-vlm qwen4_exp module unavailable; PLE hashing unchecked")
        return
    defaults = vlm_config.TextConfig.__dataclass_fields__
    base = int(defaults["ngram_vocab_size_base"].default)
    seed = int(defaults["seed"].default)
    heads = text_config["heads_per_ngram"] * (text_config["ngram_size"] - 1)
    sizes = [
        language._find_nth_prime_after(base - 1, head + 1) for head in range(heads)
    ]
    multipliers = language._build_layer_multipliers(
        text_config["vocab_size"], text_config["ngram_size"], 0, seed
    ).tolist()
    stated_sizes = [int(v) for v in metadata.get("qwen4exp.ple.head_vocab_sizes", [])]
    stated_multipliers = [
        int(v) for v in metadata.get("qwen4exp.ple.layer_multipliers", [])
    ]
    if sizes != stated_sizes or multipliers != stated_multipliers:
        raise Edge0ConversionError(
            "the pack's n-gram hashing (head sizes / multipliers) differs from "
            "mlx-vlm's qwen4_exp derivation; the PLE table would be misread"
        )


def _vision_config(mmproj: GGUFFile | None) -> dict[str, Any]:
    config = dict(templates.DEFAULT_VISION_CONFIG)
    if mmproj is None:
        return config
    md = mmproj.metadata
    for key, source in (
        ("depth", "clip.vision.block_count"),
        ("hidden_size", "clip.vision.embedding_length"),
        ("intermediate_size", "clip.vision.feed_forward_length"),
        ("num_heads", "clip.vision.attention.head_count"),
        ("out_hidden_size", "clip.vision.projection_dim"),
        ("patch_size", "clip.vision.patch_size"),
        ("spatial_merge_size", "clip.vision.spatial_merge_size"),
    ):
        if source in md:
            config[key] = int(md[source])
    position = mmproj.tensors.get("v.position_embd.weight")
    if position is not None:
        config["num_position_embeddings"] = position.shape[0]
    return config


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_GGML_TOKEN_NORMAL = 1
_GGML_TOKEN_CONTROL = 3
_GGML_TOKEN_USER_DEFINED = 4


def _build_tokenizer(metadata: dict[str, Any], path: Path) -> dict[str, int]:
    """Write a Hugging Face ``tokenizer.json`` from the GGUF vocabulary.

    Returns ``{content: id}`` of the added (special/user-defined) tokens.
    """
    from tokenizers import (
        AddedToken,
        Regex,
        Tokenizer,
        decoders,
        models,
        normalizers,
        pre_tokenizers,
        processors,
    )

    tokens = metadata["tokenizer.ggml.tokens"]
    types = metadata["tokenizer.ggml.token_type"]
    vocab = {
        token: index
        for index, (token, kind) in enumerate(zip(tokens, types))
        if kind == _GGML_TOKEN_NORMAL
    }
    if sorted(vocab.values()) != list(range(len(vocab))):
        raise Edge0ConversionError("pack vocabulary's base tokens are not contiguous")
    merges = [tuple(merge.split(" ", 1)) for merge in metadata["tokenizer.ggml.merges"]]

    tokenizer = Tokenizer(
        models.BPE(
            vocab=vocab,
            merges=merges,
            unk_token=None,
            continuing_subword_prefix="",
            end_of_word_suffix="",
            fuse_unk=False,
            byte_fallback=False,
        )
    )
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(
                Regex(templates.PRETOKENIZE_REGEX), behavior="isolated", invert=False
            ),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.decoder = decoders.ByteLevel()

    added: dict[str, int] = {}
    for index, (token, kind) in enumerate(zip(tokens, types)):
        if kind not in (_GGML_TOKEN_CONTROL, _GGML_TOKEN_USER_DEFINED):
            continue  # normal tokens above; unused padding slots are dropped
        special = (
            kind == _GGML_TOKEN_CONTROL
            and token not in templates.NON_SPECIAL_CONTROL_TOKENS
        )
        entry = AddedToken(token, special=special, normalized=False)
        if special:
            tokenizer.add_special_tokens([entry])
        else:
            tokenizer.add_tokens([entry])
        if tokenizer.token_to_id(token) != index:
            raise Edge0ConversionError(
                f"added token {token!r} did not land on its pack id {index}"
            )
        added[token] = index
    tokenizer.save(str(path))
    return added


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def convert_pack(
    model_dir: Path | str,
    *,
    projection_bits: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Convert the Edge0 pack at *model_dir* in place. Returns *model_dir*.

    *projection_bits* selects the pack's 4- or 8-bit D/U projections; ``None``
    keeps the pack manifest's choice.
    """
    model_dir = Path(model_dir)

    def report(message: str) -> None:
        logger.info("edge0 convert: %s", message)
        if progress is not None:
            progress(message)

    pack = Edge0Pack(model_dir, projection_bits=projection_bits)
    geometry = pack.geometry
    for path in (pack.skeleton_path, pack.ple_path):
        if not path.is_file():
            raise Edge0ConversionError(f"pack is missing {path.name}")

    # Drop the previous conversion first: without config.json the directory
    # is not a model, so an interrupted run can never be served half-written.
    (model_dir / "config.json").unlink(missing_ok=True)
    for stale in model_dir.glob("model-*.safetensors"):
        stale.unlink()
    flash_dir = model_dir / FLASH_MOE_DIRNAME
    latent_dir = flash_dir / LATENT_DIRNAME
    latent_dir.mkdir(parents=True, exist_ok=True)
    for stale in latent_dir.glob("*.safetensors"):
        stale.unlink()

    writer = _ShardWriter(model_dir, "model")
    with GGUFFile(pack.skeleton_path) as skeleton:
        metadata = skeleton.metadata
        if metadata.get("general.architecture") != "qwen4exp":
            raise Edge0ConversionError(
                "skeleton GGUF architecture is "
                f"{metadata.get('general.architecture')!r}, expected 'qwen4exp'"
            )

        with GGUFFile(pack.ple_path) as ple_gguf:
            table = ple_gguf.tensors.get("per_layer_token_embd.weight")
            if table is None or ple_gguf.metadata.get("strata.ple.format") != "f8_e4m3":
                raise Edge0ConversionError(
                    f"{pack.ple_path.name} is not an FP8 E4M3 n-gram table"
                )
            row_count, row_width = table.shape
            ple.write_manifest(
                model_dir / ple.MANIFEST_NAME,
                data_file=pack.ple_path.name,
                data_offset=table.offset,
                row_count=row_count,
                row_width=row_width,
                scale=float(ple_gguf.metadata["strata.ple.scale"]),
            )

        text_config = _text_config(metadata, row_width)
        if text_config["num_hidden_layers"] != geometry.n_layers or (
            text_config["hidden_size"],
            text_config["num_experts"],
            text_config["moe_intermediate_size"],
        ) != (geometry.hidden, geometry.n_experts, geometry.ff):
            raise Edge0ConversionError(
                "skeleton GGUF and expert pool disagree on the model's shape"
            )
        _check_ple_hashing(metadata, text_config)

        report("converting trunk weights")
        _convert_trunk(
            skeleton,
            writer,
            geometry.n_layers,
            {
                "key_heads": text_config["linear_num_key_heads"],
                "value_heads": text_config["linear_num_value_heads"],
                "key_head_dim": text_config["linear_key_head_dim"],
            },
        )

        report("building tokenizer")
        added_tokens = _build_tokenizer(metadata, model_dir / "tokenizer.json")
        chat_template = metadata.get("tokenizer.chat_template")
        eos_id = int(metadata["tokenizer.ggml.eos_token_id"])
        pad_id = int(metadata.get("tokenizer.ggml.padding_token_id", eos_id))

    mmproj_files = sorted(model_dir.glob("mmproj-*.gguf"))
    mmproj = GGUFFile(mmproj_files[0]) if mmproj_files else None
    try:
        vision_config = _vision_config(mmproj)
        if mmproj is not None:
            report("converting vision tower")
            _convert_vision(mmproj, writer)
    finally:
        if mmproj is not None:
            mmproj.close()
    writer.flush()

    draft_dirname = None
    if draft_head.has_draft_head(model_dir):
        report("converting draft head")
        draft_dirname = draft_head.convert_draft_head(
            model_dir, model_dir, text_config
        ).name

    report("packing latent projections")
    bits = 0
    for layer in range(geometry.n_layers):
        tensors = pack_projections(pack, layer)
        bits = int(tensors.pop("bits"))
        payload = {
            key: (value.astype(mx.float16) if value.dtype == mx.float32 else value)
            for key, value in tensors.items()
        }
        payload["inverse_scale"] = mx.array(pack.inverse_column_scale(layer))
        payload["labels"] = mx.array(pack.labels[layer].astype(np.int32))
        mx.save_safetensors(
            str(latent_dir / f"layer_{layer:03d}.safetensors"),
            payload,
            metadata={"format": "mlx"},
        )

    latent = {
        "n_layers": geometry.n_layers,
        "n_experts": geometry.n_experts,
        "n_groups": geometry.n_groups,
        "hidden": geometry.hidden,
        "latent": geometry.latent,
        "ff": geometry.ff,
        "gu_type": geometry.gu_type,
        "down_type": geometry.down_type,
        "projection_bits": bits,
    }
    _write_json(
        flash_dir / "flash_moe_config.json",
        {
            "format": FLASH_MOE_FORMAT,
            "moe_layer_indices": list(range(geometry.n_layers)),
            "hidden_size": geometry.hidden,
            "intermediate_size": geometry.ff,
            "num_experts": geometry.n_experts,
            "num_experts_per_tok": text_config["num_experts_per_tok"],
            "latent": latent,
        },
    )
    _write_json(
        flash_dir / "flash_moe_layout.json",
        {"format": FLASH_MOE_FORMAT, "pack_root": ".."},
    )

    if chat_template:
        (model_dir / "chat_template.jinja").write_text(chat_template)
    _write_json(
        model_dir / "tokenizer_config.json",
        {
            **templates.TOKENIZER_CONFIG,
            "model_max_length": text_config["max_position_embeddings"],
        },
    )
    _write_json(model_dir / "preprocessor_config.json", templates.PREPROCESSOR_CONFIG)
    _write_json(
        model_dir / "video_preprocessor_config.json",
        templates.VIDEO_PREPROCESSOR_CONFIG,
    )
    _write_json(model_dir / "processor_config.json", templates.PROCESSOR_CONFIG)
    eos_ids = list(dict.fromkeys([eos_id, pad_id]))
    _write_json(
        model_dir / "generation_config.json",
        {"eos_token_id": eos_ids, "pad_token_id": pad_id},
    )

    config: dict[str, Any] = {
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "model_type": "qwen4_exp",
        "eos_token_id": eos_ids,
        "tie_word_embeddings": False,
        "text_config": text_config,
        "vision_config": vision_config,
        "quantization": {"group_size": 32, "bits": 8, "mode": "affine"},
        CONFIG_KEY: {
            "converter_version": CONVERTER_VERSION,
            "model_name": pack.manifest.get("model_name"),
            "has_vision": bool(mmproj_files),
            "draft_head": draft_dirname,
            "latent": latent,
        },
    }
    for field, token in templates.SPECIAL_TOKEN_NAMES.items():
        if token in added_tokens:
            config[field] = added_tokens[token]

    # config.json is the model marker: write it last, atomically.
    temporary = model_dir / "config.json.tmp"
    _write_json(temporary, config)
    os.replace(temporary, model_dir / "config.json")
    report("done")
    return model_dir
