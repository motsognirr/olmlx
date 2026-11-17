"""Synthetic Edge0 latent packs for tests.

Every ggml block format decodes any bit pattern, so a pack is built from
random block bytes (with sane float16 scales) and the expected float weights
come from the decoders in ``olmlx.engine.edge0.ggml_quants`` — themselves
checked against scalar ports of ggml's reference in
``test_edge0_ggml_quants.py``.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from olmlx.engine.edge0 import ggml_quants as gq

#: ggml type -> byte offset of the block's float16 scale.
_SCALE_OFFSET = {gq.GGML_Q3_K: 108, gq.GGML_IQ4_NL: 0, gq.GGML_Q8_0: 0, gq.GGML_Q4_0: 0}


def random_blocks(
    rng, ggml_type: int, n_values: int, scale: float = 0.05
) -> np.ndarray:
    """Random valid blocks of *ggml_type* holding *n_values* weights."""
    per_block, block_bytes = gq.block_geometry(ggml_type)
    assert n_values % per_block == 0
    n_blocks = n_values // per_block
    raw = rng.integers(0, 256, size=(n_blocks, block_bytes), dtype=np.uint8)
    d = rng.uniform(scale / 2, scale, size=n_blocks).astype(np.float16)
    offset = _SCALE_OFFSET[ggml_type]
    raw[:, offset : offset + 2] = d.view(np.uint8).reshape(n_blocks, 2)
    return raw.reshape(-1)


def gguf_string(text: str) -> bytes:
    data = text.encode()
    return struct.pack("<Q", len(data)) + data


def write_gguf(path, metadata, tensors, *, alignment=32, version=3) -> int:
    """Write a GGUF file; returns the data-section offset.

    *metadata*: ``{key: (value_type, payload_bytes)}``.
    *tensors*: list of ``(name, gguf_dims, ggml_type, data_bytes)`` where
    ``gguf_dims`` is innermost-first, as stored in the file.
    """
    out = bytearray(b"GGUF")
    out += struct.pack("<IQQ", version, len(tensors), len(metadata))
    for key, (value_type, payload) in metadata.items():
        out += gguf_string(key) + struct.pack("<I", value_type) + payload
    offset = 0
    offsets = []
    for name, dims, ggml_type, data in tensors:
        out += gguf_string(name) + struct.pack("<I", len(dims))
        out += struct.pack(f"<{len(dims)}Q", *dims)
        out += struct.pack("<IQ", ggml_type, offset)
        offsets.append(offset)
        offset += len(data)
        offset += (-offset) % alignment
    out += b"\0" * ((-len(out)) % alignment)
    data_start = len(out)
    for (_, _, _, data), tensor_offset in zip(tensors, offsets):
        out += b"\0" * (data_start + tensor_offset - len(out))
        out += bytes(data)
    Path(path).write_bytes(bytes(out))
    return data_start


@dataclass
class SyntheticPack:
    root: Path
    layers: int
    experts: int
    groups: int
    hidden: int
    latent: int
    ff: int
    labels: np.ndarray
    #: ``[layers, latent]`` raw ``s`` as written to ``scales/``.
    scales: np.ndarray
    #: ``(layer, expert) -> (gate, up, down)`` float32 matrices.
    expert_weights: dict = field(default_factory=dict)
    #: per layer ``[groups, latent, hidden]`` and ``[groups, hidden, latent]``.
    down_proj: list = field(default_factory=list)
    up_proj: list = field(default_factory=list)
    #: dense.bin tensors by index name (float32, numpy shape).
    dense: dict = field(default_factory=dict)
    #: projection bits -> (down_proj list, up_proj list) for every file written.
    projections: dict = field(default_factory=dict)

    @property
    def record_bytes(self) -> int:
        return (
            2 * self.ff * (self.latent // 256) * 110
            + self.latent * (self.ff // 32) * 18
        )


def make_pack(
    tmp_path,
    *,
    layers=2,
    experts=8,
    groups=2,
    hidden=64,
    latent=256,
    ff=32,
    dense_fmt=8,
    seed=0,
    with_scales=True,
    extra_dense_fmts=(),
) -> SyntheticPack:
    """Write a tiny latent pack under *tmp_path* and return its ground truth."""
    rng = np.random.default_rng(seed)
    root = Path(tmp_path) / "pack"
    root.mkdir(parents=True)
    labels = rng.integers(0, groups, size=(layers, experts)).astype("<i4")
    scales = rng.uniform(0.5, 2.0, size=(layers, latent)).astype("<f4")
    pack = SyntheticPack(
        root, layers, experts, groups, hidden, latent, ff, labels, scales
    )

    with open(root / "experts.bin", "wb") as f:
        for layer in range(layers):
            for expert in range(experts):
                gate = random_blocks(rng, gq.GGML_Q3_K, ff * latent)
                up = random_blocks(rng, gq.GGML_Q3_K, ff * latent)
                down = random_blocks(rng, gq.GGML_IQ4_NL, latent * ff, scale=0.002)
                f.write(gate.tobytes() + up.tobytes() + down.tobytes())
                pack.expert_weights[(layer, expert)] = (
                    gq.dequantize_q3_k(gate).reshape(ff, latent),
                    gq.dequantize_q3_k(up).reshape(ff, latent),
                    gq.dequantize_iq4_nl(down).reshape(latent, ff),
                )
    record = pack.record_bytes
    (root / "native_experts.txt").write_text(
        "# strata native experts v6: latent A-only, Q3_K gu + shared colscale "
        f"(expert_embd {latent}, expert_ff {ff})\n"
        + "".join(
            f"{layer} 11 20 {layer * experts * record} {record}\n"
            for layer in range(layers)
        )
    )

    dense_name = f"latent_dense_q{dense_fmt}.bin"
    for fmt in (dense_fmt, *extra_dense_fmts):
        dense_type = gq.GGML_Q8_0 if fmt == 8 else gq.GGML_Q4_0
        down_list, up_list = [], []
        with open(root / f"latent_dense_q{fmt}.bin", "wb") as f:
            for _ in range(layers):
                d_raw = random_blocks(
                    rng, dense_type, groups * latent * hidden, scale=0.01
                )
                u_raw = random_blocks(
                    rng, dense_type, groups * hidden * latent, scale=0.01
                )
                f.write(d_raw.tobytes() + u_raw.tobytes())
                down_list.append(
                    gq.dequantize(d_raw, dense_type).reshape(groups, latent, hidden)
                )
                up_list.append(
                    gq.dequantize(u_raw, dense_type).reshape(groups, hidden, latent)
                )
        pack.projections[fmt] = (down_list, up_list)
    pack.down_proj, pack.up_proj = pack.projections[dense_fmt]

    labels.tofile(root / "labels.bin")
    if with_scales:
        (root / "scales").mkdir()
        for layer in range(layers):
            scales[layer].tofile(root / "scales" / f"l{layer}.f32")

    # index.txt + dense.bin: a BF16 router, an F32 norm, an F16 conv and a
    # skeleton-only (shape-only) matrix per layer.
    rows = []
    blob = bytearray()

    def add(name, values, kind):
        shape = values.shape
        if kind == 2:
            data = values.astype("<f4").tobytes()
            stored = values.astype(np.float32)
        elif kind == 5:
            data = values.astype("<f2").tobytes()
            stored = values.astype(np.float16).astype(np.float32)
        else:  # 4: bfloat16 = top half of the float32 pattern
            bits = (values.astype("<f4").view("<u4") >> 16).astype("<u2")
            data = bits.tobytes()
            stored = (bits.astype("<u4") << 16).view("<f4").reshape(shape)
        ne0, ne1 = (shape[1], shape[0]) if len(shape) == 2 else (shape[0], 0)
        rows.append(
            f"{name} 0 {kind} {len(blob)} {len(data)} {len(blob)} {len(data)} "
            f"{ne0} {ne1} 0 0 1 0 0 0 0 0 0 0"
        )
        blob.extend(data)
        blob.extend(b"\0" * ((-len(blob)) % 64))
        pack.dense[name] = stored

    for layer in range(layers):
        add(
            f"blk.{layer}.ffn_gate_inp.weight",
            rng.standard_normal((experts, hidden)).astype(np.float32),
            4,
        )
        add(
            f"blk.{layer}.hc_ffn_norm.weight",
            rng.standard_normal(hidden).astype(np.float32),
            2,
        )
        rows.append(
            f"blk.{layer}.ffn_up_shexp.weight 0 0 0 0 0 0 {hidden} {ff} 8 0 32 "
            "0 0 0 0 0 0 0"
        )
    add(
        "blk.1.ple_conv1d.weight",
        rng.standard_normal((hidden, 4)).astype(np.float32),
        5,
    )
    (root / "index.txt").write_text(
        "# strata pack index v3 -- synthetic\n"
        f"# align 64 pool {len(blob)} tensors {len(rows)}\n" + "\n".join(rows) + "\n"
    )
    (root / "dense.bin").write_bytes(bytes(blob))

    (root / "model.json").write_text(
        json.dumps(
            {
                "model_name": "edge0-synthetic",
                "architecture": "latent sparse MoE",
                "pack_format": "strata pack index v3",
                "args": [
                    "--pack",
                    ".",
                    "--native",
                    "./model-skeleton.gguf",
                    "--ple-gguf",
                    "./ple-fp8.gguf",
                ],
                "tokenizer": "./tokenizer",
                "env": {
                    "STRATA_LATENT_DENSE": f"./{dense_name}",
                    "STRATA_LATENT_DENSE_FMT": str(dense_fmt),
                    "STRATA_LATENT_A": "./experts.bin",
                    "STRATA_LATENT_LABELS": "./labels.bin",
                },
            }
        )
    )
    return pack


# ---------------------------------------------------------------------------
# Full pack: latent pool + skeleton GGUF + n-gram table + vision projector
# ---------------------------------------------------------------------------

#: Skeleton tensors the real pack stores as Q8_0 (everything else is float).
_Q8_SUFFIXES = (
    "attn_qkv.weight",
    "attn_gate.weight",
    "ssm_out.weight",
    "ffn_gate_shexp.weight",
    "ffn_up_shexp.weight",
    "ffn_down_shexp.weight",
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_output.weight",
    "token_embd.weight",
    "output.weight",
)

TINY_LAYERS = 4
TINY_HIDDEN = 64
#: Routed-expert width. 64 so the draft head's Q2_0 experts (64-value blocks)
#: have whole blocks on both sides.
TINY_FF = 64
#: Sixteen n-gram heads of ~20M rows each (mlx-vlm fixes the table size).
PLE_ROWS = 320001536
PLE_ROW_WIDTH = TINY_HIDDEN // 16
PLE_SCALE = 0.25


def _kv_u32(value):
    return (4, struct.pack("<I", value))


def _kv_f32(value):
    return (6, struct.pack("<f", value))


def _kv_str(value):
    return (8, gguf_string(value))


def _kv_array(item_type, fmt, values):
    return (
        9,
        struct.pack("<IQ", item_type, len(values))
        + struct.pack(f"<{len(values)}{fmt}", *values),
    )


def _kv_strings(values):
    return (
        9,
        struct.pack("<IQ", 8, len(values)) + b"".join(gguf_string(v) for v in values),
    )


def tiny_tokens():
    """A byte-level vocabulary with a few merges, specials and unused slots."""
    from tokenizers import pre_tokenizers

    base = sorted(pre_tokenizers.ByteLevel.alphabet())
    merges = ["Ġ t", "h e", "Ġt he"]
    merged = ["Ġt", "he", "Ġthe"]
    specials = [
        ("<|endoftext|>", 3),
        ("<|im_start|>", 3),
        ("<|im_end|>", 3),
        ("<|vision_start|>", 3),
        ("<|vision_end|>", 3),
        ("<|image_pad|>", 3),
        ("<|video_pad|>", 3),
        ("<tool_call>", 4),
        ("<|fim_prefix|>", 3),
        ("<think>", 4),
        ("</think>", 4),
    ]
    tokens = base + merged + [name for name, _ in specials]
    types = [1] * (len(base) + len(merged)) + [kind for _, kind in specials]
    while len(tokens) % 32:
        tokens.append(f"[PAD{len(tokens)}]")
        types.append(5)
    return tokens, types, merges


def tiny_metadata():
    tokens, types, merges = tiny_tokens()
    recurrent = [(layer + 1) % 4 != 0 for layer in range(TINY_LAYERS)]
    md = {
        "general.architecture": _kv_str("qwen4exp"),
        "qwen4exp.block_count": _kv_u32(TINY_LAYERS),
        "qwen4exp.context_length": _kv_u32(4096),
        "qwen4exp.embedding_length": _kv_u32(TINY_HIDDEN),
        "qwen4exp.attention.head_count": _kv_u32(2),
        "qwen4exp.attention.head_count_kv": _kv_u32(1),
        "qwen4exp.attention.key_length": _kv_u32(32),
        "qwen4exp.attention.value_length": _kv_u32(32),
        "qwen4exp.rope.dimension_sections": _kv_array(5, "i", [2, 1, 1, 0]),
        "qwen4exp.rope.dimension_count": _kv_u32(8),
        "qwen4exp.rope.freq_base": _kv_f32(10000000.0),
        "qwen4exp.expert_count": _kv_u32(8),
        "qwen4exp.expert_used_count": _kv_u32(2),
        "qwen4exp.expert_feed_forward_length": _kv_u32(TINY_FF),
        "qwen4exp.expert_shared_feed_forward_length": _kv_u32(32),
        "qwen4exp.ssm.conv_kernel": _kv_u32(4),
        "qwen4exp.ssm.state_size": _kv_u32(32),
        "qwen4exp.ssm.group_count": _kv_u32(2),
        "qwen4exp.ssm.time_step_rank": _kv_u32(4),
        "qwen4exp.ssm.inner_size": _kv_u32(128),
        "qwen4exp.attention.recurrent_layers": (
            9,
            struct.pack("<IQ", 7, len(recurrent)) + bytes(recurrent),
        ),
        "qwen4exp.full_attention_interval": _kv_u32(4),
        "qwen4exp.hyper_connection.count": _kv_u32(4),
        "qwen4exp.hyper_connection.low_rank": _kv_u32(8),
        "qwen4exp.attention.indexer.head_count": _kv_u32(2),
        "qwen4exp.attention.indexer.key_length": _kv_u32(8),
        "qwen4exp.attention.indexer.top_k": _kv_u32(16),
        "qwen4exp.attention.compress_ratios": _kv_array(
            5, "i", [0 if r else 4 for r in recurrent]
        ),
        "qwen4exp.ple.layers": _kv_array(5, "i", [1]),
        "qwen4exp.ple.ngram_size": _kv_u32(3),
        "qwen4exp.ple.heads_per_ngram": _kv_u32(8),
        "qwen4exp.ple.conv_kernel": _kv_u32(4),
        "qwen4exp.ple.eos_token_id": _kv_u32(tokens.index("<|endoftext|>")),
        "tokenizer.ggml.model": _kv_str("gpt2"),
        "tokenizer.ggml.tokens": _kv_strings(tokens),
        "tokenizer.ggml.token_type": _kv_array(5, "i", types),
        "tokenizer.ggml.merges": _kv_strings(merges),
        "tokenizer.ggml.eos_token_id": _kv_u32(tokens.index("<|im_end|>")),
        "tokenizer.ggml.padding_token_id": _kv_u32(tokens.index("<|endoftext|>")),
        "tokenizer.ggml.bos_token_id": _kv_u32(tokens.index("<|endoftext|>")),
        "tokenizer.chat_template": _kv_str(
            "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}"
            "<|im_end|>\n{% endfor %}<|im_start|>assistant\n"
        ),
    }
    return md, tokens


def _ple_hash_metadata(vocab_size):
    from mlx_vlm.models.qwen4_exp import language

    sizes = [language._find_nth_prime_after(20_000_000 - 1, h + 1) for h in range(16)]
    offsets = [sum(sizes[:h]) for h in range(16)]
    multipliers = language._build_layer_multipliers(vocab_size, 3, 0, 1234).tolist()
    return {
        "qwen4exp.ple.head_vocab_sizes": _kv_array(5, "i", sizes),
        "qwen4exp.ple.head_offsets": _kv_array(5, "i", offsets),
        "qwen4exp.ple.layer_multipliers": _kv_array(11, "q", multipliers),
    }


def _float_tensor(rng, name, shape, ggml_type, *, center=0.0):
    values = (rng.standard_normal(shape) * 0.05 + center).astype(np.float32)
    if ggml_type == gq.GGML_F32:
        data = values.astype("<f4").tobytes()
    elif ggml_type == gq.GGML_F16:
        values = values.astype(np.float16).astype(np.float32)
        data = values.astype("<f2").tobytes()
    else:
        bits = (values.view("<u4") >> 16).astype("<u2")
        values = (bits.astype("<u4") << 16).view("<f4").reshape(shape)
        data = bits.tobytes()
    return (name, list(reversed(shape)), ggml_type, data), values


def write_tiny_skeleton(path, rng, shapes):
    """Write a skeleton GGUF. *shapes*: ``{gguf_name: numpy shape}``.

    Returns ``{gguf_name: float32 values}`` as decoded from the file.
    """
    metadata, tokens = tiny_metadata()
    metadata.update(_ple_hash_metadata(len(tokens)))
    tensors, truth = [], {}
    for name, shape in shapes.items():
        if name.endswith(_Q8_SUFFIXES):
            raw = random_blocks(rng, gq.GGML_Q8_0, int(np.prod(shape)), scale=0.002)
            tensors.append((name, list(reversed(shape)), gq.GGML_Q8_0, raw.tobytes()))
            truth[name] = gq.dequantize_q8_0(raw).reshape(shape)
            continue
        if name.endswith("ssm_a"):
            # llama.cpp stores the GDN decay as -exp(A_log): strictly negative.
            entry, values = _float_tensor(rng, name, shape, gq.GGML_F32, center=-2.0)
        elif name.endswith("norm.weight") and "ssm_norm" not in name:
            entry, values = _float_tensor(rng, name, shape, gq.GGML_F32, center=1.0)
        elif len(shape) == 1 or "conv1d" in name or "ssm_norm" in name:
            kind = gq.GGML_F16 if "ple_conv1d" in name else gq.GGML_F32
            entry, values = _float_tensor(rng, name, shape, kind)
        else:
            entry, values = _float_tensor(rng, name, shape, gq.GGML_BF16)
        tensors.append(entry)
        truth[name] = values
    write_gguf(path, metadata, tensors)
    return truth


def write_tiny_ple(path, rng, known_rows=()):
    """Write a (sparse) FP8 n-gram table; *known_rows* get random content."""
    data_start = write_gguf(
        path,
        {
            "general.architecture": _kv_str("strata-ple"),
            "strata.ple.format": _kv_str("f8_e4m3"),
            "strata.ple.scale": _kv_f32(PLE_SCALE),
        },
        [("per_layer_token_embd.weight", [PLE_ROW_WIDTH, PLE_ROWS], gq.GGML_I8, b"")],
    )
    rows = {}
    with open(path, "r+b") as f:
        f.truncate(data_start + PLE_ROWS * PLE_ROW_WIDTH)
        for row in known_rows:
            content = rng.integers(0, 256, size=PLE_ROW_WIDTH, dtype=np.uint8)
            f.seek(data_start + row * PLE_ROW_WIDTH)
            f.write(content.tobytes())
            rows[row] = content
    return rows


def write_tiny_mmproj(path, rng, *, out_hidden=TINY_HIDDEN):
    hidden, inter, patch, positions = 16, 32, 4, 16
    metadata = {
        "general.architecture": _kv_str("clip"),
        "clip.vision.block_count": _kv_u32(1),
        "clip.vision.embedding_length": _kv_u32(hidden),
        "clip.vision.feed_forward_length": _kv_u32(inter),
        "clip.vision.attention.head_count": _kv_u32(2),
        "clip.vision.projection_dim": _kv_u32(out_hidden),
        "clip.vision.patch_size": _kv_u32(patch),
        "clip.vision.spatial_merge_size": _kv_u32(2),
    }
    shapes = {
        "v.patch_embd.weight": (hidden, 3, patch, patch),
        "v.patch_embd.weight.1": (hidden, 3, patch, patch),
        "v.patch_embd.bias": (hidden,),
        "v.position_embd.weight": (positions, hidden),
        "v.blk.0.attn_qkv.weight": (3 * hidden, hidden),
        "v.blk.0.attn_qkv.bias": (3 * hidden,),
        "v.blk.0.attn_out.weight": (hidden, hidden),
        "v.blk.0.attn_out.bias": (hidden,),
        "v.blk.0.ffn_up.weight": (inter, hidden),
        "v.blk.0.ffn_up.bias": (inter,),
        "v.blk.0.ffn_down.weight": (hidden, inter),
        "v.blk.0.ffn_down.bias": (hidden,),
        "v.blk.0.ln1.weight": (hidden,),
        "v.blk.0.ln1.bias": (hidden,),
        "v.blk.0.ln2.weight": (hidden,),
        "v.blk.0.ln2.bias": (hidden,),
        "v.post_ln.weight": (hidden,),
        "v.post_ln.bias": (hidden,),
        "mm.0.weight": (4 * hidden, 4 * hidden),
        "mm.0.bias": (4 * hidden,),
        "mm.2.weight": (out_hidden, 4 * hidden),
        "mm.2.bias": (out_hidden,),
    }
    tensors, truth = [], {}
    for name, shape in shapes.items():
        kind = (
            gq.GGML_BF16 if name.endswith("weight") and len(shape) == 2 else gq.GGML_F32
        )
        entry, values = _float_tensor(rng, name, shape, kind)
        tensors.append(entry)
        truth[name] = values
    write_gguf(path, metadata, tensors)
    return truth


def tiny_skeleton_shapes():
    """GGUF tensor shapes of the tiny model, per the metadata above."""
    hidden, ff, experts = TINY_HIDDEN, 32, 8
    hc, low_rank = 4 * hidden, 8
    shapes = {}
    for layer in range(TINY_LAYERS):
        p = f"blk.{layer}."
        for branch in ("attn", "ffn"):
            shapes[p + f"hc_{branch}_inject.weight"] = (4, hc)
            shapes[p + f"hc_{branch}_norm.weight"] = (hc,)
            shapes[p + f"hc_{branch}_down.weight"] = (low_rank, hc)
            shapes[p + f"hc_{branch}_up.weight"] = (hc, low_rank)
        if (layer + 1) % 4:
            shapes[p + "ssm_a"] = (4,)
            shapes[p + "ssm_dt.bias"] = (4,)
            shapes[p + "ssm_conv1d.weight"] = (256, 4)
            shapes[p + "ssm_alpha.weight"] = (4, hidden)
            shapes[p + "ssm_beta.weight"] = (4, hidden)
            shapes[p + "attn_qkv.weight"] = (256, hidden)
            shapes[p + "attn_gate.weight"] = (128, hidden)
            shapes[p + "ssm_norm.weight"] = (32,)
            shapes[p + "ssm_out.weight"] = (hidden, 128)
        else:
            shapes[p + "attn_q.weight"] = (128, hidden)
            shapes[p + "attn_k.weight"] = (32, hidden)
            shapes[p + "attn_v.weight"] = (32, hidden)
            shapes[p + "attn_output.weight"] = (hidden, 64)
            shapes[p + "attn_q_norm.weight"] = (32,)
            shapes[p + "attn_k_norm.weight"] = (32,)
            shapes[p + "indexer.q_proj.weight"] = (16, hidden)
            shapes[p + "indexer.k_proj.weight"] = (8, hidden)
            shapes[p + "indexer.q_norm.weight"] = (8,)
            shapes[p + "indexer.k_norm.weight"] = (8,)
        shapes[p + "ffn_gate_inp.weight"] = (experts, hidden)
        shapes[p + "ffn_gate_inp_shexp.weight"] = (1, hidden)
        shapes[p + "ffn_gate_shexp.weight"] = (ff, hidden)
        shapes[p + "ffn_up_shexp.weight"] = (ff, hidden)
        shapes[p + "ffn_down_shexp.weight"] = (hidden, ff)
    shapes["blk.1.ple_conv1d.weight"] = (hc, 4)
    shapes["blk.1.ple_key.weight"] = (hc, hidden)
    shapes["blk.1.ple_value.weight"] = (hidden, hidden)
    for part in ("conv", "key", "query"):
        shapes[f"blk.1.ple_norm_{part}.weight"] = (hc,)
    vocab = len(tiny_tokens()[0])
    shapes["token_embd.weight"] = (vocab, hidden)
    shapes["output.weight"] = (vocab, hidden)
    shapes["output_hc_norm.weight"] = (hc,)
    shapes["output_hc_down.weight"] = (low_rank, hc)
    shapes["output_hc_up.weight"] = (hc, low_rank)
    return shapes


def _hyper_entries(prefix, hc, low_rank, inject):
    entries = {}
    if inject:
        entries[f"{prefix}.block_inject_weight.weight"] = ("bf16", 4, hc)
    entries[f"{prefix}.hc_norm.weight"] = ("f32", 1, hc)
    entries[f"{prefix}.input_mix_weight_down.weight"] = ("bf16", low_rank, hc)
    entries[f"{prefix}.input_mix_weight_up.weight"] = ("bf16", hc, low_rank)
    return entries


def write_tiny_mtp(root, rng):
    """Write a tiny ``mtp/`` draft head; returns (dense truth, expert truth).

    Dense truth is ``{name: float32}`` as decoded from the file; expert truth
    is ``(gate, up, down)`` float32 arrays of shape ``[experts, out, in]``.
    """
    hidden, hc, low_rank, shared_ff, experts = TINY_HIDDEN, 4 * TINY_HIDDEN, 8, 32, 8
    entries = {
        "fc_embedding.weight": ("q8_0", hidden, hidden),
        "fc_hidden.weight": ("q8_0", hidden, hidden),
        **_hyper_entries("hyper_connection_mixer", hc, low_rank, False),
        **_hyper_entries("attn_hyper_connection", hc, low_rank, True),
        "mlp.gate.weight": ("bf16", experts, hidden),
        "mlp.shared_expert.down_proj.weight": ("q8_0", hidden, shared_ff),
        "mlp.shared_expert.gate_proj.weight": ("q8_0", shared_ff, hidden),
        "mlp.shared_expert.up_proj.weight": ("q8_0", shared_ff, hidden),
        "mlp.shared_expert_gate.weight": ("bf16", 1, hidden),
        **_hyper_entries("mlp_hyper_connection", hc, low_rank, True),
        "self_attn.indexer.index_qk_proj.weight": ("q8_0", 24, hidden),
        "self_attn.indexer.k_layernorm.weight": ("f32", 1, 8),
        "self_attn.indexer.q_layernorm.weight": ("f32", 1, 8),
        "self_attn.k_norm.weight": ("f32", 1, 32),
        "self_attn.k_proj.weight": ("q8_0", 32, hidden),
        "self_attn.o_proj.weight": ("q8_0", hidden, 64),
        "self_attn.q_norm.weight": ("f32", 1, 32),
        "self_attn.q_proj.weight": ("q8_0", 128, hidden),
        "self_attn.v_proj.weight": ("q8_0", 32, hidden),
        "pre_fc_norm_embedding.weight": ("f32", 1, hidden),
        "pre_fc_norm_hidden.weight": ("f32", 1, hc),
    }
    mtp = Path(root) / "mtp"
    mtp.mkdir()
    blob, lines, dense_truth = bytearray(), [], {}
    for name, (kind, rows, cols) in entries.items():
        if kind == "q8_0":
            raw = random_blocks(rng, gq.GGML_Q8_0, rows * cols, scale=0.002)
            data, values = raw.tobytes(), gq.dequantize_q8_0(raw).reshape(rows, cols)
        elif kind == "f32":
            values = (rng.standard_normal(cols) * 0.05 + 1.0).astype(np.float32)
            data = values.astype("<f4").tobytes()
        else:
            (_, _, _, data), values = _float_tensor(
                rng, name, (rows, cols), gq.GGML_BF16
            )
        lines.append(f"{name} {kind} {rows} {cols} {len(blob)} {len(data)}")
        blob.extend(data)
        blob.extend(b"\0" * ((-len(blob)) % 256))
        dense_truth[name] = values
    (mtp / "dense.txt").write_text("\n".join(lines) + "\n")
    (mtp / "dense.bin").write_bytes(bytes(blob))

    ff = TINY_FF
    gu_blocks, down_blocks = hidden // 64, ff // 64
    gate, up, down = [], [], []
    shifts = np.arange(4, dtype=np.uint8) * 2

    def decode(codes, scales, rows, blocks):
        values = ((codes.reshape(rows, blocks, 16, 1) >> shifts) & 3).astype(np.float32)
        return ((values.reshape(rows, blocks, 64) - 1.0) * scales[..., None]).reshape(
            rows, blocks * 64
        )

    with open(mtp / "experts.bin", "wb") as f:
        for _ in range(experts):
            gu_codes = rng.integers(
                0, 256, size=(2 * ff, gu_blocks, 16), dtype=np.uint8
            )
            down_codes = rng.integers(
                0, 256, size=(hidden, down_blocks, 16), dtype=np.uint8
            )
            gu_scales = rng.uniform(0.01, 0.02, size=(2 * ff, gu_blocks)).astype("<f2")
            down_scales = rng.uniform(0.01, 0.02, size=(hidden, down_blocks)).astype(
                "<f2"
            )
            f.write(gu_codes.tobytes() + down_codes.tobytes())
            f.write(gu_scales.tobytes() + down_scales.tobytes())
            interleaved = decode(
                gu_codes, gu_scales.astype(np.float32), 2 * ff, gu_blocks
            )
            gate.append(interleaved[0::2])
            up.append(interleaved[1::2])
            down.append(
                decode(down_codes, down_scales.astype(np.float32), hidden, down_blocks)
            )
    np.arange(16, dtype="<i4").tofile(mtp / "draft_vocab.bin")
    return dense_truth, (np.stack(gate), np.stack(up), np.stack(down))


@dataclass
class FullPack:
    latent: SyntheticPack
    skeleton: dict
    vision: dict
    ple_rows: dict
    draft_dense: dict = field(default_factory=dict)
    draft_experts: tuple = ()

    @property
    def root(self) -> Path:
        return self.latent.root


def make_full_pack(
    tmp_path, *, with_vision=True, seed=0, ple_rows=(), with_draft_head=True
) -> FullPack:
    """A complete tiny Edge0 pack that converts and loads as ``qwen4_exp``."""
    latent = make_pack(
        tmp_path, layers=TINY_LAYERS, seed=seed, extra_dense_fmts=(4,), ff=TINY_FF
    )
    rng = np.random.default_rng(seed + 1)
    skeleton = write_tiny_skeleton(
        latent.root / "model-skeleton.gguf", rng, tiny_skeleton_shapes()
    )
    rows = write_tiny_ple(latent.root / "ple-fp8.gguf", rng, ple_rows)
    vision = (
        write_tiny_mmproj(latent.root / "mmproj-BF16.gguf", rng) if with_vision else {}
    )
    draft_dense, draft_experts = (
        write_tiny_mtp(latent.root, rng) if with_draft_head else ({}, ())
    )
    return FullPack(latent, skeleton, vision, rows, draft_dense, draft_experts)
