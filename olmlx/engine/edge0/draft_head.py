"""Convert an Edge0 pack's MTP draft head for mlx-vlm speculative decoding.

Qwen3.8-Flash-Next ships a one-layer "multi-token prediction" head: given the
target's last hidden state it cheaply proposes the next few tokens, which the
target then verifies in a single forward. mlx-vlm implements both halves for
``qwen4_exp`` (``Qwen4ExpMTPDraftModel`` + ``mlx_vlm.speculative``); an Edge0
pack carries the head under ``mtp/`` in strata's own layout:

``dense.txt`` / ``dense.bin``
    ``name kind rows cols offset bytes`` per tensor, ``kind`` in
    ``q8_0 | bf16 | f32``. Names are the Hugging Face ones without the
    ``layers.0.`` prefix. ``f32`` entries are RMSNorm weights stored as
    ``1 + w``.
``experts.bin``
    One fixed-size record per (ordinary, full-width) expert in ``Q2_0`` —
    2-bit codes, value ``(code - 1) * d`` with one float16 ``d`` per 64
    weights: gate/up codes (rows interleaved gate, up, gate, ...), down
    codes, gate/up scales, down scales.

Both map exactly onto MLX affine quantization (8-bit / group 32 and 2-bit /
group 64), so the result is a lossless repack into the drafter directory
mlx-vlm's ``load_drafter`` reads.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from olmlx.engine.edge0 import ggml_quants
from olmlx.engine.edge0.affine import bf16 as _bf16
from olmlx.engine.edge0.affine import blocks_to_affine, packed_q8_0_to_bf16_affine
from olmlx.engine.edge0.pack import Edge0PackError

DRAFT_DIRNAME = "mtp_draft"
#: Verify length mlx-vlm starts from (the bonus token plus three drafts);
#: matches the published MLX drafters and Edge0's own ``--spec 4``.
BLOCK_SIZE = 4

_KINDS = {
    "q8_0": ggml_quants.GGML_Q8_0,
    "bf16": ggml_quants.GGML_BF16,
    "f32": ggml_quants.GGML_F32,
}
# Tensors that sit on the drafter itself; everything else is its one layer's.
_TOP_LEVEL = ("fc_embedding.", "fc_hidden.", "pre_fc_norm_", "hyper_connection_mixer.")
_EXPERT_PREFIX = "layers.0.mlp.switch_mlp."
_Q2_GROUP = 64


def has_draft_head(pack_root: Path | str) -> bool:
    mtp = Path(pack_root) / "mtp"
    return (mtp / "dense.txt").is_file() and (mtp / "experts.bin").is_file()


def _convert_dense(mtp: Path) -> dict[str, mx.array]:
    dense = np.memmap(mtp / "dense.bin", dtype=np.uint8, mode="r")
    out: dict[str, mx.array] = {}
    for line in (mtp / "dense.txt").read_text().splitlines():
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        if len(fields) != 6 or fields[1] not in _KINDS:
            raise Edge0PackError(f"mtp/dense.txt: malformed row {line!r}")
        name, kind = fields[0], fields[1]
        rows, cols, offset, nbytes = (int(v) for v in fields[2:])
        raw = dense[offset : offset + nbytes]
        if raw.size != nbytes:
            raise Edge0PackError(f"mtp/dense.bin is truncated at {name!r}")
        key = name if name.startswith(_TOP_LEVEL) else f"layers.0.{name}"
        if kind == "q8_0":
            codes, scales, _ = blocks_to_affine(raw, ggml_quants.GGML_Q8_0, rows)
            if codes.shape[1] * 32 != cols:
                raise Edge0PackError(f"mtp tensor {name!r} is not {rows}x{cols}")
            out.update(
                packed_q8_0_to_bf16_affine(codes, scales, key[: -len(".weight")])
            )
            continue
        values = ggml_quants.dequantize(raw, _KINDS[kind])
        if kind == "f32":
            if "norm" not in name:
                raise Edge0PackError(f"unexpected float32 mtp tensor {name!r}")
            # RMSNorm weight stored as 1 + w; mlx-vlm's norm adds the 1 back.
            out[key] = _bf16(values - 1.0)
        else:
            out[key] = _bf16(values.reshape(rows, cols))
    return out


def _convert_experts(
    mtp: Path, n_experts: int, hidden: int, ff: int
) -> dict[str, mx.array]:
    if hidden % _Q2_GROUP or ff % _Q2_GROUP:
        raise Edge0PackError("mtp experts need widths that are multiples of 64")
    gu_blocks, down_blocks = hidden // _Q2_GROUP, ff // _Q2_GROUP
    gu_code_bytes = 2 * ff * gu_blocks * 16
    down_code_bytes = hidden * down_blocks * 16
    gu_scale_bytes = 2 * ff * gu_blocks * 2
    down_scale_bytes = hidden * down_blocks * 2
    record = gu_code_bytes + down_code_bytes + gu_scale_bytes + down_scale_bytes
    path = mtp / "experts.bin"
    if path.stat().st_size != n_experts * record:
        raise Edge0PackError(
            f"mtp/experts.bin is {path.stat().st_size} bytes, expected "
            f"{n_experts * record} (incomplete download?)"
        )
    blobs = np.memmap(path, dtype=np.uint8, mode="r").reshape(n_experts, record)
    bounds = np.cumsum(
        [0, gu_code_bytes, down_code_bytes, gu_scale_bytes, down_scale_bytes]
    )
    # 2-bit codes are packed four per byte, least significant first — the
    # same bit order as MLX's packed uint32 words, so the bytes are reused.
    gu_codes = np.ascontiguousarray(blobs[:, bounds[0] : bounds[1]]).view("<u4")
    gu_codes = gu_codes.reshape(n_experts, 2 * ff, gu_blocks * 4)
    down_codes = np.ascontiguousarray(blobs[:, bounds[1] : bounds[2]]).view("<u4")
    down_codes = down_codes.reshape(n_experts, hidden, down_blocks * 4)
    gu_scales = np.ascontiguousarray(blobs[:, bounds[2] : bounds[3]]).view("<f2")
    gu_scales = gu_scales.reshape(n_experts, 2 * ff, gu_blocks).astype(np.float32)
    down_scales = np.ascontiguousarray(blobs[:, bounds[3] : bounds[4]]).view("<f2")
    down_scales = down_scales.reshape(n_experts, hidden, down_blocks).astype(np.float32)

    out: dict[str, mx.array] = {}

    def put(name: str, codes: np.ndarray, scales: np.ndarray) -> None:
        # value = (code - 1) * d  ==  d * code + (-d)
        scales_bf16 = _bf16(scales)
        out[f"{_EXPERT_PREFIX}{name}.weight"] = mx.array(np.ascontiguousarray(codes))
        out[f"{_EXPERT_PREFIX}{name}.scales"] = scales_bf16
        out[f"{_EXPERT_PREFIX}{name}.biases"] = scales_bf16 * -1.0

    # Gate and up rows are interleaved: row 2r is gate r, row 2r + 1 is up r.
    put("gate_proj", gu_codes[:, 0::2], gu_scales[:, 0::2])
    put("up_proj", gu_codes[:, 1::2], gu_scales[:, 1::2])
    put("down_proj", down_codes, down_scales)
    return out


def convert_draft_head(
    pack_root: Path | str, model_dir: Path | str, text_config: dict[str, Any]
) -> Path:
    """Write ``<model_dir>/mtp_draft`` from ``<pack_root>/mtp``. Returns it."""
    mtp = Path(pack_root) / "mtp"
    out_dir = Path(model_dir) / DRAFT_DIRNAME
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").unlink(missing_ok=True)

    tensors = _convert_dense(mtp)
    tensors.update(
        _convert_experts(
            mtp,
            int(text_config["num_experts"]),
            int(text_config["hidden_size"]),
            int(text_config["moe_intermediate_size"]),
        )
    )
    mx.save_safetensors(
        str(out_dir / "model.safetensors"), tensors, metadata={"format": "mlx"}
    )

    two_bit = {"group_size": _Q2_GROUP, "bits": 2, "mode": "affine"}
    draft_text_config = {
        key: value for key, value in text_config.items() if key != "ple_storage"
    }
    config = {
        "model_type": "qwen4_exp_mtp",
        "block_size": BLOCK_SIZE,
        "tie_word_embeddings": False,
        "text_config": draft_text_config,
        # Modules with ``.scales`` in the file take these 8-bit defaults; the
        # experts are 2-bit and must say so (mlx merges nothing).
        "quantization": {
            "group_size": 32,
            "bits": 8,
            "mode": "affine",
            **{
                f"{_EXPERT_PREFIX}{name}": two_bit
                for name in ("gate_proj", "up_proj", "down_proj")
            },
        },
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    return out_dir
