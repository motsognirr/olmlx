from __future__ import annotations

import glob
import json
import os
from typing import Any

import mlx.core as mx
import numpy as np

from olmlx.engine.rerank.config import RerankerConfig
from olmlx.engine.rerank.model import XLMRobertaCrossEncoder


def detect_layout(keys: list[str]) -> str:
    for k in keys:
        if "mixer.Wqkv" in k or "emb_ln" in k:
            return "flash"
    return "standard"


def _emb_and_head(sd: dict[str, Any], emb_ln_prefix: str) -> dict[str, Any]:
    e = "roberta.embeddings."
    return {
        "embeddings.word_embeddings.weight": sd[f"{e}word_embeddings.weight"],
        "embeddings.position_embeddings.weight": sd[f"{e}position_embeddings.weight"],
        "embeddings.token_type_embeddings.weight": sd[
            f"{e}token_type_embeddings.weight"
        ],
        "embeddings.LayerNorm.weight": sd[f"{emb_ln_prefix}.weight"],
        "embeddings.LayerNorm.bias": sd[f"{emb_ln_prefix}.bias"],
        "classifier.dense.weight": sd["classifier.dense.weight"],
        "classifier.dense.bias": sd["classifier.dense.bias"],
        "classifier.out_proj.weight": sd["classifier.out_proj.weight"],
        "classifier.out_proj.bias": sd["classifier.out_proj.bias"],
    }


def _missing_key_error(layout: str, exc: KeyError) -> KeyError:
    return KeyError(
        f"{layout}-layout checkpoint is missing expected key {exc} "
        "(was the wrong layout detected?)"
    )


def remap_standard(sd: dict[str, Any], cfg: RerankerConfig) -> dict[str, mx.array]:
    try:
        out = _emb_and_head(sd, "roberta.embeddings.LayerNorm")
        for i in range(cfg.num_hidden_layers):
            p = f"roberta.encoder.layer.{i}."
            q = f"layers.{i}."
            for proj in ("query", "key", "value"):
                out[f"{q}attention_self.{proj}.weight"] = sd[
                    f"{p}attention.self.{proj}.weight"
                ]
                out[f"{q}attention_self.{proj}.bias"] = sd[
                    f"{p}attention.self.{proj}.bias"
                ]
            out[f"{q}attention_output_dense.weight"] = sd[
                f"{p}attention.output.dense.weight"
            ]
            out[f"{q}attention_output_dense.bias"] = sd[
                f"{p}attention.output.dense.bias"
            ]
            out[f"{q}attention_output_norm.weight"] = sd[
                f"{p}attention.output.LayerNorm.weight"
            ]
            out[f"{q}attention_output_norm.bias"] = sd[
                f"{p}attention.output.LayerNorm.bias"
            ]
            out[f"{q}intermediate_dense.weight"] = sd[f"{p}intermediate.dense.weight"]
            out[f"{q}intermediate_dense.bias"] = sd[f"{p}intermediate.dense.bias"]
            out[f"{q}output_dense.weight"] = sd[f"{p}output.dense.weight"]
            out[f"{q}output_dense.bias"] = sd[f"{p}output.dense.bias"]
            out[f"{q}output_norm.weight"] = sd[f"{p}output.LayerNorm.weight"]
            out[f"{q}output_norm.bias"] = sd[f"{p}output.LayerNorm.bias"]
    except KeyError as exc:
        raise _missing_key_error("standard", exc) from exc
    return {k: mx.array(np.asarray(v)) for k, v in out.items()}


def remap_flash(sd: dict[str, Any], cfg: RerankerConfig) -> dict[str, mx.array]:
    h = cfg.hidden_size
    try:
        out = _emb_and_head(sd, "roberta.emb_ln")
        for i in range(cfg.num_hidden_layers):
            p = f"roberta.encoder.layers.{i}."
            q = f"layers.{i}."
            wqkv = np.asarray(sd[f"{p}mixer.Wqkv.weight"])
            bqkv = np.asarray(sd[f"{p}mixer.Wqkv.bias"])
            if wqkv.shape[0] != 3 * h:
                raise ValueError(
                    f"layer {i}: expected fused Wqkv rows == 3 * hidden_size "
                    f"({3 * h}), got {wqkv.shape[0]}"
                )
            out[f"{q}attention_self.query.weight"] = wqkv[:h]
            out[f"{q}attention_self.key.weight"] = wqkv[h : 2 * h]
            out[f"{q}attention_self.value.weight"] = wqkv[2 * h :]
            out[f"{q}attention_self.query.bias"] = bqkv[:h]
            out[f"{q}attention_self.key.bias"] = bqkv[h : 2 * h]
            out[f"{q}attention_self.value.bias"] = bqkv[2 * h :]
            out[f"{q}attention_output_dense.weight"] = sd[f"{p}mixer.out_proj.weight"]
            out[f"{q}attention_output_dense.bias"] = sd[f"{p}mixer.out_proj.bias"]
            out[f"{q}attention_output_norm.weight"] = sd[f"{p}norm1.weight"]
            out[f"{q}attention_output_norm.bias"] = sd[f"{p}norm1.bias"]
            out[f"{q}intermediate_dense.weight"] = sd[f"{p}mlp.fc1.weight"]
            out[f"{q}intermediate_dense.bias"] = sd[f"{p}mlp.fc1.bias"]
            out[f"{q}output_dense.weight"] = sd[f"{p}mlp.fc2.weight"]
            out[f"{q}output_dense.bias"] = sd[f"{p}mlp.fc2.bias"]
            out[f"{q}output_norm.weight"] = sd[f"{p}norm2.weight"]
            out[f"{q}output_norm.bias"] = sd[f"{p}norm2.bias"]
    except KeyError as exc:
        raise _missing_key_error("flash", exc) from exc
    return {k: mx.array(np.asarray(v)) for k, v in out.items()}


def _load_state_dict(path: str) -> dict[str, mx.array]:
    files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no .safetensors weights in {path}")
    sd: dict[str, mx.array] = {}
    for f in files:
        sd.update(mx.load(f))  # mx.load returns {name: mx.array}
    return sd


def load_cross_encoder(path: str) -> XLMRobertaCrossEncoder:
    """Build an XLMRobertaCrossEncoder from a local model directory."""
    with open(os.path.join(path, "config.json")) as fh:
        cfg = RerankerConfig.from_dict(json.load(fh))
    sd = _load_state_dict(path)
    layout = detect_layout(list(sd.keys()))
    flat = remap_flash(sd, cfg) if layout == "flash" else remap_standard(sd, cfg)
    model = XLMRobertaCrossEncoder(cfg)
    model.load_weights(list(flat.items()))
    model.eval()
    mx.eval(model.parameters())
    return model
