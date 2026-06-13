"""Train a DFlash draft from target-generated JSONL sequences.

Implements the upstream DFlash recipe at local scale: target-generated
training text (see dflash_gen_data.py), position-decay loss weighting,
many anchor windows per step, ~6 epochs.

Uses prepare_dflash_draft's `_batch_iterator` injection hook to feed
custom data; everything else (pivot selection, loss, optimizer, save)
is the production training path.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import mlx.core as mx

from olmlx.engine.dflash.prepare import prepare_dflash_draft

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def _get_pad_id(target: str) -> int:
    """Read pad_token_id from tokenizer_config.json; fall back to eos_token_id."""
    import json as _json

    cfg_path = Path(target).expanduser() / "tokenizer_config.json"
    tok_path = Path(target).expanduser() / "tokenizer.json"
    if cfg_path.exists():
        cfg = _json.loads(cfg_path.read_text())
        pad_name = cfg.get("pad_token")
        eos_name = cfg.get("eos_token")
        if tok_path.exists():
            tok = _json.loads(tok_path.read_text())
            vocab = tok.get("model", {}).get("vocab", {})
            if pad_name and pad_name in vocab:
                return int(vocab[pad_name])
            if eos_name and eos_name in vocab:
                return int(vocab[eos_name])
    return 0


def batch_iterator(path: Path, batch_size: int, min_len: int, pad_id: int = 0):
    seqs = []
    with path.open() as f:
        for line in f:
            ids = json.loads(line)["ids"]
            if len(ids) >= min_len:
                seqs.append(ids)
    # Sort by length, fixed buckets of batch_size, pad to in-batch max.
    seqs.sort(key=len)
    batches = []
    for i in range(0, len(seqs) - batch_size + 1, batch_size):
        rows = seqs[i : i + batch_size]
        m = max(len(r) for r in rows)
        batches.append([r + [pad_id] * (m - len(r)) for r in rows])
    print(f"{len(seqs)} seqs -> {len(batches)} batches/epoch", flush=True)
    rng = random.Random(0)
    while True:  # cycle epochs; prepare stops at --steps
        rng.shuffle(batches)
        for b in batches:
            yield mx.array(b)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--windows", type=int, default=12)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--gamma", type=float, default=8.0)
    ap.add_argument(
        "--pad-id",
        type=int,
        default=None,
        help="Pad token ID for within-batch padding (auto-detected from tokenizer if omitted)",
    )
    args = ap.parse_args()

    pad_id = args.pad_id if args.pad_id is not None else _get_pad_id(args.target)
    logging.getLogger(__name__).info("Using pad_id=%d for batch padding", pad_id)

    out = prepare_dflash_draft(
        model_path=Path(args.target).expanduser(),
        steps=args.steps,
        batch_size=args.batch_size,
        block_size=args.block_size,
        lr=args.lr,
        position_decay_gamma=args.gamma,
        train_windows_per_step=args.windows,
        output_dir=Path(args.output).expanduser(),
        _batch_iterator=batch_iterator(
            Path(args.data).expanduser(),
            args.batch_size,
            min_len=2 * args.block_size + 1 + 16,
            pad_id=pad_id,
        ),
    )
    print(f"saved draft to {out}")


if __name__ == "__main__":
    main()
