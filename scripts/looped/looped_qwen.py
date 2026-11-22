"""Convert a Qwen (qwen2 / qwen3, also llama-style) model into a looped
transformer and recover it by distillation from the original — in MLX.

Layout (example Qwen2.5-3B, 36 layers, --prelude 6 --core 8 --loops 3 --coda 6):

    embed -> prelude (layers 0-5, frozen, shared with teacher)
          -> core (8 shared layers) x 3 passes, a message gate between passes
          -> coda (copy of layers 30-35, trainable) -> norm -> lm_head

Same FLOPs per token as the original, ~40 % fewer layer parameters.

Two init modes:
  teacher  gate starts OPEN (pass t gets pass t-1's output). Core weights are the
           mean of the 3 original layer groups; each pass gets a per-loop LoRA
           initialised by SVD of (original - mean), plus exact per-loop biases and
           norms. With rank >= full the student reproduces the teacher exactly at
           step 0. A practical rank does not: on real models the layer
           differences are near full-rank, and at rank 64-256 the init is about
           as far from the teacher as rank 0 (see README).
  inkling  gate starts CLOSED (every pass sees only the prelude output, as in
           Inkling 2 Mini) and opens during training.

Commands:
  train     distill student from teacher
  eval      perplexity of teacher vs student on a held-out text file
  generate  sample text from a trained student (KV cache, via mlx_lm.generate)

Data: --data accepts .jsonl ({"text": ...}) or .txt, or --hf-dataset for
streaming from the Hugging Face hub (needs `pip install datasets`).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
from mlx import nn
from mlx.utils import tree_flatten, tree_map
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import KVCache

# --------------------------------------------------------------------------
# Per-loop adapted modules
# --------------------------------------------------------------------------


class LoopLinear(nn.Module):
    """Shared Linear + per-loop LoRA delta and per-loop bias delta."""

    def __init__(self, shared: nn.Linear, n_loops: int, rank: int):
        super().__init__()
        self.shared = shared
        out_dim, in_dim = shared.weight.shape
        self.rank = rank
        if rank > 0:
            self.lora_a = mx.zeros((n_loops, rank, in_dim))
            self.lora_b = mx.zeros((n_loops, out_dim, rank))
        if "bias" in shared:
            self.bias_delta = mx.zeros((n_loops, out_dim))
        self._t = 0

    def __call__(self, x):
        t = self._t
        y = self.shared(x)
        if self.rank > 0:
            y = y + (x @ self.lora_a[t].T) @ self.lora_b[t].T
        if "bias_delta" in self:
            y = y + self.bias_delta[t]
        return y


class LoopRMSNorm(nn.Module):
    """RMSNorm with a separate weight per loop."""

    def __init__(self, base: nn.RMSNorm, n_loops: int):
        super().__init__()
        self.weight = mx.repeat(base.weight[None], n_loops, axis=0)
        self.eps = base.eps
        self._t = 0

    def __call__(self, x):
        return mx.fast.rms_norm(x, self.weight[self._t], self.eps)


def _replace_modules(module: nn.Module, n_loops: int, rank: int):
    """Swap every nn.Linear / nn.RMSNorm inside `module` for its loop version."""
    for name, child in list(module.items()):
        if isinstance(child, nn.Linear):
            module[name] = LoopLinear(child, n_loops, rank)
        elif isinstance(child, nn.RMSNorm):
            module[name] = LoopRMSNorm(child, n_loops)
        elif isinstance(child, nn.Module):
            _replace_modules(child, n_loops, rank)
        elif isinstance(child, list):
            for c in child:
                if isinstance(c, nn.Module):
                    _replace_modules(c, n_loops, rank)


def _set_loop(module: nn.Module, t: int):
    for _, m in module.named_modules():
        if isinstance(m, (LoopLinear, LoopRMSNorm)):
            m._t = t


class MessageGate(nn.Module):
    """s = (1-g)*message + g*h0, g = sigmoid(W [norm(h0); norm(m)] + b).

    W starts at zero, so g is the constant sigmoid(bias_init) at step 0:
    bias -8 -> message passes through (teacher mode),
    bias +8 -> message blocked, every pass sees h0 (inkling mode).

    sigmoid never reaches 0, so with exact_open (teacher mode) g is shifted by
    its step-0 value, g' = (g - g0) / (1 - g0) with g0 = sigmoid(bias_init) a
    constant: g' is exactly 0 at init, and its gradient differs from g's only by
    the constant 1/(1-g0), so training behaves the same. Without the shift the
    gate leaks ~3e-4 of h0 into every pass and the student can't match the
    teacher exactly, even at full rank.
    """

    def __init__(self, dim: int, eps: float, bias_init: float, exact_open=False):
        super().__init__()
        self.norm_h = nn.RMSNorm(dim, eps=eps)
        self.norm_m = nn.RMSNorm(dim, eps=eps)
        self.proj = nn.Linear(2 * dim, dim)
        self.proj.weight = mx.zeros_like(self.proj.weight)
        self.proj.bias = mx.full((dim,), bias_init)
        self._bias_init = bias_init
        self._exact_open = exact_open

    def __call__(self, h0, m):
        z = self.proj(mx.concatenate([self.norm_h(h0), self.norm_m(m)], axis=-1))
        g = mx.sigmoid(z)
        if self._exact_open:
            # same op and dtype as g, so g - g0 is exactly 0 while z == bias_init
            g0 = mx.sigmoid(mx.array(self._bias_init, dtype=z.dtype))
            g = (g - g0) / (1 - g0)
        return (1 - g) * m + g * h0


class Core(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def __call__(self, h, mask, caches):
        for layer, c in zip(self.layers, caches):
            h = layer(h, mask, c)
        return h


# --------------------------------------------------------------------------
# Low-rank init
# --------------------------------------------------------------------------


def low_rank_factors(delta: mx.array, rank: int, oversample: int = 8, iters: int = 2):
    """Randomized SVD: delta (out,in) ~= B @ A with B (out,r), A (r,in)."""
    m, n = delta.shape
    k = min(rank + oversample, m, n)
    y = delta @ mx.random.normal((n, k))
    for _ in range(iters):
        q, _ = mx.linalg.qr(y, stream=mx.cpu)
        y = delta @ (delta.T @ q)
    q, _ = mx.linalg.qr(y, stream=mx.cpu)
    b = q.T @ delta  # (k, n)
    # svd of the small b via eigh of b b^T (avoids an n x n Vt)
    evals, evecs = mx.linalg.eigh(b @ b.T, stream=mx.cpu)
    order = mx.argsort(-evals)[: min(rank, k)]
    s = mx.sqrt(mx.maximum(evals[order], 0.0))
    ub = evecs[:, order]
    inv_s = mx.where(s > 1e-8, 1.0 / mx.maximum(s, 1e-8), 0.0)
    vt = (ub.T @ b) * inv_s[:, None]
    u = q @ ub
    sq = mx.sqrt(s)
    B = u * sq[None, :]
    A = vt * sq[:, None]
    r = A.shape[0]
    if r < rank:  # pad when rank > matrix rank
        A = mx.concatenate([A, mx.zeros((rank - r, n))], axis=0)
        B = mx.concatenate([B, mx.zeros((m, rank - r))], axis=1)
    return B, A


# --------------------------------------------------------------------------
# Student
# --------------------------------------------------------------------------


class _Ref:
    """Holds a module without registering it as a child of an nn.Module."""

    def __init__(self, obj):
        self.obj = obj


def _head(base, h):
    if base.args.tie_word_embeddings:
        return base.model.embed_tokens.as_linear(h)
    return base.lm_head(h)


def _new_block(base, src=None, dtype=mx.float32):
    blk = type(base.model.layers[0])(base.args)
    if src is not None:
        blk.update(tree_map(lambda a: a.astype(dtype), src.parameters()))
    return blk


class LoopedLM(nn.Module):
    def __init__(
        self,
        base,
        prelude: int,
        core: int,
        coda: int,
        loops: int,
        rank: int,
        mode: str = "teacher",
        init: bool = True,
        grad_checkpoint: bool = True,
        train_dtype=mx.float32,
    ):
        super().__init__()
        quantized = [
            name
            for name, m in base.named_modules()
            if isinstance(m, (nn.QuantizedLinear, nn.QuantizedEmbedding))
        ]
        if quantized:
            # the core is a weight mean and per-loop adapters wrap nn.Linear only:
            # packed weights would be averaged as integers and left unadapted
            raise ValueError(
                f"quantized base model ({quantized[0]}, ...): use an unquantized "
                "(bf16/fp16) checkpoint"
            )
        layers = base.model.layers
        n = len(layers)
        if min(prelude, coda, rank) < 0 or core < 1 or loops < 1:
            raise ValueError(
                "need prelude, coda, rank >= 0 and core, loops >= 1 "
                f"(got prelude={prelude} core={core} coda={coda} loops={loops} "
                f"rank={rank})"
            )
        if prelude + core * loops + coda != n:
            raise ValueError(
                f"prelude + core*loops + coda must equal {n} layers "
                f"(got {prelude}+{core}*{loops}+{coda})"
            )
        self.cfg = {
            "prelude": prelude,
            "core": core,
            "coda": coda,
            "loops": loops,
            "rank": rank,
            "mode": mode,
        }
        self.base = _Ref(
            base
        )  # plain object => not a child module (teacher stays out of the tree)
        self.n_prelude, self.n_core, self.n_coda, self.loops = (
            prelude,
            core,
            coda,
            loops,
        )
        self.grad_checkpoint = grad_checkpoint
        args = base.args
        eps = getattr(args, "rms_norm_eps", 1e-6)

        # core: mean of the `loops` corresponding original layers
        groups = [
            [layers[prelude + t * core + i] for t in range(loops)] for i in range(core)
        ]
        core_layers = []
        for i in range(core):
            blk = _new_block(base)
            if init:
                ps = [dict(tree_flatten(layer.parameters())) for layer in groups[i]]
                mean = {
                    k: sum(p[k].astype(mx.float32) for p in ps) / loops for k in ps[0]
                }
                blk.load_weights(list(mean.items()))
            core_layers.append(blk)
        for blk in core_layers:
            _replace_modules(blk, loops, rank)
        self.core = Core(core_layers)

        if init:
            self._init_adapters(groups, rank)

        bias = -8.0 if mode == "teacher" else 8.0
        self.gates = [
            MessageGate(args.hidden_size, eps, bias, exact_open=mode == "teacher")
            for _ in range(loops - 1)
        ]
        self.coda = [_new_block(base, layers[n - coda + j]) for j in range(coda)]

        self.set_dtype(train_dtype)
        mx.eval(self.parameters())

    # ----- init ----------------------------------------------------------
    def _init_adapters(self, groups, rank):
        for i, blk in enumerate(self.core.layers):
            orig = [dict(tree_flatten(layer.parameters())) for layer in groups[i]]
            for path, m in blk.named_modules():
                pre = f"{path}." if path else ""
                if isinstance(m, LoopRMSNorm):
                    m.weight = mx.stack(
                        [o[f"{pre}weight"].astype(mx.float32) for o in orig]
                    )
                elif isinstance(m, LoopLinear):
                    w_sh = m.shared.weight
                    if "bias_delta" in m:
                        m.bias_delta = mx.stack(
                            [
                                o[f"{pre}bias"].astype(mx.float32) - m.shared.bias
                                for o in orig
                            ]
                        )
                    if rank > 0:
                        As, Bs = [], []
                        for o in orig:
                            B, A = low_rank_factors(
                                o[f"{pre}weight"].astype(mx.float32) - w_sh, rank
                            )
                            As.append(A)
                            Bs.append(B)
                        m.lora_a, m.lora_b = mx.stack(As), mx.stack(Bs)
                        mx.eval(m.lora_a, m.lora_b)

    def trainable_modules(self):
        return [self.core, self.gates, self.coda]

    # ----- forward -------------------------------------------------------
    def prelude_forward(self, inputs, cache=None):
        base = self.base.obj
        h = base.model.embed_tokens(inputs)
        mask = create_attention_mask(h, cache[0] if cache else None)
        for j, layer in enumerate(base.model.layers[: self.n_prelude]):
            h = layer(h, mask, cache[j] if cache else None)
        return h

    def _core_pass(self, t, h, mask, caches):
        if self.grad_checkpoint and caches is None and self.training:
            core = self.core

            def inner(params, h):
                core.update(params)
                _set_loop(core, t)  # also set on recompute during backward
                return core(h, mask, [None] * self.n_core)

            return mx.checkpoint(inner)(core.trainable_parameters(), h)
        _set_loop(self.core, t)
        return self.core(h, mask, caches or [None] * self.n_core)

    def from_prelude(self, h0, cache=None):
        base = self.base.obj
        p, k = self.n_prelude, self.n_core
        # core/coda caches advance together; cache[0] (prelude) is already past h0
        mask = create_attention_mask(h0, cache[p] if cache else None)
        m = h0
        for t in range(self.loops):
            s = h0 if t == 0 else self.gates[t - 1](h0, m)
            cs = cache[p + t * k : p + (t + 1) * k] if cache else None
            m = self._core_pass(t, s, mask, cs)
        h = m
        off = p + self.loops * k
        for j, layer in enumerate(self.coda):
            h = layer(h, mask, cache[off + j] if cache else None)
        return _head(base, base.model.norm(h))

    def __call__(self, inputs, cache=None, **_):
        return self.from_prelude(self.prelude_forward(inputs, cache), cache)

    def make_cache(self):
        return [
            KVCache()
            for _ in range(self.n_prelude + self.n_core * self.loops + self.n_coda)
        ]

    @property
    def layers(self):  # for tools that only need len(model.layers)
        return [None] * (self.n_prelude + self.n_core * self.loops + self.n_coda)


def load_teacher(path, tok, base):
    """A separate (e.g. larger) teacher. KL is taken token by token over the
    vocabulary, so it must share the student's tokenizer exactly."""
    teacher, t_tok = load(path)
    teacher.freeze()
    if t_tok.get_vocab() != tok.get_vocab():
        raise ValueError(f"teacher {path} has a different tokenizer vocabulary")
    t_vocab, s_vocab = teacher.args.vocab_size, base.args.vocab_size
    if t_vocab != s_vocab:
        raise ValueError(f"teacher logits have {t_vocab} classes, student {s_vocab}")
    return teacher


def teacher_from_prelude(base, h0, prelude):
    mask = create_attention_mask(h0, None)
    h = h0
    for layer in base.model.layers[prelude:]:
        h = layer(h, mask, None)
    return _head(base, base.model.norm(h))


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def iter_texts(args) -> Iterator[str]:
    if args.hf_dataset:
        from datasets import load_dataset

        while True:  # loop over epochs, like a local file (no StopIteration mid-run)
            ds = load_dataset(
                args.hf_dataset, args.hf_subset, split="train", streaming=True
            )
            for row in ds:
                yield row[args.text_field]
    path = Path(args.data)
    while True:  # loop over epochs
        with path.open() as f:
            if path.suffix == ".jsonl":
                for line in f:
                    if line.strip():
                        yield json.loads(line)[args.text_field]
            else:
                yield f.read()


def iter_batches(
    tokenizer, texts, batch: int, seq: int, shuffle_buf: int = 512, seed: int = 0
):
    rng = random.Random(seed)
    eos = tokenizer.eos_token_id
    stream: list[int] = []
    buf: list[list[int]] = []
    for text in texts:
        stream.extend(tokenizer.encode(text) + [eos])
        pos = 0
        while len(stream) - pos >= seq + 1:
            buf.append(stream[pos : pos + seq + 1])
            pos += seq + 1
            if len(buf) >= max(shuffle_buf, batch):
                if shuffle_buf:  # 0 = keep stream order (eval)
                    rng.shuffle(buf)
                yield mx.array(buf[:batch])
                buf = buf[batch:]
        # drop consumed tokens once per text: slicing per window copied the
        # remaining tail every time, quadratic for a long text (a whole .txt)
        del stream[:pos]
    # only reached for a finite `texts` (iter_texts loops forever); a final
    # partial batch is dropped so every batch keeps its shape
    while len(buf) >= batch:
        yield mx.array(buf[:batch])
        buf = buf[batch:]


# --------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------


def distill_loss(student, h0, t_logits, targets, temp, ce_weight):
    s_logits = student.from_prelude(h0).astype(mx.float32)
    t_logits = t_logits.astype(mx.float32)
    t_lp = t_logits / temp - mx.logsumexp(t_logits / temp, axis=-1, keepdims=True)
    s_lp = s_logits / temp - mx.logsumexp(s_logits / temp, axis=-1, keepdims=True)
    kl = (mx.exp(t_lp) * (t_lp - s_lp)).sum(-1).mean() * temp**2
    ce = nn.losses.cross_entropy(s_logits, targets).mean()
    return (1 - ce_weight) * kl + ce_weight * ce, (kl, ce)


# --------------------------------------------------------------------------
# Save / load
# --------------------------------------------------------------------------


# training settings recorded in config.json (compared on --resume)
_TRAIN_KEYS = (
    "lr",
    "warmup",
    "steps",
    "batch",
    "seq",
    "accum",
    "temp",
    "ce_weight",
    "clip",
    "weight_decay",
    "seed",
    "data",
    "hf_dataset",
    "hf_subset",
    "text_field",
    "grad_checkpoint",
    "teacher",
)


def save_student(student, out: Path, base_path: str, step: int):
    out.mkdir(parents=True, exist_ok=True)
    # every student parameter (the teacher is a _Ref, not in the tree), frozen
    # ones included, so the file fully describes the model and loads strictly
    flat = dict(tree_flatten(student.parameters()))
    # write both to temp files, then rename: an interrupted save (OOM, Ctrl-C)
    # must not leave a truncated checkpoint that --resume can't load
    tmp_w = out / "student.tmp.safetensors"  # mlx appends .safetensors otherwise
    tmp_c = out / "config.tmp.json"
    mx.save_safetensors(str(tmp_w), flat)
    tmp_c.write_text(
        json.dumps({"base": base_path, "step": step, **student.cfg}, indent=2)
    )
    os.replace(tmp_w, out / "student.safetensors")
    os.replace(tmp_c, out / "config.json")


def load_student(out: Path, grad_checkpoint=False):
    cfg = json.loads((out / "config.json").read_text())
    base, tok = load(cfg["base"])
    base.freeze()
    st = LoopedLM(
        base,
        cfg["prelude"],
        cfg["core"],
        cfg["coda"],
        cfg["loops"],
        cfg["rank"],
        cfg["mode"],
        init=False,
        grad_checkpoint=grad_checkpoint,
    )
    for key in ("freeze_coda", "train"):
        if key in cfg:
            st.cfg[key] = cfg[key]
    # strict: a missing parameter would otherwise keep its random init=False value
    st.load_weights(list(mx.load(str(out / "student.safetensors")).items()))
    mx.eval(st.parameters())
    return st, base, tok, cfg


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def cmd_train(args):
    mx.random.seed(args.seed)
    out = Path(args.out)
    if args.resume:
        student, base, tok, cfg = load_student(
            out, grad_checkpoint=args.grad_checkpoint
        )
        start = cfg.get("step", 0)
        if args.steps <= start:
            raise ValueError(
                f"checkpoint is at step {start}: pass --steps > {start} to continue"
            )
        # a frozen coda stays frozen; --freeze-coda can still add it
        was_frozen = cfg.get("freeze_coda", False)
        freeze_coda = args.freeze_coda or was_frozen
        if freeze_coda != was_frozen:
            print("note: --freeze-coda freezes the checkpoint's trained coda")
        elif was_frozen and not args.freeze_coda:
            print("note: the coda stays frozen (checkpoint used --freeze-coda)")
    else:
        freeze_coda = args.freeze_coda
        base, tok = load(args.model)
        base.freeze()
        t0 = time.time()
        student = LoopedLM(
            base,
            args.prelude,
            args.core,
            args.coda,
            args.loops,
            args.rank,
            args.mode,
            grad_checkpoint=args.grad_checkpoint,
        )
        print(f"student built in {time.time() - t0:.0f}s")
        start = 0
    base.freeze()
    student.freeze()
    for m in student.trainable_modules():
        if isinstance(m, list):
            for x in m:
                x.unfreeze()
        else:
            m.unfreeze()
    teacher = load_teacher(args.teacher, tok, base) if args.teacher else None
    student.cfg["freeze_coda"] = freeze_coda
    if freeze_coda:
        for x in student.coda:
            x.freeze()
    student.train()

    n_train = sum(v.size for _, v in tree_flatten(student.trainable_parameters()))
    print(f"trainable params: {n_train / 1e6:.1f}M")

    train_cfg = {k: getattr(args, k) for k in _TRAIN_KEYS}
    if args.resume:
        # the architecture comes from the checkpoint; say so if the CLI differs
        ignored = [
            f"--{k} {getattr(args, k)} (checkpoint: {cfg[k]})"
            for k in ("prelude", "core", "coda", "loops", "rank", "mode")
            if getattr(args, k) != cfg[k]
        ]
        if ignored:
            print("note: ignored on --resume: " + ", ".join(ignored))
        # training settings may change (extending --steps is the usual reason
        # to resume), but visibly
        prev = student.cfg.get("train", {})
        changed = [
            f"{k} {prev[k]} -> {v}"
            for k, v in train_cfg.items()
            if k in prev and prev[k] != v
        ]
        if changed:
            print("note: training settings change: " + ", ".join(changed))
    student.cfg["train"] = train_cfg
    sched = optim.join_schedules(
        [
            optim.linear_schedule(0.0, args.lr, args.warmup),
            optim.cosine_decay(
                args.lr, max(args.steps - args.warmup, 1), args.lr * 0.1
            ),
        ],
        [args.warmup],
    )
    # on --resume, continue the schedule where it left off (optimizer moments restart)
    opt = optim.AdamW(
        learning_rate=lambda i: sched(i + start), weight_decay=args.weight_decay
    )
    lvg = nn.value_and_grad(student, distill_loss)
    base_path = cfg["base"] if args.resume else args.model
    batches = iter_batches(tok, iter_texts(args), args.batch, args.seq, seed=args.seed)
    if start:
        # Skip what the checkpoint already trained on, counted with the
        # checkpoint's own settings. With the same batch/seq/seed/data the
        # resumed run sees exactly the batches it would have seen (a changed
        # --accum is exact too); otherwise windows differ, so skip by tokens.
        old = cfg.get("train", {})
        o = {k: old.get(k, getattr(args, k)) for k in ("accum", "batch", "seq")}
        n_skip = math.ceil(
            start * o["accum"] * o["batch"] * o["seq"] / (args.batch * args.seq)
        )
        moved = [
            k
            for k in ("batch", "seq", "seed", "data", "hf_dataset", "hf_subset")
            if k in old and old[k] != getattr(args, k)
        ]
        if moved:
            print(f"note: {', '.join(moved)} changed: the data skip is approximate")
        print(f"skipping {n_skip} batches already trained on")
        for _ in range(n_skip):
            next(batches)

    tokens_per_step = args.batch * args.seq * args.accum
    t_last, toks = time.time(), 0
    for step in range(start, args.steps):
        acc, kl_sum, ce_sum = None, 0.0, 0.0
        for _ in range(args.accum):
            x = next(batches)
            inp, tgt = x[:, :-1], x[:, 1:]
            h0 = mx.stop_gradient(student.prelude_forward(inp))
            if teacher is None:
                # the base continues from the student's own prelude output
                # (on --resume, args.prelude is the CLI default, not the layout)
                t_logits = teacher_from_prelude(base, h0, student.n_prelude)
            else:
                t_logits = teacher(inp)
            t_logits = mx.stop_gradient(t_logits)
            (_, (kl, ce)), g = lvg(
                student, h0, t_logits, tgt, args.temp, args.ce_weight
            )
            acc = g if acc is None else tree_map(mx.add, acc, g)
            mx.eval(acc, kl, ce)
            kl_sum += kl.item()
            ce_sum += ce.item()
        acc = tree_map(lambda a: a / args.accum, acc)
        acc, gnorm = optim.clip_grad_norm(acc, args.clip)
        opt.update(student, acc)
        mx.eval(student.trainable_parameters(), opt.state)
        toks += tokens_per_step

        if (step + 1) % args.log_every == 0:
            dt = time.time() - t_last
            print(
                f"step {step + 1:6d} | kl {kl_sum / args.accum:.4f} | ce {ce_sum / args.accum:.4f}"
                f" | gnorm {gnorm.item():.2f} | lr {opt.learning_rate.item():.2e}"
                f" | {toks / dt:.0f} tok/s | peak {mx.get_peak_memory() / 1e9:.1f} GB",
                flush=True,
            )
            t_last, toks = time.time(), 0
        if (step + 1) % args.save_every == 0 or step + 1 == args.steps:
            save_student(student, out, base_path, step + 1)


def cmd_eval(args):
    student, base, tok, cfg = load_student(Path(args.out))
    student.eval()
    t_path = cfg.get("train", {}).get("teacher")
    teacher = load_teacher(t_path, tok, base) if t_path else None
    batches = iter_batches(tok, iter_texts(args), args.batch, args.seq, shuffle_buf=0)
    sums = {"base": 0.0, "student": 0.0, "teacher": 0.0}
    for _ in range(args.batches):
        x = next(batches)
        inp, tgt = x[:, :-1], x[:, 1:]
        h0 = student.prelude_forward(inp)
        logits = {
            "base": teacher_from_prelude(base, h0, cfg["prelude"]),
            "student": student.from_prelude(h0),
        }
        if teacher is not None:
            logits["teacher"] = teacher(inp)
        ces = {
            k: nn.losses.cross_entropy(v.astype(mx.float32), tgt).mean()
            for k, v in logits.items()
        }
        mx.eval(ces)
        for k, v in ces.items():
            sums[k] += v.item()
    n = args.batches
    # "base" is the model the student was converted from (the teacher too,
    # unless --teacher named a separate one)
    line = f"base ppl {math.exp(sums['base'] / n):.3f}"
    line += f" | student ppl {math.exp(sums['student'] / n):.3f}"
    if teacher is not None:
        line += f" | teacher ppl {math.exp(sums['teacher'] / n):.3f} ({t_path})"
    print(line)


def cmd_generate(args):
    from mlx_lm import generate

    student, _, tok, _ = load_student(Path(args.out))
    student.eval()
    prompt = args.prompt
    if args.chat and tok.chat_template:
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
    print(generate(student, tok, prompt, max_tokens=args.max_tokens, verbose=True))


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def data_args(sp):
        sp.add_argument("--data", help=".jsonl ({'text':...}) or .txt")
        sp.add_argument("--hf-dataset", help="e.g. HuggingFaceFW/fineweb-edu")
        sp.add_argument("--hf-subset", default=None, help="e.g. sample-10BT")
        sp.add_argument("--text-field", default="text")
        sp.add_argument("--batch", type=int, default=4)
        sp.add_argument("--seq", type=int, default=1024)

    t = sub.add_parser("train")
    t.add_argument("--model", default="Qwen/Qwen2.5-3B")
    t.add_argument(
        "--teacher",
        default=None,
        help="separate (e.g. larger) teacher with the same tokenizer; "
        "default: the base model itself",
    )
    t.add_argument("--out", default="looped-out")
    t.add_argument("--resume", action="store_true")
    t.add_argument("--prelude", type=int, default=6)
    t.add_argument("--core", type=int, default=8)
    t.add_argument("--coda", type=int, default=6)
    t.add_argument("--loops", type=int, default=3)
    t.add_argument(
        "--rank", type=int, default=64, help="per-loop LoRA rank (0 = fully shared)"
    )
    t.add_argument("--mode", choices=["teacher", "inkling"], default="teacher")
    t.add_argument("--freeze-coda", action="store_true")
    t.add_argument("--no-grad-checkpoint", dest="grad_checkpoint", action="store_false")
    t.add_argument("--steps", type=int, default=5000)
    t.add_argument("--accum", type=int, default=4)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--warmup", type=int, default=200)
    t.add_argument("--weight-decay", type=float, default=0.0)
    t.add_argument("--clip", type=float, default=1.0)
    t.add_argument("--temp", type=float, default=1.0)
    t.add_argument("--ce-weight", type=float, default=0.1)
    t.add_argument("--log-every", type=int, default=10)
    t.add_argument("--save-every", type=int, default=250)
    t.add_argument("--seed", type=int, default=0)
    data_args(t)
    t.set_defaults(fn=cmd_train)

    e = sub.add_parser("eval")
    e.add_argument("--out", default="looped-out")
    e.add_argument("--batches", type=int, default=20)
    data_args(e)
    e.set_defaults(fn=cmd_eval)

    g = sub.add_parser("generate")
    g.add_argument("--out", default="looped-out")
    g.add_argument("--prompt", required=True)
    g.add_argument("--max-tokens", type=int, default=200)
    g.add_argument("--chat", action="store_true")
    g.set_defaults(fn=cmd_generate)

    a = p.parse_args()
    if a.cmd in ("train", "eval") and not (a.data or a.hf_dataset):
        p.error("--data or --hf-dataset required")
    # fail now, not as a ZeroDivisionError hours into a run
    positive = ["batch", "seq", "steps", "accum", "log_every", "save_every", "batches"]
    for k in positive:
        if getattr(a, k, 1) < 1:
            p.error(f"--{k.replace('_', '-')} must be >= 1")
    a.fn(a)
    if getattr(a, "hf_dataset", None):
        # A streamed HF dataset leaves pyarrow IO threads waiting on fsspec's
        # daemon event-loop thread, which Python stops at shutdown, so exit()
        # hangs forever in libarrow's static destructors. Everything is saved and
        # printed by now: flush and skip the C++ teardown.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
