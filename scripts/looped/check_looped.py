"""Self-check for looped_qwen.py on tiny random Qwen2/Qwen3 models (CPU is fine).

Run: python scripts/looped/check_looped.py   (not a pytest module on purpose)
"""

import importlib

import looped_qwen as lq
import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten, tree_map


def make_base(kind):
    mod = importlib.import_module(f"mlx_lm.models.{kind}")
    cfg = {
        "model_type": kind,
        "hidden_size": 64,
        "num_hidden_layers": 12,
        "intermediate_size": 96,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6,
        "vocab_size": 101,
        "tie_word_embeddings": (kind == "qwen2"),
    }
    if kind == "qwen3":
        cfg.update(head_dim=16, max_position_embeddings=512, rope_theta=10000.0)
    m = mod.Model(mod.ModelArgs(**cfg))
    # perturb norms/biases so they differ per layer
    m.update(tree_map(lambda a: a + 0.05 * mx.random.normal(a.shape), m.parameters()))
    m.set_dtype(mx.float32)
    mx.eval(m.parameters())
    m.freeze()
    return m


mx.random.seed(0)
x = mx.random.randint(0, 101, (2, 17))
for kind in ["qwen2", "qwen3"]:
    base = make_base(kind)
    ref = base(x)
    # full rank -> exact teacher at init
    st = lq.LoopedLM(base, 2, 3, 1, 3, rank=96, mode="teacher")
    st.eval()
    err = mx.abs(st(x) - ref).max().item()
    print(
        kind,
        "full-rank teacher-init max err",
        err,
        "logit scale",
        mx.abs(ref).max().item(),
    )
    assert err < 2e-3, err
    # low rank -> close, inkling -> far
    lo = mx.abs(lq.LoopedLM(base, 2, 3, 1, 3, rank=8)(x) - ref).mean().item()
    ink = (
        mx.abs(lq.LoopedLM(base, 2, 3, 1, 3, rank=8, mode="inkling")(x) - ref)
        .mean()
        .item()
    )
    print(kind, "rank8 mean err", lo, "inkling mean err", ink)
    # KV cache path == full forward
    st = lq.LoopedLM(base, 2, 3, 1, 3, rank=8)
    st.eval()
    full = st(x[:1])
    cache = st.make_cache()
    a = st(x[:1, :10], cache=cache)
    outs = [a[:, -1]]
    for i in range(10, 17):
        outs.append(st(x[:1, i : i + 1], cache=cache)[:, -1])
    inc = mx.stack(outs[:-1], 1)
    err = mx.abs(inc - full[:, 9:16]).max().item()
    print(kind, "kv-cache vs full err", err)
    assert err < 1e-3

    # grads: checkpoint == no checkpoint, teacher stays out of tree
    def grads(ckpt, base=base):
        mx.random.seed(1)
        s = lq.LoopedLM(base, 2, 3, 1, 3, rank=8, grad_checkpoint=ckpt)
        s.freeze()
        for m in [s.core, *s.gates, *s.coda]:
            m.unfreeze()
        s.train()
        names = [k for k, _ in tree_flatten(s.parameters())]
        assert not any(k.startswith("base") for k in names)
        # nudge gates so their grads aren't trivially tiny
        h0 = mx.stop_gradient(s.prelude_forward(x[:, :-1]))
        tl = mx.stop_gradient(lq.teacher_from_prelude(base, h0, 2))
        lvg = nn.value_and_grad(s, lq.distill_loss)
        (l, _), g = lvg(s, h0, tl, x[:, 1:], 1.0, 0.1)
        return l, dict(tree_flatten(g))

    l1, g1 = grads(True)
    l2, g2 = grads(False)
    md = max(mx.abs(g1[k] - g2[k]).max().item() for k in g1)
    nz = sum(1 for k in g1 if mx.abs(g1[k]).max().item() > 0)
    print(
        kind,
        "loss",
        l1.item(),
        l2.item(),
        "ckpt grad diff",
        md,
        f"nonzero grads {nz}/{len(g1)}",
    )
    assert md < 1e-4
print("ALL OK")
