"""Reproduction harness for issue #380 — olmlx serve memory accumulation
across model swaps.

The real-world OOM took ~5-7 h and hard-rebooted a 64 GB machine. We do NOT
reproduce that literally. Instead we reproduce the *root cause* — per-swap
leaked state — by cycling small models through ensure_loaded -> generate ->
unload many times and watching the memory FLOOR (Metal + process RSS measured
when nothing is loaded). A leak shows as a monotonically climbing floor; a
clean fix shows a flat floor.

Safe by construction: only small models (<= a few GB) are used, so even a
real leak over the configured rounds stays far below 64 GB.

Usage:
    .venv/bin/python scripts/repro_oom_380.py [--rounds N] [--models a,b,c]
                                              [--gens-per-load N] [--no-cache]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys

# Default to a long-lived prompt cache exactly like the bench did, unless the
# caller overrides. Must be set before importing olmlx.config / settings.
os.environ.setdefault("OLMLX_PROMPT_CACHE", "true")


def _rss_bytes() -> int:
    """Current resident set size of THIS process, in bytes (via ps)."""
    try:
        out = subprocess.check_output(
            ["ps", "-o", "rss=", "-p", str(os.getpid())], text=True
        )
        return int(out.strip()) * 1024  # ps reports KiB on macOS
    except Exception:
        return 0


def _gib(n: int) -> float:
    return n / 1024**3


# Small, locally-available models. The gemma VLM is included on purpose:
# issue #380 hypothesis #3 blames mlx-vlm processor/tokenizer leaks on the
# gemma-4 family (a VLM even when used text-only).
DEFAULT_MODELS = [
    "mlx-community/Qwen3-0.6B-4bit",
    "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit",
    "mlx-community/Qwen3.5-0.8B-MLX-4bit",
    "mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
    "mlx-community/gemma-4-e2b-it-OptiQ-4bit",
]

# A non-trivial shared prefix so the PromptCacheStore actually stores a
# reusable entry per model (the leak hypothesis #1 centres on this store).
PREFIX = (
    "You are a careful assistant. Consider the following background carefully "
    "before answering. " + ("The quick brown fox jumps over the lazy dog. " * 40)
)


async def _run_one_generation(generate_chat, manager, model: str, n: int) -> None:
    messages = [
        {
            "role": "user",
            "content": f"{PREFIX}\n\nReply with the single word: ok ({n})",
        },
    ]
    result = await generate_chat(
        manager,
        model,
        messages,
        options={"temperature": 0.0},
        stream=False,
        max_tokens=8,
        cache_id=f"repro-{model}",  # stable id -> prompt cache reuse path
    )
    # result is a dict for stream=False; just drop it.
    del result


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--gens-per-load", type=int, default=2)
    parser.add_argument("--models", type=str, default="")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable prompt cache (the recommended #380 mitigation #1).",
    )
    args = parser.parse_args()

    if args.no_cache:
        os.environ["OLMLX_PROMPT_CACHE"] = "false"

    # Import after env is set so settings pick up OLMLX_PROMPT_CACHE.
    from olmlx.engine.inference import generate_chat
    from olmlx.engine.model_manager import ModelManager
    from olmlx.engine.registry import ModelRegistry
    from olmlx.models.store import ModelStore
    from olmlx.utils.memory import get_metal_memory

    models = (
        [m.strip() for m in args.models.split(",") if m.strip()]
        if args.models
        else DEFAULT_MODELS
    )

    registry = ModelRegistry()
    store = ModelStore(registry)
    manager = ModelManager(registry, store)

    cache_on = os.environ.get("OLMLX_PROMPT_CACHE", "true").lower() == "true"
    print(
        f"prompt_cache={'ON' if cache_on else 'OFF'}  rounds={args.rounds}  "
        f"gens/load={args.gens_per_load}  models={len(models)}"
    )
    print(
        f"{'round':>5} {'metal_floor_GiB':>16} {'rss_floor_GiB':>14} "
        f"{'d_metal_MiB':>12} {'d_rss_MiB':>10}"
    )

    floors: list[tuple[int, int]] = []
    base_metal = base_rss = None

    for r in range(1, args.rounds + 1):
        for model in models:
            try:
                for n in range(args.gens_per_load):
                    await _run_one_generation(generate_chat, manager, model, n)
            except Exception as exc:  # keep the sweep alive, like the bench
                print(f"  [warn] {model}: {type(exc).__name__}: {exc}", file=sys.stderr)
            finally:
                try:
                    await manager.unload(model)
                except Exception:
                    pass

        # Floor measurement: nothing should be loaded now. Drain Metal so we
        # measure reclaimed memory, not transient cache.
        await manager._flush_metal()
        metal = get_metal_memory()
        rss = _rss_bytes()
        if base_metal is None:
            base_metal, base_rss = metal, rss
        d_metal = (metal - base_metal) / 1024**2
        d_rss = (rss - base_rss) / 1024**2
        floors.append((metal, rss))
        print(
            f"{r:>5} {_gib(metal):>16.3f} {_gib(rss):>14.3f} "
            f"{d_metal:>12.1f} {d_rss:>10.1f}"
        )

    # Verdict: the process working set warms up over the first full round
    # (lazy library imports, allocator arenas), so we baseline off the SECOND
    # floor (steady state) and look for sustained per-round growth after that.
    # Baselining off round 1 would amortise a one-time warmup jump into a fake
    # per-round leak rate.
    if len(floors) >= 3:
        base_metal_f, base_rss_f = floors[1]
        last_metal, last_rss = floors[-1]
        n_steady = len(floors) - 2  # rounds compared after the warmup round
        grow_metal = (last_metal - base_metal_f) / 1024**2
        grow_rss = (last_rss - base_rss_f) / 1024**2
        per_round_rss = grow_rss / n_steady
        # Monotonic climb is the real leak signature; oscillation is allocator
        # noise. Count strictly-increasing steps among steady-state floors.
        steady = [f[1] for f in floors[1:]]
        ups = sum(1 for a, b in zip(steady, steady[1:]) if b > a)
        print("\n=== verdict (steady state, baselined on round 2) ===")
        print(
            f"warmup jump round1->2 RSS: "
            f"{(floors[1][1] - floors[0][1]) / 1024**2:+.1f} MiB (one-time)"
        )
        print(
            f"metal floor growth: {grow_metal:+.1f} MiB over {n_steady} steady rounds"
        )
        print(
            f"rss   floor growth: {grow_rss:+.1f} MiB over {n_steady} steady rounds "
            f"({per_round_rss:+.1f} MiB/round); monotonic-up steps: {ups}/{n_steady}"
        )
        if per_round_rss > 50 and ups >= n_steady - 1:
            print("LEAK SUSPECTED: floor RSS climbs monotonically across swaps.")
        else:
            print("OK: floor memory plateaus across swaps (no per-swap leak detected).")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
