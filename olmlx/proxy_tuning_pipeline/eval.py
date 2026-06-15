"""Stage-3 eval orchestration: aggregate, ship gate, run_eval."""

from __future__ import annotations

import json
from typing import Any, Callable

from olmlx.proxy_tuning_pipeline.eval_driver import generate_one
from olmlx.proxy_tuning_pipeline.eval_schema import (
    AlphaSummary,
    EvalPrompt,
    EvalScore,
    ShipDecision,
)


def aggregate(scores: list[EvalScore]) -> list[AlphaSummary]:
    by_alpha: dict[float, list[EvalScore]] = {}
    for s in scores:
        by_alpha.setdefault(s.alpha, []).append(s)
    out: list[AlphaSummary] = []
    for alpha in sorted(by_alpha):
        group = by_alpha[alpha]
        n = len(group)
        out.append(
            AlphaSummary(
                alpha=alpha,
                n=n,
                mean_convention=sum(s.convention for s in group) / n,
                mean_coherence=sum(s.coherence for s in group) / n,
            )
        )
    return out


def ship_decision(
    summaries: list[AlphaSummary],
    *,
    conv_margin: float = 0.5,
    coh_drop: float = 0.2,
) -> ShipDecision:
    """Requires an alpha=0.0 baseline and at least one alpha>0 entry (both guaranteed by run_eval)."""
    by_alpha = {s.alpha: s for s in summaries}
    if 0.0 not in by_alpha:
        raise ValueError("ship_decision requires an alpha=0.0 baseline summary")
    steered = [s for s in summaries if s.alpha > 0.0]
    if not steered:
        raise ValueError("ship_decision requires at least one alpha>0 summary")
    base = by_alpha[0.0]
    best = max(steered, key=lambda s: s.mean_convention)

    conv_ok = best.mean_convention >= base.mean_convention + conv_margin
    coh_ok = best.mean_coherence >= base.mean_coherence - coh_drop
    ship = conv_ok and coh_ok
    if ship:
        reason = (
            f"best α={best.alpha}: convention {best.mean_convention:.2f} ≥ base "
            f"{base.mean_convention:.2f}+{conv_margin}, coherence held."
        )
    elif not conv_ok:
        reason = (
            f"insufficient convention lift at best α={best.alpha}: "
            f"{best.mean_convention:.2f} vs base {base.mean_convention:.2f} "
            f"(need +{conv_margin})."
        )
    else:
        reason = (
            f"coherence dropped at best α={best.alpha}: "
            f"{best.mean_coherence:.2f} vs base {base.mean_coherence:.2f} "
            f"(max drop {coh_drop})."
        )
    return ShipDecision(
        ship=ship,
        best_alpha=best.alpha,
        base_convention=base.mean_convention,
        best_convention=best.mean_convention,
        base_coherence=base.mean_coherence,
        best_coherence=best.mean_coherence,
        conv_margin=conv_margin,
        coh_drop=coh_drop,
        reason=reason,
        per_alpha=summaries,
    )


def _default_loader(base_dir: str, expert_dir: str, antiexpert_dir: str):
    from mlx_lm import load

    base, tok = load(base_dir)
    expert, _ = load(expert_dir)
    anti, _ = load(antiexpert_dir)
    return base, expert, anti, tok


def _default_decoder_factory(base, expert, anti, alpha):
    from olmlx.engine.proxy_tuning import ProxyTuningDecoder

    return ProxyTuningDecoder(base, expert, anti, alpha=alpha)


def _default_preflight(antiexpert_dir: str, expert_dir: str) -> None:
    from olmlx.proxy_tuning_pipeline.verify import assert_serveable_pair

    # Steered base config vocab_size (Qwen3 dense = 151936).
    assert_serveable_pair(antiexpert_dir, expert_dir, 151936)


def run_eval(
    *,
    base_dir: str,
    expert_dir: str,
    antiexpert_dir: str,
    prompts: list[EvalPrompt],
    alphas: list[float],
    judge: Any,
    out_path: str,
    max_tokens: int = 256,
    loader: Callable[..., Any] = _default_loader,
    decoder_factory: Callable[..., Any] = _default_decoder_factory,
    preflight: Callable[..., None] = _default_preflight,
) -> ShipDecision:
    """Load once, sweep α, generate + judge each prompt, aggregate, decide.

    The base+expert+anti-expert load once; per α a fresh decoder is built over
    the same model objects (cheap — caches are per-decoder). Calls run on the
    default stream (decoder contract).
    """
    preflight(antiexpert_dir, expert_dir)
    base, expert, anti, tokenizer = loader(base_dir, expert_dir, antiexpert_dir)

    scores: list[EvalScore] = []
    for alpha in alphas:
        decoder = decoder_factory(base, expert, anti, alpha)
        for p in prompts:
            completion = generate_one(
                decoder, tokenizer, p.messages, max_tokens=max_tokens
            )
            conv, coh, rationale = judge.score(
                prompt=p.messages[-1]["content"], completion=completion
            )
            scores.append(
                EvalScore(p.id, p.category, alpha, conv, coh, rationale, completion)
            )

    summaries = aggregate(scores)
    decision = ship_decision(summaries)

    with open(out_path, "w") as f:
        json.dump(
            {
                "ship": decision.ship,
                "best_alpha": decision.best_alpha,
                "reason": decision.reason,
                "per_alpha": [vars(s) for s in summaries],
                "scores": [vars(s) for s in scores],
            },
            f,
            indent=2,
        )
    return decision


def main(argv: list[str] | None = None) -> None:
    import argparse

    from olmlx.proxy_tuning_pipeline.eval_judge import ProxyEvalJudge
    from olmlx.proxy_tuning_pipeline.eval_schema import load_eval_prompts
    from olmlx.proxy_tuning_pipeline.expand import OpenAIGenerator

    ap = argparse.ArgumentParser(
        description="Stage-3 proxy-tuning α-sweep eval + ship gate."
    )
    ap.add_argument("--base", required=True, help="dense Qwen3 base model dir")
    ap.add_argument("--expert", required=True, help="M+ (4-bit) dir")
    ap.add_argument("--antiexpert", required=True, help="M- (4-bit) dir")
    ap.add_argument("--prompts", required=True, help="eval_prompts.jsonl path")
    ap.add_argument("--alphas", default="0,0.5,1.0,1.5")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--out", default="eval_results.json")
    args = ap.parse_args(argv)

    prompts = load_eval_prompts(args.prompts)
    judge = ProxyEvalJudge(OpenAIGenerator())
    decision = run_eval(
        base_dir=args.base,
        expert_dir=args.expert,
        antiexpert_dir=args.antiexpert,
        prompts=prompts,
        alphas=[float(a) for a in args.alphas.split(",")],
        judge=judge,
        out_path=args.out,
        max_tokens=args.max_tokens,
    )
    print(f"\n=== {'SHIP' if decision.ship else 'NO-SHIP'} ===")
    print(decision.reason)
    for s in decision.per_alpha:
        print(
            f"  α={s.alpha:>4}: convention {s.mean_convention:.2f}  "
            f"coherence {s.mean_coherence:.2f}  (n={s.n})"
        )


if __name__ == "__main__":
    main()
