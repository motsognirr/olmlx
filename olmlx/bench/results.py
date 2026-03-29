"""Result storage, loading, and comparison for benchmark runs."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_BENCH_DIR = Path.home() / ".olmlx" / "bench" / "runs"


@dataclass
class PromptResult:
    prompt_name: str
    category: str
    output_text: str
    status_code: int
    error: str | None = None
    eval_count: int = 0
    eval_duration_ns: int = 0
    prompt_eval_count: int = 0
    prompt_eval_duration_ns: int = 0
    total_duration_ns: int = 0

    @property
    def tokens_per_second(self) -> float:
        if self.eval_duration_ns <= 0:
            return 0.0
        return self.eval_count / (self.eval_duration_ns / 1e9)

    @property
    def prompt_tokens_per_second(self) -> float:
        if self.prompt_eval_duration_ns <= 0:
            return 0.0
        return self.prompt_eval_count / (self.prompt_eval_duration_ns / 1e9)

    def to_dict(self) -> dict:
        return {
            "prompt_name": self.prompt_name,
            "category": self.category,
            "output_text": self.output_text,
            "status_code": self.status_code,
            "error": self.error,
            "eval_count": self.eval_count,
            "eval_duration_ns": self.eval_duration_ns,
            "prompt_eval_count": self.prompt_eval_count,
            "prompt_eval_duration_ns": self.prompt_eval_duration_ns,
            "total_duration_ns": self.total_duration_ns,
            "tokens_per_second": round(self.tokens_per_second, 2),
            "prompt_tokens_per_second": round(self.prompt_tokens_per_second, 2),
        }

    @classmethod
    def from_dict(cls, d: dict) -> PromptResult:
        return cls(
            prompt_name=d["prompt_name"],
            category=d["category"],
            output_text=d["output_text"],
            status_code=d["status_code"],
            error=d.get("error"),
            eval_count=d.get("eval_count", 0),
            eval_duration_ns=d.get("eval_duration_ns", 0),
            prompt_eval_count=d.get("prompt_eval_count", 0),
            prompt_eval_duration_ns=d.get("prompt_eval_duration_ns", 0),
            total_duration_ns=d.get("total_duration_ns", 0),
        )


@dataclass
class ScenarioResult:
    scenario_name: str
    scenario_description: str
    env_overrides: dict[str, str]
    prompt_results: list[PromptResult]
    skipped: bool = False
    skip_reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "scenario_name": self.scenario_name,
            "scenario_description": self.scenario_description,
            "env_overrides": self.env_overrides,
            "prompt_results": [r.to_dict() for r in self.prompt_results],
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ScenarioResult:
        return cls(
            scenario_name=d["scenario_name"],
            scenario_description=d["scenario_description"],
            env_overrides=d.get("env_overrides", {}),
            prompt_results=[
                PromptResult.from_dict(r) for r in d.get("prompt_results", [])
            ],
            skipped=d.get("skipped", False),
            skip_reason=d.get("skip_reason"),
        )


@dataclass
class RunResult:
    model: str
    timestamp: str
    git_sha: str | None
    scenarios: list[ScenarioResult]
    max_tokens_override: int | None = None

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "timestamp": self.timestamp,
            "git_sha": self.git_sha,
            "max_tokens_override": self.max_tokens_override,
            "scenarios": [s.to_dict() for s in self.scenarios],
        }

    @classmethod
    def from_dict(cls, d: dict) -> RunResult:
        return cls(
            model=d["model"],
            timestamp=d["timestamp"],
            git_sha=d.get("git_sha"),
            max_tokens_override=d.get("max_tokens_override"),
            scenarios=[ScenarioResult.from_dict(s) for s in d.get("scenarios", [])],
        )


def _git_sha() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def save_run(run: RunResult, bench_dir: Path = DEFAULT_BENCH_DIR) -> Path:
    """Save a run result to disk. Returns the run directory."""
    run_dir = bench_dir / run.timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.json").write_text(
        json.dumps(run.to_dict(), indent=2, ensure_ascii=False)
    )
    return run_dir


def load_run(run_path: Path) -> RunResult:
    """Load a run result from a directory or JSON file."""
    if run_path.is_dir():
        run_path = run_path / "results.json"
    data = json.loads(run_path.read_text())
    return RunResult.from_dict(data)


def list_runs(bench_dir: Path = DEFAULT_BENCH_DIR) -> list[dict]:
    """List all saved runs with summary info."""
    if not bench_dir.exists():
        return []
    runs = []
    for run_dir in sorted(bench_dir.iterdir()):
        results_path = run_dir / "results.json"
        if not results_path.exists():
            continue
        data = json.loads(results_path.read_text())
        scenario_names = [s["scenario_name"] for s in data.get("scenarios", [])]
        skipped = sum(1 for s in data.get("scenarios", []) if s.get("skipped"))
        runs.append(
            {
                "timestamp": data.get("timestamp", run_dir.name),
                "model": data.get("model", "unknown"),
                "git_sha": data.get("git_sha"),
                "scenarios": len(scenario_names),
                "skipped": skipped,
                "dir": str(run_dir),
            }
        )
    return runs


def compare_runs(run1: RunResult, run2: RunResult) -> str:
    """Format a comparison between two runs as a table."""
    lines: list[str] = []
    lines.append(f"Run 1: {run1.timestamp}  model={run1.model}  git={run1.git_sha}")
    lines.append(f"Run 2: {run2.timestamp}  model={run2.model}  git={run2.git_sha}")
    lines.append("")

    # Build lookup: scenario_name -> prompt_name -> PromptResult
    def _index(run: RunResult) -> dict[str, dict[str, PromptResult]]:
        idx: dict[str, dict[str, PromptResult]] = {}
        for sc in run.scenarios:
            if sc.skipped:
                continue
            idx[sc.scenario_name] = {pr.prompt_name: pr for pr in sc.prompt_results}
        return idx

    idx1 = _index(run1)
    idx2 = _index(run2)
    all_scenarios = sorted(set(idx1) | set(idx2))

    # Performance comparison table
    lines.append("## Performance (tokens/sec)")
    lines.append(
        f"{'Scenario':<20} {'Prompt':<15} {'Run1':>10} {'Run2':>10} {'Diff':>10}"
    )
    lines.append("-" * 67)

    for sc_name in all_scenarios:
        prompts1 = idx1.get(sc_name, {})
        prompts2 = idx2.get(sc_name, {})
        all_prompts = sorted(set(prompts1) | set(prompts2))
        for p_name in all_prompts:
            r1 = prompts1.get(p_name)
            r2 = prompts2.get(p_name)
            tps1 = f"{r1.tokens_per_second:.1f}" if r1 else "—"
            tps2 = f"{r2.tokens_per_second:.1f}" if r2 else "—"
            if r1 and r2 and r1.tokens_per_second > 0:
                pct = (
                    (r2.tokens_per_second - r1.tokens_per_second) / r1.tokens_per_second
                ) * 100
                diff = f"{pct:+.1f}%"
            else:
                diff = "—"
            lines.append(f"{sc_name:<20} {p_name:<15} {tps1:>10} {tps2:>10} {diff:>10}")

    # Output differences
    lines.append("")
    lines.append("## Output Differences")
    has_diffs = False
    for sc_name in all_scenarios:
        prompts1 = idx1.get(sc_name, {})
        prompts2 = idx2.get(sc_name, {})
        for p_name in sorted(set(prompts1) | set(prompts2)):
            r1 = prompts1.get(p_name)
            r2 = prompts2.get(p_name)
            if r1 and r2 and r1.output_text != r2.output_text:
                has_diffs = True
                lines.append(f"\n### {sc_name} / {p_name}")
                lines.append(f"Run 1: {r1.output_text[:200]}")
                lines.append(f"Run 2: {r2.output_text[:200]}")

    if not has_diffs:
        lines.append("No output differences found.")

    return "\n".join(lines)


def create_run_result(
    model: str,
    scenarios: list[ScenarioResult],
    max_tokens_override: int | None = None,
) -> RunResult:
    return RunResult(
        model=model,
        timestamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        git_sha=_git_sha(),
        scenarios=scenarios,
        max_tokens_override=max_tokens_override,
    )
