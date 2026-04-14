"""Benchmark a running olmlx server across API surfaces.

Targets an already-running server (default http://localhost:11434) and sweeps over
API surface × streaming mode × model × prompt, reporting TTFT and decode throughput.

Usage:
    python -m olmlx.bench.api_bench --models qwen3:8b --apis openai-chat,anthropic-messages
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Protocol

import httpx

from olmlx.bench.prompts import PROMPTS, BenchPrompt


@dataclass
class StreamEvent:
    text: str = ""
    done: bool = False
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    prompt_eval_duration_ns: int | None = None
    eval_duration_ns: int | None = None


@dataclass
class ApiMetrics:
    text: str = ""
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    prompt_eval_duration_ns: int | None = None
    eval_duration_ns: int | None = None


class ApiAdapter(Protocol):
    name: str

    def build_request(
        self, prompt: BenchPrompt, model: str, max_tokens: int, stream: bool
    ) -> tuple[str, dict, dict]: ...

    def parse_nonstream(self, resp_json: dict) -> ApiMetrics: ...

    def iter_stream(self, lines: Iterable[str]) -> Iterator[StreamEvent]: ...


_JSON_HEADERS = {"Content-Type": "application/json"}


class OllamaChatAdapter:
    name = "ollama-chat"

    def build_request(self, prompt, model, max_tokens, stream):
        body = {
            "model": model,
            "stream": stream,
            "messages": prompt.messages,
            "options": {
                "seed": 42,
                "temperature": 0.0,
                "num_predict": max_tokens,
            },
        }
        return "/api/chat", body, dict(_JSON_HEADERS)

    def parse_nonstream(self, d):
        return ApiMetrics(
            text=(d.get("message") or {}).get("content", ""),
            prompt_tokens=d.get("prompt_eval_count"),
            output_tokens=d.get("eval_count"),
            prompt_eval_duration_ns=d.get("prompt_eval_duration"),
            eval_duration_ns=d.get("eval_duration"),
        )

    def iter_stream(self, lines):
        for line in lines:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("done"):
                yield StreamEvent(
                    done=True,
                    prompt_tokens=d.get("prompt_eval_count"),
                    output_tokens=d.get("eval_count"),
                    prompt_eval_duration_ns=d.get("prompt_eval_duration"),
                    eval_duration_ns=d.get("eval_duration"),
                )
            else:
                yield StreamEvent(text=(d.get("message") or {}).get("content", ""))


class OllamaGenerateAdapter:
    name = "ollama-generate"

    @staticmethod
    def _flatten(messages: list[dict]) -> str:
        return "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages
        )

    def build_request(self, prompt, model, max_tokens, stream):
        body = {
            "model": model,
            "stream": stream,
            "prompt": self._flatten(prompt.messages),
            "options": {
                "seed": 42,
                "temperature": 0.0,
                "num_predict": max_tokens,
            },
        }
        return "/api/generate", body, dict(_JSON_HEADERS)

    def parse_nonstream(self, d):
        return ApiMetrics(
            text=d.get("response", ""),
            prompt_tokens=d.get("prompt_eval_count"),
            output_tokens=d.get("eval_count"),
            prompt_eval_duration_ns=d.get("prompt_eval_duration"),
            eval_duration_ns=d.get("eval_duration"),
        )

    def iter_stream(self, lines):
        for line in lines:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("done"):
                yield StreamEvent(
                    done=True,
                    prompt_tokens=d.get("prompt_eval_count"),
                    output_tokens=d.get("eval_count"),
                    prompt_eval_duration_ns=d.get("prompt_eval_duration"),
                    eval_duration_ns=d.get("eval_duration"),
                )
            else:
                yield StreamEvent(text=d.get("response", ""))


class OpenAIChatAdapter:
    name = "openai-chat"

    def build_request(self, prompt, model, max_tokens, stream):
        body = {
            "model": model,
            "messages": prompt.messages,
            "stream": stream,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "seed": 42,
        }
        if stream:
            body["stream_options"] = {"include_usage": True}
        return "/v1/chat/completions", body, dict(_JSON_HEADERS)

    def parse_nonstream(self, d):
        usage = d.get("usage") or {}
        choices = d.get("choices") or [{}]
        msg = (choices[0] or {}).get("message") or {}
        return ApiMetrics(
            text=msg.get("content") or "",
            prompt_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
        )

    def iter_stream(self, lines):
        usage: dict = {}
        for raw in lines:
            raw = raw.strip()
            if not raw or not raw.startswith("data:"):
                continue
            payload = raw[len("data:") :].strip()
            if payload == "[DONE]":
                yield StreamEvent(
                    done=True,
                    prompt_tokens=usage.get("prompt_tokens"),
                    output_tokens=usage.get("completion_tokens"),
                )
                return
            try:
                d = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if d.get("usage"):
                usage = d["usage"]
            choices = d.get("choices") or []
            if not choices:
                continue
            delta = (choices[0] or {}).get("delta") or {}
            text = delta.get("content") or ""
            if text:
                yield StreamEvent(text=text)


class AnthropicMessagesAdapter:
    name = "anthropic-messages"

    def build_request(self, prompt, model, max_tokens, stream):
        body = {
            "model": model,
            "messages": prompt.messages,
            "max_tokens": max_tokens,
            "stream": stream,
            "temperature": 0.0,
        }
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        return "/v1/messages", body, headers

    def parse_nonstream(self, d):
        usage = d.get("usage") or {}
        text_parts = [
            (block.get("text") or "")
            for block in (d.get("content") or [])
            if block.get("type") == "text"
        ]
        return ApiMetrics(
            text="".join(text_parts),
            prompt_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
        )

    def iter_stream(self, lines):
        current_event: str | None = None
        input_tokens: int | None = None
        for raw in lines:
            raw = raw.rstrip("\n").rstrip("\r")
            if raw.startswith("event:"):
                current_event = raw.split(":", 1)[1].strip()
            elif raw.startswith("data:"):
                payload = raw[len("data:") :].strip()
                try:
                    d = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                et = current_event or d.get("type")
                if et == "message_start":
                    usage = ((d.get("message") or {}).get("usage")) or {}
                    input_tokens = usage.get("input_tokens")
                elif et == "content_block_delta":
                    delta = d.get("delta") or {}
                    if delta.get("type") == "text_delta":
                        text = delta.get("text") or ""
                        if text:
                            yield StreamEvent(text=text)
                elif et == "message_delta":
                    usage = d.get("usage") or {}
                    yield StreamEvent(
                        done=True,
                        prompt_tokens=input_tokens,
                        output_tokens=usage.get("output_tokens"),
                    )
            elif raw == "":
                current_event = None


ADAPTERS: dict[str, type] = {
    "ollama-chat": OllamaChatAdapter,
    "ollama-generate": OllamaGenerateAdapter,
    "openai-chat": OpenAIChatAdapter,
    "anthropic-messages": AnthropicMessagesAdapter,
}


@dataclass
class RunRecord:
    api: str
    mode: str
    model: str
    prompt: str
    run_index: int
    ttft_ms: float | None
    total_ms: float
    prompt_tokens: int | None
    output_tokens: int | None
    tokens_per_sec: float | None
    tokens_estimated: bool
    tps_source: str | None
    error: str | None


def _estimate_tokens(text: str) -> int:
    # Whitespace split ≈ words; multiply by 1.3 for sub-word tokens.
    words = len(text.split())
    return int(words * 1.3) if words else max(len(text) // 4, 0)


def run_single(
    client: httpx.Client,
    base_url: str,
    adapter: ApiAdapter,
    prompt: BenchPrompt,
    model: str,
    max_tokens: int,
    stream: bool,
    timeout: float,
    run_index: int,
) -> RunRecord:
    url_path, body, headers = adapter.build_request(prompt, model, max_tokens, stream)
    url = base_url.rstrip("/") + url_path
    t_request = time.perf_counter_ns()
    ttft_ns: int | None = None
    t_end: int | None = None
    error: str | None = None
    metrics = ApiMetrics()
    try:
        if stream:
            with client.stream(
                "POST", url, json=body, headers=headers, timeout=timeout
            ) as resp:
                resp.raise_for_status()
                agg_text: list[str] = []
                got_done = False
                for ev in adapter.iter_stream(resp.iter_lines()):
                    if ev.text and ttft_ns is None:
                        ttft_ns = time.perf_counter_ns() - t_request
                    if ev.text:
                        agg_text.append(ev.text)
                    if ev.done:
                        got_done = True
                        if ev.prompt_tokens is not None:
                            metrics.prompt_tokens = ev.prompt_tokens
                        if ev.output_tokens is not None:
                            metrics.output_tokens = ev.output_tokens
                        if ev.prompt_eval_duration_ns is not None:
                            metrics.prompt_eval_duration_ns = ev.prompt_eval_duration_ns
                        if ev.eval_duration_ns is not None:
                            metrics.eval_duration_ns = ev.eval_duration_ns
                metrics.text = "".join(agg_text)
            t_end = time.perf_counter_ns()
            if not got_done:
                error = "stream ended without done event (truncated?)"
        else:
            resp = client.post(url, json=body, headers=headers, timeout=timeout)
            resp.raise_for_status()
            t_end = time.perf_counter_ns()
            metrics = adapter.parse_nonstream(resp.json())
    except Exception as exc:
        if t_end is None:
            t_end = time.perf_counter_ns()
        error = f"{type(exc).__name__}: {exc}"

    total_ms = (t_end - t_request) / 1e6
    ttft_ms = ttft_ns / 1e6 if ttft_ns is not None else None

    estimated = False
    output_tokens = metrics.output_tokens
    if output_tokens is None and metrics.text:
        output_tokens = _estimate_tokens(metrics.text)
        estimated = True

    tps: float | None = None
    tps_source: str | None = None
    if output_tokens is not None and output_tokens > 0:
        if metrics.eval_duration_ns:
            tps = output_tokens / (metrics.eval_duration_ns / 1e9)
            tps_source = "eval_duration"
        else:
            if stream:
                decode_ms = total_ms - (ttft_ms or 0.0)
                tps_source = "decode_estimate"
            elif metrics.prompt_eval_duration_ns:
                decode_ms = ((total_ms * 1e6) - metrics.prompt_eval_duration_ns) / 1e6
                tps_source = "decode_estimate"
            else:
                decode_ms = total_ms
                tps_source = "rtt_fallback"
            if decode_ms > 0:
                tps = output_tokens / (decode_ms / 1000.0)
            else:
                tps_source = None

    return RunRecord(
        api=adapter.name,
        mode="stream" if stream else "nostream",
        model=model,
        prompt=prompt.name,
        run_index=run_index,
        ttft_ms=ttft_ms,
        total_ms=total_ms,
        prompt_tokens=metrics.prompt_tokens,
        output_tokens=output_tokens,
        tokens_per_sec=tps,
        tokens_estimated=estimated,
        tps_source=tps_source,
        error=error,
    )


def _pick_prompts(csv: str | None) -> list[BenchPrompt]:
    by_name = {p.name: p for p in PROMPTS}
    if not csv:
        return list(PROMPTS)
    out = []
    for n in csv.split(","):
        n = n.strip()
        if not n:
            continue
        if n not in by_name:
            raise SystemExit(f"unknown prompt: {n} (choices: {sorted(by_name)})")
        out.append(by_name[n])
    return out


def _pick_apis(csv: str | None) -> list[ApiAdapter]:
    if not csv:
        return [cls() for cls in ADAPTERS.values()]
    out = []
    for n in csv.split(","):
        n = n.strip()
        if not n:
            continue
        if n not in ADAPTERS:
            raise SystemExit(f"unknown api: {n} (choices: {sorted(ADAPTERS)})")
        out.append(ADAPTERS[n]())
    return out


def _pick_modes(csv: str) -> list[str]:
    out = []
    for m in csv.split(","):
        m = m.strip()
        if not m:
            continue
        if m not in ("stream", "nostream"):
            raise SystemExit(f"unknown mode: {m} (choices: stream, nostream)")
        out.append(m)
    if not out:
        raise SystemExit("no modes specified")
    return out


def summarize(records: list[RunRecord]) -> list[dict]:
    groups: dict[tuple, list[RunRecord]] = defaultdict(list)
    for r in records:
        if r.error:
            continue
        groups[(r.api, r.mode, r.model, r.prompt)].append(r)
    summary = []
    for (api, mode, model, prompt), recs in groups.items():
        ttfts = [r.ttft_ms for r in recs if r.ttft_ms is not None]
        tpses = [r.tokens_per_sec for r in recs if r.tokens_per_sec is not None]
        totals = [r.total_ms for r in recs]
        summary.append(
            {
                "api": api,
                "mode": mode,
                "model": model,
                "prompt": prompt,
                "runs": len(recs),
                "ttft_p50_ms": statistics.median(ttfts) if ttfts else None,
                "tps_p50": statistics.median(tpses) if tpses else None,
                "total_p50_ms": statistics.median(totals) if totals else None,
            }
        )
    return summary


def _fmt(v: float | None, d: int = 2) -> str:
    return "-" if v is None else f"{v:.{d}f}"


def _print_table(summary: list[dict], errors: list[RunRecord]) -> None:
    try:
        from rich.console import Console
        from rich.table import Table
    except ImportError:
        cols = (
            "api",
            "mode",
            "model",
            "prompt",
            "runs",
            "ttft_p50_ms",
            "tps_p50",
            "total_p50_ms",
        )
        print("\t".join(cols))
        for row in summary:
            print(
                "\t".join(
                    _fmt(row[c]) if isinstance(row[c], float) else str(row[c])
                    for c in cols
                )
            )
        if errors:
            print(f"\n{len(errors)} errored runs", file=sys.stderr)
        return
    console = Console()
    table = Table(title="olmlx API benchmark")
    for col in (
        "api",
        "mode",
        "model",
        "prompt",
        "runs",
        "TTFT p50 (ms)",
        "tok/s p50",
        "total p50 (ms)",
    ):
        table.add_column(col)
    for row in summary:
        table.add_row(
            row["api"],
            row["mode"],
            row["model"],
            row["prompt"],
            str(row["runs"]),
            _fmt(row["ttft_p50_ms"]),
            _fmt(row["tps_p50"]),
            _fmt(row["total_p50_ms"]),
        )
    console.print(table)
    if errors:
        console.print(
            f"[yellow]{len(errors)} errored runs — see JSON output for details[/yellow]"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m olmlx.bench.api_bench",
        description="Benchmark a running olmlx server across API surfaces.",
    )
    p.add_argument(
        "--url", default="http://localhost:11434", help="Base URL of running server"
    )
    p.add_argument("--models", required=True, help="Comma-separated model names")
    p.add_argument(
        "--apis",
        default=None,
        help=f"Comma-separated subset of: {','.join(ADAPTERS)}",
    )
    p.add_argument(
        "--modes", default="stream,nostream", help="Comma-separated: stream,nostream"
    )
    p.add_argument(
        "--prompts",
        default=None,
        help="Comma-separated prompt names from olmlx.bench.prompts",
    )
    p.add_argument("--runs", type=int, default=1, help="Repetitions per cell")
    p.add_argument(
        "--max-tokens", type=int, default=None, help="Override per-prompt max_tokens"
    )
    p.add_argument(
        "--warmup", type=int, default=1, help="Warmup iterations per model (discarded)"
    )
    p.add_argument("--output", default=None, help="Explicit JSON output path")
    p.add_argument("--no-json", action="store_true", help="Skip writing JSON")
    p.add_argument(
        "--timeout", type=float, default=300.0, help="Per-request timeout seconds"
    )
    return p


def _warmup(
    client: httpx.Client,
    base_url: str,
    model: str,
    timeout: float,
    adapter: ApiAdapter,
) -> None:
    warm = BenchPrompt(
        name="_warmup",
        category="_warmup",
        messages=[{"role": "user", "content": "Hi."}],
        max_tokens=4,
    )
    path, body, headers = adapter.build_request(warm, model, 4, stream=False)
    try:
        client.post(
            base_url.rstrip("/") + path, json=body, headers=headers, timeout=timeout
        )
    except Exception as exc:  # noqa: BLE001 - warmup failures shouldn't abort the run
        print(f"[warmup] ignored error: {exc}", file=sys.stderr)


def _get_olmlx_version() -> str:
    try:
        from importlib.metadata import version

        return version("olmlx")
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    adapters = _pick_apis(args.apis)
    prompts = _pick_prompts(args.prompts)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        raise SystemExit("--models is empty")
    modes = _pick_modes(args.modes)

    started_at = datetime.now(timezone.utc).isoformat()
    records: list[RunRecord] = []

    with httpx.Client() as client:
        for model in models:
            for warmup_adapter in adapters:
                for _ in range(max(0, args.warmup)):
                    print(
                        f"[warmup] {model} via {warmup_adapter.name}", file=sys.stderr
                    )
                    _warmup(client, args.url, model, args.timeout, warmup_adapter)
            for adapter in adapters:
                for mode in modes:
                    stream = mode == "stream"
                    for prompt in prompts:
                        for run_index in range(args.runs):
                            max_tokens = args.max_tokens or prompt.max_tokens
                            print(
                                f"[run] {adapter.name} {mode} {model} {prompt.name} #{run_index}",
                                file=sys.stderr,
                            )
                            rec = run_single(
                                client,
                                args.url,
                                adapter,
                                prompt,
                                model,
                                max_tokens,
                                stream,
                                args.timeout,
                                run_index,
                            )
                            records.append(rec)

    summary = summarize(records)
    errors = [r for r in records if r.error]
    _print_table(summary, errors)

    if not args.no_json:
        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            out_dir = Path.home() / ".olmlx" / "bench"
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            out_path = out_dir / f"api_bench_{ts}.json"
        payload = {
            "olmlx_version": _get_olmlx_version(),
            "server_url": args.url,
            "started_at": started_at,
            "records": [asdict(r) for r in records],
            "summary": summary,
        }
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"[saved] {out_path}", file=sys.stderr)

    return 1 if records and len(errors) == len(records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
