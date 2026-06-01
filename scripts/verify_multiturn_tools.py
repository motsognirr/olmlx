#!/usr/bin/env python3
"""Verify multi-turn tool-use works for every model olmlx serves.

For each registered model this drives a real two-round tool-calling
conversation against the OpenAI-compatible endpoint and reports PASS/FAIL with
a reason. The scenario uses two tools where the second depends on the first's
output, so a well-behaved model must make two genuine sequential tool rounds:

    user: "What's the weather where I am, in celsius? Use the tools."
      round 1 -> get_user_location()            -> "Paris"
      round 2 -> get_weather("Paris", celsius)  -> canned weather
      final   -> a coherent answer using the result

A model passes only if it does both rounds correctly and ends with a
non-empty, non-repetitive answer. Distinct failure reasons are captured so the
known trouble modes are legible (HTTP 503 model-load/memory, no tool call,
tool-call loops, and the verbatim-repetition signature of prompt-cache /
speculative multi-turn drift).

Usage:
    python scripts/verify_multiturn_tools.py
    python scripts/verify_multiturn_tools.py --model mlx-community/Qwen3-8B-4bit:latest
    OLMLX_URL=http://localhost:11434/v1 python scripts/verify_multiturn_tools.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
from typing import Any

DEFAULT_BASE_URL = os.environ.get("OLMLX_URL", "http://localhost:11434/v1")
LOCATION = "Paris"
USER_PROMPT = (
    "What's the weather where I am right now, in celsius? Use the tools to find out."
)

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_user_location",
            "description": "Get the city the user is currently in. Takes no arguments.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit",
                    },
                },
                "required": ["city", "unit"],
            },
        },
    },
]


def dispatch_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Return a canned result for a tool call."""
    if name == "get_user_location":
        return {"city": LOCATION}
    if name == "get_weather":
        city = args.get("city", "")
        unit = args.get("unit", "celsius")
        temp = 18 if unit == "celsius" else 64
        return {"city": city, "temperature": temp, "unit": unit, "conditions": "sunny"}
    return {"error": f"unknown tool {name!r}"}


def looks_repetitive(text: str) -> bool:
    """Heuristic detector for runaway / verbatim repetition in a final answer."""
    if not text:
        return False
    # Any single non-trivial line repeated many times.
    lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) > 8]
    if lines:
        most = max(lines.count(ln) for ln in set(lines))
        if most >= 5:
            return True
    # Low diversity over a long output (shingled).
    if len(text) > 1500:
        shingles = [text[i : i + 50] for i in range(0, len(text) - 50, 25)]
        if shingles and len(set(shingles)) / len(shingles) < 0.3:
            return True
    return False


class HTTPError(Exception):
    def __init__(self, code: int, body: str) -> None:
        super().__init__(f"HTTP {code}")
        self.code = code
        self.body = body


def chat(
    base_url: str, model: str, messages: list[dict], temperature: float, timeout: float
) -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "tools": TOOLS,
        "temperature": temperature,
        "max_tokens": 1024,
    }
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:300]
        raise HTTPError(e.code, body) from None


def assistant_history_message(msg: dict) -> dict:
    """Rebuild a clean assistant message (content + tool_calls) for chat history."""
    out: dict[str, Any] = {"role": "assistant", "content": msg.get("content")}
    if msg.get("tool_calls"):
        out["tool_calls"] = [
            {
                "id": tc["id"],
                "type": tc.get("type", "function"),
                "function": {
                    "name": tc["function"]["name"],
                    "arguments": tc["function"]["arguments"],
                },
            }
            for tc in msg["tool_calls"]
        ]
    return out


def run_conversation(
    base_url: str, model: str, temperature: float, timeout: float, max_rounds: int = 6
) -> dict[str, Any]:
    """Drive the tool loop. Returns a result dict with verdict and reason."""
    messages: list[dict] = [{"role": "user", "content": USER_PROMPT}]
    rounds: list[list[dict[str, Any]]] = []  # per round: list of {name, args}
    signatures: dict[str, int] = {}
    final_text: str | None = None
    malformed = False

    for _ in range(max_rounds):
        try:
            resp = chat(base_url, model, messages, temperature, timeout)
        except HTTPError as e:
            return _verdict(
                model, rounds, None, reason=f"HTTP {e.code}: {e.body.strip()}"
            )
        except (urllib.error.URLError, TimeoutError) as e:
            return _verdict(model, rounds, None, reason=f"request failed: {e}")

        try:
            msg = resp["choices"][0]["message"]
        except (KeyError, IndexError):
            return _verdict(
                model, rounds, None, reason=f"malformed response: {str(resp)[:200]}"
            )

        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            final_text = (msg.get("content") or "").strip()
            break

        round_calls: list[dict[str, Any]] = []
        messages.append(assistant_history_message(msg))
        for tc in tool_calls:
            name = tc["function"]["name"]
            raw_args = tc["function"].get("arguments") or "{}"
            try:
                args = (
                    json.loads(raw_args)
                    if isinstance(raw_args, str)
                    else dict(raw_args)
                )
            except (json.JSONDecodeError, TypeError):
                args, malformed = {}, True
            round_calls.append({"name": name, "args": args})
            sig = f"{name}:{json.dumps(args, sort_keys=True)}"
            signatures[sig] = signatures.get(sig, 0) + 1
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": json.dumps(dispatch_tool(name, args)),
                }
            )
        rounds.append(round_calls)

        if any(count >= 3 for count in signatures.values()):
            return _verdict(
                model, rounds, None, reason="tool-call loop (same call repeated)"
            )
    else:
        return _verdict(
            model, rounds, None, reason=f"no convergence within {max_rounds} rounds"
        )

    return _verdict(model, rounds, final_text, malformed=malformed)


def _flatten(rounds: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [call for rnd in rounds for call in rnd]


def _verdict(
    model: str,
    rounds: list[list[dict[str, Any]]],
    final_text: str | None,
    reason: str | None = None,
    malformed: bool = False,
) -> dict[str, Any]:
    calls = _flatten(rounds)
    tools_called = [c["name"] for c in calls]
    base = {
        "model": model,
        "rounds": len(rounds),
        "tools_called": tools_called,
        "final": (final_text or "")[:200],
    }

    # Hard failures detected mid-loop are passed in via `reason`.
    if reason is not None:
        return {**base, "passed": False, "reason": reason}
    if malformed:
        return {
            **base,
            "passed": False,
            "reason": "malformed tool arguments (invalid JSON)",
        }

    located = any(c["name"] == "get_user_location" for c in calls)
    weather_paris = any(
        c["name"] == "get_weather"
        and LOCATION.lower() in str(c["args"].get("city", "")).lower()
        for c in calls
    )

    if not tools_called:
        return {**base, "passed": False, "reason": "no tool call (answered directly)"}
    if not located:
        return {**base, "passed": False, "reason": "never called get_user_location"}
    if len(rounds) < 2:
        return {
            **base,
            "passed": False,
            "reason": "only one tool round (no second call)",
        }
    if not weather_paris:
        return {
            **base,
            "passed": False,
            "reason": "did not call get_weather with the located city",
        }
    if not final_text:
        return {**base, "passed": False, "reason": "empty final answer"}
    if looks_repetitive(final_text):
        return {
            **base,
            "passed": False,
            "reason": "verbatim repetition in final answer",
        }
    if (
        LOCATION.lower() not in final_text.lower()
        and "18" not in final_text
        and "sunny" not in final_text.lower()
    ):
        return {
            **base,
            "passed": False,
            "reason": "final answer does not reference the tool result",
        }

    return {**base, "passed": True, "reason": "ok"}


def list_models(base_url: str, timeout: float = 30) -> list[str]:
    with urllib.request.urlopen(f"{base_url}/models", timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    return [m["id"] for m in data.get("data", [])]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument(
        "--model",
        action="append",
        help="Specific model id(s); default: all registered.",
    )
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Per-request timeout (giant models load slowly).",
    )
    ap.add_argument(
        "--out",
        default=os.path.join(tempfile.gettempdir(), "olmlx_multiturn_verify.json"),
    )
    args = ap.parse_args()

    models = args.model or list_models(args.base_url)
    if not models:
        print("No models found.", file=sys.stderr)
        return 2

    print(
        f"Verifying multi-turn tool use across {len(models)} model(s) at {args.base_url}\n"
    )
    results: list[dict[str, Any]] = []
    for i, model in enumerate(models, 1):
        print(f"[{i:>2}/{len(models)}] {model} ... ", end="", flush=True)
        t0 = time.time()
        res = run_conversation(args.base_url, model, args.temperature, args.timeout)
        res["seconds"] = round(time.time() - t0, 1)
        results.append(res)
        mark = "PASS" if res["passed"] else "FAIL"
        print(f"{mark}  ({res['rounds']} rounds, {res['seconds']}s) - {res['reason']}")

    passed = sum(1 for r in results if r["passed"])
    print("\n" + "=" * 78)
    print(f"SUMMARY: {passed}/{len(results)} passed\n")
    print(f"{'RESULT':<6} {'ROUNDS':<7} {'MODEL':<46} REASON")
    print("-" * 78)
    for r in sorted(results, key=lambda x: (x["passed"], x["model"])):
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"{mark:<6} {r['rounds']:<7} {r['model'][:45]:<46} {r['reason']}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull transcripts/results: {args.out}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
