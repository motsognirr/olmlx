"""Subprocess worker — runs prompts against a single scenario configuration.

Invoked as: python -m olmlx.bench.worker --model MODEL --prompts-json PATH --results-json PATH
Environment variables control the scenario's feature flags (set by runner before spawn).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


async def _run_prompts(
    model: str, prompts: list[dict], max_tokens_override: int | None
) -> list[dict]:
    """Create a real ASGI app and run each prompt through /api/chat."""
    from olmlx.app import create_app
    from olmlx.engine.model_manager import ModelManager
    from olmlx.engine.registry import ModelRegistry
    from olmlx.models.store import ModelStore

    registry = ModelRegistry()
    registry.load()
    store = ModelStore(registry)
    manager = ModelManager(registry, store)
    manager.start_expiry_checker()

    app = create_app()
    app.state.registry = registry
    app.state.model_manager = manager
    app.state.model_store = store

    from httpx import ASGITransport, AsyncClient

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    results = []
    async with AsyncClient(transport=transport, base_url="http://bench") as client:
        for prompt in prompts:
            max_tokens = max_tokens_override or prompt.get("max_tokens", 256)
            body = {
                "model": model,
                "stream": False,
                "messages": prompt["messages"],
                "options": {
                    "seed": 42,
                    "temperature": 0.0,
                    "num_predict": max_tokens,
                },
            }
            try:
                resp = await client.post("/api/chat", json=body, timeout=300.0)
                if resp.status_code == 200:
                    data = resp.json()
                    results.append(
                        {
                            "prompt_name": prompt["name"],
                            "category": prompt["category"],
                            "output_text": data.get("message", {}).get("content", ""),
                            "status_code": resp.status_code,
                            "error": None,
                            "eval_count": data.get("eval_count", 0),
                            "eval_duration_ns": data.get("eval_duration", 0),
                            "prompt_eval_count": data.get("prompt_eval_count", 0),
                            "prompt_eval_duration_ns": data.get(
                                "prompt_eval_duration", 0
                            ),
                            "total_duration_ns": data.get("total_duration", 0),
                        }
                    )
                else:
                    results.append(
                        {
                            "prompt_name": prompt["name"],
                            "category": prompt["category"],
                            "output_text": "",
                            "status_code": resp.status_code,
                            "error": resp.text[:500],
                            "eval_count": 0,
                            "eval_duration_ns": 0,
                            "prompt_eval_count": 0,
                            "prompt_eval_duration_ns": 0,
                            "total_duration_ns": 0,
                        }
                    )
            except Exception as e:
                results.append(
                    {
                        "prompt_name": prompt["name"],
                        "category": prompt["category"],
                        "output_text": "",
                        "status_code": 0,
                        "error": str(e),
                        "eval_count": 0,
                        "eval_duration_ns": 0,
                        "prompt_eval_count": 0,
                        "prompt_eval_duration_ns": 0,
                        "total_duration_ns": 0,
                    }
                )

    await manager.stop()
    return results


def main():
    parser = argparse.ArgumentParser(description="Bench worker subprocess")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts-json", required=True, type=Path)
    parser.add_argument("--results-json", required=True, type=Path)
    parser.add_argument("--max-tokens", type=int, default=None)
    args = parser.parse_args()

    prompts = json.loads(args.prompts_json.read_text())
    results = asyncio.run(_run_prompts(args.model, prompts, args.max_tokens))
    args.results_json.write_text(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
