# #503 Phase 0 — Prefill measurement findings

**Status:** awaiting numbers (fill in after running `olmlx bench run --prompts agentic-69k`).

## Setup
- Model: <agentic model, e.g. Qwen3-32B-4bit>
- Prompt: `agentic-69k` (~69k tokens, tool-defs system segment)
- Command: `olmlx bench run --prompts agentic-69k --model <model>`
- Read: server log `prefill Xs (fresh F/N tok, cache-covered C)` + `decode`/`spec.prefill` span attrs.

## Numbers (TO FILL)
| metric | value |
| --- | --- |
| measured prefill (ttft_ns) | |
| decode wall-clock | |
| prefill / total | |
| fresh vs cache-covered tokens | |
| target_lane_ns / draft_lane_ns (spec only) | |

## Decision (TO FILL)
- Is prefill the measured bottleneck on the agentic case? yes/no
- Build #503 proper / scope down / defer? <decision + rationale>
