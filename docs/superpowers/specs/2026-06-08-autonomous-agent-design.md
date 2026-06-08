# Autonomous Agent — Design

**Date:** 2026-06-08
**Status:** Approved design, pending implementation
**Topic:** Run the `olmlx` chat agent as an autonomous, self-directing agent in the style of the Nous Research *Hermes Agent* (self-improving loop + memory + skills + subagents), exposed over HTTP in the *OpenClaw* gateway style.

## Goal

Turn the existing interactive `olmlx chat` agent into an autonomous agent that pursues a high-level goal across many turns **without a human at each step**, accumulates knowledge over time, and is driven over an HTTP API.

## Context

`olmlx`'s `ChatSession.send_message()` (`olmlx/chat/session.py`) is already a ReAct loop: generate → parse tool calls → execute tools → feed results back, repeating until the model emits no more tool calls or hits `max_turns` (default 25). It already has builtin tools (file read/write/edit, glob, grep, bash, web search/fetch, todo_write, create/read/update_plan, an `ask question` tool), MCP tools, on-demand markdown **skills** (`chat/skills.py`), tool-safety policies (`chat/tool_safety.py`), and an LLM judge for auto-approving tool calls (`chat/llm_judge.py`). It is driven interactively by the TUI — one human message per "turn".

The gap between this and an *autonomous* agent is **self-direction**: pursuing a goal across many turns without human input, with the agent's own termination/safety conditions, plus Hermes-style persistence (cross-session memory, self-improving skills, subagent delegation).

### Reference agents
- **Hermes Agent (Nous Research)** — the closer model: a ReAct loop plus cross-session SQLite memory (FTS5 + LLM summarization), self-improving skills (the agent writes its own skill files after complex tasks), and subagent delegation. "Runs on your server" and self-improves the longer it runs.
- **OpenClaw** — a multi-channel gateway/orchestrator (chat surfaces → pluggable agent backends); the autonomy lives in the backend. We adopt its *separate the always-on HTTP surface from the agent reasoning* pattern.

## Requirements (from brainstorming)

- **Autonomy depth:** Full Hermes-style — loop + persistent memory + self-improving skills + subagent delegation.
- **Surface:** HTTP API endpoint (OpenClaw gateway pattern), background long-running runs.
- **Persistence:** SQLite-persisted and **resumable** — runs, checkpoints, memory, and learned skills all survive a server restart.
- **Stop conditions / guardrails (all in scope):**
  1. Self-judged goal completion (explicit `finish` tool the model must call to end).
  2. Hard budgets (max iterations, token budget, wall-clock timeout) enforced by the orchestrator regardless of model behavior.
  3. Stall / no-progress detection (loops, repeated failed actions, no memory progress).
  4. Existing tool-safety policy + LLM-judge approval preserved during autonomous runs.

## Key constraint: the single inference lock

`olmlx` serializes **all** generation behind one inference lock (single Metal GPU, single-user — see `CLAUDE.md` "Usage Context"). Therefore "parallel subagents" can orchestrate concurrently but their *generation* serializes on the GPU. Subagents are modeled as a **concurrency-limited queue**, not true wall-clock parallelism. This is honest and matches the existing backpressure model.

## Architecture — Approach A: wrap & extend `ChatSession`

The autonomous orchestrator drives the **existing** `ChatSession`, reusing its entire ReAct loop, tool execution, tool-safety + LLM-judge, thinking handling, and memory-truncation. All new autonomy logic is isolated in a new `engine/agent/` package, leaving the interactive chat path untouched.

> Approaches considered and rejected: **B** (standalone engine loop calling `generate_chat` directly) re-implements `ChatSession`'s hard-won logic — duplication and drift risk. **C** (extract a shared `AgentLoop` consumed by both chat and autonomous runs) is the best long-term shape but has the largest blast radius on the working chat path. We pick **A** for maximum reuse and zero risk to chat; if coupling later hurts, extract the shared loop (C) as a follow-up.

### Components

**1. Orchestrator — `engine/agent/orchestrator.py`**
Owns the goal-pursuit loop around a `ChatSession`. Per iteration:
1. Inject the tiered memory/skill context block into the system prompt ("stable → context → volatile": identity/goal, recalled memory, recent progress).
2. Run one ReAct step via `ChatSession.send_message` (which itself runs the inner tool loop until the model stops).
3. Check termination: model called the **`finish` tool** (success) OR a hard budget tripped OR stall detected.
4. Persist a checkpoint (full message history + budget counters) to the store.
5. Emit events to the run's event stream.

**Continuation semantics (the one extension to the existing loop):** a stop *without* `finish` is not the end of the run — the orchestrator injects a continuation nudge ("you stopped without calling `finish`; continue toward the goal or call `finish` if complete"). This is the single behavioral change layered over `ChatSession`; the inner loop is otherwise unmodified. Supports **resume** from the last persisted checkpoint.

**2. Persistence — `engine/agent/store.py` → SQLite at `~/.olmlx/agent.db`**
Tables:
- `runs` — `id`, `goal`, `status` (queued/running/paused/finished/failed/cancelled), `config` (JSON), budget counters (iterations/tokens/elapsed), `created_at`/`updated_at`.
- `events` / `checkpoints` — full message history per run, for resume + SSE replay.
- `memory` — FTS5 full-text table of memory entries (scope, text, embedding-free FTS).
- `skills` — learned skill files (name, description, body, source run).

All disk I/O is offloaded via `asyncio.to_thread`, matching the prompt-cache disk pattern (`engine/prompt_cache/`). No new threads owning the GPU.

**3. Memory — `engine/agent/memory.py`**
Hermes-style record/recall: FTS5 search + LLM summarization when the working set overflows. Exposed as `remember` / `recall` builtin tools and auto-injected into the system prompt as the tiered block. Survives restart and context truncation. Bounded by `OLMLX_AGENT_MEMORY_*` knobs.

**4. Skills (self-improving) — extend `chat/skills.py`**
Add a writable path: a `create_skill` tool lets the agent author a markdown skill after a complex task; persisted to SQLite + the skills dir; loaded on demand by the existing loader (`load_skills_from_dir`). Closes the Hermes learning loop with minimal new code — the read path already exists.

**5. Subagents — `engine/agent/delegate.py`**
A `delegate` tool spawns a child `AgentRun` with an isolated goal and its own context; results return to the parent. **Bounded depth and fan-out** (`OLMLX_AGENT_MAX_SUBAGENT_DEPTH`, `OLMLX_AGENT_MAX_SUBAGENT_FANOUT`). Children share the inference lock — concurrency-limited queue, serialized generation. Children are themselves persisted runs (parent_id link), so the whole tree is resumable.

**6. HTTP surface — `routers/agent.py`** (registered in `app.py`, gated on `OLMLX_AGENT_ENABLED`)
- `POST /v1/agent/runs` — create + start (background `asyncio` task), returns run id.
- `GET /v1/agent/runs/{id}` — status + budget usage.
- `GET /v1/agent/runs/{id}/events` — SSE stream (replays persisted events, then live).
- `POST /v1/agent/runs/{id}/cancel` — request cancellation (cooperative, checked at iteration boundaries).
- `POST /v1/agent/runs/{id}/resume` — resume a paused/interrupted run from its last checkpoint.
- `GET /v1/agent/runs` — list runs.

A run registry tracks in-flight `asyncio` tasks. Runs execute under the existing inference lock (natural backpressure). Schemas live in `schemas/agent.py`.

**7. Config — `config.py`, `OLMLX_AGENT_*` prefix**
`enabled` (default off), `db_path`, `max_iterations`, `token_budget`, `wallclock_timeout`, `max_subagent_depth`, `max_subagent_fanout`, `memory_max_entries` / `memory_recall_k`.

### Guardrails (all four requirements)
- **Self-judged completion:** the `finish` tool is the only clean success-path terminator; the model must call it.
- **Hard budgets:** orchestrator enforces max iterations / token budget / wall-clock timeout independent of the model; tripping any sets status `failed` (budget-exhausted) with a clear reason.
- **Stall detection:** extends the existing repetition detector (`_detect_repetition`) and consecutive-tool-failure counter to the *run* level — abort on N iterations with no new memory progress / repeated identical actions.
- **Tool-safety preserved:** the existing `ToolSafetyPolicy` + LLM-judge gating stays active during autonomous runs (auto-approve safe, gate/deny dangerous), so headless runs can't execute unbounded destructive commands.

## Data flow

```
POST /v1/agent/runs {goal, config}
  → store.create_run() (status=queued)
  → spawn background asyncio task: Orchestrator.run(run_id)
      loop (bounded by budgets):
        inject tiered memory/skill context
        ChatSession.send_message(continuation prompt)   # inner ReAct loop, tools, safety
        on finish tool      → status=finished, break
        on budget trip      → status=failed, break
        on stall            → status=failed, break
        else                → checkpoint + continue
      emit events throughout
GET /v1/agent/runs/{id}/events  → SSE: replay persisted events, then live tail
POST .../cancel                 → cooperative cancel at next iteration boundary
POST .../resume                 → rehydrate ChatSession from last checkpoint, continue loop
```

## Error handling
- Model load failure → run status `failed`, reason surfaced (reuse `ChatSession`'s `model_load_error`).
- Tool execution errors → fed back to the model (existing behavior); run-level consecutive-failure cap aborts a wedged run.
- Server restart mid-run → run left `running` in DB is marked `interrupted` on startup scan; resumable via `/resume`.
- SSE client disconnect → run continues in background (decoupled from the request, OpenClaw-style); reconnect replays from the store.
- Subagent failure → reported to the parent as a tool error; parent decides whether to continue or `finish`.

## Testing strategy (TDD per `CLAUDE.md`)
- **Store:** unit tests for run lifecycle, checkpoint round-trip, FTS5 memory search, skills persistence (in-memory SQLite).
- **Orchestrator:** termination on each path (finish / each budget / stall) with a mocked `ChatSession`; resume from checkpoint reproduces state; continuation-nudge fires on stop-without-finish.
- **Guardrails:** budget enforcement is independent of model output; tool-safety gating still applies in autonomous mode.
- **Subagents:** depth/fanout bounds enforced; serialized execution under a mocked lock; parent_id linkage + resumability.
- **Router:** create/get/list/cancel/resume happy paths + 422s; SSE replay-then-live; gated off when `OLMLX_AGENT_ENABLED=false`.
- **Live (real_model, outside `tests/integration/`):** a short end-to-end autonomous run against a small model reaching `finish`.

## Delivery — phased (parent epic + sub-tasks)

This is six subsystems; land incrementally behind `OLMLX_AGENT_ENABLED` (default off):

- **Phase 1 — Core loop + HTTP skeleton.** `engine/agent/` orchestrator with continuation/`finish` semantics, SQLite store (runs + checkpoints), `routers/agent.py` (create/get/list/cancel/resume + SSE), config, hard budgets, stall detection, tool-safety preserved, resume-on-restart. *This is a usable autonomous agent on its own.*
- **Phase 2 — Persistent memory.** FTS5 memory table, `remember`/`recall` tools, tiered system-prompt injection, summarization-on-overflow.
- **Phase 3 — Self-improving skills.** Writable skills path, `create_skill` tool, SQLite + dir persistence, on-demand load.
- **Phase 4 — Subagent delegation.** `delegate` tool, child runs with parent linkage, depth/fanout bounds, serialized-queue execution, resumable tree.

## Out of scope (YAGNI)
- Multi-channel chat surfaces (WhatsApp/Telegram/etc — the full OpenClaw gateway). The HTTP API is the integration seam if wanted later.
- Distributed / multi-machine autonomous runs.
- True parallel GPU execution for subagents (precluded by the single inference lock).
- Authentication / multi-tenant run isolation (single-user, localhost-only per `CLAUDE.md`).
```
