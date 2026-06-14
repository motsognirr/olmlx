# Multi-model panel + judge coordinator (sequential, single-box)

Issue: #520

## Summary

A new synthetic `type: "panel"` "model" that answers a request with a **panel** of
several models and a separate **judge** that reconciles the panel's output into one
response. It is a **drop-in replacement** for any tool-calling model: the standard
OpenAI / Ollama contract is preserved, and **the caller (coding agent) executes the
tools**, exactly as with a single model.

Scope here is the **sequential, single-box** variant: classifier, panelists, and judge
run one at a time under the existing global inference lock. Distributed/parallel
execution is an explicit follow-up (out of scope).

## Key design decision: panel as a per-turn reconciler

The issue's literal premise — "each panelist runs its own agent loop with its own tool
calls and its own results" — is **incompatible** with two hard constraints we adopt:
drop-in tool-calling semantics, and **client-side** tool execution. If the client
executes tools, the server cannot drive N divergent tool loops to completion, because
each loop would need results only one client conversation can produce.

The reconciliation: **do not build a server-side agent loop. Reuse the client's loop.**
The panel "model" only has to satisfy the contract on each turn — return either
`tool_calls` or `content`. All multi-model logic is internal to a single turn:

```
request ──> [stateless coordinator call]
              route (classifier) → pick panel members
              run each panelist on shared history (sequential, under the lock)
              reconcile:
                 any panelist wants tools?  → emit deduped UNION of tool_calls ──┐
                 nobody wants tools?         → judge synthesizes fresh answer    │
                                                                                 │
client executes tools, appends results, calls again  <───────────────────────────┘
```

From the coding agent's perspective this is indistinguishable from one tool-calling
model: same request shape, same `tool_calls`/`content` responses, same
resubmit-with-results loop. No server-side MCP, no new tool executor.

**What we give up:** panelists do not have independent tool traces — they share one
tool-result history (one client, one conversation). Per turn each *proposes* tool calls,
but only the reconciled union executes, and everyone sees those results next turn. The
judge can still verify groundedness against the shared tool results; it just cannot
compare divergent tool *paths*. This is the only shape consistent with "drop-in + client
executes tools."

## Decisions (locked)

| Question | Decision |
|---|---|
| Entry surface | HTTP synthetic model (OpenAI + Ollama routers), selectable by name |
| Tool execution | Client-side; panel is a per-turn reconciler riding the client's loop; server stateless across calls |
| Tool-call merge | Deduped **union** of all panelists' proposed calls |
| Panel composition | **Per-task routing** via a dedicated small classifier model |
| Judge output | **Synthesize a fresh merged answer** |
| Config surface | Synthetic `type: "panel"` entry in `models.json` |
| Stop condition | Stop issuing tools only when **no panelist** requests any |
| Failure handling | **Fail the request** (HTTP error) on any classifier/panelist/judge failure |

## Architecture

### Component & placement

New module `engine/panel.py` exposing `PanelCoordinator` with a
**`generate_chat`-compatible** entry point — same arguments and same return shapes
(`AsyncGenerator[dict]` when `stream=True`, `dict` otherwise). It calls
`engine/inference.py:generate_chat` for **every** model (classifier, panelists, judge),
so all Metal-stream and inference-lock handling is reused unchanged. The coordinator
never touches MLX directly.

### Per-turn flow (stateless)

The server stays stateless across HTTP calls — the client carries history and executes
tools. Each coordinator call:

1. **Route.** Run the classifier model on the **first user message** (stable across the
   conversation → routing is deterministically re-derived every turn, no stored state).
   Output is grammar-constrained (`engine/grammar.py` xgrammar) to one of the configured
   route keys → yields the panel member list. Unknown/edge → `default` route.
2. **Run panelists.** For each member: `generate_chat(stream=False)` on the current
   history with the client's declared tools; `parse_model_output` →
   `(thinking, answer, tool_uses)`. Sequential under the lock.
3. **Reconcile.**
   - If **any** panelist proposed tool calls → emit the **deduped union** as this turn's
     `tool_calls` and return.
   - If **none** did → run the **judge** (no tools passed to it, so it can only write
     prose) to synthesize a fresh merged answer from the panelists' answers + thinking +
     the shared tool results already present in history. Emit as `content`.

### Tool-call union

Dedup panelist `tool_uses` by normalized `(name, arguments)`: identical calls collapse
to one execution; different args both run. Assign fresh unique `tool_call` IDs. The
client executes all and returns results keyed by ID; next turn the history holds every
tool_call+result pair, so all panelists see all gathered evidence. The emitted assistant
turn carries the union (a panelist may "see" calls it did not make — the accepted
shared-history fiction).

### Config schema (`models.json`)

```jsonc
"my-panel:latest": {
  "type": "panel",
  "classifier": "qwen3-0.6b",        // dedicated small router model
  "judge": "gpt-oss-20b",            // synthesizer; runs WITHOUT tools
  "routes": {
    "code":    ["qwen3-coder", "devstral", "deepseek-coder"],
    "math":    ["qwen3", "deepseek-r1-distill", "magistral"],
    "general": ["qwen3", "llama3.1", "gpt-oss-20b"],
    "default": ["qwen3", "mistral", "llama3.1"]
  }
}
```

Members / judge / classifier reference existing `models.json` entries by name. Registry
load **validates** they all exist and **warns** if the judge is also a panelist
(self-preference bias). `routes` must contain `default`.

### Router integration & streaming

In `routers/openai.py` and `routers/chat.py`, before `ensure_loaded`, branch on
`registry.is_panel(model_name)` → dispatch to the coordinator instead of `generate_chat`.
The coordinator returns the same post-parse structure both routers already format, so
tool-call conversion / SSE / NDJSON paths are reused.

Streaming: the classifier and panelists run buffered — reuse the keepalive in
`routers/streaming_common.py` so long internal compute does not time the client out —
then proxy the judge's token stream straight through. Tool-call turns emit a single
tool-calls chunk (existing tools-mode buffering handles it).

### Failure handling

Any classifier / panelist (load, generate, timeout) / judge failure raises and surfaces
as an HTTP error. No silent degradation.

## Performance & memory caveats

- Every **tool turn** re-runs the classifier + all N panelists; the **final turn** adds
  the judge. A coding agent doing 20 tool round-trips pays ~20×(1+N) sequential model
  calls. This is inherent to stateless + drop-in and is the cost the issue accepts.
- With `max_loaded_models=1`, every panelist call evicts the prior model → reload thrash.
  Panel use effectively requires `max_loaded_models ≥ panel size + judge + classifier` so
  diverse 4-bit models co-reside. The coordinator **warns** (does not auto-change) if the
  limit looks too low for the configured panel.

## Testing (TDD, mocked models)

Unit tests mock `generate_chat` for determinism:

- Route selection from (grammar-constrained) classifier output → correct member list;
  unknown category → `default`.
- Tool-call union: dedup by `(name, args)`, unique IDs assigned.
- Reconcile decision: any-tools → emit union; none → judge synthesis.
- Deterministic re-routing on a continuation history (tool results present) → same panel.
- Judge invoked **without** tools.
- Failure (classifier/panelist/judge) → raises, request fails.
- Registry validation of `panel` entries (missing member errors; judge-in-panel warns;
  missing `default` errors).
- One router interception test (panel model name routes to coordinator).

Heavy real-model runs stay out of the default suite, matching the existing canary
pattern.

## Scope

**In scope:** sequential single-box panel + per-task routing + judge synthesis,
HTTP-surfaced, client-executed tools.

**Out of scope (issue follow-up):** distributed / parallel panelist execution.

**Docs:** add a CLAUDE.md invariant once implemented — the panel coordinator must route
every model call through `generate_chat`; never bypass the generation-stream handling.
