# olmlx agentic software factory

A mostly-autonomous loop that turns problems into merged fixes. The **only**
routine manual step is approving a plan.

```
[Explore]  nightly, inference runner
  exercises real use cases → files deduped `source:explorer` issues
        │
        ▼
[Plan]  on issue opened
  explores code → posts a plan, labels `plan:proposed`   (or triage-closes)
        │
        ▼
   ★ HUMAN: flip `plan:proposed` → `plan:approved` ★      ← the one gate
        │
        ▼
[Implement]  on `plan:approved`
  TDD on `factory/issue-N` → mocked tests green → opens PR (Closes #N)
        │
        ▼
[Babysit]  event-driven, on CI / review / comment
  triage every finding → {fix → push → loop | dismiss}
  clean + green → summary comment → MERGE
  stuck / ambiguous → label `needs-human`, stop
        │
        ▼
     merge → Explore picks it up again
```

## Workflows

| File | Stage | Trigger | Runner |
|---|---|---|---|
| `factory-explore.yml` | Explore | nightly cron + manual | `[self-hosted, inference]` |
| `factory-plan.yml` | Plan | `issues: opened/reopened` | `[self-hosted, agent]` |
| `factory-implement.yml` | Implement | `issues: labeled = plan:approved` | `[self-hosted, agent]` |
| `factory-babysit.yml` | Babysit | PR review / comment / CI completion | `[self-hosted, agent]` |
| `factory-labels.yml` | setup | manual | ubuntu |

Existing `ci.yml` (required checks) and `aider-code-review.yml` (GPT + DeepSeek
reviewers) are reused unchanged. Real inference work (`factory-explore` +
`real-model-smoke`) shares the `inference-serial` concurrency group so only one
runs at a time on the Mac.

## One-time setup (manual prerequisites)

1. **Runner labels.** Add `agent` and `inference` labels to your self-hosted
   runner(s).
   - One machine: give it **both** `agent` and `inference`.
   - Two machines: make the second `inference`-only; heavy MLX work migrates
     there automatically with no YAML changes.

2. **`FACTORY_PAT` secret.** A fine-grained PAT (or GitHub App installation
   token) with `contents:write`, `issues:write`, `pull_requests:write` on this
   repo. **Required**, because anything created with the default `GITHUB_TOKEN`
   (issues, PRs, pushes) does **not** trigger other workflows — so the explorer's
   issues wouldn't reach Plan, and the implementer's PR wouldn't reach CI or
   Babysit. The factory would silently stall without it.

   *Least privilege (optional):* the explorer only files issues, so for tighter
   scoping you can give `factory-explore.yml` a second, issues-only PAT instead
   of the full `FACTORY_PAT`. The explorer prompt also hard-forbids pushing.

3. **`CLAUDE_CODE_OAUTH_TOKEN` secret.** Already used by `claude.yml`.

4. **Create the labels.** Run the **Factory labels** workflow once.

5. **Branch protection on `main`.** Mark `ci.yml`'s jobs (Ruff / Pyright /
   Tests) as **required**. The babysitter never merges red, but required checks
   are a hard backstop.

## Design notes

- **Auto-merge on green = the babysitter merges.** Native GitHub auto-merge is
  intentionally NOT used. The babysitter reads and *judges* reviewer findings
  (fix vs dismiss) instead of treating them as a binary gate, so AI reviews
  don't need to be wired as required status checks.
- **The approval gate is the throttle.** It bounds both spend (Claude + Aider
  tokens — the real cost; the Mac compute is free) and the rate of change.
- **`needs-human` is the only unhappy-path escape.** Implementer (infeasible
  plan) and babysitter (stuck after ~3 loops, or ambiguous/architectural
  finding) escalate here instead of merging or spinning forever.
- **CLAUDE.md is law.** Every agent reads it first; many reviewer "bugs" are
  deliberate invariants, and defending one is a valid dismissal.
- **The gate trusts the approver.** Applying `plan:approved` starts a
  privileged run (`FACTORY_PAT` + arbitrary code on the self-hosted runner), so
  the security of the whole factory reduces to the security of whoever can
  apply that label. Keep approval rights to trusted human accounts with 2FA;
  treat a compromised approver account as a full runner compromise.
- **A `plan:approved` you didn't apply is unsafe.** The planner has
  `issues: write`, so a prompt-injected planner *can* stamp `plan:approved` on
  an issue. It can't start the implementer that way (its label is applied with
  `GITHUB_TOKEN`, which doesn't trigger `issues: labeled` workflows — see
  factory-plan.yml), but the stray label can still mislead a human into opening
  a PR by hand. If you see `plan:approved` on an issue no human approved, remove
  it and investigate.
- **Agents reach GitHub via the `gh` CLI, not an MCP server.** On a non-@claude
  trigger, claude-code-action runs in *agent mode*, which provides no GitHub MCP
  tools — so each workflow sets `GH_TOKEN` and the agents shell out to `gh`. The
  planner only needs `gh`, so its Bash is scoped to `Bash(gh:*)`. Explore /
  Implement / Babysit need a **full shell** (pytest, git, uv), so they run with
  unrestricted Bash on the self-hosted runner: a prompt injection in issue/PR/CI
  content they read could execute arbitrary commands on that machine. The prompts
  harden against this, but treat the runner as exposed to whoever can open issues
  or PRs the factory acts on — run it on a dedicated/isolated host, not a daily
  driver, and keep `FACTORY_PAT` minimally scoped.
