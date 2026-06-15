"""Autonomous agent subsystem (issue #445).

Drives the existing ``ChatSession`` ReAct loop across many turns toward a
goal without a human at each step, with SQLite-persisted resumable runs,
hard budgets, stall detection, cross-session memory, self-improving skills,
and bounded subagent delegation. Gated behind ``OLMLX_AGENT_ENABLED``.
"""
