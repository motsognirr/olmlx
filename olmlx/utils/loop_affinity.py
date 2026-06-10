"""Explicit enforcement of the event-loop-affinity contract (issue #463).

Much of olmlx's correctness rests on an invariant that used to live only in
comments: every mutation of the registry mappings (read-modify-write of
``models.json`` state), the model manager's ``_loaded`` dict (``unload``'s
check-then-pop), and the prompt-cache stores happens on the single asyncio
event-loop thread, so those critical sections need no locking.  Worker
threads genuinely exist nearby (the decode thread, ``to_thread`` disk I/O,
prefetch pools), which makes a future refactor that moves a caller off-loop
a silent-corruption hazard rather than an error.

This module turns the invariant into an immediate, attributable failure:
the app lifespan calls :func:`bind_loop_thread` on the event-loop thread at
startup (and :func:`unbind_loop_thread` at shutdown), and mutating entry
points call :func:`assert_loop_thread`.

When no thread is bound the assert is a no-op — CLI paths (``olmlx models``,
``olmlx chat``) never bind, and they have no concurrent mutators.  The check
itself is a thread-ident comparison, cheap enough to stay on in production.
"""

from __future__ import annotations

import threading

_bound_thread_ident: int | None = None


def bind_loop_thread() -> None:
    """Record the current thread as the event-loop thread.

    Called from the app lifespan startup, which runs on the loop.
    """
    global _bound_thread_ident
    _bound_thread_ident = threading.get_ident()


def unbind_loop_thread() -> None:
    """Clear the binding (lifespan shutdown / test teardown)."""
    global _bound_thread_ident
    _bound_thread_ident = None


def assert_loop_thread(what: str) -> None:
    """Raise ``RuntimeError`` if called off the bound event-loop thread.

    ``what`` names the violated operation (e.g. ``"ModelRegistry.add_mapping"``)
    so the failure is attributable at the call site.  No-op when unbound.
    """
    # Single read: an unbind racing this check (lifespan shutdown) can't flip
    # the comparison to None mid-expression and raise a spurious error.
    bound = _bound_thread_ident
    if bound is not None and threading.get_ident() != bound:
        raise RuntimeError(
            f"{what} must run on the event-loop thread: its read-modify-write "
            f"is only safe because all mutators are serialized on the loop "
            f"(issue #463). Called from thread {threading.get_ident()!r} "
            f"(loop thread is {bound!r})."
        )
