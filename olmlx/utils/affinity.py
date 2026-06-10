"""Debug tripwires that make implicit mutation-affinity contracts explicit.

A lot of olmlx correctness rests on "all mutations of this structure happen
on the asyncio event-loop thread" (registry read-modify-write of models.json,
``ModelManager.unload``'s check-then-pop, the prompt-cache store's in-memory
paths) or on "mutations are serialized by the inference lock even though they
roam across decode worker threads" (``_SpecCacheStore``). Those invariants
were previously documented in comments but invisible at the call sites, so a
future refactor (moving a caller into a thread, adding an ``await`` inside a
critical section) would turn into silent state corruption rather than an
error. These guards convert a violated invariant into an immediate,
attributable failure. Issue #463.

A literal ``asyncio.get_running_loop()`` assert would be wrong for both
contracts: registry mutators also run in sync CLI contexts with no loop at
all (``olmlx models``), and the spec store legitimately runs on a different
worker thread each request. Hence the two shapes below.
"""

from __future__ import annotations

import threading


class ThreadAffinityGuard:
    """Tripwire for the "all mutations happen on one thread" contract.

    The first :meth:`check` pins the calling thread as the owner (the event
    loop's thread in the server, the main thread in the CLI and tests); any
    later check from a different thread raises ``RuntimeError``. The
    unsynchronized pin is deliberate — this is a tripwire, not a lock, and
    a racy first pin still leaves one of the racing threads to trip it.
    """

    __slots__ = ("_what", "_owner_ident", "_owner_name")

    def __init__(self, what: str) -> None:
        self._what = what
        self._owner_ident: int | None = None
        self._owner_name: str | None = None

    def check(self) -> None:
        ident = threading.get_ident()
        if self._owner_ident is None:
            self._owner_ident = ident
            self._owner_name = threading.current_thread().name
            return
        if self._owner_ident != ident:
            raise RuntimeError(
                f"{self._what} called from thread "
                f"{threading.current_thread().name!r} but is pinned to "
                f"thread {self._owner_name!r} — all mutations must happen "
                f"on the owning (event-loop) thread (issue #463)"
            )


class SerializedMutationGuard:
    """Tripwire for the "mutations never overlap" contract.

    Context manager wrapping each mutating method body. Entry while another
    entry is still in progress — from any thread, including re-entrantly from
    the same one — raises ``RuntimeError``. Unlike
    :class:`ThreadAffinityGuard` it allows the mutating thread to change
    between calls, which is the actual contract of structures serialized by
    the inference lock but driven from per-request decode worker threads.
    """

    __slots__ = ("_what", "_lock")

    def __init__(self, what: str) -> None:
        self._what = what
        self._lock = threading.Lock()

    def __enter__(self) -> SerializedMutationGuard:
        if not self._lock.acquire(blocking=False):
            raise RuntimeError(
                f"{self._what} mutated concurrently from thread "
                f"{threading.current_thread().name!r} — mutations must be "
                f"serialized (by the inference lock) (issue #463)"
            )
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._lock.release()
