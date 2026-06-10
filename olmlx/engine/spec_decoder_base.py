"""Shared base class for speculative decoders (#467).

The five strategies (classic, pld, dflash, eagle, mtp — plus
self-speculative) share a protocol (``prefill``/``step``/``reset`` +
``verify_draft_greedy``) but historically had no base class, and the
mechanical parts drifted: only some decoders wrapped ``step()`` in
try/reset (#460), each wired ``cancel_event`` by hand, and the
``_patch_model``/``GDNStateCapture`` install/teardown existed as three
near-identical copies.

:class:`SpecDecoderBase` owns the mechanical parts, **not** the
algorithms:

- ``step()`` is a try/reset wrapper around the abstract ``_step_impl()``
  so a mid-step exception (Metal error, OOM, shape mismatch) can never
  leave partially-modified KV caches or a held GDN patch lock behind.
- ``prefill(prompt, *, segmented=None, cancel_event=None)`` is the
  canonical signature with accept-and-ignore defaults, so
  ``speculative_stream_generate`` passes both kwargs unconditionally.
  ``prefill`` resets before delegating to ``_prefill_impl`` and resets
  again on failure.
- ``reset()`` performs the ordered hook/capture teardown (GDN capture
  close **first** — it holds ``_GDN_PATCH_LOCK`` — then layer-hook
  unpatch, then draft unbind), clears the shared stats counters, and
  calls the abstract ``_reset_state()`` for subclass per-request state.
- The ``spec.prefill``/``spec.step``/``spec.verify`` tracing seams live
  here, so every strategy is instrumented uniformly.

The verify/draft algorithms, cache-trim arithmetic, and the classic/PLD
``_SpecCacheStore`` cross-request reuse stay in the subclasses.

``_LayerHook``/``_patch_model``/``_unpatch_model`` (the universal
target-layer hidden-state capture used by dflash/eagle/mtp) live here
because the base teardown needs ``_unpatch_model``; they are re-exported
from :mod:`olmlx.engine.dflash.decoder` for compatibility.
"""

from __future__ import annotations

import abc
import logging
import threading
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from olmlx.engine.gdn_rollback import (
    GDNBuffer,
    GDNStateCapture,
    get_model_layers as _get_layers,
)
from olmlx.utils import tracing as _tracing

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Layer hooks (universal target hidden-state capture)
# ---------------------------------------------------------------------------


class _LayerHook:
    """Wrap a target layer to capture its output hidden state.

    Transparently proxies attribute access via ``__getattr__`` so the
    rest of the model sees the original layer's interface.
    """

    def __init__(self, layer: Any, idx: int, storage: list[Any]):
        self._layer = layer
        self._idx = idx
        self._storage = storage

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        out = self._layer(*args, **kwargs)
        self._storage[self._idx] = out[0] if isinstance(out, tuple) else out
        return out

    def __getattr__(self, name: str) -> Any:
        return getattr(self._layer, name)


def _patch_model(model: nn.Module, layer_ids: list[int], storage: list[Any]) -> None:
    """Install ``_LayerHook`` on the target's selected layers (idempotent).

    The caller owns *storage* — a pre-allocated list of length
    ``len(layer_ids)`` that each hook writes its slot into. Keeping the
    storage off the ``nn.Module`` avoids contaminating ``model.parameters()``
    once captured ``mx.array``s land in it (mlx's ``Module.__setattr__``
    puts ``list`` attributes into the parameter tree, which would corrupt
    ``mx.eval(model.parameters())`` in distributed setups and serialize
    transient tensors via ``save_weights``).

    Idempotency is detected by checking whether all requested layers
    are already ``_LayerHook``; the second call is a no-op. When
    partially patched (some indices wrapped, others not — e.g. a prior
    call raised mid-loop), only the unwrapped indices are wrapped. We
    must avoid double-wrapping (``_LayerHook(_LayerHook(layer))``)
    because ``_unpatch_model`` peels exactly one level and would leave
    a stale outer hook writing into a dead storage slot.
    """
    layers = _get_layers(model)
    if all(isinstance(layers[lid], _LayerHook) for lid in layer_ids):
        return
    for i, lid in enumerate(layer_ids):
        if isinstance(layers[lid], _LayerHook):
            continue
        layers[lid] = _LayerHook(layers[lid], i, storage)


def _unpatch_model(model: nn.Module) -> None:
    """Remove ``_LayerHook`` wrappers from the target. Safe to call twice."""
    layers = _get_layers(model)
    for i, layer in enumerate(layers):
        if isinstance(layer, _LayerHook):
            layers[i] = layer._layer


# ---------------------------------------------------------------------------
# Greedy verification (shared by every strategy)
# ---------------------------------------------------------------------------


def verify_draft_greedy(
    draft_tokens: list[int],
    target_logits: mx.array,
) -> list[int]:
    """Verify draft tokens against target model logits (greedy).

    Accept draft tokens that match the target's greedy argmax. On first
    mismatch, return the target's preferred token instead.

    Args:
        draft_tokens: list of draft token IDs, length n.
        target_logits: (n+1, vocab) target model logits.

    Returns:
        List of accepted token IDs (1 to n+1 tokens).
    """
    n = len(draft_tokens)

    target_choices = mx.argmax(target_logits, axis=-1)
    mx.eval(target_choices)

    accepted: list[int] = []
    for i in range(n):
        target_token = int(target_choices[i].item())
        if draft_tokens[i] == target_token:
            accepted.append(draft_tokens[i])
        else:
            accepted.append(target_token)
            return accepted

    # All draft tokens accepted — add bonus token
    accepted.append(int(target_choices[n].item()))
    return accepted


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class SpecDecoderBase(abc.ABC):
    """Lifecycle and exception-safety scaffolding for speculative decoders.

    Subclasses implement ``_prefill_impl`` / ``_step_impl`` /
    ``_reset_state`` and set ``self._target`` (and ``self._draft`` if
    they call :meth:`_bind_draft`). Subclass ``__init__`` must call
    ``super().__init__()`` before touching the shared state.

    Two hook/capture lifetimes coexist under one teardown:

    - **Request-lifetime** (dflash/eagle/mtp): layer hooks, draft bind,
      and the GDN capture are installed during ``_prefill_impl`` via
      :meth:`_install_layer_hooks` / :meth:`_bind_draft` /
      :meth:`_install_gdn_capture`, and torn down by ``reset()``.
    - **Decoder-lifetime** (classic/pld): the GDN capture is created in
      the subclass ``__init__`` under its own field (``_gdn_capture``)
      and released by an overridden ``close()`` — base ``reset()`` never
      touches it because the request-lifetime fields stay unset.

    Not thread-safe: one decoder instance serves one request at a time.
    """

    #: Set by subclasses; the model whose layers ``reset()`` unpatches.
    _target: Any
    #: Set by subclasses that bind a draft to the target.
    _draft: Any

    def __init__(self) -> None:
        # Request-lifetime hook/capture state (see class docstring).
        self._patched: bool = False
        self._bound: bool = False
        self._capture: GDNStateCapture | None = None
        self._capture_buffer: GDNBuffer | None = None

        # Diagnostic counters shared by every strategy (reset on prefill).
        self._stats_steps: int = 0
        self._stats_proposed: int = 0
        self._stats_accepted_draft: int = 0

    # ----- canonical protocol ----------------------------------------------

    def prefill(
        self,
        prompt: mx.array,
        *,
        segmented: Any = None,
        cancel_event: threading.Event | None = None,
    ) -> int:
        """Process the prompt, populating KV caches; return the first token.

        Args:
            prompt: (1, seq_len) input token IDs.
            segmented: optional ``SegmentedPrompt`` carrying message
                boundaries for cross-request KV reuse (#421). Strategies
                without a snapshot store accept and ignore it.
            cancel_event: if set during prefill, strategies with a
                sub-chunked prefill raise ``PrefillCancelled`` at the next
                chunk boundary; single-forward strategies accept and
                ignore it.

        Any exception out of the strategy's prefill triggers a full
        ``reset()`` so a failed prefill can never leave layer hooks
        installed or the GDN patch lock held.
        """
        with _tracing.span(
            "spec.prefill",
            strategy=self._strategy(),
            prompt_tokens=int(prompt.shape[-1]),
        ):
            self.reset()
            try:
                return self._prefill_impl(
                    prompt, segmented=segmented, cancel_event=cancel_event
                )
            except Exception:
                # Best-effort cleanup; ``reset()`` swallows nested teardown
                # errors per step so the caller sees the original error.
                self.reset()
                raise

    def step(self) -> tuple[list[int], int]:
        """One speculative decoding step: draft, verify, trim.

        Returns ``(accepted_tokens, num_draft)`` from the strategy's
        ``_step_impl``. Any exception forces a ``reset()`` — a mid-step
        failure would otherwise leave the KV caches partially modified
        and the GDN capture buffer with leftover entries, which the next
        ``step()`` would silently consume (#460). The caller must
        re-``prefill`` to recover, the right semantic for a hard
        inference failure.
        """
        try:
            with _tracing.span("spec.step", strategy=self._strategy()) as _sp:
                proposed_before = self._stats_proposed
                accepted, num_draft = self._step_impl()
                _sp.set_attributes(
                    {
                        "proposed": self._stats_proposed - proposed_before,
                        "accepted": len(accepted),
                    }
                )
                return accepted, num_draft
        except Exception:
            self.reset()
            raise

    def reset(self) -> None:
        """Tear down request state; safe to call repeatedly.

        Ordering is load-bearing: the GDN capture is closed **first**
        because it holds ``_GDN_PATCH_LOCK`` — if a later step raises we
        still want the lock released so subsequent decoder instances can
        patch. Each step's flag/field is cleared in a ``finally`` so a
        double-call never double-unpatches.
        """
        if self._capture is not None:
            try:
                self._capture.close()
            finally:
                self._capture = None
                self._capture_buffer = None
        if self._patched:
            try:
                _unpatch_model(self._target)
            finally:
                self._patched = False
        if self._bound:
            try:
                self._draft.unbind()
            finally:
                self._bound = False
        self._stats_steps = 0
        self._stats_proposed = 0
        self._stats_accepted_draft = 0
        self._reset_state()

    def close(self) -> None:
        """Release resources; default is :meth:`reset`.

        ``ModelManager`` holds whichever decoder type the strategy
        resolved to and calls ``close()`` uniformly on unload, eviction,
        and keep-alive expiry. Decoders with decoder-lifetime resources
        (classic/pld's ``__init__``-created GDN capture and snapshot
        store) override this.
        """
        self.reset()

    def __del__(self) -> None:
        # Belt-and-braces: if the decoder is dropped without explicit
        # close()/reset() (cancelled request, exception unwinding past
        # the streaming bridge), the GDN patch lock would otherwise
        # leak — the patched ``__call__`` holds a strong reference to
        # the capture through its closure, so cyclic GC never runs the
        # capture's own finalizer.
        try:
            self.close()
        except Exception:
            # Finalizers must never raise — the interpreter may already
            # be tearing down.
            pass

    # ----- strategy hooks ----------------------------------------------------

    @abc.abstractmethod
    def _prefill_impl(
        self,
        prompt: mx.array,
        *,
        segmented: Any,
        cancel_event: threading.Event | None,
    ) -> int:
        """Strategy prefill. State is already ``reset()``; raise freely —
        the base wrapper guarantees teardown."""

    @abc.abstractmethod
    def _step_impl(self) -> tuple[list[int], int]:
        """Strategy step body. Raise freely — the base wrapper resets."""

    @abc.abstractmethod
    def _reset_state(self) -> None:
        """Clear strategy-specific per-request state (caches, seeds,
        pending tokens). Called at the end of every ``reset()``; must be
        idempotent and must not raise."""

    # ----- shared helpers ----------------------------------------------------

    def _install_layer_hooks(self, layer_ids: list[int], storage: list[Any]) -> None:
        """Install hidden-capture hooks on the target; ``reset()`` unpatches."""
        _patch_model(self._target, layer_ids, storage)
        self._patched = True

    def _bind_draft(self) -> None:
        """Bind the draft to the target; ``reset()`` unbinds."""
        self._draft.bind(self._target)
        self._bound = True

    def _install_gdn_capture(self) -> None:
        """Create the request-lifetime GDN capture + buffer for non-trimmable
        (hybrid linear-attention) targets; ``reset()`` closes it.

        The buffer is **not** activated — strategies that need the prompt
        forward captured call ``self._capture.use_buffer(self._capture_buffer)``
        themselves (dflash/eagle do; mtp deliberately defers activation
        past its chunked prefix).
        """
        self._capture, self._capture_buffer = GDNStateCapture.for_model(self._target)

    def _verify_greedy(
        self, draft_tokens: list[int], target_logits: mx.array
    ) -> list[int]:
        """:func:`verify_draft_greedy` under a ``spec.verify`` span."""
        with _tracing.span("spec.verify", strategy=self._strategy()):
            return verify_draft_greedy(draft_tokens, target_logits)

    def _strategy(self) -> str:
        """Tracing ``strategy`` attribute, reusing the metrics class→label
        map (classic/pld/dflash/eagle/mtp/self)."""
        from olmlx.utils import metrics as _metrics

        return _metrics._STRATEGY_BY_CLASS.get(type(self).__name__, "unknown")

    # ----- stats --------------------------------------------------------------

    def stats_summary(self) -> dict[str, Any]:
        steps = self._stats_steps
        proposed = self._stats_proposed
        accepted_draft = self._stats_accepted_draft
        summary: dict[str, Any] = {
            "steps": steps,
            "proposed": proposed,
            "accepted_draft": accepted_draft,
            "acceptance_rate": accepted_draft / proposed if proposed else 0.0,
            "avg_tokens_per_step": (accepted_draft + steps) / steps if steps else 0.0,
        }
        summary.update(self._stats_extra())
        return summary

    def _stats_extra(self) -> dict[str, Any]:
        """Strategy-specific ``stats_summary`` keys (``lambda``,
        ``block_size``, ``ema_acceptance_rate``)."""
        return {}
