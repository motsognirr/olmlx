"""DFlash block-diffusion speculative decoder.

Implements the same ``prefill``/``step``/``reset`` protocol as
``SpeculativeDecoder`` so the existing ``speculative_stream_generate``
streaming bridge works unchanged.

Universal target support is achieved by monkey-patching the target's
selected layers with ``_LayerHook`` (see ``_patch_model``) — this works
for any model whose layers list lives at one of three known locations
(``model.layers`` / ``model.model.layers`` / ``model.language_model.layers``).
No per-architecture adapter code is needed.

For target architectures whose KV cache cannot be trimmed in-place
(notably Qwen3.5 / Qwen3-Coder-Next with ``GatedDeltaNet`` linear-attention
layers), ``_GDNStateCapture`` monkey-patches ``GatedDeltaNet.__call__``
to snapshot the conv + GDN state per draft step and replays
``gated_delta_update`` on the accepted prefix to restore the correct
state on rejection.
"""

from __future__ import annotations

import logging
from threading import RLock
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import (
    RotatingKVCache,
    can_trim_prompt_cache,
    make_prompt_cache,
)

from olmlx.engine.dflash.draft_model import DFlashDraftModel, DraftConfig
from olmlx.engine.speculative import verify_draft_greedy

try:
    import mlx_lm.models.gated_delta as _gd_mod  # type: ignore[import-not-found]

    _HAS_GDN = True
except ImportError:
    _gd_mod = None  # type: ignore[assignment]
    _HAS_GDN = False


# ``_trim_recent_cache`` reaches into ``RotatingKVCache._temporal_order`` and
# ``._idx`` to reorder + slice the rotating buffer. These are private to mlx-lm
# and may be renamed without a semver bump. Probe at import time so an
# incompatible mlx-lm release fails fast (rather than mid-generation when
# DFlash first hits a sliding-window draft cache). ``_temporal_order`` is a
# method (class-level), but ``_idx`` is set in ``__init__`` (instance-level),
# so it must be probed via a sentinel instance.
def _probe_rotating_kv_privates() -> bool:
    if not hasattr(RotatingKVCache, "_temporal_order"):
        return False
    try:
        return hasattr(RotatingKVCache(max_size=1, keep=0), "_idx")
    except Exception:
        return False


_HAS_ROTATING_KV_PRIVATES = _probe_rotating_kv_privates()

logger = logging.getLogger(__name__)

# Module-level lock guarding the GatedDeltaNet monkey-patch. Prevents
# two ``_GDNStateCapture`` instances from racing on the class-level
# ``__call__`` attribute.
_GDN_PATCH_LOCK = RLock()


# ---------------------------------------------------------------------------
# Layer hooks
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


def _get_layers(model: nn.Module) -> list[Any]:
    """Find the layers list on a target model.

    Tries (in order): ``model.model.layers``, ``model.language_model.layers``,
    ``model.layers``. Raises ``AttributeError`` if none is found.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers  # type: ignore[no-any-return]
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model.layers  # type: ignore[no-any-return]
    if hasattr(model, "layers"):
        return model.layers  # type: ignore[no-any-return]
    raise AttributeError(
        f"Cannot find layers in {type(model).__name__}; tried .model.layers, "
        ".language_model.layers, .layers"
    )


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


def _trim_recent_cache(cache: list[Any], num_tokens: int) -> None:
    """Trim trailing ``num_tokens`` from each layer's KV cache.

    Special-cases ``RotatingKVCache`` (must reorder before slicing).
    Skips entries with no ``trim`` method — the GDN rollback path
    handles those separately.
    """
    if num_tokens <= 0:
        return
    for c in cache:
        n = min(getattr(c, "offset", num_tokens), num_tokens)
        if n <= 0:
            continue
        if isinstance(c, RotatingKVCache) and c.keys is not None:
            if not _HAS_ROTATING_KV_PRIVATES:
                raise RuntimeError(
                    "DFlash rollback for sliding-window draft caches relies "
                    "on the private mlx-lm API ``RotatingKVCache._temporal_order`` "
                    "/ ``._idx``. The installed mlx-lm version no longer "
                    "exposes them — pin a compatible version or file an "
                    "olmlx bug to update the private-API access pattern."
                )
            # When the cache has been in rotating mode (``offset > max_size``),
            # the underlying buffer holds at most ``max_size`` entries even
            # though ``offset`` is larger. Trimming requires us to drop back
            # to non-rotating mode with a consistent ``offset == _idx ==
            # keys.shape[2]``; otherwise ``update_and_fetch`` would write at
            # a stale ``_idx`` into a buffer it believes is full and corrupt
            # the KV state.
            actual_stored = min(c.offset, c.keys.shape[2])
            n = min(n, actual_stored)
            c.keys = c._temporal_order(c.keys)
            c.values = c._temporal_order(c.values)
            c.keys = c.keys[..., :-n, :]
            c.values = c.values[..., :-n, :]
            c.offset = actual_stored - n
            c._idx = c.keys.shape[2]
        elif hasattr(c, "trim"):
            c.trim(n)


# ---------------------------------------------------------------------------
# Gated-delta rollback (Qwen3.5 / Qwen3-Coder-Next)
# ---------------------------------------------------------------------------


# Attributes that ``_capturing_gdn_call`` reads off the patched layer.
# Used by ``_find_gdn_class`` as a structural check so the patch only
# attaches to a class that actually exposes the GDN interface — a
# same-named but unrelated class would be skipped at discovery time
# rather than crashing mid-inference on a missing attribute.
_GDN_REQUIRED_ATTRS = (
    # Projections.
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
    # Conv stack.
    "conv1d",
    "conv_kernel_size",
    "conv_dim",
    # State / bias tensors.
    "A_log",
    "dt_bias",
    # Output norm (called as ``self_layer.norm(out, z)``).
    "norm",
    # Head-shape metadata used to reshape the conv output.
    "num_k_heads",
    "num_v_heads",
    "head_k_dim",
    "head_v_dim",
    "key_dim",
)


def _find_gdn_class(model: nn.Module) -> type | None:
    """Locate the ``GatedDeltaNet`` class actually used by *model*.

    Hybrid linear-attention layers in mlx-lm are conventionally named
    ``GatedDeltaNet`` and live in the model's module file. Walking the
    target's submodules and matching by class name avoids hardcoding a
    single source module (e.g. ``qwen3_5``) and transparently covers
    future hybrid models that subclass or redefine the layer in their
    own module. We additionally check for the attributes the capturing
    ``__call__`` reads — a same-named class without those attributes is
    not a GDN layer and patching it would silently corrupt inference.
    Returns the first match, or ``None`` if no GDN-style layer is in
    the model.
    """
    for _name, mod in model.named_modules():
        if type(mod).__name__ != "GatedDeltaNet":
            continue
        if all(hasattr(mod, a) for a in _GDN_REQUIRED_ATTRS):
            return type(mod)
    return None


class _GDNStateCapture:
    """Monkey-patch ``GatedDeltaNet.__call__`` to enable rejection rollback.

    Snapshots ``(q, k, v, a, b, A_log, dt_bias, init_state, mask)`` and
    the conv input per layer call. ``rollback`` re-runs
    ``gated_delta_update`` on the first ``accepted + 1`` tokens to
    reconstruct the post-acceptance state, then writes it back into the
    cache. Holds ``_GDN_PATCH_LOCK`` for the lifetime of the capture so
    only one instance is active at a time.

    The actual ``GatedDeltaNet`` class is discovered from *model* via
    ``_find_gdn_class`` rather than imported from a fixed module — this
    keeps the patch correct when a hybrid model defines its own subclass
    in a different module (e.g. a future Qwen3.5-MoE variant).
    """

    def __init__(self, model: nn.Module) -> None:
        if not _HAS_GDN:
            raise RuntimeError(
                "mlx_lm.models.gated_delta is unavailable; cannot capture "
                "GatedDeltaNet state for DFlash rollback"
            )
        gdn_cls = _find_gdn_class(model)
        if gdn_cls is None:
            raise RuntimeError(
                "Target model declares non-trimmable KV caches but no "
                "``GatedDeltaNet`` submodule was found — DFlash rollback "
                "currently only supports GDN-based hybrid linear-attention "
                "models. Open an olmlx issue with the model's class name."
            )
        self._gdn_cls: type = gdn_cls
        self.conv_data: list[tuple[mx.array, int]] = []
        self._gdn_inputs: list[tuple[Any, ...]] = []
        self._orig_call: Any = None
        self._patched_call: Any = None
        self._closed = False
        _GDN_PATCH_LOCK.acquire()
        try:
            self._patch()
        except Exception:
            # Mark closed *before* releasing so a subsequent ``close()``
            # (e.g. via ``DFlashDecoder.reset()`` or ``__del__``) is a
            # no-op rather than calling ``release()`` on the now-unlocked
            # ``RLock`` — that would raise ``RuntimeError`` in ``reset``
            # and be silently swallowed by ``__del__``.
            self._closed = True
            _GDN_PATCH_LOCK.release()
            raise

    def _patch(self) -> None:
        gdn_cls = self._gdn_cls
        self._orig_call = gdn_cls.__call__
        capture = self
        assert _gd_mod is not None  # guarded by _HAS_GDN

        # WARNING: ``_capturing_gdn_call`` MUST stay bit-for-bit in sync
        # with ``mlx_lm.models.qwen3_5.GatedDeltaNet.__call__`` (which
        # ``mlx_lm.models.qwen3_5_moe`` reuses). If mlx-lm changes the
        # GDN forward — adds an argument to ``gated_delta_update``,
        # changes the conv-state buffer shape, introduces a bias, or
        # reorders the projection calls — inference still uses the
        # patched ``__call__``, but ``_gdn_inputs`` / ``conv_data``
        # captured here will no longer reflect what ran, and
        # ``rollback`` will replay a different operation than the
        # forward pass. The result is silent generation corruption.
        # When bumping mlx-lm: diff this closure against
        # ``GatedDeltaNet.__call__`` and update both halves together.
        def _capturing_gdn_call(self_layer, inputs, mask=None, cache=None):  # type: ignore[no-untyped-def]
            B, S, _ = inputs.shape
            # ``sharding_group`` is only set when the model is sharded for
            # distributed inference; it is initialised to ``None`` on the
            # standard ``GatedDeltaNet`` and may be entirely absent on a
            # third-party GDN subclass — ``getattr`` covers both. The
            # ``sum_gradients`` import is also deferred so non-distributed
            # builds don't pay for it.
            if getattr(self_layer, "sharding_group", None) is not None:
                from mlx.nn.layers.distributed import sum_gradients

                # ``sum_gradients`` returns an ``mx.custom_function``-decorated
                # callable whose pyright-inferred signature picks up the inner
                # ``vjp(x, dx, _)``; the runtime call takes a single tensor.
                inputs = sum_gradients(self_layer.sharding_group)(inputs)  # type: ignore[call-arg]
            qkv = self_layer.in_proj_qkv(inputs)
            z = self_layer.in_proj_z(inputs).reshape(
                B, S, self_layer.num_v_heads, self_layer.head_v_dim
            )
            b, a = self_layer.in_proj_b(inputs), self_layer.in_proj_a(inputs)
            conv_state = (
                cache[0]
                if (cache is not None and cache[0] is not None)
                else mx.zeros(
                    (B, self_layer.conv_kernel_size - 1, self_layer.conv_dim),
                    dtype=inputs.dtype,
                )
            )
            if mask is not None:
                qkv = mx.where(mask[..., None], qkv, 0)
            conv_input = mx.concatenate([conv_state, qkv], axis=1)
            capture.conv_data.append((conv_input, self_layer.conv_kernel_size))
            if cache is not None:
                cache[0] = conv_input[:, -(self_layer.conv_kernel_size - 1) :]
            conv_out = nn.silu(self_layer.conv1d(conv_input))
            q, k, v = (
                t.reshape(B, S, h, d)
                for t, h, d in zip(
                    mx.split(
                        conv_out,
                        [self_layer.key_dim, 2 * self_layer.key_dim],
                        -1,
                    ),
                    [
                        self_layer.num_k_heads,
                        self_layer.num_k_heads,
                        self_layer.num_v_heads,
                    ],
                    [
                        self_layer.head_k_dim,
                        self_layer.head_k_dim,
                        self_layer.head_v_dim,
                    ],
                    strict=True,
                )
            )
            state = cache[1] if cache else None
            inv_scale = k.shape[-1] ** -0.5
            q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
            k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
            # Mirror upstream's ``use_kernel=not self.training`` so any
            # future eval/train switch on the layer instance flows through
            # both the patched forward and the rollback replay below.
            # Capturing it alongside the inputs guarantees the rollback
            # replays with the exact same kernel choice — otherwise a
            # training-mode forward followed by an inference-mode replay
            # would produce mismatched GDN states and corrupt the cache.
            use_kernel = not getattr(self_layer, "training", False)
            capture._gdn_inputs.append(
                (
                    q,
                    k,
                    v,
                    a,
                    b,
                    self_layer.A_log,
                    self_layer.dt_bias,
                    state,
                    mask,
                    use_kernel,
                )
            )
            out, new_state = _gd_mod.gated_delta_update(
                q,
                k,
                v,
                a,
                b,
                self_layer.A_log,
                self_layer.dt_bias,
                state,
                mask,
                use_kernel=use_kernel,
            )
            if cache is not None:
                cache[1] = new_state
            out = self_layer.norm(out, z)
            out = self_layer.out_proj(out.reshape(B, S, -1))
            if getattr(self_layer, "sharding_group", None) is not None:
                out = mx.distributed.all_sum(out, group=self_layer.sharding_group)
            return out

        self._patched_call = _capturing_gdn_call
        gdn_cls.__call__ = _capturing_gdn_call

    def clear(self) -> None:
        self.conv_data.clear()
        self._gdn_inputs.clear()

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._gdn_cls.__call__ is self._patched_call:
                self._gdn_cls.__call__ = self._orig_call
        finally:
            self._closed = True
            self._orig_call = None
            self._patched_call = None
            _GDN_PATCH_LOCK.release()

    def __del__(self) -> None:
        # Belt-and-braces: if the owning ``DFlashDecoder`` is garbage-
        # collected without an explicit ``reset()``/``close()`` (e.g. a
        # generation task is cancelled and the decoder is dropped), the
        # ``_GDN_PATCH_LOCK`` would otherwise stay held until the GC
        # finalises this object. ``close()`` is idempotent.
        try:
            self.close()
        except Exception:
            # Finalisers must never raise — at this point the interpreter
            # may already be shutting down.
            pass

    def rollback(self, cache: list[Any], accepted: int, trim: int) -> None:
        """Restore the GDN state to reflect ``accepted + 1`` accepted tokens."""
        if _gd_mod is None:
            raise RuntimeError(
                "mlx_lm.models.gated_delta is unavailable; cannot perform "
                "DFlash rollback (this should have been caught at prefill)."
            )
        n_non_trimmable = sum(1 for c in cache if not c.is_trimmable())
        # Use ``RuntimeError`` rather than ``assert`` — ``assert`` is stripped
        # under ``python -O`` and a count mismatch must surface as a hard
        # error, not silently corrupt KV state. The structural assumption
        # (every non-trimmable cache is a GatedDeltaNet layer) holds today;
        # any future hybrid SSM that adds another non-trimmable cache type
        # will trip this check and need its own rollback path.
        if n_non_trimmable != len(self._gdn_inputs):
            raise RuntimeError(
                f"non-trimmable cache count ({n_non_trimmable}) != "
                f"captured GDN inputs ({len(self._gdn_inputs)}); "
                "DFlash rollback assumes every non-trimmable cache is a "
                "GatedDeltaNet layer"
            )
        # Ordering invariant: ``_gdn_inputs`` and ``conv_data`` are
        # populated in the order the verification forward pass invokes
        # ``_capturing_gdn_call``, which must match the order in which
        # ``cache`` was built by ``make_prompt_cache``. mlx-lm walks
        # ``model.layers`` in index order in both cases, so the j-th
        # captured input lines up with the j-th non-trimmable cache.
        # If a future hybrid model visits layers out of cache order
        # (tied layers, model-parallel reordering), this loop would
        # silently apply state to the wrong slot — the count check
        # above only catches mismatched cardinality, not ordering.
        j = 0
        for c in cache:
            if c.is_trimmable():
                c.trim(trim)
            else:
                # Replay with the same ``use_kernel`` flag the forward
                # used; the value was captured alongside the inputs in
                # ``_capturing_gdn_call`` so a training-mode forward
                # followed by an inference-mode replay can't drift.
                (
                    q,
                    k,
                    v,
                    a,
                    b,
                    A_log,
                    dt_bias,
                    init_state,
                    mask,
                    use_kernel,
                ) = self._gdn_inputs[j]
                n = accepted + 1
                _, state = _gd_mod.gated_delta_update(
                    q[:, :n],
                    k[:, :n],
                    v[:, :n],
                    a[:, :n],
                    b[:, :n],
                    A_log,
                    dt_bias,
                    init_state,
                    None if mask is None else mask[:, :n],
                    use_kernel=use_kernel,
                )
                # Cache layout: mlx-lm's GDN models build the per-layer
                # cache as ``ArraysCache(size=2)`` with index 0 holding
                # the conv state and index 1 holding the delta state
                # (mirrors ``mlx_lm.models.qwen3_5`` line 148+). The same
                # convention is used by ``_capturing_gdn_call``'s writes
                # above, so the two stay in lockstep. If a future GDN
                # variant rearranges the tuple, both halves of this
                # module need updating together. Going through the
                # public ``__setitem__`` (``c[i] = ...``) instead of
                # ``c.cache[i]`` is forward-compatible if mlx-lm adds
                # validation or lazy semantics to the cache.
                c[1] = state
                conv_input, K = self.conv_data[j]
                c[0] = conv_input[:, accepted + 1 : accepted + K]
                j += 1


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


class DFlashDecoder:
    """Block-diffusion speculative decoder.

    Each ``step()`` builds a masked block ``[pending_token, MASK,
    MASK, ...]`` of length ``block_size + 1``, runs one parallel draft
    forward pass to produce ``block_size`` candidate tokens, then runs
    one verification forward pass through the target. Greedy verification
    accepts the longest matching prefix; the bonus token comes from the
    target. On rejection the target cache (or GDN state for hybrid
    models) is rolled back.

    ``block_size`` is the number of *draft* tokens per step (matches
    ``SpeculativeDecoder``'s ``num_speculative_tokens``). The total
    block length passed through the draft is ``block_size + 1`` because
    the pending token occupies position 0 (sliced off via ``logits_start=1``).
    """

    def __init__(
        self,
        target_model: nn.Module,
        draft_model: DFlashDraftModel,
        draft_config: DraftConfig,
        block_size: int = 4,
    ):
        self._target = target_model
        self._draft = draft_model
        self._config = draft_config
        self._block_size = block_size

        # State (populated by prefill())
        self._target_cache: list[Any] | None = None
        self._draft_cache: list[Any] | None = None
        self._target_can_trim: bool = True
        self._capture: _GDNStateCapture | None = None
        self._hidden: mx.array | None = None
        self._pending_token: int | None = None
        self._prompt_size: int = 0
        # Per-layer hidden state captured by ``_LayerHook``. Owned by the
        # decoder (not the target ``nn.Module``) so captured ``mx.array``s
        # never appear in ``target.parameters()``.
        self._hidden_capture: list[Any] = []
        # Counts the *generated* tokens (matching upstream's ``n``);
        # used to compute draft-cache trim amounts.
        self._n_generated: int = 0

        # Diagnostic counters (reset on prefill, mirrors SpeculativeDecoder)
        self._stats_steps: int = 0
        self._stats_proposed: int = 0
        self._stats_accepted_draft: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        if self._capture is not None:
            self._capture.close()
            self._capture = None
        _unpatch_model(self._target)
        self._draft.unbind()
        self._target_cache = None
        self._draft_cache = None
        self._target_can_trim = True
        self._hidden = None
        self._pending_token = None
        self._prompt_size = 0
        self._hidden_capture = []
        self._n_generated = 0
        self._stats_steps = 0
        self._stats_proposed = 0
        self._stats_accepted_draft = 0

    def stats_summary(self) -> dict[str, Any]:
        steps = self._stats_steps
        proposed = self._stats_proposed
        accepted_draft = self._stats_accepted_draft
        acceptance_rate = accepted_draft / proposed if proposed else 0.0
        avg_tokens_per_step = (accepted_draft + steps) / steps if steps else 0.0
        return {
            "steps": steps,
            "proposed": proposed,
            "accepted_draft": accepted_draft,
            "acceptance_rate": acceptance_rate,
            "avg_tokens_per_step": avg_tokens_per_step,
            "lambda": self._block_size,
        }

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def prefill(self, prompt: mx.array) -> int:
        """Process the prompt through the target, capturing hidden states.

        Returns the first generated token (target greedy argmax).
        """
        self.reset()

        target_layer_ids = list(self._config.target_layer_ids)
        # Build the target cache before patching: ``make_prompt_cache``
        # walks ``model.layers`` to pick a per-layer cache type (sliding
        # vs. full attention) by probing the layer object. Today it uses
        # ``hasattr``, which ``_LayerHook.__getattr__`` proxies through,
        # but a future ``isinstance`` check would silently get the wrong
        # cache type for patched layers. Doing the cache build first
        # decouples cache selection from the patch.
        self._target_cache = make_prompt_cache(self._target)
        self._hidden_capture = [None] * len(target_layer_ids)
        _patch_model(self._target, target_layer_ids, self._hidden_capture)
        self._draft.bind(self._target)

        # Call the draft's own ``make_cache`` directly. ``make_prompt_cache``
        # would defer to it via ``hasattr(model, "make_cache")`` today, but
        # going through the public method keeps the per-layer
        # ``RotatingKVCache`` / ``KVCache`` selection (driven by
        # ``DraftConfig.layer_types``) explicit at the call site rather
        # than relying on mlx-lm's dispatch.
        self._draft_cache = self._draft.make_cache()
        self._target_can_trim = can_trim_prompt_cache(self._target_cache)
        if not self._target_can_trim:
            if not _HAS_GDN:
                raise RuntimeError(
                    "Target model has non-trimmable KV cache (likely "
                    "GatedDeltaNet linear-attention layers) but "
                    "mlx_lm.models.gated_delta is unavailable. Cannot "
                    "perform DFlash rejection rollback."
                )
            self._capture = _GDNStateCapture(self._target)

        target_out = self._target(prompt, cache=self._target_cache)
        logits = _logits(target_out)
        captured = list(self._hidden_capture)
        if any(h is None for h in captured):
            raise RuntimeError(
                "Target forward did not populate all configured "
                "target_layer_ids — check that the layer indices exist on "
                f"this model (got {target_layer_ids})."
            )
        self._hidden = mx.concatenate(captured, axis=-1)
        last_logit = logits[:, -1, :]
        mx.eval(last_logit, self._hidden)

        self._prompt_size = int(prompt.shape[1])
        first_token = int(mx.argmax(last_logit, axis=-1).item())
        self._pending_token = first_token
        self._n_generated = 1
        return first_token

    def step(self) -> tuple[list[int], int]:
        """One block-diffusion speculative step."""
        assert self._target_cache is not None, "Call prefill() before step()"
        assert self._draft_cache is not None, "Call prefill() before step()"
        assert self._pending_token is not None, "Call prefill() before step()"
        assert self._hidden is not None, "Call prefill() before step()"

        pending = self._pending_token
        bs_total = self._block_size + 1  # block length including pending token
        mask_id = int(self._config.mask_token_id)

        # 1. Draft a block in one parallel forward pass.
        block = mx.array([[pending] + [mask_id] * self._block_size])
        draft_logits = self._draft(
            block, self._hidden, self._draft_cache, logits_start=1
        )
        # Invariant (full-attention layers only): the draft cache stores
        # exactly the context tokens corresponding to ``prompt_size +
        # n_generated - 1`` target positions. The draft processes
        # ``S = _hidden.shape[1]`` ctx tokens per step (= previous
        # step's ``num_accepted``), so a full-attention layer's offset
        # advances in lockstep with ``_n_generated``. Sliding-window
        # layers truncate ``x_ctx`` to ``window - 1`` before
        # ``update_and_fetch`` and grow by ``min(S, window-1)`` instead,
        # so the invariant doesn't apply there. We only check the first
        # full-attention layer if one exists; the assertion catches
        # bookkeeping bugs without firing falsely on sliding drafts.
        ft_idx = next(
            (
                i
                for i, t in enumerate(self._config.layer_types)
                if t == "full_attention"
            ),
            None,
        )
        if ft_idx is not None:
            draft_offset = self._draft_cache[ft_idx].offset
            target_offset = self._prompt_size + self._n_generated - 1
            if draft_offset != target_offset:
                raise RuntimeError(
                    f"Draft cache offset ({draft_offset}) at full-attention "
                    f"layer {ft_idx} drifted from expected target offset "
                    f"({target_offset}); DFlash bookkeeping bug."
                )
        draft_tokens_arr = mx.argmax(draft_logits, axis=-1)
        mx.eval(draft_tokens_arr)
        # ``mx.array.tolist()`` is typed as ``list_or_scalar``; for a 1-D
        # array of ints it always returns ``list[int]`` at runtime.
        draft_tokens: list[int] = draft_tokens_arr[0].tolist()  # type: ignore[assignment]

        # 2. Verify with the target in one parallel forward pass.
        if self._capture is not None:
            self._capture.clear()
        # Zero the capture slots before the verification forward. Each
        # hook fires once per ``__call__`` and overwrites its slot, but
        # a hook that *stops* firing mid-generation (e.g. layer skipped
        # by a future model change) would otherwise leave the previous
        # step's tensor in the slot — the ``None``-check below would
        # pass and we would silently feed stale hiddens to the next
        # draft step. ``prefill`` allocates the list fresh; ``step``
        # has to clear it explicitly.
        for i in range(len(self._hidden_capture)):
            self._hidden_capture[i] = None
        verify_input = mx.array([[pending] + draft_tokens])
        target_out = self._target(verify_input, cache=self._target_cache)
        logits = _logits(target_out)
        captured = list(self._hidden_capture)
        # Same guard as ``prefill``: an unfired hook here would surface
        # downstream as a cryptic ``mx.concatenate`` type error rather
        # than an actionable message.
        if any(h is None for h in captured):
            raise RuntimeError(
                "Target verification forward did not populate all "
                f"configured target_layer_ids "
                f"({list(self._config.target_layer_ids)}); a layer hook may "
                "have been removed mid-generation."
            )
        new_hidden = mx.concatenate(captured, axis=-1)
        mx.eval(logits, new_hidden)

        # 3. Greedy verification.
        verification_logits = logits[0]  # (block_size + 1, vocab)
        accepted = verify_draft_greedy(draft_tokens, verification_logits)
        num_accepted = len(accepted)  # 1..block_size+1
        # accepted_drafts is the count BEFORE the bonus position
        # (excludes the target's correction or all-accepted bonus).
        accepted_drafts = num_accepted - 1

        # 4. Roll back caches: remove the unused tail of the verify block.
        trim = bs_total - num_accepted  # 0..block_size
        if trim > 0:
            if self._target_can_trim:
                _trim_recent_cache(self._target_cache, trim)
            else:
                assert self._capture is not None
                self._capture.rollback(self._target_cache, accepted_drafts, trim)

        # 5. Slice hidden state to the accepted prefix; this becomes the
        # new "ctx" length for the next draft call. Length = num_accepted
        # which spans positions [pending_token, accepted_drafts...].
        self._hidden = new_hidden[:, :num_accepted, :]

        # 6. Update state.
        self._pending_token = accepted[-1]
        self._n_generated += num_accepted

        # Diagnostics.
        self._stats_steps += 1
        self._stats_proposed += self._block_size
        self._stats_accepted_draft += accepted_drafts

        return accepted, self._block_size


def _logits(out: Any) -> mx.array:
    """Unwrap mlx-vlm ``LanguageModelOutput`` if needed."""
    return getattr(out, "logits", out)
