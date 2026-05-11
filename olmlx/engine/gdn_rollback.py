"""GatedDeltaNet (GDN) state capture and rollback for speculative decoding.

Hybrid linear-attention models (Qwen3.5, Qwen3.6, Qwen3-Coder-Next)
interleave full-attention layers (trimmable ``KVCache``) with GDN
linear-attention layers (non-trimmable ``ArraysCache``). On speculative
draft rejection, ``trim_prompt_cache`` is a no-op for any ``CacheList``
containing ``ArraysCache`` — the GDN recurrent state can't be inverted.

This module provides a class-level monkey-patch of
``GatedDeltaNet.__call__`` that snapshots the inputs to each
``gated_delta_update`` call. On rollback, it replays the update over
just the accepted-prefix inputs to reconstruct the correct
post-acceptance state, then writes it back into the cache.

Two rollback modes:

- ``GDNStateCapture.rollback_single`` — the forward pass concatenated
  all candidate tokens into one parallel call (one capture per GDN layer).
  Used by DFlash and by the *target* side of classic speculative.
- ``GDNStateCapture.rollback_autoregressive`` — the forward was a loop
  of single-token calls (M captures per GDN layer, where M is the
  number of draft steps). Used by the *draft* side of classic
  speculative when the draft is also a hybrid model.

Both modes write the post-acceptance state back into the cache and
delegate the trimmable layers to ``c.trim(...)``.

A separate "buffer" abstraction holds the captured tensors. One
``GDNStateCapture`` patches one ``GatedDeltaNet`` class and can route
each call to a chosen buffer via ``use_buffer``. Classic speculative
uses two buffers (target + draft) sharing the same patch when both
models use the same GDN class.
"""

from __future__ import annotations

import logging
from threading import Lock
from typing import Any

import mlx.core as mx
import mlx.nn as nn

try:
    import mlx_lm.models.gated_delta as _gd_mod  # type: ignore[import-not-found]

    _HAS_GDN = True
except ImportError:
    _gd_mod = None  # type: ignore[assignment]
    _HAS_GDN = False


logger = logging.getLogger(__name__)


# ``_capturing_gdn_call`` reproduces ``GatedDeltaNet.__call__`` verbatim
# and calls ``gated_delta_update`` with a fixed positional+keyword
# signature. If mlx-lm renames or reorders parameters, the captured
# tensors no longer reflect what ran and rollback silently replays the
# wrong operation. These are the parameter names we depend on; any
# mismatch raises at ``_patch()`` time rather than corrupting state.
#
# **Sync point**: this contract was last verified against mlx-lm 0.31.2
# (``mlx_lm.models.qwen3_5.GatedDeltaNet.__call__`` and
# ``mlx_lm.models.gated_delta.gated_delta_update``). When bumping
# mlx-lm, diff both functions against the current copies and update
# this list, ``_capturing_gdn_call``, and ``_GDN_REQUIRED_ATTRS``
# together.
_GATED_DELTA_UPDATE_EXPECTED_PARAMS: tuple[str, ...] = (
    "q",
    "k",
    "v",
    "a",
    "b",
    "A_log",
    "dt_bias",
    "state",
    "mask",
    "use_kernel",
)


# Attributes that ``_capturing_gdn_call`` reads off the patched layer.
# Used by ``_find_gdn_class`` as a structural check so the patch only
# attaches to a class that actually exposes the GDN interface — a
# same-named but unrelated class would be skipped at discovery time
# rather than crashing mid-inference on a missing attribute.
_GDN_REQUIRED_ATTRS = (
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
    "conv1d",
    "conv_kernel_size",
    "conv_dim",
    "A_log",
    "dt_bias",
    "norm",
    "num_k_heads",
    "num_v_heads",
    "head_k_dim",
    "head_v_dim",
    "key_dim",
)


# Module-level lock guarding the GatedDeltaNet monkey-patch. Prevents
# two ``GDNStateCapture`` instances from racing on the class-level
# ``__call__`` attribute. ``threading.Lock`` (not ``RLock``) because the
# lock can be released by a different thread than the one that
# acquired it — specifically, ``__del__`` runs on whichever thread
# drops the last reference (often the asyncio event-loop thread, while
# ``acquire`` happened on a worker thread via ``asyncio.to_thread``).
# ``RLock.release`` raises on a non-owner thread; ``Lock.release`` does
# not. Recursive acquisition is unused.
_GDN_PATCH_LOCK = Lock()


def get_model_layers(model: nn.Module) -> list[Any]:
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


def _order_matches(captured: list[Any], expected: list[Any]) -> bool:
    """Identity-based equality for two ordered module lists.

    Replaces ``captured == expected``. The natural Python list ``==``
    falls through to elementwise ``__eq__`` on the contained ``nn.Module``
    instances; mlx's ``Module.__eq__`` returns an ``mx.array`` (broadcast
    over its array attributes) and Python then tries to coerce that
    array to a scalar via ``bool()``, which raises
    ``ValueError: [convert] Only length-1 arrays can be converted to
    Python scalars`` for any module holding a multi-element tensor.
    Identity comparison sidesteps the overload entirely and matches the
    actual invariant we want (the forward visited the same module
    instances in the same order).
    """
    if len(captured) != len(expected):
        return False
    return all(a is b for a, b in zip(captured, expected))


def validate_gated_delta_update_signature() -> None:
    """Raise if mlx-lm's ``gated_delta_update`` no longer accepts the
    parameter names ``_capturing_gdn_call`` and rollback depend on.

    Without this probe, a parameter rename in mlx-lm would let
    inference continue with the captured tensors stored under stale
    keys; rollback would then call ``gated_delta_update`` with keyword
    arguments the function doesn't accept (or, worse, accept silently
    if the rename only shuffled positions). Fail loudly at patch-install
    time instead.
    """
    import inspect

    assert _gd_mod is not None  # guarded by ``_HAS_GDN`` at call sites
    try:
        sig = inspect.signature(_gd_mod.gated_delta_update)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "Cannot introspect ``gated_delta_update`` signature; the "
            "GDN capture/rollback path can't validate upstream compatibility."
        ) from exc
    actual = set(sig.parameters)
    expected = set(_GATED_DELTA_UPDATE_EXPECTED_PARAMS)
    missing = expected - actual
    if missing:
        raise RuntimeError(
            "mlx-lm's ``gated_delta_update`` is missing parameters that "
            "the GDN capture/rollback path depends on: "
            f"{sorted(missing)}. The pinned mlx-lm contract changed; "
            "update ``_capturing_gdn_call`` / rollback methods and "
            "``_GATED_DELTA_UPDATE_EXPECTED_PARAMS`` together."
        )


def find_gdn_class(model: nn.Module) -> type | None:
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


def collect_gdn_modules(model: nn.Module, gdn_cls: type) -> list[Any]:
    """Collect *model*'s GDN module instances in forward-pass order.

    Walks ``get_model_layers(model)`` in natural index order — NOT
    ``model.named_modules()``, which uses mlx's LIFO stack traversal and
    visits list children in reverse, producing a sequence that
    contradicts the forward pass. Recurses into each layer to find its
    GDN submodule(s).

    Raises ``RuntimeError`` if any layer contains more than one GDN
    submodule (every hybrid family we know about ships at most one
    per layer; multi-GDN layouts would expose the same LIFO-traversal
    bug at sub-layer granularity).

    Raises ``RuntimeError`` if no GDN modules are reachable via
    ``get_model_layers`` while ``model.named_modules()`` does find some
    — that means GDN modules live outside the layers list and rollback
    can't address them.
    """
    out: list[Any] = []
    ordered_layers = get_model_layers(model)
    for layer in ordered_layers:
        per_layer_gdns = [
            mod for _name, mod in layer.named_modules() if isinstance(mod, gdn_cls)
        ]
        if len(per_layer_gdns) > 1:
            raise RuntimeError(
                f"Layer {type(layer).__name__} contains "
                f"{len(per_layer_gdns)} GatedDeltaNet submodules; "
                "GDN rollback currently assumes at most one GDN per "
                "layer because ``layer.named_modules()`` uses mlx's "
                "LIFO stack traversal — multiple GDNs would come back "
                "in the reverse of definition order and be mis-aligned "
                "with the forward pass. File an olmlx issue with the "
                "model class name."
            )
        out.extend(per_layer_gdns)
    if not out:
        orphaned = [
            mod for _name, mod in model.named_modules() if isinstance(mod, gdn_cls)
        ]
        if orphaned:
            raise RuntimeError(
                f"Found {len(orphaned)} GatedDeltaNet module(s) in the "
                "model but none are reachable via ``get_model_layers``. "
                "Layer-hook installation uses the same helper, so "
                "rollback cannot work for this configuration. File an "
                "olmlx issue with the model class name."
            )
    return out


class GDNBuffer:
    """Per-model storage of GDN inputs for rollback.

    One buffer per (model, logical-scope) pair — e.g. one buffer for the
    target's verify forward and another for the draft's autoregressive
    loop in classic speculative, even when both share the same
    ``GDNStateCapture`` because they use the same GDN class.

    ``expected_modules`` is the list of GDN module instances belonging
    to this model (in forward-pass order). Captures land in
    ``gdn_inputs``/``conv_data``/``captured_modules`` in the order
    ``_capturing_gdn_call`` is invoked — matching the cache layout
    because mlx-lm walks layers in index order.
    """

    def __init__(self, expected_modules: list[Any]) -> None:
        self.expected_modules: list[Any] = expected_modules
        self.conv_data: list[tuple[mx.array, int]] = []
        self.gdn_inputs: list[tuple[Any, ...]] = []
        # Parallel list of layer instances ``_capturing_gdn_call`` saw
        # at each step; used by rollback to verify ordering.
        self.captured_modules: list[Any] = []

    @property
    def num_gdn_layers(self) -> int:
        return len(self.expected_modules)

    def clear(self) -> None:
        self.conv_data.clear()
        self.gdn_inputs.clear()
        self.captured_modules.clear()


class GDNStateCapture:
    """Class-level patch of ``GatedDeltaNet.__call__`` for rollback support.

    On construction, locates the GDN class actually used by *model*,
    validates the mlx-lm signature, and installs a closure on
    ``GatedDeltaNet.__call__`` that:

    1. Reproduces the original forward bit-for-bit (so inference is
       unaffected by the patch when no rollback is needed).
    2. Appends the inputs to ``gated_delta_update`` to the *active*
       buffer (set via ``use_buffer``).

    Hold ``_GDN_PATCH_LOCK`` for the lifetime of the capture so two
    instances cannot race on the class-level attribute. A second
    ``GDNStateCapture`` on the same class blocks until the first is
    closed; design accordingly (e.g. one capture instance shared
    between target and draft when both use the same GDN class).
    """

    def __init__(self, gdn_cls: type) -> None:
        if not _HAS_GDN:
            raise RuntimeError(
                "mlx_lm.models.gated_delta is unavailable; cannot capture "
                "GatedDeltaNet state for rollback"
            )
        validate_gated_delta_update_signature()
        self._gdn_cls: type = gdn_cls
        self._active_buffer: GDNBuffer | None = None
        self._orig_call: Any = None
        self._patched_call: Any = None
        self._closed = False
        _GDN_PATCH_LOCK.acquire()
        try:
            self._patch()
        except Exception:
            # Mark closed *before* releasing so a subsequent ``close()``
            # is a no-op rather than calling ``release()`` on the now-
            # unlocked lock.
            self._closed = True
            _GDN_PATCH_LOCK.release()
            raise

    @classmethod
    def for_model(cls, model: nn.Module) -> tuple["GDNStateCapture", GDNBuffer]:
        """Convenience: locate the GDN class in *model*, create a capture
        and a buffer pre-populated with *model*'s expected modules.

        Use when only one model needs rollback (DFlash's target).
        For classic speculative with hybrid target+draft sharing the
        same GDN class, construct one ``GDNStateCapture(gdn_cls)``
        directly and call ``create_buffer(model)`` per model.

        Raises ``RuntimeError`` if *model* has no ``GatedDeltaNet`` submodule.
        """
        gdn_cls = find_gdn_class(model)
        if gdn_cls is None:
            raise RuntimeError(
                "Model declares non-trimmable KV caches but no "
                "``GatedDeltaNet`` submodule was found — GDN rollback "
                "currently only supports GDN-based hybrid linear-attention "
                "models. Open an olmlx issue with the model's class name."
            )
        capture = cls(gdn_cls)
        try:
            buffer = capture.create_buffer(model)
        except Exception:
            capture.close()
            raise
        return capture, buffer

    def create_buffer(self, model: nn.Module) -> GDNBuffer:
        """Create a fresh buffer pre-populated with *model*'s GDN modules.

        Each model gets its own buffer because the expected-module list
        is model-specific (different instances for target vs draft, even
        when they share the GDN class).
        """
        return GDNBuffer(collect_gdn_modules(model, self._gdn_cls))

    # ------------------------------------------------------------------
    # Buffer routing
    # ------------------------------------------------------------------

    def use_buffer(self, buffer: GDNBuffer | None) -> None:
        """Set the active buffer that subsequent forward passes write to.

        Pass ``None`` to suppress capture (forwards still run normally
        but nothing is recorded). Useful during prefill where no
        rollback is needed.
        """
        self._active_buffer = buffer

    @property
    def gdn_cls(self) -> type:
        return self._gdn_cls

    # ------------------------------------------------------------------
    # Patch lifecycle
    # ------------------------------------------------------------------

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
        # patched ``__call__``, but ``gdn_inputs`` / ``conv_data``
        # captured here will no longer reflect what ran, and rollback
        # will replay a different operation than the forward pass. The
        # result is silent generation corruption. When bumping mlx-lm:
        # diff this closure against ``GatedDeltaNet.__call__`` and
        # update both halves together.
        def _capturing_gdn_call(self_layer, inputs, mask=None, cache=None):  # type: ignore[no-untyped-def]
            B, S, _ = inputs.shape
            if getattr(self_layer, "sharding_group", None) is not None:
                from mlx.nn.layers.distributed import sum_gradients

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
            buf = capture._active_buffer
            if buf is not None:
                buf.conv_data.append((conv_input, self_layer.conv_kernel_size))
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
            use_kernel = not getattr(self_layer, "training", False)
            if buf is not None:
                buf.captured_modules.append(self_layer)
                buf.gdn_inputs.append(
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
            self._active_buffer = None
            _GDN_PATCH_LOCK.release()

    def __del__(self) -> None:
        # Belt-and-braces: if the owner is garbage-collected without an
        # explicit close (e.g. a generation task is cancelled and the
        # decoder is dropped), the lock would otherwise stay held until
        # GC finalises this object. ``close()`` is idempotent.
        try:
            self.close()
        except Exception:
            # Finalisers must never raise.
            pass

    # ------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------

    def rollback_single(
        self,
        buffer: GDNBuffer,
        cache: list[Any],
        accepted: int,
        trim: int,
    ) -> None:
        """Restore cache state after a single parallel forward pass.

        The forward fed N tokens in one call and we keep the first
        ``accepted + 1`` of them. For trimmable layers, ``c.trim(trim)``
        does the job. For ArraysCache (GDN) layers, replay
        ``gated_delta_update`` over the first ``accepted + 1`` captured
        tokens and write the resulting state + conv state back.

        Args:
            buffer: the buffer that captured the single forward.
            cache: the model's full cache list.
            accepted: number of *additional* tokens to keep beyond the
                first (mirrors DFlash's API; pass ``num_accepted - 1``
                from classic speculative).
            trim: amount to pass to ``c.trim()`` for trimmable layers.
        """
        self._check_buffer_alignment(buffer, single=True)
        if _gd_mod is None:
            raise RuntimeError(
                "mlx_lm.models.gated_delta is unavailable; cannot perform "
                "GDN rollback (should have been caught at patch install)."
            )
        j = 0
        for c in cache:
            if c.is_trimmable():
                c.trim(trim)
            else:
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
                ) = buffer.gdn_inputs[j]
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
                # the conv state and index 1 holding the delta state.
                c[1] = state
                conv_input, K = buffer.conv_data[j]
                c[0] = conv_input[:, accepted + 1 : accepted + K]
                j += 1

    def rollback_autoregressive(
        self,
        buffer: GDNBuffer,
        cache: list[Any],
        num_steps: int,
        num_keep_steps: int,
        trim: int,
    ) -> None:
        """Restore cache state after an autoregressive loop of single-token forwards.

        The forward ran ``num_steps`` single-token calls, each appending
        one capture per GDN layer (so the buffer holds
        ``num_steps * num_gdn_layers`` entries). We keep the first
        ``num_keep_steps`` of those calls.

        For trimmable layers, ``c.trim(trim)`` works as usual. For
        ArraysCache layers, the post-step-``num_keep_steps`` state can
        be reconstructed by concatenating the first ``num_keep_steps``
        captures' inputs along the sequence dim and replaying
        ``gated_delta_update`` once with the FIRST capture's pre-state
        as ``init_state``.

        Args:
            buffer: the buffer that captured all ``num_steps`` calls.
            cache: the model's full cache list.
            num_steps: how many autoregressive calls were captured.
            num_keep_steps: how many of those to keep (1 ≤ keep ≤ steps).
            trim: amount to pass to ``c.trim()`` for trimmable layers.
        """
        if num_keep_steps < 1 or num_keep_steps > num_steps:
            raise ValueError(
                f"num_keep_steps ({num_keep_steps}) must be in [1, {num_steps}]"
            )
        if num_keep_steps == num_steps:
            # No rollback needed; trim is also 0 in this case. Buffer
            # already reflects the final state.
            return
        self._check_buffer_alignment(buffer, single=False, num_steps=num_steps)
        if _gd_mod is None:
            raise RuntimeError(
                "mlx_lm.models.gated_delta is unavailable; cannot perform "
                "GDN rollback (should have been caught at patch install)."
            )
        N = buffer.num_gdn_layers
        j_gdn = 0  # index into the per-layer GDN sequence (0 .. N-1)
        for c in cache:
            if c.is_trimmable():
                c.trim(trim)
                continue
            # Gather the first ``num_keep_steps`` captures for this GDN
            # layer. Each capture is one autoregressive step with S=1.
            captures = [
                buffer.gdn_inputs[step * N + j_gdn] for step in range(num_keep_steps)
            ]
            # Concatenate per-token tensors along the sequence dim. The
            # constant tensors (A_log, dt_bias, use_kernel) come from
            # any capture — pick the first.
            q_cat = mx.concatenate([cap[0] for cap in captures], axis=1)
            k_cat = mx.concatenate([cap[1] for cap in captures], axis=1)
            v_cat = mx.concatenate([cap[2] for cap in captures], axis=1)
            a_cat = mx.concatenate([cap[3] for cap in captures], axis=1)
            b_cat = mx.concatenate([cap[4] for cap in captures], axis=1)
            A_log = captures[0][5]
            dt_bias = captures[0][6]
            init_state = captures[0][7]  # state BEFORE the loop started
            masks = [cap[8] for cap in captures]
            if all(m is None for m in masks):
                mask_cat = None
            else:
                # If any mask is present, all must be (or we can't
                # concatenate). Fail loudly rather than silently masking
                # the wrong positions — autoregressive draft generation
                # generally passes mask=None for single-token inference,
                # so this branch shouldn't fire in practice.
                if any(m is None for m in masks):
                    raise RuntimeError(
                        "GDN autoregressive rollback: some captures had "
                        "mask=None and others had a mask. Mixed-mask "
                        "concatenation is ambiguous; please report this "
                        "model as an olmlx bug."
                    )
                mask_cat = mx.concatenate(masks, axis=1)
            use_kernel = captures[0][9]
            _, state = _gd_mod.gated_delta_update(
                q_cat,
                k_cat,
                v_cat,
                a_cat,
                b_cat,
                A_log,
                dt_bias,
                init_state,
                mask_cat,
                use_kernel=use_kernel,
            )
            c[1] = state
            # Conv state after step ``num_keep_steps``: last K-1 elements
            # of the conv_input captured at that step. The conv state
            # captured INSIDE step k already accumulates step (k-1)'s
            # contribution, so step num_keep_steps's conv_input ends
            # with the post-step-num_keep_steps conv state.
            conv_input, K = buffer.conv_data[(num_keep_steps - 1) * N + j_gdn]
            c[0] = conv_input[:, -(K - 1) :]
            j_gdn += 1

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_buffer_alignment(
        self, buffer: GDNBuffer, single: bool, num_steps: int = 1
    ) -> None:
        n_layers = buffer.num_gdn_layers
        expected = n_layers * num_steps
        if len(buffer.gdn_inputs) != expected:
            raise RuntimeError(
                f"GDN capture buffer size mismatch: expected {expected} "
                f"entries ({n_layers} layers × {num_steps} "
                f"forward call{'s' if num_steps != 1 else ''}), got "
                f"{len(buffer.gdn_inputs)}. Either capture missed some "
                "GDN calls or extra calls were made outside the "
                "expected scope."
            )
        if single:
            # Identity ordering check identical to DFlash's existing
            # invariant: forward must visit GDN modules in the same
            # order they appear in ``get_model_layers(model)``.
            if not _order_matches(buffer.captured_modules, buffer.expected_modules):
                self._raise_ordering_error(buffer)
        else:
            # For autoregressive: each step should visit the same
            # modules in the same order. Check the first step's
            # ordering and that subsequent steps match it.
            first_step = buffer.captured_modules[:n_layers]
            if not _order_matches(first_step, buffer.expected_modules):
                self._raise_ordering_error(buffer)
            for step in range(1, num_steps):
                this_step = buffer.captured_modules[
                    step * n_layers : (step + 1) * n_layers
                ]
                if not all(a is b for a, b in zip(this_step, first_step)):
                    raise RuntimeError(
                        f"GDN autoregressive rollback: step {step} visited "
                        "GDN layers in a different order than step 0. "
                        "Cache slot assignment would be wrong. Please "
                        "report as an olmlx bug."
                    )

    def _raise_ordering_error(self, buffer: GDNBuffer) -> None:
        captured_types = [type(m).__name__ for m in buffer.captured_modules]
        expected_types = [type(m).__name__ for m in buffer.expected_modules]
        raise RuntimeError(
            "GDN rollback ordering invariant violated: forward pass "
            "visited GDN layers in a different order than "
            "``get_model_layers(model)``. Captures are positional and "
            "would be applied to the wrong cache slots. This is an "
            "olmlx bug — please report at "
            "https://github.com/motsognirr/olmlx/issues with the model "
            f"name. Diagnostic: captured order={captured_types}, "
            f"expected order={expected_types}."
        )
