"""Latent MoE forward for Edge0 packs: SSD-streamed experts + shared D/U.

For one token ``x`` in one layer, with router picks ``e_k`` (weights ``w_k``)
and ``g_k`` the group of expert ``e_k``::

    xl      = (D @ x).reshape(G, latent) * (1 / s)      # shared, per group
    h_k     = silu(Gate_{e_k} @ xl[g_k]) * (Up_{e_k} @ xl[g_k])
    acc[g]  = sum_{k: g_k = g} w_k * (Down_{e_k} @ h_k)
    routed  = U @ acc.reshape(G * latent)               # U is linear: pool first

Experts are never loaded as parameters. Their records are read on demand
from the pack's ``experts.bin`` and cached **as raw bytes** (uint8 arrays) —
the ggml ``Q3_K`` / ``IQ4_NL`` blocks, ~1.6 MB per expert, with no parsing.
The Metal kernels multiply straight from those blocks; the dense-decode path
is the CPU reference.

``D`` and ``U`` are stored in the pack as ``Q8_0`` / ``Q4_0`` rows, which map
*exactly* onto MLX's affine quantization at group size 32, so they stay
packed and go through ``mx.quantized_matmul``.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from olmlx.engine.edge0 import ggml_quants, latent_kernels
from olmlx.engine.edge0.affine import blocks_to_affine
from olmlx.engine.flash._ssd_base import LayerLruCache, full_pread
from olmlx.engine.flash.moe_weight_store import ExpertCacheStats
from olmlx.utils import tracing as _tracing

if TYPE_CHECKING:
    from olmlx.engine.edge0.pack import Edge0Pack, LatentGeometry

logger = logging.getLogger(__name__)

#: Group size of the pack's D/U blocks (and of MLX's matching affine layout).
PROJECTION_GROUP_SIZE = 32

#: Experts handled per batch within one layer call. A long prompt routes to
#: nearly every expert of a layer, so a call streams them through in batches
#: instead of holding them all: each expert is still read once per call.
#: Fused kernels keep a batch as packed records (~1.6 MB each).
DEFAULT_MAX_KERNEL_EXPERTS = 64
#: The dense reference path decodes each expert to ~14 MB of float32 plus
#: several times that in intermediates.
DEFAULT_MAX_DECODE_EXPERTS = 24


# ---------------------------------------------------------------------------
# ggml block decoding as MLX ops (mirrors ggml_quants, on the device)
# ---------------------------------------------------------------------------


def _half_scale(two_bytes: mx.array) -> mx.array:
    """``[..., 2]`` uint8 (little-endian float16) -> ``[...]`` float32."""
    return two_bytes.view(mx.float16).squeeze(-1).astype(mx.float32)


def decode_q3_k(blocks: mx.array) -> mx.array:
    """Decode ``[..., n_blocks, 110]`` uint8 ``Q3_K`` -> ``[..., n_blocks*256]`` float32."""
    lead = blocks.shape[:-2]
    n_blocks = blocks.shape[-2]
    hmask = blocks[..., 0:32]
    qs = blocks[..., 32:96].reshape(*lead, n_blocks, 2, 1, 32)
    shifts = mx.array([0, 2, 4, 6], dtype=mx.uint8).reshape(4, 1)
    low = mx.bitwise_and(mx.right_shift(qs, shifts), 3)
    planes = mx.arange(8, dtype=mx.uint8).reshape(8, 1)
    high = mx.bitwise_and(mx.right_shift(hmask[..., None, :], planes), 1)
    high = high.reshape(*lead, n_blocks, 2, 4, 32)
    codes = (low + high * 4).astype(mx.float32) - 4.0

    packed = blocks[..., 96:104]
    low4 = mx.concatenate(
        [mx.bitwise_and(packed, 15), mx.right_shift(packed, 4)], axis=-1
    )
    pair_shifts = mx.array([0, 2, 4, 6], dtype=mx.uint8).reshape(4, 1)
    high2 = mx.bitwise_and(
        mx.right_shift(blocks[..., 104:108][..., None, :], pair_shifts), 3
    ).reshape(*lead, n_blocks, 16)
    six_bit = (low4 + high2 * 16).astype(mx.float32) - 32.0
    scales = six_bit * _half_scale(blocks[..., 108:110])[..., None]

    values = codes.reshape(*lead, n_blocks, 16, 16) * scales[..., None]
    return values.reshape(*lead, n_blocks * 256)


def decode_iq4_nl(blocks: mx.array) -> mx.array:
    """Decode ``[..., n_blocks, 18]`` uint8 ``IQ4_NL`` -> ``[..., n_blocks*32]`` float32."""
    lead = blocks.shape[:-2]
    n_blocks = blocks.shape[-2]
    qs = blocks[..., 2:18]
    nibbles = mx.concatenate(
        [mx.bitwise_and(qs, 15), mx.right_shift(qs, 4)], axis=-1
    ).astype(mx.uint32)
    codebook = mx.array(ggml_quants.KVALUES_IQ4NL, dtype=mx.float32)
    values = codebook[nibbles] * _half_scale(blocks[..., 0:2])[..., None]
    return values.reshape(*lead, n_blocks * 32)


def decode_records(
    raw: mx.array, geometry: LatentGeometry
) -> tuple[mx.array, mx.array, mx.array]:
    """Decode ``[n, record_bytes]`` expert records to dense float32 weights.

    Returns ``(gate [n, ff, latent], up [n, ff, latent], down [n, latent, ff])``.
    """
    if (geometry.gu_type, geometry.down_type) != (
        ggml_quants.GGML_Q3_K,
        ggml_quants.GGML_IQ4_NL,
    ):
        raise NotImplementedError(
            "only Q3_K gate/up + IQ4_NL down expert records are supported, got "
            f"ggml types {geometry.gu_type}/{geometry.down_type}"
        )
    n = raw.shape[0]
    g = geometry
    gate_up = decode_q3_k(
        raw[:, : g.down_offset].reshape(n, 2, g.ff, g.latent // 256, 110)
    )
    down = decode_iq4_nl(raw[:, g.down_offset :].reshape(n, g.latent, g.ff // 32, 18))
    return gate_up[:, 0], gate_up[:, 1], down


# ---------------------------------------------------------------------------
# D / U projections
# ---------------------------------------------------------------------------


def pack_projections(pack: Edge0Pack, layer: int) -> dict[str, Any]:
    """Build one layer's D/U as MLX affine-quantized matrices.

    ``down``: ``[G*latent, hidden]`` (rows grouped by projection group).
    ``up``: ``[hidden, G*latent]`` — the per-group ``U_g`` laid side by side,
    so one matmul applies every group and sums them.
    """
    g = pack.geometry
    bits = 8 if pack.latent_dense_type == ggml_quants.GGML_Q8_0 else 4

    codes, scales, biases = blocks_to_affine(
        np.asarray(pack.down_projection_raw(layer)),
        pack.latent_dense_type,
        g.n_groups * g.latent,
    )
    out: dict[str, Any] = {
        "bits": bits,
        "down.weight": mx.array(codes.reshape(g.n_groups * g.latent, -1).view("<u4")),
        "down.scales": mx.array(scales),
        "down.biases": mx.array(biases),
    }

    codes, scales, biases = blocks_to_affine(
        np.asarray(pack.up_projection_raw(layer)),
        pack.latent_dense_type,
        g.n_groups * g.hidden,
    )

    def side_by_side(values: np.ndarray) -> np.ndarray:
        # rows are (group, hidden); make each output row hold its G groups.
        shaped = values.reshape(g.n_groups, g.hidden, *values.shape[1:])
        return np.ascontiguousarray(np.swapaxes(shaped, 0, 1)).reshape(g.hidden, -1)

    out["up.weight"] = mx.array(side_by_side(codes).view("<u4"))
    out["up.scales"] = mx.array(side_by_side(scales))
    out["up.biases"] = mx.array(side_by_side(biases))
    return out


# ---------------------------------------------------------------------------
# Expert record store
# ---------------------------------------------------------------------------


class LatentExpertStore:
    """Reads expert records from ``experts.bin`` with an LRU of raw records.

    Cached records are MLX arrays on purpose. Keeping them as host ``bytes``
    (to stay clear of Metal's cap on MLX memory) was tried and is much worse:
    under memory pressure macOS compresses and swaps ordinary process memory
    but not MLX's wired buffers, so decode fell from ~6.8 to 4.4 tok/s at the
    same budget and to 0.5 tok/s at twice the budget.
    """

    def __init__(
        self,
        pack: Edge0Pack,
        *,
        num_io_threads: int = 16,
        cache_budget_experts: int = 48,
    ):
        self.geometry = pack.geometry
        self._experts_per_layer = pack.geometry.n_experts
        self._record_bytes = pack.geometry.record_bytes
        self._cache: LayerLruCache[int, mx.array] = LayerLruCache(
            max_per_layer=cache_budget_experts
        )
        self.stats = ExpertCacheStats()
        # Executor first: if opening the file fails there is no descriptor to
        # leak, and ``__del__`` always finds both attributes.
        self._fd: int | None = None
        self._executor = ThreadPoolExecutor(max_workers=num_io_threads)
        self._fd = os.open(pack.experts_path, os.O_RDONLY)

    def _read(self, layer: int, expert: int) -> mx.array:
        if self._fd is None:
            raise RuntimeError("LatentExpertStore is closed")
        offset = (layer * self._experts_per_layer + expert) * self._record_bytes
        raw = full_pread(self._fd, self._record_bytes, offset)
        # mx.array copies host data, so the array never aliases ``raw``.
        return mx.array(np.frombuffer(raw, dtype=np.uint8))

    def load(self, layer: int, experts: list[int]) -> mx.array:
        """Return ``[len(experts), record_bytes]`` uint8, in request order."""
        return mx.stack(self.load_records(layer, experts))

    def load_records(self, layer: int, experts: list[int]) -> list[mx.array]:
        """Return each expert's ``[record_bytes]`` uint8 record, in request order."""
        if not experts:
            raise ValueError("experts must not be empty")
        with _tracing.span("flash.weight_load", layer_idx=layer, experts=len(experts)):
            cached = self._cache.get_batch(layer, experts)
            missing = [e for e in experts if e not in cached]
            failures = 0
            first_exc: Exception | None = None
            try:
                futures = {
                    e: self._executor.submit(self._read, layer, e) for e in missing
                }
                for expert, future in futures.items():
                    try:
                        record = future.result()
                    except Exception as exc:  # noqa: BLE001 — re-raised below
                        failures += 1
                        first_exc = first_exc or exc
                        continue
                    cached[expert] = record
                    self._cache.put(layer, expert, record)
            finally:
                self.stats.record(len(experts) - len(missing), len(missing), failures)
            if first_exc is not None:
                raise first_exc
            return [cached[e] for e in experts]

    def close(self) -> None:
        self._executor.shutdown(wait=True)
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def __del__(self) -> None:
        try:
            self._executor.shutdown(wait=False)
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None
        except Exception:  # noqa: BLE001 — interpreter teardown
            pass

    def __enter__(self) -> "LatentExpertStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------


class LatentMoE(nn.Module):
    """Routed-expert dispatch for one latent MoE layer.

    Drop-in for :class:`olmlx.engine.flash.flash_moe.FlashMoE`: called with the
    router's ``(x, inds, scores)`` and returns the weighted sum of the routed
    experts (the shared expert is the caller's).
    """

    def __init__(
        self,
        *,
        layer_idx: int,
        geometry: LatentGeometry,
        labels: np.ndarray,
        tensors: dict[str, Any],
        store: LatentExpertStore,
        max_decode_experts: int = DEFAULT_MAX_DECODE_EXPERTS,
        max_kernel_experts: int = DEFAULT_MAX_KERNEL_EXPERTS,
        use_kernels: bool = True,
        separate_record_limit: int = latent_kernels.MAX_SEPARATE_RECORDS,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self._kernels_allowed = use_kernels
        self._use_kernels = False
        # Underscore attributes: fixed tables, not trainable parameters. They
        # are built from host data, so they are materialized leaves that any
        # thread may read.
        self._geometry = geometry
        self._store = store
        self._max_decode_experts = max_decode_experts
        self._max_kernel_experts = max_kernel_experts
        self._separate_record_limit = min(
            separate_record_limit, latent_kernels.MAX_SEPARATE_RECORDS
        )
        self._labels_host = np.asarray(labels, dtype=np.int64)
        self._bits = int(tensors["bits"])
        self._down = (
            tensors["down.weight"],
            tensors["down.scales"],
            tensors["down.biases"],
        )
        self._up = (tensors["up.weight"], tensors["up.scales"], tensors["up.biases"])
        self._inverse_scale = tensors["inverse_scale"].astype(mx.float32)
        mx.eval(self._down, self._up, self._inverse_scale)

    def _project(self, x: mx.array, projection: tuple) -> mx.array:
        weight, scales, biases = projection
        return mx.quantized_matmul(
            x,
            weight,
            scales,
            biases,
            transpose=True,
            group_size=PROJECTION_GROUP_SIZE,
            bits=self._bits,
        )

    def _route_outputs(
        self, experts: list[int], x: mx.array, local: mx.array, scores: mx.array
    ) -> mx.array:
        """Score-weighted latent output ``[routes, latent]`` of each route."""
        g = self._geometry
        records = self._store.load_records(self.layer_idx, experts)
        if self._use_kernels:
            # A decode step routes to a handful of experts: hand each record
            # to the kernel as its own input rather than copying them into
            # one stacked buffer per layer per token.
            separate = len(records) <= self._separate_record_limit
            return latent_kernels.routed_latent(
                records if separate else mx.stack(records), x, local, scores, g
            )
        gate, up, down = decode_records(mx.stack(records), g)
        x = x[:, None, :]
        gate_out = mx.gather_mm(  # pyright: ignore[reportCallIssue]
            x, gate.swapaxes(-1, -2), rhs_indices=local
        )
        up_out = mx.gather_mm(  # pyright: ignore[reportCallIssue]
            x, up.swapaxes(-1, -2), rhs_indices=local
        )
        hidden = nn.silu(gate_out) * up_out
        out = mx.gather_mm(  # pyright: ignore[reportCallIssue]
            hidden, down.swapaxes(-1, -2), rhs_indices=local
        )
        return out.squeeze(-2) * scores[:, None]

    def __call__(self, x: mx.array, inds: mx.array, scores: mx.array) -> mx.array:
        g = self._geometry
        lead = x.shape[:-1]
        top_k = inds.shape[-1]
        inds = inds.reshape(-1, top_k)
        scores = scores.reshape(-1).astype(mx.float32)
        # One host sync per layer: the expert read list has to exist on the
        # host. Same pattern (and same cost) as FlashMoE.
        mx.eval(inds)
        route_expert = np.array(inds, copy=False).astype(np.int64).reshape(-1)
        n = route_expert.size // top_k
        self._use_kernels = self._kernels_allowed and latent_kernels.kernels_supported(
            g
        )
        cap = (
            self._max_kernel_experts if self._use_kernels else self._max_decode_experts
        )

        xl = self._project(x.reshape(-1, g.hidden).astype(mx.float32), self._down)
        xl = (xl.reshape(n, g.n_groups, g.latent) * self._inverse_scale).reshape(
            n * g.n_groups, g.latent
        )

        # Work per expert batch, not per token: sort the routes by expert so
        # each batch of experts is read once and serves every route that
        # picked it, however many tokens the call carries.
        experts = np.unique(route_expert)
        if experts.size <= cap:
            batches = [(experts, np.arange(route_expert.size))]
        else:
            order = np.argsort(route_expert, kind="stable")
            bounds = np.searchsorted(route_expert[order], experts[::cap])
            bounds = np.append(bounds, order.size)
            batches = [
                (experts[b * cap : (b + 1) * cap], order[bounds[b] : bounds[b + 1]])
                for b in range(bounds.size - 1)
            ]

        # Route outputs are pooled per (token, group) so every group's U is
        # applied once, by one matmul. A scatter-add, not a one-hot matmul:
        # it is exact (a multi-row float matmul on Metal is only good to
        # ~1e-3) and allocates nothing beyond the pooled rows.
        pooled = mx.zeros((n * g.n_groups, g.latent), dtype=mx.float32)
        for batch_experts, routes in batches:
            route_ids = route_expert[routes]
            slot = (routes // top_k) * g.n_groups + self._labels_host[route_ids]
            slot_mx = mx.array(slot.astype(np.uint32))
            out = self._route_outputs(
                batch_experts.tolist(),
                xl[slot_mx],
                mx.array(np.searchsorted(batch_experts, route_ids).astype(np.uint32)),
                scores[mx.array(routes.astype(np.uint32))],
            )
            pooled = pooled.at[slot_mx].add(out)
            if len(batches) > 1:
                # Release this batch's records before loading the next.
                mx.eval(pooled)
        routed = self._project(pooled.reshape(n, g.n_groups * g.latent), self._up)
        return routed.reshape(*lead, g.hidden).astype(x.dtype)
