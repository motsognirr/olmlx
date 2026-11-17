"""Metal kernels for latent expert dispatch straight from the packed records.

The reference path in :mod:`olmlx.engine.edge0.latent_moe` decodes every
routed expert to dense float32 and then multiplies — ~3.4M weights decoded
per expert for three matrix-vector products, which caps decode at ~1.5 tok/s
on a 48-layer model. These kernels fuse the two: one thread per output
element walks the ggml blocks of its weight row and accumulates the dot
product, so nothing is materialized and the work per token is just the
routed experts' multiply-adds.

* ``gate_up``: thread ``(ff row, route)`` -> ``silu(gate . x) * (up . x)``
  over ``Q3_K`` rows.
* ``down``: thread ``(latent row, route)`` -> ``score * (down . h)`` over
  ``IQ4_NL`` rows.

Same arithmetic as the reference path (float32 activations against exactly
decoded weights); ``tests/test_edge0_latent_kernels.py`` holds them to it.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import mlx.core as mx

from olmlx.engine.edge0 import ggml_quants
from olmlx.engine.edge0.pack import LatentGeometry

_HEADER = r"""
inline float edge0_half(const device uchar* p) {
    return float(as_type<half>(ushort(ushort(p[0]) | (ushort(p[1]) << 8))));
}

// Dot of one Q3_K weight row (n values, 110 bytes per 256) with x.
inline float edge0_q3k_dot(const device uchar* w, const device float* x, int n) {
    float acc = 0.0f;
    for (int blk = 0; blk < n / 256; ++blk) {
        const device uchar* b = w + blk * 110;
        const device float* xb = x + blk * 256;
        float d = edge0_half(b + 108);
        for (int k = 0; k < 16; ++k) {
            // k = half*8 + j*2 + part: 16 consecutive values share a scale.
            int half_i = k >> 3;
            int j = (k >> 1) & 3;
            int part = k & 1;
            int lo4 = (k < 8) ? (b[96 + k] & 15) : (b[96 + k - 8] >> 4);
            int hi2 = (b[104 + (k & 3)] >> (2 * (k >> 2))) & 3;
            float scale = d * float((lo4 | (hi2 << 4)) - 32);
            int shift = 2 * j;
            int bit = half_i * 4 + j;
            const device uchar* qs = b + 32 + half_i * 32 + part * 16;
            const device uchar* hm = b + part * 16;
            const device float* xs = xb + k * 16;
            float s = 0.0f;
            for (int l = 0; l < 16; ++l) {
                int code = ((qs[l] >> shift) & 3) + 4 * ((hm[l] >> bit) & 1) - 4;
                s += float(code) * xs[l];
            }
            acc += scale * s;
        }
    }
    return acc;
}

constant float EDGE0_IQ4NL[16] = {
    -127.0f, -104.0f, -83.0f, -65.0f, -49.0f, -35.0f, -22.0f, -10.0f,
    1.0f, 13.0f, 25.0f, 38.0f, 53.0f, 69.0f, 89.0f, 113.0f};

// Dot of one IQ4_NL weight row (n values, 18 bytes per 32) with x.
inline float edge0_iq4nl_dot(const device uchar* w, const device float* x, int n) {
    float acc = 0.0f;
    for (int blk = 0; blk < n / 32; ++blk) {
        const device uchar* b = w + blk * 18;
        const device float* xb = x + blk * 32;
        float s = 0.0f;
        for (int j = 0; j < 16; ++j) {
            uchar q = b[2 + j];
            s += EDGE0_IQ4NL[q & 15] * xb[j] + EDGE0_IQ4NL[q >> 4] * xb[j + 16];
        }
        acc += edge0_half(b) * s;
    }
    return acc;
}
"""

# params: [latent, ff, gu_row_bytes, down_row_bytes, record_bytes]
#
# Each kernel comes in two forms that differ only in how a route finds its
# expert record (``EDGE0_RECORD``): one stacked ``[n, record_bytes]`` buffer,
# or one input per expert. The second avoids copying the records into a
# stacked buffer on every layer call, which costs more than the kernels
# themselves when a single token routes to ten experts.
_GATE_UP_SRC = r"""
    uint row = thread_position_in_grid.x;
    uint route = thread_position_in_grid.y;
    int latent = params[0];
    int ff = params[1];
    ulong gu_row = (ulong)params[2];
    EDGE0_RECORD
    const device uchar* gate = record + (ulong)row * gu_row;
    const device float* xr = x + (ulong)route * (ulong)latent;
    float g = edge0_q3k_dot(gate, xr, latent);
    float u = edge0_q3k_dot(gate + (ulong)ff * gu_row, xr, latent);
    out[(ulong)route * (ulong)ff + row] = (g / (1.0f + metal::exp(-g))) * u;
"""

_DOWN_SRC = r"""
    uint row = thread_position_in_grid.x;
    uint route = thread_position_in_grid.y;
    int latent = params[0];
    int ff = params[1];
    ulong gu_row = (ulong)params[2];
    ulong down_row = (ulong)params[3];
    EDGE0_RECORD
    const device uchar* w = record + 2 * (ulong)ff * gu_row + (ulong)row * down_row;
    float y = edge0_iq4nl_dot(w, hidden + (ulong)route * (ulong)ff, ff);
    out[(ulong)route * (ulong)latent + row] = scores[route] * y;
"""

_STACKED_RECORD = (
    "const device uchar* record = records + (ulong)local[route] * (ulong)params[4];"
)

#: Most experts the one-input-per-expert kernels are built for.
MAX_SEPARATE_RECORDS = 16


def _separate_record(count: int) -> str:
    cases = " ".join(f"case {i}: record = rec{i}; break;" for i in range(1, count))
    return (
        "const device uchar* record = rec0; "
        f"switch (local[route]) {{ {cases} default: break; }}"
    )


def kernels_supported(geometry: LatentGeometry) -> bool:
    """True when the fused kernels can serve *geometry* on the current device."""
    if not mx.metal.is_available():
        return False
    if mx.default_device().type != mx.DeviceType.gpu:
        return False
    return (geometry.gu_type, geometry.down_type) == (
        ggml_quants.GGML_Q3_K,
        ggml_quants.GGML_IQ4_NL,
    )


@lru_cache(maxsize=None)
def _kernels(separate: int) -> tuple[Any, Any]:
    """The (gate_up, down) kernels; *separate* > 0 selects the per-expert-input form."""
    if separate:
        record_inputs = [f"rec{i}" for i in range(separate)]
        record = _separate_record(separate)
    else:
        record_inputs = ["records"]
        record = _STACKED_RECORD
    gate_up = mx.fast.metal_kernel(
        name=f"edge0_latent_gate_up_{separate}",
        input_names=[*record_inputs, "x", "local", "params"],
        output_names=["out"],
        source=_GATE_UP_SRC.replace("EDGE0_RECORD", record),
        header=_HEADER,
    )
    down = mx.fast.metal_kernel(
        name=f"edge0_latent_down_{separate}",
        input_names=[*record_inputs, "hidden", "local", "scores", "params"],
        output_names=["out"],
        source=_DOWN_SRC.replace("EDGE0_RECORD", record),
        header=_HEADER,
    )
    return gate_up, down


def routed_latent(
    records: mx.array | list[mx.array],
    x: mx.array,
    local: mx.array,
    scores: mx.array,
    geometry: LatentGeometry,
) -> mx.array:
    """Score-weighted latent output of every route.

    ``records``: ``[n_experts_loaded, record_bytes]`` uint8, or a list of up
    to :data:`MAX_SEPARATE_RECORDS` ``[record_bytes]`` arrays (no stacking).
    ``x``: ``[routes, latent]`` float32 — each route's (column-scaled) input.
    ``local``: ``[routes]`` uint32 index into ``records`` each route uses.
    ``scores``: ``[routes]`` float32 router weights.
    Returns ``[routes, latent]`` float32.
    """
    g = geometry
    routes = x.shape[0]
    params = mx.array(
        [g.latent, g.ff, g.gu_row_bytes, g.down_row_bytes, g.record_bytes],
        dtype=mx.int32,
    )
    if isinstance(records, list):
        if not 0 < len(records) <= MAX_SEPARATE_RECORDS:
            raise ValueError(
                f"expected 1..{MAX_SEPARATE_RECORDS} separate records, got {len(records)}"
            )
        record_inputs = records
        gate_up, down = _kernels(len(records))
    else:
        record_inputs = [records]
        gate_up, down = _kernels(0)
    hidden = gate_up(
        inputs=[*record_inputs, x, local, params],
        grid=(g.ff, routes, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(routes, g.ff)],
        output_dtypes=[mx.float32],
    )[0]
    return down(
        inputs=[*record_inputs, hidden, local, scores, params],
        grid=(g.latent, routes, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(routes, g.latent)],
        output_dtypes=[mx.float32],
    )[0]
