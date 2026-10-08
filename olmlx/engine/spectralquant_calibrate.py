"""SpectralQuant calibration: eigenspectral analysis of KV cache vectors.

Collects key/value vectors during a calibration pass, computes per-head
covariance matrices, eigendecomposes them, and derives the spectral
rotation matrices and non-uniform codebooks needed for SpectralQuant
compression.
"""

from __future__ import annotations

import contextlib
import json
import logging
import threading
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from olmlx.config import SPECTRAL_CALIBRATIONS
from olmlx.engine.turboquant_cache import _is_plain_kv_cache

from olmlx.engine.spectralquant import _PACKABLE_BITS, allocate_bits, fit_codebook

logger = logging.getLogger(__name__)

#: Calibration dirs already warned about as pre-#761 (column) layouts.
_LEGACY_WARNED: set[Path] = set()

#: Default max tokens collected per head during calibration.  Also duplicated
#: in ``model_manager.py`` (which avoids importing this module eagerly) —
#: both copies must be kept in sync.
_SPECTRAL_DEFAULT_MAX_TOKENS_PER_HEAD = 8192
_SPECTRAL_DEFAULT_NUM_SAMPLES = 256

#: Conservative expert LRU budget during calibration. Calibration is
#: latency-tolerant and the budget only affects *which* experts are resident,
#: never the K/V output, so we minimize footprint to leave room for the
#: K/V-collection buffers. Bump if calibration is too slow.
_CALIBRATION_CACHE_BUDGET_EXPERTS = 8
_CALIBRATION_IO_THREADS = 32

#: Calibration objective (#749) -> the model-relative directory it writes.
#: ``reconstruction`` minimizes ||k - k_hat||; ``attention`` (``spectral-qa``)
#: minimizes the error in the attention products q·k. Derived from
#: ``config.SPECTRAL_CALIBRATIONS``.
SPECTRAL_DIR_BY_OBJECTIVE: dict[str, str] = dict(SPECTRAL_CALIBRATIONS.values())

#: Attention-objective candidate selection fits each candidate's codebooks on
#: the remaining keys and scores it on this held-out fraction.
_QA_HOLDOUT_FRACTION = 0.25
#: A candidate must beat the reconstruction baseline's held-out q·k error by
#: this relative margin to replace it, so split noise and Lloyd-Max local
#: optima can't trade a known-good calibration for an equivalent-or-worse one.
_QA_SWITCH_MARGIN = 0.02


def _resolve_config_holder(inner: Any, model: Any) -> Any:
    # Some architectures (e.g. Qwen3Next) expose the config namespace only on
    # the top-level model, not on the backbone returned by `_get_backbone`.
    # mlx-lm models expose it as `.args`; some mlx-vlm LanguageModel wrappers
    # expose it as `.config`. Prefer `.args` across both holders before falling
    # back to `.config` — otherwise an unrelated `.config` on `inner` (e.g.
    # inherited from a framework mixin) could shadow the real `.args` on
    # `model`, defeating the Qwen3Next fix. Use `is not None` rather than
    # `hasattr`: partially-constructed wrappers may set `self.args = None` as
    # a class attribute, which would pass `hasattr` but yield a `None`
    # namespace downstream and silently miscalibrate.
    for obj in (inner, model):
        if getattr(obj, "args", None) is not None:
            return obj
    for obj in (inner, model):
        if getattr(obj, "config", None) is not None:
            return obj
    raise RuntimeError(
        "Cannot detect model configuration: neither the backbone nor the "
        "top-level model exposes '.args' or '.config'. Unsupported architecture."
    )


def _config_namespace(cfg_holder: Any) -> Any:
    # Return the `.args` or `.config` object carrying architecture fields.
    # `_resolve_config_holder` normally enforces that one of these is present,
    # but raise explicitly here so the function's contract is enforceable in
    # isolation (matches the `_detect_head_dim` pattern in turboquant_cache).
    args = getattr(cfg_holder, "args", None)
    result = args if args is not None else getattr(cfg_holder, "config", None)
    if result is None:
        raise RuntimeError(
            f"_config_namespace: {type(cfg_holder).__name__} has neither "
            "'.args' nor '.config'"
        )
    return result


def _build_empty_collection_error(first_exc: Exception | None) -> RuntimeError:
    """Build the error raised when calibration collected zero KV vectors.

    Chains `first_exc` via `__cause__` and sets `__suppress_context__=True` so
    the behavior matches `raise ... from first_exc`: the traceback shows the
    forward-pass cause and nothing else. Without suppression, raising this
    from inside an `except` block in the future would also surface the
    unrelated implicit `__context__`.
    """
    if first_exc is not None:
        err = RuntimeError(
            "No KV vectors were collected during calibration — "
            "see cause above for the forward-pass error."
        )
        err.__cause__ = first_exc
        err.__suppress_context__ = True
        return err
    return RuntimeError(
        "No KV vectors were collected during calibration. "
        "No attention-layer cache entries were found — the model may have no "
        "attention layers, or all attention layers fell outside the "
        "calibration window."
    )


def _is_attention_cache(cache_entry: Any, expected_head_dim: int | None = None) -> bool:
    # Combined filter: must be a standard KVCache (not an SSM cache type such
    # as ArraysCache) AND expose a plausible 4D attention state. The isinstance
    # guard is load-bearing — shape alone cannot reject Mamba2 states where
    # `d_state == head_dim`. Keeping both checks inside this function means
    # the signature enforces the full contract and a future refactor can't
    # accidentally drop the type guard. Also guards against:
    # - empty caches not yet populated (len(state) < 2)
    # - caches seeded with < 2 tokens (seq < 2; already excluded by the
    #   `len(tokens) < 2` guard in `calibrate_model`, but kept as defense)
    # - head_dim mismatch (e.g. model weights loaded with a mismatched config),
    #   only when ``expected_head_dim`` is given. Calibration passes None: a
    #   layer's dims come from its own cache tensors, since they can differ
    #   from the model-wide config (Gemma 4 full-attention layers: head_dim
    #   512 and 2-4 KV heads vs config 256 and 8-16).
    # ``_is_plain_kv_cache`` also accepts mlx-vlm's plain KVCache (a VLM-only
    # checkpoint calibrates through ``load_vlm``), but not its subclasses.
    if not _is_plain_kv_cache(cache_entry):
        return False
    state: Any = cache_entry.state
    if not state or len(state) < 2:
        return False
    keys = state[0]
    if not (hasattr(keys, "ndim") and keys.ndim == 4):
        return False
    shape = keys.shape
    return shape[2] >= 2 and (
        expected_head_dim is None or shape[3] == expected_head_dim
    )


def _resolve_cache_owner(inner: Any, model: Any) -> Any:
    # `make_prompt_cache` defers to `make_cache()` on the passed object. When
    # the top-level model defines `make_cache`, it's always the authoritative
    # source for per-layer cache types — required for hybrid SSM+attention
    # architectures (Qwen3Next) and harmless for homogeneous ones. Falling back
    # to the backbone preserves legacy behavior for models that don't define
    # `make_cache` at the top level.
    if hasattr(model, "make_cache"):
        return model
    return inner


def compute_covariance(data: mx.array) -> mx.array:
    """Compute centered sample covariance matrix.

    Args:
        data: (N, D) matrix of vectors.

    Returns:
        (D, D) covariance matrix in float32.
    """
    data = data.astype(mx.float32)
    n = data.shape[0]
    mean = mx.mean(data, axis=0, keepdims=True)
    data_c = data - mean
    return (data_c.T @ data_c) / n


def eigendecompose(cov: mx.array) -> tuple[mx.array, mx.array]:
    """Eigendecompose a symmetric covariance matrix.

    Args:
        cov: (D, D) symmetric matrix.

    Returns:
        (eigenvalues, eigenvectors) sorted descending by eigenvalue.
        eigenvalues: (D,), eigenvectors: (D, D) — columns are eigenvectors.
    """
    eigenvalues, eigenvectors = mx.linalg.eigh(cov, stream=mx.cpu)
    mx.eval(eigenvalues, eigenvectors)

    # eigh returns ascending order — reverse to descending
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]

    # Clamp negative eigenvalues (numerical noise)
    eigenvalues = mx.maximum(eigenvalues, mx.array(0.0))

    return eigenvalues, eigenvectors


def compute_d_eff(eigenvalues: mx.array) -> int:
    """Compute effective dimensionality via participation ratio.

    d_eff = (sum(lambda))^2 / sum(lambda^2)

    This measures how many dimensions carry significant signal.

    Args:
        eigenvalues: Sorted eigenvalues (descending).

    Returns:
        Effective dimensionality (integer, at least 1).
    """
    ev = eigenvalues.astype(mx.float32)
    total = mx.sum(ev)
    total_sq = mx.sum(ev * ev)

    if float(total_sq) < 1e-12:
        return len(eigenvalues)

    d_eff = float(total * total / total_sq)
    # Round and clamp to [1, dim]
    d_eff = max(1, min(round(d_eff), len(eigenvalues)))
    return d_eff


def calibrate_head(
    kv_data: mx.array,
    avg_bits: int = 4,
) -> dict[str, Any]:
    """Calibrate spectral quant for a single attention head.

    Args:
        kv_data: (N, head_dim) key or value vectors from calibration.
        avg_bits: Target average bits per dimension.

    Returns:
        Dict with keys: eigenvectors, d_eff, codebook_sem, codebook_tail,
        bits_high, bits_low.
    """
    return _fit_pca_plan(*_pca_plan(kv_data, avg_bits))


def _pca_plan(
    kv_data: mx.array, avg_bits: int
) -> tuple[mx.array, mx.array, int, int, int]:
    """``calibrate_head``'s basis and bit split, without the codebooks.

    Returns ``(data_norm, basis, d_eff, bits_high, bits_low)``; ``basis`` has
    one eigenvector per row.
    """
    head_dim = kv_data.shape[-1]

    # Step 1: Covariance and eigendecomposition
    cov = compute_covariance(kv_data)
    eigenvalues, eigenvectors = eigendecompose(cov)

    # Step 2: Effective dimensionality
    d_eff = compute_d_eff(eigenvalues)

    # Step 3: Bit allocation
    bits_high, bits_low = allocate_bits(d_eff, head_dim, avg_bits)

    # Step 4: Normalize data (matching spectral_quantize which normalizes to
    # unit sphere before rotating), then rotate into spectral basis.
    data_f32 = kv_data.astype(mx.float32)
    norms = mx.sqrt(mx.sum(data_f32**2, axis=-1, keepdims=True))
    data_norm = data_f32 / mx.maximum(norms, mx.array(1e-8))
    # ``eigendecompose`` returns eigenvectors as COLUMNS; store the basis with
    # one eigenvector per ROW, which is what ``SpectralRotation.rotate``
    # (``x @ V.T``) projects onto. Using the column matrix directly projected
    # onto its rows, so the leading "semantic" coordinates weren't the
    # high-variance directions (#761).
    basis = eigenvectors.T
    return data_norm, basis, d_eff, bits_high, bits_low


def _fit_pca_plan(
    data_norm: mx.array, basis: mx.array, d_eff: int, bits_high: int, bits_low: int
) -> dict[str, Any]:
    """Fit a ``_pca_plan``'s codebooks; returns ``calibrate_head``'s result."""
    codebook_sem, codebook_tail = _fit_regime_codebooks(
        data_norm @ basis.T, d_eff, bits_high, bits_low
    )
    return {
        "eigenvectors": basis,  # (head_dim, head_dim), rows = eigvecs
        "d_eff": d_eff,
        "codebook_sem": codebook_sem,
        "codebook_tail": codebook_tail,
        "bits_high": bits_high,
        "bits_low": bits_low,
    }


def allocate_bits_weighted(
    importance: np.ndarray, avg_bits: int
) -> tuple[int, int, int]:
    """Pick ``(d_eff, bits_high, bits_low)`` minimizing weighted distortion.

    ``importance`` is per-coordinate, sorted descending; coordinate ``i``'s
    error contributes ``importance[i] * 4**-bits`` at high rate. The leading
    ``d_eff`` coordinates get ``bits_high``, the rest ``bits_low``, and the
    total never exceeds the ``len(importance) * avg_bits`` budget (unlike
    ``allocate_bits``, which minimizes absolute slack and may overshoot).
    Both widths come from ``_PACKABLE_BITS``. Falls back to uniform
    ``avg_bits`` (always within budget), which also wins ties.
    """
    imp = np.asarray(importance, dtype=np.float64)
    D = len(imp)
    budget = D * avg_bits
    prefix = np.concatenate([[0.0], np.cumsum(imp)])
    total = prefix[-1]

    best = (D, avg_bits, avg_bits)
    best_cost = total * 4.0**-avg_bits
    # Relative tolerance so float noise can't pick a non-uniform split that
    # is no better than uniform.
    tol = 1e-9 * max(total, 1e-30)
    for b_high in sorted(_PACKABLE_BITS, reverse=True):
        for b_low in sorted((b for b in _PACKABLE_BITS if b <= b_high), reverse=True):
            for d in range(1, D + 1):
                if d * b_high + (D - d) * b_low > budget:
                    continue
                cost = prefix[d] * 4.0**-b_high + (total - prefix[d]) * 4.0**-b_low
                if cost < best_cost - tol:
                    best_cost = cost
                    best = (d, b_high, b_low)
    return best


def _sym_eig_rows(mat: np.ndarray) -> np.ndarray:
    """Eigenvectors of a symmetric matrix as ROWS, descending eigenvalue."""
    _vals, vecs = np.linalg.eigh((mat + mat.T) / 2.0)
    return vecs[:, ::-1].T


def _psd_sqrt(mat: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh((mat + mat.T) / 2.0)
    return (vecs * np.sqrt(np.maximum(vals, 0.0))) @ vecs.T


def _nearest_centroid(values: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """Snap each value to its nearest entry of a sorted 1-D codebook."""
    if codebook.size == 1:
        return np.full_like(values, codebook[0])
    mids = (codebook[:-1] + codebook[1:]) / 2.0
    return codebook[np.searchsorted(mids, values)]


def _fit_regime_codebooks(
    rotated: mx.array | np.ndarray,
    d_eff: int,
    bits_high: int,
    bits_low: int,
) -> tuple[mx.array, mx.array]:
    """Fit the semantic/tail Lloyd-Max codebooks on rotated unit vectors.

    A full-rank semantic regime (``d_eff == head_dim``) gets the ``[0.0]``
    tail sentinel.
    """

    def _flat(x: mx.array | np.ndarray) -> mx.array:
        x = x.reshape(-1)
        return mx.array(x.astype(np.float32)) if isinstance(x, np.ndarray) else x

    codebook_sem = fit_codebook(_flat(rotated[:, :d_eff]), bits=bits_high)
    if d_eff < rotated.shape[1]:
        codebook_tail = fit_codebook(_flat(rotated[:, d_eff:]), bits=bits_low)
    else:
        codebook_tail = mx.array([0.0])
    return codebook_sem, codebook_tail


def calibrate_head_qa(
    kv_data: mx.array,
    query_cov: np.ndarray,
    avg_bits: int = 4,
    seed: int = 0,
) -> dict[str, Any]:
    """Attention-aware (``spectral-qa``) key calibration (#749).

    Minimizes the error in attention products, ``E[(q·(k - k_hat))^2] =
    E[e^T C_q e]`` with ``C_q`` the uncentered query second moment, instead
    of ``||k - k_hat||^2``. The runtime codec is unchanged, so the basis
    stays orthonormal (the cache unrotates with ``V``, not ``V^-1``) — only
    the basis, the coordinate order and the bit split differ.

    The baseline is ``calibrate_head`` itself (key PCA), returned verbatim
    unless a candidate beats it by ``_QA_SWITCH_MARGIN``. Other candidates:
    key PCA re-ordered by query importance, query PCA, the eigenbasis of
    ``C_q^1/2 M_k C_q^1/2`` and of the symmetrized ``C_q M_k``. In each,
    coordinate ``i``'s importance is ``(u_i^T C_q u_i) * (u_i^T M_k u_i)``
    (``M_k``: second moment of the unit-normalized keys, which is what the
    codec quantizes), and ``allocate_bits_weighted`` picks the split.

    Every candidate, baseline included, is scored the same way: converged
    codebooks fit on the keys outside a held-out split, q·k error measured
    on the held-out keys. (Scoring with cheaper, unconverged codebooks
    understated the baseline and let worse bases win on real models.) The
    winner's codebooks are refit on all the keys.

    Returns the same fields as ``calibrate_head`` plus ``objective_basis``
    (the winning candidate's name).
    """
    # Only the baseline's plan up front: its full-data codebooks are fit
    # only if it wins.
    plan = _pca_plan(kv_data, avg_bits)

    def _baseline() -> dict[str, Any]:
        result = _fit_pca_plan(*plan)
        result["objective_basis"] = "key_pca"
        return result

    c_q = np.asarray(query_cov, dtype=np.float64)
    if not np.isfinite(c_q).all() or np.trace(c_q) <= 0.0:
        return _baseline()

    keys = np.array(kv_data.astype(mx.float32), dtype=np.float64)
    N, _D = keys.shape
    norms = np.sqrt(np.sum(keys**2, axis=-1, keepdims=True))
    unit = keys / np.maximum(norms, 1e-8)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    n_hold = int(N * _QA_HOLDOUT_FRACTION) if N >= 16 else 0
    hold_idx, fit_idx = perm[:n_hold], perm[n_hold:]
    if n_hold == 0:
        hold_idx = fit_idx

    m_k = unit[fit_idx].T @ unit[fit_idx] / len(fit_idx)
    _data_norm, base_basis, base_d, base_bh, base_bl = plan
    key_pca = np.array(base_basis, dtype=np.float64)

    candidates: list[tuple[str, np.ndarray, int, int, int]] = [
        ("key_pca", key_pca, base_d, base_bh, base_bl)
    ]
    w = _psd_sqrt(c_q)
    for name, basis in (
        ("key_pca_weighted", key_pca),
        ("query_pca", _sym_eig_rows(c_q)),
        ("whitened", _sym_eig_rows(w @ m_k @ w)),
        ("product", _sym_eig_rows(c_q @ m_k + m_k @ c_q)),
    ):
        # diag(U C U^T) as BLAS matmuls (a 3-operand einsum is a scalar loop)
        imp = np.sum((basis @ c_q) * basis, axis=1) * np.sum(
            (basis @ m_k) * basis, axis=1
        )
        order = np.argsort(-imp, kind="stable")
        d, bh, bl = allocate_bits_weighted(imp[order], avg_bits)
        candidates.append((name, basis[order], d, bh, bl))

    hold_unit, hold_keys, hold_norms = unit[hold_idx], keys[hold_idx], norms[hold_idx]
    scores: list[float] = []
    for name, basis, d, bh, bl in candidates:
        cb_sem, cb_tail = _fit_regime_codebooks(unit[fit_idx] @ basis.T, d, bh, bl)
        y = hold_unit @ basis.T
        y_hat = np.concatenate(
            [
                _nearest_centroid(y[:, :d], np.array(cb_sem, dtype=np.float64)),
                _nearest_centroid(y[:, d:], np.array(cb_tail, dtype=np.float64)),
            ],
            axis=-1,
        )
        err = hold_keys - hold_norms * (y_hat @ basis)
        scores.append(float(np.mean(np.sum((err @ c_q) * err, axis=1))))
        logger.debug(
            "spectral-qa candidate %s: d_eff=%d bits=(%d,%d) qk_mse=%.4g",
            name,
            d,
            bh,
            bl,
            scores[-1],
        )

    best = min(range(1, len(candidates)), key=scores.__getitem__)
    if not scores[best] < (1.0 - _QA_SWITCH_MARGIN) * scores[0]:
        return _baseline()

    name, basis, d, bh, bl = candidates[best]
    codebook_sem, codebook_tail = _fit_regime_codebooks(unit @ basis.T, d, bh, bl)
    return {
        "eigenvectors": mx.array(basis.astype(np.float32)),
        "d_eff": d,
        "codebook_sem": codebook_sem,
        "codebook_tail": codebook_tail,
        "bits_high": bh,
        "bits_low": bl,
        "objective_basis": name,
    }


# ---------------------------------------------------------------------------
# Query capture (attention objective)
# ---------------------------------------------------------------------------

# One process-wide wrapper around ``mx.fast.scaled_dot_product_attention``,
# installed while any capture is active (refcounted, so overlapping captures
# on different threads can't restore each other's patch out of order). Each
# capture records only its own thread's calls, so an auto-calibration inside
# the server never sees another model's generation traffic.
_sdpa_lock = threading.Lock()
_sdpa_tls = threading.local()
_sdpa_refs = 0
#: The real SDPA, saved on first install. Never reset: another thread may have
#: fetched ``_recording_sdpa`` from ``mx.fast`` just before the last capture
#: restored the original, and still call it afterwards.
_sdpa_orig: Any = None


def _recording_sdpa(*args: Any, **kwargs: Any) -> Any:
    records = getattr(_sdpa_tls, "records", None)
    if records is not None:
        q = args[0] if args else kwargs.get("q")
        k = args[1] if len(args) > 1 else kwargs.get("k")
        if q is not None and k is not None:
            records.append((q, k))
    return _sdpa_orig(*args, **kwargs)


@contextlib.contextmanager
def _capture_sdpa_queries():
    """Record ``(queries, keys)`` of this thread's SDPA calls.

    Patches ``mx.fast.scaled_dot_product_attention``, which mlx-lm's (and
    mlx-vlm's) attention helpers look up at call time. Yields the list the
    records are appended to; always restores the original function.
    """
    global _sdpa_refs, _sdpa_orig
    records: list[tuple[mx.array, mx.array]] = []
    prev = getattr(_sdpa_tls, "records", None)
    with _sdpa_lock:
        if _sdpa_refs == 0:
            current = mx.fast.scaled_dot_product_attention
            if current is not _recording_sdpa:
                _sdpa_orig = current
            mx.fast.scaled_dot_product_attention = _recording_sdpa
        _sdpa_refs += 1
    _sdpa_tls.records = records
    try:
        yield records
    finally:
        _sdpa_tls.records = prev
        with _sdpa_lock:
            _sdpa_refs -= 1
            if _sdpa_refs == 0:
                mx.fast.scaled_dot_product_attention = _sdpa_orig


def _match_layer(
    layer_keys: list[tuple[int, mx.array]], k: mx.array, cursor: int
) -> int | None:
    """Index into ``layer_keys`` of the cache whose keys equal ``k``.

    Tries the next layer in forward order first (the usual match, one sync);
    otherwise compares every other same-shape cache in a single ``mx.eval``,
    so a call that matches nothing (e.g. a sliding-window layer with the same
    key shape) costs one sync rather than one per layer.
    """
    n = len(layer_keys)
    cands = [
        pos
        for pos in (*range(cursor, n), *range(0, cursor))
        if layer_keys[pos][1].shape == k.shape
    ]
    if not cands:
        return None
    if bool(mx.array_equal(layer_keys[cands[0]][1], k)):
        return cands[0]
    rest = cands[1:]
    if not rest:
        return None
    equal = [mx.array_equal(layer_keys[pos][1], k) for pos in rest]
    mx.eval(equal)
    for pos, eq in zip(rest, equal):
        if eq.item():
            return pos
    return None


def _accumulate_query_stats(
    records: list[tuple[mx.array, mx.array]],
    prompt_cache: list,
    num_layers: int,
    query_stats: dict[int, dict[str, Any]],
) -> None:
    """Attribute captured SDPA queries to layers and add ``sum q q^T``.

    A call belongs to the layer whose cached keys it attended over: matched
    by shape, then exact equality. Calls run in forward order, so the next
    layer after the last match is tried first (which also disambiguates
    identical caches); any other matching layer is accepted too, which
    covers KV-shared layers (their queries do attend over the owner's
    keys). Unmatched calls (sliding-window layers, models that repeat KV
    heads before SDPA, ...) are dropped.
    """
    layer_keys: list[tuple[int, mx.array]] = [
        (i, prompt_cache[i].state[0])
        for i in range(min(num_layers, len(prompt_cache)))
        if _is_attention_cache(prompt_cache[i])
    ]
    if not layer_keys:
        return
    cursor = 0
    for q, k in records:
        try:
            if getattr(q, "ndim", 0) != 4 or q.shape[-1] != k.shape[-1]:
                continue
            chosen = _match_layer(layer_keys, k, cursor)
            if chosen is None:
                continue
            if chosen >= cursor:
                cursor = chosen + 1
            qf = q.reshape(-1, q.shape[-1]).astype(mx.float32)
            second = np.array(qf.T @ qf, dtype=np.float64)
        except Exception as exc:  # e.g. a tracer recorded under mx.compile
            logger.debug("Skipping SDPA query record: %s", exc)
            continue
        layer = layer_keys[chosen][0]
        st = query_stats.get(layer)
        if st is None:
            st = query_stats[layer] = {"sum": np.zeros_like(second), "count": 0}
        if st["sum"].shape != second.shape:
            continue
        st["sum"] += second
        st["count"] += qf.shape[0]


# ---------------------------------------------------------------------------
# Calibration data persistence
# ---------------------------------------------------------------------------

# Key format: (layer_idx, head_idx, "key"|"value")
CalibrationKey = tuple[int, int, str]
CalibrationData = dict[CalibrationKey, dict[str, Any]]


def save_calibration(calibration: CalibrationData, output_dir: Path) -> None:
    """Save calibration data to disk.

    Writes:
      - spectral_config.json: metadata (d_eff, bit allocations per head)
      - calibration.safetensors: eigenvectors + codebooks
    """
    import safetensors.numpy

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ``basis: rows`` marks calibrations whose eigenvectors are stored one per
    # row (#761); older files stored columns, see ``load_calibration``.
    config: dict[str, Any] = {"basis": "rows", "heads": {}}
    tensors: dict[str, np.ndarray] = {}

    for (layer, head, kind), data in calibration.items():
        prefix = f"layer_{layer}_head_{head}_{kind}"
        config["heads"][prefix] = {
            "d_eff": data["d_eff"],
            "bits_high": data["bits_high"],
            "bits_low": data["bits_low"],
        }
        if "objective_basis" in data:
            config["heads"][prefix]["objective_basis"] = data["objective_basis"]
        tensors[f"{prefix}_eigvecs"] = np.array(data["eigenvectors"])
        tensors[f"{prefix}_codebook_sem"] = np.array(data["codebook_sem"])
        tensors[f"{prefix}_codebook_tail"] = np.array(data["codebook_tail"])

    (output_dir / "spectral_config.json").write_text(json.dumps(config, indent=2))
    safetensors.numpy.save_file(tensors, str(output_dir / "calibration.safetensors"))


def load_calibration(calibration_dir: Path) -> CalibrationData:
    """Load calibration data from disk.

    Returns:
        Dict mapping (layer, head, kind) → calibration result.
    """
    import safetensors.numpy

    calibration_dir = Path(calibration_dir)
    config = json.loads((calibration_dir / "spectral_config.json").read_text())
    tensors = safetensors.numpy.load_file(
        str(calibration_dir / "calibration.safetensors")
    )
    if config.get("basis") != "rows" and calibration_dir not in _LEGACY_WARNED:
        # Cache builds reload calibration per request; warn once per dir.
        _LEGACY_WARNED.add(calibration_dir)
        # Pre-#761 calibrations projected onto the eigenvector matrix's rows
        # while storing eigenvectors as columns. Their codebooks were fit on
        # that same projection, so they still round-trip consistently; they
        # just don't concentrate variance in the semantic coordinates.
        logger.warning(
            "SpectralQuant calibration at %s predates the eigenbasis fix "
            "(#761) and compresses worse than it should; re-calibrate with "
            "'olmlx spectral prepare <model>' (add --avg-bits if not 4).",
            calibration_dir,
        )

    result: CalibrationData = {}
    for prefix, meta in config["heads"].items():
        # Parse prefix: "layer_{i}_head_{j}_{kind}"
        parts = prefix.split("_")
        layer = int(parts[1])
        head = int(parts[3])
        kind = parts[4]  # "key" or "value"

        result[(layer, head, kind)] = {
            "eigenvectors": mx.array(tensors[f"{prefix}_eigvecs"]),
            "d_eff": meta["d_eff"],
            "codebook_sem": mx.array(tensors[f"{prefix}_codebook_sem"]),
            "codebook_tail": mx.array(tensors[f"{prefix}_codebook_tail"]),
            "bits_high": meta["bits_high"],
            "bits_low": meta["bits_low"],
        }

    return result


# ---------------------------------------------------------------------------
# Full model calibration pipeline
# ---------------------------------------------------------------------------


def _load_calibration_model(model_path: str):
    """Load a model for calibration; returns model + architecture facts.

    Shared by the spectral and shard calibration pipelines.  Imports stay
    inside the function (call-time) so tests can patch the helpers on their
    home modules.
    """
    from pathlib import Path

    from olmlx.engine.flash.prepare import (
        _get_backbone,
        load_model_with_strict_fallback,
    )
    from olmlx.engine.turboquant_cache import _detect_head_dim

    flash_moe_dir = Path(model_path) / "flash_moe"
    store = None
    if (flash_moe_dir / "flash_moe_layout.json").exists():
        if not (flash_moe_dir / "flash_moe_config.json").exists():
            # The bundler writes the layout as its last act and the config
            # afterwards, so layout-without-config means an interrupted
            # `olmlx flash prepare`. Committing to the flash path would die
            # with a bare FileNotFoundError; a full load could OOM.
            raise ValueError(
                f"Incomplete Flash-MoE bundle at {flash_moe_dir}: "
                "flash_moe_layout.json exists but flash_moe_config.json is "
                "missing (interrupted preparation?). Re-run `olmlx flash "
                "prepare` to rebuild the bundle."
            )
        from olmlx.engine.flash.flash_moe_model import load_flash_moe_model

        # Bundle present: commit to the Flash-MoE path. Do NOT fall back to a
        # full load on failure — that would OOM on the large models this path
        # exists to support.
        model, tokenizer, store = load_flash_moe_model(
            model_path,
            flash_moe_dir,
            cache_budget_experts=_CALIBRATION_CACHE_BUDGET_EXPERTS,
            io_threads=_CALIBRATION_IO_THREADS,
        )
    else:
        try:
            model, tokenizer = load_model_with_strict_fallback(model_path, lazy=False)
        except ValueError:
            from olmlx.engine.vlm_load import load_vlm

            model, processor = load_vlm(model_path, lazy=False)
            tokenizer = (
                processor.tokenizer if hasattr(processor, "tokenizer") else processor
            )

    try:
        inner = _get_backbone(model)
        num_layers = len(inner.layers)
        cfg_holder = _resolve_config_holder(inner, model)
        cfg_ns = _config_namespace(cfg_holder)
        head_dim = _detect_head_dim(cfg_holder, layers_hint=inner)
        logger.debug("calibration: resolved head_dim=%d", head_dim)

        n_kv_heads = getattr(cfg_ns, "num_key_value_heads", None)
        if n_kv_heads is None:
            n_kv_heads = getattr(cfg_ns, "num_attention_heads", 1)
    except Exception:
        if store is not None:
            store.close()
        raise
    return model, tokenizer, inner, head_dim, n_kv_heads, num_layers, store


def _load_and_collect_kv(
    model_path: str,
    *,
    num_samples: int,
    calibration_dataset: str | None,
    max_tokens_per_head: int,
    progress_callback: Any | None,
    query_stats: dict[int, dict[str, Any]] | None = None,
) -> tuple[Any, Any, Any, int, int, int, dict]:
    """Fetch calibration texts, load the model, and collect K/V vectors.

    The shared front half of ``calibrate_model`` (spectral) and
    ``calibrate_model_shard``. Texts are fetched BEFORE the model load: they
    need nothing from the model, and fetching first means a failing or slow
    dataset download never wastes a multi-GB load nor runs under an open
    Flash-MoE store. The store (if any) is open only for the collection
    forwards and is always closed.

    ``query_stats`` is forwarded to ``collect_kv_vectors``.

    Returns ``(model, tokenizer, inner, head_dim, n_kv_heads, num_layers,
    kv_collectors)``.
    """
    if progress_callback:
        progress_callback("Generating calibration data", 0.0)

    from olmlx.engine.flash.prepare import (
        _get_c4_calibration_data,
        _get_calibration_data,
    )

    if calibration_dataset == "synthetic":
        texts = _get_calibration_data(num_samples)
    else:
        texts = _get_c4_calibration_data(num_samples)

    if progress_callback:
        progress_callback("Loading model", 0.05)

    (
        model,
        tokenizer,
        inner,
        head_dim,
        n_kv_heads,
        num_layers,
        store,
    ) = _load_calibration_model(model_path)

    try:
        if progress_callback:
            progress_callback("Collecting KV vectors", 0.1)

        kv_collectors = collect_kv_vectors(
            model,
            tokenizer,
            inner,
            num_layers=num_layers,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            texts=texts,
            max_tokens_per_head=max_tokens_per_head,
            progress_callback=progress_callback,
            query_stats=query_stats,
        )
    finally:
        if store is not None:
            store.close()

    return model, tokenizer, inner, head_dim, n_kv_heads, num_layers, kv_collectors


def collect_kv_vectors(
    model: Any,
    tokenizer: Any,
    inner: Any,
    *,
    num_layers: int,
    n_kv_heads: int,
    head_dim: int,
    texts: list[str],
    max_tokens_per_head: int,
    progress_callback: Any | None = None,
    progress_lo: float = 0.1,
    progress_hi: float = 0.5,
    query_stats: dict[int, dict[str, Any]] | None = None,
) -> dict[int, dict[int, dict[str, list[mx.array]]]]:
    """Collect post-RoPE K/V vectors per (layer, head) from forward passes.

    Returns kv_collectors[layer][head]["key"|"value"] = list of
    (seq, head_dim) chunks.  Each chunk starts at position 0 of its sample
    (relevant for de-roping in the shard pipeline).  Raises if nothing was
    collected.

    When ``query_stats`` is a dict, the forwards also capture each attention
    layer's queries (see ``_capture_sdpa_queries``) and fill
    ``query_stats[layer] = {"sum": sum q q^T, "count": n}`` (float64, all
    query heads pooled) for the attention objective (#749).
    """
    # Heads are discovered per layer from the cache tensors (``n_kv_heads`` /
    # ``head_dim`` are the model-wide config values, which per-layer layouts
    # such as Gemma 4's full-attention layers don't follow), so each layer's
    # collectors are sized lazily.
    del n_kv_heads, head_dim
    kv_collectors: dict[int, dict[int, dict[str, list[mx.array]]]] = {
        i: {} for i in range(num_layers)
    }
    tokens_collected: dict[tuple[int, int, str], int] = {}

    # Collect post-RoPE K/V vectors by running each sample with a fresh
    # KV cache, then extracting the cached tensors.  This captures keys and
    # values *after* rotary positional embeddings — the actual distribution
    # that gets stored in the KV cache at inference time.
    cache_model = _resolve_cache_owner(inner, model)
    with contextlib.ExitStack() as stack:
        records = (
            stack.enter_context(_capture_sdpa_queries())
            if query_stats is not None
            else None
        )
        first_exc = _collect_samples(
            model,
            tokenizer,
            cache_model,
            texts,
            num_layers=num_layers,
            max_tokens_per_head=max_tokens_per_head,
            kv_collectors=kv_collectors,
            tokens_collected=tokens_collected,
            records=records,
            query_stats=query_stats,
            progress_callback=progress_callback,
            progress_lo=progress_lo,
            progress_hi=progress_hi,
        )

    # Guard: if no KV vectors were collected, fail early with a clear message
    if sum(tokens_collected.values()) == 0:
        raise _build_empty_collection_error(first_exc)

    return kv_collectors


def _collect_samples(
    model: Any,
    tokenizer: Any,
    cache_model: Any,
    texts: list[str],
    *,
    num_layers: int,
    max_tokens_per_head: int,
    kv_collectors: dict[int, dict[int, dict[str, list[mx.array]]]],
    tokens_collected: dict[tuple[int, int, str], int],
    records: list | None,
    query_stats: dict[int, dict[str, Any]] | None,
    progress_callback: Any | None,
    progress_lo: float,
    progress_hi: float,
) -> Exception | None:
    """The per-sample forward loop of ``collect_kv_vectors``.

    Returns the first forward-pass exception (if any), for error chaining.
    """
    from mlx_lm.models.cache import make_prompt_cache

    from olmlx.engine.flash.prepare import _encode_tokens

    first_exc: Exception | None = None
    for sample_idx, text in enumerate(texts):
        tokens = _encode_tokens(tokenizer, text)
        if len(tokens) < 2:
            # Single-token prefills produce (1, n_kv, 1, head_dim) caches that
            # are easy to confuse with per-step SSM states, and carry no
            # meaningful KV statistics anyway. Skip them.
            logger.debug(
                "Skipping sample %d: too short (%d tokens)", sample_idx, len(tokens)
            )
            continue
        if len(tokens) > 512:
            tokens = tokens[:512]
        input_ids = mx.array([tokens])

        # Create a fresh cache and run the forward pass
        prompt_cache = make_prompt_cache(cache_model)
        try:
            model(input_ids, cache=prompt_cache)
        except Exception as exc:
            if first_exc is None:
                first_exc = exc
            logger.debug("Skipping sample %d: %s", sample_idx, exc)
            del prompt_cache
            if records is not None:
                records.clear()
            continue
        mx.eval([c.state for c in prompt_cache if hasattr(c, "state")])
        if records is not None and query_stats is not None:
            _accumulate_query_stats(records, prompt_cache, num_layers, query_stats)
            records.clear()

        # Extract K/V from each layer's cache
        advanced = False
        for layer_idx in range(min(num_layers, len(prompt_cache))):
            cache_entry = prompt_cache[layer_idx]
            # Combined type + shape filter. Rejects SSM cache types (ArraysCache,
            # etc.) and KVCaches whose state doesn't match attention shape.
            if not _is_attention_cache(cache_entry):
                continue
            state = cache_entry.state
            # KVCache.state returns [keys, values] with shape
            # (1, n_kv_heads, seq_len, head_dim)
            cached_keys = state[0]  # (1, n_kv_heads, seq, head_dim)
            cached_values = state[1]  # (1, n_kv_heads, seq, head_dim)

            for h in range(cached_keys.shape[1]):
                kv_collectors[layer_idx].setdefault(h, {"key": [], "value": []})
                tokens_collected.setdefault((layer_idx, h, "key"), 0)
                tokens_collected.setdefault((layer_idx, h, "value"), 0)
                if tokens_collected[(layer_idx, h, "key")] < max_tokens_per_head:
                    k_h = cached_keys[0, h, :, :]  # (seq, head_dim)
                    remaining = (
                        max_tokens_per_head - tokens_collected[(layer_idx, h, "key")]
                    )
                    k_h = k_h[:remaining]
                    kv_collectors[layer_idx][h]["key"].append(k_h)
                    tokens_collected[(layer_idx, h, "key")] += k_h.shape[0]
                    advanced = True

                if tokens_collected[(layer_idx, h, "value")] < max_tokens_per_head:
                    v_h = cached_values[0, h, :, :]  # (seq, head_dim)
                    remaining = (
                        max_tokens_per_head - tokens_collected[(layer_idx, h, "value")]
                    )
                    v_h = v_h[:remaining]
                    kv_collectors[layer_idx][h]["value"].append(v_h)
                    tokens_collected[(layer_idx, h, "value")] += v_h.shape[0]
                    advanced = True

        del prompt_cache
        if not advanced:
            # Every collectable head is at max_tokens_per_head (heads on
            # filtered layers never advance and never will), so further
            # forwards contribute nothing. On the Flash-MoE path each one
            # re-streams routed experts from SSD — stop here.
            logger.info(
                "KV collection saturated after %d/%d samples; stopping early",
                sample_idx + 1,
                len(texts),
            )
            break
        if progress_callback:
            frac = progress_lo + (sample_idx + 1) / len(texts) * (
                progress_hi - progress_lo
            )
            progress_callback(f"Collected {sample_idx + 1}/{len(texts)} samples", frac)

    return first_exc


def calibrate_model(
    model_path: str,
    output_dir: Path | None = None,
    num_samples: int = _SPECTRAL_DEFAULT_NUM_SAMPLES,
    calibration_dataset: str | None = None,
    avg_bits: int = 4,
    max_tokens_per_head: int = _SPECTRAL_DEFAULT_MAX_TOKENS_PER_HEAD,
    progress_callback: Any | None = None,
    objective: str = "reconstruction",
) -> Path:
    """Run spectral calibration on a model.

    Loads the model, runs calibration text through it to collect K/V vectors
    from attention layers, then performs eigenspectral analysis per head.

    Args:
        model_path: HF model path or local directory.
        output_dir: Where to write calibration files. Defaults to
            model_dir/spectral (model_dir/spectral_qa for the attention
            objective).
        num_samples: Number of calibration text samples.
        calibration_dataset: "c4", "synthetic", or None (defaults to c4).
        avg_bits: Target average bits per dimension.
        max_tokens_per_head: Max tokens to collect per head for covariance.
        progress_callback: Called with (description, fraction).
        objective: "reconstruction" (key PCA, ``spectral:N``) or "attention"
            (query-weighted keys, ``spectral-qa:N``, #749). Values are
            calibrated the same way under both.

    Returns:
        Path to the spectral calibration directory.
    """
    import gc
    import time

    if objective not in SPECTRAL_DIR_BY_OBJECTIVE:
        raise ValueError(
            f"Unknown spectral calibration objective {objective!r}; "
            f"expected one of {sorted(SPECTRAL_DIR_BY_OBJECTIVE)}"
        )
    query_stats: dict[int, dict[str, Any]] | None = (
        {} if objective == "attention" else None
    )

    if output_dir is None:
        output_dir = Path(model_path) / SPECTRAL_DIR_BY_OBJECTIVE[objective]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (
        model,
        tokenizer,
        inner,
        head_dim,
        n_kv_heads,
        num_layers,
        kv_collectors,
    ) = _load_and_collect_kv(
        model_path,
        num_samples=num_samples,
        calibration_dataset=calibration_dataset,
        max_tokens_per_head=max_tokens_per_head,
        progress_callback=progress_callback,
        query_stats=query_stats,
    )

    if query_stats is not None and not query_stats:
        raise RuntimeError(
            "Attention-objective calibration captured no queries: no "
            "mx.fast.scaled_dot_product_attention call could be matched to an "
            "attention layer's cached keys. This model's attention is not "
            "supported by spectral-qa; use 'spectral' instead."
        )

    if progress_callback:
        progress_callback("Running eigenspectral analysis", 0.5)

    # Calibrate per layer by aggregating all heads.
    # The KV cache operates on all heads simultaneously (shape: B, n_heads,
    # seq, head_dim), so a single rotation per layer is applied to every head.
    # Aggregating across heads produces a rotation that captures the shared
    # eigenstructure rather than being tuned to one head's statistics.
    calibration: CalibrationData = {}
    total_items = num_layers * 2  # key + value per layer
    done = 0

    for layer_idx in range(num_layers):
        for kind in ("key", "value"):
            # Concatenate all heads' data for this layer+kind
            all_chunks = []
            for head_chunks in kv_collectors[layer_idx].values():
                all_chunks.extend(head_chunks[kind])
            if not all_chunks:
                logger.debug(
                    "No KV data for layer %d %s, skipping",
                    layer_idx,
                    kind,
                )
                continue

            kv_data = mx.concatenate(all_chunks, axis=0)
            if query_stats is not None and kind == "key":
                stats = query_stats.get(layer_idx)
                if stats is not None and stats["count"] > 0:
                    result = calibrate_head_qa(
                        kv_data, stats["sum"] / stats["count"], avg_bits=avg_bits
                    )
                    logger.debug(
                        "spectral-qa layer %d: basis=%s d_eff=%d bits=(%d,%d)",
                        layer_idx,
                        result["objective_basis"],
                        result["d_eff"],
                        result["bits_high"],
                        result["bits_low"],
                    )
                else:
                    logger.warning(
                        "spectral-qa: no queries captured for layer %d; "
                        "calibrating its keys by reconstruction error",
                        layer_idx,
                    )
                    result = calibrate_head(kv_data, avg_bits=avg_bits)
            else:
                result = calibrate_head(kv_data, avg_bits=avg_bits)
            calibration[(layer_idx, 0, kind)] = result

            done += 1
            if progress_callback:
                frac = 0.5 + done / total_items * 0.4
                progress_callback(f"Calibrated {done}/{total_items} layer-kinds", frac)

    # Free collectors
    del kv_collectors
    gc.collect()
    mx.clear_cache()

    if progress_callback:
        progress_callback("Saving calibration", 0.9)

    # Save
    save_calibration(calibration, output_dir)

    # Write additional metadata
    meta = {
        "num_layers": num_layers,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "avg_bits": avg_bits,
        "objective": objective,
        "num_samples": num_samples,
        "max_tokens_per_head": max_tokens_per_head,
        "calibration_dataset": calibration_dataset or "c4",
        "calibrated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    # Merge into spectral_config.json
    config_path = output_dir / "spectral_config.json"
    config = json.loads(config_path.read_text())
    config["meta"] = meta
    config_path.write_text(json.dumps(config, indent=2))

    if progress_callback:
        progress_callback("Done", 1.0)

    logger.info("Spectral calibration complete: %s", output_dir)
    return output_dir
