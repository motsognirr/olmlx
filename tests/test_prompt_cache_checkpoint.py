import threading

import mlx.core as mx
from mlx_lm.models.cache import KVCache

from olmlx.engine.prompt_cache.checkpoint import (
    SegmentedPrompt,
    Segment,
    snapshot_cache_for_persistence,
)


def test_segmented_prompt_total_tokens_is_sum_of_segments():
    sp = SegmentedPrompt(
        segments=[
            Segment(tokens=[1, 2, 3], role="system"),
            Segment(tokens=[4, 5], role="user"),
        ]
    )
    assert sp.total_tokens == 5
    assert sp.flatten() == [1, 2, 3, 4, 5]


def test_segmented_prompt_boundary_offsets_are_cumulative():
    sp = SegmentedPrompt(
        segments=[
            Segment(tokens=[1, 2, 3], role="system"),
            Segment(tokens=[4, 5], role="user"),
            Segment(tokens=[6], role="user"),
        ]
    )
    assert sp.boundary_offsets() == [3, 5, 6]


def test_segmented_prompt_empty_is_valid():
    sp = SegmentedPrompt(segments=[])
    assert sp.total_tokens == 0
    assert sp.boundary_offsets() == []
    assert sp.flatten() == []


def test_flatten_cache_state_unpacks_tuple_state():
    """flatten_cache_state must extend tuple state (keys, values) flatly,
    not nest the tuple inside the result. Callers pass the result to
    mx.eval, which would silently skip nested containers it doesn't know."""
    from olmlx.engine.prompt_cache.checkpoint import flatten_cache_state

    cache = [KVCache()]
    keys = mx.zeros((1, 4, 8, 16))
    values = mx.ones((1, 4, 8, 16))
    cache[0].update_and_fetch(keys, values)
    states = flatten_cache_state(cache)
    # KVCache.state is (keys, values); flattened result must be two
    # arrays, not one tuple.
    assert len(states) == 2
    for s in states:
        assert hasattr(s, "shape"), f"expected mlx array, got {type(s).__name__}"


def test_snapshot_cache_returns_deepcopy_of_arrays():
    cache = [KVCache()]
    keys = mx.zeros((1, 4, 8, 16))
    values = mx.ones((1, 4, 8, 16))
    cache[0].update_and_fetch(keys, values)
    snap = snapshot_cache_for_persistence(cache, eager_eval=True)
    assert snap is not cache, "must return a new list, not the input"
    assert snap[0] is not cache[0], "must deepcopy the layer object"
    # The snapshot's arrays must still represent the same data.
    snap_keys, _ = snap[0].state
    cache_keys, _ = cache[0].state
    assert mx.allclose(snap_keys, cache_keys).item()
    # Mutating the original must not affect the snapshot.
    cache[0].update_and_fetch(mx.zeros_like(keys), mx.zeros_like(values))
    snap_keys_after, _ = snap[0].state
    assert mx.allclose(snap_keys_after, snap_keys).item(), (
        "snapshot must not see the post-snapshot update"
    )


def test_snapshot_cache_eager_eval_materializes_state():
    """eager_eval=True should materialize state so cross-thread eval is safe."""
    cache = [KVCache()]
    keys = mx.zeros((1, 4, 8, 16))
    values = mx.ones((1, 4, 8, 16))
    cache[0].update_and_fetch(keys, values)
    snap = snapshot_cache_for_persistence(cache, eager_eval=True)
    snap_keys, snap_values = snap[0].state
    err: list[Exception] = []

    def read_in_thread() -> None:
        try:
            mx.eval(snap_keys)
            mx.eval(snap_values)
        except Exception as e:  # pragma: no cover
            err.append(e)

    t = threading.Thread(target=read_in_thread)
    t.start()
    t.join()
    assert not err, f"cross-thread eval failed: {err}"


def test_cached_prompt_state_defaults_match_pre_checkpoint_behavior():
    """Existing call sites that pass only tokens+cache get assistant terminal."""
    from olmlx.engine.prompt_cache.state import CachedPromptState

    state = CachedPromptState(tokens=[1, 2, 3], cache=[])
    assert state.cache_type == "assistant"
    assert state.is_checkpoint is False


def test_cached_prompt_state_can_be_marked_as_checkpoint():
    """New fields allow marking a state as a checkpoint with explicit role."""
    from olmlx.engine.prompt_cache.state import CachedPromptState

    state = CachedPromptState(
        tokens=[1, 2, 3], cache=[], cache_type="system", is_checkpoint=True
    )
    assert state.cache_type == "system"
    assert state.is_checkpoint is True


# ---------------------------------------------------------------------------
# TurboQuant / SpectralQuant snapshot path (gh #284/#343 + KV-quant unblock)
# ---------------------------------------------------------------------------


def _make_turboquant_cache(head_dim: int = 128, bits: int = 4):
    from olmlx.engine.turboquant import TurboQuantRotation
    from olmlx.engine.turboquant_cache import TurboQuantKVCache

    rot_k = TurboQuantRotation(head_dim=head_dim, seed=0)
    rot_v = TurboQuantRotation(head_dim=head_dim, seed=1)
    return TurboQuantKVCache(bits=bits, rotation_key=rot_k, rotation_value=rot_v)


def _drive_turboquant_update(cache, *, B=1, H=2, T=8, head_dim=128, dtype=mx.float16):
    keys = mx.random.normal((B, H, T, head_dim)).astype(dtype)
    values = mx.random.normal((B, H, T, head_dim)).astype(dtype)
    mx.eval(keys, values)
    return cache.update_and_fetch(keys, values)


def test_turboquant_cache_deepcopies_after_first_update():
    """After ``update_and_fetch`` locks ``_dequant_dtype`` to an ``mx.Dtype``,
    ``copy.deepcopy`` must still succeed. Pre-fix this raises
    ``TypeError: cannot pickle 'Dtype' object`` because the default deepcopy
    walks ``__dict__`` and ``mx.Dtype`` has no ``__reduce__``."""
    import copy

    cache = _make_turboquant_cache()
    _drive_turboquant_update(cache)
    snap = copy.deepcopy(cache)
    assert snap is not cache
    assert snap._dequant_dtype is cache._dequant_dtype, (
        "mx.Dtype is an immutable singleton; the copy should share the "
        "reference rather than attempt to deep-copy (which fails)."
    )
    assert snap.offset == cache.offset


def test_snapshot_turboquant_cache_preserves_state_independently():
    """``snapshot_cache_for_persistence`` on a TurboQuant-only cache list
    must produce a deepcopy whose subsequent updates do not bleed into the
    original. Covers the typical mixed Rotating+TQ layout's TQ layers."""
    cache = [_make_turboquant_cache()]
    _drive_turboquant_update(cache[0])
    pre_keys, pre_values = cache[0].update_and_fetch(
        mx.zeros((1, 2, 1, 128), dtype=mx.float16),
        mx.zeros((1, 2, 1, 128), dtype=mx.float16),
    )
    mx.eval(pre_keys, pre_values)
    pre_offset = cache[0].offset

    snap = snapshot_cache_for_persistence(cache, eager_eval=False)
    assert snap is not cache
    assert snap[0] is not cache[0]
    assert snap[0].offset == pre_offset

    # Mutate the original; snapshot must not see it.
    _drive_turboquant_update(cache[0], T=4)
    assert cache[0].offset == pre_offset + 4
    assert snap[0].offset == pre_offset, (
        "snapshot must not see post-snapshot updates on the original"
    )


def test_turboquant_deepcopy_walks_dict_for_array_attributes():
    """``__deepcopy__`` walks ``self.__dict__`` for ``mx.array`` attributes
    rather than hard-coding the buffer names, so a hypothetical future
    array attribute on the cache is covered by the eager-eval pass
    automatically. Regression guard against re-introducing a hard-coded
    list that silently misses new attributes — the #284 hazard would
    return for any buffer the list forgot."""
    import copy

    cache = _make_turboquant_cache()
    _drive_turboquant_update(cache)
    # Simulate a future buffer added by a refactor.  Use a real mx.array
    # so the dynamic ``isinstance`` filter must pick it up.
    extra = mx.zeros((4, 8), dtype=mx.float16)
    cache._future_extra_buffer = extra  # type: ignore[attr-defined]
    snap = copy.deepcopy(cache)
    assert hasattr(snap, "_future_extra_buffer")
    assert isinstance(snap._future_extra_buffer, mx.array)
    assert snap._future_extra_buffer is not extra, (
        "the dynamic walk must produce an independent copy of the new "
        "attribute, not share the reference"
    )
    assert snap._future_extra_buffer.shape == extra.shape


def test_snapshot_turboquant_cache_safe_in_other_thread():
    """A snapshot taken on this thread must be readable from a worker
    thread without re-evaluating any lazy graph bound to the originating
    Metal stream — the #284 hazard generalised to TQ's side buffers."""
    cache = [_make_turboquant_cache()]
    keys_out, values_out = _drive_turboquant_update(cache[0])
    mx.eval(keys_out, values_out)

    snap = snapshot_cache_for_persistence(cache, eager_eval=True)
    state = snap[0].state
    err: list[Exception] = []

    def _read() -> None:
        try:
            for arr in state:
                mx.eval(arr)
        except Exception as e:  # pragma: no cover
            err.append(e)

    t = threading.Thread(target=_read)
    t.start()
    t.join()
    assert not err, f"cross-thread eval failed: {err}"


def _make_spectralquant_cache(head_dim: int = 32, d_eff: int = 4):
    """Minimal SpectralQuantKVCache with synthetic-but-valid calibration."""
    import numpy as np

    from olmlx.engine.spectralquant import SpectralRotation, fit_codebook
    from olmlx.engine.spectralquant_cache import SpectralQuantKVCache

    bits_high, bits_low = 4, 2
    rng = np.random.RandomState(42)
    q, _ = np.linalg.qr(rng.randn(head_dim, head_dim).astype(np.float32))
    rotation_k = SpectralRotation(mx.array(q))
    q2, _ = np.linalg.qr(rng.randn(head_dim, head_dim).astype(np.float32))
    rotation_v = SpectralRotation(mx.array(q2))

    data = mx.random.normal((500, head_dim))
    norms = mx.linalg.norm(data, axis=-1, keepdims=True)
    data_n = data / mx.maximum(norms, mx.array(1e-8))
    rotated = rotation_k.rotate(data_n)
    cb_sem = fit_codebook(rotated[..., :d_eff].reshape(-1), bits=bits_high)
    cb_tail = fit_codebook(rotated[..., d_eff:].reshape(-1), bits=bits_low)

    return SpectralQuantKVCache(
        rotation_key=rotation_k,
        rotation_value=rotation_v,
        codebook_sem_key=cb_sem,
        codebook_tail_key=cb_tail,
        codebook_sem_value=cb_sem,
        codebook_tail_value=cb_tail,
        d_eff=d_eff,
        bits_high=bits_high,
        bits_low=bits_low,
    )


def _make_shardquant_cache(D: int = 16, H: int = 2, sink: int = 4, window: int = 8):
    """Minimal ShardKVCache with synthetic-but-valid calibration."""
    import numpy as np

    from olmlx.engine.shardquant import fit_vq_codebooks, make_v_rotation
    from olmlx.engine.shardquant_cache import ShardKVCache
    from olmlx.engine.spectralquant import fit_codebook

    bits = 4
    rng = np.random.RandomState(0)
    basis = mx.array(
        np.stack(
            [np.linalg.qr(rng.randn(D, D).astype(np.float32))[0] for _ in range(H)]
        )
    )
    k_codebook = fit_codebook(
        mx.array(rng.randn(4096).astype(np.float32) * 0.3), bits=bits
    )
    v_rot = make_v_rotation(D)
    sample = rng.randn(4096, D).astype(np.float32)
    sample /= np.linalg.norm(sample, axis=-1, keepdims=True)
    v_codebooks = fit_vq_codebooks(sample @ np.array(v_rot).T, group_size=8 // bits)
    return ShardKVCache(
        rope_spec=None,
        k_basis=basis,
        k_rank=D,
        k_codebook=k_codebook,
        k_bits=bits,
        v_rotation=v_rot,
        v_codebooks=v_codebooks,
        sink_size=sink,
        window_size=window,
    )


def test_snapshot_spectralquant_cache_safe_in_other_thread():
    """A spectral snapshot taken on this thread must be readable from a
    worker thread — same contract as the TurboQuant test above. Spectral's
    ``.state`` property rebuilds ``[..., :offset, :]`` slices off the
    step-aligned (256) backing buffers on every access, so without pinning
    the snapshot to exactly ``offset`` on the creating thread, a
    cross-thread read hits the thread-local-streams partial-slice hazard
    (#499). Reachable in production via ``_SpecCacheStore`` reuse and the
    checkpoint store — ``_KV_QUANT_PREFIXES_BLOCKING_SNAPSHOT`` is empty,
    so spectral caches flow through the in-memory snapshot path (only the
    DISK path is gated by ``_is_serializable_cache``)."""
    cache = [_make_spectralquant_cache()]
    head_dim = 32
    keys = mx.random.normal((1, 2, 8, head_dim)).astype(mx.float16)
    values = mx.random.normal((1, 2, 8, head_dim)).astype(mx.float16)
    mx.eval(keys, values)
    k_out, v_out = cache[0].update_and_fetch(keys, values)
    mx.eval(k_out, v_out)

    snap = snapshot_cache_for_persistence(cache, eager_eval=True)
    state = snap[0].state
    err: list[Exception] = []

    def _read() -> None:
        try:
            for arr in state:
                mx.eval(arr)
        except Exception as e:  # pragma: no cover
            err.append(e)

    t = threading.Thread(target=_read)
    t.start()
    t.join()
    assert not err, f"cross-thread eval failed: {err}"


def test_snapshot_shardquant_cache_safe_in_other_thread():
    """A shard snapshot taken on this thread must be readable from a worker
    thread — same contract as the TurboQuant/Spectral tests. Shard's
    ``.state`` returns the sink/window buffers raw (full-range, safe) but
    slices the compressed middle ``[..., :_mid_len, :]`` off step-aligned
    buffers, so the middle needs the same creating-thread pin (#499)."""
    cache = [_make_shardquant_cache()]
    D, H = 16, 2
    # 20 tokens: sink takes 4, window keeps 8, overflow of 8 is compressed
    # into the middle — so all three regions (and the partial-slice hazard)
    # are populated.
    keys = mx.random.normal((1, H, 20, D)).astype(mx.float16)
    values = mx.random.normal((1, H, 20, D)).astype(mx.float16)
    mx.eval(keys, values)
    k_out, v_out = cache[0].update_and_fetch(keys, values)
    mx.eval(k_out, v_out)
    assert cache[0]._mid_len > 0, "test setup must populate the compressed middle"

    snap = snapshot_cache_for_persistence(cache, eager_eval=True)
    state = snap[0].state
    err: list[Exception] = []

    def _read() -> None:
        try:
            for arr in state:
                mx.eval(arr)
        except Exception as e:  # pragma: no cover
            err.append(e)

    t = threading.Thread(target=_read)
    t.start()
    t.join()
    assert not err, f"cross-thread eval failed: {err}"


def test_snapshot_spectralquant_cache_deepcopies():
    """Regression guard: ``SpectralQuantKVCache`` has no ``mx.Dtype`` attr
    and already deepcopies cleanly. It was excluded from the checkpoint
    path defensively by analogy with the (unrelated) disk-save block."""
    import copy

    from olmlx.engine.spectralquant import SpectralRotation
    from olmlx.engine.spectralquant_cache import SpectralQuantKVCache

    head_dim = 64
    V = mx.eye(head_dim)
    sem = mx.zeros((16,), dtype=mx.float32)
    tail = mx.zeros((4,), dtype=mx.float32)
    mx.eval(V, sem, tail)

    sq = SpectralQuantKVCache(
        rotation_key=SpectralRotation(V),
        rotation_value=SpectralRotation(V),
        codebook_sem_key=sem,
        codebook_tail_key=tail,
        codebook_sem_value=sem,
        codebook_tail_value=tail,
        d_eff=head_dim // 2,
        bits_high=8,
        bits_low=2,
    )
    snap = copy.deepcopy(sq)
    assert snap is not sq
    snap_via_path = snapshot_cache_for_persistence([sq], eager_eval=False)
    assert snap_via_path[0] is not sq
