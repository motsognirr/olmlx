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
