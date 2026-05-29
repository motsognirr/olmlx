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
