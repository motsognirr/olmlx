from olmlx.engine.prompt_cache.checkpoint import SegmentedPrompt, Segment


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
