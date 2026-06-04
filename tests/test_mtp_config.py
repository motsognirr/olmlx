from olmlx.engine.registry import _VALID_SPECULATIVE_STRATEGIES


def test_mtp_is_a_valid_strategy():
    assert "mtp" in _VALID_SPECULATIVE_STRATEGIES
