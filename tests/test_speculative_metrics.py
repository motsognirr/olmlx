from olmlx.utils import metrics


class _FakeDecoder:
    # Class name not in the strategy map -> "unknown" label.
    def stats_summary(self):
        return {"proposed": 12, "accepted_draft": 9, "steps": 3}


def test_log_stats_records_speculative_counts(monkeypatch):
    recorded = []
    monkeypatch.setattr(
        metrics,
        "record_speculative",
        lambda decoder, proposed, accepted: recorded.append((proposed, accepted)),
    )
    from olmlx.engine import speculative_stream

    decoder = _FakeDecoder()
    speculative_stream._emit_speculative_metrics(decoder)
    assert recorded == [(12, 9)]
