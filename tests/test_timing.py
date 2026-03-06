"""Tests for mlx_ollama.utils.timing."""

import time

from mlx_ollama.utils.timing import Timer, TimingStats


class TestTimingStats:
    def test_defaults(self):
        stats = TimingStats()
        assert stats.total_duration == 0
        assert stats.load_duration == 0
        assert stats.prompt_eval_count == 0
        assert stats.prompt_eval_duration == 0
        assert stats.eval_count == 0
        assert stats.eval_duration == 0

    def test_to_dict(self):
        stats = TimingStats(
            total_duration=100,
            load_duration=10,
            prompt_eval_count=5,
            prompt_eval_duration=20,
            eval_count=50,
            eval_duration=70,
        )
        d = stats.to_dict()
        assert d == {
            "total_duration": 100,
            "load_duration": 10,
            "prompt_eval_count": 5,
            "prompt_eval_duration": 20,
            "eval_count": 50,
            "eval_duration": 70,
        }

    def test_to_dict_keys(self):
        stats = TimingStats()
        d = stats.to_dict()
        assert set(d.keys()) == {
            "total_duration",
            "load_duration",
            "prompt_eval_count",
            "prompt_eval_duration",
            "eval_count",
            "eval_duration",
        }


class TestTimer:
    def test_duration_ns(self):
        with Timer() as t:
            time.sleep(0.01)
        assert t.duration_ns > 0
        assert t.duration_ns >= 10_000_000  # at least 10ms in ns

    def test_context_manager_returns_self(self):
        timer = Timer()
        result = timer.__enter__()
        assert result is timer
        timer.__exit__(None, None, None)

    def test_initial_values(self):
        t = Timer()
        assert t._start == 0
        assert t._end == 0
        assert t.duration_ns == 0
