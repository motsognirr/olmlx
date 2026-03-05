import time
from dataclasses import dataclass, field


@dataclass
class TimingStats:
    total_duration: int = 0
    load_duration: int = 0
    prompt_eval_count: int = 0
    prompt_eval_duration: int = 0
    eval_count: int = 0
    eval_duration: int = 0

    def to_dict(self) -> dict:
        return {
            "total_duration": self.total_duration,
            "load_duration": self.load_duration,
            "prompt_eval_count": self.prompt_eval_count,
            "prompt_eval_duration": self.prompt_eval_duration,
            "eval_count": self.eval_count,
            "eval_duration": self.eval_duration,
        }


class Timer:
    """Nanosecond timer context manager."""

    def __init__(self):
        self._start: int = 0
        self._end: int = 0

    def __enter__(self):
        self._start = time.perf_counter_ns()
        return self

    def __exit__(self, *exc):
        self._end = time.perf_counter_ns()

    @property
    def duration_ns(self) -> int:
        return self._end - self._start
