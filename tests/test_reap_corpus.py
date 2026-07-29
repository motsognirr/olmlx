from __future__ import annotations

import pytest

from olmlx.engine.reap.corpus import (
    BUILTIN_SOURCES,
    build_calibration_corpus,
    cjk_fraction,
    synthetic_source_texts,
)


class TestCjkFraction:
    def test_pure_ascii(self):
        assert cjk_fraction("hello world") == 0.0

    def test_pure_cjk(self):
        assert cjk_fraction("你好世界") == 1.0

    def test_mixed(self):
        assert 0.4 < cjk_fraction("abc你好中文字") < 0.8

    def test_empty(self):
        assert cjk_fraction("") == 0.0


class _FakeStream:
    """Stands in for datasets.load_dataset(..., streaming=True)."""

    def __init__(self, rows):
        self.rows = rows
        self.closed = False

    def __iter__(self):
        return iter(self.rows)

    def close(self):
        self.closed = True


def _fake_load_dataset(rows_by_key):
    def load_dataset(dataset, config=None, split=None, streaming=False, data_dir=None):
        return _FakeStream(rows_by_key[(dataset, config)])

    return load_dataset


class TestBuildCorpus:
    def _patch(self, monkeypatch, rows_by_key):
        import datasets

        monkeypatch.setattr(datasets, "load_dataset", _fake_load_dataset(rows_by_key))

    def test_interleaved_and_tagged(self, monkeypatch):
        en = [{"text": f"english document number {i} " * 10} for i in range(5)]
        code = [{"content": f"def fn_{i}(): return {i} " * 10} for i in range(5)]
        self._patch(
            monkeypatch,
            {
                ("allenai/c4", "en"): en,
                (BUILTIN_SOURCES["code"].dataset, BUILTIN_SOURCES["code"].config): code,
            },
        )
        out = build_calibration_corpus(["english", "code"], 3)
        assert len(out) == 6
        # round-robin: e, c, e, c, e, c
        assert [s for s, _ in out] == ["english", "code"] * 3
        assert "english document" in out[0][1] and "def fn_" in out[1][1]

    def test_min_chars_filter(self, monkeypatch):
        rows = [{"text": "short"}] + [{"text": "x" * 200}] * 3
        self._patch(monkeypatch, {("allenai/c4", "en"): rows})
        out = build_calibration_corpus(["english"], 2)
        assert all(len(t) >= 100 for _, t in out)

    def test_cjk_filter_for_chinese(self, monkeypatch):
        mojibake = {"text": "a" * 200}  # claims zh, no CJK
        good = {"text": "中文测试" * 60}
        self._patch(monkeypatch, {("allenai/c4", "zh"): [mojibake, good, good]})
        out = build_calibration_corpus(["chinese"], 2)
        assert len(out) == 2
        assert all(cjk_fraction(t) >= 0.3 for _, t in out)

    def test_skip_reserves_held_out(self, monkeypatch):
        rows = [{"text": f"document number {i:03d} " * 10} for i in range(10)]
        self._patch(monkeypatch, {("allenai/c4", "en"): rows})
        cal = build_calibration_corpus(["english"], 3)
        held = build_calibration_corpus(["english"], 2, skip=3)
        cal_texts = {t for _, t in cal}
        assert all(t not in cal_texts for _, t in held)

    def test_char_cap(self, monkeypatch):
        rows = [{"text": "y" * 10_000}] * 2
        self._patch(monkeypatch, {("allenai/c4", "en"): rows})
        out = build_calibration_corpus(["english"], 2, char_cap=1000)
        assert all(len(t) == 1000 for _, t in out)

    def test_fallback_to_synthetic_on_error(self, monkeypatch):
        import datasets

        def boom(*a, **k):
            raise ConnectionError("offline")

        monkeypatch.setattr(datasets, "load_dataset", boom)
        out = build_calibration_corpus(["english"], 4)
        assert len(out) == 4 and all(s == "english" for s, _ in out)

    def test_unknown_source_raises(self):
        with pytest.raises(ValueError, match="klingon"):
            build_calibration_corpus(["klingon"], 2)


class TestSynthetic:
    def test_deterministic_and_distinct(self):
        a = synthetic_source_texts("code", 5)
        b = synthetic_source_texts("code", 5)
        assert a == b and len(set(a)) == 5
        assert synthetic_source_texts("english", 3) != synthetic_source_texts("code", 3)
