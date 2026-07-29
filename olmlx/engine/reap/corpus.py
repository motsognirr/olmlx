"""Calibration corpus builder: per-source tagging, CJK filter, round-robin interleave.

Ports the kimi-k3-mlx corpus lessons: per-language C4 configs (pooled multilingual
C4 is ~97% Latin script), sources interleaved not concatenated (calibration reads
a prefix), and a CJK-fraction filter to drop mojibake documents.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger(__name__)

_CJK_RANGES = (
    (0x4E00, 0x9FFF),
    (0x3400, 0x4DBF),  # CJK unified + ext A
    (0x3040, 0x30FF),  # hiragana + katakana
    (0xAC00, 0xD7AF),  # hangul
)


@dataclass(frozen=True)
class SourceSpec:
    name: str
    dataset: str
    config: str | None
    split: str
    text_key: str
    min_chars: int = 100
    min_cjk_fraction: float = 0.0
    data_dir: str | None = None


BUILTIN_SOURCES: dict[str, SourceSpec] = {
    "english": SourceSpec("english", "allenai/c4", "en", "train", "text"),
    "chinese": SourceSpec(
        "chinese", "allenai/c4", "zh", "train", "text", min_cjk_fraction=0.3
    ),
    "code": SourceSpec(
        "code",
        "bigcode/the-stack-smol",
        None,
        "train",
        "content",
        data_dir="data/python",
    ),
}


def cjk_fraction(text: str) -> float:
    if not text:
        return 0.0
    hits = sum(1 for ch in text if any(lo <= ord(ch) <= hi for lo, hi in _CJK_RANGES))
    return hits / len(text)


_SYNTH_TEMPLATES = {
    "english": "The {0} of {1} is a widely studied subject in modern research. ",
    "chinese": "关于{0}与{1}的研究是现代学术领域的重要课题。",
    "code": "def process_{0}(data):\n    result = [x for x in data if x.{1}]\n    return result\n",
}
_SYNTH_TOPICS = [
    "history",
    "structure",
    "analysis",
    "theory",
    "design",
    "behavior",
    "origin",
    "measurement",
    "classification",
    "evolution",
    "dynamics",
    "composition",
]


def synthetic_source_texts(source: str, n: int) -> list[str]:
    template = _SYNTH_TEMPLATES.get(source, _SYNTH_TEMPLATES["english"])
    texts = []
    for i in range(n):
        a = _SYNTH_TOPICS[i % len(_SYNTH_TOPICS)]
        b = _SYNTH_TOPICS[(i + 7) % len(_SYNTH_TOPICS)]
        texts.append((template.format(a, b)) * 12)
    return texts


def _stream_source(spec: SourceSpec, n: int, *, skip: int, char_cap: int) -> list[str]:
    import datasets

    kwargs = {"split": spec.split, "streaming": True}
    if spec.data_dir:
        kwargs["data_dir"] = spec.data_dir
    stream = (
        datasets.load_dataset(spec.dataset, spec.config, **kwargs)
        if spec.config
        else datasets.load_dataset(spec.dataset, **kwargs)
    )
    accepted: list[str] = []
    to_skip = skip
    try:
        for row in stream:
            text = row.get(spec.text_key, "")
            if len(text) < spec.min_chars:
                continue
            if spec.min_cjk_fraction and cjk_fraction(text) < spec.min_cjk_fraction:
                continue
            if to_skip > 0:
                to_skip -= 1
                continue
            accepted.append(text[:char_cap])
            if len(accepted) >= n:
                break
    finally:
        close = getattr(stream, "close", None)
        if close is not None:
            close()
    if len(accepted) < n:
        raise RuntimeError(
            f"source {spec.name!r} yielded only {len(accepted)}/{n} samples"
        )
    return accepted


def build_calibration_corpus(
    sources: list[str],
    samples_per_source: int,
    *,
    skip: int = 0,
    char_cap: int = 4096,
    progress_callback: Callable[[str, float], None] | None = None,
) -> list[tuple[str, str]]:
    unknown = [s for s in sources if s not in BUILTIN_SOURCES]
    if unknown:
        raise ValueError(
            f"unknown calibration sources {unknown}; "
            f"available: {sorted(BUILTIN_SOURCES)}"
        )
    per_source: dict[str, list[str]] = {}
    for i, name in enumerate(sources):
        spec = BUILTIN_SOURCES[name]
        try:
            per_source[name] = _stream_source(
                spec, samples_per_source, skip=skip, char_cap=char_cap
            )
        except Exception as exc:  # noqa: BLE001 — mirror _get_c4_calibration_data
            logger.warning(
                "calibration source %s unavailable (%s); using synthetic fallback",
                name,
                exc,
            )
            per_source[name] = [
                t[:char_cap] for t in synthetic_source_texts(name, samples_per_source)
            ]
        if progress_callback:
            progress_callback(f"Fetched source {name}", (i + 1) / len(sources))
    # round-robin interleave so a prefix read still covers every source
    tagged: list[tuple[str, str]] = []
    for row in itertools.zip_longest(
        *([(name, t) for t in per_source[name]] for name in sources)
    ):
        tagged.extend(item for item in row if item is not None)
    return tagged
