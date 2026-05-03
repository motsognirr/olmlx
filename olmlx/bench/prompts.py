"""Standard prompt sets for benchmarking."""

from __future__ import annotations

from dataclasses import dataclass

# ~500-char paragraph repeated to build a long-context prompt. Repetition is
# fine for microbenching attention/KV-cache cost — content variety doesn't
# meaningfully change prefill or decode throughput, but determinism does.
_LONG_CONTEXT_PARAGRAPH = (
    "The migratory patterns of Arctic terns span the entire globe, with these "
    "remarkable birds traveling from breeding grounds near the North Pole to "
    "wintering sites in Antarctica. A single tern can cover more than seventy "
    "thousand kilometers in a year, navigating by a combination of celestial "
    "cues, magnetic fields, and learned coastal landmarks. Researchers have "
    "tagged individuals to study route choice, fueling stops, and longevity, "
    "uncovering surprising fidelity to specific staging islands across decades. "
    "The species' endurance is matched only by its sensitivity to changes in "
    "sea-ice extent and prey availability, making the tern an unusually clear "
    "indicator of climate-driven shifts at both polar extremes. "
)
# Target ~70_000 characters → roughly 16-18k tokens with a typical BPE
# tokenizer. Truncated to a fixed length so the prompt is identical across
# runs regardless of paragraph length tweaks.
_LONG_CONTEXT_BODY = (
    _LONG_CONTEXT_PARAGRAPH * (70_000 // len(_LONG_CONTEXT_PARAGRAPH) + 1)
)[:70_000]


@dataclass(frozen=True)
class BenchPrompt:
    name: str
    category: str
    messages: list[dict[str, str]]
    max_tokens: int = 256

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "category": self.category,
            "messages": self.messages,
            "max_tokens": self.max_tokens,
        }

    @classmethod
    def from_dict(cls, d: dict) -> BenchPrompt:
        return cls(
            name=d["name"],
            category=d["category"],
            messages=d["messages"],
            max_tokens=d.get("max_tokens", 256),
        )


PROMPTS: list[BenchPrompt] = [
    BenchPrompt(
        name="factual",
        category="factual",
        messages=[
            {
                "role": "user",
                "content": "What is the capital of France? Answer in one sentence.",
            },
        ],
        max_tokens=64,
    ),
    BenchPrompt(
        name="reasoning",
        category="reasoning",
        messages=[
            {
                "role": "user",
                "content": (
                    "If all roses are flowers and some flowers fade quickly, "
                    "can we conclude that some roses fade quickly? Explain briefly."
                ),
            },
        ],
        max_tokens=256,
    ),
    BenchPrompt(
        name="coding",
        category="coding",
        messages=[
            {
                "role": "user",
                "content": "Write a Python function that checks if a string is a palindrome.",
            },
        ],
        max_tokens=256,
    ),
    BenchPrompt(
        name="creative",
        category="creative",
        messages=[
            {"role": "user", "content": "Write a haiku about programming."},
        ],
        max_tokens=64,
    ),
    BenchPrompt(
        name="instruction",
        category="instruction",
        messages=[
            {
                "role": "user",
                "content": "List exactly 3 benefits of exercise. Use numbered format.",
            },
        ],
        max_tokens=128,
    ),
    BenchPrompt(
        name="multi-turn",
        category="multi-turn",
        messages=[
            {"role": "user", "content": "What is the Fibonacci sequence?"},
            {
                "role": "assistant",
                "content": "The Fibonacci sequence is a series where each number is the sum of the two preceding ones: 0, 1, 1, 2, 3, 5, 8, 13, ...",
            },
            {
                "role": "user",
                "content": "Write a Python function to compute the nth Fibonacci number.",
            },
        ],
        max_tokens=256,
    ),
    BenchPrompt(
        name="long-context",
        category="long-context",
        messages=[
            {
                "role": "user",
                "content": (
                    "Below is a long passage. After it, answer the question "
                    "in a single sentence.\n\n"
                    f"{_LONG_CONTEXT_BODY}\n\n"
                    "Question: What is the recurring subject of the passage?"
                ),
            },
        ],
        max_tokens=64,
    ),
]
