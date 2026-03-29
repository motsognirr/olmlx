"""Standard prompt sets for benchmarking."""

from __future__ import annotations

from dataclasses import dataclass


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
]
