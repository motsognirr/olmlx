"""Tests for olmlx.bench.prompts."""

from olmlx.bench.prompts import PROMPTS, BenchPrompt


class TestBenchPrompt:
    def test_to_dict_roundtrip(self):
        p = BenchPrompt(
            name="test",
            category="unit",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=100,
        )
        d = p.to_dict()
        restored = BenchPrompt.from_dict(d)
        assert restored.name == p.name
        assert restored.category == p.category
        assert restored.messages == p.messages
        assert restored.max_tokens == p.max_tokens

    def test_from_dict_default_max_tokens(self):
        p = BenchPrompt.from_dict({"name": "x", "category": "y", "messages": []})
        assert p.max_tokens == 256


class TestPromptsList:
    def test_has_prompts(self):
        assert len(PROMPTS) >= 6

    def test_all_have_required_fields(self):
        for p in PROMPTS:
            assert p.name, "Missing name"
            assert p.category, f"Missing category for {p.name}"
            assert len(p.messages) > 0, f"No messages for {p.name}"
            assert p.max_tokens > 0, f"Invalid max_tokens for {p.name}"

    def test_all_messages_have_role_and_content(self):
        for p in PROMPTS:
            for msg in p.messages:
                assert "role" in msg, f"Missing role in {p.name}"
                assert "content" in msg, f"Missing content in {p.name}"

    def test_unique_names(self):
        names = [p.name for p in PROMPTS]
        assert len(names) == len(set(names))

    def test_categories_covered(self):
        cats = {p.category for p in PROMPTS}
        for expected in (
            "factual",
            "reasoning",
            "coding",
            "creative",
            "instruction",
            "multi-turn",
        ):
            assert expected in cats, f"Missing category {expected}"

    def test_multi_turn_has_multiple_messages(self):
        multi = [p for p in PROMPTS if p.category == "multi-turn"]
        assert len(multi) > 0
        assert len(multi[0].messages) >= 3
