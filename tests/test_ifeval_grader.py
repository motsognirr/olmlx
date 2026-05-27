"""Tests for the vendored IFEval verifiable-constraint grader."""

from __future__ import annotations

from olmlx.bench.quality import grade


class TestKeywordsExistence:
    def test_pass(self):
        result = grade(
            "ifeval",
            "I had a banana for breakfast.",
            {
                "instruction_id_list": ["keywords:existence"],
                "kwargs": [{"keywords": ["banana"]}],
            },
        )
        assert result.passed is True
        assert result.score == 1.0

    def test_fail(self):
        result = grade(
            "ifeval",
            "I had an apple for breakfast.",
            {
                "instruction_id_list": ["keywords:existence"],
                "kwargs": [{"keywords": ["banana"]}],
            },
        )
        assert result.passed is False


class TestLengthConstraints:
    def test_min_words_pass(self):
        result = grade(
            "ifeval",
            "one two three four five",
            {
                "instruction_id_list": ["length_constraints:number_words"],
                "kwargs": [{"num_words": 5, "relation": "at least"}],
            },
        )
        assert result.passed is True

    def test_max_words_fail(self):
        result = grade(
            "ifeval",
            "one two three four five six seven",
            {
                "instruction_id_list": ["length_constraints:number_words"],
                "kwargs": [{"num_words": 5, "relation": "at most"}],
            },
        )
        assert result.passed is False


class TestPunctuation:
    def test_no_commas_pass(self):
        result = grade(
            "ifeval",
            "Hello world",
            {"instruction_id_list": ["punctuation:no_comma"], "kwargs": [{}]},
        )
        assert result.passed is True

    def test_no_commas_fail(self):
        result = grade(
            "ifeval",
            "Hello, world",
            {"instruction_id_list": ["punctuation:no_comma"], "kwargs": [{}]},
        )
        assert result.passed is False


class TestStartEnd:
    def test_end_checker_pass(self):
        result = grade(
            "ifeval",
            "My reply ends here. END",
            {
                "instruction_id_list": ["startend:end_checker"],
                "kwargs": [{"end_phrase": "END"}],
            },
        )
        assert result.passed is True


class TestMultipleConstraints:
    def test_all_must_pass(self):
        result = grade(
            "ifeval",
            "Hello banana world",
            {
                "instruction_id_list": ["keywords:existence", "punctuation:no_comma"],
                "kwargs": [{"keywords": ["banana"]}, {}],
            },
        )
        assert result.passed is True

    def test_one_fail_means_overall_fail(self):
        result = grade(
            "ifeval",
            "Hello, banana world",
            {
                "instruction_id_list": ["keywords:existence", "punctuation:no_comma"],
                "kwargs": [{"keywords": ["banana"]}, {}],
            },
        )
        assert result.passed is False
        assert result.score == 0.5


class TestUnknownConstraint:
    def test_returns_ungraded(self):
        result = grade(
            "ifeval",
            "anything",
            {"instruction_id_list": ["totally:unknown"], "kwargs": [{}]},
        )
        # Unknown constraints leave the prompt ungraded (passed=None) so they
        # don't silently inflate failure counts.
        assert result.passed is None


class TestLanguageConstraintFiltering:
    def test_language_constraint_is_unknown_in_grader(self):
        """language:response_language must be treated as ungraded (not always-False)."""
        result = grade(
            "ifeval",
            "Bonjour le monde",
            {
                "instruction_id_list": ["language:response_language"],
                "kwargs": [{"language": "fr"}],
            },
        )
        # Since language:response_language is no longer in CONSTRAINT_CHECKS,
        # it hits the unknown-constraint path and returns passed=None.
        assert result.passed is None

    def test_language_constraints_are_filtered_out_of_ifeval_load(
        self, tmp_path, monkeypatch
    ):
        """load_ifeval should drop prompts whose only constraint is language:."""
        import json

        from olmlx.bench.extended_suites import load_ifeval

        monkeypatch.setenv("OLMLX_BENCH_CACHE_DIR", str(tmp_path))
        # Write a synthetic ifeval cache with one verifiable and one language-only row.
        records = [
            {
                "key": 1,
                "prompt": "Write containing the word banana.",
                "instruction_id_list": ["keywords:existence"],
                "kwargs": [{"keywords": ["banana"]}],
            },
            {
                "key": 2,
                "prompt": "Reply in French.",
                "instruction_id_list": ["language:response_language"],
                "kwargs": [{"language": "fr"}],
            },
        ]
        (tmp_path / "ifeval.json").write_text(json.dumps(records), encoding="utf-8")
        prompts = load_ifeval(n=None)
        names = [p.name for p in prompts]
        # Only the verifiable row (key=1) should survive; language-only is filtered.
        assert len(prompts) == 1
        assert "ifeval-0001" in names
