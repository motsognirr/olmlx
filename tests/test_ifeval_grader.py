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
