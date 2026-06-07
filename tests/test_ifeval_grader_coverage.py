"""Regression coverage for the vendored IFEval verifiable-constraint grader.

Each test feeds a crafted string through ``grade("ifeval", ...)`` and asserts
the pass/fail outcome for one grader branch. The goal is to pin down the exact
behaviour of every constraint check (and every ``_compare`` relation) so future
edits to ``olmlx.bench.ifeval_grader`` can't silently change the pass rate.
"""

from __future__ import annotations

from olmlx.bench.quality import grade


def _grade(output: str, iid: str, kw: dict | None = None):
    return grade(
        "ifeval",
        output,
        {"instruction_id_list": [iid], "kwargs": [kw or {}]},
    )


class TestKeywordsForbidden:
    def test_pass_when_absent(self):
        r = _grade(
            "a clean reply", "keywords:forbidden_words", {"forbidden_words": ["nope"]}
        )
        assert r.passed is True

    def test_fail_case_insensitive(self):
        # "Banana" present though forbidden is lowercase -> casefold match -> fail.
        r = _grade(
            "I ate a Banana",
            "keywords:forbidden_words",
            {"forbidden_words": ["banana"]},
        )
        assert r.passed is False


class TestKeywordsFrequency:
    def test_at_least_pass(self):
        r = _grade(
            "go go go",
            "keywords:frequency",
            {"keyword": "go", "relation": "at least", "frequency": 3},
        )
        assert r.passed is True

    def test_at_least_fail(self):
        r = _grade(
            "go go",
            "keywords:frequency",
            {"keyword": "go", "relation": "at least", "frequency": 3},
        )
        assert r.passed is False

    def test_case_insensitive_count(self):
        # findall with re.IGNORECASE counts both cases.
        r = _grade(
            "Cat cat CAT",
            "keywords:frequency",
            {"keyword": "cat", "relation": "equal to", "frequency": 3},
        )
        assert r.passed is True


class TestNumberSentences:
    def test_counts_sentence_terminators(self):
        r = _grade(
            "One. Two! Three?",
            "length_constraints:number_sentences",
            {"num_sentences": 3, "relation": "equal to"},
        )
        assert r.passed is True

    def test_more_than_fail(self):
        r = _grade(
            "Only one sentence.",
            "length_constraints:number_sentences",
            {"num_sentences": 1, "relation": "more than"},
        )
        assert r.passed is False


class TestNumberParagraphs:
    def test_double_newline_split(self):
        r = _grade(
            "para one\n\npara two\n\npara three",
            "length_constraints:number_paragraphs",
            {"num_paragraphs": 3, "relation": "equal to"},
        )
        assert r.passed is True

    def test_at_most_fail(self):
        r = _grade(
            "a\n\nb\n\nc",
            "length_constraints:number_paragraphs",
            {"num_paragraphs": 2, "relation": "at most"},
        )
        assert r.passed is False


class TestNthParagraphFirstWord:
    def test_match(self):
        r = _grade(
            "First para.\n\nSecond starts here.",
            "length_constraints:nth_paragraph_first_word",
            {"nth_paragraph": 2, "first_word": "Second"},
        )
        assert r.passed is True

    def test_out_of_range_fails(self):
        r = _grade(
            "only one para",
            "length_constraints:nth_paragraph_first_word",
            {"nth_paragraph": 5, "first_word": "only"},
        )
        assert r.passed is False

    def test_strips_punctuation_and_casefolds(self):
        r = _grade(
            '"Hello, there.',
            "length_constraints:nth_paragraph_first_word",
            {"nth_paragraph": 1, "first_word": "hello"},
        )
        assert r.passed is True


class TestPostscript:
    def test_present(self):
        r = _grade("Body text\nP.S. extra", "detectable_content:postscript")
        assert r.passed is True

    def test_custom_marker_absent(self):
        r = _grade(
            "no marker", "detectable_content:postscript", {"postscript_marker": "PPS"}
        )
        assert r.passed is False


class TestNumberPlaceholders:
    def test_meets_target(self):
        r = _grade(
            "Dear [name], your [item] is ready.",
            "detectable_content:number_placeholders",
            {"num_placeholders": 2},
        )
        assert r.passed is True

    def test_below_target_fails(self):
        r = _grade(
            "Dear [name].",
            "detectable_content:number_placeholders",
            {"num_placeholders": 2},
        )
        assert r.passed is False


class TestBulletLists:
    def test_exact_count(self):
        r = _grade(
            "- one\n- two\n* three",
            "detectable_format:number_bullet_lists",
            {"num_bullets": 3},
        )
        assert r.passed is True

    def test_wrong_count_fails(self):
        # Requires exact equality, not >=.
        r = _grade(
            "- one\n- two",
            "detectable_format:number_bullet_lists",
            {"num_bullets": 1},
        )
        assert r.passed is False


class TestJsonFormat:
    def test_plain_json(self):
        r = _grade('{"a": 1}', "detectable_format:json_format")
        assert r.passed is True

    def test_fenced_json(self):
        r = _grade('```json\n{"a": 1}\n```', "detectable_format:json_format")
        assert r.passed is True

    def test_invalid_json_fails(self):
        r = _grade("not json at all", "detectable_format:json_format")
        assert r.passed is False


class TestTitle:
    def test_angle_title_present(self):
        r = _grade("<<My Title>>\nbody", "detectable_format:title")
        assert r.passed is True

    def test_no_title_fails(self):
        r = _grade("just text", "detectable_format:title")
        assert r.passed is False


class TestConstrainedResponse:
    def test_allowed_option(self):
        r = _grade("My answer is maybe.", "detectable_format:constrained_response")
        assert r.passed is True

    def test_disallowed_fails(self):
        r = _grade("Sure thing.", "detectable_format:constrained_response")
        assert r.passed is False


class TestStartEndQuotation:
    def test_wrapped_in_quotes(self):
        r = _grade('  "fully quoted"  ', "startend:quotation")
        assert r.passed is True

    def test_unquoted_fails(self):
        r = _grade('"only opens', "startend:quotation")
        assert r.passed is False


class TestEndChecker:
    def test_trailing_whitespace_tolerated(self):
        r = _grade(
            "ends with marker END  \n", "startend:end_checker", {"end_phrase": "END"}
        )
        assert r.passed is True

    def test_wrong_end_fails(self):
        r = _grade("ends differently", "startend:end_checker", {"end_phrase": "END"})
        assert r.passed is False


class TestChangeCase:
    def test_all_caps_pass(self):
        r = _grade("HELLO WORLD 123!", "change_case:english_capital")
        assert r.passed is True

    def test_all_caps_fail_on_lower(self):
        r = _grade("HELLO world", "change_case:english_capital")
        assert r.passed is False

    def test_no_letters_fails_caps(self):
        # bool(letters) guard: a string with no letters cannot be "all capital".
        r = _grade("12345 !!!", "change_case:english_capital")
        assert r.passed is False

    def test_all_lower_pass(self):
        r = _grade("hello world 123", "change_case:english_lowercase")
        assert r.passed is True

    def test_all_lower_fail_on_upper(self):
        r = _grade("hello World", "change_case:english_lowercase")
        assert r.passed is False


class TestCapitalWordFrequency:
    def test_counts_all_caps_words(self):
        r = _grade(
            "NASA and FBI met",
            "change_case:capital_word_frequency",
            {"capital_frequency": 2, "capital_relation": "at least"},
        )
        assert r.passed is True

    def test_single_letter_not_counted(self):
        # Regex requires 2+ uppercase letters, so "A" is not a capital word.
        r = _grade(
            "A lone letter",
            "change_case:capital_word_frequency",
            {"capital_frequency": 1, "capital_relation": "at least"},
        )
        assert r.passed is False


class TestCombination:
    def test_two_responses_separator(self):
        r = _grade("first\n******\nsecond", "combination:two_responses")
        assert r.passed is True

    def test_two_responses_missing(self):
        r = _grade("first\nsecond", "combination:two_responses")
        assert r.passed is False

    def test_repeat_prompt_present(self):
        r = _grade(
            "Write a poem. Then the poem.",
            "combination:repeat_prompt",
            {"prompt_to_repeat": "Write a poem."},
        )
        assert r.passed is True

    def test_repeat_prompt_empty_kwarg_fails(self):
        # Empty/missing prompt_to_repeat is treated as malformed -> fail
        # (not an unconditional pass via "" in str).
        r = _grade("anything goes", "combination:repeat_prompt", {})
        assert r.passed is False


class TestLanguageAndUnknown:
    def test_no_instruction_list_ungraded(self):
        r = grade("ifeval", "text", {"instruction_id_list": [], "kwargs": []})
        assert r.passed is None
        assert r.score is None
        assert r.detail == "no instruction_id_list"

    def test_unknown_constraint_ungraded(self):
        r = _grade("text", "language:response_language", {"language": "fr"})
        # language:response_language is not in CONSTRAINT_CHECKS -> ungraded.
        assert r.passed is None
        assert r.score is None
        assert "unknown constraint" in r.detail


class TestCompareRelations:
    def test_less_than(self):
        r = _grade(
            "one two three",
            "length_constraints:number_words",
            {"num_words": 5, "relation": "less than"},
        )
        assert r.passed is True

    def test_at_most_boundary(self):
        r = _grade(
            "one two three four five",
            "length_constraints:number_words",
            {"num_words": 5, "relation": "at most"},
        )
        assert r.passed is True

    def test_unknown_relation_fails(self):
        # _compare returns False for any relation it doesn't recognise.
        r = _grade(
            "one two three",
            "length_constraints:number_words",
            {"num_words": 3, "relation": "approximately"},
        )
        assert r.passed is False


class TestScoringAggregation:
    def test_partial_score(self):
        r = grade(
            "ifeval",
            "banana, and more",
            {
                "instruction_id_list": ["keywords:existence", "punctuation:no_comma"],
                "kwargs": [{"keywords": ["banana"]}, {}],
            },
        )
        # keyword passes, comma fails -> 1/2.
        assert r.passed is False
        assert r.score == 0.5
        assert "keywords:existence=ok" in r.detail
        assert "punctuation:no_comma=fail" in r.detail
