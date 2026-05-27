"""Vendored IFEval verifiable-constraint checks.

Subset of the checks from the IFEval reference implementation
(https://github.com/google-research/google-research/tree/master/instruction_following_eval,
Apache-2.0). We only vendor the prefixes listed in
``_VERIFIABLE_IFEVAL_PREFIXES`` in extended_suites.py — those that admit a
purely-rule-based pass/fail decision. Rubric-graded constraints are not
supported and any unknown constraint returns ``passed=None``.

The grader returns a fractional ``score`` (fraction of constraints in the
list that passed) plus a boolean ``passed`` requiring all constraints to
pass. Detail field lists per-constraint outcomes for debugging.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from olmlx.bench.quality import QualityResult

ConstraintCheck = Callable[[str, dict[str, Any]], bool]


def _keywords_existence(output: str, kwargs: dict[str, Any]) -> bool:
    keywords = kwargs.get("keywords", [])
    out_low = output.casefold()
    return all(k.casefold() in out_low for k in keywords)


def _keywords_forbidden_words(output: str, kwargs: dict[str, Any]) -> bool:
    forbidden = kwargs.get("forbidden_words", [])
    out_low = output.casefold()
    return all(k.casefold() not in out_low for k in forbidden)


def _keywords_frequency(output: str, kwargs: dict[str, Any]) -> bool:
    keyword = kwargs.get("keyword", "")
    relation = kwargs.get("relation", "at least")
    target = int(kwargs.get("frequency", 0))
    actual = len(re.findall(re.escape(keyword), output, flags=re.IGNORECASE))
    return _compare(actual, target, relation)


def _length_constraints_number_words(output: str, kwargs: dict[str, Any]) -> bool:
    n_words = len(re.findall(r"\b\w+\b", output))
    return _compare(
        n_words, int(kwargs["num_words"]), kwargs.get("relation", "at least")
    )


def _length_constraints_number_sentences(output: str, kwargs: dict[str, Any]) -> bool:
    sentences = [s for s in re.split(r"[.!?]+", output) if s.strip()]
    return _compare(
        len(sentences), int(kwargs["num_sentences"]), kwargs.get("relation", "at least")
    )


def _length_constraints_nth_paragraph_first_word(
    output: str, kwargs: dict[str, Any]
) -> bool:
    paras = [p for p in output.split("\n\n") if p.strip()]
    n = int(kwargs.get("nth_paragraph", 1))
    if n < 1 or n > len(paras):
        return False
    first = paras[n - 1].strip().split(None, 1)[0].strip(".,;:!?\"'")
    return first.casefold() == str(kwargs.get("first_word", "")).casefold()


def _length_constraints_number_paragraphs(output: str, kwargs: dict[str, Any]) -> bool:
    paras = [p for p in output.split("\n\n") if p.strip()]
    return _compare(
        len(paras), int(kwargs["num_paragraphs"]), kwargs.get("relation", "at least")
    )


def _punctuation_no_comma(output: str, kwargs: dict[str, Any]) -> bool:
    return "," not in output


def _detectable_content_postscript(output: str, kwargs: dict[str, Any]) -> bool:
    marker = kwargs.get("postscript_marker", "P.S.")
    return marker in output


def _detectable_content_number_placeholders(
    output: str, kwargs: dict[str, Any]
) -> bool:
    target = int(kwargs.get("num_placeholders", 0))
    actual = len(re.findall(r"\[[^\[\]\n]+\]", output))
    return actual >= target


def _detectable_format_number_bullet_lists(output: str, kwargs: dict[str, Any]) -> bool:
    target = int(kwargs.get("num_bullets", 0))
    actual = len(re.findall(r"(?m)^\s*[-*]\s+", output))
    return actual == target


def _detectable_format_json_format(output: str, kwargs: dict[str, Any]) -> bool:
    import json

    s = output.strip()
    if s.startswith("```"):
        # Strip code fence.
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    try:
        json.loads(s)
        return True
    except json.JSONDecodeError:
        return False


def _detectable_format_title(output: str, kwargs: dict[str, Any]) -> bool:
    return bool(re.search(r"<<[^<>]+>>", output))


def _detectable_format_constrained_response(
    output: str, kwargs: dict[str, Any]
) -> bool:
    allowed = ["My answer is yes.", "My answer is no.", "My answer is maybe."]
    return any(opt in output for opt in allowed)


def _startend_end_checker(output: str, kwargs: dict[str, Any]) -> bool:
    end_phrase = kwargs.get("end_phrase", "")
    return output.rstrip().endswith(end_phrase)


def _startend_quotation(output: str, kwargs: dict[str, Any]) -> bool:
    s = output.strip()
    return s.startswith('"') and s.endswith('"')


def _change_case_english_capital(output: str, kwargs: dict[str, Any]) -> bool:
    letters = [c for c in output if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters)


def _change_case_english_lowercase(output: str, kwargs: dict[str, Any]) -> bool:
    letters = [c for c in output if c.isalpha()]
    return bool(letters) and all(c.islower() for c in letters)


def _change_case_capital_word_frequency(output: str, kwargs: dict[str, Any]) -> bool:
    target = int(kwargs.get("capital_frequency", 0))
    relation = kwargs.get("capital_relation", "at least")
    actual = len(re.findall(r"\b[A-Z]{2,}\b", output))
    return _compare(actual, target, relation)


def _combination_two_responses(output: str, kwargs: dict[str, Any]) -> bool:
    return "******" in output


def _combination_repeat_prompt(output: str, kwargs: dict[str, Any]) -> bool:
    prompt_to_repeat = kwargs.get("prompt_to_repeat", "")
    # An empty/missing kwarg used to pass unconditionally because "" is in
    # every string. Treat missing-or-empty as "constraint malformed → fail"
    # rather than silently inflating the IFEval pass rate.
    if not prompt_to_repeat:
        return False
    return prompt_to_repeat in output


def _language_response_language(output: str, kwargs: dict[str, Any]) -> bool:
    # We don't ship a language detector; treat as ungraded by returning False
    # so an incorrect-by-construction reply is never accidentally marked True.
    # Plan-stage decision: rather than report a wrong pass on language
    # constraints, we let these consistently fail. Future work could vendor a
    # cheap detector (langid) if this constraint family becomes important.
    return False


def _compare(actual: int, target: int, relation: str) -> bool:
    if relation == "at least":
        return actual >= target
    if relation == "at most":
        return actual <= target
    if relation == "less than":
        return actual < target
    if relation == "more than":
        return actual > target
    if relation == "equal to":
        return actual == target
    return False


CONSTRAINT_CHECKS: dict[str, ConstraintCheck] = {
    "keywords:existence": _keywords_existence,
    "keywords:forbidden_words": _keywords_forbidden_words,
    "keywords:frequency": _keywords_frequency,
    "length_constraints:number_words": _length_constraints_number_words,
    "length_constraints:number_sentences": _length_constraints_number_sentences,
    "length_constraints:nth_paragraph_first_word": _length_constraints_nth_paragraph_first_word,
    "length_constraints:number_paragraphs": _length_constraints_number_paragraphs,
    "punctuation:no_comma": _punctuation_no_comma,
    "detectable_content:postscript": _detectable_content_postscript,
    "detectable_content:number_placeholders": _detectable_content_number_placeholders,
    "detectable_format:number_bullet_lists": _detectable_format_number_bullet_lists,
    "detectable_format:json_format": _detectable_format_json_format,
    "detectable_format:title": _detectable_format_title,
    "detectable_format:constrained_response": _detectable_format_constrained_response,
    "startend:end_checker": _startend_end_checker,
    "startend:quotation": _startend_quotation,
    "change_case:english_capital": _change_case_english_capital,
    "change_case:english_lowercase": _change_case_english_lowercase,
    "change_case:capital_word_frequency": _change_case_capital_word_frequency,
    "combination:two_responses": _combination_two_responses,
    "combination:repeat_prompt": _combination_repeat_prompt,
}


def grade_ifeval(output: str, expected: dict[str, Any]) -> QualityResult:
    ids = expected.get("instruction_id_list", [])
    kwargs_list = expected.get("kwargs", [])
    if not ids:
        return QualityResult(
            grader="ifeval",
            passed=None,
            score=None,
            detail="no instruction_id_list",
        )
    # If any constraint id is unknown, the whole prompt is ungraded (so a
    # vendor gap doesn't silently fail prompts that are actually fine).
    unknown = [iid for iid in ids if iid not in CONSTRAINT_CHECKS]
    if unknown:
        return QualityResult(
            grader="ifeval",
            passed=None,
            score=None,
            detail=f"unknown constraint(s): {','.join(unknown)}",
        )
    outcomes: list[tuple[str, bool]] = []
    for iid, kw in zip(ids, kwargs_list, strict=True):
        check = CONSTRAINT_CHECKS[iid]
        outcomes.append((iid, bool(check(output, kw))))
    n_passed = sum(1 for _, ok in outcomes if ok)
    total = len(outcomes)
    detail = "; ".join(f"{iid}={'ok' if ok else 'fail'}" for iid, ok in outcomes)
    return QualityResult(
        grader="ifeval",
        passed=(n_passed == total),
        score=n_passed / total,
        detail=detail,
    )
