import pytest

from llm_compare.providers.base import LLMResult
from rag_notes.llm_judge import build_judge_prompt, clean_result, normalize_score


def test_build_judge_prompt():
    question = "What is the question?"
    context = "This is context."
    answer = "Here is the answer."
    expected_answer = "The expected answer."

    judge_prompt = build_judge_prompt(question, context, answer, expected_answer)
    assert question in judge_prompt
    assert context in judge_prompt
    assert answer in judge_prompt
    assert expected_answer in judge_prompt


def test_clean_result_without_fence():
    result = LLMResult(
        provider="claude",
        model="llm_model",
        text="Response text string",
        tokens_in=5,
        tokens_out=3,
        latency_ms=0.0123
    )
    cleaned = clean_result(result)
    assert cleaned == result.text


def test_clean_result_with_fence():
    result = LLMResult(
        provider="claude",
        model="llm_model",
        text="```json\nResponse text string```",
        tokens_in=5,
        tokens_out=3,
        latency_ms=0.0123
    )
    cleaned = clean_result(result)
    assert cleaned == "Response text string"


def test_normalize_score_normal_score():
    normalized_score = normalize_score(score=0.5, scale=3)
    assert normalized_score == 1/6


def test_normalize_score_score_none_returns_none():
    normalized_score = normalize_score(score=None, scale=3)
    assert normalized_score is None


def test_normalize_score_scale_zero_raises_value_error():
    with pytest.raises(ValueError):
        normalize_score(score=0.987, scale=0)
