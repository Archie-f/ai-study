import json

from llm_compare.providers.base import LLMResult, LLMProvider
from rag_notes.llm_judge import JUDGE_SYSTEM_PROMPT
from rag_notes.models import JudgeResult


def build_judge_prompt(question: str, context: str, answer: str, expected_answer: str) -> str:
    """Build the prompt sent to the judge model for RAG faithfulness scoring.

    Args:
        question: the original question that was asked
        context: the retrieved context the answer was supposed to be grounded in
        answer: the generated answer to evaluate
        expected_answer: golden_qa.json's expected_answer_text for this question
    Returns:
        A single prompt string with all four pieces, ready to send alongside JUDGE_SYSTEM_PROMPT
    """
    return (f"Question: {question.strip()}\n\n"
            f"Retrieved Context: \n{context.strip()}\n\n"
            f"Generated Answer: \n{answer.strip()}\n\n"
            f"Reference Answer: \n{expected_answer.strip()}\n\n"
            f"Score this answer.")


def clean_result(result: LLMResult) -> str:
    """Strip LLM response text before JSON parsing.

    Args:
        result: LLMResult object
    Returns:
        Stripped JSON parsing result
    """
    text = result.text.strip()
    if text.startswith("```"):
        text = text.replace("```", "")
        text = text.replace("json\n", "", 1)

    return text


def normalize_score(score: float, scale: int) -> float | None:
    """Normalize given score from a given scale to the 0.0–1.0 range.

    Args:
        score: score value to normalize
        scale: scale to normalize
    Returns:
        Normalized score in 0.0-1.0 range
    Raises:
        ValueError: if the score is zero
    """
    if scale == 0:
        raise ValueError("Scale cannot be 0")
    elif score is None or scale is None:
        return None
    else:
        return score / scale


def judge_answer(
    question: str,
    context: str,
    answer: str,
    expected_answer: str,
    judge: LLMProvider,
) -> JudgeResult:
    """Score a generated answer's faithfulness using a second LLM as judge.

    Args:
        question: the original question that was asked
        context: the retrieved context the answer was supposed to be grounded in
        answer: the generated answer to evaluate
        expected_answer: golden_qa.json's expected_answer_text for this question
        judge: an LLMProvider instance to call as the judge (reused, not a new provider)
    Returns:
        A JudgeResult — score=None and passed=False if the judge's response
        couldn't be parsed as valid JSON
    """
    judge_prompt = build_judge_prompt(question, context, answer, expected_answer)
    judge_reply = judge.ask(judge_prompt, JUDGE_SYSTEM_PROMPT)
    judge_result = clean_result(judge_reply)

    try:
        judge_response = json.loads(judge_result.strip())
        score = judge_response["score"]
        reason = judge_response["reason"]
        is_passed = score >=2
    except (json.JSONDecodeError, KeyError, ValueError):
        score = None
        reason = f"Judge returned invalid JSON: {judge_result!r}"
        is_passed = False

    return JudgeResult(
        question=question,
        answer=answer,
        score=normalize_score(score, scale=3),
        reason=reason,
        passed=is_passed
    )
