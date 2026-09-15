import json

from llm_compare.providers.anthropic_provider import AnthropicProvider
from llm_compare.providers.base import LLMProvider, LLMResult
from rag_notes.models import JudgeResult

JUDGE_SYSTEM_PROMPT = '''
You are an impartial evaluator scoring whether a generated answer is faithful
to its retrieved context and consistent with a reference answer.

Score the generated answer from 0 to 3:
  0 = Contradicts the context, or is fabricated (not supported by the context at all)
  1 = Partially supported by the context but missing key facts the reference answer has
  2 = Supported by the context and consistent with the reference answer, but incomplete or unclear
  3 = Fully supported by the context and matches the reference answer's key facts

Respond with ONLY a JSON object in this exact format:
  {"score": <0-3>, "reason": "<one sentence>"}
Do not add any other text.
'''

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
    return f"""Question: {question.strip()}

Retrieved Context:
{context.strip()}

Generated Answer:
{answer.strip()}

Reference Answer:
{expected_answer.strip()}

Score this answer."""


def clean_result(result: LLMResult) -> str:
    """Strip LLM response text before JSON parsing."""
    text_res = result.text.strip()
    if text_res.startswith("```"):
        text_res = text_res.replace("```", "")
        text_res = text_res.replace("json\n", "", 1)
    return text_res


def normalize_score(score: float, scale: int) -> float | None:
    """Normalize given score from a given scale to the 0.0–1.0 range."""
    if scale == 0:
        raise ValueError('Scale cannot be zero')

    return None if score is None else score / scale


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
    raw_result = judge.ask(judge_prompt, JUDGE_SYSTEM_PROMPT, temperature=0.0) if isinstance(judge, AnthropicProvider) \
        else judge.ask(judge_prompt, JUDGE_SYSTEM_PROMPT)
    judge_result = clean_result(raw_result)

    try:
        judge_response = json.loads(judge_result.strip())
        score = judge_response["score"]
        reason = judge_response["reason"]
        is_passed = score >= 2
    except (json.JSONDecodeError, KeyError, ValueError):
        score, reason = None, f'Judge returned invalid JSON: {judge_result!r}'
        is_passed = False

    normalized_score = normalize_score(score, scale=3)
    return JudgeResult(
        question=question,
        answer=answer,
        score=normalized_score,
        reason=reason,
        passed=is_passed,
    )
