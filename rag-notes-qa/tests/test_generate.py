from pathlib import Path
from typing import Literal, Generator, Any

import pytest

from llm_compare.providers.base import LLMResult, LLMProvider
from rag_notes.generate import build_context, normalize_answer, NO_ANSWER_TOKEN, GUARDRAIL_ANSWER, answer_question
from rag_notes.models import Chunk, DocumentMetadata

_metadata = DocumentMetadata(week=12, day=1, file_path=Path("fake.docx"), title="fake")

chunk_one = Chunk(text="Type hints help readability.", source=_metadata, heading="Type Hints", chunk_index=0)
chunk_two = Chunk(text="RRF merges rankings by position.", source=_metadata, heading="Hybrid Search", chunk_index=1)


class FakeRetriever:
    """A stand-in Retriever for tests: returns fixed results, records every call."""

    def __init__(self, results: list[tuple[str, float, Chunk]]):
        """Store the results to return and start with an empty call log.

        Args:
            results: the (chunk_id, score, chunk) tuples search() will return
        """
        self.results = results
        self.calls = []

    def search(
        self,
        query: str,
        n: int = 3,
        mode: Literal["hybrid", "vector", "bm25"] = "hybrid",
    ) -> list[tuple[str, float, Chunk]]:
        """Record (query, n, mode) in self.calls and return the stored results."""
        self.calls.append((query, n, mode))
        return self.results


class FakeProvider(LLMProvider):
    """A stand-in LLM provider for tests: always answers with the same text."""

    def __init__(self, text: str):
        """Store the text to answer with and start the call counter at zero."""
        self.text = text
        self.calls = 0

    def ask(self, user_input: str, system_prompt: str = "") -> LLMResult:
        """Count the call and return an LLMResult whose text is the stored text."""
        self.calls += 1
        return LLMResult(
            provider="fake_provider",
            model="fake_model",
            text=self.text,
            tokens_in=130,
            tokens_out=130,
            latency_ms=0.016
        )

    def ask_stream(self, user_input: str, system_prompt: str = '') -> Generator[str, Any, LLMResult]:
        raise NotImplementedError


def test_build_context_single_result():
    """build_context() with one (chunk_id, score, Chunk) result should produce
    exactly one "[Source 01: <label>]\\n<text>" block, with no trailing content."""
    single_result = ("fake-0", 0.5639, chunk_one)
    context = build_context([single_result])
    expected_context = "[Source 01: week-12-day-01]\nType hints help readability."
    assert context == expected_context


def test_build_context_multiple_results_numbered_in_order():
    """build_context() with two results should number them 01 and 02 in the
    given order, joined by a blank line, with no trailing separator."""
    results = [
        ("fake-0", 0.5639, chunk_one),
        ("fake-1", 0.3845, chunk_two)
    ]
    context = build_context(results)
    expected_context = ("[Source 01: week-12-day-01]\nType hints help readability.\n\n"
                        "[Source 02: week-12-day-01]\nRRF merges rankings by position.")
    assert context == expected_context


def test_build_context_raises_on_empty_results():
    """build_context() with an empty results list should raise ValueError —
    there's no context to build, and callers need to know explicitly."""
    empty_results = []
    with pytest.raises(ValueError):
        build_context(empty_results)


def test_normalize_answer_returns_guardrail_when_token_present():
    """normalize_answer() should return GUARDRAIL_ANSWER when the model's
    raw text contains NO_ANSWER_TOKEN anywhere in it."""
    answer = f"The model puts some textx before {NO_ANSWER_TOKEN} and after it."
    normalized_answer = normalize_answer(answer)
    assert normalized_answer == GUARDRAIL_ANSWER


def test_normalize_answer_returns_text_unchanged_when_token_absent():
    """normalize_answer() should return the text unchanged when NO_ANSWER_TOKEN
    does not appear anywhere in it."""
    answer = "This is the answer from the model"
    normalized_answer = normalize_answer(answer)
    assert normalized_answer == answer


def test_answer_question_returns_answer_and_citations():
    """answer_question() with a retriever that returns two results should
    return the provider's text as the answer, one citation per result, and
    should have called the retriever exactly once with the question and n."""
    question = "What does Type hints do?"
    answer = "Type hints help. [Source 01]"
    n = 3

    fake_retriever = FakeRetriever(results=[("fake-1", 0.9, chunk_one), ("fake-2", 0.7, chunk_two)])
    fake_provider = FakeProvider(text=answer)

    answer_query = answer_question(
        retriever=fake_retriever,
        question=question,
        provider=fake_provider,
        n=n
    )

    assert answer_query.answer == answer
    assert len(answer_query.citations) == 2
    assert fake_retriever.calls == [(question, n, "hybrid")]
    assert fake_provider.calls == 1


def test_answer_question_returns_guardrail_when_no_results():
    """answer_question() with a retriever that returns no results should
    return the guardrail answer with no citations, and should never call
    the provider."""
    question = "What does Type hints do?"
    answer = "Type hints help. [Source 01]"
    n = 3

    fake_retriever = FakeRetriever(results=[])
    fake_provider = FakeProvider(text=answer)

    answer_query = answer_question(
        retriever=fake_retriever,
        question=question,
        provider=fake_provider,
        n=n
    )

    assert answer_query.answer == GUARDRAIL_ANSWER
    assert len(answer_query.citations) == 0
    assert fake_provider.calls == 0
