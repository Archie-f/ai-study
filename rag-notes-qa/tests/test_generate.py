from pathlib import Path

import pytest

from rag_notes.generate import build_context, normalize_answer, NO_ANSWER_TOKEN, GUARDRAIL_ANSWER
from rag_notes.models import Chunk, DocumentMetadata

_metadata = DocumentMetadata(week=12, day=1, file_path=Path("fake.docx"), title="fake")

chunk_one = Chunk(text="Type hints help readability.", source=_metadata, heading="Type Hints", chunk_index=0)
chunk_two = Chunk(text="RRF merges rankings by position.", source=_metadata, heading="Hybrid Search", chunk_index=1)


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
