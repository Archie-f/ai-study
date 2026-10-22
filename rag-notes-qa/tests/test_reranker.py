import re
from pathlib import Path

import pytest

from rag_notes.models import DocumentMetadata, Chunk
from rag_notes.reranker import pick_top_by_score

_metadata = DocumentMetadata(week=12, day=1, file_path=Path("fake.docx"), title="fake")
chunk_one = Chunk(text="First chunk's text.", source=_metadata, heading="A", chunk_index=0)
chunk_two = Chunk(text="Second chunk's text.", source=_metadata, heading="B", chunk_index=1)
chunk_three = Chunk(text="Third chunk's text.", source=_metadata, heading="C", chunk_index=2)
chunk_four = Chunk(text="Fourth chunk's text.", source=_metadata, heading="D", chunk_index=3)

candidates_list = [
    ("a", 0.50689, chunk_one),
    ("b", 0.41297, chunk_two),
    ("c", 0.37643, chunk_three),
    ("d", 0.31563, chunk_four)
]
scores_list = [0.44444, 0.22222, 0.33333, 0.11111]


def test_candidates_come_back_best_score_first():
    """Candidates come back with the highest new score first."""
    results = pick_top_by_score(candidates_list, scores_list, top_k=4)
    assert [r[0] for r in results] == ["a", "c", "b", "d"]


def test_only_top_k_come_back():
    """Only the top_k best candidates come back."""
    results = pick_top_by_score(candidates_list, scores_list, top_k=2)
    assert len(results) == 2
    assert [r[0] for r in results] == ["a", "c"]


def test_empty_candidates_return_empty_list():
    """Empty candidates return an empty list."""
    empty_candidates_list = []
    empty_scores_list = []
    results = pick_top_by_score(empty_candidates_list, empty_scores_list, top_k=3)
    assert results == []


def test_returned_tuples_carry_the_new_score():
    """The returned tuples carry the new scores, not the old ones."""
    results = pick_top_by_score(candidates_list, scores_list, top_k=4)
    assert [r[1] for r in results] == [0.44444, 0.33333, 0.22222, 0.11111]


def test_equal_scores_keep_original_order():
    """Candidates with equal scores keep their original order."""
    scores_with_equal_values = [0.33333, 0.33333, 0.22222, 0.44444]
    results = pick_top_by_score(candidates_list, scores_with_equal_values, top_k=4)
    assert [r[0] for r in results] == ["d", "a", "b", "c"]


def test_mismatched_lengths_raise_value_error():
    """ValueError is raised if candidates and scores differ in length."""
    mismatching_scores = [0.33333, 0.33333, 0.22222, 0.44444, 0.11111]
    expected_error_message = "Number of candidates (4) and number of scores (5) do not match."
    with pytest.raises(ValueError, match=re.escape(expected_error_message)):
        pick_top_by_score(candidates_list, mismatching_scores, top_k=4)
