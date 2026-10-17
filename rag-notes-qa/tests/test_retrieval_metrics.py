from pathlib import Path

import pytest

from rag_notes.models import Chunk, DocumentMetadata
from rag_notes.retrieval_metrics import recall_at_k, reciprocal_rank, mean_reciprocal_rank, normalize_text, \
    find_relevant_chunk_ids

chunk_one = Chunk(
    text="Session Goals\n",
    source=DocumentMetadata(week=1, day=1, file_path=Path("fake.docx"), title="week-01-day-01-takeaway-notes"),
    heading="Heading_1",
    chunk_index=0
)
chunk_two = Chunk(
    text="Type hints let you\nannotate your code.\n",
    source=DocumentMetadata(week=1, day=1, file_path=Path("fake.docx"), title="week-01-day-01-takeaway-notes"),
    heading="Heading_2",
    chunk_index=1
)
chunk_three = Chunk(
    text="Recap: TYPE HINTS let you annotate your code — again.",
    source=DocumentMetadata(week=5, day=2, file_path=Path("fake.docx"), title="week-05-day-02-takeaway-notes"),
    heading="Heading_3",
    chunk_index=3
)

def test_recall_at_k_hit_within_top_k():
    retrieved_ids = ["a", "b", "c", "d"]
    relevant_ids = {"c"}
    assert recall_at_k(retrieved_ids, relevant_ids, k=3) == 1.0


def test_recall_at_k_hit_out_of_top_k():
    retrieved_ids = ["a", "b", "c", "d"]
    relevant_ids = {"d"}
    assert recall_at_k(retrieved_ids, relevant_ids, k=3) == 0.0


def test_recall_at_k_empty_relevant_ids():
    retrieved_ids = ["a", "b", "c", "d"]
    relevant_ids = set()
    assert recall_at_k(retrieved_ids, relevant_ids, k=3) == 0.0


def test_recall_at_k_empty_retrieved_ids():
    retrieved_ids = []
    relevant_ids = {"c"}
    assert recall_at_k(retrieved_ids, relevant_ids, k=3) == 0.0


def test_reciprocal_rank_hit_at_rank_one():
    retrieved_ids = ["a", "b", "c", "d"]
    relevant_ids = {"a"}
    assert reciprocal_rank(retrieved_ids, relevant_ids) == 1.0


def test_reciprocal_rank_hit_at_rank_three():
    retrieved_ids = ["a", "b", "c", "d"]
    relevant_ids = {"c"}
    assert reciprocal_rank(retrieved_ids, relevant_ids) == 1/3


def test_reciprocal_rank_empty_relevant_ids():
    retrieved_ids = ["a", "b", "c", "d"]
    relevant_ids = set()
    assert reciprocal_rank(retrieved_ids, relevant_ids) == 0.0


def test_reciprocal_rank_first_retrieved_id_counted():
    retrieved_ids = ["a", "b", "c", "d"]
    relevant_ids = {"b", "d"}
    assert reciprocal_rank(retrieved_ids, relevant_ids) == 0.5


def test_reciprocal_rank_empty_retrieved_ids():
    retrieved_ids = []
    relevant_ids = {"a"}
    assert reciprocal_rank(retrieved_ids, relevant_ids) == 0.0


def test_mean_reciprocal_rank():
    scores = [1.0, 0.3, 0.3, 0.8]
    assert mean_reciprocal_rank(scores) == pytest.approx(0.6)


def test_mean_reciprocal_rank_empty_scores_list():
    scores = []
    assert mean_reciprocal_rank(scores) == 0.0


def test_reciprocal_rank_one_score():
    scores = [1.0]
    assert mean_reciprocal_rank(scores) == 1.0


def test_normalize_text_with_escape_characters():
    text = "A\n  B\tC"
    normalized_text = normalize_text(text)
    assert normalized_text == "a b c"


def test_normalize_text_for_paddings():
    text = "  x  "
    normalized_text = normalize_text(text)
    assert normalized_text == "x"


def test_find_relevant_chunk_ids():
    chunks = [chunk_one, chunk_two, chunk_three]
    answer_key = "Type hints let you annotate your code\n"
    relevant_ids = find_relevant_chunk_ids(chunks, answer_key)
    assert relevant_ids == {"week-01-day-01-takeaway-notes-1", "week-05-day-02-takeaway-notes-3"}


def test_find_relevant_chunk_ids_non_existing_key():
    chunks = [chunk_one, chunk_two, chunk_three]
    answer_key = "kubernetes"
    relevant_ids = find_relevant_chunk_ids(chunks, answer_key)
    assert relevant_ids == set()


def test_find_relevant_chunk_ids_empty_key():
    chunks = [chunk_one, chunk_two, chunk_three]
    answer_key = "  "
    with pytest.raises(ValueError):
        find_relevant_chunk_ids(chunks, answer_key)
