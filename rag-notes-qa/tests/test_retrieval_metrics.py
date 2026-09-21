import pytest

from rag_notes.retrieval_metrics import recall_at_k, reciprocal_rank, mean_reciprocal_rank


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
