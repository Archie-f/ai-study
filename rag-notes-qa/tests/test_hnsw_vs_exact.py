import re

import pytest

from verify.verify_hnsw_vs_exact import overlap_at_k, exact_top_k


def test_overlap_at_k_with_normal_input():
    """Duplicate ids in the approximate list must not inflate the overlap (2 shared ids out of 3)."""
    overlap = overlap_at_k(['a','a','b'], ['a','b','c'], 3)
    assert overlap ==  pytest.approx(0.67, abs=1e-2)


def test_overlap_at_k_counts_first_k_elements():
    """Only the first k elements of each list are compared."""
    overlap = overlap_at_k(['a','a','b','c','d'], ['a','b','c','d'], 2)
    assert overlap == 0.5

@pytest.mark.parametrize("k", [0, -1])
def test_overlap_at_k_raises_value_error(k):
    """k of zero or below is rejected with a ValueError."""
    with pytest.raises(ValueError, match="k must be greater than 0"):
        overlap_at_k(['a','b','c'], ['b','a','z'], k)


def test_overlap_at_k_returns_zero_for_no_common_in_first_k_elements():
    """A shared id outside the first k of the exact list does not count."""
    overlap = overlap_at_k(['a','z'], ['b','x','z'], 2)
    assert overlap == 0.0


def test_overlap_at_k_calculates_according_to_k_value():
    """The overlap is divided by k, so a short exact list lowers the result."""
    overlap = overlap_at_k(['a','b'], ['a'], 2)
    assert overlap == 0.5


def test_exact_top_k_cosine_calculated_correctly():
    """Ranking uses cosine similarity (with length division), not the raw dot product."""
    exact_top_k_ids = exact_top_k(['p','r','s'], [[10,0],[1,1],[0,-3]], [1,1], 3)
    assert exact_top_k_ids == ['r', 'p', 's']


def test_exact_top_k_returns_first_k_ids():
    """Only the k most similar ids are returned."""
    exact_top_k_ids = exact_top_k(['a','b','c'], [[1,0],[0.8,0.6],[0,1]], [1,0], 2)
    assert exact_top_k_ids == ['a','b']


def test_exact_top_k_breaks_ties_by_id():
    """Equal cosine scores are ordered by id ascending, whatever the input order."""
    exact_top_k_ids = exact_top_k(['c', 'a', 'b'], [[1,0], [3,0], [2,0]], [1,0], 3)
    assert exact_top_k_ids == ['a', 'b', 'c']


def test_exact_top_k_raises_value_error_for_different_list_length():
    """ids and vectors of different lengths raise a ValueError."""
    with pytest.raises(ValueError):
        exact_top_k(['a', 'b', 'c', 'd'], [[1, 0], [0.8, 0.6], [0, 1]], [1, 0], 2)


def test_exact_top_k_raises_value_error_for_vector_length():
    """A vector whose length differs from the query vector raises a ValueError."""
    with pytest.raises(ValueError):
        exact_top_k(['a'], [[1,0,0]], [1,0], 1)


def test_exact_top_k_raises_value_error_for_zero_vector_value():
    """A zero vector (cosine undefined) raises a ValueError with a clear message."""
    error_message = re.escape("Denominator ('norm_vector * norm_query)' is calculated as 0, but must be greater than 0")
    with pytest.raises(ValueError, match=error_message):
        exact_top_k(['p'], [[0, 0]], [1, 1], 3)
