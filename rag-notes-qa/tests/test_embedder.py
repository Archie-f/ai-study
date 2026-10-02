import numpy as np
import pytest

from rag_notes.embedder import cosine_similarity


def test_cosine_similarity_identical_vectors_returns_one():
    """Two identical vectors point in exactly the same direction -- similarity 1.0."""
    assert cosine_similarity(np.array([1, 0]), np.array([1, 0])) == 1.0

def test_cosine_similarity_orthogonal_vectors_returns_zero():
    """Two perpendicular vectors share no direction -- similarity 0.0."""
    assert cosine_similarity(np.array([1, 0]), np.array([0, 1])) == 0.0

def test_cosine_similarity_opposite_vectors_returns_negative_one():
    """Two opposite-direction vectors -- similarity -1.0."""
    assert cosine_similarity(np.array([1, 0]), np.array([-1, 0])) == -1.0

def test_cosine_similarity_zero_vector_raises_value_error():
    """A zero vector has no direction -- cosine_similarity() should raise, not divide by zero silently."""
    with pytest.raises(ValueError):
        cosine_similarity(np.array([-1, 0]), np.array([0, 0]))