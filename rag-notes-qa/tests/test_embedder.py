from pathlib import Path

import numpy as np
import pytest

from rag_notes.embedder import cosine_similarity, chunk_ratios
from rag_notes.models import Chunk, DocumentMetadata


class FakeTokenizer:
    counts = {"first": 260, "second": 120, "empty": 0}

    def tokenize(self, text):
        return [0] * self.counts[text]

class FakeEncoding:
    counts = {"first": 200, "second": 100, "empty": 0}

    def encode(self, text):
        return [0] * self.counts[text]


_metadata = DocumentMetadata(week=12, day=1, file_path=Path("fake.docx"), title="fake")

chunk_one = Chunk(text="first", source=_metadata, heading="A", chunk_index=0)
chunk_two = Chunk(text="second", source=_metadata, heading="B", chunk_index=1)
chunk_three = Chunk(text="empty", source=_metadata, heading="C", chunk_index=2)


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


def test_chunk_ratios_returns_one_ratio_per_chunk():
    """Returns WordPiece/tiktoken ratio for each chunk, in chunk order"""
    chunks = [chunk_one, chunk_two]
    chunk_ratios_result = chunk_ratios(
        chunks=chunks,
        tokenizer=FakeTokenizer(),
        encoding=FakeEncoding()
    )

    assert chunk_ratios_result == pytest.approx([1.3, 1.2])


def test_chunk_ratios_skips_chunk_with_no_tiktoken_tokens():
    """A chunk with 0 tiktoken tokens is skipped instead of dividing by zero."""
    chunks = [chunk_one, chunk_two, chunk_three]
    chunk_ratios_result = chunk_ratios(
        chunks=chunks,
        tokenizer=FakeTokenizer(),
        encoding=FakeEncoding()
    )

    assert chunk_ratios_result == pytest.approx([1.3, 1.2])
