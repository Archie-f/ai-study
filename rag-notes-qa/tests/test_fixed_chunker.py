from pathlib import Path

import pytest

from rag_notes.fixed_chunker import chunk_fixed_size, chunk_fixed_size_document, encoding
from rag_notes.models import DocumentMetadata


def test_normal_multiple_chunks_with_overlap():
    """Happy path: several chunks; each consecutive pair overlaps by exactly `o` tokens."""
    n, o = 5, 2
    text = "one two three four five six seven eight"
    result = chunk_fixed_size(text, n, o)
    assert len(result) == 2

    for i in range(len(result) - 1):
        current_tail = encoding.encode(result[i])[-o:]
        next_head = encoding.encode(result[i + 1])[:o]
        assert current_tail == next_head

def test_text_fits_in_a_single_chunk():
    """total_tokens <= n -> exactly one chunk containing the whole text."""
    n, o = 5, 2
    text = "one two three four five"
    result = chunk_fixed_size(text, n, o)
    assert len(result) == 1
    assert result[0] == text

def test_text_one_token_over_single_chunk():
    """total_tokens == n + 1 -> boundary case, must become 2 chunks, not 1."""
    n, o = 5, 2
    text = "one two three four five six"
    result = chunk_fixed_size(text, n, o)
    assert len(result) == 2


def test_trailing_remainder_equal_to_overlap_is_dropped():
    """Last chunk's remainder == o tokens -> dropped (fully duplicate of previous chunk)."""
    n, o = 5, 2
    text = "one two three four five six seven eight nine ten eleven"  # 11 tokens
    result = chunk_fixed_size(text, n, o)

    tokens = encoding.encode(text)
    step = n - o
    last_start = list(range(0, len(tokens), step))[-1]
    dropped_candidate = encoding.decode(tokens[last_start:])

    assert len(result) == 3
    assert dropped_candidate not in result


def test_trailing_remainder_one_over_overlap_is_kept():
    """Last chunk's remainder == o + 1 tokens -> kept (adds at least one new token)."""
    n, o = 5, 2
    text = "one two three four five six seven eight nine ten eleven twelve"  # 12 tokens
    result = chunk_fixed_size(text, n, o)

    tokens = encoding.encode(text)
    step = n - o
    last_start = list(range(0, len(tokens), step))[-1]
    kept_tail = encoding.decode(tokens[last_start:])

    assert len(result) == 4
    assert result[-1] == kept_tail


def test_overlap_greater_or_equal_chunk_size_raises():
    """o >= n must raise ValueError before any tokenization happens."""
    with pytest.raises(ValueError):
        chunk_fixed_size("irrelevant text here", n=5, o=5)   # o == n
    with pytest.raises(ValueError):
        chunk_fixed_size("irrelevant text here", n=5, o=6)   # o > n


def test_document_chunks_have_sequential_index_and_matching_text():
    """chunk_fixed_size_document assigns chunk_index 0, 1, 2, ... in the order chunk_texts was given."""
    metadata = DocumentMetadata(week=9, day=5, file_path=Path("fake.docx"), title="fake")
    chunk_texts = ["first chunk", "second chunk", "third chunk"]

    result = chunk_fixed_size_document(chunk_texts, metadata)

    assert [chunk.chunk_index for chunk in result] == [0, 1, 2]
    assert [chunk.text for chunk in result] == chunk_texts


def test_document_heading_defaults_to_none():
    """heading is not passed in -> every Chunk's heading is None, since standalone fixed-size
    chunking has no structural heading information of its own to attach."""
    metadata = DocumentMetadata(week=9, day=5, file_path=Path("fake.docx"), title="fake")
    chunk_texts = ["only chunk"]

    result = chunk_fixed_size_document(chunk_texts, metadata)

    assert result[0].heading is None


def test_document_heading_is_passed_through_when_given():
    """heading is passed in explicitly -> every resulting Chunk carries that same heading."""
    metadata = DocumentMetadata(week=9, day=5, file_path=Path("fake.docx"), title="fake")
    chunk_texts = ["chunk one", "chunk two"]

    result = chunk_fixed_size_document(chunk_texts, metadata, heading="1.1 Some Section")

    assert all(chunk.heading == "1.1 Some Section" for chunk in result)


def test_document_source_metadata_is_carried_through_unchanged():
    """Every Chunk's source is the exact same metadata object that was passed in."""
    metadata = DocumentMetadata(week=9, day=5, file_path=Path("fake.docx"), title="fake")
    chunk_texts = ["a chunk"]

    result = chunk_fixed_size_document(chunk_texts, metadata)

    assert result[0].source is metadata


def test_document_empty_chunk_texts_returns_empty_list():
    """No chunk texts given -> no Chunks produced."""
    metadata = DocumentMetadata(week=9, day=5, file_path=Path("fake.docx"), title="fake")

    result = chunk_fixed_size_document([], metadata)

    assert result == []
