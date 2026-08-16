from pathlib import Path

from rag_notes.bm25_index import tokenize, build_doc_frequencies, build_bm25_index, score_document
from rag_notes.models import Chunk, DocumentMetadata

_fake_metadata = DocumentMetadata(week=10, day=5, file_path=Path("fake.docx"), title="fake")

movie_chunks: list[Chunk] = [
    Chunk(text="dark knight rises", source=_fake_metadata, heading=None, chunk_index=0),
    Chunk(text="man of steel returns forever", source=_fake_metadata, heading=None, chunk_index=1),
]
movie_index = build_bm25_index(movie_chunks)


def test_tokenize_lowercases_and_strips_punctuation():
    text = "The Dark Knight (2008) is great!"
    assert tokenize(text) == ["the", "dark", "knight", "2008", "is", "great"]


def test_tokenize_empty_string():
    assert tokenize("") == []


def test_build_doc_frequencies_counts_documents_not_occurrences():
    tokenized_docs = [
        ["dark", "knight", "dark"],
        ["knight", "rises"],
    ]
    doc_freq = build_doc_frequencies(tokenized_docs)
    assert doc_freq["dark"] == 1
    assert doc_freq["knight"] == 2
    assert doc_freq["rises"] == 1


def test_build_doc_frequencies_empty_list():
    assert build_doc_frequencies([]) == {}


def test_score_document_matches_hand_calculated_score():
    query_terms = tokenize("dark knight")
    score = score_document(query_terms, 0, movie_index)
    assert score == 1.5442266301082324


def test_score_document_is_zero_when_query_terms_absent_from_doc():
    query_terms = tokenize("dark knight")
    assert score_document(query_terms, 1, movie_index) == 0.0


def test_score_document_is_zero_when_query_term_unseen_in_corpus():
    query_terms = tokenize("batman")
    assert score_document(query_terms, 0, movie_index) == 0.0
