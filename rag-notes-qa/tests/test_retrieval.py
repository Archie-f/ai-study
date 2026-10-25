from pathlib import Path

import pytest

from rag_notes.models import DocumentMetadata, Chunk
from rag_notes.retrieval import search_with_rerank

_metadata = DocumentMetadata(week=12, day=1, file_path=Path("fake.docx"), title="fake")
chunk_one = Chunk(text="First chunk's text.", source=_metadata, heading="A", chunk_index=0)
chunk_two = Chunk(text="Second chunk's text.", source=_metadata, heading="B", chunk_index=1)
chunk_three = Chunk(text="Third chunk's text.", source=_metadata, heading="C", chunk_index=2)
chunk_four = Chunk(text="Fourth chunk's text.", source=_metadata, heading="D", chunk_index=3)

CANDIDATES_LIST = [
    ("a", 0.50689, chunk_one),
    ("b", 0.41297, chunk_two),
    ("c", 0.37643, chunk_three),
    ("d", 0.31563, chunk_four)
]
RERANKED = [
    ("d", 0.44444, chunk_one),
    ("a", 0.33333, chunk_one),
    ("b", 0.22222, chunk_one),
    ("c", 0.11111, chunk_one),
]


class FakeReranker:
    """Fake Reranker"""

    def __init__(self):
        self.query: str = ""
        self.candidates = []
        self.top_k: int = 0

    def rerank(self, query, candidates, top_k):
        self.query = query
        self.candidates = candidates
        self.top_k = top_k
        return RERANKED


@pytest.fixture
def patched_search(monkeypatch):
    calls = {}

    def fake_search(retrieval_index, query, n=3, mode="hybrid"):
        calls["n"] = n
        calls["mode"] = mode
        return CANDIDATES_LIST

    monkeypatch.setattr("rag_notes.retrieval.search", fake_search)
    return calls


def test_search_with_rerank_asks_first_stage_for_n_candidates(patched_search):
    """The first stage is asked for n_candidates results, in the given mode."""
    search_with_rerank(
        retrieval_index=None,
        reranker=FakeReranker(),
        query="Some query",
        n_candidates=4,
        top_k=2,
        mode="bm25"
    )

    assert patched_search["n"] == 4
    assert patched_search["mode"] == "bm25"


def test_search_with_rerank_passes_query_candidates_and_top_k_to_reranker(patched_search):
    """The reranker receives the query, the first-stage candidates and top_k."""
    reranker = FakeReranker()
    search_with_rerank(
        retrieval_index=None,
        reranker=reranker,
        query="Some query",
        n_candidates=4,
        top_k=2,
        mode="bm25"
    )

    assert reranker.query == "Some query"
    assert reranker.candidates == CANDIDATES_LIST
    assert reranker.top_k == 2


def test_search_with_rerank_returns_the_reranker_output(patched_search):
    """The result is exactly what the reranker returned."""
    results = search_with_rerank(
        retrieval_index=None,
        reranker=FakeReranker(),
        query="Some query",
        n_candidates=4,
        top_k=2,
        mode="bm25"
    )

    assert results == RERANKED
