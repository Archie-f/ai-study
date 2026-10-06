from pathlib import Path

import pytest

from rag_notes.bm25_index import build_bm25_index
from rag_notes.hybrid_search import build_rank_map, rrf_merge, hybrid_search
from rag_notes.models import Chunk, DocumentMetadata

box_office_ranking: dict[str, int] = {
    "Avatar": 1,
    "The Dark Knight": 2,
    "Inception": 3,
    "The Matrix": 4,
    "The Avengers": 5
}

critic_ratings_ranking: dict[str, int] = {
    "Pulp Fiction": 1,
    "The Dark Knight": 2,
    "Inception": 3,
    "The Matrix": 4,
    "Interstellar": 5
}

_metadata = DocumentMetadata(week=12, day=1, file_path=Path("fake.docx"), title="fake")
chunk_one = Chunk(text="Type hints help readability.", source=_metadata, heading="Type Hints", chunk_index=0)
chunk_two = Chunk(text="RRF merges rankings by position.", source=_metadata, heading="Hybrid Search", chunk_index=1)
chunk_three = Chunk(text="BM25 uses term frequency.", source=_metadata, heading="BM25 Term Frequency", chunk_index=2)


class FakeVectorStore:
    """A stand-in VectorStore for tests: returns a fixed query result, records every query."""

    def __init__(self, result: dict):
        """Store the result to return and start with an empty call log.

        Args:
            result: a Chroma-shaped dict, e.g. {"ids": [[...]], "distances": [[...]]}
        """
        self.calls = []
        self.result = result

    def add(self, embedded_chunks: list) -> None:
        """Not used by hybrid_search(); present only to match the Protocol."""

    def query(self, query: str, model, n_results: int = 3) -> dict:
        """Record (query, model, n_results) in self.calls and return the stored result."""
        self.calls.append((query, model, n_results))
        return self.result

    def reset(self) -> None:
        """Not used by hybrid_search(); present only to match the Protocol."""


def test_build_rank_map():
    movies: list[str] = [
        "Inception",
        "The Dark Knight",
        "Interstellar",
        "Pulp Fiction",
        "The Matrix"
    ]
    rank_map = build_rank_map(movies)
    for index, movie in enumerate(movies):
        assert rank_map[movie] == index+1


def test_build_rank_map_empty_list():
    movies: list[str] = []
    rank_map = build_rank_map(movies)
    assert rank_map == {}


def test_rrf_merge():
    all_rankings = [box_office_ranking, critic_ratings_ranking]
    merged_ranking = rrf_merge(all_rankings)

    all_titles = set()
    for ranking in all_rankings:
        all_titles.update(ranking.keys())
    assert len(merged_ranking) == len(all_titles)

    merged = dict(merged_ranking)
    for title, actual_score in merged.items():
        expected_score = 0.0
        for ranking in all_rankings:
            if title in ranking:
                rank = ranking[title]
                expected_score += 1 / (60 + rank)
        assert expected_score == actual_score

    assert all(merged_ranking[i][1] >= merged_ranking[i+1][1] for i in range(len(merged_ranking)-1))


def test_rrf_merge_empty_dict():
    empty_dict = dict()
    merged_rankings_with_empty = rrf_merge([box_office_ranking, empty_dict])
    all_rankings_without_empty = rrf_merge([box_office_ranking])
    assert merged_rankings_with_empty == all_rankings_without_empty


def test_hybrid_search_bm25_mode_never_queries_vector_store():
    """hybrid_search() in "bm25" mode should return results ranked by BM25
    alone and should never call the vector store."""
    bm25_index = build_bm25_index([chunk_one, chunk_two, chunk_three])
    fake_vector_store = FakeVectorStore(result={"ids": [["fake-0", "fake-1"]], "distances": [[0.20, 0.35]]})
    query = "RRF rankings"
    hybrid_search_result = hybrid_search(query, fake_vector_store, None, bm25_index, mode="bm25")

    assert fake_vector_store.calls == []
    assert hybrid_search_result[0][2].text == "RRF merges rankings by position."


def test_hybrid_search_vector_mode_uses_store_order_and_converts_distance():
    """hybrid_search() in "vector" mode should return chunks in the order the
    vector store gave them, with each score equal to 1 - distance, and should
    query the store exactly once with the caller's query, model and n."""
    bm25_index = build_bm25_index([chunk_one, chunk_two, chunk_three])
    results = {"ids": [["fake-2", "fake-0", "fake-1"]], "distances": [[0.20, 0.35, 0.40]]}
    fake_vector_store = FakeVectorStore(result=results)
    query = "RRF rankings"
    model = "fake_model"
    n = 3
    hybrid_search_result = hybrid_search(query, fake_vector_store, model, bm25_index, n, mode="vector")
    ids = [hybrid_search_result[i][0] for i in range(len(hybrid_search_result))]
    scores = [hybrid_search_result[i][1] for i in range(len(hybrid_search_result))]

    assert ids == ["fake-2", "fake-0", "fake-1"]
    assert scores == pytest.approx([1 - 0.20, 1 - 0.35, 1 - 0.40])
    assert fake_vector_store.calls == [(query, model, n)]


def test_hybrid_search_hybrid_mode_merges_both_and_respects_n():
    """hybrid_search() in "hybrid" mode should query the vector store exactly
    once with the caller's query, model and n, return at most n results, and
    rank first the chunk that both the vector store and BM25 ranked first."""
    bm25_index = build_bm25_index([chunk_one, chunk_two, chunk_three])
    results = {"ids": [["fake-1", "fake-2", "fake-0"]], "distances": [[0.20, 0.35, 0.40]]}
    fake_vector_store = FakeVectorStore(result=results)
    query = "RRF rankings"
    model = "fake_model"
    n = 2
    hybrid_search_result = hybrid_search(query, fake_vector_store, model, bm25_index, n, mode="hybrid")

    assert fake_vector_store.calls == [(query, model, n)]
    assert len(hybrid_search_result) == n
    assert hybrid_search_result[0][0] == results["ids"][0][0]
