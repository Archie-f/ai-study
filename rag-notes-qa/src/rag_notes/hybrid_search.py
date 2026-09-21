from typing import Literal

from rag_notes.models import BM25Index, Chunk
from rag_notes.bm25_index import bm25_search
from rag_notes.vector_store import get_query_result


def build_rank_map(ordered_ids: list[str]) -> dict[str, int]:
    """Turn an ordered id list into an id -> 1-based rank lookup.

        Args:
            ordered_ids (list[str]): ordered list of ids to rank
        Returns:
            dict[str, int]: rank map
    """
    return {chunk_id: rank for rank, chunk_id in enumerate(ordered_ids, start=1)}


def rrf_merge(rank_maps: list[dict[str, int]], k: int = 60) -> list[tuple[str, float]]:
    """Merge multiple id -> rank maps into one RRF-scored, sorted list.

    Args:
        rank_maps: one dict per ranker, each id -> 1-based rank
        k: RRF damping constant
    Returns:
        list of (chunk_id, rrf_score) tuples, highest score first
    """
    all_ids: set[str] = set()
    for rank_map in rank_maps:
        all_ids.update(rank_map.keys())

    scored = []
    for chunk_id in all_ids:
        score = 0.0
        for rank_map in rank_maps:
            rank = rank_map.get(chunk_id)
            if rank is not None:
                score += 1 / (k + rank)
        scored.append((chunk_id, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def hybrid_search(
        query: str,
        collection,
        model,
        bm25_index: BM25Index,
        n: int = 3,
        mode: Literal["hybrid", "vector", "bm25"] = "hybrid",
) -> list[tuple[str, float, Chunk]]:
    """Search with a selectable retrieval mode; "hybrid" is the existing RRF-fused behavior.

    Args:
        query: raw query string
        collection: an already-populated Chroma collection
        model: the SentenceTransformer model used to embed the query
        bm25_index: a built BM25Index over the same chunks as the collection
        n: how many merged results to return
        mode: "hybrid" (default, current behavior), "vector" (dense-only, no fusion),
            or "bm25" (sparse-only, no fusion)
    Returns:
        top n (chunk_id, score, chunk) tuples, same shape regardless of mode
    """
    if mode == "bm25":
        sparse_results = bm25_search(
            query=query,
            index=bm25_index,
            n_results=n
        )
        return [
            (f"{chunk.source.title}-{chunk.chunk_index}", score, chunk)
            for score, chunk in sparse_results
        ]
    elif mode == "vector":
        dense_results = get_query_result(
            collection=collection,
            query=query,
            model=model,
            n_results=n
        )
        dense_ids = dense_results["ids"][0]
        distances = dense_results["distances"][0]
        id_to_chunk = {f"{c.source.title}-{c.chunk_index}": c for c in bm25_index.chunks}
        return [
            (chunk_id, (1 - distance), id_to_chunk[chunk_id])
            for distance, chunk_id in zip(distances, dense_ids)
        ]
    else:
        dense_results = get_query_result(
            collection=collection,
            query=query,
            model=model,
            n_results=n
        )
        sparse_results = bm25_search(
            query=query,
            index=bm25_index,
            n_results=n
        )

        dense_ids = dense_results["ids"][0]
        sparse_ids = [f"{chunk.source.title}-{chunk.chunk_index}" for _, chunk in sparse_results]

        rank_maps = [build_rank_map(dense_ids), build_rank_map(sparse_ids)]
        merged = rrf_merge(rank_maps)

        id_to_chunk = {f"{c.source.title}-{c.chunk_index}": c for c in bm25_index.chunks}

        return [
            (chunk_id, score, id_to_chunk[chunk_id])
            for chunk_id, score in merged[:n]
        ]
