from typing import Literal

from rag_notes.bm25_index import bm25_search
from rag_notes.hybrid_search import build_rank_map, rrf_merge
from rag_notes.models import BM25Index, Chunk
from rag_notes.vector_store import get_query_result


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
    bm25_results = []
    if mode == "bm25" or mode == "hybrid":
        bm25_raw_results = bm25_search(query, bm25_index, n)
        for res in bm25_raw_results:
            chunk_id = f"{res[1].source.title}-{res[1].chunk_index}"
            bm25_results.append((chunk_id, res[0], res[1]))
        if mode == "bm25":
            return bm25_results

    vector_raw_results = get_query_result(collection, query, model, n_results=n)
    vector_results_chunk_ids = vector_raw_results["ids"][0]
    id_chunk_dict = {f"{chunk.source.title}-{chunk.chunk_index}": chunk for chunk in bm25_index.chunks}

    if mode == "vector" or mode == "hybrid":
        distances = vector_raw_results["distances"][0]
        vector_results = []
        for chunk_id, distance in zip(vector_results_chunk_ids, distances):
            vector_results.append((chunk_id, (1 - distance), id_chunk_dict[chunk_id]))
        if mode == "vector":
            return vector_results

    bm25_results_chunk_ids = [result[0] for result in bm25_results]
    ranked_maps = [build_rank_map(vector_results_chunk_ids), build_rank_map(bm25_results_chunk_ids)]
    rrf_merged = rrf_merge(ranked_maps)
    hybrid_results = []
    for chunk_id, score in rrf_merged[:n]:
        hybrid_results.append((chunk_id, score, id_chunk_dict[chunk_id]))

    return hybrid_results
