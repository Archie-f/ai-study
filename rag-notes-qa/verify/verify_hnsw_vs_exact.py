import json
import math

from config import PERSIST_PATH, EVAL_CORPUS_PATH, GOLDEN_QA_PATH
from rag_notes.adapters import ChromaVectorStore
from rag_notes.embedder import load_embedding_model
from rag_notes.pipeline import make_default_chunker
from rag_notes.retrieval import build_retrieval_index


TOP_K = 20


def overlap_at_k(approx_ids: list[str], exact_ids: list[str], k: int) -> float:
    """Fraction of the first k exact ids that also appear in the first k approximate ids."""
    if k <= 0:
        raise ValueError("k must be greater than 0")

    return len(set(approx_ids[:k]) & set(exact_ids[:k])) / k


def exact_top_k(ids: list[str], vectors: list[list[float]], query_vector: list[float], k: int) -> list[str]:
    """Return the k ids whose vectors have the highest cosine similarity to query_vector.

    Args:
        ids: chunk ids, same order as vectors
        vectors: stored chunk vectors
        query_vector: the embedded question
        k: how many ids to return
    Returns:
        k ids, most similar first
    """
    norm_query = math.sqrt(sum(q * q for q in query_vector))
    cosine_similarities = []
    for vector, chunk_id in zip(vectors, ids, strict=True):
        dot_product = sum(v * q for v, q in zip(vector, query_vector, strict=True))
        norm_vector = math.sqrt(sum(n * n for n in vector))
        denominator = norm_vector * norm_query

        if denominator == 0:
            raise ValueError("Denominator ('norm_vector * norm_query)' is calculated as 0, but must be greater than 0")

        similarity = dot_product / denominator
        cosine_similarities.append((similarity, chunk_id))

    cosine_similarities = sorted(cosine_similarities, key=lambda item: (-item[0], item[1]))
    return [chunk_id for _, chunk_id in cosine_similarities[:k]]


def main():
    model = load_embedding_model()
    chunker = make_default_chunker(model)
    vector_store = ChromaVectorStore(PERSIST_PATH)
    build_retrieval_index(
        notes_root=EVAL_CORPUS_PATH,
        vector_store=vector_store,
        chunker=chunker,
        model=model,
    )
    stored = vector_store.collection.get(include=["embeddings"])
    vectors = stored["embeddings"].tolist()
    ids = stored["ids"]

    with open(GOLDEN_QA_PATH, "r") as f:
        golden_qa = json.load(f)

    differences = []
    for entry in golden_qa:
        query = entry["question"]
        query_vector = model.encode(query).tolist()
        query_result_ids = vector_store.query(query, model, n_results=TOP_K)["ids"][0]
        exact_top_k_ids = exact_top_k(
            ids=ids,
            vectors=vectors,
            query_vector=query_vector,
            k=TOP_K
        )

        overlap_value = overlap_at_k(query_result_ids, exact_top_k_ids, k=TOP_K)

        if overlap_value < 1.0:
            difference = {
                "query" : query,
                "overlap": overlap_value,
                "chroma": query_result_ids,
                "exact": exact_top_k_ids,
                "only_in_chroma": list(set(query_result_ids) - set(exact_top_k_ids)),
                "only_in_exact": list(set(exact_top_k_ids) - set(query_result_ids)),
            }
            differences.append(difference)

    print(f"# {len(differences)} difference(s) found")
    for i, difference in enumerate(differences, start=1):
        print(
            f"Difference-{i}:\n\n"
            f"query: {difference['query']}\n\n"
            f"overlap value: {difference["overlap"]}\n\n"
            f"Chroma: {difference['chroma']}\n\n"
            f"Exact: {difference['exact']}\n\n"
            f"Only in Chroma: {difference['only_in_chroma']}\n\n"
            f"Only in Exact: {difference['only_in_exact']}\n\n"
            f"---"
        )


if '__main__' == __name__:
    main()
