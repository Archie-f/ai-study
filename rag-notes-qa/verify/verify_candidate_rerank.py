import json

from config import GOLDEN_QA_PATH
from rag_notes.bm25_index import bm25_search
from rag_notes.retrieval_metrics import (
    find_relevant_chunk_ids, recall_at_k, reciprocal_rank, mean_reciprocal_rank,
)
from verify.verify_candidate_options import CONFIGS, build_candidates
from verify.verify_rerank_eval import get_retrieval_index_and_reranker

DEPTH = 100
TOP_K = 5
CONFIG_NAMES = ["baseline d20 c20", "depth 100, cut 20", "depth 100, cut 40"]


def main() -> None:
    """Send the candidates of three configs through the real reranker and compare.

    Dense and BM25 lists (depth 100) are fetched once per question and stored.
    For each config the candidates are reranked to the top TOP_K (no relevance
    threshold). Prints, per config: average candidates, questions whose right
    chunk is in the candidates, in the top 5, at rank 1, and MRR (out of 48);
    then the questions where the configs disagree about the top-5 hit.
    """
    retrieval_index, reranker = get_retrieval_index_and_reranker()
    chunks = retrieval_index.bm25_index.chunks
    id_to_chunk = {f"{chunk.source.title}-{chunk.chunk_index}": chunk for chunk in chunks}

    with open(GOLDEN_QA_PATH, "r") as f:
        golden_qa_list = json.load(f)

    stored_questions = []
    for entry in golden_qa_list:
        query_id = entry["id"]
        query = entry["question"]
        dense_ids = retrieval_index.vector_store.query(query, retrieval_index.model, n_results=DEPTH)["ids"][0]
        sparse_results = bm25_search(query, retrieval_index.bm25_index, n_results=DEPTH)
        sparse_ids = [f"{chunk.source.title}-{chunk.chunk_index}" for _, chunk in sparse_results]
        relevant_ids = find_relevant_chunk_ids(chunks, entry["answer_key"])
        stored_questions.append((query_id, query, dense_ids, sparse_ids, relevant_ids))

    stats = {}
    top5_hits = {}
    for name, depth, cut, weights in CONFIGS:
        if name in CONFIG_NAMES:
            stat = {"candidates": 0, "in_candidates": 0, "in_top5": 0, "at_rank1": 0, "ranks": []}
            for query_id, query, dense_ids, sparse_ids, relevant_ids in stored_questions:
                candidate_ids = build_candidates(dense_ids, sparse_ids, depth, cut, weights)
                candidates = [(chunk_id, 0.0, id_to_chunk[chunk_id]) for chunk_id in candidate_ids]
                reranked = reranker.rerank(query, candidates, TOP_K)
                reranked_ids = [chunk_id for chunk_id, _, _ in reranked]
                hit_top5 = recall_at_k(reranked_ids, relevant_ids, TOP_K) > 0

                stat["candidates"] += len(candidate_ids)
                stat["in_candidates"] += bool(relevant_ids & set(candidate_ids))
                stat["in_top5"] += hit_top5
                stat["at_rank1"] += recall_at_k(reranked_ids, relevant_ids, 1) > 0
                stat["ranks"].append(reciprocal_rank(reranked_ids, relevant_ids))
                top5_hits.setdefault(query_id, {})[name] = hit_top5
            stats[name] = stat

    total = len(golden_qa_list)
    print(f"{total} questions, reranked top {TOP_K}")
    print(f"{'config':<20}{'avg cand.':>10}{'in cand.':>10}{'in top 5':>10}{'rank 1':>8}{'MRR':>8}")
    for name, stat in stats.items():
        print(
            f"{name:<20}{stat['candidates'] / total:>10.2f}{stat['in_candidates']:>10}"
            f"{stat['in_top5']:>10}{stat['at_rank1']:>8}{mean_reciprocal_rank(stat['ranks']):>8.3f}"
        )

    print("\nQuestions where the configs disagree about the top-5 hit:")
    for question_id, hits in top5_hits.items():
        if len(set(hits.values())) > 1:
            print(f"{question_id}: {hits}")


if __name__ == "__main__":
    main()