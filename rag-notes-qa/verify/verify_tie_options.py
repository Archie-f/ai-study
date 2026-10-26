import json

from config import GOLDEN_QA_PATH
from rag_notes.bm25_index import bm25_search
from rag_notes.hybrid_search import build_rank_map, rrf_merge
from rag_notes.retrieval_metrics import (
    find_relevant_chunk_ids,
    mean_reciprocal_rank,
    recall_at_k,
    reciprocal_rank,
)
from verify.verify_rerank_eval import get_retrieval_index_and_reranker
from verify.verify_rrf_ties import CUT

TOP_K = 5


def prefer_vector_side(scored: list[tuple[str, float]], vector_ids: list[str]) -> list[tuple[str, float]]:
    """Re-sort a merged RRF list so that, among equal scores, ids found by the vector side come first.

    Example: scored = [('c', 0.3), ('a', 0.2), ('b', 0.2)], vector_ids = ['b']
    -> [('c', 0.3), ('b', 0.2), ('a', 0.2)]. Scores still decide first; remaining
    ties are ordered by id, as in rrf_merge().

    Args:
        scored: (chunk_id, score) tuples from rrf_merge(), highest score first
        vector_ids: ids returned by the vector (dense) search
    Returns:
        a new list with the same items
    """
    vector_set = set(vector_ids)
    return sorted(scored, key=lambda item: (-item[1], item[0] not in vector_set, item[0]))


def keep_tied_at_cut(scored: list[tuple[str, float]], cut: int) -> list[tuple[str, float]]:
    """Keep the first cut items, plus every following item whose score equals the last kept score.

    Example: scored = [('a', .5), ('b', .4), ('c', .4), ('d', .4), ('e', .3)], cut = 2
    -> [('a', .5), ('b', .4), ('c', .4), ('d', .4)]. With cut = 1 -> [('a', .5)].

    Args:
        scored: (chunk_id, score) tuples, highest score first
        cut: how many items to keep at least
    Returns:
        a new list with at least min(cut, len(scored)) items
    Raises:
        ValueError: if cut is not greater than 0
    """
    if cut <= 0:
        raise ValueError(f"cut value must be greater than 0 but found: {cut}")

    if cut >= len(scored):
        return list(scored)

    last_kept_score = scored[cut - 1][1]
    end = cut
    while end < len(scored) and scored[end][1] == last_kept_score:
        end += 1

    return scored[:end]


RULES = {
    "A id order": lambda merged, vector_ids: merged[:CUT],
    "B vector first": lambda merged, vector_ids: prefer_vector_side(merged, vector_ids)[:CUT],
    "C keep ties": lambda merged, vector_ids: keep_tied_at_cut(merged, CUT),
}


def main() -> None:
    """Compare three ways of cutting the merged RRF list at CUT, on all golden questions.

    Every rule sees the same dense/BM25 lists per question, so differences come
    only from the cut rule. Each candidate set goes through the real reranker
    (top TOP_K). The relevance threshold is not applied: this measures ranking only.
    Prints, per rule: average candidate count, questions whose right chunk is in the
    candidates, in the top 5, at rank 1 (all out of the question count), and MRR;
    then the questions where the rules disagree about the top-5 hit.
    """
    retrieval_index, reranker = get_retrieval_index_and_reranker()
    chunks = retrieval_index.bm25_index.chunks
    id_to_chunk = {f"{chunk.source.title}-{chunk.chunk_index}": chunk for chunk in chunks}

    with open(GOLDEN_QA_PATH, "r") as f:
        golden_qa_list = json.load(f)

    stats = {
        name: {"candidates": 0, "in_candidates": 0, "in_top5": 0, "at_rank1": 0, "ranks": []}
        for name in RULES
    }
    top5_hits: dict[str, dict[str, bool]] = {}

    for entry in golden_qa_list:
        query = entry["question"]
        relevant_ids = find_relevant_chunk_ids(chunks, entry["answer_key"])

        dense_ids = retrieval_index.vector_store.query(
            query, retrieval_index.model, n_results=CUT
        )["ids"][0]
        bm25_results = bm25_search(query, retrieval_index.bm25_index, n_results=CUT)
        sparse_ids = [f"{chunk.source.title}-{chunk.chunk_index}" for _, chunk in bm25_results]
        merged = rrf_merge([build_rank_map(dense_ids), build_rank_map(sparse_ids)])

        top5_hits[entry["id"]] = {}
        for name, rule in RULES.items():
            chosen = rule(merged, dense_ids)
            candidates = [(chunk_id, score, id_to_chunk[chunk_id]) for chunk_id, score in chosen]
            reranked = reranker.rerank(query, candidates, TOP_K)
            reranked_ids = [chunk_id for chunk_id, _, _ in reranked]

            stat = stats[name]
            stat["candidates"] += len(chosen)
            stat["in_candidates"] += bool(relevant_ids & {chunk_id for chunk_id, _ in chosen})
            stat["in_top5"] += recall_at_k(reranked_ids, relevant_ids, TOP_K) > 0
            stat["at_rank1"] += recall_at_k(reranked_ids, relevant_ids, 1) > 0
            stat["ranks"].append(reciprocal_rank(reranked_ids, relevant_ids))
            top5_hits[entry["id"]][name] = recall_at_k(reranked_ids, relevant_ids, TOP_K) > 0

    total = len(golden_qa_list)
    print(f"{total} questions, cut = {CUT}, reranked top {TOP_K}")
    print(f"{'rule':<16}{'avg cand.':>10}{'in cand.':>10}{'in top 5':>10}{'rank 1':>8}{'MRR':>8}")
    for name, stat in stats.items():
        print(
            f"{name:<16}{stat['candidates'] / total:>10.2f}{stat['in_candidates']:>10}"
            f"{stat['in_top5']:>10}{stat['at_rank1']:>8}{mean_reciprocal_rank(stat['ranks']):>8.3f}"
        )

    print("\nQuestions where the rules disagree about the top-5 hit:")
    for question_id, hits in top5_hits.items():
        if len(set(hits.values())) > 1:
            print(f"{question_id}: {hits}")


if __name__ == "__main__":
    main()
