import json

from config import GOLDEN_QA_PATH
from rag_notes.retrieval import search_with_rerank
from verify.verify_rerank_eval import get_retrieval_index_and_reranker
from verify.verify_rerank_scores import UNANSWERABLE_OFF_TOPIC, UNANSWERABLE_NEAR_TOPIC

THRESHOLDS = [-2.0, 0.0, 1.0, 2.0, 3.0]
TOP_K = 5


def main():
    with open(GOLDEN_QA_PATH, "r") as f:
        golden_qa = json.load(f)
    retrieval_index, reranker = get_retrieval_index_and_reranker()

    # (id, rank of the right chunk, its score), rank None = not in the 20 candidates
    right_chunks = []
    for golden in golden_qa:
        results = search_with_rerank(retrieval_index, reranker, golden["question"], top_k=20)
        found = [(rank, score) for rank, (_, score, chunk) in enumerate(results, start=1)
                 if golden["answer_key"] in chunk.text]
        rank, score = found[0] if found else (None, None)
        right_chunks.append((golden["id"], rank, score))

    unanswerable_tops = []
    for question in UNANSWERABLE_OFF_TOPIC + UNANSWERABLE_NEAR_TOPIC:
        top = search_with_rerank(retrieval_index, reranker, question, top_k=1)
        unanswerable_tops.append((question, top[0][1]))

    print(f"{'id':<6} {'rank':<6} {'score':<8}")
    for qid, rank, score in sorted(right_chunks, key=lambda r: (r[2] is None, r[2])):
        print(f"{qid:<6} {rank or '-':<6} {f'{score:.2f}' if score is not None else '-':<8}")

    in_top = [(q, r, s) for q, r, s in right_chunks if r is not None and r <= TOP_K]
    print(f"\nRight chunk in top {TOP_K}: {len(in_top)}/{len(right_chunks)}; "
          f"in top 20 only: {sum(1 for _, r, _ in right_chunks if r is not None and r > TOP_K)}; "
          f"not retrieved: {sum(1 for _, r, _ in right_chunks if r is None)}")

    print(f"\n{'threshold':<10} {'right chunk kept':<18} {'unanswerable passing':<22}")
    for t in THRESHOLDS:
        kept = sum(1 for _, _, s in in_top if s >= t)
        passing = sum(1 for _, s in unanswerable_tops if s >= t)
        print(f"{t:<10} {kept}/{len(in_top):<16} {passing}/{len(unanswerable_tops)}")


if __name__ == "__main__":
    main()