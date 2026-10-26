import json

from config import PERSIST_PATH, EVAL_CORPUS_PATH, GOLDEN_QA_PATH
from rag_notes.adapters import ChromaVectorStore
from rag_notes.bm25_index import bm25_search
from rag_notes.embedder import load_embedding_model
from rag_notes.hybrid_search import rrf_merge, build_rank_map
from rag_notes.pipeline import make_default_chunker
from rag_notes.retrieval import build_retrieval_index


CUT = 20


def count_tied_pairs(scored: list[tuple[str, float]]) -> int:
    """Count the positions whose score equals the score of the item right before them.

    Args:
        scored: (chunk_id, score) tuples, already sorted by score, highest first
    Returns:
        number of positions i >= 1 where scored[i][1] == scored[i - 1][1]
    """
    return sum(scored[i][1] == scored[i - 1][1] for i in range(1, len(scored)))


def tie_crosses_cut(scored: list[tuple[str, float]], cut: int) -> bool:
    """Tell whether the last kept item ties with the first dropped item.

    Args:
        scored: (chunk_id, score) tuples, sorted by score, highest first
        cut: how many items are kept
    Returns:
        True if scored[cut - 1][1] == scored[cut][1]; False if there is no item after the cut
    """
    if cut <= 0:
        raise ValueError(f"cut value must be greater than 0 but found: {cut}")

    if cut >= len(scored):
        return False

    return scored[cut - 1][1] == scored[cut][1]


def main() -> None:
    """Count RRF ties in the merged first-stage list for every golden question.

    For each question: take the dense top 20 and the BM25 top 20, merge them with
    rrf_merge (the full, uncut list), then check for ties.
    Prints: questions with at least one tie anywhere in the merged list,
    questions with a tie inside the kept part (first CUT items),
    and questions where a tie crosses the cut.
    """
    model = load_embedding_model()
    chunker = make_default_chunker(model)
    vector_store = ChromaVectorStore(PERSIST_PATH)
    retrieval_index = build_retrieval_index(
        notes_root=EVAL_CORPUS_PATH,
        vector_store=vector_store,
        chunker=chunker,
        model=model,
    )

    with open(GOLDEN_QA_PATH, "r") as f:
        golden_qa_list = json.load(f)

    has_at_least_one_tie = 0
    has_ties_inside_cut_value = 0
    with_tie_crossing_the_cut = 0
    crossing_cut_list = []
    for entry in golden_qa_list:
        query = entry["question"]
        chroma_results = vector_store.query(query, model=model, n_results=CUT)
        dense_ids = chroma_results["ids"][0]

        bm25_results = bm25_search(query, retrieval_index.bm25_index, n_results=CUT)
        sparse_ids = [f"{chunk.source.title}-{chunk.chunk_index}" for (_, chunk) in bm25_results]

        merged = rrf_merge([build_rank_map(dense_ids), build_rank_map(sparse_ids)])

        has_at_least_one_tie += bool(count_tied_pairs(merged))
        has_ties_inside_cut_value += bool(count_tied_pairs(merged[:CUT]))
        if tie_crosses_cut(merged, CUT):
            with_tie_crossing_the_cut += 1
            crossing_cut_list.append((query, merged[CUT - 1], merged[CUT]))

    print(f"{len(golden_qa_list)} questions\n"
          f"{'with at least one tie in the merged list:':<43} {has_at_least_one_tie}\n"
          f"{'with a tie inside the kept first 20:':<43} {has_ties_inside_cut_value}\n"
          f"{'with a tie crossing the cut (20|21):':<43} {with_tie_crossing_the_cut}\n"
          f"ties crossing the cut:\n"
          f"{crossing_cut_list}"
    )


if __name__ == "__main__":
    main()
