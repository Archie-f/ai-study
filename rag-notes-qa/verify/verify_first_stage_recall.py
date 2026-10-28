import json

from config import GOLDEN_QA_PATH, EVAL_CORPUS_PATH, PERSIST_PATH
from rag_notes.adapters import ChromaVectorStore
from rag_notes.bm25_index import bm25_search
from rag_notes.embedder import load_embedding_model
from rag_notes.hybrid_search import rrf_merge, build_rank_map
from rag_notes.pipeline import make_default_chunker
from rag_notes.retrieval import build_retrieval_index
from rag_notes.retrieval_metrics import find_relevant_chunk_ids

DEPTH = 100
CUT = 20
CUTOFFS = [20, 30, 40, 60, 100]


def first_rank(ranked_ids: list[str], relevant_ids: set[str]) -> int | None:
    """Return the 1-based position of the first id that is in relevant_ids, or None."""
    for index, ranked_id in enumerate(ranked_ids, start=1):
        if ranked_id in relevant_ids:
            return index
    return None


def count_within(ranks: list[int | None], cutoffs: list[int]) -> dict[int, int]:
    """For each cutoff, count the ranks that are not None and are <= that cutoff."""
    valid_ranks = [rank for rank in ranks if rank is not None]

    result = {}
    for cutoff in cutoffs:
        if cutoff <= 0:
            raise ValueError(f"cutoff must be > 0 but found: {cutoff}")
        result[cutoff] = sum(1 for rank in valid_ranks if rank <= cutoff)
    return result


def main() -> None:
    """Find where the right chunk sits in the dense, BM25 and merged lists (depth 100).

    For each golden question: first_rank in the three lists.
    Prints: a table of count_within for each list at CUTOFFS, then every question
    whose merged rank is None or above CUT (id, notes label, three ranks,
    first 80 characters of the question).
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

    dense_ranks = []
    sparse_ranks = []
    merged_ranks = []
    missed = []
    for entry in golden_qa_list:
        query = entry["question"]
        # TODO 1: dense_ids  (vector_store.query, n_results=DEPTH)
        dense_ids = vector_store.query(query, model, n_results=DEPTH)["ids"][0]
        # TODO 2: sparse_ids (bm25_search, n_results=DEPTH)
        sparse_ids = [f"{chunk.source.title}-{chunk.chunk_index}" for _, chunk in bm25_search(query, retrieval_index.bm25_index, n_results=DEPTH)]
        # TODO 3: merged_ids (rrf_merge of the two rank maps, ids only)
        merged_ids = [chunk_id for chunk_id, _ in rrf_merge([build_rank_map(dense_ids), build_rank_map(sparse_ids)])]
        # TODO 4: relevant_ids (find_relevant_chunk_ids on retrieval_index.bm25_index.chunks)
        relevant_ids = find_relevant_chunk_ids(retrieval_index.bm25_index.chunks, entry["answer_key"])
        # TODO 5: three first_rank values, appended to the three lists
        dense_rank = first_rank(dense_ids, relevant_ids)
        sparse_rank = first_rank(sparse_ids, relevant_ids)
        merged_rank = first_rank(merged_ids, relevant_ids)
        dense_ranks.append(dense_rank)
        sparse_ranks.append(sparse_rank)
        merged_ranks.append(merged_rank)
        # TODO 6: if merged rank is None or > CUT, append what you want to print to missed
        if merged_rank is None or merged_rank > CUT:
            missed.append((entry["id"], entry["notes"], dense_rank, sparse_rank, merged_rank, query[:80]))

    # TODO 7: print the table (count_within for each of the three lists)
    count_dense = count_within(dense_ranks, CUTOFFS)
    count_sparse = count_within(sparse_ranks, CUTOFFS)
    count_merged = count_within(merged_ranks, CUTOFFS)
    body_counts = f"{'dense':<10}{count_dense}\n{'sparse':<10}{count_sparse}\n{'merged':<10}{count_merged}"
    print("Counts Within Cutoff Values:")
    print(body_counts)
    print("-" * 20)
    # TODO 8: print the missed questions
    print("Missed Questions:")
    body_missed = [f"{m[0]:<6}| {m[1][:11]:<12}| {'dense':<7}{m[2]} | "
                   f"{'sparse':<7}{m[3]} | {'merged':<7}{m[4]} | "
                   f"{m[5]}..." for m in missed]
    print(*body_missed, sep="\n")
    print("-" * 20)
    # (TODO 9, optional: how many missed have notes == "paraphrase")
    paraphrase = sum(1 for m in missed if m[1] == "paraphrase")
    print(f"{paraphrase} of {len(missed)} missed questions have 'paraphrase' tag.")


if __name__ == "__main__":
    main()
