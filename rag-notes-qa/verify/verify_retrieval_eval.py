import json
from pathlib import Path
from typing import Literal

from config import PERSIST_PATH, EVAL_CORPUS_PATH, GOLDEN_QA_PATH
from rag_notes.adapters import StructureChunker, ChromaVectorStore
from rag_notes.models import RetrievalIndex, RetrievalEvalReport
from rag_notes.retrieval import build_retrieval_index, search
from rag_notes.retrieval_metrics import recall_at_k, reciprocal_rank, mean_reciprocal_rank, find_relevant_chunk_ids

mode_list = ["hybrid", "vector", "bm25"]

def run_retrieval_eval(
    retrieval_index: RetrievalIndex,
    golden_qa_path: Path,
    k_values: list[int] | None = None,
    mode: Literal["hybrid", "vector", "bm25"] = "hybrid",
    level: Literal["document", "chunk"] = "document",
) -> RetrievalEvalReport:
    """Run every golden_qa.json question through search() and score retrieval quality.

    Args:
        retrieval_index: a built RetrievalIndex, as returned by build_retrieval_index()
        golden_qa_path: path to golden_qa.json
        k_values: which recall@k cutoffs to report
        mode: "hybrid" (default), "vector", or "bm25" — forwarded to hybrid_search()
        level: "document" (default) - a hit is any chunk from source_document;
               "chunk" - a hit is a chunk whose text contains answer_key
    Returns:
        a RetrievalEvalReport with recall@k for each k in k_values, plus MRR,
        each averaged across every question in the golden set
    Raises:
        ValueError: at chunk level, if a question has no relevant chunk
    """
    if k_values is None:
        k_values = [1, 3, 5]

    with open(golden_qa_path, "r") as golden_qa_file:
        golden_qa = json.load(golden_qa_file)

    recall_k: dict[int, float] = {}
    recall_values: list[list[float]] = []
    ranks: list[float] = []
    for index, entry in enumerate(golden_qa, start=1):
        results = search(retrieval_index, entry["question"], max(k_values), mode=mode)
        if level == "document":
            retrieved_ids = [result[2].source.title for result in results]
            relevant_ids = {entry["source_document"]}
        elif level == "chunk":
            retrieved_ids = [result[0] for result in results]
            relevant_ids = find_relevant_chunk_ids(retrieval_index.bm25_index.chunks, entry["answer_key"])
            if len(relevant_ids) == 0:
                raise ValueError(f"No relevant chunks found for question: '{entry["id"]}'")
        else:
            raise ValueError(f"Unknown level: {level}")

        recalls = []
        for k in k_values:
            recall = recall_at_k(retrieved_ids, relevant_ids, k)
            recalls.append(recall)

        recall_values.append(recalls)
        rank = reciprocal_rank(retrieved_ids, relevant_ids)
        ranks.append(rank)

    for index, k in enumerate(k_values):
        numerator = sum(recall[index] for recall in recall_values)
        denominator = len(recall_values)
        recall_k[k] = numerator / denominator

    mrr = mean_reciprocal_rank(ranks)

    return RetrievalEvalReport(
        recall_k=recall_k,
        mrr=mrr
    )


def compare_retrieval_modes(
    retrieval_index: RetrievalIndex,
    golden_qa_path: Path,
    k_values: list[int] | None = None,
    modes: list[Literal["hybrid", "vector", "bm25"]] | None = None,
) -> dict[str, RetrievalEvalReport]:
    """Run run_retrieval_eval() once per retrieval mode and collect the results.

    Args:
        retrieval_index: a built RetrievalIndex, as returned by build_retrieval_index()
        golden_qa_path: path to golden_qa.json
        k_values: which recall@k cutoffs to report
        modes: which modes to compare (defaults to hybrid, vector, and bm25)
    Returns:
        a dict mapping each mode name to its RetrievalEvalReport
    """
    if modes is None:
        modes = ["hybrid", "vector", "bm25"]
    return {mode: run_retrieval_eval(retrieval_index, golden_qa_path, k_values, mode) for mode in modes}


def main():
    vector_store = ChromaVectorStore(PERSIST_PATH)
    index = build_retrieval_index(EVAL_CORPUS_PATH, vector_store, StructureChunker())
    report = run_retrieval_eval(index, GOLDEN_QA_PATH)
    print(report)
    print("-" * 20)
    reports = compare_retrieval_modes(index, GOLDEN_QA_PATH, modes=mode_list)
    print(reports)



if __name__ == "__main__":
    main()


