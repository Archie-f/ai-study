import json
import os
from pathlib import Path

from dotenv import load_dotenv

from rag_notes.models import RetrievalIndex, RetrievalEvalReport
from rag_notes.retrieval import build_retrieval_index, search
from rag_notes.retrieval_metrics import recall_at_k, reciprocal_rank, mean_reciprocal_rank

load_dotenv()
NOTES_ROOT = Path(os.getenv("NOTES_ROOT"))
PERSIST_PATH = str(Path(__file__).parent.parent / "persistent")
GOLDEN_QA_PATH = Path(__file__).parent.parent / "data" / "golden_qa.json"

def run_retrieval_eval(
    retrieval_index: RetrievalIndex,
    golden_qa_path: Path,
    k_values: list[int] | None = None,
) -> RetrievalEvalReport:
    """Run every golden_qa.json question through search() and score retrieval quality.

    Args:
        retrieval_index: a built RetrievalIndex, as returned by build_retrieval_index()
        golden_qa_path: path to golden_qa.json
        k_values: which recall@k cutoffs to report
    Returns:
        a RetrievalEvalReport with recall@k for each k in k_values, plus MRR,
        each averaged across every question in the golden set
    """
    if k_values is None:
        k_values = [1, 3, 5]

    with open(golden_qa_path, "r") as golden_qa_file:
        golden_qa = json.load(golden_qa_file)

    recall_k: dict[int, float] = {}
    recall_values: list[list[float]] = []
    ranks: list[float] = []
    for index, entry in enumerate(golden_qa, start=1):
        results = search(retrieval_index, entry["question"], max(k_values))
        retrieved_ids = [result[2].source.title for result in results]
        relevant_id = {entry["source_document"]}

        recalls = []
        for k in k_values:
            recall = recall_at_k(retrieved_ids, relevant_id, k)
            recalls.append(recall)

        recall_values.append(recalls)
        rank = reciprocal_rank(retrieved_ids, relevant_id)
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


def main():
    index = build_retrieval_index(NOTES_ROOT, PERSIST_PATH)
    report = run_retrieval_eval(index, GOLDEN_QA_PATH)
    print(report)


if __name__ == "__main__":
    main()


