from pathlib import Path
from typing import Literal

from rag_notes.adapters import ChromaVectorStore, StructureChunker, FixedSizeChunker
from rag_notes.models import RetrievalEvalReport
from rag_notes.protocols import VectorStore, Chunker
from rag_notes.retrieval import build_retrieval_index
from verify.verify_retrieval_eval import compare_retrieval_modes, EVAL_CORPUS_PATH, GOLDEN_QA_PATH, PERSIST_PATH


def compare_chunking_strategies(
    notes_root: Path,
    vector_store: VectorStore,
    golden_qa_path: Path,
    chunkers: dict[str, Chunker],
    k_values: list[int] | None = None,
    modes: list[Literal["hybrid", "vector", "bm25"]] | None = None,
) -> dict[str, dict[str, RetrievalEvalReport]]:
    """Evaluate several chunking strategies on the same corpus and questions.

    For each named chunker: build a fresh index over notes_root (the vector
    store is reused and emptied by build_retrieval_index()), then score it
    with compare_retrieval_modes().

    Args:
        notes_root: folder containing the corpus to evaluate on
        vector_store: one VectorStore, reused for every strategy
        golden_qa_path: path to golden_qa.json
        chunkers: strategy name -> Chunker, e.g. {"structure": StructureChunker()}
        k_values: which recall@k cutoffs to report
        modes: which retrieval modes to compare
    Returns:
        {strategy name: {mode name: RetrievalEvalReport}}
    """
    results = {}
    for name, chunker in chunkers.items():
        retrieval_index = build_retrieval_index(notes_root, vector_store, chunker)
        result = compare_retrieval_modes(retrieval_index, golden_qa_path, k_values, modes)
        results[name] = result

    return results


def format_comparison_table(
    results: dict[str, dict[str, RetrievalEvalReport]],
) -> str:
    """Render chunking-comparison results as a plain-text table.

    One header line, then one line per (strategy, mode) pair, in the order
    the dicts give them. Each data line shows the strategy name, the mode,
    recall@k for every k in ascending order, and MRR, all to two decimals.

    Args:
        results: as returned by compare_chunking_strategies()
    Returns:
        the table as a single string, lines joined with newlines
    """
    if not results:
        return "Results do not exist."

    stg_mode = []
    k_values = []
    k_results = []
    mrr_results = []
    for strategy, r_eval in results.items():
        for mode, r_retrieval_report in r_eval.items():
            k_result = []
            values = []
            stg_mode.append([strategy, mode])
            mrr_results.append(r_retrieval_report.mrr)
            recall_dict_sorted = dict(sorted(r_retrieval_report.recall_k.items(), key=lambda item: item[0]))
            for element in list(recall_dict_sorted.items()):
                values.append(element[0])
                k_result.append(element[1])
            k_values.append(values)
            k_results.append(k_result)

    body = []
    for (st, m), k, mrr in zip(stg_mode, k_results, mrr_results):
        k_line = "\t".join([f"{x:.2f}" for x in k])
        body.append(f"{st:<10}\t{m:<7}\t{k_line}\t{mrr:.2f}")
    body_text = "\n".join(body)

    k_values = list(
        dict.fromkeys(k_values[0]))  # k_values[0] is used because all reports in results produce same k values
    k_header = "\t".join(f"k({k_value})" for k_value in k_values)
    header = f"{'Strategy':<10}\t{'Mode':<7}\t{k_header}\t{'MRR':<10}"

    return "\n".join([header, body_text])


def main() -> None:
    """Run the chunking comparison on the frozen eval corpus and print the table.

    Builds one ChromaVectorStore, compares StructureChunker ("structure")
    against FixedSizeChunker ("fixed") with default k values and all three
    retrieval modes, and prints format_comparison_table() of the results.
    """
    vector_s = ChromaVectorStore(PERSIST_PATH)
    chunker_st = {"structure": StructureChunker(), "fixed": FixedSizeChunker()}

    comparison_results = compare_chunking_strategies(
        EVAL_CORPUS_PATH,
        vector_s,
        GOLDEN_QA_PATH,
        chunker_st
    )
    print(format_comparison_table(comparison_results))


if __name__ == "__main__":
    main()
