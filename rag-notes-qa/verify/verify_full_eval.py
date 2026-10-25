from pathlib import Path

from config import PERSIST_PATH, EVAL_CORPUS_PATH, GOLDEN_QA_PATH
from llm_compare.providers.anthropic_provider import AnthropicProvider
from llm_compare.providers.openai_provider import OpenAIProvider
from rag_notes.adapters import StructureChunker, ChromaVectorStore
from rag_notes.eval_report import save_full_eval_html
from rag_notes.pipeline import build_default_retriever
from rag_notes.retrieval import build_retrieval_index
from verify.verify_llm_judge import run_full_judge_eval
from verify.verify_retrieval_eval import compare_retrieval_modes


REPORT_PATH = Path(__file__).parent.parent / "reports"
provider_llm = OpenAIProvider()
judge_llm = AnthropicProvider()


def main() -> None:
    vector_store = ChromaVectorStore(PERSIST_PATH)
    retriever = build_default_retriever(EVAL_CORPUS_PATH, vector_store)
    mode_reports = compare_retrieval_modes(
        retriever.retrieval_index, GOLDEN_QA_PATH, level="chunk", reranker=retriever.reranker
    )
    judge_results = run_full_judge_eval(retriever, GOLDEN_QA_PATH, provider_llm, judge_llm)

    report_path = save_full_eval_html(mode_reports, judge_results, REPORT_PATH)
    print(f"Eval report saved to {report_path}")


if __name__ == "__main__":
    main()
