import os
from pathlib import Path

from dotenv import load_dotenv

from llm_compare.providers.anthropic_provider import AnthropicProvider
from llm_compare.providers.openai_provider import OpenAIProvider
from rag_notes.adapters import StructureChunker, ChromaVectorStore
from rag_notes.eval_report import save_full_eval_html
from rag_notes.retrieval import build_retrieval_index
from verify.verify_llm_judge import run_full_judge_eval
from verify.verify_retrieval_eval import compare_retrieval_modes, GOLDEN_QA_PATH

load_dotenv()
notes_root_env = os.getenv("NOTES_ROOT")
if notes_root_env is None:
    raise RuntimeError("NOTES_ROOT not set — check your .env file")
NOTES_ROOT = Path(notes_root_env)
PERSIST_PATH = str(Path(__file__).parent.parent / "persistent")
REPORT_PATH = Path(__file__).parent.parent / "reports"
provider_llm = OpenAIProvider()
judge_llm = AnthropicProvider()


def main() -> None:
    vector_store = ChromaVectorStore(PERSIST_PATH)
    retrieval_index = build_retrieval_index(NOTES_ROOT, vector_store, StructureChunker())
    mode_reports = compare_retrieval_modes(retrieval_index, GOLDEN_QA_PATH)
    judge_results = run_full_judge_eval(retrieval_index, GOLDEN_QA_PATH, provider_llm, judge_llm)

    report_path = save_full_eval_html(mode_reports, judge_results, REPORT_PATH)
    print(f"Eval report saved to {report_path}")


if __name__ == "__main__":
    main()
