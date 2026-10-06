import json
import os
from pathlib import Path

from dotenv import load_dotenv

from llm_compare.providers.anthropic_provider import AnthropicProvider
from llm_compare.providers.base import LLMProvider
from llm_compare.providers.openai_provider import OpenAIProvider
from rag_notes.adapters import StructureChunker, ChromaVectorStore
from rag_notes.generate import build_context, generate_answer, normalize_answer
from rag_notes.llm_judge import judge_answer
from rag_notes.models import RetrievalIndex, JudgeResult
from rag_notes.retrieval import search, build_retrieval_index
from verify.verify_retrieval_eval import GOLDEN_QA_PATH

load_dotenv()
notes_root_env = os.getenv("NOTES_ROOT")
if notes_root_env is None:
    raise RuntimeError("NOTES_ROOT not set — check your .env file")
NOTES_ROOT = Path(notes_root_env)
PERSIST_PATH = str(Path(__file__).parent.parent / "persistent")
provider_llm = OpenAIProvider()
judge_llm = AnthropicProvider()


def run_judge_eval(
        retrieval_index: RetrievalIndex,
        question: str,
        expected_answer: str,
        provider: LLMProvider,
        judge: LLMProvider
) -> JudgeResult:
    results = search(retrieval_index, question, n=5)
    context = build_context(results)

    result = generate_answer(question, context=context, provider=provider)
    normalized_answer = normalize_answer(result.text)

    return judge_answer(
        question=question,
        context=context,
        answer=normalized_answer,
        expected_answer=expected_answer,
        judge=judge
    )


def run_full_judge_eval(
        retrieval_index: RetrievalIndex,
        golden_qa_path: Path,
        provider: LLMProvider,
        judge: LLMProvider
) -> list[JudgeResult]:
    with open(golden_qa_path, "r") as golden_qa_file:
        golden_qa = json.load(golden_qa_file)

    return [
        run_judge_eval(
            retrieval_index=retrieval_index,
            question=entry["question"],
            expected_answer=entry["expected_answer_text"],
            provider=provider,
            judge=judge
        )
        for entry in golden_qa
    ]


def main() -> None:
    vector_store = ChromaVectorStore(PERSIST_PATH)
    retrieval_index = build_retrieval_index(NOTES_ROOT, vector_store, StructureChunker())

    judge_results = run_full_judge_eval(
        retrieval_index=retrieval_index,
        golden_qa_path=GOLDEN_QA_PATH,
        provider=provider_llm,
        judge=judge_llm
    )
    print(*(f"Score: {judge_result.score} \nReason: {judge_result.reason} \nPassed?: {judge_result.passed}"
          for judge_result in judge_results), sep="\n---\n")


if __name__ == "__main__":
    main()
