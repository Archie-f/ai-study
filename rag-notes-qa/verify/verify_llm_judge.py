import json
from pathlib import Path


from config import PERSIST_PATH, EVAL_CORPUS_PATH, GOLDEN_QA_PATH
from llm_compare.providers.anthropic_provider import AnthropicProvider
from llm_compare.providers.base import LLMProvider
from llm_compare.providers.openai_provider import OpenAIProvider
from rag_notes.adapters import StructureChunker, ChromaVectorStore
from rag_notes.generate import build_context, generate_answer, normalize_answer, GUARDRAIL_ANSWER
from rag_notes.llm_judge import judge_answer
from rag_notes.models import RetrievalIndex, JudgeResult
from rag_notes.pipeline import build_default_retriever
from rag_notes.protocols import Retriever
from rag_notes.retrieval import search, build_retrieval_index


provider_llm = OpenAIProvider()
judge_llm = AnthropicProvider()


def run_judge_eval(
        retriever: Retriever,
        question: str,
        expected_answer: str,
        provider: LLMProvider,
        judge: LLMProvider
) -> JudgeResult:
    results = retriever.search(question, n=5)
    try:
        context = build_context(results)
    except ValueError:
        context = "(no context retrieved)"
        answer = GUARDRAIL_ANSWER
    else:
        result = generate_answer(question, context=context, provider=provider)
        answer = normalize_answer(result.text)

    return judge_answer(
        question=question,
        context=context,
        answer=answer,
        expected_answer=expected_answer,
        judge=judge
    )


def run_full_judge_eval(
        retriever: Retriever,
        golden_qa_path: Path,
        provider: LLMProvider,
        judge: LLMProvider
) -> list[JudgeResult]:
    with open(golden_qa_path, "r") as golden_qa_file:
        golden_qa = json.load(golden_qa_file)

    return [
        run_judge_eval(
            retriever=retriever,
            question=entry["question"],
            expected_answer=entry["expected_answer_text"],
            provider=provider,
            judge=judge
        )
        for entry in golden_qa
    ]


def main() -> None:
    vector_store = ChromaVectorStore(PERSIST_PATH)
    retriever = build_default_retriever(EVAL_CORPUS_PATH, vector_store)

    judge_results = run_full_judge_eval(
        retriever=retriever,
        golden_qa_path=GOLDEN_QA_PATH,
        provider=provider_llm,
        judge=judge_llm
    )
    print(*(f"Score: {judge_result.score} \nReason: {judge_result.reason} \nPassed?: {judge_result.passed}"
          for judge_result in judge_results), sep="\n---\n")


if __name__ == "__main__":
    main()
