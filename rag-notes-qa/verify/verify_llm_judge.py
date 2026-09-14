import json
import os
from pathlib import Path

from dotenv import load_dotenv

from llm_compare.providers.anthropic_provider import AnthropicProvider
from llm_compare.providers.ollama_provider import OllamaProvider
from llm_compare.providers.openai_provider import OpenAIProvider
from rag_notes.generate import build_context, generate_answer, normalize_answer
from rag_notes.llm_judge import judge_answer
from rag_notes.retrieval import search, build_retrieval_index
from verify.verify_retrieval_eval import GOLDEN_QA_PATH

load_dotenv()
notes_root_env = os.getenv("NOTES_ROOT")
if notes_root_env is None:
    raise RuntimeError("NOTES_ROOT not set — check your .env file")
NOTES_ROOT = Path(notes_root_env)
PERSIST_PATH = str(Path(__file__).parent.parent / "persistent")
GOLDEN_QA_PATH = str(Path(__file__).parent.parent / "data" / "golden_qa.json")
provider_llm = OpenAIProvider()
judge_llm = AnthropicProvider()


def main() -> None:
    with open(GOLDEN_QA_PATH, "r") as golden_qa_file:
        golden_qa = json.load(golden_qa_file)

    retrieval_index = build_retrieval_index(NOTES_ROOT, PERSIST_PATH)
    for index, entry in enumerate(golden_qa, start=1):
        question = entry["question"]
        expected_answer = entry["expected_answer_text"]

        results = search(retrieval_index, entry["question"], n=5)
        context = build_context(results)
        result = generate_answer(question, context, provider=provider_llm)
        normalized_answer = normalize_answer(result.text)
        judge_result = judge_answer(
            question=question,
            context=context,
            answer=normalized_answer,
            expected_answer=expected_answer,
            judge=judge_llm
        )
        print(f"Score: {judge_result.score} \nReason: {judge_result.reason} \nPassed?: {judge_result.passed}")


if __name__ == "__main__":
    main()
