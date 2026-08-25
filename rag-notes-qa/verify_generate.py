import os
from pathlib import Path

from dotenv import load_dotenv

from llm_compare.providers.ollama_provider import OllamaProvider
from rag_notes.retrieval import build_retrieval_index, search
from rag_notes.generate import build_context, generate_answer


load_dotenv()

notes_root_env = os.getenv("NOTES_ROOT")
if notes_root_env is None:
    raise RuntimeError("NOTES_ROOT not set — check your .env file")
NOTES_ROOT = Path(notes_root_env)
PERSIST_PATH = str(Path(__file__).parent / "persistent")

QUESTIONS = [
    "Why do type hints matter in Python even though the language doesn't enforce them at runtime?", # in-scope question
    "What is the capital of France?",                                                               # out-of-scope question
]


def main() -> None:
    index = build_retrieval_index(NOTES_ROOT, PERSIST_PATH
                                  )
    provider = OllamaProvider()

    for question in QUESTIONS:
        results = search(index, question, n=5)
        context = build_context(results)
        result = generate_answer(question, context, provider)

        print(f"Question: {question}")
        print(f"Answer  : {result.text}")
        print(f"(provider={result.provider}, cost=${result.cost_usd():.6f}, latency={result.latency_ms}ms)")
        print("-" * 60)


if __name__ == "__main__":
    main()
