from pathlib import Path

from config import get_notes_root
from llm_compare.providers.ollama_provider import OllamaProvider
from rag_notes.retrieval import build_retrieval_index
from rag_notes.generate import display_answer, answer_question

notes_root_env = get_notes_root()
NOTES_ROOT = notes_root_env
PERSIST_PATH = str(Path(__file__).parent / "persistent")

QUESTIONS = [
    "Why do type hints matter in Python even though the language doesn't enforce them at runtime?", # in-scope question
    "What is the capital of France?",                                                               # out-of-scope question
]


def main() -> None:
    index = build_retrieval_index(NOTES_ROOT, PERSIST_PATH)
    provider = OllamaProvider(temperature=0)

    for question in QUESTIONS:
        answered = answer_question(
            index=index,
            question=question,
            provider=provider,
        )
        display = display_answer(answered)
        print(display)

        print("-" * 60)


if __name__ == "__main__":
    main()
