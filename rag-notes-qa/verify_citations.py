import os
from pathlib import Path

from dotenv import load_dotenv

from rag_notes.retrieval import build_retrieval_index, search
from rag_notes.generate import build_context
from rag_notes.citations import build_citations, format_citations


load_dotenv()

notes_root_env = os.getenv("NOTES_ROOT")
if notes_root_env is None:
    raise RuntimeError("NOTES_ROOT not set — check your .env file")
NOTES_ROOT = Path(notes_root_env)
PERSIST_PATH = str(Path(__file__).parent / "persistent")

QUESTION = "Why do type hints matter in Python even though the language doesn't enforce them at runtime?"


def main() -> None:
    index = build_retrieval_index(NOTES_ROOT, PERSIST_PATH)
    results = search(index, QUESTION, n=5)

    context = build_context(results)
    citations = build_citations(results)

    print("--- build_context() output ---")
    print(context)
    print()
    print("--- format_citations() output ---")
    print(format_citations(citations))


if __name__ == "__main__":
    main()