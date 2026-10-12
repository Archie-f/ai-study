from config import get_notes_root, PERSIST_PATH
from rag_notes.adapters import StructureChunker, ChromaVectorStore
from rag_notes.retrieval import build_retrieval_index, search
from rag_notes.generate import build_context
from rag_notes.citations import build_citations, format_citations

NOTES_ROOT = get_notes_root()

QUESTION = "Why do type hints matter in Python even though the language doesn't enforce them at runtime?"


def main() -> None:
    vector_store = ChromaVectorStore(PERSIST_PATH)
    index = build_retrieval_index(NOTES_ROOT, vector_store, StructureChunker())
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