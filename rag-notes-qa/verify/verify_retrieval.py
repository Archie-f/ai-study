from config import get_notes_root, PERSIST_PATH
from rag_notes.adapters import StructureChunker, ChromaVectorStore
from rag_notes.retrieval import build_retrieval_index, search


NOTES_ROOT = get_notes_root()
QUERY = "Why do type hints matter in Python even though the language doesn't enforce them at runtime?"


def main() -> None:
    vector_store = ChromaVectorStore(PERSIST_PATH)
    index = build_retrieval_index(NOTES_ROOT, vector_store, StructureChunker())
    results = search(index, QUERY, n=5)
    for rank, (chunk_id, score, chunk) in enumerate(results, start=1):
        heading = chunk.heading or "(no heading)"
        print(f"Rank: {rank} | Score: {score:.6f}| Heading: {heading}")
        # ... prints rank, score, heading, and flags the expected match





if __name__ == "__main__":
    main()