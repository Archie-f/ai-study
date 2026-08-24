from pathlib import Path

from rag_notes.bm25_index import build_bm25_index
from rag_notes.embedder import load_embedding_model, embed_chunks
from rag_notes.loader import load_corpus
from rag_notes.models import RetrievalIndex, Chunk
from rag_notes.structure_chunker import chunk_document, BOUNDARY_STYLES
from rag_notes.vector_store import get_collection, add_chunks, delete_collection
from rag_notes.hybrid_search import hybrid_search


def build_retrieval_index(notes_root: Path, persist_path: str) -> RetrievalIndex:
    """Load, chunk, embed, and index an entire notes corpus, ready for search().

    Args:
        notes_root: folder containing the week-*/*.docx corpus
        persist_path: folder to store the Chroma collection in
    Returns:
        a RetrievalIndex bundling the populated collection, the loaded
        embedding model, and the built BM25 index
    """
    model = load_embedding_model()

    documents = load_corpus(notes_root)
    chunks = []
    for document in documents:
        document_chunks = chunk_document(document, BOUNDARY_STYLES)
        chunks.extend(document_chunks)
    embedded_chunks = embed_chunks(chunks, model)
    delete_collection(persist_path)
    collection = get_collection(persist_path)
    if collection:
        print(f"Collection '{collection.name}' rebuilt.")
    add_chunks(collection, embedded_chunks)

    bm25_index = build_bm25_index(chunks)

    return RetrievalIndex(collection=collection, model=model, bm25_index=bm25_index)


def search(retrieval_index: RetrievalIndex, query: str, n: int = 3) -> list[tuple[str, float, Chunk]]:
    """Run a hybrid search against an already-built RetrievalIndex.

    Args:
        retrieval_index: a RetrievalIndex from build_retrieval_index()
        query: raw query string
        n: how many merged results to return
    Returns:
        top n (chunk_id, rrf_score, chunk) tuples, highest combined score first
    """
    return hybrid_search(
        query=query,
        collection=retrieval_index.collection,
        model=retrieval_index.model,
        bm25_index=retrieval_index.bm25_index,
        n=n,
    )
