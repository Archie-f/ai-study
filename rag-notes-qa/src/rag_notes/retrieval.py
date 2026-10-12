from pathlib import Path
from typing import Literal

from rag_notes.bm25_index import build_bm25_index
from rag_notes.embedder import load_embedding_model, embed_chunks
from rag_notes.loader import load_corpus
from rag_notes.models import RetrievalIndex, Chunk
from rag_notes.protocols import Chunker, VectorStore
from rag_notes.hybrid_search import hybrid_search


def build_retrieval_index(
        notes_root: Path,
        vector_store: VectorStore,
        chunker: Chunker
) -> RetrievalIndex:
    """Load, chunk, embed, and index an entire notes corpus, ready for search().
    The vector store is emptied first, so calling this twice with the same
    store leaves only the chunks from the second call.

    Args:
        notes_root: folder containing the week-*/*.docx corpus
        vector_store: a VectorStore Protocol object that receives the embedded chunks
        chunker: any object satisfying the Chunker Protocol. Decides how
            each document will be split into chunks
    Returns:
        a RetrievalIndex bundling the populated vector store, the loaded
        embedding model, and the built BM25 index
    """
    model = load_embedding_model()

    documents = load_corpus(notes_root)
    chunks = []
    for document in documents:
        document_chunks = chunker.chunk(document)
        chunks.extend(document_chunks)
    embedded_chunks = embed_chunks(chunks, model)
    vector_store.reset()
    vector_store.add(embedded_chunks)

    bm25_index = build_bm25_index(chunks)

    return RetrievalIndex(
        vector_store=vector_store,
        model=model,
        bm25_index=bm25_index
    )


def search(
        retrieval_index: RetrievalIndex,
        query: str,
        n: int = 3,
        mode: Literal["hybrid", "vector", "bm25"] = "hybrid",
) -> list[tuple[str, float, Chunk]]:
    """Run a hybrid search against an already-built RetrievalIndex.

    Args:
        retrieval_index: a RetrievalIndex from build_retrieval_index()
        query: raw query string
        n: how many merged results to return
        mode: "hybrid" (default), "vector", or "bm25" — forwarded to hybrid_search()
    Returns:
        top n (chunk_id, rrf_score, chunk) tuples, highest combined score first
    """
    return hybrid_search(
        query=query,
        vector_store=retrieval_index.vector_store,
        model=retrieval_index.model,
        bm25_index=retrieval_index.bm25_index,
        n=n,
        mode=mode
    )
