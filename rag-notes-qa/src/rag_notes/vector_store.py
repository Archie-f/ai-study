import chromadb
from chromadb import Collection, QueryResult
from sentence_transformers import SentenceTransformer

from rag_notes.models import EmbeddedChunk

COLLECTION_NAME = "ai_study_notes"


def get_collection(persist_path: str, name: str = COLLECTION_NAME) -> Collection:
    """Open (or create) a cosine-configured Chroma collection on disk.

    Args:
        persist_path: folder to store the Chroma database in
        name: collection name
    Returns:
        the collection, ready for add() or query()
    """
    client = chromadb.PersistentClient(path=persist_path)
    return client.get_or_create_collection(
        name=name,
        configuration={"hnsw": {"space": "cosine"}},
    )


def build_metadata(chunk) -> dict:
    """Convert a Chunk's fields into a Chroma-safe metadata dict.

    Chroma rejects None at runtime despite what the type hints say,
    so any optional field needs a sentinel instead.

    Args:
        chunk: the Chunk this metadata describes
    Returns:
        dict with only str | int | float | bool values, no None
    """
    return {
        "source": chunk.source.title,
        "heading": chunk.heading or "",
        "chunk_index": chunk.chunk_index,
    }


def add_chunks(collection, embedded_chunks: list[EmbeddedChunk]) -> None:
    """Add a batch of EmbeddedChunks to a Chroma collection.

    Args:
        collection: a Chroma collection (already created/opened)
        embedded_chunks: chunks paired with their vectors, from embed_chunks()
    """
    ids = [f"{embedded_chunk.chunk.source.title}-{embedded_chunk.chunk.chunk_index}" for embedded_chunk in embedded_chunks]
    collection.add(
        ids=ids,
        embeddings=[embedded_chunk.vector for embedded_chunk in embedded_chunks],
        metadatas=[build_metadata(embedded_chunk.chunk) for embedded_chunk in embedded_chunks],
        documents=[embedded_chunk.chunk.text for embedded_chunk in embedded_chunks],
    )


def get_query_result(
        collection,
        query: str,
        model: SentenceTransformer,
        n_results: int = 3,
        includes: tuple[str, ...] = ("metadatas", "documents", "distances")
) -> QueryResult:
    """Get a QueryResult from a query string.
    Args:
        collection: a Chroma collection (already created/opened)
        query: query string
        model: SentenceTransformer model
        n_results: number of results to return (3 by default)
        includes: tuple of strings indicating which fields to include
    Returns:
        a QueryResult object
    """
    query_vector = model.encode(query)
    return collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=n_results,
        include=list(includes),
    )
