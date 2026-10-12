from typing import Protocol, Literal

from chromadb import QueryResult
from sentence_transformers import SentenceTransformer

from rag_notes.models import SourceDocument, Chunk, EmbeddedChunk


class Chunker(Protocol):
    def chunk(self, document: SourceDocument) -> list[Chunk]:
        """Split one SourceDocument into Chunks."""
        ...


class VectorStore(Protocol):
    def add(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        """Add a batch of embedded chunks to the store."""
        ...

    def query(self, query: str, model: SentenceTransformer, n_results: int = 3) -> QueryResult:
        """Return the n_results nearest chunks to query."""
        ...

    def reset(self) -> None:
        """Remove everything in the store, leaving it empty and ready for add()."""
        ...


class Retriever(Protocol):
    def search(
        self,
        query: str,
        n: int = 3,
        mode: Literal["hybrid", "vector", "bm25"] = "hybrid",
    ) -> list[tuple[str, float, Chunk]]:
        """Return the top n (chunk_id, score, chunk) results for query."""
        ...
