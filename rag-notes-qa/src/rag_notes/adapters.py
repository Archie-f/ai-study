from typing import Literal

from chromadb import QueryResult
from sentence_transformers import SentenceTransformer

from rag_notes.fixed_chunker import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, chunk_fixed_size, chunk_fixed_size_document
from rag_notes.models import SourceDocument, Chunk, EmbeddedChunk, RetrievalIndex
from rag_notes.retrieval import search
from rag_notes.structure_chunker import BOUNDARY_STYLES, chunk_document
from rag_notes.vector_store import COLLECTION_NAME, get_collection, add_chunks, get_query_result


class StructureChunker:
    def __init__(self, boundary_styles: set[str] = BOUNDARY_STYLES):
        self.boundary_styles = boundary_styles

    def chunk(self, document: SourceDocument) -> list[Chunk]:
        """Chunk a document along its heading boundaries (wraps chunk_document()).

        Args:
            document (SourceDocument): The document to chunk.
        Returns:
            list[Chunk]: A list of Chunks.
        """
        return chunk_document(document, self.boundary_styles)


class FixedSizeChunker:
    def __init__(self, n: int = DEFAULT_CHUNK_SIZE, o: int = DEFAULT_OVERLAP):
        self.n = n
        self.o = o

    def chunk(self, document: SourceDocument) -> list[Chunk]:
        """Chunk a document by fixed token count (wraps chunk_fixed_size() + chunk_fixed_size_document()).

        Args:
            document (SourceDocument): The document to chunk.
        Returns:
            list[Chunk]: A list of Chunks.
        """
        paragraph_texts = [paragraph[0] for paragraph in document.paragraphs]
        doc_text = "\n".join(paragraph_texts)
        chunk_texts = chunk_fixed_size(doc_text, self.n, self.o)
        return chunk_fixed_size_document(
            chunk_texts,
            document.metadata
        )


class ChromaVectorStore:
    def __init__(self, persist_path: str, name: str = COLLECTION_NAME):
        self.collection = get_collection(persist_path, name)

    def add(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        """Add embedded chunks to the underlying Chroma collection (wraps add_chunks()).

        Args:
            embedded_chunks (list[EmbeddedChunk]): The chunks to add.
        """
        add_chunks(self.collection, embedded_chunks)

    def query(self, query: str, model: SentenceTransformer, n_results: int = 3) -> QueryResult:
        """Query the underlying Chroma collection (wraps get_query_result()).

        Args:
            query (str): The query to query.
            model (SentenceTransformer): The SentenceTransformer model.
            n_results (int): The number of nearest chunks to return.
        Returns:
            QueryResult: The query result.
        """
        return get_query_result(
            self.collection,
            query,
            model,
            n_results
        )


class HybridRetriever:
    def __init__(self, retrieval_index: RetrievalIndex):
        self.retrieval_index = retrieval_index

    def search(
        self,
        query: str,
        n: int = 3,
        mode: Literal["hybrid", "vector", "bm25"] = "hybrid",
    ) -> list[tuple[str, float, Chunk]]:
        """Search this retriever's index (wraps search()/hybrid_search())."""
        return search(
            self.retrieval_index,
            query,
            n,
            mode
        )
