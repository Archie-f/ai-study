from pathlib import Path

from sentence_transformers import CrossEncoder, SentenceTransformer

from rag_notes.adapters import CrossEncoderReranker, RerankingRetriever, StructureThenSizeChunker
from rag_notes.embedder import load_embedding_model
from rag_notes.protocols import VectorStore
from rag_notes.retrieval import build_retrieval_index

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MIN_RERANK_SCORE = 0.0  #calibrated using 48 golden questions (Week 17 Day 05)


def make_default_chunker(model: SentenceTransformer) -> StructureThenSizeChunker:
    """Structure-first chunker whose size limit fits the embedding model's window."""
    return StructureThenSizeChunker(tokenizer=model.tokenizer, limit=model.max_seq_length - 2)


def build_default_retriever(
        notes_root: Path,
        vector_store: VectorStore,
        min_score: float = MIN_RERANK_SCORE,
) -> RerankingRetriever:
    """Build the production pipeline: new chunker, hybrid first stage, cross-encoder, threshold.

    Each model is loaded once here.
    """
    model = load_embedding_model()
    index = build_retrieval_index(notes_root, vector_store, make_default_chunker(model), model=model)
    reranker = CrossEncoderReranker(CrossEncoder(RERANKER_MODEL_NAME))
    return RerankingRetriever(index, reranker, min_score)