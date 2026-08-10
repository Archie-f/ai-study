import numpy as np
from sentence_transformers import SentenceTransformer

from rag_notes.models import Chunk, EmbeddedChunk


DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


def load_embedding_model(model_name: str = DEFAULT_MODEL_NAME) -> SentenceTransformer:
    """Load a local sentence-transformers model by name."""
    return SentenceTransformer(model_name)


def embed_chunks(chunks: list[Chunk], model: SentenceTransformer) -> list[EmbeddedChunk]:
    """Embed every chunk's text with the given model, pairing each
    Chunk with its vector.

    Args:
        chunks: chunks produced by either chunker
        model: a loaded SentenceTransformer instance
    Returns:
        list of EmbeddedChunk, same order and length as chunks
    """
    text = [chunk.text for chunk in chunks]
    vectors = model.encode(text)

    return [EmbeddedChunk(chunk, vector.tolist()) for chunk, vector in zip(chunks, vectors)]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return the cosine similarity between two equal-length vectors,
    from -1.0 (opposite direction) to 1.0 (identical direction).

    Args:
        a: first vector
        b: second vector
    Returns:
        cosine similarity
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return float(dot_product / (norm_a * norm_b))
