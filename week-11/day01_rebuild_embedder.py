from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from rag_notes.models import EmbeddedChunk, Chunk, DocumentMetadata

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


def load_embedding_model(model_name: str = DEFAULT_MODEL_NAME) -> SentenceTransformer:
    """Load a local sentence-transformers model by name."""
    return SentenceTransformer(model_name)


def embed_chunks(chunks: list[Chunk], model: SentenceTransformer) -> list[EmbeddedChunk]:
    """Embed every chunk's text with the given model, pairing each Chunk with its vector."""
    vectors = model.encode([chunk.text for chunk in chunks])
    return [EmbeddedChunk(chunk=chunk, vector=vector.tolist()) for chunk, vector in zip(chunks, vectors)]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return the cosine similarity between two equal-length vectors, from -1.0 to 1.0."""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        raise ValueError(f"Cannot calculate cosine similarity between two vectors. Check the results: norm_a = {norm_a}, norm_b = {norm_b}")

    return float(dot_product / (norm_a * norm_b))


if __name__ == "__main__":
    metadata = DocumentMetadata(week=1, day=1, file_path=Path("test.docx"), title="test")
    chunks = [
        Chunk(text="Type hints in Python are optional at runtime.", source=metadata, heading="1.1", chunk_index=0),
        Chunk(text="Static type checkers catch bugs before you run the code.", source=metadata, heading="1.2",
              chunk_index=1),
        Chunk(text="The espresso machine needs descaling every three months.", source=metadata, heading="2.1",
              chunk_index=2),
    ]

    model = SentenceTransformer(DEFAULT_MODEL_NAME)
    embedded_chunks = embed_chunks(chunks, model)
    for embedded_chunk in embedded_chunks:
        print(embedded_chunk)

    print("-" * 40)
    identical = (np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))  # expected ~1.0
    opposite = (np.array([1.0, 0.0]), np.array([-1.0, 0.0]))  # expected ~-1.0
    orthogonal = (np.array([1.0, 0.0]), np.array([0.0, 1.0]))  # expected ~0.0
    zero_vec = (np.array([1.0, 2.0]), np.array([0.0, 0.0]))  # should raise ValueError
    arrays = [identical, opposite, orthogonal, zero_vec]

    for array in arrays:
        similarity = cosine_similarity(array[0], array[1])
        print(similarity)




