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


def count_wordpiece_tokens(text: str, tokenizer) -> int:
    """Count the WordPiece pieces of one text, without special tokens.

    Args:
        text: any string
        tokenizer: the embedder's tokenizer (model.tokenizer)
    Returns:
        number of pieces, e.g. 'embeddings' -> 4
    """
    return len(tokenizer.tokenize(text))


def wordpiece_counts(chunks: list[Chunk], tokenizer) -> list[int]:
    """Count the WordPiece pieces of every chunk text.

    Args:
        chunks: chunks from any chunker
        tokenizer: the embedder's tokenizer
    Returns:
        one count per chunk, same order and length as chunks
    """
    return [count_wordpiece_tokens(chunk.text, tokenizer) for chunk in chunks]


def chunk_ratios(chunks: list[Chunk], tokenizer, encoding) -> list[float]:
    """WordPiece-to-tiktoken ratio of every chunk.

    Args:
        chunks: chunks from any chunker
        tokenizer: the embedder's tokenizer (model.tokenizer)
        encoding: the tiktoken encoding used by the fixed chunker
    Returns:
        one ratio per chunk with a non-empty text, in chunk order,
        e.g. a chunk with 260 WordPiece and 200 tiktoken tokens -> 1.3
    """
    wordpiece_counts_list = wordpiece_counts(chunks, tokenizer)
    tiktoken_counts_list = [len(encoding.encode(chunk.text)) for chunk in chunks]

    return [(w / t) for w, t in zip(wordpiece_counts_list, tiktoken_counts_list) if t > 0]


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

    if norm_a == 0 or norm_b == 0:
        raise ValueError(f"Cannot calculate cosine similarity between two vectors. Check the results: norm_a = {norm_a}, norm_b = {norm_b}")

    return float(dot_product / (norm_a * norm_b))
