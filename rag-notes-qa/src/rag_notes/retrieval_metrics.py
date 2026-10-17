from rag_notes.models import Chunk


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Return 1.0 if any of the top-k retrieved IDs is relevant, else 0.0.

    Args:
        retrieved_ids: chunk IDs in ranked order, best match first
        relevant_ids: the set of chunk IDs considered relevant for this query
        k: how many of the top results to check
    Returns:
        1.0 on a hit within the top k, 0.0 otherwise
    """
    return 1.0 if any(retrieved_id in relevant_ids for retrieved_id in retrieved_ids[:k]) else 0.0


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Return 1 / rank of the first relevant ID in retrieved_ids (1-indexed).

    Args:
        retrieved_ids: chunk IDs in ranked order, best match first
        relevant_ids: the set of chunk IDs considered relevant for this query
    Returns:
        1 / rank of the first hit, or 0.0 if nothing relevant was retrieved
    """
    for retrieved_id in retrieved_ids:
        if retrieved_id in relevant_ids:
            return 1 / (1 + retrieved_ids.index(retrieved_id))
    return 0.0


def mean_reciprocal_rank(scores: list[float]) -> float:
    """Average a list of per-query reciprocal_rank() scores into one MRR value.

    Args:
        scores: one reciprocal_rank() result per query
    Returns:
        the mean of scores, or 0.0 for an empty list
    """
    if len(scores) == 0:
        return 0.0
    return sum(scores) / len(scores)


def normalize_text(text: str) -> str:
    """Lowercase text and collapse every run of whitespace into one space.

    Leading and trailing whitespace is removed.

    Args:
        text: any string, e.g. a chunk's text or an answer key
    Returns:
        the normalized string
    """
    return " ".join(text.lower().split()).strip()


def find_relevant_chunk_ids(chunks: list[Chunk], answer_key: str) -> set[str]:
    """Return the ids of every chunk whose text contains answer_key.

    Both the chunk text and the key are compared after normalize_text().
    Ids use the same format as hybrid_search(): "<source.title>-<chunk_index>".

    Args:
        chunks: every chunk in the index (e.g. retrieval_index.bm25_index.chunks)
        answer_key: the golden entry's short answer span
    Returns:
        the set of matching chunk ids; empty if no chunk contains the key
    Raises:
        ValueError: if answer_key is empty after normalization
    """
    answer_key = normalize_text(answer_key)
    if answer_key == "":
        raise ValueError("Answer_key cannot be an empty string")

    relevant_chunk_ids: set[str] = set()
    for chunk in chunks:
        chunk_text = normalize_text(chunk.text)
        if answer_key in chunk_text:
            chunk_id = f"{chunk.source.title}-{chunk.chunk_index}"
            relevant_chunk_ids.add(chunk_id)

    return relevant_chunk_ids
